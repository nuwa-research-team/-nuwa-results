import gzip
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / 'd4_context_vote'
sys.path.insert(0, str(OLD))
from pair_timing import describe, estimator
from experiment import DATA, DEV, FULL, LABEL, RATES, ds, read_panels, sha

FILES = (frozenset(('SMB-1','Weibo-1')), frozenset(('SMB-2','Weibo-2','Weibo-3','Weibo-4')))
ARMS = ('summary','ordered')


def file_of(r):
    return r['id'].split('.pcap#')[0]


def inputs(r, q):
    s = np.asarray(r['s'])
    t = np.asarray(r['t'], dtype=np.int64)
    visible = np.ones(len(s), bool)
    visible[q] = False
    pos = np.flatnonzero(visible)
    dt = np.diff(t[pos])
    seq = np.full((len(pos), 3), np.nan)
    seq[:,0] = np.sign(s[pos])
    seq[1:,1] = np.where(dt>=0, np.log1p(np.maximum(0, dt)), np.nan)
    seq[1:,2] = np.diff(pos)-1
    assert len(pos)<=512
    return describe(s,t,q),seq.ravel()


def fixed_mask(r,rate):
    n = len(r['s'])
    return np.sort(np.random.default_rng(42+r['i']).choice(n,max(1,round(n*rate/100)),replace=False))


def matrix(x, width):
    result = np.full((len(x),47+width),np.nan)
    for i,(summary,seq) in enumerate(x):
        result[i,:47] = summary
        count = min(len(seq),width)
        result[i,47:47+count] = seq[:count]
    return result


def fit(x,y):
    width=max(len(seq) for _,seq in x)
    xx=matrix(x,width)
    use=np.isfinite(xx).any(axis=0)
    return width,use,estimator().fit(xx[:,:47],y),estimator().fit(xx[:,use],y)


def predict(heads,x):
    width,use,old,new=heads
    xx=matrix(x,width)
    return np.stack([old.predict(xx[:,:47]),new.predict(xx[:,use])])


def cross_predict(heads,folds,x):
    out=np.full((2,len(x)),-1)
    for fold in np.unique(folds):
        ix=np.flatnonzero(folds==fold)
        out[:,ix]=predict(heads[int(fold)],[x[i] for i in ix])
    assert (out>=0).all()
    return out


def metrics(y,p,classes):
    cm=confusion_matrix(y,p,labels=np.arange(classes))
    assert cm.sum()==len(y)
    return dict(n=len(y),acc=100*accuracy_score(y,p),f1=100*f1_score(y,p,labels=np.arange(classes),average='macro',zero_division=0),
                smb=100*cm[1,1]/cm[1].sum(),weibo=100*cm[0,0]/cm[0].sum(),
                sw=int(cm[1,0]),ws=int(cm[0,1]),cm=cm.tolist())


def index_folds(rows):
    return np.array([next((j for j,files in enumerate(FILES) if file_of(r) in files),2) for r in rows])


def save(name, data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=1),encoding='utf-8')


def strict(rows,heads,fold,y):
    res={}
    archive={'ids':np.array([r['id'] for r in rows]),'y':y,'fold':fold}
    for rate in RATES:
        xx=[inputs(r,fixed_mask(r,rate)) for r in rows]
        pp=cross_predict(heads,fold,xx)
        res[rate]={}
        for arm,p in zip(ARMS,pp):
            archive[f'{arm}{rate}']=p
            res[rate][arm]={}
            for split,select in [('all',np.ones(len(y),bool)),('A',fold==0),('B',fold==1)]:
                res[rate][arm][split]=metrics(y[select],p[select],2)
    np.savez_compressed(OUT/'binary_predictions.npz',**archive)
    save('binary_metrics.json',res)
    return res


def pipeline(name,rows,heads,flags):
    folds=index_folds(rows)
    with np.load(OLD/f'{name}_predictions.npz') as orig:
        ids=np.array([r['id'] for r in rows])
        y=orig['y']
        assert np.array_equal(ids,orig['ids'])
        archive={'ids':ids,'y':y}
        result={}
        for rate in RATES:
            if name=='full':
                with np.load(DATA/f'mask{rate}.npz') as mf:
                    assert np.array_equal(ids,mf['ids'])
                    masks=[np.flatnonzero(mask) for mask in mf['mask']]
            else:
                masks=[r['masks'][str(rate)] for r in rows]
            xx=[inputs(r,q) for r,q in zip(rows,masks,strict=True)]
            result[rate]={}
            for flag in flags:
                pp=cross_predict(heads,folds,xx) if flag=='held' else predict(heads[2],xx)
                result[rate][flag]={}
                for arm,rerank in zip(ARMS,pp):
                    result[rate][flag][arm]={}
                    for kind in ds.NAMES:
                        base=orig[f'{kind}_joint{rate}']
                        p=np.where(base<2,rerank,base)
                        archive[f'{rate}_{flag}_{arm}_{kind}']=p
                        result[rate][flag][arm][kind]=metrics(y,p,6)
                        if flag=='pooled' and arm=='summary':
                            with np.load(OLD/f'pair_{name}_predictions.npz') as ref:
                                assert np.array_equal(p,ref[f'{kind}_augmented{rate}'])
    np.savez_compressed(OUT/f'pipeline_{name}.npz',**archive)
    save(f'pipeline_{name}.json',result)


def main():
    panel=read_panels()
    rows=[r for r in panel['train'] if LABEL[file_of(r)] in (0,1)]
    fold=index_folds(rows)
    y=np.array([LABEL[file_of(r)] for r in rows])
    xx=[]
    for r in rows:
        (item,)=r['masks'].items()
        k,q=item
        assert np.array_equal(q,fixed_mask(r,int(k)))
        xx.append(inputs(r,q))
    # Changing every hidden value must leave the training input identical.
    for i in range(0,len(rows),73):
        r=rows[i]; q=next(iter(r['masks'].values()))
        s=np.array(r['s']); t=np.array(r['t'])
        s[q]=31415; t[q]=-999999
        for a,b in zip(xx[i],inputs(dict(r,s=s,t=t),q)):
            assert np.array_equal(a,b,equal_nan=True)
    heads={}
    for f in (0,1,2):
        ix=np.flatnonzero(fold!=f)
        assert f==2 or not {file_of(rows[i]) for i in ix}.intersection(FILES[f])
        heads[f]=fit([xx[i] for i in ix],y[ix])
    joblib.dump(heads,OUT/'downstream_heads.joblib')
    res=strict(rows,heads,fold,y)
    change={k:np.mean([res[r]['ordered']['all'][k]-res[r]['summary']['all'][k] for r in RATES]) for k in ('acc','f1')}
    passed=min(change.values())>0
    save('decision.json',dict(mean_delta=change,expand=bool(passed),file_groups=[sorted(x) for x in FILES],
        hashes={str(p):sha(p) for p in [Path(__file__),OUT/'protocol.txt',DEV/'panels.json.gz',OLD/'pair_timing.py',OUT/'downstream_heads.joblib']}))
    print('Gate',change,passed,flush=True)
    for name in ('screen','confirm'):
        pipeline(name,panel[name],heads,['held'])
    if passed:
        t=json.loads(gzip.decompress((FULL/'verified.json.gz').read_bytes()))
        full=[dict(id=r['id'],s=r['signed'],t=t[r['id']][1]) for r in map(json.loads,(DATA/'flows.jsonl').open())]
        assert len(full)==13150 and not {r['id'] for r in rows}.intersection(r['id'] for r in full)
        pipeline('full',full,heads,['held','pooled'])
    print('Finished',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=3): main()
