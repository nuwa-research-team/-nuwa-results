import gzip
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
PREV=OUT.parent/'d4_segment_pcap'
sys.path.insert(0,str(PREV))
import compare as seg
old=seg.old
ARMS=(*seg.ARMS,'combined')


def prepare(rows,masks):
    return [(old.inputs(r,q),seg.segments(r,q)) for r,q in zip(rows,masks,strict=True)]


def matrix(examples,schema):
    width,used,vec=schema
    packet,runs=zip(*examples)
    return np.column_stack((old.matrix(packet,width)[:,used],seg.dense(vec,runs)))


def predict(examples,folds,heads,schemas):
    out=np.empty(len(folds),int)
    for f in np.unique(folds):
        idx=np.flatnonzero(folds==f)
        out[idx]=heads[f].predict(matrix([examples[i] for i in idx],schemas[f]))
    return out


def downstream(tag,rows,heads,schemas,ordered,segments):
    with np.load(old.OLD/f'{tag}_predictions.npz') as router:
        ids=np.array([r['id'] for r in rows])
        assert np.array_equal(ids,router['ids'])
        result={'ids':ids,'y':router['y']}
        folds=old.index_folds(rows)
        for rate in old.RATES:
            if tag=='full':
                with np.load(old.DATA/f'mask{rate}.npz') as f:
                    assert np.array_equal(ids,f['ids'])
                    qs=[np.flatnonzero(m) for m in f['mask']]
            else:
                qs=[r['masks'][str(rate)] for r in rows]
            x=prepare(rows,qs)
            p=predict(x,folds,heads,schemas)
            if tag=='full':
                pp=old.cross_predict(ordered,folds,[packet for packet,_ in x])
                sp=seg.predict(segments,rows,qs,folds)
                votes=dict(zip(ARMS,(*pp,sp,p)))
            else:
                votes={'combined':p}
            for kind in old.ds.NAMES:
                base=router[f'{kind}_joint{rate}']
                for arm,v in votes.items():
                    result[f'{rate}_{arm}_{kind}']=np.where(base<2,v,base)
        if tag!='full':
            with np.load(PREV/f'{tag}.npz') as prior:
                assert np.array_equal(prior['ids'],ids) and np.array_equal(prior['y'],result['y'])
                for k in prior.files:
                    if k not in result: result[k]=prior[k]
        np.savez_compressed(OUT/f'{tag}.npz',**result)


def main():
    old_hash=json.loads((PREV/'decision.json').read_bytes())['hashes']
    for filename,digest in old_hash.items():
        assert old.sha(Path(filename))==digest,filename
    panels=old.read_panels()
    rows=[r for r in panels['train'] if old.LABEL[old.file_of(r)] in (0,1)]
    ids=np.array([r['id'] for r in rows])
    folds=old.index_folds(rows)
    y=np.array([old.LABEL[old.file_of(r)] for r in rows])
    ordered=joblib.load(seg.PREV/'downstream_heads.joblib')
    segments=joblib.load(PREV/'segment_heads.joblib')
    schemas={f:(ordered[f][0],ordered[f][1],segments[f][0]) for f in ordered}
    train_masks=[]
    for row in rows:
        (rate,q),=row['masks'].items()
        assert np.array_equal(q,old.fixed_mask(row,int(rate)))
        train_masks.append(q)
    train=prepare(rows,train_masks)
    heads={}
    for f in (0,1,2):
        idx=np.flatnonzero(folds!=f)
        if f<2: assert {old.file_of(rows[i]) for i in idx}.isdisjoint(old.FILES[f])
        heads[f]=old.estimator().fit(matrix([train[i] for i in idx],schemas[f]),y[idx])
    joblib.dump(heads,OUT/'combined_heads.joblib')
    with np.load(PREV/'paired.npz') as prior:
        preds={k:prior[k] for k in prior.files}
    assert np.array_equal(ids,preds['ids']) and np.array_equal(y,preds['y']) and np.array_equal(folds,preds['fold'])
    for rate in old.RATES:
        qs=[old.fixed_mask(row,rate) for row in rows]
        x=prepare(rows,qs)
        # Replay controls to check the mask sampler, features, fold and order together.
        a,b=old.cross_predict(ordered,folds,[packet for packet,_ in x])
        c=seg.predict(segments,rows,qs,folds)
        for arm,v in zip(seg.ARMS,(a,b,c)): assert np.array_equal(v,preds[f'{arm}{rate}'])
        preds[f'combined{rate}']=predict(x,folds,heads,schemas)
    np.savez_compressed(OUT/'paired.npz',**preds)
    ordered50=preds['ordered50']; joint=preds['combined50']
    rescue=(y==1)&(preds['summary50']==0)&(ordered50==1)
    assert int(rescue.sum())==143
    decisions={}
    for f,name in enumerate(('A','B')):
        sel=folds==f
        left=old.metrics(y[sel],ordered50[sel],2)
        right=old.metrics(y[sel],joint[sel],2)
        decisions[name]=dict(rescued=int(sum(rescue&sel)),retained=int(sum(rescue&sel&(joint==1))),
            smb_error_delta=right['sw']-left['sw'],weibo_error_delta=right['ws']-left['ws'],
            passed=right['sw']<=left['sw'] and right['ws']<left['ws'])
    (OUT/'decision.json').write_text(json.dumps(decisions,indent=1))
    (OUT/'identity.json').write_text(json.dumps({str(p):old.sha(p) for p in
        [Path(__file__),OUT/'protocol.txt',OUT/'combined_heads.joblib',PREV/'paired.npz',old.DEV/'panels.json.gz']},indent=1))
    print(decisions,flush=True)
    for tag in ('screen','confirm'):
        downstream(tag,panels[tag],heads,schemas,ordered,segments)
    if all(d['passed'] for d in decisions.values()):
        timing=json.loads(gzip.decompress((old.FULL/'verified.json.gz').read_bytes()))
        full=[dict(id=r['id'],s=r['signed'],t=timing[r['id']][1]) for r in map(json.loads,(old.DATA/'flows.jsonl').open())]
        assert len(full)==13150 and not set(ids).intersection(r['id'] for r in full)
        downstream('full',full,heads,schemas,ordered,segments)


if __name__=='__main__':
    with threadpool_limits(limits=3): main()
