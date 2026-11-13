import gzip
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
PREV = OUT.parent / 'd4_ordered_pcap'
spec = importlib.util.spec_from_file_location('ordered_protocol', PREV/'run.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
ARMS = (*old.ARMS, 'segment')


def segments(row, q):
    visible = np.ones(len(row['s']), bool)
    visible[q] = False
    pos = np.flatnonzero(visible)
    signs = np.sign(np.asarray(row['s'])[pos])
    times = np.asarray(row['t'], dtype=np.int64)[pos]
    starts = np.r_[0, np.flatnonzero(np.diff(signs)) + 1]
    gaps = np.diff(pos) - 1
    dt = np.diff(times)
    log_dt = np.where(dt >= 0, np.log1p(np.maximum(dt, 0)), np.nan)
    out = {}
    for j, (start, end) in enumerate(zip(starts, np.r_[starts[1:], len(pos)])):
        prefix = f'r{j}_'
        out[prefix+'sign'] = signs[start]
        out[prefix+'count'] = end-start
        internal = log_dt[start:end-1]
        valid = internal[np.isfinite(internal)]
        out[prefix+'missing'] = gaps[start:end-1].sum()
        if len(valid):
            out[prefix+'mean_dt'] = valid.mean()
            out[prefix+'max_dt'] = valid.max()
        if start:
            out[prefix+'incoming_dt'] = log_dt[start-1]
            out[prefix+'incoming_missing'] = gaps[start-1]
        for offset,value in enumerate(internal):
            out[prefix+f'dt{offset}'] = value
            out[prefix+f'gap{offset}'] = gaps[start+offset]
    return out


def dense(vectorizer, records):
    # Missing run fields and genuine zero-duration/zero-gap pairs are distinct.
    sparse = vectorizer.transform(records).tocoo()
    array = np.full(sparse.shape, np.nan)
    array[sparse.row, sparse.col] = sparse.data
    return array


def fit_heads(rows, y, folds):
    packed = []
    summaries = []
    for row in rows:
        (rate, q), = row['masks'].items()
        assert np.array_equal(q,old.fixed_mask(row,int(rate)))
        packed.append(segments(row,q))
        summaries.append(old.describe(row['s'],row['t'],q))
    summaries = np.array(summaries)
    heads = {}
    for fold in (0,1,2):
        select = folds!=fold
        train = [packed[i] for i in np.flatnonzero(select)]
        vec = DictVectorizer().fit(train)
        x = np.column_stack([summaries[select], dense(vec,train)])
        heads[fold] = (vec, old.estimator().fit(x,y[select]))
    return heads


def predict(heads, rows, qs, folds):
    p = np.empty(len(rows), int)
    for f in np.unique(folds):
        ix = np.flatnonzero(folds==f)
        vec, model = heads[f]
        data = [segments(rows[i],qs[i]) for i in ix]
        summaries = [old.describe(rows[i]['s'],rows[i]['t'],qs[i]) for i in ix]
        p[ix] = model.predict(np.column_stack([summaries,dense(vec,data)]))
    return p


def dump(name, data):
    (OUT/name).write_text(json.dumps(data,indent=1),encoding='utf-8')


def pipeline(tag, rows, heads, old_heads):
    result = {}
    archive = {}
    with np.load(old.OLD/f'{tag}_predictions.npz') as router:
        y = router['y']
        ids = np.array([r['id'] for r in rows])
        assert np.array_equal(router['ids'], ids)
        archive.update(ids=ids,y=y)
        folds = old.index_folds(rows)
        for rate in old.RATES:
            if tag=='full':
                with np.load(old.DATA/f'mask{rate}.npz') as mf:
                    assert np.array_equal(mf['ids'],ids)
                    qs = [np.flatnonzero(m) for m in mf['mask']]
            else:
                qs = [r['masks'][str(rate)] for r in rows]
            candidate = predict(heads, rows, qs, folds)
            prior = old.cross_predict(old_heads,folds,[old.inputs(r,q) for r,q in zip(rows,qs)])
            result[rate] = {}
            for arm, votes in zip(ARMS, [*prior, candidate]):
                result[rate][arm] = {}
                for kind in old.ds.NAMES:
                    base = router[f'{kind}_joint{rate}']
                    p = np.where(base<2,votes,base)
                    if arm in old.ARMS and tag!='full':
                        with np.load(PREV/f'pipeline_{tag}.npz') as ref:
                            assert np.array_equal(p,ref[f'{rate}_held_{arm}_{kind}'])
                    archive[f'{rate}_{arm}_{kind}']=p
                    result[rate][arm][kind]=old.metrics(y,p,6)
    np.savez_compressed(OUT/f'{tag}.npz',**archive)
    dump(f'{tag}.json',result)


def main():
    # Known example: the first two visible packets share direction across a hole.
    z=segments(dict(s=[52,-999,1500,-52],t=[0,-999,9,18]),[1])
    assert z['r0_count']==2 and z['r0_gap0']==1 and z['r1_incoming_missing']==0 and z['r0_dt0']==np.log1p(9)
    panel=old.read_panels()
    rows=[r for r in panel['train'] if old.LABEL[old.file_of(r)] in (0,1)]
    y=np.array([old.LABEL[old.file_of(r)] for r in rows])
    folds=old.index_folds(rows)
    assert len(rows)==3596 and set(folds)=={0,1}
    for i in range(0,len(rows),71):
        r=rows[i]; q=next(iter(r['masks'].values()))
        s=np.array(r['s']); t=np.array(r['t']); s[q]=988; t[q]=-2000
        a,b=segments(r,q),segments(dict(s=s,t=t),q)
        assert a.keys()==b.keys() and np.array_equal(list(a.values()),list(b.values()),equal_nan=True)
    heads=fit_heads(rows,y,folds)
    joblib.dump(heads,OUT/'segment_heads.joblib')
    old_heads=joblib.load(PREV/'downstream_heads.joblib')
    metrics={}; rescued={}; archive=dict(y=y,fold=folds,ids=np.array([r['id'] for r in rows]))
    with np.load(PREV/'binary_predictions.npz') as ref:
        for k in archive: assert np.array_equal(archive[k],ref[k])
        for rate in old.RATES:
            qs=[old.fixed_mask(r,rate) for r in rows]
            prior=old.cross_predict(old_heads,folds,[old.inputs(r,q) for r,q in zip(rows,qs)])
            p=predict(heads,rows,qs,folds)
            metrics[rate]={}
            for arm,pred in zip(ARMS,[*prior,p]):
                if arm in old.ARMS: assert np.array_equal(pred,ref[f'{arm}{rate}'])
                archive[f'{arm}{rate}']=pred
                metrics[rate][arm]={key:old.metrics(y[sel],pred[sel],2) for key,sel in
                    [('all',np.ones(len(y),bool)),('A',folds==0),('B',folds==1)]}
            if rate==50:
                for f,key in enumerate(('A','B')):
                    mask=(folds==f)&(y==1)&(prior[0]==0)&(prior[1]==1)
                    rescued[key]=dict(previous=int(mask.sum()),kept=int(np.sum(mask&(p==1))))
    np.savez_compressed(OUT/'paired.npz',**archive)
    dump('metrics.json',metrics)
    passed={key:metrics[50]['segment'][key]['sw']<=metrics[50]['ordered'][key]['sw'] and
                metrics[50]['segment'][key]['ws']<metrics[50]['ordered'][key]['ws'] for key in ('A','B')}
    dump('decision.json',dict(passed=passed,rescued=rescued,hashes={str(file):old.sha(file) for file in
        [OUT/'protocol.txt',Path(__file__),PREV/'run.py',PREV/'binary_predictions.npz',PREV/'downstream_heads.joblib',old.DEV/'panels.json.gz',OUT/'segment_heads.joblib']}))
    print('50%',passed,rescued,flush=True)
    for tag in ('screen','confirm'):
        pipeline(tag,panel[tag],heads,old_heads)
    if all(passed.values()):
        times=json.loads(gzip.decompress((old.FULL/'verified.json.gz').read_bytes()))
        full=[dict(id=r['id'],s=r['signed'],t=times[r['id']][1]) for r in map(json.loads,(old.DATA/'flows.jsonl').open())]
        pipeline('full',full,heads,old_heads)


if __name__=='__main__':
    with threadpool_limits(limits=3): main()
