import gzip
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from experiment import DATA, FULL, LABEL, P, RATES, ds, features, read_panels, scores


def describe(signed, times, hidden):
    visible = np.ones(len(signed), dtype=bool)
    visible[hidden] = False
    ix = np.flatnonzero(visible)
    direction = np.sign(np.asarray(signed)[ix])
    dt = np.diff(np.asarray(times, dtype=np.int64)[ix])
    gaps = np.diff(ix)
    valid = dt >= 0
    per_slot = np.log1p(np.maximum(dt, 0) / gaps)
    masks = [(direction[:-1] == a) & (direction[1:] == b)
             for a in (-1, 1) for b in (-1, 1)]
    thirds = np.minimum(2, 3 * ix[1:] // len(signed))
    masks.extend(thirds == j for j in range(3))
    extra = []
    for mask in masks:
        selected = per_slot[mask & valid]
        extra.append(len(selected) / max(1, len(dt)))
        extra.extend(np.quantile(selected, [.25, .5, .75]) if len(selected) else [np.nan]*3)
    return np.r_[features(times, visible)[0], extra]


def estimator():
    return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, min_samples_leaf=25,
        l2_regularization=2., learning_rate=.08, early_stopping=False, random_state=42)


def train_heads(rows):
    x, y, group = [], [], []
    for row in rows:
        file = row['id'].split('.pcap#')[0]
        if LABEL[file] not in (0, 1):
            continue
        (q,) = row['masks'].values()
        x.append(describe(row['s'], row['t'], q))
        y.append(LABEL[file])
        group.append(file in ('SMB-1','Weibo-1'))
    x, y, group = np.array(x), np.array(y), np.array(group)
    heads = {arm: estimator().fit(x[:, :width], y) for arm,width in [('coarse',19),('augmented',47)]}
    transfers = []
    for hold in (group, ~group):
        p = estimator().fit(x[~hold], y[~hold]).predict(x[hold])
        recalls = [float(np.mean(p[y[hold]==k]==k)) for k in (0,1)]
        transfers.append(dict(n=int(hold.sum()),recalls=recalls,balanced_accuracy=np.mean(recalls)))
    joblib.dump(heads, P/'pair_heads.joblib')
    return heads, transfers


def run_panel(tag, rows, heads):
    result = {}
    with np.load(P/f'{tag}_predictions.npz') as before:
        y = before['y']
        assert before['ids'].tolist() == [r['id'] for r in rows]
        preds = {'ids':before['ids'], 'y':y}
        for rate in RATES:
            if tag=='full':
                with np.load(DATA/f'mask{rate}.npz') as mf:
                    assert np.array_equal(mf['ids'], before['ids'])
                    masks = [np.flatnonzero(m) for m in mf['mask']]
            else:
                masks = [r['masks'][str(rate)] for r in rows]
            xx = np.array([describe(r['s'],r['t'],q) for r,q in zip(rows,masks,strict=True)])
            result[rate] = {}
            for arm,head in heads.items():
                rerank = head.predict(xx[:, :head.n_features_in_])
                models = {}
                for kind in ds.NAMES:
                    original = before[f'{kind}_joint{rate}']
                    p = np.where(original < 2,rerank,original)
                    preds[f'{kind}_{arm}{rate}'] = p
                    models[kind] = scores(y.astype(int),p.astype(int))
                result[rate][arm] = dict(models=models, **{k:np.mean([m[k] for m in models.values()]) for k in ('acc','f1')})
    np.savez_compressed(P/f'pair_{tag}_predictions.npz',**preds)
    (P/f'pair_{tag}.json').write_text(json.dumps(result,indent=1))
    return result


def main():
    panels = read_panels()
    heads,transfer = train_heads(panels['train'])
    (P/'pair_transfer.json').write_text(json.dumps(transfer,indent=1))
    passed = []
    for tag in ('screen','confirm'):
        res = run_panel(tag,panels[tag],heads)
        baseline = json.loads((P/f'{tag}.json').read_bytes())['rates']
        delta = [np.mean([res[r]['augmented'][k]-baseline[str(r)]['joint'][k] for r in RATES]) for k in ('acc','f1')]
        print(tag, delta, flush=True)
        passed.append(min(delta)>0)
    if all(passed):
        times = json.loads(gzip.decompress((FULL/'verified.json.gz').read_bytes()))
        rows = [dict(id=r['id'],s=r['signed'],t=times[r['id']][1]) for r in map(json.loads,(DATA/'flows.jsonl').open())]
        run_panel('full',rows,heads)
    print('done',passed,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=3):
        main()
