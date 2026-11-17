from collections import Counter
import gzip
import json
import numpy as np
import torch
from torch import nn
from logits import BASE, OUT, KINDS, load_inputs, sha

FULL = BASE.parent/'d4_full_aligned_20261113'
DATA = BASE.parent/'d4_downstream_random'
RATES = (10,30,50)
METHODS = ('dual','tfi')


def features(times, visible):
    dt = np.diff(np.asarray(times,dtype=np.int64)[visible])
    valid = dt >= 0
    values = np.full(len(dt),np.nan)
    values[valid] = np.log1p(dt[valid])
    out = np.full(19,np.nan)
    n = min(12,len(dt))
    out[:n] = values[:n]
    if valid.any():
        x = values[valid]
        out[12:] = [x.mean(),x.std(),*np.quantile(x,[0,.25,.5,.75,1])]
    return out, int((~valid).sum())


def prepare(data):
    panel_file = FULL/'dev_readout/panels.json.gz'
    assert sha(panel_file)==json.loads((FULL/'dev_readout/identity.json').read_bytes())['panels']
    panels = json.loads(gzip.decompress(panel_file.read_bytes()))
    flows = [json.loads(line) for line in (DATA/'flows.jsonl').open()]
    ids = [r['id'] for r in flows]
    times = json.loads(gzip.decompress((FULL/'verified.json.gz').read_bytes()))
    assert len(ids)==len(set(ids))==len(times)==13150 and set(ids)==times.keys()
    dev_ids = [r['id'] for panel in panels.values() for r in panel]
    assert len(dev_ids)==len(set(dev_ids)) and set(dev_ids).isdisjoint(ids)
    labels = json.loads((FULL/'six_label_audit.json').read_bytes())['mapping']
    def label(r):
        return labels[r['id'].split('.pcap#')[0]]
    X, offsets, audit = {}, {}, {}
    for tag in ('train','screen'):
        rows = panels[tag]
        feat, indices = [], []
        for i,r in enumerate(rows):
            assert np.array_equal(r['s'],data[tag][0][i]) and label(r)==data[tag][1][i]
            for q in r['masks'].values():
                visible = np.ones(len(r['s']),bool)
                visible[q] = False
                feat.append(features(r['t'],visible)[0])
                indices.append(i)
        X[tag] = np.array(feat)
        offsets[tag] = np.array(indices)
    for rate in RATES:
        path = DATA/f'mask{rate}.npz'
        with np.load(path) as mf:
            assert mf['ids'].tolist()==ids
            masks = mf['mask']
        feat, bad, nbad = [],0,0
        for i,(r,m) in enumerate(zip(flows,masks,strict=True)):
            n = len(r['signed'])
            assert not m[n:].any() and len(times[r['id']][1])==n
            vis = ~m[:n]
            for method in METHODS:
                seq, y = data[f'{method}{rate}']
                assert len(seq[i])==n and label(r)==y[i]
                assert np.array_equal(np.asarray(r['signed'])[vis],seq[i][vis])
            ts = times[r['id']][1]
            fv, inv = features(ts,vis)
            if i<10:
                other = np.array(ts)
                other[~vis] = -999999
                assert np.array_equal(features(other,vis)[0],fv,equal_nan=True)
            feat.append(fv)
            bad+=inv
            nbad+=bool(inv)
        X[rate] = np.array(feat)
        audit[rate] = dict(mask_sha=sha(path),hidden=int(masks.sum()),negative_visible_intervals=bad,affected_flows=nbad)
    mean = np.nanmean(X['train'],axis=0)
    std = np.maximum(.01,np.nanstd(X['train'],axis=0))
    for tag,rows in X.items():
        x = np.nan_to_num((rows-mean)/std,nan=0.)
        X[tag] = torch.from_numpy(x.astype('float32'))
    return X, offsets, ids, audit, mean, std


class TimeReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(19,64)
        self.last = nn.Linear(64,6,bias=False)
        nn.init.zeros_(self.last.weight)

    def forward(self,x):
        activate = nn.functional.gelu
        return self.last(activate(self.first(x))-activate(self.first.bias))


def fit(X, base, y):
    torch.manual_seed(42)
    head = TimeReadout()
    opt = torch.optim.AdamW(head.parameters(),lr=.001)
    score = nn.functional.cross_entropy
    best = float('inf')
    for ep in range(21):
        if ep:
            for ids in torch.randperm(len(y['train'])).split(128):
                loss = score(head(X['train'][ids])+base['train'][ids],y['train'][ids])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            val = score(head(X['screen'])+base['screen'],y['screen']).item()
        if val<best:
            epoch,best,state = ep,val,{k:v.detach().clone() for k,v in head.state_dict().items()}
    head.load_state_dict(state)
    return head,epoch


def scores(y,p):
    cm = np.bincount(6*y+p,minlength=36).reshape(6,6)
    f1 = 2*np.diag(cm)/np.maximum(1,cm.sum(1)+cm.sum(0))
    return dict(acc=100*float(np.trace(cm))/len(y),f1=100*float(f1.mean()),sw=int(cm[1,0]),ws=int(cm[0,1]),cm=cm.tolist())


def main():
    torch.set_num_threads(2)
    data = load_inputs()
    X, index, ids, audit, mean, std = prepare(data)
    original = np.load(BASE/'six_predictions.npz')
    records, weights, preds, epochs = [], {}, {}, {}
    for kind in KINDS:
        with np.load(OUT/('logits_'+kind+'.npz')) as lg:
            base = {tag:torch.from_numpy(lg[tag][ix]) for tag,ix in index.items()}
            ys = {tag:torch.LongTensor(data[tag][1][ix]) for tag,ix in index.items()}
            head,epoch = fit(X,base,ys)
            epochs[kind] = epoch
            weights[kind] = head.state_dict()
            with torch.no_grad():
                assert torch.all(head(torch.zeros(1,19))==0)
                for rate in RATES:
                    delta = head(X[rate])
                    for method in METHODS:
                        tag = method+str(rate)
                        y = data[tag][1]
                        before = lg[tag].argmax(1)
                        after = (torch.from_numpy(lg[tag])+delta).argmax(1).numpy()
                        old = original[kind+'_'+tag]
                        rec = dict(model=kind,rate=rate,method=method,baseline_mismatches=int((before!=old).sum()),before=scores(y,before),after=scores(y,after))
                        records.append(rec)
                        preds[kind+'_'+tag+'_before'] = before.astype('int8')
                        preds[kind+'_'+tag+'_after'] = after.astype('int8')
                        preds['y'] = y
        print(kind,'screen-selected epoch',epoch,flush=True)
    np.savez_compressed(OUT/'predictions.npz',ids=np.array(ids),**preds)
    torch.save(dict(weights=weights,mean=torch.tensor(mean),std=torch.tensor(std)),OUT/'timing_heads.pt')
    provenance = {str(p.name):sha(p) for p in (DATA/'flows.jsonl',FULL/'verified.json.gz',FULL/'dev_readout/panels.json.gz',BASE/'six_data.pkl.gz',OUT/'protocol.txt',OUT/'ablate.py',OUT/'logits.py')}
    (OUT/'metrics.json').write_text(json.dumps(dict(audit=audit,hashes=provenance,epochs=epochs,records=records),indent=1))


if __name__=='__main__':
    main()
