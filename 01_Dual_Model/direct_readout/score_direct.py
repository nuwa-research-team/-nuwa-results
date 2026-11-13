"""Score B in the archived order, with the same batch widths as the TFI controls."""
import gzip,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
F=P.parent/'d4_full_aligned_20261113'
D=P.parent/'d4_downstream_random'
sys.path.insert(0,str(D))
import downstream as ds
torch.set_num_threads(4)
flows=ds.read(D/'flows.jsonl');ids=[r['id'] for r in flows]
ids_idx=dict(zip(ids,range(len(ids))));assert len(ids_idx)==13150
raw=pickle.loads(ds.RAW.read_bytes())
orig_idx=dict(zip(raw['ids'],range(len(raw['ids']))))
identity=json.loads((F/'cloud_code/identity.json').read_bytes())
readout_hash=json.loads((P/'done.json').read_bytes())['head']
assert hashlib.sha256((F/'dev_readout/B.pt').read_bytes()).hexdigest()==readout_hash
allslots=[]
refs_old=json.loads((F/'slot_comparison.json').read_bytes())
for rate in (10,30,50):
    z=np.load(P/f'direct_{rate}.npz');assert z['ids'].astype(str).tolist()==ids
    path=D/f'mask{rate}.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==str(z['mask_sha'])
    m=np.load(path);assert m['ids'].astype(str).tolist()==ids
    if rate==50:
        with gzip.open(F/'tfi_reference/tfi_50.jsonl.gz','rt') as f:
            ref={r['id']:np.asarray(r['signed']) for r in map(json.loads,f)}
    else:
        a=np.load(ds.PE/f'tfi_imputed_{rate}.npz')
        ref={k:v[:n] for k,v,n in zip(a['ids'].astype(str),a['values'],a['lengths'])}
    nq=z['nq'];positions=z['q'];predicted=z['pred'];maskrows=m['mask']
    off=0;completed={};byflow=[]
    for i,r in enumerate(flows):
        k=r['id'];s=np.asarray(r['signed'],dtype=int);n=int(nq[i]);q=positions[off:off+n];v=predicted[off:off+n];off+=n
        mask=maskrows[i][:len(s)]
        assert np.array_equal(np.flatnonzero(mask),q) and np.array_equal(ref[k][~mask],s[~mask])
        true=s[q];tf=ref[k][q].astype(int)
        e=np.column_stack([abs(abs(v)-abs(true)),abs(abs(tf)-abs(true)),abs(v-true),abs(tf-true),np.sign(v)==np.sign(true),np.sign(tf)==np.sign(true),v==true,tf==true])
        byflow.append(e.sum(0))
        s[q]=v;completed[k]=s.tolist()
    assert off==len(z['pred'])
    sums=np.array(byflow);np.savez_compressed(P/f'errors_{rate}.npz',ids=ids,nq=z['nq'],sums=sums)
    value=sums.sum(axis=0)/off
    old=next(v for v in refs_old if v['rate']==rate and v['group']=='All' and v['length']=='all')
    assert np.isclose(value[1],old['TFI']['abs_mae'])
    allslots.append(dict(rate=rate,n=off,amp=value[:2].tolist(),signed=value[2:4].tolist(),direction=(100*value[4:6]).tolist(),exact=(100*value[6:8]).tolist()))
    (P/'packet_metrics.json').write_text(json.dumps(allslots,indent=2))
    print('Packet',allslots[-1],flush=True)
    for name in ds.NAMES:
        target=P/f'{name}_direct_{rate}.npz'
        if target.exists():continue
        weights=ds.SNAP/f'work/res_aligned_{name}/{name}.pt' if name in ds.NAMES[:3] else ds.REPAIRED/f'repaired/repaired_{name}.pt'
        assert hashlib.sha256(weights.read_bytes()).hexdigest()==identity['classifiers'][name]
        base=np.load(ds.archive(name,'clean',0));keys=base['ids'].astype(str)
        seqs=[completed[k] if k in completed else raw['seqs'][orig_idx[k]] for k in keys]
        yy=base['y'];assert [int(raw['y'][orig_idx[k]]) for k in keys]==yy.tolist()
        model=ds.model_for(name,8)
        inds=np.arange(len(keys))
        if name in ds.NAMES[:3]:pred=ds.pv.classify(model,seqs,inds,'cpu',512)
        else:pred=ds.er.classify(model,seqs,inds,'cpu',16 if name=='tst' else 48,1/1600)
        valid=np.isin(keys,ids);assert sum(valid)==len(ids)
        np.savez_compressed(target,ids=keys[valid],y=yy[valid],pred=np.asarray(pred)[valid])
        print(rate,name,'done',flush=True)
