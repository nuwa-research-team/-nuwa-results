"""Frozen D4 10% and 30% replay: substitute TFI signed sizes at B-predicted short slots."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
P = ROOT.parent
D = P.parent/'d4_downstream_random'
F = P.parent/'d4_full_aligned_20261113'
sys.path.insert(0,str(D))
import downstream as ds

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

@torch.inference_mode()
def main(rate):
    OUT = ROOT / str(rate)
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    flowrows=ds.read(D/'flows.jsonl')
    ids=np.array([r['id'] for r in flowrows])
    y=np.array([r['y'] for r in flowrows])
    assert len(ids)==len(set(ids))==13150
    maskpath=D/f'mask{rate}.npz'
    mask_sha=sha(maskpath)
    with np.load(P/f'direct_{rate}.npz') as z, np.load(maskpath) as m:
        assert mask_sha==str(z['mask_sha']) and np.array_equal(ids,z['ids']) and np.array_equal(ids,m['ids'])
        masks=m['mask']
        nq,q,b=z['nq'],z['q'],z['pred'].astype(int)
    assert np.all(b!=0) and len(b)==len(q)==nq.sum()
    with np.load(ds.PE / f'tfi_imputed_{rate}.npz') as tfi:
        ref_ids = tfi['ids'].astype(str)
        assert len(ref_ids) == len(set(ref_ids))
        ref = {k: v[:n] for k, v, n in zip(ref_ids, tfi['values'], tfi['lengths'])}
    assert set(ids).issubset(ref)
    split=np.r_[0,np.cumsum(nq)]
    hybrid=b.copy()
    eligible=np.zeros(b.size,dtype=bool)
    completed={}
    changedflows=np.zeros(len(ids),dtype=bool)
    for i,r in enumerate(flowrows):
        s=np.asarray(r['signed']).copy()
        mask=masks[i,:len(s)]
        sec=slice(split[i],split[i+1])
        pos=q[sec]
        t=ref[r['id']]
        assert len(t)==len(s) and np.array_equal(np.flatnonzero(mask),pos) and np.array_equal(t[~mask],s[~mask])
        assert np.all(t[pos] == t[pos].astype(int))
        eligible[sec]=(abs(b[sec])<=128)&(t[pos]!=0)
        hybrid[sec]=np.where(eligible[sec],t[pos],b[sec])
        changedflows[i]=np.any(hybrid[sec]!=b[sec])
        s[pos]=hybrid[sec]
        completed[r['id']]=s.tolist()
    assert np.array_equal(hybrid[~eligible],b[~eligible])
    np.savez_compressed(OUT/'joint_packets.npz',ids=ids,mask_sha=mask_sha,nq=nq,q=q,eligible=eligible,pred=hybrid)
    audit=dict(flows=len(ids),masked=len(b),eligible=int(eligible.sum()),changed=int((hybrid!=b).sum()),flows_changed=int(changedflows.sum()),mask_sha=mask_sha)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(audit,flush=True)
    raw=pickle.loads(ds.RAW.read_bytes())
    orig=dict(zip(raw['ids'],range(len(raw['ids']))))
    identity=json.loads((F/'cloud_code/identity.json').read_bytes())
    for k in ['fsnet','cnn','lstm','gru','sdae','tst']:
        w=ds.SNAP/f'work/res_aligned_{k}/{k}.pt' if k in ds.NAMES[:3] else ds.REPAIRED/f'repaired/repaired_{k}.pt'
        assert sha(w)==identity['classifiers'][k]
        with np.load(ds.archive(k,'clean',0)) as archive:
            keys=archive['ids'].astype(str)
            assert np.array_equal(archive['y'],[raw['y'][orig[key]] for key in keys])
        bykey={key:i for i,key in enumerate(keys)}
        assert len(bykey)==len(keys)
        order=[bykey[key] for key in ids]
        assert np.array_equal(y,[raw['y'][orig[key]] for key in ids])
        seqs=[completed[key] if key in completed else raw['seqs'][orig[key]] for key in keys]
        model=ds.model_for(k,8).eval()
        indices=np.arange(len(keys))
        if k in ds.NAMES[:3]:
            pred=ds.pv.classify(model,seqs,indices,'cpu',512)
        else:
            pred=ds.er.classify(model,seqs,indices,'cpu',16 if k=='tst' else 48,1/1600)
        pred=np.asarray(pred)[order]
        with np.load(P/f'{k}_direct_{rate}.npz') as baseline:
            lookup=dict(zip(baseline['ids'],range(len(baseline['ids']))))
            before=baseline['pred'][[lookup[key] for key in ids]]
            assert np.array_equal(before[~changedflows],pred[~changedflows])
        np.savez_compressed(OUT/f'{k}.npz',ids=ids,y=y,pred=pred)
        print(k,ds.summarize(y,pred),flush=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rate", choices=[10,30], type=int)
    main(parser.parse_args().rate)
