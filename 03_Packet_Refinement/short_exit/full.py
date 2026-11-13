"""Evaluate the locked short head in original batch order with identical padding."""
import json,pickle,sys,joblib
from pathlib import Path
import numpy as np
import torch
from replay import variants, OUT, OLD
from dev import ds,sha,score

P=OUT.parent
F=P.parent/'d4_full_aligned_20261113'
D=P.parent/'d4_downstream_random'

@torch.inference_mode()
def main():
    dev=json.loads((OUT/'results.json').read_bytes())
    assert dev['selected'] == 'both' and sha(OLD/'head.joblib')==dev['hashes']['head']
    head=joblib.load(OLD/'head.joblib')
    torch.set_num_threads(4)
    rows=ds.read(D/'flows.jsonl')
    ids=np.array([r['id'] for r in rows])
    y=np.array([r['y'] for r in rows])
    seqs=[r['signed'] for r in rows]
    assert len(set(ids))==len(ids)==13150
    raw=pickle.loads(ds.RAW.read_bytes())
    assert set(ids)<=set(str(raw['ids'][i]) for i in raw['split']['test'])
    original=dict(zip(raw['ids'],raw['seqs']))
    weights=json.loads((F/'cloud_code/identity.json').read_bytes())['classifiers']
    for rate in [50,30,10]:
        with np.load(P/f'direct_{rate}.npz') as z,np.load(D/f'mask{rate}.npz') as mask:
            assert np.array_equal(ids,z['ids']) and np.array_equal(ids,mask['ids'])
            assert str(z['mask_sha'])==sha(D/f'mask{rate}.npz')
            positions=[np.flatnonzero(m[:len(s)]) for m,s in zip(mask['mask'],seqs,strict=True)]
            nq=z['nq']; b=z['pred']
            assert np.array_equal(np.concatenate(positions),z['q']) and np.array_equal(list(map(len,positions)),nq)
            arms=variants(head,seqs,positions,b)
            with np.load(OLD/f'packets_{rate}.npz') as ref:
                assert np.array_equal(ref['ids'], ids) and np.array_equal(ref['q'],z['q']) and np.array_equal(ref['pred'], arms['v1'])
            own=arms[dev['selected']]
            np.savez_compressed(OUT/f'packets_{rate}.npz',ids=ids,nq=nq,q=z['q'],pred=own,mask_sha=z['mask_sha'])
        completed={};changed=set();off=0
        for key,s,q in zip(ids,seqs,positions,strict=True):
            packet=own[off:off+len(q)]
            if np.any(packet!=b[off:off+len(q)]):changed.add(key)
            seq=np.array(s);seq[q]=packet;completed[key]=seq;off+=len(q)
        assert off==len(own)
        print(rate,'changed',len(changed),'flows',int((own!=b).sum()),'slots',flush=True)
        for name in ['fsnet','cnn','lstm','gru','sdae','tst']:
            model=ds.model_for(name,8)
            legacy=name in ds.NAMES[:3]
            w=ds.SNAP/f'work/res_aligned_{name}/{name}.pt' if legacy else ds.REPAIRED/f'repaired/repaired_{name}.pt'
            assert sha(w)==weights[name]
            with np.load(ds.archive(name,'clean',0)) as archive:
                keys=archive['ids'].astype(str); labels=archive['y']
            lookup=dict(zip(keys,range(len(keys))))
            order=np.array([lookup[k] for k in ids])
            assert np.array_equal(y,labels[order])
            with np.load(P/f'{name}_direct_{rate}.npz') as base:
                ix=dict(zip(base['ids'].astype(str),range(len(base['ids']))))
                inds=[ix[k] for k in ids]
                assert np.array_equal(y,base['y'][inds])
                before=base['pred'][inds]
            allpred=np.full(len(keys),-1,dtype=int)
            allpred[order]=before
            sequences=[completed[k] if k in completed else original[k] for k in keys]
            batch=512 if legacy else 16 if name=='tst' else 48
            # Identical inputs in unchanged batches already scored by this checkpoint.
            for start in range(0,len(keys),batch):
                sel=np.arange(start,min(start+batch,len(keys)))
                if changed.isdisjoint(keys[sel]):continue
                pred=(ds.pv.classify(model,sequences,sel,'cpu',batch) if legacy else
                      ds.er.classify(model,sequences,sel,'cpu',batch,1/1600))
                allpred[sel]=pred
            result=allpred[order]
            unchanged=np.array([k not in changed for k in ids])
            assert np.array_equal(result[unchanged],before[unchanged]) and (result>=0).all()
            np.savez_compressed(OUT/f'{name}_{rate}.npz',ids=ids,y=y,pred=result)
            print(rate,name,score(y,result),flush=True)

if __name__=='__main__':main()

