import json,gzip,sys,joblib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
from short_head import fit,apply
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'dev_sign_swap'))
from run import ds,DEV,sha,metrics
arms=['原双路','自有短包头','短包换TFI','完整TFI']

def classify(model,seqs,pos,packed):
    completed=[];off=0
    for seq,q in zip(seqs,pos,strict=True):
        s=np.array(seq);s[q]=packed[off:off+len(q)]
        completed.append(s);off+=len(q)
    assert off==len(packed)
    return ds.er.classify(model,completed,np.arange(len(seqs)),'cpu',48,1/1600)

def score(y,p):
    return dict(**metrics(y,p),acc=100*accuracy_score(y,p),
                macro8=100*f1_score(y,p,labels=range(8),average='macro',zero_division=0))

def run():
    identity=json.loads((DEV/'identity.json').read_bytes())
    assert sha(DEV/'panels.json.gz')==identity['panels']
    panels=json.loads(gzip.decompress((DEV/'panels.json.gz').read_bytes()))
    ids=[r['id'] for rows in panels.values() for r in rows]
    assert len(ids)==len(set(ids))
    m,meta=fit(panels['train'])
    joblib.dump(m,OUT/'head.joblib',compress=3)
    torch.set_num_threads(4)
    w=ds.REPAIRED/'repaired/repaired_fsnet.pt'
    assert sha(w)==json.loads((DEV.parent/'cloud_code/identity.json').read_bytes())['classifiers']['fsnet']
    fs=ds.model_for('fsnet',8)
    groups={}; saved={};counts={}
    for tag in ['screen','confirm']:
        rows=panels[tag]
        with np.load(DEV/f'{tag}_pred.npz') as z:
            y=z['y']; ids=z['ids']
            assert list(ids)==[r['id'] for r in rows] and list(y)==[r['y'] for r in rows]
            pos=[r['masks']['30'] for r in rows]
            seqs=[r['s'] for r in rows]
            assert np.array_equal(np.concatenate(pos),z['q_30']) and np.array_equal(list(map(len,pos)),z['nq_30'])
            b=z['B_30_packet']
            own=apply(m,seqs,pos,b)
            t=z['TFI_30_packet']
            blend=np.where((abs(b)<=128)&(t!=0),t,b)
            pred=[classify(fs,seqs,pos,p) for p in [b,own,blend,t]]
            assert np.array_equal(pred[0],z['B_30_fsnet']) and np.array_equal(pred[3],z['TFI_30_fsnet'])
            groups[tag]={arm:score(y,p) for arm,p in zip(arms,pred)}
            counts[tag]={'gated':int((abs(b)<=128).sum()),'changed':int((b!=own).sum())}
            for k,p in enumerate(pred):saved[f'{tag}_{k}']=p
            saved.update({tag+'_ids':ids,tag+'_y':y,tag+'_q':z['q_30'],tag+'_own_packet':own})
    def pool(key):return np.concatenate([saved[f'{tag}_{key}'] for tag in groups if tag!='pooled'])
    groups['pooled']={arm:score(pool('y'),pool(k)) for k,arm in enumerate(arms)}
    def delta(tag):
        a,b=[groups[tag][k] for k in arms[:2]]
        return (b['tp']-a['tp'],a['htbot_to_virut']-b['htbot_to_virut'])
    d=delta('pooled')
    expand=min(d)>=0 and max(d)>0 and min(delta('confirm'))>=0
    result=dict(training=meta,counts=counts,groups=groups,expand=bool(expand),
        hashes={k:sha(p) for k,p in {'panel':DEV/'panels.json.gz','fsnet':w,'head':OUT/'head.joblib','protocol':OUT/'protocol.txt'}.items()})
    (OUT/'dev_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    np.savez_compressed(OUT/'dev_predictions.npz',**saved)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':run()
