import gzip
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from heads import BASE, ARMS, RATES

O=BASE.parent
PREV=O/'d4_numeric_iat_20261113'
DEV=O/'d4_full_aligned_20261113/dev_readout'
sys.path[:0]=[str(PREV/'short_exit_replay'),str(PREV/'context_candidates_50/confirm_signed_rule')]
from replay import variants, ds
from validate import apply_rule

def current_predictions(head,seqs,qs,b):
    current=variants(head,seqs,qs,b)['both']
    start=0
    for seq,q in zip(seqs,qs,strict=True):
        v=np.setdiff1d(np.arange(len(seq)),q)
        sl=slice(start,start+len(q))
        current[sl]=apply_rule(v,seq[v],q,current[sl])[0]
        start+=len(q)
    return current

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    panels=json.loads(gzip.decompress((DEV/'panels.json.gz').read_bytes()))
    assert hashlib.sha256((DEV/'panels.json.gz').read_bytes()).hexdigest()==json.loads((DEV/'identity.json').read_bytes())['panels']
    head=joblib.load(PREV/'own_short_v1/head.joblib')
    models={k:ds.model_for(k,8) for k in ds.NAMES}
    output={}
    with np.load(BASE/'override.npz') as ov:
        for tag in ['screen','confirm']:
            rows=panels[tag]
            seqs=[np.asarray(r['s']) for r in rows]
            y=[r['y'] for r in rows]
            idx=np.arange(len(rows))
            output[tag]={}
            with np.load(DEV/f'{tag}_pred.npz') as z:
                assert list(z['ids'])==[r['id'] for r in rows] and np.array_equal(y,z['y'])
                for r in RATES:
                    out={}
                    qs=[np.array(v['masks'][str(r)]) for v in rows]
                    assert np.array_equal(z[f'q_{r}'],np.concatenate(qs)) and np.array_equal(z[f'nq_{r}'],list(map(len,qs)))
                    c=current_predictions(head,seqs,qs,z[f'B_{r}_packet'])
                    truth=np.concatenate([s[q] for s,q in zip(seqs,qs,strict=True)]).astype(float)
                    for arm in ('current',*ARMS):
                        pred=c
                        if arm!='current':
                            o=ov[f'{arm}_{tag}_{r}']
                            pred=np.where(o!=0,o,c)
                        filled=[seq.copy() for seq in seqs]
                        for seq,q,part in zip(filled,qs,np.split(pred,np.cumsum(list(map(len,qs)))[:-1]),strict=True):
                            seq[q]=part
                        scores={}
                        for k,model in models.items():
                            cl=(ds.pv.classify(model,filled,idx,'cpu',512) if k in ds.NAMES[:3]
                                else ds.er.classify(model,filled,idx,'cpu',16 if k=='tst' else 48,1/1600))
                            metric=ds.summarize(y,cl)
                            scores[k]=[metric['accuracy'],metric['macro']]
                        out[arm]=dict(mean=np.mean(list(scores.values()),axis=0).tolist(),scores=scores,
                             mae=float(np.abs(abs(pred)-abs(truth)).mean()),changed=int((pred!=c).sum()))
                    output[tag][r]=out
                    print(tag,r,{k:v['mean'] for k,v in out.items()},flush=True)
    av={}
    for tag,rr in output.items():
        av[tag]={arm:np.mean([[*rr[r][arm]['mean'],rr[r][arm]['mae']] for r in RATES],0).tolist() for arm in ('current',*ARMS)}
    def improves(arm,tag):
        acc,f1,mae=np.subtract(av[tag][arm],av[tag]['current'])
        return acc>=0 and f1>0 and mae<=0
    eligible=sorted([a for a in ARMS if improves(a,'screen')],key=lambda a:-av['screen'][a][1])
    winner=eligible[0] if eligible and improves(eligible[0],'confirm') else None
    (BASE/'dev_scores.json').write_text(json.dumps(dict(results=output,average=av,screen_order=eligible,selected=winner),indent=2))
    print(av,winner,flush=True)
if __name__=='__main__':main()
