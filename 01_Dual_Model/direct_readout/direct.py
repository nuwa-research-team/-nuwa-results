"""Evaluate the previously fitted direct B head. No fitting and no TFI inputs."""
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

P = Path(__file__).resolve().parent
R = P.parent
D = R / 'outputs/d4_downstream_random'
sys.path[:0] = [str(D/'dual4b'), str(D), str(R)]
from previous import dualfeat
from evaluate import TrafficFrontend, collate_features, digest

start = time.monotonic()
torch.set_num_threads(2)
identity = json.loads((R/'dev_readout/identity.json').read_bytes())
ckpt = R/'runs/formal_a_full1882_s17_v1/ckpt_u1000.pt'
assert digest(ckpt) == identity['checkpoint']
ck = torch.load(ckpt, map_location='cpu', weights_only=False)
assert ck['stage']=='a' and ck['update']==1000
front = TrafficFrontend(2560)
front.load_state_dict(ck['frontend'])
del ck
front.cuda().eval()
vocab = identity['vocab']
head = nn.Module()
head.local = nn.Sequential(nn.LayerNorm(522), nn.Linear(522,128), nn.GELU())
head.output = nn.Linear(128,len(vocab)+5)
head.load_state_dict(torch.load(R/'dev_readout/B.pt', map_location='cpu', weights_only=True))
head.cuda().eval()

@torch.inference_mode()
def infer(rows, rate):
    tasks=[]
    packed=[{} for _ in rows]
    for i,r in enumerate(rows):
        n=len(r['s'])
        cuts=[0,n] if n<=256 else [0,n//2,n]
        for a,b in zip(cuts,cuts[1:]):
            q=[j-a for j in r['masks'][str(rate)] if a<=j<b]
            if q:tasks.append((i,a,b,q))
    tasks.sort(key=lambda t:t[2]-t[1])
    for lo in range(0,len(tasks),32):
        batch=tasks[lo:lo+32]
        feats=[]; xs=[]; candidates=[]
        for i,a,b,q in batch:
            r=rows[i];s=np.asarray(r['s'][a:b])*r['f']
            feats.append(dualfeat(s.tolist(),r['t'][a:b],q)[1])
        inputs=collate_features(feats,device='cuda')
        z=front(**inputs)
        for j,(i,a,b,q) in enumerate(batch):
            r=rows[i];s=np.array(r['s'][a:b])*r['f'];n=len(s)
            idx=torch.tensor(q,device='cuda');ranges=inputs['ranges'][j]
            cover=((idx[:,None]>=ranges[:,0])&(idx[:,None]<ranges[:,1])).float()
            beh=(cover/cover.sum(1,keepdim=True).clamp_min(1))@z['behavior_hidden'][j]
            visible=np.flatnonzero(~np.isin(np.arange(n),q))
            extra=[]; vals=[]
            for pos in q:
                k=np.searchsorted(visible,pos)
                neighbor=[k-1,k,k-2,k+1]
                at=[int(visible[t]) if 0<=t<len(visible) else None for t in neighbor]
                lengths=[int(s[t]) if t is not None else 0 for t in at]
                extra.append([(t-pos)/max(1,n) if t is not None else 0 for t in at]+[x/1600 for x in lengths]+[pos/max(1,n-1),n/512])
                row=vocab+lengths+[0]
                vals.append([x if x and x not in row[:k] else 0 for k,x in enumerate(row)])
            xs.append(torch.cat([z['packet_hidden'][j,idx],beh,torch.tensor(extra,device='cuda')],dim=1))
            candidates.extend(vals)
        cand=torch.tensor(candidates,device='cuda')
        scores=head.output(head.local(torch.cat(xs))).masked_fill(cand==0,-1e6)
        selected=cand.gather(1,scores.argmax(1)[:,None]).flatten().tolist()
        offset=0
        for i,a,b,q in batch:
            for pos,v in zip(q,selected[offset:offset+len(q)],strict=True):packed[i][a+pos]=v*rows[i]['f']
            offset+=len(q)
        assert offset==len(selected)
        if lo and lo%2048==0:print(rate,lo,len(tasks),round(time.monotonic()-start),flush=True)
    return [[bypos[j] for j in r['masks'][str(rate)]] for bypos,r in zip(packed,rows,strict=True)]

panels=json.loads(gzip.decompress((R/'dev_readout/panels.json.gz').read_bytes()))
reference=np.load(R/'dev_readout/confirm_pred.npz')
for rate in (10,30,50):
    pred=infer(panels['confirm'],rate)
    assert np.array_equal(reference[f'B_{rate}_packet'],np.concatenate(pred)),('confirm replay',rate)
print('All confirmation packet predictions match',flush=True)
flows=[json.loads(l) for l in (D/'flows.jsonl').open()]
orients=json.loads(gzip.decompress((R/'directions.json.gz').read_bytes()))
ids=[r['id'] for r in flows]; assert len(set(ids))==13150 and set(ids)==set(orients)
assert not set(ids)&{r['id'] for r in panels['train']}
rows=[dict(s=r['signed'],f=orients[r['id']][0],t=orients[r['id']][1],masks={}) for r in flows]
for rate in (10,30,50):
    path=D/f'mask{rate}.npz'
    z=np.load(path);assert np.array_equal(z['ids'].astype(str),ids)
    for row,m in zip(rows,z['mask'],strict=True):
        q=np.flatnonzero(m).tolist();assert q and q[-1]<len(row['s'])
        row['masks'][str(rate)]=q
    pred=infer(rows,rate)
    with (P/f'direct_{rate}.npz').open('wb') as file:
        np.savez_compressed(file,ids=ids,nq=[len(x) for x in pred],q=np.concatenate([r['masks'][str(rate)] for r in rows]),pred=np.concatenate(pred),mask_sha=digest(path))
    print('done',rate,round(time.monotonic()-start),flush=True)
(P/'done.json').write_text(json.dumps(dict(checkpoint=digest(ckpt),head=digest(R/'dev_readout/B.pt'),confirmed=512,rates=[10,30,50],seconds=time.monotonic()-start)))
