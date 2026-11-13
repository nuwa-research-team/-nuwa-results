import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

BASE = Path(__file__).resolve().parent
ARMS = ('local', 'scales', 'timed')
RATES = (10, 30, 50)

def features(s, t, q):
    s, t, q = np.asarray(s), np.asarray(t, dtype=float), np.asarray(q, dtype=int)
    v = np.setdiff1d(np.arange(len(s)), q)
    assert len(v)
    sv, tv = s[v], t[v]
    mid = np.searchsorted(v, q)
    col = [q / (len(s)-1), np.full(len(q),len(s)/512), np.full(len(q),len(q)/len(s))]
    timing = []
    for delta in (-3,-2,-1,0,1,2):
        p = mid + delta
        valid = (p>=0) & (p<len(v))
        ix = p.clip(0,len(v)-1)
        col.extend([np.where(valid,sv[ix]/1600,0),np.where(valid,(v[ix]-q)/64,0),valid])
    local_end = len(col)
    for span in (8,32,512):
        lo, hi = np.searchsorted(v,q-span), np.searchsorted(v,q+span,side='right')
        count = hi-lo
        col.append(count/np.minimum(len(s),2*span+1))
        for x in (sv>0, abs(sv)<=128, abs(sv)>=1400, sv/1600):
            acc = np.r_[0,np.cumsum(x)]
            col.append((acc[hi]-acc[lo])/np.maximum(1,count))
    scales_end = len(col)
    for a,b in ((-2,-1),(-1,0),(0,1),(-3,2)):
        left, right = mid+a, mid+b
        valid = (left>=0)&(right<len(v))
        elapsed = tv[right.clip(0,len(v)-1)]-tv[left.clip(0,len(v)-1)]
        timing.extend([np.where(valid,np.log1p(np.maximum(0,elapsed))/16,0),valid])
    return np.column_stack(col+timing).astype('float32'), local_end, scales_end

def design(rows, rate=None):
    xs, ys = [], []
    for r in rows:
        key = str(rate) if rate else next(iter(r['masks']))
        q = r['masks'][key]
        x,a,b = features(r['s'],r['t'],q)
        xs.append(x)
        ys.append(np.asarray(r['s'])[q])
    return np.concatenate(xs),np.concatenate(ys),a,b

def main():
    panel=json.loads(gzip.decompress((BASE/'panels.json.gz').read_bytes()))
    ids=[r['id'] for rs in panel.values() for r in rs]
    assert len(set(ids))==len(ids)
    x,y,a,b=design(panel['train'])
    vocab=[v for v,n in Counter(y.tolist()).most_common(128)]
    mapping={v:i for i,v in enumerate(vocab)}
    dev={(split,rate):design(panel[split],rate)[0] for split in ('screen','confirm') for rate in RATES}
    device='cuda'
    yt=torch.tensor([mapping.get(v,len(vocab)) for v in y],device=device)
    outs={}
    for arm,width in zip(ARMS,(a,b,x.shape[1])):
        torch.manual_seed(42)
        net=nn.Sequential(nn.Linear(x.shape[1],192),nn.LayerNorm(192),nn.GELU(),nn.Linear(192,192),nn.GELU(),nn.Linear(192,len(vocab)+1)).to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=.002)
        xt=torch.tensor(x,device=device)
        xt[:,width:]=0
        for epoch in range(20):
            for ix in torch.randperm(len(x),device=device).split(512):
                loss=nn.functional.cross_entropy(net(xt[ix]),yt[ix])
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        torch.save(dict(weights=net.state_dict(),width=width,vocab=vocab,dim=x.shape[1]),BASE/(arm+'.pt'))
        with torch.inference_mode():
            for (split,r),xd in dev.items():
                tx=torch.tensor(xd,device=device)
                tx[:,width:]=0
                p,ix=torch.softmax(net(tx),1).max(1)
                vals=np.asarray(vocab+[0])[ix.cpu().numpy()]
                outs[f'{arm}_{split}_{r}']=np.where(p.cpu().numpy()>=.95,vals,0)
        print(arm,len(y),float(loss.detach()),flush=True)
    np.savez_compressed(BASE/'override.npz',**outs)
    print(torch.__version__,torch.cuda.get_device_name(),flush=True)

if __name__=='__main__':main()
