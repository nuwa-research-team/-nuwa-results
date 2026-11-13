import gzip
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'repaired'))
import models
import models_repaired as mr
from train_repaired import SDAERepaired
MODELS=('cnn','lstm','gru','tst','sdae','fsnet')

def backbone(k):
    net=(models.build(k,8) if k in MODELS[:3] else SDAERepaired(8) if k=='sdae' else mr.build(k,8))
    net.load_state_dict(torch.load(P/(k+'.pt'),weights_only=True,map_location='cpu')['state'])
    if k in MODELS[:3]: net.fc=nn.Identity()
    elif k=='fsnet': net.head[-1]=nn.Identity()
    elif k=='sdae': net.head=nn.Identity()
    else: net.base.output_layer=nn.Identity()
    return net.cuda().eval()

@torch.inference_mode()
def encode(net,k,keys):
    buckets={}
    for i,key in enumerate(keys):
        n=len(key)//2
        width=max(32,1<<(n-1).bit_length())
        buckets.setdefault(width,[]).append(i)
    for width,inds in buckets.items():
        for start in range(0,len(inds),48):
            ix=inds[start:start+48]
            rows=[np.frombuffer(keys[i],dtype=np.int16) for i in ix]
            pad=np.zeros((len(ix),width),dtype=np.float32)
            for j,s in enumerate(rows):pad[j,:len(s)]=s
            x=torch.from_numpy(pad).cuda()
            z=(net(x[:,None,:]) if k in MODELS[:3] else net(x/1600,torch.tensor(list(map(len,rows)),device='cuda')))
            yield ix,z

def score(y,p):
    cm=np.bincount(6*np.array(y)+p,minlength=36).reshape(6,6)
    f=2*np.diag(cm)/np.maximum(1,cm.sum(0)+cm.sum(1))
    return [100*np.trace(cm)/cm.sum(),100*f.mean(),100*np.average(f,weights=cm.sum(1))]

def main():
    with gzip.open(P/'six_data.pkl.gz','rb') as f:data=pickle.load(f)
    keys=[];index={};ixs={}; ys={}
    for name,(rows,y) in data.items():
        inds=[]
        for row in rows:
            key=np.asarray(row,dtype=np.int16).tobytes()
            if key not in index:index[key]=len(keys);keys.append(key)
            inds.append(index[key])
        ixs[name]=np.array(inds);ys[name]=np.asarray(y)
    ndev=max(ixs['screen'].max(),ixs['train'].max())+1
    results={};preds={}
    for k in MODELS:
        net=backbone(k)
        reps=None
        for ix,z in encode(net,k,keys[:ndev]):
            if reps is None:reps=torch.zeros(ndev,z.shape[1],device='cuda')
            reps[ix]=z
        tr=reps[ixs['train']];va=reps[ixs['screen']]
        mean=tr.mean(0);scale=tr.std(0).clamp_min(.01)
        tr=(tr-mean)/scale;va=(va-mean)/scale
        yt=torch.as_tensor(ys['train'],device='cuda');yv=torch.as_tensor(ys['screen'],device='cuda')
        torch.manual_seed(42)
        head=nn.Linear(tr.shape[1],6).cuda()
        opt=torch.optim.AdamW(head.parameters(),lr=.001)
        best=float('inf')
        for ep in range(20):
            for idx in torch.randperm(len(tr),device='cuda').split(128):
                loss=nn.functional.cross_entropy(head(tr[idx]),yt[idx])
                opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():val=nn.functional.cross_entropy(head(va),yv).item()
            if val<best:
                best=val;epoch=ep+1;state={n:t.detach().clone() for n,t in head.state_dict().items()}
        head.load_state_dict(state)
        torch.save(dict(state=state,mean=mean,scale=scale,epoch=epoch),P/('six_'+k+'.pt'))
        p=np.empty(len(keys),dtype=np.int64)
        with torch.inference_mode():
            for ix,z in encode(net,k,keys):p[ix]=head((z-mean)/scale).argmax(1).cpu().numpy()
        results[k]={name:score(ys[name],p[ix]) for name,ix in ixs.items()}
        for name,ix in ixs.items():preds[k+'_'+name]=p[ix]
        print(k,epoch,results[k],flush=True)
    np.savez_compressed(P/'six_predictions.npz',**preds,**{'y_'+k:y for k,y in ys.items()})
    (P/'six_scores.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
