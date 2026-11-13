import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

P=Path(__file__).resolve().parent
BASE=P.parent/'d4_5090_visible_20261113'
IAT=BASE/'downstream_iat'
sys.path[:0]=[str(BASE),str(IAT)]
from score_dev import current_predictions, DEV, PREV, ds
from ablate import TimeReadout, features, scores, FULL, DATA
from logits import sha, load_inputs

RATES=(10,30,50)
ARMS=('base','direction','magnitude','joint','tfi','clean')
LABEL=json.loads((FULL/'six_label_audit.json').read_bytes())['mapping']


class TemplateBank:
    def __init__(self,rows):
        self.count=Counter(tuple(r['s']) for r in rows)
        self.groups={}
        for n in {len(k) for k in self.count}:
            keys=sorted(k for k in self.count if len(k)==n)
            self.groups[n]=(np.asarray(keys),np.array([self.count[k] for k in keys]))

    def vote(self, positions, visible, hidden, base):
        d=base.copy()
        m=base.copy()
        n=len(positions)+len(hidden)
        if n not in self.groups: return d,m,base.copy()
        seq,freq=self.groups[n]
        dist=np.mean(seq[:,positions]!=visible,axis=1)
        best=dist.min()
        if best>.15: return d,m,base.copy()
        cutoff=min(best+.10,np.partition(dist,min(7,len(dist)-1))[min(7,len(dist)-1)])
        keep=dist<=cutoff
        if freq[keep].sum()<3: return d,m,base.copy()
        weights=np.sqrt(freq[keep])*np.exp(-20*dist[keep])
        weights/=weights.sum()
        vals=seq[keep][:,hidden]
        pos=weights@(vals>0)
        d[pos>=.85]=abs(base[pos>=.85])
        d[pos<=.15]=-abs(base[pos<=.15])
        for i,v in enumerate(abs(vals).T):
            codes,inv=np.unique(v,return_inverse=True)
            votes=np.bincount(inv,weights=weights)
            if votes.max()>=.85: m[i]=np.sign(base[i])*codes[votes.argmax()]
        return d,m,np.sign(d)*abs(m)


def read_panels():
    assert sha(DEV/'panels.json.gz')==json.loads((DEV/'identity.json').read_bytes())['panels']
    panels=json.loads(gzip.decompress((DEV/'panels.json.gz').read_bytes()))
    ids=[r['id'] for panel in panels.values() for r in panel]
    assert len(set(ids))==len(ids)
    return panels


def dev_sources(tag,rows,short_head):
    with np.load(DEV/f'{tag}_pred.npz') as f:
        assert list(f['ids'])==[r['id'] for r in rows]
        for rate in RATES:
            qs=[np.array(r['masks'][str(rate)]) for r in rows]
            assert np.array_equal(np.concatenate(qs),f[f'q_{rate}'])
            seq=[np.asarray(r['s']) for r in rows]
            base=current_predictions(short_head,seq,qs,f[f'B_{rate}_packet'])
            splits=np.cumsum([len(q) for q in qs])[:-1]
            yield rate,qs,np.split(base,splits),np.split(f[f'TFI_{rate}_packet'],splits)


@torch.inference_mode()
def encode(net,kind,head,keys):
    logit=np.zeros((len(keys),6),np.float32)
    buckets={}
    for i,s in enumerate(keys): buckets.setdefault(max(32,1<<(len(s)-1).bit_length()),[]).append(i)
    for w,group in buckets.items():
        for j in range(0,len(group),24):
            ix=group[j:j+24]
            x=np.zeros((len(ix),w),np.float32)
            for row,i in enumerate(ix): x[row,:len(keys[i])]=keys[i]
            x=torch.from_numpy(x)
            z=net(x[:,None,:]) if kind in ds.NAMES[:3] else net(x/1600,torch.tensor([len(keys[i]) for i in ix]))
            logit[ix]=nn.functional.linear((z-head['mean'])/head['scale'],**head['state']).numpy()
    return logit


def build_cases(bank,rows,sources,arms):
    keys={}
    cases={}
    cache=[]
    for rate,qs,bs,ts in sources:
        ix={k:[] for k in arms}; timing=[]; packets={k:[] for k in arms}
        for r,q,b,t in zip(rows,qs,bs,ts,strict=True):
            s=np.array(r['s'],int); q=np.asarray(q,int); b=np.asarray(b,int)
            visible=np.ones(len(s),bool);visible[q]=False
            d,m,joint=bank.vote(np.flatnonzero(visible),s[visible],q,b)
            timing.append(features(r['t'],visible)[0])
            cand=dict(zip(ARMS,(b,d,m,joint,t,s[q]),strict=True))
            for k in arms:
                filled=s.copy(); v=cand[k];filled[q]=v
                key=tuple(filled)
                ix[k].append(keys.setdefault(key,len(keys)))
                packets[k].append([len(q),int(np.abs(abs(v)-abs(s[q])).sum()),int((np.sign(v)==np.sign(s[q])).sum()),int((v!=b).sum())])
        cases[rate]=(ix,np.array(timing),{k:np.array(v) for k,v in packets.items()})
    return list(keys),cases


@torch.inference_mode()
def evaluate(tag,bank,rows,sources,arms):
    keys,cases=build_cases(bank,rows,sources,arms)
    print(tag,'flows',len(rows),'unique classifier inputs',len(keys),flush=True)
    y=np.array([LABEL[r['id'].split('.pcap#')[0]] for r in rows])
    seen=np.array([tuple(r['s']) in bank.count for r in rows])
    weights=torch.load(IAT/'timing_heads.pt',weights_only=True,map_location='cpu')
    predictions={'ids':np.array([r['id'] for r in rows]),'y':y,'exact_train_template':seen}
    result={r:{} for r in RATES}
    for kind in ds.NAMES:
        net=ds.model_for(kind,8)
        if kind in ds.NAMES[:3]: net.fc=nn.Identity()
        elif kind=='sdae': net.head=nn.Identity()
        elif kind=='fsnet': net.head[-1]=nn.Identity()
        else: net.base.output_layer=nn.Identity()
        head=torch.load(BASE/f'six_{kind}.pt',map_location='cpu',weights_only=True)
        logits=encode(net,kind,head,keys)
        time=TimeReadout().eval()
        time.load_state_dict(weights['weights'][kind])
        for rate,(ix,x,packets) in cases.items():
            xx=np.nan_to_num((x-weights['mean'].numpy())/weights['std'].numpy(),nan=0.)
            bias=time(torch.from_numpy(xx.astype('float32'))).numpy()
            for arm in arms:
                p=(logits[ix[arm]]+bias).argmax(1)
                predictions[f'{kind}_{arm}{rate}']=p.astype('int8')
                slots,loss,corr,changed=map(int,packets[arm].sum(0))
                out=result[rate].setdefault(arm,dict(mae=loss/slots,direction=100*corr/slots,changed=changed,slots=slots,models={}))
                out['models'][kind]=scores(y,p)
        print(tag,kind,flush=True)
    for results in result.values():
        for row in results.values():
            for field in ('acc','f1'): row[field]=float(np.mean([m[field] for m in row['models'].values()]))
    np.savez_compressed(P/f'{tag}_predictions.npz',**predictions)
    (P/f'{tag}.json').write_text(json.dumps(dict(n=len(y),seen=int(seen.sum()),rates=result),indent=1))
    return result


def passes(res):
    def delta(field):return np.mean([res[r]['joint'][field]-res[r]['base'][field] for r in RATES])
    return delta('acc')>=0 and delta('f1')>0 and delta('mae')<=0


def test_inputs():
    data=load_inputs()
    timing=json.loads(gzip.decompress((FULL/'verified.json.gz').read_bytes()))
    rows=[dict(id=r['id'],s=r['signed'],t=timing[r['id']][1]) for r in map(json.loads,(DATA/'flows.jsonl').open())]
    def source():
        for rate in RATES:
            with np.load(DATA/f'mask{rate}.npz') as f:
                assert list(f['ids'])==[r['id'] for r in rows]
                qs=[np.flatnonzero(mask) for mask in f['mask']]
            dual=data['dual'+str(rate)][0]; tfi=data['tfi'+str(rate)][0]
            yield rate,qs,[s[q] for s,q in zip(dual,qs)],[s[q] for s,q in zip(tfi,qs)]
    return rows,source()


def main():
    torch.set_num_threads(4)
    panels=read_panels()
    bank=TemplateBank(panels['train'])
    short=joblib.load(PREV/'own_short_v1/head.joblib')
    passed=[]
    sources=[P/'protocol.txt',Path(__file__),DEV/'panels.json.gz',DEV/'screen_pred.npz',DEV/'confirm_pred.npz',
             IAT/'timing_heads.pt',PREV/'own_short_v1/head.joblib',*[BASE/f'six_{k}.pt' for k in ds.NAMES]]
    (P/'identity.json').write_text(json.dumps({str(s):sha(s) for s in sources},indent=1))
    for tag in ('screen','confirm'):
        rows=panels[tag]
        res=evaluate(tag,bank,rows,dev_sources(tag,rows,short),ARMS)
        passed.append(passes(res));print(tag,'gate',passed[-1],flush=True)
    if all(passed):
        rows,src=test_inputs()
        assert {r['id'] for r in rows}.isdisjoint({r['id'] for g in panels.values() for r in g})
        evaluate('full',bank,rows,src,('base','joint','tfi','clean'))
    print('done',passed,flush=True)


if __name__=='__main__': main()
