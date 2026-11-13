"""Freeze train and two unused development panels. No test sequences are used."""
import gzip, json, hashlib, pickle, tarfile
from pathlib import Path
from collections import Counter
import numpy as np
p=Path(__file__).resolve().parent
root=p.parents[2]; outputs=root/'outputs'
data=root/'nuwa_reproduction_20261109/audit/paired_snapshot_20261110/work/d4_aligned/raw.pkl'
assert hashlib.sha256(data.read_bytes()).hexdigest()=='312617767197cad4c931dbc57f8fdaead8185c13e6e9e10b07458ce3ffabb3d6'
d=pickle.loads(data.read_bytes())
exact=json.loads(gzip.decompress((p/'dev_exact.json.gz').read_bytes()))
prior=set()
for k in (1,2,3):
    m=json.loads((outputs/f'd4_fast_dev_20261113/round{k}/manifest.json').read_bytes())
    prior.update(row['id'] for phase in ('screen','confirm') for row in m[phase])
for panel in json.loads((outputs/'d4_iat_20261113/cohorts.json').read_bytes()).values():prior.update(panel['validation'])
with tarfile.open(outputs/'d4_constrained_u1000_20261112/cloud_input.tar') as tar:
    for line in tar.extractfile('data/windows.jsonl'):
        w=json.loads(line)
        c,n=w['flow_id'].rsplit(':',1);prior.add(Path(c).name+'#'+n)
def rank(i):
    return hashlib.sha256(('nov13-query-readout:'+str(d['ids'][i])).encode()).digest()
def take(split,n,skip,balanced=False):
    pool=[int(i) for i in d['split'][split] if str(d['ids'][i]) in exact and str(d['ids'][i]) not in skip]
    pool.sort(key=rank)
    groups=[[i for i in pool if int(d['y'][i])==j] for j in range(8)]
    quota=np.array([min(len(g)//(2 if split=='val' and balanced else 1), 6 if split=='val' else 96) for g in groups])
    while sum(quota)<n:
        score=np.array([len(g)/max(1,len(pool))*n-q for g,q in zip(groups,quota)])
        for j,g in enumerate(groups):
            if quota[j]>=len(g)//(2 if split=='val' and balanced else 1):score[j]=-1e20
        assert max(score)>-1e19
        quota[np.argmax(score)]+=1
    vals=sorted([i for g,q in zip(groups,quota) for i in g[:q]],key=rank)
    return vals
train=take('train',4096,prior);screen=take('val',384,prior,True)
confirm=take('val',512,prior|{str(d['ids'][i]) for i in screen})
panels={}
for tag,rows in [('train',train),('screen',screen),('confirm',confirm)]:
    panels[tag]=[]
    for idx,i in enumerate(rows):
        key=str(d['ids'][i]);f,t=exact[key];s=list(map(int,d['seqs'][i]));n=len(s)
        assert f in (-1,1) and len(t)==n and 20<=n<=512
        rates=([10,30,50][idx%3],) if tag=='train' else (10,30,50)
        masks={r:sorted(np.random.default_rng(42+i).choice(n,max(1,round(n*r/100)),replace=False).tolist()) for r in rates}
        panels[tag].append(dict(id=key,i=i,y=int(d['y'][i]),s=s,t=t,f=f,masks=masks))
    print(tag,len(rows),Counter(d['labels'][int(d['y'][i])] for i in rows))
ids=[r['id'] for rr in panels.values() for r in rr];assert len(set(ids))==len(ids)
assert not set(ids)&{str(d['ids'][int(i)]) for i in d['split']['test']}
with gzip.open(p/'panels.json.gz','wt') as out:json.dump(panels,out,separators=(',',':'))
