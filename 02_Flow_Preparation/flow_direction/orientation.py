import json,pickle,gzip,hashlib,subprocess,collections,re,concurrent.futures
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
old=P.parent/'d4_behavior_iat_20261113'
seq=P.parents[1]/'nuwa_reproduction_20261109/data/processed/USTC-TFC2016/sequences'
db=pickle.loads((P.parent/'d4_iat_20261113/aligned.pkl').read_bytes())
sets={k:json.loads(gzip.decompress((old/(k+'.json.gz')).read_bytes())) for k in ('pilot','expansion')}
panel={r['id']:r for rr in sets.values() for r in rr}
assert len(panel)==sum(map(len,sets.values()))
train=json.loads((P/'train_check.json').read_bytes())
def digest(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def scan(p):
 m=json.loads(p.read_bytes());src=Path(m['source']);name=src.name
 assert digest(src)==m['source_sha256']
 keys={k.split('#')[1]:k for k in db if k.startswith(name+'#')}
 wins=collections.defaultdict(list)
 for r in train:
  cap,stream=r['flow_id'].rsplit(':',1)
  if Path(cap).name==name:wins[stream].append(r)
 fields='tcp.stream ip.src ip.dst tcp.srcport tcp.dstport ip.len frame.time_epoch tcp.analysis.retransmission ip.flags.mf ip.frag_offset frame.number'.split()
 cmd=[str(Path.home()/'Desktop/Wireshark/tshark.exe'),'-n','-2','-r',str(src),'-Y','tcp && ip','-T','fields','-E','occurrence=f']
 for x in fields:cmd+=['-e',x]
 proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,text=True,encoding='ascii')
 factors={};checked=collections.defaultdict(list); ww=collections.defaultdict(list)
 for l in proc.stdout:
  st,a,b,ap,bp,n,t,ret,mf,frag,frame=l.rstrip('\r\n').split('\t')
  if st not in keys and st not in wins:continue
  if not n:continue
  A=(a,int(ap or 0));B=(b,int(bp or 0));assert A!=B
  sign=1 if A<B else -1
  if st in wins and mf.lower() in ('0','','false') and frag in ('0','') and 40<=int(n)<=65535:
   ww[st].append((float(t),int(frame),sign*int(n)))
  if ret or st not in keys:continue
  k=keys[st];f=factors.setdefault(k,sign)
  if k in panel:checked[k].append((f*sign*int(n),float(t),sign))
 assert proc.wait()==0 and len(factors)==len(keys)
 for k,v in checked.items():
  r=panel[k];v=v[:512];assert [x[0] for x in v]==r['s'],k
  us=np.rint(np.array([x[1] for x in v])*1e6).astype('int64');assert (us-us[0]).tolist()==r['t']
  for q in r['q'].values():
   visible=[j for j in range(len(v)) if j not in q];assert visible and {int(np.sign(r['s'][j]))*v[j][2] for j in visible}=={factors[k]}
 assert len(checked)==sum(k.startswith(name+'#') for k in panel)
 matched=total=0
 for st,rs in wins.items():
  vv=sorted(ww[st]); signed=[x[2] for x in vv]
  for r in rs:
   lo,hi=map(int,r['sample_id'].split('#w')[1].split('-'));total+=1;matched+=signed[lo:hi]==r['signed_lengths']
 print(name,'flows',len(factors),'panel',len(checked),'checkpoint windows',matched,total,flush=True)
 assert matched==total, (name,matched,total)
 return factors,dict(hash=m['source_sha256'],count=len(factors),panel=len(checked),matched=matched,total=total)
if __name__=='__main__':
 pp=sorted(p for p in seq.glob('*.audit.json') if p.name.startswith(('SMB','Weibo')))
 with concurrent.futures.ThreadPoolExecutor(2) as pool:res=list(pool.map(scan,pp))
 factors={k:v for d,_ in res for k,v in d.items()};assert len(factors)==len(db)
 counts={}
 for tag,ids in [('eligible',list(db)),*[(k,[r['id'] for r in v]) for k,v in sets.items()]]:
  counts[tag]={c:[sum(factors[k]==-1 for k in ids if k.startswith(c)),sum(k.startswith(c) for k in ids)] for c in ('SMB','Weibo','')}
 (P/'orientation.json').write_text(json.dumps({k:factors[k] for k in panel}))
 (P/'audit_capture.json').write_text(json.dumps(dict(counts=counts,captures={p.name:m for p,(_,m) in zip(pp,res)}),indent=2))
 print(counts,flush=True)


