import os,gzip,json,sys,time,pickle,copy,hashlib,collections
from pathlib import Path
import numpy as np,torch
P=Path(__file__).resolve().parent; R=P.parent
O=R/'behavior_iat_13'
D=R/'outputs/d4_downstream_random'; sys.path[:0]=[str(D/'dual4b'),str(D)]
import evaluate as ev, downstream as ds
from eval_packing import parse_p1,pack_single
from constrained import ArrayGrammar
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoModelForMaskedLM,PreTrainedTokenizerFast,LogitsProcessor
from peft import get_peft_model,LoraConfig
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def out(p,j):(P/p).write_text(json.dumps(j,ensure_ascii=False,indent=1))
def loadj(p):return json.loads((P/p).read_text(encoding='utf-8-sig'))
def line(p,j):
 with (P/p).open('a') as f: print(json.dumps(j,separators=(',',':')),file=f)
class Processor(LogitsProcessor):
 def __init__(self,tok,counts):
  self.gram=[ArrayGrammar(tok,n) for n in counts]; self.eos=tok.eos_token_id
 def __call__(self,input_ids,scores):
  cpu=input_ids.cpu(); ids=[]; rr=[]
  for i,(x,g) in enumerate(zip(cpu,self.gram)):
   a=[self.eos] if self.eos in x.tolist() else g.allowed(i,x);ids.extend(a);rr.extend([i]*len(a))
  r=torch.tensor(rr,device=scores.device);t=torch.tensor(ids,device=scores.device)
  v=scores[r,t];scores.fill_(-float('inf'));scores[r,t]=v;return scores
def dualfeat(s,t,q):
 base=ev.features(s,[None]*len(s),set(q)); timed=ev.features(s,t,set(q)); other=copy.deepcopy(base)
 for field in ('stats','stat_valid'):
  for a,b in zip(other[field],timed[field]):a[7:9]=b[7:9]
 return [base,other]
@torch.inference_mode()
def main(stage):
 factor=loadj('orientation.json'); rows=json.loads(gzip.decompress((O/(stage+'.json.gz')).read_bytes())); ids=[r['id'] for r in rows]; y=[r['y'] for r in rows]
 assert sha(O/(stage+'.json.gz'))==loadj('protocol.json')['panel_hashes'][stage]
 raw=pickle.loads(ds.RAW.read_bytes()); assert sha(ds.RAW)=='312617767197cad4c931dbc57f8fdaead8185c13e6e9e10b07458ce3ffabb3d6'
 for r in rows:
  i=r['i'];assert str(raw['ids'][i])==r['id'] and int(raw['y'][i])==r['y'] and list(raw['seqs'][i])==r['s'] and raw['labels'][r['y']] in ('SMB','Weibo')
  for rate in [10,30,50]:assert sorted(np.random.default_rng(42+i).choice(len(r['s']),max(1,round(len(r['s'])*rate/100)),replace=False).tolist())==r['q'][str(rate)]
 del raw
 mdir=R/'model'; path=R/'runs/formal_a_full1882_s17_v1/ckpt_u1000.pt'
 assert sha(path)=='2bcd21e4534bf8fa79f9f8ad30630d2178ab4c2fafab9cedc977f139c20623f8'
 ck=torch.load(path,map_location='cpu',weights_only=False)
 for field,path2 in [('data_windows_sha256',R/'data/windows.jsonl'),('model_config_sha256',mdir/'config.json'),('tokenizer_sha256',mdir/'tokenizer.json')]:assert ck['meta'][field]==sha(path2)
 w=ev.read(R/'data/windows.jsonl');ex={}
 for split in ['train','val']:
  kk={Path(a['flow_id'].rsplit(':',1)[0]).name+'#'+a['flow_id'].rsplit(':',1)[1] for a in w if a['split']==split};ex[split]=sorted(set(ids)&kk)
 out(stage+'_identity.json',dict(checkpoint=sha(path),stage=ck['stage'],step=ck['update'],flow_overlap=ex,n=len(ids),code={p.name:sha(p) for p in [Path(__file__),D/'dual4b/evaluate.py',D/'dual4b/eval_frontend.py',D/'dual4b/constrained.py',D/'downstream.py']},tfi=sha(R/'tfi_model/pytorch_model.bin'),classifiers={k:sha(ds.SNAP/f'work/res_aligned_{k}/{k}.pt' if k in ds.NAMES[:3] else ds.REPAIRED/f'repaired/repaired_{k}.pt') for k in ds.NAMES}))
 print(stage,'attested',len(ids),'overlap',[len(v) for v in ex.values()],flush=True)
 tok=AutoTokenizer.from_pretrained(mdir,local_files_only=True); model=AutoModelForCausalLM.from_pretrained(mdir,torch_dtype=torch.bfloat16,local_files_only=True)
 model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',target_modules=['q_proj','k_proj','v_proj','o_proj']))
 assert set(ck['lora'])=={k for k in model.state_dict() if 'lora_' in k};model.load_state_dict(ck['lora'],strict=False)
 front=ev.TrafficFrontend(2560);front.load_state_dict(ck['frontend']);front.cuda().eval();model.cuda().eval();del ck
 start=time.monotonic()
 for rate in [10,30,50]:
  name=f'{stage}_{rate}_pairs.jsonl'; existing={v['id'] for v in ev.read(P/name)} if (P/name).exists() else set()
  pending=sorted([r for r in rows if r['id'] not in existing],key=lambda r:(len(r['q'][str(rate)]),len(r['s'])))
  for startrow in range(0,len(pending),32):
   batch=pending[startrow:startrow+32]; embeds=[];counts=[]
   for r in batch:
    s=r['s'];n=len(s);q=r['q'][str(rate)]; f=factor[r['id']]; assert f in (-1,1); ss1=[f*x for x in s]; fs=[dualfeat(u,r['t'],q)[1] for u in (s,ss1)]; assert n<=256
    if startrow==0:
     dirty=list(r['t']);ss=list(s)
     for pos in q:dirty[pos]=-1234567;ss[pos]=0
     assert [dualfeat(u,dirty,q)[1] for u in (ss,[f*x for x in ss])]==fs
    z=front(**ev.collate_features(fs,device='cuda'))
    for j in range(2):
     pack=pack_single(tok,model.get_input_embeddings(),z['packet_embeds'][j],z['behavior_embeds'][j],n,q,canonical=True)
     embeds.append(pack['inputs_embeds'][0]);counts.append(len(q))
   width=max(map(len,embeds)); E=embeds[0].new_zeros((len(embeds),width,2560)); A=torch.zeros(E.shape[:2],device='cuda',dtype=torch.long)
   for i,e in enumerate(embeds):E[i,-len(e):]=e;A[i,-len(e):]=1
   gen=model.generate(input_ids=torch.empty((len(embeds),0),device='cuda',dtype=torch.long),inputs_embeds=E,attention_mask=A,logits_processor=[Processor(tok,counts)],do_sample=False,num_beams=1,max_new_tokens=7*max(counts)+2,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,use_cache=True).tolist()
   vals=[]
   for g,k in zip(gen,counts):
    end=g.index(tok.eos_token_id); res=parse_p1(tok.decode(g[:end],clean_up_tokenization_spaces=False),k);assert res['status']=='valid'; vals.append(res['parsed_values'])
   for i,r in enumerate(batch):line(name,dict(id=r['id'],prompt=vals[2*i],aligned=[factor[r['id']]*x for x in vals[2*i+1]]))
   print(stage,rate,len(existing)+startrow+len(batch),'/',len(rows),'s',round(time.monotonic()-start),flush=True)
 del model,front;torch.cuda.empty_cache()
 summary=[];y=np.array(y)
 for rate in [10,30,50]:
  seq={};vals={v['id']:v for v in ev.read(P/f'{stage}_{rate}_pairs.jsonl')}; assert set(vals)==set(ids)
  for arm in ['prompt','aligned']:
   seq[arm]=[]
   for r in rows:
    v=np.array(r['s']); v[r['q'][str(rate)]]=vals[r['id']][arm];seq[arm].append(v.tolist())
  predfile={};ix=np.arange(len(rows))
  for name in ds.NAMES:
   clf=ds.model_for(name,8).eval()
   for arm,seqs in seq.items():
    pred=ds.pv.classify(clf,seqs,ix,'cpu',512) if name in ds.NAMES[:3] else ds.er.classify(clf,seqs,ix,'cpu',16 if name=='tst' else 48,1/1600)
    pp=np.array(pred).astype(int);predfile[name+'_'+arm]=pp.tolist();summary.append(dict(rate=rate,model=name,arm=arm,**ds.summarize(y,pp),other=int((~np.isin(pp,[4,6])).sum()),smb_weibo=int(((y==4)&(pp==6)).sum()),weibo_smb=int(((y==6)&(pp==4)).sum())))
  out(f'{stage}_{rate}_pred.json',dict(ids=ids,y=y.tolist(),pred=predfile))
 out(stage+'_scores.json',summary)
 print(stage,'COMPLETE',round(time.monotonic()-start),flush=True)
if __name__=='__main__':
 torch.set_num_threads(3)
 main('pilot');main('expansion')
