"""Paired development experiment for a signed-packet-length readout."""
import gzip, json, time, sys, collections, hashlib, pickle
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM, PreTrainedTokenizerFast
from peft import get_peft_model, LoraConfig
P=Path(__file__).resolve().parent
root=P.parent
sys.path[:0]=[str(root),str(root/'outputs/d4_downstream_random/dual4b'),str(root/'outputs/d4_downstream_random')]
import downstream as ds
import evaluate as ev
from previous import dualfeat, Processor
from eval_packing import pack_single, parse_p1
torch.set_num_threads(2)
torch.manual_seed(17)
start=time.monotonic()
def log(*args):print(round(time.monotonic()-start),*args,flush=True)
def save(n,v):
    path=P/n;tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(v,indent=1));tmp.replace(path)
panels=json.loads(gzip.decompress((P/'panels.json.gz').read_bytes()))
pin=json.loads((root/'identity.json').read_bytes())
weights=root/'runs/formal_a_full1882_s17_v1/ckpt_u1000.pt'
assert ev.digest(weights)==pin['checkpoint'] and ev.digest(root/'tfi_model/pytorch_model.bin')==pin['tfi']
ck=torch.load(weights,map_location='cpu',weights_only=False)
frontend=ev.TrafficFrontend(2560);frontend.load_state_dict(ck['frontend']);frontend.cuda().eval()
tokenizer=AutoTokenizer.from_pretrained(root/'model',local_files_only=True)
llm=AutoModelForCausalLM.from_pretrained(root/'model',torch_dtype=torch.bfloat16,local_files_only=True)
llm=get_peft_model(llm,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',target_modules=['q_proj','k_proj','v_proj','o_proj']))
assert set(ck['lora'])=={k for k in llm.state_dict() if 'lora_' in k}
llm.load_state_dict(ck['lora'],strict=False);del ck
llm.cuda().eval();backbone=llm.get_base_model().model
token=PreTrainedTokenizerFast.from_pretrained(str(root/'tfi_model'),local_files_only=True)
tfi,info=AutoModelForMaskedLM.from_pretrained(str(root/'tfi_model'),local_files_only=True,output_loading_info=True)
assert not any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys'))
tfi.cuda().eval()
classifiers={k:ds.model_for(k,8).eval() for k in ds.NAMES}
vocab=[x for x,_ in collections.Counter(v*r['f'] for r in panels['train'] for v in r['s']).most_common(128)]
vsize=len(vocab); assert all(40<=abs(v)<=65535 for v in vocab)
save('identity.json',dict(script=ev.digest(Path(__file__)),panels=ev.digest(P/'panels.json.gz'),protocol=ev.digest(P/'protocol.txt'),checkpoint=pin['checkpoint'],TFI=pin['tfi'],vocab=vocab,device=torch.cuda.get_device_name(),torch=str(torch.__version__)))
# Numeric logits use the original TFI token mapping. Special predictions use its original zero fallback.
v2id={int(k)-1600:i for k,i in token.get_vocab().items() if k.isdigit() and 0<=int(k)<=3200}
from collections import namedtuple
Sample=namedtuple('Sample','idx offset signed times q flip')
@torch.inference_mode()
def extract(rows,rate,anchor):
    tasks=[];tf_by=[];truth=[]; counts=[]; positions=[]
    for ix,r in enumerate(rows):
        s=r['s'];f=r['f'];q=r['masks'][str(rate)] if rate else next(iter(r['masks'].values()))
        counts.append(len(q));positions.append(q);truth.extend(s[j] for j in q)
        n=len(s); cuts=[0,n] if n<=256 else [0,n//2,n]
        for a,b in zip(cuts,cuts[1:]):
            qq=[j-a for j in q if a<=j<b]
            if qq:tasks.append(Sample(ix,a,[f*v for v in s[a:b]],r['t'][a:b],qq,f))
    # Score TFI in its own full-flow context, including flows > 256 packets.
    byrow={}
    for lo in range(0,len(rows),24):
        rr=rows[lo:lo+24];texts=[]
        for i,r in enumerate(rr):
            q=set(positions[lo+i]);texts.append(' '.join(token.mask_token if j in q else str(v+1600) for j,v in enumerate(r['s'])))
        enc=token(texts,add_special_tokens=False,padding=True,return_tensors='pt')
        assert enc.attention_mask.sum(1).tolist()==[len(r['s']) for r in rr]
        scores=tfi(**enc.to('cuda')).logits
        for i,r in enumerate(rr):
            q=positions[lo+i];ss=scores[i,q].float().cpu();td=token.convert_ids_to_tokens(ss.argmax(1).tolist())
            values=[(int(v)-1600) if v.isdigit() and 0<=int(v)<=3200 else 0 for v in td]
            tf_by.extend(values); byrow[lo+i]={j:(ss[l],values[l]) for l,j in enumerate(q)}
        del scores
    features=[];language=[];cands=[];priors=[];ys=[];hybrid=[];llm_results={}
    # Sorting changes batching only. Results are restored by original flow and packet slot.
    cached={};tasks.sort(key=lambda x:(len(x.q),len(x.signed)))
    for lo in range(0,len(tasks),24):
        batch=tasks[lo:lo+24];packs=[];pkt_features=[];prefix_offsets=[]
        for w in batch:
            feat=dualfeat(w.signed,w.times,w.q)[1];b=ev.collate_features([feat],device='cuda');z=frontend(**b)
            ranges=b['ranges'][0];q=torch.tensor(w.q,device='cuda')
            cover=((q[:,None]>=ranges[None,:,0])&(q[:,None]<ranges[None,:,1])).float()
            beh=(cover/cover.sum(1,keepdim=True).clamp_min(1))@z['behavior_hidden'][0]
            pkt_features.append(torch.cat([z['packet_hidden'][0,q],beh],1).cpu())
            pack=pack_single(tokenizer,llm.get_input_embeddings(),z['packet_embeds'][0],z['behavior_embeds'][0],len(w.signed),w.q,canonical=True)
            packs.append(pack['inputs_embeds'][0]);prefix_offsets.append(len(packs[-1])-len(pack['layout']['segments'][2])-len(w.signed))
        width=max(map(len,packs));E=packs[0].new_zeros(len(batch),width,2560);A=torch.zeros(len(batch),width,device='cuda',dtype=torch.long)
        for j,p in enumerate(packs):E[j,-len(p):]=p;A[j,-len(p):]=1
        h=backbone(inputs_embeds=E,attention_mask=A,position_ids=(A.cumsum(1)-1).clamp_min(0),use_cache=False).last_hidden_state
        predarrays=[None]*len(batch)
        if anchor:
            gen=llm.generate(input_ids=torch.empty((len(batch),0),device='cuda',dtype=torch.long),inputs_embeds=E,attention_mask=A,logits_processor=[Processor(tokenizer,[len(w.q) for w in batch])],do_sample=False,num_beams=1,max_new_tokens=7*max(len(w.q) for w in batch)+2,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,use_cache=True).tolist()
            for j,(ids,w) in enumerate(zip(gen,batch)):
                result=parse_p1(tokenizer.decode(ids[:ids.index(tokenizer.eos_token_id)],clean_up_tokenization_spaces=False),len(w.q))
                assert result['status']=='valid';predarrays[j]=result['parsed_values']
        for j,w in enumerate(batch):
            hidden=set(w.q);visible=[v for v in range(len(w.signed)) if v not in hidden]
            hh=h[j,torch.tensor(w.q,device='cuda')+width-len(packs[j])+prefix_offsets[j]].half().cpu()
            for qi,pos in enumerate(w.q):
                left=[v for v in visible if v<pos];right=[v for v in visible if v>pos]
                neigh=[left[-1:] ,right[:1],left[-2:-1],right[1:2]]
                nc=[w.signed[l[0]] if l else 0 for l in neigh]
                ts,tp=byrow[w.idx][pos+w.offset]; vals=vocab+nc+[tp*w.flip if abs(tp)>=40 else 0]
                valid=[v!=0 and v not in vals[:k] for k,v in enumerate(vals)]
                # Missing candidates and duplicate indices carry no probability mass.
                values=[v if ok else 0 for ok,v in zip(valid,vals)]
                pr=torch.tensor([float(ts[v2id[v*w.flip]]) if v and v*w.flip in v2id else -100. for v in values])
                extra=[(l[0]-pos)/max(1,len(w.signed)) if l else 0 for l in neigh]+[v/1600 for v in nc]+[pos/max(1,len(w.signed)-1),len(w.signed)/512]
                cached[w.idx,pos+w.offset]=(torch.cat([pkt_features[j][qi],torch.tensor(extra)]),hh[qi],values,pr,w.signed[pos],w.flip,predarrays[j][qi]*w.flip if anchor else 0)
        del h
    index=[(i,j) for i,q in enumerate(positions) for j in q];assert len(index)==len(cached)==len(tf_by)
    x,l,c,p,y,f,orig=zip(*(cached[k] for k in index))
    assert np.array_equal(np.array(y)*np.array(f),truth)
    return dict(x=torch.stack(x).float(),l=torch.stack(l),c=torch.tensor(c),p=torch.stack(p),y=torch.tensor(y),f=np.array(f),tf=np.array(tf_by),orig=np.array(orig),truth=np.array(truth),positions=positions)
class Head(nn.Module):
    def __init__(self,with_language):
        super().__init__()
        self.local=nn.Sequential(nn.LayerNorm(522),nn.Linear(522,128),nn.GELU())
        self.output=nn.Linear(128,vsize+5)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        if with_language:self.lang=nn.Sequential(nn.LayerNorm(2560),nn.Linear(2560,128,bias=False))
    def forward(self,x,l):
        h=self.local(x)
        if hasattr(self,'lang'):h=h+self.lang(l)
        return self.output(h)
def logits(head,data,ix,arm):
    c=data['c'][ix].cuda();out=head(data['x'][ix].cuda(),data['l'][ix].float().cuda())
    valid=c!=0
    if arm=='B':valid[:,-1]=False
    else:out=out+data['p'][ix].cuda()
    return out.masked_fill(~valid,-1e6),c,valid
def train(data):
    heads={}
    for arm in ('B','C','D'):
        torch.manual_seed(17);h=Head(arm=='D').cuda()
        opt=torch.optim.AdamW(h.parameters(),lr=.001);g=torch.Generator().manual_seed(17);hist=[]
        for epoch in range(8):
            total=0.;num=0
            for ix in torch.randperm(len(data['y']),generator=g).split(512):
                scores,c,valid=logits(h,data,ix,arm)
                hits=c.eq(data['y'][ix].cuda()[:,None])&valid;ok=hits.any(1)
                if not ok.any():continue
                loss=nn.functional.cross_entropy(scores[ok],hits[ok].long().argmax(1))
                opt.zero_grad();loss.backward();opt.step();num+=int(ok.sum());total+=loss.item()*int(ok.sum())
            hist.append(total/num)
        heads[arm]=h.eval();torch.save(h.state_dict(),P/(arm+'.pt'))
        save(arm+'_fit.json',dict(n=int(len(data['y'])),covered=num,loss=hist))
    return heads
@torch.inference_mode()
def evaluate(tag,heads):
    rows=panels[tag];yy=np.asarray([r['y'] for r in rows]);results=[];archive={'ids':[r['id'] for r in rows],'y':yy}
    for rate in (10,30,50):
        data=extract(rows,rate,True);t=data['truth']
        methods={'clean':t,'TFI':data['tf'],'4B':data['orig']}
        for arm,h in heads.items():
            preds=[]
            for ix in torch.arange(len(t)).split(512):
                scores,c,_=logits(h,data,ix,arm);preds.extend(c.gather(1,scores.argmax(1)[:,None]).flatten().tolist())
            methods[arm]=np.array(preds)*data['f']
        archive[f'q_{rate}']=np.concatenate(data['positions']);archive[f'nq_{rate}']=[len(q) for q in data['positions']]
        for arm,pred in methods.items():
            mae=float(np.abs(np.abs(t)-np.abs(pred)).mean());direction=float(np.mean(np.sign(t)==np.sign(pred))*100)
            rebuilt=[];p=0
            for r,q in zip(rows,data['positions']):
                s=np.array(r['s']);s[q]=pred[p:p+len(q)];p+=len(q);rebuilt.append(s.tolist())
            assert p==len(pred);archive[f'{arm}_{rate}_packet']=pred
            ix=np.arange(len(rows))
            for j,k in enumerate(ds.NAMES):
                cls=classifiers[k];pr=ds.pv.classify(cls,rebuilt,ix,'cpu',512) if j<3 else ds.er.classify(cls,rebuilt,ix,'cpu',16 if k=='tst' else 48,1/1600)
                archive[f'{arm}_{rate}_{k}']=pr
                results.append(dict(arm=arm,rate=rate,clf=k,accuracy=float(np.mean(pr==yy)*100),f1=float(f1_score(yy,pr,average='macro',labels=np.arange(8),zero_division=0)*100),present_f1=float(f1_score(yy,pr,average='macro',labels=np.unique(yy),zero_division=0)*100),mae=mae,direction=direction))
        log(tag,rate,'complete')
    save(tag+'.json',results);np.savez_compressed(P/(tag+'_pred.npz'),**archive)
    return {a:{k:float(np.mean([r[k] for r in results if r['arm']==a])) for k in ('accuracy','f1','present_f1','mae','direction')} for a in methods}
log('extracting training reps')
tr=extract(panels['train'],None,False);log('fitting',len(tr['y']));heads=train(tr);del tr
screen=evaluate('screen',heads);qualified=[a for a in heads if screen[a]['accuracy']>=screen['TFI']['accuracy']]
winner=max(qualified,key=lambda a:(screen[a]['f1'],screen[a]['accuracy'])) if qualified else max(heads,key=lambda a:(screen[a]['accuracy'],screen[a]['f1']))
save('locked.json',dict(winner=winner,screen=screen));log('locked',winner)
confirm=evaluate('confirm',heads)
gate=all(rows[winner]['accuracy']>rows['TFI']['accuracy'] and rows[winner]['f1']>rows['TFI']['f1'] for rows in (screen,confirm))
save('done.json',dict(winner=winner,screen=screen,confirm=confirm,gate=gate,seconds=time.monotonic()-start));log('finished')
