"""Frozen D4 paired evaluation. Record each whole-window query before scoring."""
import sys, json, time, random, gzip, hashlib, subprocess, os
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
D=P/'outputs/d4_downstream_random'
sys.path[:0]=[str(D/'dual4b'),str(D)]
import evaluate as ev
from eval_packing import pack_single,parse_p1
from constrained import ArrayGrammar
from transformers import AutoTokenizer,AutoModelForCausalLM,LogitsProcessor
from peft import get_peft_model,LoraConfig
from previous import Processor,dualfeat
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return [json.loads(l) for l in p.open() if l.strip()]
@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    ident=json.loads((P/'identity.json').read_bytes())
    ckpath=P/'runs/formal_a_full1882_s17_v1/ckpt_u1000.pt'
    assert sha(ckpath)==ident['checkpoint']
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    assert ck['stage']=='a' and ck['update']==1000
    md=P/'model'
    for field,p in [('model_config_sha256',md/'config.json'),('tokenizer_sha256',md/'tokenizer.json'),('data_windows_sha256',P/'data/windows.jsonl')]: assert sha(p)==ck['meta'][field],field
    for n,codehash in ident['code'].items():
        if n!='run.py':assert sha((D if n=='downstream.py' else D/'dual4b')/n)==codehash,n
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(md,torch_dtype=torch.bfloat16,local_files_only=True)
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',target_modules=['q_proj','k_proj','v_proj','o_proj']))
    assert set(ck['lora'])=={k for k in model.state_dict() if 'lora_' in k}
    model.load_state_dict(ck['lora'],strict=False)
    front=ev.TrafficFrontend(2560)
    front.load_state_dict(ck['frontend'])
    front.cuda().eval();model.cuda().eval();del ck
    flows={r['id']:r for r in read(D/'flows.jsonl')}
    assert len(flows)==13150
    directions=json.loads(gzip.decompress((P/'directions.json.gz').read_bytes()))
    rates=(10,30,50)
    batches=[]; byrate={}
    for rate in rates:
        z=np.load(D/f'mask{rate}.npz')
        assert z['ids'].astype(str).tolist()==list(flows)
        path=P/f'windows_{rate}.jsonl'
        saved=read(path) if path.exists() else []
        keys={(r['id'],r['a']) for r in saved}
        assert len(keys)==len(saved)
        tasks=[]
        for (key,r),mask in zip(flows.items(),z['mask']):
            if key not in directions:continue
            n=len(r['signed'])
            cuts=[0,n] if n<=256 else [0,n//2,n]
            for a,b in zip(cuts,cuts[1:]):
                q=np.flatnonzero(mask[a:b]).tolist()
                if q and (key,a) not in keys:tasks.append((key,a,b,q))
        tasks.sort(key=lambda v:(len(v[3]),v[2]-v[1]))
        byrate[rate]=len(tasks)
        for i in range(0,len(tasks),64):batches.append((rate,tasks[i:i+64]))
    random.Random(42).shuffle(batches)
    print('START',len(directions),'flows; pending windows',byrate,flush=True)
    cost=lambda tasks: 6+max(len(v[3]) for v in tasks)
    total=sum(cost(batch) for rate,batch in batches);work=0;done=0;start=time.monotonic()
    for rate,batch in batches:
        vec=[];counts=[]
        for key,a,b,q in batch:
            f,t=directions[key]; assert f in (-1,1)
            signs=[v*f for v in flows[key]['signed'][a:b]]
            feat=dualfeat(signs,t[a:b],q)[1]
            z=front(**ev.collate_features([feat],device='cuda'))
            pack=pack_single(tok,model.get_input_embeddings(),z['packet_embeds'][0],z['behavior_embeds'][0],b-a,q,canonical=True)
            vec.append(pack['inputs_embeds'][0]);counts.append(len(q))
        width=max(map(len,vec))
        E=vec[0].new_zeros((len(vec),width,2560));A=torch.zeros(E.shape[:2],device='cuda',dtype=torch.long)
        for j,v in enumerate(vec):E[j,-len(v):]=v;A[j,-len(v):]=1
        args=dict(input_ids=torch.empty((len(vec),0),device='cuda',dtype=torch.long),inputs_embeds=E,attention_mask=A,logits_processor=[Processor(tok,counts)],do_sample=False,num_beams=1,max_new_tokens=7*max(counts)+2,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,use_cache=True)
        gen=model.generate(**args).tolist()
        results=[]
        for (key,a,b,q),g in zip(batch,gen):
            raw=tok.decode(g[:g.index(tok.eos_token_id)],clean_up_tokenization_spaces=False)
            result=parse_p1(raw,len(q));assert result['status']=='valid',(key,raw)
            results.append(dict(id=key,a=a,q=q,pred=[v*directions[key][0] for v in result['parsed_values']]))
        with (P/f'windows_{rate}.jsonl').open('a') as file:
            for r in results:print(json.dumps(r,separators=(',',':')),file=file)
        work+=cost(batch);done+=len(batch);elapsed=time.monotonic()-start
        print('PROGRESS',done,'/',sum(byrate.values()),'windows',rate,'elapsed_s',round(elapsed),'eta_s',round(elapsed*(total-work)/work),flush=True)
    if len(directions)==len(flows):subprocess.run([sys.executable,'score.py'],check=True)
    else:print('awaiting verified directions',len(flows)-len(directions),flush=True)
if __name__=='__main__':main()