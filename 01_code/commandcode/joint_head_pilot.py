"""D4 gold-supervised frozen dual-encoder head pilot. No LLM calls or classifier fitting."""
import argparse,csv,hashlib,html,io,json,math,msvcrt,os,pickle,sys,tarfile,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import prompt_ablation as p

ROOT=p.ROOT;DATA=p.DATA;OUT=DATA/'joint_head_dev200_20261116';PREV=p.OUT
sys.path.insert(0,str(ROOT/'outputs/d4_downstream_random/dual4b'))
from eval_frontend import TrafficFrontend,featurize,collate_features
OLDHEAD=ROOT/'outputs/d4_model_iterations_20261112/cloud_snapshot/readout01/pb.pt'
ARCHIVE=ROOT/'outputs/d4_constrained_u1000_20261112/cloud_input.tar'
CKMEMBER='runs/formal_a_full1882_s17_v1/ckpt_u1000.pt'
CKHASH='2bcd21e4534bf8fa79f9f8ad30630d2178ab4c2fafab9cedc977f139c20623f8'

def status(stage,**kwargs):
    p.api.save(OUT/'progress.json',{'utc':p.api.utc(),'pid':os.getpid(),'stage':stage,**kwargs})
    print(stage,json.dumps(kwargs),flush=True)

def inputs(seq,q):
    hidden=set(q)
    # No timestamps are exposed in the frozen teacher comparison. Mark timing statistics unavailable.
    records=[{'slot':i,'signed_len':int(v),'time_us':0} for i,v in enumerate(seq) if i not in hidden]
    f=featurize(records,len(seq))
    for v in f['stat_valid']:v[6:9]=[0.,0.,0.]
    return f

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():
        for path,h in p.api.read(OUT/'manifest.json')['files'].items():assert p.api.sha(Path(path).read_bytes())==h,path
        return
    for path,h in p.api.read(PREV/'manifest.json')['files'].items():assert p.api.sha(Path(path).read_bytes())==h,path
    rawpath=p.exp.SNAP/'work/d4_aligned/raw.pkl';raw=rawpath.read_bytes();d=pickle.loads(raw)
    assert p.api.sha(raw)==p.api.read(DATA/'protocol.json')['source_sha256']
    sets={s:{tuple(map(int,d['seqs'][i])) for i in d['split'][s]} for s in ('train','val','test')}
    panels={};identities={}
    for split,n,seed in [('train',2048,2026111701),('val',256,2026111702)]:
        other=set().union(*(v for k,v in sets.items() if k!=split));seen=set();candidates=[]
        for idx in map(int,d['split'][split]):
            seq=tuple(map(int,d['seqs'][idx]))
            if seq in other or seq in seen:continue
            seen.add(seq);candidates.append(idx)
        rng=np.random.default_rng(seed);chosen=sorted(rng.choice(candidates,n,False).tolist())
        rows=[]
        for i in chosen:
            seq=list(map(int,d['seqs'][i]));rate=(.1,.3,.5) if split=='train' else (.3,)
            for rr in rate:
                mask=sorted(np.random.default_rng([17,i,round(rr*100)]).choice(len(seq),max(1,round(rr*len(seq))),False).tolist())
                rows.append({'flow':str(d['ids'][i]),'raw_index':i,'truth':seq,'pos':mask,'rate':rr})
        panels[split]=rows;identities[split]={'source_eligible_unique':len(candidates),'raw_indices':chosen}
    panels['dev']=p.api.read(PREV/'cohorts.json')['dev']
    assert not {r['flow'] for r in panels['train']} & {r['flow'] for r in panels['val']+panels['dev']}
    targets=[abs(r['truth'][j]) for r in panels['train'] for j in r['pos']]
    vocab=[int(v) for v,n in sorted(Counter(targets).items(),key=lambda a:(-a[1],a[0]))[:64]]
    with tarfile.open(ARCHIVE) as tar:
        blob=tar.extractfile(CKMEMBER).read()
    assert p.api.sha(blob)==CKHASH
    ck=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
    assert ck['stage']=='a' and ck['update']==1000
    torch.save(ck['frontend'],OUT/'frozen_frontend.pt')
    p.api.save(OUT/'panels.json',panels);p.api.save(OUT/'split_identity.json',identities)
    protocol={'utc':p.api.utc(),'purpose':'Gold-supervised direction/length head pilot on frozen packet+behavior encoders',
        'training_flows':2048,'validation_flows':256,'dev_flows':200,'train_masks':[.1,.3,.5],
        'train_windows':len(panels['train']),'dev_mask':'Original 30% positions from prompt dev200; unchanged.',
        'split':'Original D4 train/val; deduplicated sequences, exclude exact complete-sequence matches across splits. No dev labels used in fitting.',
        'frontend':{'stage':'a','update':1000,'checkpoint_sha256':CKHASH,'frozen':True},
        'information':'Visible signed packet length, original slot, missingness and behavior statistics only. Timing unknown, validity disabled. First-packet D4 sign convention unchanged.',
        'old_dual':'Archived PB scalar readout, no-time D4-input adaptation. Not the later B candidate head or Qwen4B generator.',
        'new_control':'R: LayerNorm512,Linear512->128,GELU,Linear128->2. BCE direction + L1 softplus magnitude/1500.',
        'new_joint':'J: same trunk; binary direction logits + direction-conditional 65-way magnitude (64 train modes + OTHER) + two fallback magnitude regressors.',
        'joint_loss':'CE(direction) + CE(magnitude bucket conditioned on true direction) + 0.2 SmoothL1(softplus conditional magnitude, abs(y)/1500).',
        'joint_decode':'Choose direction from direction marginal; conditional mode, OTHER uses positive scalar regression. Round and clamp to [1,1600].',
        'vocab_from_train_only':vocab,'seed':17,'epochs':12,'batch_slots':512,'lr':.001,'weight_decay':.01,
        'budget':'Two new heads, one seed, exactly 12 epochs each. Fixed final epoch; validation monitored, no checkpoint/hyperparameter selection.',
        'evaluation':'Same six frozen classifier weights, six-label macro-F1, mean classifier Accuracy primary; magnitude MAE and signed/direction metrics.',
        'expansion_gate':'J exceeds Luna P0 and the matched R control in mean Accuracy, macro-F1 does not decrease, magnitude MAE is no worse than R; first fresh confirmation, no automatic full-cohort run.',
        'limitation':'Exploratory historical test dev200; not untouched final evaluation. Frontend pretraining provenance remains historical. Timing/LLM removed from this head pilot.',
        'not_new_invention':'Earlier numeric/candidate heads exist. This is a controlled head objective comparison, not a novelty claim.'}
    p.api.save(OUT/'protocol.json',protocol)
    files=dict(p.api.read(PREV/'manifest.json')['files'])
    for path in [Path(__file__),ROOT/'outputs/d4_downstream_random/dual4b/eval_frontend.py',OLDHEAD,OUT/'frozen_frontend.pt',OUT/'panels.json',OUT/'protocol.json',OUT/'split_identity.json']:
        files[str(path)]=p.api.sha(path.read_bytes())
    p.api.save(OUT/'manifest.json',{'files':files})
    status('prepared',train_flows=2048,training_views=len(panels['train']),vocab=len(vocab))

@torch.inference_mode()
def encode(front,b):
    n=b['valid'].shape[1];pos=torch.arange(n,dtype=torch.float32)[:,None]
    freq=torch.exp(torch.arange(0,256,2)*(-math.log(10000)/256));pe=torch.zeros(n,256)
    pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
    hp=front.packet_encoder(front.packet_input(b['numeric'])+pe,src_key_padding_mask=~b['valid'])
    hp=hp.masked_fill(~b['valid'].unsqueeze(-1),0)
    ix=torch.arange(n)[None,None,:]
    support=(ix>=b['ranges'][...,0,None])&(ix<b['ranges'][...,1,None])&b['observation'][:,None,:]&b['valid'][:,None,:]&b['behavior_valid'][...,None]
    weights=support.float()/support.sum(-1,keepdim=True).clamp_min(1)
    pooled=weights@hp;bounds=b['ranges'].float()/b['valid'].sum(-1).clamp_min(1)[:,None,None]
    raw=torch.cat([pooled,b['stats'],b['stat_valid'],bounds],-1)
    hb=front.behavior_encoder(front.behavior_input(raw)+front.scale_embedding(b['scale_ids']),src_key_padding_mask=~b['behavior_valid'])
    hb=hb.masked_fill(~b['behavior_valid'].unsqueeze(-1),0)
    return hp,hb

def get_features(front,rows,tag):
    path=OUT/(tag+'_features.pt')
    if path.exists():return torch.load(path,weights_only=True)
    tasks=[]
    for i,r in enumerate(rows):
        n=len(r['truth']);cuts=[0,n] if n<=256 else [0,n//2,n]
        for a,z in zip(cuts,cuts[1:]):
            q=[j-a for j in r['pos'] if a<=j<z]
            if q:tasks.append((i,a,z,q))
    tasks.sort(key=lambda t:(t[2]-t[1],t[0],t[1]));cache={};start=time.time()
    for lo in range(0,len(tasks),16):
        batch=tasks[lo:lo+16];fs=[inputs(rows[i]['truth'][a:z],q) for i,a,z,q in batch]
        b=collate_features(fs);hp,hb=encode(front,b)
        for j,(i,a,z,q) in enumerate(batch):
            qi=torch.tensor(q);ranges=b['ranges'][j]
            cover=((qi[:,None]>=ranges[:,0])&(qi[:,None]<ranges[:,1])&b['behavior_valid'][j]).float()
            beh=(cover/cover.sum(1,keepdim=True).clamp_min(1))@hb[j]
            x=torch.cat([hp[j,qi],beh],-1)
            for k,pos in enumerate(q):cache[i,a+pos]=x[k].clone()
        if lo%256==0 or lo+16>=len(tasks):
            done=min(lo+16,len(tasks));elapsed=time.time()-start
            status('encoding_'+tag,done=done,total=len(tasks),elapsed_seconds=elapsed,remaining_seconds=elapsed*(len(tasks)-done)/done)
    order=[(i,j) for i,r in enumerate(rows) for j in r['pos']]
    data={'x':torch.stack([cache[k] for k in order]),'y':torch.tensor([rows[i]['truth'][j] for i,j in order]),
          'nq':torch.tensor([len(r['pos']) for r in rows])}
    torch.save(data,path);return data

class Joint(nn.Module):
    def __init__(self,k):
        super().__init__();self.k=k
        self.trunk=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,128),nn.GELU())
        self.direction=nn.Linear(128,2);self.magnitude=nn.Linear(128,2*k);self.fallback=nn.Linear(128,2)
    def forward(self,x):
        h=self.trunk(x);return self.direction(h),self.magnitude(h).reshape(-1,2,self.k),self.fallback(h)

def regression():return nn.Sequential(nn.LayerNorm(512),nn.Linear(512,128),nn.GELU(),nn.Linear(128,2))

def loss(net,x,y,arm,vocab):
    if arm=='R':
        o=net(x);return F.binary_cross_entropy_with_logits(o[:,0],(y>0).float())+F.l1_loss(F.softplus(o[:,1]),y.abs()/1500)
    d,m,r=net(x);sign=(y>0).long();match=y.abs()[:,None]==vocab[None,:]
    target=torch.where(match.any(1),match.long().argmax(1),len(vocab));ii=torch.arange(len(y))
    return F.cross_entropy(d,sign)+F.cross_entropy(m[ii,sign],target)+.2*F.smooth_l1_loss(F.softplus(r[ii,sign]),y.abs()/1500)

@torch.inference_mode()
def predict(net,x,arm,vocab):
    result=[]
    for xx in x.split(512):
        if arm=='J':
            d,m,r=net(xx);sg=d.argmax(1);ii=torch.arange(len(xx));k=m[ii,sg].argmax(1)
            magnitude=torch.where(k<len(vocab),vocab[k.clamp_max(len(vocab)-1)],(F.softplus(r[ii,sg])*1500).round())
            values=(2*sg-1)*magnitude.clamp(1,1600)
        else:
            o=net(xx);values=torch.where(o[:,0]>=0,1,-1)*(F.softplus(o[:,1])*1500).round().clamp(40 if arm=='OLD_PB' else 1,1600)
        result.extend(values.long().tolist())
    return result

def fit(train,val,arm,vocab):
    torch.manual_seed(17);net=regression() if arm=='R' else Joint(len(vocab)+1)
    checkpoint=OUT/(arm+'.pt')
    if checkpoint.exists():net.load_state_dict(torch.load(checkpoint,weights_only=True));return net.eval()
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);rng=torch.Generator().manual_seed(17);log=[]
    for epoch in range(12):
        net.train();s=0.;count=0
        for ix in torch.randperm(len(train['y']),generator=rng).split(512):
            cost=loss(net,train['x'][ix],train['y'][ix],arm,vocab)
            opt.zero_grad();cost.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            s+=float(cost.detach())*len(ix);count+=len(ix)
        net.eval();vv=0.
        with torch.inference_mode():
            for ix in torch.arange(len(val['y'])).split(512):vv+=float(loss(net,val['x'][ix],val['y'][ix],arm,vocab))*len(ix)
        item={'epoch':epoch+1,'train_loss':s/count,'val_loss':vv/len(val['y'])};log.append(item)
        p.api.save(OUT/(arm+'_training.json'),log);status('fit_'+arm,**item)
    torch.save(net.state_dict(),checkpoint)
    p.api.save(OUT/(arm+'_identity.json'),{'sha256':p.api.sha(checkpoint.read_bytes()),'parameters':sum(x.numel() for x in net.parameters()),'epochs':12,'seed':17})
    return net.eval()

def score(rows,values,arm):
    stats=p.metrics(rows,values);p.api.save(OUT/(arm+'_predictions.json'),{'values':values,'classes':stats.pop('classes')})
    p.api.save(OUT/(arm+'_scores.json'),stats);status('scored_'+arm,accuracy=stats['mean_accuracy'],macro_f1=stats['mean_macro_f1'],mae=stats['magnitude_mae'])

def finish():
    results={a:p.api.read(OUT/(a+'_scores.json')) for a in ('clean','OLD_PB','R','J')}
    results.update({a:p.api.read(PREV/'dev'/(b+'_scores.json')) for a,b in [('Luna','P0'),('TFI','TFI')]})
    names={'clean':'完整真实流','OLD_PB':'旧双路 PB 头（无时间输入适配）','R':'双路＋同预算方向/包长回归头','J':'双路＋新方向条件包长头','Luna':'Luna 原提示词','TFI':'本地 TFI'}
    cols=['方法','Accuracy (%)','macro-F1 (%)','包长 MAE','带符号 MAE','方向正确率 (%)']
    table=[];details=[['分类器','方法','Accuracy (%)','macro-F1 (%)']]
    for a,s in results.items():
        table.append([names[a],s['mean_accuracy'],s['mean_macro_f1'],s['magnitude_mae'],s['signed_mae'],s['direction_pct']])
        for k,v in s['classifiers'].items():details.append([k.upper(),names[a],v['accuracy'],v['macro_f1']])
    for name,rows in [('summary.csv',[cols,*table]),('classifiers.csv',details)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(rows)
    j=results['J'];r=results['R'];l=results['Luna']
    gate=j['mean_accuracy']>max(r['mean_accuracy'],l['mean_accuracy']) and j['mean_macro_f1']>=max(r['mean_macro_f1'],l['mean_macro_f1']) and j['magnitude_mae']<=r['magnitude_mae']
    p.api.save(OUT/'decision.json',{'eligible_for_fresh_confirmation':bool(gate),'full_run_started':False,'reason':'Predeclared accuracy/F1/MAE gate; no hyperparameter search on dev200.'})
    p.api.save(OUT/'comparison.json',results)
    def tab(rows):return '<table>'+''.join('<tr>'+''.join('<'+('th' if i==0 else 'td')+'>'+html.escape(f'{v:.2f}' if isinstance(v,float) else str(v))+'</'+('th' if i==0 else 'td')+'>' for v in row)+'</tr>' for i,row in enumerate(rows))+'</table>'
    text='<html lang="zh"><meta charset="utf-8"><title>D4 真实监督双路输出头实验</title><style>body{font:16px Arial,"Microsoft YaHei";margin:35px;line-height:1.7}table{border-collapse:collapse;border-top:2px solid;border-bottom:2px solid;margin:25px 0}th{border-bottom:1px solid}td,th{padding:9px}</style><h1>D4 真实监督双路输出头实验</h1><p>固定原 200 条开发流、30% 缺包位置、六个分类器。完整真实流是参考成绩，不是严格数学上限。Accuracy 和 macro-F1 为六分类器等权平均；包长误差只统计 2,045 个缺失位置。</p>'+tab([cols,*table])+tab(details)+'<p>第1000步前端冻结，仅训练输出头；本轮没有调用 Qwen4B 或 Luna API。2,048 条原 D4 训练流，三种固定遮挡，256 条原 val 流监测；两头均固定12轮、seed17，不按开发成绩选轮。删除跨划分完全相同序列并去重。</p><p>旧 PB 是早期 512维数值头的无时间输入适配；不是后来的 B 候选头，也不是旧4B生成版本。其原训练分布与本轮不同；R 与 J 才是同数据同预算的结构对照。历史4B归档仅覆盖198/200，未将其不完整指标混入主表。所有当前输入均无时间戳。</p><p>历史开发池不是独立最终测试；流ID分离不等于独立PCAP采集。本轮不声称超过原论文。</p><p><a href="protocol.json">预登记配置</a> · <a href="CHANGELOG.txt">改动记录</a></p></html>'
    (OUT/'comparison.html').write_text(text,encoding='utf-8')
    (OUT/'CHANGELOG.txt').write_text('本轮改动：复用冻结双路包/行为编码器；R 用真实方向和包长训练普通回归头；J 用真实方向分类和方向条件的包长分布，训练集64个高频幅值加OTHER，OTHER采用回归。\n无语言模型调用、无教师伪标签、无测试真值输入、无时间戳。完整真实流仅用于诊断评分。\n旧PB单独注明输入适配；R/J为同预算对照。\n'+json.dumps(p.api.read(OUT/'protocol.json'),ensure_ascii=False,indent=2)+'\n'+json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    torch.set_num_threads(4);torch.manual_seed(17);prepare()
    if args.prepare:return
    panels=p.api.read(OUT/'panels.json');vocab=torch.tensor(p.api.read(OUT/'protocol.json')['vocab_from_train_only'])
    front=TrafficFrontend();front.load_state_dict(torch.load(OUT/'frozen_frontend.pt',weights_only=True));front.eval()
    for x in front.parameters():x.requires_grad_(False)
    rr=panels['dev'][0];f=inputs(rr['truth'],rr['pos']);changed=rr['truth'].copy()
    for q in rr['pos']:changed[q]=-1234
    assert f==inputs(changed,rr['pos'])
    b=collate_features([f])
    with torch.inference_mode():
        ref=front(**b);hp,hb=encode(front,b)
        assert torch.allclose(hp,ref['packet_hidden'],atol=1e-6) and torch.allclose(hb,ref['behavior_hidden'],atol=1e-6)
    p.api.save(OUT/'preflight.json',{'hidden_target_mutation_pass':True,'fast_encoder_equals_original':True,'checkpoint_sha256':CKHASH,'device':'cpu','threads':4})
    dev=get_features(front,panels['dev'],'dev')
    old=regression();old.load_state_dict(torch.load(OLDHEAD,weights_only=True,map_location='cpu'));old.eval()
    def byflow(flat,rows):
        out=[];start=0
        for rr in rows:out.append(flat[start:start+len(rr['pos'])]);start+=len(rr['pos'])
        assert start==len(flat);return out
    score(panels['dev'],byflow(predict(old,dev['x'],'OLD_PB',vocab),panels['dev']),'OLD_PB')
    train=get_features(front,panels['train'],'train');val=get_features(front,panels['val'],'val')
    del front
    for arm in ('R','J'):
        net=fit(train,val,arm,vocab);score(panels['dev'],byflow(predict(net,dev['x'],arm,vocab),panels['dev']),arm)
    finish();p.api.save(OUT/'complete.json',{'utc':p.api.utc(),'new_fits':2,'epochs_each':12});status('complete')

if __name__=='__main__':main()
