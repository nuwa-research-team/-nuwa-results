"""Finite hyperparameter study of original 64-mode head; train/validation only selection."""
import csv,json,html,math,pickle,time,msvcrt
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score,confusion_matrix
import joint_head_pilot as j

p=j.p;BASE=j.OUT;OUT=BASE/'original_head_tuning';j.OUT=OUT
CONFIGS=[dict(id='C0',lr=.001,wd=.01,dirw=1.,regw=.2),
         dict(id='C1',lr=.0003,wd=.01,dirw=1.,regw=.2),
         dict(id='C2',lr=.001,wd=.001,dirw=1.,regw=.2),
         dict(id='C3',lr=.001,wd=.01,dirw=2.,regw=.2),
         dict(id='C4',lr=.001,wd=.01,dirw=1.,regw=1.)]
def read(path):return p.api.read(path)
def save(name,obj):p.api.save(OUT/name,obj)
def sha(path):return p.api.sha(Path(path).read_bytes())
def group(flat,rows):
    result=[];at=0
    for r in rows:n=len(r['pos']);result.append(flat[at:at+n]);at+=n
    assert at==len(flat)
    return result
def loss(net,x,y,v,c):
    d,m,r=net(x);s=(y>0).long();matches=y.abs()[:,None]==v[None,:]
    target=torch.where(matches.any(1),matches.long().argmax(1),len(v));ii=torch.arange(len(y))
    return c['dirw']*F.cross_entropy(d,s)+F.cross_entropy(m[ii,s],target)+c['regw']*F.smooth_l1_loss(F.softplus(r[ii,s]),y.abs()/1500)
def score(stage,arm,rows,values,classes=None):
    result=p.metrics(rows,values,classes);cs=result.pop('classes');y=np.array([r['label'] for r in rows]);focal={}
    for lab,name in [(4,'Miuref'),(1,'SMB')]:
        ix=y==lab;truth=np.array([r['truth'][q] for r in rows if r['label']==lab for q in r['pos']]);pr=np.array([x for r,vs in zip(rows,values) if r['label']==lab for x in vs])
        focal[name]={'flows':int(ix.sum()),'correct_events':sum(int((np.array(c)[ix]==lab).sum()) for c in cs.values()),'recall':float(np.mean([z['recall_by_class'][lab] for z in result['classifiers'].values()])),
            'f1':float(np.mean([100*f1_score(y,c,labels=range(6),average=None,zero_division=0)[lab] for c in cs.values()])),
            'magnitude_mae':float(np.abs(np.abs(truth)-np.abs(pr)).mean()),'direction_wrong':int(((truth*pr)<=0).sum()),'slots':len(truth)}
    result['focal']=focal;save(f'{stage}/{arm}_scores.json',result);save(f'{stage}/{arm}_predictions.json',{'values':values,'classes':cs})
    print(stage,arm,*(round(result[k],4) for k in ['mean_accuracy','mean_macro_f1','magnitude_mae']),flush=True)
    return result
def tab(rows):
    return '<table>'+''.join('<tr>'+''.join(f'<{tag}>'+html.escape(f'{v:.2f}' if isinstance(v,float) else str(v))+f'</{tag}>' for v in row)+'</tr>' for tag,row in [('th',rows[0])]+[('td',row) for row in rows[1:]])+'</table>'
def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'complete.json').exists(),'Already complete, do not restart'
    torch.set_num_threads(4);torch.manual_seed(17)
    for path,h in read(BASE/'manifest.json')['files'].items():assert sha(path)==h,path
    panels=read(BASE/'panels.json');rawpath=p.exp.SNAP/'work/d4_aligned/raw.pkl';raw=pickle.loads(rawpath.read_bytes())
    assert sha(rawpath)==read(j.DATA/'protocol.json')['source_sha256']
    sets={s:{tuple(map(int,raw['seqs'][i])) for i in raw['split'][s]} for s in ('train','val','test')}
    mapping=read(j.ROOT/'outputs/d4_full_aligned_20261113/six_label_audit.json')['mapping'];seen=set();rows=[]
    for i in sorted(map(int,raw['split']['val'])):
        seq=tuple(map(int,raw['seqs'][i]))
        if seq in sets['train'] or seq in sets['test'] or seq in seen:continue
        seen.add(seq);q=sorted(np.random.default_rng([17,i,30]).choice(len(seq),max(1,round(.3*len(seq))),False).tolist());flow=str(raw['ids'][i])
        rows.append(dict(flow=flow,raw_index=i,truth=list(seq),pos=q,rate=.3,label=mapping[flow.split('.pcap#')[0]]))
    assert len(rows)==340 and Counter(r['label'] for r in rows)[4]==2
    lookup={r['flow']:r for r in rows}
    for r in panels['val']:assert r['truth']==lookup[r['flow']]['truth'] and r['pos']==lookup[r['flow']]['pos']
    assert not set(lookup)&{r['flow'] for r in panels['train']+panels['dev']}
    train=torch.load(BASE/'train_features.pt',weights_only=True);v=torch.tensor(read(BASE/'protocol.json')['vocab_from_train_only'])
    assert train['y'].tolist()==[r['truth'][q] for r in panels['train'] for q in r['pos']]
    assert all(r['raw_index'] in set(map(int,raw['split']['train'])) for r in panels['train'])
    watch=[BASE/'frozen_frontend.pt',BASE/'J.pt',BASE/'R.pt',BASE/'panels.json',BASE/'train_features.pt',BASE/'dev_features.pt',rawpath]
    watch+=list(p.exp.SIX.glob('six_*.pt'))
    # Include all six frozen feature extractors as well as final classifier heads.
    import sys;sys.path.insert(0,str(j.ROOT/'outputs/d4_downstream_random'));import downstream as ds
    watch += [ds.SNAP/f'work/res_aligned_{k}/{k}.pt' for k in ('cnn','lstm','gru')]
    watch += [ds.REPAIRED/f'repaired/repaired_{k}.pt' for k in ('tst','sdae','fsnet')]
    hashes={str(x):sha(x) for x in watch};save('val_rows.json',rows)
    protocol={'utc':p.api.utc(),'architecture':'Original J: 64 train-derived modes + OTHER, hard direction then magnitude; no expanded vocabulary, no joint MAP.',
        'train_flows':2048,'train_views':6144,'seed':17,'batch':512,'max_epochs':24,'evaluation_epochs':[12,24],'configs':CONFIGS,
        'budget':'Five trajectories, 24 epochs each, ten checkpoint evaluations; one seed. C0 epoch12 must reproduce archived J exactly. No adaptive additions.',
        'val_flows':340,'val_class_support':dict(Counter(r['label'] for r in rows)),
        'validation_change':'All eligible deduplicated original D4 val flows; same masks on old256. Exclude train/test exact sequence overlap. No historical test used to rank.',
        'selection':'Eligible if validation Accuracy >= old J, macroF1 >= old J, magnitudeMAE <= old J and at least one strict improvement. Choose max Accuracy then macroF1 then min MAE then arm name. If none eligible, descriptive candidate max macroF1 then Accuracy then min MAE; label tradeoff, not promotion.',
        'post_selection':'Freeze one checkpoint ID before evaluating original dev200; report alongside R/J/TFI. No confirmation reuse or full run in this batch.',
        'fixed':'Same train examples/masks, frozen dual frontend and six classifiers, no class-label input to imputer. All changes are optimizer/epoch/loss weights.',
        'limitations':'One seed, small validation, only two Miuref; historical dev repeatedly examined. No independent final performance claim.',
        'source_hashes':hashes,'code_sha256':sha(__file__)}
    save('protocol.json',protocol)
    front=j.TrafficFrontend();front.load_state_dict(torch.load(BASE/'frozen_frontend.pt',weights_only=True));front.eval()
    for param in front.parameters():param.requires_grad_(False)
    val=j.get_features(front,rows,'selection340');original=j.Joint(65);original.load_state_dict(torch.load(BASE/'J.pt',weights_only=True));original.eval()
    baseline=score('val','J-original',rows,group(j.predict(original,val['x'],'J',v),rows));results={};replay=False
    for c in CONFIGS:
        torch.manual_seed(17);net=j.Joint(65);optimizer=torch.optim.AdamW(net.parameters(),lr=c['lr'],weight_decay=c['wd']);rng=torch.Generator().manual_seed(17);log=[]
        for epoch in range(1,25):
            net.train();total=0.;start=time.time()
            for ix in torch.randperm(len(train['y']),generator=rng).split(512):
                cost=loss(net,train['x'][ix],train['y'][ix],v,c);optimizer.zero_grad();cost.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();total+=float(cost.detach())*len(ix)
            net.eval();entry={'epoch':epoch,'train_loss':total/len(train['y']),'seconds':time.time()-start};log.append(entry);save(c['id']+'_training.json',log)
            print('fit',c['id'],epoch,round(entry['train_loss'],5),flush=True)
            if epoch in (12,24):
                arm=c['id']+f'-e{epoch}';torch.save(net.state_dict(),OUT/(arm+'.pt'))
                if c['id']=='C0' and epoch==12:
                    assert all(torch.equal(x,original.state_dict()[key]) for key,x in net.state_dict().items());replay=True
                results[arm]=score('val',arm,rows,group(j.predict(net,val['x'],'J',v),rows))
    keys=['mean_accuracy','mean_macro_f1','magnitude_mae']
    eligible=[a for a,s in results.items() if s[keys[0]]>=baseline[keys[0]] and s[keys[1]]>=baseline[keys[1]] and s[keys[2]]<=baseline[keys[2]] and any(s[k]!=baseline[k] for k in keys)]
    pool=eligible or list(results)
    selected=sorted(pool,key=lambda a:(-results[a][keys[0 if eligible else 1]],-results[a][keys[1 if eligible else 0]],results[a][keys[2]],a))[0]
    decision={'utc_before_dev':p.api.utc(),'selected':selected,'eligible':eligible,'validation_all_three_improved_or_tied':bool(eligible),'selection_source':'original D4 val340 only','confirmation_run':False,'full_run':False}
    save('selection.json',decision);print('SELECTED',json.dumps(decision),flush=True)
    net=j.Joint(65);net.load_state_dict(torch.load(OUT/(selected+'.pt'),weights_only=True));net.eval();dev=torch.load(BASE/'dev_features.pt',weights_only=True)
    devresults={}
    for arm,path in [('J-original',BASE/'J_predictions.json'),('R',BASE/'R_predictions.json'),('TFI',j.PREV/'dev/TFI_predictions.json')]:
        z=read(path);devresults[arm]=score('dev',arm,panels['dev'],z['values'],z['classes'])
    devresults[selected]=score('dev',selected,panels['dev'],group(j.predict(net,dev['x'],'J',v),panels['dev']))
    for path,h in hashes.items():assert sha(path)==h,path
    for stage,rs,cohort in [('val',{'J-original':baseline,**results},rows),('dev',devresults,panels['dev'])]:
        y=[r['label'] for r in cohort];truth=np.array([r['truth'][q] for r in cohort for q in r['pos']])
        for arm,s in rs.items():
            z=read(OUT/stage/(arm+'_predictions.json'));pred=np.array(sum(z['values'],[]));assert abs(np.abs(abs(truth)-abs(pred)).mean()-s['magnitude_mae'])<1e-9
            for k,cs in z['classes'].items():
                assert abs(100*np.mean(np.array(cs)==y)-s['classifiers'][k]['accuracy'])<1e-9
                assert abs(100*f1_score(y,cs,labels=range(6),average='macro',zero_division=0)-s['classifiers'][k]['macro_f1'])<1e-9
    save('audit.json',{'passed':True,'frozen_sources_unchanged':True,'C0_e12_exact_reproduces_original':replay,'train_identity_and_masks_verified':True,'selection_before_dev':True,'new_fit_trajectories':5,'epochs_each':24,'scoring_recomputed':True})
    save('comparison.json',{'val':{'J-original':baseline,**results},'dev':devresults})
    columns=['方法','Accuracy %','macro-F1 %','幅值MAE/字节','带符号MAE','方向正确率 %','Miuref召回 %','Miuref F1 %']
    body='<html lang="zh"><meta charset="utf-8"><style>body{font:16px Arial,"Microsoft YaHei";margin:32px;line-height:1.7}table{border-collapse:collapse;border-top:2px solid;border-bottom:2px solid;margin:24px 0}th{border-bottom:1px solid}th,td{padding:9px}</style><h1>原联合头有限调参实验</h1><p>仅调整学习率、权重衰减、方向/幅值损失权重与轮数。64类原结构和先定方向解码不变。2,048条训练流；340条原验证流选参；原200条开发流只在选择冻结后报告。固定30%验证/开发掩码，训练三档固定掩码。</p>'
    for stage,rs in [('val',{'J-original':baseline,**results}),('dev',devresults)]:
        table=[columns]+[[a,s['mean_accuracy'],s['mean_macro_f1'],s['magnitude_mae'],s['signed_mae'],s['direction_pct'],s['focal']['Miuref']['recall'],s['focal']['Miuref']['f1']] for a,s in rs.items()]
        with (OUT/(stage+'_summary.csv')).open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(table)
        body+='<h2>'+stage+'</h2>'+tab(table)
    body+='<p>'+html.escape(json.dumps(CONFIGS,ensure_ascii=False))+'</p><p>'+html.escape(json.dumps(decision,ensure_ascii=False))+'</p><p>结果是六个冻结分类器平均；Miuref验证/开发各仅2条。验证类比例与开发不同。单种子搜索，不宣称独立最终提升；没有扩大或重用确认集选参。<a href="protocol.json">预登记</a> · <a href="audit.json">核验</a></p></html>'
    (OUT/'report.html').write_text(body,encoding='utf-8')
    (OUT/'CHANGELOG.txt').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n'+json.dumps(decision,ensure_ascii=False,indent=2)+'\n'+json.dumps(devresults,ensure_ascii=False,indent=2),encoding='utf-8')
    save('complete.json',{'utc':p.api.utc(),'selection':decision});print('COMPLETE',flush=True)
if __name__=='__main__':main()
