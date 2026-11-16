"""Bounded train-only vocabulary expansion and joint-MAP decoding study."""
import csv, html, json, time, pickle
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score, recall_score, confusion_matrix
import joint_head_pilot as j

p=j.p; BASE=j.OUT; OUT=BASE/'tail_joint_map'; j.OUT=OUT

def save(name,obj): p.api.save(OUT/name,obj)
def sha(path): return p.api.sha(Path(path).read_bytes())
def byflow(flat,rows):
    result=[]; at=0
    for r in rows:
        n=len(r['pos']); result.append(flat[at:at+n]); at+=n
    assert at==len(flat)
    return result

@torch.inference_mode()
def jointmap(net,x,vocab):
    result=[]
    for xx in x.split(512):
        d,m,r=net(xx)
        logp=F.log_softmax(d,1)[:,:,None]+F.log_softmax(m,2)
        best=logp.flatten(1).argmax(1); sign=best//net.k; k=best%net.k
        ii=torch.arange(len(xx))
        mag=torch.where(k<len(vocab),vocab[k.clamp_max(len(vocab)-1)],(F.softplus(r[ii,sign])*1500).round())
        result.extend(((2*sign-1)*mag.clamp(1,1600)).long().tolist())
    return result

def evaluate(stage,arm,rows,values,classes=None):
    stats=p.metrics(rows,values,classes); cs=stats.pop('classes')
    target=np.array([r['label'] for r in rows]); focal={}
    for label,name in [(4,'Miuref'),(1,'SMB')]:
        mask=target==label
        truth=np.array([r['truth'][pos] for r in rows if r['label']==label for pos in r['pos']])
        pred=np.array([v for r,vs in zip(rows,values) if r['label']==label for v in vs])
        focal[name]={'flows':int(mask.sum()),'slots':len(truth),
            'correct_events':sum(int((np.array(c)[mask]==label).sum()) for c in cs.values()),
            'mean_recall':float(np.mean([s['recall_by_class'][label] for s in stats['classifiers'].values()])),
            'mean_f1':float(np.mean([100*f1_score(target,c,labels=range(6),average=None,zero_division=0)[label] for c in cs.values()])),
            'direction_wrong':int(((truth*pred)<=0).sum()),'magnitude_mae':float(np.abs(np.abs(truth)-np.abs(pred)).mean())}
    stats['focal']=focal
    save(f'{stage}/{arm}_predictions.json',{'values':values,'classes':cs})
    save(f'{stage}/{arm}_scores.json',stats)
    print(stage,arm,json.dumps({k:stats[k] for k in ['mean_accuracy','mean_macro_f1','magnitude_mae','focal']}),flush=True)
    return stats

def table(rs):
    return '<table>'+''.join('<tr>'+''.join(f'<{tag}>'+html.escape(f'{v:.2f}' if isinstance(v,float) else str(v))+f'</{tag}>' for v in row)+'</tr>' for tag,row in [('th',rs[0])]+[('td',r) for r in rs[1:]])+'</table>'

def report(results,decision):
    rows=[['方法','Accuracy %','macro-F1 %','幅值 MAE (字节)','带符号 MAE','Miuref召回 %','Miuref F1 %','Miuref判对次数/12']]
    for a,s in results.items():
        f=s['focal']['Miuref'];rows.append([a,s['mean_accuracy'],s['mean_macro_f1'],s['magnitude_mae'],s['signed_mae'],f['mean_recall'],f['mean_f1'],f['correct_events']])
    with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f: csv.writer(f).writerows(rows)
    details=[['方法','分类器','Accuracy %','macro-F1 %','Miuref召回 %','SMB召回 %']]
    for a,s in results.items():
        for c,ss in s['classifiers'].items(): details.append([a,c,ss['accuracy'],ss['macro_f1'],ss['recall_by_class'][4],ss['recall_by_class'][1]])
    with (OUT/'classifiers.csv').open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(details)
    text='<html lang="zh"><meta charset="utf-8"><style>body{font:16px Arial,"Microsoft YaHei";margin:32px;line-height:1.7}table{border-collapse:collapse;border-top:2px solid;border-bottom:2px solid;margin:24px 0}th{border-bottom:1px solid}td,th{padding:9px}</style><h1>低频包长覆盖与联合概率解码：200条开发流</h1><p>固定30%缺包、2045个缺失位置。Accuracy和macro-F1为六个冻结分类器等权平均；Miuref只有2条，12次为2流×6分类器，并非12个独立样本。</p>'+table(rows)+table(details)
    text+='<p>J = 原64类加OTHER，W = 训练目标全部1397幅值加OTHER。hard = 先方向后包长；map = 在两个方向的联合概率中选最大项。仅新训练W一次、12轮、seed17。两种解码共享相同权重。编码器、分类器、训练样本和掩码不变；W输出层更大，属于容量变化，不将收益完全归因于机制。</p><p>OTHER是含多种幅值的合并事件，其概率与单一幅值的概率存在粒度差异；联合MAP是本轮有界近似，不是校准完成的概率模型。未按开发分数调阈值。</p><p>确认决策：'+html.escape(json.dumps(decision,ensure_ascii=False))+'</p><p>历史开发池不是独立最终测试；本轮不声称超过原论文。<a href="protocol.json">预登记</a> · <a href="audit.json">核验</a></p></html>'
    (OUT/'report.html').write_text(text,encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    assert not (OUT/'complete.json').exists(),'Already completed; do not refit.'
    panels=p.api.read(BASE/'panels.json')
    for path,h in p.api.read(BASE/'manifest.json')['files'].items():assert sha(path)==h,path
    train=torch.load(BASE/'train_features.pt',weights_only=True);val=torch.load(BASE/'val_features.pt',weights_only=True);dev=torch.load(BASE/'dev_features.pt',weights_only=True)
    for split,data in [('train',train),('val',val),('dev',dev)]:
        assert data['y'].tolist()==[r['truth'][pos] for r in panels[split] for pos in r['pos']]
    raw=pickle.loads((p.exp.SNAP/'work/d4_aligned/raw.pkl').read_bytes())
    trainids=set(map(int,raw['split']['train']))
    assert all(r['raw_index'] in trainids and r['truth']==list(map(int,raw['seqs'][r['raw_index']])) for r in panels['train'])
    vocab=train['y'].abs().unique(sorted=True)
    oldv=torch.tensor(p.api.read(BASE/'protocol.json')['vocab_from_train_only'])
    watch=[BASE/'frozen_frontend.pt',BASE/'panels.json',BASE/'train_features.pt',BASE/'val_features.pt',BASE/'dev_features.pt',BASE/'R.pt',BASE/'J.pt']
    watch+=list(p.exp.PE.rglob('*.pt')) if hasattr(p.exp,'PE') else []
    hashes={str(path):sha(path) for path in watch}
    protocol={'utc':p.api.utc(),'new_fits':1,'epochs':12,'seed':17,'batch':512,'lr':.001,'weight_decay':.01,
        'train_flows':2048,'train_views':6144,'train_missing_slots':len(train['y']),'train_only_vocab':vocab.tolist(),
        'change_1':'Replace top64 vocabulary by all distinct training-target magnitudes; keep OTHER and same regression fallback, trunk, losses.',
        'change_2':'Joint MAP: argmax_s,k log p(s|x)+log p(k|s,x), no direction confidence threshold; OTHER uses selected direction regression.',
        'arms':['R reused','J-hard reused','J-map no fit','W-hard','W-map','TFI reused'],
        'fixed':'Frozen encoder features, six classifiers, samples, masks, optimizer, fixed final epoch; no development labels in fitting.',
        'budget':'One new head fit, two decoders; no seeds/threshold/hyperparameter retries. Validation loss monitoring only.',
        'confirmation_gate':'Only preregistered combined W-map: Accuracy>J-hard, macroF1>J-hard and >=R, magnitudeMAE<=J-hard, Miuref recall>J-hard. If passed evaluate fixed confirmation200 once; no full run.',
        'capacity_note':'W larger output layer; computational budget equal epochs/examples, not equal parameters/FLOPs.',
        'sources':hashes,'code_sha256':sha(__file__)}
    save('protocol.json',protocol)
    print('Prepared vocabulary',len(vocab),'new fit starts',flush=True)
    net=j.fit(train,val,'W',vocab)
    old=j.Joint(len(oldv)+1);old.load_state_dict(torch.load(BASE/'J.pt',weights_only=True));old.eval()
    # Reload checkpoint for exact independent inference replay.
    replay=j.Joint(len(vocab)+1);replay.load_state_dict(torch.load(OUT/'W.pt',weights_only=True));replay.eval()
    results={}
    for a,path in [('R',BASE/'R_predictions.json'),('J-hard',BASE/'J_predictions.json'),('TFI',j.PREV/'dev/TFI_predictions.json')]:
        saved=p.api.read(path);results[a]=evaluate('dev',a,panels['dev'],saved['values'],saved['classes'])
    for a,n,v,decode in [('J-map',old,oldv,jointmap),('W-hard',net,vocab,lambda n,x,v:j.predict(n,x,'J',v)),('W-map',net,vocab,jointmap)]:
        flat=decode(n,dev['x'],v)
        if a.startswith('W'):assert flat==decode(replay,dev['x'],v)
        results[a]=evaluate('dev',a,panels['dev'],byflow(flat,panels['dev']))
    w=results['W-map'];o=results['J-hard'];r=results['R']
    checks={'accuracy_above_J':w['mean_accuracy']>o['mean_accuracy'],'f1_above_J':w['mean_macro_f1']>o['mean_macro_f1'],
        'f1_at_least_R':w['mean_macro_f1']>=r['mean_macro_f1'],'mae_no_worse_J':w['magnitude_mae']<=o['magnitude_mae'],
        'miuref_recall_above_J':w['focal']['Miuref']['mean_recall']>o['focal']['Miuref']['mean_recall']}
    decision={'gate_checks':checks,'confirmation_run':all(checks.values()),'full_run':False}
    save('decision.json',decision);save('comparison.json',results)
    if decision['confirmation_run']:
        rows=p.api.read(j.PREV/'cohorts.json')['confirm']
        assert not {r['flow'] for r in rows}&{r['flow'] for r in panels['train']+panels['dev']}
        front=j.TrafficFrontend();front.load_state_dict(torch.load(BASE/'frozen_frontend.pt',weights_only=True));front.eval()
        for param in front.parameters():param.requires_grad_(False)
        data=j.get_features(front,rows,'confirm');save('confirm/rows.json',rows)
        for a,n,v,decode in [('W-map',net,vocab,jointmap),('J-hard',old,oldv,lambda n,x,v:j.predict(n,x,'J',v))]:evaluate('confirm',a,rows,byflow(decode(n,data['x'],v),rows))
        rn=j.regression();rn.load_state_dict(torch.load(BASE/'R.pt',weights_only=True));rn.eval()
        evaluate('confirm','R',rows,byflow(j.predict(rn,data['x'],'R',oldv),rows))
        with np.load(p.exp.PE/'tfi_imputed_30.npz') as z: lookup=dict(zip(z['ids'],z['values']))
        evaluate('confirm','TFI',rows,[[int(lookup[r['flow']][pos]) for pos in r['pos']] for r in rows])
    unchanged=all(sha(path)==h for path,h in hashes.items());assert unchanged
    for a in results:
        saved=p.api.read(OUT/f'dev/{a}_predictions.json');s=results[a];truth=np.array([r['truth'][pos] for r in panels['dev'] for pos in r['pos']]);pred=np.array(sum(saved['values'],[]));y=[r['label'] for r in panels['dev']]
        assert abs(np.abs(np.abs(truth)-np.abs(pred)).mean()-s['magnitude_mae'])<1e-9
        for c,cl in saved['classes'].items():
            assert abs(100*np.mean(np.array(cl)==y)-s['classifiers'][c]['accuracy'])<1e-9
            assert abs(100*f1_score(y,cl,labels=range(6),average='macro',zero_division=0)-s['classifiers'][c]['macro_f1'])<1e-9
    save('audit.json',{'passed':True,'source_hashes_unchanged':unchanged,'train_source_verified':True,'cached_targets_alignment':True,'new_checkpoint_replay_exact':True,'metrics_recomputed':True,'only_new_fit':'W','new_parameters':sum(x.numel() for x in net.parameters()),'old_parameters':sum(x.numel() for x in old.parameters())})
    report(results,decision)
    (OUT/'CHANGELOG.txt').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\nRESULTS\n'+json.dumps(results,ensure_ascii=False,indent=2)+'\nDECISION\n'+json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
    save('complete.json',{'utc':p.api.utc(),'decision':decision})
    print('COMPLETE',json.dumps(decision),flush=True)

if __name__=='__main__':main()
