"""No-fit diagnosis of frozen dev200 results, with bounded oracle component repairs."""
import csv,html,json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score,f1_score
import joint_head_pilot as j

p=j.p;SRC=j.OUT;OUT=SRC/'diagnosis_miuref_smb';OUT.mkdir(exist_ok=True)
rows=p.api.read(SRC/'panels.json')['dev'];y=np.array([r['label'] for r in rows]);K=p.exp.KINDS
NAMES=['Weibo','SMB','Htbot','Virut','Miuref','OtherMerged']
FILES={'J':SRC/'J_predictions.json','R':SRC/'R_predictions.json',
       'B':SRC/'current_B_198_predictions.json','TFI':j.PREV/'dev/TFI_predictions.json'}
D={a:p.api.read(path) for a,path in FILES.items()}
common=D['B']['indices'];byB={v:i for i,v in enumerate(common)}
initial={str(path):p.api.sha(path.read_bytes()) for path in [*FILES.values(),SRC/'panels.json',SRC/'clean_predictions.json',SRC/'J.pt',SRC/'R.pt']}

def values(a,i):return D[a]['values'][byB[i] if a=='B' else i]
def classes(a,k,ix):return np.array([D[a]['classes'][k][byB[i] if a=='B' else i] for i in ix])
def writecsv(name,records):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def perclass(a,ix):
    result=[]
    for k in K:
        pp=classes(a,k,ix);yy=y[ix];pr,re,ff,support=precision_recall_fscore_support(yy,pp,labels=range(6),zero_division=0)
        cm=confusion_matrix(yy,pp,labels=range(6))
        for c,name in enumerate(NAMES):
            result.append({'model':a,'classifier':k,'class':name,'support':int(support[c]),'tp':int(cm[c,c]),
                'fp':int(cm[:,c].sum()-cm[c,c]),'fn':int(cm[c].sum()-cm[c,c]),
                'precision':100*pr[c],'recall':100*re[c],'f1':100*ff[c],
                'predicted_classes':json.dumps({NAMES[t]:int(n) for t,n in enumerate(cm[c]) if n})})
    return result

full={a:perclass(a,list(range(200))) for a in ('R','J','TFI')}
paired={a:perclass(a,common) for a in ('R','J','B','TFI')}
writecsv('per_classifier_class200.csv',[r for v in full.values() for r in v])
writecsv('per_classifier_class198.csv',[r for v in paired.values() for r in v])
decomposition=[]
for c,name in enumerate(NAMES):
    rr=[x for x in full['R'] if x['class']==name];jj=[x for x in full['J'] if x['class']==name]
    rt=sum(x['tp'] for x in rr);jt=sum(x['tp'] for x in jj)
    rf=np.mean([x['f1'] for x in rr]);jf=np.mean([x['f1'] for x in jj])
    decomposition.append({'class':name,'support':int(sum(y==c)),'R_correct_events':rt,'J_correct_events':jt,
        'net_correct_events':jt-rt,'accuracy_contribution_pp':100*(jt-rt)/1200,
        'R_mean_class_f1':rf,'J_mean_class_f1':jf,'macro_f1_contribution_pp':(jf-rf)/6})
scorej=p.api.read(SRC/'J_scores.json');scorer=p.api.read(SRC/'R_scores.json')
assert abs(sum(x['accuracy_contribution_pp'] for x in decomposition)-(scorej['mean_accuracy']-scorer['mean_accuracy']))<1e-9
assert abs(sum(x['macro_f1_contribution_pp'] for x in decomposition)-(scorej['mean_macro_f1']-scorer['mean_macro_f1']))<1e-9
writecsv('metric_decomposition.csv',decomposition)

summary=[];packetrows=[];packetstats=[];flowstats=[]
for c in (4,1):
    ix=np.flatnonzero(y==c).tolist();assert all(i in byB for i in ix)
    for a in ('R','J','B','TFI'):
        cc=[x for x in paired[a] if x['class']==NAMES[c]]
        summary.append({'class':NAMES[c],'model':a,'support_flows':len(ix),'correct_events_out_of_6N':sum(x['tp'] for x in cc),
            'mean_recall':float(np.mean([x['recall'] for x in cc])),'mean_precision':float(np.mean([x['precision'] for x in cc])),
            'mean_f1_paired198':float(np.mean([x['f1'] for x in cc])),
            'SMB_to_Weibo_events':sum(int(np.sum(classes(a,k,ix)==0)) for k in K) if c==1 else ''})
        all_t=[];all_v=[]
        for i in ix:
            r=rows[i];t=np.array([r['truth'][q] for q in r['pos']]);v=np.array(values(a,i));assert len(t)==len(v)
            all_t.extend(t);all_v.extend(v)
            flowstats.append({'flow':r['flow'],'class':NAMES[c],'model':a,'missing':len(t),
                'direction_wrong':int(sum(t*v<=0)),'magnitude_mae':float(np.mean(abs(abs(t)-abs(v)))),
                'correct_classifiers':','.join(k for k in K if classes(a,k,[i])[0]==c),
                'predictions':json.dumps({k:NAMES[int(classes(a,k,[i])[0])] for k in K})})
            for q,tt,vv in zip(r['pos'],t,v):
                packetrows.append({'flow':r['flow'],'class':NAMES[c],'position_zero_based':q,'model':a,'truth':int(tt),'prediction':int(vv),
                    'direction_wrong':bool(tt*vv<=0),'magnitude_error':int(abs(abs(tt)-abs(vv))),
                    'signed_error':int(abs(tt-vv)),'magnitude_bias':int(abs(vv)-abs(tt))})
        t=np.array(all_t);v=np.array(all_v);mt=abs(t);mv=abs(v)
        for tag,mask in [('all',np.ones(len(t),bool)),('short_le128',mt<=128),('middle129_1023',(mt>128)&(mt<1024)),('long_ge1024',mt>=1024)]:
            tt=t[mask];vv=v[mask]
            packetstats.append({'class':NAMES[c],'model':a,'stratum':tag,'positions':len(tt),
                'direction_wrong':int(sum(tt*vv<=0)),'direction_accuracy':float(100*np.mean(tt*vv>0)) if len(tt) else None,
                'magnitude_mae':float(np.mean(abs(abs(tt)-abs(vv)))) if len(tt) else None,
                'signed_mae':float(np.mean(abs(tt-vv))) if len(tt) else None,
                'signed_exact':int(sum(tt==vv)),'abs_exact':int(sum(abs(tt)==abs(vv))),
                'underestimated':int(sum(abs(vv)<abs(tt))),'overestimated':int(sum(abs(vv)>abs(tt))),
                'mean_magnitude_bias':float(np.mean(abs(vv)-abs(tt))) if len(tt) else None,
                'short_to_long':int(sum((abs(tt)<=128)&(abs(vv)>=1024))),
                'long_to_short':int(sum((abs(tt)>=1024)&(abs(vv)<=128)))})
writecsv('class_summary198.csv',summary);writecsv('packet_strata.csv',packetstats);writecsv('packets.csv',packetrows);writecsv('flows.csv',flowstats)

# Oracle repairs alter missing slots of the 80 target-class flows only. No fit or selection.
target=np.flatnonzero(np.isin(y,[1,4])).tolist();targetrows=[rows[i] for i in target]
protocol={'utc':p.api.utc(),'target_flows':len(target),'indices':target,'mask_rate':.3,
    'arms':['direction_only','magnitude_only','short_magnitude_only','middle_magnitude_only','long_magnitude_only','both'],
    'strata':'By TRUE absolute packet length: short<=128, middle129..1023, long>=1024. All three strata reported; no threshold tuning.',
    'direction_only':'True sign times J absolute length.','magnitude_only':'J sign times true absolute length.',
    'both':'True signed length at missing positions of targeted flows; other observations unchanged.',
    'scope':'Same 80 flows/positions and frozen classifiers; others retain original J classification for full-cohort metrics.',
    'boundary':'Oracle diagnostics using truth and target class, not deployable predictions or model performance. Missing values repaired only. No training, API, full dataset run, or threshold change.'}
if (OUT/'repair_protocol.json').exists():
    old=p.api.read(OUT/'repair_protocol.json');assert {k:v for k,v in old.items() if k!='utc'}=={k:v for k,v in protocol.items() if k!='utc'}
else:p.api.save(OUT/'repair_protocol.json',protocol)
repair_stats=[];repair_events=[]
original={k:classes('J',k,target) for k in K};clean=p.api.read(SRC/'clean_predictions.json')['classes']
for arm in ['baseline_replay',*protocol['arms']]:
    path=OUT/(arm+'_predictions.json')
    if path.exists():saved=p.api.read(path);preds=saved['classes']
    else:
        seqs=[]
        for i in target:
            r=rows[i];s=r['truth'].copy()
            for pos,val in zip(r['pos'],values('J',i)):
                truth=r['truth'][pos];mag=abs(truth)
                if arm=='direction_only':val=int(np.sign(truth))*abs(val)
                elif arm=='magnitude_only':val=int(np.sign(val))*mag
                elif arm=='short_magnitude_only' and mag<=128:val=int(np.sign(val))*mag
                elif arm=='middle_magnitude_only' and 128<mag<1024:val=int(np.sign(val))*mag
                elif arm=='long_magnitude_only' and mag>=1024:val=int(np.sign(val))*mag
                elif arm=='both':val=truth
                s[pos]=val
            assert all(s[q]==r['truth'][q] for q in range(len(s)) if q not in r['pos'])
            seqs.append(s)
        preds=p.exp.classify(seqs);p.api.save(path,{'indices':target,'classes':preds})
    for k in K:
        if arm=='baseline_replay':assert np.array_equal(preds[k],original[k]),k
        if arm=='both':assert preds[k]==[clean[k][i] for i in target],k
    for c in (4,1):
        local=np.flatnonzero(y[target]==c);rescued=lost=correct=before=toWeibo=0
        for k in K:
            pp=np.array(preds[k]);base=original[k];yy=y[target]
            for z in local:
                rescued+=int(base[z]!=c and pp[z]==c);lost+=int(base[z]==c and pp[z]!=c)
                correct+=int(pp[z]==c);before+=int(base[z]==c);toWeibo+=int(c==1 and pp[z]==0)
                repair_events.append({'arm':arm,'class':NAMES[c],'flow':rows[target[z]]['flow'],'classifier':k,'before':NAMES[int(base[z])],'after':NAMES[int(pp[z])],
                    'rescued':bool(base[z]!=c and pp[z]==c),'newly_wrong':bool(base[z]==c and pp[z]!=c)})
        repair_stats.append({'arm':arm,'class':NAMES[c],'flows':len(local),'baseline_correct_events':before,'correct_events':correct,
            'rescued_events':rescued,'newly_wrong_events':lost,'recall_pct':100*correct/(6*len(local)),'SMB_to_Weibo_events':toWeibo if c==1 else ''})
    print('ORACLE',arm,repair_stats[-2:],flush=True)
writecsv('oracle_repair_summary.csv',repair_stats);writecsv('oracle_repair_events.csv',repair_events)

# Every source artifact must remain unchanged.
for path,h in initial.items():assert p.api.sha(Path(path).read_bytes())==h,path
p.api.save(OUT/'audit.json',{'passed':True,'source_hashes':initial,'no_fitting':True,'same_target_indices_and_masks':True,
    'baseline_replay_matches_original':True,'both_repair_matches_clean':True,'decomposition_sums_match':True})
p.api.save(OUT/'analysis.json',{'class_summary198':summary,'decomposition200':decomposition,'packet_strata':packetstats,'oracle_repairs':repair_stats})
def table(records,cols):
    header='<tr>'+''.join('<th>'+html.escape(label)+'</th>' for key,label in cols)+'</tr>'
    def fmt(v):return f'{v:.2f}' if isinstance(v,(float,np.floating)) else str(v)
    return '<table>'+header+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r[key]))+'</td>' for key,label in cols)+'</tr>' for r in records)+'</table>'
text='<!doctype html><html lang="zh"><meta charset="utf-8"><title>联合头错误诊断</title><style>body{font:16px Arial,"Microsoft YaHei";margin:35px;line-height:1.7}table{border-collapse:collapse;border-top:2px solid;border-bottom:2px solid;margin:25px 0}th{border-bottom:1px solid}th,td{padding:8px;text-align:left}</style><h1>200条开发流：准确率与macro-F1分歧诊断</h1><p>R=同预算普通回归头；J=新方向条件包长头；B=现有双路候选数值头；TFI=本地TFI。无重训、无全量、无API。200条分解比较R/J；含B的分类表按共同198条计算。Miuref两条、SMB78条均完整覆盖。B使用历史可见时间，属于系统参考，R/J才是同条件结构对照。</p>'
text+='<h2>准确率与macro-F1变化的来源：J−R，完整200条</h2><p>“正确次数”是6个分类器×流的判断事件，不是独立新增流数。Accuracy按流数加权，macro-F1先对各类等权，再对六分类器等权。</p>'+table(decomposition,[('class','类别'),('support','流数'),('R_correct_events','R正确次数'),('J_correct_events','J正确次数'),('accuracy_contribution_pp','准确率贡献pp'),('R_mean_class_f1','R类F1'),('J_mean_class_f1','J类F1'),('macro_f1_contribution_pp','macro-F1贡献pp')])
text+='<h2>重点类别：四方法对比（共同198条计算F1）</h2>'+table(summary,[('class','类别'),('model','方法'),('support_flows','流数'),('correct_events_out_of_6N','正确次数'),('mean_recall','平均召回%'),('mean_precision','平均精确率%'),('mean_f1_paired198','平均类F1%'),('SMB_to_Weibo_events','SMB→Weibo次数')])
text+='<h2>缺包重建：方向和幅值分开看</h2><p>短包≤128，中间包129–1023，长包≥1024；分组依据真值，只用于诊断。MAE单位字节。所有方法均使用同一批Miuref/SMB缺失位置。</p>'+table([r for r in packetstats if r['stratum']=='all'],[('class','类别'),('model','方法'),('positions','缺包位置'),('direction_wrong','方向错误数'),('magnitude_mae','幅值MAE'),('signed_mae','带符号MAE'),('short_to_long','短→长'),('long_to_short','长→短')])
text+='<details><summary>按真实包长分组展开</summary>'+table(packetstats,[('class','类别'),('model','方法'),('stratum','包长组'),('positions','位置数'),('direction_wrong','方向错误'),('magnitude_mae','幅值MAE'),('mean_magnitude_bias','平均偏差:负为补短')])+'</details>'
text+='<h2>仅修复一个因素的诊断推理</h2><p>只修改Miuref/SMB缺包位置，真值仅用于oracle诊断。direction_only只修方向；magnitude_only只修幅值；short/middle/long_magnitude_only分别只修该真值包长组的幅值；both全部恢复。救回/新错按分类器×流计数，不可当作可部署模型收益。</p>'+table(repair_stats,[('arm','诊断修改'),('class','类别'),('baseline_correct_events','原判对次数'),('correct_events','修复后判对次数'),('rescued_events','救回'),('newly_wrong_events','原对变错'),('recall_pct','召回%'),('SMB_to_Weibo_events','SMB→Weibo次数')])
text+='<p>Miuref仅2条，不能外推为稳定总体结论。Oracle修复的交互可能非线性，单因素救回不等于已找到可部署修复规则。所有原始预测和模型文件哈希保持一致。</p><p><a href="flows.csv">逐流分类结果</a> · <a href="packets.csv">逐包真值与预测</a> · <a href="oracle_repair_events.csv">逐条修复事件</a> · <a href="per_classifier_class198.csv">六分类器逐类指标</a></p></html>'
(OUT/'report.html').write_text(text,encoding='utf-8')
print('DONE',str(OUT/'report.html'),flush=True)
