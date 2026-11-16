"""Independent saved-output metric and frozen-weight check, no fitting."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
import joint_head_tail as t

out=t.OUT;read=t.p.api.read;checks={};allresults={}
for stage in ('dev','confirm'):
    rows=read(t.BASE/'panels.json')['dev'] if stage=='dev' else read(out/'confirm/rows.json')
    results={};y=[r['label'] for r in rows];truth=np.array([r['truth'][pos] for r in rows for pos in r['pos']])
    for path in sorted((out/stage).glob('*_scores.json')):
        arm=path.name.removesuffix('_scores.json');s=read(path);saved=read(out/stage/(arm+'_predictions.json'))
        pred=np.array([v for vs in saved['values'] for v in vs]);assert len(pred)==len(truth)
        assert np.isfinite(pred).all() and ((abs(pred)>=1)&(abs(pred)<=1600)).all()
        assert abs(np.abs(abs(pred)-abs(truth)).mean()-s['magnitude_mae'])<1e-9
        for c,cs in saved['classes'].items():
            assert abs(100*accuracy_score(y,cs)-s['classifiers'][c]['accuracy'])<1e-9
            assert abs(100*f1_score(y,cs,average='macro',labels=range(6),zero_division=0)-s['classifiers'][c]['macro_f1'])<1e-9
        results[arm]=s
    allresults[stage]=results;checks[stage]={'flows':len(rows),'slots':len(truth),'methods_verified':list(results)}
for path,h in read(out/'classifier_hashes_before_evaluation.json').items():assert t.sha(path)==h
checks['six_classifier_heads_unchanged']=True
checks['passed']=True
t.save('independent_metrics_audit.json',checks)

summary=[['分组','方法','Accuracy %','macro-F1 %','幅值MAE/字节','Miuref召回 %','Miuref F1 %','Miuref判对次数/12']]
for stage,rs in allresults.items():
    for arm,s in rs.items():
        f=s['focal']['Miuref'];summary.append([stage,arm,s['mean_accuracy'],s['mean_macro_f1'],s['magnitude_mae'],f['mean_recall'],f['mean_f1'],f['correct_events']])
with (out/'dev_confirm_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(summary)
details=[['分组','方法','分类器','Accuracy %','macro-F1 %','Miuref召回 %','SMB召回 %']]
for stage,rs in allresults.items():
    for arm,s in rs.items():
        for c,ss in s['classifiers'].items():details.append([stage,arm,c,ss['accuracy'],ss['macro_f1'],ss['recall_by_class'][4],ss['recall_by_class'][1]])
with (out/'dev_confirm_classifiers.csv').open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(details)
conclusion=('本轮只训练一个扩大包长类别的新头，使用原D4训练子集2048条流、三种固定掩码、12轮、seed17。编码器和六个分类器未训练。'
    '\n改动一：64高频幅值扩为训练目标全部1397幅值，保留OTHER。改动二：从先定方向改为比较两个方向下的联合概率，不调阈值。'
    '\n开发200：组合版相对原联合头 Accuracy +0.58个百分点、macro-F1 +5.51个百分点、幅值MAE -20.64字节；Miuref从2/12到4/12。通过预登记门槛后进行了固定确认200。'
    '\n确认200：组合版相对原联合头 Accuracy -0.50个百分点、macro-F1 -5.53个百分点、幅值MAE -23.56字节；Miuref从11/12降到8/12。分类收益没有复现，不扩大、不替换原模型。'
    '\n相对TFI：开发组合版 Accuracy +1.58个百分点、macro-F1 -5.49个百分点、MAE -1.85字节；确认 Accuracy -0.83个百分点、macro-F1 -6.13个百分点、MAE约持平（-0.019字节）。'
    '\n结论：扩大候选改善总体幅值误差，但包长覆盖不等于低频包长学会；联合MAP并未稳定解决方向与包长的联合判断。不能把开发Miuref仅2条的恢复外推为泛化收益。'
    '\n后续若继续，应先只读检查确认组Miuref的变化，区分低频包长概率仍偏低和联合MAP偏向高频尖峰；本轮不据确认结果调阈值或重新拟合。'
    '\n限制：单种子小实验；扩大类别同时增加参数；OTHER与单点幅值概率粒度不同。确认集属于既有历史测试池中的固定留出，不是独立采集或最终测试。')
(out/'结论与改动记录.txt').write_text(conclusion,encoding='utf-8')
with (out/'CHANGELOG.txt').open('a',encoding='utf-8') as f:f.write('\n'+conclusion)
path=out/'report.html';body=path.read_text(encoding='utf-8').removesuffix('</html>')
body+='<h2>固定确认组：完整对照与结论</h2>'+t.table(summary)+'<p>'+conclusion.replace('\n','</p><p>')+'</p>'+t.table(details)+'</html>'
path.write_text(body,encoding='utf-8')
t.save('final_decision.json',{'confirmation_classification_improvement_reproduced':False,'promote_new_head':False,'full_run':False,'further_fit':False})
print(json.dumps(checks,ensure_ascii=False))
