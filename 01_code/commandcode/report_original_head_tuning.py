"""Summarize completed tuning results only; no inference or training."""
import csv,json,html
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=root/'outputs/d4_teacher_api_20261114/joint_head_dev200_20261116/original_head_tuning'
d=json.loads((out/'comparison.json').read_text());decision=json.loads((out/'selection.json').read_text());arm=decision['selected']
rows=[['分组','配置','分类器','Accuracy %','macro-F1 %','Miuref召回 %','SMB召回 %']]
for stage,rs in d.items():
    for a,s in rs.items():
        for c,z in s['classifiers'].items():rows.append([stage,a,c,z['accuracy'],z['macro_f1'],z['recall_by_class'][4],z['recall_by_class'][1]])
with (out/'classifier_details.csv').open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(rows)
s=d['dev'][arm];b=d['dev']['J-original'];t=d['dev']['TFI']
diff={k:s[k]-b[k] for k in ('mean_accuracy','mean_macro_f1','magnitude_mae','signed_mae','direction_pct')}
notes=('本轮停止扩大包长候选与联合MAP路线，保持原64类+OTHER头、先方向后包长解码，仅调训练参数。\n'
       '完成5条训练轨迹，每条24轮，检查12/24轮，共10个候选；C0第12轮与历史原头所有权重逐项完全相等。\n'
       '选参只使用原D4验证集340条，训练仍仅原train子集2048条；340条中含原256条，新增84条均来自原val。原200条开发流在配置选择冻结后报告，未用来排序候选。\n'
       '满足验证Accuracy和macro-F1不降且幅值MAE不增的配置是C0-e24与C4-e12；按预登记Accuracy优先规则选中C0-e24。\n'
       '选中参数：学习率0.001，weight_decay=0.01，方向CE权重1，幅值分类CE权重1，回归SmoothL1权重0.2，batch512，seed17，24轮。与原头唯一区别为训练12轮延长到24轮。\n'
       f'原200条开发流三成丢包的变化：Accuracy {diff["mean_accuracy"]:+.4f}个百分点；macro-F1 {diff["mean_macro_f1"]:+.4f}个百分点；幅值MAE {diff["magnitude_mae"]:+.4f}字节；带符号MAE {diff["signed_mae"]:+.4f}字节；方向正确率 {diff["direction_pct"]:+.4f}个百分点。\n'
       'Miuref判对次数从2/12到3/12，普通回归头7/12、TFI9/12；该类只有2条流。SMB方向错30/758降到16/758，类别召回16.45%到16.88%。\n'
       f'相对本地TFI：Accuracy {s["mean_accuracy"]-t["mean_accuracy"]:+.4f}个百分点；macro-F1 {s["mean_macro_f1"]-t["mean_macro_f1"]:+.4f}个百分点；幅值MAE {s["magnitude_mae"]-t["magnitude_mae"]:+.4f}字节。仍非全指标胜出。\n'
       '结论：保持原结构延长训练有小幅正向效果，可保留为调参候选；没有证明稳定泛化改善或超过原文。单种子，验证类比例不同，历史开发反复使用。未追加搜索、未重用确认集选参、未跑全量，冻结编码器和六个分类器全部哈希未变。')
(out/'调参结论.txt').write_text(notes,encoding='utf-8')
with (out/'CHANGELOG.txt').open('a',encoding='utf-8') as f:f.write('\n'+notes)
path=out/'report.html';text=path.read_text(encoding='utf-8').removesuffix('</html>')
text+='<h2>完成结论</h2><p>'+html.escape(notes).replace('\n','</p><p>')+'</p><p><a href="classifier_details.csv">全部配置六分类器明细</a> · <a href="调参结论.txt">中文结论</a></p></html>'
path.write_text(text,encoding='utf-8')
(out/'final_decision.json').write_text(json.dumps({'selected':arm,'keep_as_pilot_candidate':True,'all_dev_summary_metrics_improved_over_original':True,'beats_TFI_on_all_metrics':False,'new_confirmation':False,'full_run':False},ensure_ascii=False),encoding='utf-8')
print(notes)
