"""Post-run paired comparisons, no API calls and no selection changes."""
import csv,json,html
from pathlib import Path
import numpy as np
import luna_parameter_compare as c

out=c.OUT;read=c.p.api.read
assert (out/'complete.json').exists(),'Wait for complete paired evaluation'
data=read(out/'comparison.json');rows=read(out/'cohorts.json')['dev'];y=np.array([r['label'] for r in rows]);truth=np.array([r['truth'][q] for r in rows for q in r['pos']])
preds={a:read(out/'dev'/(a+'_predictions.json')) for a in c.ARMS}
base=preds['P0_repeat'];basevals=np.array([v for r,vs in zip(rows,base['values']) for v in (vs if vs is not None else [0]*len(r['pos']))])
paired={};classrows=[['方案','类别','流数','平均召回率%','判对次数(流x6分类器)']]
labels=['Weibo','SMB','Htbot','Virut','Miuref','OtherMerged']
for a,z in preds.items():
    s=data['current'][a];gained=lost=0
    for k,cs in z['classes'].items():
        old=np.array(base['classes'][k])==y;new=np.array(cs)==y
        gained+=int((~old&new).sum());lost+=int((old&~new).sum())
    vals=np.array([v for r,vs in zip(rows,z['values']) for v in (vs if vs is not None else [0]*len(r['pos']))])
    paired[a]={'classifier_flow_rescued_vs_fresh_P0':gained,'classifier_flow_newly_wrong_vs_fresh_P0':lost,
               'same_packet_output_pct_vs_fresh_P0':float(100*np.mean(vals==basevals)),
               'changed_flow_outputs_vs_fresh_P0':sum(x!=b for x,b in zip(z['values'],base['values'])),
               'exact_signed_pct':float(100*np.mean(vals==truth))}
    for lab,name in enumerate(labels):
        correct=sum(int((np.array(cs)[y==lab]==lab).sum()) for cs in z['classes'].values())
        classrows.append([c.NAMES[a],name,int((y==lab).sum()),float(np.mean([ss['recall_by_class'][lab] for ss in s['classifiers'].values()])),correct])
with (out/'class_recall.csv').open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(classrows)
historical=read(c.OLD/'dev/P0_predictions.json')
paired['fresh_P0_vs_historical']={'changed_flow_outputs':sum(a!=b for a,b in zip(base['values'],historical['values'])),
    'delta':{k:data['current']['P0_repeat'][k]-data['historical']['P0'][k] for k in ('mean_accuracy','mean_macro_f1','magnitude_mae')}}
c.save('paired_comparison.json',paired)
chosen=read(out/'selection.json')['selected'];param=data['current'][chosen] if chosen else None;prompt=data['current']['P2_repeat'];baseline=data['current']['P0_repeat']
lines=['本轮仅调用Luna API，未训练本地模型；同200条流、相同30%缺包位置、六分类器冻结。',
       '原提示词P0保持messages完全相同，只改变reasoning_effort或temperature。参数默认组为low且不显式指定temperature；所有组max_completion_tokens=4096，JSON object输出。',
       '优化提示词P2为之前按准确率规则选出的固定行为摘要版本，本轮没有继续改提示词。P0与P2本轮重新调用，历史结果单独保留。',
       f'按预登记规则选出的参数组：{chosen}。全部五组结果公开；没有在看到结果后追加参数组合。']
if param:
    delta={k:param[k]-prompt[k] for k in ('mean_accuracy','mean_macro_f1','magnitude_mae')}
    lines.append('选中参数组相对本轮P2：'+json.dumps(delta,ensure_ascii=False)+'；Accuracy/F1为百分点，幅值MAE为字节。')
    dominates=delta['mean_accuracy']>=0 and delta['mean_macro_f1']>=0 and delta['magnitude_mae']<=0
    lines.append('本轮选中参数组三项总体指标均不差于P2。' if dominates else '参数与提示词之间存在指标权衡，不能宣布选中参数组三项全部胜出。')
else:lines.append('没有参数组同时满足相对本轮原参数Accuracy提高且macro-F1不降低，不宣称已找到更优调用参数。')
lines.append('原参数同提示词重跑与历史差异：'+json.dumps(paired['fresh_P0_vs_historical'],ensure_ascii=False))
lines.extend(['这些差异包含生成随机性、平台可能的时变因素以及有限样本效应，不能仅凭一轮归因于内部推理能力。',
              '每种设置每条流一次调用；格式无效不重抽，按缺包位置填零并计入指标。只记录平台报告token，不虚构费用或未报告的reasoning tokens。',
              '平台HTTP200只证明请求被接受；上游模型身份与参数执行未独立认证。历史开发池多次使用，非独立最终测试。未新增确认集或跑全量。'])
text='\n'.join(lines);(out/'两种方案对比结论.txt').write_text(text,encoding='utf-8')
with (out/'CHANGELOG.txt').open('a',encoding='utf-8') as f:f.write('\n'+text)
path=out/'report.html';body=path.read_text(encoding='utf-8').removesuffix('</html>')
body+='<h2>配对变化与结论</h2><p>'+html.escape(text).replace('\n','</p><p>')+'</p>'+c.tab(classrows)+'</html>';path.write_text(body,encoding='utf-8')
print(json.dumps({'selected':chosen,'paired':paired},ensure_ascii=False))
