"""Paired 198-flow historical dual baselines, separate from the full 200-flow pilot."""
import csv,html
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
import joint_head_pilot as j

p=j.p;out=j.OUT;rows=p.api.read(out/'panels.json')['dev']
bp=p.api.read(out/'current_B_198_predictions.json');ix=bp['indices'];subset=[rows[i] for i in ix]
assert len(ix)==198 and len(set(r['flow'] for r in subset))==198
values={};byid={r['flow']:{} for r in subset}
archive=j.ROOT/'outputs/d4_full_aligned_20261113/windows_30.jsonl'
for line in archive.open():
    rr=__import__('json').loads(line)
    if rr['id'] in byid:
        for pos,v in zip(rr['q'],rr['pred'],strict=True):
            assert rr['a']+pos not in byid[rr['id']]
            byid[rr['id']][rr['a']+pos]=v
for r in subset:assert set(byid[r['flow']])==set(r['pos'])
vals=[[byid[r['flow']][pos] for pos in r['pos']] for r in subset]
gp=out/'generator4B_198_predictions.json'
if not gp.exists():
    stats=p.metrics(subset,vals);p.api.save(gp,{'indices':ix,'values':vals,'classes':stats.pop('classes')});p.api.save(out/'generator4B_198_scores.json',stats)
results={'current_B':p.api.read(out/'current_B_198_scores.json'),'generator4B':p.api.read(out/'generator4B_198_scores.json')}
for a in ('R','J','Luna','TFI','clean'):
    path=out/(a+'_predictions.json') if a not in ('Luna','TFI') else j.PREV/'dev'/(('P0' if a=='Luna' else 'TFI')+'_predictions.json')
    saved=p.api.read(path)
    vs=[[r['truth'][pos] for pos in r['pos']] for r in subset] if a=='clean' else [saved['values'][i] for i in ix]
    classes={k:[v[i] for i in ix] for k,v in saved['classes'].items()}
    s=p.metrics(subset,vs,classes);s.pop('classes');results[a]=s
names={'current_B':'现有双路 B 候选数值头','generator4B':'方向对齐双路4B第1000步整串生成','R':'本轮同预算回归头 R','J':'本轮方向条件包长头 J','Luna':'Luna 原提示词','TFI':'本地 TFI','clean':'完整真实流'}
columns=['方法（相同198条）','Accuracy (%)','macro-F1 (%)','包长 MAE']
table=[[names[a],s['mean_accuracy'],s['mean_macro_f1'],s['magnitude_mae']] for a,s in results.items()]
with (out/'historical_paired198.csv').open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows([columns,*table])
checks={}
for a,file in [('current_B','current_B_198_predictions.json'),('generator4B','generator4B_198_predictions.json')]:
    saved=p.api.read(out/file);y=[r['label'] for r in subset]
    truth=np.array([r['truth'][q] for r in subset for q in r['pos']]);pred=np.array([v for vs in saved['values'] for v in vs])
    assert abs(float(np.abs(np.abs(truth)-np.abs(pred)).mean())-results[a]['magnitude_mae'])<1e-9
    for k,c in saved['classes'].items():
        assert abs(accuracy_score(y,c)*100-results[a]['classifiers'][k]['accuracy'])<1e-9
        assert abs(f1_score(y,c,labels=range(6),average='macro',zero_division=0)*100-results[a]['classifiers'][k]['macro_f1'])<1e-9
    checks[a]=True
p.api.save(out/'historical_paired198.json',{'results':results,'indices':ix,'mask_identity_passed':True,'metric_checks':checks,
    'missing_flows':[r['flow'] for i,r in enumerate(rows) if i not in ix],
    'source_sha256':{str(path):p.api.sha(path.read_bytes()) for path in [archive,j.ROOT/'outputs/d4_numeric_iat_20261113/direct_30.npz']},
    'boundary':'Historical dual methods used visible timing and verified endpoint-direction metadata. New pilot/Luna/TFI use signed lengths only. This is an archival system comparison, not a head-only ablation.'})
def tab(rs):return '<table>'+''.join('<tr>'+''.join('<'+('th' if i==0 else 'td')+'>'+html.escape(f'{v:.2f}' if isinstance(v,float) else str(v))+'</'+('th' if i==0 else 'td')+'>' for v in row)+'</tr>' for i,row in enumerate(rs))+'</table>'
body='<h2>补充：与现有双路的相同198条对照</h2><p>旧双路归档缺少 Virut.pcap#167 和 Neris.pcap#11200。因此以下所有方法都限定为相同198条、相同缺包位置，不能与上方200条直接相减。历史双路使用可见时间和端点方向元数据；本轮无时间输入，这里是系统参考，不是只换输出头的消融。</p>'+tab([columns,*table])
recalls=[['类别','支持流数','R平均召回率','J平均召回率','Luna平均召回率','TFI平均召回率']]
allscore=p.api.read(out/'comparison.json');labels=['Weibo','SMB','Htbot','Virut','Miuref','其他合并类']
for c,name in enumerate(labels):
    recalls.append([name,sum(r['label']==c for r in rows),*[float(np.mean([z['recall_by_class'][c] for z in allscore[a]['classifiers'].values()])) for a in ('R','J','Luna','TFI')]])
with (out/'class_recall200.csv').open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(recalls)
body+='<h2>200条开发组的逐类召回</h2>'+tab(recalls)+'<p>新头的准确率收益主要伴随 Weibo 召回上升；SMB 未改善，Miuref 等类别退步。Miuref仅2条，结果不作稳定泛化估计。方向和包长总体误差不能代替宏平均分类指标。</p>'
path=out/'comparison.html';text=path.read_text(encoding='utf-8');text=text[:text.index('<h2>补充：')] if '<h2>补充：' in text else text.removesuffix('</html>');path.write_text(text+body+'</html>',encoding='utf-8')
with (out/'CHANGELOG.txt').open('a',encoding='utf-8') as f:
    f.write('\n补充历史双路：相同198条和相同掩码比较，独立列示，不能混入200条表。缺少流ID：Virut.pcap#167、Neris.pcap#11200。历史双路使用可见时间，当前试验无时间。\n结论：J虽准确率上涨，但macro-F1下降，未通过扩大条件。两个训练头检查点重推逐包一致，指标独立复核通过；未开全量或新确认训练。\n')
print(__import__('json').dumps({a:{k:s[k] for k in ('mean_accuracy','mean_macro_f1','magnitude_mae')} for a,s in results.items()},ensure_ascii=False))
