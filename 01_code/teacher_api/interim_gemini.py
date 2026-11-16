"""Read-only paired interim scoring; never changes API requests or final scores."""
import json
import html
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
import experiment as e
import gemini_compare_v2 as g

now=datetime.now(timezone.utc)
dest=g.DEST/('interim_'+now.strftime('%Y%m%dT%H%M%SZ'))
dest.mkdir()
rows,bodies=g.prepare()
records={m:{int(p.stem):json.loads(p.read_bytes()) for p in (g.DEST/m).glob('[0-9][0-9][0-9][0-9].json')} for m in g.MODELS}
ids=sorted(set.intersection(*(set(rs) for rs in records.values())))
assert ids
selected=[rows[i] for i in ids]
progress={}
values={}
for model,rs in records.items():
    recent=sum(0 <= (now-datetime.fromisoformat(r['completed_utc'])).total_seconds() < 600 for r in rs.values())
    progress[model]={'processed':len(rs),'valid':sum(r['values'] is not None for r in rs.values()),
        'http_success':sum(r['http_status']==200 for r in rs.values()),
        'errors':dict(Counter(r['validation_error'] for r in rs.values() if r['validation_error'])),
        'recent_10min_completed':recent,'eta_minutes_first_pass':(1000-len(rs))*10/recent if recent else None}
    values[model]=[]
    for i in ids:
        r=rs[i]
        endpoint=f'https://aiplatform.googleapis.com/v1/projects/{g.PROJECT}/locations/global/publishers/google/models/{model}:generateContent'
        assert r['draw']==i and r['request_sha256']==g.sha(g.encoded({'endpoint':endpoint,'body':bodies[i]}))
        v,err=g.validate(r['response'],len(rows[i]['pos'])) if r['http_status']==200 else (None,'http_failure')
        assert v==r['values'] and err==r['validation_error']
        values[model].append(v)
baseline=e.read('A_local_tfi_predictions.json')
values={'local_TFI':[baseline['values'][i] for i in ids],**values}
g.save(dest/'snapshot.json',{'utc':now.isoformat(),'draw_ids':ids,'progress':progress,'values':values,
    'records_sha256':{m:{str(i):g.sha(g.encoded(rs[i])) for i in ids} for m,rs in records.items()},
    'source_A_sha256':g.sha((e.OUT/'local_A.json').read_bytes()),
    'code_sha256':g.sha(Path(__file__).read_bytes())})
print(json.dumps({'snapshot':str(dest),'common_flows':len(ids),'progress':progress}),flush=True)
results={}
truth=np.array([r['truth'][p] for r in selected for p in r['pos']])
y=np.array([r['label'] for r in selected])
for model,vs in values.items():
    seqs=[];flat=[]
    for row,v in zip(selected,vs):
        seq=row['truth'].copy()
        fill=v if v is not None else [0]*len(row['pos'])
        assert len(fill)==len(row['pos'])
        for p,x in zip(row['pos'],fill):seq[p]=x
        assert all(seq[j]==row['truth'][j] for j in range(len(seq)) if j not in row['pos'])
        seqs.append(seq);flat.extend(fill)
    preds=e.classify(seqs)
    if model=='local_TFI':
        assert all(preds[k]==[baseline['classes'][k][i] for i in ids] for k in e.KINDS), 'Subset baseline changed'
    classifiers={k:{'accuracy':100*float((np.array(p)==y).mean()),
        'macro_f1':100*float(f1_score(y,p,average='macro',labels=range(6),zero_division=0))} for k,p in preds.items()}
    flat=np.array(flat)
    results[model]={'n':len(ids),'hidden':len(truth),'valid_n':sum(v is not None for v in vs),
        'mean_accuracy':float(np.mean([s['accuracy'] for s in classifiers.values()])),
        'mean_macro_f1':float(np.mean([s['macro_f1'] for s in classifiers.values()])),
        'magnitude_mae':float(np.abs(np.abs(truth)-np.abs(flat)).mean()),
        'signed_mae':float(np.abs(truth-flat).mean()),
        'direction_accuracy':100*float((np.sign(truth)==np.sign(flat)).mean()),
        'exact_pct':100*float((truth==flat).mean()),'classifiers':classifiers}
    g.save(dest/(model+'_predictions.json'),preds)
    print(json.dumps({'model':model,**results[model]}),flush=True)
g.save(dest/'results.json',{'snapshot_utc':now.isoformat(),'progress':progress,'draw_ids':ids,
    'label_counts':dict(Counter(map(int,y))),'results':results})
def table(head,rs):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rs)+'</tbody></table>'
summary=[[m,s['valid_n'],*[f'{s[k]:.2f}' for k in ('mean_accuracy','mean_macro_f1','magnitude_mae','signed_mae','direction_accuracy')]] for m,s in results.items()]
detail=[[k.upper(),*[f'{s["classifiers"][k]["accuracy"]:.2f} / {s["classifiers"][k]["macro_f1"]:.2f}' for s in results.values()]] for k in e.KINDS]
page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 Gemini 临时配对比较</title><style>body{{font:16px Arial,"Microsoft YaHei",sans-serif;max-width:1400px;margin:32px auto;padding:20px;color:#172332}}table{{border-collapse:collapse;width:100%;border-top:2px solid;border-bottom:2px solid;margin:25px 0}}th{{border-bottom:1px solid}}th,td{{text-align:left;padding:12px}}p{{line-height:1.8}}</style>
<h1>D4 Gemini 临时配对比较：30% 缺包，{len(ids)} 条共同流</h1><p>快照 UTC：{now.isoformat(timespec='seconds')}。只比较三组均已有终态记录的相同流、相同缺包位置；未完成调用不进入本次快照。接口失败和格式无效均保留并回填 0，之后统一恢复接口失败可能改变结果。本表不是完整 1,000 条排名，也不是独立最终测试。</p>
{table(['模型','有效回答数','六分类器平均准确率 %','平均 macro-F1 %','包长幅值 MAE 字节','带方向 MAE 字节','方向正确率 %'],summary)}
<h2>六个冻结分类器：准确率 / macro-F1（%）</h2>{table(['分类器',*results],detail)}
<p>本地 TFI 已在相同子集重新推理并与原有预测逐条核对一致。TFI 和分类器是本地复现/适配权重。本次不更换教师、不调整提示词，等待完整批次后再选择。</p></html>'''
(dest/'comparison.html').write_text(page,encoding='utf-8')
print('REPORT '+str(dest/'comparison.html'),flush=True)
