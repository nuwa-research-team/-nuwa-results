"""Score the frozen paired B prompt experiment without additional API calls."""
import html
import json
from collections import Counter
from pathlib import Path
import numpy as np
import experiment as e
import gemini_compare_v2 as g
import gemini_prompt_B as b


def enrich(scores, rows, values, indices):
    t=np.array([rows[i]['truth'][p] for i in indices for p in rows[i]['pos']])
    v=np.array([x for i in indices for x in (values[i] if values[i] is not None else [0]*len(rows[i]['pos']))])
    scores=dict(scores)
    scores.update(signed_mae=float(np.abs(t-v).mean()),
        direction_accuracy=100*float((np.sign(t)==np.sign(v)).mean()),
        mean_macro_f1=float(np.mean([s['macro_f1'] for s in scores['classifiers'].values()])))
    return scores


def table(head, rows):
    esc=lambda x:html.escape(str(x))
    return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'


def main():
    assert (b.DEST/'api_complete.json').exists()
    for path,digest in b.read(b.DEST/'execution_manifest.json')['files'].items():
        assert g.sha(Path(path).read_bytes())==digest, 'Frozen execution file changed: '+path
    rows,bodies=b.prepare()
    selection=b.read(b.DEST/'selection.json');model=selection['winner']
    a_flows={r['flow'] for r in e.read('local_A.json')}
    subsets={'all':list(range(1000)), 'flow_absent_from_A':[i for i,r in enumerate(rows) if r['flow'] not in a_flows]}
    assert len(subsets['flow_absent_from_A'])==933
    baseline=e.read('B_local_tfi_predictions.json')
    seqs=[]
    for row,values in zip(rows,baseline['values']):
        assert len(values)==len(row['pos'])
        seq=row['truth'].copy()
        for p,v in zip(row['pos'],values):seq[p]=v
        seqs.append(seq)
    recheck=e.classify(seqs)
    assert recheck==baseline['classes'], 'Frozen classifier baseline changed'
    g.save(b.DEST/'classifier_recheck.json',{'utc':b.utc(),'draws':1000,'classifiers':6,'identical':True})
    scores=e.read('B_local_tfi_scores.json')
    results={'local_TFI':{s:enrich(scores[s],rows,baseline['values'],ix) for s,ix in subsets.items()}}
    api={}
    for variant in b.VARIANTS:
        recs=[b.read(b.DEST/variant/f'{i:04d}.json') for i in range(1000)]
        values=[]
        for i,(r,body) in enumerate(zip(recs,bodies[variant])):
            assert r['model']==model and r['variant']==variant and r['draw']==i
            assert r['request_sha256']==g.sha(g.encoded({'endpoint':b.endpoint(model),'body':body}))
            vs,err=g.validate(r['response'],len(rows[i]['pos'])) if r['http_status']==200 else (None,'http_failure')
            assert vs==r['values'] and err==r['validation_error']
            values.append(vs)
        method='gemini_prompt_v1_'+variant
        e.score('B',method,values)
        scores=e.read('B_'+method+'_scores.json')
        results[variant]={s:enrich(scores[s],rows,values,ix) for s,ix in subsets.items()}
        api[variant]={'http_success':sum(r['http_status']==200 for r in recs),
            'http_attempts':sum(len(r['attempts']) for r in recs),
            'errors':dict(Counter(r['validation_error'] for r in recs if r['validation_error'])),
            'model_versions':dict(Counter(r['response'].get('modelVersion','unknown') for r in recs)),
            'usage':{k:sum(r['response'].get('usageMetadata',{}).get(k,0) for r in recs) for k in ('promptTokenCount','candidatesTokenCount','thoughtsTokenCount','totalTokenCount')}}
    metrics=('mean_accuracy','mean_macro_f1','magnitude_mae','signed_mae','packet_exact_pct','direction_accuracy','valid_pct')
    deltas={s:{ref:{k:results['behavior'][s][k]-results[ref][s][k] for k in metrics} for ref in ('plain','local_TFI')} for s in subsets}
    report={'utc':b.utc(),'teacher':model,'selection':selection,'results':results,'behavior_minus_reference':deltas,'API':api,
        'interpretation':'Accuracy/F1/hit deltas in percentage points; MAE deltas in bytes (negative is better). Historical test-pool development comparison, not independent final testing.'}
    g.save(b.DEST/'comparison.json',report)
    parts=[]
    for subset,ix in subsets.items():
        parts.append('<h2>'+('全部 1,000 次抽样（967 条唯一流）' if subset=='all' else '剔除与 A 重合的流：933 次抽样')+'</h2>')
        parts.append(table(['方法','Accuracy %','macro-F1 %','幅值 MAE/字节','带方向 MAE/字节','精确命中 %','方向正确 %','有效 JSON %'],
            [[m,*[f'{results[m][subset][k]:.3f}' for k in metrics]] for m in results]))
        parts.append(table(['分类器','本地 TFI Acc / F1','原提示词 Acc / F1','行为提示词 Acc / F1'],
            [[kind.upper(),*[f'{results[m][subset]["classifiers"][kind]["accuracy"]:.3f} / {results[m][subset]["classifiers"][kind]["macro_f1"]:.3f}' for m in results]] for kind in e.KINDS]))
    content='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 paired prompt comparison</title>
    <style>body{font:16px Arial,"Microsoft YaHei",sans-serif;max-width:1450px;margin:35px auto;padding:20px;color:#172332}table{border-collapse:collapse;width:100%;margin:24px 0;border-top:2px solid;border-bottom:2px solid}th{border-bottom:1px solid;text-align:left}td,th{padding:10px}p{line-height:1.8}</style>'''
    content+='<h1>D4 第二批：原提示词与可见行为提示词</h1><p>教师：'+html.escape(model)+'。固定 30% 缺包、10,060 个隐藏位置，两组使用完全相同的流、缺包位置、输出约束和六个冻结分类器。</p>'
    content+='<p>唯一改动：加入可见包的带方向包长频次、正负方向数量、相邻可见包同向/换向数量及对应指令。没有时间间隔输入，没有真实缺失包长或类别标签。所有失败样本保留，缺失位置回填 0。</p>'+''.join(parts)
    content+='<p>Accuracy 与 macro-F1 为六分类器等权平均；各分类器数值分别列出。该批来自历史测试池，用于提示词开发；包含有放回重复与 A 组重合，不作为独立最终测试。对照是本地 TFI 复现权重，不是原文数字。</p></html>'
    (b.DEST/'comparison.html').write_text(content,encoding='utf-8')
    change=b.read(b.DEST/'change_log.json');change.update(status='complete',completed_utc=b.utc(),teacher=model,results=deltas)
    g.save(b.DEST/'change_log.json',change)
    print(json.dumps({'teacher':model,'behavior_minus_reference':deltas}),flush=True)


if __name__=='__main__':main()
