"""Same frozen classifier evaluation for cross-platform A and paired-prompt B."""
import html
import json
import sys
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import experiment as e
import gemini_compare_v2 as g
import cross_teacher_chain as c

def enrich(stats,rows,values):
    truth=np.array([r['truth'][p] for r in rows for p in r['pos']])
    pred=np.array([x for r,vs in zip(rows,values) for x in (vs if vs is not None else [0]*len(r['pos']))])
    return dict(stats,mean_macro_f1=float(np.mean([s['macro_f1'] for s in stats['classifiers'].values()])),
        signed_mae=float(np.abs(truth-pred).mean()),direction_accuracy=100*float((np.sign(truth)==np.sign(pred)).mean()))

def baseline(phase,rows):
    values=e.read(phase+'_local_tfi_predictions.json')['values']
    stats=e.read(phase+'_local_tfi_scores.json')
    result={'all':enrich(stats['all'],rows,values)}
    if phase=='B':
        a={r['flow'] for r in e.read('local_A.json')};ids=[i for i,r in enumerate(rows) if r['flow'] not in a]
        result['flow_absent_from_A']=enrich(stats['flow_absent_from_A'],[rows[i] for i in ids],[values[i] for i in ids])
    return result

def evaluate(phase,variant,model,rows,bodies):
    values=[];records=[]
    for row,google_body in zip(rows,bodies):
        rec=c.read(c.DEST/phase/variant/f"{row['draw']:04d}.json")
        endpoint,body=c.request_for(model,google_body)
        assert rec['draw']==row['draw'] and rec['model']==model
        assert rec['request_sha256']==g.sha(g.encoded({'endpoint':endpoint,'body':body}))
        v,err=c.validate(model,rec['response'],len(row['pos'])) if rec['http_status']==200 else (None,'http_failure')
        assert v==rec['values'] and err==rec['validation_error']
        values.append(v);records.append(rec)
    method='cross_v1_'+variant
    e.score(phase,method,values)
    raw=e.read(phase+'_'+method+'_scores.json')
    classes=e.read(phase+'_'+method+'_predictions.json')['classes']
    result={'all':enrich(raw['all'],rows,values)}
    if phase=='B':
        a={r['flow'] for r in e.read('local_A.json')};ids=[i for i,r in enumerate(rows) if r['flow'] not in a]
        result['flow_absent_from_A']=enrich(raw['flow_absent_from_A'],[rows[i] for i in ids],[values[i] for i in ids])
    result['model']=model
    result['http_success']=sum(r['http_status']==200 for r in records)
    result['errors']=dict(Counter(r['validation_error'] for r in records if r['validation_error']))
    result['http_attempts']=sum(len(r['attempts']) for r in records)
    result['usage']={}
    for r in records:
        for k,v in r['response'].get('usage',r['response'].get('usageMetadata',{})).items():
            if type(v) in (int,float):result['usage'][k]=result['usage'].get(k,0)+v
    result['reported_models']=dict(Counter(r['response'].get('model',r['response'].get('modelVersion','unknown')) for r in records))
    labels=[r['label'] for r in rows]
    g.save(c.DEST/phase/variant/'class_diagnostics.json',{k:{
        'confusion_matrix_true_rows_pred_columns':confusion_matrix(labels,p,labels=range(6)).tolist(),
        'recall':recall_score(labels,p,labels=range(6),average=None,zero_division=0).tolist(),
        'f1':f1_score(labels,p,labels=range(6),average=None,zero_division=0).tolist()} for k,p in classes.items()})
    g.save(c.DEST/phase/variant/'scores.json',result)
    return result

def table(headers,rows):
    esc=lambda x:html.escape(str(x))
    return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

def render(phase,results,winner):
    cols=('valid_pct','packet_exact_pct','direction_accuracy','magnitude_mae','signed_mae','mean_accuracy','mean_macro_f1')
    heads=['Method','Valid JSON (%)','Packet exact (%)','Direction (%)','Magnitude MAE (bytes)','Signed MAE (bytes)','Mean accuracy (%)','Mean macro-F1 (%)']
    sections=[]
    for subset in ('all','flow_absent_from_A') if phase=='B' else ('all',):
        stats={m:s[subset] for m,s in results.items()}
        sections.append('<h2>'+html.escape(subset)+'</h2>'+table(heads,[[m,*[f'{s[k]:.4f}' for k in cols]] for m,s in stats.items()]))
        sections.append(table(['Classifier',*[m+' Accuracy / macro-F1 (%)' for m in stats]],
            [[k.upper(),*[f'{s["classifiers"][k]["accuracy"]:.2f} / {s["classifiers"][k]["macro_f1"]:.2f}' for s in stats.values()]] for k in e.KINDS]))
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 Teacher Comparison</title><style>body{font:16px Arial,"Microsoft YaHei",sans-serif;color:#172332;max-width:1700px;margin:35px auto;padding:20px}table{border-collapse:collapse;width:100%;border-top:2px solid;border-bottom:2px solid;margin:25px 0}th{border-bottom:1px solid}th,td{text-align:left;padding:12px}p{line-height:1.8}</style>'''
    text+=f'<h1>D4 {phase}：教师 / 提示词配对比较</h1><p>固定 1,000 条抽样记录，30% 缺包，六个冻结分类器。A 组主指标为平均 Accuracy；选中教师：{html.escape(winner)}。</p>'
    text+=''.join(sections)
    text+='<p>包长幅值 MAE 不含方向，Signed MAE 含方向。失败或格式无效的记录不剔除，缺包位置填零；可见包不变。DeepSeek 使用 JSON-object 模式和本地严格检查，Gemini 使用原生 schema；相同 low reasoning 和 4096 token 上限不代表内部算力相同。</p>'
    text+='<p>这是历史测试池上的探索性选择，不是独立最终测试。B 为有放回抽样：967 条独特流、33 次重复；67 条流与 A 重叠，另外报告 933 条 A 未见抽样记录。本地 TFI 与分类器是已有复现/适配权重，不将这些成绩冒充原论文同协议结果。</p></html>'
    (c.DEST/phase/'comparison.html').write_text(text,encoding='utf-8')

def main():
    phase=sys.argv[1];assert phase in ('A','B')
    assert (c.DEST/phase/'api_complete.json').exists()
    rows,abodies,brows,bbodies=c.prepare()
    if phase=='A':
        ds=evaluate('A','plain',c.MODEL,rows,abodies)
        old=c.read(g.DEST/'comparison.json')
        results={'local_TFI':baseline('A',rows),**{m:{'all':old['results'][m]} for m in g.MODELS},c.MODEL:ds}
        ranking=sorted((*g.MODELS,c.MODEL),key=lambda m:(-results[m]['all']['mean_accuracy'],-results[m]['all']['packet_exact_pct'],-results[m]['all']['valid_pct'],m))
        out={'primary':'equal-weight mean Accuracy of six frozen classifiers','ranking':ranking,'results':results}
        winner=ranking[0]
    else:
        winner=c.read(c.DEST/'selection.json')['winner']
        results={'local_TFI':baseline('B',brows),**{v:evaluate('B',v,winner,brows,bbodies[v]) for v in bbodies}}
        delta={sub:{k:results['behavior'][sub][k]-results['plain'][sub][k] for k in ('mean_accuracy','mean_macro_f1','magnitude_mae','signed_mae','packet_exact_pct')} for sub in ('all','flow_absent_from_A')}
        out={'winner_selected_on_A':winner,'results':results,'behavior_minus_plain':delta}
        g.save(c.DEST/'change_log.json',{'utc':c.utc(),'change':'Only prompt instruction and visible-only behavior summary added; no model/classifier training','delta':delta,'units':'Accuracy/F1/hit differences in percentage points; MAE differences in bytes'})
    g.save(c.DEST/phase/'comparison.json',out)
    render(phase,results,winner)
    print(json.dumps({'phase':phase,'winner':winner,'complete':True}),flush=True)

if __name__=='__main__':main()
