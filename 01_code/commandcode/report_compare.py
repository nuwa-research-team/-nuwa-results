"""Evaluate only complete frozen cohorts; combine with archived teacher results."""
import argparse,csv,html,json,sys
from pathlib import Path
from collections import Counter
from compare import DATA,OUT,MODELS,read,save,sha,enc,validate,utc

def evaluate(model):
    sys.path.insert(0,str(DATA))
    import experiment as e
    rows=read(DATA/'local_A.json');requests=read(OUT/'requests.json')[model];records=[]
    for i,r in enumerate(rows):
        record=read(OUT/model/f'{i:04d}.json')
        assert record['draw']==i and record['model']==model
        assert record['request_sha256']==sha(enc(requests[i]))
        v,err=validate(record['response'],len(r['pos'])) if record['http_status']==200 else (None,'http_failure')
        assert v==record['values'] and err==record['validation_error']
        records.append(record)
    vs=[r['values'] for r in records];method='commandcode_v1_'+model.replace('-','_')
    stats=e.score('A',method,vs)
    stats['mean_macro_f1']=sum(s['macro_f1'] for s in stats['classifiers'].values())/6
    truth=[r['truth'][p] for r in rows for p in r['pos']]
    pred=[x for r,v in zip(rows,vs) for x in (v if v is not None else [0]*len(r['pos']))]
    stats['signed_mae']=sum(abs(a-b) for a,b in zip(truth,pred))/len(truth)
    stats['direction_accuracy']=100*sum((a>0)==(b>0) and b!=0 for a,b in zip(truth,pred))/len(truth)
    save(OUT/model/'scores.json',{'all':stats,'errors':dict(Counter(r['validation_error'] for r in records if r['validation_error'])),
        'reported_models':dict(Counter(r['response'].get('model','unknown') for r in records)),
        'http_success':sum(r['http_status']==200 for r in records),'utc':utc()})

def report():
    results=read(DATA/'cross_teacher_v1/A/comparison.json')['results'].copy()
    for model in MODELS:
        p=OUT/model/'scores.json'
        if p.exists():results[model]=read(p)
    columns=['Model','Accuracy (%)','Macro-F1 (%)','Magnitude MAE (bytes)','Signed MAE (bytes)','Valid output (%)']
    rows=[]
    for m,r in results.items():
        s=r['all'];f1=s.get('mean_macro_f1',sum(x['macro_f1'] for x in s['classifiers'].values())/6)
        rows.append([m,s['mean_accuracy'],f1,s['magnitude_mae'],s.get('signed_mae',''),s['valid_pct']])
    for m in MODELS:
        if m not in results:
            blocked=read(OUT/'blocked.json') if (OUT/'blocked.json').exists() else {}
            state='Not run: plan unavailable' if blocked.get('model')==m else 'Pending'
            rows.append([m,state,state,state,state,state])
    details=[['Classifier','Model','Accuracy (%)','Macro-F1 (%)']]
    for k in ('cnn','lstm','gru','tst','sdae','fsnet'):
        for m,r in results.items():
            s=r['all']['classifiers'][k];details.append([k,m,s['accuracy'],s['macro_f1']])
    for name,rs in [('summary.csv',[columns,*rows]),('classifiers.csv',details)]:
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:csv.writer(f).writerows(rs)
    def table(headers,rs):
        fmt=lambda v:html.escape(f'{v:.4f}' if type(v) is float else str(v))
        return '<table><thead><tr>'+''.join('<th>'+fmt(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+fmt(v)+'</td>' for v in row)+'</tr>' for row in rs)+'</tbody></table>'
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 teacher comparison</title><style>body{font:16px Arial,"Microsoft YaHei";margin:35px auto;max-width:1500px;padding:20px;color:#172332}table{border-collapse:collapse;width:100%;border-top:2px solid;border-bottom:2px solid;margin:25px 0}th{border-bottom:1px solid}th,td{text-align:left;padding:10px}p{line-height:1.8}</style><h1>D4：Luna / Astra / Gemini / DeepSeek / 本地 TFI</h1><p>A 批固定 1,000 条流，30% 缺包，共 10,488 个缺失位置；提示词与输入逐条一致。仅完整 1,000 条评分，Pending 表示尚未完成。</p>'''
    text+=table(columns,rows)+table(details[0],details[1:])
    text+='<p>Accuracy 与 macro-F1 是六个冻结分类器指标的等权均值；下表列出各分类器。幅值 MAE 不含方向，Signed MAE 含方向。无效或失败样本保留并填零，可见值不变。DeepSeek 有较多输出截断，须结合有效率解读。</p><p>这是历史测试池上的探索性教师比较，不是独立最终测试，也不是原论文同协议成绩。CommandCode 返回的 Luna / Astra 模型身份未经独立上游认证；各平台 JSON 约束机制不同，4096 token 与 low 推理设置不能保证等算力。</p></html>'
    (OUT/'comparison.html').write_text(text,encoding='utf-8')
    rank=sorted((m for m in results if m!='local_TFI'),key=lambda m:(-results[m]['all']['mean_accuracy'],-results[m]['all']['packet_exact_pct'],-results[m]['all']['valid_pct'],m))
    save(OUT/'comparison.json',{'utc':utc(),'all_teachers_complete':all(m in results for m in MODELS),'ranking_of_completed_teachers':rank,'results':results})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluate',choices=MODELS);a=p.parse_args()
    if a.evaluate:evaluate(a.evaluate)
    else:report()
