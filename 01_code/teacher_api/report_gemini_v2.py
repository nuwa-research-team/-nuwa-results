"""Evaluate completed teacher outputs with the existing six frozen classifiers."""
import argparse
import csv
import hashlib
import html
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score

import experiment as e
import gemini_compare_v2 as g


def evaluate(model):
    files=[g.DEST/model/f'{i:04d}.json' for i in range(1000)]
    if not all(p.exists() for p in files):
        return False
    score_path=g.DEST/model/'scores.json'
    if score_path.exists():
        return True
    rows,bodies=g.prepare()
    recs=[json.loads(p.read_bytes()) for p in files]
    values=[]
    for i,(r,b) in enumerate(zip(recs,bodies)):
        endpoint=f'https://aiplatform.googleapis.com/v1/projects/{g.PROJECT}/locations/global/publishers/google/models/{model}:generateContent'
        assert r['draw']==i and r['request_sha256']==g.sha(g.encoded({'endpoint':endpoint,'body':b}))
        v,err=g.validate(r['response'],len(rows[i]['pos'])) if r['http_status']==200 else (None,'http_failure')
        assert v==r['values'] and err==r['validation_error']
        values.append(v)
    stats=e.score('A','gemini_v2_'+model,values)
    preds=e.read('A_gemini_v2_'+model+'_predictions.json')['classes']
    truth=[r['truth'][p] for r in rows for p in r['pos']]
    flat=[v for r,vs in zip(rows,values) for v in (vs if vs is not None else [0]*len(r['pos']))]
    truth=np.array(truth); flat=np.array(flat)
    stats.update(signed_mae=float(np.abs(truth-flat).mean()),
        direction_accuracy=100*float((np.sign(truth)==np.sign(flat)).mean()),
        mean_macro_f1=float(np.mean([s['macro_f1'] for s in stats['classifiers'].values()])),
        http_success=sum(r['http_status']==200 for r in recs),
        model_versions=dict(Counter(r['response'].get('modelVersion','unknown') for r in recs)),
        errors=dict(Counter(r['validation_error'] for r in recs if r['validation_error'])),
        http_attempts=sum(len(r['attempts']) for r in recs),
        usage={k:sum(r['response'].get('usageMetadata',{}).get(k,0) for r in recs)
               for k in ('promptTokenCount','candidatesTokenCount','thoughtsTokenCount','totalTokenCount')})
    labels=[r['label'] for r in rows]
    g.save(g.DEST/model/'class_diagnostics.json', {k:{
        'confusion_matrix_true_rows_pred_columns':confusion_matrix(labels,p,labels=range(6)).tolist(),
        'recall':recall_score(labels,p,labels=range(6),average=None,zero_division=0).tolist(),
        'f1':f1_score(labels,p,labels=range(6),average=None,zero_division=0).tolist()}
        for k,p in preds.items()})
    g.save(score_path,stats)
    return True


def report():
    rows=e.read('local_A.json')
    baseline=e.read('A_local_tfi_scores.json')['all']
    bv=e.read('A_local_tfi_predictions.json')['values']
    t=np.array([r['truth'][p] for r in rows for p in r['pos']]); v=np.array([x for a in bv for x in a])
    baseline.update(mean_macro_f1=float(np.mean([s['macro_f1'] for s in baseline['classifiers'].values()])),
        signed_mae=float(np.abs(t-v).mean()),direction_accuracy=100*float((np.sign(t)==np.sign(v)).mean()))
    results={m:json.loads((g.DEST/m/'scores.json').read_bytes()) for m in g.MODELS}
    ranking=sorted(results,key=lambda m:(-results[m]['mean_accuracy'],-results[m]['packet_exact_pct'],-results[m]['valid_pct'],m))
    allresults={'local_TFI':baseline,**results}
    g.save(g.DEST/'comparison.json', {'setting':'Exploratory historical test cohort A; 1000 unique flows; fixed 30% masks',
        'primary':'equal-weight mean accuracy of six frozen local classifiers', 'ranking':ranking,'results':allresults})
    headers=['Model','Valid JSON (%)','Packet exact (%)','Direction (%)','Magnitude MAE (bytes)','Signed MAE (bytes)','Mean accuracy (%)','Mean macro-F1 (%)']
    summary=[[m,*[f'{s[k]:.4f}' for k in ('valid_pct','packet_exact_pct','direction_accuracy','magnitude_mae','signed_mae','mean_accuracy','mean_macro_f1')]] for m,s in allresults.items()]
    detail_headers=['Classifier','local_TFI Acc/F1',*[m+' Acc/F1' for m in g.MODELS]]
    detail=[[k.upper(),*[f'{s["classifiers"][k]["accuracy"]:.2f} / {s["classifiers"][k]["macro_f1"]:.2f}' for s in allresults.values()]] for k in e.KINDS]
    for filename,head,data in [('summary.csv',headers,summary),('classifiers.csv',detail_headers,detail)]:
        with (g.DEST/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(head);w.writerows(data)
    def table(head,data):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in data)+'</tbody></table>'
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 Gemini Teacher Comparison</title>
    <style>body{{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1500px;margin:35px auto;padding:20px;color:#172332}}table{{border-collapse:collapse;width:100%;margin:24px 0;border-top:2px solid;border-bottom:2px solid}}th{{border-bottom:1px solid;padding:12px;text-align:left}}td{{padding:12px}}h1{{font-size:25px}}p{{line-height:1.7}}.note{{color:#455266}}</style>
    <h1>D4 教师模型配对比较：1,000 条流，30% 随机缺包</h1>
    <p>按预登记主指标（六个冻结分类器的平均准确率），本轮最高教师：<b>{ranking[0]}</b>。同一批流、相同缺包位置、同一提示词与 JSON 约束；无额外训练、无格式修复。</p>
    {table(headers,summary)}<h2>六个分类器：Accuracy / macro-F1（%）</h2>{table(detail_headers,detail)}
    <p class="note">共 10,488 个缺失位置。JSON 无效或接口失败的流仍保留：缺失位置回填 0、精确命中计错；可见值保持原样。包长幅值 MAE 不含方向，Signed MAE 包含方向。模型版本与 token 用量保存在各模型 scores.json。</p>
    <p class="note">这是历史测试池上的探索性教师选择，不是独立最终测试。TFI 和六个分类器均为现有本地复现/适配权重，不能把本表与论文数值解释为严格同协议比较。Gemini 3.1 Pro 为 preview；相同 LOW 思考设置不代表内部计算量完全相同。B 组未用于本轮。</p></html>'''
    (g.DEST/'comparison.html').write_text(text,encoding='utf-8')
    print(json.dumps({'ranking':ranking,'summary':summary},ensure_ascii=False),flush=True)


def progress_page():
    now=datetime.now(timezone.utc)
    records=[]
    for model in g.MODELS:
        rs=[json.loads(p.read_bytes()) for p in (g.DEST/model).glob('[0-9][0-9][0-9][0-9].json')]
        recent=sum((now-datetime.fromisoformat(r['completed_utc'])).total_seconds()<300 for r in rs)
        eta=f'{(1000-len(rs))*5/recent:.0f} min' if recent and len(rs)<1000 else ('Scoring/recovery' if len(rs)==1000 else 'Estimating')
        records.append([model,len(rs),sum(r['http_status']==200 for r in rs),sum(r['values'] is not None for r in rs),sum(r['http_status']!=200 for r in rs),eta])
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in records)
    content=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>D4 Gemini comparison — running</title>
    <style>body{{font:17px Arial,"Microsoft YaHei",sans-serif;max-width:1200px;margin:50px auto;padding:20px;color:#172332}}table{{width:100%;border-collapse:collapse;border-top:2px solid;border-bottom:2px solid}}th,td{{text-align:left;padding:14px}}th{{border-bottom:1px solid}}p{{line-height:1.8}}</style>
    <h1>D4 Gemini 教师比较：运行中</h1><p>目标：每个模型 1,000 条相同流，固定 30% 缺包，共 10,488 个隐藏位置。暂未产生完整排名。</p>
    <table><tr><th>Model</th><th>已处理 / 1,000</th><th>HTTP 成功</th><th>JSON 有效</th><th>待恢复接口失败</th><th>近期速度估计剩余</th></tr>{table}</table>
    <p>统一输出：<code>{{"values":[1500,-52,...]}}</code>。只允许指定数量的非零整数，范围 −1600 至 1600。模型只收到遮挡后的序列及缺包位置，未收到真值或类别标签。</p>
    <p>所有结果收到后，先统一恢复接口失败，再运行六个冻结分类器。本页将自动替换为包长误差、方向正确率、Accuracy 和 macro-F1 对比表。限流与代理故障可能延长 ETA；预测格式无效不会通过修改答案补救。</p>
    <p>更新（UTC）：{now.isoformat(timespec='seconds')}。本轮为历史测试池上的探索性教师选择，B 组未使用。</p></html>'''
    (g.DEST/'comparison.html').write_text(content,encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    while True:
        if not (g.DEST/'api_complete.json').exists():progress_page()
        finished={m:evaluate(m) if (g.DEST/'api_complete.json').exists() else False for m in g.MODELS}
        if all(finished.values()):report();return
        print(json.dumps({'scoring_complete':finished}),flush=True)
        if not args.watch:return
        time.sleep(30)


if __name__=='__main__':main()
