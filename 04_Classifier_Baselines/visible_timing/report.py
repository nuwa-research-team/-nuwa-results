import json
from pathlib import Path
from statistics import mean

import numpy as np

P = Path(__file__).resolve().parent
doc = json.loads((P/'metrics.json').read_bytes())
records = doc['records']
names = {'cnn':'CNN','lstm':'LSTM','gru':'GRU','tst':'TST','sdae':'SDAE','fsnet':'FS-Net'}
methods = {'dual':'双路','tfi':'本地 TFI'}
assert len(records)==len({(r['rate'],r['method'],r['model']) for r in records})==36
head = '<tr><th>丢包</th><th>补全</th><th>分类器</th><th>准确率 %<br>无 → 有</th><th>macro-F1 %<br>无 → 有</th><th>SMB → Weibo<br>无 → 有</th><th>减少</th><th>Weibo → SMB<br>无 → 有</th><th>减少</th></tr>'
def line(rate,method,name,a,b):
    cells = [f'{rate}%',methods[method],name,
             f"{a['acc']:.2f} → {b['acc']:.2f}",f"{a['f1']:.2f} → {b['f1']:.2f}",
             f"{a['sw']:,} → {b['sw']:,}",f"{a['sw']-b['sw']:+,}",
             f"{a['ws']:,} → {b['ws']:,}",f"{a['ws']-b['ws']:+,}"]
    return '<tr>'+''.join(f'<td>{s}</td>' for s in cells)+'</tr>'
rows,summary = [],[]
with np.load(P/'predictions.npz') as archive:
    y=archive['y']
    assert len(set(archive['ids']))==13150 and np.bincount(y).tolist()==[7044,5286,254,255,127,184]
    for r in records:
        for arm in ('before','after'):
            pred=archive[f"{r['model']}_{r['method']}{r['rate']}_{arm}"].astype(int)
            cm=np.bincount(6*y+pred,minlength=36).reshape(6,6)
            m=r[arm]
            assert cm.tolist()==m['cm'] and cm[1,0]==m['sw'] and cm[0,1]==m['ws']
            assert abs(np.mean(pred==y)*100-m['acc'])<1e-8
            assert abs(np.mean(2*cm.diagonal()/(cm.sum(0)+cm.sum(1)))*100-m['f1'])<1e-8
for rate in (10,30,50):
    for method in methods:
        subset = [r for r in records if (r['rate'],r['method'])==(rate,method)]
        for k in names:
            r = next(r for r in subset if r['model']==k)
            rows.append(line(rate,method,names[k],r['before'],r['after']))
        a,b=[{metric:(mean if metric in ('acc','f1') else sum)(r[arm][metric] for r in subset)
              for metric in ('acc','f1','sw','ws')} for arm in ('before','after')]
        summary.append(line(rate,method,'六模型汇总',a,b))
        print(rate,method,a,b)

diffs=[(r['model'],r['method'],r['rate'],r['baseline_mismatches']) for r in records if r['baseline_mismatches']]
print('precision replay differences',diffs)
page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 下游可见包时间间隔配对实验</title>
<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1170px;margin:2rem auto;padding:0 1rem;color:#172331}
h1{font-size:1.7rem}table{width:100%;border-collapse:collapse;border-block:2px solid #25364a;margin:1rem 0 2rem;font-variant-numeric:tabular-nums}th{border-bottom:1px solid #25364a}td,th{text-align:right;padding:.42rem}td:nth-child(-n+3),th:nth-child(-n+3){text-align:left}tbody tr:nth-child(6n+1) td{padding-top:.85rem}p{max-width:86ch}</style>
<h1>D4 · 把可见包的时间间隔加入下游分类器</h1>
<p>13,150 条流，包含 5,286 条 SMB、7,044 条 Weibo。每一对结果使用相同的流、掩码和补全包长。模型用连续两个可见包的实际时间差，跨缺口的间隔保留；零间隔保留，负间隔按缺失处理。已核对逐流预测和全部 72 个混淆矩阵。</p>
<p>六类口径。准确率和 macro-F1 按分类器等权平均，两个方向的互错按六个分类器累计，同一条流可以计入六次。“减少”为加间隔前减去加间隔后，负数表示该方向错判增加。</p>'''
def table(items):
    return '<table><thead>'+head+'</thead><tbody>'+''.join(items)+'</tbody></table>'
page += '<h2>三档丢包汇总</h2>'+table(summary)+'<h2>六个分类器明细</h2>'+table(rows)
page += '<p>冻结双路、本地 TFI 和六个包长分类器。给每个分类器的末端接入一个单独训练的时间分支，将其输出加到六类分数上。使用原 4,096 条训练流的干净包长路径和遮盖后的时间特征拟合；用 384 条开发流在三档掩码下的平均交叉熵选轮。两个补全方法共享同一时间分支。该流级划分复用同一批抓包文件，没有第二次独立采集。</p>'
page += '<p>无间隔列使用本次 CPU 重放，保证两臂使用完全相同的包长分类路径。与上一轮 GPU 归档逐条核对，存在预测差异的组合（模型、补全、丢包%、条数）为 '+str(diffs)+'。</p>'
(P/'D4_IAT.html').write_text(page,encoding='utf-8')
