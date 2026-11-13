import json
from html import escape

from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ARMS=('summary','ordered')
RATES=(10,30,50)
KINDS=('cnn','lstm','gru','tst','sdae','fsnet')


def metrics(y,p,nclasses):
    cm=np.zeros((nclasses,nclasses),int)
    np.add.at(cm,(y,p),1)
    support=cm.sum(1)
    return dict(n=len(y),acc=100*cm.trace()/len(y),f1=100*np.mean(2*cm.diagonal()/np.maximum(1,support+cm.sum(0))),
                smb=100*cm[1,1]/support[1],weibo=100*cm[0,0]/support[0],
                sw=int(cm[1,0]),ws=int(cm[0,1]),cm=cm.tolist())


def table(headers, rows):
    def cells(tag, values):
        return ''.join('<{0}>{1}</{0}>'.format(tag,escape(str(x))) for x in values)
    return '<table><thead><tr>'+cells('th',headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+cells('td',r)+'</tr>' for r in rows)+'</tbody></table>'


def format_row(result):
    return [round(result[k],2) for k in ('acc','f1','smb','sw','weibo','ws')]


def main():
    names={'summary':'当前汇总特征','ordered':'汇总＋逐包三元组'}
    headers=['丢包率 %','版本','准确率 %','macro-F1 %','SMB 召回 %','SMB → Weibo','Weibo 召回 %','Weibo → SMB']
    binary=[]; folds=[]; captures=[]; six=[]; avgs=[]
    record=json.loads((OUT/'binary_metrics.json').read_bytes())
    with np.load(OUT/'binary_predictions.npz') as saved:
        y=saved['y']
        f=np.array([sid.split('.pcap#')[0] for sid in saved['ids']])
        for rate in RATES:
            for arm in ARMS:
                p=saved[f'{arm}{rate}']
                for label,sel in [('all',np.ones(len(y),bool)),('A',saved['fold']==0),('B',saved['fold']==1)]:
                    m=metrics(y[sel],p[sel],2)
                    ref=record[str(rate)][arm][label]
                    assert all(np.allclose(v,ref[k]) for k,v in m.items())
                    prefix=[rate,names[arm]]
                    if label=='all': binary.append(prefix+format_row(m))
                    else: folds.append([label]+prefix+format_row(m))
                for file in sorted(set(f)):
                    sel=f==file
                    hit=np.sum(y[sel]==p[sel])
                    captures.append([file,rate,names[arm],int(sel.sum()),int(hit),int(sel.sum()-hit),round(100*hit/sel.sum(),2)])
    panels=[np.load(OUT/f'pipeline_{tag}.npz') for tag in ('screen','confirm')]
    assert not set(panels[0]['ids']).intersection(panels[1]['ids'])
    y=np.concatenate([s['y'] for s in panels])
    for rate in RATES:
        for arm in ARMS:
            rows=[]
            for kind in KINDS:
                p=np.concatenate([s[f'{rate}_held_{arm}_{kind}'] for s in panels])
                m=metrics(y,p,6)
                six.append([kind,rate,names[arm]]+format_row(m))
                rows.append([m[k] for k in ('acc','f1','smb','sw','weibo','ws')])
            avgs.append([rate,names[arm]]+np.round(np.mean(rows,axis=0),2).tolist())
    for panel in panels: panel.close()
    print('binary',binary,'folds',folds,'six',avgs)
    body='<h1>D4 逐包时序输入的 PCAP 留出验证</h1><p>二分类头按文件两折交叉验证。A 为 SMB-1 / Weibo-1，B 为 SMB-2 / Weibo-2、3、4。方向、间隔和缺包数仅来自可见槽位。旧版与新版逐流配对，三档复用同一批流。</p>'
    body+='<h2>严格文件留出，两类合并</h2>'+table(headers,binary)
    body+='<h2>按留出折</h2>'+table(['文件组']+headers,folds)
    body+='<h2>各 PCAP</h2>'+table(['文件','丢包率 %','版本','流数','正确','错判至另一类','召回率 %'],captures)
    body+='<h2>六分类系统，896 条开发流</h2><p>改动仅在下游二分类头。六个固定上游分类器沿用已有权重。本表为六个分类器等权均值，错判数可为小数。</p>'+table(headers,avgs)
    body+='<h2>六个分类器逐项结果</h2>'+table(['分类器']+headers,six)
    body+='<p>三档合并的二分类准确率和 macro-F1 变化量均小于零，全量扩展门槛未通过。</p>'
    (OUT/'D4_pcap_validation.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 PCAP 验证</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:15px}table{border-collapse:collapse;border-block:2px solid}thead{border-bottom:1px solid}th,td{padding:.35em .65em;text-align:left}h2{margin-top:2em}</style>'+body+'</html>',encoding='utf-8')


if __name__=='__main__': main()
