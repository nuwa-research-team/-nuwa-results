import json
from html import escape
from pathlib import Path

import numpy as np

P=Path(__file__).resolve().parent
ARMS={'summary':'当前版','ordered':'逐包版','segment':'分段版'}
HEAD=['丢包率','版本','准确率 %','macro-F1 %','SMB 召回 %','SMB → Weibo','Weibo 召回 %','Weibo → SMB']
KINDS=('cnn','lstm','gru','tst','sdae','fsnet')


def table(headers, rows):
    def cells(tag,values):
        return ''.join(f'<{tag}>{escape(str(v))}</{tag}>' for v in values)
    return '<table><thead><tr>'+cells('th',headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+cells('td',r)+'</tr>' for r in rows)+'</tbody></table>'


def scores(y,p,n):
    cm=np.zeros((n,n),int)
    np.add.at(cm,(y,p),1)
    den=cm.sum(1)
    return cm,np.array([100*cm.trace()/len(y),100*np.mean(2*cm.diagonal()/np.maximum(1,den+cm.sum(0))),
        100*cm[1,1]/den[1],cm[1,0],100*cm[0,0]/den[0],cm[0,1]])


def fmt(nums,avg=False):
    return [int(v) if i in (3,5) and not avg else f'{v:.2f}' for i,v in enumerate(nums)]


def downstream(tags):
    preds=[np.load(P/f'{t}.npz') for t in tags]
    ids=np.concatenate([p['ids'] for p in preds]); y=np.concatenate([p['y'] for p in preds])
    assert len(ids)==len(set(ids))
    mean=[]; detail=[]
    for rate in (10,30,50):
        for arm,title in ARMS.items():
            values=[]
            for k in KINDS:
                v=np.concatenate([p[f'{rate}_{arm}_{k}'] for p in preds])
                _,s=scores(y,v,6)
                values.append(s)
                detail.append([rate,title,k,*fmt(s)])
            mean.append([rate,title,*fmt(np.mean(values,axis=0),True)])
    for p in preds:p.close()
    return mean,detail


def main():
    summary=json.loads((P/'metrics.json').read_bytes())
    decision=json.loads((P/'decision.json').read_bytes())
    both=[]; folds=[]
    with np.load(P/'paired.npz') as pred:
        for rate,results in summary.items():
            for arm,name in ARMS.items():
                for group,data in results[arm].items():
                    mask=np.ones(len(pred['y']),bool) if group=='all' else pred['fold']==('A','B').index(group)
                    cm,s=scores(pred['y'][mask],pred[f'{arm}{rate}'][mask],2)
                    assert cm.tolist()==data['cm'] and np.allclose(s,[data[k] for k in ('acc','f1','smb','sw','weibo','ws')])
                    row=[rate,name,*fmt(s)]
                    if group=='all': both.append(row)
                    else: folds.append([group,*row])
    means,detail=downstream(('screen','confirm'))
    print('binary',both,'fold50',[r for r in folds if r[1]=='50'],'pipeline',means,flush=True)
    sections=[('两折合并，3,596 条流',table(HEAD,both)),('按留出文件组',table(['组']+HEAD,folds)),
        ('896 条开发流，六分类器等权平均',table(HEAD,means)),
        ('各分类器明细',table(HEAD[:2]+['分类器']+HEAD[2:],detail))]
    if all(decision['passed'].values()):
        full,_=downstream(('full',));sections.append(('13,150 条流',table(HEAD,full)))
    info='A 留出 SMB-1、Weibo-1；B 留出 SMB-2、Weibo-2/3/4。二分类错判单位为条。开发流水线只替换二分类判别，使用原六分类器，均值可为小数。'
    body='<h1>D4 同向可见包分段验证</h1><p>'+info+'</p>'+''.join('<h2>'+k+'</h2>'+t for k,t in sections)
    (P/'D4_segment_comparison.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 同向收发段对照</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:16px;color:#20252b}table{border-collapse:collapse;border-block:2px solid}thead{border-bottom:1px solid}th,td{text-align:left;padding:.34em .75em}h2{margin-top:2em}</style>'+body+'</html>',encoding='utf-8')


if __name__=='__main__': main()
