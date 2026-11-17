"""Read-only audit of archived predictions at the original hidden packet slots."""
import gzip
import hashlib
import json
import pickle
import re
from collections import Counter
from html import escape, unescape
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
D = P.parent.parent/'d4_downstream_random'


def collect():
    meta = json.loads((P/'metrics.json').read_bytes())
    maskfile = D/'mask50.npz'
    flowfile = D/'flows.jsonl'
    seqfile = P.parent/'six_data.pkl.gz'
    for path, h in [(maskfile, meta['audit']['50']['mask_sha']),
                    (flowfile, meta['hashes'][flowfile.name]), (seqfile, meta['hashes'][seqfile.name])]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==h
    with gzip.open(seqfile,'rb') as f:
        data = pickle.load(f)
    cohort = {}
    for row in re.findall(r'<tr>(.*?)</tr>',(P/'D4_IAT_50_swap.html').read_text(encoding='utf-8')):
        model,key,target,base,direction,size,both = map(unescape,re.findall(r'<td>(.*?)</td>',row))
        if model=='lstm' and target=='Miuref' and direction==target and size!=target:
            cohort[key]=(model,target,4)
        if model=='fsnet' and target=='Htbot' and direction!=target and size==target:
            assert base=='其他合并类'
            cohort[key]=(model,target,2)
        if key in cohort:
            assert base!=target and both==target
    assert Counter(v[1] for v in cohort.values())=={'Miuref':59,'Htbot':10}
    records=[]
    with np.load(maskfile) as masks, np.load(P/'predictions.npz') as pred,flowfile.open(encoding='utf-8') as flows:
        ids = pred['ids'].tolist()
        assert ids==masks['ids'].tolist() and len(set(ids))==len(ids)==13150
        masked = masks['mask']
        for i,(key,line) in enumerate(zip(ids,flows,strict=True)):
            if key not in cohort: continue
            model,target,label=cohort[key]
            row=json.loads(line)
            assert row['id']==key and pred['y'][i]==label and pred[model+'_tfi50_after'][i]==label
            assert pred[model+'_dual50_after'][i]!=label
            if target=='Htbot': assert pred[model+'_dual50_after'][i]==5
            g=np.asarray(row['signed'],dtype=int)
            a,b=[np.asarray(data[tag][0][i],dtype=int) for tag in ('dual50','tfi50')]
            n=len(g)
            assert n==len(a)==len(b) and not masked[i,n:].any()
            mask=masked[i,:n]
            assert np.array_equal(g[~mask],a[~mask]) and np.array_equal(g[~mask],b[~mask])
            assert all(v.all() for v in (g,a,b))
            for rank,pos in enumerate(np.flatnonzero(mask),1):
                records.append((key,target,n,pos+1,rank,g[pos],a[pos],b[pos]))
    z=pd.DataFrame(records,columns=['flow','class','n','pos','rank','truth','dual','tfi'])
    assert set(z.flow)==set(cohort) and not z.duplicated(['flow','pos']).any()
    return z


def counts(z):
    a=np.sign(z.dual)!=np.sign(z.truth)
    b=np.sign(z.tfi)!=np.sign(z.truth)
    return [len(z),a.sum(),b.sum(),(a&~b).sum(),(b&~a).sum()]


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+escape(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    z=collect()
    m=z[z['class']=='Miuref']; h=z[z['class']=='Htbot']
    bins=[0,8,16,32,64,128,float('inf')]
    direction=[]
    for left,right in zip(bins[:-1],bins[1:]):
        sub=m[(m.pos>left)&(m.pos<=right)]
        if len(sub):
            direction.append([f'{left+1}–{int(right)}' if right!=float('inf') else f'{left+1}+',*counts(sub)])
    direction.append(['全部',*counts(m)])
    exact=sorted(([i,*counts(s)] for i,s in m.groupby('pos')),key=lambda r:(-r[2],r[0]))
    print('position bands',direction,'exact positions',exact[:15])
    print('sign splits',[(v,counts(m[np.sign(m.truth)==v])) for v in (-1,1)])
    print('top Miuref truth/dual/tfi',Counter(zip(m.truth,m.dual,m.tfi)).most_common(10))
    detail=[]
    for key in ('dual','tfi'):
        g=h.truth.abs(); p=h[key].abs(); e=p-g
        for name,sel in [('短包 ≤128 B',g<=128),('非短包 >128 B',g>128),('全部',g>0)]:
            d=e[sel]
            detail.append([key,name,len(d),(d>0).sum(),(d<0).sum(),(d==0).sum(),int(d.clip(lower=0).sum()),int(-d.clip(upper=0).sum()),round(d.abs().mean(),2)])
        print(key,'cross short→long, long→short',((g<=128)&(p>128)).sum(),((g>128)&(p<=128)).sum(),
              'transitions',Counter(zip(g,p)).most_common(15),'top by abs error',
              sorted(((n*abs(i-j),i,j,n) for (i,j),n in Counter(zip(g,p)).items() if i!=j),reverse=True)[:10])
    print('Htbot',detail)
    head=['流内包序号（从 1 起）','缺包数','双路判反','TFI 判反','仅双路判反','仅 TFI 判反']
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 · 50% 缺包逐包核查</title><style>body{max-width:1150px;margin:32px auto;font:16px/1.55 system-ui;padding:0 1rem}table{border-collapse:collapse;border-block:2px solid;margin:1rem 0}td,th{padding:.2rem .7rem;text-align:right}td:first-child,th:first-child{text-align:left}thead{border-bottom:1px solid}summary{cursor:pointer}</style><h1>D4 · 59 条 Miuref 与 10 条 Htbot 的逐包真值核查</h1><p>流和 50% 缺包掩码沿用上轮。所有序号从 1 开始；正号是该流首包的方向。双路和本地 TFI 均与同位置的归档真值比较。</p><h2>Miuref · 方向错误</h2>',table(head,direction),
           '<details><summary>按单个包序号查看（按双路错误数排序）</summary>',table(head,exact),'</details><h2>Htbot · 绝对包长误差</h2>',
           table(['方法','真实包长','缺包数','补长','补短','相等','多补字节','少补字节','MAE（B）'],detail)]
    for label,group in [('Miuref',m),('Htbot',h)]:
        parts.append(f'<h2>{label} · 每条流的缺包位置</h2>')
        for flow,s in group.groupby('flow',sort=True):
            parts.extend([f'<details><summary>{escape(flow)}（{len(s)} 个缺包）</summary>',
                table(['流内包序号','缺包序号','真值 ±B','双路 ±B','TFI ±B','双路方向','TFI 方向','双路幅值偏差 B','TFI 幅值偏差 B'],
                      [[r.pos,r.rank,r.truth,r.dual,r.tfi,'对' if r.dual*r.truth>0 else '反','对' if r.tfi*r.truth>0 else '反',abs(r.dual)-abs(r.truth),abs(r.tfi)-abs(r.truth)] for r in s.itertuples()]),'</details>'])
    (P/'D4_50_packet_truth.html').write_text(''.join(parts)+'</html>',encoding='utf-8')


if __name__=='__main__': main()
