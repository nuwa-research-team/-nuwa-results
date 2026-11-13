import json
from pathlib import Path
import sys

import numpy as np

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'d4_segment_pcap'))
import report as layout

layout.P=P
layout.ARMS={**layout.ARMS,'combined':'逐包＋分段'}
decision=json.loads((P/'decision.json').read_bytes())
baseline=json.loads((P.parent/'d4_segment_pcap/metrics.json').read_bytes())
combined=[]; folds=[]; audit=[]
with np.load(P/'paired.npz') as data:
    y=data['y']
    for rate in (10,30,50):
        for arm,title in layout.ARMS.items():
            pred=data[f'{arm}{rate}']
            for label,sel in [('all',np.ones(len(y),bool)),('A',data['fold']==0),('B',data['fold']==1)]:
                cm,s=layout.scores(y[sel],pred[sel],2)
                if arm!='combined':
                    assert cm.tolist()==baseline[str(rate)][arm][label]['cm']
                row=[rate,title,*layout.fmt(s)]
                if label=='all': combined.append(row)
                else: folds.append([label,*row])
    pool=(y==1)&(data['summary50']==0)&(data['ordered50']==1)
    assert pool.sum()==143
    for i,label in enumerate(('A','B')):
        sel=data['fold']==i
        kept=int(np.sum(pool&sel&(data['combined50']==1)))
        assert kept==decision[label]['retained']
        ws_corrected=int(np.sum(sel&(y==0)&(data['ordered50']==1)&(data['combined50']==0)))
        ws_added=int(np.sum(sel&(y==0)&(data['ordered50']==0)&(data['combined50']==1)))
        assert ws_added-ws_corrected==decision[label]['weibo_error_delta']
        audit.append([label,int(np.sum(pool&sel)),kept,ws_corrected,ws_added,ws_added-ws_corrected,decision[label]['smb_error_delta']])
means,detail=layout.downstream(('screen','confirm'))
print('combined',combined,'fold50',[r for r in folds if r[1]==50 and r[2]=='逐包＋分段'],'paired audit',audit,'pipeline50',means[-4:])
sections=[('两折合并，3,596 条',layout.table(layout.HEAD,combined)),
          ('50% 逐流配对，相对逐包版',layout.table(['折','原救回 SMB','仍判对','纠正 Weibo→SMB','新增 Weibo→SMB','Weibo 错判净增','SMB 错判净增'],audit)),
          ('逐折',layout.table(['折']+layout.HEAD,folds)),
          ('六个分类器，896 条开发流，等权平均',layout.table(layout.HEAD,means)),
          ('开发集分类器明细',layout.table(layout.HEAD[:2]+['分类器']+layout.HEAD[2:],detail))]
if all(d['passed'] for d in decision.values()):
    full,_=layout.downstream(('full',))
    sections.append(('13,150 条全量流',layout.table(layout.HEAD,full)))
html='<h1>D4 逐包与收发段联合输入验证</h1><p>A 留出 SMB-1 和 Weibo-1，B 留出 SMB-2 和 Weibo-2、3、4。仅拟合下游 SMB/Weibo 判别器。逐折错判数按流计，六分类器平均值可为小数。</p>'
html+=''.join('<h2>'+title+'</h2>'+table for title,table in sections)
(P/'D4_packet_segment.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 四版对照</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:16px}table{border-collapse:collapse;border-block:2px solid}th,td{padding:.35em .8em;text-align:left}thead{border-bottom:1px solid}</style>'+html+'</html>',encoding='utf-8')
