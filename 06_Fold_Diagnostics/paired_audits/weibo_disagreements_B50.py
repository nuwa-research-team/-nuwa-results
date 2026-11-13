"""Summarize archived B-fold Weibo disagreements without running either head."""
import json
from statistics import mean

import numpy as np

from audit_A50 import observation, table
import evaluate as e


def summaries(records):
    for name in ('Weibo-4','Weibo-2','Weibo-3'):
        for trigger in (True,False):
            rows = [r for r in records if r['pcap']==name and r['trigger']==trigger]
            scores = [mean(r[k] for r in rows) for k in ('packet','joint','margin')]
            features = {}
            for k in rows[0]['visible']:
                values = [r['visible'][k] for r in rows if r['visible'][k] is not None]
                if k=='每次换向都跨缺口':
                    continue
                features[k] = [len(values), *np.percentile(values,[25,50,75])]
            yield dict(pcap=name,trigger=trigger,n=len(rows),scores=scores,features=features)


def main():
    by_id={r['id']:r for r in e.old.read_panels()['train']}
    records=[]
    with np.load(e.OUT/'B50_fallback_predictions.npz') as b, np.load(e.OUT/'paired.npz') as p:
        fold=p['fold']==1
        for a,c in [('ids','ids'),('y','y'),('ordered','ordered50'),('before','combined50')]:
            assert np.array_equal(b[a],p[c][fold])
        choose=(b['y']==0)&(b['ordered']==1)&(b['before']==0)
        for i in np.flatnonzero(choose):
            name=str(b['ids'][i])
            flow=by_id[name]
            packet,joint=float(b['ordered_score'][i]),float(b['joint_score'][i])
            margin=abs(packet)-abs(joint)
            trigger=bool(b['trigger'][i])
            assert packet>0 and joint<=0 and trigger==(margin>0) and trigger==bool(b['after'][i])
            records.append(dict(id=name,pcap=e.old.file_of(flow),trigger=trigger,packet=packet,joint=joint,
                margin=margin,visible=observation(flow,e.old.fixed_mask(flow,50))))
    assert len({r['id'] for r in records})==50
    groups=list(summaries(records))
    assert [g['n'] for g in groups]==[11,8,5,11,7,8]
    obs=list(groups[0]['features'])
    head=['PCAP','回退','流数','逐包分','联合分','绝对分差']
    def group_name(r):
        return '触发' if r['trigger'] else '未触发'
    dist=lambda q: f'{q[2]:g} [{q[1]:g}, {q[3]:g}] (n={q[0]})'
    doc=table(head+obs,[[g['pcap'],group_name(g),g['n'],*[f'{v:+.3f}' for v in g['scores']],
        *[dist(g['features'][k]) for k in obs]] for g in groups])
    doc+='<h2>逐流对照</h2>'+table(['流 ID','回退',*head[3:],*obs],[
        [r['id'],group_name(r),*[f'{r[k]:+.4f}' for k in ('packet','joint','margin')],
         *[r['visible'][k] if r['visible'][k] is not None else '无' for k in obs]]
        for r in sorted(records,key=lambda r:(-int(r['pcap'][-1]),not r['trigger'],r['id']))])
    (e.OUT/'B50_Weibo_pcap_disagreements.html').write_text('<!doctype html><meta charset=utf-8><title>B 折 Weibo 分歧流</title><style>body{font:16px system-ui;margin:2%}table{border-collapse:collapse;border-block:2px solid}thead{border-bottom:1px solid}td,th{padding:.35em .6em}</style><h1>B 折 50% 丢包下的 Weibo 分歧流</h1><p>逐包判 SMB，联合判 Weibo。分数列为组均值。绝对分差等于逐包绝对分减联合绝对分。特征列为逐流中位数 [Q1, Q3] 和有效流数，间隔以微秒计。</p>'+doc,encoding='utf-8')
    for g in groups:
        print(json.dumps(g,ensure_ascii=False))


if __name__=='__main__':
    main()
