"""Describe the frozen SMB fallback on the original two held-out PCAP folds."""
from collections import Counter
import json

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import audit_A50 as audit
import evaluate as e


def examine(fold, paired, lookup):
    pick = paired['fold'] == fold
    ids, y, packet, joint = [paired[k][pick] for k in ('ids','y','ordered50','combined50')]
    rows = [lookup[i] for i in ids]
    files = [e.old.file_of(row) for row in rows]
    assert set(files) == e.old.FILES[fold] and np.array_equal(y,[e.old.LABEL[f] for f in files])
    masks = [e.old.fixed_mask(row,50) for row in rows]
    width, kept, _, pm = joblib.load(e.seg.PREV/'downstream_heads.joblib')[fold]
    vec = joblib.load(e.PREV/'segment_heads.joblib')[fold][0]
    jm = joblib.load(e.OUT/'combined_heads.joblib')[fold]
    x = e.matrix(e.prepare(rows,masks),(width,kept,vec))
    px = x[:,:kept.sum()]
    ps, js = pm.decision_function(px), jm.decision_function(x)
    assert np.array_equal(packet,pm.predict(px)) and np.array_equal(joint,jm.predict(x))
    candidate = (packet == 1) & (joint == 0)
    advantage = abs(ps) - abs(js)
    fires = candidate & (advantage > 0)
    if fold == 1:
        with np.load(e.OUT/'B50_fallback_predictions.npz') as saved:
            for key,value in [('ids',ids),('trigger',fires),('ordered_score',ps),('joint_score',js)]:
                assert np.array_equal(value,saved[key])
    p0,ap = audit.contributions(pm,px[fires])
    j0,aj = audit.contributions(jm,x[fires])
    aj[:,:ap.shape[1]] -= ap
    groups = audit.features_by_group(width,kept,vec)
    by_feature = {k:aj[:,groups==k].sum(axis=1) for k in dict.fromkeys(groups)}
    assert np.allclose(j0-p0+sum(by_feature.values()),(js-ps)[fires])
    details = []
    summary = []
    for order,i in enumerate(np.flatnonzero(fires)):
        details.append(dict(fold='AB'[fold],label='SMB' if y[i] else 'Weibo',id=str(ids[i]),
            pcap=files[i],packet=float(ps[i]),joint=float(js[i]),advantage=float(advantage[i]),
            **audit.observation(rows[i],masks[i]),path_delta={k:float(v[order]) for k,v in by_feature.items()}))
    for label in ['SMB','Weibo']:
        subset = [r for r in details if r['label']==label]
        truth = y == (label=='SMB')
        stats = {}
        for key in ['packet','joint','advantage','可见包','收发段','最长段包数','段内间隔中位数 μs','换向间隔中位数 μs','段内跨过缺包总数','换向跨过缺包总数']:
            values = [r[key] for r in subset if r[key] is not None]
            stats[key] = [len(values),*map(float,np.percentile(values,[25,50,75]))] if values else [0]
        summary.append(dict(fold='AB'[fold],label=label,n=int(truth.sum()),candidate=int((candidate & truth).sum()),
            trigger=len(subset),single=sum(r['收发段']==1 for r in subset),
            all_cross=sum(r['每次换向都跨缺口'] for r in subset),stat=stats,
            files=dict(Counter(r['pcap'] for r in subset)),
            mean_path_delta={k:round(float(np.mean([r['path_delta'][k] for r in subset])),3) for k in by_feature},
            base_delta=j0-p0))
    assert [s['trigger'] for s in summary] == ([27,15] if fold==0 else [12,23])
    return details,summary


def main():
    lookup = {r['id']:r for r in e.old.read_panels()['train']}
    with np.load(e.OUT/'paired.npz') as data:
        parts = [examine(f,data,lookup) for f in (0,1)]
    rows = parts[0][0]+parts[1][0]
    summary = parts[0][1]+parts[1][1]
    (e.OUT/'AB50_trigger_audit.json').write_text(json.dumps(dict(summary=summary,records=rows),ensure_ascii=False,indent=1),encoding='utf-8')
    hdr=['折','真类','该类流数','逐包 SMB / 联合 Weibo','触发回退','仅一段','换向均跨缺口']
    html = audit.table(hdr,[[s[k] for k in ('fold','label','n','candidate','trigger','single','all_cross')] for s in summary])
    hdr = ['逐流指标','A · SMB','A · Weibo','B · SMB','B · Weibo']
    def distribution(q):
        return f'{q[2]:.3g} [{q[1]:.3g}, {q[3]:.3g}] (n={q[0]})' if q[0] else '无观测'
    names={'packet':'逐包分','joint':'联合分','advantage':'逐包绝对分减联合绝对分'}
    html+=audit.table(hdr,[[names.get(k,k),*[distribution(s['stat'][k]) for s in summary]] for k in summary[0]['stat']])
    html+='<p>上表为中位数 [Q1, Q3]。时间间隔先在每条流内取中位数。正分指向 SMB。下表为联合减逐包的平均路径贡献，正数向 SMB。</p>'
    html+=audit.table(hdr,[[k,*[s['mean_path_delta'][k] for s in summary]] for k in summary[0]['mean_path_delta']]+[['基准分变化',*[round(s['base_delta'],3) for s in summary]]])
    keys=['fold','label','id','pcap','packet','joint','advantage','可见包','收发段','最长段包数','段内间隔中位数 μs','换向间隔中位数 μs']
    html+=audit.table([names.get(k,k) for k in keys],[[round(r[k],4) if isinstance(r[k],float) else ('无' if r[k] is None else r[k]) for k in keys] for r in rows])
    (e.OUT/'AB50_trigger_audit.html').write_text('<!doctype html><meta charset=utf-8><title>50% 丢包回退诊断</title><style>body{font:16px system-ui;margin:3%}table{border-collapse:collapse;border-block:2px solid;margin:1.5em 0}thead{border-bottom:1px solid}td,th{padding:.4em .6em;text-align:left}</style><h1>D4 两折 50% 丢包回退诊断</h1>'+html,encoding='utf-8')
    for s in summary:
        print(json.dumps(s,ensure_ascii=False))


if __name__=='__main__':
    with threadpool_limits(limits=3):
        main()
