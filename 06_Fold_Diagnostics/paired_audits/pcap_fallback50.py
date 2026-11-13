"""Aggregate archived, paired predictions by capture and standardize class mix."""
import json
from html import escape
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent


def scores(cm, smb_share):
    # Rows are truth (Weibo, SMB), columns are prediction in the same order.
    weighted = cm / cm.sum(axis=1, keepdims=True) * np.array([[1-smb_share], [smb_share]])
    return 100*np.array([weighted.trace(), np.mean(2*weighted.diagonal()/(weighted.sum(0)+weighted.sum(1)))])


def html_table(headers, rows):
    def cells(tag, values):
        return ''.join(f'<{tag}>{escape(str(v))}</{tag}>' for v in values)
    return '<table><thead><tr>'+cells('th',headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+cells('td',row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    audit = json.loads((OUT/'AB50_trigger_audit.json').read_text(encoding='utf-8'))
    records = {r['id']:r for r in audit['records']}
    assert len(records)==len(audit['records'])==77
    file_rows, fold_counts = [], {}
    with np.load(OUT/'paired.npz') as data:
        ids = data['ids']
        assert len(set(ids))==len(ids) and set(records)<=set(ids)
        for f,fold in enumerate('AB'):
            select = data['fold']==f
            names = ids[select]
            files = np.array([name.split('.pcap#')[0] for name in names])
            truth,packet,joint = [data[k][select] for k in ('y','ordered50','combined50')]
            possible = (packet==1)&(joint==0)
            trigger = np.isin(names,list(records))
            for name,pcap,label in zip(names[trigger],files[trigger],truth[trigger]):
                rec=records[name]
                assert (rec['fold'],rec['pcap'],rec['label'])==(fold,pcap,('Weibo','SMB')[label])
                assert rec['packet']>0>rec['joint'] and abs(rec['packet'])>abs(rec['joint'])
            assert possible[trigger].all()
            if f==1:
                with np.load(OUT/'B50_fallback_predictions.npz') as b:
                    assert np.array_equal(b['ids'],names) and np.array_equal(b['trigger'],trigger)
            switched=np.where(trigger,packet,joint)
            for pcap in sorted(set(files)):
                mask=files==pcap
                count=int(mask.sum())
                diff=int(possible[mask].sum())
                t=int(trigger[mask].sum())
                net=int((switched[mask]==truth[mask]).sum()-(joint[mask]==truth[mask]).sum())
                file_rows.append([fold,pcap,count,diff,t,f'{100*t/count:.2f}%',f'{net:+d}'])
            fold_counts[fold]=[np.bincount(2*truth+p,minlength=4).reshape(2,2) for p in (joint,switched)]
    total=sum(c[0] for c in fold_counts.values())
    pooled=total[1].sum()/total.sum()
    weighted_rows=[]
    for mix in ('各折实际比例','两折合并比例','两类等权'):
        for f,(before,after) in fold_counts.items():
            share={'各折实际比例':before[1].sum()/before.sum(),'两折合并比例':pooled,'两类等权':0.5}[mix]
            a=scores(before,share)
            b=scores(after,share)
            weighted_rows.append([mix,f,f'{share*100:.2f}%',f'{a[0]:.3f}',f'{b[0]:.3f}',f'{b[0]-a[0]:+.3f}',
                                  f'{a[1]:.3f}',f'{b[1]:.3f}',f'{b[1]-a[1]:+.3f}'])
    h=html_table(['折','PCAP','总流数','定向分歧','触发','触发/总流数','净多对'],file_rows)
    h+=html_table(['口径','折','SMB占比','联合准确率','回退准确率','变化(pp)','联合 macro-F1','回退 macro-F1','变化(pp)'],weighted_rows)
    (OUT/'AB50_pcap_fallback.html').write_text('<!doctype html><meta charset=utf-8><title>D4 回退规则按 PCAP 复核</title><style>body{font:16px system-ui;margin:3%}table{border-collapse:collapse;border-block:2px solid;margin:1.5em 0}thead{border-bottom:1px solid}td,th{padding:.45em .7em;text-align:right}</style><h1>D4 两折 50% 丢包</h1><p>分歧指逐包判 SMB、联合判 Weibo。触发指分歧且逐包绝对分数严格更大。净多对相对联合版。</p>'+h+'<p>标准化以各真类在整折的流数为分母，对联合与回退的混淆矩阵按相同真类比例加权，再计算准确率与 macro-F1。</p>',encoding='utf-8')
    print(json.dumps([file_rows,weighted_rows],ensure_ascii=False))


if __name__=='__main__':
    main()
