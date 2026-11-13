"""Paired frozen FS-Net replay using the previous short-sign experiment's cohort."""
import gzip
import json
import sys
from html import escape
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
PREV = OUT.parent / 'dev_sign_swap'
sys.path.insert(0, str(PREV))
from run import DEV, ds, metrics, sha

ARMS = {'B': '原双路', 'swap': '只换方向', 'length': '只换包长', 'TFI': '完整本地 TFI'}
GROUPS = {'confirm': '确认组', 'screen': '筛选组', 'pooled': '合并'}

def html_table(headers, rows):
    head = ''.join('<th>'+escape(h)+'</th>' for h in headers)
    body = ''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)
    return '<table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table>'

@torch.inference_mode()
def main():
    old = json.loads((PREV/'results.json').read_bytes())
    assert old['cutoff_bytes']==128 and sha(DEV/'panels.json.gz')==old['panels']
    assert sha(ds.REPAIRED/'repaired/repaired_fsnet.pt')==old['fsnet']
    panels = json.loads(gzip.decompress((DEV/'panels.json.gz').read_bytes()))
    torch.set_num_threads(4)
    model = ds.model_for('fsnet',8).eval()
    outputs, details, counts, data = {}, {}, {}, {}
    with np.load(PREV/'paired_predictions.npz') as previous:
        for tag in ('screen','confirm'):
            rows = panels[tag]
            path = DEV/f'{tag}_pred.npz'
            assert sha(path)==old['predictions'][tag]
            with np.load(path) as z:
                ids,y,q,nq = (z[k] for k in ('ids','y','q_50','nq_50'))
                for key,v in [('ids',ids),('y',y),('q',q),('nq',nq)]:
                    assert np.array_equal(previous[f'{tag}_{key}'],v)
                    data[f'{tag}_{key}']=v
                assert ids.tolist()==[r['id'] for r in rows] and y.tolist()==[r['y'] for r in rows]
                assert np.array_equal(q,np.concatenate([r['masks']['50'] for r in rows]))
                assert nq.tolist()==[len(r['masks']['50']) for r in rows]
                b,t=(z[k].astype(int) for k in ('B_50_packet','TFI_50_packet'))
                keep=(np.abs(b)<=128)&(t!=0)
                assert (b!=0).all() and keep.sum()==old['panels_detail'][tag]['eligible']
                sign=b.copy()
                sign[keep]=np.sign(t[keep])*np.abs(b[keep])
                assert np.array_equal(sign!=b,previous[f'{tag}_changed'])
                length=b.copy()
                length[keep]=np.sign(b[keep])*np.abs(t[keep])
                assert np.array_equal(np.sign(length),np.sign(b))
                assert np.array_equal(length[~keep],b[~keep]) and np.array_equal(abs(length[keep]),abs(t[keep]))
                counts[tag]=dict(eligible=int(keep.sum()),length_changed=int((length!=b).sum()),masked=len(q))
                data[tag+'_eligible']=keep
                data[tag+'_length_packet']=length
                outputs[tag]={}
                offsets=np.r_[0,np.cumsum(nq)]
                assert offsets[-1]==len(q)==len(b)==len(t)
                for arm,values in [('B',b),('swap',sign),('length',length),('TFI',t)]:
                    seqs=[]
                    for i,r in enumerate(rows):
                        seq=np.array(r['s'])
                        sec=slice(offsets[i],offsets[i+1])
                        seq[q[sec]]=values[sec]
                        seqs.append(seq)
                    pred=ds.er.classify(model,seqs,np.arange(len(rows)),'cpu',48,1/1600)
                    if arm!='length':
                        assert np.array_equal(pred,previous[f'{tag}_{arm}'])
                    data[f'{tag}_{arm}']=pred
                    outputs[tag][arm]=metrics(y,pred)
                details[tag]=[]
                for i in np.flatnonzero((y==1)&(data[tag+'_B']!=1)&(data[tag+'_TFI']==1)):
                    details[tag].append(dict(id=str(ids[i]),**{a:int(data[f'{tag}_{a}'][i]) for a in ARMS}))
    assert len(details['confirm'])==5
    both=lambda key: np.concatenate([data[tag+'_'+key] for tag in ('screen','confirm')])
    assert len(set(both('ids')))==896
    outputs['pooled']={arm:metrics(both('y'),both(arm)) for arm in ARMS}
    result=dict(fsnet=old['fsnet'],panels=old['panels'],previous_predictions=old['predictions'],
                original_masks_and_batching_replayed=True,cutoff=128,counts=counts,metrics=outputs,
                miuref_tfi_correct_b_incorrect=details)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    np.savez_compressed(OUT/'paired_predictions.npz',**data)
    table=[]
    for group,name in GROUPS.items():
        for arm,label in ARMS.items():
            m=outputs[group][arm]
            table.append([name,label,f"{m['tp']}/{m['miuref']}",m['fp'],f"{m['recall']:.2f}%",f"{m['f1']:.2f}%",f"{m['htbot_to_virut']}/{m['htbot']}"])
    cls=['Htbot','Miuref','Neris','Nsis-ay','SMB','Virut','Weibo','Zeus']
    five=[[r['id']]+[cls[r[a]] for a in ARMS] for r in details['confirm']]
    html='<!doctype html><html lang="zh"><meta charset="utf-8"><title>FS-Net 短包方向与包长对照</title><style>body{max-width:1100px;margin:40px auto;padding:0 20px;font:16px system-ui;color:#182939}table{border-collapse:collapse;border-block:2px solid;width:100%}thead{border-bottom:1px solid}th,td{text-align:right;padding:7px}td:first-child,th:first-child{text-align:left}</style><h1>FS-Net · D4 开发集 · 50% 丢包</h1><p>两种替换均使用上次原双路预测幅值 ≤128 字节且 TFI 非零的缺包位置。只换包长保留双路正负号。合并指标按逐流预测计算。</p>'
    html+=html_table(['分组','方法','Miuref 判对/总数','Miuref 误报','Miuref 召回率','Miuref 类 F1','Htbot → Virut / Htbot 总数'],table)
    html+='<h2>确认组中原双路判错、完整 TFI 判对的五条 Miuref</h2>'+html_table(['流 ID',*ARMS.values()],five)+'</html>'
    (OUT/'FSNet_short_length.html').write_text(html,encoding='utf8')
    print(json.dumps({'counts':counts,'metrics':outputs,'confirm5':details['confirm']}))

if __name__=='__main__':
    main()
