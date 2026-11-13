"""Replay FS-Net on paired development flows, changing only short imputed signs."""
import gzip
import hashlib
import json
import sys
from html import escape
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
DEV = BASE / 'd4_full_aligned_20261113/dev_readout'
sys.path.insert(0, str(BASE / 'd4_downstream_random'))
import downstream as ds

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def metrics(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y != 1) & (pred == 1)).sum())
    n = int((y == 1).sum())
    return dict(miuref=n, tp=tp, fp=fp, recall=100*tp/n, f1=200*tp/(n+tp+fp),
                htbot=int((y == 0).sum()), htbot_to_virut=int(((y == 0) & (pred == 5)).sum()))

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    identity = json.loads((DEV / 'identity.json').read_bytes())
    assert sha(DEV/'panels.json.gz') == identity['panels']
    weight_sha = sha(ds.REPAIRED/'repaired/repaired_fsnet.pt')
    assert weight_sha == json.loads((DEV.parent/'cloud_code/identity.json').read_bytes())['classifiers']['fsnet']
    model = ds.model_for('fsnet', 8).eval()
    panels = json.loads(gzip.decompress((DEV/'panels.json.gz').read_bytes()))
    ids = [r['id'] for tag in ('train','screen','confirm') for r in panels[tag]]
    assert len(ids)==len(set(ids))
    outputs = {}
    saved = {}
    paired = []
    hashes = {}
    for tag in ('screen','confirm'):
        rows = panels[tag]
        path = DEV/f'{tag}_pred.npz'
        hashes[tag] = sha(path)
        with np.load(path) as z:
            keys = z['ids'].astype(str)
            y = z['y']
            assert keys.tolist()==[r['id'] for r in rows] and y.tolist()==[r['y'] for r in rows]
            assert all(r['id'].startswith({0:'Htbot',1:'Miuref',5:'Virut'}.get(r['y'],'')) for r in rows)
            q = z['q_50']
            nq = z['nq_50']
            b = z['B_50_packet'].astype(int)
            t = z['TFI_50_packet'].astype(int)
            assert np.all(b != 0) and np.array_equal(q,np.concatenate([r['masks']['50'] for r in rows]))
            assert nq.tolist()==[len(r['masks']['50']) for r in rows] and q.size==b.size==t.size
            original = [np.asarray(r['s']) for r in rows]
            for seq,r in zip(original,rows,strict=True):
                pos = r['masks']['50']
                assert len(pos)==len(set(pos)) and min(pos)>=0 and max(pos)<len(seq)
            assert np.array_equal(np.concatenate([seq[r['masks']['50']] for seq,r in zip(original,rows)]),z['clean_50_packet'])
            eligible = (np.abs(b)<=128) & (t!=0)
            hybrid = b.copy()
            hybrid[eligible] = np.abs(b[eligible]) * np.sign(t[eligible])
            changed = hybrid != b
            assert np.array_equal(abs(hybrid),abs(b)) and not changed[~eligible].any()
            assert np.all(np.sign(hybrid[eligible])==np.sign(t[eligible]))
            arms = {}
            for arm, packed in [('B',b),('swap',hybrid),('TFI',t)]:
                seqs = []
                offset = 0
                for i,src in enumerate(original):
                    seq = src.copy()
                    end = offset+nq[i]
                    pos = q[offset:end]
                    seq[pos] = packed[offset:end]
                    seqs.append(seq)
                    offset = end
                assert offset==len(packed)
                pred = ds.er.classify(model,seqs,np.arange(len(rows)),'cpu',48,1/1600)
                if arm != 'swap':
                    assert np.array_equal(pred,z[f'{arm}_50_fsnet']), (tag,arm)
                arms[arm] = pred
            yy = np.repeat(y,nq)
            saved.update({tag+'_ids':keys, tag+'_y':y, tag+'_q':q, tag+'_nq':nq, tag+'_changed':changed,
                          **{tag+'_'+k:v for k,v in arms.items()}})
            paired.append((y,arms))
            outputs[tag] = {'flows':len(y), 'masked':len(q), 'eligible':int(eligible.sum()),
                            'no_tfi_direction':int(((abs(b)<=128)&(t==0)).sum()),'flipped':int(changed.sum()),
                            'miuref_flips':int(changed[yy==1].sum()),'htbot_flips':int(changed[yy==0].sum()),
                            'arms':{k:metrics(y,pr) for k,pr in arms.items()}}
    y = np.concatenate([v[0] for v in paired])
    preds = {k:np.concatenate([v[1][k] for v in paired]) for k in ('B','swap','TFI')}
    out = {k:metrics(y,v) for k,v in preds.items()}
    meta = dict(panels=identity['panels'], predictions=hashes, fsnet=weight_sha, cutoff_bytes=128,
                short_definition='0 < abs(B prediction) <= 128 at archived missing slots; preserve B if TFI has no sign',
                panels_detail=outputs, pooled=out)
    (OUT/'results.json').write_text(json.dumps(meta,indent=2), encoding='utf8')
    np.savez_compressed(OUT/'paired_predictions.npz',**saved)
    title = 'D4 开发集 · 50% 丢包 · FS-Net 短包方向替换'
    content = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px system-ui;max-width:1040px;margin:44px auto;padding:0 20px;color:#182939}table{border-collapse:collapse;border-top:2px solid;border-bottom:2px solid;width:100%}th,td{padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}thead{border-bottom:1px solid}</style><h1>'+title+'</h1>',
               '<p>仅替换原缺包位置中双路预测幅值 ≤128 字节、且本地 TFI 有方向的包。FS-Net 冻结，保留原面板的顺序与批次。合并结果由两组逐流预测计算。F1 为 Miuref 的单类 F1，误报统计覆盖面板全部流。</p>']
    labels={'B':'改前 · 双路直接数值','swap':'改后 · 仅换短包方向','TFI':'本地 TFI'}
    for label, group in [('开发集合并',out),('开发筛选组',outputs['screen']['arms']),('开发确认组',outputs['confirm']['arms'])]:
        content.append('<h2>'+label+'</h2><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ['方法','Miuref 判对/总数','Miuref 误报','Miuref 召回率','Miuref F1','Htbot → Virut / Htbot 总数'])+'</tr></thead><tbody>')
        for k,m in group.items():
            cells=[labels[k],f"{m['tp']}/{m['miuref']}",str(m['fp']),f"{m['recall']:.2f}%",f"{m['f1']:.2f}%",f"{m['htbot_to_virut']}/{m['htbot']}"]
            content.append('<tr>'+''.join('<td>'+escape(c)+'</td>' for c in cells)+'</tr>')
        content.append('</tbody></table>')
    (OUT/'FSNet_short_sign.html').write_text('\n'.join(content)+'</html>',encoding='utf8')
    print(json.dumps(meta,indent=2))

if __name__=='__main__':
    main()
