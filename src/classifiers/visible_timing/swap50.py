"""Paired signed-length substitutions on two prespecified error cohorts."""
import gzip
import html
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from ablate import DATA, FULL, TimeReadout, features
from logits import BASE, OUT, load_inputs, sha

ARMS = ('dual', 'direction', 'magnitude', 'both')
LABELS = ('Weibo', 'SMB', 'Htbot', 'Virut', 'Miuref', '其他合并类')


def variants(a, b, mask):
    a,b = np.asarray(a,dtype=np.int32), np.asarray(b,dtype=np.int32)
    assert len(a)==len(b)==len(mask) and np.array_equal(a[~mask],b[~mask])
    assert (a[mask]!=0).all() and (b[mask]!=0).all()
    d,m,joint = a.copy(),a.copy(),a.copy()
    d[mask] = np.sign(b[mask])*np.abs(a[mask])
    m[mask] = np.sign(a[mask])*np.abs(b[mask])
    joint[mask] = np.sign(d[mask])*np.abs(m[mask])
    assert np.array_equal(joint,b)
    return a,d,m,joint


def length_logits(net,kind,head,seqs):
    groups = {}
    for i,s in enumerate(seqs):
        groups.setdefault(max(32,1<<(len(s)-1).bit_length()),[]).append(i)
    out = torch.empty(len(seqs),6)
    for width, group in groups.items():
        for start in range(0,len(group),24):
            ix = group[start:start+24]
            xs = torch.zeros(len(ix),width)
            lens = torch.tensor([len(seqs[i]) for i in ix])
            for j,i in enumerate(ix):
                xs[j,:lens[j]] = torch.as_tensor(seqs[i])
            rep = net(xs[:,None,:]) if kind=='lstm' else net(xs/1600,lens)
            out[ix] = nn.functional.linear((rep-head['mean'])/head['scale'],**head['state'])
    return out


@torch.inference_mode()
def run():
    torch.set_num_threads(4)
    archive = np.load(OUT/'predictions.npz')
    meta = json.loads((OUT/'metrics.json').read_bytes())
    mask_path = DATA/'mask50.npz'
    assert sha(mask_path)==meta['audit']['50']['mask_sha']
    with np.load(mask_path) as f:
        assert np.array_equal(f['ids'],archive['ids'])
        masks = f['mask']
    for path in [FULL/'verified.json.gz',BASE/'six_data.pkl.gz']:
        assert sha(path)==meta['hashes'][path.name]
    data = load_inputs()
    assert all(np.array_equal(data[tag][1],archive['y']) for tag in ('dual50','tfi50'))
    times = json.loads(gzip.decompress((FULL/'verified.json.gz').read_bytes()))
    timing = torch.load(OUT/'timing_heads.pt',map_location='cpu',weights_only=True)
    verified = json.loads((BASE/'verified.json').read_bytes())['hashes']
    assert sha(BASE/'six_inputs.tgz')==verified['six_inputs.tgz']
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        with tarfile.open(BASE/'six_inputs.tgz') as tf:
            for name in ('models.py','repaired/models_repaired.py','fsnet.pt','lstm.pt'):
                dest = folder/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(tf.extractfile(name).read())
        sys.path[:0]=[tmp,str(folder/'repaired')]
        import models
        import models_repaired
        for kind,target in [('lstm',4),('fsnet',2)]:
            a,b = (archive[f'{kind}_{tag}50_after'] for tag in ('dual','tfi'))
            subset = np.flatnonzero((archive['y']==target)&(b==target)&((a!=target) if kind=='lstm' else (a==5)))
            assert len(subset)==(71 if kind=='lstm' else 38)
            groups = {arm:[] for arm in ARMS}
            inputs = []
            for i in subset:
                key = str(archive['ids'][i])
                sa,sb = (data[tag][0][i] for tag in ('dual50','tfi50'))
                ts = times[key][1]
                assert len(ts)==len(sa) and not masks[i,len(sa):].any()
                mask = masks[i,:len(sa)]
                for arm,seq in zip(ARMS,variants(sa,sb,mask),strict=True):
                    groups[arm].append(seq)
                inputs.append(features(ts,~mask)[0])
            x = np.nan_to_num((np.array(inputs)-timing['mean'].numpy())/timing['std'].numpy(),nan=0.)
            time_head = TimeReadout().eval()
            time_head.load_state_dict(timing['weights'][kind])
            delta = time_head(torch.tensor(x,dtype=torch.float32))
            net = (models if kind=='lstm' else models_repaired).build(kind,8)
            net.load_state_dict(torch.load(folder/(kind+'.pt'),map_location='cpu',weights_only=True)['state'])
            if kind=='lstm':
                net.fc = nn.Identity()
            else:
                net.head[-1] = nn.Identity()
            net.eval()
            hp = BASE/f'six_{kind}.pt'
            assert sha(hp)==verified[hp.name]
            head = torch.load(hp,map_location='cpu',weights_only=True)
            preds = {arm:(length_logits(net,kind,head,seqs)+delta).argmax(1).numpy()
                     for arm,seqs in groups.items()}
            assert np.array_equal(preds['dual'],a[subset]) and np.array_equal(preds['both'],b[subset])
            d = preds['direction']==target
            m = preds['magnitude']==target
            print(kind,len(subset),'direction',d.sum(),'magnitude',m.sum(),'only D, only M, both alone, neither',[(d&~m).sum(),(m&~d).sum(),(d&m).sum(),(~d&~m).sum()])
            for j,i in enumerate(subset):
                records.append([kind,str(archive['ids'][i]),LABELS[target],*[LABELS[preds[k][j]] for k in ARMS]])
    return records


if __name__=='__main__':
    records = run()
    rows = [['分类器','流 ID','真类','原双路','只换方向','只换绝对包长','同时换两项'],*records]
    table = ''.join('<tr>'+''.join(f'<td>{html.escape(v)}</td>' for v in r)+'</tr>' for r in rows)
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 50% 丢包逐流核对</title><style>body{max-width:1050px;margin:3rem auto;font:16px/1.5 system-ui}table{border-collapse:collapse;border-block:2px solid}td{padding:.32rem .8rem}tr:first-child{font-weight:bold;border-bottom:1px solid}</style><h1>D4 · 50% 丢包逐流核对</h1><p>固定可见包时间分支，替换值均取自同位置的本地 TFI 预测。</p><table>'+table+'</table></html>'
    (OUT/'D4_IAT_50_swap.html').write_text(page,encoding='utf-8')
