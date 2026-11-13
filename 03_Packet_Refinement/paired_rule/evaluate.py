"""Apply the unchanged confirmation rule to the archived D4 50% predictions."""
import gzip
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
P = OUT.parents[1]
OLD = P/'short_exit_replay'
FULL = P.parent/'d4_full_aligned_20261113'
D = P.parent/'d4_downstream_random'
sys.path[:0] = [str(OUT.parent/'confirm_signed_rule'), str(P/'full_joint_all'), str(D)]
from validate import apply_rule
from report import aligned, table, NAMES
import downstream as ds


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(rows, ids, hashes, rate=50):
    with np.load(OLD/f'packets_{rate}.npz') as p, np.load(D/f'mask{rate}.npz') as m:
        assert np.array_equal(ids,p['ids']) and np.array_equal(ids,m['ids'])
        assert hashes['mask']==str(p['mask_sha'])
        original=p['pred'].astype(int)
        positions=np.split(p['q'],np.cumsum(p['nq'])[:-1])
        masks=m['mask']
    if rate == 50:
        with gzip.open(FULL/'tfi_reference/tfi_50.jsonl.gz','rt') as f:
            tfi={r['id']:np.array(r['signed']) for r in map(json.loads,f)}
    else:
        with np.load(ds.PE/f'tfi_imputed_{rate}.npz') as z:
            lookup=dict(zip(z['ids'],range(len(z['ids']))))
            values,lengths=z['values'],z['lengths']
            assert len(lookup)==len(lengths)==len(values)
            tfi={key:values[lookup[key],:lengths[lookup[key]]] for key in ids}
    assert set(tfi)==set(ids)
    off=0
    events=[]
    changed=set()
    completed={}
    parts=[]
    for row,q,mask in zip(rows,positions,masks,strict=True):
        key=row['id']
        seq=np.array(row['signed'])
        visible=np.flatnonzero(~mask[:len(seq)])
        assert np.array_equal(np.flatnonzero(mask[:len(seq)]),q)
        assert len(seq)==len(tfi[key]) and np.array_equal(seq[visible],tfi[key][visible])
        before=original[off:off+len(q)]
        after,fired=apply_rule(visible,seq[visible],q,before)
        assert np.array_equal(np.flatnonzero(before!=after),[j for j,_ in fired])
        if fired:
            changed.add(key)
        for j,run in fired:
            events.append(dict(flow=key,label=row['y'],packet=int(q[j])+1,left_run=run,truth=int(seq[q[j]])))
        parts.append(np.stack([seq[q],before,after,tfi[key][q]]))
        seq[q]=after
        completed[key]=seq
        off+=len(q)
    assert off==len(original) and (rate!=50 or off==218419)
    true,*predictions=np.concatenate(parts,axis=1)
    packet=[[float(abs(abs(pred)-abs(true)).mean()),float(100*np.mean(pred==true))] for pred in predictions]
    return packet,events,completed,changed


@torch.inference_mode()
def classify(name, ids, y, completed, changed, rawseqs, weights, rate=50):
    legacy=name in ds.NAMES[:3]
    path=ds.SNAP/f'work/res_aligned_{name}/{name}.pt' if legacy else ds.REPAIRED/f'repaired/repaired_{name}.pt'
    assert sha(path)==weights[name]
    with np.load(ds.archive(name,'clean',0)) as z:
        keys=z['ids'].astype(str)
        archive_y=z['y']
    idx={key:i for i,key in enumerate(keys)}
    order=np.array([idx[key] for key in ids])
    assert len(keys)==len(idx) and np.array_equal(y,archive_y[order])
    before=aligned(OLD/f'{name}_{rate}.npz',ids,y)
    ref=aligned(FULL/f'baseline/{name}_tfi_{rate}.npz',ids,y,True)
    after=np.full(len(keys),-1,dtype=int)
    after[order]=before
    seqs=[completed[k] if k in completed else rawseqs[k] for k in keys]
    model=ds.model_for(name,8)
    batch=512 if legacy else 16 if name=='tst' else 48
    for start in sorted({idx[k]//batch*batch for k in changed}):
        sel=np.arange(start,min(start+batch,len(keys)))
        after[sel]=(ds.pv.classify(model,seqs,sel,'cpu',batch) if legacy else
                    ds.er.classify(model,seqs,sel,'cpu',batch,1/1600))
    after=after[order]
    unchanged=np.array([k not in changed for k in ids])
    assert np.array_equal(before[unchanged],after[unchanged]) and np.all((after>=0)&(after<8))
    vals=[]
    for pred in [before,after,ref]:
        scores=ds.summarize(y,pred)
        vals.append([scores['accuracy'],scores['macro']])
    return vals,after


def main():
    torch.set_num_threads(4)
    flows=ds.read(D/'flows.jsonl')
    ids=np.array([r['id'] for r in flows])
    y=np.array([r['y'] for r in flows])
    assert len(set(ids))==len(ids)==13150 and set(y)==set(range(8))
    hashes={k:sha(path) for k,path in {'mask':D/'mask50.npz','before':OLD/'packets_50.npz',
            'tfi':FULL/'tfi_reference/tfi_50.jsonl.gz','rule':OUT.parent/'confirm_signed_rule/validate.py'}.items()}
    packet,events,completed,changed=replay(flows,ids,hashes)
    counts={}
    for label,group in [(4,'SMB'),(6,'Weibo'),(None,'其他类别')]:
        subset=[e for e in events if (e['label']==label if label is not None else e['label'] not in (4,6))]
        for is33 in [True,False]:
            chosen=[e for e in subset if (e['packet']==33)==is33]
            counts[group+('·第33包' if is33 else '·其他位置')]=[
                len(chosen),len({e['flow'] for e in chosen}),sum(e['truth']==1500 for e in chosen),
                sum(e['truth']==-52 for e in chosen),sum(e['truth'] not in (1500,-52) for e in chosen)]
    print('packet',packet,'events',counts,flush=True)
    raw=pickle.loads(ds.RAW.read_bytes())
    assert set(ids)<=set(raw['ids'][i] for i in raw['split']['test'])
    rawseqs=dict(zip(raw['ids'],raw['seqs']))
    weights=json.loads((FULL/'cloud_code/identity.json').read_bytes())['classifiers']
    previous=json.loads((OLD/'full_results.json').read_bytes())['rates']['50']
    assert np.allclose(np.array(packet)[[0,2],0],np.array(previous['packet'])[[2,4],0])
    scores={}
    saved={}
    for key,name in NAMES.items():
        vals,saved[key]=classify(key,ids,y,completed,changed,rawseqs,weights)
        assert np.allclose(np.array(vals)[[0,2]],np.array(previous['classifiers'][name])[[2,4]])
        scores[name]=vals
        print(name,vals,flush=True)
    scores['六分类器均值']=np.mean(list(scores.values()),axis=0).tolist()
    np.savez_compressed(OUT/'classifier_predictions.npz',ids=ids,y=y,**saved)
    result=dict(flows=len(ids),masked=218419,changed_flows=len(changed),hashes=hashes,weights=weights,
                arms=['before','rule','TFI'],packet_columns=['MAE_bytes','signed_exact_pct'],packet=packet,
                classifier_columns=['accuracy_pct','macro_F1_pct'],classifiers=scores,counts_columns=[
                'triggers','flows','fixed','harmed','both_wrong'],counts=counts,events=events)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    hdr=['指标','改前','冻结规则','本地 TFI']
    data=[['包长 MAE（字节）',*[f'{a:.3f}' for a,b in packet]],
          ['带方向精确命中率（%）',*[f'{b:.3f}' for a,b in packet]]]
    data.extend([name,*[f'{a:.3f} / {b:.3f}' for a,b in vals]] for name,vals in scores.items())
    markup=('<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 50% 缺包回放</title>'
        '<style>body{font:16px system-ui;max-width:960px;margin:32px auto}table{width:100%;border-collapse:collapse;'
        'border-block:2px solid;margin:16px 0 32px}thead{border-bottom:1px solid}th,td{text-align:right;padding:6px}'
        'th:first-child,td:first-child{text-align:left}</style><h1>D4 原 50% 掩码 · 冻结规则回放</h1>'
        '<p>13,150 条流，218,419 个缺包位置。分类器单元格为准确率 / 八类 macro-F1（%）。</p>')
    markup+=table(hdr,data)+table(['类别与包序','触发','涉及流','改对','原对改错','前后均错'],[[k,*v] for k,v in counts.items()])
    (OUT/'comparison.html').write_text(markup+'</html>',encoding='utf8')


if __name__=='__main__':
    main()
