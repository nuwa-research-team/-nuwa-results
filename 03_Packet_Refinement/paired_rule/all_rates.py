"""Evaluate 10/30% and join the archived 50% replay for the same D4 flows."""
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from evaluate import OUT, FULL, D, OLD, NAMES, ds, sha, replay, classify, table

DEST = OUT/'all_rates'


def counts(events):
    groups = {}
    for group,label in [('SMB',4),('Weibo',6),('其他类别',None)]:
        selected=[e for e in events if (e['label']==label if label is not None else e['label'] not in (4,6))]
        for position,at33 in [('第33包',True),('其他位置',False)]:
            section=[e for e in selected if (e['packet']==33)==at33]
            fixed=sum(e['truth']==1500 for e in section)
            harmed=sum(e['truth']==-52 for e in section)
            groups[group+'·'+position]=[len(section),len({e['flow'] for e in section}),fixed,harmed,len(section)-fixed-harmed]
    return groups


def run_rate(rate, flows, ids, y, rawseqs, weights, anchor):
    paths={'mask':D/f'mask{rate}.npz','before':OLD/f'packets_{rate}.npz','tfi':ds.PE/f'tfi_imputed_{rate}.npz',
           'rule':OUT.parent/'confirm_signed_rule/validate.py'}
    hashes={key:sha(path) for key,path in paths.items()}
    packet,events,completed,changed=replay(flows,ids,hashes,rate)
    with np.load(paths['before']) as z:
        nslots=len(z['q'])
    net=sum((e['truth']==1500)-(e['truth']==-52) for e in events)
    assert np.isclose((packet[1][1]-packet[0][1])*nslots/100,net)
    assert np.allclose(np.array(packet)[[0,2],0],np.array(anchor['packet'])[[2,4],0])
    groups=counts(events)
    print(rate,nslots,groups,packet,flush=True)
    scores={}
    predictions={}
    for key,name in NAMES.items():
        vals,predictions[key]=classify(key,ids,y,completed,changed,rawseqs,weights,rate)
        assert np.allclose(np.array(vals)[[0,2]],np.array(anchor['classifiers'][name])[[2,4]])
        scores[name]=vals
        print(rate,name,vals[1],flush=True)
    scores['六分类器均值']=np.mean(list(scores.values()),axis=0).tolist()
    np.savez_compressed(DEST/f'predictions_{rate}.npz',ids=ids,y=y,**predictions)
    return dict(flows=len(ids),masked=nslots,changed_flows=len(changed),events=events,counts=groups,
                hashes=hashes,packet=packet,classifiers=scores)


def render(rates):
    rows=[]
    event_rows=[]
    for rate,r in rates.items():
        prefix=f'{rate}%'
        rows.append([prefix,'包长 MAE（字节）',*[f'{a:.3f}' for a,b in r['packet']]])
        rows.append([prefix,'带方向精确命中率（%）',*[f'{b:.3f}' for a,b in r['packet']]])
        for name,vals in r['classifiers'].items():
            rows.append([prefix,name,*[f'{a:.3f} / {b:.3f}' for a,b in vals]])
        event_rows.extend([[prefix,group,*vals] for group,vals in r['counts'].items()])
    html = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 三档丢包对照</title>'
            '<style>body{font:16px system-ui;max-width:1000px;margin:32px auto}table{border-collapse:collapse;'
            'width:100%;border-block:2px solid;margin:16px 0 32px}thead{border-bottom:1px solid}th,td{'
            'padding:5px 8px;text-align:right}td:nth-child(2),th:nth-child(2){text-align:left}</style>'
            '<h1>D4 同流同掩码 · 三档回放</h1><p>各档均为 13,150 条流；分类器数字为准确率 / 八类 macro-F1（%）。'
            '包级指标按该档全部缺包位置计算。带方向精确命中要求包长和正负号同时正确。</p>')
    html+=table(['丢包率','指标或分类器','改前双路','加入规则','完整本地 TFI'],rows)
    html+=table(['丢包率','触发位置','触发','涉及流','改对','原对改错','前后均错'],event_rows)
    (DEST/'comparison.html').write_text(html+'</html>',encoding='utf8')


def main():
    DEST.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    flows=ds.read(D/'flows.jsonl')
    ids=np.array([r['id'] for r in flows])
    y=np.array([r['y'] for r in flows])
    with np.load(OUT/'classifier_predictions.npz') as z:
        assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['y'])
    assert len(ids)==len(set(ids))==13150 and set(y)==set(range(8))
    raw=pickle.loads(ds.RAW.read_bytes())
    assert set(ids)<=set(raw['ids'][i] for i in raw['split']['test'])
    rawseqs=dict(zip(raw['ids'],raw['seqs']))
    archive=json.loads((OUT/'results.json').read_bytes())
    weights=json.loads((FULL/'cloud_code/identity.json').read_bytes())['classifiers']
    assert archive['weights']==weights and sha(OUT.parent/'confirm_signed_rule/validate.py')==archive['hashes']['rule']
    assert sha(D/'mask50.npz')==archive['hashes']['mask'] and counts(archive['events'])==archive['counts']
    anchor=json.loads((OLD/'full_results.json').read_bytes())['rates']
    rates={str(r):run_rate(r,flows,ids,y,rawseqs,weights,anchor[str(r)]) for r in (10,30)}
    rates['50']=archive
    result=dict(rates=rates,arms=archive['arms'],counts_columns=archive['counts_columns'],
                packet_columns=archive['packet_columns'],classifier_columns=archive['classifier_columns'],
                weights=weights,reused_50_sha256=sha(OUT/'results.json'))
    (DEST/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    render(rates)


if __name__=='__main__':
    main()
