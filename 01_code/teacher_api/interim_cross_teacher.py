"""Read-only common-subset snapshot; never alters the API experiment."""
import json
from datetime import datetime,timezone
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score
import experiment as e
import gemini_compare_v2 as g
import cross_teacher_chain as c

now=datetime.now(timezone.utc)
dest=c.DEST/('interim_'+now.strftime('%Y%m%dT%H%M%SZ'))
dest.mkdir()
rows,bodies,_,_=c.prepare()
records=[c.read(p) for p in sorted((c.DEST/'A/plain').glob('[0-9][0-9][0-9][0-9].json'))]
ids=[r['draw'] for r in records]
assert ids and len(ids)==len(set(ids))
for r in records:
    url,body=c.request_for(c.MODEL,bodies[r['draw']])
    assert r['request_sha256']==g.sha(g.encoded({'endpoint':url,'body':body}))
    values,error=c.validate(c.MODEL,r['response'],len(rows[r['draw']]['pos'])) if r['http_status']==200 else (None,'http_failure')
    assert values==r['values'] and error==r['validation_error']
selected=[rows[i] for i in ids]
values=[r['values'] for r in records]
seqs=[]
for row,vs in zip(selected,values):
    seq=row['truth'].copy()
    for p,v in zip(row['pos'],vs if vs is not None else [0]*len(row['pos'])):seq[p]=v
    seqs.append(seq)
predictions=e.classify(seqs)
datasets={c.MODEL:{'values':values,'classes':predictions}}
for model,method in [('local_TFI','local_tfi'),*[(m,'gemini_v2_'+m) for m in g.MODELS]]:
    raw=e.read('A_'+method+'_predictions.json')
    datasets[model]={'values':[raw['values'][i] for i in ids],
                     'classes':{k:[raw['classes'][k][i] for i in ids] for k in e.KINDS}}
truth=np.array([row['truth'][p] for row in selected for p in row['pos']])
y=[row['label'] for row in selected]
results={}
for model,data in datasets.items():
    flat=np.array([v for row,vs in zip(selected,data['values']) for v in (vs if vs is not None else [0]*len(row['pos']))])
    classifiers={k:{'accuracy':100*float(np.mean(np.array(v)==y)),
        'macro_f1':100*float(f1_score(y,v,labels=range(6),average='macro',zero_division=0))} for k,v in data['classes'].items()}
    results[model]={'n':len(ids),'hidden':len(truth),'classifiers':classifiers,
        'mean_accuracy':float(np.mean([v['accuracy'] for v in classifiers.values()])),
        'mean_macro_f1':float(np.mean([v['macro_f1'] for v in classifiers.values()])),
        'magnitude_mae':float(np.abs(np.abs(truth)-np.abs(flat)).mean()),
        'signed_mae':float(np.abs(truth-flat).mean())}
times=[datetime.fromisoformat(a['utc']) for r in records for a in r['attempts']]
elapsed=(now-min(times)).total_seconds()
recent=sum(0<=(now-datetime.fromisoformat(r['completed_utc'])).total_seconds()<300 for r in records)
rate=len(records)/elapsed*60
out={'utc':now.isoformat(),'draw_ids':ids,'label_counts':dict(Counter(y)),
     'errors':dict(Counter(r['validation_error'] for r in records if r['validation_error'])),
     'processed':len(records),'valid':sum(r['values'] is not None for r in records),
     'elapsed_minutes':elapsed/60,'flows_per_minute':rate,'recent_5min_completed':recent,
     'eta_minutes_first_pass':(1000-len(records))/rate,
     'results':results,'boundary':'Tiny completion-order subset, not full-cohort ranking; failed answers kept as zero'}
g.save(dest/'results.json',out)
g.save(dest/'predictions.json',datasets)
print(json.dumps(dict(out,path=str(dest)),ensure_ascii=False))
