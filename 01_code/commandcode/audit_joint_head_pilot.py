"""Read-only verification of pilot inputs, head replay, and downstream metric accounting."""
import json,pickle
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import joint_head_pilot as j

torch.set_num_threads(4)
out=j.OUT;p=j.p
for path,h in p.api.read(out/'manifest.json')['files'].items():assert p.api.sha(j.Path(path).read_bytes())==h,path
panels=p.api.read(out/'panels.json');raw=pickle.loads((p.exp.SNAP/'work/d4_aligned/raw.pkl').read_bytes())
splitsets={s:set(map(int,raw['split'][s])) for s in ('train','val','test')}
seqsets={s:{tuple(map(int,raw['seqs'][i])) for i in splitsets[s]} for s in splitsets}
for split in ('train','val'):
    for r in panels[split]:
        i=r['raw_index'];assert i in splitsets[split]
        assert r['truth']==list(map(int,raw['seqs'][i])) and r['flow']==str(raw['ids'][i])
        assert all(tuple(r['truth']) not in seqsets[k] for k in seqsets if k!=split)
        assert len(r['pos'])==round(r['rate']*len(r['truth']))
        assert r['pos']==sorted(set(r['pos']))
    assert len({r['flow'] for r in panels[split]})=={'train':2048,'val':256}[split]
assert panels['dev']==p.api.read(j.PREV/'cohorts.json')['dev']
counts=Counter(abs(r['truth'][i]) for r in panels['train'] for i in r['pos'])
vocab=[v for v,n in sorted(counts.items(),key=lambda a:(-a[1],a[0]))[:64]]
assert vocab==p.api.read(out/'protocol.json')['vocab_from_train_only']
for r in panels['train'][::64]+panels['dev']:
    n=len(r['truth']);cuts=[0,n] if n<=256 else [0,n//2,n]
    for a,b in zip(cuts,cuts[1:]):
        q=[i-a for i in r['pos'] if a<=i<b];s=r['truth'][a:b];changed=s.copy()
        for i in q:changed[i]=-1599
        assert j.inputs(s,q)==j.inputs(changed,q)
dev=torch.load(out/'dev_features.pt',weights_only=True)
report={'passed':True,'train_flows':2048,'train_views':6144,'val_flows':256,'dev_flows':200,'head_replays':{},'metrics':{},'complete_flow_confusion':{}}
for arm in ('R','J'):
    net=j.regression() if arm=='R' else j.Joint(65)
    net.load_state_dict(torch.load(out/(arm+'.pt'),weights_only=True));net.eval()
    flat=j.predict(net,dev['x'],arm,torch.tensor(vocab))
    saved=p.api.read(out/(arm+'_predictions.json'))
    assert flat==[v for row in saved['values'] for v in row]
    log=p.api.read(out/(arm+'_training.json'));assert [r['epoch'] for r in log]==list(range(1,13))
    assert all(np.isfinite(r['train_loss']) and np.isfinite(r['val_loss']) for r in log)
    report['head_replays'][arm]={'passed':True,'epochs':12,'packet_mismatches':0}
truth=np.array([r['truth'][i] for r in panels['dev'] for i in r['pos']]);y=np.array([r['label'] for r in panels['dev']])
for arm in ('clean','OLD_PB','R','J'):
    s=p.api.read(out/(arm+'_scores.json'));saved=p.api.read(out/(arm+'_predictions.json'))
    pred=truth if arm=='clean' else np.array([v for row in saved['values'] for v in row])
    assert len(pred)==len(truth)==2045
    assert abs(np.mean(np.abs(np.abs(pred)-np.abs(truth)))-s['magnitude_mae'])<1e-9
    for k,c in saved['classes'].items():
        assert len(c)==200
        assert abs(100*accuracy_score(y,c)-s['classifiers'][k]['accuracy'])<1e-9
        assert abs(100*f1_score(y,c,labels=range(6),average='macro',zero_division=0)-s['classifiers'][k]['macro_f1'])<1e-9
        if arm=='clean':report['complete_flow_confusion'][k]=confusion_matrix(y,c,labels=range(6)).tolist()
    report['metrics'][arm]={'passed':True,'mean_accuracy':s['mean_accuracy'],'mean_macro_f1':s['mean_macro_f1'],'magnitude_mae':s['magnitude_mae']}
report['training_mode_coverage']=sum(counts[v] for v in vocab)/sum(counts.values())
report['dev_mode_coverage']=float(np.isin(np.abs(truth),vocab).mean())
p.api.save(out/'independent_audit.json',report)
print(json.dumps(report,ensure_ascii=False))
