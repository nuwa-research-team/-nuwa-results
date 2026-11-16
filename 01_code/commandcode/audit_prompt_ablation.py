"""Independent accounting and recomputation of completed prompt-study artifacts."""
import csv, json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import prompt_ablation as p

out=p.OUT
co=p.api.read(out/'cohorts.json')
requests=p.api.read(out/'requests.json')
files=p.api.read(out/'manifest.json')['files']
for name,digest in files.items():
    assert p.api.sha(Path(name).read_bytes())==digest,name
audit={'utc':p.api.utc(),'frozen_files_checked':len(files),'stages':{}}
cost=[]
for stage,rows in co.items():
    stats={};base=None
    for path in sorted((out/stage).glob('*_scores.json')):
        arm=path.name.removesuffix('_scores.json')
        score=p.api.read(path);saved=p.api.read(out/stage/(arm+'_predictions.json'))
        y=np.array([r['label'] for r in rows]);truth=[];pred=[];records=[]
        for i,(r,v) in enumerate(zip(rows,saved['values'],strict=True)):
            recordpath=out/stage/arm/f'{i:04d}.json'
            if recordpath.exists():
                rec=p.api.read(recordpath);records.append(rec)
                assert rec['request_sha256']==p.api.sha(p.api.enc(requests[stage][arm][i]))
                assert rec['draw']==i and rec['model']==arm
                if rec['http_status']==200:
                    assert rec['response'].get('model')==p.MODEL
                    vs,error=p.api.validate(rec['response'],len(r['pos']))
                    assert vs==v and error==rec['validation_error']
                else:assert v is None
            truth.extend(r['truth'][k] for k in r['pos'])
            pred.extend(v if v is not None else [0]*len(r['pos']))
        truth=np.array(truth);pred=np.array(pred)
        mae=float(np.mean(np.abs(np.abs(truth)-np.abs(pred))))
        assert abs(mae-score['magnitude_mae'])<1e-9
        for k,c in saved['classes'].items():
            assert abs(100*accuracy_score(y,c)-score['classifiers'][k]['accuracy'])<1e-9
            assert abs(100*f1_score(y,c,labels=range(6),average='macro',zero_division=0)-score['classifiers'][k]['macro_f1'])<1e-9
        usage=Counter();errors=Counter();seconds=[]
        for r in records:
            errors[str(r['validation_error'])]+=1
            for k,v in r['response'].get('usage',{}).items():
                if isinstance(v,(float,int)):usage[k]+=v
            seconds.extend(a.get('seconds',0) for a in r['attempts'])
        stats[arm]={'passed':True,'draws':len(rows),'records':len(records),'errors':dict(errors),'usage':dict(usage)}
        cost.append([stage,arm,len(records),usage.get('prompt_tokens',0),usage.get('completion_tokens',0),usage.get('total_tokens',0),sum(seconds)])
    audit['stages'][stage]=stats
p.api.save(out/'independent_result_audit.json',audit)
with (out/'usage.csv').open('w',encoding='utf-8-sig',newline='') as f:
    csv.writer(f).writerows([['Stage','Arm','New responses','Prompt tokens','Completion tokens','Total tokens','Sum request seconds'],*cost])
lines=['Luna prompt optimization / 2026-11-16',
       'Task: D4 packet-length imputation, 30% original random masks, frozen six-class classifiers.',
       'No retraining. No added timestamps. Outputs retain the original signed-integer JSON schema.',
       'P0: Original messages and saved predictions reused for development; matching request hash verified.',
       'P1: P0 + three observed neighbors on either side, original indices/distances, gap start and size.',
       'P2: P1 + visible-only per-sign length modes, adjacent same-sign counts, runs, 16-slot summaries and support.',
       'P3: P2 + two training examples retrieved by visible-only length histograms, length, coverage and adjacency.',
       'Training examples: original train split only; deduplicated; excludes every complete sequence equal to a test sequence.',
       'Development: 200 class-proportionate draws from A, seed 2026111601. Masks unchanged.',
       'Confirmation: 200 fixed flows outside A and B, seed 2026111602. Original 30% masks.',
       'All cohorts, requests, source paths and hashes frozen before first new API request.',
       'Selection: mean Accuracy improves over P0, macro-F1 does not decrease. Best eligible Accuracy wins.',
       'Tie order: macro-F1, lower magnitude MAE, arm name. No prompt edits after selection.',
       'Budget: 600 development calls, conditional 400 confirmation calls; up to 3 transport attempts; no invalid-output retries.',
       'Invalid outputs: retained and zero-filled only at missing slots, as in the original comparison.',
       'Usage.csv records tokens, not a billed currency estimate. Provider billing is not independently available.',
       'Boundary: historical test pool used for exploratory study. Confirmation is held out from this prompt selection,',
       'not guaranteed independent acquisition/PCAP or an untouched final test. Rare classes have small support.',
       'No original-paper superiority claim. Tables compare local TFI under the identical local protocol.',
       'Engineering: initial launch found the per-user encrypted credential store missing; restored the previously',
       'user-provided credential to Windows DPAPI and resumed before any new API call. No key is saved in this folder.',
       'Verification: input mutation audit and train example reconstruction passed; result audit recomputes metrics',
       'from stored predictions. This is not a new training or independent classifier inference reproduction.']
for stage in ('dev','confirm'):
    scores={f.name.removesuffix('_scores.json'):p.api.read(f) for f in (out/stage).glob('*_scores.json')}
    if not scores:continue
    lines.extend(['',stage.upper(),'Method | Accuracy % | Macro-F1 % | Magnitude MAE bytes | Valid %'])
    for arm in ('TFI',*p.ARMS):
        if arm not in scores:continue
        s=scores[arm];lines.append(f"{arm} | {s['mean_accuracy']:.4f} | {s['mean_macro_f1']:.4f} | {s['magnitude_mae']:.4f} | {s['valid_pct']:.2f}")
if (out/'selection.json').exists():lines.extend(['','Selection: '+json.dumps(p.api.read(out/'selection.json'),ensure_ascii=False)])
(out/'CHANGELOG.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False))
