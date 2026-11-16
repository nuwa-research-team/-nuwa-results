"""Offline check of published metrics; standard library only, no API or model execution."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def macro(y,pr):
    f=[]
    for c in range(6):
        tp=sum(a==c and b==c for a,b in zip(y,pr));fp=sum(a!=c and b==c for a,b in zip(y,pr));fn=sum(a==c and b!=c for a,b in zip(y,pr));den=2*tp+fp+fn;f.append(2*tp/den if den else 0.)
    return 100*sum(f)/6
rows=read(ROOT/'02_data/prompt_cohorts/dev_200.json');y=[r['label'] for r in rows];checks=0
for directory in [ROOT/'04_prompt_study/dev',ROOT/'05_integrated_prompt/dev',ROOT/'06_luna_parameters/dev',ROOT/'07_head_pilot',ROOT/'07_head_pilot/tail_joint_map/dev',ROOT/'07_head_pilot/original_head_tuning/dev']:
    for scorefile in sorted(directory.glob('*_scores.json')):
        predfile=scorefile.with_name(scorefile.name.replace('_scores.json','_predictions.json'))
        if not predfile.exists():continue
        s=read(scorefile);z=read(predfile)
        if not isinstance(z,dict) or 'values' not in z or len(z['values'])!=200:continue
        assert len(z['classes'])==6
        ae=[]
        for r,vs in zip(rows,z['values']):
            vs=vs if vs is not None else [0]*len(r['pos']);assert len(vs)==len(r['pos'])
            ae.extend(abs(abs(r['truth'][q])-abs(v)) for q,v in zip(r['pos'],vs))
        assert abs(sum(ae)/len(ae)-s['magnitude_mae'])<1e-8,str(scorefile)
        accs=[];fs=[]
        for c,pr in z['classes'].items():
            assert len(pr)==len(y)
            acc=100*sum(a==b for a,b in zip(y,pr))/len(y);f=macro(y,pr)
            assert abs(acc-s['classifiers'][c]['accuracy'])<1e-8 and abs(f-s['classifiers'][c]['macro_f1'])<1e-8,str(scorefile)
            accs.append(acc);fs.append(f)
        assert abs(sum(accs)/6-s['mean_accuracy'])<1e-8 and abs(sum(fs)/6-s['mean_macro_f1'])<1e-8
        checks+=1
assert checks>=20
print('PASS:',checks,'saved 200-flow method results; no API calls, no training.')
