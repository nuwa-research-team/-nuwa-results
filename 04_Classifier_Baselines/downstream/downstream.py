"""Score frozen classifiers on identical flows under each imputer and loss rate."""
import argparse
import csv
import importlib.util
import json
import pickle
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from prepare import OUT, NUWA, PE, RAW, SNAP, RATES, read

REPAIRED = NUWA/'audit/repaired_snapshot_20261110'
sys.path.insert(0,str(NUWA/'shared'))
import paired_evaluation_v3 as pv
sys.path.insert(0,str(REPAIRED/'repaired'))
import eval_repaired as er
NAMES = ['cnn','lstm','gru','tst','sdae','fsnet']
DISPLAY = dict(zip(NAMES,['CNN','LSTM','GRU','TST','SDAE','FS-Net']))


def archive(name,method,r):
    if name in NAMES[:3]:
        return PE/f'{name}_{method}_{r:02}.npz'
    return REPAIRED/'work/repaired_eval_20261110'/f'{name}_{method}_{r}.npz'


def summarize(y,pred):
    return dict(accuracy=100*accuracy_score(y,pred),
                macro=100*f1_score(y,pred,labels=np.unique(y),average='macro',zero_division=0),
                weighted=100*f1_score(y,pred,average='weighted',zero_division=0),n=len(y))


def model_for(name,n):
    if name in NAMES[:3]:
        spec=importlib.util.spec_from_file_location('legacy',SNAP/'src/models.py')
        legacy=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        model=legacy.build(name,n)
        path=SNAP/f'work/res_aligned_{name}/{name}.pt'
    else:
        model=er.SDAERepaired(n) if name=='sdae' else er.MR.build(name,n)
        path=REPAIRED/f'repaired/repaired_{name}.pt'
    ck=torch.load(path,weights_only=True,map_location='cpu')
    model.load_state_dict(ck['state'])
    return model.eval()


def predict(name,method,r,flows):
    dest=OUT/f'{name}_{method}_{r}.npz'
    if dest.exists():
        return
    d=pickle.loads(RAW.read_bytes())
    lookup=dict(zip(d['ids'],range(len(d['ids']))))
    baseline=np.load(archive(name,'clean',0))
    ids=baseline['ids'].astype(str)
    chosen={f['id']:f for f in flows}
    completed={}
    if method!='deletion':
        completed={f['id']:f['signed'] for f in read(OUT/f'{method}_{r}.jsonl')}
        assert completed.keys()==chosen.keys()
    seqs=[d['seqs'][lookup[k]] for k in ids]
    assert all(y==d['y'][lookup[k]] for k,y in zip(ids,baseline['y']))
    z=np.load(OUT/f'mask{r}.npz')
    masks=dict(zip(z['ids'].astype(str),z['mask']))
    for i,(k,s) in enumerate(zip(ids,seqs)):
        if k not in chosen:
            # Preserve original batch width, even for rows excluded from scoring.
            if method=='deletion':
                seqs[i]=s[max(1,round(len(s)*r/100)):]
            continue
        m=masks[k][:len(s)]
        if method=='deletion':
            seqs[i]=np.array(s)[~m].tolist()
        else:
            seqs[i]=completed[k]
            assert len(seqs[i])==len(s) and np.array_equal(np.array(seqs[i])[~m],np.array(s)[~m])
    model=model_for(name,len(d['labels']))
    idx=np.arange(len(ids))
    if name in NAMES[:3]:
        pred=pv.classify(model,seqs,idx,'cpu',512)
    else:
        pred=er.classify(model,seqs,idx,'cpu',16 if name=='tst' else 48,1/1600)
    valid=np.isin(ids,list(chosen))
    np.savez_compressed(dest,ids=ids[valid],y=baseline['y'][valid],pred=pred[valid])


def report(flows):
    cohort={f['id']:f['y'] for f in flows}
    ref={}
    file=NUWA/'viewer/Nuwa_Experiment_Results_20261110/paper_reference_extraction.csv'
    for r in csv.DictReader(file.open(encoding='utf-8-sig')):
        if r['dataset']=='d4' and r['environment']=='simulated' and r['table']=='XVIII' and r['imputer']=='TFI':
            ref[r['classifier'], round(float(r['loss_rate'])*100),r['metric']]=r['paper_value_percent']
    txt=['D4 随机丢包，下游分类结果，Accuracy / macro-F1（%）。论文 F1 的平均方式未注明。', '',
         f'本地有效测试流 {len(flows)} 条，覆盖 {len(set(cohort.values()))} 类。论文 D4 为 6 类，50% 未报告。',
         '本地 TFI 为已有复现权重。Qwen 为 D4 真实包长训练的 0.6B 学生。无效 JSON 的缺失位置按 0 回填，保留全部会话。',
         '', '| 丢包 | 分类器 | 仅删除 | 本地 TFI | Qwen 普通 | Qwen 行为 | 论文 TFI |',
         '|---|---|---|---|---|---|---|']
    metrics=[]
    for r in RATES:
        for name in NAMES:
            cells=[]
            for method in ['deletion','tfi','plain','behavior']:
                path=archive(name,method,r) if r in (10,30) and method in ['tfi','deletion'] else OUT/f'{name}_{method}_{r}.npz'
                if not path.exists():
                    cells.append('待测')
                    continue
                z=np.load(path)
                ids=z['ids'].astype(str)
                select=np.isin(ids,list(cohort))
                assert set(ids[select])==cohort.keys() and select.sum()==len(cohort)
                y=z['y'][select]
                assert y.tolist()==[cohort[k] for k in ids[select]]
                s=summarize(y,z['pred'][select])
                metrics.append(dict(classifier=name,method=method,loss_pct=r,**s))
                cells.append(f"{s['accuracy']:.2f} / {s['macro']:.2f}")
            paper=' / '.join(ref[DISPLAY[name],r,m] for m in ['Accuracy','F1']) if r!=50 else '未报告'
            txt.append('| '+' | '.join([f'{r}%',DISPLAY[name],*cells,paper])+' |')
    (OUT/'comparison.txt').write_text('\n'.join(txt)+'\n',encoding='utf-8-sig')
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('method',choices=['report','deletion','plain','behavior'])
    ap.add_argument('--rate',type=int,choices=RATES)
    a=ap.parse_args()
    torch.set_num_threads(3)
    flows=read(OUT/'flows.jsonl')
    if a.method!='report':
        assert a.rate in RATES
        for name in NAMES:
            predict(name,a.method,a.rate,flows)
            print(a.method,a.rate,name,flush=True)
            report(flows)
    else:
        report(flows)

