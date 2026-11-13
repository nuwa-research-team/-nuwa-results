from pathlib import Path
import sys,json,numpy as np,torch,csv,hashlib
P=Path(__file__).resolve().parent;D=P/'outputs/d4_downstream_random'
sys.path.insert(0,str(D))
import downstream as ds
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    torch.set_num_threads(3)
    identity=json.loads((P/'identity.json').read_bytes())
    flows=ds.read(D/'flows.jsonl')
    out=[]
    for rate in (10,30,50):
        z=np.load(D/f'mask{rate}.npz');masks=dict(zip(z['ids'].astype(str),z['mask']))
        pred={r['id']:{} for r in flows}
        for r in ds.read(P/f'windows_{rate}.jsonl'):
            for pos,v in zip(r['q'],r['pred'],strict=True):
                k=r['a']+pos;assert k not in pred[r['id']];pred[r['id']][k]=v
        with (D/f'aligned_{rate}.jsonl').open('w') as file:
            for r in flows:
                key=r['id']; assert set(pred[key])==set(np.flatnonzero(masks[key]))
                s=list(r['signed'])
                for k,v in pred[key].items():s[k]=v
                print(json.dumps(dict(id=key,signed=s),separators=(',',':')),file=file)
        for name in ds.NAMES:
            path=ds.SNAP/f'work/res_aligned_{name}/{name}.pt' if name in ds.NAMES[:3] else ds.REPAIRED/f'repaired/repaired_{name}.pt'
            assert sha(path)==identity['classifiers'][name]
            ds.predict(name,'aligned',rate,flows)
            a=np.load(D/f'{name}_aligned_{rate}.npz');b=np.load(P/'baseline'/f'{name}_tfi_{rate:02}.npz')
            ids=a['ids'].astype(str);look={v:i for i,v in enumerate(b['ids'].astype(str))}
            idx=[look[k] for k in ids];assert len(ids)==len(set(ids))==13150 and np.array_equal(a['y'],b['y'][idx])
            y=a['y'];x=ds.summarize(y,a['pred']);ref=ds.summarize(y,b['pred'][idx])
            out.append(dict(rate=rate,clf=name,accuracy=x['accuracy'],macro_f1=x['macro'],tfi_accuracy=ref['accuracy'],tfi_macro_f1=ref['macro'],delta_accuracy=x['accuracy']-ref['accuracy'],delta_f1=x['macro']-ref['macro']))
            print(out[-1],flush=True)
    (P/'scores.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()