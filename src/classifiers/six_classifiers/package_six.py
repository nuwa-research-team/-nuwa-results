import gzip
import json
import pickle
import sys
import tarfile
from pathlib import Path
import numpy as np
from heads import BASE,RATES

O=BASE.parent
F=O/'d4_full_aligned_20261113'
N=O/'d4_numeric_iat_20261113'
sys.path.insert(0,str(N/'context_candidates_50/full_signed_rule'))
from evaluate import ds,replay,sha

def main():
    a=json.loads((F/'six_label_audit.json').read_bytes())
    d=pickle.loads(ds.RAW.read_bytes())
    m=np.array([a['mapping'][n+'-1' if n in ('Weibo','SMB') else n] for n in d['labels']])
    rows=ds.read(ds.OUT/'flows.jsonl')
    ids=np.array([r['id'] for r in rows])
    assert len(set(ids))==len(ids)==13150
    y=np.array([r['y'] for r in rows])
    dev=json.loads(gzip.decompress((F/'dev_readout/panels.json.gz').read_bytes()))
    assert set(ids).isdisjoint({r['id'] for rr in dev.values() for r in rr})
    data={}
    for tag in ('train','screen'):
        data[tag]=([r['s'] for r in dev[tag]],m[[r['y'] for r in dev[tag]]])
    for rate in RATES:
        _,_,seqs,_=replay(rows,ids,dict(mask=sha(ds.OUT/f'mask{rate}.npz')),rate)
        if rate==50:
            with gzip.open(F/'tfi_reference/tfi_50.jsonl.gz','rt') as file:
                refs={r['id']:r['signed'] for r in map(json.loads,file)}
        else:
            with np.load(ds.PE/f'tfi_imputed_{rate}.npz') as z:
                refs={k:s[:n] for k,s,n in zip(z['ids'],z['values'],z['lengths'],strict=True)}
        for key,source in [('dual',seqs),('tfi',refs)]:
            data[f'{key}{rate}']=([np.asarray(source[k],dtype=np.int16) for k in ids],m[y])
    with gzip.open(BASE/'six_data.pkl.gz','wb',compresslevel=2) as f:pickle.dump(data,f,protocol=4)
    with tarfile.open(BASE/'six_inputs.tgz','w:gz',compresslevel=1) as tar:
        for name in ['six_protocol.txt','six_gpu.py','six_data.pkl.gz']:
            tar.add(BASE/name,arcname=name)
        tar.add(ds.SNAP/'src/models.py',arcname='models.py')
        for name in ['models_repaired.py','train_repaired.py']:
            tar.add(ds.REPAIRED/'repaired'/name,arcname='repaired/'+name)
        src='upstream/mvts_transformer/src/models/ts_transformer.py'
        tar.add(ds.REPAIRED/src,arcname=src)
        weights=json.loads((F/'cloud_code/identity.json').read_bytes())['classifiers']
        for k in ds.NAMES:
            path=ds.SNAP/f'work/res_aligned_{k}/{k}.pt' if k in ds.NAMES[:3] else ds.REPAIRED/f'repaired/repaired_{k}.pt'
            assert sha(path)==weights[k]
            tar.add(path,arcname=k+'.pt')
    print('package bytes', (BASE/'six_inputs.tgz').stat().st_size,flush=True)
if __name__=='__main__':main()
