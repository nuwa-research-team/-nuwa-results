"""Join frozen D4 classifier flows to timestamps and freeze three evaluation masks."""
import hashlib
import json
import pickle
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
NUWA = ROOT / 'nuwa_reproduction_20261109'
SNAP = NUWA / 'audit/paired_snapshot_20261110'
PE = SNAP / 'work/paired_eval_v2_20261110'
RAW = SNAP / 'work/d4_aligned/raw.pkl'
RATES = (10, 30, 50)


def read(path):
    return [json.loads(line) for line in path.open(encoding='utf-8')]


def extract(capture, desired, used):
    from collections import Counter
    import shutil
    exe = shutil.which('tshark') or str(Path.home() / 'Desktop/Wireshark/tshark.exe')
    fields = ['tcp.stream','ip.src','tcp.srcport','ip.len','frame.time_epoch','tcp.analysis.retransmission']
    cmd = [exe,'-n','-2','-r',str(capture),'-Y','tcp && ip','-T','fields','-E','occurrence=f']
    for f in fields:
        cmd.extend(['-e', f])
    variants = [defaultdict(list), defaultdict(list)]
    origins = {}
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding='utf8') as proc:
        for line in proc.stdout:
            stream,ip,port,size,ts,retrans = line.rstrip('\r\n').split('\t')
            if not stream or not size:
                continue
            endpoint = (ip,port)
            origin = origins.setdefault(stream,endpoint)
            v = min(1600,int(size)) * (1 if origin == endpoint else -1)
            pair = (v, round(float(ts)*1e6))
            for idx in ([0] if retrans else [0,1]):
                if len(variants[idx][stream]) < 512:
                    variants[idx][stream].append(pair)
        if proc.wait():
            raise RuntimeError(f'tshark failed for {capture}')
    result = {}
    wanted = {tuple(s) for s in desired.values()}
    multiplicity = Counter(tuple(s) for s in desired.values())
    matches = defaultdict(set)
    for variant in variants:
        for key, pairs in variant.items():
            values = tuple(x for x,t in pairs)
            if values in wanted:
                matches[values].add((key, tuple(t for x,t in pairs)))
    for k, seq in desired.items():
        hits = matches[tuple(seq)]
        direct = [h for h in hits if h[0]==k]
        candidates = direct or (list(hits) if multiplicity[tuple(seq)]==1 else [])
        if len(candidates)==1:
            stream, ts = candidates[0]
            if f'{capture.name}#{stream}' not in used:
                result[k] = list(ts)
    return result


def main():
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == '312617767197cad4c931dbc57f8fdaead8185c13e6e9e10b07458ce3ffabb3d6'
    d = pickle.loads(RAW.read_bytes())
    test = np.array(d['split']['test'])
    assert not set(test) & (set(d['split']['train']) | set(d['split']['val']))
    ids = np.array(d['ids'])[test]
    seq = {key: list(map(int,d['seqs'][i])) for key,i in zip(ids,test)}
    used = set()
    pd = ROOT/'outputs/d4_prompt_distill_v1/data'
    for role in ('train','val'):
        for m in read(pd/f'{role}_manifest.jsonl'):
            cap,stream = re.fullmatch(r'.*/([^/]+\.pcap):(\d+)#w.*',m['source']).groups()
            used.add(f'{cap}#{stream}')
    archive = NUWA/'data/processed/USTC-TFC2016'
    times = {}
    for file in (archive/'sequences').glob('*.jsonl'):
        cap = file.stem.rsplit('_',1)[0]+'.pcap'
        for r in read(file):
            k = f"{cap}#{r['tcp_stream']}"
            if k in seq and k not in used and r['lengths']==seq[k]:
                times[k] = [round(t*1e6) for t in r['timestamps']]
    missing = defaultdict(dict)
    for k,s in seq.items():
        if k not in used and k not in times:
            cap,stream=k.split('#')
            missing[cap][stream]=s
    for cap, desired in missing.items():
        candidates = list((NUWA/'data/raw/USTC-TFC2016').rglob(cap)) + list((archive/'expanded').rglob(cap))
        found = extract(candidates[0],desired,used)
        times.update({f'{cap}#{k}':t for k,t in found.items()})
        print(cap, len(found),'/',len(desired),flush=True)
    te = np.array([i for i,k in enumerate(ids) if k not in used],dtype=int)
    with (OUT/'flows.jsonl').open('w') as f:
        for i in te:
            k=ids[i]
            time_values = times.get(k,[None]*len(seq[k]))
            assert len(time_values)==len(seq[k])
            print(json.dumps(dict(id=k,y=int(d['y'][test[i]]),signed=seq[k],us=time_values,verified_times=k in times),separators=(',',':')),file=f)
    sizes=np.array([len(seq[k]) for k in ids])
    audit = dict(raw_test=len(ids), excluded_shared_development_flows=len(set(ids)&used),
                 unmatched=[k for k in ids if k not in used and k not in times], retained=len(te),
                 labels=d['labels'],support=dict(Counter(d['labels'][d['y'][test[i]]] for i in te)),rates={})
    for r in RATES:
        mask=np.zeros((len(ids),512),dtype=bool)
        for j,i in enumerate(test):
            n=sizes[j]
            ix=np.random.default_rng(42+int(i)).choice(n,max(1,round(n*r/100)),replace=False)
            mask[j,ix]=True
        if r in (10,30):
            z=np.load(PE/f'mask_{r}.npz')
            assert np.array_equal(z['ids'].astype(str),ids) and np.array_equal(z['mask'],mask)
        np.savez_compressed(OUT/f'mask{r}.npz',ids=ids[te],mask=mask[te])
        audit['rates'][r]=dict(hidden=int(mask[te].sum()),total=int(sizes[te].sum()))
    (OUT/'cohort.json').write_text(json.dumps(audit,indent=2))
    print({k:v for k,v in audit.items() if k!='unmatched'},'unmatched',len(audit['unmatched']),flush=True)


if __name__=='__main__':
    main()


