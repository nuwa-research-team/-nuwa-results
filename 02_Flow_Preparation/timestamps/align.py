import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / 'nuwa_reproduction_20261109'
RAW = PROJECT / 'audit/paired_snapshot_20261110/work/d4_aligned/raw.pkl'
SEQ = PROJECT / 'data/processed/USTC-TFC2016/sequences'

def digest(path):
    with path.open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()

def align():
    raw_hash = digest(RAW)
    assert raw_hash == '312617767197cad4c931dbc57f8fdaead8185c13e6e9e10b07458ce3ffabb3d6'
    raw = pickle.loads(RAW.read_bytes())
    eligible = {}
    for role in ('train', 'val'):
        for i in raw['split'][role]:
            name = raw['labels'][raw['y'][i]]
            if name in ('SMB', 'Weibo'):
                key = str(raw['ids'][i])
                assert key not in eligible
                eligible[key] = (int(i), role, name)
    aligned, audit = {}, {}
    for path in sorted(SEQ.glob('*.jsonl')):
        name = path.stem.rsplit('_',1)[0]
        if name.split('-')[0] not in ('SMB','Weibo'):
            continue
        meta = json.loads(path.with_suffix('.audit.json').read_text(encoding='utf8'))
        source_hash = digest(Path(meta['source']))
        assert source_hash == meta['source_sha256'] and digest(path) == meta['output_sha256']
        counts = Counter()
        counts['eligible'] = sum(k.startswith(name+'.pcap#') for k in eligible)
        for line in path.open(encoding='utf8'):
            row = json.loads(line)
            key = name+'.pcap#'+str(row['tcp_stream'])
            if key not in eligible:
                continue
            i, role, label = eligible[key]
            counts['found_id'] += 1
            assert key not in aligned and row['source_sha256'] == source_hash
            if list(raw['seqs'][i]) != row['lengths']:
                counts['sequence_mismatch'] += 1
                continue
            times = np.asarray(row['timestamps'], dtype=float)
            if len(times) != len(row['lengths']) or not np.isfinite(times).all():
                counts['bad_time'] += 1
                continue
            us = np.rint(times*1e6).astype(np.int64)
            if np.any(np.diff(us)<0):
                counts['clock_reversal'] += 1
                continue
            rel = us - us[0]
            lengths = np.asarray(row['lengths'], dtype='<i2')
            fingerprint = hashlib.sha256(lengths.tobytes()+rel.astype('<i8').tobytes()).hexdigest()
            aligned[key] = (i, role, int(label=='Weibo'), lengths, rel, fingerprint)
            counts['aligned'] += 1
        audit[name] = dict(counts, source_sha=source_hash)
    assert len(audit)==6
    (HERE/'aligned.pkl').write_bytes(pickle.dumps(aligned,protocol=5))
    (HERE/'alignment.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print({s:(v['eligible'],v.get('aligned',0)) for s,v in audit.items()},flush=True)
if __name__=='__main__':
    align()
