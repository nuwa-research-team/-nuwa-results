"""Descriptive audit restricted to the 50% screen cohort, 52 -> 1500 changes."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
DEV = OUT.parents[2] / 'd4_full_aligned_20261113' / 'dev_readout'


def screen_panel():
    # The original JSON archive orders train, screen, confirm. Decode screen alone.
    marker = b'"screen":'
    buf = b''
    with gzip.open(DEV/'panels.json.gz', 'rb') as f:
        while marker not in buf:
            chunk = f.read(32768)
            assert chunk
            buf = buf[-len(marker):] + chunk
        buf = buf.split(marker, 1)[1]
        while True:
            try:
                return json.JSONDecoder().raw_decode(buf.decode('ascii'))[0]
            except json.JSONDecodeError:
                chunk = f.read(32768)
                assert chunk
                buf += chunk


def side(lengths, times, k, step):
    if k < 0 or k >= len(lengths):
        return 0, 0, None
    val = lengths[k]
    direction = np.sign(val)
    j = k + step
    while 0 <= j < len(lengths) and np.sign(lengths[j]) == direction:
        j += step
    other = k + step
    gap = int((times[other]-times[k])*step) if 0 <= other < len(times) else None
    return int(val), int(abs(j-k)), gap


def collect():
    panel = screen_panel()
    archive = OUT.parent/'screen.npz'
    with np.load(archive,allow_pickle=False) as data:
        ids,y,qs,ns,old,new,truth = [data[k] for k in ('ids','y','q','nq','current','context','truth')]
    assert len(panel)==len(ids)==len(set(ids))==384
    off = 0
    entries = []
    for row, key, label, n in zip(panel, ids, y, ns, strict=True):
        assert row['id']==key and row['y']==label
        s = np.asarray(row['s'])
        q = qs[off:off+n]
        assert np.array_equal(q,row['masks']['50']) and np.array_equal(s[q],truth[off:off+n])
        flip = (abs(old[off:off+n])==52) & (abs(new[off:off+n])==1500)
        observed = np.setdiff1d(np.arange(len(s)),q)
        visible = s[observed]
        times = np.asarray(row['t'])[observed]
        for j in np.flatnonzero(flip):
            i = q[j]
            k = np.searchsorted(observed,i)
            left,lr,lg = side(visible,times,k-1,-1)
            right,rr,rg = side(visible,times,k,1)
            span = int(times[k]-times[k-1]) if left and right else None
            actual = int(abs(truth[off+j]))
            entries.append(dict(flow=str(key),slot=int(i),truth=actual,
                signed_truth=int(truth[off+j]), signed_old=int(old[off+j]), signed_new=int(new[off+j]),
                outcome='correct' if actual==1500 else ('damaged' if actual==52 else 'other'),
                left=left,right=right,left_run=lr,right_run=rr,left_iat_us=lg,right_iat_us=rg,
                span_us=span,interior_slots=int(observed[k]-observed[k-1]-1) if span is not None else None,
                predicted_dir=int(np.sign(old[off+j]))))
        off += n
    assert off==len(qs) and len(entries)==int(((abs(old)==52)&(abs(new)==1500)).sum())
    return entries,hashlib.sha256(archive.read_bytes()).hexdigest()


def median_iqr(vals):
    vals=[v for v in vals if v is not None]
    return [len(vals),*np.percentile(vals,[25,50,75]).tolist()] if vals else [0]


def direction(r):
    l,rgt=r['left'],r['right']
    return 'edge' if not (l and rgt) else ('same' if l*rgt>0 else 'opposite')


def summarize(items):
    flows=collections.Counter(r['flow'] for r in items)
    return dict(n=len(items),flows=len(flows),max_slots=max(flows.values(), default=0),
                dirs=dict(collections.Counter(direction(r) for r in items)),
                signs=dict(collections.Counter((str(np.sign(r['left']))+','+str(np.sign(r['right']))) for r in items)),
                feature={k:median_iqr(r[k] for r in items) for k in
                         ('left_run','right_run','span_us','left_iat_us','right_iat_us','interior_slots')},
                top_sources=collections.Counter(r['flow'].split('#')[0] for r in items).most_common(6))


if __name__=='__main__':
    entries,hashval=collect()
    groups={name:summarize([r for r in entries if r['outcome']==name])
            for name in ('correct','damaged','other')}
    rule=dict(left=1500,right=-52,left_run_strictly_below=16)
    selected=[r for r in entries if r['left']==rule['left'] and r['right']==rule['right']
              and r['left_run']<rule['left_run_strictly_below']]
    assert len({(r['flow'],r['slot']) for r in entries})==len(entries)
    (OUT/'positions.json').write_text(json.dumps(dict(screen_sha256=hashval,groups=groups,
                                                exploratory_magnitude_rule=rule,
                                                rule_screen_summary=summarize(selected),rows=entries),
                                                indent=2),encoding='utf8')
    print(json.dumps(groups,indent=1))
