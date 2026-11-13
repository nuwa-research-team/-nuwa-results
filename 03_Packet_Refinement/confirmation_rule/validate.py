"""Replay the frozen visible-context rule on the original confirmation mask."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ARCHIVE = OUT.parent / 'confirm.npz'
PANEL = OUT.parents[2] / 'd4_full_aligned_20261113/dev_readout/panels.json.gz'


def apply_rule(visible_slots, visible, missing, original):
    result = original.copy()
    fired = []
    for j in np.flatnonzero(original == -52):
        k = int(np.searchsorted(visible_slots, missing[j]))
        if not 0 < k < len(visible) or visible[k-1] != 1500 or visible[k] != -52:
            continue
        start = k - 1
        while start > 0 and visible[start-1] > 0:
            start -= 1
        if k - start < 16:
            result[j] = 1500
            fired.append((int(j), k-start))
    return result, fired


def metrics(rows):
    truth, old, new = (np.concatenate([r[k] for r in rows]) for k in ('truth','old','new'))
    changed = old != new
    result = dict(flows=len(rows), slots=len(old), triggers=int(changed.sum()),
                  triggered_flows=sum(bool(np.any(r['new']!=r['old'])) for r in rows),
                  fixed=int(np.sum(changed & (truth==new))), harmed=int(np.sum(changed & (truth==old))),
                  both_wrong=int(np.sum(changed & (truth!=new) & (truth!=old))))
    for key, pred in [('before',old),('after',new)]:
        errors=np.abs(np.abs(truth)-np.abs(pred))
        result[key]=dict(absolute_error=int(errors.sum()),mae=float(errors.mean()),
                         signed_hits=int(np.sum(pred==truth)),signed_hit_pct=float(100*np.mean(pred==truth)))
    assert result['after']['signed_hits']-result['before']['signed_hits']==result['fixed']-result['harmed']
    return result


def main():
    with gzip.open(PANEL, 'rt') as f:
        panel=json.load(f)['confirm']
    rows, events = [], []
    off=0
    with np.load(ARCHIVE, allow_pickle=False) as z:
        ids,y,flat,counts,baseline,truth=(z[key] for key in ('ids','y','q','nq','current','truth'))
        assert len(panel)==len(set(ids))==512
        for row, key, label, n in zip(panel, ids, y, counts, strict=True):
            assert row['id']==key and row['y']==label
            sl=slice(off,off+n)
            q=flat[sl]
            assert np.array_equal(q,row['masks']['50']) and len(np.unique(q))==n
            seq=np.asarray(row['s'],dtype=int)
            old=baseline[sl].astype(int)
            observed=np.setdiff1d(np.arange(len(seq)),q)
            new, fired=apply_rule(observed,seq[observed],q,old)
            t=truth[sl].astype(int)
            assert np.array_equal(seq[q],t)
            family=key.split('.pcap#')[0].split('-')[0]
            rows.append(dict(family=family,old=old,new=new,truth=t))
            for j,run in fired:
                events.append(dict(flow=str(key),slot=int(q[j]),left_run=run,truth=int(t[j]),
                                   before=-52,after=1500))
            off+=n
        assert off==len(flat)==len(baseline)==len(truth)
    groups={g:metrics([r for r in rows if r['family']==g]) for g in sorted({r['family'] for r in rows})}
    groups['ALL']=metrics(rows)
    assert len(events)==groups['ALL']['triggers']
    report=dict(rule='original == -52, nearest visible neighbors (+1500,-52), outward left visible positive run <16; set to +1500',
                panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),
                predictions_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),groups=groups,events=events)
    (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(groups,indent=1))


if __name__=='__main__':
    main()
