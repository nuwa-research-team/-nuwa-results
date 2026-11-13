"""Frozen within-flow matching between visible short and long packet examples."""
import gzip
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
LAST = OUT.parent / 'short_exit_replay'
sys.path.insert(0, str(LAST))
from replay import DEV, OLD, classify, ds, features, sha

DATA = OUT.parent.parent / 'd4_downstream_random'
FULL = DEV.parent
ARMS = ('current', 'context', 'TFI')


def labels_at(x):
    mag = np.abs(x)
    return np.sign(x) * np.where(mag <= 128, 1, np.where(mag < 1400, 2, 3))


def proposals(visible, q):
    """Return winner's unsigned exemplar length, or zero on insufficient support."""
    code = labels_at(visible)
    examples = np.flatnonzero(np.isin(abs(code), [1, 3]))
    chosen = np.zeros(len(q), dtype=int)
    if examples.size < 2:
        return chosen
    scores = np.zeros((len(q), len(examples)), dtype=np.int16)
    counts = np.zeros_like(scores)
    for offset in (-4, -3, -2, -1, 1, 2, 3, 4):
        pos_q, pos_e = q + offset, examples + offset
        qcode = np.where((pos_q >= 0) & (pos_q < len(code)), code[pos_q.clip(0, len(code)-1)], 0)
        ecode = np.where((pos_e >= 0) & (pos_e < len(code)), code[pos_e.clip(0, len(code)-1)], 0)
        usable = (qcode[:, None] != 0) & (ecode != 0) & (pos_q[:, None] != examples)
        counts += usable
        scores += np.where(usable, np.where(qcode[:, None] == ecode, 1, -1), 0) * (12 // abs(offset))
    scores[counts < 2] = 0
    totals = np.zeros((2, len(q)), dtype=int)
    best = np.zeros_like(totals)
    eligible = np.zeros_like(totals, dtype=bool)
    for band, cat in enumerate((1, 3)):
        keep = abs(code[examples]) == cat
        positions = examples[keep]
        if not len(positions):
            continue
        sc = np.maximum(scores[:, keep], 0)
        order = np.lexsort((abs(q[:, None]-positions), -sc), axis=1)[:, :2]
        top = np.take_along_axis(sc, order, axis=1)
        totals[band] = top.sum(axis=1)
        eligible[band] = (top > 0).sum(axis=1) == 2
        best[band] = np.abs(visible[positions[order[:, 0]]])
    for band in range(2):
        win = eligible[band] & (totals[band] >= 2 * totals[1-band])
        chosen[win] = best[band, win]
    return chosen


def route(head, sequences, positions, base, current):
    output = current.copy()
    short_indices, short_features = [], []
    off = 0
    for sequence, pos in zip(sequences, positions, strict=True):
        q = np.asarray(pos, dtype=int)
        seq = np.array(sequence, dtype=int)
        seq[q] = 0
        ix = np.arange(off, off + len(q))
        cand = proposals(seq, q)
        direct_band = np.abs(labels_at(base[ix]))
        old_band = np.abs(labels_at(current[ix]))
        ok = (direct_band != 2) & (old_band != 2)
        long = (cand >= 1400) & (old_band == 1) & ok
        output[ix[long]] = np.sign(current[ix[long]]) * cand[long]
        short = (cand > 0) & (cand <= 128) & (old_band == 3) & ok
        if short.any():
            short_indices.append(ix[short])
            short_features.append(features(seq, q[short]))
        off += len(q)
    assert off == len(base) == len(current)
    if short_indices:
        indices = np.concatenate(short_indices)
        contexts = np.concatenate(short_features)
        for start in range(0, len(indices), 4096):
            end = start + 4096
            ix = indices[start:end]
            output[ix] = np.sign(current[ix]) * np.abs(head.predict(contexts[start:end]))
    assert np.array_equal(np.sign(current), np.sign(output))
    return output


def audit(split, ids, y, sequences, positions, packets, preds):
    truth = np.concatenate([np.array(s)[q] for s, q in zip(sequences, positions, strict=True)])
    rows = {}
    before = abs(packets['current'].astype(int))
    for method, predicted in packets.items():
        length = abs(predicted.astype(int))
        error = abs(abs(truth) - length)
        classification = preds[method]
        h = y == 0
        r = dict(flows=len(ids), slots=len(truth), ht_total=int(h.sum()),
                 short_support=int((abs(truth)==40).sum()), long_support=int((abs(truth)==1500).sum()),
                 mae=float(error.mean()), abs_error=int(error.sum()), changed=int((before!=length).sum()),
                 ht_correct=int((classification[h]==0).sum()), ht_virut=int((classification[h]==5).sum()))
        for src, dest in ((40,1500),(1500,40)):
            old_wrong = (abs(truth)==src) & (before==dest)
            wrong = (abs(truth)==src) & (length==dest)
            r[f'err{src}'] = int(wrong.sum())
            r[f'removed{src}'] = int((old_wrong & ~wrong).sum())
            r[f'added{src}'] = int((~old_wrong & wrong).sum())
            r[f'exact_fix{src}'] = int((old_wrong & (length==src)).sum())
        rows[method] = r
    np.savez_compressed(OUT/f'{split}.npz', ids=ids, y=y, q=np.concatenate(positions),
                        nq=list(map(len,positions)), truth=truth, **packets,
                        **{k+'_fsnet':v for k,v in preds.items()})
    return rows


def dev(split, rows, saved, head, model):
    ids = np.array([r['id'] for r in rows])
    y = np.array([r['y'] for r in rows])
    seq = [r['s'] for r in rows]
    q = [r['masks']['50'] for r in rows]
    with np.load(DEV/f'{split}_pred.npz') as z:
        assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['y'])
        assert np.array_equal(np.concatenate(q), z['q_50']) and np.array_equal(list(map(len,q)),z['nq_50'])
        assert np.array_equal(np.concatenate([np.array(s)[pos] for s,pos in zip(seq,q)]),z['clean_50_packet'])
        assert np.array_equal(saved[split+'_ids'],ids) and np.array_equal(saved[split+'_q'],z['q_50'])
        base = z['B_50_packet']
        current = saved[split+'_both_packets']
        context = route(head,seq,q,base,current)
        packets = dict(current=current,context=context,TFI=z['TFI_50_packet'])
        preds = {k:classify(model,seq,q,p) for k,p in packets.items()}
        assert np.array_equal(preds['current'],saved[split+'_both'])
        assert np.array_equal(preds['TFI'],z['TFI_50_fsnet'])
    return audit(split,ids,y,seq,q,packets,preds)


def pooled(first, second):
    out = {}
    for m in ARMS:
        total = {k:v+second[m][k] for k,v in first[m].items() if k!='mae'}
        total['mae'] = total['abs_error']/total['slots']
        out[m] = total
    return out


def passed(rows):
    new, old = rows['context'], rows['current']
    return (new['err40']+new['err1500'] < old['err40']+old['err1500']
            and new['abs_error'] < old['abs_error'] and new['ht_correct'] >= old['ht_correct']
            and new['ht_virut'] <= old['ht_virut'])


def run():
    hashes = json.loads((LAST/'results.json').read_bytes())['hashes']
    assert sha(DEV/'panels.json.gz')==hashes['panel'] and sha(OLD/'head.joblib')==hashes['head']
    assert sha(ds.REPAIRED/'repaired/repaired_fsnet.pt')==hashes['fsnet']
    groups = json.loads(gzip.decompress((DEV/'panels.json.gz').read_bytes()))
    keys = [r['id'] for rows in groups.values() for r in rows]
    assert len(keys)==len(set(keys))
    head = joblib.load(OLD/'head.joblib')
    torch.set_num_threads(4)
    model = ds.model_for('fsnet',8).eval()
    with np.load(LAST/'dev_predictions.npz') as z:
        results = {s:dev(s,groups[s],z,head,model) for s in ('screen','confirm')}
    results['pooled'] = pooled(results['screen'], results['confirm'])
    expand = passed(results['confirm']) and passed(results['pooled'])
    log = dict(results=results,expand=expand,hashes=hashes,protocol_sha=sha(OUT/'protocol.txt'),
               code_sha=sha(Path(__file__)),base_model='stage A u1000 dual numeric readout')
    (OUT/'dev.json').write_text(json.dumps(log,indent=2),encoding='utf8')
    print(json.dumps(log,indent=2))


if __name__ == '__main__':
    run()
