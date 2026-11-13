import gzip
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / 'own_short_v1'
sys.path.insert(0, str(OLD))
from dev import ds, DEV, sha, score, classify
from short_head import features


def variants(head, seqs, positions, base):
    out = {k: base.copy() for k in ('v1', 'open', 'exit', 'both')}
    chunks, inds = [], []
    off = 0
    for seq, pos in zip(seqs, positions, strict=True):
        q = np.asarray(pos, dtype=int)
        idx = np.flatnonzero((np.abs(base[off:off+len(q)]) <= 128) & (base[off:off+len(q)] != 0))
        if len(idx):
            visible = np.array(seq)
            visible[q] = 0
            chunks.append(features(visible, q[idx]))
            inds.append(idx + off)
        off += len(q)
    assert off == len(base)
    if not inds:
        return out
    ix = np.concatenate(inds)
    x = np.concatenate(chunks)
    for j in range(0, len(ix), 4096):
        xx, ii = x[j:j+4096], ix[j:j+4096]
        p = head.predict_proba(xx)
        k = p.argmax(1)
        top = head.classes_[k]
        old = np.where(p[np.arange(len(k)), k] >= .45, top, base[ii])
        left, right = xx[:,9], xx[:,11]
        agree = (left == right) & (np.abs(left) > 128)
        out['v1'][ii] = old
        out['open'][ii] = top
        out['exit'][ii] = np.where(agree, left, old)
        out['both'][ii] = np.where(agree, left, top)
    return out


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    identity = json.loads((OLD/'dev_results.json').read_bytes())['hashes']
    assert sha(DEV/'panels.json.gz') == identity['panel']
    assert sha(OLD/'head.joblib') == identity['head']
    assert sha(ds.REPAIRED/'repaired/repaired_fsnet.pt') == identity['fsnet']
    groups = json.loads(gzip.decompress((DEV/'panels.json.gz').read_bytes()))
    ids = [r['id'] for rows in groups.values() for r in rows]
    assert len(ids) == len(set(ids))
    head = joblib.load(OLD/'head.joblib')
    fs = ds.model_for('fsnet', 8).eval()
    record, saved = {}, {}
    for split in ('screen','confirm'):
        rows = groups[split]
        s = [r['s'] for r in rows]
        q = [r['masks']['50'] for r in rows]
        with np.load(DEV/f'{split}_pred.npz') as z:
            y = z['y']
            assert list(z['ids']) == [r['id'] for r in rows] and list(y) == [r['y'] for r in rows]
            assert np.array_equal(z['q_50'], np.concatenate(q)) and np.array_equal(z['nq_50'], list(map(len,q)))
            b, t = z['B_50_packet'], z['TFI_50_packet']
            arms = variants(head,s,q,b)
            arms.update(base=b,TFI=t)
            truth = np.concatenate([np.array(seq)[pos] for seq,pos in zip(s,q)])
            saved[split+'_ids'] = z['ids']
            saved[split+'_q'] = z['q_50']
            saved[split+'_y'] = y
            record[split] = {}
            for name, packets in arms.items():
                pred = classify(fs,s,q,packets)
                if name in ('base','TFI'):
                    assert np.array_equal(pred,z[('B' if name=='base' else 'TFI')+'_50_fsnet'])
                m = score(y,pred)
                m['changed_vs_v1'] = int((packets!=arms['v1']).sum())
                m['length_mae'] = float(np.abs(np.abs(packets.astype(float))-np.abs(truth)).mean())
                record[split][name] = m
                saved[split+'_'+name] = pred
                saved[split+'_'+name+'_packets'] = packets
        print(split,record[split],flush=True)
    def passes(split, arm):
        a,b = record[split]['v1'],record[split][arm]
        return b['tp']>=a['tp'] and b['htbot_to_virut']<=a['htbot_to_virut'] and b['macro8']>=a['macro8'] and (b['tp']>a['tp'] or b['htbot_to_virut']<a['htbot_to_virut'])
    passed = [a for a in ('open','exit','both') if all(passes(s,a) for s in record)]
    selected = max(passed,key=lambda a:record['screen'][a]['macro8'],default=None)
    np.savez_compressed(OUT/'dev_predictions.npz', **saved)
    (OUT/'results.json').write_text(json.dumps(dict(groups=record,selected=selected,hashes=identity,protocol_sha=sha(OUT/'protocol.txt')),indent=2),encoding='utf8')
    print('selected:',selected,flush=True)

if __name__ == '__main__':
    main()
