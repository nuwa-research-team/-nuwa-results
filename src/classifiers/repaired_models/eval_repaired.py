# -*- coding: utf-8 -*-
"""Evaluate the three repaired classifiers on the SAME paired masks/imputations.

Reuses work/paired_eval_v2_20261110/mask_*.npz and *_imputed_*.npz (no new MLM
training, no new imputation values). Produces 3 models x 22 conditions = 66 cells.
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models_repaired as MR
from train_repaired import SDAERepaired

ROOT = Path('/data1/lhy/nuwa')
PE = ROOT / 'work/paired_eval_v2_20261110'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def markov_table(seqs, idx):
    t = {}
    for i in idx:
        s = seqs[i]
        for a, b in zip(s[:-1], s[1:]):
            c = t.setdefault(int(a), {})
            c[int(b)] = c.get(int(b), 0) + 1
    return {a: max(c.items(), key=lambda kv: kv[1])[0] for a, c in t.items()}


def fill(sequences, mask, method, table=None):
    out = []
    for row, s in enumerate(sequences):
        pos = np.flatnonzero(mask[row, :len(s)])
        if method == 'deletion':
            out.append([v for j, v in enumerate(s) if not mask[row, j]])
            continue
        v = list(s)
        for j in pos:
            if method == 'ZeroPad':
                v[j] = 0
            else:
                prev = v[j - 1] if j else 0
                v[j] = table.get(prev, prev)
        out.append(v)
    return out


@torch.inference_mode()
def classify(model, rows, order, dev, batch, scale):
    preds = []
    for start in range(0, len(order), batch):
        idx = order[start:start + batch]
        b = [rows[i] for i in idx]
        L = max(map(len, b))
        x = torch.zeros(len(b), L)
        for j, s in enumerate(b):
            x[j, :len(s)] = torch.as_tensor(s, dtype=torch.float32) * scale
        lengths = torch.tensor([len(s) for s in b])
        preds.extend(model(x.to(dev), lengths.to(dev)).argmax(-1).cpu().tolist())
    return np.asarray(preds)


def summarize(y, p, n_cls):
    cm = confusion_matrix(y, p, labels=list(range(n_cls)))
    return {'accuracy': float(accuracy_score(y, p)),
            **{'f1_' + a: float(f1_score(y, p, average=a, zero_division=0))
               for a in ('macro', 'weighted', 'micro')},
            'support': cm.sum(1).tolist(), 'confusion': cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'work/repaired_eval_20261110'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=512)
    args = ap.parse_args()

    torch.set_num_threads(4)
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    d = pickle.load(open(ROOT / 'work/d4_aligned/raw.pkl', 'rb'))
    seqs, all_y, ids, splits = d['seqs'], d['y'], d['ids'], d['split']
    te = list(map(int, splits['test']))
    sequences = [list(seqs[i]) for i in te]
    lengths = np.asarray([len(s) for s in sequences])
    order = np.argsort(lengths)
    y = np.asarray(all_y)[np.asarray(te)[order]]
    ordered_ids = np.asarray(ids)[np.asarray(te)[order]].astype(str)
    n_cls = len(d['labels'])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    table = markov_table(seqs, splits['train'])
    test_ids = np.asarray(ids)[np.asarray(te)].astype(str)
    masks = {}
    for r in (10, 20, 30):
        z = np.load(PE / f'mask_{r}.npz')
        assert np.array_equal(z['ids'].astype(str), test_ids), 'mask ids order mismatch'
        assert np.array_equal(z['lengths'], lengths), 'mask lengths mismatch'
        masks[r] = z['mask']
    imputed = {}
    for m in ('tfi', 'bert', 'roberta', 'deberta'):
        imputed[m] = {}
        for r in (10, 20, 30):
            z = np.load(PE / f'{m}_imputed_{r}.npz')
            assert np.array_equal(z['ids'].astype(str), test_ids), f'{m} imputed ids mismatch'
            imputed[m][r] = z['values']

    config = {
        'checkpoint_sha256': {n: sha(ROOT / 'repaired' / f'repaired_{n}.pt')
                              for n in ('tst', 'fsnet', 'sdae')},
        'source_sha256': {'train': sha(Path(__file__).with_name('train_repaired.py')),
                          'models': sha(Path(MR.__file__)),
                          'eval': sha(Path(__file__))},
        'protocol_sha256': sha(ROOT / 'repaired/protocol_repaired_20261110.json'),
        'paired_source_sha256': {**{p.name: sha(p) for p in sorted(PE.glob('mask_*.npz'))},
                                 **{p.name: sha(p) for p in sorted(PE.glob('*_imputed_*.npz'))}},
        'row_order': 'server test order, np.argsort(lengths) fixed at evaluation time',
        'uses_test_for_selection': False,
        'fits': 0, 'mlm_retrained': False,
    }
    (out / 'run.json').write_text(json.dumps(config, indent=2))

    results = []
    for name in ('tst', 'fsnet', 'sdae'):
        ck = torch.load(ROOT / 'repaired' / f'repaired_{name}.pt', map_location='cpu',
                        weights_only=True)
        if name == 'tst':
            model = MR.build('tst', n_cls, seq_len=512)
            scale = 1.0 / 1600.0
        elif name == 'fsnet':
            model = MR.build('fsnet', n_cls)
            scale = 1.0 / 1600.0
        else:
            model = SDAERepaired(n_cls)
            scale = 1.0 / 1600.0
        model.load_state_dict(ck['state'])
        model.to(dev).eval()

        def run(method, rate, rows):
            pred = classify(model, rows, order, dev, args.batch, scale)
            p = out / f'{name}_{method}_{rate}.npz'
            np.savez_compressed(p, ids=ordered_ids, y=y, pred=pred)
            results.append({'classifier': name, 'method': method,
                            'loss_percent': rate, 'prediction_sha256': sha(p),
                            **summarize(y, pred, n_cls)})
            (out / 'metrics.partial.json').write_text(json.dumps(results, indent=2))
            print(json.dumps({'cells': len(results), 'clf': name, 'method': method,
                              'rate': rate}), flush=True)

        run('clean', 0, sequences)
        for r in (10, 20, 30):
            for meth in ('deletion', 'ZeroPad', 'Markov'):
                run(meth, r, fill(sequences, masks[r], meth, table))
            for m in ('tfi', 'bert', 'roberta', 'deberta'):
                vals = imputed[m][r]
                rows = [list(map(int, vals[i][:lengths[i]])) for i in range(len(sequences))]
                run(m, r, rows)
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert len(results) == 66, len(results)
    (out / 'metrics.json').write_text(json.dumps(results, indent=2))
    (out / 'evaluation.done').write_text('66 cells; repaired classifiers; paired masks/imputations\n')
    print('done', len(results))


if __name__ == '__main__':
    main()
