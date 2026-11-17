"""Evaluation-only repair: correct MLM classes, paired masks and saved flow IDs.

Reuses frozen DSH checkpoints. No optimizer, training or parameter search.
The DSH classifier implementations remain declared adaptations, not exact paper models.
"""
import argparse
import base64
import zlib
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def paired_mask(indices, sequences, ratio, seed=42):
    """Match the existing DSH zero/delete baseline's seed+global-row rule."""
    width = max(len(sequences[i]) for i in indices)
    mask = np.zeros((len(indices), width), dtype=bool)
    for j, i in enumerate(indices):
        length = len(sequences[i])
        n = max(1, int(round(length * ratio)))
        mask[j, np.random.default_rng(seed + int(i)).choice(length, n, replace=False)] = True
    return mask


def markov_table(sequences, indices):
    table = {}
    for i in indices:
        s = sequences[i]
        for a, b in zip(s[:-1], s[1:]):
            counter = table.setdefault(int(a), {})
            counter[int(b)] = counter.get(int(b), 0) + 1
    return {a: max(counter.items(), key=lambda item: item[1])[0] for a, counter in table.items()}


def deterministic_fill(sequences, mask, method, table=None):
    result = []
    for row, s in enumerate(sequences):
        pos = np.flatnonzero(mask[row, :len(s)])
        if method == 'deletion':
            result.append([v for j, v in enumerate(s) if not mask[row, j]])
            continue
        values = list(s)
        for j in pos:
            if method == 'ZeroPad':
                values[j] = 0
            elif method == 'Markov':
                prev = values[j - 1] if j else 0
                values[j] = table.get(prev, prev)
            else:
                raise ValueError(method)
        result.append(values)
    return result


@torch.inference_mode()
def mlm_fill(model, tokenizer, sequences, mask, device, batch):
    filled = [list(s) for s in sequences]
    special_predictions = 0
    for start in range(0, len(sequences), batch):
        rows = sequences[start:start + batch]
        texts = [' '.join(tokenizer.mask_token if mask[start + j, p] else str(int(v) + 1600)
                          for p, v in enumerate(s)) for j, s in enumerate(rows)]
        enc = tokenizer(texts, add_special_tokens=False, padding=True, truncation=False,
                        return_tensors='pt')
        # One numeric token or one mask per actual packet: never infer lengths from zeros.
        assert enc['attention_mask'].sum(1).tolist() == [len(s) for s in rows]
        for j, s in enumerate(rows):
            observed = enc['input_ids'][j, :len(s)].eq(tokenizer.mask_token_id).numpy()
            assert np.array_equal(observed, mask[start + j, :len(s)])
        outputs = model(input_ids=enc['input_ids'].to(device),
                        attention_mask=enc['attention_mask'].to(device)).logits.argmax(-1).cpu()
        for j, s in enumerate(rows):
            for p in np.flatnonzero(mask[start + j, :len(s)]):
                value = tokenizer.convert_ids_to_tokens(int(outputs[j, p]))
                try:
                    value = int(value) - 1600
                    if not -1600 <= value <= 1600:
                        raise ValueError('non-length token')
                except (ValueError, TypeError):
                    value = 0  # Preserve the existing upstream-inspired decoding fallback.
                    special_predictions += 1
                filled[start + j][p] = value
            keep = ~mask[start + j, :len(s)]
            assert np.array_equal(np.asarray(filled[start + j])[keep], np.asarray(s)[keep])
        if start % (batch * 40) == 0:
            print(json.dumps({'imputed_rows': start, 'total': len(sequences)}), flush=True)
    return filled, special_predictions


@torch.inference_mode()
def classify(model, sequences, order, device, batch):
    predictions = []
    for start in range(0, len(order), batch):
        indices = order[start:start + batch]
        rows = [sequences[i] for i in indices]
        length = max(map(len, rows))
        x = torch.zeros(len(rows), 1, length, dtype=torch.float32)
        for j, s in enumerate(rows):
            x[j, 0, :len(s)] = torch.as_tensor(s, dtype=torch.float32)
        predictions.extend(model(x.to(device)).argmax(-1).cpu().tolist())
    return np.asarray(predictions, dtype=np.int64)


def summarize(y, pred, classes):
    cm = confusion_matrix(y, pred, labels=list(range(classes)))
    return {'accuracy': float(accuracy_score(y, pred)),
            **{'f1_' + avg: float(f1_score(y, pred, average=avg, zero_division=0))
               for avg in ('macro', 'weighted', 'micro')},
            'support': cm.sum(1).tolist(), 'confusion': cm.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--impute-batch', type=int, default=64)
    parser.add_argument('--classifier-batch', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--order-file', type=Path, help='Server global test row order (.b64 zlib int64)')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.root.resolve()
    raw_path = root / 'work/d4_aligned/raw.pkl'
    # Trusted artifact from the reviewed DSH preprocessing, not a downloaded pickle.
    with raw_path.open('rb') as f:
        data = pickle.load(f)
    seqs, all_y, ids, splits = data['seqs'], data['y'], data['ids'], data['split']
    roles = [set(map(int, splits[role])) for role in ('train', 'val', 'test')]
    assert not any(roles[i] & roles[j] for i in range(3) for j in range(i + 1, 3))
    assert sum(map(len, roles)) == len(seqs)
    assert len(set(ids)) == len(ids)
    te = list(map(int, splits['test']))
    sequences = [list(seqs[i]) for i in te]
    lengths = np.asarray([len(s) for s in sequences])
    assert lengths.min() > 0 and lengths.max() <= 512
    # Preserve this environment's original DSH tie ordering, and archive the actual IDs.
    if args.order_file:
        global_order = np.frombuffer(zlib.decompress(base64.b64decode(args.order_file.read_text())), dtype='<i8')
        assert len(global_order) == len(te) and set(global_order.tolist()) == set(te)
        positions = {int(row): pos for pos, row in enumerate(te)}
        order = np.asarray([positions[int(row)] for row in global_order], dtype=np.int64)
    else:
        order = np.argsort(lengths)
    y = np.asarray(all_y)[np.asarray(te)[order]]
    ordered_ids = np.asarray(ids)[np.asarray(te)[order]].astype(str)
    names = ['cnn', 'gru', 'lstm', 'tst', 'sdae', 'fsnet']
    ckpts = {name: root / ('work/res_aligned_' + name) / (name + '.pt') for name in names}
    for path in ckpts.values():
        assert path.is_file(), str(path)
    model_roots = {name: root / 'work/d4_aligned' / name / 'model'
                   for name in ('tfi', 'bert', 'roberta', 'deberta')}
    for path in model_roots.values():
        assert (path / 'config.json').is_file(), str(path)
    masks = {r: paired_mask(te, seqs, r / 100, args.seed) for r in (10, 20, 30)}
    print(json.dumps({'audit': 'PASS', 'rows': len(te), 'classes': len(data['labels']),
                      'expected_metric_cells': 132, 'fits_requested': 0,
                      'scope': 'USTC 8-class adaptation, random loss only'}), flush=True)
    if args.audit_only:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    spec = importlib.util.spec_from_file_location('dsh_legacy_models', root / 'src/models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classifiers = {}
    for name, path in ckpts.items():
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['labels'] == data['labels']
        model = module.build(name, len(data['labels']))
        model.load_state_dict(checkpoint['state'], strict=True)
        classifiers[name] = model.eval()

    config = {'data_sha256': sha(raw_path), 'code_sha256': sha(__file__),
              'classifier_code_sha256': sha(root / 'src/models.py'),
              'checkpoint_sha256': {k: sha(v) for k, v in ckpts.items()},
              'seed': args.seed, 'classifier_batch': args.classifier_batch,
              'order_file_sha256': sha(args.order_file) if args.order_file else None,
              'mask_rule': 'default_rng(seed + original_global_row), choice without replacement',
              'metric_averages': ['macro', 'weighted', 'micro'], 'uses_test_for_selection': False,
              'strict_paper_reproduction': False, 'fits': 0,
              'disclosures': ['USTC8 class/data mismatch; raw signed IP length inputs',
                              'legacy classifier architectures retained; TST/FSNet/SDAE are approximations',
                              'uniform random mask only; not Poisson or real network replay',
                              'original DSH sequence order and batch size retained and IDs saved',
                              'GPU-trained standard MLM checkpoints fixed; extra CPU BERT not selected']}
    (args.output / 'run.json').write_text(json.dumps(config, indent=2))
    for rate, mask in masks.items():
        np.savez_compressed(args.output / ('mask_%d.npz' % rate),
                            ids=np.asarray(ids)[te].astype(str), mask=mask, lengths=lengths)

    def evaluate(method, rate, rows):
        for name, model in classifiers.items():
            model.to(args.device)
            pred = classify(model, rows, order, args.device, args.classifier_batch)
            model.cpu()
            path = args.output / ('%s_%s_%02d.npz' % (name, method, rate))
            np.savez_compressed(path, ids=ordered_ids, y=y, pred=pred)
            results.append({'classifier': name, 'method': method, 'loss_percent': rate,
                            'prediction_sha256': sha(path), **summarize(y, pred, len(data['labels']))})
            (args.output / 'metrics.partial.json').write_text(json.dumps(results, indent=2))
            print(json.dumps({'completed_cells': len(results), 'classifier': name,
                              'method': method, 'rate': rate}), flush=True)

    evaluate('clean', 0, sequences)
    table = markov_table(seqs, splits['train'])
    for rate, mask in masks.items():
        for method in ('deletion', 'ZeroPad', 'Markov'):
            evaluate(method, rate, deterministic_fill(sequences, mask, method, table))

    from transformers import AutoModelForMaskedLM, PreTrainedTokenizerFast
    reference_vocab = None
    for name, model_path in model_roots.items():
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_path), local_files_only=True)
        if reference_vocab is None:
            reference_vocab = tokenizer.get_vocab()
        assert tokenizer.get_vocab() == reference_vocab, 'MLM vocabulary mismatch'
        model, loading = AutoModelForMaskedLM.from_pretrained(str(model_path), local_files_only=True,
                                                            output_loading_info=True)
        assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading['mismatched_keys'], loading
        expected_type = 'deberta' if name == 'tfi' else name
        assert model.config.model_type == expected_type
        (args.output / (name + '_loading.json')).write_text(json.dumps({
            'class': type(model).__name__, 'model_type': model.config.model_type,
            'loading_info': loading, 'config_sha256': sha(model_path / 'config.json')}, indent=2))
        model.to(args.device).eval()
        for rate, mask in masks.items():
            rows, count = mlm_fill(model, tokenizer, sequences, mask, args.device, args.impute_batch)
            np.savez_compressed(args.output / ('%s_imputed_%02d.npz' % (name, rate)),
                                values=np.asarray([s + [0] * (512 - len(s)) for s in rows], dtype=np.int16),
                                ids=np.asarray(ids)[te].astype(str), lengths=lengths,
                                special_prediction_fallback_count=count)
            evaluate(name, rate, rows)
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert len(results) == 132
    (args.output / 'metrics.json').write_text(json.dumps(results, indent=2))
    (args.output / 'evaluation.done').write_text('132 cells; evaluation only\n')


if __name__ == '__main__':
    main()
