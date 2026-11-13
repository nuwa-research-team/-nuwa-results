"""Bounded P1 formal Stage A/B; reuse the audited architecture, fix run control.

Stage A: all 1,882 training windows, 1,000 optimizer updates, frontend only.
Stage B: 10,000 updates maximum, frontend + attention LoRA. No test inference.
The engineering pilot is not reused as Stage A initialization.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import random
from pathlib import Path
import time

import numpy as np
import torch
import train_sft_v6 as implementation
from audited_control import empty_best, choose_best, make_control, validate_resume


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def evaluate(runner, windows, manifest, update):
    """Greedy generation first; evaluator reads hidden gold only afterwards."""
    assert len(windows) == len(manifest['rows']) == 256
    started = time.monotonic()
    records, packet_errors = [], []
    output = runner.run_dir / f'generations_u{update}.jsonl'
    with output.open('x', encoding='utf-8') as f:
        for i, (w, m) in enumerate(zip(windows, manifest['rows']), 1):
            assert w['sample_id'] == m['sample_id']
            generated = runner.generate(w, m['hidden'])
            gold = [w['signed_lengths'][j] for j in m['hidden']]
            mae = None
            if generated['status'] == 'valid':
                errors = [abs(a - b) for a, b in zip(generated['parsed'], gold)]
                assert len(errors) == len(gold) > 0
                mae = sum(errors) / len(errors)
                packet_errors.extend(errors)
            record = {'sample_id': w['sample_id'], 'hidden': m['hidden'], 'gold': gold,
                      **generated, 'signed_mae': mae}
            records.append(record)
            f.write(json.dumps(record, allow_nan=False) + '\n')
            f.flush()
            if i % 8 == 0:
                runner.emit(event='validation_progress', update=update, done=i, total=256,
                            elapsed_seconds=round(time.monotonic() - started, 1))
    valid = [x for x in records if x['status'] == 'valid']
    from collections import Counter
    summary = {'n': 256, 'update': update, 'valid_rate': len(valid) / 256,
               'mae_valid_only': sum(x['signed_mae'] for x in valid) / len(valid) if valid else None,
               'mae_packet_micro_valid_only': sum(packet_errors) / len(packet_errors) if packet_errors else None,
               'status_counts': dict(Counter(x['status'] for x in records)),
               'latency_ms_p50': float(np.percentile([x['latency_ms'] for x in records], 50)),
               'latency_ms_p95': float(np.percentile([x['latency_ms'] for x in records], 95)),
               'wall_seconds': time.monotonic() - started,
               'outputs_sha256': sha(output),
               'note': 'Invalid outputs remain failures. MAE is conditional on valid outputs.'}
    atomic_json(runner.run_dir / f'validation_u{update}.json', summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data1/lhy/behavior_lora_d4_20261111')
    ap.add_argument('--stage', required=True, choices=['a', 'b'])
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--model-dir', default='')
    ap.add_argument('--resume', default='')
    ap.add_argument('--init-from', default='')
    args = ap.parse_args()
    if args.resume and args.init_from:
        raise ValueError('Choose same-stage resume or cross-stage initialization, not both')
    if args.stage == 'a' and args.init_from:
        raise ValueError('Formal Stage A starts from its registered seed, not the pilot checkpoint')
    if args.stage == 'b' and not (args.resume or args.init_from):
        raise ValueError('Stage B needs a completed Stage A checkpoint')
    root = Path(args.root).resolve()
    assert root.name == 'behavior_lora_d4_20261111'
    implementation.ROOT = root
    args.model_dir = args.model_dir or str(root / 'model/qwen3-4b-instruct-2507')
    args.accum = 32
    args.total_updates = 1000 if args.stage == 'a' else 10000
    args.select_every = 500
    args.roll_eval_windows = 256
    args.final_eval_windows = 256
    run_dir = root / 'runs' / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    source = None
    source_path = Path(args.resume or args.init_from).resolve() if (args.resume or args.init_from) else None
    if source_path:
        source_path.relative_to(root)
        source = torch.load(source_path, map_location='cpu', weights_only=False)
        if args.init_from and (source.get('stage') != 'a' or source.get('update') != 1000):
            raise ValueError('Stage B initialization must be a completed 1,000-update Stage A')
    train, val = implementation.load_split('train'), implementation.load_split('val')[:256]
    assert len(train) == 1882 and len(val) == 256
    manifest = implementation.mask_manifest(val, args.seed)
    atomic_json(run_dir / 'validation_mask_manifest.json', manifest)
    model = Path(args.model_dir)
    revision = (root / 'model/revision.txt').read_text().strip()
    assert revision == 'cdbee75f17c01a7cc42f958dc650907174af0554'
    files = [Path(__file__), root / 'src/train_sft_v6.py', root / 'src/audited_control.py',
             root / 'src/audited_frontend.py', root / 'src/audited_packing.py']
    code_hashes = {str(p.relative_to(root)): sha(p) for p in files}
    initialization = f'seed:{args.seed}:fresh_frontend_and_zero_B_lora'
    if args.init_from:
        initialization = sha(source_path)
    elif args.resume:
        initialization = source['control']['immutable']['identities']['initialization']
    identities = {'model_revision': revision, 'model_config': sha(model / 'config.json'),
                  'tokenizer': sha(model / 'tokenizer.json'),
                  'tokenizer_template': sha(model / 'tokenizer_config.json'),
                  'data': sha(root / 'data/windows.jsonl'),
                  'split': sha(root / 'data/split_manifest.json'),
                  'masks': manifest['sha256'], 'initialization': initialization,
                  'code': hashlib.sha256(json.dumps(code_hashes, sort_keys=True).encode()).hexdigest()}
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'owner': 'Codex',
                'args': vars(args), 'identities': identities, 'code_sha256': code_hashes,
                'train_windows': len(train), 'validation_windows': len(val),
                'train_mask_families': implementation.FAMILIES,
                'train_mask_rates': implementation.RATES_TRAIN,
                'validation_family': 'random', 'validation_rate': 0.3,
                'checkpoint_rule': 'valid_rate>=0.99, then minimum window-macro signed MAE',
                'test_inference': False, 'pilot_initialization_reused': False,
                'full_state_resume_verified_on_full_model': False,
                'environment': {k: importlib.metadata.version(k) for k in
                                ['torch', 'transformers', 'peft', 'accelerate', 'numpy']}}
    atomic_json(run_dir / 'protocol.json', protocol)
    best = empty_best()
    start_update, train_seconds, generation_seconds = 0, 0.0, 0.0
    best_ref = None
    if args.resume:
        start_update, best = validate_resume(source['control'], vars(args), identities, len(train))
        best_ref = source.get('best_checkpoint_ref')
        train_seconds = source.get('train_seconds', 0.0)
        generation_seconds = source.get('generation_seconds', 0.0)
        if best_ref:
            best_path = (root / best_ref['path']).resolve()
            best_path.relative_to(root)
            assert best_path.is_file()
            expected = best_ref.get('sha256')
            if expected:
                assert sha(best_path) == expected
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    r = implementation.Runner(args)
    joint = args.stage == 'b'
    r.set_lora_trainable(joint)
    if source:
        assert source['meta']['data_windows_sha256'] == identities['data']
        assert source['meta']['model_config_sha256'] == identities['model_config']
        assert source['meta']['tokenizer_sha256'] == identities['tokenizer']
        r.frontend.load_state_dict(source['frontend'])
        expected = {n for n in r.model.state_dict() if 'lora_' in n}
        assert set(source['lora']) == expected
        r.model.load_state_dict(source['lora'], strict=False)
    frontend = list(r.frontend.parameters())
    adapter = r.lora_params()
    frozen = [p for n, p in r.model.named_parameters() if 'lora_' not in n]
    assert all(not p.requires_grad for p in frozen)
    groups = [{'params': frontend, 'lr': 5e-5 if joint else 1e-4}]
    if joint:
        groups.append({'params': adapter, 'lr': 1e-4})
    optimized = [p for g in groups for p in g['params']]
    assert not ({id(p) for p in optimized} & {id(p) for p in frozen})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)
    total, warm = args.total_updates, max(1, int(args.total_updates * 0.03))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s:
        min(1.0, (s + 1) / warm) * 0.5 *
        (1 + math.cos(math.pi * max(0, s - warm) / max(1, total - warm))))
    if args.resume:
        opt.load_state_dict(source['optim'])
        sched.load_state_dict(source['sched'])
        implementation.set_rng(source['meta']['rng'])
    del source
    r.model.train()
    r.frontend.train()
    r.emit(event='formal_started', owner='Codex', stage=args.stage, start_update=start_update,
           total_updates=total, effective_batch=32, train_windows=len(train),
           validation_windows=256, code_sha256=code_hashes,
           model_base_frozen=True, lora_trainable=joint)

    def save(path, update, ref):
        r.save(path, {'update': update, 'optim': opt.state_dict(), 'sched': sched.state_dict(),
                      'stage': args.stage, 'seed': args.seed, 'args': vars(args),
                      'control': make_control(vars(args), identities, update, best, len(train)),
                      'best_checkpoint_ref': ref, 'train_seconds': train_seconds,
                      'generation_seconds': generation_seconds})
        return {'path': str(path.relative_to(root)), 'sha256': sha(path)}

    last_validation = None
    for update in range(start_update, total):
        step_start = time.monotonic()
        opt.zero_grad(set_to_none=True)
        losses = []
        # No extra stochastic wiring step is run on resume. The first actual
        # formal steps provide gradients; RNG/sampler resume remains untouched.
        for micro in range(32):
            sample_number = update * 32 + micro
            w = train[sample_number % len(train)]
            hidden = implementation.mask_indices(w['n'], random.choice(implementation.RATES_TRAIN),
                     random.choice(implementation.FAMILIES), w['sample_id'], seed=args.seed,
                     epoch=sample_number // len(train))
            loss = r.train_step(w, hidden)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite formal training loss')
            (loss / 32).backward()
            losses.append(float(loss.detach()))
        norms = {}
        for name, parameters in {**r.groups, **({'lora': adapter} if joint else {})}.items():
            grads = [p.grad for p in parameters if p.grad is not None]
            assert grads
            norm = torch.sqrt(sum(g.detach().float().square().sum() for g in grads))
            norms[name] = float(norm)
            if not math.isfinite(norms[name]) or norms[name] <= 0:
                raise FloatingPointError(f'Invalid gradient group {name}')
        assert all(p.grad is None for p in frozen)
        lr_used = [g['lr'] for g in opt.param_groups]
        torch.nn.utils.clip_grad_norm_(optimized, 1.0, error_if_nonfinite=True)
        opt.step()
        sched.step()
        step_seconds = time.monotonic() - step_start
        train_seconds += step_seconds
        done = update + 1
        if done <= 3 or done % 10 == 0:
            r.emit(event='formal_update', update=done, total=total, loss=sum(losses) / 32,
                   seconds_per_update=step_seconds, train_seconds=train_seconds,
                   generation_seconds=generation_seconds, lr_used=lr_used, grad_norms=norms,
                   peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                   peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
        if done % 500 == 0:
            last_validation = evaluate(r, val, manifest, done)
            generation_seconds += last_validation['wall_seconds']
            best, improved = choose_best(best, last_validation, done)
            path = run_dir / f'ckpt_u{done}.pt'
            reference = {'path': str(path.relative_to(root)), 'sha256': None} if improved else best_ref
            saved = save(path, done, reference)
            if improved:
                best_ref = saved
            r.emit(event='formal_validation_finished', update=done, best=best,
                   accepted_new_best=improved, **{k: v for k, v in last_validation.items() if k != 'update'})
        if done % 100 == 0:
            save(run_dir / 'ckpt_latest.pt', done, best_ref)
    final_ref = save(run_dir / 'ckpt_final.pt', total, best_ref)
    outcome = {'passed': True, 'engineering_completed': True, 'stage': args.stage,
               'updates': total, 'formal_checkpoint_eligible': best['update'] is not None,
               'best': best, 'best_checkpoint_ref': best_ref, 'final_checkpoint': final_ref,
               'last_validation': last_validation, 'train_seconds': train_seconds,
               'generation_seconds': generation_seconds, 'test_inference': False,
               'note': 'Stage completion is separate from the 99% output-validity requirement.'}
    atomic_json(run_dir / 'completion.json', outcome)
    r.emit(event='formal_stage_finished', **outcome)


if __name__ == '__main__':
    main()
