"""Meaningful regression checks for expensive-run selection and resume rules."""
from copy import deepcopy
import json
from pathlib import Path
from audited_control import choose_best, empty_best, make_control, validate_resume, config_digest


def rejected(fn):
    try:
        fn()
    except ValueError:
        return True
    raise AssertionError('Expected an explicit rejection')


def main():
    checks = {}
    b = empty_best()
    same, save = choose_best(b, {'n': 256, 'valid_rate': 0.5, 'mae_valid_only': 1.0}, 500)
    assert same == b and not save and same is not b
    checks['ineligible_low_mae_not_selected'] = True
    b, save = choose_best(b, {'n': 256, 'valid_rate': 1.0, 'mae_valid_only': 100.0}, 500)
    assert save and b['update'] == 500
    b, save = choose_best(b, {'n': 256, 'valid_rate': 0.995, 'mae_valid_only': 80.0}, 1000)
    assert save and b['mae'] == 80.0
    checks['above_threshold_selects_mae_not_lexicographic_valid_rate'] = True
    b, save = choose_best(b, {'n': 256, 'valid_rate': 1.0, 'mae_valid_only': 0.0}, 1500)
    assert save and b['mae'] == 0
    same, save = choose_best(b, {'n': 256, 'valid_rate': 1.0, 'mae_valid_only': 20.0}, 2000)
    assert not save and same == b
    same, save = choose_best(b, {'n': 256, 'valid_rate': 1.0, 'mae_valid_only': 0.0}, 2000)
    assert not save and same == b
    checks['zero_mae_preserved_no_worse_or_tied_overwrite'] = True
    checks['smaller_validation_rejected'] = rejected(
        lambda: choose_best(b, {'n': 32, 'valid_rate': 1.0, 'mae_valid_only': 0}, 2000))
    checks['eligible_nan_rejected'] = rejected(
        lambda: choose_best(b, {'n': 256, 'valid_rate': 1.0, 'mae_valid_only': float('nan')}, 2000))
    args = {'stage': 'b', 'seed': 17, 'accum': 32, 'total_updates': 10000,
            'run_id': 'old', 'resume': '', 'init_from': 'stage_a', 'model_dir': '/old/model'}
    ids = {'model_revision': 'fixed-revision', 'data': 'fixed-data', 'split': 'fixed-split',
           'code': 'fixed-code', 'tokenizer_template': 'fixed-template', 'masks': 'fixed-mask'}
    control = make_control(args, ids, 2000, b, 1882)
    moved = {**args, 'run_id': 'cloud', 'resume': '/new/checkpoint.pt', 'model_dir': '/cloud/model'}
    update, restored = validate_resume(control, moved, ids, 1882)
    assert update == 2000 and restored == b and restored is not b
    restored['mae'] = 10
    assert control['best']['mae'] == 0
    checks['cloud_paths_change_but_best_and_counters_resume'] = True
    checks['stage_change_rejected'] = rejected(
        lambda: validate_resume(control, {**args, 'stage': 'a'}, ids, 1882))
    checks['content_identity_change_rejected'] = rejected(
        lambda: validate_resume(control, args, {**ids, 'data': 'different'}, 1882))
    checks['accum_change_rejected'] = rejected(
        lambda: validate_resume(control, {**args, 'accum': 16}, ids, 1882))
    broken = deepcopy(control); broken['next_sample'] += 1
    checks['sampler_counter_mismatch_rejected'] = rejected(
        lambda: validate_resume(broken, args, ids, 1882))
    checks['legacy_weights_only_is_not_full_resume'] = rejected(
        lambda: validate_resume({}, args, ids, 1882))
    out = {'passed': all(checks.values()), 'scope': 'Pure software control tests; zero model fitting',
           'checks': checks, 'config_digest': config_digest(args, ids)}
    Path(__file__).with_name('control_checks.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
