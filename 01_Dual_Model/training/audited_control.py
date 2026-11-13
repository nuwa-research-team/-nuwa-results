"""Pure control rules for formal runs; no fitting, I/O or device operations.

Old engineering checkpoints may initialize weights. Faithful same-stage resume
requires the explicit control record below, so missing fields cannot silently pass.
"""
from copy import deepcopy
import hashlib
import json
import math

SCHEMA = 1
OPERATIONAL_FIELDS = frozenset(('run_id', 'resume', 'init_from', 'model_dir', 'root'))


def immutable_config(args, identities):
    """Paths may move for cloud continuation; immutable content identities may not."""
    config = {k: v for k, v in dict(args).items() if k not in OPERATIONAL_FIELDS}
    if not identities or any(not isinstance(v, str) or not v for v in identities.values()):
        raise ValueError('Immutable content identities are required')
    return {'schema': SCHEMA, 'config': config, 'identities': dict(identities)}


def config_digest(args, identities):
    blob = json.dumps(immutable_config(args, identities), sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def empty_best():
    return {'update': None, 'mae': None, 'valid_rate': None}


def choose_best(previous, evaluation, update, required_n=256, min_valid_rate=0.99):
    """Eligibility first, then minimum window-macro signed MAE; ties keep the first.

    Returns (new_record, must_save). Save ckpt_best only when must_save is true.
    A larger valid rate above the threshold does not override a smaller MAE.
    """
    best = deepcopy(previous)
    if evaluation.get('n') != required_n:
        raise ValueError('Formal selection requires the registered validation size')
    rate, mae = evaluation.get('valid_rate'), evaluation.get('mae_valid_only')
    if not isinstance(rate, (float, int)) or not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError('Invalid output-rate statistic')
    if rate < min_valid_rate:
        return best, False
    if not isinstance(mae, (float, int)) or not math.isfinite(mae) or mae < 0:
        raise ValueError('An eligible checkpoint needs a finite nonnegative MAE')
    if best['update'] is None or mae < best['mae']:
        return {'update': int(update), 'mae': float(mae), 'valid_rate': float(rate)}, True
    return best, False


def make_control(args, identities, update, best, train_size):
    """Checkpoint only at a completed optimizer-update boundary (micro=0)."""
    args = dict(args)
    total = int(args['total_updates'])
    if not 0 <= update <= total or train_size < 1 or int(args['accum']) < 1:
        raise ValueError('Invalid update/sampler boundary')
    return {'schema': SCHEMA, 'immutable': immutable_config(args, identities),
            'config_sha256': config_digest(args, identities), 'stage': args['stage'],
            'update': int(update), 'micro': 0, 'train_size': int(train_size),
            'next_sample': int(update) * int(args['accum']), 'best': deepcopy(best)}


def validate_resume(control, args, identities, train_size):
    """Validate before restoring optimizer, scheduler and RNG; never reset best."""
    args = dict(args)
    if control.get('schema') != SCHEMA:
        raise ValueError('No compatible full-state control record; weights-only init is separate')
    if control.get('stage') != args['stage']:
        raise ValueError('Cross-stage optimizer resume is forbidden; use fresh stage initialization')
    if control.get('config_sha256') != config_digest(args, identities):
        raise ValueError('Immutable configuration or content identity changed')
    update = control.get('update')
    if type(update) is not int or not 0 <= update <= int(args['total_updates']):
        raise ValueError('Invalid optimizer update counter')
    if control.get('micro') != 0 or control.get('train_size') != train_size:
        raise ValueError('Sampler boundary or training size differs')
    if control.get('next_sample') != update * int(args['accum']):
        raise ValueError('Sampler counter differs from the optimizer counter')
    best = control.get('best')
    if not isinstance(best, dict) or set(best) != set(empty_best()):
        raise ValueError('Missing or incompatible best-selection state')
    if best['update'] is not None:
        if not 0 < best['update'] <= update:
            raise ValueError('Best checkpoint update is inconsistent')
        if not isinstance(best['mae'], (int, float)) or not math.isfinite(best['mae']):
            raise ValueError('Best MAE is not finite')
        if not isinstance(best['valid_rate'], (int, float)) or not 0.99 <= best['valid_rate'] <= 1:
            raise ValueError('Best checkpoint was never eligible')
    return int(update), deepcopy(best)
