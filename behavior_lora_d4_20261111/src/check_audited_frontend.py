"""Meaningful synthetic contract/gradient tests. No data fit or experiment score."""
import hashlib
import json
from pathlib import Path
import time

import torch
from audited_frontend import mask_indices, redact, featurize, collate_features, TrafficFrontend

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.manual_seed(17)
t0 = time.monotonic()
checks = {}
fixture = json.loads((ROOT/'independent_feature_fixture.json').read_text())
f = featurize(fixture['redacted_records'], fixture['N'])
assert torch.allclose(torch.tensor(f['numeric']), torch.tensor(fixture['expected_numeric_input']), atol=1e-7)
assert torch.allclose(torch.tensor(f['stats'][-1]), torch.tensor(fixture['expected_global_stats']), atol=1e-7)
assert f['stat_valid'][-1] == fixture['expected_global_stat_valid']
checks['pdf8_numeric_and16_stats_golden'] = True
cases = 0
for n in (64, 65, 127, 128, 255, 256):
    for rate in (.1, .2, .3, .5, .7, .9):
        for family in ('random', 'block', 'multispan'):
            for seed in (17, 29, 43):
                hidden = mask_indices(n, rate, family, 'fixed', seed)
                assert len(hidden) == int(rate*n+.5) and 0 not in hidden and n-1 not in hidden
                runs = 1+sum(b != a+1 for a, b in zip(hidden, hidden[1:]))
                if family == 'block': assert runs == 1
                if family == 'multispan': assert 2 <= runs <= 4
                assert hidden == mask_indices(n, rate, family, 'fixed', seed)
                cases += 1
checks['mask_budget_boundaries_run_count_deterministic_cases'] = cases
n = 256
signed = [60 if i%2 == 0 else -1500 for i in range(n)]
times = [i*1000 for i in range(n)]
hidden = list(range(16, 64))
red = redact(signed, times, hidden)
base = featurize(red, n)
changed_s, changed_t = signed[:], times[:]
for i in hidden:
    changed_s[i] = -72 if i%2 == 0 else 65535
    changed_t[i] = i*1000+1
assert base == featurize(redact(changed_s, changed_t, hidden), n)
checks['hidden_values_counterfactual_bitwise_inputs'] = True
# Entire 16-slot units empty, but coverage, adjacency support and gap geometry defined.
assert base['stats'][1][0] == 0. and base['stat_valid'][1][1:13] == [0.]*12
assert base['stat_valid'][1][0] == 1. and base['stat_valid'][1][14:16] == [1., 1.]
checks['empty_unit_stat_validity'] = True
reverse = [{**r, 'signed_len': -r['signed_len']} for r in red]
rev = featurize(reverse, n)
for a, b in zip(base['stats'], rev['stats']):
    assert abs(a[1]-b[2]) < 1e-12 and abs(a[5]+b[5]) < 1e-12
    assert a[9:13] == list(reversed(b[9:13]))
checks['direction_reverse_equivariance_features'] = True
short = featurize(redact(signed[:64], times[:64], [3, 4]), 64)
batch = collate_features([base, short])
assert not batch['numeric'][1, 64:].any() and not batch['valid'][1, 64:].any()
checks['variable_length_padding_zero_and_excluded'] = True
model = TrafficFrontend().eval()
out = model(**batch)
assert out['packet_embeds'].shape == (2, 256, 2560)
assert out['behavior_embeds'].shape == (2, 21, 2560)
assert not out['visible_aggregate'][0, 1].any()
assert all(torch.isfinite(v).all() for v in out.values())
assert not out['packet_embeds'][1, 64:].any()
assert not out['behavior_embeds'][1, 6:].any()
checks['architecture_shapes_empty_aggregate_padding_finite'] = True
# Both branch paths and all parameter groups must receive a real differentiable signal.
loss = out['packet_embeds'].square().mean() + out['behavior_embeds'].square().mean()
loss.backward()
groups = {}
for key in ('packet_input', 'packet_encoder', 'behavior_input', 'scale_embedding',
            'behavior_encoder', 'packet_projector', 'behavior_projector', 'type_packet', 'type_behavior'):
    grads = [p.grad for name, p in model.named_parameters() if name.startswith(key)]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    total = sum(float(g.float().square().sum()) for g in grads)
    assert total > 0., key
    groups[key] = total**.5
checks['frontend_gradient_groups_finite_nonzero'] = groups
reloaded = TrafficFrontend().eval()
reloaded.load_state_dict(model.state_dict())
# Match grad mode, since Transformer eval fastpaths can differ under no_grad.
again = reloaded(**batch)
for key in ('packet_embeds', 'behavior_embeds'):
    assert torch.equal(out[key], again[key])
checks['frontend_state_reload_bitwise'] = True
result = {'scope': 'Synthetic frontend only; not Qwen integration, training, or performance evidence',
          'checks': checks, 'elapsed_seconds': time.monotonic()-t0,
          'parameters': sum(p.numel() for p in model.parameters()),
          'source_sha256': hashlib.sha256((ROOT/'audited_frontend.py').read_bytes()).hexdigest(),
          'torch': torch.__version__}
(ROOT/'frontend_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
