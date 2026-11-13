"""PDF v1.0 numerical/behavior frontend, implemented by Codex.

No training entry point. featurize() ONLY accepts visible records, N and a
valid-slot contract. No clean hidden values, labels, endpoints or payloads.

records: list[dict(slot: int, signed_len: int, time_us: int)]
N: actual valid slots, not padded max length. Caller redacts before calling.
"""
import hashlib
import json
import math
import random
from statistics import mean, pstdev

import torch
from torch import nn


def mask_indices(n, rate, family, sample_id, seed=17, epoch=0):
    k = math.floor(rate*n + 0.5)
    if not 1 <= k <= n-2:
        raise ValueError('Infeasible exact mask budget')
    key = json.dumps([seed, str(sample_id), epoch, family, rate], separators=(',', ':'))
    rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))
    if family == 'random':
        result = rng.sample(range(1, n-1), k)
    elif family == 'block':
        first = rng.randint(1, n-1-k)
        result = list(range(first, first+k))
    elif family == 'multispan':
        rmax = min(4, k, n-k-1)
        if rmax < 2:
            raise ValueError('Multispan needs at least two feasible runs')
        runs = rng.randint(2, rmax)
        cuts = [0] + sorted(rng.sample(range(1, k), runs-1)) + [k]
        lengths = [b-a for a, b in zip(cuts, cuts[1:])]
        remaining = n-2-k-(runs-1)
        # Uniform weak composition via uniformly selected stars-and-bars bars.
        bars = [-1] + sorted(rng.sample(range(remaining+runs), runs)) + [remaining+runs]
        gaps = [b-a-1 for a, b in zip(bars, bars[1:])]
        result = []
        pos = 1+gaps[0]
        for r, length in enumerate(lengths):
            result.extend(range(pos, pos+length))
            pos += length
            if r+1 < runs:
                pos += 1+gaps[r+1]
        assert pos+gaps[-1] == n-1
    else:
        raise ValueError(f'Unknown mask family {family}')
    result = sorted(result)
    assert len(result) == len(set(result)) == k and min(result)>0 and max(result)<n-1
    return result


def redact(signed_lengths, times_us, hidden):
    """Boundary adapter ONLY; subsequent featurizers do not accept these arrays."""
    assert len(signed_lengths) == len(times_us)
    n = len(signed_lengths)
    hidden = set(hidden)
    assert 0 not in hidden and n-1 not in hidden
    return [dict(slot=i, signed_len=int(signed_lengths[i]), time_us=int(times_us[i]))
            for i in range(n) if i not in hidden]


def _run_lengths(missing):
    out = []
    current = 0
    for is_missing in missing:
        if is_missing:
            current += 1
        elif current:
            out.append(current)
            current = 0
    if current:
        out.append(current)
    return out


def featurize(records, n, max_packets=256):
    if not 2 <= n <= max_packets:
        raise ValueError('Invalid actual window length')
    rows = sorted(records, key=lambda r: r['slot'])
    visible = {r['slot']: r for r in rows}
    assert len(visible) == len(rows)
    assert all(type(r['slot']) is int and 0 <= r['slot'] < n for r in rows)
    assert all(type(r['time_us']) is int and r['time_us'] >= 0 for r in rows)
    assert not rows or rows[0]['time_us'] == 0
    assert all(40 <= abs(r['signed_len']) <= 65535 for r in rows)
    assert all(a['time_us'] <= b['time_us'] for a, b in zip(rows, rows[1:]))
    slots = sorted(visible)
    slot_set = set(slots)
    numeric = []
    prev = None
    for i in range(n):
        r = visible.get(i)
        if r is None:
            values = [0., 0., 0., 0.]
        else:
            gap = 0 if prev is None else r['time_us']-prev['time_us']
            values = [math.log1p(abs(r['signed_len'])/1500),
                      1. if r['signed_len'] > 0 else -1.,
                      math.log1p(r['time_us']/1000), math.log1p(gap/1000)]
            prev = r
        left = [s for s in slots if s < i]
        right = [s for s in slots if s > i]
        numeric.append(values + [i/(n-1), float(r is not None), float(bool(left)),
                                  (i-left[-1])/n if left else 0., float(bool(right)),
                                  (right[0]-i)/n if right else 0.])
    ranges = [(a, min(n, a+w), scale) for scale, w in enumerate((16, 64))
              for a in range(0, n, w)] + [(0, n, 2)]
    stats, stat_valid = [], []
    for start, end, _ in ranges:
        local = [visible[i] for i in range(start, end) if i in visible]
        m, width = len(local), end-start
        s, v = [0.]*16, [0.]*16
        s[0], v[0] = m/width, 1.
        if m:
            lens = [abs(r['signed_len']) for r in local]
            logs = [math.log1p(x/1500) for x in lens]
            s[1:6] = [sum(r['signed_len']>0 for r in local)/m,
                      sum(r['signed_len']<0 for r in local)/m,
                      mean(logs), pstdev(logs), sum(r['signed_len'] for r in local)/sum(lens)]
            v[1:6] = [1.]*5
        if m >= 2:
            gaps = [math.log1p((b['time_us']-a['time_us'])/1000)
                    for a, b in zip(local, local[1:])]
            s[6:9] = [math.log1p((local[-1]['time_us']-local[0]['time_us'])/1000),
                      mean(gaps), pstdev(gaps)]
            v[6:9] = [1.]*3
        pairs = [(visible[i]['signed_len']>0, visible[i+1]['signed_len']>0)
                 for i in range(start, end-1) if i in slot_set and i+1 in slot_set]
        if pairs:
            s[9:13] = [pairs.count(pattern)/len(pairs)
                        for pattern in [(True, True), (True, False), (False, True), (False, False)]]
            v[9:13] = [1.]*4
        if width >= 2:
            s[13], v[13] = len(pairs)/(width-1), 1.
        runs = _run_lengths([i not in slot_set for i in range(start, end)])
        s[14:16] = [max(runs, default=0)/width, len(runs)/width]
        v[14:16] = [1., 1.]
        stats.append(s)
        stat_valid.append(v)
    return dict(numeric=numeric, observation=[i in visible for i in range(n)],
                valid=[True]*n, stats=stats, stat_valid=stat_valid,
                ranges=[[a, b] for a, b, _ in ranges], scale_ids=[c for _, _, c in ranges],
                queries=[i for i in range(n) if i not in visible])


def collate_features(features, device='cpu'):
    """Dynamic padding only at the batch tail; original N retained per sample."""
    b = len(features)
    n = max(len(f['numeric']) for f in features)
    k = max(len(f['stats']) for f in features)
    result = dict(numeric=torch.zeros(b, n, 10), observation=torch.zeros(b, n, dtype=torch.bool),
                  valid=torch.zeros(b, n, dtype=torch.bool), stats=torch.zeros(b, k, 16),
                  stat_valid=torch.zeros(b, k, 16), ranges=torch.zeros(b, k, 2, dtype=torch.long),
                  scale_ids=torch.zeros(b, k, dtype=torch.long),
                  behavior_valid=torch.zeros(b, k, dtype=torch.bool))
    for i, f in enumerate(features):
        ni, ki = len(f['numeric']), len(f['stats'])
        for key in ('numeric', 'observation', 'valid'):
            result[key][i, :ni] = torch.tensor(f[key], dtype=result[key].dtype)
        for key in ('stats', 'stat_valid', 'ranges', 'scale_ids'):
            result[key][i, :ki] = torch.tensor(f[key], dtype=result[key].dtype)
        result['behavior_valid'][i, :ki] = True
    return {key: value.to(device) for key, value in result.items()}


class TrafficFrontend(nn.Module):
    def __init__(self, llm_dim=2560):
        super().__init__()
        self.packet_input = nn.Linear(10, 256)
        layer = nn.TransformerEncoderLayer(256, 4, 1024, dropout=.1, batch_first=True,
                                           norm_first=True)
        self.packet_encoder = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
        self.behavior_input = nn.Linear(290, 256)
        self.scale_embedding = nn.Embedding(3, 256)
        layer_b = nn.TransformerEncoderLayer(256, 4, 1024, dropout=.1, batch_first=True,
                                             norm_first=True)
        self.behavior_encoder = nn.TransformerEncoder(layer_b, 2, enable_nested_tensor=False)
        def projector():
            return nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 512), nn.GELU(),
                                 nn.Linear(512, llm_dim))
        self.packet_projector = projector()
        self.behavior_projector = projector()
        self.type_packet = nn.Parameter(torch.zeros(llm_dim))
        self.type_behavior = nn.Parameter(torch.zeros(llm_dim))
        nn.init.normal_(self.type_packet, std=.02)
        nn.init.normal_(self.type_behavior, std=.02)

    def forward(self, numeric, observation, valid, stats, stat_valid, ranges,
                scale_ids, behavior_valid):
        b, n = valid.shape
        pos = torch.arange(n, device=numeric.device, dtype=torch.float32).unsqueeze(1)
        freq = torch.exp(torch.arange(0, 256, 2, device=numeric.device)*(-math.log(10000)/256))
        pe = torch.zeros(n, 256, device=numeric.device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos*freq), torch.cos(pos*freq)
        hp = self.packet_input(numeric) + pe.to(numeric.dtype)
        hp = self.packet_encoder(hp, src_key_padding_mask=~valid)
        hp = hp.masked_fill(~valid.unsqueeze(-1), 0.)
        indices = torch.arange(n, device=numeric.device)[None, None, :]
        support = ((indices >= ranges[..., 0, None]) & (indices < ranges[..., 1, None]) &
                   observation[:, None, :] & valid[:, None, :] & behavior_valid[..., None])
        counts = support.sum(-1, keepdim=True).clamp_min(1)
        weights = support.to(hp.dtype)/counts.to(hp.dtype)
        pooled = torch.bmm(weights, hp)
        actual_n = valid.sum(-1).clamp_min(1)[:, None, None]
        bounds = ranges.to(hp.dtype)/actual_n.to(hp.dtype)
        inp = torch.cat([pooled, stats.to(hp.dtype), stat_valid.to(hp.dtype), bounds], dim=-1)
        hb = self.behavior_input(inp) + self.scale_embedding(scale_ids)
        hb = self.behavior_encoder(hb, src_key_padding_mask=~behavior_valid)
        hb = hb.masked_fill(~behavior_valid.unsqueeze(-1), 0.)
        ep = self.packet_projector(hp) + self.type_packet
        eb = self.behavior_projector(hb) + self.type_behavior
        return dict(packet_embeds=ep.masked_fill(~valid.unsqueeze(-1), 0.),
                    behavior_embeds=eb.masked_fill(~behavior_valid.unsqueeze(-1), 0.),
                    packet_valid=valid, behavior_valid=behavior_valid,
                    packet_hidden=hp, behavior_hidden=hb, visible_aggregate=pooled)
