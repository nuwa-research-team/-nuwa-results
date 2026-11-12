# -*- coding: utf-8 -*-
"""Flow -> non-overlapping windows -> group split (plan sections 5).

Rules: window length 64..256 (tail <64 dropped), non-overlap, split by capture
group (all windows of one capture/archive shard go to one split), 70/10/20 target.
Outputs windows.jsonl + split_manifest.json; never reads hidden values.
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

MIN_W = 64
MAX_W = 256
CAPTURES = {"SMB": "SMB", "Weibo": "Weibo"}  # multi-part captures keep one group


def capture_group(capture_id: str) -> str:
    cls, name = capture_id.split("/", 1)
    stem = Path(name).stem
    for k in CAPTURES:
        if stem.startswith(k):
            return f"{cls}/{CAPTURES[k]}"
    return f"{cls}/{stem}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", default="/data1/lhy/behavior_lora_d4_20261111/data/flows.jsonl")
    ap.add_argument("--out", default="/data1/lhy/behavior_lora_d4_20261111/data")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)          # group -> windows
    flow_counts = Counter()
    excl = Counter()
    n_flows = 0

    with open(args.flows) as fh:
        for line in fh:
            f = json.loads(line)
            n_flows += 1
            pk = f["packets"]
            n = len(pk)
            if n < MIN_W:
                excl["short_flow"] += 1
                continue
            g = capture_group(f["capture_id"])
            flow_counts[g] += 1
            cls = f["capture_id"].split("/", 1)[0]
            label = "malicious" if cls.lower().startswith("malware") else "benign"
            idx = 0
            while idx + MIN_W <= n:
                end = min(idx + MAX_W, n)
                seg = pk[idx:end]
                t0 = seg[0]["time_us"]
                times = [p["time_us"] - t0 for p in seg]
                signed = [p["signed_len"] for p in seg]
                sid = f"{f['flow_id']}#w{idx}-{end}"
                groups[g].append({
                    "sample_id": sid, "flow_id": f["flow_id"], "group_id": g,
                    "capture_id": f["capture_id"], "label": label,
                    "n": len(seg), "signed_lengths": signed, "times_us": times,
                })
                idx = end
            if idx < n:
                excl["tail_below_64"] += 1

    keys = sorted(groups)
    total = sum(len(groups[k]) for k in keys)
    target = {"train": 0.7, "val": 0.1, "test": 0.2}
    names = list(target)
    best, best_cost = None, 1e18
    # exhaustive 3^G assignment (G<=12); require every split non-empty
    import itertools
    for combo in itertools.product(range(3), repeat=len(keys)):
        acc = [0, 0, 0]
        for k, c in zip(keys, combo):
            acc[c] += len(groups[k])
        if min(acc) == 0:
            continue
        cost = sum(abs(acc[i] - target[names[i]] * total) for i in range(3))
        if cost < best_cost:
            best_cost, best = cost, combo
    split = {k: names[c] for k, c in zip(keys, best)}
    acc = {s: 0 for s in names}
    for k in keys:
        acc[split[k]] += len(groups[k])

    with open(out / "windows.jsonl", "w") as fh:
        for k in keys:
            for w in groups[k]:
                w["split"] = split[k]
                fh.write(json.dumps(w) + "\n")

    manifest = {
        "seed": args.seed, "min_window": MIN_W, "max_window": MAX_W,
        "flows_total": n_flows,
        "flows_used": int(sum(flow_counts.values())),
        "windows_total": total,
        "windows_by_split": acc,
        "groups": {k: {"split": split[k], "windows": len(groups[k]),
                       "flows": flow_counts[k]} for k in keys},
        "exclusions": dict(excl),
        "label_by_group": {k: groups[k][0]["label"] for k in keys},
        "note": ("split is by capture/archive-shard group; if a class has too few "
                 "independent captures this is an engineering pilot, not group-held-out proof"),
        "windows_sha256": hashlib.sha256((out / "windows.jsonl").read_bytes()).hexdigest(),
    }
    json.dump(manifest, open(out / "split_manifest.json", "w"), indent=2)
    print(json.dumps({k: manifest[k] for k in
                      ("flows_total", "flows_used", "windows_total", "windows_by_split", "exclusions")},
                     indent=2))
    for k in keys:
        print(k, split[k], len(groups[k]))


if __name__ == "__main__":
    main()
