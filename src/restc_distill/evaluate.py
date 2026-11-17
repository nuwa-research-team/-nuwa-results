"""Packet metrics and opt-in replay through the existing six classifier heads."""
from collections import defaultdict
from pathlib import Path
import importlib.util
import sys
from .common import read, read_lines, save, checked_rows, restore, digest, file_hash


def packet_metrics(rows, answers):
    missing = sum(len(r["positions"]) for r in rows)
    valid_flows = valid_slots = exact = signs = 0
    length_error = 0
    for row, answer in zip(rows, answers, strict=True):
        if answer is None:
            continue
        completed = restore(row, answer)
        valid_flows += 1
        for p in row["positions"]:
            value, truth = completed[p], row["truth"][p]
            valid_slots += 1
            exact += value == truth
            signs += (value > 0) == (truth > 0)
            length_error += abs(abs(value) - abs(truth))
    return {"flows": len(rows), "valid_flows": valid_flows, "missing_slots": missing, "valid_slots": valid_slots,
            "valid_flow_rate": valid_flows / len(rows), "exact_accuracy_all_slots": exact / missing,
            "direction_accuracy_all_slots": signs / missing,
            "length_mae_valid_slots": length_error / valid_slots if valid_slots else None,
            "invalid_policy": "abstain; exact/direction count invalid slots as errors; MAE reports valid slots only"}


def classifier_metrics(labels, predictions, num_classes=6):
    correct = sum(y == p for y, p in zip(labels, predictions, strict=True))
    f1 = []
    for c in range(num_classes):
        tp = sum(y == c and p == c for y, p in zip(labels, predictions))
        fp = sum(y != c and p == c for y, p in zip(labels, predictions))
        fn = sum(y == c and p != c for y, p in zip(labels, predictions))
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"accuracy": 100 * correct / len(labels), "macro_f1": 100 * sum(f1) / num_classes}


def load_legacy(config_path):
    config = read(config_path)
    for filename, checksum in config["files"].items():
        if file_hash(filename) != checksum:
            raise ValueError("classifier asset identity mismatch: " + filename)
    path = Path(config["module"])
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("restc_legacy_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, Path(config["heads"]), config


def evaluate(prepared, predictions, out, split="test", classifiers=None, allow_development=False):
    rows, manifest = checked_rows(prepared, split)
    if Path(out).exists():
        raise FileExistsError("evaluation output already exists")
    methods, identities = {}, {}
    required = {r["id"] for r in rows}
    for item in predictions:
        name, path = item.split("=", 1)
        if name in methods or name in ("Normal", "Baseline"):
            raise ValueError("duplicate/reserved method name")
        records = read_lines(path)
        mapping = {r["id"]: r for r in records}
        if len(mapping) != len(records) or set(mapping) != required:
            raise ValueError("predictions must cover exactly the frozen split; no omitted difficult samples")
        answers = []
        for row in rows:
            pred = mapping[row["id"]]
            if pred["input_sha256"] != digest(row["messages"]):
                raise ValueError("prediction input identity mismatch")
            answer = pred["answer"]
            if answer is not None:
                restore(row, answer)
            answers.append(answer)
        methods[name] = answers
        identities[name] = {"path": str(path), "sha256": file_hash(path)}
    result = {"prepared_manifest_sha256": file_hash(Path(prepared) / "manifest.json"), "split": split,
              "prediction_files": identities, "packet_metrics": {}, "classification": None}
    rates = sorted({r["loss_rate"] for r in rows})
    for rate in rates:
        indices = [i for i, r in enumerate(rows) if r["loss_rate"] == rate]
        result["packet_metrics"][str(rate)] = {name: packet_metrics([rows[i] for i in indices], [answers[i] for i in indices])
                                               for name, answers in methods.items()}
    if classifiers:
        if not allow_development:
            raise ValueError("historical classifiers were trained on overlapping original data; require --allow-development-classifiers")
        module, heads, config = load_legacy(classifiers)
        label_map = config["label_map"]
        classified = {}
        for rate in rates:
            ix = [i for i, r in enumerate(rows) if r["loss_rate"] == rate]
            group = [rows[i] for i in ix]
            labels = [label_map[r["label"]] for r in group]
            sequences = {"Normal": [r["truth"] for r in group],
                         "Baseline": [[v for v in r["observed"] if v is not None] for r in group]}
            for name, answers in methods.items():
                sequences[name] = [restore(rows[i], answers[i]) if answers[i] is not None else None for i in ix]
            scored = {}
            for name, seqs in sequences.items():
                valid = [i for i, seq in enumerate(seqs) if seq is not None]
                partial = module.classify([seqs[i] for i in valid], heads) if valid else {k: [] for k in module.KINDS}
                scores = {}
                for kind, p in partial.items():
                    full = [-1] * len(group)
                    for index, value in zip(valid, p, strict=True):
                        full[index] = value
                    scores[kind] = {**classifier_metrics(labels, full), "predictions": full}
                scored[name] = scores
            classified[str(rate)] = scored
        result["classification"] = {"independent_test": False,
            "reason": "Historical classifier training overlaps the regrouped data; development comparison only.",
            "assets_config_sha256": file_hash(classifiers), "metrics": classified}
    save(out, result)
    print(str(out))
