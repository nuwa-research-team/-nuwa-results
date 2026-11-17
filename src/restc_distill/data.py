"""Group-disjoint data preparation and visible-only retrieval for the teacher."""
from collections import Counter, defaultdict
import pickle
import random
from pathlib import Path
from .common import (canonical, digest, file_hash, read, resolve, save, write_lines,
                     checked_rows, target_input, student_messages, schema, parse_answer)

SPLITS = ("train", "validation", "test")


def split_for(group, seed, fractions):
    u = int(digest(["split", seed, group]), 16) / 2**256
    return "train" if u < fractions[0] else "validation" if u < sum(fractions[:2]) else "test"


def retrieve(observed, positions, bank, excluded_group, k, matrix=None):
    # Exact nearest neighbours under mean absolute signed-length distance.
    # The candidate set is defined by length; an empty set yields no evidence.
    visible = [i for i, v in enumerate(observed) if v is not None]
    import numpy as np
    if not bank:
        return [], []
    if matrix is None:
        matrix = np.asarray([g["truth"] for g in bank], dtype=np.int32)
    distances = np.abs(matrix[:, visible] - np.asarray([observed[i] for i in visible], dtype=np.int32)).mean(axis=1) / 3200
    candidates = []
    for j, g in enumerate(bank):
        if g["group_id"] == excluded_group or len(g["truth"]) != len(observed):
            continue
        distance = float(distances[j])
        candidates.append((distance, g["group_id"], g))
    candidates.sort(key=lambda x: (x[0], x[1]))
    examples, sources = [], []
    for distance, group_id, g in candidates[:k]:
        masked = [None if i in positions else v for i, v in enumerate(g["truth"])]
        example = target_input(masked, positions)
        example.update(answer={"values": {str(i): g["truth"][i] for i in positions}},
                       observed_distance=distance, training_frequency=len(g["indices"]))
        examples.append(example)
        sources.append(group_id)
    return examples, sources


def prepare(config_path, out):
    config = read(config_path)
    out = Path(out)
    if out.exists():
        raise FileExistsError("Use a new prepared-data directory; frozen inputs are never overwritten")
    raw_path = resolve(config["raw_data"])
    actual_hash = file_hash(raw_path)
    if actual_hash != config["raw_sha256"]:
        raise ValueError("raw dataset hash mismatch")
    # This is the user's trusted local pickle, pinned by SHA-256 above.
    with raw_path.open("rb") as f:
        raw = pickle.load(f)
    fractions = config["split_fractions"]
    if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-8 or min(fractions) <= 0:
        raise ValueError("three positive split fractions summing to one are required")
    if config["retrieval_k"] < 1 or any(not 0 < r < 1 for r in config["loss_rates"]):
        raise ValueError("invalid retrieval count or loss rates")
    groups = {}
    for index, seq in enumerate(raw["seqs"]):
        truth = [int(x) for x in seq]
        if len(truth) < 2 or any(x != original for x, original in zip(truth, seq)) or any(not 0 < abs(x) <= 1600 for x in truth):
            raise ValueError(f"invalid sequence at raw index {index}; no clipping or coercion")
        group_id = digest(truth)
        g = groups.setdefault(group_id, {"group_id": group_id, "truth": truth, "indices": [], "label_counts": Counter()})
        g["indices"].append(index)
        g["label_counts"][str(raw["labels"][int(raw["y"][index])])] += 1
    split_groups = {s: [] for s in SPLITS}
    for g in groups.values():
        split_groups[split_for(g["group_id"], config["seed"], fractions)].append(g)
    for seqs in split_groups.values():
        seqs.sort(key=lambda g: digest(["order", config["seed"], g["group_id"]]))
    bank = split_groups["train"]
    by_length = defaultdict(list)
    for g in bank:
        by_length[len(g["truth"])].append(g)
    import numpy as np
    matrices = {n: np.asarray([g["truth"] for g in v], dtype=np.int32) for n, v in by_length.items()}
    teacher_system = resolve(config["teacher_prompt"]).read_text()
    student_system = resolve(config["student_prompt"]).read_text().strip()
    manifests, files = {}, {}
    out.mkdir(parents=True)
    for split in SPLITS:
        limit = config["flow_limits"][split]
        if limit < 1 or limit > len(split_groups[split]):
            raise ValueError(f"flow limit unavailable for {split}: {limit}")
        records = []
        for g in split_groups[split][:limit]:
            n = len(g["truth"])
            for rate in config["loss_rates"]:
                count = max(1, min(n - 1, round(rate * n)))
                rng = random.Random(int(digest(["mask", config["seed"], g["group_id"], rate]), 16))
                positions = sorted(rng.sample(range(n), count))
                observed = [None if i in positions else v for i, v in enumerate(g["truth"])]
                examples, source_groups = retrieve(observed, positions, by_length[n], g["group_id"], config["retrieval_k"], matrices.get(n))
                payload = {"training_examples": examples, "target": target_input(observed, positions)}
                teacher = config["teacher"]
                request = {"systemInstruction": {"parts": [{"text": teacher_system}]},
                           "contents": [{"role": "user", "parts": [{"text": canonical(payload)}]}],
                           "generationConfig": {"candidateCount": 1, "maxOutputTokens": teacher["max_output_tokens"],
                                                "thinkingConfig": {"thinkingLevel": teacher["thinking_level"]},
                                                "responseMimeType": "application/json", "responseJsonSchema": schema(positions)}}
                records.append({"id": digest([g["group_id"], rate, positions]), "split": split,
                                "group_id": g["group_id"], "raw_index": g["indices"][0],
                                "label": str(raw["labels"][int(raw["y"][g["indices"][0]])]),
                                "group_label_counts": dict(g["label_counts"]), "loss_rate": rate,
                                "positions": positions, "observed": observed, "truth": g["truth"],
                                "messages": student_messages(observed, positions, student_system),
                                "teacher_request": request, "retrieval_group_ids": source_groups,
                                "request_sha256": digest(request)})
        path = out / (split + ".jsonl")
        write_lines(path, records)
        files[path.name] = file_hash(path)
        manifests[split] = {"all_unique_groups": len(split_groups[split]), "selected_unique_groups": limit,
                            "records": len(records), "empty_retrieval_records": sum(not r["retrieval_group_ids"] for r in records)}
    save(out / "groups.json", {s: [{"group_id": g["group_id"], "raw_indices": g["indices"],
                                    "label_counts": dict(g["label_counts"])} for g in v] for s, v in split_groups.items()})
    files["groups.json"] = file_hash(out / "groups.json")
    manifest = {"format": "restc-distillation-v1", "config": config, "config_sha256": digest(config),
                "raw_sha256": actual_hash, "files": files, "splits": manifests,
                "conflicting_label_groups": sum(len(g["label_counts"]) > 1 for g in groups.values()),
                "protocol": "full-sequence-group-disjoint; each unique target once per rate; teacher retrieval from training groups only; student target only",
                "student_system": student_system, "teacher_system_sha256": digest(teacher_system)}
    save(out / "manifest.json", manifest)
    print(canonical(manifests))


def build_sft(prepared, responses, out, limit=None, target="teacher"):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Use a new SFT directory")
    train, manifest = checked_rows(prepared, "train")
    validation, _ = checked_rows(prepared, "validation")
    if limit is not None:
        if not 0 < limit <= len(train):
            raise ValueError("invalid explicit training subset size")
        train = train[:limit]
    output = {}
    teacher_records = {}
    for split, rows in (("train", train), ("validation", validation)):
        built = []
        for row in rows:
            source = "teacher" if split == "train" and target == "teacher" else "ground_truth"
            if source == "teacher":
                record = read(Path(responses) / (row["id"] + ".json"))
                if record["request_sha256"] != row["request_sha256"] or record["teacher"] != manifest["config"]["teacher"]:
                    raise ValueError("teacher provenance mismatch")
                if record["error"] is not None:
                    raise ValueError("invalid teacher response; dataset build stopped, no answer substitution")
                # Revalidate the raw provider text, not a mutable parsed cache alone.
                from .teacher import extract_answer
                answer = extract_answer(record["response"], row["positions"])
                teacher_records[row["id"]] = digest(record)
            else:
                answer = {"values": {str(p): row["truth"][p] for p in row["positions"]}}
            parse_answer(canonical(answer), row["positions"])
            built.append({"id": row["id"], "group_id": row["group_id"], "split": split,
                          "target_source": source, "messages": row["messages"] + [{"role": "assistant", "content": canonical(answer)}]})
        output[split] = built
    if {r["group_id"] for r in output["train"]} & {r["group_id"] for r in output["validation"]}:
        raise ValueError("train/validation leakage")
    out.mkdir(parents=True)
    for split, rows in output.items():
        write_lines(out / (split + ".jsonl"), rows)
    save(out / "manifest.json", {"format": "restc-sft-v1", "prepared_manifest_sha256": file_hash(Path(prepared) / "manifest.json"),
                                "objective": "sequence-level distillation" if target == "teacher" else "ground-truth supervised ablation",
                                "validation_target": "ground_truth", "counts": {s: len(v) for s, v in output.items()},
                                "files": {s + ".jsonl": file_hash(out / (s + ".jsonl")) for s in output},
                                "teacher_record_sha256": teacher_records, "student_system": manifest["student_system"]})
    print(canonical({"sft": str(out), "counts": {s: len(v) for s, v in output.items()}}))
