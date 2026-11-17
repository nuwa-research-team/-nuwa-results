"""Canonical artifacts and strict signed-packet schema. No output repair."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def write_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(canonical(row) + "\n")
    tmp.replace(path)


def read_lines(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def unique_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("duplicate JSON key")
        out[k] = v
    return out


def parse_answer(text, positions):
    obj = json.loads(text, object_pairs_hook=unique_keys)
    if not isinstance(obj, dict) or set(obj) != {"values"} or not isinstance(obj["values"], dict):
        raise ValueError("expected exactly {values: object}")
    values = obj["values"]
    if set(values) != {str(p) for p in positions}:
        raise ValueError("missing/extra position")
    if any(type(v) is not int or not 0 < abs(v) <= 1600 for v in values.values()):
        raise ValueError("expected nonzero signed integer within [-1600,1600]")
    return obj


def schema(positions):
    integer = {"anyOf": [{"type": "integer", "minimum": -1600, "maximum": -1},
                         {"type": "integer", "minimum": 1, "maximum": 1600}]}
    return {"type": "object", "properties": {"values": {
        "type": "object", "properties": {str(p): integer for p in positions},
        "required": [str(p) for p in positions], "additionalProperties": False}},
        "required": ["values"], "additionalProperties": False}


def rle(values):
    rows = []
    for i, v in enumerate(values):
        direction = None if v is None else "P" if v > 0 else "N"
        magnitude = None if v is None else abs(v)
        if rows and rows[-1][2:] == [direction, magnitude]:
            rows[-1][1] = i
        else:
            rows.append([i, i, direction, magnitude])
    return rows


def target_input(observed, positions):
    if len(set(positions)) != len(positions) or sorted(positions) != positions:
        raise ValueError("positions must be unique and sorted")
    if [i for i, x in enumerate(observed) if x is None] != positions:
        raise ValueError("observation mask mismatch")
    return {"total_slots": len(observed), "intervals_start_end_direction_bytes": rle(observed),
            "missing_positions_zero_based": positions, "missing_count": len(positions)}


def student_messages(observed, positions, system):
    return [{"role": "system", "content": system},
            {"role": "user", "content": canonical({"target": target_input(observed, positions)})}]


def restore(row, answer):
    values = parse_answer(canonical(answer), row["positions"])["values"]
    out = row["observed"].copy()
    for key, v in values.items():
        out[int(key)] = v
    return out


def checked_rows(directory, split):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    filename = split + ".jsonl"
    if file_hash(directory / filename) != manifest["files"][filename]:
        raise ValueError("prepared input hash mismatch: " + filename)
    return read_lines(directory / filename), manifest
