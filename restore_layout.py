from pathlib import Path
import hashlib
import json
import shutil
import sys

base = Path(__file__).resolve().parent
if len(sys.argv) != 2:
    sys.exit("Usage: python restore_layout.py NEW_DIRECTORY")
target = Path(sys.argv[1]).resolve()
if target.exists() or base == target or base in target.parents:
    sys.exit("Choose a new directory outside the upload folder.")
items = json.loads((base / "source_manifest.json").read_text(encoding="utf-8"))
for item in items:
    source = base / item["file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
        sys.exit(f"File differs from manifest: {source.name}")
for item in items:
    dest = target / item["project_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base / item["file"], dest)
print(f"Restored {len(items)} files to {target}")
