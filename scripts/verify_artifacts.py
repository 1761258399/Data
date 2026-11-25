"""Verify the supplied code-and-data release without loading a model."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


manifest = json.loads((ROOT / "MANIFEST_SHA256.json").read_text(encoding="utf-8"))
failed = []
for record in manifest["files"]:
    path = (ROOT / record["path"]).resolve()
    path.relative_to(ROOT)
    if (not path.is_file() or path.stat().st_size != record["bytes"]
            or digest(path) != record["sha256"]):
        failed.append(record["path"])
if failed:
    print(json.dumps({"passed": False, "mismatches": failed}))
    sys.exit(1)
print(json.dumps({"passed": True, "verified_files": len(manifest["files"]),
                  "scope": "code_data_reference_results", "pretrained_model_included": False}))
