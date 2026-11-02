"""One bounded local CPU census; preserve both success and failure receipts."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sources"))
from avatar_assets.runner import run_bounded


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(name, value):
    with (HERE / name).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


prepared = json.loads((HERE / "prepared.json").read_text())
for entry in prepared["inputs"] + prepared["sources"]:
    if sha(entry["path"]) != entry["sha256"]:
        raise ValueError("Prepared source changed: " + entry["path"])
job = run_bounded([prepared["python"], str(HERE / "census.py")], HERE / "cpu.log", HERE,
                  timeout=30, memory_mb=512, output_mb=16)
write("cpu-job.json", job)
unchanged = all(sha(p["path"]) == p["sha256"] for p in prepared["inputs"] + prepared["sources"])
passed = job["exit_code"] == 0 and not job["failure"] and unchanged and (HERE / "census.json").is_file()
artifacts = [{"path": str(p.relative_to(HERE)), "bytes": p.stat().st_size, "sha256": sha(p)}
             for p in sorted(HERE.rglob("*")) if p.is_file()]
write("manifest.json", {"schema_version": 1, "status": "measured" if passed else "failed",
                       "prepared_sha256": sha(HERE / "prepared.json"), "inputs_unchanged": unchanged,
                       "artifacts": artifacts, "provider_calls": 0, "native_calls": 0,
                       "geometry_created": False, "quality_approved": False})
print(json.dumps({"passed": passed, "job": job, "manifest_sha256": sha(HERE / "manifest.json")}))
sys.exit(0 if passed else 1)
