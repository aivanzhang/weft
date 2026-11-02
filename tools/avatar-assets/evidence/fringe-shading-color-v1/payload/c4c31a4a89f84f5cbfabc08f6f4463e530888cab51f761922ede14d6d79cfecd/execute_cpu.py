"""One bounded ownership read; no native execution or texture decode."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"sources"))
from avatar_assets.runner import run_bounded


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(name, value):
    with (HERE/name).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


prepared = json.loads((HERE/"prepared.json").read_text())
for row in prepared["inputs"]+prepared["sources"]:
    if sha(row["path"]) != row["sha256"]:
        raise ValueError("Pinned input changed: "+row["path"])
job = run_bounded([prepared["python"], str(HERE/"audit.py")], HERE/"cpu.log", HERE,
                  timeout=15, memory_mb=256, output_mb=8)
write("cpu-job.json", job)
unchanged = all(sha(row["path"]) == row["sha256"] for row in prepared["inputs"]+prepared["sources"])
completed = job["exit_code"] == 0 and not job["failure"] and unchanged and (HERE/"ownership.json").is_file()
result = json.loads((HERE/"ownership.json").read_text()) if completed else {}
artifacts = [{"path": str(p.relative_to(HERE)), "bytes": p.stat().st_size, "sha256": sha(p)}
             for p in sorted(HERE.rglob("*")) if p.is_file()]
write("manifest.json", {"schema_version": 1, "status": result.get("status", "ownership_audit_failed"),
                       "measurement_completed": completed, "candidate_accepted": False,
                       "quality_approved": False, "inputs_unchanged": unchanged,
                       "prepared_sha256": sha(HERE/"prepared.json"), "artifacts": artifacts,
                       "provider_calls": 0, "native_calls": 0, "texture_decodes": 0})
print(json.dumps({"measurement_completed": completed, "status": result.get("status"),
                  "job": job, "manifest_sha256": sha(HERE/"manifest.json")}))
sys.exit(0 if completed else 1)
