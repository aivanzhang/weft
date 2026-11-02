"""Run the single prepared panda finish after exact local verification is sealed."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from avatar_assets.safety import require, sha256, write_json

release = json.loads((HERE/"verified-release.json").read_text())
require(release["status"] == "verified_local_preconditions", "Local checks are not complete")
require(release["native_attempts_max"] == 1 and release["new_paid_calls"] == 0, "Execution scope changed")
require(not (HERE/"execution-started.json").exists(), "Prepared actual finish already started")
for row in release["pinned_inputs"]:
    require(sha256(ROOT/row["path"]) == row["sha256"], "Prepared input changed: "+row["path"])
target = ROOT/"outputs/panda-adaptive-undercoat-finish-v2"
require(not target.exists(), "Preserve prior actual output")
write_json(HERE/"execution-started.json", {"release_sha256": sha256(HERE/"verified-release.json"),
    "output": str(target), "new_paid_calls": 0, "native_attempts_max": 1})
from avatar_assets.finish_generated import finish
try:
    result = finish(HERE/"finish-design.json", target)
    write_json(HERE/"execution-result.json", {"result": result, "new_paid_calls": 0,
        "quality_approved": False, "full_media_rendered": False})
    print(json.dumps(result, indent=2))
except Exception as error:
    write_json(HERE/"execution-result.json", {"status": "failed_preserved", "new_paid_calls": 0,
        "error": type(error).__name__+": "+str(error), "automatic_retry": False,
        "quality_approved": False})
    raise
