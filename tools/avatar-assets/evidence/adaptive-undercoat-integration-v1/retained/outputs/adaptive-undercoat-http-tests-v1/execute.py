"""Rerun only the 27 tests whose loopback binds the sandbox refused."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import require, sha256, write_json

previous = ROOT/"outputs/adaptive-undercoat-integration-tests-v2"
sources = json.loads((previous/"sources.json").read_text())["files"]
require(all(sha256(ROOT/row["path"]) == row["sha256"] for row in sources), "Suite inputs changed")
log = (previous/"tests.txt").read_text()
require(log.count("ERROR: ") == 27 and log.count("PermissionError: [Errno 1]") == 27
        and "FAILED (errors=27)" in log and "Ran 668 tests" in log, "Failure set changed")
require(not (HERE/"receipt.json").exists(), "Preserve prior test receipt")
result = run_bounded([sys.executable, "-m", "unittest", "tests.test_gallery.HTTPTests",
                     "tests.test_studio.StudioHTTPTests", "-q"], HERE/"tests.txt", HERE,
                     timeout=90, memory_mb=768, output_mb=64)
unchanged = all(sha256(ROOT/row["path"]) == row["sha256"] for row in sources)
receipt = {"schema_version": 1, "job": result, "sources_unchanged": unchanged,
           "original_suite_log_sha256": sha256(previous/"tests.txt"),
           "source_inventory_sha256": sha256(previous/"sources.json"),
           "log_sha256": sha256(HERE/"tests.txt"), "new_paid_calls": 0,
           "scope": "Temporary loopback HTTP fixtures only; no browser or generation"}
write_json(HERE/"receipt.json", receipt)
print(json.dumps(receipt, indent=2))
require(result["exit_code"] == 0 and not result["failure"] and unchanged, "HTTP checks failed")
