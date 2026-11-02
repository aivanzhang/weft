"""Single bounded local suite after the adaptive adapter is frozen; no provider calls."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from avatar_assets.runner import run_bounded
from avatar_assets.safety import require, sha256, write_json


def inventory():
    return [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for folder in ("avatar_assets", "scripts", "tests")
            for path in sorted((ROOT/folder).rglob("*.py"))]


require(not (HERE/"sources.json").exists(), "Test run already prepared; preserve its evidence")
before = inventory()
write_json(HERE/"sources.json", {"files": before})
result = run_bounded([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
                     HERE/"tests.txt", HERE, timeout=180, memory_mb=1536, output_mb=64)
unchanged = before == inventory()
receipt = {"schema_version": 1, "job": result, "sources_unchanged": unchanged,
           "source_count": len(before), "log_sha256": sha256(HERE/"tests.txt"),
           "new_paid_calls": 0}
write_json(HERE/"receipt.json", receipt)
print(json.dumps(receipt, indent=2))
require(result["exit_code"] == 0 and not result["failure"] and unchanged,
        "Tests failed or source inputs changed; inspect retained evidence")
