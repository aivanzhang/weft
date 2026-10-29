"""Re-export frozen journals with foot close-ups; never rerun or alter dynamics."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from weft.benchmark import seal_suite
from weft.io import digest, write_json
from weft.pipeline import verify_artifacts
from weft.recording import export_recording, load_recording, render_snapshot


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/foot-body-view-01")
    args = parser.parse_args()
    out, source = Path(args.out).resolve(), ROOT / "artifacts/foot-body-01"
    verified = verify_artifacts(source)
    if out.exists() or out.is_relative_to(source):
        raise ValueError("Use a new output outside the frozen source study")
    out.mkdir(parents=True)
    shutil.copyfile(ROOT / "src/weft/recording.py", out / "recording.py")
    shutil.copyfile(Path(__file__), out / "export_foot_views.py")
    shutil.copyfile(ROOT / ".local/foot-body-replay-tests.xml", out / "replay-tests.xml")
    results = {}
    for case in ("single-leg-squat", "fullbody-squat"):
        original, target = source / case / "session", out / case
        target.mkdir()
        for filename in ("scene.json", "recording.json", "events.jsonl"):
            shutil.copyfile(original / filename, target / filename)
        results[case] = export_recording(target)
        saved = load_recording(target)
        assert saved["events"] == load_recording(original)["events"]
        before = digest(target / "events.jsonl")
        display = deepcopy(saved["events"][0]["snapshot"])
        points = np.array([display["human"]["joints"][side + "_" + part]
                           for side in ("left", "right") for part in ("ankle", "heel", "foot_index")])
        display["human"].update(view_origin=points.mean(axis=0).tolist(),
                                 view_height=float(max(.35, 1.8 * np.ptp(points, axis=0).max())))
        Image.fromarray(render_snapshot(display, saved["scene"])).save(out / (case + "-foot-closeup.png"))
        assert digest(target / "events.jsonl") == before == digest(original / "events.jsonl")
    write_json(out / "report.json", {"source": verified, "results": results,
                                     "scope": "Camera-only replay export; exact recorded actions/states unchanged, zero simulation or API calls"})
    print(json.dumps({"out": str(out), "seal": seal_suite(out), "results": results}, indent=2))


if __name__ == "__main__":
    main()
