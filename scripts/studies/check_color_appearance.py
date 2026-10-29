"""Compare immutable appearance evidence on retained development sources."""

import argparse
import json
from pathlib import Path
import shutil

from weft.appearance_consistency import COLOR_CONTRACT
from weft.benchmark import seal_suite
from weft.generic_vision import evaluate_manual, track_video
from weft.io import digest, local_time, write_json


CASES = [
    ("quick-pool-break", "data/balls/quick-pool-break.webm", "data/balls/quick-pool-break.initialization.json",
     "data/balls/quick-pool-break.manual.json", "artifacts/real-ball-followup-01/quick-pool-break/appearance/appearance-tracks.json"),
    ("air-table-collision", "data/balls/air-table-collision.webm", "data/balls/air-table-collision.initialization.json",
     "data/balls/air-table-collision.manual.json", "artifacts/real-ball-followup-01/air-table-collision/appearance/appearance-tracks.json"),
    ("pendulum", "data/general/elastic-pendulum.webm", "data/general/annotations/pendulum.initialization.json",
     "data/general/annotations/pendulum.manual.json", "artifacts/appearance-tracking-01/pendulum/appearance-tracks.json"),
    ("electrostatic", "data/balls/bouncing-ball.webm", "data/general/annotations/electrostatic.initialization.json",
     "data/general/annotations/electrostatic.manual.json", "artifacts/appearance-tracking-01/electrostatic/appearance-tracks.json"),
    ("rolling-grayscale", "data/general/rolling-ball-grayscale.avi", "data/general/annotations/rolling-grayscale.initialization.json",
     None, "artifacts/explicit-modes-e2e-03/rolling-grayscale/run/appearance/appearance-tracks.json"),
]


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), *[Path("src/weft") / name for name in (
        "appearance_consistency.py", "generic_vision.py", "scene_vision.py", "video_io.py")],
        Path("tests/test_appearance_consistency.py"), Path("tests/test_generic_vision.py")]
    hashes = {str(path): digest(path) for path in source_paths}
    for path in source_paths:
        shutil.copyfile(path, out / path.name)
    write_json(out / "pre-evaluation-freeze.json", {"created_at": local_time(), "hashes": hashes,
               "contract": COLOR_CONTRACT, "cases": CASES,
               "scope": "Fixed analytic color rules before real candidate scoring; exposed development labels, not final holdouts"})
    reports = []
    for name, video, config_path, labels_path, baseline_path in CASES:
        config = json.loads(Path(config_path).read_text())
        baseline = json.loads(Path(baseline_path).read_text())
        candidate = track_video(video, out / name, config)
        inputs = {path: digest(path) for path in (video, config_path, baseline_path)}
        report = {"case": name, "inputs": inputs, "candidate_seconds": candidate["seconds"],
                  "baseline_seconds": baseline["seconds"]}
        if labels_path:
            labels = json.loads(Path(labels_path).read_text())
            inputs[labels_path] = digest(labels_path)
            shutil.copyfile(labels_path, out / name / "manual-labels.json")
            for key, value in (("baseline", baseline), ("candidate", candidate)):
                report[key] = evaluate_manual(value, labels)
        else:
            def accepted(tracks):
                return [(row["frame"], row["object_id"], row["x_px"], row["y_px"])
                        for row in tracks["observations"] if row["status"] == "tracked"]
            report.update(unchanged_accepted_positions=accepted(candidate) == accepted(baseline),
                          candidate_accepted=len(accepted(candidate)), baseline_accepted=len(accepted(baseline)))
        write_json(out / name / "comparison.json", report)
        reports.append(report)
    if hashes != {str(path): digest(path) for path in source_paths}:
        raise ValueError("Study implementation changed while running; preserve output but do not promote")
    write_json(out / "report.json", {"created_at": local_time(), "cases": reports, "hashes": hashes,
               "external_api_usd": 0, "scope": "Tracking development comparison only; no reconstructed physics or verified identity claim"})
    print(json.dumps(seal_suite(out)))
    for report in reports:
        print(report["case"], {key: ({field: value[field] for field in (
            "accepted_visible_labels", "visible_labels", "false_visible_labels", "accepted_rmse_source_px", "camera_check")}
            if key in ("baseline", "candidate") else value) for key, value in report.items() if key != "inputs"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out)
