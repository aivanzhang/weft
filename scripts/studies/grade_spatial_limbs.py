"""Compare frozen source-view limb directions without fitting a camera to labels."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from weft.benchmark import seal_suite
from weft.human import HumanEnv
from weft.io import digest, local_time, write_json
from weft.limb_fidelity import load_annotations, grade_limb_projection


def run(baseline, candidate, out, *, nested=False):
    labels_path = Path("data/limb-orientation/annotations.json")
    labels = load_annotations(labels_path, project_root=".")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(labels_path, out / "annotations.json")
    shutil.copyfile(__file__, out / "study.py")
    shutil.copyfile("src/weft/limb_fidelity.py", out / "grader.py")
    reports, inputs = {}, {}
    for kind, root in (("baseline", baseline), ("candidate", candidate)):
        predictions, matching = [], []
        for clip in sorted({row["clip_id"] for row in labels["frames"]}):
            directory = root / clip / ("run" if kind == "baseline" or nested else "")
            spec_path = directory / "human.json"
            spec = json.loads(spec_path.read_text())
            inputs[f"{kind}/{clip}"] = {"path": str(spec_path), "sha256": digest(spec_path)}
            shutil.copyfile(spec_path, out / f"{kind}-{clip}.human.json")
            env = HumanEnv(spec)
            try:
                times = np.array([row["time_s"] for row in spec["reference"]])
                for label in (row for row in labels["frames"] if row["clip_id"] == clip):
                    nearest = int(np.argmin(abs(times - label["timestamp_seconds"])))
                    offset = float(times[nearest] - label["timestamp_seconds"])
                    # Samples are every other source frame; larger gaps remain missing.
                    matched = abs(offset) <= 1 / 23 + 1e-8
                    matching.append({"id": label["id"], "reference_index": nearest,
                                     "offset_seconds": offset, "matched": matched})
                    segments = {}
                    if matched:
                        state = env.reference_snapshot(nearest)
                        for segment in state["human"]["segments"]:
                            if segment["id"] in label["segments"]:
                                axis = np.subtract(segment["b"], segment["a"])
                                segments[segment["id"]] = {"axis_xy": [float(axis[0]), float(-axis[2])]}
                    predictions.append({"id": label["id"], "source_sha256": spec["source_sha256"],
                                        "segments": segments})
            finally:
                env.close()
        write_json(out / f"{kind}-predictions.json", predictions)
        write_json(out / f"{kind}-time-matching.json", matching)
        reports[kind] = grade_limb_projection(labels, predictions)
    write_json(out / "report.json", {"created_at": local_time(), "inputs": inputs, **reports,
               "scope": "Frozen assistant-estimated limb directions; no camera alignment, contact or dynamics validation",
               "projection": "Simulation x and -z; no translation, scale, yaw or per-frame fit to annotations",
               "time_match_bound_seconds": 1 / 23})
    print(json.dumps({key: report["summary"] for key, report in reports.items()}, indent=2))
    print(json.dumps(seal_suite(out)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nested", action="store_true", help="Candidate is a full benchmark with case/run directories")
    args = parser.parse_args()
    run(args.baseline, args.candidate, args.out, nested=args.nested)
