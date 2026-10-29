"""Grade retained source labels after pipeline execution; never feed labels into construction."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from weft.benchmark import seal_suite
from weft.generic_vision import evaluate_manual
from weft.io import digest, local_time, write_json
from weft.observations import adapt_run


def run(candidate, out):
    out.mkdir(parents=True, exist_ok=False)
    label_path = Path("data/balls/triple-collision-240p.manual.json")
    labels = json.loads(label_path.read_text())
    normalized = {"source_sha256": labels["source_sha256"], "coordinate_system": "original_source_pixels", "annotations": []}
    for row in labels["observations"]:
        for identifier, center, visibility in zip(labels["objects"], row["centers"], row["visibility"], strict=True):
            if visibility not in {"visible", "occluded", "out_of_frame"} or (visibility == "visible") != (center is not None):
                raise ValueError("Unexpected frozen source-label visibility")
            normalized["annotations"].append({"frame": row["frame"], "object_id": identifier,
                "visible": visibility == "visible", "center_xy": center,
                "source_identity_status": row["identity_status"], "source_visibility": visibility})
    shutil.copyfile(label_path, out / "frozen-source-labels.json")
    write_json(out / "normalized-labels.json", normalized)
    traces = {}
    for name in ("long-analytic-interval", "rolling-grayscale-interval", "triple-collision-interval"):
        path = candidate / name / "run/appearance/appearance-tracks.json"
        traces[name] = json.loads(path.read_text())
        graph = adapt_run(candidate / name)
        write_json(out / f"{name}.observations.json", graph)
    grade = evaluate_manual(traces["triple-collision-interval"], normalized)
    baseline_path = Path("artifacts/appearance-guards-e2e-06/rolling-grayscale/run/appearance/appearance-tracks.json")
    baseline = json.loads(baseline_path.read_text())
    def positions(trace):
        return [{key: row.get(key) for key in ("id", "frame", "time_s", "object_id", "status", "box_xywh", "x_px", "y_px")}
                for row in trace["observations"]]
    same_positions = positions(baseline) == positions(traces["rolling-grayscale-interval"])
    if not same_positions:
        raise ValueError("Interval selection changed retained grayscale observations")
    inputs = {str(label_path): digest(label_path), str(baseline_path): digest(baseline_path)}
    for name in traces:
        p = candidate / name / "run/appearance/appearance-tracks.json"
        inputs[str(p)] = digest(p)
    audits = {name: trace["decode_audit"] for name, trace in traces.items()}
    write_json(out / "report.json", {"created_at": local_time(), "inputs": inputs,
        "real_source_grade": grade, "grayscale_observations_exactly_unchanged": same_positions,
        "real_status_counts": dict(Counter(row["status"] for row in traces["triple-collision-interval"]["observations"])),
        "decode_audits": audits, "scope": "Frozen approximate source-only development labels; initialized and pre-initialization points are retained in coverage, not fabricated detections. Post-contact source identities remain inferred. No whole-source encoded completeness or physical reconstruction claim."})
    for path in (Path(__file__), Path("src/weft/generic_vision.py"), Path("src/weft/observations.py")):
        shutil.copyfile(path, out / path.name)
    print({key: grade[key] for key in ("accepted_visible_labels", "visible_labels", "false_visible_labels", "accepted_rmse_source_px", "camera_check")})
    print(seal_suite(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.candidate, args.out)
