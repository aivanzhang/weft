"""Score frozen source pair directions; corrections never redefine the labels."""

import argparse
import json
import math
from pathlib import Path
import shutil

from weft.body_fidelity import angular_error_degrees
from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json


def baseline_path(clip):
    study = "exercise-body-e2e-02" if clip == "lunge-twist" else "spatial-human-e2e-04"
    return Path("artifacts") / study / clip / "run" / "evidence.json"


def grade(labels, evidence_by_clip):
    rows = []
    for label in labels["frames"]:
        evidence = evidence_by_clip[label["clip_id"]]
        if evidence["source_sha256"] != label["source_sha256"]:
            raise ValueError("Identity benchmark source mismatch")
        sample = min(evidence["frames"], key=lambda row: abs(row["time_s"] - label["timestamp_seconds"]))
        offset = sample["time_s"] - label["timestamp_seconds"]
        nearby = abs(offset) <= 1 / evidence["fps"] + 1e-8
        for name, (left, right) in {"shoulder": (11, 12), "hip": (23, 24)}.items():
            target = label["pairs"][name]
            row = {"id": label["id"], "clip_id": label["clip_id"], "pair": name,
                   "source_observable": target["observable"], "status": "source_unobservable",
                   "sample_id": sample["id"], "time_offset_s": offset,
                   "reassigned": "identity_raw" in sample, "angular_error_degrees": None,
                   "tolerance_degrees": target.get("angular_tolerance_degrees"), "passes": False}
            if target["observable"]:
                row["status"] = "prediction_missing"
                points = sample.get("landmarks", [])
                if nearby and sample["status"] in {"tracked", "tracked_partial"} and len(points) == 33:
                    visible = all(min(points[i]["visibility"], points[i]["presence"]) >= .5
                                  and -.05 <= points[i]["x"] <= 1.05 and -.05 <= points[i]["y"] <= 1.05
                                  for i in (left, right))
                    if visible:
                        axis = [(points[right]["x"] - points[left]["x"]) * label["width"],
                                (points[right]["y"] - points[left]["y"]) * label["height"]]
                        if math.hypot(*axis) > 1e-8:
                            error = angular_error_degrees(target["axis_xy"], axis)
                            row.update(status="scored", angular_error_degrees=error,
                                       passes=error <= target["angular_tolerance_degrees"], prediction_axis_xy=axis)
                        else:
                            row["status"] = "degenerate_prediction"
            rows.append(row)
    errors = [row["angular_error_degrees"] for row in rows if row["status"] == "scored"]
    return {"observations": rows, "summary": {
        "pair_slots": len(rows), "source_observable_pairs": sum(row["source_observable"] for row in rows),
        "source_unobservable_pairs": sum(not row["source_observable"] for row in rows),
        "scored_pairs": len(errors), "missing_or_degenerate": sum(row["status"] in {"prediction_missing", "degenerate_prediction"} for row in rows),
        "within_tolerance": sum(row["passes"] for row in rows),
        "mean_angular_error_degrees": sum(errors) / len(errors) if errors else None,
        "max_angular_error_degrees": max(errors) if errors else None,
        "reassigned_source_unobservable": sum(row["reassigned"] and not row["source_observable"] for row in rows)}}


def run(candidate, out, nested=False):
    labels_path = Path("data/pose-identity-labels/annotations.json")
    labels = json.loads(labels_path.read_text())
    for source in labels["sources"]:
        if digest(Path(source["path"])) != source["sha256"]:
            raise ValueError("Frozen source video changed")
    for frame in labels["frames"]:
        if digest(Path(frame["image_path"])) != frame["image_sha256"]:
            raise ValueError("Frozen source image changed")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(labels_path, out / "annotations.json")
    shutil.copyfile(__file__, out / "study.py")
    reports, inputs = {}, {}
    for kind in ("raw", "candidate"):
        evidence = {}
        for clip in sorted({row["clip_id"] for row in labels["frames"]}):
            path = baseline_path(clip) if kind == "raw" else candidate / clip / ("run" if nested else "") / "evidence.json"
            evidence[clip] = json.loads(path.read_text())
            inputs[f"{kind}/{clip}"] = {"path": str(path), "sha256": digest(path)}
        reports[kind] = grade(labels, evidence)
    improvements, regressions = [], []
    for old, new in zip(reports["raw"]["observations"], reports["candidate"]["observations"]):
        if old["passes"] and not new["passes"]:
            regressions.append({"id": old["id"], "pair": old["pair"]})
        if not old["passes"] and new["passes"]:
            improvements.append({"id": old["id"], "pair": old["pair"]})
    write_json(out / "report.json", {"created_at": local_time(), "inputs": inputs, **reports,
               "new_passes": improvements, "regressions": regressions,
               "scope": "Assistant-labelled development pair directions; nearest sample within one source frame. Overlapping windows are not independent trials. Unobservable pairs do not prove identity correctness."})
    print(json.dumps({kind: report["summary"] for kind, report in reports.items()}, indent=2))
    print("new passes", len(improvements), "regressions", len(regressions))
    print(json.dumps(seal_suite(out)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--nested", action="store_true")
    args = parser.parse_args()
    run(args.candidate, args.out, args.nested)
