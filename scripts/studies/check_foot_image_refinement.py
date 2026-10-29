"""Frozen image-direction proposal, unchanged floor checks and fixed proxy assignments."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.foot_body import audit_foot_body
from weft.foot_image_refinement import refine_foot_image
from weft.foot_reference import FOOT_IMAGE_CONTRACT
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import verify_artifacts
from check_foot_reference import pairs_error


ROOT = Path(__file__).resolve().parents[2]
CASES = ("single-leg-squat", "squat", "kettlebell-pushup", "fullbody-squat")


def projected(reference, frame, scale):
    row = next((row for row in reference["frames"] if row["frame"] == frame), None)
    result = {}
    if row:
        for side, foot in row["feet"].items():
            if foot["status"] != "kinematic_reference":
                continue
            anchor = np.array(foot["raw_image_reference"]["points_px"]["ankle"])
            result[side] = [(anchor + np.array(foot["endpoint_geometry"][key])[[0, 2]] * [1, -1] / scale).tolist()
                            for key in ("heel_relative", "foot_index_relative")]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/foot-image-refinement-01")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    sources = {name: ROOT / "artifacts" / name for name in ("foot-reference-01", "foot-body-01", "foot-reference-source-labels-01")}
    verified = {name: verify_artifacts(path) for name, path in sources.items()}
    if out.exists() or any(out.is_relative_to(path) for path in sources.values()):
        raise ValueError("Use a new output outside all frozen source studies")
    out.mkdir(parents=True)
    for path in [*sorted((ROOT / "src/weft").glob("*.py")), Path(__file__),
                 ROOT / "scripts/studies/check_foot_reference.py", ROOT / "tests/test_foot_image_refinement.py"]:
        target = out / "implementation-source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for name in ("foot-image-tests.xml", "foot-image-final-tests.xml"):
        shutil.copyfile(ROOT / ".local" / name, out / name)
    write_json(out / "pre-fit-freeze.json", {"created_at": local_time(), "contract": FOOT_IMAGE_CONTRACT,
        "sources": verified, "proxy_assignment": "Reuse baseline anonymous assignments without rematching",
        "promotion": "No promotion unless full fixed floor gates pass; source proxies are not anatomy truth",
        "paid_requests": 0, "fit_input_excludes": ["source-only labels", "floor scores", "controller outcomes"]})
    started, results, references = time.monotonic(), {}, {}
    for case in CASES:
        folder = out / case
        (folder / "inputs").mkdir(parents=True)
        source = sources["foot-reference-01"] / case
        for original, name in ((source / "inputs/human.json", "human.json"), (source / "foot-reference.json", "foot-reference.json"),
                               (source / "inputs/root-motion.json", "root-motion.json")):
            shutil.copyfile(original, folder / "inputs" / name)
        spec = json.loads((source / "inputs/human.json").read_text())
        original = json.loads((source / "foot-reference.json").read_text())
        before_hash = canonical_hash(original)
        candidate = refine_foot_image(spec, original)
        assert canonical_hash(original) == before_hash
        write_json(folder / "foot-reference.json", candidate)
        audit = audit_foot_body(spec, candidate)
        write_json(folder / "candidate-audit.json", audit)
        baseline = json.loads((sources["foot-body-01"] / case / "candidate-audit.json").read_text())
        assert audit["baseline"] == baseline["baseline"]
        assert len(audit["checks"]) == len(baseline["checks"])
        paired = []
        for before, after in zip(baseline["checks"], audit["checks"]):
            assert (before["time_s"], before["status"]) == (after["time_s"], after["status"])
            if after["status"] == "checked":
                paired.append({"body_reference_index": after["body_reference_index"], "time_s": after["time_s"],
                               "before": before["max_penetration_model_units"], "after": after["max_penetration_model_units"],
                               "passed_before": before["passes_floor_limit"], "passed_after": after["passes_floor_limit"]})
        write_json(folder / "paired-floor-checks.json", paired)
        results[case] = {"refinement": candidate["orientation_refinement"], "floor_before": baseline["summary"],
                         "floor_after": audit["summary"], "interactive_start_available": audit["interactive_start_available"],
                         "new_floor_failures": sum(row["passed_before"] and not row["passed_after"] for row in paired),
                         "resolved_floor_failures": sum(not row["passed_before"] and row["passed_after"] for row in paired),
                         "accepted": False, "promoted": False, "physical_validation": "unverified"}
        references[case] = (original, candidate)
    # Previously exposed proxy labels never enter the proposal or its parameter selection.
    labels = json.loads((sources["foot-reference-source-labels-01"] / "labels.json").read_text())
    baseline_report = json.loads((sources["foot-reference-01"] / "report.json").read_text())
    prior_rows = {(row["case"], row["frame"]): row for row in baseline_report["source_label_comparisons"]}
    comparisons, tiles = [], []
    for label in labels["rows"]:
        case, frame = label["case"], label["frame"]
        original, candidate = references[case]
        root = json.loads((sources["foot-reference-01"] / case / "inputs/root-motion.json").read_text())
        scale = root["scale"]["model_units_per_pixel"]
        before, after = projected(original, frame, scale), projected(candidate, frame, scale)
        assert set(before) == set(after)
        prior = prior_rows[(case, frame)]
        slots = {slot["id"]: slot for slot in label["feet"]}
        for assignment in prior["anonymous_assignments"]:
            name, side = assignment["label_id"], assignment["model_side_hypothesis"]
            before_error = pairs_error(slots[name]["endpoints_px"], before[side])
            after_error = pairs_error(slots[name]["endpoints_px"], after[side])
            np.testing.assert_allclose(before_error, [assignment["endpoint_distance_px"], assignment["axis_error_degrees"]], atol=1e-10)
            comparisons.append({"case": case, "frame": frame, "label_id": name, "fixed_model_side_hypothesis": side,
                                "before_endpoint_px": before_error[0], "after_endpoint_px": after_error[0],
                                "before_axis_degrees": before_error[1], "after_axis_degrees": after_error[1],
                                "label_uncertainty_px": slots[name]["endpoint_uncertainty_px"],
                                "scope": "Reused anonymous silhouette proxies; not anatomical identity or calibrated ground truth"})
        image_path = sources["foot-reference-source-labels-01"] / label["source_png"]
        assert digest(image_path) == label["source_png_sha256"]
        source_image = Image.open(image_path).convert("RGB")
        canvas = Image.new("RGB", (source_image.width, source_image.height + 60), "white")
        canvas.paste(source_image)
        draw = ImageDraw.Draw(canvas)
        for values, color in ((before, "#B000B5"), (after, "#1179BE")):
            for points in values.values():
                draw.line([tuple(point) for point in points], fill=color, width=2)
        for slot in slots.values():
            if slot["status"] == "axis_estimated":
                draw.line([tuple(point) for point in slot["endpoints_px"]], fill="#E78A00", width=2)
        draw.text((6, source_image.height + 5), f"{case} frame {frame}: purple=before, blue=proposal", fill="black")
        draw.text((6, source_image.height + 23), "Orange=silhouette proxy, not anatomy/contact truth", fill="black")
        draw.text((6, source_image.height + 41), "Source-ankle alignment; unchanged uncalibrated scale", fill="black")
        canvas.save(out / case / f"overlay-{frame:04d}.png")
        canvas.thumbnail((640, 520))
        tile = Image.new("RGB", (660, 540), "white")
        tile.paste(canvas, ((660 - canvas.width) // 2, 0))
        tiles.append(tile)
    sheet = Image.new("RGB", (1320, 540 * 4), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2) * 660, (i // 2) * 540))
    sheet.save(out / "comparison-sheet.png")
    write_json(out / "proxy-comparisons.json", {"comparisons": comparisons, "original_labels": labels,
        "fixed_baseline_assignments": True, "unavailable_proxy_slots": sum(slot["status"] != "axis_estimated" for row in labels["rows"] for slot in row["feet"]),
        "label_sha256": digest(sources["foot-reference-source-labels-01"] / "labels.json")})
    write_json(out / "report.json", {"cases": results, "runtime_seconds": time.monotonic() - started,
        "proxy_comparisons": comparisons, "paid_requests": 0, "body_or_floor_changed": False,
        "scope": "Bounded source-image agreement proposal; all original availability and floor rules preserved"})
    print(json.dumps({"out": str(out), "seal": seal_suite(out), "cases": results}, indent=2))


if __name__ == "__main__":
    main()
