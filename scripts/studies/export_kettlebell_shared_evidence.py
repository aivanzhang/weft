"""Export saved three-object proposals and masks without another model request."""

import argparse
from collections import Counter
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.segmentation_masks import CONTRACT as MASK_CONTRACT, export_masks
from weft.segmentation_observations import export_tracking_evidence
from weft.world_evidence import load_world_evidence


ROOT = Path(__file__).resolve().parents[2]
TRACKING = ROOT / "artifacts/kettlebell-shared-tracking-01/tracking"
PREPARATION = ROOT / "artifacts/kettlebell-tracking-input-02"
ORIGINAL_SHA = "4664b16bab5fbefd66f3dc9116ced8dfeb6f3b979635403b715bbc5848b0afbc"
DERIVATIVE_SHA = "34e74d9ab854a474ba647395f232e61dfde5cfa166bb449fd3057084c6dd5416"
LABELS = {"1": "person", "2": "working kettlebell", "3": "idle kettlebell"}
SELECTED = (0, 7, 8, 16, 64, 128, 192, 256)
MAX_OUTPUT_BYTES = 192_000_000


def read(path):
    return json.loads(Path(path).read_text())


def stats(values):
    if not values:
        return {"count": 0, "minimum": None, "median": None, "p95": None, "maximum": None}
    return {"count": len(values), "minimum": min(values), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "maximum": max(values)}


def intervals(frames):
    result = []
    for frame in frames:
        if result and frame == result[-1][1] + 1:
            result[-1][1] = frame
        else:
            result.append([frame, frame])
    return result


def verify_slots(out):
    evidence = read(out / "evidence/segmentation-evidence.json")
    shared_raw = (out / "evidence/scene-observations.json").read_bytes()
    shared_sha = sha256(shared_raw).hexdigest()
    shared = json.loads(shared_raw)
    registry = load_world_evidence({shared_sha: shared_raw})
    write_json(out / "registry-inspection.json", registry.summary())
    if (evidence["source"]["original_source_sha256"] != ORIGINAL_SHA
            or evidence["source"]["submitted_video_sha256"] != DERIVATIVE_SHA
            or shared["source"]["sha256"] != DERIVATIVE_SHA
            or shared["provenance"]["original_source_sha256"] != ORIGINAL_SHA):
        raise ValueError("Original versus submitted source identities were changed")
    if [obj["id"] for obj in evidence["objects"]] != ["sam-1", "sam-2", "sam-3"]:
        raise ValueError("Model object identities changed")
    if len(evidence["frames"]) != 257 or len(evidence["observations"]) != 771 or len(shared["observations"]) != 771:
        raise ValueError("All 257 x 3 original slots must remain")
    raw = read(TRACKING / "call/segmentation.json")
    mapping = read(out / "inputs/preparation/frame-mapping-and-pixels.json")
    shared_by_id = {row["id"]: row for row in shared["evidence"]}
    rows = {(row["frame"], row["raw_model_id"]): row for row in evidence["observations"]}
    for frame in range(257):
        original = mapping["frames"][frame]
        if original["source_frame"] != frame or original["relative_frame"] != frame:
            raise ValueError("Source mapping changed original frame indices")
        for raw_row in raw["frames"][frame]["objects"]:
            row = rows[(frame, raw_row["object_id"])]
            if row["raw_model_output"] != raw_row or shared_by_id[row["id"]]["payload"] != row:
                raise ValueError("Export changed a raw proposal or lost its evidence payload")
            if row["original_source"] != {"sha256": ORIGINAL_SHA, "frame": frame, "time_s": original["source_pts"] / 1000}:
                raise ValueError("Original native source clock was rewritten")
            if not math.isclose(row["time_s"], original["relative_pts"] / 1000, rel_tol=0, abs_tol=1e-9):
                raise ValueError("Submitted decoder clock changed")
            expected_phase = "pre_initialization" if frame < 8 else "initialization" if frame == 8 else "post_initialization"
            if row["phase"] != expected_phase or (frame < 8 and row["proposal_status"] != "missing"):
                raise ValueError("Pre-initialization slots were omitted or fabricated")
    return evidence, shared_sha


def inspect_masks(out, evidence):
    decoded = out / "masks"
    document = read(decoded / "mask-evidence.json")
    if (document["original_evidence_sha256"] != canonical_hash(evidence)
            or document["identity_groups"] != evidence["identity_groups"]
            or len(document["frames"]) != 257 or len(document["observations"]) != 771):
        raise ValueError("Decoded masks changed input evidence or source coverage")
    raw_by_id = {row["id"]: row for row in evidence["observations"]}
    maximum_centroid_error = 0.
    verified_pngs = 0
    for row in document["observations"]:
        original = raw_by_id[row["id"]]
        if row["raw_observation_sha256"] != canonical_hash(original):
            raise ValueError("Decoded mask input hash changed")
        for key in ("frame", "time_s", "object_id", "raw_model_id", "phase", "proposal_status", "identity", "confidence", "original_source"):
            if row[key] != original[key]:
                raise ValueError(f"Decoded mask changed {key}")
        if not row["raster_path"]:
            if row["silhouette"] is not None:
                raise ValueError("Missing masks cannot fabricate silhouettes")
            continue
        with Image.open(decoded / row["raster_path"]) as image:
            raster = np.array(image)
        if (raster.dtype != np.uint8 or list(raster.shape) != row["raster_dimensions"]
                or np.any(raster > 1) or sha256(raster.tobytes()).hexdigest() != row["raster_sha256"]):
            raise ValueError("Saved raster bytes/dimensions changed")
        ys, xs = np.nonzero(raster)
        if len(xs) != row["silhouette"]["foreground_pixels"]:
            raise ValueError("Silhouette foreground count changed")
        if len(xs):
            left, top, right, bottom = row["box"]["box_xyxy_halfopen"]
            expected = [left + (float(xs.mean()) + .5) * (right - left) / raster.shape[1],
                        top + (float(ys.mean()) + .5) * (bottom - top) / raster.shape[0]]
            error = math.dist(expected, row["silhouette"]["centroid_px"])
            if error > 1e-8:
                raise ValueError("Independent cell-center centroid calculation differs")
            maximum_centroid_error = max(maximum_centroid_error, error)
        verified_pngs += 1
    reports = {}
    for identifier, label in LABELS.items():
        source_rows = [row for row in evidence["observations"] if row["raw_model_id"] == identifier]
        mask_rows = [row for row in document["observations"] if row["raw_model_id"] == identifier]
        centers = [(row["frame"], row["silhouette"]["centroid_px"]) for row in mask_rows
                   if row["silhouette"] and row["silhouette"]["centroid_px"] is not None]
        steps = [{"from_frame": a[0], "to_frame": b[0], "distance_px": math.dist(a[1], b[1])}
                 for a, b in zip(centers, centers[1:]) if b[0] == a[0] + 1]
        reports[identifier] = {
            "object_id": "sam-" + identifier, "authored_prompt_label": label, "source_slots": len(source_rows),
            "proposal_counts": dict(Counter(row["proposal_status"] for row in source_rows)),
            "phase_counts": dict(Counter(row["phase"] for row in source_rows)),
            "mask_counts": dict(Counter(row["mask_status"] for row in mask_rows)),
            "pre_initialization_frames": intervals([row["frame"] for row in source_rows if row["phase"] == "pre_initialization"]),
            "post_initialization_missing_frames": intervals([row["frame"] for row in source_rows if row["phase"] == "post_initialization" and row["proposal_status"] == "missing"]),
            "unresolved_association_slots": sum(row["identity"]["status"] == "unresolved_association" for row in source_rows),
            "gap_seen_slots": sum(row["identity"]["gap_seen"] for row in source_rows),
            "mask_confidence": stats([row["confidence"]["mask"] for row in source_rows if row["confidence"]["mask"] is not None]),
            "silhouette_area_px2": stats([row["silhouette"]["mapped_area_px2"] for row in mask_rows if row["silhouette"]]),
            "adjacent_centroid_step_px": stats([row["distance_px"] for row in steps]),
            "largest_adjacent_centroid_steps": sorted(steps, key=lambda row: row["distance_px"], reverse=True)[:5],
            "physical_identity_verified": False,
        }
    return document, {"verified_raster_pngs": verified_pngs, "maximum_independent_centroid_difference_px": maximum_centroid_error,
                      "entities": reports, "identity_groups": evidence["identity_groups"],
                      "proximity_events": len(evidence["proximity_events"]),
                      "scope": "Model coverage/consistency diagnostics, not independent segmentation or physical accuracy"}


def panels(out, document):
    paths = []
    for frame in SELECTED:
        image = Image.open(out / "masks" / document["frames"][frame]["preview"]).convert("RGB")
        overlay = image.convert("RGBA")
        for index in document["frames"][frame]["observation_indices"]:
            row = document["observations"][index]
            if row["raster_path"]:
                left, top, right, bottom = row["box"]["box_xyxy_halfopen"]
                with Image.open(out / "masks" / row["raster_path"]) as mask:
                    patch = mask.convert("RGBA").resize((right - left, bottom - top), Image.Resampling.NEAREST)
                overlay.alpha_composite(patch, (left, top))
                draw = ImageDraw.Draw(overlay)
                draw.rectangle((left, top, right - 1, bottom - 1), outline=tuple(row["color"]), width=1)
                draw.text((left, max(0, top - 12)), row["raw_model_id"], fill="white", stroke_width=1, stroke_fill="black")
        panel = Image.new("RGB", (1280, 528), "white")
        panel.paste(image, (0, 48)); panel.paste(overlay.convert("RGB"), (640, 48))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 5), f"Frame {frame}: original decoder preview | proposed silhouettes, not verified identity or contact", fill="black")
        draw.text((8, 23), "Authored labels: 1 person; 2 working kettlebell; 3 idle kettlebell. Frames 0-7 precede initialization.", fill="black")
        name = f"panel-{frame:04d}.png"; panel.save(out / name); paths.append(name)
    return paths


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve()
    if out.exists():
        raise FileExistsError("Use a new export directory")
    started = time.monotonic()
    tracking_seal, preparation_seal = verify_artifacts(TRACKING), verify_artifacts(PREPARATION)
    if digest(TRACKING / "source.mp4") != DERIVATIVE_SHA:
        raise ValueError("Tracking source differs from the prepared derivative")
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out / "export_kettlebell_shared_evidence.py")
    shutil.copytree(PREPARATION, out / "inputs/preparation")
    if verify_artifacts(out / "inputs/preparation") != preparation_seal:
        raise ValueError("Copied mapping evidence differs from the original seal")
    copy_names = ("call/response.sse", "call/segmentation.json", "call/call.json", "report.json", "video.json")
    tracking_copies = {}
    for name in copy_names:
        target = out / "inputs/tracking" / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TRACKING / name, target); tracking_copies[name] = digest(target)
    shutil.copyfile(TRACKING / "manifest.json", out / "inputs/tracking/original-manifest-reference.json")
    write_json(out / "pre-run.json", {"created_at": local_time(), "command": [sys.executable, *sys.argv],
        "tracking_directory": str(TRACKING), "tracking_verification": tracking_seal,
        "preparation_directory": str(PREPARATION), "preparation_verification": preparation_seal,
        "copied_tracking_subset_hashes": tracking_copies, "source_mapping_copied_whole": True,
        "python": sys.version, "opencv": cv2.__version__, "numpy": np.__version__,
        "mask_codec": importlib.metadata.version("meta-sam-parser"), "mask_contract": MASK_CONTRACT,
        "maximum_output_bytes": MAX_OUTPUT_BYTES, "maximum_mask_workers": 1,
        "selected_panel_frames": list(SELECTED), "new_api_calls": 0, "new_model_inference": 0,
        "source_identity_rule": "Submitted derivative identity is retained; original native source is linked by sealed per-frame mapping, never relabeled equal"})
    write_json(out / "authored-labels.json", {"labels_by_raw_model_id": LABELS,
        "scope": "Authored initialization labels, not verified semantic identity, physical role or contact"})
    try:
        evidence_report = export_tracking_evidence(TRACKING, out / "evidence", source_mapping_bundle=out / "inputs/preparation")
        evidence, shared_sha = verify_slots(out)
        mask_report = export_masks(out / "evidence", TRACKING / "source.mp4", out / "masks")
        document, quality = inspect_masks(out, evidence)
        write_json(out / "tracking-quality.json", quality)
        images = panels(out, document)
        if verify_artifacts(TRACKING) != tracking_seal or verify_artifacts(PREPARATION) != preparation_seal:
            raise ValueError("Immutable tracking or media preparation changed")
        total = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
        if total > MAX_OUTPUT_BYTES:
            raise ValueError("Evidence export exceeded its byte bound")
        report = {"created_at": local_time(), "status": "evidence_and_masks_only", "frames": 257, "entities": 3,
            "source_slots": 771, "source_sha256": ORIGINAL_SHA, "submitted_derivative_sha256": DERIVATIVE_SHA,
            "scene_observations": {"path": "evidence/scene-observations.json", "sha256": shared_sha,
                                   "source_sha256": DERIVATIVE_SHA, "original_source_link_is_provenance_not_identity_rewrite": True},
            "segmentation_evidence": {"path": "evidence/segmentation-evidence.json", "sha256": digest(out / "evidence/segmentation-evidence.json")},
            "mask_evidence": {"path": "masks/mask-evidence.json", "sha256": digest(out / "masks/mask-evidence.json")},
            "evidence_export": evidence_report, "mask_export": mask_report,
            "verified_pngs": quality["verified_raster_pngs"],
            "maximum_independent_centroid_difference_px": quality["maximum_independent_centroid_difference_px"],
            "original_native_clock_pts": [3, 8536], "submitted_decoder_clock_pts": [0, 8533], "time_base": "1/1000",
            "explicit_decoder_origin_offset_pts": 3, "mapping_file": "inputs/preparation/frame-mapping-and-pixels.json",
            "panels": images, "bytes_before_report_and_seal": total, "seconds": time.monotonic() - started,
            "new_api_calls": 0, "new_model_inference": 0, "new_external_cost_usd": 0,
            "physics_ready": False, "physical_identity_verified": False, "failures": [],
            "limitations": ["Authored person/working/idle labels remain separate from provider IDs.",
                            "Every source slot remains, including all 24 pre-initialization slots and any later gaps.",
                            "No independent segmentation labels, metric depth, mass, contact, camera or physical geometry are recovered.",
                            "Proximity groups are conservative ambiguity flags, not measured contact or identity switches.",
                            "Pose/source and derivative/SAM evidence require a separately validated equivalence bridge; their hashes are not equal."]}
        write_json(out / "report.json", report)
        (out / "index.html").write_text("<!doctype html><title>Kettlebell evidence inspection</title><h1>Saved model proposals, not verified physics</h1>"
            + '<p><a href="tracking-quality.json">Full-slot quality report</a> | <a href="masks/index.html">All-frame mask viewer</a></p>'
            + "".join(f'<p><img width="1280" src="{name}" alt="Source and model-mask diagnostic"/></p>' for name in images))
        print(json.dumps({"report": report, "verification": seal_suite(out)}, indent=2))
    except Exception as error:
        if not (out / "manifest.json").exists():
            write_json(out / "failure.json", {"created_at": local_time(), "type": type(error).__name__, "reason": str(error),
                                              "new_api_calls": 0, "preserved_directory": str(out)})
        raise


if __name__ == "__main__":
    main()
