"""Freeze calibrated two-view estimates, then compare with a separate published reference."""

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from weft.camera_reference import JOINTS, compare_frame, project_world
from weft.exercise_vision import validate_person_region_evidence
from weft.io import digest, local_time, write_json
from weft.multiview_pose import triangulate_pair


VIEWS = ("left", "mid")
DISPLAY_FRAMES = (0, 60, 120, 180)
# This is a coordinate convention, not a gravity estimate.
CAMERA_TO_NOMINAL = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
EDGES = [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")] + [
    (f"{side}_{a}", f"{side}_{b}") for side in ("left", "right")
    for a, b in (("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"))]


def read(path, maximum=16 << 20):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError(f"Expected bounded regular input: {path}")
    return json.loads(path.read_text())


def seal(root):
    files = {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}
    if len(files) > 128 or sum((root / name).stat().st_size for name in files) > 128 << 20:
        raise ValueError("Comparison exceeded its output budget")
    write_json(root / "manifest.json", {"schema_version": 1, "files": files})
    return digest(root / "manifest.json")


def verify_member(root, name, manifest):
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular sealed member: {name}")
    entry = manifest["files"][name]
    expected = entry["sha256"] if isinstance(entry, dict) else entry
    if digest(path) != expected:
        raise ValueError(f"Changed sealed member: {name}")
    return path


def describe(values):
    array = np.asarray(values, dtype=float)
    return {"count": len(values), "mean": float(array.mean()) if len(values) else None,
            "rmse": float(np.sqrt((array * array).mean())) if len(values) else None,
            "p95": float(np.percentile(array, 95)) if len(values) else None,
            "maximum": float(array.max()) if len(values) else None}


def make_camera(source, view, evidence):
    value = read(source / f"camera-{view}.json")
    intrinsic = np.asarray(value["intrinsic_matrix"], dtype=float).reshape(3, 4)
    extrinsic = np.asarray(value["extrinsic_matrix"], dtype=float).reshape(4, 4)
    if not np.array_equal(intrinsic[:, 3], np.zeros(3)) or not np.array_equal(extrinsic[3], [0, 0, 0, 1]):
        raise ValueError("Unsupported publisher homogeneous camera matrices")
    rotation = extrinsic[:3, :3] @ CAMERA_TO_NOMINAL.T
    translation = extrinsic[:3, 3] * .001
    return {"schema_version": 1, "source_sha256": evidence["source_sha256"],
        **{key: evidence[key] for key in ("width", "height", "fps", "frame_count")},
        "world_axes": "right_handed_z_up", "units": "m", "quality": "reconstructed",
        "provenance": "Publisher pinhole calibration, unchanged six-decimal rotation; mm-to-m units follow published C3D header and shared camera projection convention. "
                      "Nominal x=left-camera right, y=forward, z=up. This authored axis convention is not measured gravity or ground; camera scale/accuracy remain provider estimates.",
        "K": intrinsic[:, :3].tolist(), "radial_k1": 0.,
        "frames": [{"frame": i, "time_s": i / evidence["fps"], "R": rotation.tolist(), "t": translation.tolist()}
                   for i in range(evidence["frame_count"])]}


def overlay(thumbnail, rows, destination, title):
    if thumbnail.is_symlink() or thumbnail.stat().st_size > 4 << 20:
        raise ValueError("Inspection thumbnail exceeds bounds")
    with Image.open(thumbnail) as original:
        if original.size != (960, 540):
            raise ValueError("Expected original-source inspection thumbnail")
        source = original.convert("RGB")
    image = Image.new("RGB", (960, 600), "#f5eadc"); image.paste(source, (0, 60))
    draw = ImageDraw.Draw(image)
    draw.text((12, 8), title, fill="#214f50")
    draw.text((12, 26), "Cyan: published reference projection | Orange: image model estimate", fill="#214f50")
    draw.text((12, 42), "Reference quality is unverified. No fitted alignment or physical-accuracy claim.", fill="#214f50")
    for key, color in (("reference_image_px", "#00d7ff"), ("predicted_image_px", "#ff9e32")):
        points = {name: (row[key][0] / 4, 60 + row[key][1] / 4) for name, row in rows.items() if row[key] is not None}
        for a, b in EDGES:
            if a in points and b in points:
                draw.line([points[a], points[b]], fill=color, width=2)
        for x, y in points.values():
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    image.save(destination)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "poses", "reference-inspection", "media-inspection", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    source, poses, reference, media, out = (p.resolve() for p in
        (args.source, args.poses, args.reference_inspection, args.media_inspection, args.out))
    if out.exists() or any(out == p or out in p.parents or p in out.parents for p in (source, poses, reference, media)):
        raise ValueError("Use a new comparison directory outside every input")
    source_manifest, pose_manifest = read(source / "manifest.json"), read(poses / "manifest.json")
    out.mkdir(parents=True); candidate = out / "candidate"; candidate.mkdir()
    shutil.copyfile(__file__, out / "check_aspset_pose.py")
    inputs, views, indexed = {}, [], {}
    for view in VIEWS:
        for name in (f"camera-{view}.json", f"{view}.mkv"):
            inputs["source/" + name] = digest(verify_member(source, name, source_manifest))
        evidence_path = verify_member(poses, f"{view}/evidence.json", pose_manifest)
        raw_path = verify_member(poses, f"{view}/pose-landmarks.json", pose_manifest)
        clock_path = verify_member(poses, f"{view}/source-clock.json", pose_manifest)
        evidence, raw, clock = read(evidence_path), read(raw_path), read(clock_path)
        if (evidence["source_sha256"] != inputs[f"source/{view}.mkv"]
                or (evidence["width"], evidence["height"], evidence["fps"], evidence["frame_count"], evidence["decoded_frames"]) != (3840, 2160, 50., 188, 188)
                or clock["source_sha256"] != evidence["source_sha256"] or clock["decoded_frames"] != 188
                or len(clock["frames"]) != 188
                or clock["decoded_to_eof"] is not True or clock["native_minus_decoder_seconds"] != 0):
            raise ValueError("Source evidence and qualified complete clock disagree")
        if any(row["frame"] != i or abs(row["decoder_time_s"] - i / 50) > 1e-9 for i, row in enumerate(clock["frames"])):
            raise ValueError("Packaged source-frame alignment changed")
        region = evidence.get("person_region", {}).get("box_xyxy")
        validate_person_region_evidence(evidence, raw, region)
        camera = make_camera(source, view, evidence)
        item = {"name": view, "camera": camera, "evidence": evidence,
                "clock_provenance": "ASPset matching subject/clip camera members, 188 complete frames at 50Hz; exact packaged indices and zero native/decoder offset. Physical synchronization remains publisher-supplied, not independently measured."}
        views.append(item); indexed[view] = {row["frame"]: row for row in evidence["frames"]}
        write_json(candidate / f"view-{view}.json", item)
        inputs[f"poses/{view}/evidence.json"] = digest(evidence_path)
        inputs[f"poses/{view}/source-clock.json"] = digest(clock_path)
    write_json(candidate / "plan.json", {"created_at": local_time(), "inputs": inputs,
        "method": "Existing calibrated-two-ray-pose-v1; same-label image observations only",
        "original_world_to_nominal_world": {"rotation": CAMERA_TO_NOMINAL.tolist(), "scale": .001,
            "input_units": "publisher_mm", "output_units": "m", "gravity_alignment": "unknown", "floor": "unknown"},
        "reference_pose_inputs": [], "maximum_frames": 188, "maximum_views": 2, "maximum_joints": 33,
        "maximum_seconds": 30, "maximum_bytes": 128 << 20, "new_model_calls": 0, "physics_calls": 0, "paid_calls": 0})
    result = triangulate_pair(*views)
    write_json(candidate / "triangulation.json", result)
    candidate_seal = seal(candidate)
    if time.monotonic() - started > 15:
        raise TimeoutError("Candidate construction exceeded its half-budget")

    # Reference coordinates become accessible only after the candidate is frozen.
    metadata = read(reference / "report.json")
    if (metadata["units"], metadata["point_rate"], metadata["frame_count"], metadata["point_count"],
            metadata["first_frame"], metadata["last_frame"]) != ("mm", 50., 188, 17, 1, 188):
        raise ValueError("Reference metadata differs from the declared comparison protocol")
    if metadata["source_sha256"] != digest(verify_member(source, "reference.c3d", source_manifest)):
        raise ValueError("Reference reader provenance differs from the complete source member")
    array_path = reference / "published-reference.npz"
    if array_path.is_symlink() or array_path.stat().st_size > 1 << 20:
        raise ValueError("Reference arrays exceed bounds")
    with zipfile.ZipFile(array_path) as archive:
        if (set(archive.namelist()) != {"points.npy", "frame_ids.npy"}
                or len(archive.infolist()) != 2 or sum(item.file_size for item in archive.infolist()) > 1 << 20):
            raise ValueError("Reference array members exceed bounds")
    with np.load(array_path, allow_pickle=False) as archive:
        points = archive["points"].copy(); frame_ids = archive["frame_ids"].copy()
    if points.shape != (188, 17, 5) or not np.isfinite(points).all() or not np.array_equal(frame_ids, np.arange(1, 189)):
        raise ValueError("Reference arrays do not preserve their complete original frame IDs")
    labels = metadata["labels"]
    if len(labels) != 17 or len(set(labels)) != 17 or not set(JOINTS) <= set(labels):
        raise ValueError("Reference must identify every compared anatomical joint")
    names = {name: labels.index(name) for name in JOINTS}
    reference_world = points[:, :, :3].astype(float) * .001 @ CAMERA_TO_NOMINAL.T
    rows, summary, images = {}, {}, []
    for view, item in zip(VIEWS, views, strict=True):
        camera = item["camera"]; projection = camera["frames"][0]
        comparisons, errors2d, errors3d, counts2d, counts3d = [], [], [], Counter(), Counter()
        for i in range(188):
            xyz, pixels, valid = project_world(reference_world[i], projection["R"], projection["t"], camera["K"])
            reference_xyz = {name: xyz[j] if valid[j] and points[i, j, 3] >= 0 else np.full(3, np.nan) for name, j in names.items()}
            reference_pixels = {name: pixels[j] if valid[j] and points[i, j, 3] >= 0 else np.full(2, np.nan) for name, j in names.items()}
            compared = compare_frame(indexed[view].get(i), reference_xyz, reference_pixels, width=3840, height=2160)
            comparisons.append({"source_frame": i, "reference_frame": int(frame_ids[i]), "time_s": i / 50, "joints": compared})
            for value in compared.values():
                counts2d[value["image_status"]] += 1; counts3d[value["relative_3d_status"]] += 1
                if value["image_distance_px"] is not None: errors2d.append(value["image_distance_px"])
                if value["relative_3d_distance_mm"] is not None: errors3d.append(value["relative_3d_distance_mm"])
            if i in DISPLAY_FRAMES:
                name = f"{view}-{i:04d}.png"
                thumbnail = media / view / f"frame-{i:04d}.jpg"
                overlay(thumbnail, compared, out / name, f"{view} view / source frame {i} / published C3D frame {i + 1}")
                images.append(name)
        rows[view] = comparisons
        summary[view] = {"image_statuses": dict(counts2d), "relative_3d_statuses": dict(counts3d),
                         "image_error_px": describe(errors2d), "monocular_root_relative_error_mm": describe(errors3d)}
    stereo_rows, stereo_errors, stereo_statuses = [], [], Counter()
    for i, frame in enumerate(result["frames"]):
        joints = {}
        for name, index in JOINTS.items():
            value = frame["joints"][index]; status = value["status"]; error = None
            if points[i, names[name], 3] < 0: status = "reference_unavailable"
            elif status == "estimated":
                error = float(np.linalg.norm(np.asarray(value["world_point_m"]) - reference_world[i, names[name]]) * 1000)
                stereo_errors.append(error)
            stereo_statuses[status] += 1; joints[name] = {"status": status, "absolute_distance_mm": error}
        stereo_rows.append({"source_frame": i, "reference_frame": i + 1, "joints": joints})
    summary["two_view"] = {"statuses": dict(stereo_statuses), "absolute_reference_error_mm": describe(stereo_errors),
                           "all_33_joint_statuses": result["joint_status_counts"]}
    write_json(out / "comparisons.json", {"monocular": rows, "two_view": stereo_rows})
    report = {"created_at": local_time(), "candidate_manifest_sha256": candidate_seal,
        "source_manifest_sha256": digest(source / "manifest.json"), "pose_manifest_sha256": digest(poses / "manifest.json"),
        "reference_array_sha256": digest(array_path), "reference_metadata_sha256": digest(reference / "report.json"),
        "reference_quality": "Published 3D keypoint annotations; capture accuracy/method unqualified here; zero residual/camera flags do not establish measured marker truth",
        "protocol": "Descriptive exposed-development comparison; exact packaged indices, C3D1→video0; no alignment, timing, scale, identity fitting or post-hoc success threshold",
        "counts": {"source_frames_per_view": 188, "sampled_frames_per_view": 47, "common_named_joints": 12},
        "summary": summary, "seconds": time.monotonic() - started, "new_model_calls": 0, "physics_calls": 0, "paid_calls": 0,
        "limits": ["Not independently measured physical accuracy", "Nominal up is not calibrated gravity; floor/contact remain unknown",
                   "Manual person-region input is not automatic detection", "Same-name anatomy correspondence is not independently verified",
                   "Two-view reconstruction is additional input, not single-video recovery"]}
    if report["seconds"] > 30: raise TimeoutError("Comparison deadline exceeded")
    write_json(out / "report.json", report)
    body = "".join(f'<figure><img src="{name}" width="960" height="600" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>' for name in images)
    page = ('<!doctype html><meta charset="utf-8"><title>Weft: calibrated pose comparison</title>'
        '<style>body{font:16px system-ui;margin:24px;max-width:1100px;background:#f5eadc;color:#214f50}img{max-width:100%;height:auto}pre{white-space:pre-wrap}figure{margin:24px 0}</style>'
        '<h1>Pose coverage is not 3D accuracy</h1><p>Two supplied camera views and explicit person regions. '
        'Published reference annotations are not verified physical truth. Ground and contact remain unknown.</p>'
        '<p><a href="report.json">Summary and limits</a> · <a href="comparisons.json">Every comparison</a> · '
        '<a href="candidate/triangulation.json">Frozen 3D estimates</a></p><pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre>' + body)
    (out / "index.html").write_text(page)
    manifest_sha = seal(out)
    print(json.dumps({"out": str(out), "manifest_sha256": manifest_sha, "report": report}))


if __name__ == "__main__":
    main()
