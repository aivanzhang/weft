"""One frozen saved-mask study; conditional image continuity, never physical identity."""

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


GEOMETRY_FRAMES = (8, 18, 28, 56, 80, 116, 152)
GEOMETRY_OBJECTS = ("sam-2", "sam-3")
LIMITS = {"wall_seconds": 30., "cpu_seconds": 20, "rss_bytes": 512 << 20,
          "output_bytes": 32 << 20, "cache_bytes": 4 << 20, "cache_entries": 8}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, document):
    raw = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    return _sha(raw)


def summarize_associations(result, masks):
    original = {row["id"]: row for row in masks["observations"]}
    by_raw, anchors, swaps, indexed = {}, {}, [], {}
    for frame in result["frames"]:
        for row in frame["slots"]:
            indexed[frame["frame"], row["raw_object_id"]] = row
            by_raw.setdefault(row["raw_object_id"], []).append((frame["frame"], row))
            if row["anchor_id"]:
                anchors.setdefault(row["anchor_id"], []).append((frame["frame"], row))
            if row["previous_id"]:
                previous = original[row["previous_id"]]
                current = original[row["id"]]
                if previous["raw_model_id"] != current["raw_model_id"]:
                    swaps.append({"frame": frame["frame"], "previous_id": row["previous_id"], "current_id": row["id"],
                        "previous_provider_id": previous["raw_model_id"], "current_provider_id": current["raw_model_id"],
                        "anchor_id": row["anchor_id"], "scope": "Accepted local image association across provider labels, not a verified identity swap"})
    summary = {}
    for name, rows in by_raw.items():
        declared = [(frame, row) for frame, row in rows if row["status"] == "declared_initialization"]
        anchor = declared[0][1]["anchor_id"] if declared else None
        lineage = anchors.get(anchor, [])
        last = lineage[-1][0] if lineage else None
        failure = None
        if last is not None and last < result["frames"][-1]["frame"]:
            following = result["frames"][last + 1]
            raw_row = indexed[last + 1, lineage[-1][1]["raw_object_id"]]
            failure = {"frame": last + 1, "raw_slot_status": raw_row["status"],
                "outgoing_pairs": [pair for pair in following["pairs"] if pair["previous_id"] == lineage[-1][1]["id"]]}
        summary[name] = {"anchor_id": anchor, "initialization_frame": declared[0][0] if declared else None,
            "anchored_continuation_slots": len(lineage), "last_anchored_frame": last,
            "first_anchor_failure": failure, "raw_slot_status_counts": dict(Counter(row["status"] for _, row in rows)),
            "anchor_provider_ids": sorted({original[row["id"]]["raw_model_id"] for _, row in lineage})}
    eligibility = []
    for frame in GEOMETRY_FRAMES:
        objects = {}
        for name in GEOMETRY_OBJECTS:
            row = indexed.get((frame, name))
            target_anchor = summary.get(name, {}).get("anchor_id")
            # Follow the image hypothesis, not the mutable provider label's current slot.
            lineage = [(current, item) for current, item in anchors.get(target_anchor, []) if current == frame]
            item = lineage[0][1] if lineage else None
            objects[name] = {"conditional_image_track_eligible": item is not None,
                "anchor_id": target_anchor, "matched_source_evidence_id": item["id"] if item else None,
                "matched_raw_object_id": item["raw_object_id"] if item else None,
                "raw_slot_status": row["status"] if row else "missing_source_slot",
                "physical_identity_verified": False, "geometry_admitted": False, "contact": "unknown"}
        eligibility.append({"frame": frame, "objects": objects,
            "both_conditional_image_tracks_eligible": all(row["conditional_image_track_eligible"] for row in objects.values())})
    return {"per_object": summary, "provider_id_changes": swaps, "geometry_frame_eligibility": eligibility,
        "raw_identity_reclassified": False, "physical_identity_verified": False, "contact": "unknown",
        "scope": "Anchored image-overlap hypotheses only; neither mask derivation nor semantic/physical identity is independently established."}


def prepare(source, out):
    from weft.mask_track_hypotheses import CONTRACT
    from weft.shared_world_packaging import read_package_file, read_package_manifest

    source = source.resolve(strict=True)
    if out.exists():
        raise FileExistsError("Use a new immutable study directory")
    root_raw, root_pins = read_package_manifest(source)
    consumed = {"manifest.json": _sha(root_raw)}
    nested_manifests = {}
    for prefix in ("evidence", "masks"):
        name = prefix + "/manifest.json"
        raw = read_package_file(source, name, expected_sha256=root_pins[name], max_bytes=8 << 20)
        consumed[name] = _sha(raw)
        nested_manifests[prefix] = json.loads(raw)["files"]
    source_name, mask_name = "evidence/scene-observations.json", "masks/mask-evidence.json"
    documents = {}
    for name in (source_name, mask_name):
        prefix, relative = name.split("/", 1)
        if root_pins[name] != nested_manifests[prefix].get(relative):
            raise ValueError("Evidence document lacks matching nested manifest custody")
        documents[name] = read_package_file(source, name, expected_sha256=root_pins[name], max_bytes=16 << 20)
        consumed[name] = root_pins[name]
    masks = json.loads(documents[mask_name])
    nested = nested_manifests["masks"]
    pngs = set()
    for row in masks["observations"]:
        if row.get("raster_path") is None:
            continue
        relative = row["raster_path"]
        name = "masks/" + relative
        if not relative.startswith("masks/") or not relative.endswith(".png") or root_pins.get(name) != nested.get(relative) or name not in root_pins:
            raise ValueError("Saved PNG lacks exact root and nested manifest custody")
        consumed[name] = root_pins[name]
        pngs.add(name)
    if len(masks["frames"]) != 257 or len(masks["observations"]) != 771 or len(pngs) != 747:
        raise ValueError("Frozen kettlebell source coverage differs; do not substitute or tune")
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    runtime = {}
    for path in (repository / "src/weft").glob("*.py"):
        raw = path.read_bytes()
        name = "src/weft/" + path.name
        target = out / "runtime" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        runtime[name] = _sha(raw)
    for name in ("pyproject.toml", "requirements.lock", "requirements-lab.lock"):
        raw = (repository / name).read_bytes()
        (out / "runtime" / name).write_bytes(raw)
        runtime[name] = _sha(raw)
    worker = Path(__file__).read_bytes()
    (out / "worker.py").write_bytes(worker)
    _write(out / "plan.json", {"schema_version": 1, "source_directory": str(source), "contract": CONTRACT,
        "limits": LIMITS, "consumed_source_members": consumed, "png_count": len(pngs),
        "runtime_files": runtime, "worker_sha256": _sha(worker), "geometry_frames": list(GEOMETRY_FRAMES),
        "geometry_objects": list(GEOMETRY_OBJECTS), "partition": "previously_exposed_development",
        "hypothesis": "Fixed adjacent mutual mask overlap may support conditional image continuity where provider labels remain unverified; stop each anchor permanently at its first unsupported edge.",
        "derivation_scope": "Trust pinned saved producer PNG receipts; independently verify exact PNG bytes, binary pixels and metadata, not encoded SAM masks or physical identity.",
        "work": {"video_decodes": 0, "inference_calls": 0, "physics_steps": 0, "paid_calls": 0}})


def worker(out):
    started = time.monotonic()
    plan = json.loads((out / "plan.json").read_bytes())
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 << 20, 32 << 20))
    def check(*_):
        if time.monotonic() - started > 28:
            raise TimeoutError("Saved-mask study wall bound")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > LIMITS["rss_bytes"]:
            raise MemoryError("Saved-mask study RSS bound")
    signal.signal(signal.SIGALRM, check)
    signal.setitimer(signal.ITIMER_REAL, .25, .25)
    for name, expected in plan["runtime_files"].items():
        if _sha((out / "runtime" / name).read_bytes()) != expected:
            raise ValueError("Frozen study runtime changed")
    sys.path.insert(0, str(out / "runtime/src"))
    from weft.mask_track_hypotheses import CONTRACT, associate_source_masks
    from weft.shared_world_packaging import read_package_file
    from weft.silhouette_geometry import _read_raster
    if CONTRACT != plan["contract"] or _sha(Path(__file__).read_bytes()) != plan["worker_sha256"]:
        raise ValueError("Frozen contract or worker changed")
    source, pins = Path(plan["source_directory"]), plan["consumed_source_members"]
    def read(name):
        check()
        return read_package_file(source, name, expected_sha256=pins[name], max_bytes=16 << 20)
    source_raw, masks_raw = read("evidence/scene-observations.json"), read("masks/mask-evidence.json")
    for name in ("manifest.json", "evidence/manifest.json", "masks/manifest.json"):
        read(name)
    cache, cache_size, loaded = OrderedDict(), 0, []
    def load(row):
        nonlocal cache_size
        name = "masks/" + row["raster_path"]
        key = (name, row["raster_sha256"], tuple(row["raster_dimensions"]), row["silhouette"]["foreground_pixels"])
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        png = read(name)
        shape, binary, png_sha = _read_raster(png, row)
        loaded.append({"path": name, "png_sha256": png_sha, "raster_sha256": _sha(binary),
                       "shape": list(shape), "png_bytes": len(png), "binary_bytes": len(binary)})
        while cache and (len(cache) >= LIMITS["cache_entries"] or cache_size + len(binary) > LIMITS["cache_bytes"]):
            cache_size -= len(cache.popitem(last=False)[1])
        if len(binary) <= LIMITS["cache_bytes"]:
            cache[key] = binary
            cache_size += len(binary)
        return binary
    report = associate_source_masks(source_raw, masks_raw, raster_loader=load)
    summary = summarize_associations(report, json.loads(masks_raw))
    for name in pins:
        read(name)
    if len(report["frames"]) != 257 or sum(len(frame["slots"]) for frame in report["frames"]) != 771:
        raise ValueError("Association dropped source slots")
    if len(loaded) != 747 or report["raw_identity_reclassified"] or report["physical_identity_verified"]:
        raise ValueError("Study changed its fixed source or identity scope")
    check()
    _write(out / "associations.json", report)
    _write(out / "loaded-rasters.json", loaded)
    _write(out / "summary.json", summary)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    _write(out / "verification.json", {"source_members_before_after": len(pins), "all_source_hashes_unchanged": True,
        "frames": 257, "slots": 771, "pngs": len(loaded), "work": report["work"],
        "wall_seconds": time.monotonic() - started, "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "peak_rss_bytes": usage.ru_maxrss, "provider_id_changes": len(summary["provider_id_changes"]),
        "derivation_scope": plan["derivation_scope"], "physical_identity_verified": False, "contact": "unknown",
        "video_decodes": 0, "inference_calls": 0, "physics_steps": 0, "paid_calls": 0})
    signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    out = args.out.absolute()
    if args.worker:
        try:
            worker(out)
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            _write(out / "failure.json", {"type": type(error).__name__, "message": str(error), "partial_evidence_preserved": True})
            raise
        return
    if args.source is None:
        parser.error("--source is required for a new study")
    started = time.monotonic()
    prepare(args.source, out)
    environment = {"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "TZ": "America/Los_Angeles"}
    with (out / "stdout.log").open("xb") as stdout, (out / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen([sys.executable, "-I", "-B", str(out / "worker.py"), "--worker", "--out", str(out)],
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=environment, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=max(.1, LIMITS["wall_seconds"] - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
    _write(out / "process.json", {"pid": process.pid, "returncode": process.returncode, "reaped": True,
        "timed_out": timed_out, "wall_seconds": time.monotonic() - started})
    size = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    if size > LIMITS["output_bytes"]:
        _write(out / "output-bound-failure.json", {"actual_bytes": size, "limit_bytes": LIMITS["output_bytes"]})
    members = {path.relative_to(out).as_posix(): _sha(path.read_bytes()) for path in out.rglob("*") if path.is_file()}
    _write(out / "manifest.json", {"schema_version": 1, "status": "complete" if process.returncode == 0 and size <= LIMITS["output_bytes"] else "failed", "files": members})
    if process.returncode:
        raise SystemExit(process.returncode)
    if size > LIMITS["output_bytes"]:
        raise SystemExit("Study output bound exceeded; artifacts preserved")
    print(json.dumps(json.loads((out / "verification.json").read_bytes())), flush=True)


if __name__ == "__main__":
    main()
