"""Bounded saved-evidence construction into the existing interactive lab."""

from hashlib import sha256
import json
import math
from pathlib import Path
import time

from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files


PROFILE = "moving-equipment-build-v1"
MAX_SECONDS = 180
MAX_RECEIPT_BYTES = 32 << 20
MAX_OUTPUT_BYTES = 832 << 20
SCOPE = "Conditional source-backed simulation exploration; no recovered grasp, physical validation or scored exercise task."


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Moving-equipment build exceeded its aggregate time bound")


def _settings(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Moving-equipment settings contain duplicate keys")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Moving-equipment settings must be finite")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (RecursionError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid bounded moving-equipment settings JSON") from error
    from weft.moving_equipment_proposal import validate_moving_settings
    return validate_moving_settings(value)


def prepare_moving_equipment_scene(base_run, evidence_run, settings_path, out, *,
                                   source_mappings=(), max_seconds=120, max_candidates=3, progress=None):
    """Share source preparation across candidates and preserve every failed branch."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= MAX_SECONDS:
        raise ValueError("Moving-equipment build requires a 1–180 second bound")
    if type(max_candidates) is not int or not 1 <= max_candidates <= 3:
        raise ValueError("Moving-equipment build permits one to three candidate slots")
    if not isinstance(source_mappings, (tuple, list)) or len(source_mappings) > 7:
        raise ValueError("At most seven exact source mappings are supported")
    started = time.monotonic()
    deadline = started + max_seconds
    base_run, evidence_run, settings_path, out = (Path(p).absolute() for p in (base_run, evidence_run, settings_path, out))
    mappings = [Path(p).absolute() for p in source_mappings]
    paths = [base_run, evidence_run, settings_path, out, *mappings]
    if any(".." in p.parts or any(parent.is_symlink() for parent in (p, *p.parents)) for p in paths):
        raise ValueError("Moving-equipment paths cannot contain traversal or links")
    if out.exists() or any(out.is_relative_to(p) or p.is_relative_to(out)
                           for p in [base_run, evidence_run, settings_path, *mappings]):
        raise ValueError("Choose a new moving-equipment output outside every input")
    settings_raw = read_package_file(settings_path.parent, settings_path.name, max_bytes=64 << 10)
    settings = _settings(settings_raw)
    base_manifest, _ = read_package_manifest(base_run)
    equipment_manifest, _ = read_package_manifest(evidence_run)
    implementation = Path(__file__).read_bytes()
    files = {"settings-input.json": settings_raw, "settings.json": _bytes(settings),
             "inputs/base-manifest.json": base_manifest, "inputs/equipment-manifest.json": equipment_manifest,
             "implementation/moving_equipment_pipeline.py": implementation,
             "limits.json": _bytes({"maximum_seconds": max_seconds, "maximum_candidates": max_candidates,
                 "maximum_output_bytes": MAX_OUTPUT_BYTES, "concurrent_constructions": 1,
                 "api_calls": 0, "inference_calls": 0, "training": False})}
    result = {"profile": PROFILE, "status": "failed", "scope": SCOPE, "candidates": [], "reasons": [],
              "selected_candidate": None, "api_calls": 0, "inference_calls": 0, "training": False,
              "task_success_available": False, "physical_validation": "unverified"}

    def report(stage, detail):
        if progress is not None:
            progress(stage, detail)

    def save(extra):
        if sum(map(len, files.values())) + sum(map(len, extra.values())) > MAX_RECEIPT_BYTES:
            raise ValueError("Moving-equipment receipts exceed their aggregate byte bound")
        write_package_files(out, extra)
        files.update(extra)

    out.mkdir(parents=True, mode=0o700)
    write_package_files(out, files)
    try:
        from weft.moving_equipment_inputs import load_moving_equipment_inputs, verify_moving_equipment_inputs_unchanged
        from weft.moving_equipment_proposal import derive_moving_proposals
        from weft.shared_world_exploration import prepare_shared_world_exploration

        _check(deadline)
        report("inputs", "Read the original sealed body, clocks and equipment tracks once")
        inputs = load_moving_equipment_inputs(base_run, evidence_run,
            base_manifest_sha256=sha256(base_manifest).hexdigest(),
            evidence_manifest_sha256=sha256(equipment_manifest).hexdigest(),
            source_mappings=mappings, max_seconds=min(30, deadline - time.monotonic()))
        _check(deadline)
        report("proposals", "Retain alternative identities and explicit physical assumptions")
        bundle = derive_moving_proposals(inputs, settings, max_candidates=max_candidates, deadline=deadline)
        if not isinstance(bundle, dict) or not isinstance(bundle.get("candidates"), list) or len(bundle["candidates"]) > max_candidates:
            raise ValueError("Moving-equipment proposal family exceeds its candidate bound")
        candidates = bundle["candidates"]
        ids = [row.get("id") for row in candidates]
        if (any(not isinstance(name, str) or not name or len(name) > 80
                or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in name) for name in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("Moving-equipment candidate identities must be unique safe names")
        save({"moving-proposals.json": _bytes(bundle)})
        reserved = MAX_RECEIPT_BYTES
        for candidate in candidates:
            row = {"id": candidate["id"], "status": "not_attempted", "lab_directory": None, "reasons": []}
            result["candidates"].append(row)
            if candidate.get("status") != "conditional_free_body_hypothesis":
                row.update(status="unsupported_input", reasons=candidate.get("reasons", []))
                continue
            left = deadline - time.monotonic()
            if left < 1:
                row.update(status="deferred", reasons=["aggregate_time_bound"])
                continue
            # Reserve the entire package ceiling, including a failed partial package.
            from weft.shared_world_exploration import MAX_LAB_BYTES
            if type(MAX_LAB_BYTES) is not int or not 0 < MAX_LAB_BYTES <= 256 << 20:
                raise ValueError("Exploration lab byte ceiling is outside the build envelope")
            if reserved + MAX_LAB_BYTES > MAX_OUTPUT_BYTES:
                row.update(status="deferred", reasons=["aggregate_output_bound"])
                continue
            reserved += MAX_LAB_BYTES
            report("candidate", candidate["id"])
            try:
                built = prepare_shared_world_exploration(inputs, bundle, candidate["id"],
                    out / "candidates" / candidate["id"], max_seconds=min(60, left), deadline=deadline)
                if (built.get("status") != "ready" or built.get("control_profile") != "shared_world_exploration"
                        or built.get("task_success_available") is not False):
                    raise ValueError("Exploration adapter did not return a complete unscored lab")
                row.update(status="ready", lab_directory=built["lab_directory"],
                           manifest_sha256=built["manifest_sha256"], scene_sha256=built["scene_sha256"])
            except (ValueError, RuntimeError, TimeoutError, OSError, MemoryError) as error:
                row.update(status="failed", reasons=[{"code": type(error).__name__, "message": str(error)[:1000]}])
        _check(deadline)
        verify_moving_equipment_inputs_unchanged(inputs, deadline=deadline)
        if (read_package_manifest(base_run)[0] != base_manifest
                or read_package_manifest(evidence_run)[0] != equipment_manifest
                or read_package_file(settings_path.parent, settings_path.name, max_bytes=64 << 10) != settings_raw
                or Path(__file__).read_bytes() != implementation):
            raise ValueError("Moving-equipment input seals, settings or implementation changed")
        result.update(status="candidates_ready" if any(row["status"] == "ready" for row in result["candidates"]) else "abstained",
                      reasons=bundle.get("reasons", []), reserved_output_bytes=reserved)
    except (ValueError, RuntimeError, TimeoutError, OSError, MemoryError) as error:
        result.update(status="failed", reasons=[{"code": type(error).__name__, "message": str(error)[:1000]}])
    result["seconds"] = time.monotonic() - started
    report("seal", "No candidate is selected automatically")
    save({"result.json": _bytes(result)})
    write_package_files(out, {"manifest.json": _bytes({"schema_version": 1, "status": result["status"],
        "seal_scope": "Build inputs and complete candidate outcomes; each lab has its own sealed source/runtime closure",
        "files": {name: sha256(raw).hexdigest() for name, raw in files.items()}})})
    return result
