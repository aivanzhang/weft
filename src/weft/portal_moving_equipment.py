"""Saved-evidence moving hypotheses: explicit assumptions, one immutable selection, no AI."""

from hashlib import sha256
import json
from pathlib import Path
import re
import time

from weft.io import canonical_hash, local_time
from weft.lab_worker import atomic_json
from weft.shared_world_packaging import read_package_file, write_package_files

PROFILE = "portal-moving-equipment-v1"
DIRECTORY = "moving-equipment"
ACTIVE = {"queued", "running", "cancelling"}
TERMINAL = {"awaiting_choices", "complete", "failed", "cancelled", "interrupted"}
MAX_SUMMARY_BYTES = 1 << 20
MAX_FAMILY_BYTES = 16 << 20
MAX_CHOICES = 3
MAX_JOB_BYTES = 1536 << 20
RECEIPT_BYTES = 32 << 20
SCOPE = "Explicit identity and depth assumptions for manual, unscored simulation; no recovered grasp, physical validation or learned task."
ARTIFACTS = {"family.json", "family-summary.json", "source-validation.json", "request.json", "build-request.json",
             "candidates.json", "selection.json", "prepare-start.json", "build-start.json"}


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _read(root, name, maximum=64 << 10, pin=None):
    from weft.shared_world_action_replay import _json
    raw = read_package_file(root, name, max_bytes=maximum, expected_sha256=pin)
    return raw, _json(raw)


def _save(root, name, value, maximum=MAX_SUMMARY_BYTES):
    raw = _bytes(value)
    if len(raw) > maximum:
        raise ValueError("Moving equipment receipt exceeds its bound")
    write_package_files(root, {name: raw})
    return sha256(raw).hexdigest()


def read_status(root):
    path = Path(root) / DIRECTORY / "status.json"
    return _read(path.parent, path.name, 32 << 10)[1] if path.exists() else {}


def _base(root):
    root = Path(root)
    job_raw, job = _read(root, "job.json")
    status_raw, status = _read(root, "status.json", 32 << 10)
    if (job.get("kind", "video") != "video" or status.get("status") != "needs_input"
            or status.get("route") != "exercise" or status.get("lab")
            or status.get("equipment_candidate_id") or status.get("moving_candidate_id")
            or (root / "equipment/selection.json").exists()):
        raise ValueError("Moving hypotheses require an unresolved, unselected original exercise upload")
    for path in ("equipment/status.json", "perception/status.json", "muse/status.json"):
        if (root / path).exists():
            state = _read(root, path, 32 << 10)[1].get("status")
            if state not in {"complete", "failed", "interrupted", "cancelled", "evidence_ready", "proposed"}:
                raise ValueError("Finish the existing preparation before creating moving hypotheses")
    descriptor_raw, descriptor = _read(root, "perception/prepared-source.json", 16 << 10)
    if descriptor.get("source_sha256") != job.get("source_sha256") or descriptor.get("job_sha256") != sha256(job_raw).hexdigest():
        raise ValueError("Prepared source metadata differs from this upload")
    return job_raw, status_raw, descriptor_raw


def summary(root):
    """Small published metadata only; source admission and geometry never run here."""
    root = Path(root)
    state = read_status(root)
    result = {**state, "available": False, "scope": SCOPE, "api_calls": 0}
    if not state:
        try:
            _base(root)
            result.update(available=True, message="Explore moving equipment using saved evidence. No new AI request.")
        except (ValueError, OSError, KeyError):
            pass
        return result
    work = root / DIRECTORY
    if state.get("family_summary_sha256"):
        result["family"] = _read(work, "family-summary.json", MAX_SUMMARY_BYTES, state["family_summary_sha256"])[1]
    if state.get("report_sha256"):
        result["report"] = _read(work, "candidates.json", 128 << 10, state["report_sha256"])[1]
    return result


def queue_prepare(root, settings):
    from weft.moving_depth_family import validate_moving_depth_settings
    settings = validate_moving_depth_settings(settings)
    root = Path(root); work = root / DIRECTORY
    if work.exists():
        if _read(work, "request.json")[1].get("settings") != settings:
            raise ValueError("This project already has different moving assumptions; its history cannot be overwritten")
        return
    job, status, descriptor = _base(root)
    request = {"profile": PROFILE, "job_sha256": sha256(job).hexdigest(),
        "parent_status_sha256": sha256(status).hexdigest(), "descriptor_sha256": sha256(descriptor).hexdigest(),
        "settings": settings, "api_calls": 0, "inference_calls": 0}
    work.mkdir(mode=0o700)
    _save(work, "storage-plan.json", {"profile": PROFILE, "max_job_bytes": MAX_JOB_BYTES, "receipt_bytes": RECEIPT_BYTES, "max_materializations": MAX_CHOICES})
    write_package_files(work, {"parent-status.json": status})
    _save(work, "request.json", request, 64 << 10)
    atomic_json(work / "status.json", {"status": "queued", "phase": "prepare", "message": "Waiting to derive complete depth alternatives from saved evidence; no API call."})


def queue_build(root, selection):
    # Full numeric membership is revalidated off-thread against the authenticated family.
    if (not isinstance(selection, dict) or set(selection) != {"schema_version", "profile", "family_sha256", "choices"}
            or selection.get("schema_version") != 1 or selection.get("profile") != "moving-depth-selection-v1"
            or not isinstance(selection.get("family_sha256"), str) or not re.fullmatch("[a-f0-9]{64}", selection["family_sha256"])
            or not isinstance(selection.get("choices"), list) or not 1 <= len(selection["choices"]) <= MAX_CHOICES
            or len(_bytes(selection)) > 16 << 10):
        raise ValueError("Choose one to three explicit, bounded identity/depth hypotheses")
    work = Path(root) / DIRECTORY
    if (work / "build-request.json").exists():
        if _read(work, "build-request.json", 16 << 10)[1] != selection:
            raise ValueError("This construction attempt is immutable; existing choices cannot be replaced")
        return
    state = read_status(root)
    if state.get("status") != "awaiting_choices":
        raise ValueError("Finish deriving the complete family before choosing assumptions")
    family = _read(work, "family-summary.json", MAX_SUMMARY_BYTES, state["family_summary_sha256"])[1]
    if family["family_sha256"] != selection["family_sha256"]:
        raise ValueError("Chosen family differs from the published source alternatives")
    _base(root)
    _save(work, "build-request.json", selection, 16 << 10)
    atomic_json(work / "status.json", {**state, "status": "queued", "phase": "build", "message": "Waiting to construct the explicitly chosen assumptions. All other alternatives remain saved."})


def _source(root, project, deadline):
    from weft.portal_equipment_candidates import validate_job_source
    from weft.moving_equipment_inputs import load_moving_equipment_inputs
    work = root / DIRECTORY
    request = _read(work, "request.json")[1]
    if set(request) != {"profile", "job_sha256", "parent_status_sha256", "descriptor_sha256", "settings", "api_calls", "inference_calls"} or request["profile"] != PROFILE:
        raise ValueError("Moving request schema differs")
    job, parent, descriptor = _base(root)
    if (sha256(job).hexdigest() != request["job_sha256"] or sha256(parent).hexdigest() != request["parent_status_sha256"]
            or sha256(descriptor).hexdigest() != request["descriptor_sha256"]
            or parent != read_package_file(work, "parent-status.json", max_bytes=32 << 10)
            or type(request["api_calls"]) is not int or request["api_calls"] != 0
            or type(request["inference_calls"]) is not int or request["inference_calls"] != 0):
        raise ValueError("Moving request original source/parent identities changed")
    paths, proof = validate_job_source(root, project=project, deadline=deadline)
    if proof["descriptor_sha256"] != request["descriptor_sha256"]:
        raise ValueError("Moving source descriptor changed during qualification")
    _, desc = _read(root, "perception/prepared-source.json", 16 << 10, request["descriptor_sha256"])
    packet = load_moving_equipment_inputs(**paths, base_manifest_sha256=desc["base"]["manifest_sha256"],
        evidence_manifest_sha256=desc["equipment"]["manifest_sha256"], max_seconds=max(.01, min(30, deadline-time.monotonic())))
    return packet, request, proof


def _family_summary(document, raw):
    factors = [{"object_id": item["object_id"], "status": item["status"], "reasons": item["reasons"],
        "certified_intervals": (item["depth_intervals"] or {}).get("certified_intervals", []),
        "possible_intervals": (item["depth_intervals"] or {}).get("possible_intervals", []),
        "boundary_ambiguity": (item["depth_intervals"] or {}).get("boundary_ambiguity", False)} for item in document["factors"]]
    return {"profile": PROFILE, "family_sha256": canonical_hash(document), "family_file_sha256": sha256(raw).hexdigest(),
        "status": document["status"], "identity_family": document["identity_family"], "settings": document["settings"],
        "factors": factors, "scope": SCOPE, "maximum_materializations": MAX_CHOICES}


def execute_job(root, *, project):
    """One existing supervised portal worker; no provider or body-inference fallback."""
    from weft.moving_depth_family import prepare_moving_depth_family, admit_moving_depth_family, materialize_moving_depth_selection
    from weft.moving_equipment_inputs import verify_moving_equipment_inputs_unchanged
    from weft.shared_world_exploration import prepare_shared_world_exploration, MAX_LAB_BYTES
    root = Path(root); work = root / DIRECTORY
    state = read_status(root); phase = state.get("phase")
    if state.get("status") in TERMINAL:
        return 0 if state["status"] in {"awaiting_choices", "complete"} else 1
    started = time.monotonic(); deadline = started + (90 if phase == "prepare" else 180)
    try:
        if phase not in {"prepare", "build"}:
            raise ValueError("Unknown moving worker phase")
        if (work / "cancel-request.json").exists():
            raise ValueError("Moving preparation cancelled before source validation")
        _save(work, phase + "-start.json", {"profile": PROFILE, "started_at": local_time(), "api_calls": 0, "inference_calls": 0})
        atomic_json(work / "status.json", {**state, "status": "running", "message": "Validating saved source and explicit moving assumptions. No new AI request."})
        request_raw = read_package_file(work, "request.json", max_bytes=64 << 10)
        storage_limit(root)
        packet, request, proof = _source(root, project, deadline)
        if phase == "prepare":
            family = prepare_moving_depth_family(packet, request["settings"], deadline=deadline)
            raw = family.document_bytes
            if len(raw) > MAX_FAMILY_BYTES:
                raise ValueError("Complete moving family exceeds its retained output bound")
            write_package_files(work, {"family.json": raw})
            info = _family_summary(family.document, raw)
            pin = _save(work, "family-summary.json", info)
            _save(work, "source-validation.json", proof)
            next_state = {**state, "family_summary_sha256": pin, "status": "awaiting_choices" if info["status"] == "ready_for_explicit_selection" else "complete",
                "message": "Choose identity and depth assumptions explicitly; no depth was automatically selected." if info["status"] == "ready_for_explicit_selection" else "No certified depth was found. All alternatives and refusal reasons are preserved."}
        else:
            info = _read(work, "family-summary.json", MAX_SUMMARY_BYTES, state["family_summary_sha256"])[1]
            _, document = _read(work, "family.json", MAX_FAMILY_BYTES, info["family_file_sha256"])
            if canonical_hash(document) != info["family_sha256"] or document["settings"] != request["settings"]:
                raise ValueError("Saved family assumptions changed")
            family = admit_moving_depth_family(packet, document, deadline=deadline)
            selection_raw, selection = _read(work, "build-request.json", 16 << 10)
            bundle = materialize_moving_depth_selection(family, selection, deadline=deadline)
            rows = []
            for candidate in bundle["candidates"]:
                identity = candidate["id"]
                row = {"id": identity, "status": "deferred", "selectable": False, "reasons": [], "scope": SCOPE}
                if time.monotonic() >= deadline or (work / "cancel-request.json").exists():
                    row["reasons"] = ["Construction deadline or cancellation; this explicit choice was not executed."]
                else:
                    try:
                        result = prepare_shared_world_exploration(packet, bundle, identity, work / "run" / identity,
                            depth_family=family, max_seconds=60, deadline=deadline)
                        lab = Path(result["lab_directory"])
                        if lab != work / "run" / identity or result["control_profile"] != "shared_world_exploration" or result["task_success_available"] is not False:
                            raise ValueError("Moving constructor returned a different lab/control scope")
                        config_raw, config = _read(lab, "lab-config.json")
                        read_package_file(lab / "run", "manifest.json", max_bytes=8 << 20, expected_sha256=config["manifest_sha256"])
                        if result["bytes_written"] > MAX_LAB_BYTES:
                            raise ValueError("Moving lab exceeds its byte allowance")
                        row.update(status="ready", selectable=True, lab=lab.relative_to(root).as_posix(),
                            manifest_sha256=config["manifest_sha256"], lab_config_sha256=sha256(config_raw).hexdigest(), scene_sha256=config["scene_sha256"])
                    except Exception as error:
                        row.update(status="rejected", reasons=[str(error)[:1000]])
                rows.append(row)
            report = {"profile": PROFILE, "family_sha256": info["family_sha256"], "selection_sha256": sha256(selection_raw).hexdigest(),
                "candidates": rows, "api_calls": 0, "inference_calls": 0, "task_success_available": False, "scope": SCOPE}
            pin = _save(work, "candidates.json", report, 128 << 10)
            next_state = {**state, "status": "complete", "report_sha256": pin, "message": "Choose one admitted world to explore; all refusals and unselected alternatives remain saved." if any(r["selectable"] for r in rows) else "No selected hypothesis passed the unchanged constructor. No world was substituted."}
        verify_moving_equipment_inputs_unchanged(packet, deadline=deadline)
        job_raw, parent_raw, descriptor_raw = _base(root)
        final_request_raw, original_request = _read(work, "request.json")
        if final_request_raw != request_raw or (phase == "build" and read_package_file(work, "build-request.json", max_bytes=16 << 10) != selection_raw):
            raise ValueError("Moving request or explicit selection changed before publication")
        if any(sha256(raw).hexdigest() != original_request[key] for raw, key in ((job_raw, "job_sha256"), (parent_raw, "parent_status_sha256"), (descriptor_raw, "descriptor_sha256"))):
            raise ValueError("Moving parent/source descriptor changed before publication")
        if time.monotonic() >= deadline:
            raise TimeoutError("Moving preparation exceeded its aggregate deadline")
        atomic_json(work / "status.json", {**next_state, "seconds": time.monotonic()-started, "api_calls": 0, "inference_calls": 0})
        return 0
    except Exception as error:
        atomic_json(work / "status.json", {**state, "status": "failed", "error": str(error)[:1000],
            "message": "Moving preparation stopped; original evidence and partial alternatives are preserved. No automatic retry.", "api_calls": 0, "inference_calls": 0})
        return 1


def select_candidate(root, identity):
    root = Path(root); work = root / DIRECTORY
    state = read_status(root)
    if state.get("status") != "complete" or not state.get("report_sha256"):
        raise ValueError("Finish moving construction before choosing a world")
    report = _read(work, "candidates.json", 128 << 10, state["report_sha256"])[1]
    row = next((r for r in report["candidates"] if r["id"] == identity), None)
    if row is None or row.get("selectable") is not True or row.get("status") != "ready":
        raise ValueError("This moving hypothesis was not admitted")
    relative = f"{DIRECTORY}/run/{identity}"
    if not re.fullmatch(r"candidate-[1-3]", identity) or row["lab"] != relative:
        raise ValueError("Moving lab path differs from its selected hypothesis")
    config = _read(root / relative, "lab-config.json", 64 << 10, row["lab_config_sha256"])[1]
    read_package_file(root / relative / "run", "manifest.json", max_bytes=8 << 20, expected_sha256=row["manifest_sha256"])
    if (config.get("control_profile") != "shared_world_exploration" or config.get("task_success_available") is not False
            or config.get("learned_policy_available") is not False or config.get("manifest_sha256") != row["manifest_sha256"]):
        raise ValueError("Moving selection requires the exact manual-only lab")
    saved = {"profile": PROFILE, "candidate_id": identity, "lab": relative, "manifest_sha256": row["manifest_sha256"],
        "lab_config_sha256": row["lab_config_sha256"], "report_sha256": state["report_sha256"], "control_profile": "shared_world_exploration", "scope": SCOPE}
    if (work / "selection.json").exists():
        if _read(work, "selection.json")[1] != saved:
            raise ValueError("This project is locked to its selected moving world")
    else:
        _, original, _ = _base(root)
        if original != read_package_file(work, "parent-status.json", max_bytes=32 << 10):
            raise ValueError("Original project changed before moving selection")
        _save(work, "selection.json", saved)
    parent = _read(work, "parent-status.json", 32 << 10)[1]
    current = _read(root, "status.json", 32 << 10)[1]
    if current.get("lab") and (current.get("lab") != relative or current.get("moving_candidate_id") != identity):
        raise ValueError("Selected project runtime cannot be replaced")
    atomic_json(root / "status.json", {**parent, "status": "ready", "stage": "moving_equipment", "lab": relative,
        "moving_candidate_id": identity, "control_profile": "shared_world_exploration", "scope": SCOPE})


def selected_profile(root, status):
    """Bind lightweight selection identities; full source/runtime validation stays in the loader."""
    if not status.get("moving_candidate_id"):
        return None
    work = Path(root) / DIRECTORY
    saved = _read(work, "selection.json")[1]
    if (set(saved) != {"profile", "candidate_id", "lab", "manifest_sha256", "lab_config_sha256", "report_sha256", "control_profile", "scope"}
            or saved["profile"] != PROFILE or saved["control_profile"] != "shared_world_exploration" or saved["scope"] != SCOPE
            or not isinstance(saved["candidate_id"], str) or not re.fullmatch(r"candidate-[1-3]", saved["candidate_id"])):
        raise ValueError("Moving selection requires its closed, contained exploration identity")
    state = read_status(root)
    report = _read(work, "candidates.json", 128 << 10, saved["report_sha256"])[1]
    row = next((r for r in report["candidates"] if r["id"] == saved["candidate_id"]), None)
    if (row is None or row.get("selectable") is not True or row.get("status") != "ready" or state.get("report_sha256") != saved["report_sha256"]
            or saved["candidate_id"] != status["moving_candidate_id"] or saved["lab"] != status.get("lab")
            or saved["lab"] != f"{DIRECTORY}/run/{saved['candidate_id']}"
            or saved["manifest_sha256"] != row.get("manifest_sha256") or saved["lab_config_sha256"] != row.get("lab_config_sha256")):
        raise ValueError("Moving selection differs from its immutable construction report")
    config = _read(root / saved["lab"], "lab-config.json", 64 << 10, saved["lab_config_sha256"])[1]
    read_package_file(root / saved["lab"] / "run", "manifest.json", max_bytes=8 << 20, expected_sha256=saved["manifest_sha256"])
    if config.get("control_profile") != "shared_world_exploration" or config.get("task_success_available") is not False:
        raise ValueError("Moving selection is not an unscored exploration")
    return "shared_world_exploration"


def storage_limit(root):
    plan = _read(Path(root) / DIRECTORY, "storage-plan.json", 8192)[1]
    if plan != {"profile": PROFILE, "max_job_bytes": MAX_JOB_BYTES, "receipt_bytes": RECEIPT_BYTES, "max_materializations": MAX_CHOICES}:
        raise ValueError("Moving storage allowance requires its exact closed workflow plan")
    request = _read(Path(root) / DIRECTORY, "request.json")[1]
    if request.get("profile") != PROFILE or request.get("api_calls") != 0 or request.get("inference_calls") != 0:
        raise ValueError("Moving storage allowance requires a saved-evidence-only request")
    return MAX_JOB_BYTES


def reserved_growth(root):
    state = read_status(root)
    if state.get("status") not in ACTIVE:
        return 0
    storage_limit(root)
    if state.get("phase") == "prepare":
        return RECEIPT_BYTES
    selection = _read(Path(root) / DIRECTORY, "build-request.json", 16 << 10)[1]
    count = len(selection.get("choices", []))
    if not 1 <= count <= MAX_CHOICES:
        raise ValueError("Moving storage reservation requires bounded explicit choices")
    return RECEIPT_BYTES + count * (256 << 20)
