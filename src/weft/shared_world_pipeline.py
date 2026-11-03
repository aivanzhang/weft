"""Package source-linked initial equipment geometry without claiming recovered contact."""

import hashlib
import json
import math
from pathlib import Path
import re
import time

from weft.io import canonical_hash, local_time
from weft.shared_world import FULLSPAN_PROFILE, audit_shared_world_feasibility, full_reference_horizon, validate_shared_world
from weft.shared_world_evidence import (
    MAX_INPUT_BYTES, MAX_SOURCE_FILES, build_shared_world_binding, verify_shared_world_sources,
)
from weft.shared_world_packaging import (
    collect_shared_source_files, read_package_file, read_package_manifest, write_package_files,
)
from weft.source_equivalence import MAX_FILES, load_source_equivalence
from weft.world_evidence import _check_tree, _nonfinite, _unique_object


MAX_OUTPUT_FILES = 512
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
METADATA_RESERVE_BYTES = 2 * 1024 * 1024


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Construction exceeded its cooperative wall-time bound")


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _human_sidecar(scene, full_raw):
    if "human_provenance" in scene:
        from weft.shared_human_projection import validate_projection
        validate_projection(scene["human_spec"], scene["human_provenance"], full_raw)
        return full_raw
    return _bytes(scene["human_spec"])


def _json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
        _check_tree(value)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Construction input contains malformed JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Construction input must be a JSON object")
    return value


def _sealed_file(directory, files, name, *, max_bytes=MAX_INPUT_BYTES):
    if name not in files:
        raise ValueError(f"Required construction input is not sealed: {name}")
    try:
        return read_package_file(directory, name, max_bytes=max_bytes, expected_sha256=files[name])
    except ValueError as error:
        raise ValueError(f"Construction input {name}: {error}") from error


def read_equipment_observations(directory, files):
    """Accept the canonical exporter and legacy study layout without rewriting either."""
    names = [name for name in ("scene-observations.json", "evidence/scene-observations.json") if name in files]
    if not names:
        raise ValueError("Equipment input requires sealed scene-observations.json")
    documents = [_sealed_file(directory, files, name) for name in names]
    if any(raw != documents[0] for raw in documents[1:]):
        raise ValueError("Equipment input contains conflicting sealed observation documents")
    return documents[0]


def construct_source_equipment_run(base_run, proposal_run, evidence_run, out, *, target_object_id,
                                   source_mappings=(), frame_translation=None, full_reference=False, deadline=None):
    """Construct a new sealed coupled candidate from explicit, verified intermediate inputs."""
    from weft.shared_world_reconstruction import MAX_BODY_JSON_BYTES, MAX_PROPOSAL_BYTES, PROFILE, construct_equipment_world

    started = time.monotonic()
    if type(full_reference) is not bool:
        raise ValueError("Full-reference construction must be explicitly enabled or disabled")
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("Construction deadline must be a finite absolute monotonic time")
    deadline = min(started + 30.0, deadline) if deadline is not None else started + 30.0
    _check_deadline(deadline)

    def checked(operation, *args, **kwargs):
        # Existing bounded/native calls are cooperative, not preemptible.
        _check_deadline(deadline)
        result = operation(*args, **kwargs)
        _check_deadline(deadline)
        return result

    if not isinstance(source_mappings, (tuple, list)) or len(source_mappings) > 7:
        raise ValueError("Use at most seven explicitly supplied source mapping bundles")
    base_run, proposal_run, evidence_run, out = map(Path, (base_run, proposal_run, evidence_run, out))
    inputs = [base_run, proposal_run, evidence_run, *(Path(path) for path in source_mappings)]
    if out.exists():
        raise FileExistsError("Choose a new source-equipment output directory")
    if any(out.resolve().is_relative_to(path.resolve()) or path.resolve().is_relative_to(out.resolve())
           for path in inputs):
        raise ValueError("Keep construction output outside every immutable input")
    base_manifest, base_files = checked(read_package_manifest, base_run)
    base_raw = checked(_sealed_file, base_run, base_files, "scene.json", max_bytes=MAX_BODY_JSON_BYTES)
    base = checked(validate_shared_world, _json(base_raw))
    base_sources = checked(collect_shared_source_files, base, base_run,
        expected_manifest_sha256=hashlib.sha256(base_manifest).hexdigest())
    if base["source_binding"]["mode"] != "source_qualified":
        raise ValueError("Equipment reconstruction requires an existing source-qualified mobile body")

    proposal_manifest, proposal_files = checked(read_package_manifest, proposal_run)
    proposal_raw = _sealed_file(proposal_run, proposal_files, "geometry-proposal.json", max_bytes=MAX_PROPOSAL_BYTES)
    proposal = _json(proposal_raw)
    equipment, body_binding = proposal.get("equipment"), proposal.get("body_binding")
    if (proposal.get("profile") != PROFILE or type(proposal.get("schema_version")) is not int
            or proposal["schema_version"] != 1 or not isinstance(equipment, list) or not 1 <= len(equipment) <= 12
            or not isinstance(body_binding, dict)
            or not isinstance(body_binding.get("human_json_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", body_binding["human_json_sha256"])
            or any(not isinstance(row, dict) or not isinstance(row.get("object_id"), str)
                   or not 1 <= len(row["object_id"]) <= 80 for row in equipment)):
        raise ValueError("Equipment proposal requires its version, pinned body and bounded object identities")
    evidence_manifest, evidence_files = checked(read_package_manifest, evidence_run)
    equipment_raw = checked(read_equipment_observations, evidence_run, evidence_files)
    equipment_sha = hashlib.sha256(equipment_raw).hexdigest()
    human_ref = next(row for row in base["source_binding"]["entity_bindings"] if row["world_entity_id"] == "human")
    human_sha = human_ref["observation_document_sha256"]
    if human_sha is None:
        raise ValueError("The body must have its own exact source observation binding")
    human_raw = base_sources.files[base["source_binding"]["evidence_documents"][human_sha]]
    del base_sources
    documents = {human_sha: human_raw, equipment_sha: equipment_raw}

    human_file_sha = proposal["body_binding"]["human_json_sha256"]
    human_files = [name for name, sha in base_files.items() if sha == human_file_sha
                   and Path(name).name == "human.json"]
    if not human_files:
        raise ValueError("Geometry proposal must pin an exact body document sealed in the base run")
    human_document = _sealed_file(base_run, base_files, sorted(human_files)[0], max_bytes=MAX_BODY_JSON_BYTES)
    mappings = []
    remaining = MAX_INPUT_BYTES - sum(map(len, documents.values()))
    remaining_files = MAX_SOURCE_FILES - len(documents)
    for path in source_mappings:
        _check_deadline(deadline)
        item = load_source_equivalence(path, max_input_bytes=min(remaining, MAX_INPUT_BYTES),
                                       max_files=min(remaining_files, MAX_FILES + 1),
                                       deadline=deadline)
        remaining -= item.total_input_bytes
        remaining_files -= len(item.file_names)
        mappings.append(item)
    ids = [row["object_id"] for row in proposal["equipment"]]
    refs = {"human": (human_sha, human_ref["observation_entity_id"]),
            **{identifier: (equipment_sha, identifier) for identifier in ids}}
    binding, construction = build_shared_world_binding(documents,
        source_sha256=base["source_binding"]["source_sha256"], world_ids=["human", *ids],
        entity_refs=refs, source_equivalences=mappings, deadline=deadline)
    frame_receipt = None
    if frame_translation is not None:
        from weft.shared_world_reconstruction import translate_shared_world_xy
        base, frame_receipt = checked(translate_shared_world_xy, base, frame_translation)
    scene = checked(construct_equipment_world, base, proposal, source_binding=binding,
        construction_evidence=construction, human_document=human_document, target_object_id=target_object_id,
        **({'deadline': deadline} if proposal.get('proposer', {}).get('frame_selection') is not None else {}))
    initial_interval = {key: scene["task"][key] for key in ("start_offset_s", "horizon_steps")}
    if full_reference:
        # Choose the new task before packaging; never stretch an existing task or policy in place.
        scene["profile"] = FULLSPAN_PROFILE
        scene["task"].update(start_offset_s=0.,
                            horizon_steps=full_reference_horizon(scene["human_spec"]["reference"], scene["dt"]))
        scene = checked(validate_shared_world, scene)
    feasibility = checked(audit_shared_world_feasibility, scene)

    files = {name: documents[sha] for sha, name in binding["evidence_documents"].items()}
    for item in mappings:
        files.update({f"source-mappings/{item.manifest_sha256}/{name}": item.raw_bytes(name)
                      for name in item.file_names})
    files.update({"construction-evidence.json": construction, "geometry-proposal.json": proposal_raw,
                  "inputs/base-scene.json": base_raw, "inputs/base-manifest.json": base_manifest,
                  "inputs/human.json": human_document, "inputs/proposal-manifest.json": proposal_manifest,
                  "inputs/evidence-manifest.json": evidence_manifest})
    if proposal.get('proposer', {}).get('frame_selection') is not None:
        from weft.equipment_box_consensus import validate_selection_reference

        selections = {item['object_id']: item['frame_selection_receipt'] for item in proposal['equipment']
                      if 'frame_selection_receipt' in item}
        references = {obj['id']: obj['provenance']['frame_selection_reference'] for obj in scene['objects']
                      if 'frame_selection_reference' in obj['provenance']}
        if set(references) != set(selections):
            raise ValueError('Scene selection references differ from the complete proposal receipts')
        for name, reference in references.items():
            validate_selection_reference(reference, selections[name])
        files['frame-selection.json'] = _bytes(selections)
    if "report.json" in proposal_files:
        files["geometry-study.json"] = _sealed_file(proposal_run, proposal_files, "report.json")
    inspection_files = [name for name in proposal_files if name in {
        "candidates.json", "body-reprojection.json", "mask-reprojection.json", "source-slot-alignment.json",
        "input-bindings.json", "pre-run.json"} or name.startswith("perspective-") and name.endswith(".png")]
    if len(inspection_files) > 16:
        raise ValueError("Geometry inspection export exceeds 16 files")
    inspection_bytes = 0
    for name in inspection_files:
        _check_deadline(deadline)
        raw = read_package_file(proposal_run, name, max_bytes=16 * 1024 * 1024 - inspection_bytes,
                                expected_sha256=proposal_files[name])
        inspection_bytes += len(raw)
        files[f"geometry-inspection/{name}"] = raw
    # Show the unchanged original video; the derivative remains a separately pinned proof input.
    original_sha = binding["source_sha256"]
    for item in mappings:
        for name in item.file_names:
            suffix = Path(name).suffix.lower()
            raw = item.raw_bytes(name)
            if suffix in {".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv"} and hashlib.sha256(raw).hexdigest() == original_sha:
                files[f"source{suffix}"] = raw
    attribution_files = sorted(f"source-mappings/{item.manifest_sha256}/source.json"
                               for item in mappings if "source.json" in item.file_names)
    if "source.json" in base_files:
        files["inputs/base-source-metadata.json"] = _sealed_file(base_run, base_files, "source.json", max_bytes=500_000)
        attribution_files.append("inputs/base-source-metadata.json")
    files["source.json"] = (json.dumps({"sha256": original_sha, "decoded_frames": binding["total_source_frames"],
        "scope": "Canonical source identity/count verified from pinned evidence; no physical calibration.",
        "attribution_metadata_files": attribution_files}, indent=2) + "\n").encode()
    if not any(name.startswith("source.") and not name.endswith(".json") for name in files):
        for name, sha in base_files.items():
            if Path(name).name.startswith("source.") and sha == original_sha:
                files[Path(name).name] = _sealed_file(base_run, base_files, name)
    depth_modules = (("equipment_proposal_config.py",) if proposal.get("proposer", {}).get("method") ==
                     "explicit-fixed-pelvis-floor-box-depth-ratio-proposal-v1" else ())
    if proposal.get('proposer', {}).get('frame_selection') is not None:
        depth_modules = tuple(dict.fromkeys((*depth_modules, 'equipment_proposal_config.py', 'equipment_box_consensus.py', 'io.py')))
    for name in ("shared_world_pipeline.py", "shared_world_reconstruction.py", "shared_world.py",
                 "shared_world_observation.py",
                 "shared_world_evidence.py", "shared_world_packaging.py", "source_equivalence.py",
                 "world_evidence.py", "observations.py", "human.py", "mobile_human.py", "whole_body.py",
                 "body_frame.py", "spatial_pose.py", "root_motion.py", "errors.py",
                 "spatial_rigid.py", *depth_modules):
        files[f"implementation/{name}"] = Path(__file__).with_name(name).read_bytes()
    if "human_provenance" in scene:
        files["implementation/shared_human_projection.py"] = Path(__file__).with_name("shared_human_projection.py").read_bytes()
    files.update({"scene.json": _bytes(scene), "human.json": _human_sidecar(scene, human_document),
                  "task.json": _bytes(scene["task"]), "reference-feasibility.json": _bytes(feasibility),
                  "repair-ledger.json": _bytes({"proposals": [],
                    "scope": "Initial geometry proposal only; retained camera alternatives are in geometry-study.json. No dynamic repair or learning executed."})})
    if frame_receipt is not None:
        files["frame-migration.json"] = _bytes(frame_receipt)
        files["inputs/framed-base-scene.json"] = _bytes(base)
    # Verification reports and the complete manifest have reserved capacity before any output.
    if (len(files) + 3 > MAX_OUTPUT_FILES or
            sum(map(len, files.values())) + METADATA_RESERVE_BYTES > MAX_OUTPUT_BYTES or
            any(len(raw) > 64 * 1024 * 1024 for raw in files.values())):
        raise ValueError("Construction complete output exceeds its file/byte bound")
    _check_deadline(deadline)
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        checked(write_package_files, out, {name: raw})
    source_check = checked(verify_shared_world_sources, scene, out)
    report = {"schema_version": 1, "profile": scene["profile"], "created_at": local_time(),
              "mode": "source_equipment_initial_geometry", "physical_validation": "unverified",
              "scene_sha256": canonical_hash(scene), "source_binding_verification": source_check,
              "base_manifest_sha256": hashlib.sha256(base_manifest).hexdigest(),
              "proposal_manifest_sha256": hashlib.sha256(proposal_manifest).hexdigest(),
              "evidence_manifest_sha256": hashlib.sha256(evidence_manifest).hexdigest(),
              "proposal_sha256": hashlib.sha256(proposal_raw).hexdigest(),
              "preserved_human_spec_sha256": canonical_hash(base["human_spec"]),
              "reference_interval": {"mode": "complete_reference" if full_reference else "initial_window",
                  "initial_window": initial_interval,
                  "selected": {key: scene["task"][key] for key in ("start_offset_s", "horizon_steps")},
                  "first_reference_time_s": scene["human_spec"]["reference"][0]["time_s"],
                  "last_reference_time_s": scene["human_spec"]["reference"][-1]["time_s"],
                  "dt": scene["dt"], "substeps": scene["substeps"],
                  "scope": "Task interval choice, not complete source observation, source-clock alignment or task success."},
              "frame_migration": frame_receipt,
              "reference_feasibility": {key: feasibility[key] for key in ("status", "summary")},
              "construction_seconds": time.monotonic() - started,
              "scope": "Source-linked collision proxies in one body/object simulator; camera, depth, materials and initial state are explicit priors. No recovered grasp, hand support, complete exercise or physical validation."}
    metadata = {"source-binding-verification.json": _bytes(source_check), "reconstruction.json": _bytes(report)}
    manifest = _bytes({"schema_version": 1, "files": {
        name: hashlib.sha256(raw).hexdigest() for name, raw in (files | metadata).items()}})
    if sum(map(len, metadata.values())) + len(manifest) > METADATA_RESERVE_BYTES:
        raise ValueError("Construction verification metadata exceeds its reserved byte bound")
    checked(write_package_files, out, metadata)
    for name, raw in (files | metadata).items():
        checked(read_package_file, out, name, max_bytes=max(1, len(raw)),
                expected_sha256=hashlib.sha256(raw).hexdigest())
    checked(write_package_files, out, {"manifest.json": manifest})
    checked(collect_shared_source_files, scene, out)
    return {"out": str(out), "scene_sha256": report["scene_sha256"],
            "profile": scene["profile"], "reference_interval": report["reference_interval"],
            "source_documents": len(documents), "source_equivalences": len(mappings),
            "reference_feasibility": report["reference_feasibility"],
            "scope": report["scope"], "manifest_sha256": hashlib.sha256(manifest).hexdigest()}
