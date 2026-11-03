"""Bounded alternative role assumptions, never repaired source identities."""

from hashlib import sha256
import json
import math
from pathlib import Path
import time

from weft.equipment_autoproposal import MAX_CONTEXT_BYTES, POLICY, derive_equipment_config
from weft.equipment_proposal_config import CONFIG_PROFILE, CONSENSUS_METHOD, validate_proposal_config
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files


PROFILE = "conditional-equipment-assembly-v1"
MAX_CANDIDATES = 4
MAX_RECEIPT_BYTES = 64 << 20
SCOPE = ("Alternative body/rigid-role assumptions under unresolved source identity. Native admission is not "
         "identity resolution, recovered contact, physical accuracy or task success.")


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Conditional assembly exceeded its cooperative deadline")


def derive_conditional_equipment_configs(documents, *, max_candidates=3, **kwargs):
    """Branch an ambiguity component; original automatic abstention remains intact."""
    if type(max_candidates) is not int or not 2 <= max_candidates <= MAX_CANDIDATES:
        raise ValueError("Conditional assembly permits two to four candidate slots")
    automatic, construction = derive_equipment_config(documents, **kwargs)
    inventory = automatic["inventory"]
    selected = [row for row in inventory if row["selected_role"] in
                {"exact_bound_body", "conditional_body_correspondence"}]
    reasons = [reason for reason in automatic["reasons"] if reason != "unresolved_source_identity_relationships"]
    relationships = [row for row in automatic["relationships"] if row["type"] == "possible_association_ambiguity"]
    if not relationships:
        reasons.append("no_construction_identity_ambiguity_use_automatic_route")
    if len(selected) != 1:
        reasons.append("no_unique_initial_body_correspondence_hypothesis")
    component = {selected[0]["object_id"]} if len(selected) == 1 else set()
    previous = None
    while previous != component:
        previous = set(component)
        for relation in relationships:
            objects = {ref["object_id"] for ref in relation["entity_refs"]}
            if component.intersection(objects):
                component.update(objects)
    if len(component) < 2:
        reasons.append("ambiguity_does_not_connect_body_hypothesis")
    if len(component) > max_candidates:
        reasons.append("complete_alternative_set_exceeds_candidate_bound")
    if any(row["exact_body_entity"] for row in inventory):
        reasons.append("exact_body_entity_cannot_be_reassigned_as_rigid")
    candidates = []
    if not reasons:
        ids = {row["object_id"] for row in inventory}
        scores = {row["object_id"]: row["full_torso_containment_fraction"] for row in inventory}
        # Score orders display only; no competing identity is removed by a score threshold.
        order = sorted(component, key=lambda oid: (-scores[oid], oid))
        for index, body in enumerate(order):
            rigid = sorted(ids - {body})
            config = validate_proposal_config({"schema_version": 1, "profile": CONFIG_PROFILE,
                "source_sha256": automatic["source_sha256"],
                "camera": {"mode": "fixed_pelvis_perspective_prior",
                           "distance_model_units": POLICY["camera_distance_model_units"]},
                "geometry_prior": "floor_box_depth_equals_width",
                "equipment": [{"object_id": oid, "frame_selection": {"method": CONSENSUS_METHOD}} for oid in rigid],
                "target_object_id": rigid[0]})
            candidates.append({"id": f"alternative-{index + 1}", "status": "conditional_config",
                "assignment": {"body_correspondence": body, "rigid_proxies": rigid},
                "assumption_basis": "Initial torso correspondence" if body == selected[0]["object_id"]
                                    else "Competing assignment in the original construction ambiguity component",
                "body_containment_fraction": scores[body], "config": config,
                "all_raw_objects_accounted": sorted([body, *rigid]) == sorted(ids),
                "task_comparable": False,
                "task_scope": "Same authored target-selection rule, not the same physical target or comparable task across role assignments"})
    result = {"schema_version": 1, "profile": PROFILE,
        "status": "conditional_configs" if candidates else "abstained", "scope": SCOPE,
        "maximum_candidates": max_candidates, "role_report": automatic,
        "ambiguity_component": sorted(component), "candidates": candidates, "reasons": sorted(set(reasons)),
        "physical_identity_verified": False, "source_contact_verified": False,
        "raw_identity_reclassified": False, "source_slots_rewritten": 0,
        "conditional_assumptions": ["Exactly one raw equipment track corresponds to the retained body in each branch; this hypothesis family is not exhaustive.",
            "Every other raw track is an authored independent rigid proxy; duplicates, nonrigid tracks and occluded identities remain possible.",
            "Camera, depth, material, floor, zero motion and target-selection priors are identical across branches; none is source-verified."],
        "construction_evidence_sha256": sha256(construction).hexdigest()}
    if len(_bytes(result)) > MAX_CONTEXT_BYTES:
        raise ValueError("Conditional role context exceeds its byte bound")
    return result, construction


def prepare_conditional_equipment_scene(base_run, evidence_run, out, *, source_mappings=(), max_seconds=90,
                                        max_candidates=3):
    """Run each frozen role branch once through the ordinary proposal/world/lab gates."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 180:
        raise ValueError("Conditional assembly requires a 1–180 second bound")
    if type(max_candidates) is not int or not 2 <= max_candidates <= MAX_CANDIDATES:
        raise ValueError("Conditional assembly permits two to four candidate slots")
    if not isinstance(source_mappings, (tuple, list)) or len(source_mappings) > 7:
        raise ValueError("At most seven source mappings are supported")
    from weft.shared_world_packaging import collect_shared_source_files
    from weft.shared_world_pipeline import read_equipment_observations
    from weft.source_equivalence import load_source_equivalence
    from weft.exercise_scene_pipeline import MAX_BYTES, MAX_FILES, prepare_exercise_scene

    base_run, evidence_run, out = (Path(p).absolute() for p in (base_run, evidence_run, out))
    inputs = [base_run, evidence_run, *(Path(p).absolute() for p in source_mappings)]
    if any(".." in p.parts or any(item.is_symlink() for item in (p, *p.parents)) for p in [out, *inputs]):
        raise ValueError("Conditional paths cannot contain traversal or links")
    if out.exists() or any(out.is_relative_to(p) or p.is_relative_to(out) for p in inputs):
        raise ValueError("Choose a new conditional output outside every input")
    started = time.monotonic()
    deadline = started + max_seconds
    implementation = Path(__file__).read_bytes()
    base_manifest, base_files = read_package_manifest(base_run)
    scene_raw = read_package_file(base_run, "scene.json", expected_sha256=base_files["scene.json"], max_bytes=16 << 20)
    scene = json.loads(scene_raw)
    proof = collect_shared_source_files(scene, base_run, expected_manifest_sha256=sha256(base_manifest).hexdigest(), deadline=deadline)
    human = next(row for row in scene["source_binding"]["entity_bindings"] if row["world_entity_id"] == "human")
    human_sha = human["observation_document_sha256"]
    human_raw = proof.files[scene["source_binding"]["evidence_documents"][human_sha]]
    equipment_manifest, equipment_files = read_package_manifest(evidence_run)
    equipment_raw = read_equipment_observations(evidence_run, equipment_files)
    equipment_sha = sha256(equipment_raw).hexdigest()
    documents = {human_sha: human_raw, equipment_sha: equipment_raw}
    remaining = MAX_RECEIPT_BYTES - sum(map(len, documents.values()))
    equivalents = []
    for path in source_mappings:
        _check(deadline)
        item = load_source_equivalence(path, max_input_bytes=remaining, deadline=deadline)
        remaining -= item.total_input_bytes
        equivalents.append(item)
    context, construction = derive_conditional_equipment_configs(documents,
        source_sha256=scene["source_binding"]["source_sha256"], human_ref=(human_sha, human["observation_entity_id"]),
        equipment_document_sha256=equipment_sha, body_references=scene["human_spec"]["reference_evidence"],
        source_equivalences=equivalents, deadline=deadline, max_candidates=max_candidates)
    files = {"conditional-context.json": _bytes(context), "construction-evidence.json": construction,
        "inputs/base-manifest.json": base_manifest, "inputs/equipment-manifest.json": equipment_manifest,
        "inputs/base-scene.json": scene_raw, "implementation/equipment_conditional_assembly.py": implementation,
        **{f"inputs/observations/{pin}.json": raw for pin, raw in documents.items()},
        "settings.json": _bytes({"maximum_seconds": max_seconds, "maximum_candidates": max_candidates,
            "maximum_output_files": max_candidates * MAX_FILES + 32,
            "maximum_output_bytes": max_candidates * MAX_BYTES + MAX_RECEIPT_BYTES,
            "concurrent_constructions": 1, "training": False, "physics_steps": 0,
            "native_initial_admission": True, "api_calls": 0, "inference_calls": 0,
            "geometry_optimization": False, "source_box_initialization": True})}
    for candidate in context["candidates"]:
        files[f"configs/{candidate['id']}.json"] = _bytes(candidate["config"])
    if sum(map(len, files.values())) > MAX_RECEIPT_BYTES:
        raise ValueError("Conditional input receipts exceed their aggregate byte bound")
    out.mkdir(parents=True, mode=0o700)
    write_package_files(out, files)
    candidates = []
    for candidate in context["candidates"]:
        row = {key: candidate[key] for key in ("id", "assignment", "assumption_basis", "task_comparable", "task_scope")}
        row.update(status="not_attempted", lab_directory=None, reasons=[], scope=SCOPE)
        left = deadline - time.monotonic()
        if left < 1:
            row.update(status="deferred", reasons=["aggregate_time_bound"])
        else:
            try:
                built = prepare_exercise_scene(base_run, evidence_run, out / "configs" / (row["id"] + ".json"),
                    out / "candidates" / row["id"], source_mappings=source_mappings, full_reference=True,
                    train_settings=None, max_seconds=left, conditional_context_path=out / "conditional-context.json",
                    conditional_candidate_id=row["id"])
                row.update(status=built["status"], lab_directory=built.get("lab_directory"), reasons=built.get("reasons", []),
                    assembly_directory=str(out / "candidates" / row["id"]),
                    assembly_manifest_sha256=built.get("manifest_sha256"))
            except Exception as error:
                row.update(status="failed", reasons=[{"code": type(error).__name__, "message": str(error)[:1000]}])
        candidates.append(row)
    if (read_package_manifest(base_run)[0] != base_manifest or read_package_manifest(evidence_run)[0] != equipment_manifest
            or read_package_file(base_run, "scene.json", expected_sha256=base_files["scene.json"], max_bytes=16 << 20) != scene_raw
            or read_equipment_observations(evidence_run, equipment_files) != equipment_raw
            or Path(__file__).read_bytes() != implementation):
        raise ValueError("Conditional assembly inputs or implementation changed")
    for name, raw in proof.files.items():
        read_package_file(base_run, name, max_bytes=max(1, len(raw)), expected_sha256=sha256(raw).hexdigest())
    for item, path in zip(equivalents, source_mappings):
        for name in item.file_names:
            raw = item.raw_bytes(name)
            read_package_file(path, name, max_bytes=max(1, len(raw)), expected_sha256=sha256(raw).hexdigest())
    for name, raw in files.items():
        read_package_file(out, name, max_bytes=max(1, len(raw)), expected_sha256=sha256(raw).hexdigest())
    result = {"profile": PROFILE, "status": "candidates_ready" if any(row["status"] == "ready" for row in candidates) else "abstained",
        "scope": SCOPE, "candidates": candidates, "reasons": context["reasons"],
        "context_path": str(out / "conditional-context.json"), "seconds": time.monotonic() - started,
        "api_calls": 0, "inference_calls": 0, "physics_steps": 0, "training": False,
        "physical_validation": "unverified", "raw_identity_reclassified": False, "selected_candidate": None}
    files["result.json"] = _bytes(result)
    write_package_files(out, {"result.json": files["result.json"], "manifest.json": _bytes({"schema_version": 1,
        "status": result["status"], "seal_scope": "conditional input/result receipts; each assembly has its own ordinary seal",
        "files": {name: sha256(raw).hexdigest() for name, raw in files.items()}})})
    return result
