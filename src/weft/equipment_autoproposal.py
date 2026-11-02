"""Conditional source-image role hypotheses feeding the ordinary equipment builder.

Torso containment proposes correspondence, never physical identity or rigidity.
All source slots and unresolved alternatives survive; ambiguity prevents assembly.
"""

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import time

from weft.equipment_proposal_config import CONFIG_PROFILE, CONSENSUS_METHOD, validate_proposal_config
from weft.io import canonical_hash
from weft.shared_world_evidence import build_shared_world_binding
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files
from weft.world_evidence import load_world_evidence


PROFILE = "source-equipment-auto-proposal-v1"
POLICY = {"profile": "torso-box-conditional-roles-v1", "torso_indices": [11, 12, 23, 24],
          "minimum_confidence": .5, "minimum_common_frames": 3, "minimum_reference_coverage": .5,
          "minimum_torso_span_px": 8.,
          "body_containment_minimum": .9, "other_containment_maximum": .1, "minimum_margin": .8,
          "camera_distance_model_units": 2., "maximum_objects": 12, "maximum_frames": 1800}
SCOPE = ("Conditional image correspondence and authored rigid proxies, not verified identity, rigidity, "
         "camera calibration, depth, floor support, material, contact or task success.")
MAX_BYTES = 64 << 20
MAX_CONTEXT_BYTES = 8 << 20


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Automatic equipment proposal exceeded its deadline")


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def derive_equipment_config(documents, *, source_sha256, human_ref, equipment_document_sha256,
                            body_references, source_equivalences=(), deadline=None):
    """Return a config proposal or deterministic abstention; never mutate source documents."""
    deadline = min(time.monotonic() + 10, deadline) if deadline is not None else time.monotonic() + 10
    _check(deadline)
    registry = load_world_evidence(documents)
    human_ref = tuple(human_ref)
    if len(human_ref) != 2 or equipment_document_sha256 not in documents:
        raise ValueError("Exact body entity and equipment document are required")
    original = json.loads(registry.source_bytes(equipment_document_sha256))
    objects = original["objects"]
    if not 1 <= len(objects) <= POLICY["maximum_objects"] or original["source"]["decoded_frames"] > POLICY["maximum_frames"]:
        raise ValueError("Equipment inventory exceeds the complete bounded source contract")
    ids = [obj["id"] for obj in objects]
    refs = {"human": human_ref}
    for index, oid in enumerate(ids):
        if (equipment_document_sha256, oid) != human_ref:
            refs[f"raw-{index}"] = (equipment_document_sha256, oid)
    binding, construction_raw = build_shared_world_binding(documents, source_sha256=source_sha256,
        world_ids=list(refs), entity_refs=refs, source_equivalences=source_equivalences, deadline=deadline)
    construction = json.loads(construction_raw)
    sources = {s["document_sha256"]: s for s in construction["sources"]}
    equipment, body = sources[equipment_document_sha256], sources[human_ref[0]]
    indexed = {(s["document_sha256"], row["object_id"], row["frame"]): row
               for s in sources.values() for row in s["observations"]}
    if not isinstance(body_references, list) or not 3 <= len(body_references) <= 900:
        raise ValueError("Preserve all ordered original body reference identities")
    frames, previous = {}, -1
    for ref in body_references:
        identifier = ref.get("evidence_id", "")
        prefix = source_sha256 + ":frame:"
        if not isinstance(identifier, str) or not identifier.startswith(prefix) or not identifier[len(prefix):].isdigit():
            raise ValueError("Body reference requires original frame identity")
        frame = int(identifier[len(prefix):])
        row = indexed.get((*human_ref, frame))
        if (frame <= previous or row is None or row["admission"] != "included" or ref.get("time_s") != row["time_s"]
                or {"document_sha256": human_ref[0], "evidence_id": identifier} not in row["evidence_refs"]):
            raise ValueError("Body reference order or construction evidence differs")
        frames[frame] = row
        previous = frame
    width, height = equipment["image"]["width_px"], equipment["image"]["height_px"]
    inventory = []
    for oid in ids:
        _check(deadline)
        slots, common, contained = [], 0, 0
        exact = (equipment_document_sha256, oid) == human_ref
        for frame in range(binding["total_source_frames"]):
            _check(deadline)
            observed = indexed[(equipment_document_sha256, oid, frame)]
            pose = frames.get(frame)
            reason = ("partition_withheld" if observed["admission"] != "included" else
                      "not_body_reference" if pose is None else
                      "source_not_visible" if observed["visibility"] not in {"detected", "partial"} or pose["visibility"] not in {"detected", "partial"} else
                      "ambiguous_evidence" if len(observed["evidence_refs"]) != 1 or len(pose["evidence_refs"]) != 1 else None)
            count = None
            if reason is None and exact:
                common += 1
                contained += 1
                count = 4
            elif reason is None:
                box = observed["measurements"].get("box", {})
                points = pose["measurements"].get("landmarks", {})
                if ((box.get("coordinate_system"), box.get("unit"), box.get("status")) != ("image", "pixels", "inferred")
                        or (points.get("coordinate_system"), points.get("unit"), points.get("status")) != ("image", "pixels", "inferred")):
                    reason = "inferred_box_or_torso_unavailable"
                else:
                    by_id = {point["index"]: point for point in points.get("value", [])}
                    torso = [by_id.get(i) for i in POLICY["torso_indices"]]
                    if any(p is None or min(p.get("visibility", 0), p.get("presence", 0)) < POLICY["minimum_confidence"]
                           or not 0 <= p["position"][0] <= width or not 0 <= p["position"][1] <= height for p in torso):
                        reason = "confident_torso_unavailable"
                    elif math.hypot(*[(torso[0]["position"][axis] + torso[1]["position"][axis]
                                      - torso[2]["position"][axis] - torso[3]["position"][axis]) / 2
                                     for axis in (0, 1)]) < POLICY["minimum_torso_span_px"]:
                        reason = "degenerate_torso"
                    else:
                        x, y, w, h = box["value"]
                        if not 0 <= x < x + w <= width or not 0 <= y < y + h <= height:
                            reason = "inferred_box_outside_image"
                        else:
                            count = sum(x <= p["position"][0] <= x + w and y <= p["position"][1] <= y + h for p in torso)
                            common += 1
                            contained += count == 4
            slots.append({"frame": frame, "observation_id": observed["id"], "time_s": observed["time_s"],
                "body_time_s": pose["time_s"] if pose else None, "partition": observed["partition"],
                "admission": observed["admission"], "evidence_refs": observed["evidence_refs"],
                "reason": reason, "contained_torso_points": count})
        inventory.append({"object_id": oid, "entity_ref": {"document_sha256": equipment_document_sha256, "object_id": oid},
            "properties": next(o["properties"] for o in equipment["objects"] if o["id"] == oid),
            "exact_body_entity": exact, "common_frames": common, "reference_coverage": common / len(frames),
            "full_torso_containment_fraction": contained / common if common else None,
            "slots": slots, "unavailable_counts": dict(Counter(s["reason"] for s in slots if s["reason"])),
            "alternatives": ["body_correspondence", "authored_rigid_proxy", "unresolved"], "selected_role": "unresolved"})
    reasons = []
    for item in inventory:
        if item["common_frames"] < POLICY["minimum_common_frames"] or item["reference_coverage"] < POLICY["minimum_reference_coverage"]:
            reasons.append("insufficient_common_evidence:" + item["object_id"])
    candidates = [item for item in inventory if item["full_torso_containment_fraction"] is not None
                  and item["full_torso_containment_fraction"] >= POLICY["body_containment_minimum"]]
    if len(candidates) > 1:
        reasons.append("multiple_body_correspondence_candidates")
    chosen = candidates[0] if len(candidates) == 1 else None
    for item in inventory:
        score = item["full_torso_containment_fraction"]
        if item is chosen:
            item["selected_role"] = "exact_bound_body" if item["exact_body_entity"] else "conditional_body_correspondence"
        elif score is not None and score <= POLICY["other_containment_maximum"]:
            item["selected_role"] = "authored_rigid_proxy"
        else:
            reasons.append("unresolved_body_correspondence:" + item["object_id"])
        if chosen and item is not chosen and score is not None and chosen["full_torso_containment_fraction"] - score < POLICY["minimum_margin"]:
            reasons.append("insufficient_role_margin:" + item["object_id"])
    ambiguous = [r for r in equipment["relationships"] if r["type"] == "possible_association_ambiguity"]
    if ambiguous:
        reasons.append("unresolved_source_identity_relationships")
    proxies = sorted(item["object_id"] for item in inventory if item["selected_role"] == "authored_rigid_proxy")
    if not proxies:
        reasons.append("no_supported_rigid_proxy_candidates")
    config = None
    if not reasons:
        config = validate_proposal_config({"schema_version": 1, "profile": CONFIG_PROFILE,
            "source_sha256": source_sha256, "camera": {"mode": "fixed_pelvis_perspective_prior",
                "distance_model_units": POLICY["camera_distance_model_units"]},
            "geometry_prior": "floor_box_depth_equals_width", "equipment": [
                {"object_id": oid, "frame_selection": {"method": CONSENSUS_METHOD}} for oid in proxies],
            "target_object_id": proxies[0]})
    report = {"schema_version": 1, "profile": PROFILE, "status": "config_proposed" if config else "abstained",
        "scope": SCOPE, "policy": dict(POLICY), "source_sha256": source_sha256,
        "input_documents": sorted(documents), "body_reference_sha256": canonical_hash(body_references),
        "source_binding_sha256": canonical_hash(binding), "construction_evidence_sha256": sha256(construction_raw).hexdigest(),
        "inventory": inventory, "relationships": equipment["relationships"], "reasons": sorted(set(reasons)),
        "all_original_objects_accounted": len(inventory) == len(objects), "source_slots_rewritten": 0,
        "physical_identity_verified": False, "source_contact_verified": False, "physical_validation": "unverified",
        "authored_defaults": {"camera_distance": "2 original body-model units, never calibrated or fitted",
            "depth": "equals width", "shape": "solid floor-supported axis-aligned box",
            "mass_and_friction": "Existing equipment constructor priors, never inferred",
            "target": "Lexicographically first proxy ID; authored task choice, not inferred intent",
            "initial_motion": "zero; later equipment initialization implies static prefix extrapolation"},
        "config": config}
    if len(_json(report)) > MAX_CONTEXT_BYTES:
        raise ValueError("Automatic equipment report exceeds its byte bound")
    _check(deadline)
    return report, construction_raw


def prepare_automatic_equipment_scene(base_run, evidence_run, out, *, source_mappings=(), max_seconds=90):
    """Derive one config, then invoke the existing strict proposal/world/lab stages once."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 180:
        raise ValueError("Automatic equipment assembly requires a 1–180 second bound")
    if not isinstance(source_mappings, (tuple, list)) or len(source_mappings) > 7:
        raise ValueError("At most seven verified source mappings are supported")
    from weft.shared_world_packaging import collect_shared_source_files
    from weft.shared_world_pipeline import read_equipment_observations
    from weft.source_equivalence import load_source_equivalence

    base_run, evidence_run, out = map(lambda p: Path(p).absolute(), (base_run, evidence_run, out))
    inputs = [base_run, evidence_run, *(Path(p).absolute() for p in source_mappings)]
    if any(".." in p.parts or any(parent.is_symlink() for parent in (p, *p.parents)) for p in [out, *inputs]):
        raise ValueError("Automatic equipment paths cannot contain traversal or links")
    if out.exists() or any(out.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(out.resolve()) for p in inputs):
        raise ValueError("Choose a new automatic proposal directory outside every input")
    started = time.monotonic()
    deadline = started + max_seconds
    implementation = Path(__file__).read_bytes()
    base_manifest, base_files = read_package_manifest(base_run)
    scene_raw = read_package_file(base_run, "scene.json", expected_sha256=base_files["scene.json"], max_bytes=16 << 20)
    scene = json.loads(scene_raw)
    proof = collect_shared_source_files(scene, base_run, expected_manifest_sha256=sha256(base_manifest).hexdigest(), deadline=deadline)
    human = next(r for r in scene["source_binding"]["entity_bindings"] if r["world_entity_id"] == "human")
    human_sha = human["observation_document_sha256"]
    human_raw = proof.files[scene["source_binding"]["evidence_documents"][human_sha]]
    equipment_manifest, equipment_files = read_package_manifest(evidence_run)
    equipment_raw = read_equipment_observations(evidence_run, equipment_files)
    equipment_sha = sha256(equipment_raw).hexdigest()
    documents = {human_sha: human_raw, equipment_sha: equipment_raw}
    remaining = MAX_BYTES - sum(map(len, documents.values()))
    equivalents = []
    for path in source_mappings:
        _check(deadline)
        item = load_source_equivalence(path, max_input_bytes=remaining, deadline=deadline)
        remaining -= item.total_input_bytes
        equivalents.append(item)
    report, construction = derive_equipment_config(documents, source_sha256=scene["source_binding"]["source_sha256"],
        human_ref=(human_sha, human["observation_entity_id"]), equipment_document_sha256=equipment_sha,
        body_references=scene["human_spec"]["reference_evidence"], source_equivalences=equivalents, deadline=deadline)
    files = {"auto-proposal.json": _json(report), "construction-evidence.json": construction,
             "inputs/base-manifest.json": base_manifest, "inputs/equipment-manifest.json": equipment_manifest,
             "inputs/base-scene.json": scene_raw, "implementation/equipment_autoproposal.py": implementation,
             **{f"inputs/observations/{sha}.json": raw for sha, raw in documents.items()}}
    if report["config"] is not None:
        files["proposal-config.json"] = _json(report["config"])
    if sum(map(len, files.values())) > MAX_BYTES:
        raise ValueError("Automatic equipment input receipt exceeds its byte bound")
    _check(deadline)
    out.mkdir(parents=True, mode=0o700)
    write_package_files(out, files)
    result = {"profile": PROFILE, "status": report["status"], "scope": SCOPE, "promoted": False,
              "api_calls": 0, "inference_calls": 0, "source_fits": 0, "automatic_config": report["config"] is not None,
              "reasons": report["reasons"], "proposal_report": str(out / "auto-proposal.json")}
    if report["config"] is not None:
        from weft.exercise_scene_pipeline import prepare_exercise_scene

        _check(deadline)
        built = prepare_exercise_scene(base_run, evidence_run, out / "proposal-config.json", out / "build",
            source_mappings=source_mappings, full_reference=True, train_settings=None,
            max_seconds=max(1., deadline - time.monotonic()), auto_context_path=out / "auto-proposal.json")
        result.update(status=built["status"], assembly=built)
        result.update({key: built[key] for key in ("reasons", "lab_directory", "launch_arguments") if key in built})
    if (read_package_manifest(base_run)[0] != base_manifest or read_package_manifest(evidence_run)[0] != equipment_manifest
            or read_package_file(base_run, "scene.json", expected_sha256=base_files["scene.json"], max_bytes=16 << 20) != scene_raw
            or read_equipment_observations(evidence_run, equipment_files) != equipment_raw):
        raise ValueError("Automatic equipment input changed")
    if Path(__file__).read_bytes() != implementation:
        raise ValueError("Automatic equipment implementation changed during assembly")
    _check(deadline)
    result["seconds"] = time.monotonic() - started
    write_package_files(out, {"result.json": _json(result)})
    files["result.json"] = _json(result)
    write_package_files(out, {"manifest.json": _json({"schema_version": 1, "status": result["status"],
        "seal_scope": "automatic_input_receipts; assembly has its own ordinary manifest",
        "files": {name: sha256(raw).hexdigest() for name, raw in files.items()}})})
    return result
