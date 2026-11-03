"""One initial free-body state, then ordinary dynamics; image trajectories never drive physics."""

from copy import deepcopy
import json

from weft.io import canonical_hash
from weft.moving_equipment_proposal import MAX_BYTES, PROFILE, WORLD_PROFILE, _check, _deadline, derive_moving_proposals


def construct_moving_equipment_world(inputs, bundle, candidate_id, *, deadline=None):
    """Return an unscored validated scene; the caller owns exact source packaging."""
    from weft.dynamic_native_interval import dynamic_native_interval
    from weft.moving_equipment_inputs import validate_moving_equipment_inputs
    from weft.shared_world import full_reference_horizon, validate_shared_world
    from weft.shared_world_reconstruction import _check_initial_geometry

    stop = _deadline(deadline)
    packet = validate_moving_equipment_inputs(inputs, deadline=stop)
    if not isinstance(bundle, dict) or bundle.get("profile") != PROFILE:
        raise ValueError("Moving world requires the complete versioned proposal bundle")
    size = 0
    for chunk in json.JSONEncoder(allow_nan=False).iterencode(bundle):
        _check(stop)
        size += len(chunk.encode())
        if size > MAX_BYTES:
            raise ValueError("Moving world proposal exceeds its byte bound")
    expected = derive_moving_proposals(packet, bundle.get("settings"), max_candidates=bundle.get("maximum_candidates"), deadline=stop)
    if canonical_hash(bundle) != canonical_hash(expected):
        raise ValueError("Moving world proposal differs from exact source/prior rederivation")
    matches = [item for item in bundle["candidates"] if item["id"] == candidate_id]
    if len(matches) != 1 or matches[0]["status"] != "conditional_free_body_hypothesis":
        raise ValueError("Moving world requires one explicit available conditional candidate")
    candidate = matches[0]
    base = packet.base_scene
    scene = deepcopy(base)
    if (base["source_binding"]["mode"] != "source_qualified" or base["controlled_id"] != "human"
            or base["task"]["start_offset_s"] != 0
            or any(row["provenance_mode"] != "authored_addition" for row in base["source_binding"]["entity_bindings"]
                   if row["world_entity_id"] != "human")):
        raise ValueError("Moving world base must preserve the original body start without prior reconstructed equipment")
    bridge = scene["coordinate_frame"]["scale_bridge"]
    scale, origin = bridge["body_height_to_world_scale"], bridge["origin_world"]
    tracks = {row["id"]: row for row in bundle["track_proposals"]}
    settings = bundle["settings"]
    proposal_sha256 = canonical_hash(bundle)
    objects = []
    for item in candidate["objects"]:
        _check(stop)
        track = tracks[item["track_proposal_id"]]
        initial = track["initial_state_hypothesis"]
        objects.append({"id": item["object_id"], "label": f"{item['object_id']} conditional sphere", "role": "passive",
            "shape": "sphere", "position": [origin[i] + scale * value for i, value in enumerate(initial["position_model_units"])],
            "radius": scale * track["shape_prior"]["radius_model_units"], "quaternion": [1., 0., 0., 0.],
            "velocity": [scale * value for value in initial["velocity_model_units_per_second"]],
            "angular_velocity": list(initial["angular_velocity_rad_per_second"]),
            "mass": settings["mass_kg"], "friction": list(settings["friction"]), "color": "#965644",
            "provenance": {"profile": WORLD_PROFILE, "proposal_sha256": proposal_sha256,
                "candidate_id": candidate_id, "track_proposal_id": track["id"], "entity_ref": track["entity_ref"],
                "motion_evidence_sha256": bundle["input_pins"]["motion_evidence_sha256"],
                "camera_prior": bundle["camera_hypothesis"], "shape_prior": track["shape_prior"],
                "initial_state_hypothesis": initial, "identity_assignment": candidate["identity_assignment"],
                "physical_validation": "unverified", "source_contact_verified": False, "grasp_supported": False,
                "trajectory_driving": False, "hidden_assistance": False, "task_success_available": False}})
    names = {obj["id"] for obj in objects}
    binding = packet.source_binding
    binding["entity_bindings"] = [row for row in binding["entity_bindings"] if row["world_entity_id"] in {"human", *names}]
    scene.pop("action_contract", None)
    scene.update(profile="shared-world-v2", objects=objects, source_binding=binding, learned_policy_available=False,
                 controlled_id="human", target=list(objects[0]["position"]))
    scene.pop("inventory", None)
    scene["task"].update(type="coupled_human_object_v1", authored_addition=True, start_offset_s=0.)
    if not isinstance(scene["task"].get("object_goal"), dict):
        raise ValueError("Moving exploration requires the existing coupled engine scaffold, not a new task")
    scene["task"]["object_goal"]["target_object_id"] = objects[0]["id"]
    marker = dynamic_native_interval(scene["human_spec"], scene["dt"], dynamic_object_ids=sorted(names))
    if marker is None:
        scene["task"].pop("native_interval", None)
        scene["task"]["horizon_steps"] = full_reference_horizon(scene["human_spec"]["reference"], scene["dt"])
    else:
        scene["task"].update(native_interval=marker, horizon_steps=marker["full_control_steps"] + 1)
    scene["assumptions"] = list(base["assumptions"]) + [
        "Original authored props are replaced by conditional moving-equipment free bodies; all original body geometry, roots and clocks remain unchanged.",
        "Only initial object states are installed. Complete image-ray proposals remain evidence sidecars, never forces or pose playback.",
        "Initial depth, sphere shape, orientation, mass, friction and prefix state are authored hypotheses; no grasp or physical identity is recovered.",
        "Exploration is unscored: existing coupled-engine rewards and terminal flags are diagnostics, not reconstructed-motion or exercise success.",
    ]
    _check(stop)
    scene = validate_shared_world(scene)
    first_frame = packet.body_references[0]["source_frame"]
    checks = _check_initial_geometry(scene, {first_frame: 0})
    _check(stop)
    for obj in scene["objects"]:
        obj["provenance"]["initial_geometry_checks"] = [row for row in checks if row["object_id"] == obj["id"]]
    result = validate_shared_world(scene)
    if (result["human_spec"] != base["human_spec"] or result["coordinate_frame"] != base["coordinate_frame"]
            or any(result[key] != base[key] for key in ("dt", "substeps", "gravity", "bounds", "contact"))):
        raise ValueError("Moving construction changed the protected body/world contract")
    _check(stop)
    return result
