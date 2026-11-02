"""Inspect saved kinematic equipment overlaps without executing the scene."""

from collections import Counter
import math
import re

from weft.io import canonical_hash


PROFILE = "saved-reference-equipment-overlap-v1"
MAX_REFERENCES, MAX_CONTACTS, MAX_EXAMPLES = 900, 256, 12
HUMAN_GEOMS = {"torso_geom", "head_geom"} | {
    f"{side}_{joint}_geom" for side in ("left", "right") for joint in ("shoulder", "elbow", "hip", "knee")}
KINDS = {"human_floor", "human_static", "human_object", "object_floor", "object_static", "object_object"}
SCOPE = ("Saved kinematic poses with equipment fixed at its nominal placement; no new geometry or physics check. "
         "Only retained reference poses are inspected, not missing frames or the intervals between poses. "
         "A negative recorded collision distance is shape overlap, not measured anatomical contact, grasp, load, "
         "controller failure or proof that no legal controller can succeed. Kinematic solver forces are not interpreted. "
         "Task gates, lab availability and the earlier floor/static/initial-placement audit are unchanged.")


def _require(condition, message):
    if not condition:
        raise ValueError("Saved reference equipment: " + message)


def _number(value, label, *, low=-1e6, high=1e6):
    _require(type(value) in (int, float) and low <= value <= high and math.isfinite(value), label)
    return value


def _text(value):
    _require(type(value) is str and 0 < len(value) <= 160 and all(ord(c) >= 32 for c in value), "bounded identity required")
    return value


def _vector(value, size):
    _require(type(value) is list and len(value) == size, "vector dimensions differ")
    return [_number(item, "finite vector required") for item in value]


def summarize_reference_equipment(scene, reference, *, check=lambda: None):
    """The caller verifies selected bytes; this checks their scene/clock/placement joins."""
    check()
    _require(type(scene) is dict and type(reference) is dict, "scene and reference objects required")
    if scene.get("profile") not in ("shared-world-v1", "shared-world-v2"):
        return {"profile": PROFILE, "status": "unavailable", "physical_validation": "unverified",
                "reason": "Saved equipment-overlap inspection is not qualified for this scene profile.", "scope": SCOPE}
    human = scene.get("human_spec", {})
    _require(type(human) is dict, "body metadata required")
    poses, evidence = human.get("reference"), human.get("reference_evidence")
    _require(type(poses) is list and 1 <= len(poses) <= MAX_REFERENCES
             and all(type(pose) is dict for pose in poses), "bounded complete body references required")
    roots = human.get("root_reference")
    _require(type(roots) is list and len(roots) == len(poses) and all(type(row) is dict for row in roots),
             "complete reference-root metadata required")
    _require(reference.get("schema_version") == 1 and type(reference["schema_version"]) is int
             and reference.get("profile") == "shared-world-kinematic-reference-v1", "unsupported reference document")
    frames = reference.get("frames")
    _require(type(frames) is list and len(frames) == len(poses)
             and all(type(reference.get(key)) is int and reference[key] == len(poses)
                     for key in ("reference_poses_total", "reference_poses_included")), "missing or clipped reference poses")
    coordinate = scene.get("coordinate_frame", {})
    _require(type(coordinate) is dict and type(coordinate.get("scale_bridge")) is dict
             and type(coordinate.get("units")) is dict, "coordinate metadata required")
    scale = _number(coordinate.get("scale_bridge", {}).get("body_height_to_world_scale"), "positive declared scale required", low=1e-6)
    _require(coordinate.get("axes") == "x_right_y_depth_z_up", "unsupported coordinate axes")
    world_unit = _text(coordinate.get("units", {}).get("length"))
    origin = _vector(coordinate["scale_bridge"].get("origin_world"), 3)
    source = scene.get("source_binding", {})
    _require(type(source) is dict, "source binding required")
    source_sha = source.get("source_sha256")
    if source.get("mode") == "source_qualified":
        _require(type(source_sha) is str and re.fullmatch(r"[0-9a-f]{64}", source_sha)
                 and human.get("source_sha256") == source_sha and type(evidence) is list
                 and len(evidence) == len(poses), "source/reference provenance differs")
    else:
        _require(source.get("mode") == "analytic_fixture" and source_sha is None
                 and human.get("source_sha256") is None, "unsupported source binding")
    objects = scene.get("objects")
    _require(type(objects) is list and 1 <= len(objects) <= 12 and all(type(obj) is dict for obj in objects), "bounded object inventory required")
    by_id = {_text(obj.get("id")): obj for obj in objects}
    _require(len(by_id) == len(objects) and not (set(by_id) & (HUMAN_GEOMS | {"mobile_floor"})), "ambiguous object identities")
    _require(all(obj.get("role") in ("passive", "static", "controlled") for obj in objects), "unsupported object role")
    dynamic = {name for name, obj in by_id.items() if obj["role"] != "static"}
    _require(type(scene.get("action_contract")) is dict, "action metadata required")
    action_names = scene["action_contract"].get("names")
    _require(type(action_names) is list and 19 <= len(action_names) <= 64, "bounded action metadata required")
    totals = {name: {"object_id": name, "poses_with_contacts": 0, "poses_with_overlap": 0,
                    "negative_contact_records": 0, "max_penetration_world": 0., "max_penetration_model_units": 0.,
                    "first_overlap": None, "human_geometries": {}} for name in sorted(dynamic)}
    previous_time, previous_frame = -math.inf, -1
    first_time = _number(poses[0].get("time_s"), "finite reference clock required", low=0., high=86400.)
    examples, negative_records, overlapping_poses, contact_poses, total_records = [], 0, 0, 0, 0
    for index, (pose, frame) in enumerate(zip(poses, frames)):
        check()
        _require(type(pose) is dict and type(frame) is dict, "reference row object required")
        timestamp = _number(pose.get("time_s"), "ordered finite source clock required", low=0., high=86400.)
        _require(timestamp > previous_time and frame.get("profile") == scene["profile"]
                 and frame.get("coordinate_frame") == coordinate and frame.get("source_time_s") == timestamp
                 and math.isclose(_number(frame.get("time_s"), "elapsed reference clock required", low=0.),
                                  timestamp - first_time, rel_tol=0., abs_tol=1e-10), "scene/reference clock or frame differs")
        previous_time = timestamp
        _require(type(frame.get("steps")) is int and frame["steps"] == index
                 and type(frame.get("episode_steps")) is int and frame["episode_steps"] == 0
                 and type(frame.get("substeps")) is int and frame["substeps"] == 0
                 and type(frame.get("human")) is dict and frame["human"].get("reference") is True
                 and all(frame.get(key) is False for key in ("success", "terminated", "truncated")), "only complete zero-step kinematic frames are admitted")
        joints = pose.get("joints")
        _require(type(joints) is dict and len(joints) == 19 and frame["human"].get("angles") == joints,
                 "saved body angles differ from the reference")
        for value in joints.values():
            _number(value, "finite reference angles required")
        mobility = frame["human"].get("mobility")
        _require(type(mobility) is dict and roots[index].get("time_s") == timestamp
                 and _vector(frame["human"].get("origin"), 3) == origin
                 and all(math.isclose(a, b, rel_tol=0., abs_tol=1e-10) for a, b in
                         zip(_vector(mobility.get("translation"), 3), _vector(roots[index].get("translation"), 3))),
                 "saved body root or origin differs from the reference")
        _require(not any(_vector(frame.get("normalized_action"), len(action_names)))
                 and not any(_vector(frame.get("virtual_force"), 3)), "kinematic frame contains applied control")
        current = frame.get("objects")
        _require(type(current) is list and len(current) == len(objects)
                 and all(type(obj) is dict and type(obj.get("id")) is str for obj in current)
                 and len({obj.get("id") for obj in current}) == len(current)
                 and {obj.get("id") for obj in current} == set(by_id), "reference object inventory differs")
        for obj in current:
            original = by_id[obj["id"]]
            _require(obj.get("role") == original["role"], "reference object role differs")
            _require(all(obj.get(key) == original[key] for key in ("shape", "size", "radius", "geometry")
                         if key in original), "reference object geometry differs")
            for field, size in (("position", 3), ("quaternion", 4)):
                _require(all(math.isclose(a, b, rel_tol=0., abs_tol=1e-10) for a, b in
                             zip(_vector(obj.get(field), size), _vector(original.get(field), size))), "equipment is not at nominal placement")
            _require(not any(_vector(obj.get("virtual_force"), 3)), "equipment has virtual force")
        provenance = {"reference_index": index, "source_time_s": timestamp, "source_frame": None, "evidence_id": None}
        if source_sha is not None:
            item = evidence[index]
            _require(type(item) is dict and item.get("time_s") == timestamp, "reference evidence clock differs")
            evidence_id = _text(item.get("evidence_id"))
            match = re.fullmatch(re.escape(source_sha) + r":frame:(0|[1-9][0-9]*)", evidence_id)
            _require(match is not None, "exact source-frame identity required")
            source_frame = int(match[1])
            _require(previous_frame < source_frame < 1000000, "source-frame order differs")
            previous_frame = source_frame
            provenance.update(source_frame=source_frame, evidence_id=item["evidence_id"])
        contacts = frame.get("contact_details")
        _require(type(contacts) is list and len(contacts) <= MAX_CONTACTS
                 and type(frame.get("contact_count")) is int and frame["contact_count"] == len(contacts), "contact records missing or out of bounds")
        touched, overlapped, counts = set(), set(), Counter()
        for contact in contacts:
            _require(type(contact) is dict and contact.get("kind") in KINDS, "unsupported contact kind")
            pair = contact.get("objects")
            _require(type(pair) is list and len(pair) == 2 and all(type(name) is str for name in pair)
                     and pair == sorted(set(pair)) and all(name in by_id or name in HUMAN_GEOMS or name == "mobile_floor" for name in pair), "unknown contact identity")
            human_names = [name for name in pair if name in HUMAN_GEOMS]
            equipment_names = [name for name in pair if name in by_id]
            if human_names and equipment_names:
                expected_kind = "human_object" if equipment_names[0] in dynamic else "human_static"
                _require(contact["kind"] == expected_kind, "human/equipment contact classification differs")
            elif contact["kind"] in ("human_object", "human_static"):
                _require(False, "contact classification lacks human/equipment identities")
            distance = _number(contact.get("distance"), "finite contact distance required")
            _vector(contact.get("position"), 3)
            counts[contact["kind"]] += 1
            if contact["kind"] != "human_object":
                continue
            identifier, geometry = equipment_names[0], human_names[0]
            touched.add(identifier)
            if distance >= 0.:
                continue
            overlapped.add(identifier)
            row = totals[identifier]
            row["negative_contact_records"] += 1
            row["max_penetration_world"] = max(row["max_penetration_world"], -distance)
            row["max_penetration_model_units"] = row["max_penetration_world"] / scale
            row["human_geometries"][geometry] = row["human_geometries"].get(geometry, 0) + 1
            item = {**provenance, "object_id": identifier, "human_geom": geometry,
                    "distance_world": distance, "penetration_model_units": -distance / scale,
                    "position_world": list(contact["position"])}
            if row["first_overlap"] is None:
                row["first_overlap"] = item
            negative_records += 1
            if len(examples) < MAX_EXAMPLES:
                examples.append(item)
        summary = frame.get("contact_summary")
        _require(type(summary) is dict and all(type(summary.get(kind)) is int and summary[kind] == counts[kind] for kind in KINDS), "saved contact counts contradict records")
        for name in touched: totals[name]["poses_with_contacts"] += 1
        for name in overlapped: totals[name]["poses_with_overlap"] += 1
        contact_poses += bool(touched)
        overlapping_poses += bool(overlapped)
        total_records += len(contacts)
    check()
    return {"profile": PROFILE, "status": "overlap_reported" if negative_records else "no_overlap_in_saved_poses",
            "scene_sha256": canonical_hash(scene), "source_sha256": source_sha, "physical_validation": "unverified",
            "reference_poses": len(frames), "dynamic_objects": len(dynamic), "contact_records_inspected": total_records,
            "poses_with_equipment_contacts": contact_poses, "poses_with_equipment_overlap": overlapping_poses,
            "negative_contact_records": negative_records, "objects": list(totals.values()),
            "examples": examples, "examples_omitted": max(0, negative_records - len(examples)),
            "distance_units": world_unit, "world_units_per_model_unit": scale,
            "clock_scope": "Declared source-reference times, not verified native-video synchronization",
            "geometric_overlap_rule": "recorded distance < 0; descriptive, not a new success threshold",
            "physics_steps": 0, "scope": SCOPE}
