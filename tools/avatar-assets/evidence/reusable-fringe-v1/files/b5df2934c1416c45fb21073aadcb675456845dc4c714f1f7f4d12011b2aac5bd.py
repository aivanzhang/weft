"""Opt-in foreign-weight removal for declared rigid surfaces; not a quality verdict."""

from __future__ import annotations

import hashlib
import re
import time

import numpy as np


VERSION = "stable-role-projection-v1"
MAX_VERTICES = 100000
MAX_BONES = 64
MAX_ROLES = 32
MAX_RECORDED_LOSSES = 100000
CORE_THRESHOLD = .5
FRINGE_THRESHOLD = .05
MIN_STABLE_VERTICES = 8
_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, *, integer):
    if isinstance(value, np.ndarray):
        _require(value.ndim == 2 and 0 < len(value) <= MAX_VERTICES and value.shape[1] == 4,
                 "Skin arrays must contain 1..100000 rows of four slots")
        _require(value.dtype.kind in ("iu" if integer else "iuf"), "Skin values must be numeric and nonboolean")
    else:
        _require(isinstance(value, (list, tuple)) and 0 < len(value) <= MAX_VERTICES,
                 "Skin arrays must contain 1..100000 rows")
        allowed = (int, np.integer) if integer else (int, float, np.integer, np.floating)
        for row in value:
            _require(isinstance(row, (list, tuple)) and len(row) == 4, "Skin rows require four slots")
            _require(all(isinstance(item, allowed) and not isinstance(item, (bool, np.bool_)) for item in row),
                     "Skin values must be numeric and nonboolean")
    array = np.asarray(value)
    _require(array.dtype.kind in ("iu" if integer else "iuf"), "Skin values exceed supported numeric bounds")
    _require(np.isfinite(array).all(), "Skin values must be finite")
    return array


def _digest(array, dtype):
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes(order="C")).hexdigest()


def _selection(weights, joints, bone_indices, threshold):
    return np.sum(weights * np.isin(joints, bone_indices), axis=1) >= threshold


def _normalized(weights):
    return weights / weights.sum(axis=1, keepdims=True)


def _hierarchy(bone_names, bone_parents):
    _require(isinstance(bone_names, (list, tuple)) and 1 <= len(bone_names) <= MAX_BONES,
             "Require 1..64 actual bone names")
    _require(all(isinstance(name, str) and 0 < len(name) <= 128 and not any(ord(c) < 32 for c in name)
                 for name in bone_names) and len(set(bone_names)) == len(bone_names),
             "Bone names must be unique bounded strings")
    _require(isinstance(bone_parents, dict) and set(bone_parents) == set(bone_names),
             "Parent mapping must cover the exact actual bones")
    for name in bone_names:
        seen, current = set(), name
        while current is not None:
            _require(isinstance(current, str) and current in bone_parents and current not in seen,
                     "Invalid or cyclic actual bone hierarchy")
            seen.add(current)
            current = bone_parents[current]
    return {name: index for index, name in enumerate(bone_names)}


def _roles(roles, bone_names, bone_parents, lookup):
    _require(isinstance(roles, list) and 1 <= len(roles) <= MAX_ROLES, "Require 1..32 bound roles")
    prepared, seen = [], set()
    for role in roles:
        _require(isinstance(role, dict) and set(role) == {"state_id", "role_id", "bones", "stable"},
                 "Role requires exact state_id, role_id, bones and stable fields")
        _require(all(isinstance(role[key], str) and _ID.fullmatch(role[key]) for key in ("state_id", "role_id")),
                 "State and role IDs must be bounded safe slugs")
        key = (role["state_id"], role["role_id"])
        _require(key not in seen and type(role["stable"]) is bool, "Duplicate role or invalid stability flag")
        seen.add(key)
        bones = role["bones"]
        _require(isinstance(bones, list) and 1 <= len(bones) <= MAX_BONES
                 and all(isinstance(name, str) and name in lookup for name in bones)
                 and len(set(bones)) == len(bones), "Role targets must be unique actual bones")
        targets = frozenset(bones)
        descendants = []
        for name in bone_names:
            current = name
            while current is not None and current not in targets:
                current = bone_parents[current]
            if current is not None:
                descendants.append(lookup[name])
        prepared.append({**role, "bones": list(bones), "target_indices": sorted(lookup[name] for name in bones),
                         "fringe_indices": descendants, "target_set": targets})
    _require(any(role["stable"] for role in prepared), "At least one explicit stable role is required")
    return prepared


def project_stable_roles(joints, weights, bone_names, roles, *, bone_parents, surface_kind):
    """Return a weight copy and evidence; require explicit rigid-surface opt-in.

    Roles are exact {state_id, role_id, bones, stable} records for *all* action roles.
    Core membership uses direct target weight >=0.5; fringe uses descendants >=0.05.
    Multi-bone projection removes foreign weights but does not prove co-rigidity.
    """
    start = time.monotonic()
    _require(surface_kind == "rigid_hard_surface", "Projection requires explicit rigid_hard_surface opt-in")
    lookup = _hierarchy(bone_names, bone_parents)
    prepared = _roles(roles, bone_names, bone_parents, lookup)
    joint_array, source = _array(joints, integer=True), _array(weights, integer=False)
    _require(joint_array.shape == source.shape, "Joint and weight shapes differ")
    _require(np.all((joint_array >= 0) & (joint_array < len(bone_names))), "Joint index outside actual bones")
    _require(np.all((source >= 0) & (source <= 1.001))
             and np.all(np.abs(source.sum(axis=1, dtype=np.float64)-1) <= .001), "Invalid normalized skin weights")
    joint_array = np.asarray(joint_array, dtype=np.int32)
    source = np.asarray(source, dtype=np.float64)
    before = _normalized(source)
    for role in prepared:
        role["core"] = _selection(before, joint_array, role["target_indices"], CORE_THRESHOLD)
        role["fringe"] = _selection(before, joint_array, role["fringe_indices"], FRINGE_THRESHOLD)
        if role["stable"]:
            _require(int(role["core"].sum()) >= MIN_STABLE_VERTICES,
                     "Stable role lacks eight majority-weighted vertices")

    # Freeze ownership before renormalizing, including equal-weight boundary ties.
    stable_groups = {}
    for role in prepared:
        if not role["stable"]:
            continue
        for other in prepared:
            if role is other:
                continue
            shared = np.any(role["core"] & other["core"])
            _require(not shared or role["target_set"] == other["target_set"] and other["stable"],
                     "Conflicting or tied role ownership at frozen majority membership")
        stable_groups.setdefault(role["target_set"], role)

    result, protected = source.copy(), np.zeros(len(source), dtype=bool)
    for role in stable_groups.values():
        selected = role["core"]
        target_slots = np.isin(joint_array[selected], role["target_indices"])
        retained = source[selected] * target_slots
        # Retain target-bone ratios; do not substitute a guessed anatomical bone.
        result[selected] = retained / retained.sum(axis=1, keepdims=True)
        protected |= selected
    after = _normalized(result)
    changed = np.any(result != source, axis=1)
    _require(not np.any(changed & ~protected) and np.isfinite(result).all(), "Projection changed an unprotected vertex")
    _require(np.all((source > 0) | (result == 0)), "Projection introduced a new influence")
    reports, recorded_losses = [], 0
    for role in prepared:
        core_after = _selection(after, joint_array, role["target_indices"], CORE_THRESHOLD)
        _require(np.array_equal(role["core"], core_after), "Projection changed a frozen role core membership")
        fringe_after = _selection(after, joint_array, role["fringe_indices"], FRINGE_THRESHOLD)
        lost, gained = role["fringe"] & ~fringe_after, ~role["fringe"] & fringe_after
        _require(not np.any(gained), "Projection gained fringe membership")
        _require(not np.any(lost & ~protected), "Projection lost fringe outside protected stable membership")
        old_fringe = np.sum(before * np.isin(joint_array, role["fringe_indices"]), axis=1)
        new_fringe = np.sum(after * np.isin(joint_array, role["fringe_indices"]), axis=1)
        removed_fringe_slots = (source > 0) & (result == 0) & np.isin(joint_array, role["fringe_indices"])
        _require(np.all(np.any(removed_fringe_slots[lost], axis=1)) and np.all(new_fringe[lost] < old_fringe[lost]),
                 "Fringe loss is not explained by removed foreign influence")
        lost_ids = np.flatnonzero(lost)
        recorded_losses += len(lost_ids)
        _require(recorded_losses <= MAX_RECORDED_LOSSES, "Recorded fringe-loss index budget exceeded")
        core_ids, fringe_ids, after_ids = map(np.flatnonzero, (role["core"], role["fringe"], fringe_after))
        mode = ("single_bone_rigid_binding" if len(role["bones"]) == 1 else
                "foreign_influence_removal_not_verified_rigid") if role["stable"] else "unchanged_role_contract"
        reports.append({"state_id": role["state_id"], "role_id": role["role_id"], "bones": role["bones"],
                        "stable": role["stable"], "mode": mode, "co_rigid_verified": False,
                        "core_vertices_before": len(core_ids), "core_vertices_after": int(core_after.sum()),
                        "core_indices_before_sha256": _digest(core_ids, "<i4"),
                        "core_indices_after_sha256": _digest(np.flatnonzero(core_after), "<i4"),
                        "fringe_bones": [bone_names[index] for index in role["fringe_indices"]],
                        "fringe_vertices_before": len(fringe_ids), "fringe_vertices_after": len(after_ids),
                        "fringe_indices_before_sha256": _digest(fringe_ids, "<i4"),
                        "fringe_indices_after_sha256": _digest(after_ids, "<i4"),
                        "lost_fringe_count": len(lost_ids), "lost_fringe_indices": lost_ids.tolist(),
                        "lost_fringe_indices_sha256": _digest(lost_ids, "<i4"), "gained_fringe_count": 0})
    report = {"schema_version": 1, "method": VERSION, "status": "projected_diagnostic_only",
              "surface_kind": surface_kind, "surface_classification_verified": False,
              "vertices": len(source), "bones": len(bone_names), "bone_names": list(bone_names),
              "core_threshold": CORE_THRESHOLD, "fringe_threshold": FRINGE_THRESHOLD,
              "protected_vertices": int(protected.sum()), "changed_vertices": int(changed.sum()),
              "removed_influences": int(np.sum((source > 0) & (result == 0))),
              "removed_source_weight_sum": float(np.sum(source[(source > 0) & (result == 0)])),
              "input_joints_sha256": _digest(joint_array, "<i4"),
              "input_weights_sha256": _digest(source, "<f8"), "output_weights_sha256": _digest(result, "<f8"),
              "protected_indices_sha256": _digest(np.flatnonzero(protected), "<i4"),
              "changed_indices_sha256": _digest(np.flatnonzero(changed), "<i4"), "roles": reports,
              "hash_encodings": {"indices": "little-endian int32", "weights": "little-endian float64 C order"},
              "index_space": "Input rows, retaining every separate exported UV/normal seam vertex",
              "inputs_mutated": False, "core_memberships_unchanged": True,
              "geometry_or_animation_supplied": False, "quality_approved": False,
              "limits": {"vertices": MAX_VERTICES, "bones": MAX_BONES, "roles": MAX_ROLES,
                         "slots_per_vertex": 4, "recorded_fringe_losses": MAX_RECORDED_LOSSES},
              "limitations": ["Authored hard-surface opt-in, never inferred from color or silently applied to organic skin.",
                              "Multi-bone targets retain their ratios; co-rigidity is not established.",
                              "Existing majority assignments can be wrong; new joint discontinuities remain possible.",
                              "Inherited root motion is unchanged. Independent action, deformation and visual checks remain required."],
              "elapsed_seconds": time.monotonic()-start}
    return result, report
