"""Optional temporal orientation proposals where the original body declared twist unknown."""

from copy import deepcopy
import math
import json
import re
import time
from hashlib import sha256

import numpy as np

from weft.spatial_pose import LIMB_ENDPOINTS, SPATIAL_JOINT_NAMES, rotation_to_zyx, rotation_zyx

PROFILE = "assumed-twist-temporal-transport-v1"
MAX_SPAN_S = .25
MAX_DIRECTION_CHANGE = .06
SCOPE = ("Offline interpolation of explicitly unobserved axial orientation, not recovered motion. "
         "Observable anchors, proximal directions, hinge bends, source slots and confidence stay fixed; "
         "no force, contact, control or physical-accuracy qualification.")


def _require(value, message):
    if not value:
        raise ValueError("Twist continuity: " + message)


def _transport(rotation, axis):
    old = rotation[:, 2]
    cosine = float(np.clip(old @ axis, -1., 1.))
    _require(cosine > -1+1e-8, "antipodal upper directions have no unique minimal transport")
    cross = np.cross(old, axis)
    matrix = np.array([[0., -cross[2], cross[1]], [cross[2], 0., -cross[0]], [-cross[1], cross[0], 0.]])
    return (np.eye(3)+matrix+matrix@matrix/(1+cosine)) @ rotation


def _cost(angles, times):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        velocity = np.diff(angles, axis=0)/np.diff(times)[:, None]
        cost = float(np.sum(velocity**2*np.diff(times)[:, None]))
        peak = float(np.max(abs(velocity), initial=0.))
    _require(np.isfinite(velocity).all() and math.isfinite(cost) and math.isfinite(peak),
             "native durations produce nonfinite chart-speed cost")
    return cost, peak


def _digest(value, check):
    digest, size = sha256(), 0
    for chunk in json.JSONEncoder(sort_keys=True, allow_nan=False).iterencode(value):
        check(); raw = chunk.encode(); size += len(raw)
        _require(size <= 4 << 20, "reference/audit metadata exceeds four MiB")
        digest.update(raw)
    return digest.hexdigest()


def _metadata(value, check):
    stack, visited = [(value, 0)], 0
    while stack:
        check(); item, depth = stack.pop(); visited += 1
        _require(depth <= 16 and visited <= 250000, "metadata depth/node bound exceeded")
        if type(item) is dict:
            _require(len(item) <= 128 and all(type(k) is str and len(k) <= 256 for k in item), "bounded metadata fields required")
            stack.extend((child, depth+1) for child in item.values())
        elif type(item) is list:
            _require(len(item) <= 1800, "metadata list bound exceeded")
            stack.extend((child, depth+1) for child in item)
        elif type(item) is str:
            _require(len(item) <= 4096, "metadata string bound exceeded")
        elif type(item) in (int, float):
            _require(abs(item) <= 1e12 and math.isfinite(item), "finite bounded metadata number required")
        else:
            _require(item is None or type(item) is bool, "plain JSON metadata required")
    return _digest(value, check)


def propose_twist_continuity(references, audit_rows, source_frame_order, *, source_sha256,
                             parent_human_sha256, anchor_rotation, max_seconds=10):
    """Return a separate full reference proposal; never fill a missing pose or change a known twist."""
    _require(type(max_seconds) in (int, float) and 0 < max_seconds <= 10 and math.isfinite(max_seconds),
             "deadline must be positive and at most ten seconds")
    deadline = time.monotonic()+max_seconds
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Twist continuity deadline exceeded")
    for pin in (source_sha256, parent_human_sha256):
        _require(type(pin) is str and re.fullmatch("[a-f0-9]{64}", pin), "exact source and parent identities required")
    _require(type(references) is list and 2 <= len(references) <= 1800
             and type(audit_rows) is list and len(audit_rows) == len(references), "bounded aligned reference/audit rows required")
    _require(type(source_frame_order) is list and len(references) <= len(source_frame_order) <= 1800
             and all(type(x) is str and re.fullmatch(source_sha256+r":frame:[0-9]{1,8}", x) for x in source_frame_order)
             and len(set(source_frame_order)) == len(source_frame_order), "complete distinct original sampled-frame order required")
    original_pin, audit_pin, order_pin = (_metadata(value, check) for value in (references, audit_rows, source_frame_order))
    _metadata(anchor_rotation, check)
    slots = {name: index for index, name in enumerate(source_frame_order)}
    _require(type(anchor_rotation) is list and len(anchor_rotation) == 3
             and all(type(row) is list and len(row) == 3 and all(type(v) in (int, float) for v in row)
                     for row in anchor_rotation), "bounded numeric anchor matrix required")
    anchor = np.asarray(anchor_rotation, dtype=float)
    _require(anchor.shape == (3, 3) and np.isfinite(anchor).all()
             and np.allclose(anchor.T@anchor, np.eye(3), atol=1e-8, rtol=0)
             and abs(np.linalg.det(anchor)-1) < 1e-8, "proper original anchor rotation required")
    times, indices, matrices = [], [], []
    for row, audit in zip(references, audit_rows):
        check()
        _require(type(row) is dict and set(row) == {"time_s", "joints"}
                 and type(row["joints"]) is dict and set(row["joints"]) == set(SPATIAL_JOINT_NAMES), "exact spatial reference fields required")
        stamp = row["time_s"]
        _require(type(stamp) in (int, float) and math.isfinite(stamp) and 0 <= stamp <= 60
                 and (not times or stamp > times[-1]), "native source times must stay ordered")
        _require(type(audit) is dict and audit.get("time_s") == stamp
                 and type(audit.get("evidence_id")) is str and audit["evidence_id"] in slots,
                 "reference evidence identity/time mismatch")
        slot = slots[audit["evidence_id"]]
        _require(not indices or slot > indices[-1], "reference order differs from retained source slots")
        for name, value in row["joints"].items():
            limit = 2.6 if name.endswith(("_elbow", "_knee")) else math.tau
            lower = 0 if limit == 2.6 else -math.tau
            _require(type(value) in (int, float) and math.isfinite(value) and lower <= value <= limit,
                     "original joint coordinate exceeds its authored chart")
        assumed = audit.get("assumed_twist")
        _require(type(assumed) is list and len(assumed) <= 4 and all(type(x) is str and x in LIMB_ENDPOINTS for x in assumed)
                 and len(set(assumed)) == len(assumed), "explicit distinct unknown-twist audit required")
        confidence = audit.get("joint_confidence")
        _require(type(confidence) is dict and set(confidence) == set(SPATIAL_JOINT_NAMES)
                 and all(type(x) in (int, float) and 0 <= x <= 1 and math.isfinite(x) for x in confidence.values()),
                 "bounded original joint confidence required")
        for limb in audit["assumed_twist"]:
            certainty = audit["joint_confidence"].get(limb+"_twist")
            _require(type(certainty) in (int, float) and certainty == 0., "assumed twist must retain zero confidence")
        times.append(float(stamp)); indices.append(slot)
        matrices.append(anchor @ rotation_zyx([row["joints"][n] for n in SPATIAL_JOINT_NAMES[:3]]))
    times = np.asarray(times)
    result, trials = deepcopy(references), []
    for limb in LIMB_ENDPOINTS:
        names = (limb+"_twist", limb+"_lateral", limb)
        distal = limb.rsplit("_", 1)[0]+("_knee" if limb.endswith("hip") else "_elbow")
        sign = -1 if limb.endswith("hip") else 1
        angles = np.array([[row["joints"][name] for name in names] for row in references])
        local = [rotation_zyx(value) for value in angles]
        world = [parent @ rotation for parent, rotation in zip(matrices, local)]
        unknown = [limb in row["assumed_twist"] for row in audit_rows]
        index = 0
        while index < len(unknown):
            check()
            if not unknown[index]:
                index += 1; continue
            first = index
            while index < len(unknown) and unknown[index]: index += 1
            last, left, right = index-1, first-1, index
            row = {"limb": limb, "first_reference": first, "last_reference": last,
                "source_evidence_ids": [a["evidence_id"] for a in audit_rows[first:last+1]], "accepted": False}
            trials.append(row)
            if left < 0 or right >= len(unknown):
                row["reason"] = "two_observable_anchors_required"; continue
            row["anchor_evidence_ids"] = [audit_rows[left]["evidence_id"], audit_rows[right]["evidence_id"]]
            if times[right]-times[left] > MAX_SPAN_S+1e-12:
                row["reason"] = "anchor_span_exceeds_bound"; continue
            if any(indices[i+1] != indices[i]+1 for i in range(left, right)):
                row["reason"] = "missing_source_pose_breaks_transport"; continue
            if any(audit_rows[i].get("identity_status") != audit_rows[left].get("identity_status") for i in range(left, right+1)):
                row["reason"] = "identity_status_change_breaks_transport"; continue
            if any(audit_rows[i].get("joint_confidence", {}).get(limb+"_twist", 0) <= 0 for i in (left, right)):
                row["reason"] = "anchors_lack_observed_bend_plane"; continue
            if any(abs(math.sin(references[i]["joints"][distal])) >= .03+1e-12 for i in range(first, last+1)):
                row["reason"] = "model_hinge_is_not_near_straight"; continue
            try:
                forward, backward = {left: world[left]}, {right: world[right]}
                for i in range(first, right):
                    check(); forward[i] = _transport(forward[i-1], world[i][:, 2])
                for i in range(last, left, -1):
                    check(); backward[i] = _transport(backward[i+1], world[i][:, 2])
                candidate = angles[left:right+1].copy()
                displacement = 0.
                for i in range(first, last+1):
                    check()
                    relative = forward[i].T @ backward[i]
                    theta = math.atan2(relative[1, 0], relative[0, 0])
                    fraction = (times[i]-times[left])/(times[right]-times[left])
                    proposed_world = forward[i] @ rotation_zyx([fraction*theta, 0., 0.])
                    proposed_local = matrices[i].T @ proposed_world
                    prior = (1-fraction)*angles[left]+fraction*angles[right]
                    value, singular = rotation_to_zyx(proposed_local, prior)
                    _require(not singular, "proposed singular Euler chart")
                    _require(np.linalg.norm(proposed_world[:, 2]-world[i][:, 2]) <= 1e-8, "upper direction changed")
                    bend = rotation_zyx([0., 0., sign*references[i]["joints"][distal]]) @ [0., 0., -1.]
                    displacement = max(displacement, float(np.linalg.norm(proposed_world@bend-world[i]@bend)))
                    candidate[i-left] = value
                old_cost, old_peak = _cost(angles[left:right+1], times[left:right+1])
                new_cost, new_peak = _cost(candidate, times[left:right+1])
                row.update(before_integrated_chart_speed_squared=old_cost, after_integrated_chart_speed_squared=new_cost,
                    before_peak_chart_speed_rad_s=old_peak, after_peak_chart_speed_rad_s=new_peak,
                    maximum_distal_direction_change=displacement)
                if displacement > MAX_DIRECTION_CHANGE+1e-10:
                    row["reason"] = "distal_direction_envelope_exceeded"; continue
                if new_cost >= old_cost-1e-10 or new_peak > old_peak+1e-10:
                    row["reason"] = "temporal_chart_cost_or_peak_not_improved"; continue
                for i in range(first, last+1):
                    for name, value in zip(names, candidate[i-left]): result[i]["joints"][name] = float(value)
                row.update(accepted=True, reason="bounded_unknown_twist_temporal_proposal")
            except ValueError as error:
                row["reason"] = str(error)
    check()
    receipt = {"schema_version": 1, "profile": PROFILE, "source_sha256": source_sha256,
        "parent_human_sha256": parent_human_sha256, "original_reference_sha256": original_pin,
        "proposed_reference_sha256": _digest(result, check), "original_audit_sha256": audit_pin,
        "source_frame_order_sha256": order_pin, "maximum_anchor_span_s": MAX_SPAN_S,
        "maximum_distal_direction_change": MAX_DIRECTION_CHANGE, "trials": trials,
        "accepted_spans": sum(row["accepted"] for row in trials), "confidence_promoted": False,
        "original_first_last_reference_unchanged": result[0] == references[0] and result[-1] == references[-1], "scope": SCOPE}
    return result, receipt
