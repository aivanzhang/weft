"""Bounded static object balance; supplied contact geometry is a hypothesis, not admission."""

from hashlib import sha256
import json
import math
import re
import time
from types import MappingProxyType

import numpy as np

from weft.constrained_qp import solve_feasible_qp

PROFILE = "contact-force-balance-v1"
LIMITS = MappingProxyType({"input_bytes": 16384, "output_bytes": 32768,
    "contacts": 2, "input_contacts": 8, "seconds": 2., "iterations": 128,
    "force_residual": 1e-7, "moment_residual": 1e-7, "inequality_residual": 1e-8})
SCOPE = ("Static object equilibrium under explicit supplied point contacts and force caps. "
    "Not source authentication, contact/actuator admission, dynamic stability, recovered grasp or task success. "
    "Refusal is not global infeasibility. Manual exploration and existing task gates are unchanged.")


def _require(value, message):
    if not value:
        raise ValueError("Contact force balance: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Contact force balance deadline exceeded")


def _number(value, label, bound, low=None):
    _require(type(value) in (int, float) and abs(value) <= bound, label + " bounded number required")
    _require(math.isfinite(value) and (low is None or value >= low), label + " range differs")
    return float(value)


def _vector(value, label, bound, low=None, size=3):
    _require(type(value) is list and len(value) == size, label + " shape differs")
    return [_number(x, label, bound, low) for x in value]


def _keys(value, keys, label):
    expected = keys.split()
    _require(type(value) is dict and len(value) == len(expected) and set(value) == set(expected), label + " fields differ")


def _bytes(value, maximum):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    _require(len(raw) <= maximum, "JSON byte bound exceeded")
    return raw


def validate_force_hypothesis(value):
    """Validate a small closed caller-supplied hypothesis, without claiming its source is true."""
    _require(type(value) is dict, "hypothesis must be an object")
    _keys(value, "schema_version profile provenance contact_model center_of_mass_world mass gravity_world external_force_world external_moment_world direct_force_max_world contacts" + (" motor_limits" if "motor_limits" in value else ""), "hypothesis")
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1 and value["profile"] == PROFILE, "profile differs")
    _keys(value["provenance"], "source_sha256 scene_sha256 task_sha256", "provenance")
    _require(all(type(x) is str and re.fullmatch("[a-f0-9]{64}", x) for x in value["provenance"].values()), "caller provenance pins required")
    _require(type(value["contact_model"]) is str and len(value["contact_model"]) <= 64, "contact model label bound")
    _vector(value["center_of_mass_world"], "COM", 100)
    _number(value["mass"], "mass", 1000, 1e-6)
    _vector(value["gravity_world"], "gravity", 100)
    _vector(value["external_force_world"], "external force", 1e5)
    _vector(value["external_moment_world"], "external moment about COM", 1e7)
    _vector(value["direct_force_max_world"], "original direct-force caps", 1e5, 0)
    contacts = value["contacts"]
    _require(type(contacts) is list and 1 <= len(contacts) <= LIMITS["input_contacts"], "contact inventory bound")
    names = set()
    for contact in contacts:
        _keys(contact, "id position_world normal_on_object_world tangent_world sliding_friction normal_force_max", "contact")
        name = contact["id"]
        _require(type(name) is str and re.fullmatch("[A-Za-z0-9_-]{1,64}", name) and name not in names, "unique contact ID required")
        names.add(name)
        _vector(contact["position_world"], "contact position", 100)
        n = _vector(contact["normal_on_object_world"], "normal", 1)
        t = _vector(contact["tangent_world"], "tangent", 1)
        _require(abs(sum(x*x for x in n)-1) <= 1e-10 and abs(sum(x*x for x in t)-1) <= 1e-10
                 and abs(sum(x*y for x, y in zip(n, t))) <= 1e-10, "unit orthogonal contact frame required")
        mu = _vector(contact["sliding_friction"], "effective sliding friction", 5, 0, size=2)
        _require(all(x == 0 or x >= 1e-6 for x in mu), "positive friction below supported conditioning bound")
        _number(contact["normal_force_max"], "original normal-force cap", 1e4, 0)
    if "motor_limits" in value:
        motor = value["motor_limits"]
        _keys(motor, "matrix bias lower upper", "motor limits")
        _require(type(motor["matrix"]) is list and len(motor["matrix"]) <= 32, "motor row bound")
        for row in motor["matrix"]:
            _vector(row, "motor world-force mapping", 1e4, size=3*len(contacts))
        for key in ("bias", "lower", "upper"):
            _vector(motor[key], "motor "+key, 1e5, size=len(motor["matrix"]))
        _require(all(a <= b for a, b in zip(motor["lower"], motor["upper"])), "motor lower exceeds upper")
    return json.loads(_bytes(value, LIMITS["input_bytes"]))


def _problem(h):
    n = 3*len(h["contacts"])
    forces, moments, hard, rhs = [], [], [], []
    com = np.asarray(h["center_of_mass_world"])
    for i, contact in enumerate(h["contacts"]):
        normal, tangent = np.asarray(contact["normal_on_object_world"]), np.asarray(contact["tangent_world"])
        basis = np.column_stack((normal, tangent, np.cross(normal, tangent)))
        forces.append(basis)
        moments.append(np.column_stack([np.cross(np.asarray(contact["position_world"])-com, basis[:, j]) for j in range(3)]))
        def row(local, lower=0.):
            result = np.zeros(n); result[3*i:3*i+3] = local
            hard.append(result); rhs.append(lower)
        row([1, 0, 0]); row([-1, 0, 0], -contact["normal_force_max"])
        mu1, mu2 = contact["sliding_friction"]
        if mu1 and mu2:
            for a in (-1, 1):
                for b in (-1, 1): row([mu1*mu2, a*mu2, b*mu1])
        else:
            for j, mu in enumerate((mu1, mu2), 1):
                for sign in (-1, 1):
                    local = [mu, 0., 0.]; local[j] = sign; row(local)
    F, M = np.hstack(forces), np.hstack(moments)
    load = h["mass"]*np.asarray(h["gravity_world"])+np.asarray(h["external_force_world"])
    torque, caps = np.asarray(h["external_moment_world"]), np.asarray(h["direct_force_max_world"])
    # Direct assistance acts at COM, so it cannot supply an unrequested torque.
    balance = np.vstack((F, -F, M, -M))
    lower = np.r_[-load-caps, load-caps, -torque, torque]
    scales = np.r_[np.full(6, max(1., float(np.max(abs(load))), float(np.max(caps)))),
                   np.full(6, max(1., float(np.max(abs(torque))), float(np.max(abs(M)))))]
    motor = h.get("motor_limits")
    if motor and motor["matrix"]:
        world = np.zeros((n, n))
        for i, basis in enumerate(forces): world[3*i:3*i+3, 3*i:3*i+3] = basis
        J = np.asarray(motor["matrix"])@world
        motor_lower = np.asarray(motor["lower"])-motor["bias"]
        motor_upper = np.asarray(motor["bias"])-motor["upper"]
        scale = max(1., float(np.max(abs(motor_lower))), float(np.max(abs(motor_upper))))
        balance = np.vstack((balance/scales[:, None], J/scale, -J/scale))
        lower = np.r_[lower/scales, motor_lower/scale, motor_upper/scale]
    else:
        balance, lower = balance/scales[:, None], lower/scales
    return F, M, np.asarray(hard), np.asarray(rhs), balance, lower


def _audit(h, point, F, M, hard, rhs):
    load = h["mass"]*np.asarray(h["gravity_world"])+np.asarray(h["external_force_world"])
    direct = -(load+F@point)
    rows = []
    for i, contact in enumerate(h["contacts"]):
        world_force = F[:, 3*i:3*i+3]@point[3*i:3*i+3]
        normal = np.asarray(contact["normal_on_object_world"]); normal = normal/np.linalg.norm(normal)
        tangent = np.asarray(contact["tangent_world"]); tangent = tangent-normal*np.dot(normal, tangent)
        tangent = tangent/np.linalg.norm(tangent)
        # Audit the published world vector, not only the solver's local coefficients.
        local = np.array([world_force@normal, world_force@tangent, world_force@np.cross(normal, tangent)])
        mu = contact["sliding_friction"]
        diamond = sum(abs(local[j+1])/mu[j] for j in range(2) if mu[j])
        zero_axis = max([abs(local[j+1]) for j in range(2) if not mu[j]]+[0.])
        elliptic = math.sqrt(sum((local[j+1]/mu[j])**2 for j in range(2) if mu[j]))
        rows.append({"id": contact["id"], "force_world": world_force.tolist(),
            "normal_force": float(local[0]), "tangent_forces": local[1:].tolist(),
            "diamond_violation": max(0., float(diamond-local[0])), "zero_friction_axis_force": float(zero_axis),
            "elliptic_violation": max(0., float(elliptic-local[0])),
            "normal_cap_violation": max(0., float(local[0]-contact["normal_force_max"])),
            "tensile_violation": max(0., float(-local[0]))})
    # Reconstruct all forces from the published world components, independently of QP slack.
    force = load+direct+sum((np.asarray(row["force_world"]) for row in rows), np.zeros(3))
    cross_moment = np.asarray(h["external_moment_world"])+sum((np.cross(np.asarray(c["position_world"])-h["center_of_mass_world"], r["force_world"]) for c, r in zip(h["contacts"], rows)), np.zeros(3))
    residuals = {"force_max_abs": float(np.max(abs(force))), "moment_max_abs": float(np.max(abs(cross_moment))),
        "direct_force_cap_violation": max(0., float(np.max(abs(direct)-h["direct_force_max_world"]))),
        "original_linear_violation": max(0., float(np.max(rhs-hard@point))),
        "contact_violation": max(row[key] for row in rows for key in ("diamond_violation", "zero_friction_axis_force", "elliptic_violation", "normal_cap_violation", "tensile_violation"))}
    motor = h.get("motor_limits")
    motor_audit = {"status": "capacity_unassessed", "loads": []}
    residuals["motor_bound_violation"] = 0.
    if motor and motor["matrix"]:
        loads = np.asarray(motor["bias"])+np.asarray(motor["matrix"])@np.concatenate([row["force_world"] for row in rows])
        residuals["motor_bound_violation"] = max(0., float(np.max(np.asarray(motor["lower"])-loads)), float(np.max(loads-motor["upper"])))
        motor_audit = {"status": "supplied_linear_rows_checked", "loads": loads.tolist()}
    ok = residuals["force_max_abs"] <= LIMITS["force_residual"] and residuals["moment_max_abs"] <= LIMITS["moment_residual"] and all(residuals[k] <= LIMITS["inequality_residual"] for k in ("direct_force_cap_violation", "original_linear_violation", "contact_violation", "motor_bound_violation"))
    return ok, {"contacts": rows, "direct_force_world": direct.tolist(), "residuals": residuals, "motor_capacity": motor_audit}


def solve_contact_force_balance(hypothesis, *, max_seconds=2., max_iterations=128):
    """Return an independently checked witness or a bounded refusal, never a grasp claim."""
    started = time.monotonic()
    _number(max_seconds, "seconds", LIMITS["seconds"], .001)
    _require(type(max_iterations) is int and 1 <= max_iterations <= LIMITS["iterations"], "iteration bound")
    deadline = started+max_seconds
    h = validate_force_hypothesis(hypothesis)
    result = {"schema_version": 1, "profile": PROFILE, "hypothesis_sha256": sha256(_bytes(h, LIMITS["input_bytes"])).hexdigest(),
        "provenance": h["provenance"], "status": "refused", "reason": None, "witness": None,
        "scope": SCOPE, "source_authenticated": False, "global_infeasibility_claim": False,
        "units": "Force = mass × world-distance/s²; moment = force × world-distance. Not independently calibrated SI units.",
        "approximation": "Conservative diamond inside the supplied anisotropic elliptic sliding cone; zero torsional/rolling torque. Not MuJoCo soft-contact dynamics.",
        "normal_caps": "Explicit supplied search caps; not independently measured contact or actuator capacity.",
        "motor_scope": "Only supplied linear bias+Jacobian load bounds are checked; no full-body or floor load-sharing equilibrium is inferred.",
        "limits": dict(LIMITS), "solver": None}
    try:
        _check(deadline)
        if h["contact_model"] != "point_sliding" or len(h["contacts"]) > LIMITS["contacts"]:
            result.update(status="unsupported", reason="Only one or two explicit sliding point contacts are implemented")
        else:
            F, M, hard, rhs, balance, lower = _problem(h)
            n = F.shape[1]
            A = np.vstack((np.column_stack((hard, np.zeros(len(hard)))), np.column_stack((balance, np.ones(len(balance)))), np.r_[np.zeros(n), 1.]))
            b = np.r_[rhs, lower, 0.]
            # An interior normal avoids the degenerate apex of every friction cone.
            anchor = np.zeros(n)
            anchor[::3] = [c["normal_force_max"]/2 for c in h["contacts"]]
            initial = np.r_[anchor, max(0., float(np.max(lower-balance@anchor)))+1e-10]
            _check(deadline)
            remaining = min(.5, deadline-time.monotonic())
            if remaining < .001: raise TimeoutError("Contact force balance deadline exceeded")
            solved = solve_feasible_qp(np.eye(n+1)*1e-8, np.r_[np.zeros(n), 1.], A, b,
                initial=initial, settings={"maximum_iterations": max_iterations, "maximum_seconds": remaining})
            _check(deadline)
            result["solver"] = {"status": solved["status"], "iterations": solved["report"]["iterations"],
                "variables": n+1, "constraints": len(b), "temporary_slack_retained_in_witness": False}
            value = solved.get("step")
            if value is None: value = solved["report"].get("last_feasible_point")
            if value is not None:
                _require(type(value) is list and len(value) == n+1 and all(type(x) in (int,float) and math.isfinite(x) and abs(x)<=1e6 for x in value), "solver returned malformed point")
                ok, audit = _audit(h, np.asarray(value[:-1]), F, M, hard, rhs)
                result["candidate_audit"] = audit
                result["solver"]["temporary_phase_slack"] = value[-1]
                if ok: result.update(status="feasible", reason="Original supplied force, moment, friction and cap residuals verified", witness=audit)
                else: result["reason"] = "No verified static witness within this conservative contact hypothesis and bounded solve"
            else: result["reason"] = "Bounded solver returned no candidate; no infeasibility conclusion"
    except TimeoutError:
        result.update(status="refused", reason="deadline_exceeded", witness=None)
    except np.linalg.LinAlgError:
        result.update(status="refused", reason="numerical_solver_refusal", witness=None)
    result["seconds"] = time.monotonic()-started
    _bytes(result, LIMITS["output_bytes"])
    if time.monotonic() >= deadline:
        result.update(status="refused", reason="deadline_exceeded", witness=None)
    return result
