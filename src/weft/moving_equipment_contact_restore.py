"""Pure bounded contact restoration; linear feasibility is never a contact certificate."""

import math
import time
from types import MappingProxyType

import numpy as np

PROFILE = "moving-contact-restoration-v1"
LIMITS = MappingProxyType({"variables": 192, "rows": 8192, "seconds": 10.,
    "iterations": 24, "working_set_steps": 192, "feasibility_tolerance": 1e-10})


def _require(ok, message):
    if not ok:
        raise ValueError("Contact restoration: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Contact restoration deadline exceeded")


def _matrix(value, label, deadline, n=None):
    # Reject oversized/object inputs before NumPy can allocate or coerce arbitrary payloads.
    if isinstance(value, np.ndarray):
        _require(value.ndim == 2 and value.dtype.kind in "fiu" and len(value) <= LIMITS["rows"]
                 and 1 <= value.shape[1] <= LIMITS["variables"], label + " matrix shape/type bound")
        width = value.shape[1]
    else:
        _require(type(value) in (list, tuple) and len(value) <= LIMITS["rows"], label + " matrix row bound")
        width = n if not value else len(value[0]) if type(value[0]) in (list, tuple) else 0
        _require(type(width) is int and 1 <= width <= LIMITS["variables"], label + " variable bound")
        for row in value:
            _check(deadline)
            _require(type(row) in (list, tuple) and len(row) == width
                     and all(type(x) in (int, float) for x in row), label + " numeric matrix required")
    _require(n is None or width == n, label + " row shape differs")
    _check(deadline)
    return np.asarray(value, dtype=float).reshape((-1, width))


def _rows(matrix, vector, n, label, deadline):
    a = _matrix(matrix, label, deadline, n)
    if isinstance(vector, np.ndarray):
        _require(vector.ndim == 1 and vector.dtype.kind in "fiu" and len(vector) == len(a), label + " vector shape/type bound")
    else:
        _require(type(vector) in (list, tuple) and len(vector) == len(a)
                 and all(type(x) in (int, float) for x in vector), label + " numeric vector required")
    b = np.asarray(vector, dtype=float)
    _require(np.isfinite(a).all() and np.isfinite(b).all(), label + " requires finite rows")
    _check(deadline)
    return a, b


def _qp(H, g, C, lower, initial, deadline, maximum):
    """Keep every iterate feasible; only the independent active rows enter the dense solve."""
    x = initial.copy()
    _require(np.min(C @ x-lower, initial=0.) >= -1e-8, "QP requires a feasible starting point")
    _check(deadline)
    inverse = np.linalg.solve(H, np.eye(len(x)))
    _check(deadline)
    active, history, status = [], [], "working_set_limit"
    for iteration in range(maximum):
        _check(deadline)
        gradient = H @ x+g
        if active:
            A = C[active]
            gram = A @ inverse @ A.T
            multiplier = np.linalg.solve(gram, A @ inverse @ gradient)
            direction = inverse @ (-gradient+A.T @ multiplier)
        else:
            multiplier = np.empty(0)
            direction = -inverse @ gradient
        _check(deadline)
        if np.linalg.norm(direction, ord=np.inf) <= 1e-9:
            if len(active) and float(np.min(multiplier)) < -1e-8:
                del active[int(np.argmin(multiplier))]
                continue
            status = "stationary_feasible"
            break
        slack, rates = C @ x-lower, C @ direction
        candidates = np.flatnonzero(rates < -1e-12)
        if active:
            candidates = candidates[~np.isin(candidates, active)]
        alpha, blocker = 1., None
        if len(candidates):
            ratios = np.maximum(0., slack[candidates])/-rates[candidates]
            which = int(np.argmin(ratios))
            if ratios[which] < alpha:
                alpha, blocker = float(ratios[which]), int(candidates[which])
        trial = x+alpha*direction
        # All original rows, including duplicates and inactive constraints, remain authoritative.
        if np.min(C @ trial-lower, initial=0.) < -1e-8:
            status = "numerical_feasibility_refusal"
            break
        x = trial
        if blocker is not None:
            row = C[blocker]
            residual = row if not active else row-np.linalg.lstsq(C[active].T, row, rcond=1e-11)[0] @ C[active]
            if np.linalg.norm(residual) <= 1e-10*max(1., np.linalg.norm(row)):
                status = "dependent_blocker_refusal"
                break
            active.append(blocker)
            _require(len(active) <= len(x), "independent active set exceeds variable count")
        history.append({"fraction": alpha, "active_count": len(active)})
    _check(deadline)
    return x, {"status": status, "iterations": iteration+1, "maximum_active_rows": max([h["active_count"] for h in history]+[len(active)]),
        "minimum_hard_residual": float(np.min(C @ x-lower, initial=0.)), "rows_squared_allocation": False}


def _feasible_anchor(C, lower, deadline, maximum):
    """Phase-I slack is temporary; no relaxed hard row enters either restoration phase."""
    n = C.shape[1]
    slack = float(np.max(lower, initial=0.))+1e-12
    augmented = np.column_stack((C, np.ones(len(C))))
    augmented = np.vstack((augmented, np.r_[np.zeros(n), 1.], np.r_[np.zeros(n), -1.]))
    rhs = np.r_[lower, 0., -slack]
    initial = np.r_[np.zeros(n), slack]
    H, g = np.eye(n+1)*1e-4, np.r_[np.zeros(n), 1.]
    point, receipt = _qp(H, g, augmented, rhs, initial, deadline, maximum)
    candidate = point[:-1]
    residual = C @ candidate-lower
    verified = bool(np.min(residual, initial=0.) >= -LIMITS["feasibility_tolerance"])
    receipt.update(phase="hard_feasibility_initialization", temporary_slack=float(point[-1]),
                   original_hard_rows_verified=verified, minimum_original_hard_residual=float(np.min(residual, initial=0.)))
    return candidate if verified else None, receipt


def _cost(A, values, x):
    deficit = np.minimum(A @ x+values, 0.)
    return float(deficit @ deficit)/max(1, len(values))


def _phase(A, values, C, lower, start, deadline, iterations, working_steps):
    x, receipts = start.copy(), []
    for iteration in range(iterations):
        _check(deadline)
        residual = A @ x+values
        violated = residual < 0
        if not np.any(violated):
            break
        # Individual deficits cannot disappear beneath an unrelated worst-row slack.
        J, v = A[violated], values[violated]
        scale = max(1, len(values))
        H = J.T @ J/scale+np.eye(len(x))*1e-6
        gradient_constant = J.T @ v/scale-1e-6*x
        try:
            candidate, receipt = _qp(H, gradient_constant, C, lower, x, deadline, working_steps)
        except np.linalg.LinAlgError:
            receipts.append({"status": "numerical_linear_algebra_refusal", "accepted": False})
            break
        before = _cost(A, values, x)
        receipt.update(before_squared_deficit=before, accepted=False)
        for power in range(8):
            _check(deadline)
            trial = x+(candidate-x)*2.**(-power)
            after = _cost(A, values, trial)
            if np.min(C @ trial-lower, initial=0.) >= -LIMITS["feasibility_tolerance"] and after < before-1e-14:
                x = trial
                receipt.update(accepted=True, after_squared_deficit=after, fraction=2.**(-power))
                break
        receipts.append(receipt)
        if not receipt["accepted"]:
            break
    return x, receipts


def restore_contact_direction(collision_matrix, collision_values, support_matrix, support_values,
                              hard_matrix, hard_lower, *, max_seconds=2, max_iterations=12,
                              max_working_set_steps=96):
    """Restore collision rows first, then support without losing any attained collision bound."""
    started = time.monotonic()
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= LIMITS["seconds"], "invalid time bound")
    _require(type(max_iterations) is int and 1 <= max_iterations <= LIMITS["iterations"], "invalid iteration bound")
    _require(type(max_working_set_steps) is int and 1 <= max_working_set_steps <= LIMITS["working_set_steps"], "invalid working-set bound")
    deadline = started+max_seconds
    matrix = _matrix(hard_matrix, "hard", deadline)
    _require(matrix.ndim == 2 and 1 <= matrix.shape[1] <= LIMITS["variables"], "invalid variable count")
    n = matrix.shape[1]
    C, lower = _rows(matrix, hard_lower, n, "hard", deadline)
    A, av = _rows(collision_matrix, collision_values, n, "collision", deadline)
    B, bv = _rows(support_matrix, support_values, n, "support", deadline)
    _require(len(A)+len(B)+len(C) <= LIMITS["rows"], "aggregate row bound exceeded")
    initial_violations = [{"row": int(i), "residual": float(-lower[i])} for i in np.flatnonzero(lower > 0)]
    x, stages, failure = np.zeros(n), [], None
    anchor_refused = False
    resource_incomplete = False
    collision_floor = None
    try:
        _check(deadline)
        if initial_violations:
            anchor, receipt = _feasible_anchor(C, lower, deadline, max_working_set_steps)
            stages.append({"phase": "hard_feasibility_initialization", "report": receipt})
            if anchor is None:
                anchor_refused = True
                raise ArithmeticError("No verified feasible hard-bound anchor within the bounded Phase-I search")
            x = anchor
        x, receipt = _phase(A, av, C, lower, x, deadline, max_iterations, max_working_set_steps)
        stages.append({"phase": "collision", "iterations": receipt})
        collision_floor = np.minimum(A @ x+av, 0.)
        combined = np.vstack((C, A))
        combined_lower = np.r_[lower, collision_floor-av]
        x, receipt = _phase(B, bv, combined, combined_lower, x, deadline, max_iterations, max_working_set_steps)
        stages.append({"phase": "support", "iterations": receipt})
        _check(deadline)
    except (ArithmeticError, np.linalg.LinAlgError) as error:
        anchor_refused = True
        failure = str(error)
    except TimeoutError as error:
        resource_incomplete = True
        failure = str(error)
    hard_residual, collision, support = C @ x-lower, A @ x+av, B @ x+bv
    hard_ok = bool(np.min(hard_residual, initial=0.) >= -LIMITS["feasibility_tolerance"])
    collision_preserved = collision_floor is None or bool(np.min(collision-collision_floor, initial=0.) >= -1e-8)
    # A refused initializer has no direction; retain its exact original deficits instead of throwing away the search trace.
    _require(collision_preserved and np.isfinite(x).all(), "final original collision-preservation verification failed")
    if not hard_ok:
        anchor_refused = True
        failure = failure or "Final original hard-bound verification refused the direction"
    full = bool(np.min(collision, initial=0.) >= -1e-8 and np.min(support, initial=0.) >= -1e-8)
    collision_cost, support_cost = _cost(A, av, x), _cost(B, bv, x)
    step = x.tolist() if hard_ok and not anchor_refused else None
    if time.monotonic() >= deadline:
        resource_incomplete = True
        failure = failure or "Contact restoration deadline exceeded during final audit"
    return {"profile": PROFILE, "status": "incomplete_resource_bound" if resource_incomplete else "unsupported_hard_anchor" if anchor_refused else "incomplete_resource_bound" if failure else "linear_restoration_complete" if full else "partial_restoration",
        "step": step, "failure": failure, "report": {"stages": stages, "seconds": time.monotonic()-started,
            "variables": n, "rows": len(A)+len(B)+len(C), "initial_hard_violations": initial_violations,
            "hard_feasibility_verified": hard_ok and not anchor_refused, "minimum_hard_residual": float(np.min(hard_residual, initial=0.)),
            "collision_squared_deficit": collision_cost, "support_squared_deficit": support_cost,
            "collision_preservation_verified": collision_preserved, "all_original_linear_rows_pass": full and hard_ok,
            "nonlinear_contact_verified": False, "task_success": False,
            "scope": "Bounded linear direction only; source geometry and original nonlinear/task audit remain mandatory."}}
