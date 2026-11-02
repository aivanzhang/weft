"""Smooth joint-only schedules around a separately authenticated same-state base action."""

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from weft.shared_exercise_equipment_feedback import _frozen, _vector, prepare_equipment_feedback


PROFILE = "shared-exercise-temporal-joint-residual-v1"
KNOTS = (.25, .5, .75)
HALF_WIDTH = .25
COEFFICIENT_BOUND = RESIDUAL_BOUND = .25
MAX_REPORT_BYTES = 8 * 1024
SCOPE = ("Experimental phase-conditioned joint correction, not a trained policy or collision guarantee. "
         "The caller must authenticate an authored reference action and observation from the same current state. "
         "Elapsed task phase is not a source observation or contact prediction. Root action bytes, equipment "
         "forces, source references, physics and task limits are unchanged; phase schedules can overfit a clip.")


def _weights(value):
    if type(value) is np.ndarray:
        valid = value.shape == (19, 3) and value.dtype.kind in "fiu"
    else:
        valid = (type(value) in (list, tuple) and len(value) == 19
                 and all(type(row) in (list, tuple) and len(row) == 3
                         and all(type(item) in (int, float) and math.isfinite(item)
                                 and abs(item) <= COEFFICIENT_BOUND for item in row) for row in value))
    if not valid:
        raise ValueError("Temporal residual requires a finite 19-by-3 coefficient array")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all() or np.any(abs(result) > COEFFICIENT_BOUND):
        raise ValueError("Temporal residual coefficients must stay in [-.25,.25]")
    return _frozen(result)


def basis(elapsed):
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= 1:
        raise ValueError("Temporal basis requires an elapsed fraction in [0,1]")
    # Overlapping raised cosines form a convex blend, with zero value/slope at task endpoints.
    distance = (np.asarray(KNOTS) - elapsed) / HALF_WIDTH
    values = np.zeros(3)
    inside = abs(distance) < 1.
    values[inside] = .5 * (1. + np.cos(math.pi * distance[inside]))
    return values


def feature_contract():
    return {"profile": PROFILE, "coefficient_shape": [19, 3], "coefficient_bound": COEFFICIENT_BOUND,
            "joint_residual_bound": RESIDUAL_BOUND, "centers_elapsed_fraction": list(KNOTS),
            "half_width_elapsed_fraction": HALF_WIDTH, "basis": "compact_raised_cosine_convex_blend",
            "phase_source": "current_observation_final_elapsed_fraction", "joint_channels": list(range(19)),
            "root_channels": [19, 20, 21], "root_mutation": "forbidden", "equipment_force_channels": [],
            "base": "caller_authenticated_same_state_authored_reference", "scope": SCOPE}


@dataclass(frozen=True, eq=False)
class TemporalJointResidual:
    _weights: np.ndarray
    _parameter_sha256: str
    _observation_size: int

    def sample(self, base_action, observation):
        base = _vector(base_action, 22, 1., "temporal base action")
        observed = _vector(observation, self._observation_size, 1e6, "temporal observation")
        reference_phase, elapsed = float(observed[154]), float(observed[-1])
        if not 0 <= reference_phase <= 1 or not 0 <= elapsed < 1:
            raise ValueError("Temporal residual requires an unfinished task and valid reference phase")
        features = basis(elapsed)
        requested = np.clip(self._weights @ features, -RESIDUAL_BOUND, RESIDUAL_BOUND)
        action = base.copy()
        changed = np.flatnonzero(requested != 0.)
        action[changed] = np.clip(base[changed].astype(float) + requested[changed], -1., 1.).astype(base.dtype)
        # Casting at a legal boundary must not enlarge the original-base residual interval.
        outside = np.flatnonzero(abs(action[:19].astype(float) - base[:19].astype(float)) > RESIDUAL_BOUND)
        action[outside] = np.nextafter(action[outside], base[outside])
        applied = action[:19].astype(float) - base[:19].astype(float)
        if (np.any(abs(applied) > RESIDUAL_BOUND) or np.any(abs(action) > 1.)
                or action[19:].tobytes() != base[19:].tobytes()):
            raise ValueError("Temporal residual could not preserve the joint/root action bounds")
        report = {"profile": PROFILE, "parameter_sha256": self._parameter_sha256,
                  "pairing": "caller_responsibility_not_authenticated", "action_dtype": action.dtype.str,
                  "reference_phase": reference_phase, "elapsed_fraction": elapsed, "basis": features.tolist(),
                  "requested_joint_residual": requested.tolist(), "applied_joint_residual": applied.tolist(),
                  "changed_joint_channels": np.flatnonzero(applied != 0.).tolist(),
                  "maximum_abs_residual": float(np.max(abs(applied))), "root_bytes_unchanged": True}
        if len(json.dumps(report, allow_nan=False, separators=(",", ":")).encode()) > MAX_REPORT_BYTES:
            raise ValueError("Temporal residual report exceeds its fixed bound")
        return action, report


def prepare_temporal_joint_residual(env, coefficients):
    """Admit metadata once; sampling never reads or modifies the live simulator."""
    weights = _weights(coefficients)
    # Reuse the exact existing 22-channel, no-required-contact observation/action admission.
    admitted = prepare_equipment_feedback(env, np.zeros((8, 2)))
    parameters = {"contract": feature_contract(), "coefficients": weights.tolist()}
    raw = json.dumps(parameters, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return TemporalJointResidual(weights, hashlib.sha256(raw).hexdigest(), admitted._observation_size)
