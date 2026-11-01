"""Pure equipment-conditioned residuals around a caller-supplied, pinned v5 action."""

from dataclasses import dataclass
import json
import math

import numpy as np

from weft.mobile_human import ROOT_AXES
from weft.shared_exercise_feedback_residual import SPLIT_GROUP_NAMES, SPLIT_POLICY_PROFILE, joint_groups


PROFILE = "shared-exercise-equipment-feedback-v1"
GROUP_NAMES = (*SPLIT_GROUP_NAMES, *ROOT_AXES)
COEFFICIENT_BOUND = RESIDUAL_BOUND = .25
PROXIMITY_DISTANCE_MODEL_UNITS = 1.5
JOINT_SPEED_RAD_S = 4.
CLOSING_SPEED_MODEL_UNITS_S = 2.
MAX_RESULT_BYTES = 16 * 1024
SCOPE = ("Experimental pure residual actor, not a trained policy or demonstrated anticipatory controller. "
         "The caller supplies a pinned v5 base and the same-current-state observation from the unchanged "
         "prepared environment; this helper cannot authenticate pairing, freshness or early termination. "
         "Nearest equipment-center distance is not limb clearance, contact prediction or a task-gate forecast. "
         "Normalizers are authored body-model-unit priors, not calibrated physical measurements. "
         "Only the existing nineteen joints and three root-assistance channels can change; no object forces.")
_EQUIPMENT_FIELDS = ("position_model_units:3", "quaternion:4", "velocity_model_units_s:3",
                     "angular_velocity_rad_s:3", "displacement_model_units:3", "orientation_error_rad:1")
_CONTACT_FIELDS = ("active", "qualifying", "load_fraction_clipped_10", "observed_coverage_fraction",
                   "current_gap_s", "max_gap_s")


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _frozen(value):
    # Bytes-backed arrays cannot be made writable by a caller retaining the prepared actor.
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _coefficients(value):
    if type(value) is np.ndarray:
        valid = value.shape == (8, 2) and value.dtype.kind in "fiu"
    else:
        valid = (type(value) in (list, tuple) and len(value) == 8
                 and all(type(row) in (list, tuple) and len(row) == 2
                         and all(_number(item, -.25, .25) for item in row) for row in value))
    if not valid:
        raise ValueError("Equipment feedback requires sixteen finite coefficients in an 8-by-2 array")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all() or np.any(abs(result) > COEFFICIENT_BOUND):
        raise ValueError("Equipment feedback coefficients must remain in [-.25,.25]")
    return _frozen(result)


def _vector(value, size, bound, label):
    # Inspect the fixed shape before any copy or conversion of caller-owned input.
    if (type(value) is not np.ndarray or value.shape != (size,)
            or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(value).all() or np.any(abs(value) > bound)):
        raise ValueError(f"Equipment feedback {label} requires {size} finite bounded float32/float64 values")
    return value.copy()


def _observation_contract(ids):
    return {"profile": "shared-exercise-observation-v1", "size": 156 + 17 * len(ids),
            "fields": [{"name": "human_reference_conditioned_state", "count": 155},
                       {"name": "equipment", "ids": list(ids), "count_each": 17,
                        "fields": list(_EQUIPMENT_FIELDS)},
                       {"name": "contact_requirements", "count_each": 6, "fields": list(_CONTACT_FIELDS)},
                       {"name": "elapsed_fraction", "count": 1}]}


@dataclass(frozen=True, eq=False)
class EquipmentFeedback:
    """Use only with the frozen environment and same-state base/observation admitted by the caller."""

    _weights: np.ndarray
    _channel_weights: np.ndarray
    _equipment_ids: tuple
    _scale: float
    _pelvis_origin: np.ndarray
    _observation_size: int

    def sample(self, base_action, observation):
        base = _vector(base_action, 22, 1., "base action")
        observed = _vector(observation, self._observation_size, 1e6, "observation")
        phase, elapsed = float(observed[154]), float(observed[-1])
        if not 0. <= phase <= 1. or not 0. <= elapsed < 1.:
            raise ValueError("Equipment feedback requires a valid reference phase and unfinished elapsed fraction")

        # The engine prefix is q22, v22, target22, target_velocity22, PD66, reference_phase1.
        pelvis = self._pelvis_origin + observed[19:22].astype(float) / self._scale
        pelvis_velocity = observed[41:44].astype(float) / self._scale
        equipment = observed[155:-1].reshape(len(self._equipment_ids), 17)
        offsets = pelvis - equipment[:, :3].astype(float)
        distances = np.hypot.reduce(offsets, axis=1)
        nearest = int(np.argmin(distances))
        distance = float(distances[nearest])
        known_direction = distance > 0.
        away = offsets[nearest] / distance if known_direction else np.zeros(3)
        relative_velocity = pelvis_velocity - equipment[nearest, 7:10].astype(float)
        closing_speed = max(0., -float(np.dot(away, relative_velocity))) if known_direction else 0.
        proximity = float(np.clip(1. - distance / PROXIMITY_DISTANCE_MODEL_UNITS, 0., 1.))
        closing = float(np.clip(closing_speed / CLOSING_SPEED_MODEL_UNITS_S, 0., 1.))
        features = np.empty((22, 2))
        features[:19, 0] = proximity * np.clip(observed[22:41].astype(float) / JOINT_SPEED_RAD_S, -1., 1.)
        features[:19, 1] = proximity * np.clip(observed[66:85].astype(float) / JOINT_SPEED_RAD_S, -1., 1.)
        features[19:, 0] = proximity * away
        features[19:, 1] = proximity * closing * away
        requested = np.clip(np.sum(features * self._channel_weights, axis=1), -RESIDUAL_BOUND, RESIDUAL_BOUND)
        action = base.copy()
        changed = requested != 0.
        # Preserve zero-residual bytes; cast rounding must not enlarge the original-base box.
        action[changed] = np.clip(base[changed].astype(float) + requested[changed], -1., 1.).astype(base.dtype)
        outside = abs(action.astype(float) - base.astype(float)) > RESIDUAL_BOUND
        action[outside] = np.nextafter(action[outside], base[outside])
        applied = action.astype(float) - base.astype(float)
        if np.any(abs(applied) > RESIDUAL_BOUND) or np.any(abs(action) > 1.):
            raise ValueError("Equipment feedback could not represent the bounded residual in the base dtype")

        report = {"profile": PROFILE, "status": "sampled", "scope": SCOPE,
                  "required_base_profile": SPLIT_POLICY_PROFILE, "pairing": "caller_responsibility_not_authenticated",
                  "action_dtype": base.dtype.str, "groups": list(GROUP_NAMES), "coefficients": self._weights.tolist(),
                  "zero_coefficients": not bool(np.any(self._weights)), "observation_size": self._observation_size,
                  "contact_requirement_count": 0, "reference_phase": phase, "elapsed_fraction": elapsed,
                  "normalizers": {"proximity_distance_model_units": PROXIMITY_DISTANCE_MODEL_UNITS,
                                  "joint_speed_rad_s": JOINT_SPEED_RAD_S,
                                  "closing_speed_model_units_s": CLOSING_SPEED_MODEL_UNITS_S},
                  "nearest": {"id": self._equipment_ids[nearest], "index": nearest,
                              "center_distance_model_units": distance, "away_direction_known": known_direction,
                              "away_world_axes": away.tolist(), "proximity": proximity,
                              "closing_speed_model_units_s": closing_speed, "normalized_closing_speed": closing},
                  "features": features.tolist(), "requested_residual": requested.tolist(),
                  "applied_residual": applied.tolist(), "max_abs_residual": float(np.max(abs(applied))),
                  "changed_channels": np.flatnonzero(applied != 0.).tolist()}
        if len(json.dumps(report, allow_nan=False, separators=(",", ":")).encode()) > MAX_RESULT_BYTES:
            raise ValueError("Equipment feedback diagnostic exceeds its fixed 16 KiB bound")
        return action, report


def prepare_equipment_feedback(env, coefficients):
    """Cache metadata from an already admitted environment; no native admission or live state is queried."""
    weights = _coefficients(coefficients)
    scene, task = env.scene, env.task
    if (scene.get("profile") not in ("shared-world-v1", "shared-world-v2") or scene.get("controlled_id") != "human"
            or task.get("profile") != "shared-exercise-task-v1"
            or task.get("control") != {"root_assistance": "explicit_3_axis", "object_forces": "forbidden"}
            or task.get("contact_requirements") != []):
        raise ValueError("Equipment feedback requires the existing coupled task, three root axes and no required contacts")
    ids = task.get("equipment_ids")
    if (type(ids) is not list or not 1 <= len(ids) <= 12
            or any(type(name) is not str or not 1 <= len(name) <= 160 for name in ids) or len(set(ids)) != len(ids)):
        raise ValueError("Equipment feedback requires one to twelve unique bounded equipment IDs")
    names = env.action_names
    if type(env.action_count) is not int or env.action_count != 22 or len(names) != 22:
        raise ValueError("Equipment feedback requires exactly nineteen joints and three root channels")
    groups = joint_groups(names, profile=SPLIT_POLICY_PROFILE)
    action_contract = env.action_contract()
    expected_action = {"names": list(names), "normalized_bounds": [-1., 1.], "joint_channels": list(range(19)),
                       "root_assistance_channels": [19, 20, 21], "root_assistance": "explicit_3_axis",
                       "object_force_channels": []}
    if (type(action_contract) is not dict or set(action_contract) != {*expected_action, "units", "physical_limits"}
            or any(action_contract[name] != value for name, value in expected_action.items())):
        raise ValueError("Equipment feedback action metadata differs from the existing human-only contract")
    try:
        declared_action = scene["action_contract"]
        if any(action_contract[name] != declared_action[name][:22] for name in ("names", "units", "physical_limits")):
            raise ValueError("Equipment feedback scene and task action metadata disagree")
        limits, units = action_contract["physical_limits"], action_contract["units"]
        if (len(limits) != 22 or any(not _number(value, 1e-12, 1e9) for value in limits)
                or len(units) != 22 or any(type(value) is not str or not 1 <= len(value) <= 240 for value in units)):
            raise ValueError("Equipment feedback requires the existing finite positive motor limits and units")
        frame = scene["coordinate_frame"]
        if (frame["axes"] != "x_right_y_depth_z_up" or frame["units"]["time"] != "second"
                or frame["units"]["orientation"] != "quaternion_wxyz"
                or frame["units"]["angular_velocity"] != "world_radians_per_second"):
            raise ValueError("Equipment feedback requires the documented shared world coordinate frame")
        bridge = frame["scale_bridge"]
        scale, origin = bridge["body_height_to_world_scale"], bridge["origin_world"]
        height = scene["human_spec"]["mobility"]["base_height"]
        if (scene["human_spec"]["profile"] != "exercise-v3" or not _number(scale, .5, 2.)
                or type(origin) not in (list, tuple) or len(origin) != 3
                or any(not _number(value, 0., 6.) for value in origin) or not _number(height, .01, 5.)):
            raise ValueError("Equipment feedback requires the existing scale, origin and mobile-root height bounds")
    except (KeyError, TypeError) as error:
        raise ValueError("Equipment feedback scene metadata is incomplete") from error
    expected_observation = _observation_contract(ids)
    actual_observation = env.observation_contract()
    # Canonical JSON equality also distinguishes boolean counts from integers.
    if (type(env.observation_size) is not int or env.observation_size != expected_observation["size"]
            or json.dumps(actual_observation, sort_keys=True, allow_nan=False, separators=(",", ":"))
            != json.dumps(expected_observation, sort_keys=True, allow_nan=False, separators=(",", ":"))):
        raise ValueError("Equipment feedback observation metadata differs from the exact documented layout")
    pelvis_origin = np.asarray(origin, dtype=float) / float(scale) + [0., 0., float(height)]
    channels = np.concatenate((groups, [5, 6, 7]))
    return EquipmentFeedback(weights, _frozen(weights[channels]), tuple(ids), float(scale),
                             _frozen(pelvis_origin), expected_observation["size"])
