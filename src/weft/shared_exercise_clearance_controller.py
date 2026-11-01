"""Authored joint-only collision response; privileged geometry is not reconstructed contact."""

from dataclasses import asdict, dataclass
import math

import mujoco
import numpy as np

from weft.spatial_pose import SPATIAL_JOINT_NAMES


PROFILE = "shared-exercise-clearance-controller-v1"
MAX_GEOMETRY_PAIRS = 512
MAX_DIAGNOSTIC_PAIRS = 32
SCOPE = ("Experimental authored joint-only residual using privileged current simulator geometry and velocity. "
         "Not a learned policy, contact measurement, collision guarantee or physical validation. "
         "Root actions and passive-object forces are unchanged; required-contact tasks are unsupported.")


@dataclass(frozen=True, slots=True)
class ClearanceConfig:
    margin_model_units: float = .025
    lookahead_seconds: float = .10
    stiffness_weight_per_model_unit: float = 8.
    maximum_force_body_weights: float = .25
    maximum_joint_residual: float = .25
    query_radius_model_units: float = .20

    def __post_init__(self):
        bounds = {"margin_model_units": (.001, .10), "lookahead_seconds": (0., .25),
                  "stiffness_weight_per_model_unit": (0., 50.), "maximum_force_body_weights": (0., 1.),
                  "maximum_joint_residual": (0., .5), "query_radius_model_units": (.001, 1.)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError("Invalid bounded clearance setting: " + name)
        if self.query_radius_model_units <= self.margin_model_units:
            raise ValueError("Clearance query radius must exceed its margin")


class ClearanceController:
    def __init__(self, env, base_action, config):
        if type(config) is not ClearanceConfig or not callable(base_action):
            raise ValueError("Clearance requires immutable settings and a prepared base-action callable")
        scene, task = env.scene, env.task
        if (scene.get("profile") not in {"shared-world-v1", "shared-world-v2"}
                or task.get("profile") != "shared-exercise-task-v1" or task.get("contact_requirements") != []):
            raise ValueError("Clearance supports only equipment tasks without required contact or static support")
        if (env.action_count not in (19, 22) or len(env.action_names) != env.action_count
                or tuple(env.action_names[:19]) != SPATIAL_JOINT_NAMES
                or env.action_count == 22 and tuple(env.action_names[19:]) != ("root_x", "root_y", "root_z")):
            raise ValueError("Clearance requires the exact nineteen joint and optional three root channels")
        engine = env._engine
        if tuple(engine.human_action_names[:env.action_count]) != tuple(env.action_names):
            raise ValueError("Clearance engine and task channels differ")
        self._env, self._engine, self._base_action, self._config = env, engine, base_action, config
        self._model, self._data = engine.model, engine.data
        if not 19 <= self._model.nv <= 128:
            raise ValueError("Clearance generalized-velocity bound exceeded")
        self._scale, self._mass = float(engine.scale), float(engine.human_mass)
        gravity = np.asarray(self._model.opt.gravity, dtype=float)
        if (not np.isfinite([self._scale, self._mass]).all() or min(self._scale, self._mass) <= 0
                or gravity.shape != (3,) or not np.isfinite(gravity).all()):
            raise ValueError("Clearance requires finite positive scale/mass and declared gravity")
        self._weight = self._mass * float(np.linalg.norm(gravity))
        if not math.isfinite(self._weight):
            raise ValueError("Clearance body-weight force exceeds finite arithmetic")
        self._limits = np.asarray(engine.human_limits[:19], dtype=float).copy()
        self._v = np.asarray(engine._v[:19], dtype=int).copy()
        if (self._limits.shape != (19,) or not np.isfinite(self._limits).all() or np.any(self._limits <= 0)
                or self._v.shape != (19,) or len(set(self._v)) != 19
                or np.any(self._v < 0) or np.any(self._v >= self._model.nv)):
            raise ValueError("Clearance requires exact positive motor limits and unique joint velocity addresses")
        self._limits.flags.writeable = self._v.flags.writeable = False
        objects = {obj["id"]: obj for obj in scene["objects"]}
        equipment = task.get("equipment_ids")
        if (not isinstance(equipment, list) or not 1 <= len(equipment) <= 12 or len(set(equipment)) != len(equipment)
                or any(name not in objects or objects[name]["role"] == "static" for name in equipment)):
            raise ValueError("Clearance requires one to twelve declared dynamic equipment objects")
        groups, count = [], 0
        for human_geom, human_name in sorted(engine._human_geom_to_name.items()):
            for name in equipment:
                parts = tuple(engine._object_geometry_ids[name])
                if not parts or len(set(parts)) != len(parts):
                    raise ValueError("Clearance equipment must have unique collision parts")
                count += len(parts)
                if count > MAX_GEOMETRY_PAIRS:
                    raise ValueError("Clearance geometry-pair bound exceeded before querying")
                groups.append((human_geom, human_name, name, parts))
        if not groups:
            raise ValueError("Clearance requires human collision geometry")
        self._groups, self._pair_count = tuple(groups), count
        self._fromto = np.zeros(6)
        self._human_jac = np.zeros((3, self._model.nv))
        self._object_jac = np.zeros((3, self._model.nv))

    def describe(self):
        return {"profile": PROFILE, "settings": asdict(self._config), "scope": SCOPE,
                "geometry_pairs": self._pair_count, "human_equipment_groups": len(self._groups),
                "maximum_geometry_pairs": MAX_GEOMETRY_PAIRS,
                "compound_rule": "nearest_signed_distance_part_per_human_geometry_and_equipment",
                "force_units": "mass_prior_times_actual_model_gravity", "body_weight_force": self._weight,
                "zero_gravity_behavior": "unchanged_base_action", "world_units_per_model_unit": self._scale,
                "action_dtype": "floating_preserved_integer_promoted_only_for_nonzero_correction",
                "base_action_identity": "caller_must_bind_and_record", "maximum_reported_pairs": MAX_DIAGNOSTIC_PAIRS}

    def __call__(self):
        return self._sample(False)[0]

    def sample(self):
        return self._sample(True)

    def _sample(self, diagnostics):
        if self._engine.model is not self._model or self._engine.data is not self._data:
            raise RuntimeError("Prepare clearance again after replacing the engine model or data")
        base = np.asarray(self._base_action())
        if (base.shape != (self._env.action_count,) or base.dtype.kind not in "fiu"
                or not np.isfinite(base).all() or np.any(abs(base) > 1)):
            raise ValueError("Clearance base action must be finite and bounded")
        config = self._config
        report = {"profile": PROFILE, "status": "unchanged", "distance_queries": 0, "jacobian_queries": 0,
                  "active_pairs": 0, "unavailable_pairs": 0, "unactuated_pairs": 0,
                  "reported_pairs": [], "omitted_pair_details": 0, "maximum_actual_joint_residual": 0.}
        if self._weight == 0 or min(config.stiffness_weight_per_model_unit, config.maximum_force_body_weights,
                                    config.maximum_joint_residual) == 0:
            report["status"] = "zero_gravity" if self._weight == 0 else "disabled"
            return base.copy(), report if diagnostics else None
        velocity = np.asarray(self._data.qvel)
        if velocity.shape != (self._model.nv,) or not np.isfinite(velocity).all():
            raise ValueError("Clearance requires finite current generalized velocity")
        torque = np.zeros(19)
        maximum = config.query_radius_model_units * self._scale
        for human_geom, human_name, object_id, parts in self._groups:
            nearest, unavailable = None, False
            for part in parts:
                self._fromto[:] = np.nan
                gap = float(mujoco.mj_geomDistance(self._model, self._data, human_geom, part, maximum, self._fromto))
                report["distance_queries"] += 1
                if not math.isfinite(gap):
                    unavailable = True
                elif gap < maximum and (nearest is None or gap < nearest[0]):
                    nearest = (gap, part, self._fromto.copy())
            detail = {"human_geom": human_name, "object_id": object_id}
            if unavailable:
                detail["unavailable_reason"] = "nonfinite_part_distance"
            elif nearest is None:
                continue
            else:
                gap, part, points = nearest
                detail.update(part_geom_id=int(part), distance_model_units=gap / self._scale)
                if abs(gap) <= 1e-10 * self._scale:
                    detail["unavailable_reason"] = "degenerate_zero_distance_normal"
                elif not np.isfinite(points).all():
                    detail["unavailable_reason"] = "unavailable_closest_points"
                else:
                    normal = (points[:3] - points[3:]) / gap
                    norm = float(np.linalg.norm(normal))
                    if not math.isfinite(norm) or not math.isclose(norm, 1., rel_tol=1e-5, abs_tol=1e-7):
                        detail["unavailable_reason"] = "inconsistent_signed_closest_points"
            if "unavailable_reason" in detail:
                report["unavailable_pairs"] += 1
            else:
                normal /= norm
                mujoco.mj_jac(self._model, self._data, self._human_jac, None, points[:3],
                              int(self._model.geom_bodyid[human_geom]))
                mujoco.mj_jac(self._model, self._data, self._object_jac, None, points[3:],
                              int(self._model.geom_bodyid[part]))
                report["jacobian_queries"] += 2
                if not np.isfinite(self._human_jac).all() or not np.isfinite(self._object_jac).all():
                    raise ValueError("Clearance point Jacobian is nonfinite")
                normal_speed = float(normal @ (self._human_jac - self._object_jac) @ velocity) / self._scale
                if not math.isfinite(normal_speed):
                    raise ValueError("Clearance relative speed exceeds finite arithmetic")
                deficit = max(0., config.margin_model_units - gap / self._scale
                              + config.lookahead_seconds * max(0., -normal_speed))
                fraction = min(config.maximum_force_body_weights, config.stiffness_weight_per_model_unit * deficit)
                if fraction == 0:
                    continue
                gradient = normal @ self._human_jac[:, self._v]
                if not np.isfinite(gradient).all():
                    raise ValueError("Clearance distance gradient exceeds finite arithmetic")
                if np.linalg.norm(gradient) < 1e-12 * self._scale:
                    report["unactuated_pairs"] += 1
                else:
                    torque += gradient * (self._weight * fraction)
                    report["active_pairs"] += 1
                if diagnostics:
                    detail.update(normal_world=normal.tolist(), closest_human_world=points[:3].tolist(),
                        closest_object_world=points[3:].tolist(), normal_speed_model_units_s=normal_speed,
                        predicted_deficit_model_units=deficit, force_body_weights=fraction,
                        distance_gradient_model_units=(gradient / self._scale).tolist())
            if diagnostics:
                if len(report["reported_pairs"]) < MAX_DIAGNOSTIC_PAIRS:
                    report["reported_pairs"].append(detail)
                else:
                    report["omitted_pair_details"] += 1
        if not np.isfinite(torque).all():
            raise ValueError("Clearance accumulated torque exceeds finite arithmetic")
        residual = np.clip(torque / self._limits, -config.maximum_joint_residual, config.maximum_joint_residual)
        if np.any(residual):
            action = base.copy() if base.dtype.kind == "f" else base.astype(float, copy=True)
            base_joints = base[:19].astype(float)
            action[:19] = np.clip(base_joints + residual, -1., 1.)
            # Low-precision rounding cannot enlarge the declared residual budget.
            excess = abs(action[:19].astype(float) - base_joints) > config.maximum_joint_residual
            action[:19][excess] = np.nextafter(action[:19][excess], base[:19][excess])
            actual = action[:19].astype(float) - base_joints
            report["maximum_actual_joint_residual"] = float(np.max(abs(actual)))
            if np.any(actual):
                report["status"] = "active_with_unavailable_pairs" if report["unavailable_pairs"] else "active"
                return action, report if diagnostics else None
        if report["unavailable_pairs"]:
            report["status"] = "unavailable_pairs"
        return base.copy(), report if diagnostics else None


def prepare_clearance_controller(env, *, base_action=None, config=None):
    return ClearanceController(env, env.authored_action if base_action is None else base_action,
                               ClearanceConfig() if config is None else config)
