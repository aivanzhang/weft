"""Read-only force diagnostics for piecewise-linear, directly actuated references."""

import math

import mujoco
import numpy as np


PROFILE = "reference-dynamics-diagnostic-v1"
MAX_POSES, MAX_AXES = 900, 64


def _array(value, name):
    if isinstance(value, (list, tuple)):
        for row in value:
            if isinstance(row, (bool, np.bool_)) or (isinstance(row, (list, tuple)) and
                    any(isinstance(item, (bool, np.bool_)) for item in row)):
                raise ValueError(f"{name} cannot contain booleans")
    result = np.asarray(value)
    if result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"{name} requires finite numeric values")
    return result.astype(float, copy=True)


def reference_samples(times, positions, transition_s):
    """Knot accelerations are declared finite-window probes, not exact path derivatives."""
    times, positions = _array(times, "Times"), _array(positions, "Positions")
    if times.ndim != 1 or not 2 <= len(times) <= MAX_POSES:
        raise ValueError("Reference requires 2–900 times")
    if positions.ndim != 2 or positions.shape[0] != len(times) or not 1 <= positions.shape[1] <= MAX_AXES:
        raise ValueError("Reference requires one row per time and 1–64 scalar axes")
    gaps = np.diff(times)
    if (np.any(times < 0) or np.any(times > 86400) or np.any(gaps < 1e-6)
            or times[-1] - times[0] > 60 or np.any(np.abs(positions) > 1e4)):
        raise ValueError("Reference exceeds its ordered time/position bounds")
    if (type(transition_s) not in (int, float) or not math.isfinite(transition_s)
            or not 1e-5 <= transition_s <= min(1., float(gaps.min()))):
        raise ValueError("Transition window must fit within every reference interval")
    slopes = np.diff(positions, axis=0) / gaps[:, None]
    if np.any(np.abs(slopes) > 1e4):
        raise ValueError("Reference velocity exceeds its numeric bound")
    samples = []
    for index, slope in enumerate(slopes):
        samples.append({"kind": "interval_midpoint", "source_indices": [index, index + 1],
            "time_s": float((times[index] + times[index + 1]) / 2),
            "position": (positions[index] + positions[index + 1]) / 2,
            "velocity": slope.copy(), "acceleration": np.zeros(positions.shape[1]),
            "velocity_jump": np.zeros(positions.shape[1])})
    # Reset begins at rest; terminal braking is outside the observed-motion task.
    for index in range(len(times) - 1):
        before = np.zeros(positions.shape[1]) if index == 0 else slopes[index - 1]
        after = slopes[index]
        samples.append({"kind": "reset_transition_probe" if index == 0 else "knot_transition_probe",
            "source_indices": list(range(max(0, index - 1), index + 2)), "time_s": float(times[index]),
            "position": positions[index].copy(), "velocity": (before + after) / 2,
            "acceleration": (after - before) / transition_s, "velocity_jump": after - before})
    return samples


def _direct_axes(model, names, limits):
    if (not isinstance(names, (list, tuple)) or not 1 <= len(names) <= MAX_AXES
            or any(not isinstance(name, str) or not 1 <= len(name) <= 128 for name in names)
            or len(set(names)) != len(names)
            or not model.nq == model.nv == model.njnt == model.nu == len(names)
            or model.nbody > 256 or model.ngeom > 256 or model.nplugin or model.nflex):
        raise ValueError("Diagnostic requires a bounded, fully actuated scalar-joint model")
    if model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE):
        raise ValueError("Diagnostic requires continuous-time inverse dynamics")
    limits = _array(limits, "Force limits")
    if limits.shape != (len(names),) or np.any(limits <= 0) or np.any(limits > 1e6):
        raise ValueError("Each axis requires its positive bounded force limit")
    q, v, engine_limits = [], [], []
    for name, limit in zip(names, limits):
        joint, actuator = model.joint(name).id, model.actuator(f"{name}_motor").id
        if (int(model.jnt_type[joint]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
                or model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
                or model.actuator_trnid[actuator, 0] != joint
                or not np.array_equal(model.actuator_gear[actuator], [1., 0., 0., 0., 0., 0.])
                or model.actuator_dyntype[actuator] != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_gainprm[actuator, 0] != 1
                or model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_NONE
                or not model.actuator_ctrllimited[actuator]
                or model.actuator_ctrlrange[actuator, 0] > -limit or model.actuator_ctrlrange[actuator, 1] < limit
                or (model.actuator_forcelimited[actuator] and
                    (model.actuator_forcerange[actuator, 0] > -limit or model.actuator_forcerange[actuator, 1] < limit))
                or (model.jnt_actfrclimited[joint] and
                    (model.jnt_actfrcrange[joint, 0] > -limit or model.jnt_actfrcrange[joint, 1] < limit))):
            raise ValueError("Diagnostic requires unit-gear direct motors supporting the declared force limits")
        q.append(int(model.jnt_qposadr[joint]))
        v.append(int(model.jnt_dofadr[joint]))
        # The runtime can impose a stricter cap than the engine's actuator range.
        engine_limits.append({"name": name, "control_range": model.actuator_ctrlrange[actuator].tolist(),
            "force_range": model.actuator_forcerange[actuator].tolist() if model.actuator_forcelimited[actuator] else None,
            "joint_force_range": model.jnt_actfrcrange[joint].tolist() if model.jnt_actfrclimited[joint] else None})
    return np.array(q), np.array(v), limits, engine_limits


def audit_reference_dynamics(model, times, positions, *, names, limits, transition_s):
    """Inspect source targets in separate MjData; never step or modify the live model/state."""
    samples = reference_samples(times, positions, transition_s)
    q, v, limits, engine_limits = _direct_axes(model, names, limits)
    if len(q) != len(samples[0]["position"]):
        raise ValueError("Reference axes must exactly match the motor list")
    data = mujoco.MjData(model)
    rows = []
    for sample in samples:
        mujoco.mj_resetData(model, data)
        data.qpos[q], data.qvel[v], data.qacc[v] = sample["position"], sample["velocity"], sample["acceleration"]
        mujoco.mj_inverse(model, data)
        required = data.qfrc_inverse[v].copy()
        impulse, jump = np.zeros(model.nv), np.zeros(model.nv)
        jump[v] = sample["velocity_jump"]
        mujoco.mj_mulM(model, data, impulse, jump)
        inertia, accel = np.zeros(model.nv), np.zeros(model.nv)
        accel[v] = sample["acceleration"]
        mujoco.mj_mulM(model, data, inertia, accel)
        residual = required - (inertia[v] + data.qfrc_bias[v] - data.qfrc_passive[v] - data.qfrc_constraint[v])
        arrays = (required, impulse, inertia, residual, data.qfrc_bias, data.qfrc_passive, data.qfrc_constraint)
        if any(not np.isfinite(array).all() or np.any(np.abs(array) > 1e12) for array in arrays):
            raise ValueError("Inverse-dynamics result exceeds the finite diagnostic envelope")
        row = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in sample.items()}
        row.update({"required_motor_force": required.tolist(), "force_limit_ratio": (np.abs(required) / limits).tolist(),
            "inertia_force": inertia[v].tolist(), "bias_force": data.qfrc_bias[v].tolist(),
            "passive_force": data.qfrc_passive[v].tolist(), "constraint_force": data.qfrc_constraint[v].tolist(),
            "frozen_configuration_impulse": impulse[v].tolist(),
            "motor_only_impulse_time_s": (np.abs(impulse[v]) / limits).tolist(),
            "contacts": int(data.ncon), "constraint_rows": int(data.nefc),
            "force_balance_residual_max": float(np.max(np.abs(residual))),
            "over_limit_axes": [name for name, force, limit in zip(names, required, limits) if abs(force) > limit]})
        rows.append(row)
    summaries = {}
    for kind in ("interval_midpoint", "reset_transition_probe", "knot_transition_probe"):
        selected = [row for row in rows if row["kind"] == kind]
        ratios = np.array([row["force_limit_ratio"] for row in selected])
        summaries[kind] = {"samples": len(selected),
            "over_limit_samples": sum(bool(row["over_limit_axes"]) for row in selected),
            "constraint_free_samples": sum(row["constraint_rows"] == 0 for row in selected),
            "constraint_free_over_limit_samples": sum(row["constraint_rows"] == 0 and bool(row["over_limit_axes"]) for row in selected),
            "maximum_force_limit_ratio": float(ratios.max()) if selected else None,
            "over_limit_counts_by_axis": {name: int(np.sum(ratios[:, i] > 1)) if selected else 0 for i, name in enumerate(names)}}
    return {"profile": PROFILE, "status": "diagnostic_only", "physical_validation": "unverified",
        "names": list(names), "force_limits": limits.tolist(), "engine_limits": engine_limits,
        "transition_window_s": transition_s,
        "source_poses": len(times), "inverse_calls": len(rows), "physics_steps": 0,
        "summary": summaries, "samples": rows,
        "limitations": ["Authored mass, damping, inertia, force and contact assumptions remain unchanged.",
            "Midpoints sample the exact linear-position path; they do not certify whole intervals.",
            "Knot velocities jump; finite-window acceleration probes are not exact path derivatives.",
            "Motor-only impulse times freeze configuration and omit other forces; they are not minimum transition times.",
            "Constraint-assisted samples are separate; no contact or physical accuracy is established.",
            "No controller, source trajectory, evaluator, task gate or terminal braking is changed."]}
