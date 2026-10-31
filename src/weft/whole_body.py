"""Direct 22-motor reference tracking; no hidden servo or recovered human intent."""

from copy import deepcopy
import math

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from weft.human import HumanEnv, reference_velocity_index
from weft.mobile_human import ROOT_AXES


GROUP_NAMES = ("torso", "proximal", "distal", "root")
SUCCESS_LIMITS = {"angular_rmse_rad": .35, "root_rmse_model_units": .15,
                  "max_floor_penetration_model_units": .025}
OBSERVATION_FIELDS = (("position", 22), ("velocity", 22), ("target_position", 22),
                      ("target_velocity", 22), ("policy_features", 66), ("source_phase", 1))
RESET_JITTER = {"angular_position": .01, "angular_velocity": .02,
                "root_horizontal_position": .002, "root_upward_position": .002, "root_velocity": .01}


class WholeBodyEnv(gym.Env):
    """All actuator commands are normalized explicit actions held for eight substeps."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, spec, *, start_time_s=0., horizon_steps=180, seed=None):
        if 'camera_reference' in spec:
            raise ValueError('Camera-world candidates need independent trajectory validation before whole-body training')
        if spec.get("profile") != "exercise-v3":
            raise ValueError("Whole-body tracking requires the mobile exercise-v3 model")
        if spec.get("dt") != 1 / 30:
            raise ValueError("Whole-body v1 requires a 30-Hz control timestep")
        if type(horizon_steps) is not int or not 1 <= horizon_steps <= 180:
            raise ValueError("Whole-body horizon must contain 1–180 control steps")
        if type(start_time_s) not in (int, float) or not math.isfinite(start_time_s) or start_time_s < 0:
            raise ValueError("Whole-body start offset must be a finite nonnegative time")
        self.base = HumanEnv(spec)
        self.model, self.data = self.base.model, self.base.data
        self.dt, self.horizon_steps = float(spec["dt"]), horizon_steps
        self._times = self.base._reference_times.copy()
        self.source_start = float(self._times[0] + start_time_s)
        self.start_time_s = float(start_time_s)
        if self.source_start + horizon_steps * self.dt > self._times[-1] + 1e-9:
            self.base.close()
            raise ValueError("Whole-body window exceeds the available reference; end holding is not evaluation coverage")
        self.action_names = tuple(self.base.joint_names) + ROOT_AXES
        self.joint_names = self.base.joint_names
        self.group_indices = np.array([0 if name.startswith("torso_") else
                                       1 if "shoulder" in name or "hip" in name else 2
                                       for name in self.joint_names] + [3, 3, 3], dtype=int)
        self.group_names = GROUP_NAMES
        mobility = self.base.mobility
        self._q = np.concatenate((self.base._q_array, mobility.q))
        self._v = np.concatenate((self.base._v_array, mobility.v))
        self._actuators = np.concatenate((self.base._a_array, mobility.actuators))
        if len(self.action_names) != 22 or len(set(self._actuators)) != self.model.nu or self.model.nu != 22:
            raise ValueError("Whole-body action contract must explicitly cover all 22 actuators")
        self.limits = np.concatenate((self.base._motor_limits, np.repeat(mobility.limit, 3)))
        self.limits.setflags(write=False)
        self._kp = np.concatenate((np.full(19, 40.), np.repeat(mobility.kp, 3)))
        self._kd = np.concatenate((np.full(19, 5.), np.repeat(mobility.kd, 3)))
        self._reference = np.vstack((self.base._reference_matrix, mobility.positions.T))
        self._slopes = np.diff(self._reference, axis=1) / np.diff(self._times)
        if not np.isfinite(self._slopes).all() or np.max(np.abs(self._slopes)) > 1e4:
            raise ValueError("Whole-body reference velocities exceed the finite feature envelope")
        self.action_space = spaces.Box(-1., 1., shape=(22,), dtype=np.float32)
        low, high = np.full(155, -1e6, np.float32), np.full(155, 1e6, np.float32)
        low[-1], high[-1] = 0., 1.
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.scene = deepcopy(self.base.scene)
        self.scene.update(profile="exercise-imitation-v1", controlled_id="torso",
                          task={"type": "whole_body_reference_tracking", "start_offset_s": self.start_time_s,
                                "source_start_time_s": self.source_start, "horizon_steps": horizon_steps,
                                "success_limits": dict(SUCCESS_LIMITS), "reference_conditioned": True,
                                "scope": "Authored simulated trajectory tracking; not recovered intent or independent balance"},
                          action_contract={"names": list(self.action_names), "normalized_bounds": [-1., 1.],
                                           "physical_limits": self.limits.tolist(), "units": ["Nm"] * 19 + ["N"] * 3,
                                           "substeps_per_action": 8, "hidden_assistance": False})
        self.scene["human"]["control_mode"] = "direct_motor_actions"
        self._initial_seed, self._initialized, self._done = seed, False, True
        self._last_action = np.zeros(22)

    def _target_at(self, elapsed):
        timestamp = float(np.clip(self.source_start + elapsed, self._times[0], self._times[-1]))
        index = min(len(self._times) - 2, max(0, int(np.searchsorted(self._times, timestamp, side="right")) - 1))
        target = self._reference[:, index] + self._slopes[:, index] * (timestamp - self._times[index])
        velocity_index = reference_velocity_index(
            self._times, timestamp, elapsed=elapsed, timestep=self.model.opt.timestep, source_start=self.source_start)
        velocity = self._slopes[:, velocity_index].copy() if velocity_index is not None else np.zeros(22)
        return target, velocity

    def target_state(self):
        return self._target_at(self.data.time)

    def _state(self):
        target, velocity = self.target_state()
        q, v = self.data.qpos[self._q], self.data.qvel[self._v]
        features = np.column_stack((self._kp * (target - q), self._kd * (velocity - v),
                                    self.data.qfrc_bias[self._v])) / self.limits[:, None]
        return q, v, target, velocity, features

    def feature_matrix(self):
        if not self._initialized:
            raise RuntimeError("Reset before reading whole-body policy features")
        return self._state()[-1].copy()

    def observation(self):
        if not self._initialized:
            raise RuntimeError("Reset before reading a whole-body observation")
        q, v, target, velocity, features = self._state()
        phase = (self.source_start + self.data.time - self._times[0]) / (self._times[-1] - self._times[0])
        result = np.concatenate((q, v, target, velocity, features.ravel(), [np.clip(phase, 0., 1.)])).astype(np.float32)
        if not self.observation_space.contains(result):
            raise RuntimeError("Whole-body observation exceeds finite declared bounds")
        return result

    def reset(self, *, seed=None, options=None):
        if seed is None and not self._initialized:
            seed = self._initial_seed
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("Whole-body seed must be a bounded nonnegative integer")
        if options is not None and not isinstance(options, dict):
            raise ValueError("Whole-body reset options must be a mapping")
        options = options or {}
        allowed = {"jitter", "jitter_scale", "qpos_offset", "qvel_offset"}
        if set(options) - allowed or type(options.get("jitter", True)) is not bool:
            raise ValueError("Whole-body reset accepts boolean jitter, jitter_scale, qpos_offset, and qvel_offset")
        use_jitter = options.get("jitter", True)
        raw_scale = options.get("jitter_scale", 1. if use_jitter else 0.)
        if type(raw_scale) not in (int, float) or not math.isfinite(raw_scale) or not 0. <= float(raw_scale) <= 5.:
            raise ValueError("Whole-body jitter_scale must be finite within [0, 5]")
        scale = float(raw_scale)
        if not use_jitter and scale > 0.:
            raise ValueError("Whole-body jitter_scale > 0 requires jitter=True")
        qpos_offset = options.get("qpos_offset")
        qvel_offset = options.get("qvel_offset")
        qpos_vec = qvel_vec = None
        if qpos_offset is not None:
            if isinstance(qpos_offset, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in qpos_offset):
                raise ValueError("Whole-body qpos_offset cannot contain booleans")
            qpos_vec = np.asarray(qpos_offset)
            if (
                qpos_vec.shape != (22,)
                or qpos_vec.dtype.kind not in "fiu"
                or not np.isfinite(qpos_vec).all()
                or np.any(np.abs(qpos_vec[:19]) > .25)
                or np.any(np.abs(qpos_vec[19:]) > .10)
            ):
                raise ValueError("Whole-body qpos_offset requires 22 finite values within [±0.25 rad, ±0.10 root]")
            qpos_vec = qpos_vec.astype(float, copy=True)
        if qvel_offset is not None:
            if isinstance(qvel_offset, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in qvel_offset):
                raise ValueError("Whole-body qvel_offset cannot contain booleans")
            qvel_vec = np.asarray(qvel_offset)
            if (
                qvel_vec.shape != (22,)
                or qvel_vec.dtype.kind not in "fiu"
                or not np.isfinite(qvel_vec).all()
                or np.any(np.abs(qvel_vec[:19]) > 1.)
                or np.any(np.abs(qvel_vec[19:]) > .5)
            ):
                raise ValueError("Whole-body qvel_offset requires 22 finite values within [±1.0 rad/s, ±0.5 root/s]")
            qvel_vec = qvel_vec.astype(float, copy=True)

        super().reset(seed=seed)
        self.base.reset(seed=seed, options={"jitter": False})
        target, _ = self._target_at(0.)
        self.data.qpos[self._q] = target
        self.data.qvel[:] = 0.
        if use_jitter and scale > 0.:
            self.data.qpos[self._q[:19]] += self.np_random.uniform(-.01 * scale, .01 * scale, 19)
            self.data.qvel[self._v[:19]] = self.np_random.uniform(-.02 * scale, .02 * scale, 19)
            self.data.qpos[self._q[19:21]] += self.np_random.uniform(-.002 * scale, .002 * scale, 2)
            self.data.qpos[self._q[21]] += self.np_random.uniform(0., .002 * scale)
            self.data.qvel[self._v[19:]] = self.np_random.uniform(-.01 * scale, .01 * scale, 3)
        if qpos_vec is not None:
            self.data.qpos[self._q] += qpos_vec
        if qvel_vec is not None:
            self.data.qvel[self._v] += qvel_vec
        if (use_jitter and scale != 1.) or qpos_vec is not None:
            for idx, name in enumerate(self.joint_names):
                low, high = self.base.joint_limits[name]
                self.data.qpos[self._q[idx]] = float(np.clip(self.data.qpos[self._q[idx]], low + 1e-4, high - 1e-4))
        self.data.ctrl[:] = 0.
        self.data.qfrc_applied[:] = 0.
        self.data.xfrc_applied[:] = 0.
        self.base.mobility.support_enabled, self.base.mobility.mode = False, "direct_motor_actions"
        mujoco.mj_forward(self.model, self.data)
        self._initialized, self._done = True, False
        self.steps, self._episode_return = 0, 0.
        self._angular_squared, self._root_squared = 0., 0.
        self._metric_samples = 0
        self._max_penetration = self._floor_penetration()
        self._failure = None
        self._last_action = np.zeros(22)
        q_delta = self.data.qpos[self._q] - target
        v_now = self.data.qvel[self._v]
        return self.observation(), {
            "source_time_s": self.source_start,
            "reset_jitter": use_jitter,
            "jitter_scale": scale if use_jitter else 0.,
            "initial_angular_offset_rmse_rad": float(np.sqrt(np.mean(q_delta[:19] ** 2))),
            "initial_root_offset_rmse_model_units": float(np.sqrt(np.mean(q_delta[19:] ** 2))),
            "initial_angular_speed_rmse_rad_s": float(np.sqrt(np.mean(v_now[:19] ** 2))),
            "initial_root_speed_rmse_model_units_s": float(np.sqrt(np.mean(v_now[19:] ** 2))),
            "initial_floor_penetration_model_units": float(self._max_penetration),
            "hidden_assistance": False,
            "action_names": list(self.action_names),
        }

    def _floor_penetration(self):
        return max((max(0., -float(self.data.contact[i].dist)) for i in range(self.data.ncon)
                    if self.base.mobility.floor in (self.data.contact[i].geom1, self.data.contact[i].geom2)), default=0.)

    def _instability(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return "nonfinite_dynamics"
        q, v = self.data.qpos[self._q], self.data.qvel[self._v]
        if np.max(np.abs(q[:19])) > 8 or np.max(np.abs(q[19:])) > 5.1:
            return "position_envelope_exceeded"
        if np.max(np.abs(v[:19])) > 30 or np.max(np.abs(v[19:])) > 10:
            return "velocity_envelope_exceeded"
        return None

    def step(self, action):
        if not self._initialized or self._done:
            raise RuntimeError("Reset before whole-body stepping and after episode completion")
        if isinstance(action, (list, tuple)) and any(isinstance(value, (bool, np.bool_)) for value in action):
            raise ValueError("Whole-body actions cannot contain booleans")
        values = np.asarray(action)
        if (values.shape != (22,) or values.dtype.kind not in "fiu" or not np.isfinite(values).all()
                or np.any(values < -1) or np.any(values > 1)):
            raise ValueError("Whole-body action requires 22 finite numeric values within [-1, 1]")
        values = values.astype(float, copy=True)
        if self._instability() == "nonfinite_dynamics":
            self._failure, self._done = "nonfinite_dynamics", True
            raise RuntimeError("Nonfinite whole-body state; physics was not advanced")
        forces = values * self.limits
        self.data.ctrl[:] = 0.
        self.data.ctrl[self._actuators] = forces
        self.data.qfrc_applied[:] = 0.
        self.data.xfrc_applied[:] = 0.
        # Direct motor commands remain unchanged across the same eight physics ticks for every policy.
        for _ in range(8):
            mujoco.mj_step(self.model, self.data)
            self._max_penetration = max(self._max_penetration, self._floor_penetration())
            self._failure = self._instability()
            if self._failure:
                break
        self.steps += 1
        self._last_action = values
        if self._failure == "nonfinite_dynamics":
            self._done = True
            raise RuntimeError("Nonfinite whole-body dynamics; no fabricated terminal position or observation")
        mujoco.mj_forward(self.model, self.data)
        self._max_penetration = max(self._max_penetration, self._floor_penetration())
        target, _ = self.target_state()
        error = self.data.qpos[self._q] - target
        angular, root = float(np.mean(error[:19] ** 2)), float(np.mean(error[19:] ** 2))
        self._angular_squared += angular
        self._root_squared += root
        self._metric_samples += 1
        tracking = math.exp(-angular / SUCCESS_LIMITS["angular_rmse_rad"] ** 2 - root / SUCCESS_LIMITS["root_rmse_model_units"] ** 2)
        effort = .5 * (float(np.mean(values[:19] ** 2)) + float(np.mean(values[19:] ** 2)))
        # Nonnegative rewards make the fixed-horizon zero tail non-profitable on failure.
        reward = max(0., tracking - .01 * effort)
        self._episode_return += reward
        terminated = self._failure is not None
        truncated = not terminated and self.steps >= self.horizon_steps
        self._done = terminated or truncated
        info = {**self.metrics(), "tracking_reward": tracking, "normalized_effort": effort,
                "applied_motor_forces": forces.tolist(), "hidden_assistance": False,
                "source_time_s": self.source_start + float(self.data.time)}
        return self.observation(), reward, terminated, truncated, info

    def metrics(self):
        if not self._initialized:
            raise RuntimeError("Reset before whole-body metrics")
        valid = self._failure != "nonfinite_dynamics"
        angular = (math.sqrt(self._angular_squared / self._metric_samples) if self._metric_samples else 0.) if valid else None
        root = (math.sqrt(self._root_squared / self._metric_samples) if self._metric_samples else 0.) if valid else None
        completed = self.steps == self.horizon_steps and self._failure is None
        failures = [name for name, value in (("angular_tracking", angular is None or angular > SUCCESS_LIMITS["angular_rmse_rad"]),
                                            ("root_tracking", root is None or root > SUCCESS_LIMITS["root_rmse_model_units"]),
                                            ("floor_penetration", self._max_penetration > SUCCESS_LIMITS["max_floor_penetration_model_units"])) if value]
        failure = self._failure or (",".join(failures) if completed and failures else None)
        return {"angular_rmse_rad": angular, "root_rmse_model_units": root,
                "max_floor_penetration_model_units": self._max_penetration,
                "completed_horizon": completed, "success": completed and not failures,
                "failure": failure, "steps": self.steps, "metric_samples": self._metric_samples,
                "tracking_metrics_valid": valid,
                "return": self._episode_return / self.horizon_steps, "episode_return": self._episode_return}

    def snapshot(self):
        if not self._initialized:
            raise RuntimeError("Reset before whole-body snapshots")
        metrics = self.metrics()
        self.base.steps = self.steps
        self.base.success = metrics["success"]
        self.base.terminated = self._failure is not None
        self.base.truncated = self._done and self._failure is None
        state = self.base.snapshot()
        state.update(controlled_id="torso", episode_steps=self.steps,
                     source_time_s=self.source_start + float(self.data.time), whole_body_metrics=metrics)
        human = state["human"]
        for key in ("selected_joint", "target_angle", "target_source"):
            human.pop(key, None)
        human.update(control_mode="direct_motor_actions", action_names=list(self.action_names),
                     normalized_action=self._last_action.tolist(), motor_forces=self.data.ctrl[self._actuators].tolist(),
                     motor_units=["Nm"] * 19 + ["N"] * 3,
                     control_scope="Explicit policy motor commands; no hidden assistance, measured muscle forces or recovered intent")
        return state

    def render(self):
        from weft.recording import render_snapshot
        return render_snapshot(self.snapshot(), self.scene, width=640)

    def close(self):
        self._done = True
        self._initialized = False
        self.base.close()
