"""Direct 28-motor posture task and terminal recording for the ankle candidate."""

from copy import deepcopy
import json
import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from PIL import Image

from weft.foot_body import ANKLE_NAMES, FOOT_BODY_CONTRACT, FootBody, audit_foot_body
from weft.io import canonical_hash, write_json
from weft.mobile_human import ROOT_AXES
from weft.recording import Recorder, export_recording, render_snapshot
from weft.whole_body import SUCCESS_LIMITS


PROFILE = "exercise-feet-posture-v1"
HELP = ("Commands: hold [1–6], release [1–6], motor NAME VALUE [1–6], reset, status, help, quit. "
        "VALUE is −1 to 1. Motor commands zero every other motor. Hold uses explicit authored feedback "
        "toward the starting pose; it does not replay the video or measure muscle activity.")


class FootEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, spec, reference, *, horizon_steps=180):
        if type(horizon_steps) is not int or not 1 <= horizon_steps <= 180:
            raise ValueError("Foot posture horizon must contain 1–180 steps")
        self.base = FootBody(spec, reference)
        self.model, self.data = self.base.model, self.base.data
        initial = self.base.source_pose(0)
        if initial is None:
            self.base.close()
            raise ValueError("Candidate unavailable: the original starting pose needs both feet; no later-window substitution")
        self._initial_qpos = initial.qpos.copy()
        self.action_names = tuple(self.base.joint_names) + ANKLE_NAMES + ROOT_AXES
        self._q = np.concatenate((self.base._q_array, self.base.ankle_q, self.base.mobility.q))
        self._v = np.concatenate((self.base._v_array, self.base.ankle_v, self.base.mobility.v))
        self._actuators = np.concatenate((self.base._a_array, self.base.ankle_actuators, self.base.mobility.actuators))
        if self.model.nu != 28 or len(set(self._actuators)) != 28:
            raise ValueError("Foot posture actions must cover all 28 native motors exactly once")
        self.limits = np.concatenate((self.base._motor_limits, np.full(6, FOOT_BODY_CONTRACT["motor_limit_nm"]),
                                      np.full(3, self.base.mobility.limit)))
        self.limits.setflags(write=False)
        self.target = self._initial_qpos[self._q].copy()
        self.target.setflags(write=False)
        self._kp = np.r_[np.full(25, 40.), np.full(3, self.base.mobility.kp)]
        self._kd = np.r_[np.full(25, 5.), np.full(3, self.base.mobility.kd)]
        self.action_space = spaces.Box(-1., 1., shape=(28,), dtype=np.float32)
        self.observation_space = spaces.Box(-1e6, 1e6, shape=(85,), dtype=np.float32)
        self.horizon_steps, self.dt = horizon_steps, float(spec["dt"])
        self.scene = deepcopy(self.base.scene)
        self.scene.update(profile=PROFILE, controlled_id="torso",
            task={"type": "starting_pose_hold", "horizon_steps": horizon_steps, "success_limits": dict(SUCCESS_LIMITS),
                  "target_source_time_s": spec["reference"][0]["time_s"],
                  "scope": "Authored starting-pose task; not source-motion imitation or independently validated balance"},
            action_contract={"names": list(self.action_names), "normalized_bounds": [-1., 1.],
                             "physical_limits": self.limits.tolist(), "units": ["Nm"] * 25 + ["N"] * 3,
                             "substeps_per_action": 8, "hidden_assistance": False},
            foot_candidate={**deepcopy(FOOT_BODY_CONTRACT), "body_sha256": canonical_hash(spec),
                            "foot_reference_sha256": canonical_hash(reference)})
        self.scene["human"].update(profile=PROFILE, control_mode="direct_motor_actions",
                                   joint_names=list(self.action_names[:25]))
        self._initialized, self._done = False, True

    def observation(self):
        if not self._initialized:
            raise RuntimeError("Reset before reading the foot environment")
        result = np.r_[self.data.qpos[self._q], self.data.qvel[self._v], self.target,
                       self.steps / self.horizon_steps].astype(np.float32)
        if not self.observation_space.contains(result):
            raise RuntimeError("Foot observation exceeds its finite declared bounds")
        return result

    def _penetration(self):
        return max((max(0., -float(c.dist)) for c in self.data.contact
                    if self.base.mobility.floor in (c.geom1, c.geom2)), default=0.)

    def reset(self, *, seed=None, options=None):
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("Seed must be a bounded nonnegative integer")
        if options not in (None, {}):
            raise ValueError("Foot posture reset has no jitter or source-pose substitution")
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._initial_qpos
        self.base.mobility.support_enabled, self.base.mobility.mode = False, "direct_motor_actions"
        mujoco.mj_forward(self.model, self.data)
        self._initialized = self.base._initialized = True
        self._done = False
        self.steps, self._angular_squared, self._root_squared, self._episode_return = 0, 0., 0., 0.
        self._last_action, self._failure = np.zeros(28), None
        self._max_penetration = self._penetration()
        return self.observation(), {"hidden_assistance": False, "source_motion_tracking": False}

    def hold_action(self):
        if not self._initialized or self._done:
            raise RuntimeError("Reset before choosing a posture command")
        command = (self._kp * (self.target - self.data.qpos[self._q])
                   - self._kd * self.data.qvel[self._v] + self.data.qfrc_bias[self._v]) / self.limits
        return np.clip(command, -1, 1)

    def _instability(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return "nonfinite_dynamics"
        q, v = self.data.qpos[self._q], self.data.qvel[self._v]
        if np.max(np.abs(q[:25])) > 8 or np.max(np.abs(q[25:])) > 5.1:
            return "position_envelope_exceeded"
        if np.max(np.abs(v[:25])) > 30 or np.max(np.abs(v[25:])) > 10:
            return "velocity_envelope_exceeded"
        return None

    def step(self, action):
        if not self._initialized or self._done:
            raise RuntimeError("Reset before stepping and after the episode ends")
        if isinstance(action, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in action):
            raise ValueError("Foot actions cannot contain booleans")
        values = np.asarray(action)
        if (values.shape != (28,) or values.dtype.kind not in "fiu" or not np.isfinite(values).all()
                or np.any(np.abs(values) > 1)):
            raise ValueError("Foot actions require 28 finite numbers within [-1, 1]")
        if self._instability():
            self._done = True
            raise RuntimeError("Foot state exceeded its envelope before stepping")
        values = values.astype(float, copy=True)
        self.data.ctrl[:] = 0.
        self.data.ctrl[self._actuators] = values * self.limits
        self.data.qfrc_applied[:] = 0.
        self.data.xfrc_applied[:] = 0.
        for _ in range(8):
            mujoco.mj_step(self.model, self.data)
            self._max_penetration = max(self._max_penetration, self._penetration())
            self._failure = self._instability()
            if self._failure:
                break
        if self._failure == "nonfinite_dynamics":
            self._done = True
            raise RuntimeError("Nonfinite foot dynamics; no terminal pose fabricated")
        mujoco.mj_forward(self.model, self.data)
        self._max_penetration = max(self._max_penetration, self._penetration())
        self.steps += 1
        self._last_action = values
        error = self.data.qpos[self._q] - self.target
        angular, root = float(np.mean(error[:25] ** 2)), float(np.mean(error[25:] ** 2))
        self._angular_squared += angular
        self._root_squared += root
        reward = max(0., math.exp(-angular / .35 ** 2 - root / .15 ** 2) - .01 * float(np.mean(values ** 2)))
        self._episode_return += reward
        terminated, truncated = self._failure is not None, self._failure is None and self.steps >= self.horizon_steps
        self._done = terminated or truncated
        return self.observation(), reward, terminated, truncated, self.metrics()

    def metrics(self):
        if not self._initialized:
            raise RuntimeError("Reset before reading posture metrics")
        angular = math.sqrt(self._angular_squared / max(1, self.steps))
        root = math.sqrt(self._root_squared / max(1, self.steps))
        completed = self.steps == self.horizon_steps and self._failure is None
        success = (completed and angular <= SUCCESS_LIMITS["angular_rmse_rad"]
                   and root <= SUCCESS_LIMITS["root_rmse_model_units"]
                   and self._max_penetration <= SUCCESS_LIMITS["max_floor_penetration_model_units"])
        return {"steps": self.steps, "angular_rmse_rad": angular, "root_rmse_model_units": root,
                "max_floor_penetration_model_units": self._max_penetration, "completed_horizon": completed,
                "success": success, "failure": self._failure, "episode_return": self._episode_return,
                "hidden_assistance": False, "scope": "Starting-pose task only; physical accuracy unverified"}

    def snapshot(self):
        metrics = self.metrics()
        self.base.steps, self.base.success = self.steps, metrics["success"]
        self.base.terminated = self._failure is not None
        self.base.truncated = self._done and self._failure is None
        state = self.base.snapshot()
        human = state["human"]
        for key in ("selected_joint", "target_angle", "target_source"):
            human.pop(key, None)
        human.update(control_mode="direct_motor_actions", action_names=list(self.action_names),
                     normalized_action=self._last_action.tolist(), motor_forces=self.data.ctrl[self._actuators].tolist(),
                     motor_units=["Nm"] * 25 + ["N"] * 3)
        state.update(controlled_id="torso", posture_metrics=metrics,
                     scope="Simulated ankle/foot candidate; authored starting-pose task, not source replay")
        return state

    def render(self):
        return render_snapshot(self.snapshot(), self.scene, width=640)

    def close(self):
        self._initialized, self._done = False, True
        self.base.close()


def command_action(command, env):
    parts = command.split()
    if not parts or parts[0] not in {"hold", "release", "motor"}:
        raise ValueError(HELP)
    count = 3 if parts[0] == "motor" else 1
    if len(parts) not in (count, count + 1):
        raise ValueError(HELP)
    steps = int(parts[-1]) if len(parts) == count + 1 else 1
    if not 1 <= steps <= 6:
        raise ValueError("Use 1–6 steps per command")
    if parts[0] == "hold":
        return env.hold_action, steps
    action = np.zeros(28)
    if parts[0] == "motor":
        if parts[1] not in env.action_names:
            raise ValueError("Unknown motor; use: " + ", ".join(env.action_names))
        value = float(parts[2])
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("Motor value must be finite and between −1 and 1")
        action[env.action_names.index(parts[1])] = value
    return lambda: action.copy(), steps


def console(body, feet, recording, *, seed=1729, input_fn=input, output_fn=print):
    body, feet, recording = Path(body).resolve(), Path(feet).resolve(), Path(recording).resolve()
    if recording.exists() or any(recording.is_relative_to(p.parent) for p in (body, feet)):
        raise ValueError("Choose a new recording directory outside both source artifact directories")
    for path in (body, feet):
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Body and foot documents must each fit within 16 MiB")
    spec, reference = json.loads(body.read_text()), json.loads(feet.read_text())
    audit = audit_foot_body(spec, reference)
    env = FootEnv(spec, reference)
    recorder, failure = Recorder(recording, env.scene, profile="exercise"), None

    def preview():
        Image.fromarray(env.render()).save(recording / "current-frame.png")
        output_fn(json.dumps(env.metrics(), allow_nan=False))

    try:
        env.reset(seed=seed)
        recorder.start(env.snapshot(), seed=seed)
        write_json(recording / "human.json", spec)
        write_json(recording / "foot-reference.json", reference)
        write_json(recording / "candidate-audit.json", audit)
        output_fn(HELP)
        output_fn("Foot motors: " + ", ".join(ANKLE_NAMES))
        output_fn("Candidate floor checks: " + json.dumps(audit["summary"]))
        output_fn("Frame updated after each command: " + str(recording / "current-frame.png"))
        preview()
        while True:
            try:
                command = input_fn("feet> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if command == "quit":
                break
            if command == "help":
                output_fn(HELP)
                continue
            if command == "status":
                output_fn(json.dumps(env.metrics()))
                continue
            if command == "reset":
                env.reset(seed=seed)
                recorder.append({"type": "reset", "seed": seed, "snapshot": env.snapshot()})
                preview()
                continue
            try:
                if env._done:
                    raise ValueError("Episode ended; reset before another action")
                choose, steps = command_action(command, env)
                for _ in range(steps):
                    recorder.check_capacity()
                    action = choose()
                    try:
                        _, reward, terminated, truncated, _ = env.step(action)
                    except RuntimeError as error:
                        failure = {"reason": str(error), "attempted_action": action.tolist(), "command": command,
                                   "completed_recorded_steps": recorder.summary()["steps"]}
                        break
                    recorder.append({"type": "step", "action": action.tolist(), "reward": reward,
                                     "terminated": terminated, "truncated": truncated, "snapshot": env.snapshot()})
                    if terminated or truncated:
                        break
                if failure:
                    output_fn(json.dumps(failure))
                    break
                preview()
            except ValueError as error:
                output_fn(str(error))
    finally:
        if recorder._started is not None:
            recorder.close()
        env.close()
    if failure:
        write_json(recording / "failed-attempt.json", failure)
    return export_recording(recording) | {"failed_attempt": failure}
