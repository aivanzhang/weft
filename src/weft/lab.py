"""Loopback-only inspection and control of the authoritative Python environment."""

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import uuid4

import numpy as np

from weft.env import RollingEnv, task_success
from weft.exercise_coverage import TERMINAL, load_coverage
from weft.io import canonical_hash, local_time, write_json
from weft.lab_inspection import load_inspection
from weft.lab_worker import atomic_json
from weft.pipeline import verify_artifacts
from weft.recording import Recorder, export_recording
from weft.world import World


class RollingSession:
    def __init__(self, root):
        self.world = World.from_dict(json.loads((root / "world.json").read_text()))
        self.env = RollingEnv(self.world)
        self.q = np.load(root / "policy.npz", allow_pickle=False)["q"]
        w = self.world
        self.scene = {"schema_version": 1, "profile": "rolling", "width": 1,
                      "height": max(w.height_px / w.width_px, 0.4), "dt": self.env.task.dt,
                      "controlled_id": "ball-0", "target": [0.75, w.center_y(0.75)],
                      "task": self.env.task.to_dict(), "support": {"slope": w.support_slope,
                      "intercept": w.support_intercept}, "world": w.to_dict(),
                      "objects": [{"id": "ball-0", "label": "Ball", "role": "active", "shape": "circle",
                                   "radius": w.radius, "color": "#C5A059"}],
                      "assumptions": ["Horizontal rolling profile; authored left/right control and parking task"]}
        self.done = False
        self.reset(0)

    def reset(self, seed):
        self.observation, _ = self.env.reset(seed=seed)
        self.done = self.terminated = self.truncated = False
        return self.snapshot()

    def policy_action(self):
        from weft.learning import state_index
        return {0: 1, 1: 0, 2: 2}[int(np.argmax(self.q[state_index(self.observation)]))]

    def step(self, action):
        if action not in (0, 1, 2):
            raise ValueError("Rolling profile supports left, right, and coast only")
        self.observation, reward, self.terminated, self.truncated, info = self.env.step({0: 1, 1: 0, 2: 2}[action])
        self.done = self.terminated or self.truncated
        return reward, self.terminated, self.truncated, info

    def snapshot(self):
        return {"objects": [{**self.scene["objects"][0], "position": [self.env.x, self.world.center_y(self.env.x)],
                             "velocity": [self.env.velocity, 0], "angle": 0}],
                "target": [self.env.target, self.world.center_y(self.env.target)], "controlled_id": "ball-0",
                "steps": self.env.steps, "time_s": self.env.steps * self.env.task.dt,
                "success": task_success(self.env.x, self.env.velocity, self.env.target, self.env.task),
                "terminated": self.terminated, "truncated": self.truncated}


class PlanarSession:
    def __init__(self, root):
        from weft.scene import SceneEnv
        self.scene = json.loads((root / "scene.json").read_text())
        self.policy = json.loads((root / "policy.json").read_text())
        self.env = SceneEnv(self.scene)
        self.scene = self.env.scene
        self.trained_controlled_id = self.policy.get("controlled_id", self.scene["controlled_id"])
        self.reset(0)

    def reset(self, seed):
        self.observation, _ = self.env.reset(seed=seed, options={"jitter": False})
        self.done = False
        return self.snapshot()

    def select_actor(self, name):
        self.observation = self.env.select_actor(name)
        self.scene = self.env.scene
        self.done = self.env._done

    def policy_action(self):
        from weft.scene_learning import policy_action
        if self.scene["controlled_id"] != self.trained_controlled_id:
            raise ValueError(f"This policy was trained for {self.trained_controlled_id}; select that object first")
        return int(policy_action(self.policy, self.observation))

    def step(self, action):
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated
        return reward, terminated, truncated, info

    def snapshot(self):
        return self.env.snapshot()


class HumanSession:
    def __init__(self, root):
        from weft.human import HumanEnv
        self.spec = json.loads((root / "human.json").read_text())
        self.policy = json.loads((root / "policy.json").read_text())
        self.env = HumanEnv(self.spec)
        self.env.select_joint(self.policy["selected_joint"])
        self.scene = self.env.scene
        self.scene["joint_names"] = list(self.env.joint_names)
        self.reset(0)

    def reset(self, seed):
        self.observation, _ = self.env.reset(seed=seed, options={"jitter": False})
        self.done = self.reference_finished = False
        return self.snapshot()

    def select_joint(self, name):
        self.observation = self.env.select_joint(name)
        self.done = False

    def policy_action(self):
        from weft.human_learning import policy_action
        if self.env.selected_joint != self.policy["selected_joint"]:
            raise ValueError(f"This policy was trained for {self.policy['selected_joint']}; select that joint first")
        return int(policy_action(self.policy, self.observation))

    def step(self, action):
        if action >= 6 and self.spec["profile"] == "exercise-v3":
            self.env.advance_physics(action)
            self.observation = self.env.observation()
            self.done = self.env.steps >= 1800
            return 0.0, False, self.done, {"scope": "Authored mobile-body intervention, not learned balance"}
        if action == 5:
            self.env.advance_physics(action)
            self.observation = self.env.observation()
            duration = self.spec["reference"][-1]["time_s"] - self.spec["reference"][0]["time_s"]
            self.done = self.reference_finished = self.env.steps * self.spec["dt"] >= max(duration, self.spec["dt"])
            return 0.0, False, self.done, {"scope": "Authored reference servo, not learned policy"}
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated
        return reward, terminated, truncated, info

    def snapshot(self):
        state = self.env.snapshot()
        if self.reference_finished:
            state["truncated"] = True
        return state


class CameraHumanSession(HumanSession):
    def __init__(self, root):
        from weft.human import HumanEnv
        self.spec = json.loads((root / 'human.json').read_text())
        task = json.loads((root / 'task.json').read_text())
        if 'camera_reference' not in self.spec:
            raise ValueError('Camera-body session requires explicit camera-world provenance')
        self.policy = None
        self.env = HumanEnv(self.spec)
        try:
            self.env.select_joint(task['selected_joint'])
            self.scene = self.env.scene
            self.scene['joint_names'] = list(self.env.joint_names)
            self.reset(0)
        except Exception:
            self.env.close()
            raise

    def policy_action(self):
        raise ValueError('This camera-aware candidate has no learned policy; use manual or reference control')


class WholeBodySession:
    def __init__(self, root):
        from weft.whole_body import WholeBodyEnv
        self.spec = json.loads((root / 'human.json').read_text())
        self.policy = json.loads((root / 'policy.json').read_text())
        settings = json.loads((root / 'settings.json').read_text())
        if self.policy.get('spec_sha256') != canonical_hash(self.spec):
            raise ValueError('Whole-body policy belongs to a different body model')
        partition = settings['partition']
        self.env = WholeBodyEnv(self.spec, start_time_s=partition['evaluation_start_times_s'][0],
                                horizon_steps=partition['horizon_steps'])
        self.scene = self.env.scene
        self.scene['joint_names'] = list(self.env.action_names)
        try:
            self.reset(0)
            self.policy_action()
        except Exception:
            self.env.close()
            raise

    def reset(self, seed):
        self.observation, _ = self.env.reset(seed=seed, options={'jitter': True})
        self.done, self.failure = False, None
        self._snapshot = self.env.snapshot()
        return self.snapshot()

    def policy_action(self):
        from weft.whole_body_learning import policy_action
        return policy_action(self.policy, self.env).tolist()

    def reference_action(self):
        return np.clip(self.env.feature_matrix().sum(axis=1), -1, 1).tolist()

    def validate_action(self, action):
        if (not isinstance(action, list) or len(action) != 22
                or any(type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1
                       for value in action)):
            raise ValueError('Whole-body action requires exactly 22 finite numbers within [-1, 1]')
        return action

    def step(self, action):
        self.validate_action(action)
        try:
            self.observation, reward, terminated, truncated, info = self.env.step(action)
            self._snapshot = self.env.snapshot()
        except RuntimeError as error:
            self.done = True
            self.failure = {'reason': str(error), 'attempted_action': list(action),
                            'scope': 'Last valid state shown; no terminal pose was fabricated.'}
            raise
        self.done = terminated or truncated
        return reward, terminated, truncated, info

    def snapshot(self):
        if self.failure:
            return {**self._snapshot, 'terminated': True, 'simulation_failure': self.failure}
        return self._snapshot


class SpatialSession:
    def __init__(self, root):
        from weft.spatial_rigid import SpatialRigidEnv
        self.scene = json.loads((root / "scene.json").read_text())
        self.policy = json.loads((root / "policy.json").read_text()) if (root / "policy.json").is_file() else None
        if self.policy is not None and self.policy.get("scene_sha256") != canonical_hash(self.scene):
            raise ValueError("Spatial policy belongs to a different 3D scene")
        self.env = SpatialRigidEnv(self.scene)
        self.scene = self.env.scene
        self.trained_controlled_id = (
            self.policy.get("controlled_id", self.scene["controlled_id"]) if self.policy is not None else None
        )
        try:
            self.reset(0)
            if self.policy is not None:
                self.policy_action()
        except Exception:
            self.env.close()
            raise

    def reset(self, seed):
        self.observation, _ = self.env.reset(seed=seed, options={"jitter": False})
        self.done, self.failure = False, None
        self._snapshot = self.env.snapshot()
        return self.snapshot()

    def select_actor(self, name):
        if self.failure is not None:
            raise ValueError("Reset the failed episode before changing the controlled object")
        self.observation = self.env.select_actor(name)
        self.scene = self.env.scene
        self.done = self.env._done
        self._snapshot = self.env.snapshot()

    def policy_action(self):
        if self.policy is None:
            raise ValueError("This 3D scene has no learned policy; use manual controls")
        from weft.spatial_learning import policy_action
        if self.env.scene["controlled_id"] != self.trained_controlled_id:
            raise ValueError(f"This policy was trained for {self.trained_controlled_id}; select that object first")
        return int(policy_action(self.policy, self.observation))

    def step(self, action):
        if type(action) is not int or not 0 <= action <= 7:
            raise ValueError("Spatial rigid control requires an integer action 0–7")
        try:
            self.observation, reward, terminated, truncated, info = self.env.step(action)
            self._snapshot = self.env.snapshot()
        except RuntimeError as error:
            self.done = True
            self.failure = {"reason": str(error), "attempted_action": action,
                            "scope": "Last valid state shown; no terminal state was fabricated."}
            raise
        self.done = terminated or truncated
        return reward, terminated, truncated, info

    def snapshot(self):
        if self.failure:
            return {**self._snapshot, "terminated": True, "simulation_failure": self.failure}
        return self._snapshot


class SharedWorldSession:
    def __init__(self, root):
        from weft.shared_world import SharedWorldEnv, validate_shared_world
        from weft.shared_world_evidence import verify_shared_world_sources
        from weft.shared_world_learning import validate_shared_policy

        self.scene = validate_shared_world(json.loads((root / "scene.json").read_text()))
        self.source_binding_verification = verify_shared_world_sources(self.scene, root)
        self.policy = json.loads((root / "policy.json").read_text()) if (root / "policy.json").is_file() else None
        if self.policy is not None:
            validate_shared_policy(self.policy, self.scene)
        self.env = SharedWorldEnv(self.scene)
        self.scene = self.env.scene
        try:
            self.reset(0)
            if self.policy is not None:
                self.policy_action()
        except Exception:
            self.env.close()
            raise

    def reset(self, seed):
        self.observation, _ = self.env.reset(seed=seed, options={"jitter": False})
        self.done, self.failure = False, None
        self._snapshot = self.env.snapshot()
        return self.snapshot()

    def select_actor(self, name):
        if self.failure is not None:
            raise ValueError("Reset the failed episode before changing the control target")
        self.observation = self.env.select_target(name)
        self.scene = self.env.scene
        self._snapshot = self.env.snapshot()

    def policy_action(self):
        if self.policy is None:
            raise ValueError("This shared-world run has no learned policy; use manual or reference controls")
        from weft.shared_world_learning import policy_action

        return policy_action(self.policy, self.env).tolist()

    def reference_action(self):
        return self.env.authored_action().tolist()

    def resolve_action(self, action):
        expected = len(self.env.action_names)
        if type(action) is int:
            if self.scene["controlled_id"] == "human":
                raise ValueError(f"Human control requires a {expected}-element normalized motor+object vector")
            if not 0 <= action <= 7:
                raise ValueError("Shared-world rigid object action must be an integer 0–7")
            return self.env.manual_action_vector(
                object_id=self.scene["controlled_id"],
                object_action_code=action,
            ).tolist()
        if (
            not isinstance(action, list)
            or len(action) != expected
            or any(type(v) not in (int, float) or not math.isfinite(v) or not -1.0 <= v <= 1.0 for v in action)
        ):
            raise ValueError(f"Shared-world action requires exactly {expected} finite numbers within [-1, 1]")
        return [float(v) for v in action]

    def step(self, action):
        vector = self.resolve_action(action)
        try:
            self.observation, reward, terminated, truncated, info = self.env.step(vector)
            self._snapshot = self.env.snapshot()
        except RuntimeError as error:
            self.done = True
            self.failure = {
                "reason": str(error),
                "attempted_action": vector,
                "scope": "Last valid state shown; no terminal state was fabricated.",
            }
            raise
        self.done = terminated or truncated
        return reward, terminated, truncated, info

    def snapshot(self):
        if self.failure:
            return {**self._snapshot, "terminated": True, "simulation_failure": self.failure}
        return self._snapshot


class Lab:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = json.loads((self.root / "lab-config.json").read_text())
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.session = None
        self.recorder = None
        self.export_future = None
        self.export_path = None
        self.coverage_future = None
        self.inspection_future = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weft-export")
        self.worker = None
        self.worker_log = None
        self.seed = 0
        self.artifacts = None
        self._evidence_provenance = None
        self.cancelled = False

    def start_worker(self):
        self.worker_log = (self.root / "worker.log").open("xb")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONUNBUFFERED": "1",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MPLCONFIGDIR": str(self.root / "runtime" / "matplotlib")}
        try:
            self.worker = subprocess.Popen([sys.executable, "-m", "weft.lab_worker", str(self.root)],
                                           env=env, stdout=self.worker_log, stderr=subprocess.STDOUT)
        except OSError:
            self.worker_log.close()
            raise
        threading.Thread(target=self._watch_worker, daemon=True, name="weft-pipeline-watch").start()

    def _watch_worker(self):
        try:
            code = self.worker.wait(timeout=360)
        except subprocess.TimeoutExpired:
            self.worker.terminate()
            try:
                self.worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait()
            code = -1
        with self.lock:
            if not self.cancelled and self.status()["status"] not in ("ready", "failed", "cancelled"):
                self.fail(f"Pipeline exited with code {code}; inspect worker.log. Hard wall-time limit: 360 seconds.")
        self.worker_log.close()

    def fail(self, message, state="failed"):
        old = self.status()
        error = {"message": message, "time": local_time()}
        write_json(self.root / "failure.json", error)
        atomic_json(self.root / "status.json", {"status": state, "events": old.get("events", []), "error": error})

    def status(self):
        path = self.root / "status.json"
        return json.loads(path.read_text()) if path.exists() else {"status": "pending", "events": []}

    def load_session(self):
        if self.session is None:
            if self.status()["status"] != "ready":
                raise ValueError("The environment is not ready; inspect the pipeline status")
            verify_artifacts(self.root / "run")
            cls = {"rolling": RollingSession, "planar": PlanarSession, "appearance": PlanarSession,
                   "spatial-rigid": SpatialSession, "shared-world": SharedWorldSession,
                   "exercise": HumanSession}[self.config["profile"]]
            if self.config.get('control_profile') == 'whole_body':
                if self.config['profile'] != 'exercise':
                    raise ValueError('Whole-body control requires an exercise lab')
                cls = WholeBodySession
            elif self.config.get('control_profile') == 'camera_body':
                if self.config['profile'] != 'exercise':
                    raise ValueError('Camera-body control requires an exercise lab')
                cls = CameraHumanSession
            self.session = cls(self.root / "run")
        return self.session

    def exercise_coverage(self):
        if self.config['profile'] != 'exercise':
            return {'status': 'not_applicable'}
        if self.status()['status'] not in TERMINAL:
            return {'status': 'pending'}
        if self.coverage_future is None:
            # Terminal evidence is immutable; parse it once off the request/control thread.
            self.coverage_future = self.executor.submit(load_coverage, self.root / 'run')
        return self.coverage_future.result() if self.coverage_future.done() else {'status': 'loading'}

    def inspection(self):
        status = self.status()['status']
        if status != 'ready':
            return {'status': 'unavailable' if status in TERMINAL else 'pending',
                    'reason': 'A completed run is required for the saved inspection summary.'}
        if self.inspection_future is None:
            # Immutable metadata is read once off the control thread; no environment is created.
            self.inspection_future = self.executor.submit(
                load_inspection, self.root / 'run', profile=self.config['profile'])
        return self.inspection_future.result() if self.inspection_future.done() else {'status': 'loading'}

    def capabilities(self, session):
        from weft.capabilities import describe_capabilities
        if self.artifacts is None and (self.root / "run").exists():
            self.artifacts = [{"name": str(p.relative_to(self.root / "run")), "bytes": p.stat().st_size}
                              for p in sorted((self.root / "run").iterdir()) if p.is_file()]
        artifacts = self.artifacts or []
        policy_ready = (session.scene.get("learned_policy_available") is not False
                        and not isinstance(session, CameraHumanSession)
                        and (not isinstance(session, (SpatialSession, SharedWorldSession)) or session.policy is not None))
        trained_target = None
        if policy_ready:
            if isinstance(session, RollingSession):
                trained_target = "ball-0"
            elif isinstance(session, (PlanarSession, SpatialSession)):
                trained_target = session.trained_controlled_id
            elif isinstance(session, SharedWorldSession):
                trained_target = "coupled_human_and_" + session.scene["task"]["object_goal"]["target_object_id"]
            elif isinstance(session, WholeBodySession):
                trained_target = "all_direct_motors"
            elif isinstance(session, HumanSession) and session.policy is not None:
                trained_target = session.policy.get("selected_joint")
        has_video = any(
            item["name"].startswith("source.")
            and Path(item["name"]).suffix.lower() in (".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv")
            for item in artifacts
        )
        has_reference = (
            any(item["name"] == "human-reference.json" for item in artifacts)
            or bool(getattr(session, "spec", {}).get("reference"))
            or bool(session.scene.get("human_spec", {}).get("reference"))
        )
        return describe_capabilities(
            session.scene,
            lab_profile=self.config.get("control_profile") or self.config["profile"],
            learned_policy_available=policy_ready,
            trained_target_id=trained_target,
            source_video_present=has_video,
            reference_available=has_reference,
        )

    def evidence_provenance(self, capabilities):
        if self._evidence_provenance is not None:
            return self._evidence_provenance
        run_dir = self.root / "run"

        def _bounded_json(name, limit=4 * 1024 * 1024):
            path = run_dir / name
            if not path.is_file() or path.stat().st_size > limit:
                return None
            try:
                value = json.loads(path.read_text())
                return value if isinstance(value, dict) else None
            except (OSError, ValueError):
                return None

        source_meta = _bounded_json("source.json")
        pose_quality = _bounded_json("pose-quality.json")
        obs_meta = _bounded_json("observations.json")
        coverage_summary = _bounded_json("coverage-summary.json")
        scene_doc = _bounded_json("scene.json")
        source_binding = scene_doc.get("source_binding") if isinstance(scene_doc, dict) else None

        has_video = bool(capabilities["evidence_separation"]["source_video_present"])
        has_obs = (run_dir / "observations.json").is_file()
        frame_count = None
        source_sha = None
        if source_meta is not None:
            source_sha = source_meta.get("sha256")
            for key in ("frames", "frame_count", "decoded_frames"):
                if type(source_meta.get(key)) is int:
                    frame_count = source_meta[key]
                    break
        if isinstance(source_binding, dict) and source_binding.get("mode") == "source_qualified":
            source_sha = source_sha or source_binding.get("source_sha256")
            if frame_count is None and type(source_binding.get("total_source_frames")) is int:
                frame_count = source_binding["total_source_frames"]
        missing_summary = None
        if coverage_summary is not None and type(coverage_summary.get("missing_observation_slots")) is int:
            missing_summary = (
                f"{coverage_summary['missing_observation_slots']}/{coverage_summary.get('observation_slots', '?')} "
                "unavailable object-frame slots preserved without interpolation"
            )
        elif isinstance(source_binding, dict) and source_binding.get("mode") == "source_qualified":
            bindings = source_binding.get("entity_bindings", [])
            tot_slots = sum(
                row.get("observed_slots", 0) + row.get("missing_slots", 0) + row.get("pre_init_slots", 0)
                for row in bindings if isinstance(row, dict)
            )
            obs_slots = sum(row.get("observed_slots", 0) for row in bindings if isinstance(row, dict))
            missing_summary = (
                f"{obs_slots}/{tot_slots} observed entity-frame slots across {len(bindings)} coupled entities "
                "(missing and authored-addition slots preserved explicitly)"
            )
        elif pose_quality is not None and type(pose_quality.get("unusable_frames")) is int:
            missing_summary = (
                f"{pose_quality.get('usable_frames', '?')} usable, "
                f"{pose_quality.get('partial_frames', 0)} partial, "
                f"{pose_quality['unusable_frames']} unusable sampled frames"
            )
        elif obs_meta is not None and isinstance(obs_meta.get("entities"), list):
            missing_slots = 0
            total_slots = 0
            for ent in obs_meta["entities"]:
                for sample in ent.get("samples", []) if isinstance(ent, dict) else []:
                    total_slots += 1
                    if isinstance(sample, dict) and sample.get("status") not in ("observed", "tracked", "measured"):
                        missing_slots += 1
            if total_slots:
                missing_summary = f"{missing_slots}/{total_slots} non-observed or uncertain entity-frame slots"

        ctrls = capabilities["controllers"]
        active_controllers = ["manual"]
        if ctrls["authored_reference_controller"]:
            active_controllers.append("authored_reference_controller")
        if ctrls["learned_policy_available"]:
            active_controllers.append("learned_policy")

        if has_video:
            recorded_desc = "Sealed source video and per-frame measurements preserved in run/"
        elif source_meta is not None or has_obs or (isinstance(source_binding, dict) and source_binding.get("mode") == "source_qualified"):
            recorded_desc = "Source-linked metadata/observations preserved; no video file sealed in run/"
        else:
            recorded_desc = "Analytic or imported scene without sealed source video"

        kinematic_desc = (
            "Source-derived joint poses projected onto simplified body geometry (no physics solver)"
            if ctrls["kinematic_pose_replay"]
            else "Not applicable (rigid-object simulation)"
        )
        self._evidence_provenance = {
            "schema_version": 1,
            "physical_validation": "unverified",
            "source_evidence": {
                "source_video_present": has_video,
                "source_sha256": source_sha,
                "frame_count": frame_count,
                "observations_artifact_present": has_obs,
                "missing_or_uncertain_summary": missing_summary,
            },
            "layers": {
                "recorded_source": recorded_desc,
                "kinematic_projection": kinematic_desc,
                "controllers": active_controllers,
                "physical_simulation": (
                    f"{capabilities['physics_backend']} forward dynamics under authored geometry, "
                    "contact, and actuator assumptions (physical_validation: unverified)"
                ),
                "muscle_display": capabilities["evidence_separation"]["muscle_display"],
            },
        }
        return self._evidence_provenance

    def summary(self):
        status = self.status()
        result = {**status, "profile": self.config["profile"], "directory": str(self.root)}
        if self.config.get('control_profile') in ('whole_body', 'camera_body') or self.config.get('profile') == 'shared-world':
            if 'reference_feasibility' in self.config:
                result['reference_feasibility'] = self.config.get('reference_feasibility')
        if self.config.get('control_profile') == 'camera_body' and 'ground_clearance' in self.config:
            result['ground_clearance'] = self.config['ground_clearance']
        scene_config = self.config.get("scene_config")
        tracking = scene_config.get("tracking") if isinstance(scene_config, dict) else None
        window = tracking.get("decode_interval") if isinstance(tracking, dict) else None
        if isinstance(window, list) and len(window) == 2 and all(type(value) is int for value in window):
            result["requested_source_interval"] = {"first_frame": window[0], "last_frame": window[1],
                "initialization_frame": tracking.get("initialization_frame", window[0]),
                "scope": "Requested original-source interval; consult decode audit for completion, not whole-video verification"}
        if (self.root / "run").exists():
            result["artifacts"] = self.artifacts if self.artifacts is not None else [
                {"name": p.name, "bytes": p.stat().st_size}
                for p in sorted((self.root / "run").iterdir()) if p.is_file()]
        if status["status"] == "ready":
            session = self.load_session()
            if self.artifacts is None:
                self.artifacts = [{"name": str(p.relative_to(self.root / "run")), "bytes": p.stat().st_size}
                                  for p in sorted((self.root / "run").iterdir()) if p.is_file()]
            capabilities = self.capabilities(session)
            policy_ready = capabilities["controllers"]["learned_policy_available"]
            result.update(scene=session.scene, snapshot=session.snapshot(), artifacts=self.artifacts,
                          learned_policy_available=policy_ready, capabilities=capabilities,
                          evidence_provenance=self.evidence_provenance(capabilities))
        result["recording"] = self.recorder is not None
        if self.export_future is not None:
            result["export"] = {"status": "running", "directory": self.export_path}
            if self.export_future.done():
                try:
                    result["export"].update(status="ready", result=self.export_future.result())
                except Exception as error:
                    result["export"].update(status="failed", error=str(error))
        sessions = self.root / "sessions"
        result["sessions"] = [{"name": p.name, "complete": (p / "manifest.json").exists()}
                              for p in sorted(sessions.iterdir())[-100:]] if sessions.exists() else []
        return result

    def control(self, path, data):
        session = self.load_session()
        if path == "/api/joint":
            if not isinstance(session, HumanSession):
                raise ValueError("Joint selection is available only for exercise runs")
            if self.recorder:
                raise ValueError("Stop and export the recording before changing its controlled joint")
            session.select_joint(data.get("joint"))
            return {"snapshot": session.snapshot(), "scene": session.scene, "capabilities": self.capabilities(session)}
        if path == "/api/actor":
            if not isinstance(session, (PlanarSession, SpatialSession, SharedWorldSession)):
                raise ValueError("Object selection is available only for rigid-object or shared-world runs")
            if self.recorder:
                raise ValueError("Stop and export the recording before changing its controlled object")
            session.select_actor(data.get("actor"))
            return {"snapshot": session.snapshot(), "scene": session.scene, "capabilities": self.capabilities(session)}
        if path in ("/api/reset", "/api/step") and self.recorder:
            try:
                self.recorder.check_capacity()
            except (ValueError, RuntimeError) as error:
                self.recorder.close()
                self.recorder = None
                raise ValueError(f"Recording stopped before the next action: {error}. The journal is preserved.") from error
        if path == "/api/reset":
            seed = data.get("seed", self.seed + 1)
            if type(seed) is not int or not 0 <= seed < 2 ** 31:
                raise ValueError("Seed must be an integer between 0 and 2147483647")
            self.seed = seed
            snapshot = session.reset(seed)
            if self.recorder:
                self.recorder.append({"type": "reset", "seed": seed, "snapshot": snapshot})
            return {"snapshot": snapshot}
        if path == "/api/step":
            count, action = data.get("steps", 1), data.get("action", 0)
            scene_profile = session.scene.get("profile")
            maximum_action = 9 if scene_profile == "exercise-v3" else 7 if scene_profile == "spatial-rigid-v1" else 5
            if type(count) is not int or not 1 <= count <= 6:
                raise ValueError('Use 1–6 steps')
            if type(data.get("policy", False)) is not bool:
                raise ValueError("policy must be a boolean")
            if isinstance(session, WholeBodySession):
                if data.get('controller') not in (None, 'reference'):
                    raise ValueError('Unknown whole-body controller')
                selectors = int('action' in data) + int(data.get('policy', False)) + int(data.get('controller') is not None)
                if selectors != 1:
                    raise ValueError('Choose one whole-body action vector, learned policy or reference controller')
                if 'action' in data:
                    session.validate_action(action)
            elif isinstance(session, SharedWorldSession):
                if data.get('controller') not in (None, 'reference'):
                    raise ValueError('Unknown shared-world controller')
                selectors = int('action' in data) + int(data.get('policy', False)) + int(data.get('controller') is not None)
                if selectors != 1:
                    raise ValueError('Choose one shared-world action, learned policy or reference controller')
                if 'action' in data:
                    action = session.resolve_action(action)
            elif ('controller' in data or type(action) is not int or not 0 <= action <= maximum_action):
                raise ValueError(f'Use action 0–{maximum_action}; whole-body controllers require a whole-body run')
            transitions = []
            for _ in range(count):
                if session.done:
                    break
                if self.recorder:
                    try:
                        self.recorder.check_capacity()
                    except (ValueError, RuntimeError) as error:
                        self.recorder.close()
                        self.recorder = None
                        return {"snapshot": session.snapshot(), "transitions": transitions, "recording": False,
                                "warning": f"Recording stopped at its bound: {error}. Prior actions are saved."}
                chosen = (session.policy_action() if data.get('policy', False) else
                          session.reference_action() if data.get('controller') == 'reference' else
                          session.resolve_action(action) if isinstance(session, SharedWorldSession) else action)
                try:
                    reward, terminated, truncated, info = session.step(chosen)
                except RuntimeError:
                    if not isinstance(session, (WholeBodySession, SpatialSession, SharedWorldSession)) or session.failure is None:
                        raise
                    if self.recorder:
                        recorder, self.recorder = self.recorder, None
                        recorder.close()
                        metadata = json.loads((recorder.root / 'recording.json').read_text())
                        atomic_json(recorder.root / 'recording.json', metadata | {'failed_attempt': session.failure})
                        self.export_path = recorder.root.name
                        self.export_future = self.executor.submit(export_recording, recorder.root)
                    return {'snapshot': session.snapshot(), 'transitions': transitions, 'recording': False,
                            'warning': 'Simulation stopped after a failed action. Last valid state and failed attempt preserved.'}
                event = {"type": "step", "action": chosen, "reward": float(reward),
                         "terminated": bool(terminated), "truncated": bool(truncated), "snapshot": session.snapshot()}
                if self.recorder:
                    self.recorder.append(event)
                transitions.append(event)
            return {"snapshot": session.snapshot(), "transitions": transitions}
        if path == "/api/record/start":
            if self.recorder:
                raise ValueError("A recording is already active")
            if self.export_future is not None and not self.export_future.done():
                raise ValueError("Wait for the current export to finish")
            # Start at a seeded reset so every saved action stream can be reproduced.
            snapshot = session.reset(self.seed)
            name = f"session-{uuid4().hex[:12]}"
            recorder_profile = (
                "shared-world"
                if session.scene.get("profile") == "shared-world-v1"
                else "spatial-rigid"
                if session.scene.get("profile") == "spatial-rigid-v1"
                else self.config["profile"]
            )
            recorder = Recorder(self.root / "sessions" / name, session.scene, profile=recorder_profile)
            recorder.start(snapshot, seed=self.seed)
            self.recorder = recorder
            return {"recording": True, "directory": name, "snapshot": snapshot}
        if path == "/api/record/stop":
            if not self.recorder:
                raise ValueError("No recording is active")
            recorder, self.recorder = self.recorder, None
            result = recorder.close()
            root = Path(result["directory"]) if "directory" in result else recorder.root
            self.export_path = Path(root).name
            self.export_future = self.executor.submit(export_recording, root)
            return {"recording": False, "export": "running", "directory": self.export_path}
        raise ValueError("Unknown control endpoint")

    def close(self):
        if self.worker and self.worker.poll() is None:
            terminal = self.status()["status"] in ("ready", "failed", "cancelled")
            with self.lock:
                self.cancelled = True
            if not terminal:
                self.worker.terminate()
            try:
                self.worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait()
            if not terminal:
                with self.lock:
                    self.fail("Cancelled by the user; partial evidence is preserved", "cancelled")
        if self.recorder:
            self.recorder.close()
            self.recorder = None
        self.executor.shutdown(wait=True)
        if self.session:
            self.session.env.close()


class LabServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, lab):
        self.lab = lab
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(address, LabHandler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class LabHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Query tokens must not enter request logs.

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def response_headers(self, code, mime, size):
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; script-src 'self'; style-src 'self'; frame-ancestors 'none'; connect-src 'self'")

    def reply(self, value, code=200):
        body = json.dumps(value, allow_nan=False).encode()
        self.response_headers(code, "application/json; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self, public=False):
        host = f"127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host") != host:
            self.reply({"error": "Use the printed 127.0.0.1 address"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + host:
            self.reply({"error": "Cross-origin access is disabled"}, 403)
            return False
        if not public:
            token = self.headers.get("Authorization", "").removeprefix("Bearer ")
            if not token:
                token = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
            if not secrets.compare_digest(token.encode("utf-8"), self.server.lab.token.encode("utf-8")):
                self.reply({"error": "Open the complete local lab link, including its access token"}, 401)
                return False
        return True

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        assets = Path(__file__).parent / "assets"
        public = {"/": assets / "lab.html", "/lab.js": assets / "lab.js", "/lab.css": assets / "lab.css",
                  "/brand.css": assets / "brand.css", "/weft-logo.png": assets / "weft-logo.png"}
        if not self.authorized(public=path in public):
            return
        try:
            if path == "/api/status":
                with self.server.lab.lock:
                    return self.reply(self.server.lab.summary())
            if path == '/api/exercise-coverage':
                with self.server.lab.lock:
                    return self.reply(self.server.lab.exercise_coverage())
            if path == '/api/inspection':
                with self.server.lab.lock:
                    return self.reply(self.server.lab.inspection())
            if path in public:
                return self.send_file(public[path])
            if path.startswith("/files/"):
                relative = path.removeprefix("/files/")
                file = (self.server.lab.root / relative).resolve()
                if not file.is_relative_to(self.server.lab.root) or not file.is_file():
                    raise ValueError("File is outside this lab or does not exist")
                resolved_relative = file.relative_to(self.server.lab.root).as_posix()
                if not resolved_relative.startswith(("run/", "sessions/")) and resolved_relative not in ("worker.log", "events.jsonl", "failure.json"):
                    raise ValueError("File is not an inspectable artifact")
                return self.send_file(file)
            self.reply({"error": "Not found"}, 404)
        except (ValueError, OSError, RuntimeError) as error:
            self.reply({"error": str(error)}, 400)

    def send_file(self, path):
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        requested = self.headers.get("Range")
        if requested:
            import re
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
            if not match:
                return self.reply({"error": "Unsupported byte range"}, 416)
            start = int(match[1])
            end = min(int(match[2]), size - 1) if match[2] else size - 1
            if not 0 <= start <= end < size:
                return self.reply({"error": "Range outside file"}, 416)
            code = 206
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.response_headers(code, mime, end - start + 1)
        self.send_header("Accept-Ranges", "bytes")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        # Saved HTML is downloaded; its offline scripts never inherit the live lab origin.
        if path.suffix == ".html" and path != Path(__file__).parent / "assets" / "lab.html":
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = stream.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self):
        if not self.authorized():
            return
        try:
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("Use application/json")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Request body must be 1–4096 bytes")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Request must be a JSON object")
            with self.server.lab.lock:
                result = self.server.lab.control(urlsplit(self.path).path, data)
            self.reply(result)
        except (ValueError, OSError, RuntimeError, TypeError) as error:
            self.reply({"error": str(error)}, 400)


def prepare_lab(video, out, *, profile="planar", config=None, episodes=1500, eval_episodes=100,
                seed=1729, source_note="Rights not independently verified; user-provided local video", pose_model=None,
                pose_mode="video", body_model="hinge", pose_identity="raw"):
    if profile not in ("planar", "appearance", "rolling", "exercise"):
        raise ValueError("Unknown profile")
    if pose_mode not in ("video", "image") or (profile != "exercise" and pose_mode != "video"):
        raise ValueError("Pose mode must be video or image and applies only to exercise")
    if body_model not in ("hinge", "spatial", "mobile") or (profile != "exercise" and body_model != "hinge"):
        raise ValueError("Body model must be hinge, spatial or mobile and applies only to exercise")
    if pose_identity not in ("raw", "temporal") or (profile != "exercise" and pose_identity != "raw"):
        raise ValueError("Pose identity must be raw or temporal and applies only to exercise")
    if type(episodes) is not int or not 1 <= episodes <= (50000 if profile == "rolling" else 5000):
        raise ValueError("Training episodes exceed the selected profile's bounds")
    if type(eval_episodes) is not int or not 1 <= eval_episodes <= 200:
        raise ValueError("Evaluation episodes must be 1–200")
    if type(seed) is not int or not 0 <= seed < 2 ** 31 or seed == 938401:
        raise ValueError("Invalid training seed or collision with evaluation seed")
    video, out = Path(video).resolve(), Path(out).resolve()
    if not video.is_file() or video.stat().st_size > 100_000_000:
        raise ValueError("Provide a local video no larger than 100 MB")
    if video.suffix.lower() not in (".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv"):
        raise ValueError("Supported video containers: mp4, webm, mov, m4v, avi, mkv")
    if profile == "exercise":
        pose_model = Path(pose_model or ".local/models/pose_landmarker_lite.task").resolve()
        if not pose_model.is_file():
            raise ValueError("Pose model missing. Run: .venv-lab/bin/python -m weft fetch-pose-model")
    scene_config = None
    if profile == "appearance" and not config:
        raise ValueError("Appearance tracking requires --config with source-pinned regions and explicit world assumptions")
    if config:
        if profile not in ("planar", "appearance"):
            raise ValueError("--config applies only to planar and appearance profiles")
        if Path(config).stat().st_size > 64_000:
            raise ValueError("Scene config exceeds 64 KB")
        scene_config = json.loads(Path(config).read_text())
        if not isinstance(scene_config, dict):
            raise ValueError("Scene config must be a JSON object")
    if out.exists():
        raise ValueError("Output already exists; use a new directory or --resume")
    out.mkdir(parents=True)
    write_json(out / "lab-config.json", {"schema_version": 1, "video": str(video), "profile": profile,
               "episodes": episodes, "eval_episodes": eval_episodes, "seed": seed,
               "scene_config": scene_config, "source_note": source_note, "created_at": local_time(),
               "pose_model": str(pose_model) if pose_model else None,
               "pose_mode": pose_mode,
               "body_model": body_model,
               "pose_identity": pose_identity,
               "bounds": {"wall_seconds": 360, "external_spend_usd": 0, "recurring_services": 0}})
    return out


def launch_lab(args):
    scene_run = getattr(args, "scene_run", None)
    shared_world_run = getattr(args, "shared_world_run", None)
    scene_learning = getattr(args, "learning", None)
    if scene_run and shared_world_run:
        raise ValueError("Choose either --scene-run or --shared-world-run, not both")
    if args.resume:
        if args.video or args.out or args.config or scene_run or shared_world_run or scene_learning:
            raise ValueError("--resume cannot be combined with --video, --out, --config, --scene-run, or --shared-world-run")
        root = Path(args.resume).resolve()
        if not (root / "lab-config.json").is_file():
            raise ValueError("--resume requires an existing lab directory")
    elif scene_run:
        if args.video or args.config or not args.out:
            raise ValueError("--scene-run requires a new --out directory and cannot be combined with --video or --config")
        from weft.spatial_lab import prepare_spatial_lab
        prepare_spatial_lab(scene_run, args.out, policy_run=scene_learning)
        root = Path(args.out).resolve()
    elif shared_world_run:
        if args.video or args.config or not args.out:
            raise ValueError("--shared-world-run requires a new --out directory and cannot be combined with --video or --config")
        from weft.shared_world_lab import prepare_shared_world_lab
        prepare_shared_world_lab(shared_world_run, args.out, policy_run=scene_learning)
        root = Path(args.out).resolve()
    else:
        if scene_learning:
            raise ValueError("--learning requires --scene-run or --shared-world-run")
        if not args.video or not args.out:
            raise ValueError("Provide --video and a new --out directory")
        root = prepare_lab(args.video, args.out, profile=args.profile, config=args.config,
                           episodes=args.episodes, eval_episodes=args.eval_episodes, seed=args.seed,
                           source_note=args.source_note, pose_model=args.pose_model, pose_mode=args.pose_mode,
                           body_model=args.body_model, pose_identity=args.pose_identity)
    if not 0 <= args.port <= 65535:
        raise ValueError("Port must be 0–65535 (0 chooses a free local port)")
    lab = Lab(root)
    server = None
    try:
        if not args.no_serve:
            server = LabServer(("127.0.0.1", args.port), lab)
            print(f"\nWeft local lab: http://127.0.0.1:{server.server_port}/#token={lab.token}", flush=True)
            print(f"Artifacts: {root}\nOpen the link in your browser. Ctrl+C stops the lab; recordings stay on disk.\n", flush=True)
        if not args.resume and not scene_run and not shared_world_run:
            lab.start_worker()
        if args.no_serve:
            last = -1
            while True:
                status = lab.status()
                for event in status.get("events", []):
                    if event["index"] > last:
                        print(f"[{event['stage']}] {event.get('message', '')}", flush=True)
                        last = event["index"]
                if status["status"] in ("ready", "failed", "cancelled"):
                    if status["status"] != "ready":
                        raise RuntimeError(status.get("error", {}).get("message", "Pipeline failed"))
                    return {"directory": str(root), "status": "ready",
                            "inspect": f"{sys.executable} -m weft lab --resume {root}"}
                if args.resume:
                    raise ValueError("This interrupted run is not ready; inspect failure.json and use a new output")
                time.sleep(0.2)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping the local lab. Saved artifacts and recording journals are preserved.", flush=True)
    finally:
        if server:
            server.server_close()
        lab.close()
    return {"directory": str(root), "status": "closed"}
