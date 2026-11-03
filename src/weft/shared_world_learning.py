"""Bounded CEM policy search for coupled human-and-object shared-world-v1 tasks."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.shared_world import (
    CAMERA_PROFILE,
    HUMAN_ONLY_PROFILE,
    PROFILE as SCENE_PROFILE,
    SharedWorldEnv,
    audit_shared_world_feasibility,
    validate_shared_world,
)
from weft.shared_world_packaging import collect_shared_source_files, read_package_file, write_package_files
from weft.whole_body import SUCCESS_LIMITS


POLICY_PROFILE = "shared-world-policy-v1"
ALGORITHM = "shared_world_cem_v1"
WEIGHT_DIM = 16
WEIGHT_BOUND = 3.0
SCOPE = (
    "Bounded CEM policy search over one coupled MuJoCo world (22 explicit human motors + 3D virtual object forces) "
    "starting from uninformed zero weights; not recovered human intent, measured muscle activation, or real-world transfer."
)


def _action_from_weights(weights, env):
    w = np.asarray(weights, dtype=float)
    if w.shape != (WEIGHT_DIM,):
        raise ValueError(f"Shared-world policy requires {WEIGHT_DIM} weights")
    act = np.zeros(len(env.action_names), dtype=float)
    human_w = w[:12].reshape(4, 3)
    act[:22] = np.clip(np.sum(env.feature_matrix() * human_w[env.group_indices], axis=1), -1.0, 1.0)

    obj_w = w[12:16]
    if np.all(obj_w == 0.0):
        return act

    obs = env.observation()
    phase = float(obs[154])
    w_pos, w_vel, w_brake, w_gate = (float(v) for v in obj_w)
    if w_gate > 0.0 and phase < min(0.65, 0.20 * w_gate):
        return act

    goal = env.scene["task"]["object_goal"]
    target_id = goal["target_object_id"]
    dyn_idx = env._dynamic_ids.index(target_id)
    all_idx = next(i for i, o in enumerate(env.scene["objects"]) if o["id"] == target_id)
    base_offset = 155 + 13 * all_idx
    vel = np.asarray(obs[base_offset + 7 : base_offset + 10], dtype=float)
    delta = np.asarray(obs[-3:], dtype=float)
    dist = float(np.linalg.norm(delta))
    speed = float(np.linalg.norm(vel))
    pos_tol = float(goal["position_tolerance"])
    speed_tol = float(goal["speed_tolerance"])
    accel = float(goal["virtual_acceleration"])

    brake_gain = max(0.0, w_brake)
    if dist <= 0.85 * pos_tol and (brake_gain > 0.0 or w_vel > 0.0):
        if speed > 0.45 * speed_tol:
            desired_acc = -(3.0 + 4.5 * brake_gain + 2.0 * max(0.0, w_vel)) * vel
        else:
            desired_acc = np.zeros(3, dtype=float)
    else:
        kp = max(0.0, w_pos) * 4.5
        kd = max(0.0, w_vel) * 2.5
        desired_acc = kp * delta - kd * vel
        if brake_gain > 0.0 and speed > 1e-9:
            approach_limit = 0.6 * speed_tol + (0.45 + 0.55 * brake_gain) * math.sqrt(max(0.0, dist - 0.5 * pos_tol))
            if speed > approach_limit:
                desired_acc -= (2.5 + 3.0 * brake_gain) * ((speed - approach_limit) / speed) * vel

    norm_force = desired_acc / max(1e-9, accel)
    mag = float(np.linalg.norm(norm_force))
    if mag > 1.0:
        norm_force = norm_force / mag
    act[22 + 3 * dyn_idx : 22 + 3 * (dyn_idx + 1)] = np.clip(norm_force, -1.0, 1.0)
    return act


def validate_shared_policy(policy, scene=None):
    if isinstance(scene, dict) and scene.get("task", {}).get("native_interval", {}).get("profile") == "native-dynamic-terminal-control-interval-v1":
        raise ValueError("Dynamic native interval learning is not qualified")
    if isinstance(scene, dict) and scene.get("profile") == CAMERA_PROFILE:
        raise ValueError("Camera-coupled worlds require separately qualified policies; existing coupled policies are unsupported")
    if not isinstance(policy, dict):
        raise ValueError("Shared-world policy must be a JSON object")
    if policy.get("schema_version") != 1 or policy.get("profile") != POLICY_PROFILE or policy.get("algorithm") != ALGORITHM:
        raise ValueError("Unsupported shared-world policy contract")
    weights = np.asarray(policy.get("weights"), dtype=float)
    if weights.shape != (WEIGHT_DIM,) or not np.isfinite(weights).all() or np.any(np.abs(weights) > WEIGHT_BOUND):
        raise ValueError(f"Shared-world policy requires {WEIGHT_DIM} finite weights within [-{WEIGHT_BOUND}, {WEIGHT_BOUND}]")
    if scene is not None:
        validated = validate_shared_world(scene)
        if validated["profile"] == HUMAN_ONLY_PROFILE:
            raise ValueError("Person-only scenes require the shared-exercise residual policy, not a coupled object policy")
        if policy.get("scene_sha256") != canonical_hash(validated):
            raise ValueError("Shared-world policy scene_sha256 does not match this scene")
        if policy.get("action_names") != validated["action_contract"]["names"]:
            raise ValueError("Shared-world policy action_names differ from scene action_contract")
    return deepcopy(policy)


def policy_action(policy, env):
    # The environment already validated its model; never compile MuJoCo again per action.
    if env.scene.get("task", {}).get("native_interval", {}).get("profile") == "native-dynamic-terminal-control-interval-v1":
        raise ValueError("Dynamic native interval learning is not qualified")
    if env.scene.get("profile") == CAMERA_PROFILE:
        raise ValueError("Camera-coupled worlds require separately qualified policies; existing coupled policies are unsupported")
    if env.scene.get("profile") == HUMAN_ONLY_PROFILE:
        raise ValueError("Person-only scenes require the shared-exercise residual policy, not a coupled object policy")
    validate_shared_policy(policy)
    if policy.get("scene_sha256") != canonical_hash(env.scene):
        raise ValueError("Shared-world policy scene_sha256 does not match this scene")
    if policy.get("action_names") != list(env.action_names):
        raise ValueError("Shared-world policy action_names differ from scene action_contract")
    return _action_from_weights(policy["weights"], env)


def _rollout(env, controller, seed, *, weights=None, reset_options=None, deadline=None, trace=False):
    opts = {"jitter": False} if reset_options is None else reset_options
    env.reset(seed=seed, options=opts)
    rng = np.random.default_rng(seed + 19)
    snapshots = [env.snapshot()] if trace else None
    act_dim = len(env.action_names)

    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Shared-world local training/evaluation deadline reached")
        if controller == "learned":
            action = _action_from_weights(weights, env)
        elif controller == "zero":
            action = np.zeros(act_dim, dtype=float)
        elif controller == "random":
            action = rng.uniform(-1.0, 1.0, act_dim)
        elif controller == "authored_feedback":
            action = env.authored_action()
        else:
            raise ValueError(f"Unknown shared-world controller: {controller!r}")

        try:
            _, reward, terminated, truncated, _ = env.step(action)
        except RuntimeError:
            if env.metrics().get("failure") is None:
                raise
            break
        if trace:
            snapshots.append(env.snapshot() | {"action": action.tolist(), "reward": float(reward)})
        if terminated or truncated:
            break
    return env.metrics(), snapshots


def _summarize_trials(trials):
    valid_ang = [row["angular_rmse_rad"] for row in trials if row["angular_rmse_rad"] is not None]
    valid_root = [row["root_rmse_model_units"] for row in trials if row["root_rmse_model_units"] is not None]
    return {
        "episodes": len(trials),
        "successes": sum(int(row["success"]) for row in trials),
        "success_rate": sum(int(row["success"]) for row in trials) / len(trials),
        "human_tracking_pass_rate": sum(int(row["human_tracking_passed"]) for row in trials) / len(trials),
        "object_parked_rate": sum(int(row["object_parked"]) for row in trials) / len(trials),
        "human_object_contact_rate": sum(int(row["human_object_contacts_total"] > 0) for row in trials) / len(trials),
        "mean_return": float(np.mean([row["return"] for row in trials])),
        "mean_angular_rmse_rad": float(np.mean(valid_ang)) if valid_ang else None,
        "mean_root_rmse_model_units": float(np.mean(valid_root)) if valid_root else None,
        "max_floor_penetration_model_units": float(max(row["max_floor_penetration_model_units"] for row in trials)),
        "mean_final_object_distance_world": float(np.mean([row["object_distance_world"] for row in trials])),
        "mean_final_object_speed_world_s": float(np.mean([row["object_speed_world_s"] for row in trials])),
        "trials": trials,
    }


def train_shared_world_policy(
    scene_or_dir,
    out,
    *,
    seed=1729,
    generations=8,
    population=16,
    eval_episodes=8,
    jitter_scale=1.0,
    position_jitter=0.02,
    velocity_jitter=0.08,
    max_seconds=120,
):
    """Train a bounded CEM policy on a shared-world-v1 scene and evaluate against matched baselines."""
    if type(seed) is not int or not 0 <= seed < 2**31 or seed == 938401:
        raise ValueError("Invalid training seed or collision with evaluation seed 938401")
    if type(generations) is not int or not 1 <= generations <= 16:
        raise ValueError("generations must be 1–16")
    if type(population) is not int or not 4 <= population <= 32:
        raise ValueError("population must be 4–32")
    if type(eval_episodes) is not int or not 2 <= eval_episodes <= 32:
        raise ValueError("eval_episodes must be 2–32")
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 180:
        raise ValueError("max_seconds must be 1–180 seconds")

    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("Choose a new output directory for shared-world training")

    source_dir = None
    source_verified = None
    candidate_dir = None
    if isinstance(scene_or_dir, (str, Path)):
        source_dir = Path(scene_or_dir).resolve()
        if out == source_dir or out.is_relative_to(source_dir):
            raise ValueError("Keep shared-world training output outside the immutable source directory")
        candidate_dir = source_dir / "run" if (source_dir / "run" / "scene.json").is_file() else source_dir
        source_verified = verify_artifacts(candidate_dir)
        scene_raw = read_package_file(candidate_dir, "scene.json", max_bytes=64 * 1024 * 1024)
        scene = json.loads(scene_raw)
    else:
        scene = deepcopy(scene_or_dir)

    scene = validate_shared_world(scene)
    if scene["task"].get("native_interval", {}).get("profile") == "native-dynamic-terminal-control-interval-v1":
        raise ValueError("Dynamic native interval learning is not qualified")
    if "native_clock" in scene.get("human_spec", {}):
        raise ValueError("Native-clock learning requires the source-bound shared-exercise pipeline; this legacy learner does not preserve clock custody")
    if scene["profile"] != SCENE_PROFILE:
        raise ValueError("train_shared_world_policy requires a shared-world-v1 scene")
    if scene["source_binding"]["mode"] == "source_qualified" and candidate_dir is None:
        raise ValueError("Source-qualified training requires a sealed source directory with observation evidence")
    from weft.shared_world_evidence import verify_shared_world_sources

    source_files = collect_shared_source_files(
        scene, candidate_dir,
        expected_manifest_sha256=source_verified["manifest_sha256"] if source_verified else None)
    if candidate_dir is not None and source_files.manifest_files.get("scene.json") != hashlib.sha256(scene_raw).hexdigest():
        raise ValueError("Training scene must match its sealed candidate identity")
    generated_paths = {"manifest.json", "implementation", "reference-feasibility.json", "settings.json",
                       "scene.json", "policy.json", "training.json", "rl-evaluation.json", "policy-rollout.json"}
    source_files.validate_layout(reserved_paths=generated_paths)
    evidence_verified = source_files.report

    feasibility = audit_shared_world_feasibility(scene)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + float(max_seconds)

    modules = (
        "shared_world.py", "shared_world_observation.py", "shared_world_learning.py", "shared_world_evidence.py", "shared_world_packaging.py",
        "source_equivalence.py", "world_evidence.py",
        "observations.py", "human.py", "mobile_human.py", "whole_body.py",
        "body_frame.py", "spatial_pose.py", "root_motion.py", "errors.py", "spatial_rigid.py",
    )
    implementation = {name: (Path(__file__).parent / name).read_bytes() for name in modules}
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in implementation.items()}
    write_package_files(out, {f"implementation/{name}": raw for name, raw in implementation.items()})
    source_files.copy_to(out, reserved_paths=generated_paths)
    del source_files

    rng = np.random.default_rng(seed)
    train_seeds = [int(x) for x in rng.integers(0, 2**31, 3)]
    eval_rng = np.random.default_rng(938401)
    eval_seeds = [int(x) for x in eval_rng.integers(0, 2**31, eval_episodes)]

    jitter_opts = {
        "jitter": True,
        "jitter_scale": float(jitter_scale),
        "position_jitter": float(position_jitter),
        "velocity_jitter": float(velocity_jitter),
    }
    scene_for_storage = deepcopy(scene)
    scene_for_storage["learned_policy_available"] = True
    scene_for_storage = validate_shared_world(scene_for_storage)
    scene_sha = canonical_hash(scene_for_storage)

    settings = {
        "schema_version": 1,
        "created_at": local_time(),
        "algorithm": ALGORITHM,
        "seed": seed,
        "evaluation_seed": 938401,
        "generations": generations,
        "population": population,
        "elite_count": max(2, population // 4),
        "initial_mean": 0.0,
        "initial_std": 0.9,
        "minimum_std": 0.08,
        "weight_bounds": [-WEIGHT_BOUND, WEIGHT_BOUND],
        "training_seeds": train_seeds,
        "evaluation_seeds": eval_seeds,
        "reset_perturbation": jitter_opts,
        "scene_sha256": scene_sha,
        "source_directory": str(source_dir) if source_dir is not None else None,
        "source_verification": source_verified,
        "evidence_verification": evidence_verified,
        "module_hashes": hashes,
        "human_limits": deepcopy(SUCCESS_LIMITS),
        "human_limits_modified": False,
        "external_api_usd": 0.0,
        "physical_validation": "unverified",
        "reference_feasibility": {k: feasibility[k] for k in ("status", "summary")},
        "scope": SCOPE,
    }
    write_json(out / "reference-feasibility.json", feasibility)
    write_json(out / "settings.json", settings)
    write_json(out / "scene.json", scene_for_storage)

    env = SharedWorldEnv(scene_for_storage)
    mean = np.zeros(WEIGHT_DIM, dtype=float)
    std = np.full(WEIGHT_DIM, 0.9, dtype=float)
    best = np.zeros(WEIGHT_DIM, dtype=float)
    best_score = -float("inf")
    history = []
    training = {
        "settings": settings,
        "history": history,
        "completed_generations": 0,
        "budget_exhausted": False,
        "seconds": 0.0,
    }

    try:
        for gen in range(generations):
            samples = np.clip(rng.normal(mean, std, (population, WEIGHT_DIM)), -WEIGHT_BOUND, WEIGHT_BOUND)
            samples[0] = best
            rows = []
            for cand_idx, cand_w in enumerate(samples):
                trials = []
                # Train on 1 nominal + 2 perturbed trials so policy handles both nominal and jittered starts.
                for t_idx, t_seed in enumerate(train_seeds):
                    opts = {"jitter": False} if t_idx == 0 else jitter_opts
                    metrics, _ = _rollout(
                        env,
                        "learned",
                        t_seed,
                        weights=cand_w,
                        reset_options=opts,
                        deadline=deadline,
                    )
                    trials.append(metrics)
                score = float(np.mean([t["return"] + 0.25 * float(t["success"]) for t in trials]))
                rows.append({"candidate": cand_idx, "weights": cand_w.tolist(), "score": score})
                if score > best_score:
                    best_score = score
                    best = cand_w.copy()

            elite_idx = np.argsort([-r["score"] for r in rows], kind="stable")[: settings["elite_count"]]
            mean = np.mean(samples[elite_idx], axis=0)
            std = np.maximum(np.std(samples[elite_idx], axis=0), settings["minimum_std"])
            history.append(
                {
                    "generation": gen,
                    "best_training_return": float(best_score),
                    "mean_candidate_score": float(np.mean([r["score"] for r in rows])),
                    "next_mean": mean.tolist(),
                    "next_std": std.tolist(),
                }
            )
            policy = {
                "schema_version": 1,
                "profile": POLICY_PROFILE,
                "algorithm": ALGORITHM,
                "scene_sha256": scene_sha,
                "action_names": list(env.action_names),
                "target_object_id": scene_for_storage["task"]["object_goal"]["target_object_id"],
                "weights": best.tolist(),
                "weight_names": [
                    *(f"{g}_{f}" for g in ("torso", "proximal", "distal", "root") for f in ("pos", "vel", "bias")),
                    "object_pos_gain",
                    "object_vel_damp",
                    "object_terminal_brake",
                    "object_phase_gate",
                ],
                "seed": seed,
                "training_score": float(best_score),
                "human_limits": deepcopy(SUCCESS_LIMITS),
                "physical_validation": "unverified",
                "scope": SCOPE,
            }
            validate_shared_policy(policy, scene_for_storage)
            training.update(completed_generations=gen + 1, seconds=time.perf_counter() - started)
            write_json(out / "policy.json", policy)
            write_json(out / "training.json", training)

        frozen_policy_sha = digest(out / "policy.json")
        controllers = ("learned", "zero", "random", "authored_feedback")
        nominal_results = {}
        changed_results = {}
        traces = {}
        for ctrl in controllers:
            nom_metrics, trace = _rollout(
                env,
                ctrl,
                eval_seeds[0],
                weights=best,
                reset_options={"jitter": False},
                deadline=deadline,
                trace=True,
            )
            traces[ctrl] = trace
            nominal_results[ctrl] = _summarize_trials([{"seed": eval_seeds[0], "jitter": False, **nom_metrics}])

            pert_trials = []
            for e_seed in eval_seeds:
                p_metrics, _ = _rollout(
                    env,
                    ctrl,
                    e_seed,
                    weights=best,
                    reset_options=jitter_opts,
                    deadline=deadline,
                )
                pert_trials.append({"seed": e_seed, "jitter": True, **p_metrics})
            changed_results[ctrl] = _summarize_trials(pert_trials)

        if digest(out / "policy.json") != frozen_policy_sha:
            raise RuntimeError("Shared-world policy changed during evaluation")

        evaluation = {
            "schema_version": 1,
            "profile": "shared-world-rl-evaluation-v1",
            "policy_sha256": frozen_policy_sha,
            "scene_sha256": scene_sha,
            "policies": changed_results,
            "regimes": {
                "nominal_exact": nominal_results,
                "changed_initial_conditions": changed_results,
            },
            "reset_perturbation": jitter_opts,
            "human_limits": deepcopy(SUCCESS_LIMITS),
            "human_limits_modified": False,
            "object_goal": deepcopy(scene_for_storage["task"]["object_goal"]),
            "reference_feasibility": settings["reference_feasibility"],
            "seconds": time.perf_counter() - started,
            "physical_validation": "unverified",
            "comparison": (
                "All four controllers issue one explicit unified (22 human motors + 3D object force) vector "
                "per 30-Hz step on identical seeded resets in one coupled MuJoCo model."
            ),
            "scope": SCOPE,
        }
        write_json(out / "rl-evaluation.json", evaluation)
        write_json(out / "policy-rollout.json", traces)
        verify_shared_world_sources(scene_for_storage, out)
        write_json(
            out / "manifest.json",
            {
                "schema_version": 1,
                "created_at": local_time(),
                "files": {
                    str(path.relative_to(out)): digest(path)
                    for path in sorted(out.rglob("*"))
                    if path.is_file() and path != out / "manifest.json"
                },
            },
        )
        verify_artifacts(out)
        return evaluation
    except TimeoutError as error:
        training.update(budget_exhausted=True, seconds=time.perf_counter() - started, failure=str(error))
        write_json(out / "training.json", training)
        raise
    finally:
        env.close()
