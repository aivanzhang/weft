"""Bounded CEM policy search and changed-initial-condition evaluation for 3D rigid scenes."""

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.spatial_rigid import ACTIONS, PROFILE, SpatialRigidEnv, validate_scene


LEGACY_POLICY_PROFILE = "spatial-policy-v1"
LEGACY_ALGORITHM = "spatial_cem_v1"
POLICY_PROFILE = "spatial-policy-v2"
ALGORITHM = "spatial_cem_v2"
POLICY_SCHEMA_VERSION = 2
WEIGHT_DIM = 11
WEIGHT_LIMIT = 6.0
SCOPE = (
    "Bounded linear-feedback and braking policy over privileged 3D simulator state; "
    "authored virtual translational forces only, physical validation unverified."
)


def _number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return float(value)


def validate_spatial_policy(policy, scene=None):
    """Validate an explicitly versioned policy against an optional spatial-rigid-v1 scene."""
    if not isinstance(policy, dict):
        raise ValueError("Spatial policy must be a JSON object")
    if type(policy.get("schema_version")) is not int:
        raise ValueError("Spatial policy schema_version must be an integer")
    contract = (policy["schema_version"], policy.get("profile"), policy.get("algorithm"))
    if contract not in ((1, LEGACY_POLICY_PROFILE, LEGACY_ALGORITHM), (POLICY_SCHEMA_VERSION, POLICY_PROFILE, ALGORITHM)):
        raise ValueError(f"Unsupported spatial policy profile/algorithm: {policy.get('profile')!r}/{policy.get('algorithm')!r}")
    if policy.get("physical_validation") != "unverified":
        raise ValueError("Spatial policy must preserve physical_validation='unverified'")
    object_ids = policy.get("object_ids")
    if not isinstance(object_ids, list) or not object_ids or any(not isinstance(item, str) for item in object_ids):
        raise ValueError("Spatial policy requires non-empty object_ids")
    controlled_id = policy.get("controlled_id")
    controlled_index = policy.get("controlled_index")
    if (
        not isinstance(controlled_id, str)
        or type(controlled_index) is not int
        or not 0 <= controlled_index < len(object_ids)
        or object_ids[controlled_index] != controlled_id
    ):
        raise ValueError("Spatial policy controlled_id/controlled_index mismatch")
    obs_dim = policy.get("observation_dim")
    if type(obs_dim) is not int or obs_dim != 13 * len(object_ids) + 4:
        raise ValueError("Spatial policy observation_dim does not match object_ids")
    weights = policy.get("weights")
    if not isinstance(weights, list) or len(weights) != WEIGHT_DIM:
        raise ValueError(f"Spatial policy requires {WEIGHT_DIM} weights")
    for w in weights:
        _number(w, "policy weight", -WEIGHT_LIMIT, WEIGHT_LIMIT)
    _number(policy.get("position_tolerance"), "position_tolerance", 0.005, 0.5)
    _number(policy.get("speed_tolerance"), "speed_tolerance", 0.01, 2.0)
    if not isinstance(policy.get("scene_sha256"), str) or len(policy["scene_sha256"]) != 64:
        raise ValueError("Spatial policy requires a 64-hex scene_sha256")
    if scene is not None:
        validated_scene = validate_scene(scene)
        if policy["scene_sha256"] != canonical_hash(validated_scene):
            raise ValueError("Spatial policy belongs to a different 3D scene")
        scene_ids = [obj["id"] for obj in validated_scene["objects"]]
        if object_ids != scene_ids:
            raise ValueError("Spatial policy object ordering differs from scene")
        target_obj = validated_scene["objects"][controlled_index]
        if target_obj["role"] == "static":
            raise ValueError("Spatial policy cannot target a static object")
        if controlled_id != validated_scene["controlled_id"]:
            raise ValueError("Spatial policy controlled actor differs from scene")
        for name in ("position_tolerance", "speed_tolerance"):
            if policy[name] != validated_scene["task"][name]:
                raise ValueError(f"Spatial policy {name} differs from scene task")
    return deepcopy(policy)


def _action_from_weights(weights, observation, controlled_index, pos_tol, speed_tol):
    # Sealed v1 weights retain their original gains and braking thresholds.
    vel_start = 13 * controlled_index + 7
    vel = observation[vel_start : vel_start + 3]
    offset = observation[-4:-1]
    dist = float(np.linalg.norm(offset))
    speed = float(np.linalg.norm(vel))

    kp = weights[0:3]
    kd = weights[3:6]
    bias = 0.2 * weights[6:9]
    brake_radius = max(0.0, 0.1 * float(weights[9]))
    deadband = max(0.0, 0.05 * (float(weights[10]) + 0.1))

    if dist <= pos_tol + brake_radius:
        if speed > 0.45 * speed_tol:
            return 7
        if dist <= 0.9 * pos_tol:
            return 0
    if float(np.dot(offset, vel)) < 0.0 and brake_radius > 0.0 and speed > max(0.5 * speed_tol, dist / max(brake_radius, 0.02) * 0.15):
        return 7

    demand = kp * offset - kd * vel + bias
    axis = int(np.argmax(np.abs(demand)))
    if abs(float(demand[axis])) <= deadband:
        return 0
    if axis == 0:
        return 2 if demand[0] > 0.0 else 1
    if axis == 1:
        return 3 if demand[1] > 0.0 else 4
    return 5 if demand[2] > 0.0 else 6


def _action_from_weights_v2(weights, observation, controlled_index, pos_tol, speed_tol):
    vel_start = 13 * controlled_index + 7
    vel = observation[vel_start : vel_start + 3]
    offset = observation[-4:-1]
    dist = float(np.linalg.norm(offset))
    speed = float(np.linalg.norm(vel))

    kp = weights[0:3]
    kd = 0.25 * weights[3:6]
    bias_scale = min(1.0, dist / max(2.0 * pos_tol, 1e-3))
    bias = 0.05 * weights[6:9] * bias_scale
    brake_gain = max(0.0, float(weights[9]))
    deadband = max(0.0, 0.02 * float(weights[10]))

    if brake_gain > 0.0:
        if dist <= 0.85 * pos_tol:
            return 7 if speed > 0.7 * speed_tol else 0
        approach_limit = 0.6 * speed_tol + (0.5 + 0.6 * brake_gain) * math.sqrt(max(0.0, dist - 0.5 * pos_tol))
        if speed > approach_limit:
            return 7
        if float(np.dot(offset, vel)) < -0.1 * speed_tol * dist and speed > speed_tol:
            return 7

    demand = kp * offset - kd * vel + bias
    axis = int(np.argmax(np.abs(demand)))
    if abs(float(demand[axis])) <= deadband:
        return 0
    if axis == 0:
        return 2 if demand[0] > 0.0 else 1
    if axis == 1:
        return 3 if demand[1] > 0.0 else 4
    return 5 if demand[2] > 0.0 else 6


def _action_kernel(algorithm):
    if algorithm == LEGACY_ALGORITHM:
        return _action_from_weights
    if algorithm == ALGORITHM:
        return _action_from_weights_v2
    raise ValueError(f"Unsupported spatial policy algorithm: {algorithm!r}")


def policy_action(policy, observation):
    """Choose action 0..7 using the sealed policy's explicit weight interpretation."""
    validate_spatial_policy(policy)
    obs = np.asarray(observation, dtype=float)
    if obs.shape != (policy["observation_dim"],) or not np.isfinite(obs).all():
        raise ValueError("Spatial observation shape or finiteness differs from policy contract")
    weights = np.asarray(policy["weights"], dtype=float)
    return int(
        _action_kernel(policy["algorithm"])(
            weights,
            obs,
            policy["controlled_index"],
            float(policy["position_tolerance"]),
            float(policy["speed_tolerance"]),
        )
    )


def _rollout_episode(
    env,
    *,
    seed,
    reset_options,
    mode="weights",
    weights=None,
    controlled_index=0,
    pos_tol=0.035,
    speed_tol=0.08,
    algorithm=ALGORITHM,
    deadline=None,
    trace=False,
):
    kernel = _action_kernel(algorithm)
    obs, reset_info = env.reset(seed=seed, options=reset_options)
    rng = np.random.default_rng(seed + 104729)
    total_reward = 0.0
    steps = 0
    failure = None
    snapshots = [env.snapshot()] if trace else None
    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Spatial 3D learning local compute deadline reached")
        if mode == "weights":
            act = kernel(weights, obs, controlled_index, pos_tol, speed_tol)
        elif mode == "random":
            act = int(rng.integers(0, len(ACTIONS)))
        elif mode == "coast":
            act = 0
        elif mode == "brake":
            act = 7
        else:
            raise ValueError(f"Unknown controller mode: {mode!r}")
        try:
            obs, reward, terminated, truncated, _ = env.step(act)
        except RuntimeError as error:
            failure = str(error)
            break
        steps += 1
        total_reward += float(reward)
        if trace:
            snapshots.append(env.snapshot() | {"action": int(act), "reward": float(reward)})
        if terminated or truncated:
            break
    final_state = env.snapshot()
    controlled = next(o for o in final_state["objects"] if o["id"] == env.scene["controlled_id"])
    final_dist = float(math.dist(controlled["position"], env.scene["target"]))
    final_speed = float(np.linalg.norm(controlled["velocity"]))
    success = bool(final_state["success"]) and failure is None
    score = total_reward + 2.0 * float(success) - 0.5 * final_dist - 0.1 * final_speed
    summary = {
        "seed": int(seed),
        "reset_jitter": bool(reset_info.get("reset_jitter", False)),
        "perturbation": reset_info.get("perturbation"),
        "steps": steps,
        "return": total_reward,
        "score": float(score),
        "final_distance_m": final_dist,
        "final_speed_mps": final_speed,
        "success": success,
        "terminated": bool(final_state["terminated"]),
        "truncated": bool(final_state["truncated"]),
        "failure": failure,
    }
    return summary, snapshots, steps


def train_spatial_policy(
    source_run_or_scene,
    out,
    *,
    seed=1729,
    eval_seed=938401,
    generations=8,
    population=16,
    eval_episodes=12,
    position_jitter=0.03,
    velocity_jitter=0.15,
    max_seconds=90,
):
    """Train and independently evaluate a bounded 3D rigid policy into a new sealed directory."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Invalid spatial training seed")
    if type(eval_seed) is not int or not 0 <= eval_seed < 2**31 or seed == eval_seed:
        raise ValueError("Invalid spatial evaluation seed or collision with training seed")
    if type(generations) is not int or not 1 <= generations <= 20:
        raise ValueError("Spatial generations must be 1–20")
    if type(population) is not int or not 4 <= population <= 32:
        raise ValueError("Spatial population must be 4–32")
    if type(eval_episodes) is not int or not 4 <= eval_episodes <= 64:
        raise ValueError("Spatial eval_episodes must be 4–64")
    pos_jitter = _number(position_jitter, "position_jitter", 0.0, 0.15)
    vel_jitter = _number(velocity_jitter, "velocity_jitter", 0.0, 1.0)
    budget_s = _number(max_seconds, "max_seconds", 1.0, 180.0)

    out = Path(out).resolve()
    source_provenance = None
    if isinstance(source_run_or_scene, (str, Path)):
        source_dir = Path(source_run_or_scene).resolve()
        if out == source_dir or out.is_relative_to(source_dir):
            raise ValueError("Choose a new output directory outside the immutable source run")
        verified = verify_artifacts(source_dir)
        scene_path = source_dir / "scene.json"
        if not scene_path.is_file():
            raise ValueError("Source 3D directory must contain a sealed scene.json")
        scene = json.loads(scene_path.read_text())
        source_provenance = {
            "source_directory": str(source_dir),
            "source_manifest_sha256": verified["manifest_sha256"],
            "source_scene_sha256": digest(scene_path),
        }
    elif isinstance(source_run_or_scene, dict):
        scene = deepcopy(source_run_or_scene)
    else:
        raise ValueError("Provide a verified 3D run directory or spatial-rigid-v1 scene dict")

    if scene.get("profile") != PROFILE:
        raise ValueError(f"Spatial training requires profile {PROFILE!r}")
    validated_scene = validate_scene(scene)
    if out.exists():
        raise FileExistsError("Choose a new spatial training output directory")

    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + budget_s

    modules = ("spatial_rigid.py", "spatial_learning.py", "spatial_lab.py")
    hashes = {name: digest(Path(__file__).parent / name) for name in modules}
    impl_dir = out / "implementation"
    impl_dir.mkdir()
    for name in modules:
        shutil.copyfile(Path(__file__).parent / name, impl_dir / name)

    rng = np.random.default_rng(seed)
    train_seeds = [int(x) for x in rng.integers(0, 2**31, 3)]
    eval_rng = np.random.default_rng(eval_seed)
    eval_changed_seeds = [int(x) for x in eval_rng.integers(0, 2**31, eval_episodes)]
    nominal_eval_seed = int(eval_rng.integers(0, 2**31))

    object_ids = [obj["id"] for obj in validated_scene["objects"]]
    controlled_id = validated_scene["controlled_id"]
    controlled_index = object_ids.index(controlled_id)
    pos_tol = float(validated_scene["task"]["position_tolerance"])
    speed_tol = float(validated_scene["task"]["speed_tolerance"])
    scene_hash = canonical_hash(validated_scene)

    settings = {
        "schema_version": 1,
        "created_at": local_time(),
        "profile": POLICY_PROFILE,
        "algorithm": ALGORITHM,
        "seed": seed,
        "eval_seed": eval_seed,
        "generations": generations,
        "population": population,
        "elite_count": max(2, population // 4),
        "eval_episodes": eval_episodes,
        "position_jitter": pos_jitter,
        "velocity_jitter": vel_jitter,
        "weight_bounds": [-WEIGHT_LIMIT, WEIGHT_LIMIT],
        "max_seconds": budget_s,
        "training_seeds": train_seeds,
        "nominal_eval_seed": nominal_eval_seed,
        "changed_init_eval_seeds": eval_changed_seeds,
        "scene_sha256": scene_hash,
        "controlled_id": controlled_id,
        "source_provenance": source_provenance,
        "module_hashes": hashes,
        "external_api_usd": 0,
        "physical_validation": "unverified",
        "scope": SCOPE,
    }
    write_json(out / "settings.json", settings)
    # Preserve exact input scene representation so canonical_hash(scene) stays identical to source run.
    write_json(out / "scene.json", scene)

    use_train_jitter = pos_jitter > 0.0 or vel_jitter > 0.0
    train_regimes = [
        {"jitter": False},
        *(
            [{"jitter": True, "position_jitter": pos_jitter, "velocity_jitter": vel_jitter}] * 2
            if use_train_jitter
            else [{"jitter": False}] * 2
        ),
    ]

    mean = np.zeros(WEIGHT_DIM, dtype=float)
    std = np.full(WEIGHT_DIM, 1.5, dtype=float)
    best_weights = np.zeros(WEIGHT_DIM, dtype=float)
    best_score = -float("inf")
    history = []
    total_steps = 0
    completed_episodes = 0
    training = {
        "schema_version": 1,
        "settings": settings,
        "history": history,
        "completed_generations": 0,
        "completed_episodes": 0,
        "episodes": generations * population * len(train_seeds),
        "steps": 0,
        "budget_exhausted": False,
        "seconds": 0.0,
    }

    env = SpatialRigidEnv(validated_scene)
    try:
        for generation in range(generations):
            samples = np.clip(rng.normal(mean, std, (population, WEIGHT_DIM)), -WEIGHT_LIMIT, WEIGHT_LIMIT)
            if generation == 0:
                samples[0] = np.zeros(WEIGHT_DIM, dtype=float)
            else:
                samples[0] = best_weights
            rows = []
            gen_successes = 0
            gen_trials = 0
            for idx, weights in enumerate(samples):
                trials = []
                for trial_seed, reset_opts in zip(train_seeds, train_regimes):
                    trial, _, steps = _rollout_episode(
                        env,
                        seed=trial_seed,
                        reset_options=reset_opts,
                        mode="weights",
                        weights=weights,
                        controlled_index=controlled_index,
                        pos_tol=pos_tol,
                        speed_tol=speed_tol,
                        deadline=deadline,
                    )
                    total_steps += steps
                    completed_episodes += 1
                    gen_trials += 1
                    gen_successes += int(trial["success"])
                    trials.append(trial)
                cand_score = float(np.mean([t["score"] for t in trials]))
                cand_success_rate = sum(int(t["success"]) for t in trials) / len(trials)
                rows.append(
                    {
                        "candidate": idx,
                        "weights": weights.tolist(),
                        "score": cand_score,
                        "success_rate": cand_success_rate,
                        "mean_return": float(np.mean([t["return"] for t in trials])),
                    }
                )
                if cand_score > best_score:
                    best_score = cand_score
                    best_weights = weights.copy()

            elite_idx = np.argsort([-row["score"] for row in rows], kind="stable")[: settings["elite_count"]]
            mean = np.mean(samples[elite_idx], axis=0)
            std = np.maximum(np.std(samples[elite_idx], axis=0), 0.1)
            elite_success_rate = float(np.mean([rows[i]["success_rate"] for i in elite_idx]))
            history.append(
                {
                    "generation": generation,
                    "episode": completed_episodes,
                    "best_training_score": best_score,
                    "recent_training_success_rate": elite_success_rate,
                    "population_success_rate": gen_successes / max(1, gen_trials),
                    "next_mean": mean.tolist(),
                    "next_std": std.tolist(),
                }
            )
            policy_doc = validate_spatial_policy(
                {
                    "schema_version": POLICY_SCHEMA_VERSION,
                    "profile": POLICY_PROFILE,
                    "algorithm": ALGORITHM,
                    "scene_sha256": scene_hash,
                    "controlled_id": controlled_id,
                    "controlled_index": controlled_index,
                    "object_ids": object_ids,
                    "observation_dim": 13 * len(object_ids) + 4,
                    "position_tolerance": pos_tol,
                    "speed_tolerance": speed_tol,
                    "weights": best_weights.tolist(),
                    "seed": seed,
                    "training_score": best_score,
                    "physical_validation": "unverified",
                    "scope": SCOPE,
                },
                validated_scene,
            )
            training.update(
                completed_generations=generation + 1,
                completed_episodes=completed_episodes,
                steps=total_steps,
                seconds=time.perf_counter() - started,
            )
            write_json(out / "policy.json", policy_doc)
            write_json(out / "training.json", training)

        # Freeze policy digest before independent evaluation across nominal and changed initial conditions.
        frozen_policy_sha = digest(out / "policy.json")
        changed_opts = (
            {"jitter": True, "position_jitter": pos_jitter, "velocity_jitter": vel_jitter}
            if use_train_jitter
            else {"jitter": False}
        )
        controllers = [
            ("learned", "weights"),
            ("random", "random"),
            ("coast", "coast"),
            ("brake", "brake"),
        ]
        policies_summary = {}
        nominal_regime = {}
        changed_regime = {}
        traces = {}

        for name, ctrl_mode in controllers:
            nom_trial, nom_trace, _ = _rollout_episode(
                env,
                seed=nominal_eval_seed,
                reset_options={"jitter": False},
                mode=ctrl_mode,
                weights=best_weights,
                controlled_index=controlled_index,
                pos_tol=pos_tol,
                speed_tol=speed_tol,
                deadline=deadline,
                trace=(name == "learned"),
            )
            changed_trials = []
            for idx, c_seed in enumerate(eval_changed_seeds):
                c_trial, c_trace, _ = _rollout_episode(
                    env,
                    seed=c_seed,
                    reset_options=changed_opts,
                    mode=ctrl_mode,
                    weights=best_weights,
                    controlled_index=controlled_index,
                    pos_tol=pos_tol,
                    speed_tol=speed_tol,
                    deadline=deadline,
                    trace=(name == "learned" and idx == 0),
                )
                changed_trials.append(c_trial)
                if c_trace is not None:
                    traces["learned_changed_initial_condition"] = c_trace
            if nom_trace is not None:
                traces["learned_nominal"] = nom_trace

            all_trials = [nom_trial, *changed_trials]
            changed_successes = sum(int(t["success"]) for t in changed_trials)
            total_successes = int(nom_trial["success"]) + changed_successes

            nominal_regime[name] = {
                "episodes": 1,
                "successes": int(nom_trial["success"]),
                "success_rate": float(nom_trial["success"]),
                "mean_return": float(nom_trial["return"]),
                "mean_final_distance_m": float(nom_trial["final_distance_m"]),
                "trials": [nom_trial],
            }
            changed_regime[name] = {
                "episodes": len(changed_trials),
                "successes": changed_successes,
                "success_rate": changed_successes / len(changed_trials),
                "mean_return": float(np.mean([t["return"] for t in changed_trials])),
                "mean_final_distance_m": float(np.mean([t["final_distance_m"] for t in changed_trials])),
                "trials": changed_trials,
            }
            policies_summary[name] = {
                "episodes": len(all_trials),
                "successes": total_successes,
                "success_rate": total_successes / len(all_trials),
                "nominal_success": bool(nom_trial["success"]),
                "changed_init_successes": changed_successes,
                "changed_init_episodes": len(changed_trials),
                "mean_return": float(np.mean([t["return"] for t in all_trials])),
                "mean_final_distance_m": float(np.mean([t["final_distance_m"] for t in all_trials])),
                "trials": all_trials,
            }

        if digest(out / "policy.json") != frozen_policy_sha or hashes != {
            name: digest(Path(__file__).parent / name) for name in modules
        }:
            raise RuntimeError("Spatial policy or implementation changed during evaluation")

        evaluation = {
            "schema_version": 1,
            "profile": POLICY_PROFILE,
            "scene_profile": PROFILE,
            "scene_sha256": scene_hash,
            "policy_sha256": frozen_policy_sha,
            "controlled_id": controlled_id,
            "policies": policies_summary,
            "regimes": {
                "nominal_exact": nominal_regime,
                "changed_initial_conditions": {
                    "position_jitter_m": pos_jitter,
                    "velocity_jitter_mps": vel_jitter,
                    "policies": changed_regime,
                },
            },
            "seconds": time.perf_counter() - started,
            "physical_validation": "unverified",
            "comparison": "Matched nominal and collision-free perturbed 3D initial states across learned, random, coast, and brake baselines; task tolerances unchanged.",
            "scope": SCOPE,
        }
        write_json(out / "rl-evaluation.json", evaluation)
        write_json(
            out / "policy-rollout.json",
            {
                "schema_version": 1,
                "policy_sha256": frozen_policy_sha,
                "controlled_id": controlled_id,
                "physical_validation": "unverified",
                "traces": traces,
            },
        )
        write_json(
            out / "manifest.json",
            {
                "schema_version": 1,
                "created_at": local_time(),
                "files": {
                    str(path.relative_to(out)): digest(path)
                    for path in sorted(out.rglob("*"))
                    if path.is_file() and path.name != "manifest.json"
                },
            },
        )
        verify_artifacts(out)
        return evaluation
    except TimeoutError as error:
        training.update(
            budget_exhausted=True,
            completed_episodes=completed_episodes,
            steps=total_steps,
            seconds=time.perf_counter() - started,
            failure=str(error),
        )
        write_json(out / "training.json", training)
        raise
    finally:
        env.close()
