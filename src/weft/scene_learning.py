"""Bounded Q-learning and matched simulation trials on one fixed scene/task."""

from pathlib import Path
import time

import numpy as np

from weft.io import canonical_hash, write_json
from weft.scene import ACTIONS, SceneEnv, task_success, validate_scene


ERROR_EDGES = [-0.6, -0.3, -0.15, -0.07, -0.025, 0.025, 0.07, 0.15, 0.3, 0.6]
VELOCITY_EDGES = [-0.8, -0.3, -0.1, 0.1, 0.3, 0.8]


def _state_key(observation, error_edges=ERROR_EDGES, velocity_edges=VELOCITY_EDGES):
    if len(observation) != 4 or not np.all(np.isfinite(observation)):
        raise ValueError("Policy requires four finite observations")
    indices = np.searchsorted(error_edges, observation[:2]).tolist()
    indices += np.searchsorted(velocity_edges, observation[2:]).tolist()
    return ",".join(str(i) for i in indices)


def policy_action(policy, observation):
    if policy.get("schema_version") != 1 or policy.get("algorithm") != "tabular_q_learning":
        raise ValueError("Unsupported policy schema")
    key = _state_key(observation, policy["error_edges"], policy["velocity_edges"])
    values = policy["q_values"].get(key)
    if values is None:
        return 0
    if len(values) != 6 or not np.all(np.isfinite(values)):
        raise ValueError("Invalid policy action values")
    return int(np.argmax(values))


def _save_policy(q, scene, out, seed):
    policy = {"schema_version": 1, "algorithm": "tabular_q_learning", "scene_sha256": canonical_hash(scene),
              "seed": seed, "actions": list(ACTIONS), "observation": ["target_dx", "target_dy", "vx", "vy"],
              "error_edges": ERROR_EDGES, "velocity_edges": VELOCITY_EDGES,
              "unknown_state_action": "coast", "q_values": {key: values.tolist() for key, values in q.items()},
              "scope": "Learned in this simulation; authored actuator and parking objective; no physical transfer claim"}
    write_json(out / "policy.json", policy)
    return policy


def train_scene(scene, out, episodes=1500, eval_episodes=100, seed=1729, progress=None):
    scene = validate_scene(scene)
    if type(episodes) is not int or not 1 <= episodes <= 5000:
        raise ValueError("Training requires 1–5000 episodes")
    if type(eval_episodes) is not int or not 1 <= eval_episodes <= 200:
        raise ValueError("Evaluation requires 1–200 episodes per policy")
    if type(seed) is not int or not 0 <= seed < 2 ** 31:
        raise ValueError("Seed must be a nonnegative 31-bit integer")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = ("policy.json", "training.json", "rl-evaluation.json", "policy-rollout.json")
    if any((out / name).exists() for name in names):
        raise FileExistsError("Training artifacts already exist; choose a new run directory")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    env = SceneEnv(scene)
    q, history, successes = {}, [], []
    steps = 0
    completed = 0
    budget_exhausted = False
    learning_rate, discount = 0.2, 0.98
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=int(rng.integers(0, 2 ** 31)))
            epsilon = max(0.06, 0.85 * (1 - episode / episodes) ** 2)
            while True:
                key = _state_key(observation)
                values = q.setdefault(key, np.zeros(6))
                if rng.random() < epsilon:
                    action = int(rng.integers(6))
                else:
                    # Random training ties avoid encoding a directional expert in the policy.
                    action = int(rng.choice(np.flatnonzero(values == values.max())))
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_values = q.setdefault(_state_key(next_observation), np.zeros(6))
                target = reward if terminated else reward + discount * float(next_values.max())
                values[action] += learning_rate * (target - values[action])
                observation = next_observation
                steps += 1
                if terminated or truncated:
                    successes.append(int(info["is_success"]))
                    break
                if steps % 128 == 0 and time.perf_counter() - started > 90:
                    budget_exhausted = True
                    break
            if budget_exhausted:
                break
            completed = episode + 1
            if completed % 100 == 0 or completed == episodes:
                record = {"episode": completed, "recent_training_success_rate": float(np.mean(successes[-100:])),
                          "epsilon": epsilon, "steps": steps, "elapsed_seconds": time.perf_counter() - started}
                history.append(record)
                if progress:
                    progress("training", {"status": "progress", **record, "requested_episodes": episodes})
        policy = _save_policy(q, scene, out, seed)
        training = {"algorithm": "tabular_q_learning", "seed": seed, "requested_episodes": episodes,
                    "completed_episodes": completed, "steps": steps, "seconds": time.perf_counter() - started,
                    "budget_exhausted": budget_exhausted, "learning_rate": learning_rate, "discount": discount,
                    "history": history, "visited_states": len(q),
                    "reset_distribution": "Up to 0.06 image-width position jitter per dynamic object, without overlap",
                    "scope": "Actual Q-learning from environment rewards; no scripted expert or prerecorded action training"}
        write_json(out / "training.json", training)
    finally:
        env.close()
    evaluation_seed = (seed + 938401) % (2 ** 31)
    evaluation_rng = np.random.default_rng(evaluation_seed)
    seeds = [int(v) for v in evaluation_rng.integers(0, 2 ** 31, size=eval_episodes)]
    policies, traces = {}, {}
    env = SceneEnv(scene)
    eval_steps = 0
    try:
        for name in ("q_learning", "random", "coast"):
            action_rng = np.random.default_rng(evaluation_seed + 1)
            trials = []
            for index, trial_seed in enumerate(seeds):
                observation, _ = env.reset(seed=trial_seed)
                trace = [env.snapshot()] if index == 0 else None
                total_reward = 0.0
                while True:
                    action = (policy_action(policy, observation) if name == "q_learning" else
                              int(action_rng.integers(6)) if name == "random" else 0)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    eval_steps += 1
                    if trace is not None:
                        trace.append(env.snapshot() | {"action": action, "reward": reward})
                    if terminated or truncated:
                        body = env.bodies[scene["controlled_id"]]
                        trials.append({"seed": trial_seed, "success": task_success(scene, body.position, body.velocity),
                                       "steps": env.steps, "return": total_reward, "final_distance": info["distance"],
                                       "final_speed": info["speed"], "terminated": terminated, "truncated": truncated})
                        break
                    if eval_steps % 128 == 0 and time.perf_counter() - started > 120:
                        raise TimeoutError("Training/evaluation exceeded 120 seconds; policy and training evidence retained")
                if trace is not None:
                    traces[name] = trace
            successes_count = sum(t["success"] for t in trials)
            policies[name] = {"successes": successes_count, "episodes": len(trials),
                              "success_rate": successes_count / len(trials),
                              "mean_return": float(np.mean([trial["return"] for trial in trials])), "trials": trials}
            if progress:
                progress("evaluation", {"status": "progress", "policy": name,
                                        "successes": successes_count, "episodes": len(trials)})
    finally:
        env.close()
    evaluation = {"schema_version": 1, "scene_sha256": canonical_hash(scene), "seed": evaluation_seed,
                  "matched_episode_seeds": seeds, "policies": policies,
                  "success_definition": "Position AND speed tolerances, scored from terminal state independently of reward",
                  "scope": "All policies evaluated on this same fitted scene/task; development simulation trials, not physical evidence"}
    write_json(out / "rl-evaluation.json", evaluation)
    write_json(out / "policy-rollout.json", {"scene_sha256": canonical_hash(scene), "seed": seeds[0], "traces": traces})
    return {"training": training, "evaluation": {key: value for key, value in evaluation.items() if key != "policies"} | {
        "policies": {name: {key: value for key, value in values.items() if key != "trials"}
                     for name, values in policies.items()}}, "seconds": time.perf_counter() - started}
