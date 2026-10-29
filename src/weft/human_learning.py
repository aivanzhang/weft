"""Learn one authored joint-position task, not a human's true muscle policy."""

from pathlib import Path
import time

import numpy as np

from weft.human import HumanEnv, joint_contract, validate_human
from weft.io import canonical_hash, write_json


def _index(observation):
    error, velocity = map(float, observation)
    if not np.isfinite(error) or not np.isfinite(velocity):
        raise ValueError("Human policy observations must be finite")
    return max(0, min(60, round((error + 3) * 10))), max(0, min(80, round((velocity + 10) * 4)))


def policy_action(policy, observation):
    if policy.get("algorithm") != "human_tabular_q_learning" or policy.get("schema_version") != 1:
        raise ValueError("Unsupported human policy")
    i, j = _index(observation)
    values = policy["q_values"][i][j]
    if len(values) != 4 or not np.all(np.isfinite(values)):
        raise ValueError("Human policy must contain four finite action values per state")
    return int(np.argmax(values))


def train_human(spec, out, episodes=1500, eval_episodes=100, seed=1729, progress=None, selected_joint="left_knee"):
    spec = validate_human(spec)
    if type(episodes) is not int or not 1 <= episodes <= 5000 or type(eval_episodes) is not int or not 1 <= eval_episodes <= 200:
        raise ValueError("Human training requires 1–5000 episodes and 1–200 evaluation episodes")
    if type(seed) is not int or not 0 <= seed < 2 ** 31 or selected_joint not in joint_contract(spec)[0]:
        raise ValueError("Invalid seed or selected human joint")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("policy.json", "training.json", "rl-evaluation.json", "policy-rollout.json")):
        raise FileExistsError("Learning artifacts exist; choose a new output directory")
    started = time.perf_counter()
    q = np.zeros((61, 81, 4))
    rng = np.random.default_rng(seed)
    env = HumanEnv(spec)
    env.select_joint(selected_joint)
    history, successes = [], []
    steps, completed = 0, 0
    exhausted = False
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=int(rng.integers(0, 2 ** 31)))
            epsilon = max(0.05, 0.8 * (1 - episode / episodes) ** 2)
            while True:
                state = _index(observation)
                action = int(rng.integers(4)) if rng.random() < epsilon else int(np.argmax(q[state]))
                next_observation, reward, done, truncated, info = env.step(action)
                target = reward if done else reward + 0.98 * float(np.max(q[_index(next_observation)]))
                q[state][action] += 0.2 * (target - q[state][action])
                observation = next_observation
                steps += 1
                if done or truncated:
                    successes.append(int(info["is_success"]))
                    break
                if steps % 128 == 0 and time.perf_counter() - started > 90:
                    exhausted = True
                    break
            if exhausted:
                break
            completed = episode + 1
            if completed % 100 == 0 or completed == episodes:
                row = {"episode": completed, "recent_training_success_rate": float(np.mean(successes[-100:])),
                       "epsilon": epsilon, "steps": steps, "elapsed_seconds": time.perf_counter() - started}
                history.append(row)
                if progress:
                    progress("training", row | {"message": f"Joint-control learning: {completed}/{episodes} episodes"})
        policy = {"schema_version": 1, "algorithm": "human_tabular_q_learning", "selected_joint": selected_joint,
                  "target_angle": env.target_angle, "spec_sha256": canonical_hash(spec), "seed": seed,
                  "actions": ["coast", "flex", "extend", "brake"], "q_values": q.tolist(),
                  "observation": ["target_angle_error", "angular_velocity"],
                  "scope": "Learned motor commands for one authored human joint task with virtual support; no balance learning or human muscle recovery"}
        training = {"algorithm": policy["algorithm"], "requested_episodes": episodes, "completed_episodes": completed,
                    "steps": steps, "seconds": time.perf_counter() - started, "seed": seed,
                    "selected_joint": selected_joint, "target_angle": env.target_angle, "history": history,
                    "budget_exhausted": exhausted, "learning_rate": 0.2, "discount": 0.98,
                    "reset_distribution": "Reference initial angles with independent ±0.25-radian jitter, clipped to authored joint limits"}
        write_json(out / "policy.json", policy)
        write_json(out / "training.json", training)
        evaluation_seed = (seed + 938401) % (2 ** 31)
        seeds = [int(v) for v in np.random.default_rng(evaluation_seed).integers(0, 2 ** 31, eval_episodes)]
        policies, traces = {}, {}
        eval_steps = 0
        for name in ("q_learning", "random", "coast"):
            trials = []
            random = np.random.default_rng(evaluation_seed + 1)
            for index, trial_seed in enumerate(seeds):
                observation, _ = env.reset(seed=trial_seed)
                trace = [env.snapshot()] if index == 0 else None
                total = 0.0
                while True:
                    action = int(np.argmax(q[_index(observation)])) if name == "q_learning" else int(random.integers(4)) if name == "random" else 0
                    observation, reward, done, truncated, info = env.step(action)
                    total += reward
                    eval_steps += 1
                    if trace is not None:
                        trace.append(env.snapshot() | {"action": action, "reward": reward})
                    if done or truncated:
                        error, velocity = map(float, env.observation())
                        trials.append({"seed": trial_seed, "success": abs(error) <= 0.08 and abs(velocity) <= 0.15,
                                       "return": total, "steps": env.steps, "final_angle_error": error, "final_speed": velocity})
                        break
                    if eval_steps % 128 == 0 and time.perf_counter() - started > 120:
                        raise TimeoutError("Human learning/evaluation exceeded 120 seconds; training and policy retained")
                if trace is not None:
                    traces[name] = trace
            count = sum(t["success"] for t in trials)
            policies[name] = {"successes": count, "episodes": len(trials), "success_rate": count / len(trials),
                              "mean_return": float(np.mean([row["return"] for row in trials])), "trials": trials}
            if progress:
                progress("evaluation", {"policy": name, "successes": count, "episodes": len(trials),
                                        "message": f"{name}: {count}/{len(trials)} joint-target successes"})
        evaluation = {"schema_version": 1, "seed": evaluation_seed, "matched_episode_seeds": seeds,
                      "selected_joint": selected_joint, "target_angle": env.target_angle, "policies": policies,
                      "success_definition": "Angle error ≤0.08 radians AND angular speed ≤0.15 radians/second",
                      "scope": "Matched simulation starts and one fixed authored task; no physiological or physical-transfer evidence"}
        write_json(out / "rl-evaluation.json", evaluation)
        write_json(out / "policy-rollout.json", {"seed": seeds[0], "traces": traces})
        return {"training": training, "evaluation": {key: value for key, value in evaluation.items() if key != "policies"} | {
            "policies": {name: {key: value for key, value in result.items() if key != "trials"}
                         for name, result in policies.items()}}, "seconds": time.perf_counter() - started}
    finally:
        env.close()
