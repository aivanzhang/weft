import time

import numpy as np

from weft.env import RollingEnv, task_success
from weft.io import write_json

ERROR_BINS = 81
VELOCITY_BINS = 81


def state_index(observation):
    error, velocity = observation
    return (max(0, min(ERROR_BINS - 1, int((float(error) + 1) * 40))),
            max(0, min(VELOCITY_BINS - 1, int((float(velocity) + 2) * 20))))


def train(world, out, episodes=15000, seed=1729):
    if not 1 <= episodes <= 50000:
        raise ValueError("Training is bounded to 1–50,000 episodes")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    q = np.zeros((ERROR_BINS, VELOCITY_BINS, 3), dtype=np.float64)
    env = RollingEnv(world)
    history, successes = [], []
    total_steps = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=int(rng.integers(0, 2 ** 31)))
        epsilon = max(0.04, 0.9 * (1 - episode / episodes) ** 2)
        while True:
            state = state_index(observation)
            action = int(rng.integers(3)) if rng.random() < epsilon else int(np.argmax(q[state]))
            next_observation, reward, terminated, truncated, info = env.step(action)
            # Time limits are not absorbing states in the continuing control task.
            target = reward if terminated else reward + 0.98 * float(np.max(q[state_index(next_observation)]))
            q[state][action] += 0.15 * (target - q[state][action])
            observation = next_observation
            total_steps += 1
            if terminated or truncated:
                successes.append(int(info["is_success"]))
                break
        if (episode + 1) % 500 == 0 or episode + 1 == episodes:
            history.append({"episode": episode + 1, "recent_training_success_rate": float(np.mean(successes[-500:])),
                            "epsilon": epsilon})
        if time.perf_counter() - started > 120:
            raise TimeoutError("Training exceeded its 120-second CPU wall-time envelope")
    np.savez_compressed(out / "policy.npz", q=q)
    record = {"algorithm": "tabular Q-learning; no scripted policy or expert actions", "seed": seed,
              "episodes": episodes, "steps": total_steps, "seconds": time.perf_counter() - started,
              "learning_rate": 0.15, "discount": 0.98, "history": history}
    write_json(out / "training.json", record)
    return q, record


def evaluate_policies(world, q, count=300, seed=938401):
    if not 1 <= count <= 2000:
        raise ValueError("Evaluation episode bound exceeded")
    rng = np.random.default_rng(seed)
    episode_seeds = [int(x) for x in rng.integers(0, 2 ** 31, size=count)]
    results = {}
    examples = {}
    for policy in ("coast", "random", "q_learning"):
        action_rng = np.random.default_rng(seed + 1)
        env = RollingEnv(world)
        trials = []
        for i, episode_seed in enumerate(episode_seeds):
            observation, _ = env.reset(seed=episode_seed)
            trace = [{"t": 0.0, "x": env.x, "velocity": env.velocity, "target": env.target}]
            total_reward = 0.0
            while True:
                action = (1 if policy == "coast" else int(action_rng.integers(3)) if policy == "random"
                          else int(np.argmax(q[state_index(observation)])))
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                trace.append({"t": env.steps * env.task.dt, "x": env.x, "velocity": env.velocity,
                              "target": env.target, "action": action - 1})
                if terminated or truncated:
                    # Evaluate terminal state against the frozen task, not accumulated reward.
                    success = task_success(env.x, env.velocity, env.target, env.task)
                    trials.append({"seed": episode_seed, "success": success, "return": total_reward,
                                   "steps": env.steps, "out_of_bounds": info["out_of_bounds"],
                                   "final_distance": abs(env.x - env.target), "final_speed": abs(env.velocity)})
                    break
            if i == 0:
                examples[policy] = trace
        successes = sum(t["success"] for t in trials)
        results[policy] = {"successes": successes, "episodes": count, "success_rate": successes / count,
                           "mean_return": float(np.mean([t["return"] for t in trials])), "trials": trials}
    return {"seed": seed, "scope": "Simulation only; same fitted world, unseen initial states; no real actuator trials",
            "matched_episode_seeds": episode_seeds, "policies": results}, examples


def changed_action_trials(world):
    results = {}
    for name, action in (("brake", 0), ("coast", 1), ("push", 2)):
        env = RollingEnv(world)
        env.reset(seed=0, options={"x": 0.25, "velocity": 0.4, "target": 0.8})
        rows = [{"t": 0.0, "x": env.x, "velocity": env.velocity}]
        for _ in range(10):
            _, _, done, truncated, _ = env.step(action)
            rows.append({"t": env.steps * env.task.dt, "x": env.x, "velocity": env.velocity})
            if done or truncated:
                break
        results[name] = rows
    return {"scope": "Simulated interventions with assumed actuator, not physical observations", "trials": results}
