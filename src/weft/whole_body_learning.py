"""Bounded policy search over explicit whole-body motors; no hidden stabilizer."""

from pathlib import Path
import shutil
import time

import numpy as np

from weft.human import validate_human
from weft.io import canonical_hash, digest, local_time, write_json
from weft.whole_body import WholeBodyEnv
from weft.whole_body_feasibility import audit_reference_feasibility


ALGORITHM = 'whole_body_cem_v1'
SCOPE = ('Structured reference-tracking feedback with privileged simulator features and authored '
         'virtual root motors; not recovered human intent, muscle activation or unassisted balance.')


def policy_action(policy, env):
    if policy.get('algorithm') != ALGORITHM or policy.get('schema_version') != 1:
        raise ValueError('Unsupported whole-body policy')
    weights = np.asarray(policy.get('weights'), dtype=float)
    if weights.shape != (4, 3) or not np.all(np.isfinite(weights)) or np.any(np.abs(weights) > 3):
        raise ValueError('Whole-body policy requires bounded finite 4-by-3 weights')
    if policy.get('action_names') != list(env.action_names):
        raise ValueError('Whole-body action ordering differs from this environment')
    return _action(weights, env)


def _action(weights, env):
    return np.clip(np.sum(env.feature_matrix() * weights[np.asarray(env.group_indices)], axis=1), -1, 1)


def windows(spec, requested_horizon=60):
    if type(requested_horizon) is not int or not 1 <= requested_horizon <= 180:
        raise ValueError('Whole-body horizon must be 1–180 steps')
    times = np.asarray([row['time_s'] for row in spec['reference']], dtype=float)
    if len(times) < 4 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Reference needs at least four ordered finite samples for guarded time partitions')
    times -= times[0]
    duration = float(times[-1])
    split_index = min(len(times) - 3, int(np.searchsorted(times, duration * .6, side='right')) - 1)
    training_end, evaluation_start = float(times[split_index]), float(times[split_index + 2])
    horizon = min(requested_horizon, int(np.floor(min(training_end, duration - evaluation_start) * 30 - 1e-6)))
    if horizon < 1:
        raise ValueError('Reference is too short for disjoint train/evaluation windows')
    span = horizon / 30
    return {'horizon_steps': horizon, 'split_time_s': duration * .6,
            'training_end_time_s': training_end, 'evaluation_start_time_s': evaluation_start,
            'training_start_times_s': np.linspace(0, training_end - span, 3).tolist(),
            'evaluation_start_times_s': np.linspace(evaluation_start, duration - span, 3).tolist(),
            'reference_duration_s': duration,
            'conditioning_guard': {'last_training_target_sample': split_index,
                                   'last_training_velocity_sample': split_index + 1,
                                   'first_evaluation_sample': split_index + 2},
            'scope': 'Disjoint time windows in an already-exposed development video, not a final holdout.'}


def rollout(env, weights, seed, *, random=False, deadline=None, trace=False):
    env.reset(seed=seed, options={'jitter': True})
    rng = np.random.default_rng(seed + 1)
    snapshots = [env.snapshot()] if trace else None
    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('Whole-body local compute deadline reached')
        action = rng.uniform(-1, 1, len(env.action_names)) if random else _action(weights, env)
        try:
            _, reward, terminated, truncated, _ = env.step(action)
        except RuntimeError:
            # Preserve a failed trial, but never replace nonfinite state with a valid-looking pose.
            if env.metrics().get('failure') is None:
                raise
            break
        if trace:
            snapshots.append(env.snapshot() | {'action': action.tolist(), 'reward': float(reward)})
        if terminated or truncated:
            break
    return env.metrics(), snapshots


def train_whole_body(spec, out, *, seed=1729, generations=8, population=16,
                     horizon_steps=60, max_seconds=120, progress=None):
    spec = validate_human(spec)
    if spec.get('profile') != 'exercise-v3':
        raise ValueError('Whole-body learning requires a mobile exercise-v3 model')
    if abs(spec['dt'] - 1 / 30) > 1e-12:
        raise ValueError('Whole-body v1 requires a 30-Hz model')
    if type(seed) is not int or not 0 <= seed < 2 ** 31:
        raise ValueError('Invalid whole-body seed')
    if type(generations) is not int or not 1 <= generations <= 12:
        raise ValueError('Whole-body generations must be 1–12')
    if type(population) is not int or not 4 <= population <= 24:
        raise ValueError('Whole-body population must be 4–24')
    if type(max_seconds) not in (int, float) or not np.isfinite(max_seconds) or not 1 <= max_seconds <= 180:
        raise ValueError('Whole-body time budget must be 1–180 seconds')
    partition = windows(spec, horizon_steps)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + max_seconds
    feasibility = audit_reference_feasibility(spec)
    write_json(out / 'reference-feasibility.json', feasibility)
    rng = np.random.default_rng(seed)
    train_seeds = [int(x) for x in rng.integers(0, 2 ** 31, 3)]
    eval_seeds = [int(x) for x in rng.integers(0, 2 ** 31, 3)]
    modules = ('whole_body.py', 'whole_body_learning.py', 'human.py', 'mobile_human.py',
               'human_task.py', 'body_frame.py', 'spatial_pose.py', 'root_motion.py', 'whole_body_feasibility.py')
    hashes = {name: digest(Path(__file__).parent / name) for name in modules}
    implementation = out / 'implementation'
    implementation.mkdir()
    for name in modules:
        shutil.copyfile(Path(__file__).parent / name, implementation / name)
    settings = {'created_at': local_time(), 'algorithm': ALGORITHM, 'seed': seed,
                'generations': generations, 'population': population, 'elite_count': max(2, population // 4),
                'initial_mean': 0, 'initial_std': .8, 'minimum_std': .08, 'weight_bounds': [-3, 3],
                'max_seconds': max_seconds, 'training_seeds': train_seeds, 'evaluation_seeds': eval_seeds,
                'partition': partition, 'spec_sha256': canonical_hash(spec), 'module_hashes': hashes,
                'maximum_training_steps': generations * population * 3 * partition['horizon_steps'],
                'maximum_evaluation_steps': 4 * 3 * partition['horizon_steps'],
                'external_api_usd': 0, 'scope': SCOPE}
    settings['reference_feasibility'] = {key: feasibility[key] for key in ('status', 'summary')}
    settings['feasibility_scope'] = 'Diagnostic learning retains all windows and failures; floor conflicts are not repaired or silently omitted.'
    write_json(out / 'settings.json', settings)
    write_json(out / 'human.json', spec)
    envs, history = [], []
    mean, std = np.zeros((4, 3)), np.full((4, 3), .8)
    best, best_score = np.zeros((4, 3)), -float('inf')
    training = {'settings': settings, 'history': history, 'completed_generations': 0,
                'budget_exhausted': False, 'seconds': 0.}
    try:
        envs = [WholeBodyEnv(spec, start_time_s=start, horizon_steps=partition['horizon_steps'])
                for start in partition['training_start_times_s']]
        for generation in range(generations):
            samples = np.clip(rng.normal(mean, std, (population, 4, 3)), -3, 3)
            samples[0] = best
            rows = []
            training.update(active_generation=generation, active_candidates=rows)
            for index, weights in enumerate(samples):
                trials = []
                row = {'candidate': index, 'weights': weights.tolist(), 'trials': trials, 'complete': False}
                rows.append(row)
                for env, trial_seed in zip(envs, train_seeds):
                    try:
                        trials.append(rollout(env, weights, trial_seed, deadline=deadline)[0])
                    except TimeoutError:
                        row['interrupted_trial'] = env.metrics()
                        raise
                score = float(np.mean([trial['return'] for trial in trials]))
                row.update(score=score, complete=True)
                if score > best_score:
                    best, best_score = weights.copy(), score
            elite = np.argsort([-row['score'] for row in rows], kind='stable')[:settings['elite_count']]
            mean = np.mean(samples[elite], axis=0)
            std = np.maximum(np.std(samples[elite], axis=0), settings['minimum_std'])
            history.append({'generation': generation, 'candidates': rows, 'best_training_return': best_score,
                            'next_mean': mean.tolist(), 'next_std': std.tolist()})
            policy = {'schema_version': 1, 'algorithm': ALGORITHM, 'weights': best.tolist(),
                      'action_names': list(envs[0].action_names), 'seed': seed, 'scope': SCOPE,
                      'spec_sha256': canonical_hash(spec), 'training_return': best_score,
                      'feature_names': ['normalized_position_feedback', 'normalized_velocity_feedback',
                                        'normalized_simulator_bias_force'], 'groups': ['torso', 'proximal', 'distal', 'root']}
            training.update(completed_generations=generation + 1, seconds=time.perf_counter() - started,
                            active_generation=None, active_candidates=[])
            write_json(out / 'policy.json', policy)
            write_json(out / 'training.json', training)
            if progress:
                progress({'generation': generation + 1, 'best_training_return': best_score})
        # Evaluation cannot choose weights; policy bytes are fixed before reading later windows.
        frozen_policy = digest(out / 'policy.json')
        for env in envs:
            env.close()
        envs = [WholeBodyEnv(spec, start_time_s=start, horizon_steps=partition['horizon_steps'])
                for start in partition['evaluation_start_times_s']]
        results, traces = {}, {}
        for name, weights in [('learned', best), ('zero', np.zeros((4, 3))),
                              ('random', np.zeros((4, 3))), ('authored_feedback', np.ones((4, 3)))]:
            trials = []
            for index, (env, trial_seed) in enumerate(zip(envs, eval_seeds)):
                metrics, trace = rollout(env, weights, trial_seed, random=name == 'random',
                                         deadline=deadline, trace=index == 0)
                trials.append({'seed': trial_seed, 'start_time_s': partition['evaluation_start_times_s'][index], **metrics})
                if trace is not None:
                    traces[name] = trace
            results[name] = {'trials': trials, 'success_rate': sum(row['success'] for row in trials) / len(trials),
                             'mean_return': float(np.mean([row['return'] for row in trials]))}
        if digest(out / 'policy.json') != frozen_policy or hashes != {
                name: digest(Path(__file__).parent / name) for name in modules}:
            raise RuntimeError('Policy or implementation changed during whole-body evaluation')
        report = {'schema_version': 1, 'policies': results, 'policy_sha256': frozen_policy,
                  'partition': partition, 'seconds': time.perf_counter() - started, 'scope': SCOPE,
                  'reference_feasibility': settings['reference_feasibility'],
                  'comparison': 'All controllers choose one explicit 22-motor vector per 30-Hz step; same seeded resets.'}
        write_json(out / 'rl-evaluation.json', report)
        write_json(out / 'policy-rollout.json', traces)
        write_json(out / 'manifest.json', {'schema_version': 1, 'created_at': local_time(),
                   'files': {str(path.relative_to(out)): digest(path)
                             for path in sorted(out.rglob('*')) if path.is_file()}})
        return report
    except TimeoutError as error:
        training.update(budget_exhausted=True, seconds=time.perf_counter() - started, failure=str(error))
        write_json(out / 'training.json', training)
        raise
    finally:
        for env in envs:
            env.close()
