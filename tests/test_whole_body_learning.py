import json
from types import SimpleNamespace

import numpy as np
import pytest

from weft.human import build_human
from weft.whole_body_learning import ALGORITHM, policy_action, train_whole_body, windows
from test_mobile_integration import evidence


def test_partitions_never_hold_last_pose_or_overlap_training_with_evaluation():
    spec = {'reference': [{'time_s': float(t)} for t in np.linspace(2, 8, 61)]}
    result = windows(spec, 60)
    span = result['horizon_steps'] / 30
    assert max(result['training_start_times_s']) + span <= result['split_time_s']
    assert min(result['evaluation_start_times_s']) >= result['split_time_s']
    assert max(result['evaluation_start_times_s']) + span <= 6.
    guard = result['conditioning_guard']
    assert guard['last_training_velocity_sample'] < guard['first_evaluation_sample']
    with pytest.raises(ValueError):
        windows({'reference': [{'time_s': 0.}, {'time_s': .01}]})


def test_policy_actions_cover_all_axes_and_respect_group_features():
    names = [str(i) for i in range(22)]
    groups = np.array([0] * 3 + [1] * 8 + [2] * 8 + [3] * 3)
    env = SimpleNamespace(action_names=names, group_indices=groups,
                          feature_matrix=lambda: np.tile([.5, -.25, 2.], (22, 1)))
    policy = {'algorithm': ALGORITHM, 'schema_version': 1, 'action_names': names,
              'weights': [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]}
    actions = policy_action(policy, env)
    np.testing.assert_allclose(actions, [.5] * 3 + [-.25] * 8 + [1] * 8 + [0] * 3)
    for mutation in ({'weights': [[float('nan')] * 3] * 4}, {'weights': [[4.] * 3] * 4},
                     {'action_names': names[::-1]}, {'algorithm': 'human_tabular_q_learning'}):
        with pytest.raises(ValueError):
            policy_action(policy | mutation, env)


def test_small_real_physics_search_retains_candidates_and_freezes_policy_before_evaluation(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    spec = build_human(evidence(), model, body_model='mobile')
    reports = []
    for name in ('a', 'b'):
        directory = tmp_path / name
        reports.append(train_whole_body(spec, directory, generations=2, population=4, horizon_steps=3, seed=5))
    first = json.loads((tmp_path / 'a' / 'training.json').read_text())
    policy = json.loads((tmp_path / 'a' / 'policy.json').read_text())
    assert len(first['history']) == 2
    assert first['history'][0]['candidates'][0]['weights'] == np.zeros((4, 3)).tolist()
    candidates = [row for generation in first['history'] for row in generation['candidates']]
    assert policy['training_return'] == max(row['score'] for row in candidates)
    assert all(len(row['trials']) == 3 for row in candidates)
    assert json.loads((tmp_path / 'b' / 'policy.json').read_text()) == policy
    assert reports[0]['policies'] == reports[1]['policies']
    assert set(reports[0]['policies']) == {'learned', 'zero', 'random', 'authored_feedback'}
    trial_seeds = [row['seed'] for row in reports[0]['policies']['learned']['trials']]
    for result in reports[0]['policies'].values():
        assert [row['seed'] for row in result['trials']] == trial_seeds
        assert all(row['completed_horizon'] for row in result['trials'])
    assert set(first['settings']['training_seeds']).isdisjoint(trial_seeds)
    with pytest.raises(FileExistsError):
        train_whole_body(spec, tmp_path / 'a', generations=1, population=4, horizon_steps=3)


@pytest.mark.parametrize('kwargs', [{'seed': True}, {'generations': 13}, {'population': 25},
                                    {'max_seconds': float('nan')}, {'max_seconds': 181}, {'horizon_steps': 0}])
def test_rejects_unbounded_work_before_creating_output(tmp_path, kwargs):
    model = tmp_path / 'model'
    model.mkdir()
    spec = build_human(evidence(), model, body_model='mobile')
    with pytest.raises(ValueError):
        train_whole_body(spec, tmp_path / 'invalid', **kwargs)
    assert not (tmp_path / 'invalid').exists()


def test_timeout_keeps_partial_trial_without_claiming_completed_evaluation(tmp_path, monkeypatch):
    from weft import whole_body_learning
    model = tmp_path / 'model'
    model.mkdir()
    spec = build_human(evidence(), model, body_model='mobile')
    ticks = iter([0., 2., 2.])
    monkeypatch.setattr(whole_body_learning.time, 'perf_counter', lambda: next(ticks))
    with pytest.raises(TimeoutError):
        train_whole_body(spec, tmp_path / 'interrupted', generations=2, population=4,
                         horizon_steps=3, max_seconds=1)
    training = json.loads((tmp_path / 'interrupted' / 'training.json').read_text())
    assert training['budget_exhausted'] and training['completed_generations'] == 0
    candidate = training['active_candidates'][0]
    assert candidate['complete'] is False and candidate['interrupted_trial']['steps'] == 0
    assert not (tmp_path / 'interrupted' / 'rl-evaluation.json').exists()
    assert not (tmp_path / 'interrupted' / 'manifest.json').exists()
