from copy import deepcopy

import pytest

from weft.evidence_scorecard import build_evidence_scorecard


def summary():
    artifacts = [{'source_name': name, 'path': 'run/' + name, 'sha256': char * 64}
                 for name, char in [('scene.json', 'a'), ('policy.json', 'b'), ('rl-evaluation.json', 'c')]]
    rigid = {'profile': 'spatial-rigid-evidence-v1', 'status': 'available', 'scene_sha256': 'd' * 64,
             'policy_sha256': 'b' * 64, 'return_unit': 'episode_return', 'physical_validation': 'unverified',
             'evidence_kind': 'simulated_development', 'partition': 'exposed_development',
             'evidence': deepcopy(artifacts), 'regimes': {}, 'limitations': []}
    for regime, episodes, learned in [('nominal_exact', 1, 1), ('changed_initial_conditions', 12, 0)]:
        rigid['regimes'][regime] = {'episodes_per_controller': episodes, 'controllers': {
            name: {'episodes': episodes, 'successes': learned if name == 'learned' else 0,
                   'mean_episode_return': -3.5 if name == 'random' else 4.25}
            for name in ['learned', 'coast', 'brake', 'random']}}
    return {'source_manifest_sha256': 'e' * 64, 'scene_sha256': 'd' * 64,
            'artifacts': artifacts, 'evidence_partition': 'construction', 'rigid_learning': rigid}


def rows(value):
    return {row['id']: row for row in build_evidence_scorecard(value)['rows']}


def metrics(row):
    return {metric['id']: metric for metric in row['metrics']}


def test_regimes_keep_counts_returns_and_evidence_boundaries_separate():
    value = summary()
    result = build_evidence_scorecard(value)
    sections = {row['id']: row for row in result['rows']}
    nominal = metrics(sections['task_learning_utility'])
    changed = metrics(sections['robustness_coverage'])
    assert len(nominal) == len(changed) == 9
    assert nominal['nominal_exact_learned_successes']['denominator'] == 1
    assert nominal['nominal_exact_learned_successes']['value'] == 1
    assert changed['changed_initial_conditions_learned_successes']['denominator'] == 12
    assert changed['changed_initial_conditions_learned_successes']['value'] == 0
    assert nominal['nominal_exact_random_mean_episode_return']['value'] == -3.5
    assert nominal['nominal_exact_random_mean_episode_return']['unit'] == 'episode_return'
    for row in (sections['task_learning_utility'], sections['robustness_coverage']):
        assert row['status'] == 'failure_reported'
        for metric in row['metrics'][1:]:
            assert metric['partition'] == 'exposed_development'
            assert metric['evidence_kind'] == 'simulated_development'
            assert metric['evidence'] == value['artifacts']
    assert nominal['physical_transfer_trials']['value'] is None
    assert result['identities']['task_sha256'] is None
    assert sections['execution_actions']['status'] == 'unavailable'
    assert sections['long_run_stability']['status'] == 'unavailable'
    assert sections['physical_relationships']['status'] == 'unavailable'


@pytest.mark.parametrize('status', ['unsupported', 'unavailable'])
def test_missing_or_unsupported_contract_never_exposes_plausible_counts(status):
    value = summary()
    value['rigid_learning']['status'] = status
    value['rigid_learning']['limitations'] = ['Unsupported saved policy version.']
    result = rows(value)
    assert result['task_learning_utility']['status'] == 'unavailable'
    assert len(result['task_learning_utility']['metrics']) == 1
    assert result['robustness_coverage']['status'] == 'unavailable'


@pytest.mark.parametrize('change', ['scene', 'policy', 'pointer', 'duplicate_pointer', 'missing_pointer',
    'units', 'partition', 'physical', 'regime', 'controllers', 'denominator', 'bool_count', 'count',
    'nan_return', 'notes'])
def test_contradictory_normalized_metadata_rejects(change):
    value = summary()
    rigid = value['rigid_learning']
    controller = rigid['regimes']['nominal_exact']['controllers']['learned']
    if change == 'scene': rigid['scene_sha256'] = 'f' * 64
    elif change == 'policy': rigid['policy_sha256'] = 'f' * 64
    elif change == 'pointer': rigid['evidence'][0]['path'] = 'run/unverified.json'
    elif change == 'duplicate_pointer': rigid['evidence'].append(rigid['evidence'][0])
    elif change == 'missing_pointer': rigid['evidence'].pop()
    elif change == 'units': rigid['return_unit'] = 'reward'
    elif change == 'partition': rigid['partition'] = 'fresh_final'
    elif change == 'physical': rigid['physical_validation'] = 'verified'
    elif change == 'regime': rigid['regimes'].pop('changed_initial_conditions')
    elif change == 'controllers': rigid['regimes']['changed_initial_conditions']['controllers'].pop('coast')
    elif change == 'denominator': controller['episodes'] = 12
    elif change == 'bool_count': controller['successes'] = True
    elif change == 'count': controller['successes'] = 2
    elif change == 'nan_return': controller['mean_episode_return'] = float('nan')
    elif change == 'notes': rigid['limitations'] = ['x'] * 9
    with pytest.raises(ValueError):
        build_evidence_scorecard(value)


def test_failure_receipt_discards_unsealed_rigid_measurements():
    value = summary()
    value['source_seal_scope'] = 'failure_receipt_only'
    value['artifacts'] = []
    result = rows(value)
    assert result['task_learning_utility']['status'] == 'unavailable'
    assert result['robustness_coverage']['status'] == 'unavailable'


def test_saved_failure_examples_and_relative_reset_settings_remain_inspectable():
    value = summary()
    changed = value['rigid_learning']['regimes']['changed_initial_conditions']
    changed.update(position_jitter=.015, position_jitter_unit='scene_width', velocity_jitter=.03,
                   velocity_jitter_unit='scene_width_per_second', distinct_seed_count=11,
                   failure_examples=[{'regime': 'changed_initial_conditions', 'controller': 'learned',
                       'seed': 22, 'kind': 'task_not_successful', 'failure': None}], failure_examples_omitted=47)
    result = rows(value)['robustness_coverage']
    measured = metrics(result)
    assert measured['saved_position_jitter']['unit'] == 'scene_width'
    assert measured['saved_velocity_jitter']['unit'] == 'scene_width_per_second'
    assert measured['saved_changed_regime_seeds']['value'] == 11
    assert result['failure_examples'][0]['seed'] == 22
    assert result['failure_examples_omitted'] == 47
    assert 'saved_changed_starts' not in measured


@pytest.mark.parametrize('change', ['count', 'controller', 'unit'])
def test_invalid_failure_examples_or_metre_claims_reject(change):
    value = summary()
    changed = value['rigid_learning']['regimes']['changed_initial_conditions']
    changed.update(failure_examples=[{'regime': 'changed_initial_conditions', 'controller': 'learned',
        'seed': 22, 'kind': 'task_not_successful', 'failure': None}], failure_examples_omitted=47)
    if change == 'count': changed['failure_examples_omitted'] = 0
    elif change == 'controller': changed['failure_examples'][0]['controller'] = []
    elif change == 'unit': changed.update(position_jitter=.1, position_jitter_unit='metres')
    with pytest.raises(ValueError):
        build_evidence_scorecard(value)


@pytest.mark.parametrize('already_shortened', [True, False])
def test_clipped_failure_text_is_explicit(already_shortened):
    value = summary()
    changed = value['rigid_learning']['regimes']['changed_initial_conditions']
    changed.update(failure_examples=[{'regime': 'changed_initial_conditions', 'controller': 'learned',
        'seed': 22, 'kind': 'execution_failure', 'failure': 'x' * (512 if already_shortened else 800),
        'failure_text_truncated': already_shortened}], failure_examples_omitted=47)
    example = rows(value)['robustness_coverage']['failure_examples'][0]
    assert len(example['failure']) == 512 and example['failure_text_truncated'] is True
