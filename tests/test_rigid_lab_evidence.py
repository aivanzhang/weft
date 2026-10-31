from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from weft.io import canonical_hash
from weft.rigid_lab_evidence import normalize_rigid_lab_evidence


def raw_document(value):
    return json.dumps(value, sort_keys=True).encode()


def rigid_documents(*, version=1, sidecar=True):
    """Small saved-only fixture; raw_document reproduces its exact pointer bytes."""
    scene = {'schema_version': 1, 'profile': 'spatial-rigid-v1',
        'units': {'length': 'scene_width', 'time': 'second', 'mass': 'relative'},
        'objects': [{'id': 'block', 'role': 'active'}, {'id': 'floor', 'role': 'static'}],
        'controlled_id': 'block', 'target': [.8, .5, .1],
        'task': {'position_tolerance': .03, 'speed_tolerance': .05, 'acceleration': 5., 'max_steps': 10}}
    policy = {'schema_version': version, 'profile': f'spatial-policy-v{version}', 'algorithm': f'spatial_cem_v{version}',
        'scene_sha256': canonical_hash(scene), 'physical_validation': 'unverified',
        'object_ids': ['block', 'floor'], 'controlled_id': 'block', 'controlled_index': 0, 'observation_dim': 30,
        'weights': [0.] * 11, 'position_tolerance': .03, 'speed_tolerance': .05}
    task = ({'profile': scene['profile'], 'controlled_id': 'block', 'target': scene['target'], **scene['task'],
             'scope': 'Authored simulation task; not physical validation.'} if sidecar else None)

    def trial(name, seed, changed, success):
        reward = 2. if success else -.5
        return {'seed': seed, 'steps': 5, 'return': reward, 'score': reward,
            'success': success, 'terminated': True, 'truncated': False, 'failure': None, 'reset_jitter': changed,
            'perturbation': {'block': {'position_delta': [.01, 0., 0.], 'velocity_delta': [.02, 0., 0.],
                                      'fallback_unperturbed': False}} if changed else None,
            'final_distance_m': .01 if success else .1, 'final_speed_mps': 0.}

    def summary(trials):
        successes = sum(row['success'] for row in trials)
        return {'episodes': len(trials), 'successes': successes, 'success_rate': successes / len(trials),
                'mean_return': sum(row['return'] for row in trials) / len(trials), 'trials': trials}

    nominal, changed, aggregate = {}, {}, {}
    for name in ('learned', 'random', 'coast', 'brake'):
        nominal[name] = summary([trial(name, 7, False, name == 'learned')])
        changed[name] = summary([trial(name, seed, True, name == 'learned' and seed == 11) for seed in (11, 12, 13, 14)])
        aggregate[name] = summary(deepcopy(nominal[name]['trials'] + changed[name]['trials']))
        aggregate[name].update(nominal_success=name == 'learned', changed_init_successes=changed[name]['successes'],
                               changed_init_episodes=4)
    evaluation = {'schema_version': 1, 'profile': policy['profile'], 'scene_profile': scene['profile'],
        'scene_sha256': canonical_hash(scene), 'policy_sha256': hashlib.sha256(raw_document(policy)).hexdigest(),
        'controlled_id': 'block', 'physical_validation': 'unverified', 'policies': aggregate,
        'regimes': {'nominal_exact': nominal, 'changed_initial_conditions':
                    {'position_jitter_m': .03, 'velocity_jitter_mps': .15, 'policies': changed}}}
    values = {'scene.json': scene, 'policy.json': policy, 'rl-evaluation.json': evaluation}
    if task is not None:
        values['task.json'] = task
    pointers = [{'source_name': name, 'path': 'run/' + name, 'sha256': hashlib.sha256(raw_document(value)).hexdigest(),
                 'bytes': len(raw_document(value))} for name, value in values.items()]
    return {'scene': scene, 'task': task, 'policy': policy, 'evaluation': evaluation, 'verified_metadata': pointers}


@pytest.mark.parametrize('version,sidecar', [(1, True), (1, False), (2, True)])
def test_saved_regimes_counts_and_episode_return_remain_distinct(version, sidecar):
    values = rigid_documents(version=version, sidecar=sidecar)
    before = deepcopy(values)
    result = normalize_rigid_lab_evidence(**values)
    assert values == before
    assert result['status'] == 'available' and result['physical_validation'] == 'unverified'
    assert result['evidence_kind'] == 'simulated_development' and result['partition'] == 'exposed_development'
    assert result['return_unit'] == 'episode_return' and 'task_sha256' not in result
    assert result['task_binding'] == ('matching_task_sidecar' if sidecar else 'scene_embedded_task')
    nominal, changed = result['regimes']['nominal_exact'], result['regimes']['changed_initial_conditions']
    assert nominal['controllers']['learned'] == {'episodes': 1, 'successes': 1, 'mean_episode_return': 2.}
    assert changed['controllers']['learned'] == {'episodes': 4, 'successes': 1, 'mean_episode_return': .125}
    assert result['aggregate']['controllers']['learned'] == {'episodes': 5, 'successes': 2, 'mean_episode_return': .5}
    assert changed['position_jitter_unit'] == 'scene_width' and changed['velocity_jitter_unit'] == 'scene_width_per_second'
    assert changed['seeds'] == [11, 12, 13, 14] and changed['distinct_seed_count'] == 4
    assert len(changed['failure_examples']) == 8 and changed['failure_examples_omitted'] == 7
    assert {item['source_name'] for item in result['evidence']} == {'scene.json', 'policy.json', 'rl-evaluation.json'} | ({'task.json'} if sidecar else set())
    assert all(set(row) == {'episodes', 'successes', 'mean_episode_return'} for row in changed['controllers'].values())


def test_unsupported_scene_does_not_inspect_opaque_policy_or_evaluation():
    result = normalize_rigid_lab_evidence(scene={'profile': 'planar-rigid-v2'}, policy=object(), evaluation=object(),
                                          verified_metadata=object())
    assert result['status'] == 'unsupported' and result['evidence'] == []


@pytest.mark.parametrize('missing', ['policy', 'evaluation'])
def test_missing_metadata_remains_unavailable(missing):
    values = rigid_documents()
    values[missing] = None
    result = normalize_rigid_lab_evidence(**values)
    assert result['status'] == 'unavailable' and 'regimes' not in result


@pytest.mark.parametrize('key', ['policy', 'evaluation'])
def test_unknown_versions_do_not_claim_supported_evidence(key):
    values = rigid_documents()
    values[key]['schema_version'] = 99
    assert normalize_rigid_lab_evidence(**values)['status'] == 'unsupported'


@pytest.mark.parametrize('change', ['scene_hash', 'policy_hash', 'controlled_id', 'object_order', 'static_actor', 'task_sidecar',
    'policy_tolerance', 'physical', 'units', 'wrong_profile', 'missing_pointer', 'unsafe_pointer', 'duplicate_pointer',
    'missing_controller', 'boolean_count', 'count_mismatch', 'rate_mismatch', 'return_mismatch', 'nonfinite',
    'wrong_reset', 'unmatched_seed', 'unmatched_perturbation', 'aggregate_trials', 'aggregate_counts',
    'aggregate_return', 'aggregate_nominal', 'aggregate_trial_type', 'too_many_trials', 'failed_success', 'unfinished_trial'])
def test_contradictory_saved_evidence_is_rejected(change):
    values = rigid_documents()
    policy, evaluation = values['policy'], values['evaluation']
    nominal = evaluation['regimes']['nominal_exact']['learned']
    changed = evaluation['regimes']['changed_initial_conditions']['policies']['learned']
    if change in ('scene_hash', 'policy_hash', 'controlled_id'):
        evaluation[{'scene_hash': 'scene_sha256', 'policy_hash': 'policy_sha256', 'controlled_id': 'controlled_id'}[change]] = '0' * 64
    elif change == 'object_order':
        policy['object_ids'].reverse()
    elif change == 'static_actor':
        policy['controlled_index'] = 1
    elif change == 'task_sidecar':
        values['task']['target'] = [0., 0., 0.]
    elif change == 'policy_tolerance':
        policy['speed_tolerance'] = .1
    elif change == 'physical':
        evaluation['physical_validation'] = 'verified'
    elif change == 'units':
        values['scene']['units']['length'] = 'metres'
    elif change == 'wrong_profile':
        evaluation['profile'] = 'spatial-policy-v2'
    elif change == 'missing_pointer':
        values['verified_metadata'].pop(1)
    elif change == 'unsafe_pointer':
        values['verified_metadata'][0]['path'] = 'https://example.invalid/leak'
    elif change == 'duplicate_pointer':
        values['verified_metadata'].append(values['verified_metadata'][0])
    elif change == 'missing_controller':
        del evaluation['regimes']['nominal_exact']['brake']
    elif change == 'boolean_count':
        nominal['episodes'] = True
    elif change == 'count_mismatch':
        changed['successes'] = 3
    elif change == 'rate_mismatch':
        changed['success_rate'] = 1.
    elif change == 'return_mismatch':
        changed['mean_return'] = .025
    elif change == 'nonfinite':
        changed['trials'][0]['return'] = float('nan')
    elif change == 'wrong_reset':
        nominal['trials'][0]['reset_jitter'] = True
    elif change == 'unmatched_seed':
        changed['trials'][0]['seed'] = 999
    elif change == 'unmatched_perturbation':
        changed['trials'][0]['perturbation']['block']['velocity_delta'][0] = .1
    elif change == 'aggregate_trials':
        evaluation['policies']['learned']['trials'].reverse()
    elif change == 'aggregate_counts':
        evaluation['policies']['learned']['changed_init_successes'] = 2
    elif change == 'aggregate_return':
        evaluation['policies']['learned']['mean_return'] = .1
    elif change == 'aggregate_nominal':
        evaluation['policies']['learned']['nominal_success'] = 1
    elif change == 'aggregate_trial_type':
        evaluation['policies']['learned']['trials'][0]['success'] = 1
    elif change == 'too_many_trials':
        changed['episodes'] = 65
        changed['trials'] *= 17
    elif change == 'failed_success':
        nominal['trials'][0]['failure'] = 'Solver failure'
    else:
        nominal['trials'][0]['terminated'] = False
    with pytest.raises(ValueError, match='saved rigid evidence'):
        normalize_rigid_lab_evidence(**values)


def test_trial_bound_rejects_before_serializing_saved_documents(monkeypatch):
    values = rigid_documents()
    values['evaluation']['policies']['learned']['trials'] *= 14

    def forbidden(*args, **kwargs):
        raise AssertionError('Oversized trial data reached serialization')

    monkeypatch.setattr('weft.rigid_lab_evidence.json.dumps', forbidden)
    with pytest.raises(ValueError, match='early trial bound'):
        normalize_rigid_lab_evidence(**values)


def test_duplicate_seeds_remain_visible_instead_of_becoming_distinct_starts():
    values = rigid_documents()
    for name in ('learned', 'random', 'coast', 'brake'):
        regime = values['evaluation']['regimes']['changed_initial_conditions']['policies'][name]
        regime['trials'][-1]['seed'] = 13
        values['evaluation']['policies'][name]['trials'][-1]['seed'] = 13
    changed = normalize_rigid_lab_evidence(**values)['regimes']['changed_initial_conditions']
    assert changed['episodes_per_controller'] == 4 and changed['distinct_seed_count'] == 3


def test_normalizer_import_has_no_native_media_or_runtime_dependency(tmp_path):
    code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'numpy','mujoco','cv2','PIL','torch','mediapipe'} or name in {'weft.spatial_rigid','weft.spatial_learning','weft.lab'}:
            raise AssertionError('Forbidden import: '+name)
sys.meta_path.insert(0,Block())
from weft.rigid_lab_evidence import normalize_rigid_lab_evidence
assert normalize_rigid_lab_evidence(scene={'profile':'planar-rigid-v1'},verified_metadata=[])['status']=='unsupported'
"""
    source = Path(__file__).parents[1] / 'src'
    completed = subprocess.run([sys.executable, '-P', '-B', '-c', code],
        env={'PATH': os.defpath, 'PYTHONPATH': str(source), 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
