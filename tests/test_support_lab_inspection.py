"""Saved authored-support metadata only; fixtures never execute physics or read media."""

import json

import pytest

from weft.io import canonical_hash
from weft.shared_support_task import build_shared_support_task
from test_lab_evidence import card_rows, digest, exercise_run, load, metrics, put, seal
from test_shared_support_task import support_fixture


def support_run(root, *, learned=True, evaluated=True):
    exercise_run(root, learned=learned, evaluated=evaluated)
    original = json.loads((root / 'scene.json').read_bytes())
    scene, task = support_fixture(horizon=9, root=True)
    scene.update(human_spec=original['human_spec'], source_binding=original['source_binding'])
    scene['source_binding']['entity_bindings'] = [{
        'world_entity_id': 'support_a', 'provenance_mode': 'authored_addition',
        'observed_slots': 0, 'missing_slots': 10, 'pre_init_slots': 0,
        'observation_document_sha256': None, 'observation_entity_id': None}]
    task = build_shared_support_task(scene, support_requirements=task['support_requirements'], root_assistance=True)
    contract = json.loads((root / 'exercise-lab.json').read_bytes())
    contract.update(scene_sha256=canonical_hash(scene), task_sha256=canonical_hash(task))
    put(root, 'scene.json', scene)
    put(root, 'exercise-task.json', task)
    if learned:
        policy = {'schema_version': 1, 'profile': 'shared-support-policy-v1',
                  'scene_sha256': canonical_hash(scene), 'task_sha256': canonical_hash(task),
                  'gates_sha256': task['gates_sha256']}
        put(root, 'policy.json', policy)
        contract['policy_sha256'] = digest(root / 'policy.json')
        contract['exercise_policy'].update(profile=policy['profile'], policy_sha256=contract['policy_sha256'])
    put(root, 'exercise-lab.json', contract)
    seal(root)
    return root


@pytest.mark.parametrize('learned,evaluated', [(False, False), (True, False), (True, True)])
def test_authored_support_inspection_keeps_scope_and_saved_counts(tmp_path, learned, evaluated):
    root = support_run(tmp_path / 'run', learned=learned, evaluated=evaluated)
    before = {p.relative_to(root): digest(p) for p in root.rglob('*') if p.is_file()}
    result = load(root)
    assert result['status'] == 'available', result
    support = next(row for row in result['actors'] if row['id'] == 'support_a')
    assert support['source_binding']['kind'] == 'authored_addition'
    assert support['scope'] == 'Authored model; no source-qualified observation binding available.'
    rows = card_rows(result)
    assert rows['physical_relationships']['status'] == 'unavailable'
    learning = metrics(rows['task_learning_utility'])
    if evaluated:
        assert learning['learned_successes']['value'] == 3
        assert learning['learned_successes']['partition'] == 'exposed_development'
    else:
        assert set(learning) == {'physical_transfer_trials'}
    assert {p.relative_to(root): digest(p) for p in root.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('damage', ['task_profile', 'gates', 'physical_claim', 'support_identity',
                                  'policy_profile', 'policy_gates', 'source_binding'])
def test_resealed_support_metadata_cannot_bypass_task_or_policy_scope(tmp_path, damage):
    root = support_run(tmp_path / 'run')
    task = json.loads((root / 'exercise-task.json').read_bytes())
    policy = json.loads((root / 'policy.json').read_bytes())
    contract = json.loads((root / 'exercise-lab.json').read_bytes())
    if damage == 'task_profile':
        task['profile'] = 'shared-exercise-task-v1'
    elif damage == 'gates':
        task['gates_sha256'] = policy['gates_sha256'] = 'a' * 64
    elif damage == 'physical_claim':
        task['source_scope']['source_contact_verified'] = True
    elif damage == 'support_identity':
        task['source_scope']['supports'][0]['object_sha256'] = 'a' * 64
    elif damage == 'policy_profile':
        policy['profile'] = 'shared-exercise-policy-v4'
    elif damage == 'policy_gates':
        policy['gates_sha256'] = 'a' * 64
    else:
        source = json.loads((root / 'source.json').read_bytes())
        source['sha256'] = 'a' * 64
        put(root, 'source.json', source)
    put(root, 'exercise-task.json', task)
    contract['task_sha256'] = policy['task_sha256'] = canonical_hash(task)
    put(root, 'policy.json', policy)
    contract['policy_sha256'] = digest(root / 'policy.json')
    contract['exercise_policy'].update(profile=policy['profile'], policy_sha256=contract['policy_sha256'])
    put(root, 'exercise-lab.json', contract)
    seal(root)
    assert load(root)['status'] == 'unavailable'
