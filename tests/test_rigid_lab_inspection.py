import hashlib
import json

import pytest

from weft.lab_inspection import load_inspection
from test_rigid_lab_evidence import raw_document, rigid_documents


def saved_run(root, *, version=1, sidecar=True, change=None):
    values = rigid_documents(version=version, sidecar=sidecar)
    if change:
        change(values)
    root.mkdir()
    documents = {'scene.json': values['scene'], 'policy.json': values['policy'],
                 'rl-evaluation.json': values['evaluation']}
    if values['task'] is not None:
        documents['task.json'] = values['task']
    for name, value in documents.items():
        (root / name).write_bytes(raw_document(value))
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in documents}
    (root / 'manifest.json').write_bytes(raw_document({'schema_version': 1, 'files': files}))
    return root


@pytest.mark.parametrize('version,sidecar', [(1, True), (2, False)])
def test_saved_rigid_evaluation_reaches_common_inspection_without_native_work(tmp_path, version, sidecar):
    root = saved_run(tmp_path / 'run', version=version, sidecar=sidecar)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    result = load_inspection(root, profile='spatial-rigid')
    assert result['status'] == 'available', result
    assert result['source_manifest_sha256'] == hashlib.sha256(before['manifest.json']).hexdigest()
    verified = {row['source_name']: row for row in result['verified_metadata']}
    assert set(verified) == set(before)
    rows = {row['id']: row for row in result['evidence_scorecard']['rows']}
    nominal = {metric['id']: metric for metric in rows['task_learning_utility']['metrics']}
    changed = {metric['id']: metric for metric in rows['robustness_coverage']['metrics']}
    assert (nominal['nominal_exact_learned_successes']['value'], nominal['nominal_exact_learned_successes']['denominator']) == (1, 1)
    assert (changed['changed_initial_conditions_learned_successes']['value'], changed['changed_initial_conditions_learned_successes']['denominator']) == (1, 4)
    assert changed['changed_initial_conditions_learned_mean_episode_return']['unit'] == 'episode_return'
    assert rows['robustness_coverage']['failure_examples_omitted'] == 7
    assert result['evidence_scorecard']['identities']['task_sha256'] is None
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    for pointer in nominal['nominal_exact_learned_successes']['evidence']:
        assert pointer['path'] == 'run/' + pointer['source_name']
        assert pointer['sha256'] == verified[pointer['source_name']]['sha256']


@pytest.mark.parametrize('change', ['tampered_file', 'different_scene', 'different_task', 'oversized', 'truncated'])
def test_contradictory_or_unreadable_rigid_evidence_is_explicitly_unavailable(tmp_path, change):
    def mutate(values):
        if change == 'different_scene': values['evaluation']['scene_sha256'] = 'f' * 64
        if change == 'different_task': values['task']['max_steps'] = 20
    root = saved_run(tmp_path / 'run', change=mutate)
    if change in ('tampered_file', 'oversized', 'truncated'):
        raw = b'{}' if change == 'tampered_file' else b' ' * (1024 * 1024 + 1) if change == 'oversized' else b'{'
        (root / 'rl-evaluation.json').write_bytes(raw)
        if change != 'tampered_file':
            manifest = json.loads((root / 'manifest.json').read_bytes())
            manifest['files']['rl-evaluation.json'] = hashlib.sha256(raw).hexdigest()
            (root / 'manifest.json').write_bytes(raw_document(manifest))
    result = load_inspection(root, profile='spatial-rigid')
    assert result['status'] == 'unavailable'
    assert 'evidence_scorecard' not in result


def test_unknown_policy_version_keeps_source_inspection_but_no_success_counts(tmp_path):
    root = saved_run(tmp_path / 'run', version=99)
    result = load_inspection(root, profile='spatial-rigid')
    assert result['status'] == 'available', result
    row = next(row for row in result['evidence_scorecard']['rows'] if row['id'] == 'task_learning_utility')
    assert row['status'] == 'unavailable' and len(row['metrics']) == 1
    assert any('version' in note for note in row['uncertainty_and_missing_measurements'])
