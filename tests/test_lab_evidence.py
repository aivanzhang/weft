import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from weft import lab_inspection
from weft.io import canonical_hash


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(root):
    put(root, 'manifest.json', {'schema_version': 1, 'status': 'ready', 'files': {
        path.relative_to(root).as_posix(): digest(path)
        for path in root.rglob('*') if path.is_file() and path.name != 'manifest.json'}})
    # Nested manifests are ordinary pinned members of the outer run.
    value = json.loads((root / 'manifest.json').read_bytes())
    for path in root.glob('*/manifest.json'):
        value['files'][path.relative_to(root).as_posix()] = digest(path)
    put(root, 'manifest.json', value)


def load(root):
    return lab_inspection.load_inspection(root, profile='shared-world')


def card_rows(result):
    return {row['id']: row for row in result['evidence_scorecard']['rows']}


def metrics(row):
    return {metric['id']: metric for metric in row['metrics']}


def exercise_run(root, *, learned=True, evaluated=True):
    root.mkdir()
    put(root, 'source.webm', b'opaque source media; no decoder may read this')
    source_sha = digest(root / 'source.webm')
    put(root, 'source.json', {'sha256': source_sha, 'decoded_frames': 10})
    scene = {'schema_version': 1, 'profile': 'shared-world-v3', 'objects': [], 'dt': 1 / 30,
        'human_spec': {'source_sha256': source_sha, 'reference': [{'time_s': 0.}, {'time_s': .3}]},
        'coordinate_frame': {'scale_bridge': {'body_height_to_world_scale': 1.25}},
        'source_binding': {'source_sha256': source_sha, 'total_source_frames': 10,
                           'evidence_documents': {}, 'entity_bindings': []},
        'task': {'start_offset_s': 0., 'horizon_steps': 9}}
    task = {'schema_version': 1, 'profile': 'shared-exercise-task-v2', 'scene_sha256': canonical_hash(scene),
            'interval': dict(scene['task']), 'control': {'root_assistance': 'explicit_3_axis', 'object_forces': 'disabled'}}
    scene_sha, task_sha = canonical_hash(scene), canonical_hash(task)
    put(root, 'scene.json', scene)
    put(root, 'exercise-task.json', task)
    contract = {'schema_version': 1, 'profile': 'shared-world', 'control_profile': 'shared_exercise',
                'scene_sha256': scene_sha, 'task_sha256': task_sha, 'learned_policy_available': learned}
    if learned:
        policy = {'schema_version': 1, 'profile': 'shared-exercise-policy-v4',
                  'scene_sha256': scene_sha, 'task_sha256': task_sha}
        put(root, 'policy.json', policy)
        policy_sha = digest(root / 'policy.json')
        contract['policy_sha256'] = policy_sha
        contract['exercise_policy'] = {'profile': policy['profile'], 'policy_sha256': policy_sha,
            'evaluation_status': 'evaluated' if evaluated else 'not_attached',
            'evaluation_successes': {name: {'successes': successes, 'episodes': 3}
                for name, successes in (('learned', 3), ('authored', 3), ('coast', 0), ('random', 0))} if evaluated else None}
        if evaluated:
            put(root, 'policy-evaluation.json', b'opaque potentially large trial arrays; must not be parsed')
            put(root, 'policy-evaluation-run/manifest.json', b'opaque evaluation manifest; not independently read')
            contract['policy_evaluation_sha256'] = digest(root / 'policy-evaluation.json')
            contract['policy_evaluation_manifest_sha256'] = digest(root / 'policy-evaluation-run/manifest.json')
    put(root, 'exercise-lab.json', contract)
    put(root, 'reference-feasibility.json', b'linked but not read; do not import another scene audit')
    put(root, 'evidence.json', b'linked raw observations; do not parse here')
    put(root, 'human-reference.json', b'large pose replay; do not parse here')
    seal(root)
    return root


def test_compact_current_policy_counts_preserve_unknown_metrics_and_exact_pointers(tmp_path):
    root = exercise_run(tmp_path / 'run')
    before = {path.relative_to(root): digest(path) for path in root.rglob('*') if path.is_file()}
    result = load(root)
    assert result['status'] == 'available', result
    rows = card_rows(result)
    assert len(rows) == 14
    learning = metrics(rows['task_learning_utility'])
    for name, successes in [('learned', 3), ('authored', 3), ('coast', 0), ('random', 0)]:
        metric = learning[name + '_successes']
        assert (metric['value'], metric['numerator'], metric['denominator']) == (successes, successes, 3)
        assert metric['partition'] == 'exposed_development' and metric['evidence_kind'] == 'simulated_development'
        assert learning[name + '_completed_intervals']['value'] is None
        assert learning[name + '_mean_reward']['value'] is None
    assert rows['execution_actions']['status'] == rows['long_run_stability']['status'] == 'unavailable'
    assert rows['physical_relationships']['status'] == 'unavailable'
    assert 'saved_changed_starts' not in metrics(rows['robustness_coverage'])
    assert learning['physical_transfer_trials']['value'] is None
    assert result['source_manifest_sha256'] == digest(root / 'manifest.json')
    verified = {item['source_name']: item for item in result['verified_metadata']}
    assert set(verified) == {'manifest.json', 'source.json', 'scene.json', 'exercise-task.json', 'exercise-lab.json', 'policy.json'}
    for name, item in verified.items():
        assert item['path'] == 'run/' + name and item['sha256'] == before[Path(name)]
    pointers = learning['learned_successes']['evidence']
    assert {item['source_name'] for item in pointers} == {'exercise-lab.json', 'policy.json', 'scene.json', 'exercise-task.json'}
    assert all(item['path'] == verified[item['source_name']]['path']
               and item['sha256'] == verified[item['source_name']]['sha256'] for item in pointers)
    assert 'evidence.json' in {item['name'] for item in result['artifact_links']}
    assert 'evidence.json' not in verified and 'policy-evaluation.json' not in verified
    assert metrics(rows['parameter_validity'])['authored_body_scale_prior']['value'] == 1.25
    assert all(digest(root / name) == value for name, value in before.items())


@pytest.mark.parametrize('learned,evaluated', [(False, False), (True, False)])
def test_untrained_or_unevaluated_labs_do_not_invent_successes(tmp_path, learned, evaluated):
    result = load(exercise_run(tmp_path / 'run', learned=learned, evaluated=evaluated))
    assert result['status'] == 'available', result
    assert card_rows(result)['task_learning_utility']['status'] == 'unavailable'
    assert set(metrics(card_rows(result)['task_learning_utility'])) == {'physical_transfer_trials'}


@pytest.mark.parametrize('profile', ['planar-rigid-v1', 'planar-rigid-v2', 'spatial-rigid-v1', 'shared-world-v1'])
def test_other_profiles_reuse_all_categories_without_exercise_or_learning_fabrication(tmp_path, profile):
    root = tmp_path / 'run'
    root.mkdir()
    put(root, 'scene.json', {'profile': profile, 'objects': [], 'assumptions': ['Authored depth and material.']})
    put(root, 'policy.json', b'opaque non-exercise policy; not read')
    put(root, 'rl-evaluation.json', b'different regimes and units; not read')
    seal(root)
    result = load(root)
    assert result['status'] == 'available', result
    rows = card_rows(result)
    assert len(rows) == 14 and rows['task_learning_utility']['status'] == 'unavailable'
    assert result['evidence_scorecard']['identities']['task_sha256'] is None
    assert result['evidence_scorecard']['identities']['source_sha256'] is None
    assert 'source_count' not in metrics(rows['robustness_coverage'])
    assert {item['source_name'] for item in result['verified_metadata']} == {'manifest.json', 'scene.json'}


@pytest.mark.parametrize('change', ['scene', 'task', 'task_scene', 'policy', 'compact_policy', 'contract_policy',
    'compact_profile', 'evaluation_hash', 'evaluation_manifest', 'missing_policy', 'missing_task', 'human_source',
    'count_bool', 'count_excess', 'denominator_zero', 'denominator_mismatch', 'unknown_controller', 'too_many_controllers',
    'status', 'unattached_counts', 'unavailable_counts', 'root_assistance', 'scale'])
def test_contradictory_compact_receipts_are_explicitly_unavailable(tmp_path, change):
    root = exercise_run(tmp_path / 'run')
    name = 'exercise-lab.json'
    contract = json.loads((root / name).read_bytes())
    counts = contract['exercise_policy']['evaluation_successes']
    if change in ('scene', 'task'):
        contract[change + '_sha256'] = '0' * 64
    elif change in ('policy', 'task_scene', 'human_source', 'root_assistance', 'scale'):
        name = 'policy.json' if change == 'policy' else 'scene.json' if change in ('human_source', 'scale') else 'exercise-task.json'
        doc = json.loads((root / name).read_bytes())
        if change in ('policy', 'task_scene'):
            doc['scene_sha256'] = '0' * 64
        elif change == 'human_source':
            doc['human_spec']['source_sha256'] = '0' * 64
        elif change == 'root_assistance':
            doc['control']['root_assistance'] = True
        else:
            doc['coordinate_frame']['scale_bridge']['body_height_to_world_scale'] = True
        put(root, name, doc)
        if change == 'policy':
            contract['policy_sha256'] = contract['exercise_policy']['policy_sha256'] = digest(root / name)
    elif change in ('compact_policy', 'contract_policy'):
        (contract['exercise_policy'] if change == 'compact_policy' else contract)['policy_sha256'] = '0' * 64
    elif change == 'compact_profile':
        contract['exercise_policy']['profile'] = 'shared-exercise-policy-v2'
    elif change in ('evaluation_hash', 'evaluation_manifest'):
        contract['policy_evaluation_sha256' if change == 'evaluation_hash' else 'policy_evaluation_manifest_sha256'] = '0' * 64
    elif change in ('missing_policy', 'missing_task'):
        (root / ('policy.json' if change == 'missing_policy' else 'exercise-task.json')).unlink()
    elif change == 'count_bool':
        counts['learned']['successes'] = True
    elif change == 'count_excess':
        counts['learned']['successes'] = 4
    elif change == 'denominator_zero':
        counts['coast']['episodes'] = 0
    elif change == 'denominator_mismatch':
        counts['coast']['episodes'] = 4
    elif change in ('unknown_controller', 'too_many_controllers'):
        counts['other'] = counts.pop('random') if change == 'unknown_controller' else {'successes': 1, 'episodes': 3}
    elif change == 'status':
        contract['exercise_policy']['evaluation_status'] = 'ready'
    elif change == 'unattached_counts':
        contract['exercise_policy']['evaluation_status'] = 'not_attached'
    elif change == 'unavailable_counts':
        contract['learned_policy_available'] = False
    put(root, 'exercise-lab.json', contract)
    seal(root)
    result = load(root)
    assert result['status'] == 'unavailable', result
    assert 'evidence_scorecard' not in result and 'verified_metadata' not in result


def test_selected_metadata_tampering_is_not_hidden_by_a_valid_compact_receipt(tmp_path):
    root = exercise_run(tmp_path / 'run')
    with (root / 'policy.json').open('ab') as stream:
        stream.write(b' ')
    result = load(root)
    assert result['status'] == 'unavailable' and 'hash does not match' in result['reason']


@pytest.mark.parametrize('limit', ['MAX_BASE_SUMMARY_BYTES', 'MAX_SUMMARY_BYTES', 'MAX_METADATA_BYTES'])
def test_base_and_combined_envelopes_stay_bounded(tmp_path, monkeypatch, limit):
    root = exercise_run(tmp_path / 'run')
    monkeypatch.setattr(lab_inspection, limit, 1)
    assert load(root)['status'] == 'unavailable'


def test_inspection_does_not_import_or_read_native_models_media_or_trial_arrays(tmp_path):
    root = exercise_run(tmp_path / 'run')
    code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'mujoco','cv2','torch','mediapipe','PIL','numpy'} or name in {'weft.lab','weft.shared_exercise_lab','weft.shared_world'}:
            raise AssertionError('Forbidden import: '+name)
sys.meta_path.insert(0,Block())
from weft import lab_inspection
original=lab_inspection._open
allowed={'manifest.json','source.json','scene.json','exercise-task.json','exercise-lab.json','policy.json','source.webm','evidence.json','reference-feasibility.json'}
def confined(fd,name):
    assert name in allowed, name
    return original(fd,name)
lab_inspection._open=confined
result=lab_inspection.load_inspection(sys.argv[1],profile='shared-world')
assert result['status']=='available',result
assert len(result['evidence_scorecard']['rows'])==14
"""
    env = {'PATH': os.defpath, 'PYTHONPATH': str(Path(lab_inspection.__file__).parents[1]), 'PYTHONDONTWRITEBYTECODE': '1'}
    completed = subprocess.run([sys.executable, '-P', '-B', '-c', code, str(root)], env=env,
                               capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
