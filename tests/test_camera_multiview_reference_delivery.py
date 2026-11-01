"""Analytical mixed recording and selected attachment custody; no native admission."""

from copy import deepcopy
from hashlib import sha256
import json
import time

import pytest

import test_camera_reference_lab as old_lab
import test_camera_reference_recording as old_recording
from test_camera_reference_controller import api, runtime
from test_camera_multiview_reference_controller import multiview_fixture
from test_camera_multiview_reference_inputs import multiview_packages
from test_camera_shared_lab import no_execution
from test_camera_shared_task import analytical_only
from test_whole_body_reference_report import encode
from weft import camera_reference_lab as attachment
from weft import camera_reference_receipt as receipt
from weft import recording_receipt
from weft.io import canonical_hash


@pytest.fixture
def multiview_build(api, runtime, analytical_only, monkeypatch):
    def data(actor, pinned, *, root, corrections):
        return multiview_fixture(actor, pinned, roots=root)
    monkeypatch.setattr(old_recording, 'fixture', data)
    prepared = old_recording.build.__wrapped__(api, runtime, analytical_only, monkeypatch)
    yield next(prepared)
    with pytest.raises(StopIteration): next(prepared)


@pytest.mark.parametrize('roots', [False, True])
def test_multiview_mixed_journal_reset_and_exact_analytical_replay(multiview_build, tmp_path, roots):
    env, scene, rows, raw = old_recording.sequence(multiview_build, roots)
    assert json.loads(raw)['profile'] == 'camera-multiview-source-fit-controller-recipe-v1'
    root = old_recording.save(tmp_path, scene, rows)
    saved = old_recording.recording.load_recording(root)
    assert saved['scene'] == scene and saved['events'] == rows
    proof = old_recording.adapter.verify_action_replay(root)
    assert proof['exact_state_replay'] is True
    assert (proof['steps'], proof['events'], proof['reference_actions_verified']) == (5, 7, 2)
    assert proof['reference_controller_sha256'] == sha256(raw).hexdigest()
    assert proof['learned_actions_verified'] is proof['policy_sha256'] is None
    assert [row.get('controller') for row in rows] == [None, 'manual', 'reference_proposal', None,
                                                     'authored', 'reference_proposal', 'coast']
    assert old_recording.adapter.validate_recording_scene(scene) == scene
    assert scene['engine_scene']['learned_policy_available'] is False
    assert env.task['scope']['learning'] == 'unqualified'


@pytest.fixture
def multiview_trial(tmp_path, monkeypatch):
    def packages(directory):
        proposal, source, task, *_ = multiview_packages(directory, monkeypatch)
        return proposal, source, task
    monkeypatch.setattr(old_lab, 'packages', packages)
    value = old_lab.saved_trial.__wrapped__(tmp_path, monkeypatch)
    files = value.files
    report, plan = (json.loads(files[name]) for name in ('report.json', 'plan.json'))
    source = report['source_identities']
    recipe = json.loads(files['run/reference-controller.json'])
    recipe.update(profile='camera-multiview-source-fit-controller-recipe-v1',
        **{key: source[key] for key in ('additional_view_sha256', 'multiview_qualification_sha256')})
    files['run/reference-controller.json'] = encode(recipe)
    pin = sha256(files['run/reference-controller.json']).hexdigest()
    report['reference_controller']['sha256'] = plan['reference_controller_sha256'] = pin
    report['controller_family'] = plan['controller_family'] = 'camera_multiview_angle_overlay'
    mode = next(row for row in report['trials'] if row['mode'] == 'reference_proposal')
    mode['action_replay']['reference_controller_sha256'] = pin
    path = mode['action_replay']['path']
    proof = json.loads(files[path]); proof['reference_controller_sha256'] = pin
    files[path] = encode(proof)
    plan['preserved_files']['run/reference-controller.json'] = pin
    files.update({'plan.json': encode(plan), 'report.json': encode(report)})
    for name in ('plan.json', 'report.json', 'run/reference-controller.json', path):
        (value.directory / name).write_bytes(files[name])
        value.manifest['files'][name] = sha256(files[name]).hexdigest()
    old_lab.seal(value.directory, value.manifest)
    return value


def test_multiview_attachment_and_resume_preserve_all_secondary_aliases(multiview_trial):
    value = multiview_trial
    result = old_lab.prepare(value)
    assert result['recipe_document'] == value.files['run/reference-controller.json']
    source = result['custody']['source_identity']
    assert source['secondary_source_sha256'] == 'f' * 64
    assert result['summary']['learned'] is False
    assert not any(name.endswith(('.mp4', '.gif', 'events.jsonl')) for name in result['files'])
    contract = {'learned_policy_available': False, **result['identity'], 'exercise_reference': result['summary'],
                'runtime_sha256': result['runtime_sha256']}
    loaded = attachment.load_reference_attachment(result['files'], value.scene, value.task,
        value.source_manifest, value.source_files, value.task_raw, contract, deadline=time.monotonic() + 5)
    assert loaded['custody'] == result['custody'] and loaded['recipe_document'] == result['recipe_document']


def test_multiview_attachment_cannot_relabel_legacy_recipe(multiview_trial):
    value = multiview_trial
    name = 'run/reference-controller.json'
    recipe = json.loads(value.files[name]); recipe['profile'] = 'camera-source-fit-controller-recipe-v1'
    value.files[name] = encode(recipe)
    (value.directory / name).write_bytes(value.files[name])
    value.manifest['files'][name] = sha256(value.files[name]).hexdigest()
    old_lab.seal(value.directory, value.manifest)
    with pytest.raises(ValueError): old_lab.prepare(value)


def receipt_arguments(monkeypatch):
    monkeypatch.setattr(receipt, 'saved_camera_bindings', lambda *args: {})
    scene, task = {'scope': 'Custody fixture'}, {'gates_sha256': 'a' * 64, **dict.fromkeys(receipt.IDENTITIES, 'b' * 64)}
    runtime = dict.fromkeys(receipt.REQUIRED_RUNTIME | receipt.ORIGINAL_RUNTIME, 'c' * 64)
    source = {'original_human_sha256': 'd' * 64, 'input_evidence_sha256': 'e' * 64,
        'angle_proposal_sha256': 'f' * 64, 'proposal_report_sha256': '1' * 64,
        'secondary_source_sha256': '2' * 64, 'secondary_camera_sha256': '3' * 64,
        'secondary_evidence_sha256': '4' * 64, 'secondary_view_config_sha256': '5' * 64,
        'secondary_world_alignment_sha256': '9' * 64,
        'secondary_observation_sha256s': ['6' * 64]}
    binding = {'profile': 'camera-multiview-evidence-extension-v1', 'views': [{}, {'source_sha256': source['secondary_source_sha256']}],
        'secondary_input_bindings': {'camera_raw_sha256': source['secondary_camera_sha256'],
            'evidence_raw_sha256': source['secondary_evidence_sha256'], 'observation_document_sha256s': ['6' * 64],
            'alignment_raw_sha256': source['secondary_world_alignment_sha256']}}
    extension = {'binding': binding, 'binding_canonical_sha256': canonical_hash(binding)}
    audit = {'binding_canonical_sha256': extension['binding_canonical_sha256']}
    source.update(additional_view_sha256=canonical_hash(extension), multiview_qualification_sha256=canonical_hash(audit))
    recipe = {'profile': 'camera-multiview-source-fit-controller-recipe-v1',
        'interpretation': 'archived_execution_only_not_source_fit_admission', 'scene_sha256': canonical_hash(scene),
        'task_sha256': canonical_hash(task), 'gates_sha256': task['gates_sha256'], **{key: task[key] for key in receipt.IDENTITIES},
        'input_human_sha256': source['original_human_sha256'], 'input_evidence_sha256': source['input_evidence_sha256'],
        'angle_proposal_sha256': source['angle_proposal_sha256'], 'proposal_report_sha256': source['proposal_report_sha256'],
        'runtime_sha256': runtime, 'angle_proposal': {'profile': 'camera-body-multiview-angle-proposal-v1', 'additional_view': extension},
        'multiview_images': audit, **{key: source[key] for key in ('additional_view_sha256', 'multiview_qualification_sha256')}}
    raw = encode(recipe); pin = sha256(raw).hexdigest()
    envelope = {'schema_version': 1, 'profile': receipt.PROFILE, 'engine_scene': scene, 'exercise_task': task,
        'dt': .03, 'objects': [], 'reference_controller': {'document_utf8': raw.decode(), 'sha256': pin, 'implementation_sha256': runtime}}
    aliases = {name: 'inputs/proposal/' + name for name in ('inputs/secondary-camera.json', 'inputs/secondary-evidence.json',
        'inputs/camera-secondary-config.json', 'inputs/camera-secondary-alignment.json', 'inputs/camera-secondary-observation-00.json')}
    selected = dict.fromkeys({'plan.json', 'report.json', 'inputs/proposal-manifest.json', 'inputs/scene-manifest.json',
        'inputs/exercise-task.json', 'run/exercise-task.json', 'inputs/scene.json', 'run/scene.json', 'run/reference-controller.json',
        'proofs/authored-action-replay.json', 'proofs/reference_proposal-action-replay.json',
        'recordings/authored/manifest.json', 'recordings/reference_proposal/manifest.json', *aliases.values()}, '7' * 64)
    for name, key in [('inputs/secondary-camera.json', 'secondary_camera_sha256'),
                      ('inputs/secondary-evidence.json', 'secondary_evidence_sha256'),
                      ('inputs/camera-secondary-config.json', 'secondary_view_config_sha256'),
                      ('inputs/camera-secondary-alignment.json', 'secondary_world_alignment_sha256')]: selected[aliases[name]] = source[key]
    selected[aliases['inputs/camera-secondary-observation-00.json']] = '6' * 64
    trials = [{'mode': mode, 'steps': 3, 'metrics': {'success': True, 'mean_reward': .5}}
              for mode in ('authored', 'reference_proposal')]
    summary = {'learned': False, 'saved_trial_outcomes': {row['mode']: {'steps': 3, **row['metrics']} for row in trials}}
    contract = {'reference_controller_sha256': pin, 'reference_trial_manifest_sha256': '8' * 64,
        'reference_proposal_manifest_sha256': '7' * 64, 'exercise_reference': summary}
    custody = {'profile': 'camera-reference-lab-attachment-v1', 'scene_sha256': canonical_hash(scene),
        'task_sha256': canonical_hash(task), **{key: contract[key] for key in contract if key != 'exercise_reference'},
        'summary': summary, 'source_identity': source, 'selected_trial_members': selected}
    members = {'reference-controller.json': pin, 'reference-trial/manifest.json': '8' * 64,
        **{'reference-trial/' + name: value for name, value in selected.items()},
        'source-run/human.json': source['original_human_sha256'], 'source-run/evidence.json': source['input_evidence_sha256'],
        'source-run/inputs/two-view-root/camera.json': source['secondary_camera_sha256'],
        'source-run/inputs/two-view-root/evidence.json': source['secondary_evidence_sha256']}
    return (scene, task, envelope, contract, runtime, members), {'recipe': recipe, 'recipe_file_sha256': pin,
        'custody': custody, 'trial_manifest': {'profile': 'shared-exercise-reference-trial-v1', 'status': 'completed', 'files': selected},
        'trial_report': {'status': 'completed', 'controller_family': 'camera_multiview_angle_overlay',
                         'source_identities': source, 'proposal_input_aliases': aliases, 'trials': trials}}


@pytest.mark.parametrize('mutation', [None, 'secondary', 'observation', 'qualification', 'missing_alignment', 'family'])
def test_saved_multiview_receipt_joins_original_and_retained_bytes(monkeypatch, mutation):
    args, kwargs = receipt_arguments(monkeypatch)
    if mutation == 'secondary': args[-1]['source-run/inputs/two-view-root/camera.json'] = '0' * 64
    elif mutation == 'observation':
        alias = kwargs['trial_report']['proposal_input_aliases']['inputs/camera-secondary-observation-00.json']
        args[-1]['reference-trial/' + alias] = '0' * 64
    elif mutation == 'qualification': kwargs['custody']['source_identity']['multiview_qualification_sha256'] = '0' * 64
    elif mutation == 'family': kwargs['trial_report']['controller_family'] = 'camera_angle_overlay'
    elif mutation == 'missing_alignment':
        alias = kwargs['trial_report']['proposal_input_aliases'].pop('inputs/camera-secondary-alignment.json')
        kwargs['custody']['selected_trial_members'].pop(alias)
        args[-1].pop('reference-trial/' + alias)
    if mutation:
        with pytest.raises(ValueError): receipt.saved_reference_bindings(*args, **kwargs)
    else:
        result = receipt.saved_reference_bindings(*args, **kwargs)
        assert result == {'reference_controller_sha256': kwargs['recipe_file_sha256']}
