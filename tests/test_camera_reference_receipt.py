"""Fitted recording receipts bind saved custody without importing native controllers."""

from hashlib import sha256
import json

import pytest

from weft import camera_reference_receipt as reference
from weft import recording_receipt as receipts
from test_camera_shared_receipt import camera_lab, metadata_only
from test_recording_receipt import NAME, canonical, digest, exporter, write
from test_shared_exercise_feet_receipt import save_source


@pytest.fixture
def fitted_lab(tmp_path, metadata_only):
    runtime = dict(metadata_only)
    for name in reference.REQUIRED_RUNTIME | {'__init__.py'}:
        path = receipts._PACKAGE / name
        if not path.exists(): path.write_text('# Saved receipt fixture only: ' + name)
        runtime[name] = digest(path)
    lab, run, session, scene, task, contract = camera_lab(tmp_path, runtime)
    raw = lambda value: (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    human, evidence = raw(scene['human_spec']), b'{"fixture":"original pose bytes"}'
    source = {'original_human_sha256': sha256(human).hexdigest(), 'input_evidence_sha256': sha256(evidence).hexdigest(),
        'angle_proposal_sha256': 'd' * 64, 'proposal_report_sha256': 'e' * 64}
    recipe = {'profile': 'camera-source-fit-controller-recipe-v1',
        'interpretation': 'archived_execution_only_not_source_fit_admission',
        'scene_sha256': canonical(scene), 'task_sha256': canonical(task), 'gates_sha256': task['gates_sha256'],
        **{key: task[key] for key in reference.IDENTITIES},
        'runtime_sha256': {name: runtime[name] for name in reference.ORIGINAL_RUNTIME | {'__init__.py'}},
        'input_human_sha256': source['original_human_sha256'], 'input_evidence_sha256': source['input_evidence_sha256'],
        'angle_proposal_sha256': source['angle_proposal_sha256'], 'proposal_report_sha256': source['proposal_report_sha256']}
    recipe_raw = raw(recipe); recipe_sha = sha256(recipe_raw).hexdigest()
    summary = {'profile': 'camera-reference-lab-attachment-v1', 'learned': False,
        'saved_trial_outcomes': {mode: {'steps': 3, 'success': True, 'mean_reward': reward}
                                 for mode, reward in [('authored', .8), ('reference_proposal', .7)]}}
    trial_report = {'status': 'completed', 'source_identities': source,
        'proposal_input_aliases': {'report.json': 'inputs/proposal-report.json'},
        'trials': [{'mode': mode, 'steps': value['steps'], 'metrics': {key: value[key] for key in ('success', 'mean_reward')}}
                   for mode, value in summary['saved_trial_outcomes'].items()]}
    names = {'plan.json', 'report.json', 'inputs/proposal-manifest.json', 'inputs/scene-manifest.json',
        'inputs/exercise-task.json', 'run/exercise-task.json', 'inputs/scene.json', 'run/scene.json', 'run/reference-controller.json',
        'proofs/authored-action-replay.json', 'proofs/reference_proposal-action-replay.json',
        'recordings/authored/manifest.json', 'recordings/reference_proposal/manifest.json', 'inputs/proposal-report.json'}
    trial_files = {name: b'{}' for name in names}
    trial_files['report.json'] = raw(trial_report)
    trial_files['run/reference-controller.json'] = recipe_raw
    trial_hashes = {name: sha256(value).hexdigest() for name, value in trial_files.items()}
    trial_manifest = {'profile': 'shared-exercise-reference-trial-v1', 'status': 'completed', 'files': trial_hashes}
    trial_raw = raw(trial_manifest)
    identity = {'reference_controller_sha256': recipe_sha, 'reference_trial_manifest_sha256': sha256(trial_raw).hexdigest(),
                'reference_proposal_manifest_sha256': trial_hashes['inputs/proposal-manifest.json']}
    custody = {'profile': summary['profile'], **identity, 'scene_sha256': canonical(scene), 'task_sha256': canonical(task),
        'summary': summary, 'source_identity': source, 'selected_trial_members': trial_hashes}
    contract.update(identity, runtime_sha256=runtime, exercise_reference=summary)
    reference_envelope = {'document_utf8': recipe_raw.decode(), 'sha256': recipe_sha,
                          'implementation_sha256': recipe['runtime_sha256']}
    save_source(run, session, scene, task, contract, envelope_profile=reference.PROFILE,
                extra={'reference_controller': reference_envelope})
    files = {'reference-controller.json': recipe_raw, 'reference-attachment.json': raw(custody),
        'source-run/human.json': human, 'source-run/evidence.json': evidence,
        'reference-trial/manifest.json': trial_raw, **{'reference-trial/' + name: value for name, value in trial_files.items()}}
    for name, value in files.items():
        path = run / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
    members = {str(path.relative_to(run)): digest(path) for path in run.rglob('*') if path.is_file() and path != run / 'manifest.json'}
    write(run / 'manifest.json', {'schema_version': 1, 'status': 'ready', 'files': members})
    return lab, run, session, task, recipe_sha


def export_reference(value, *, change=None, replay=True):
    _, run, session, task, pin = value
    def transform(result):
        if not replay:
            result.pop('action_replay')
        else:
            result['action_replay'].update(protocol=reference.PROTOCOL, reference_controller_sha256=pin,
                reference_actions_verified=2, policy_sha256=None, learned_actions_verified=None,
                **{key: task[key] for key in reference.IDENTITIES})
        if change: change(result)
    perform, calls = exporter(shared=True, transform=transform)
    return receipts.export_recording_with_receipt(session, export=perform, run_directory=run,
        expected_run_manifest_sha256=digest(run / 'manifest.json')), calls


def test_reference_receipt_counts_fitted_not_learned_and_saved_inspection_is_metadata_only(fitted_lab, monkeypatch):
    result, calls = export_reference(fitted_lab)
    assert result['recording_evidence']['status'] == 'available', result['recording_evidence']
    lab, _, session, task, recipe_sha = fitted_lab
    observed = []
    read = receipts._Budget.read
    def saved(self, root, name, *args, **kwargs):
        observed.append(name)
        assert not name.endswith('.py') and name not in {'video.mp4', 'preview.gif', 'replay.html'}
        return read(self, root, name, *args, **kwargs)
    monkeypatch.setattr(receipts._Budget, 'read', saved)
    view = receipts.load_recording_evidence(lab, NAME)
    assert view['status'] == 'available', view
    assert view['identities']['reference_controller_sha256'] == recipe_sha
    assert all(view['identities'][key] == task[key] for key in reference.IDENTITIES)
    assert view['replay']['protocol'] == reference.PROTOCOL
    assert view['replay']['reference_actions_verified'] == 2 and view['replay']['learned_actions_verified'] is None
    assert observed.count('events.jsonl') == 1 and len(calls) == 1


@pytest.mark.parametrize('replay', [False, True])
def test_reference_receipt_does_not_infer_replay_from_a_card(fitted_lab, replay):
    result, _ = export_reference(fitted_lab, replay=replay)
    assert result['recording_evidence']['status'] == 'available', result['recording_evidence']
    view = receipts.load_recording_evidence(fitted_lab[0], NAME)
    assert view['replay']['status'] == ('passed' if replay else 'not_performed')
    assert view['replay']['reference_actions_verified'] == (2 if replay else None)


def test_attached_recipe_with_manual_authored_only_retains_zero_fitted_count(fitted_lab):
    lab, _, session, _, recipe_sha = fitted_lab
    rows = [{'type': 'start'}, *({'type': 'step', 'controller': mode}
            for mode in ('manual', 'authored', 'manual')), {'type': 'reset'}]
    journal = b''.join((json.dumps(row) + '\n').encode() for row in rows)
    (session / 'events.jsonl').write_bytes(journal)
    metadata = json.loads((session / 'recording.json').read_bytes())
    metadata['journal_bytes'] = len(journal)
    write(session / 'recording.json', metadata)
    result, _ = export_reference(fitted_lab,
        change=lambda result: result['action_replay'].update(reference_actions_verified=0))
    assert result['recording_evidence']['status'] == 'available', result['recording_evidence']
    view = receipts.load_recording_evidence(lab, NAME)
    assert view['identities']['reference_controller_sha256'] == recipe_sha
    assert view['replay']['reference_actions_verified'] == 0 and view['replay']['learned_actions_verified'] is None
    assert view['recorded'] == {'events': 5, 'actions': 3, 'resets': 1}


@pytest.mark.parametrize('fault', ['recipe', 'protocol', 'count', 'learning'])
def test_replay_identity_or_mode_mismatch_never_seals_success(fitted_lab, fault):
    def change(result):
        proof = result['action_replay']
        if fault == 'recipe': proof['reference_controller_sha256'] = 'f' * 64
        elif fault == 'protocol': proof['protocol'] = 'camera-shared-full-state-replay-v1'
        elif fault == 'count': proof['reference_actions_verified'] = 4
        else: proof['learned_actions_verified'] = 2
    result, _ = export_reference(fitted_lab, change=change)
    assert result['recording_evidence']['status'] == 'unavailable'
    assert not (fitted_lab[2] / 'evidence').exists()


def test_removed_selected_proof_cannot_be_hidden_by_resealing_outer_metadata(fitted_lab):
    _, run, session, _, _ = fitted_lab
    path = run / 'reference-attachment.json'; custody = json.loads(path.read_bytes())
    custody['selected_trial_members'].pop('proofs/reference_proposal-action-replay.json')
    write(path, custody)
    manifest = json.loads((run / 'manifest.json').read_bytes())
    manifest['files']['reference-attachment.json'] = digest(path)
    manifest['files'].pop('reference-trial/proofs/reference_proposal-action-replay.json')
    write(run / 'manifest.json', manifest)
    result, _ = export_reference(fitted_lab)
    assert result['recording_evidence']['status'] == 'unavailable'
    assert not (session / 'evidence').exists()
