"""Fitted lab attachment and controls use saved analytical inputs, never physics."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from weft import camera_reference_lab as attachment
from weft import camera_reference_recording as recording
from weft import shared_exercise_lab as shared
from weft import shared_exercise_reference_trial as trial
from weft.io import canonical_hash
from test_camera_reference_inputs import packages
from test_camera_shared_lab import no_execution, inputs as mock_lab_inputs
from test_whole_body_reference_report import encode


def seal(directory, value):
    raw = encode(value)
    (directory / 'manifest.json').write_bytes(raw)
    return raw


@pytest.fixture
def saved_trial(tmp_path, monkeypatch):
    from weft.camera_reference_inputs import load_camera_reference_inputs

    proposal, source, task_path, *_ = packages(tmp_path)
    task = json.loads(task_path.read_bytes())
    task['control'] = {'root_assistance': 'disabled', 'object_forces': 'forbidden'}
    task_path.write_bytes(encode(task))
    source_manifest = json.loads((source / 'manifest.json').read_bytes())
    source_manifest['files']['exercise-task.json'] = sha256(task_path.read_bytes()).hexdigest()
    source_manifest_raw = seal(source, source_manifest)
    source_files = {name: (source / name).read_bytes() for name in source_manifest['files'] if (source / name).is_file()}
    inputs = load_camera_reference_inputs(proposal, source, task_path)
    scene = json.loads(inputs.scene_bytes)
    runtime = {'camera_reference_controller.py': sha256(b'analytical-runtime').hexdigest()}
    versions = {'python': 'analytical', 'executable': 'none', 'packages': {}}
    monkeypatch.setattr(trial, '_runtime_versions', lambda: deepcopy(versions))
    calls = []
    def prepare(raw, actual, original_task):
        calls.append(raw)
        assert actual == scene and original_task == task
        assert json.loads(raw)['runtime_sha256'] == runtime
        return {'document_utf8': raw.decode(), 'sha256': sha256(raw).hexdigest(),
                'implementation_sha256': runtime}, None
    monkeypatch.setattr(recording, '_prepare', prepare)
    monkeypatch.setattr(attachment, 'execution_runtime', lambda value, **kwargs: dict(value))
    identities = inputs.identities
    recipe = {'profile': 'camera-source-fit-controller-recipe-v1', 'runtime_sha256': runtime,
        'input_human_sha256': identities['original_human_sha256'],
        'input_evidence_sha256': identities['input_evidence_sha256'],
        'angle_proposal_sha256': identities['angle_proposal_sha256'],
        'proposal_report_sha256': identities['proposal_report_sha256']}
    recipe_raw = encode(recipe)
    recipe_sha = sha256(recipe_raw).hexdigest()
    aliases = {name: attachment._ALIASES.get(name, 'inputs/proposal/' + name) for name in inputs.proposal_members}
    files = {aliases[name]: raw for name, raw in inputs.proposal_members.items()}
    files.update({'inputs/proposal-manifest.json': inputs.proposal_manifest_bytes,
        'inputs/scene-manifest.json': source_manifest_raw, 'inputs/scene.json': inputs.scene_bytes,
        'run/scene.json': inputs.scene_bytes, 'inputs/exercise-task.json': inputs.task_bytes,
        'run/exercise-task.json': inputs.task_bytes, 'run/reference-controller.json': recipe_raw,
        'runtime/weft/camera_reference_controller.py': b'analytical-runtime'})
    plan = {'profile': attachment.TRIAL_PROFILE + '-plan', 'status': 'frozen', 'controller_family': 'camera_angle_overlay',
        'modes': ['authored', 'reference_proposal'], 'seed': 1729, 'reset_options': dict(shared.RESET_OPTIONS),
        'source_identities': identities, 'runtime_sha256': runtime, 'runtime_versions': versions,
        'reference_controller_sha256': recipe_sha, 'preserved_files': {name: sha256(raw).hexdigest() for name, raw in files.items()}}
    report = {'profile': attachment.TRIAL_PROFILE, 'status': 'completed', 'controller_family': 'camera_angle_overlay',
        'seed': 1729, 'reset_options': dict(shared.RESET_OPTIONS), 'error': None, 'same_initial_state': True,
        'source_identities': identities, 'proposal_input_aliases': aliases,
        'reference_controller': {'document': 'run/reference-controller.json', 'sha256': recipe_sha}, 'trials': []}
    for mode in plan['modes']:
        prefix = 'recordings/' + mode + '/'
        recording_files = {name: sha256(('not read ' + name).encode()).hexdigest() for name in (
            'scene.json', 'recording.json', 'events.jsonl', 'video.mp4', 'preview.gif', 'replay.html')}
        files[prefix + 'manifest.json'] = encode({'schema_version': 1, 'files': recording_files})
        replay = {'protocol': 'camera-shared-reference-full-state-replay-v1' if mode == 'reference_proposal'
                  else 'camera-shared-full-state-replay-v1', 'exact_state_replay': True, 'steps': 4, 'events': 5,
            'scene_sha256': canonical_hash(scene), 'task_sha256': canonical_hash(task),
            **{key: task[key] for key in recording.IDENTITIES}, 'policy_sha256': None, 'learned_actions_verified': None}
        if mode == 'reference_proposal':
            replay.update(reference_controller_sha256=recipe_sha, reference_actions_verified=4)
        path = 'proofs/' + mode + '-action-replay.json'
        files[path] = encode(replay)
        report['trials'].append({'mode': mode, 'status': 'verified', 'steps': 4, 'initial_state_sha256': 'a' * 64,
            'recording': {'directory': prefix[:-1], 'manifest_sha256': sha256(files[prefix + 'manifest.json']).hexdigest()},
            'action_replay': {**replay, 'path': path, 'status': 'passed'},
            'metrics': {'success': True, 'mean_reward': .7 if mode == 'reference_proposal' else .8}})
    files.update({'plan.json': encode(plan), 'report.json': encode(report)})
    directory = tmp_path / 'trial'; directory.mkdir()
    for name, raw in files.items():
        path = directory / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    hashes = {name: sha256(raw).hexdigest() for name, raw in files.items()}
    for mode in plan['modes']:
        prefix = 'recordings/' + mode + '/'
        hashes.update({prefix + name: pin for name, pin in json.loads(files[prefix + 'manifest.json'])['files'].items()})
    manifest = {'schema_version': 1, 'profile': attachment.TRIAL_PROFILE, 'status': 'completed',
        'scene_sha256': canonical_hash(scene), 'task_sha256': canonical_hash(task), 'files': hashes}
    seal(directory, manifest)
    return SimpleNamespace(directory=directory, files=files, manifest=manifest, scene=scene, task=task,
        source_manifest=source_manifest_raw, source_files=source_files, task_raw=inputs.task_bytes, calls=calls)


def prepare(value):
    return attachment.prepare_reference_attachment(value.directory, value.scene, value.task,
        value.source_manifest, value.source_files, value.task_raw, deadline=time.monotonic() + 5)


def test_selected_attachment_resume_retains_original_recipe_and_worse_result(saved_trial):
    value = saved_trial
    result = prepare(value)
    assert result['recipe_document'] == value.files['run/reference-controller.json']
    assert result['summary']['learned'] is False
    outcomes = result['summary']['saved_trial_outcomes']
    assert outcomes['reference_proposal']['mean_reward'] < outcomes['authored']['mean_reward']
    assert not any(name.endswith(('.mp4', '.gif', 'events.jsonl')) for name in result['files'])
    assert 'omitted trial media/journals' in result['custody']['scope']
    contract = {'learned_policy_available': False, **result['identity'], 'exercise_reference': result['summary'],
                'runtime_sha256': result['runtime_sha256']}
    loaded = attachment.load_reference_attachment(result['files'], value.scene, value.task, value.source_manifest,
        value.source_files, value.task_raw, contract, deadline=time.monotonic() + 5)
    assert loaded['recipe_document'] == result['recipe_document'] and loaded['custody'] == result['custody']


@pytest.mark.parametrize('change', ['missing_alias', 'changed_source', 'changed_runtime', 'unmatched_start', 'missing_recording'])
def test_attachment_rejects_inconsistent_proofs(saved_trial, change):
    value = saved_trial
    if change == 'changed_source':
        value.source_files['human.json'] += b' '
    elif change == 'changed_runtime':
        plan = json.loads(value.files['plan.json']); plan['runtime_versions']['python'] = 'changed'
        value.files['plan.json'] = encode(plan)
    elif change == 'missing_recording':
        del value.manifest['files']['recordings/reference_proposal/events.jsonl']
    else:
        report = json.loads(value.files['report.json'])
        if change == 'missing_alias': report['proposal_input_aliases'].pop('inputs/evidence-original.json')
        else: report['trials'][1]['initial_state_sha256'] = 'b' * 64
        value.files['report.json'] = encode(report)
    for name, raw in value.files.items():
        (value.directory / name).write_bytes(raw)
        value.manifest['files'][name] = sha256(raw).hexdigest()
    seal(value.directory, value.manifest)
    with pytest.raises(ValueError): prepare(value)


def test_expired_attachment_and_conflicting_modes_fail_before_reads(saved_trial, monkeypatch):
    monkeypatch.setattr(attachment, 'read_package_file', lambda *a, **k: pytest.fail('expired read'))
    with pytest.raises(ValueError, match='deadline'):
        attachment.prepare_reference_attachment(saved_trial.directory, {}, {}, b'', {}, b'', deadline=0.)
    for options in ({'root_assistance': True}, {'policy_run': 'policy'}, {'evaluation_run': 'eval', 'policy_run': 'policy'}):
        with pytest.raises(ValueError, match='cannot mix'):
            shared.prepare_shared_exercise_lab('unused', 'unused-out', reference_trial='trial', **options)


@pytest.mark.parametrize('fault', ['missing_task', 'stale_attachment'])
def test_fresh_attachment_rejects_before_native_admission(mock_lab_inputs, tmp_path, monkeypatch, fault):
    options = {}
    if fault != 'missing_task':
        path = tmp_path / 'task.json'; path.write_bytes(encode(mock_lab_inputs.task))
        options['task_path'] = path
    def reject(*args, **kwargs):
        raise ValueError('stale fitted metadata')
    monkeypatch.setattr(attachment, 'prepare_reference_attachment', reject)
    monkeypatch.setattr(shared, 'validate_shared_world', lambda *args: pytest.fail('native admission before metadata'))
    monkeypatch.setattr(shared, 'collect_shared_source_files', lambda *a, **k: pytest.fail('source admission before metadata'))
    with pytest.raises(ValueError, match='exact task|stale fitted metadata'):
        shared.prepare_shared_exercise_lab(mock_lab_inputs.source, tmp_path / 'out', reference_trial=tmp_path / 'trial', **options)
    assert not (tmp_path / 'out').exists()


def test_stale_recipe_rejects_before_large_proposal_read(saved_trial, monkeypatch):
    from weft import camera_reference_inputs
    def reject(*args, **kwargs):
        raise ValueError('stale runtime')
    monkeypatch.setattr(recording, '_prepare', reject)
    monkeypatch.setattr(camera_reference_inputs, '_load_camera', lambda *a, **k: pytest.fail('proposal loaded before runtime'))
    with pytest.raises(ValueError, match='stale runtime'):
        prepare(saved_trial)


def test_ordinary_lab_keeps_validated_numeric_identity(mock_lab_inputs, tmp_path, monkeypatch):
    fixture = mock_lab_inputs
    fixture.scene['gravity'] = [0, 0, -9.81]
    raw_scene = encode(fixture.scene)
    (fixture.source / 'scene.json').write_bytes(raw_scene)
    manifest = json.loads((fixture.source / 'manifest.json').read_bytes())
    manifest['files']['scene.json'] = sha256(raw_scene).hexdigest()
    seal(fixture.source, manifest)
    def normalize(scene):
        result = deepcopy(scene)
        result['gravity'] = [float(value) for value in result['gravity']]
        return result
    normalized = normalize(fixture.scene)
    fixture.task['scene_sha256'] = canonical_hash(normalized)
    assert canonical_hash(fixture.scene) != canonical_hash(normalized)
    monkeypatch.setattr(shared, 'validate_shared_world', normalize)
    out = tmp_path / 'ordinary-lab'
    shared.prepare_shared_exercise_lab(fixture.source, out)
    loaded = shared.load_shared_exercise_lab(out / 'run')
    assert loaded['contract']['scene_sha256'] == canonical_hash(normalized)
    assert (out / 'run/scene.json').read_bytes() == raw_scene
    assert 'reference_document' not in loaded


def test_cached_fitted_manual_authored_and_recording_boundaries(monkeypatch):
    from weft.lab import Lab
    from weft.lab_console import Controls

    session = shared.SharedExerciseSession.__new__(shared.SharedExerciseSession)
    session.scene_profile, session.policy, session.done = 'shared-world-v6', None, False
    session.reference_controller_sha256 = 'a' * 64
    session._fitted_sample = None
    calls = []
    session._fitted_action = lambda: (calls.append('fitted') or np.full(19, .02))
    session.env = SimpleNamespace(action_count=19, authored_action=lambda: np.full(19, .01))
    session.step = lambda action: (.25, False, False, {})
    session.snapshot = lambda: {'analytical': True}
    session.resolve_action = list
    app, events = Lab.__new__(Lab), []
    app.load_session = lambda: session
    app.recorder = SimpleNamespace(check_capacity=lambda: None, append=events.append)
    caps = {'lab_profile': 'shared_exercise', 'action_contract': {'space_type': 'continuous_box', 'dimension': 19, 'actuator_names': ['joint'] * 19},
            'controllers': {'learned_policy_available': False, 'authored_reference_controller': True,
                            'fitted_reference_controller': True}}
    controls = Controls(caps)
    for mode in ('manual', 'reference', 'fitted', 'coast'):
        controls.select_mode(mode); app.control('/api/step', controls.step(1))
    assert [event['controller'] for event in events] == ['manual', 'authored', 'reference_proposal', 'coast']
    assert events[2]['reference_controller_sha256'] == session.reference_controller_sha256
    assert all('reference_controller_sha256' not in row for row in (events[0], events[1], events[3]))
    assert calls == ['fitted']
    with pytest.raises(ValueError, match='exactly one'):
        app.control('/api/step', {'controller': 'reference_proposal', 'action': [0.] * 19})
    with pytest.raises(ValueError, match='cannot change'):
        app.control('/api/step', {'action': [0.] * 19, 'reference_controller_sha256': 'a' * 64})
    with pytest.raises(ValueError, match='no available learned'):
        controls.select_mode('learned')
    session.reference_controller_sha256 = None
    with pytest.raises(ValueError, match='no attached fitted'):
        app.control('/api/step', {'controller': 'reference_proposal'})


def test_new_lab_and_resume_keep_reference_separate_from_policy(mock_lab_inputs, tmp_path, monkeypatch):
    fixture = mock_lab_inputs
    task_path = tmp_path / 'exact-task.json'; task_path.write_bytes(encode(fixture.task))
    package = Path(shared.__file__).parent
    runtime = {name: sha256((package / name).read_bytes()).hexdigest() for name in shared.runtime_files(fixture.scene)}
    recipe = encode({'runtime_sha256': runtime})
    identity = {'reference_controller_sha256': sha256(recipe).hexdigest(),
                'reference_trial_manifest_sha256': sha256(b'{}').hexdigest(), 'reference_proposal_manifest_sha256': 'b' * 64}
    summary = {'profile': attachment.PROFILE, 'scope': attachment.SCOPE, 'learned': False}
    files = {'reference-controller.json': recipe, 'reference-attachment.json': b'{}', 'reference-trial/manifest.json': b'{}'}
    result = {'recipe_document': recipe, 'identity': identity, 'summary': summary, 'files': files, 'runtime_sha256': runtime}
    calls, deadlines, source_deadlines = [], [], []
    def prepare_attachment(directory, scene, task, source_manifest, source_files, task_raw, **kwargs):
        calls.append('prepare')
        deadlines.append(kwargs['deadline'])
        assert scene == fixture.scene and task == fixture.task and task_raw == task_path.read_bytes()
        return deepcopy(result)
    def load_attachment(saved, scene, task, source_manifest, source_files, task_raw, contract, **kwargs):
        calls.append('resume')
        deadlines.append(kwargs['deadline'])
        assert all(saved.get(name) == raw for name, raw in files.items())
        assert contract['exercise_reference'] == summary and not contract['learned_policy_available']
        return deepcopy(result)
    monkeypatch.setattr(attachment, 'prepare_reference_attachment', prepare_attachment)
    monkeypatch.setattr(attachment, 'load_reference_attachment', load_attachment)
    monkeypatch.setattr(attachment, 'execution_runtime', lambda value, **kwargs: dict(value))
    monkeypatch.setattr(recording, '_runtime_hashes', lambda value: dict(value))
    collect = shared.collect_shared_source_files
    def bounded_collect(*args, **kwargs):
        source_deadlines.append(kwargs['deadline'])
        assert kwargs['deadline'] == deadlines[-1]
        return collect(*args, **kwargs)
    monkeypatch.setattr(shared, 'collect_shared_source_files', bounded_collect)
    out = tmp_path / 'new-lab'
    prepared = shared.prepare_shared_exercise_lab(fixture.source, out, task_path=task_path, reference_trial=tmp_path / 'trial')
    assert prepared['learned_policy_available'] is False and prepared['reference_controller_sha256'] == identity['reference_controller_sha256']
    config = json.loads((out / 'lab-config.json').read_bytes())
    loaded = shared.load_shared_exercise_lab(out / 'run', config=config)
    assert loaded['reference_document'] == recipe and 'policy' not in loaded
    assert calls == ['prepare', 'resume', 'resume']
    assert len(source_deadlines) == 3 and source_deadlines[:2] == [deadlines[0]] * 2
    config['reference_controller_sha256'] = 'c' * 64
    with pytest.raises(ValueError, match='fitted controller identity'):
        shared.load_shared_exercise_lab(out / 'run', config=config)
    assert not (out / 'run' / 'policy.json').exists()
    def reject(*args, **kwargs):
        raise ValueError('retained fitted proof is invalid')
    monkeypatch.setattr(attachment, 'load_reference_attachment', reject)
    monkeypatch.setattr(shared, 'validate_shared_world', lambda *a: pytest.fail('native admission before retained proof'))
    with pytest.raises(ValueError, match='retained fitted proof'):
        shared.load_shared_exercise_lab(out / 'run')


def test_session_prepares_fitted_action_once_and_closes_on_preparation_failure(monkeypatch, record_property):
    from weft import camera_shared_task, capabilities

    class Engine:
        action_count = 19
        def __init__(self, scene, task):
            self.scene, self.task = scene, task
        def reset(self, **kwargs): return [], {}
        def snapshot(self): return {'test': 'no physics'}
        def close(self): closed.append(True)
        def action_contract(self): return {'dimension': 19}
    closed, preparations, decisions = [], [], []
    loaded = {'scene': {'profile': 'shared-world-v6'}, 'task': {'fixture': True}, 'artifacts': [],
        'source_verification': {}, 'reference_document': b'fixed-recipe',
        'contract': {'exercise_reference': {'scope': attachment.SCOPE}, 'reference_controller_sha256': 'a' * 64}}
    monkeypatch.setattr(shared, 'load_shared_exercise_lab', lambda *a, **k: deepcopy(loaded))
    monkeypatch.setattr(camera_shared_task, 'CameraSharedTaskEnv', Engine)
    monkeypatch.setattr(recording, 'recording_scene', lambda *a, **k: {'reference_controller': {'sha256': 'a' * 64}})
    def prepare(*args):
        preparations.append(True)
        return lambda: (decisions.append(True) or np.zeros(19))
    monkeypatch.setattr(recording, 'prepare_recorded_reference_controller', prepare)
    monkeypatch.setattr(capabilities, 'describe_capabilities', lambda *a, **k: {'controllers': {'learned_policy_available': False}})
    session = shared.SharedExerciseSession('unused')
    assert session.capabilities()['controllers']['fitted_reference_controller'] is True
    monkeypatch.setattr(attachment, 'read_package_file', lambda *a, **k: pytest.fail('per-action disk read'))
    monkeypatch.setattr(recording, '_prepare', lambda *a, **k: pytest.fail('per-action recipe preparation'))
    began = time.monotonic()
    for _ in range(1000):
        chosen = session.fitted_action()
        assert session.fitted_recording_metadata(chosen) == {'reference_controller_sha256': 'a' * 64}
    record_property('cached_actions', json.dumps({'actions': 1000, 'seconds': time.monotonic() - began,
        'scope': 'Analytical cached action adapter only; not native or UI performance'}))
    assert len(preparations) == 1 and len(decisions) == 1000
    session.reset(4)
    with pytest.raises(ValueError, match='actual prepared'):
        session.fitted_recording_metadata(chosen)
    def fail(*args): raise ValueError('recipe preparation failed')
    monkeypatch.setattr(recording, 'prepare_recorded_reference_controller', fail)
    with pytest.raises(ValueError, match='recipe preparation failed'):
        shared.SharedExerciseSession('unused')
    assert closed == [True]


def test_cli_forwards_explicit_attachment_without_changing_other_options(monkeypatch, capsys):
    from weft import cli
    calls = []
    monkeypatch.setattr(shared, 'prepare_shared_exercise_lab', lambda *a, **k: calls.append((a, k)) or {'status': 'ready'})
    monkeypatch.setattr('sys.argv', ['weft', 'shared-exercise-lab', '--run', 'original', '--task', 'exact-task.json',
        '--out', 'new-lab', '--reference-trial', 'completed-trial', '--reference-manifest-sha256', 'a' * 64, '--max-seconds', '20'])
    cli.main()
    assert calls == [(('original', 'new-lab'), {'root_assistance': False, 'task_path': 'exact-task.json',
        'policy_run': None, 'evaluation_run': None, 'max_seconds': 20.,
        'reference_trial': 'completed-trial', 'reference_manifest_sha256': 'a' * 64})]
