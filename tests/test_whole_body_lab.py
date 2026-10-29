import json

import numpy as np
import pytest

from weft.human import build_human
from weft.io import digest, write_json
from weft.lab import Lab, WholeBodySession
from weft.pipeline import verify_artifacts
from weft.recording import load_recording
from weft.whole_body_lab import prepare_whole_body_lab
from weft.whole_body_learning import train_whole_body
from test_lab_http import request, serving
from test_mobile_integration import evidence


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / 'source'
    run = source / 'run'
    run.mkdir(parents=True)
    spec = build_human(evidence(), run, body_model='mobile')
    write_json(run / 'policy.json', {'selected_joint': 'left_knee', 'scope': 'legacy fixture'})
    write_json(run / 'manifest.json', {'files': {p.name: digest(p) for p in run.iterdir() if p.is_file()}})
    write_json(source / 'lab-config.json', {'profile': 'exercise'})
    write_json(source / 'status.json', {'status': 'ready', 'events': []})
    learning = tmp_path / 'learning'
    train_whole_body(spec, learning, generations=1, population=4, horizon_steps=3)
    out = tmp_path / 'lab'
    before = [verify_artifacts(path) for path in (run, learning)]
    prepare_whole_body_lab(source, learning, out)
    assert before == [verify_artifacts(path) for path in (run, learning)]
    assert json.loads((out / 'run/legacy-joint/policy.json').read_text())['scope'] == 'legacy fixture'
    return out, source, learning


def test_packaged_lab_loads_all_motor_actions_and_reference_audit(prepared):
    out, source, learning = prepared
    lab = Lab(out)
    try:
        summary = lab.summary()
        assert isinstance(lab.session, WholeBodySession)
        assert summary['scene']['profile'] == 'exercise-imitation-v1'
        assert len(summary['scene']['joint_names']) == 22
        assert summary['reference_feasibility']['status'] in ('feasible', 'infeasible')
        assert 'selected_joint' not in summary['snapshot']['human']
        assert summary['events'][-1]['stage'] == 'ready'
        assert summary['events'][-1]['elapsed_seconds'] >= 0
        with pytest.raises(ValueError):
            lab.control('/api/joint', {'joint': 'root_x'})
    finally:
        lab.close()
    with pytest.raises(FileExistsError):
        prepare_whole_body_lab(source, learning, out)


def test_http_records_exact_vectors_and_replays_all_controllers(prepared):
    out, _, _ = prepared
    before = verify_artifacts(out / 'run')
    with serving(Lab(out)) as server:
        assert request(server, '/api/status', token=False)[0] == 401
        assert request(server, '/api/status', headers={'Origin': 'https://invalid.example'})[0] == 403
        code, _, body = request(server, '/api/record/start', method='POST', data={})
        assert code == 200
        name = json.loads(body)['directory']
        manual = [0.] * 22
        manual[19] = .1
        transitions = []
        for data in ({'action': manual, 'steps': 2}, {'controller': 'reference', 'steps': 1}):
            code, _, body = request(server, '/api/step', method='POST', data=data)
            assert code == 200
            transitions.extend(json.loads(body)['transitions'])
        assert transitions[-1]['truncated']
        assert request(server, '/api/reset', method='POST', data={'seed': 19})[0] == 200
        code, _, body = request(server, '/api/step', method='POST', data={'policy': True, 'steps': 3})
        assert code == 200
        transitions.extend(json.loads(body)['transitions'])
        assert request(server, '/api/record/stop', method='POST', data={})[0] == 200
        result = server.lab.export_future.result(timeout=10)
        saved = load_recording(out / 'sessions' / name)
        rows = [row for row in saved['events'] if row['type'] == 'step']
        assert [{key: value for key, value in row.items() if key != 'index'} for row in rows] == transitions
        assert result['steps'] == 6 and result['video_frames'] == 8
        assert rows[0]['action'] == manual
        replay = WholeBodySession(out / 'run')
        try:
            for row in saved['events']:
                if row['type'] in ('start', 'reset'):
                    replay.reset(row['seed'])
                else:
                    reward, terminated, truncated, _ = replay.step(row['action'])
                    assert (reward, terminated, truncated) == (row['reward'], row['terminated'], row['truncated'])
                assert replay.snapshot() == row['snapshot']
        finally:
            replay.env.close()
    assert before == verify_artifacts(out / 'run')


@pytest.mark.parametrize('data', [
    {}, {'action': 0}, {'action': [0.] * 21}, {'action': [True] * 22}, {'action': [float('nan')] * 22},
    {'action': [1.1] * 22}, {'policy': True, 'action': [0.] * 22},
    {'policy': True, 'controller': 'reference'}, {'controller': 'unknown'}, {'policy': 'yes'},
    {'policy': True, 'steps': 7}, {'policy': True, 'steps': True},
])
def test_invalid_controls_reject_before_mutating_physics(prepared, data):
    lab = Lab(prepared[0])
    try:
        initial = lab.summary()['snapshot']
        with pytest.raises(ValueError):
            lab.control('/api/step', data)
        assert lab.session.snapshot() == initial
        assert np.all(lab.session.env.data.ctrl == 0)
    finally:
        lab.close()


def test_numerical_failure_keeps_last_valid_pose_and_exports_prefix(prepared, monkeypatch):
    lab = Lab(prepared[0])
    try:
        started = lab.control('/api/record/start', {})
        initial = started['snapshot']

        def fail(action):
            raise RuntimeError('nonfinite fixture')

        monkeypatch.setattr(lab.session.env, 'step', fail)
        action = [0.] * 22
        action[21] = .2
        response = lab.control('/api/step', {'action': action})
        assert response['recording'] is False and response['transitions'] == []
        assert response['snapshot']['human'] == initial['human']
        assert response['snapshot']['simulation_failure']['attempted_action'] == action
        result = lab.export_future.result(timeout=10)
        saved = load_recording(prepared[0] / 'sessions' / started['directory'])
        assert result['steps'] == 0 and result['video_frames'] == 1
        assert saved['metadata']['failed_attempt']['reason'] == 'nonfinite fixture'
        assert lab.session.done
    finally:
        lab.close()


def test_unsealed_files_are_excluded_and_absolute_manifest_paths_rejected(prepared, tmp_path):
    _, source, learning = prepared
    (source / 'run/unrelated.txt').write_text('Not part of the source run')
    (learning / 'implementation/unrelated.txt').write_text('Not part of the learner')
    out = tmp_path / 'sealed-only'
    prepare_whole_body_lab(source, learning, out)
    assert not (out / 'run/unrelated.txt').exists()
    assert not (out / 'run/whole-body-implementation/unrelated.txt').exists()
    manifest_path = source / 'run/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    path = source / 'run/human.json'
    manifest['files'][str(path.resolve())] = digest(path)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='canonical relative'):
        prepare_whole_body_lab(source, learning, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()
