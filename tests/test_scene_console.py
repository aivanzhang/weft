"""Terminal controls operate a real SceneEnv; source seals remain unchanged."""

import cv2
import pytest

from weft.io import canonical_hash, digest, write_json
from weft.recording import load_recording
from weft.scene import SceneEnv
from weft.scene_console import command_action, console
from test_scene import scene


def sealed_run(path, scene, *, candidate=False, mutate_report=None):
    path.mkdir()
    write_json(path / 'scene.json', scene)
    if candidate:
        value = {'profile': 'segmentation-planar-candidate-v1', 'status': 'executable_hypothesis',
                 'physics_ready': False, 'physical_validation': 'unverified', 'scene': scene}
        report = {'status': 'completed_diagnostic_replay', 'physics_ready': False,
                  'physical_validation': 'unverified', 'candidate_sha256': canonical_hash(value)}
        if mutate_report:
            mutate_report(report)
        write_json(path / 'candidate.json', value)
        write_json(path / 'report.json', report)
    write_json(path / 'manifest.json', {'files': {p.name: digest(p) for p in path.iterdir()}})
    return path


def run_commands(run, recording, commands, **kwargs):
    iterator = iter(commands)
    output = []
    result = console(run, recording, input_fn=lambda _: next(iterator), output_fn=output.append, **kwargs)
    return result, output


@pytest.mark.parametrize('name,index', [('coast', 0), ('left', 1), ('right', 2), ('up', 3), ('down', 4), ('brake', 5)])
def test_command_names_are_explicit_and_step_counts_bounded(name, index):
    assert command_action(name) == (index, 1)
    assert command_action(name + ' 6') == (index, 6)


@pytest.mark.parametrize('command', ['', '1', 'right 7', 'left 0', 'up -1', 'down 1.5',
                                     'coast nan', 'brake 2 3', 'unknown', 'right ' + '1' * 300])
def test_invalid_commands_do_not_become_actions(command):
    with pytest.raises(ValueError):
        command_action(command)


def test_actual_actions_record_export_and_replay_exactly_without_mutating_run(tmp_path, scene):
    run = sealed_run(tmp_path / 'run', scene, candidate=True)
    before = {p.name: digest(p) for p in run.iterdir()}
    commands = ['status', 'help', 'bogus', 'right 2', 'left 2', 'up 2', 'down 2', 'coast 2', 'brake 2', 'reset', 'quit']
    result, output = run_commands(run, tmp_path / 'recording', commands)
    saved = load_recording(tmp_path / 'recording')
    assert result['steps'] == 12 and result['resets'] == 1 and result['video_frames'] == 14
    assert [event['action'] for event in saved['events'] if event['type'] == 'step'] == [2, 2, 1, 1, 3, 3, 4, 4, 0, 0, 5, 5]
    assert any('Physical validation: unverified' in line for line in output)
    assert saved['events'][0]['snapshot'] == saved['events'][-1]['snapshot']
    env = SceneEnv(scene)
    for event in saved['events']:
        if event['type'] in {'start', 'reset'}:
            env.reset(seed=event['seed'], options={'jitter': False})
        else:
            _, reward, terminated, truncated, _ = env.step(event['action'])
            assert (reward, terminated, truncated) == (event['reward'], event['terminated'], event['truncated'])
        assert env.snapshot() == event['snapshot']
    cap = cv2.VideoCapture(str(tmp_path / 'recording/video.mp4'))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    assert count == 14
    assert (tmp_path / 'recording/current-frame.png').is_file()
    assert before == {p.name: digest(p) for p in run.iterdir()}


def test_status_help_and_eof_never_advance_and_still_export(tmp_path, scene):
    run = sealed_run(tmp_path / 'run', scene)
    commands = iter(['status', 'help'])
    def read(_):
        try:
            return next(commands)
        except StopIteration:
            raise EOFError
    output = []
    result = console(run, tmp_path / 'recording', input_fn=read, output_fn=output.append)
    assert result['steps'] == 0 and result['video_frames'] == 1
    assert any('Authored planar simulation' in line for line in output)


def test_task_end_requires_reset_and_stops_requested_batch(tmp_path, scene):
    scene['task']['max_steps'] = 10
    run = sealed_run(tmp_path / 'run', scene)
    result, output = run_commands(run, tmp_path / 'recording', ['coast 6', 'coast 6', 'left', 'reset', 'right', 'quit'])
    saved = load_recording(tmp_path / 'recording')
    assert result['steps'] == 11 and result['resets'] == 1
    assert saved['events'][10]['truncated'] is True
    assert any('Task ended' in line for line in output)


@pytest.mark.parametrize('change', [lambda r: r.update(status='failed_replay'),
    lambda r: r.update(status='unsupported_retained'), lambda r: r.update(physics_ready=True),
    lambda r: r.update(physical_validation='verified'), lambda r: r.update(candidate_sha256='a' * 64)])
def test_candidate_reports_cannot_promote_or_hide_failures(tmp_path, scene, change):
    run = sealed_run(tmp_path / 'run', scene, candidate=True, mutate_report=change)
    with pytest.raises(ValueError):
        run_commands(run, tmp_path / 'recording', ['quit'])
    assert not (tmp_path / 'recording').exists()


def test_unsealed_scene_and_report_rejected(tmp_path, scene):
    run = sealed_run(tmp_path / 'run', scene)
    write_json(run / 'report.json', {'status': 'completed'})
    with pytest.raises(ValueError, match='sealed'):
        run_commands(run, tmp_path / 'recording', ['quit'])
    write_json(run / 'manifest.json', {'files': {'report.json': digest(run / 'report.json')}})
    with pytest.raises(ValueError, match='scene.json'):
        run_commands(run, tmp_path / 'recording', ['quit'])


@pytest.mark.parametrize('name', ['candidate.json', 'report.json'])
def test_null_candidate_metadata_cannot_fall_back_to_authored_scene(tmp_path, scene, name):
    run = sealed_run(tmp_path / 'run', scene, candidate=True)
    write_json(run / name, None)
    write_json(run / 'manifest.json', {'files': {p.name: digest(p) for p in run.iterdir() if p.name != 'manifest.json'}})
    with pytest.raises(ValueError):
        run_commands(run, tmp_path / 'recording', ['quit'])


def test_changed_sealed_input_and_escape_rejected(tmp_path, scene):
    run = sealed_run(tmp_path / 'run', scene)
    write_json(run / 'scene.json', scene | {'damping': .5})
    with pytest.raises(ValueError, match='integrity'):
        run_commands(run, tmp_path / 'recording', ['quit'])
    other = tmp_path / 'outside.json'
    write_json(other, scene)
    linked = tmp_path / 'linked'
    linked.mkdir()
    (linked / 'scene.json').symlink_to(other)
    write_json(linked / 'manifest.json', {'files': {'scene.json': digest(other)}})
    with pytest.raises(ValueError, match='integrity'):
        run_commands(linked, tmp_path / 'recording', ['quit'])
    with pytest.raises(ValueError, match='immutable'):
        run_commands(run, run / 'session', ['quit'])


def test_existing_recording_preserved_and_invalid_seed_rejected(tmp_path, scene):
    run = sealed_run(tmp_path / 'run', scene)
    directory = tmp_path / 'existing'
    directory.mkdir()
    with pytest.raises(FileExistsError):
        run_commands(run, directory, ['quit'])
    for seed in (-1, True, 2**32):
        with pytest.raises(ValueError, match='Seed'):
            run_commands(run, tmp_path / 'new', ['quit'], seed=seed)


def test_failed_step_retains_last_valid_record_without_fake_terminal(tmp_path, scene, monkeypatch):
    run = sealed_run(tmp_path / 'run', scene)
    def fail(self, action):
        raise RuntimeError('analytic dynamics failure')
    monkeypatch.setattr(SceneEnv, 'step', fail)
    result, _ = run_commands(run, tmp_path / 'recording', ['left'])
    saved = load_recording(tmp_path / 'recording')
    assert result['steps'] == 0 and result['video_frames'] == 1
    assert saved['metadata']['failed_attempt']['reason'] == 'analytic dynamics failure'
    assert saved['metadata']['failed_attempt']['command'] == 'left'


@pytest.mark.parametrize('commands', [['right 6'], ['right 2', 'reset']])
def test_recorder_limit_checked_before_step_or_reset(tmp_path, scene, monkeypatch, commands):
    run = sealed_run(tmp_path / 'run', scene)
    monkeypatch.setattr('weft.recording.MAX_EVENTS', 3)
    result, _ = run_commands(run, tmp_path / 'recording', commands)
    saved = load_recording(tmp_path / 'recording')
    assert result['steps'] == 2 and result['resets'] == 0 and result['video_frames'] == 3
    assert 'Recording limit' in saved['metadata']['failed_attempt']['reason']
