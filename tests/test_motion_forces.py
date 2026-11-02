"""CLI packaging, independent force arithmetic and immutable-source boundaries."""

from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from weft import camera_shared_task, reference_dynamics_run as workflow, shared_world
from weft.io import canonical_hash, digest, write_json
from weft.pipeline import verify_artifacts
from weft.reference_dynamics import audit_reference_dynamics


def seal(run):
    write_json(run / 'manifest.json', {'files': {name: digest(run / name) for name in ('scene.json', 'exercise-task.json')}})


@pytest.fixture
def package(tmp_path, monkeypatch):
    run = tmp_path / 'run'; run.mkdir()
    scene = {'profile': 'shared-world-v6', 'objects': [], 'analytic_fixture': True}
    task = {'profile': 'camera-shared-task-v1', 'scene_sha256': canonical_hash(scene),
            'control': {'root_assistance': 'explicit_3_axis'}, 'gates_sha256': 'a' * 64}
    write_json(run / 'scene.json', scene); write_json(run / 'exercise-task.json', task); seal(run)
    names = [f'axis{i}' for i in range(22)]
    bodies = ''.join(f'<body pos="{i} 0 0"><joint name="{name}" type="slide" damping="2"/>'
                     '<geom type="sphere" size=".1" mass="2" contype="0" conaffinity="0"/></body>'
                     for i, name in enumerate(names))
    motors = ''.join(f'<motor name="{name}_motor" joint="{name}" ctrllimited="true" ctrlrange="-10 10"/>' for name in names)
    model = mujoco.MjModel.from_xml_string(f'<mujoco><option gravity="0 0 0"/><worldbody>{bodies}</worldbody><actuator>{motors}</actuator></mujoco>')
    closed = []
    env = SimpleNamespace(model=model, scene=deepcopy(scene), human_motor_count=22, start_offset_s=0,
        horizon_steps=10, dt=.02, _times=np.array([0., .1, .2]), _reference=np.tile([0., .2, 0.], (22, 1)),
        human_action_names=names, human_limits=np.full(22, 10.), close=lambda: closed.append(True))
    monkeypatch.setattr(shared_world, 'SharedWorldEnv', lambda actual: env)
    monkeypatch.setattr(camera_shared_task, 'validate_camera_shared_task', lambda task, scene: task)
    monkeypatch.setattr(workflow, '_runtime', lambda: {'implementation/fixture.txt': b'Analytic test runtime'})
    monkeypatch.setattr(workflow, '_rss', lambda: 64 << 20)
    def denied(*args, **kwargs):
        pytest.fail('Force inspection must not step physics, render or use network')
    for name in ('mj_step', 'mj_step1', 'mj_step2', 'Renderer'):
        monkeypatch.setattr(mujoco, name, denied)
    monkeypatch.setattr(socket, 'socket', denied)
    return SimpleNamespace(run=run, out=tmp_path / 'forces', scene=scene, task=task, env=env, closed=closed)


def test_analytic_cli_workflow_preserves_inputs_and_reports_separate_probes(package):
    before = {p.name: p.read_bytes() for p in package.run.iterdir()}
    result = workflow.run_motion_force_check(package.run, package.out)
    assert result['status'] == 'diagnostic_only' and result['source_and_task_unchanged'] is True
    assert result['physics_steps'] == result['api_cost_usd'] == result['inference_calls'] == 0
    assert result['inverse_calls'] == 4 and package.closed == [True]
    midpoint, knot = result['summary']['interval_midpoint'], result['summary']['knot_transition_probe']
    assert midpoint['samples'] == midpoint['constraint_free_samples'] == 2
    assert midpoint['over_limit_samples'] == 0
    assert midpoint['constraint_free_maximum_force_limit_ratio'] == pytest.approx(.4)
    assert knot['constraint_free_maximum_force_limit_ratio'] == pytest.approx(40.)
    assert {p.name: p.read_bytes() for p in package.run.iterdir()} == before
    assert verify_artifacts(package.out)['verified_files'] == 9
    diagnostic = json.loads((package.out / 'diagnostic.json').read_bytes())
    assert diagnostic['samples'][0]['required_motor_force'] == [4.] * 22
    assert diagnostic['samples'][-1]['required_motor_force'] == [-400.] * 22
    assert result['physical_validation'] == 'unverified'
    assert 'finite-window' in (package.out / 'index.html').read_text()
    assert 'not hard preemption' in result['budget_scope']


@pytest.mark.parametrize('case', ['changed_member', 'missing_task', 'duplicate_key', 'nan', 'bad_hash',
    'self_seal', 'escape', 'linked_member', 'too_many', 'failed', 'wrong_scene', 'equipment', 'disabled_root'])
def test_invalid_or_unsupported_input_never_constructs(package, monkeypatch, case):
    run = package.run
    if case == 'changed_member': (run / 'scene.json').write_text('{}')
    elif case == 'duplicate_key': (run / 'manifest.json').write_text('{"files":{},"files":{}}')
    elif case == 'nan': (run / 'manifest.json').write_text('{"files":{},"extra":NaN}')
    elif case == 'linked_member':
        target = run.parent / 'linked.json'; target.write_text('{}')
        (run / 'link.json').symlink_to(target)
        value = json.loads((run / 'manifest.json').read_bytes()); value['files']['link.json'] = digest(target)
        write_json(run / 'manifest.json', value)
    elif case in {'equipment', 'wrong_scene'}:
        if case == 'equipment': package.scene['objects'] = [{'id': 'object'}]
        else: package.scene['profile'] = 'shared-world-v5'
        package.task['scene_sha256'] = canonical_hash(package.scene)
        write_json(run / 'scene.json', package.scene); write_json(run / 'exercise-task.json', package.task); seal(run)
    elif case == 'disabled_root':
        package.task['control']['root_assistance'] = 'disabled'
        write_json(run / 'exercise-task.json', package.task); seal(run)
    else:
        value = json.loads((run / 'manifest.json').read_bytes())
        if case == 'missing_task': del value['files']['exercise-task.json']
        if case == 'bad_hash': value['files']['scene.json'] = 'not a hash'
        if case == 'self_seal': value['files']['manifest.json'] = 'a' * 64
        if case == 'escape': value['files']['../outside.json'] = 'a' * 64
        if case == 'too_many': value['files'].update({f'{i}.json': 'a' * 64 for i in range(513)})
        if case == 'failed': value['status'] = 'failed'
        write_json(run / 'manifest.json', value)
    monkeypatch.setattr(shared_world, 'SharedWorldEnv', lambda *a: pytest.fail('Invalid source must not compile'))
    with pytest.raises(ValueError): workflow.run_motion_force_check(run, package.out)
    assert not package.out.exists()


@pytest.mark.parametrize('seconds', [0, -.1, 61, True, float('nan'), float('inf')])
def test_invalid_budget_rejected_before_any_source_read(package, seconds, monkeypatch):
    monkeypatch.setattr(workflow, '_inputs', lambda *a: pytest.fail('No source work for invalid budget'))
    with pytest.raises(ValueError): workflow.run_motion_force_check(package.run, package.out, max_seconds=seconds)


@pytest.mark.parametrize('kind', ['existing', 'nested', 'linked', 'parent'])
def test_output_cannot_overwrite_or_enter_source(package, tmp_path, kind):
    out = package.out
    if kind == 'existing': out.mkdir(); (out / 'saved.txt').write_text('preserve')
    elif kind == 'nested': out = package.run / 'forces'
    elif kind == 'linked': out.symlink_to(package.run, target_is_directory=True)
    else: out = package.run.parent
    before = (package.run / 'manifest.json').read_bytes()
    with pytest.raises(ValueError): workflow.run_motion_force_check(package.run, out)
    assert (package.run / 'manifest.json').read_bytes() == before
    if kind == 'existing': assert (out / 'saved.txt').read_text() == 'preserve'


@pytest.mark.parametrize('kind', ['compile', 'inverse', 'source_changed', 'runtime_changed', 'deadline', 'memory'])
def test_failure_preserves_new_receipt_and_closes_native_state(package, monkeypatch, kind):
    inverse = mujoco.mj_inverse
    def call(*args):
        if kind == 'inverse': raise RuntimeError('native diagnostic failure')
        if kind == 'source_changed': (package.run / 'scene.json').write_text('{}')
        if kind == 'runtime_changed': monkeypatch.setattr(workflow, '_runtime', lambda: {})
        if kind == 'deadline': monkeypatch.setattr(workflow.time, 'monotonic', lambda: 1e20)
        if kind == 'memory': monkeypatch.setattr(workflow, '_rss', lambda: (512 << 20) + 1)
        return inverse(*args)
    monkeypatch.setattr(mujoco, 'mj_inverse', call)
    if kind == 'compile':
        def fail(*args): raise ValueError('compile failed')
        monkeypatch.setattr(shared_world, 'SharedWorldEnv', fail)
    result = workflow.run_motion_force_check(package.run, package.out)
    assert result['status'] == 'failed' and result['error'] is not None
    assert result['source_and_task_unchanged'] is None and result['summary'] == {}
    assert result['inverse_calls'] is None
    assert package.closed == ([] if kind == 'compile' else [True])
    assert verify_artifacts(package.out)['verified_files'] == 6
    assert not (package.out / 'index.html').exists()


def test_absent_constraint_free_samples_remain_unknown(package, monkeypatch):
    from weft import reference_dynamics
    def constrained(*args, **kwargs):
        result = audit_reference_dynamics(*args, **kwargs)
        for row in result['samples']: row['constraint_rows'] = 1
        for row in result['summary'].values():
            row['constraint_free_samples'] = row['constraint_free_over_limit_samples'] = 0
        return result
    monkeypatch.setattr(reference_dynamics, 'audit_reference_dynamics', constrained)
    result = workflow.run_motion_force_check(package.run, package.out)
    assert result['status'] == 'diagnostic_only'
    assert all(row['constraint_free_maximum_force_limit_ratio'] is None for row in result['summary'].values())
    assert 'unavailable' in (package.out / 'index.html').read_text()


def test_budget_callback_can_stop_before_native_allocation(package, monkeypatch):
    monkeypatch.setattr(mujoco, 'MjData', lambda *a: pytest.fail('Expired work must not allocate'))
    def expired(): raise TimeoutError('expired')
    with pytest.raises(TimeoutError):
        audit_reference_dynamics(package.env.model, [0., .1], np.zeros((2, 22)), names=package.env.human_action_names,
            limits=package.env.human_limits, transition_s=.02, check_budget=expired)


@pytest.mark.parametrize('failed', [False, True])
def test_command_routes_and_reports_failure_without_serving(package, monkeypatch, capsys, failed):
    from weft.cli import main
    calls = []
    def execute(run, out, **kwargs):
        calls.append((run, out, kwargs))
        return {'status': 'failed' if failed else 'diagnostic_only', 'report_path': str(package.out / 'index.html')}
    monkeypatch.setattr(workflow, 'run_motion_force_check', execute)
    monkeypatch.setattr(sys, 'argv', ['weft', 'motion-forces', '--run', str(package.run), '--out', str(package.out), '--max-seconds', '12'])
    if failed:
        with pytest.raises(SystemExit) as error: main()
        assert error.value.code == 1
    else: main()
    assert calls == [(str(package.run), str(package.out), {'max_seconds': 12.})]
    captured = capsys.readouterr()
    assert json.loads(captured.out)['status'] == ('failed' if failed else 'diagnostic_only')
    assert ('preserved output' if failed else 'Saved motion force report') in captured.err
