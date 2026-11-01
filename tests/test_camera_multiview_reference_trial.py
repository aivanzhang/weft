"""Matched-loop routing and bounded saved custody; all dynamics/exporters are mocked."""

import ast
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from test_shared_exercise_reference_trial import (
    no_native_or_network, velocity_index, setup_case, run,
)
from weft import shared_exercise_reference_trial as trial


@pytest.mark.parametrize('fault', [None, 'live_step'])
def test_multiview_reuses_matched_loop_and_retains_actual_failure(setup_case, monkeypatch, fault):
    state = setup_case(camera=True, fault=fault)
    receipt = state.inputs.receipt()
    receipt['profile'] = 'camera-multiview-reference-inputs-v1'
    state.inputs = replace(state.inputs, _receipt_json=json.dumps(receipt))
    old = trial._runtime_snapshot
    seen = []
    def runtime(deadline, *, camera=False, multiview=False):
        seen.append((camera, multiview))
        assert camera and multiview
        return old(deadline, camera=camera)
    monkeypatch.setattr(trial, '_runtime_snapshot', runtime)
    result = run(state)
    assert seen == [(True, True)]
    assert result['controller_family'] == 'camera_multiview_angle_overlay'
    assert result['limits']['maximum_selected_metadata_read_bytes'] == 80 << 20
    assert state.loader_options['multiview_max_input_bytes'] == 2 * ((80 << 20) // 3)
    assert state.loader_options['camera_max_input_bytes'] == 2 * ((64 << 20) // 3)
    assert (state.out / 'run/scene.json').read_bytes() == state.inputs.scene_bytes
    assert (state.out / 'run/exercise-task.json').read_bytes() == state.inputs.task_bytes
    assert (state.out / 'inputs/angle-proposal.json').read_bytes() == state.inputs.angle_proposal_bytes
    assert all(event.get('controller') != 'learned' for recorder in state.recorders.values() for event in recorder.events)
    assert len(state.envs) == 1 and state.envs[0].closed
    if fault:
        assert result['status'] == 'failed' and result['counts']['live_control_steps'] < 18
    else:
        assert result['status'] == 'completed' and result['same_initial_state'] is True
        assert result['counts']['live_control_steps'] == 18
        assert result['counts']['exports_completed'] == result['counts']['replays_completed'] == 2
        assert state.envs[0].resets == [(1729, {'jitter': False, 'jitter_scale': 0.})] * 2
        assert [row['metrics']['success'] for row in result['trials']] == [True, False]
        assert result['trials'][1]['action_replay']['reference_actions_verified'] == 9


@pytest.mark.parametrize('family,size,passes', [
    ('camera-multiview-reference-inputs-v1', 23_106_000, True),
    ('camera-multiview-reference-inputs-v1', (80 << 20) // 3, True),
    ('camera-multiview-reference-inputs-v1', (80 << 20) // 3 + 1, False),
    ('camera-reference-inputs-v1', 23_106_000, False),
    ('camera-reference-inputs-v1', 18_825_425, True),
    ('shared-exercise-reference-inputs-v1', 18_825_425, False)])
def test_final_readback_keeps_family_caps(tmp_path, monkeypatch, family, size, passes):
    # Logical read accounting only: no large fixture allocation or physical input scan.
    task = tmp_path / 'exercise-task.json'
    sizes = [min(8 << 20, size - i) for i in range(0, size, 8 << 20)]
    names = ['part-' + str(i) + '.json' for i in range(len(sizes) - 1)] + [task.name]
    receipt = {'profile': family, 'input_bytes_read': 2 * size,
        'selected_files': [{'directory': str(tmp_path), 'name': name, 'sha256': 'a' * 64, 'bytes': count}
                           for name, count in zip(names, sizes, strict=True)]}
    inputs = SimpleNamespace(receipt=lambda: receipt, source_directory=tmp_path)
    proof = SimpleNamespace(manifest_bytes=b'', files={})
    calls = []
    def read(*args):
        calls.append(args)
        return args[3]
    monkeypatch.setattr(trial, '_read_identity', read)
    monkeypatch.setattr(trial, '_runtime_versions', lambda: {})
    if passes:
        result = trial._recheck(inputs, proof, {}, {}, task, float('inf'))
        assert result['selected_metadata_bytes'] == 3 * size and len(calls) == len(sizes)
    else:
        with pytest.raises(ValueError, match='aggregate selected metadata'):
            trial._recheck(inputs, proof, {}, {}, task, float('inf'))
        limit = (80 << 20 if family == 'camera-multiview-reference-inputs-v1' else
                 64 << 20 if family == 'camera-reference-inputs-v1' else 32 << 20)
        assert 2 * size + sum(args[3] for args in calls) <= limit or not calls


def test_multiview_runtime_archives_new_metadata_validation_without_native(monkeypatch):
    from weft import camera_reference_controller as actor
    from weft.camera_shared_receipt import RUNTIME_REQUIRED
    package = Path(trial.__file__).parent
    tree = ast.parse((package / 'camera_reference_recording.py').read_bytes())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'REQUIRED_RUNTIME_FILES' for target in node.targets))
    required = eval(compile(ast.Expression(assignment.value), '<exact-recorder-closure>', 'eval'),
                    {'RUNTIME_REQUIRED': RUNTIME_REQUIRED, 'controller': actor, 'frozenset': frozenset})
    monkeypatch.setitem(sys.modules, 'weft.camera_reference_recording', SimpleNamespace(REQUIRED_RUNTIME_FILES=required))
    runtime = trial._runtime_snapshot(float('inf'), camera=True, multiview=True)
    assert required | {'camera_multiview_report.py', 'camera_multiview_inputs.py', 'camera_multiview_fit.py'} <= runtime.keys()
    assert all(sha256(runtime[name]).hexdigest() == pin for name, pin in actor.RUNTIME_SHA256.items())
    assert len(runtime) <= 128 and 3 * sum(map(len, runtime.values())) <= trial.MAX_RUNTIME_BYTES
    with pytest.raises(ValueError, match='authenticated camera inputs'):
        trial._runtime_snapshot(float('inf'), multiview=True)
