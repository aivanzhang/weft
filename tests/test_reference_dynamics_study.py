"""Source seals and presentation guards for the read-only native study."""

import importlib.util
import json
from pathlib import Path

import pytest

from weft.io import canonical_hash, digest, write_json


spec = importlib.util.spec_from_file_location('reference_dynamics_study',
    Path(__file__).parents[1] / 'scripts/studies/audit_camera_reference_dynamics.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def fixture(tmp_path):
    run = tmp_path / 'run'; run.mkdir()
    scene = {'profile': 'shared-world-v6', 'objects': []}
    task = {'profile': 'camera-shared-task-v1', 'scene_sha256': canonical_hash(scene)}
    write_json(run / 'scene.json', scene)
    write_json(run / 'exercise-task.json', task)
    seal(run)
    return run, tmp_path / 'new-output'


def seal(run):
    write_json(run / 'manifest.json', {'files': {name: digest(run / name) for name in ('scene.json', 'exercise-task.json')}})


def test_valid_input_verified_without_physics(tmp_path, monkeypatch):
    run, out = fixture(tmp_path)
    monkeypatch.setattr(study, 'SharedWorldEnv', lambda *a: pytest.fail('Input checks do not compile'))
    actual, target, scene, task, verification = study.validate_inputs(run, out)
    assert actual == run and target == out and not out.exists()
    assert verification['verified_files'] == 2 and task['scene_sha256'] == canonical_hash(scene)


@pytest.mark.parametrize('kind', ['changed', 'missing_task', 'wrong_scene', 'linked', 'escape', 'oversized', 'invalid_names'])
def test_malformed_sources_rejected_before_compilation(tmp_path, monkeypatch, kind):
    run, out = fixture(tmp_path)
    if kind == 'changed': (run / 'scene.json').write_text('{}')
    elif kind == 'missing_task': write_json(run / 'manifest.json', {'files': {'scene.json': digest(run / 'scene.json')}})
    elif kind == 'wrong_scene':
        write_json(run / 'exercise-task.json', {'profile': 'camera-shared-task-v1', 'scene_sha256': 'a' * 64}); seal(run)
    elif kind == 'linked':
        source = tmp_path / 'other.json'; source.write_text('{}')
        (run / 'linked.json').symlink_to(source)
        write_json(run / 'manifest.json', {'files': {'linked.json': digest(source)}})
    elif kind == 'escape': write_json(run / 'manifest.json', {'files': {'../other.json': 'a' * 64}})
    elif kind == 'oversized': write_json(run / 'manifest.json', {'files': {str(i): 'a' * 64 for i in range(513)}})
    else: write_json(run / 'manifest.json', {'files': []})
    monkeypatch.setattr(study, 'SharedWorldEnv', lambda *a: pytest.fail('Rejected input must not compile'))
    with pytest.raises(ValueError): study.run_audit(run, out)
    assert not out.exists()


@pytest.mark.parametrize('nested', [False, True])
def test_existing_or_nested_output_rejected(tmp_path, nested):
    run, out = fixture(tmp_path)
    if nested: out = run / 'output'
    else: out.mkdir()
    with pytest.raises(ValueError): study.validate_inputs(run, out)


def test_report_separates_exact_samples_from_finite_window_probes():
    from weft.reference_dynamics import audit_reference_dynamics
    from test_reference_dynamics import slider
    diagnostic = audit_reference_dynamics(slider(), [0., .1], [[0.], [.1]],
        names=['slide'], limits=[10.], transition_s=.02)
    diagnostic['names'][0] = '<unsafe>'
    text = study.render_report({'diagnostic': diagnostic})
    assert '<unsafe>' not in text and '&lt;unsafe&gt;' in text
    assert 'finite-window probe' in text and 'Over limit without constraints' in text
    assert 'Largest without constraints' in text
    assert 'not minimum transition times' in text
    assert 'diagnostic.json' in text and text.count('<table>') == 2
    json.dumps(diagnostic, allow_nan=False)
