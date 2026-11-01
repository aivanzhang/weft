"""Parent budgets restrict complete lab output; native admission is mocked here."""

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from weft import equipment_box_consensus as consensus
from weft import exercise_scene_pipeline as pipeline
from weft import shared_exercise_lab as lab
from test_equipment_selection_reference import packaged
from test_equipment_box_consensus import pipeline as world_pipeline
from test_exercise_scene_pipeline import _fake_stages


def source(monkeypatch, tmp_path):
    body, equipment = packaged(monkeypatch, tmp_path)
    candidate = tmp_path / 'candidate'
    world_pipeline.construct_source_equipment_run(body, tmp_path / 'proposal', equipment, candidate, target_object_id='weight')
    return candidate


def mock_native(monkeypatch, module=lab):
    monkeypatch.setattr(module, 'validate_shared_world', deepcopy)
    monkeypatch.setattr(module, 'build_shared_exercise_task', lambda scene, **_: {'scene_sha256': module.canonical_hash(scene)})
    monkeypatch.setattr(module, '_reference_frames', lambda *_: {'frames': [], 'scope': 'mocked native boundary'})
    monkeypatch.setattr(module, 'load_shared_exercise_lab', lambda *_, **__: None)
    monkeypatch.setattr(module, 'runtime_files', lambda _: ())
    monkeypatch.setattr(module, '_implementation_snapshot', lambda: {'equipment_box_consensus.py': Path(consensus.__file__).read_bytes()})
    monkeypatch.setattr(module, 'local_time', lambda: '2026-11-01T15:30:00-08:00')


@pytest.mark.parametrize('limits', [(), [], (1,), (1, 1, 1), '1,1', {'files': 1, 'bytes': 1},
    (0, 1), (-1, 1), (True, 1), (1., 1), (1, 0), (1, True), (1, float('inf')),
    (lab.MAX_OUTPUT_FILES + 1, 1), (1, lab.MAX_OUTPUT_BYTES + 1)])
def test_invalid_or_expanded_lab_limits_reject_before_input_io(monkeypatch, tmp_path, limits):
    monkeypatch.setattr(lab, '_verified_inputs', lambda *_args, **_kwargs: pytest.fail('Input read before limit validation'))
    with pytest.raises(ValueError, match='existing maxima'):
        lab.prepare_shared_exercise_lab(tmp_path / 'missing', tmp_path / 'out', output_limits=limits)
    assert not (tmp_path / 'out').exists()


def test_exact_complete_output_is_allowed_and_one_less_rejects_before_any_write(monkeypatch, tmp_path):
    candidate = source(monkeypatch, tmp_path); mock_native(monkeypatch)
    baseline = tmp_path / 'baseline'
    lab.prepare_shared_exercise_lab(candidate, baseline)
    expected = {p.relative_to(baseline).as_posix(): p.read_bytes() for p in baseline.rglob('*') if p.is_file()}
    count, total = len(expected), sum(map(len, expected.values()))
    assert count < lab.MAX_OUTPUT_FILES and total < lab.MAX_OUTPUT_BYTES
    exact = tmp_path / 'exact'
    lab.prepare_shared_exercise_lab(candidate, exact, output_limits=(count, total))
    assert {p.relative_to(exact).as_posix(): p.read_bytes() for p in exact.rglob('*') if p.is_file()} == expected
    for name, limits in [('short-files', (count - 1, total)), ('short-bytes', (count, total - 1))]:
        with pytest.raises(ValueError, match='closure exceeds its output bound'):
            lab.prepare_shared_exercise_lab(candidate, tmp_path / name, output_limits=limits)
        assert not (tmp_path / name).exists()


def test_parent_caps_final_lab_from_existing_files_without_dropping_evidence(monkeypatch, tmp_path):
    calls = _fake_stages(monkeypatch)
    original = pipeline.construct_source_equipment_run
    def retained(*args, **kwargs):
        result = original(*args, **kwargs)
        for index in range(127): (args[3] / f'kept-{index}.txt').write_bytes(b'x')
        return result
    monkeypatch.setattr(pipeline, 'construct_source_equipment_run', retained)
    out = tmp_path / 'assembly'
    result = pipeline.prepare_exercise_scene(tmp_path / 'base', tmp_path / 'evidence', tmp_path / 'config', out)
    assert result['status'] == 'ready'
    assert dict(calls)['interactive_lab']['output_limits'] == (1461, lab.MAX_OUTPUT_BYTES)
    row = next(row for row in result['stages'] if row['stage'] == 'interactive_lab' and row['status'] == 'running')
    assert row['output_limits'] == {'files': 1461, 'bytes': lab.MAX_OUTPUT_BYTES}
    assert all((out / f'candidate/kept-{index}.txt').read_bytes() == b'x' for index in range(127))
    assert len(json.loads((out / 'manifest.json').read_bytes())['files']) + 1 <= pipeline.MAX_FILES


@pytest.mark.parametrize('dimension', ['files', 'bytes'])
def test_final_allowance_is_clamped_to_remaining_parent_capacity(monkeypatch, tmp_path, dimension):
    (tmp_path / 'prior').write_bytes(b'x' * 100)
    limits = (20, 1000 + pipeline.METADATA_RESERVE_BYTES)
    allowance = (40, 2000) if dimension == 'files' else (2, 2000)
    expected = (11, 900) if dimension == 'files' else (2, 900)
    assert pipeline._reserve_stage(tmp_path, 'interactive_lab', time.monotonic() + 2,
        limits=limits, allowance=allowance, cap_to_remaining=True) == expected
    with pytest.raises(ValueError, match='cannot reserve'):
        pipeline._reserve_stage(tmp_path, 'interactive_lab', time.monotonic() + 2,
            limits=limits, allowance=allowance)


@pytest.mark.parametrize('limits', [(9, pipeline.MAX_BYTES), (pipeline.MAX_FILES, pipeline.METADATA_RESERVE_BYTES + 1)])
def test_exhausted_parent_allowance_stops_before_lab(monkeypatch, tmp_path, limits):
    (tmp_path / 'prior').write_bytes(b'x')
    with pytest.raises(ValueError, match='cannot reserve'):
        pipeline._reserve_stage(tmp_path, 'interactive_lab', time.monotonic() + 2,
            limits=limits, cap_to_remaining=True)


def test_standalone_explicit_maxima_preserve_default_bytes(monkeypatch, tmp_path):
    candidate = source(monkeypatch, tmp_path); mock_native(monkeypatch)
    for name, limits in [('default', None), ('explicit', (lab.MAX_OUTPUT_FILES, lab.MAX_OUTPUT_BYTES))]:
        lab.prepare_shared_exercise_lab(candidate, tmp_path / name, output_limits=limits)
    read = lambda directory: {p.relative_to(directory).as_posix(): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    assert read(tmp_path / 'default') == read(tmp_path / 'explicit')
