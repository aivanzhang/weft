"""Local matrix capture preserves the frozen candidate and cannot run a source fit."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pytest

from test_arm_equipment_refinement import fixture
from weft import constrained_arm_alignment as arm


def diagnostic_fixture(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts/studies'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('active_set_study', scripts / 'check_arm_active_set.py')
    study = importlib.util.module_from_spec(spec); spec.loader.exec_module(study)
    human, evidence, options = fixture()
    raw = json.dumps(human).encode()
    previous = arm.propose_constrained_arm_alignment(raw, evidence, **options)
    return study, raw, evidence, options, previous


def test_capture_has_exact_matrices_and_preserves_candidate_without_optimizer(monkeypatch):
    study, raw, evidence, options, previous = diagnostic_fixture(monkeypatch)
    candidate = previous['human_candidate']; frozen = deepcopy(candidate)
    monkeypatch.setattr(arm, '_propose', lambda *a, **k: pytest.fail('Local capture reran the full optimizer'))
    matrices, check = study.capture_local_problems(raw, evidence, candidate, frames=(4, 6), deadline=time.monotonic() + 5., **options)
    assert candidate == frozen and len(matrices) == 2
    for data in matrices:
        assert not data['historical_subproblem']
        assert data['all_source_slots'] == 11 and data['all_references'] == 6
        assert len(data['constraint_rows']) == len(data['matrix']) == len(data['lower'])
        assert np.asarray(data['hessian']).shape == (8, 8)
        assert any(len(s['reference_indices']) == 2 for s in data['affected_samples'])
        assert all(data['reference_index'] in s['reference_indices'] for s in data['affected_samples'])
        # Actual nonlinear cost differences check the captured gradient, not a second quadratic implementation.
        numeric = []
        for axis in range(8):
            step = np.zeros(8); step[axis] = 1e-6
            plus = check(data['reference_index'], step)['attempts'][0]['cost']
            minus = check(data['reference_index'], -step)['attempts'][0]['cost']
            numeric.append((plus - minus) / 2e-6)
        np.testing.assert_allclose(data['gradient'], numeric, atol=2e-5, rtol=1e-6)
    assert candidate == frozen


def parent_fixture(tmp_path, monkeypatch):
    study, raw, evidence, options, previous = diagnostic_fixture(monkeypatch)
    source, out = tmp_path / 'parent', tmp_path / 'diagnostic'; source.mkdir()
    files = {'proposal.json': study.custody._bytes(previous),
        'pre-fit.json': study.custody._bytes({k: options[k] for k in ('human_entity_ref', 'camera_hypothesis', 'equipment_boxes')}),
        'inputs/human.json': raw, 'inputs/construction-evidence.json': evidence, 'inputs/human.xml': options['human_xml_document'],
        'inputs/scene.json': b'{}', 'inputs/geometry-proposal.json': b'{}', 'inputs/source.json': b'{"scope":"analytic fixture"}',
        'implementation/constrained_arm_alignment.py': Path(arm.__file__).read_bytes()}
    for name, value in files.items():
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
    (source / 'manifest.json').write_text(json.dumps({'files': {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}}))
    monkeypatch.setattr(study, 'FRAMES', (4, 6))
    return study, source, out


def test_single_capture_is_sealed_before_solver_and_all_inputs_read_back(tmp_path, monkeypatch):
    study, source, out = parent_fixture(tmp_path, monkeypatch)
    original_capture, original_solve = study.capture_local_problems, study.solve_feasible_qp
    calls = []
    def capture(*args, **kwargs):
        plan = json.loads((out / 'pre-capture.json').read_text())
        assert plan['capture_count'] == 1 and plan['full_source_fits'] == 0
        calls.append('capture'); return original_capture(*args, **kwargs)
    def solve(*args, **kwargs):
        assert len(list((out / 'subproblems').glob('*.json'))) == 2
        calls.append('solve'); return original_solve(*args, **kwargs)
    monkeypatch.setattr(study, 'capture_local_problems', capture)
    monkeypatch.setattr(study, 'solve_feasible_qp', solve)
    report = study.run(source, out)
    assert calls == ['capture', 'solve', 'solve']
    assert report['status'] == 'complete_local_diagnostic' and report['full_source_fits'] == 0
    manifest = json.loads((out / 'manifest.json').read_text())
    assert report['readback_verified_files'] == len(manifest['files'])
    assert sum((out / name).stat().st_size for name in manifest['files']) < study.MAX_OUTPUT_BYTES
    for name, digest in manifest['files'].items(): assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_preflight_reservation_rejects_before_capture_or_output(tmp_path, monkeypatch):
    study, source, out = parent_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(study, 'MAX_OUTPUT_BYTES', 1024 * 1024)
    monkeypatch.setattr(study, 'MAX_RESULT_BYTES', 1024 * 1024)
    monkeypatch.setattr(study, 'capture_local_problems', lambda *a, **k: pytest.fail('Unreserved capture'))
    with pytest.raises(ValueError, match='reserves exceed'):
        study.run(source, out)
    assert not out.exists()


def test_capture_failure_keeps_failed_receipt_without_admission(tmp_path, monkeypatch):
    study, source, out = parent_fixture(tmp_path, monkeypatch)
    def failed(*a, **k): raise TimeoutError('injected local-capture deadline')
    monkeypatch.setattr(study, 'capture_local_problems', failed)
    with pytest.raises(TimeoutError): study.run(source, out)
    receipt = json.loads((out / 'report.json').read_text()); manifest = json.loads((out / 'manifest.json').read_text())
    assert receipt['status'] == 'failed' and receipt['custody'] == 'failure_receipt_only'
    assert manifest['custody'] == 'failure_receipt_only' and not receipt['promoted']
    assert (out / 'pre-capture.json').exists()
