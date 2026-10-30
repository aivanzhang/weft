"""Second-order feasibility restoration without relaxed gates or repeated source fits."""

import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pytest

from test_arm_active_set_diagnostic import diagnostic_fixture, parent_fixture


def load_study(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts/studies'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('curvature_study', scripts / 'check_arm_curvature_restoration.py')
    study = importlib.util.module_from_spec(spec); spec.loader.exec_module(study)
    return study


def test_second_order_correction_restores_curved_feasibility_with_original_tolerance(monkeypatch):
    study = load_study(monkeypatch)
    delta = np.zeros(8); delta[0] = .1 / 32
    # y-x² >= 0 is tangent-feasible at the origin, but the first-order step enters it.
    gap = -delta[0] ** 2
    jac = np.zeros(8); jac[:2] = [-2 * delta[0], 1.]
    matrix = np.vstack((jac, np.eye(8), -np.eye(8)))
    lower = np.r_[-gap - 1e-8, np.full(8, -.1) - delta, delta - .1]
    labels = [{'kind': 'pair_gap'}] + [{'kind': 'correction_bound'}] * 16
    result = study.minimum_norm_restoration(matrix, lower, delta, labels)
    assert gap < -1e-8 and result['status'] == 'optimal'
    assert result['problem']['return_witness_feasible']
    assert result['problem']['lower'][0] == pytest.approx(-gap, abs=1e-18)
    proposed = delta + np.asarray(result['correction'])
    assert proposed[1] - proposed[0] ** 2 >= -1e-8
    assert 0 < proposed[0] <= delta[0] and np.linalg.norm(result['correction']) < np.linalg.norm(delta)


def test_infeasible_return_witness_abstains_without_solver_or_relaxed_constraint(monkeypatch):
    study = load_study(monkeypatch)
    monkeypatch.setattr(study, 'solve_feasible_qp', lambda *a, **k: pytest.fail('Infeasible witness entered solver'))
    matrix = np.zeros((1, 8)); matrix[0, 0] = 1.
    result = study.minimum_norm_restoration(matrix, [.001 - 1e-8], np.zeros(8), [{'kind': 'pair_gap'}])
    assert result['status'] == 'abstained_infeasible_return_witness' and result['solver'] is None
    assert result['problem']['return_witness_maximum_violation'] == pytest.approx(.001)
    assert result['problem']['linear_pair_tolerance_spending'] == 0.


def test_prepare_context_never_recaptures_or_solves_the_saved_quadratic(monkeypatch):
    local, raw, evidence, options, previous = diagnostic_fixture(monkeypatch)
    monkeypatch.setattr(local.arm, '_evaluate_arm_pose', lambda *a, **k: pytest.fail('Preparation recaptured a quadratic'))
    context = local.capture_local_problems(raw, evidence, previous['human_candidate'], frames=(4, 6),
                                          deadline=time.monotonic() + 5., prepare_only=True, **options)
    assert context['indices'] == [2, 3] and len(context['source_rows']) == 11
    assert len(context['rows']) == 6


def restoration_parent(tmp_path, monkeypatch):
    local, source, output = parent_fixture(tmp_path, monkeypatch)
    local.run(source, output)
    study = load_study(monkeypatch); monkeypatch.setattr(study, 'FRAMES', (4, 6))
    return study, output, tmp_path / 'restoration'


def test_restoration_driver_retains_all_pair_checks_and_seals_without_installation(tmp_path, monkeypatch):
    study, source, output = restoration_parent(tmp_path, monkeypatch)
    old = study.local.capture_local_problems; calls = []
    def prepare(*args, **kwargs):
        assert kwargs['prepare_only']
        calls.append('prepare'); return old(*args, **kwargs)
    monkeypatch.setattr(study.local, 'capture_local_problems', prepare)
    report = study.run(source, output)
    assert calls == ['prepare'] and report['status'] == 'complete_local_restoration_diagnostic'
    assert report['full_source_fits'] == 0 and not report['candidate_installed']
    for path in (output / 'frames').glob('*.json'):
        result = json.loads(path.read_text())
        assert len(result['corrections']) <= 2
        for audit in (result['initial'], result['final']):
            assert audit['pair_comparisons'] == len(audit['pair_checks'])
            assert audit['affected_unique_samples'] > 1
            assert len(audit['source_endpoints']) == 4
        assert not result['candidate_installed']
    manifest = json.loads((output / 'manifest.json').read_text())
    assert report['readback_verified_files'] == len(manifest['files'])


def test_restoration_reserves_all_outputs_before_geometry(tmp_path, monkeypatch):
    study, source, output = restoration_parent(tmp_path, monkeypatch)
    monkeypatch.setattr(study, 'MAX_RESULT_BYTES', study.MAX_OUTPUT_BYTES)
    monkeypatch.setattr(study.local, 'capture_local_problems', lambda *a, **k: pytest.fail('Unreserved geometry'))
    with pytest.raises(ValueError, match='reserves exceed'):
        study.run(source, output)
    assert not output.exists()


def test_failed_restoration_keeps_explicit_receipt_and_original_inputs(tmp_path, monkeypatch):
    study, source, output = restoration_parent(tmp_path, monkeypatch)
    def failed(*a, **k): raise TimeoutError('injected ten-second restoration deadline')
    monkeypatch.setattr(study, 'restore_frame', failed)
    with pytest.raises(TimeoutError): study.run(source, output)
    report = json.loads((output / 'report.json').read_text()); manifest = json.loads((output / 'manifest.json').read_text())
    assert report['status'] == 'failed' and manifest['custody'] == 'failure_receipt_only'
    assert not report['promoted'] and (output / 'pre-restoration.json').exists()
