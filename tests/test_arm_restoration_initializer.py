"""Saved-matrix feasibility routes never imply nonlinear or global feasibility."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load_study(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts/studies'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('restoration_initializer', scripts / 'check_arm_restoration_initializer.py')
    study = importlib.util.module_from_spec(spec); spec.loader.exec_module(study)
    return study


def problem():
    matrix = np.zeros((2, 8)); matrix[0, 0] = 1.; matrix[1, 0] = -1.
    return {'hessian': np.eye(8).tolist(), 'gradient': [0.] * 8, 'matrix': matrix.tolist(), 'lower': [.4, -.8],
            'initial': [1.] + [0.] * 7, 'constraint_rows': [{'kind': 'pair_gap'}, {'kind': 'correction_upper'}],
            'linear_pair_tolerance_spending': 0.}


def test_verified_dual_initializer_then_active_minimum_norm_preserves_problem(monkeypatch):
    study = load_study(monkeypatch); data = problem(); original = deepcopy(data)
    monkeypatch.setattr(study, 'return_ray_interval', lambda *a, **k: pytest.fail('Unneeded fallback'))
    result = study.solve_saved_problem(data)
    assert result['status'] == 'verified_minimum_norm_linear_restoration'
    assert result['initializer_method'] == 'verified_dual_projection'
    np.testing.assert_allclose(result['qualified_linear_correction'], [.4] + [0.] * 7, atol=1e-10, rtol=0)
    assert result['independent_active_set_checks']['primal']['passes']
    assert data == original and result['nonlinear_admission'] == 'not_executed'


@pytest.mark.parametrize('claimed_converged', [False, True])
def test_all_row_midpoint_fallback_after_failed_or_incorrect_dual_claim(monkeypatch, claimed_converged):
    study = load_study(monkeypatch)
    monkeypatch.setattr(study, '_project_step', lambda *a, **k: (np.zeros(8), {'converged': claimed_converged, 'cycles': 48, 'maximum_violation': 0.}))
    result = study.solve_saved_problem(problem())
    assert not result['dual']['independent_primal']['passes']
    assert result['initializer_method'] == 'verified_all_row_return_ray_midpoint'
    interval = result['return_ray_fallback']
    assert interval['rows_checked'] == 2 and interval['independent_primal']['passes']
    assert interval['midpoint_fraction'] == pytest.approx(.6)
    assert result['status'] == 'verified_minimum_norm_linear_restoration'


def test_empty_return_ray_does_not_claim_eight_dimensional_infeasibility(monkeypatch):
    study = load_study(monkeypatch); data = problem()
    data['matrix'][0][1] = 1.; data['matrix'][1][1] = 1.; data['lower'] = [.6, -.4]
    # x+y>=.6 and -x+y>=-.4 contradict on y=0, but (.5,.2) is feasible.
    point = np.array([.5, .2] + [0.] * 6)
    assert np.all(np.asarray(data['matrix']) @ point >= data['lower'])
    monkeypatch.setattr(study, '_project_step', lambda *a, **k: (np.zeros(8), {'converged': False, 'cycles': 48, 'maximum_violation': .1}))
    result = study.solve_saved_problem(data)
    assert result['status'] == 'no_verified_initializer_within_fixed_routes'
    assert not result['return_ray_fallback']['nonempty'] and not result['global_infeasibility_claim']
    assert result['active_set'] is None


def test_nonidentity_restoration_is_not_silently_regularized(monkeypatch):
    study = load_study(monkeypatch); data = problem(); data['hessian'][0][0] = 2.
    with pytest.raises(ValueError, match='exact eight-dimensional identity'):
        study.solve_saved_problem(data)


def parent_fixture(tmp_path, monkeypatch):
    study = load_study(monkeypatch); source, out = tmp_path / 'parent', tmp_path / 'result'; source.mkdir()
    files = {'pre-restoration.json': study.custody._bytes({'contract': {'method': 'two-frame-minimum-norm-curvature-restoration-v1'}})}
    for frame in (240, 244): files[f'frames/frame-{frame:05d}.json'] = study.custody._bytes({'frame': frame, 'corrections': [{'problem': problem()}]})
    for name, raw in files.items():
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    (source / 'manifest.json').write_text(json.dumps({'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}))
    return study, source, out


def test_two_problem_study_seals_exact_inputs_without_geometry(tmp_path, monkeypatch):
    study, source, out = parent_fixture(tmp_path, monkeypatch)
    import weft.arm_equipment_refinement as geometry
    monkeypatch.setattr(geometry, '_Geometry', lambda *a, **k: pytest.fail('Saved matrices reached geometry'))
    calls = []; original = study.solve_saved_problem
    def solve(data): calls.append(1); return original(data)
    monkeypatch.setattr(study, 'solve_saved_problem', solve)
    report = study.run(source, out)
    assert calls == [1, 1] and report['geometry_queries'] == report['full_source_fits'] == 0
    assert report['status'] == 'complete_saved_matrix_diagnostic' and not report['candidate_installed']
    manifest = json.loads((out / 'manifest.json').read_text())
    assert report['readback_verified_files'] == len(manifest['files'])
    for name, expected in manifest['files'].items(): assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected


def test_archive_reservation_rejects_before_solver_or_output(tmp_path, monkeypatch):
    study, source, out = parent_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(study, 'MAX_RESULT_BYTES', study.MAX_OUTPUT_BYTES)
    monkeypatch.setattr(study, 'solve_saved_problem', lambda *a, **k: pytest.fail('Unreserved solver'))
    with pytest.raises(ValueError, match='reserves exceed'):
        study.run(source, out)
    assert not out.exists()


def test_budget_failure_preserves_failed_receipt(tmp_path, monkeypatch):
    study, source, out = parent_fixture(tmp_path, monkeypatch)
    def failed(*a, **k): raise TimeoutError('injected five-second budget')
    monkeypatch.setattr(study, 'solve_saved_problem', failed)
    with pytest.raises(TimeoutError): study.run(source, out)
    report = json.loads((out / 'report.json').read_text()); manifest = json.loads((out / 'manifest.json').read_text())
    assert report['status'] == 'failed' and manifest['custody'] == 'failure_receipt_only'
    assert not report['promoted']
