"""Saved-matrix Phase I must preserve its problem and fail closed before source geometry."""

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
    spec = importlib.util.spec_from_file_location('phase_one_study', scripts / 'check_arm_phase_one.py')
    study = importlib.util.module_from_spec(spec); spec.loader.exec_module(study)
    return study


def problem():
    coupled = np.zeros((2, 8)); coupled[:, :2] = [[1., 1.], [-1., 1.]]
    matrix = np.vstack((coupled, np.eye(8), -np.eye(8)))
    return {'hessian': np.eye(8).tolist(), 'gradient': [0.] * 8, 'matrix': matrix.tolist(),
        'lower': [.6, -.4] + [-1.] * 16, 'initial': [1.] + [0.] * 7,
        'constraint_rows': [{'kind': 'pair_gap'}] * 2 + [{'kind': 'correction_lower'}] * 8 + [{'kind': 'correction_upper'}] * 8,
        'linear_pair_tolerance_spending': 0.}


def test_phase_one_then_exact_minimum_norm_problem(monkeypatch):
    study = load_study(monkeypatch); data = problem(); original = deepcopy(data)
    result = study.solve_saved_problem(data)
    assert result['status'] == 'verified_minimum_norm_linear_restoration'
    np.testing.assert_allclose(result['qualified_linear_correction'], [.3, .3] + [0.] * 6, atol=1e-10, rtol=0)
    assert result['independent_checks']['primal']['passes'] and data == original
    assert result['nonlinear_admission'] == 'not_executed' and not result['candidate_installed']


@pytest.mark.parametrize('change', ['missing', 'scaled', 'duplicate'])
def test_original_coordinate_bound_rows_cannot_be_replaced(monkeypatch, change):
    study = load_study(monkeypatch); data = problem()
    if change == 'missing': data['constraint_rows'][2] = {'kind': 'pair_gap'}
    if change == 'scaled': data['matrix'][2][0] = 2.
    if change == 'duplicate': data['matrix'][3] = data['matrix'][2].copy()
    with pytest.raises(ValueError): study.exact_box_bounds(data)


def test_unqualified_phase_point_cannot_reach_active_solver(monkeypatch):
    study = load_study(monkeypatch)
    monkeypatch.setattr(study, 'find_bounded_feasible_point', lambda *a, **k: {'point': [0.] * 8})
    monkeypatch.setattr(study, 'solve_feasible_qp', lambda *a, **k: pytest.fail('Invalid initializer reached QP'))
    with pytest.raises(ValueError, match='independent original-unit'):
        study.solve_saved_problem(problem())


def parent_fixture(tmp_path, monkeypatch):
    study = load_study(monkeypatch); data = problem()
    encode = study.previous.custody._bytes; sha = lambda raw: hashlib.sha256(raw).hexdigest()
    def seal(name, files):
        directory = tmp_path / name; directory.mkdir()
        for relative, raw in files.items():
            path = directory / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        manifest = encode({'files': {name: sha(raw) for name, raw in files.items()}})
        (directory / 'manifest.json').write_bytes(manifest)
        return directory, sha(manifest)
    restored, manifest = seal('restoration', {'pre-restoration.json': b'{}',
        'frames/frame-00240.json': encode({'frame': 240, 'corrections': [{'problem': data}]})})
    initializer, _ = seal('initializer', {'pre-initializer.json': encode({'parent': str(restored), 'parent_manifest_sha256': manifest}),
        'results.json': encode({'240': {'problem_sha256': study.canonical_hash(data)}})})
    return study, initializer, tmp_path / 'phase-one'


def test_exact_parent_chain_is_sealed_after_one_matrix_solve(tmp_path, monkeypatch):
    study, source, out = parent_fixture(tmp_path, monkeypatch)
    import weft.arm_equipment_refinement as geometry
    monkeypatch.setattr(geometry, '_Geometry', lambda *a, **k: pytest.fail('Saved matrices reached geometry'))
    calls = []; original = study.solve_saved_problem
    def solve(data): calls.append(1); return original(data)
    monkeypatch.setattr(study, 'solve_saved_problem', solve)
    report = study.run(source, out)
    assert calls == [1] and report['geometry_queries'] == report['full_source_fits'] == 0
    assert report['status'] == 'verified_minimum_norm_linear_restoration'
    assert not report['candidate_installed'] and not report['global_infeasibility_claim']
    plan = json.loads((out / 'pre-phase-one.json').read_text())
    assert set(plan['input_studies']) == {'initializer', 'restoration'}
    manifest = json.loads((out / 'manifest.json').read_text())
    assert report['readback_verified_files'] == len(manifest['files'])
    for name, expected in manifest['files'].items(): assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected


def test_changed_ancestor_seal_rejects_before_output_or_solver(tmp_path, monkeypatch):
    study, source, out = parent_fixture(tmp_path, monkeypatch)
    (tmp_path / 'restoration/manifest.json').write_bytes(b'{}')
    monkeypatch.setattr(study, 'solve_saved_problem', lambda *a, **k: pytest.fail('Changed seal reached solver'))
    with pytest.raises(ValueError): study.run(source, out)
    assert not out.exists()


def test_output_reservation_rejects_before_solver_or_mkdir(tmp_path, monkeypatch):
    study, source, out = parent_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(study, 'MAX_RESULT_BYTES', study.MAX_OUTPUT_BYTES)
    monkeypatch.setattr(study, 'solve_saved_problem', lambda *a, **k: pytest.fail('Unreserved solver'))
    with pytest.raises(ValueError, match='reserves exceed'): study.run(source, out)
    assert not out.exists()


def test_solver_budget_failure_preserves_failed_receipt(tmp_path, monkeypatch):
    study, source, out = parent_fixture(tmp_path, monkeypatch)
    def failed(*a, **k): raise TimeoutError('injected phase-one budget')
    monkeypatch.setattr(study, 'solve_saved_problem', failed)
    with pytest.raises(TimeoutError): study.run(source, out)
    report = json.loads((out / 'report.json').read_text()); manifest = json.loads((out / 'manifest.json').read_text())
    assert report['status'] == 'failed' and manifest['custody'] == 'failure_receipt_only'
    assert not report['promoted']
