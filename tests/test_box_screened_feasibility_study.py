"""The matrix study binds saved seeds/boxes and retains failures without geometry or baseline reruns."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def study_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/studies/check_box_screened_feasibility.py'
    spec = importlib.util.spec_from_file_location('screened_study', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def proposal(study):
    diagnostics, restorations = [], []
    for index in range(4):
        a = np.zeros((3, 8)); a[0, :2] = [1., 1.]; a[1, :2] = [-1., 1.]; a[2, 0] = (index + 1) * 1e-19
        lo, hi = np.array([0.] + [-1.] * 7), np.ones(8)
        a = np.vstack((a, np.eye(8), -np.eye(8))); b = np.r_[.6, -.4, -.6, lo, -hi]
        problem = {'kind': 'restoration', 'hessian': np.eye(8).tolist(), 'gradient': [0.] * 8, 'matrix': a.tolist(), 'lower': b.tolist()}
        identity = study.canonical_hash(problem)
        diagnostics.append({'sha256': identity, 'problem': problem})
        restorations.append({'reference_index': index, 'sweep': 0, 'rounds': [{'problem_sha256': identity,
            'constraint_rows': len(b), 'linear_pair_tolerance_spending': 0., 'trial_delta_rad': [-1.] + [0.] * 7,
            'iteration': 0, 'phase_one': {'status': 'normalization_bound', 'iterations': 0}}]})
    return {'report': {'profile': 'active-set-curvature-restored-arm-alignment-v1',
        'saved_diagnostic_problems': diagnostics, 'restorations': restorations}}


def parent_fixture(tmp_path):
    study = study_module(); data = proposal(study); source = tmp_path / 'parent'; source.mkdir()
    files = {'proposal.json': study._bytes(data), 'pre-fit.json': b'{}', 'report.json': b'{}'}
    study.write_package_files(source, files)
    manifest = study._bytes({'files': {name: study._sha(raw) for name, raw in files.items()}})
    study.write_package_files(source, {'manifest.json': manifest})
    return study, source, tmp_path / 'new-study', study._sha(manifest)


def test_saved_matrix_and_exact_seed_are_unchanged_and_fraction_verified():
    study = study_module(); data = proposal(study); original = deepcopy(data)
    saved = study.extract_problems(data)[0]; result = study.solve_saved_problem(saved)
    assert result['status'] == 'verified_original_feasible_initializer'
    assert result['independent_checks']['original_primal']['passes']
    assert result['independent_checks']['original_primal']['rows_checked'] == 19
    assert all(p['enclosure_valid'] for p in result['independent_checks']['redundancy_proofs'])
    assert result['helper']['report']['omitted_rows'] == [2]
    assert data == original and not result['candidate_installed']
    assert saved['seed'] == [1.] + [0.] * 7


@pytest.mark.parametrize('mutation', ['hash', 'box', 'seed', 'objective', 'match', 'tolerance'])
def test_wrong_saved_binding_rejects_before_a_solver(mutation):
    study = study_module(); data = proposal(study); report = data['report']
    diagnostic = report['saved_diagnostic_problems'][0]; row = report['restorations'][0]['rounds'][0]
    if mutation == 'hash': diagnostic['sha256'] = '0' * 64
    elif mutation == 'seed':
        other = deepcopy(report['restorations'][0]); other['rounds'][0]['trial_delta_rad'][0] = -2.
        report['restorations'].append(other)
    elif mutation == 'match': row['problem_sha256'] = '0' * 64
    elif mutation == 'tolerance': row['linear_pair_tolerance_spending'] = 1e-8
    else:
        if mutation == 'box': diagnostic['problem']['matrix'][-16][0] = 2.
        else: diagnostic['problem']['hessian'][0][0] = 2.
        diagnostic['sha256'] = row['problem_sha256'] = study.canonical_hash(diagnostic['problem'])
    with pytest.raises(ValueError): study.extract_problems(data)


def test_forged_redundancy_claim_fails_exact_fraction_verification():
    study = study_module(); saved = study.extract_problems(proposal(study))[0]
    result = study.solve_saved_problem(saved)['helper']
    result['report']['row_lower_enclosures'][0] = 2.
    result['report']['omitted_rows'].append(0)
    with pytest.raises(ValueError, match='exact-rational'): study.independent_checks(saved, result)


def test_four_calls_archive_all_inputs_and_results_without_baseline_or_geometry(tmp_path, monkeypatch):
    study, source, out, seal = parent_fixture(tmp_path); calls = []; original = study.solve_saved_problem
    def solve(saved): calls.append(saved['sha256']); return original(saved)
    monkeypatch.setattr(study, 'solve_saved_problem', solve)
    report = study.run(source, out, expected_manifest_sha256=seal)
    assert len(calls) == len(set(calls)) == 4
    assert report['qualified_initializers'] == report['helper_calls'] == 4
    assert report['geometry_queries'] == report['physics_steps'] == report['full_source_fits'] == 0
    assert not report['candidate_installed']
    assert (out / 'inputs/proposal.json').read_bytes() == (source / 'proposal.json').read_bytes()
    manifest = json.loads((out / 'manifest.json').read_text())
    for name, expected in manifest['files'].items(): assert study._sha((out / name).read_bytes()) == expected
    assert report['readback_verified_files'] == len(manifest['files'])


@pytest.mark.parametrize('failure', ['seal', 'reserve'])
def test_preflight_rejects_before_output_and_calls(tmp_path, monkeypatch, failure):
    study, source, out, seal = parent_fixture(tmp_path)
    monkeypatch.setattr(study, 'solve_saved_problem', lambda *a: pytest.fail('Invalid preflight reached solver'))
    if failure == 'reserve': monkeypatch.setattr(study, 'MAX_RESULT_BYTES', study.MAX_OUTPUT_BYTES)
    else: seal = '0' * 64
    with pytest.raises(ValueError): study.run(source, out, expected_manifest_sha256=seal)
    assert not out.exists()


def test_failed_second_call_preserves_first_result_and_bounded_receipt(tmp_path, monkeypatch):
    study, source, out, seal = parent_fixture(tmp_path); original = study.solve_saved_problem; count = [0]
    def solve(saved):
        count[0] += 1
        if count[0] == 2: raise TimeoutError('Injected study budget')
        return original(saved)
    monkeypatch.setattr(study, 'solve_saved_problem', solve)
    with pytest.raises(TimeoutError): study.run(source, out, expected_manifest_sha256=seal)
    assert count == [2] and len(list((out / 'results').glob('*.json'))) == 1
    report = json.loads((out / 'report.json').read_text()); manifest = json.loads((out / 'manifest.json').read_text())
    assert report['status'] == 'failed' and manifest['custody'] == 'failure_receipt_only'
    assert report['unsealed_outputs_preserved'] and not report['promoted']


def test_pinned_project_config_change_cannot_publish_success(tmp_path, monkeypatch):
    study, source, out, seal = parent_fixture(tmp_path)
    config = tmp_path / 'config'; config.write_bytes(b'frozen')
    monkeypatch.setattr(study, '_implementation_paths', lambda: {'project/config': config})
    original = study.solve_saved_problem
    def solve(saved):
        result = original(saved); config.write_bytes(b'changed'); return result
    monkeypatch.setattr(study, 'solve_saved_problem', solve)
    with pytest.raises(ValueError, match='configuration changed'): study.run(source, out, expected_manifest_sha256=seal)
    assert json.loads((out / 'report.json').read_text())['status'] == 'failed'


def test_post_seal_failure_preserves_prior_publication_and_explicit_failure(tmp_path, monkeypatch):
    study, source, out, seal = parent_fixture(tmp_path); original = study._publish
    def failed(out, report): original(out, report); raise ValueError('Injected final readback failure')
    monkeypatch.setattr(study, '_publish', failed)
    with pytest.raises(ValueError, match='readback failure'): study.run(source, out, expected_manifest_sha256=seal)
    assert (out / 'manifest.publication-failed.json').is_file()
    assert (out / 'report.publication-failed.json').is_file()
    assert json.loads((out / 'manifest.json').read_text())['custody'] == 'failure_receipt_only'
