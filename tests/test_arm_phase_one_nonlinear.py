"""Four-study custody; synthetic mocked geometry is never reported as source acceptance."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def fixture(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts/studies'; monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('phase_nonlinear_study', scripts / 'check_arm_phase_one_nonlinear.py')
    study = importlib.util.module_from_spec(spec); spec.loader.exec_module(study)
    encode = study.restoration.local.custody._bytes; sha = lambda raw: hashlib.sha256(raw).hexdigest()
    def seal(name, files):
        directory = tmp_path / name; directory.mkdir()
        for relative, raw in files.items():
            path = directory / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        manifest = encode({'files': {name: sha(raw) for name, raw in files.items()}})
        (directory / 'manifest.json').write_bytes(manifest)
        return directory, sha(manifest), files | {'manifest.json': manifest}
    names = [f'joint{i}' for i in range(8)]
    candidate = {'reference': [{'time_s': t, 'joints': dict.fromkeys(names, 0.)} for t in (0., 8.)]}
    prefit = {'human_entity_ref': {'document_sha256': 'a' * 64, 'object_id': 'synthetic'}, 'camera_hypothesis': {}, 'equipment_boxes': []}
    inputs = {'proposal.json': encode({'human_candidate': candidate}), 'pre-fit.json': encode(prefit),
              'inputs/human.json': encode(candidate), 'inputs/construction-evidence.json': b'{}', 'inputs/human.xml': b'<mujoco/>'}
    captured = {'candidate_model_sha256': study.canonical_hash(candidate), 'arm_contract': study.arm.CONTRACT,
                'input_sha256': {name: sha(raw) for name, raw in inputs.items()}, **prefit}
    matrix = encode({'frame': 240, 'variables': names, 'current_angle_correction_rad': [0.] * 8})
    active, active_hash, _ = seal('active', {'pre-capture.json': encode(captured),
        'comparisons.json': encode({'comparisons': [{'frame': 240, 'matrix_sha256': sha(matrix),
            'active_set': {'status': 'optimal', 'step': [0.] * 8}}]}), 'subproblems/frame-00240.json': matrix,
        **{'inputs/parent/' + name: raw for name, raw in inputs.items()}})
    problem = {'matrix': np.vstack((np.eye(8), -np.eye(8))).tolist(), 'lower': [-.1] * 16,
        'constraint_rows': [{'kind': 'correction_lower'}] * 8 + [{'kind': 'correction_upper'}] * 8}
    restored, restored_hash, restored_files = seal('restored', {'pre-restoration.json': encode({'parent': str(active), 'parent_manifest_sha256': active_hash}),
        'frames/frame-00240.json': encode({'frame': 240, 'reference_index': 1, 'direction_scale': 1 / 32,
            'saved_direction_rad': [0.] * 8, 'corrections': [{'problem': problem, 'trial_delta_rad': [0.] * 8}]})})
    initializer, initializer_hash, initializer_files = seal('initializer', {
        'pre-initializer.json': encode({'parent': str(restored), 'parent_manifest_sha256': restored_hash}),
        'results.json': encode({'240': {'problem_sha256': study.canonical_hash(problem)}})})
    source, _, _ = seal('phase-one', {'pre-phase-one.json': encode({'frame': 240, 'phase_one_contract': study.phase.CONTRACT,
        'problem_sha256': study.canonical_hash(problem),
        'input_studies': {'initializer': {'path': str(initializer), 'manifest_sha256': initializer_hash},
                         'restoration': {'path': str(restored), 'manifest_sha256': restored_hash}},
        'input_sha256': {label + '/' + name: sha(raw) for label, files in [('initializer', initializer_files), ('restoration', restored_files)]
                         for name, raw in files.items()}}),
        'result.json': encode({'status': 'verified_minimum_norm_linear_restoration', 'problem_sha256': study.canonical_hash(problem),
            'qualified_linear_correction': [0.] * 8, 'active_set': {'status': 'optimal', 'step': [0.] * 8}})})
    calls = []
    def prepare(*args, **kwargs):
        assert kwargs['prepare_only'] and kwargs['frames'] == (240,)
        calls.append('prepare')
        return {'indices': [1], 'changes': np.zeros((2, 8)), 'rows': [{}, {}], 'source_rows': {0: {}, 1: {}},
            'geometry': SimpleNamespace(arm_names=names), 'evaluate': lambda *a: (1., None, None, True, None, None, None)}
    def inspect(*args):
        calls.append('inspect')
        return {'local_step_acceptable': False, 'pair_comparisons': 180, 'affected_unique_samples': 5,
            'maximum_violation_model_units': .01, 'cost': 2., 'baseline_cost': 1.,
            'maximum_angle_correction_rad': 0., 'maximum_correction_rate_rad_s': 0.,
            'maximum_endpoint_depth_change_model_units': 0., 'pair_checks': [{} for _ in range(180)],
            'scope': 'mocked rejected custody fixture'}
    monkeypatch.setattr(study.restoration.local, 'capture_local_problems', prepare)
    monkeypatch.setattr(study.restoration, 'inspect_local', inspect)
    monkeypatch.setattr(study.restoration, 'solve_feasible_qp', lambda *a, **k: pytest.fail('Audit called a solver'))
    monkeypatch.setattr(study.phase, 'find_bounded_feasible_point', lambda *a, **k: pytest.fail('Audit called Phase I'))
    return study, source, tmp_path / 'audit', calls


def test_four_exact_seals_loaded_for_one_complete_nonlinear_check(tmp_path, monkeypatch):
    study, source, out, calls = fixture(tmp_path, monkeypatch)
    report = study.run(source, out)
    assert calls == ['prepare', 'inspect'] and report['status'] == 'local_step_rejected'
    assert report['new_solver_calls'] == report['new_qp_captures'] == report['full_source_fits'] == 0
    assert not report['candidate_installed'] and report['other_frames_status'] == 'not_modified_or_audited'
    plan = json.loads((out / 'pre-audit.json').read_text())
    assert set(plan['input_studies']) == {'phase-one', 'initializer', 'restoration', 'active-set'}
    manifest = json.loads((out / 'manifest.json').read_text())
    assert report['readback_verified_files'] == len(manifest['files'])
    for name, digest in manifest['files'].items(): assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_changed_parent_seal_rejects_before_geometry(tmp_path, monkeypatch):
    study, source, out, calls = fixture(tmp_path, monkeypatch)
    (tmp_path / 'restored/manifest.json').write_bytes(b'{}')
    with pytest.raises(ValueError): study.run(source, out)
    assert calls == [] and not out.exists()


def test_sealed_but_mismatched_correction_cannot_reach_geometry(tmp_path, monkeypatch):
    study, source, out, calls = fixture(tmp_path, monkeypatch)
    result = json.loads((source / 'result.json').read_bytes()); result['qualified_linear_correction'][0] = .01
    raw = study.restoration.local.custody._bytes(result); (source / 'result.json').write_bytes(raw)
    manifest = json.loads((source / 'manifest.json').read_bytes()); manifest['files']['result.json'] = hashlib.sha256(raw).hexdigest()
    (source / 'manifest.json').write_bytes(study.restoration.local.custody._bytes(manifest))
    with pytest.raises(ValueError, match='correction origin differs'): study.run(source, out)
    assert calls == [] and not out.exists()


def test_reserved_output_bound_rejects_before_geometry(tmp_path, monkeypatch):
    study, source, out, calls = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(study, 'MAX_RESULT_BYTES', study.MAX_OUTPUT_BYTES)
    with pytest.raises(ValueError, match='reserves exceed'): study.run(source, out)
    assert calls == [] and not out.exists()


def test_missing_pair_rows_preserve_failed_receipt(tmp_path, monkeypatch):
    study, source, out, calls = fixture(tmp_path, monkeypatch)
    inspect = study.restoration.inspect_local
    def incomplete(*args):
        result = inspect(*args); result['pair_checks'].pop(); return result
    monkeypatch.setattr(study.restoration, 'inspect_local', incomplete)
    with pytest.raises(ValueError, match='all 180 pair checks'): study.run(source, out)
    report = json.loads((out / 'report.json').read_bytes()); manifest = json.loads((out / 'manifest.json').read_bytes())
    assert report['status'] == 'failed' and manifest['custody'] == 'failure_receipt_only'
