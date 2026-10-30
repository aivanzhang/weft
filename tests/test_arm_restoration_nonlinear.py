"""Three-study custody for a single audit; mocked geometry never represents source success."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest


def fixture(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts/studies'; monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('nonlinear_study', scripts / 'check_arm_restoration_nonlinear.py')
    study = importlib.util.module_from_spec(spec); spec.loader.exec_module(study)
    encode = study.restoration.local.custody._bytes; sha = lambda raw: hashlib.sha256(raw).hexdigest()
    def seal(name, files):
        directory = tmp_path / name; directory.mkdir()
        for relative, raw in files.items():
            path = directory / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        manifest = encode({'files': {name: sha(raw) for name, raw in files.items()}})
        (directory / 'manifest.json').write_bytes(manifest)
        return directory, sha(manifest)
    names = [f'joint{i}' for i in range(8)]
    candidate = {'reference': [{'time_s': t, 'joints': dict.fromkeys(names, 0.)} for t in (0., 8.133333333333333)]}
    prefit = {'human_entity_ref': {'document_sha256': 'a' * 64, 'object_id': 'synthetic'}, 'camera_hypothesis': {}, 'equipment_boxes': []}
    inputs = {'proposal.json': encode({'human_candidate': candidate}), 'pre-fit.json': encode(prefit),
              'inputs/human.json': encode(candidate), 'inputs/construction-evidence.json': b'{}', 'inputs/human.xml': b'<mujoco/>'}
    captured = {'candidate_model_sha256': study.canonical_hash(candidate), 'arm_contract': study.arm.CONTRACT,
                'input_sha256': {name: sha(raw) for name, raw in inputs.items()}, **prefit}
    active, active_hash = seal('active', {'pre-capture.json': encode(captured),
        'comparisons.json': encode({'comparisons': [{'frame': 244, 'active_set': {'status': 'optimal', 'step': [0.] * 8}}]}),
        **{'inputs/parent/' + name: raw for name, raw in inputs.items()}})
    problem = {'matrix': np.eye(8).tolist(), 'lower': [-.1] * 8}
    restored, restored_hash = seal('restored', {'pre-restoration.json': encode({'parent': str(active), 'parent_manifest_sha256': active_hash}),
        'frames/frame-00244.json': encode({'frame': 244, 'reference_index': 1, 'direction_scale': 1 / 32,
            'saved_direction_rad': [0.] * 8, 'corrections': [{'problem': problem, 'trial_delta_rad': [0.] * 8}]})})
    source, _ = seal('initializer', {'pre-initializer.json': encode({'parent': str(restored), 'parent_manifest_sha256': restored_hash}),
        'results.json': encode({'244': {'status': 'verified_minimum_norm_linear_restoration', 'problem_sha256': study.canonical_hash(problem),
            'qualified_linear_correction': [0.] * 8, 'active_set': {'step': [0.] * 8}}})})
    calls = []
    def prepare(*args, **kwargs):
        assert kwargs['prepare_only'] and kwargs['frames'] == (244,)
        calls.append('prepare')
        return {'indices': [1], 'changes': np.zeros((2, 8)), 'rows': [{}, {}], 'source_rows': {0: {}, 1: {}},
            'geometry': SimpleNamespace(arm_names=names), 'evaluate': lambda *a: (1., None, None, True, None, None, None)}
    def inspect(*args):
        calls.append('inspect')
        return {'local_step_acceptable': False, 'pair_comparisons': 180, 'affected_unique_samples': 5,
            'maximum_violation_model_units': .01, 'cost': 2., 'baseline_cost': 1.,
            'maximum_angle_correction_rad': 0., 'maximum_correction_rate_rad_s': 0.,
            'maximum_endpoint_depth_change_model_units': 0., 'pair_checks': [], 'scope': 'mocked rejected custody fixture'}
    monkeypatch.setattr(study.restoration.local, 'capture_local_problems', prepare)
    monkeypatch.setattr(study.restoration, 'inspect_local', inspect)
    monkeypatch.setattr(study.restoration, 'solve_feasible_qp', lambda *a, **k: pytest.fail('Audit called a solver'))
    return study, source, tmp_path / 'audit', calls


def test_three_exact_seals_are_loaded_and_only_one_nonlinear_check_runs(tmp_path, monkeypatch):
    study, source, out, calls = fixture(tmp_path, monkeypatch)
    report = study.run(source, out)
    assert calls == ['prepare', 'inspect'] and report['status'] == 'local_step_rejected'
    assert report['new_solver_calls'] == report['new_qp_captures'] == report['full_source_fits'] == 0
    assert not report['candidate_installed'] and report['frame240_status'] == 'unresolved_not_audited'
    plan = json.loads((out / 'pre-audit.json').read_text())
    assert set(plan['input_studies']) == {'initializer', 'restoration', 'active-set'}
    manifest = json.loads((out / 'manifest.json').read_text())
    assert report['readback_verified_files'] == len(manifest['files'])
    for name, digest in manifest['files'].items(): assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_changed_parent_seal_rejects_before_geometry(tmp_path, monkeypatch):
    study, source, out, calls = fixture(tmp_path, monkeypatch)
    (tmp_path / 'restored' / 'manifest.json').write_bytes(b'{}')
    with pytest.raises(ValueError): study.run(source, out)
    assert calls == [] and not out.exists()


def test_reserved_output_bound_rejects_before_geometry(tmp_path, monkeypatch):
    study, source, out, calls = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(study, 'MAX_RESULT_BYTES', study.MAX_OUTPUT_BYTES)
    with pytest.raises(ValueError, match='reserves exceed'):
        study.run(source, out)
    assert calls == [] and not out.exists()
