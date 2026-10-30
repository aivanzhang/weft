"""The new full-source profile retains the original custody and explicit nominal-box boundary."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from test_restored_arm_alignment_study import study_fixture


def screened_fixture(tmp_path, monkeypatch):
    _, source, out, files = study_fixture(tmp_path, monkeypatch)
    path = Path(__file__).resolve().parents[1] / 'scripts/studies/check_box_screened_arm_alignment.py'
    spec = importlib.util.spec_from_file_location('screened_arm_study', path)
    study = importlib.util.module_from_spec(spec); spec.loader.exec_module(study)
    return study, source, out, files


def test_one_original_source_call_has_frozen_contract_inputs_and_readback(tmp_path, monkeypatch):
    study, source, out, files = screened_fixture(tmp_path, monkeypatch)
    calls = []; actual = study.propose_box_screened_arm_alignment
    def propose(*args, **kwargs):
        plan = json.loads((out / 'pre-fit.json').read_text())
        assert plan['contract'] == study.CONTRACT
        assert plan['contract']['method'] == 'box-screened-curvature-restored-arm-alignment-v1'
        assert plan['contract']['maximum_fit_seconds'] == 30. and plan['maximum_study_seconds'] == 45
        assert 'authored proxy' in plan['geometry_version'] and 'not the later sam-3 compound' in plan['geometry_version']
        assert 'numerical feasibility is not physical recovery' in plan['scope']
        assert plan['candidate_count'] == 1 and plan['paid_calls'] == plan['physics_steps'] == 0
        assert args == (files['human.json'], files['construction-evidence.json'])
        calls.append(1); return actual(*args, **kwargs)
    monkeypatch.setattr(study, 'propose_box_screened_arm_alignment', propose)
    report = study.run(source, out)
    assert calls == [1] and report['input_and_implementation_unchanged']
    assert 'screened_initializer_status_counts' in report
    assert report['geometry_admission'].startswith('unadmitted') and not report['promoted']
    manifest = json.loads((out / 'manifest.json').read_text())
    assert report['readback_verified_files'] == len(manifest['files'])
    for name, expected in manifest['files'].items(): assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    for name, raw in files.items(): assert (out / 'inputs' / name).read_bytes() == raw


def test_complete_output_reservation_precedes_creation_and_fit(tmp_path, monkeypatch):
    study, source, out, _ = screened_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(study.custody, 'MAX_OUTPUT_BYTES', study.MAX_PROPOSAL_BYTES)
    monkeypatch.setattr(study, 'propose_box_screened_arm_alignment', lambda *a, **k: pytest.fail('Unreserved source fit'))
    with pytest.raises(ValueError, match='archive plus proposal/metadata'): study.run(source, out)
    assert not out.exists()


def test_full_source_fit_failure_retains_failed_custody_without_retry(tmp_path, monkeypatch):
    study, source, out, _ = screened_fixture(tmp_path, monkeypatch); calls = []
    def fail(*args, **kwargs): calls.append(1); raise TimeoutError('Injected bounded fit exhaustion')
    monkeypatch.setattr(study, 'propose_box_screened_arm_alignment', fail)
    with pytest.raises(TimeoutError): study.run(source, out)
    assert calls == [1]
    report = json.loads((out / 'report.json').read_text())
    assert report['status'] == 'failed' and report['custody'] == 'failure_receipt_only'
    assert not report['promoted'] and report['unsealed_outputs_preserved']


def test_project_config_change_does_not_publish_success(tmp_path, monkeypatch):
    study, source, out, _ = screened_fixture(tmp_path, monkeypatch)
    actual, read = study.propose_box_screened_arm_alignment, Path.read_bytes
    changed = False
    def propose(*args, **kwargs):
        nonlocal changed
        result = actual(*args, **kwargs); changed = True; return result
    def read_bytes(path):
        raw = read(path)
        return raw + b'\n' if changed and path.name == 'pyproject.toml' and path.parent.name == 'weft' else raw
    monkeypatch.setattr(study, 'propose_box_screened_arm_alignment', propose)
    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    with pytest.raises(ValueError, match='configuration changed'): study.run(source, out)
    report = json.loads((out / 'report.json').read_text())
    assert report['status'] == 'failed' and 'project/pyproject.toml' in report['error']
    assert (out / 'proposal.json').is_file()


def test_compound_geometry_cannot_enter_nominal_box_method(tmp_path, monkeypatch):
    study, source, _, _ = screened_fixture(tmp_path, monkeypatch)
    scene = json.loads((source / 'scene.json').read_text()); scene['objects'][0]['shape'] = 'compound'
    with pytest.raises(ValueError, match='not replacement/compound'):
        study._boxes(scene, scene['objects'][0]['provenance']['camera_hypothesis'])
