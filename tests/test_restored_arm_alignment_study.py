"""Opt-in restored source-study publication keeps original-box scope and complete custody."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from test_arm_equipment_refinement import fixture


def study_fixture(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts/studies'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('restored_study', scripts / 'check_restored_arm_alignment.py')
    study = importlib.util.module_from_spec(spec); spec.loader.exec_module(study)
    human, evidence, options = fixture()
    raw = json.dumps(human).encode(); camera = options['camera_hypothesis']; entity = options['human_entity_ref']
    box = options['equipment_boxes'][0]
    scene = {'human_spec': human, 'coordinate_frame': {'scale_bridge': {'body_height_to_world_scale': 1., 'origin_world': [0., 0., 0.]}},
             'objects': [{'id': box['id'], 'shape': 'box', 'quaternion': [1., 0., 0., 0.], 'position': box['center'],
                 'size': [h * 2 for h in box['half_extents']],
                 'provenance': {'camera_hypothesis': camera, 'geometry_model_units': {k: box[k] for k in ('center', 'half_extents')}}}],
             'source_binding': {'entity_bindings': [{'world_entity_id': 'human', 'observation_entity_id': entity['object_id'],
                 'observation_document_sha256': entity['document_sha256']}]}}
    files = {'human.json': raw, 'construction-evidence.json': evidence, 'scene.json': json.dumps(scene).encode(),
             'geometry-proposal.json': json.dumps({'camera_hypothesis': camera, 'body_binding': {'human_json_sha256': hashlib.sha256(raw).hexdigest()}}).encode(),
             'source.json': b'{"license":"authored synthetic fixture"}'}
    source, out = tmp_path / 'source', tmp_path / 'study'; source.mkdir()
    for name, value in files.items(): (source / name).write_bytes(value)
    (source / 'manifest.json').write_text(json.dumps({'files': {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}}))
    return study, source, out, files


def test_one_frozen_fit_seals_exact_inputs_configuration_and_readback(tmp_path, monkeypatch):
    study, source, out, files = study_fixture(tmp_path, monkeypatch)
    calls = []; original = study.propose_restored_arm_alignment
    def proposed(*args, **kwargs):
        plan = json.loads((out / 'pre-fit.json').read_text())
        assert plan['contract'] == study.CONTRACT
        assert 'not the later sam-3 compound' in plan['geometry_version']
        assert plan['candidate_count'] == 1 and plan['paid_calls'] == 0
        assert plan['contract']['method'] == 'active-set-curvature-restored-arm-alignment-v1'
        calls.append(1); return original(*args, **kwargs)
    monkeypatch.setattr(study, 'propose_restored_arm_alignment', proposed)
    report = study.run(source, out)
    assert calls == [1] and report['input_and_implementation_unchanged']
    assert report['numerical_summary']['direction_qp_calls'] == 30
    assert sum(report['direction_solver_status_counts'].values()) == report['projection_attempts']
    manifest = json.loads((out / 'manifest.json').read_text())
    assert report['readback_verified_files'] == len(manifest['files'])
    for name, expected in manifest['files'].items(): assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    for name, raw in files.items(): assert (out / 'inputs' / name).read_bytes() == raw
    assert not report['promoted'] and report['geometry_admission'].startswith('unadmitted')


def test_preflight_output_reservation_prevents_creation_and_fit(tmp_path, monkeypatch):
    study, source, out, _ = study_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(study.custody, 'MAX_OUTPUT_BYTES', study.MAX_PROPOSAL_BYTES)
    monkeypatch.setattr(study, 'propose_restored_arm_alignment', lambda *a, **k: pytest.fail('Unreserved fit'))
    with pytest.raises(ValueError, match='archive plus proposal/metadata'):
        study.run(source, out)
    assert not out.exists()


def test_project_configuration_change_publishes_only_failed_receipt(tmp_path, monkeypatch):
    study, source, out, _ = study_fixture(tmp_path, monkeypatch)
    original_fit, original_read = study.propose_restored_arm_alignment, Path.read_bytes
    changed = False
    def proposed(*args, **kwargs):
        nonlocal changed
        result = original_fit(*args, **kwargs); changed = True; return result
    def read(path):
        raw = original_read(path)
        return raw + b'\n' if changed and path.name == 'pyproject.toml' and path.parent.name == 'weft' else raw
    monkeypatch.setattr(study, 'propose_restored_arm_alignment', proposed)
    monkeypatch.setattr(Path, 'read_bytes', read)
    with pytest.raises(ValueError, match='implementation/configuration changed'):
        study.run(source, out)
    manifest = json.loads((out / 'manifest.json').read_text()); report = json.loads((out / 'report.json').read_text())
    assert manifest['custody'] == report['custody'] == 'failure_receipt_only'
    assert set(manifest['files']) == {'report.json', 'failure.json'}
    assert 'project/pyproject.toml' in report['error']
    assert (out / 'proposal.json').exists()


def test_compound_geometry_is_rejected_not_silently_dropped(tmp_path, monkeypatch):
    study, source, _, _ = study_fixture(tmp_path, monkeypatch)
    scene = json.loads((source / 'scene.json').read_text())
    scene['objects'][0]['shape'] = 'compound'
    with pytest.raises(ValueError, match='not replacement/compound'):
        study._boxes(scene, scene['objects'][0]['provenance']['camera_hypothesis'])
