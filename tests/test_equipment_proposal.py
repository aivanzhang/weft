"""Unfamiliar saved observations reach the strict constructor without study directories."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from test_shared_world_reconstruction import fixture, raw
from weft import equipment_proposal as proposal
from weft.io import digest, write_json
from weft.shared_world import validate_shared_world
from weft.shared_world_evidence import build_shared_world_binding
from weft.shared_world_pipeline import construct_source_equipment_run
from weft.shared_world_reconstruction import construct_equipment_world


def inputs(*, overlap=False):
    base, old, options, documents = fixture(overlap=overlap, include_documents=True)
    source = json.loads(next(iter(documents.values())))
    names = {'equipment-a': 'foam-pad', 'equipment-b': 'training-weight'}
    # Replace synthetic entity identities, including references, without changing source frames or values.
    document = json.dumps(source)
    for before, after in names.items(): document = document.replace(before, after)
    document = document.encode(); document_sha = hashlib.sha256(document).hexdigest()
    documents = {document_sha: document}
    base_ids = ['human', *(o['id'] for o in base['objects'])]
    base['source_binding'], _ = build_shared_world_binding(documents, source_sha256=base['human_spec']['source_sha256'],
        world_ids=base_ids, entity_refs={name: (document_sha, 'person') if name == 'human' else None for name in base_ids})
    base = validate_shared_world(base)
    binding, construction = build_shared_world_binding(documents, source_sha256=base['human_spec']['source_sha256'],
        world_ids=['human', *names.values()], entity_refs={name: (document_sha, 'person' if name == 'human' else name)
                                                         for name in ['human', *names.values()]})
    config = {'schema_version': 1, 'profile': proposal.CONFIG_PROFILE,
        'source_sha256': base['human_spec']['source_sha256'],
        'camera': {'mode': 'fixed_pelvis_perspective_prior', 'distance_model_units': 2.},
        'geometry_prior': 'floor_box_depth_equals_width',
        'equipment': [{'object_id': 'foam-pad', 'source_frame': 12}, {'object_id': 'training-weight', 'source_frame': 18}],
        'target_object_id': 'foam-pad'}
    kwargs = {'source_binding': binding, 'construction_evidence': construction, 'human_document': options['human_document']}
    return base, config, kwargs, documents, old


def seal(directory):
    write_json(directory / 'manifest.json', {'schema_version': 1, 'files': {
        p.relative_to(directory).as_posix(): digest(p) for p in directory.rglob('*') if p.is_file() and p.name != 'manifest.json'}})


def packages(root, *, overlap=False):
    base, config, kwargs, documents, _ = inputs(overlap=overlap)
    body, equipment = root / 'body', root / 'equipment'
    body.mkdir(); equipment.mkdir()
    write_json(body / 'scene.json', base); (body / 'human.json').write_bytes(kwargs['human_document'])
    for sha, name in base['source_binding']['evidence_documents'].items():
        path = body / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(documents[sha])
    (equipment / 'scene-observations.json').write_bytes(next(iter(documents.values())))
    write_json(equipment / 'report.json', {'status': 'evidence_only', 'physics_ready': False})
    seal(body); seal(equipment)
    config_path = root / 'config.json'; config_path.write_text(json.dumps(config, indent=1) + '\n\n')
    return body, equipment, config_path, base, config


@pytest.mark.parametrize('center_x', [-.6, 0., .6])
def test_analytic_inverse_preserves_all_eight_vertex_bounds_in_each_projection_branch(center_x):
    base, config, kwargs, _, old = inputs()
    camera = old['camera_hypothesis']; center = np.array([center_x, .45, .12]); half = np.array([.1, .1, .12])
    bounds = proposal._project_box(center, half, camera, base['human_spec']['mobility']['base_height']).tolist()
    result = proposal.floor_box_from_bounds(bounds, camera, base['human_spec']['mobility']['base_height'], 0.)
    np.testing.assert_allclose(result['center_model_units'], center, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result['half_extents_model_units'], half, atol=1e-12, rtol=0)
    assert max(abs(v) for v in result['construction_bbox_edge_error_px']) < 1e-10


def test_unfamiliar_entity_names_and_explicit_frames_reach_unchanged_constructor():
    base, config, kwargs, _, old = inputs()
    original = deepcopy((base, config, kwargs))
    candidate = proposal.build_equipment_proposal(base, config, **kwargs)
    scene = construct_equipment_world(base, candidate, **kwargs, target_object_id=config['target_object_id'])
    assert (base, config, kwargs) == original
    assert [obj['id'] for obj in scene['objects']] == ['foam-pad', 'training-weight']
    assert [obj['source_frame'] for obj in candidate['equipment']] == [12, 18]
    for before, after in zip(old['equipment'], candidate['equipment']):
        np.testing.assert_allclose(before['geometry']['center_model_units'], after['geometry']['center_model_units'], atol=1e-12, rtol=0)
    assert scene['human_spec'] == base['human_spec']
    for obj in scene['objects']:
        assert obj['velocity'] == obj['angular_velocity'] == obj['provenance']['reference_object_force'] == [0., 0., 0.]
        assert not obj['provenance']['source_reference']['object_id'].startswith('sam-')
    assert len(json.loads(kwargs['construction_evidence'])['sources'][0]['observations']) == 63 * 3


def test_sealed_canonical_export_proposal_constructs_without_masks_studies_or_inference(tmp_path):
    body, equipment, config_path, original, config = packages(tmp_path)
    before = {p: p.read_bytes() for directory in (body, equipment) for p in directory.rglob('*') if p.is_file()}
    config_bytes = config_path.read_bytes(); out = tmp_path / 'proposal'
    report = proposal.prepare_equipment_proposal(body, equipment, config_path, out)
    assert report['status'] == 'proposal_ready' and report['reasons'] == []
    assert (out / 'input-config.json').read_bytes() == config_bytes
    assert (out / 'geometry-proposal.json').is_file() and not (out / 'geometry-candidate.rejected.json').exists()
    assert report['construction_ready_arguments']['target_object_id'] == 'foam-pad'
    assert report['construction_ready_arguments']['frame_translation'] is None
    assert report['full_reference_clearance_checked'] is False and report['promoted'] is False
    assert report['source_binding']['total_source_frames'] == 63
    constructor = tmp_path / 'constructed'
    construct_source_equipment_run(body, out, equipment, constructor, target_object_id=config['target_object_id'])
    scene = json.loads((constructor / 'scene.json').read_bytes())
    assert scene['human_spec'] == original['human_spec']
    assert all(path.read_bytes() == value for path, value in before.items())
    manifest = json.loads((out / 'manifest.json').read_bytes())
    assert all(digest(out / name) == sha for name, sha in manifest['files'].items())
    assert all(report['contract'][name] == 0 for name in ('fits', 'inference_calls', 'api_calls', 'video_decodes'))


def test_initial_overlap_is_sealed_as_nonready_and_cannot_feed_standard_constructor(tmp_path):
    body, equipment, config_path, _, config = packages(tmp_path, overlap=True)
    out = tmp_path / 'rejected'
    report = proposal.prepare_equipment_proposal(body, equipment, config_path, out)
    assert report['status'] == 'rejected_initial_geometry'
    assert report['construction_ready_arguments'] is None
    assert (out / 'geometry-candidate.rejected.json').is_file()
    assert not (out / 'geometry-proposal.json').exists()
    with pytest.raises(ValueError, match='not sealed'):
        construct_source_equipment_run(body, out, equipment, tmp_path / 'must-not-construct', target_object_id=config['target_object_id'])


@pytest.mark.parametrize('change', ['source', 'camera', 'dimension_prior', 'duplicate', 'frame', 'target', 'negative_translation', 'extra'])
def test_bad_configuration_has_a_sealed_reviewable_unsupported_state(tmp_path, change):
    body, equipment, config_path, _, config = packages(tmp_path)
    if change == 'source': config['source_sha256'] = 'b' * 64
    elif change == 'camera': config['camera']['mode'] = 'calibrated_intrinsics'
    elif change == 'dimension_prior': config['geometry_prior'] = 'handheld_sphere'
    elif change == 'duplicate': config['equipment'].append(deepcopy(config['equipment'][0]))
    elif change == 'frame': config['equipment'][0]['source_frame'] = 1
    elif change == 'target': config['target_object_id'] = 'unknown'
    elif change == 'negative_translation': config['frame_translation_xy'] = [-.2, 0.]
    elif change == 'extra': config['assume_contact'] = True
    write_json(config_path, config); out = tmp_path / 'unsupported'
    result = proposal.prepare_equipment_proposal(body, equipment, config_path, out)
    assert result['status'] == 'unsupported_input' and result['reasons']
    assert (out / 'manifest.json').is_file() and (out / 'input-config.json').read_bytes() == config_path.read_bytes()
    assert not (out / 'geometry-proposal.json').exists()


def test_withheld_selected_frame_is_not_replaced_or_leaked():
    base, config, kwargs, _, _ = inputs()
    evidence = json.loads(kwargs['construction_evidence'])
    row = next(r for r in evidence['sources'][0]['observations'] if r['object_id'] == 'foam-pad' and r['frame'] == 12)
    row.update(partition='audit', admission='withheld', measurements={}, evidence_refs=[], visibility=None)
    entity = next(r for r in kwargs['source_binding']['entity_bindings'] if r['world_entity_id'] == 'foam-pad')
    entity['observed_slots'] -= 1; entity['missing_slots'] += 1
    kwargs['construction_evidence'] = raw(evidence)
    with pytest.raises(proposal.EquipmentProposalError, match='admitted construction'):
        proposal.build_equipment_proposal(base, config, **kwargs)


def test_missing_torso_reference_is_not_interpolated_or_dropped():
    base, config, kwargs, _, _ = inputs()
    evidence = json.loads(kwargs['construction_evidence'])
    row = next(r for r in evidence['sources'][0]['observations'] if r['object_id'] == 'person' and r['frame'] == 12)
    row['measurements']['landmarks']['value'][0]['visibility'] = .2
    kwargs['construction_evidence'] = raw(evidence)
    with pytest.raises(ValueError, match='confident torso'):
        proposal.build_equipment_proposal(base, config, **kwargs)


def test_explicit_translation_is_recorded_and_never_inferred(tmp_path):
    body, equipment, config_path, original, config = packages(tmp_path)
    config['frame_translation_xy'] = [.1, 0.]; write_json(config_path, config)
    report = proposal.prepare_equipment_proposal(body, equipment, config_path, tmp_path / 'translated')
    assert report['status'] == 'proposal_ready'
    assert report['construction_ready_arguments']['frame_translation'] == [.1, 0.]
    receipt = json.loads((tmp_path / 'translated/frame-migration.json').read_bytes())
    assert receipt['translation_world'] == [.1, 0., 0.]
    assert receipt['full_task_equivalence_verified'] is False
    assert report['frame_translation_verification']['status'] == 'passed_bounded_nominal_translation_check'
    assert json.loads((body / 'scene.json').read_bytes()) == original


def test_same_media_name_different_source_hash_is_not_a_source_equivalence(tmp_path):
    body, equipment, config_path, _, _ = packages(tmp_path)
    path = equipment / 'scene-observations.json'
    raw_document = path.read_bytes().replace(b'a' * 64, b'b' * 64)
    path.write_bytes(raw_document); seal(equipment)
    report = proposal.prepare_equipment_proposal(body, equipment, config_path, tmp_path / 'mismatched')
    assert report['status'] == 'unsupported_input'
    assert 'verified derivative' in report['reasons'][0]['message']


def test_tampered_canonical_export_is_preserved_as_rejection_not_resealed_as_valid(tmp_path):
    body, equipment, config_path, _, _ = packages(tmp_path)
    (equipment / 'scene-observations.json').write_bytes(b'changed after seal')
    report = proposal.prepare_equipment_proposal(body, equipment, config_path, tmp_path / 'tampered')
    assert report['status'] == 'unsupported_input'
    assert 'SHA256' in report['reasons'][0]['message']
    assert not (tmp_path / 'tampered/geometry-proposal.json').exists()


@pytest.mark.parametrize('duplicate', [False, True])
def test_nested_sealed_human_document_is_resolved_without_study_specific_paths(tmp_path, duplicate):
    body, equipment, config_path, _, _ = packages(tmp_path)
    nested = body / 'some-import/original/human.json'; nested.parent.mkdir(parents=True)
    nested.write_bytes((body / 'human.json').read_bytes())
    if not duplicate: (body / 'human.json').unlink()
    seal(body); out = tmp_path / 'nested-proposal'
    report = proposal.prepare_equipment_proposal(body, equipment, config_path, out)
    assert report['status'] == 'proposal_ready'
    binding = json.loads((out / 'inputs/human-document-binding.json').read_bytes())
    assert len(binding['identical_sealed_paths']) == (2 if duplicate else 1)
    assert binding['selected_path'] == ('human.json' if duplicate else 'some-import/original/human.json')
    construct_source_equipment_run(body, out, equipment, tmp_path / 'nested-constructed', target_object_id='foam-pad')


def test_different_exact_human_documents_are_not_silently_selected(tmp_path):
    body, equipment, config_path, _, _ = packages(tmp_path)
    nested = body / 'other/human.json'; nested.parent.mkdir()
    nested.write_bytes((body / 'human.json').read_bytes() + b'\n')
    seal(body); out = tmp_path / 'ambiguous-body'
    report = proposal.prepare_equipment_proposal(body, equipment, config_path, out)
    assert report['status'] == 'unsupported_input'
    assert report['reasons'][0]['code'] == 'ambiguous_human_document'
    assert not (out / 'geometry-proposal.json').exists()


def test_failed_explicit_translation_preserves_rejected_candidate_and_reason(tmp_path, monkeypatch):
    body, equipment, config_path, _, config = packages(tmp_path)
    config['frame_translation_xy'] = [.1, 0.]; write_json(config_path, config)
    def fail(*args): raise RuntimeError('Nominal migration exceeded the original rigid-speed envelope')
    monkeypatch.setattr(proposal, 'translate_shared_world_xy', fail)
    out = tmp_path / 'migration-failed'
    report = proposal.prepare_equipment_proposal(body, equipment, config_path, out)
    assert report['status'] == 'rejected_initial_geometry'
    assert report['reasons'][0]['error_type'] == 'RuntimeError'
    assert 'rigid-speed' in report['reasons'][0]['message']
    assert (out / 'geometry-candidate.rejected.json').is_file()
    assert not (out / 'geometry-proposal.json').exists()
    assert report['construction_ready_arguments'] is None
