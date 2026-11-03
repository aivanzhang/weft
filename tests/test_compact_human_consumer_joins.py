"""Compact runtime bodies retain exact full-sidecar identity at downstream intake."""

from copy import deepcopy
import hashlib
import json
import time

import pytest

from test_shared_world_reconstruction import fixture, raw
from test_person_body_learning_admission import document, reseal
from test_exercise_body_native_clock import native_pose
from test_exercise_body_source import forbid_runtime
from test_shared_world_human import inventory
from test_source_clock import clock_fixture
from weft import equipment_proposal, exercise_learning_pipeline, shared_world_reconstruction, shared_world_pipeline
from weft.io import canonical_hash, write_json
from weft.shared_human_projection import create_projection
from weft.source_clock import clock_summary


def projected_fixture():
    scene, proposal, options = fixture()
    full = json.loads(options['human_document'])
    _, binding, _ = clock_fixture()
    binding.update(metadata_frames=63, decoded_frames=63, time_base='1/30', native_origin_pts=0,
                   native_to_reconstruction_offset_s=0, decoder_minus_native_s=0,
                   rows=[{'frame': i, 'pts': i, 'decoder_timestamp_s': i / 30} for i in range(63)])
    full['native_clock'] = clock_summary(binding)
    full['body_frame'] = {'schema_version': 1, 'status': 'assumed', 'evidence_ids': [], 'reference': [],
                          'rotation': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}
    scene['coordinate_frame']['clock_binding'].update(native_clock=full['native_clock'],
        alignment_method='native_pts_elapsed_time_v1', source_fps=30.)
    full['reconstruction_coverage'] = {'scope': 'Analytic provenance, not measured coverage'}
    for row in full['reference_evidence']:
        row['joint_confidence'] = {'right_elbow': .75}
    options['human_document'] = raw(full)
    proposal['body_binding']['human_json_sha256'] = hashlib.sha256(options['human_document']).hexdigest()
    scene['human_spec'], scene['human_provenance'] = create_projection(options['human_document'])
    return scene, proposal, options


@pytest.fixture(scope='module')
def native_candidate(tmp_path_factory):
    from weft import exercise_body_source, shared_world_human
    root = tmp_path_factory.mktemp('compact-consumer-native')
    original, clock, _, _ = native_pose(root / 'poses')
    with pytest.MonkeyPatch.context() as patch:
        forbid_runtime(patch)
        exercise_body_source.prepare_exercise_body_source_from_evidence(original, root / 'body', native_clock=clock)
        shared_world_human.construct_human_only_run(root / 'body', inventory(root / 'body'), root / 'candidate')
    return root / 'candidate'


def test_discovery_matches_exact_full_bytes_not_compact_or_other_body(tmp_path):
    scene, _, options = projected_fixture()
    full = options['human_document']
    contents = {'human.json': full, 'copy/human.json': full, 'wrong/human.json': raw(scene['human_spec'])}
    selected, receipt = equipment_proposal._find_human_document(
        tmp_path, contents, scene['human_spec'], lambda _root, _names, name, **_: contents[name],
        provenance=scene['human_provenance'])
    assert selected == full
    assert receipt['identical_sealed_paths'] == ['copy/human.json', 'human.json']
    assert receipt['candidate_documents'] == 3


def test_writer_preserves_full_sidecar_bytes_and_rejects_reencoded_copy():
    scene, _, options = projected_fixture()
    original = options['human_document']
    assert shared_world_pipeline._human_sidecar(scene, original) == original
    with pytest.raises(ValueError):
        shared_world_pipeline._human_sidecar(scene, original + b' ')
    legacy = {'human_spec': json.loads(original)}
    assert shared_world_pipeline._human_sidecar(legacy, b'unused') == shared_world_pipeline._bytes(legacy['human_spec'])


@pytest.mark.parametrize('mutation', ['full_detail', 'runtime_physical', 'descriptor'])
def test_discovery_rejects_changed_projection_join(tmp_path, mutation):
    scene, _, options = projected_fixture()
    if mutation == 'full_detail':
        full = json.loads(options['human_document'])
        full['reference_evidence'][0]['joint_confidence']['right_elbow'] = .9
        options['human_document'] = raw(full)
    elif mutation == 'runtime_physical':
        scene['human_spec']['lengths']['torso'] += .01
    else:
        scene['human_provenance']['full_human_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='exact human-document'):
        equipment_proposal._find_human_document(tmp_path, {'human.json': None}, scene['human_spec'],
            lambda *_a, **_k: options['human_document'], provenance=scene['human_provenance'])


def test_constructor_retains_full_join_before_native_geometry(monkeypatch):
    scene, proposal, options = projected_fixture()
    before = deepcopy((scene, proposal, options))
    class ReachedGeometry(Exception):
        pass
    def stop_at_geometry(candidate, _frames):
        assert candidate['human_spec'] == scene['human_spec']
        assert candidate['human_provenance'] == scene['human_provenance']
        raise ReachedGeometry
    monkeypatch.setattr(shared_world_reconstruction, '_check_initial_geometry', stop_at_geometry)
    with pytest.raises(ReachedGeometry):
        shared_world_reconstruction.construct_equipment_world(scene, proposal, **options)
    assert (scene, proposal, options) == before
    options['human_document'] += b' '
    with pytest.raises(ValueError):
        shared_world_reconstruction.construct_equipment_world(scene, proposal, **options)


def test_constructor_validates_externalized_orientation_samples(monkeypatch):
    scene, proposal, options = projected_fixture()
    full = json.loads(options['human_document'])
    full['body_frame']['reference'] = [{'rotation': [[0., 0., 0.]] * 3, 'time_s': .1, 'id': 'bad'}]
    options['human_document'] = raw(full)
    proposal['body_binding']['human_json_sha256'] = hashlib.sha256(options['human_document']).hexdigest()
    scene['human_spec'], scene['human_provenance'] = create_projection(options['human_document'])
    monkeypatch.setattr(shared_world_reconstruction, '_check_initial_geometry',
                        lambda *_: pytest.fail('Invalid externalized provenance must fail before native checks'))
    with pytest.raises(ValueError, match='orthonormal'):
        shared_world_reconstruction.construct_equipment_world(scene, proposal, **options)


def test_legacy_discovery_still_requires_canonical_full_body(tmp_path):
    scene, _, options = fixture()
    raw_body = options['human_document']
    result, _ = equipment_proposal._find_human_document(tmp_path, {'human.json': None}, scene['human_spec'],
        lambda *_a, **_k: raw_body)
    assert result == raw_body
    changed = deepcopy(scene['human_spec'])
    changed['lengths']['torso'] += .01
    with pytest.raises(ValueError, match='exact human-document'):
        equipment_proposal._find_human_document(tmp_path, {'human.json': None}, changed,
            lambda *_a, **_k: raw_body)


def proposal_inputs():
    scene, _, options = projected_fixture()
    options.pop('target_object_id')
    config = {'schema_version': 1, 'profile': equipment_proposal.CONFIG_PROFILE,
              'source_sha256': scene['human_spec']['source_sha256'],
              'camera': {'mode': 'fixed_pelvis_perspective_prior', 'distance_model_units': 2.},
              'geometry_prior': 'floor_box_depth_equals_width',
              'equipment': [{'object_id': 'equipment-a', 'source_frame': 12},
                            {'object_id': 'equipment-b', 'source_frame': 18}],
              'target_object_id': 'equipment-a'}
    return scene, config, options


def test_actual_proposal_builder_projection_matches_full_body_output_exactly():
    scene, config, options = proposal_inputs()
    original = deepcopy((scene, config, options))
    projected = equipment_proposal.build_equipment_proposal(scene, config, **options)
    full_scene = deepcopy(scene)
    full_scene.pop('human_provenance')
    full_scene['human_spec'] = json.loads(options['human_document'])
    complete = equipment_proposal.build_equipment_proposal(full_scene, config, **options)
    assert projected == complete
    assert projected['body_binding']['human_json_sha256'] == hashlib.sha256(options['human_document']).hexdigest()
    assert (scene, config, options) == original


@pytest.mark.parametrize('damage', ['bytes', 'confidence', 'physical', 'omitted_orientation'])
def test_actual_proposal_builder_rejects_changed_full_provenance(damage):
    scene, config, options = proposal_inputs()
    full = json.loads(options['human_document'])
    if damage == 'bytes':
        options['human_document'] += b' '
    else:
        if damage == 'confidence':
            full['reference_evidence'][0]['joint_confidence']['right_elbow'] = .9
        elif damage == 'physical':
            full['lengths']['torso'] += .01
        else:
            full['body_frame']['reference'] = [{'rotation': [[0., 0., 0.]] * 3, 'time_s': .1, 'id': 'bad'}]
        options['human_document'] = raw(full)
        if damage == 'omitted_orientation':
            scene['human_spec'], scene['human_provenance'] = create_projection(options['human_document'])
    with pytest.raises(ValueError):
        equipment_proposal.build_equipment_proposal(scene, config, **options)


@pytest.mark.parametrize('tamper', [False, True])
def test_learning_receipt_requires_exact_full_body_projection(native_candidate, tmp_path, tamper):
    import shutil
    candidate = tmp_path / 'candidate'
    shutil.copytree(native_candidate, candidate)
    scene = document(candidate, 'scene.json')
    full = (candidate / 'human.json').read_bytes()
    scene['human_spec'], scene['human_provenance'] = create_projection(full)
    if tamper:
        scene['human_spec']['reference'][0]['joints']['right_elbow'] += .001
    write_json(candidate / 'scene.json', scene)
    reconstruction = document(candidate, 'reconstruction.json')
    reconstruction['scene_sha256'] = canonical_hash(scene)
    write_json(candidate / 'reconstruction.json', reconstruction)
    reseal(candidate)
    manifest = (candidate / 'manifest.json').read_bytes()
    names = json.loads(manifest)['files']
    call = lambda: exercise_learning_pipeline._admit_preserved_body_source(
        candidate, manifest, names, scene, time.monotonic() + 5)
    if tamper:
        with pytest.raises(ValueError):
            call()
    else:
        assert call()['human_file_sha256'] == hashlib.sha256(full).hexdigest()
