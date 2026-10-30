"""Source-qualified geometry construction without hidden contact or object actuation."""

from copy import deepcopy
import hashlib
import itertools
import json

import numpy as np
import pytest

from weft.io import canonical_hash
from weft.observations import COORDINATES
from weft.shared_world import SharedWorldEnv, validate_shared_world
from weft.shared_world_evidence import build_shared_world_binding
from weft.shared_world_lab import authored_shared_world_scene
from weft.shared_world_reconstruction import construct_equipment_world, translate_shared_world_xy


SOURCE = 'a' * 64


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def fixture(*, overlap=False, scale=1., include_documents=False):
    scene = authored_shared_world_scene()
    scene.pop('action_contract')
    scene['coordinate_frame']['scale_bridge']['body_height_to_world_scale'] = scale
    human = scene['human_spec']
    human['source_sha256'] = SOURCE
    human['reference_evidence'] = []
    for index, (pose, root) in enumerate(zip(human['reference'], human['root_reference'])):
        pose['time_s'] = root['time_s'] = (index + 2) / 30
        human['reference_evidence'].append({'evidence_id': f'{SOURCE}:frame:{index + 2}', 'time_s': pose['time_s']})
    human['mobility']['floor_fit']['reference_time_s'] = 2 / 30
    scene['coordinate_frame']['clock_binding']['source_time_offset_s'] = 2 / 30
    unknown = {'value': None, 'status': 'unknown', 'evidence_ids': []}
    names = ['person', 'equipment-a', 'equipment-b']
    document = {'schema_version': 1, 'profile': 'scene-observations-v1',
        'source': {'sha256': SOURCE, 'width_px': 640, 'height_px': 480, 'fps': 30, 'metadata_frames': 63, 'decoded_frames': 63},
        'time': {'unit': 'seconds', 'origin': 'video_start', 'basis': 'frame_index_divided_by_fps', 'status': 'assumed'},
        'image': {'width_px': 640, 'height_px': 480}, 'coordinates': deepcopy(COORDINATES),
        'camera': {'motion': deepcopy(unknown), 'calibration': deepcopy(unknown), 'screen': {}},
        'objects': [{'id': name, 'label': name, 'properties': {'kind': deepcopy(unknown), 'role': deepcopy(unknown)}, 'unknowns': ['depth', 'identity', 'contacts']} for name in names],
        'observations': [], 'evidence': [], 'relationships': [], 'provenance': {'scope': 'analytic source fixture'}, 'limits': ['Not a real video']}
    base_height = human['mobility']['base_height']
    camera = {'distance_model_units': 2., 'focal_px': 200., 'original_scale_at_anchor_model_units_per_pixel': .01,
              'pixel_origin': [320., 240.], 'world_anchor_model_units': [0., 0., base_height], 'world_axes': ['x_right', 'y_depth', 'z_up']}
    equipment = []
    for index, name in enumerate(names[1:]):
        center = [0., 0., .12] if overlap and index == 0 else [(.6 if index == 0 else -.6), .45, .12]
        half = [.1, .1, .12]
        vertices = np.asarray(center) + np.array(list(itertools.product((-1, 1), repeat=3))) * half
        pixels = np.array([320., 240.]) + 200 * np.column_stack((vertices[:, 0], base_height - vertices[:, 2])) / (2 + vertices[:, 1, None])
        bounds = np.r_[pixels.min(axis=0), pixels.max(axis=0)].tolist()
        evidence_id = f'{SOURCE}:frame:8:object:{name}'
        equipment.append({'object_id': name, 'source_frame': 8, 'source_evidence_id': evidence_id,
            'source_identity_verified': False, 'source_contact_verified': False,
            'initial_velocity_model_units_per_second': [0., 0., 0.], 'initial_angular_velocity': [0., 0., 0.],
            'external_force_channels': [0., 0., 0.], 'simulator_mass_prior': {'value': 1., 'status': 'authored_not_measured'},
            'geometry': {'status': 'authored_floor_box_candidate', 'source_original_frame': 8, 'source_evidence_id': evidence_id,
                         'center_model_units': center, 'half_extents_model_units': half,
                         'dimensions_model_units': (2 * np.array(half)).tolist(), 'source_bounds_px': bounds, 'projected_bounds_px': bounds}})
    for frame in range(63):
        for name in names:
            present = frame >= (2 if name == 'person' else 8)
            identifier = f'{SOURCE}:frame:{frame}' if name == 'person' else f'{SOURCE}:frame:{frame}:object:{name}'
            measurements = {}
            if present and name == 'person':
                translation = human['root_reference'][frame - 2]['translation']
                hip = np.array([320. + translation[0] / .01, 240. - translation[2] / .01])
                positions = {11: hip + [-10, -50], 12: hip + [10, -50], 23: hip + [-10, 0], 24: hip + [10, 0]}
                measurements['landmarks'] = {'coordinate_system': 'image', 'unit': 'pixels', 'status': 'inferred',
                    'value': [{'index': index, 'position': pos.tolist(), 'visibility': 1., 'presence': 1.} for index, pos in positions.items()]}
            elif present:
                bounds = next(obj['geometry']['source_bounds_px'] for obj in equipment if obj['object_id'] == name)
                measurements['box'] = {'coordinate_system': 'image', 'unit': 'pixels', 'status': 'inferred',
                                       'value': [bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]]}
            if present:
                document['evidence'].append({'id': identifier, 'frame': frame, 'time_s': frame / 30, 'object_id': name,
                                             'kind': 'pose_model_output' if name == 'person' else 'model_segmentation_proposal', 'status': 'inferred', 'payload': {}})
            document['observations'].append({'id': f'{SOURCE}:observation:{frame}:{name}', 'frame': frame, 'time_s': frame / 30,
                'object_id': name, 'partition': 'construction', 'visibility': 'detected' if present else 'not_sampled',
                'evidence_ids': [identifier] if present else [], 'measurements': measurements})
    source_raw = raw(document)
    digest = hashlib.sha256(source_raw).hexdigest()
    documents = {digest: source_raw}
    base_ids = ['human', *(obj['id'] for obj in scene['objects'])]
    original_binding, _ = build_shared_world_binding(documents, source_sha256=SOURCE, world_ids=base_ids,
        entity_refs={name: (digest, 'person') if name == 'human' else None for name in base_ids})
    binding, construction = build_shared_world_binding(documents, source_sha256=SOURCE, world_ids=['human', *names[1:]],
        entity_refs={name: (digest, 'person' if name == 'human' else name) for name in ['human', *names[1:]]})
    scene['source_binding'] = original_binding
    scene = validate_shared_world(scene)
    human_bytes = raw(scene['human_spec'])
    ids = [row['evidence_id'] for row in human['reference_evidence']]
    proposal = {'schema_version': 1, 'profile': 'equipment-geometry-study-proposal-v1',
        'status': 'authored_initial_geometry_proposal_not_physical_reconstruction',
        'body_binding': {'source_sha256': SOURCE, 'human_json_sha256': hashlib.sha256(human_bytes).hexdigest(),
            'reference_count': 61, 'first_frame': 2, 'preserve_all_source_references_and_clocks': True,
            'mobility_contract_unchanged': deepcopy(human['mobility']), 'body_lengths_unchanged': deepcopy(human['lengths']),
            'root_origin_unchanged': {'evidence_id': ids[0], 'time_s': 2 / 30, 'pelvis_image_px': [320., 240.]},
            'root_scale_unchanged': {'status': 'authored_uncalibrated_model_scale', 'method': 'model_torso_divided_by_median_image_torso',
                'model_torso_length': .5, 'torso_image_median_px': 50., 'model_units_per_pixel': .01, 'samples': 61,
                'torso_image_p10_px': 50., 'torso_image_p90_px': 50., 'evidence_ids': ids}},
        'camera_hypothesis': camera, 'equipment': equipment}
    result = (scene, proposal, {'source_binding': binding, 'construction_evidence': construction,
                               'human_document': human_bytes, 'target_object_id': 'equipment-a'})
    return (*result, documents) if include_documents else result


@pytest.mark.parametrize('scale', [1., 1.25])
def test_constructor_preserves_physics_body_evidence_and_inputs_while_replacing_props(scale):
    base, proposal, kwargs = fixture(scale=scale)
    saved = deepcopy((base, proposal, kwargs))
    result = construct_equipment_world(base, proposal, **kwargs)
    assert (base, proposal, kwargs) == saved
    for key in ('human_spec', 'bounds', 'gravity', 'contact', 'dt', 'substeps', 'coordinate_frame'):
        assert result[key] == base[key]
    assert result['task']['start_offset_s'] == 0 and result['task']['human_limits'] == base['task']['human_limits']
    assert result['learned_policy_available'] is False and result['controlled_id'] == 'human'
    assert [obj['id'] for obj in result['objects']] == ['equipment-a', 'equipment-b']
    assert result['target'] == result['objects'][0]['position']
    assert not result['task']['object_goal']['require_human_contact']
    for obj, source in zip(result['objects'], proposal['equipment']):
        np.testing.assert_allclose(obj['position'], np.array(source['geometry']['center_model_units']) * scale + base['coordinate_frame']['scale_bridge']['origin_world'])
        np.testing.assert_allclose(obj['size'], np.array(source['geometry']['dimensions_model_units']) * scale)
        assert obj['role'] == 'passive' and obj['mass'] == 1.
        assert obj['provenance']['source_prefix']['body_start_frame'] == 2
        assert obj['provenance']['source_prefix']['equipment_observation_frame'] == 8
        assert obj['provenance']['source_prefix']['source_slots_filled'] == 0
    original, env = SharedWorldEnv(base), SharedWorldEnv(result)
    try:
        np.testing.assert_array_equal(original.model.opt.gravity, env.model.opt.gravity)
        for field in ('geom_pos', 'geom_quat', 'geom_size'):
            np.testing.assert_array_equal(getattr(original.model, field)[original._floor_geom], getattr(env.model, field)[env._floor_geom])
        env.reset(seed=3)
        for _ in range(3):
            action = env.authored_action()
            np.testing.assert_array_equal(action[22:], np.zeros(6))
            env.step(action)
            assert all(obj['virtual_force'] == [0., 0., 0.] for obj in env.snapshot()['objects'])
        assert env.action_space.shape == (28,) and env.model.nu == 22
    finally:
        original.close(); env.close()


@pytest.mark.parametrize('mutation', ['human_bytes', 'canonical_body', 'origin', 'scale', 'root', 'body_count', 'source_frame', 'evidence_id', 'box', 'focal', 'force', 'mass'])
def test_constructor_rejects_mismatched_body_geometry_source_and_authored_controls(mutation):
    base, proposal, kwargs = fixture()
    if mutation == 'human_bytes': kwargs['human_document'] += b' '
    elif mutation == 'canonical_body': base['human_spec']['lengths']['torso'] += .01
    elif mutation == 'origin': proposal['body_binding']['root_origin_unchanged']['pelvis_image_px'][0] += 1
    elif mutation == 'scale': proposal['body_binding']['root_scale_unchanged']['model_units_per_pixel'] *= 2
    elif mutation == 'root': base['human_spec']['root_reference'][1]['translation'][0] += .01
    elif mutation == 'body_count': proposal['body_binding']['reference_count'] -= 1
    elif mutation == 'source_frame': proposal['equipment'][0]['source_frame'] = 9
    elif mutation == 'evidence_id': proposal['equipment'][0]['source_evidence_id'] = 'unresolved'
    elif mutation == 'box': proposal['equipment'][0]['geometry']['source_bounds_px'][0] += 1
    elif mutation == 'focal': proposal['camera_hypothesis']['focal_px'] += 1
    elif mutation == 'force': proposal['equipment'][0]['external_force_channels'][0] = .1
    elif mutation == 'mass': proposal['equipment'][0]['simulator_mass_prior']['value'] = 2
    with pytest.raises(ValueError):
        construct_equipment_world(base, proposal, **kwargs)


def test_constructor_rejects_withheld_geometry_even_when_counts_are_recomputed():
    base, proposal, kwargs = fixture()
    document = json.loads(kwargs['construction_evidence'])
    source = document['sources'][0]
    row = next(row for row in source['observations'] if row['frame'] == 8 and row['object_id'] == 'equipment-a')
    row.update(partition='audit', admission='withheld', visibility=None, measurements={}, evidence_refs=[])
    source['evidence'] = [row for row in source['evidence'] if row['id'] != proposal['equipment'][0]['source_evidence_id']]
    entity = next(row for row in kwargs['source_binding']['entity_bindings'] if row['world_entity_id'] == 'equipment-a')
    entity['observed_slots'] -= 1; entity['missing_slots'] += 1
    kwargs['construction_evidence'] = raw(document)
    with pytest.raises(ValueError, match='unavailable or withheld'):
        construct_equipment_world(base, proposal, **kwargs)


@pytest.mark.parametrize('change', ['counts', 'document', 'duplicate_slot', 'missing_slot'])
def test_constructor_requires_actual_complete_document_qualified_evidence(change):
    base, proposal, kwargs = fixture()
    if change == 'counts':
        row = kwargs['source_binding']['entity_bindings'][1]
        row['observed_slots'] += 1; row['missing_slots'] -= 1
    elif change == 'document':
        kwargs['source_binding']['entity_bindings'][1]['observation_document_sha256'] = 'b' * 64
    else:
        document = json.loads(kwargs['construction_evidence'])
        rows = document['sources'][0]['observations']
        if change == 'duplicate_slot': rows.append(deepcopy(rows[-1]))
        else: rows.pop()
        kwargs['construction_evidence'] = raw(document)
    with pytest.raises(ValueError):
        construct_equipment_world(base, proposal, **kwargs)


def test_initial_overlap_is_rejected_without_repositioning_or_trimming_source():
    base, proposal, kwargs = fixture(overlap=True)
    original = canonical_hash((base, proposal))
    with pytest.raises(ValueError, match='intersect|overlap'):
        construct_equipment_world(base, proposal, **kwargs)
    assert canonical_hash((base, proposal)) == original


def test_frozen_world_bounds_reject_proxy_without_moving_body_or_clipping_equipment():
    base, proposal, kwargs = fixture()
    base['coordinate_frame']['scale_bridge']['origin_world'][0] = .65
    original = deepcopy((base, proposal))
    with pytest.raises(ValueError, match='bounds'):
        construct_equipment_world(base, proposal, **kwargs)
    assert (base, proposal) == original


def test_observation_clock_cannot_be_retimed_with_its_evidence_row():
    base, proposal, kwargs = fixture()
    document = json.loads(kwargs['construction_evidence'])
    source = document['sources'][0]
    row = next(row for row in source['observations'] if row['frame'] == 8 and row['object_id'] == 'equipment-a')
    evidence = next(row for row in source['evidence'] if row['id'] == proposal['equipment'][0]['source_evidence_id'])
    row['time_s'] += .01; evidence['time_s'] += .01
    kwargs['construction_evidence'] = raw(document)
    with pytest.raises(ValueError, match='clock'):
        construct_equipment_world(base, proposal, **kwargs)


def test_malformed_or_rejected_proposals_fail_before_construction():
    base, proposal, kwargs = fixture()
    del proposal['body_binding']['root_origin_unchanged']
    with pytest.raises(ValueError, match='Malformed'):
        construct_equipment_world(base, proposal, **kwargs)
    base, proposal, kwargs = fixture()
    proposal['equipment'][0]['geometry']['status'] = 'rejected'
    with pytest.raises(ValueError, match='uncertainty'):
        construct_equipment_world(base, proposal, **kwargs)


def test_explicit_xy_frame_migration_preserves_nominal_dynamics_and_discloses_envelope_change():
    base = authored_shared_world_scene()
    original = deepcopy(base)
    moved, receipt = translate_shared_world_xy(base, [.2, .15])
    assert base == original
    shift = np.array([.2, .15, 0.])
    for key in ('human_spec', 'gravity', 'contact', 'source_binding', 'dt', 'substeps', 'task', 'action_contract'):
        assert moved[key] == base[key]
    assert moved['coordinate_frame']['clock_binding'] == base['coordinate_frame']['clock_binding']
    assert moved['coordinate_frame']['scale_bridge']['body_height_to_world_scale'] == base['coordinate_frame']['scale_bridge']['body_height_to_world_scale']
    np.testing.assert_allclose(moved['bounds'], np.array(base['bounds']) + shift)
    np.testing.assert_allclose(moved['target'], np.array(base['target']) + shift)
    np.testing.assert_allclose(moved['coordinate_frame']['scale_bridge']['origin_world'], np.array(base['coordinate_frame']['scale_bridge']['origin_world']) + shift)
    for before, after in zip(base['objects'], moved['objects']):
        for key in set(before) - {'position'}:
            assert before[key] == after[key]
        np.testing.assert_allclose(after['position'], np.array(before['position']) + shift)
    assert receipt['before_scene_sha256'] == canonical_hash(base)
    assert receipt['after_scene_sha256'] == canonical_hash(moved)
    assert receipt['translation_world'] == shift.tolist()
    assert 'lower' in receipt['numerical_envelope_change'] and not receipt['full_task_equivalence_verified']
    verification = receipt['nominal_translation_verification']
    assert verification['matched_control_steps'] == 8
    assert verification['compared_contact_records'] > 0
    assert verification['maximum_state_error'] <= verification['state_tolerance']
    assert verification['maximum_contact_geometry_error'] <= verification['state_tolerance']
    assert all(np.array(action)[22:].tolist() == [0.] * 6 for action in verification['matched_controls'])


def test_explicit_migration_then_equipment_construction_retains_original_source_geometry():
    base, proposal, kwargs = fixture()
    moved, receipt = translate_shared_world_xy(base, [.2, 0.])
    result = construct_equipment_world(moved, proposal, **kwargs)
    assert result['human_spec'] == base['human_spec']
    assert result['coordinate_frame'] == moved['coordinate_frame']
    assert result['bounds'] == moved['bounds']
    assert result['task']['start_offset_s'] == 0 and result['human_spec']['reference'][0]['time_s'] == 2 / 30
    assert result['objects'][0]['provenance']['construction']['base_scene_sha256'] == receipt['after_scene_sha256']
    assert any(note.startswith('Explicit authored XY frame migration:') for note in result['assumptions'])
    for obj, source in zip(result['objects'], proposal['equipment']):
        assert obj['provenance']['geometry_model_units']['center'] == source['geometry']['center_model_units']


@pytest.mark.parametrize('translation', [[-.1, 0.], [0., -.1], [0., 0.], [.2, 0., 0.],
                                        [float('nan'), 0.], [True, 0.], [5., 0.], {'x': .2, 'y': 0.}])
def test_xy_migration_rejects_z_scale_negative_oversize_and_invalid_inputs(translation):
    base = authored_shared_world_scene()
    original = canonical_hash(base)
    with pytest.raises(ValueError):
        translate_shared_world_xy(base, translation)
    assert canonical_hash(base) == original
