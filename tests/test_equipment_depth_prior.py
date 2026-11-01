"""Analytical depth geometry and custody; model/native admission is explicitly mocked."""

from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from weft import equipment_proposal as proposal
from weft import equipment_proposal_config as config_api
from weft import shared_world_pipeline as pipeline
from weft import shared_world_reconstruction as reconstruction
from weft.observations import COORDINATES
from weft.shared_world_evidence import build_shared_world_binding


SOURCE = 'a' * 64
PRIOR = {'depth_to_width_ratio': .5, 'provenance': 'Authored analytical dimensions, not a physical measurement.'}


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def project(center, half, camera, base):
    # Independent homogeneous projection generates pixels before the inverse is called.
    transform = np.array([[1., 0., 0., 0.], [0., 0., -1., base],
                          [0., 1., 0., camera['distance_model_units']]])
    intrinsic = np.array([[camera['focal_px'], 0., camera['pixel_origin'][0]],
                          [0., camera['focal_px'], camera['pixel_origin'][1]], [0., 0., 1.]])
    vertices = np.asarray(center) + np.asarray(list(itertools.product((-1., 1.), repeat=3))) * half
    pixels = (intrinsic @ transform @ np.column_stack((vertices, np.ones(8))).T).T
    pixels = pixels[:, :2] / pixels[:, 2:]
    return np.r_[pixels.min(axis=0), pixels.max(axis=0)].tolist()


def prior(ratio):
    return {**PRIOR, 'depth_to_width_ratio': ratio}


def config(*, ratio_mode=True):
    result = {'schema_version': 1, 'profile': config_api.CONFIG_PROFILE, 'source_sha256': SOURCE,
        'camera': {'mode': 'fixed_pelvis_perspective_prior', 'distance_model_units': 2.},
        'geometry_prior': config_api.DEPTH_RATIO_PRIOR if ratio_mode else 'floor_box_depth_equals_width',
        'equipment': [{'object_id': 'weight', 'source_frame': 2}, {'object_id': 'pad', 'source_frame': 4}],
        'target_object_id': 'weight'}
    if ratio_mode:
        for item, ratio in zip(result['equipment'], (.5, 2.), strict=True):
            item['depth_prior'] = prior(ratio)
    return result


def pure_inputs(monkeypatch, *, ratio_mode=True):
    """Keep actual source/body/projection joins; replace only model schema/native boundaries."""
    checks = []
    def initial(scene, frames):
        checks.append((deepcopy(scene), dict(frames)))
        return [{'object_id': obj['id'], 'test_scope': 'native_gate_mock_not_executed'} for obj in scene['objects']]
    monkeypatch.setattr(proposal, 'validate_shared_world', deepcopy)
    monkeypatch.setattr(reconstruction, 'validate_shared_world', deepcopy)
    monkeypatch.setattr(reconstruction, '_check_initial_geometry', initial)
    monkeypatch.setattr(pipeline, 'validate_shared_world', deepcopy)
    monkeypatch.setattr(pipeline, 'audit_shared_world_feasibility',
                        lambda scene: {'status': 'analytical_mock_not_executed', 'summary': {}})
    human = {'profile': 'exercise-v3', 'source_sha256': SOURCE, 'dt': 1 / 30,
        'lengths': {'torso': .5}, 'mobility': {'base_height': 1., 'floor_height': 0.},
        'reference': [{'time_s': frame / 30, 'joints': {}} for frame in (0, 2, 4)],
        'root_reference': [{'time_s': frame / 30, 'translation': [frame * .01, 0., 0.]} for frame in (0, 2, 4)],
        'reference_evidence': [{'time_s': frame / 30, 'evidence_id': f'{SOURCE}:frame:{frame}'} for frame in (0, 2, 4)],
        'assumptions': ['Incomplete analytical body; native model/schema admission is mocked.']}
    unknown = {'value': None, 'status': 'unknown', 'evidence_ids': []}
    names = ('person', 'weight', 'pad')
    document = {'schema_version': 1, 'profile': 'scene-observations-v1',
        'source': {'sha256': SOURCE, 'width_px': 640, 'height_px': 480, 'fps': 30, 'metadata_frames': 6, 'decoded_frames': 6},
        'time': {'unit': 'seconds', 'origin': 'video_start', 'basis': 'frame_index_divided_by_fps', 'status': 'assumed'},
        'image': {'width_px': 640, 'height_px': 480}, 'coordinates': deepcopy(COORDINATES),
        'camera': {'motion': deepcopy(unknown), 'calibration': deepcopy(unknown), 'screen': {}},
        'objects': [{'id': name, 'label': name, 'properties': {'kind': deepcopy(unknown), 'role': deepcopy(unknown)},
                     'unknowns': ['depth', 'contact']} for name in names],
        'observations': [], 'evidence': [], 'relationships': [],
        'provenance': {'scope': 'authored analytical fixture; no source video exists'}, 'limits': ['Not physical evidence']}
    camera = {'distance_model_units': 2., 'focal_px': 200., 'pixel_origin': [320., 240.]}
    boxes = {name: project([x, .45, .12], [.1, .1, .12], camera, 1.) for name, x in [('weight', .6), ('pad', -.6)]}
    for frame in range(6):
        for name in names:
            present = frame in (0, 2, 4) if name == 'person' else frame >= 2
            identifier = f'{SOURCE}:frame:{frame}' if name == 'person' else f'{SOURCE}:frame:{frame}:object:{name}'
            measurements = {}
            if present and name == 'person':
                hip = np.array([320. + frame, 240.])
                points = {11: hip + [-10, -50], 12: hip + [10, -50], 23: hip + [-10, 0], 24: hip + [10, 0]}
                measurements['landmarks'] = {'coordinate_system': 'image', 'unit': 'pixels', 'status': 'inferred',
                    'value': [{'index': key, 'position': point.tolist(), 'visibility': 1., 'presence': 1.} for key, point in points.items()]}
            elif present:
                left, top, right, bottom = boxes[name]
                measurements['box'] = {'coordinate_system': 'image', 'unit': 'pixels', 'status': 'inferred',
                                       'value': [left, top, right - left, bottom - top]}
            if present:
                document['evidence'].append({'id': identifier, 'frame': frame, 'time_s': frame / 30, 'object_id': name,
                    'kind': 'pose_model_output' if name == 'person' else 'model_segmentation_proposal', 'status': 'inferred', 'payload': {}})
            document['observations'].append({'id': f'{SOURCE}:observation:{frame}:{name}', 'frame': frame, 'time_s': frame / 30,
                'object_id': name, 'partition': 'construction', 'visibility': 'detected' if present else 'not_sampled',
                'evidence_ids': [identifier] if present else [], 'measurements': measurements})
    document_raw = raw(document)
    sha = hashlib.sha256(document_raw).hexdigest()
    documents = {sha: document_raw}
    original_binding, _ = build_shared_world_binding(documents, source_sha256=SOURCE, world_ids=['human', 'dummy'],
        entity_refs={'human': (sha, 'person'), 'dummy': None})
    binding, construction = build_shared_world_binding(documents, source_sha256=SOURCE, world_ids=['human', 'weight', 'pad'],
        entity_refs={'human': (sha, 'person'), 'weight': (sha, 'weight'), 'pad': (sha, 'pad')})
    base = {'schema_version': 1, 'profile': 'shared-world-v1', 'human_spec': human, 'source_binding': original_binding,
        'controlled_id': 'human', 'objects': [{'id': 'dummy'}], 'bounds': [6., 6., 6.], 'gravity': [0., 0., -9.81],
        'contact': {}, 'dt': 1 / 30, 'substeps': 16,
        'coordinate_frame': {'scale_bridge': {'body_height_to_world_scale': 1., 'origin_world': [1., 1., 0.]},
                             'clock_binding': {'source_time_offset_s': 0.}},
        'task': {'start_offset_s': 0., 'horizon_steps': 4, 'human_limits': {'frozen': True}, 'object_goal': {}},
        'assumptions': ['Authored analytical fixture']}
    kwargs = {'source_binding': binding, 'construction_evidence': construction, 'human_document': raw(human)}
    return base, config(ratio_mode=ratio_mode), kwargs, documents, checks


@pytest.mark.parametrize('ratio', [config_api.MIN_DEPTH_TO_WIDTH_RATIO, .25, 1, 3., config_api.MAX_DEPTH_TO_WIDTH_RATIO])
def test_explicit_prior_preserves_exact_supplied_values_and_is_detached(ratio):
    value = prior(ratio)
    result = config_api.validate_depth_prior(value)
    assert result == value and type(result['depth_to_width_ratio']) is type(ratio)
    result['provenance'] = 'changed'
    assert value['provenance'] == PRIOR['provenance']


@pytest.mark.parametrize('ratio', [None, True, '1', 0, -1, math.nan, math.inf, 10 ** 400,
                                  math.nextafter(config_api.MIN_DEPTH_TO_WIDTH_RATIO, 0.), 120.00001])
def test_invalid_ratio_rejects_without_numeric_normalization(ratio):
    with pytest.raises(config_api.EquipmentProposalError, match='ratio'):
        config_api.validate_depth_prior(prior(ratio))


@pytest.mark.parametrize('value', [None, {}, {'depth_to_width_ratio': 1},
    {**PRIOR, 'status': 'measured'}, {**PRIOR, 'provenance': ''}, {**PRIOR, 'provenance': '  '},
    {**PRIOR, 'provenance': 'x' * 2001}])
def test_prior_requires_bounded_explicit_provenance(value):
    with pytest.raises(config_api.EquipmentProposalError):
        config_api.validate_depth_prior(value)


def test_config_requires_opt_in_and_each_objects_own_prior():
    value = config()
    assert config_api.validate_proposal_config(value) == value
    del value['equipment'][1]['depth_prior']
    with pytest.raises(config_api.EquipmentProposalError, match='depth_prior'):
        config_api.validate_proposal_config(value)
    legacy = config(ratio_mode=False)
    assert config_api.validate_proposal_config(legacy) == legacy
    legacy['equipment'][0]['depth_prior'] = prior(1)
    with pytest.raises(config_api.EquipmentProposalError, match='object_id and source_frame only'):
        config_api.validate_proposal_config(legacy)


@pytest.mark.parametrize('center_x', [-.6, 0., .6])
@pytest.mark.parametrize('ratio', [.25, .5, 1., 2., 4.])
def test_inverse_recovers_independently_projected_non_square_boxes(center_x, ratio):
    camera = {'distance_model_units': 2., 'focal_px': 200., 'pixel_origin': [320., 240.]}
    center, half = [center_x, .45, .12], [.1, .1 * ratio, .12]
    bounds = project(center, half, camera, 1.)
    result = proposal.floor_box_from_bounds(bounds, camera, 1., 0., depth_prior=prior(ratio))
    np.testing.assert_allclose(result['center_model_units'], center, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result['half_extents_model_units'], half, atol=1e-12, rtol=0)
    assert max(map(abs, result['construction_bbox_edge_error_px'])) < 1e-10
    assert result['depth_prior'] == prior(ratio)


@pytest.mark.parametrize('center_x', [-.6, 0., .6])
def test_explicit_ratio_one_has_exact_legacy_numerical_geometry(center_x):
    camera = {'distance_model_units': 2., 'focal_px': 200., 'pixel_origin': [320., 240.]}
    bounds = project([center_x, .45, .12], [.1, .1, .12], camera, 1.)
    legacy = proposal.floor_box_from_bounds(bounds, camera, 1., 0.)
    explicit = proposal.floor_box_from_bounds(bounds, camera, 1., 0., depth_prior=prior(1))
    assert explicit.pop('depth_prior') == prior(1)
    assert explicit.pop('status') == 'authored_floor_box_ratio_candidate'
    assert legacy.pop('status') == 'authored_floor_box_candidate'
    assert raw(explicit) == raw(legacy)


@pytest.mark.parametrize('half', [[.6, .005, .12], [.005, .6, .12]])
def test_ratio_limits_do_not_skip_actual_projection_or_dimensions(half):
    camera = {'distance_model_units': 2., 'focal_px': 200., 'pixel_origin': [320., 240.]}
    center = [0., 1., .12]
    ratio = half[1] / half[0]
    result = proposal.floor_box_from_bounds(project(center, half, camera, 1.), camera, 1., 0., depth_prior=prior(ratio))
    np.testing.assert_allclose(result['half_extents_model_units'], half, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result['center_model_units'], center, atol=1e-12, rtol=0)


@pytest.mark.parametrize('bounds,ratio,error', [
    ([300., 230., 340., 300.], 1., 'below the original pelvis'),
    ([300., 280., 340., 300.], 120., 'nonpositive or nonfinite'),
    ([300., 5000., 340., 6000.], 1., 'behind or too near'),
])
def test_ratio_mode_retains_projection_domain_rejections_without_fallback(bounds, ratio, error):
    camera = {'distance_model_units': 2., 'focal_px': 200., 'pixel_origin': [320., 240.]}
    with pytest.raises(ValueError, match=error):
        proposal.floor_box_from_bounds(bounds, camera, 1., 0., depth_prior=prior(ratio))


def test_actual_body_source_joins_and_constructor_preserve_invariants(monkeypatch):
    base, value, kwargs, documents, checks = pure_inputs(monkeypatch)
    originals = deepcopy((base, value, kwargs, documents))
    candidate = proposal.build_equipment_proposal(base, value, **kwargs)
    scene = reconstruction.construct_equipment_world(base, candidate, **kwargs, target_object_id='weight')
    assert (base, value, kwargs, documents) == originals
    for key in ('human_spec', 'bounds', 'gravity', 'contact', 'dt', 'substeps', 'coordinate_frame'):
        assert scene[key] == base[key]
    assert scene['task']['human_limits'] == base['task']['human_limits']
    assert scene['task']['horizon_steps'] == base['task']['horizon_steps']
    assert len(checks) == 1 and checks[0][1] == {0: 0, 2: 1, 4: 2}
    assert len(json.loads(kwargs['construction_evidence'])['sources'][0]['observations']) == 18
    for selected, obj in zip(value['equipment'], scene['objects'], strict=True):
        assert obj['provenance']['depth_prior'] == selected['depth_prior']
        assert obj['size'][1] == pytest.approx(obj['size'][0] * selected['depth_prior']['depth_to_width_ratio'])
        assert obj['provenance']['physical_validation'] == 'unverified'
        assert obj['provenance']['reference_object_force'] == [0., 0., 0.]
        assert obj['position'][2] - obj['size'][2] / 2 == 0.


@pytest.mark.parametrize('change', ['ratio', 'note', 'method', 'status', 'dimensions', 'floor', 'projection', 'too_large', 'clock'])
def test_constructor_independently_rejects_forged_or_out_of_bounds_geometry(monkeypatch, change):
    base, value, kwargs, _, checks = pure_inputs(monkeypatch)
    candidate = proposal.build_equipment_proposal(base, value, **kwargs)
    geometry = candidate['equipment'][0]['geometry']
    if change == 'ratio': geometry['depth_prior']['depth_to_width_ratio'] = 3.
    elif change == 'note': geometry['depth_prior']['provenance'] = ''
    elif change == 'method': del candidate['proposer']
    elif change == 'status': geometry['status'] = 'authored_floor_box_candidate'
    elif change == 'dimensions': geometry['dimensions_model_units'][1] += .01
    elif change == 'floor': geometry['center_model_units'][2] += .01
    elif change == 'projection': geometry['projected_bounds_px'][0] += .01
    elif change == 'too_large': geometry['half_extents_model_units'][0] = .61
    elif change == 'clock': candidate['body_binding']['root_origin_unchanged']['time_s'] = 1 / 30
    with pytest.raises(ValueError):
        reconstruction.construct_equipment_world(base, candidate, **kwargs, target_object_id='weight')
    assert checks == []


def test_initial_collision_rejection_is_not_waived(monkeypatch):
    base, value, kwargs, _, _ = pure_inputs(monkeypatch)
    candidate = proposal.build_equipment_proposal(base, value, **kwargs)
    calls = []
    def reject(scene, frames):
        calls.append(dict(frames))
        raise ValueError('Equipment initial overlap at source frame 0: weight')
    monkeypatch.setattr(reconstruction, '_check_initial_geometry', reject)
    with pytest.raises(ValueError, match='initial overlap'):
        reconstruction.construct_equipment_world(base, candidate, **kwargs, target_object_id='weight')
    assert calls == [{0: 0, 2: 1, 4: 2}]


def seal(directory):
    files = {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in directory.rglob('*') if p.is_file()}
    (directory / 'manifest.json').write_bytes(raw({'schema_version': 1, 'files': files}))


@pytest.mark.parametrize('ratio_mode', [False, True])
def test_public_producer_and_constructor_archive_only_required_dependency(monkeypatch, tmp_path, ratio_mode):
    base, value, kwargs, documents, checks = pure_inputs(monkeypatch, ratio_mode=ratio_mode)
    body, equipment = tmp_path / 'body', tmp_path / 'equipment'
    body.mkdir(); equipment.mkdir()
    (body / 'scene.json').write_bytes(raw(base)); (body / 'human.json').write_bytes(kwargs['human_document'])
    for sha, name in base['source_binding']['evidence_documents'].items():
        target = body / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(documents[sha])
    (equipment / 'scene-observations.json').write_bytes(next(iter(documents.values())))
    (equipment / 'report.json').write_bytes(raw({'status': 'evidence_only', 'physics_ready': False}))
    seal(body); seal(equipment)
    config_path = tmp_path / 'config.json'
    config_path.write_bytes(raw(value))
    initial = {p: p.read_bytes() for folder in (body, equipment) for p in folder.rglob('*') if p.is_file()}
    report = proposal.prepare_equipment_proposal(body, equipment, config_path, tmp_path / 'proposal')
    assert report['status'] == 'proposal_ready', report['reasons']
    assert (tmp_path / 'proposal/input-config.json').read_bytes() == raw(value)
    expected_method = ('explicit-fixed-pelvis-floor-box-depth-ratio-proposal-v1' if ratio_mode
                       else 'explicit-fixed-pelvis-floor-box-proposal-v1')
    assert report['contract']['method'] == expected_method
    pipeline.construct_source_equipment_run(body, tmp_path / 'proposal', equipment, tmp_path / 'candidate', target_object_id='weight')
    archive = tmp_path / 'candidate/implementation/equipment_proposal_config.py'
    assert archive.exists() is ratio_mode
    if ratio_mode:
        assert archive.read_bytes() == Path(config_api.__file__).read_bytes()
    assert all(path.read_bytes() == saved for path, saved in initial.items())
    for folder in (tmp_path / 'proposal', tmp_path / 'candidate'):
        manifest = json.loads((folder / 'manifest.json').read_bytes())
        assert all(hashlib.sha256((folder / name).read_bytes()).hexdigest() == sha for name, sha in manifest['files'].items())
    assert len(checks) == 2
