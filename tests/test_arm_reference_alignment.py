"""Analytic source-image targets for bounded, construction-only full-arm repair."""

from copy import deepcopy
import hashlib
import json

import mujoco
import numpy as np
import pytest

from weft import arm_reference_alignment as alignment
from weft.body_refinement import JOINT_LANDMARKS, _kinematics
from weft.human import HumanEnv, joint_contract
from weft.observations import COORDINATES
from weft.shared_world_lab import _analytic_mobile_human_spec
from weft.world_evidence import load_world_evidence


def inputs(*, withheld=False, secret_shift=0., low_confidence=False, near_limit=False):
    spec = _analytic_mobile_human_spec()
    selected = list(range(0, 12, 2))
    spec['reference'] = [spec['reference'][index] for index in selected]
    spec['root_reference'] = [spec['root_reference'][index] for index in selected]
    spec['source_sha256'] = 'a' * 64
    spec['reference_evidence'] = [{'evidence_id': f"{'a' * 64}:frame:{frame}", 'time_s': frame / 30} for frame in selected]
    if near_limit:
        for row in spec['reference']: row['joints']['left_elbow'] = .01
    camera = {'distance_model_units': 2., 'focal_px': 400., 'original_scale_at_anchor_model_units_per_pixel': .005,
              'pixel_origin': [320., 380.], 'world_anchor_model_units': [0., 0., spec['mobility']['base_height']],
              'world_axes': ['x_right', 'y_depth', 'z_up']}
    unknown = {'value': None, 'status': 'unknown', 'evidence_ids': []}
    document = {'schema_version': 1, 'profile': 'scene-observations-v1',
        'source': {'sha256': 'a' * 64, 'width_px': 640, 'height_px': 640, 'fps': 30, 'metadata_frames': 11, 'decoded_frames': 11},
        'time': {'unit': 'seconds', 'origin': 'video_start', 'basis': 'frame_index_divided_by_fps', 'status': 'assumed'},
        'image': {'width_px': 640, 'height_px': 640}, 'coordinates': deepcopy(COORDINATES),
        'camera': {'motion': deepcopy(unknown), 'calibration': deepcopy(unknown), 'screen': {}},
        'objects': [{'id': 'person', 'label': 'Analytic person', 'properties': {'kind': deepcopy(unknown), 'role': deepcopy(unknown)}, 'unknowns': ['depth']}],
        'evidence': [], 'observations': [], 'relationships': [], 'provenance': {}, 'limits': ['Analytic fixture, not physical observations']}
    env = HumanEnv(spec)
    try:
        data = mujoco.MjData(env.model)
        for frame in range(11):
            measurements, identifiers = {}, []
            sampled = frame in selected
            partition = 'audit' if withheld and frame == 6 else 'construction' if sampled else 'unsampled'
            if sampled:
                index = selected.index(frame)
                env.mobility.reference(data, index)
                truth = np.array([spec['reference'][index]['joints'][name] for name in env.joint_names])
                gain = min(1., index / 3)
                for name, correction in {'left_shoulder_lateral': .12, 'left_elbow': -.06,
                                         'right_shoulder': .14, 'right_shoulder_twist': -.08, 'right_elbow': -.10}.items():
                    truth[env.joint_names.index(name)] += correction * gain
                relative, _ = _kinematics(env, data, truth)
                pelvis = data.site_xpos[env._site_ids['pelvis']].copy()
                world = np.tile(pelvis, (33, 1))
                for name, point in zip(JOINT_LANDMARKS, relative): world[JOINT_LANDMARKS[name]] = pelvis + point
                pixels, _ = alignment.perspective_points(world, camera)
                relative = world - pelvis
                camera_world = relative[:, [0, 2, 1]] * [1, -1, 1]
                if partition == 'audit': pixels += secret_shift
                def confidence(index): return .2 if low_confidence and frame == 4 and index == 15 else 1.
                def points(values):
                    return [{'index': i, 'position': point.tolist(), 'visibility': confidence(i), 'presence': 1.} for i, point in enumerate(values)]
                measurements = {
                    'landmarks': {'coordinate_system': 'image', 'unit': 'pixels', 'status': 'inferred', 'value': points(pixels)},
                    'world_landmarks': {'coordinate_system': 'model_world', 'unit': 'model_meters', 'status': 'inferred', 'value': points(camera_world)}}
                identifier = f"{'a' * 64}:frame:{frame}"; identifiers = [identifier]
                document['evidence'].append({'id': identifier, 'frame': frame, 'time_s': frame / 30, 'object_id': 'person',
                                            'kind': 'pose_model_output', 'status': 'inferred', 'payload': {'private': secret_shift}})
            document['observations'].append({'id': f"{'a' * 64}:observation:{frame}:person", 'frame': frame, 'time_s': frame / 30, 'object_id': 'person',
                'visibility': 'detected' if sampled else 'not_sampled', 'partition': partition, 'evidence_ids': identifiers, 'measurements': measurements})
    finally:
        env.close()
    raw = json.dumps(document, sort_keys=True).encode()
    sha = hashlib.sha256(raw).hexdigest()
    construction = load_world_evidence({sha: raw}).construction_bytes()
    return spec, construction, {'human_entity_ref': {'document_sha256': sha, 'object_id': 'person'}, 'camera_hypothesis': camera}


def test_perspective_derivatives_match_finite_difference_and_reject_behind_camera():
    camera = {'distance_model_units': 2., 'focal_px': 400., 'world_anchor_model_units': [0., 0., .8], 'pixel_origin': [320., 380.]}
    points = np.array([[.2, .3, .5], [-.1, -.2, .9]])
    _, jac = alignment.perspective_points(points, camera, jacobian=True)
    for axis in range(3):
        offset = np.zeros_like(points); offset[:, axis] = 1e-6
        plus, _ = alignment.perspective_points(points + offset, camera)
        minus, _ = alignment.perspective_points(points - offset, camera)
        np.testing.assert_allclose((plus - minus) / 2e-6, jac[:, :, axis], atol=1e-7, rtol=0)
    with pytest.raises(ValueError, match='behind'):
        alignment.perspective_points([[0., -2., .5]], camera)


def test_full_arm_candidate_reduces_analytic_image_error_without_changing_other_contracts():
    spec, evidence, kwargs = inputs()
    original = deepcopy((spec, evidence, kwargs))
    candidate, audit = alignment.align_arm_reference(spec, evidence, **kwargs)
    assert (spec, evidence, kwargs) == original
    assert candidate['reference'][0] == spec['reference'][0]
    for key in audit['fixed_fields']:
        assert candidate.get(key) == spec.get(key)
    names, limits = joint_contract(spec)
    for before, after in zip(spec['reference'], candidate['reference']):
        assert before['time_s'] == after['time_s']
        for name in names:
            assert limits[name][0] <= after['joints'][name] <= limits[name][1]
            if 'shoulder' not in name and not name.endswith('_elbow'):
                assert before['joints'][name] == after['joints'][name]
    points = [p for row in audit['references'] for p in row['endpoints'] if p['fit_weight']]
    assert sum(p['error_after_px'] ** 2 for p in points) < sum(p['error_before_px'] ** 2 for p in points)
    assert 0 < audit['changed_reference_poses'] <= 5
    assert len(audit['source_slots']) == 11 and len(audit['references']) == 6
    assert audit['maximum_correction_rate_rad_s'] <= 1.5 + 1e-10
    assert max(row['maximum_angle_correction_rad'] for row in audit['references']) <= .3 + 1e-10
    assert max(abs(p['depth_change_model_units']) for row in audit['references'] for p in row['endpoints']) <= .08 + 1e-10
    assert audit['physics_steps'] == 0 and not audit['promoted']
    assert all(audit['bound_checks'].values())


def test_withheld_measurements_cannot_influence_angles_and_missing_reference_is_not_dropped():
    one = inputs(withheld=True, secret_shift=0.)
    two = inputs(withheld=True, secret_shift=100.)
    candidate, audit = alignment.align_arm_reference(one[0], one[1], **one[2])
    other, _ = alignment.align_arm_reference(two[0], two[1], **two[2])
    assert candidate['reference'] == other['reference']
    assert candidate['reference'][3] == one[0]['reference'][3]
    assert audit['references'][3]['source_admission'] == 'withheld'
    assert len(candidate['reference']) == 6


def test_low_confidence_wrist_freezes_entire_side_and_respects_gap_continuity():
    spec, evidence, kwargs = inputs(low_confidence=True)
    candidate, audit = alignment.align_arm_reference(spec, evidence, **kwargs)
    assert not audit['references'][2]['eligible_sides'][0]
    for name in spec['reference'][2]['joints']:
        if name.startswith('left_'):
            assert candidate['reference'][2]['joints'][name] == spec['reference'][2]['joints'][name]
    assert audit['maximum_correction_rate_rad_s'] <= 1.5 + 1e-10


def test_joint_limit_is_preserved_when_image_target_requests_negative_elbow_angle():
    spec, evidence, kwargs = inputs(near_limit=True)
    candidate, _ = alignment.align_arm_reference(spec, evidence, **kwargs)
    assert all(row['joints']['left_elbow'] >= 0 for row in candidate['reference'])


@pytest.mark.parametrize('change', ['identity', 'clock', 'camera_anchor', 'camera_scale', 'partition', 'repeat_fit'])
def test_mismatched_source_camera_clock_and_fit_stacking_reject(change):
    spec, evidence, kwargs = inputs()
    if change == 'identity': spec['source_sha256'] = 'b' * 64
    elif change == 'clock': spec['reference_evidence'][1]['time_s'] += .001
    elif change == 'camera_anchor': kwargs['camera_hypothesis']['world_anchor_model_units'][2] += .01
    elif change == 'camera_scale': kwargs['camera_hypothesis']['focal_px'] *= 2
    elif change == 'repeat_fit': spec['arm_reference_alignment'] = {'status': 'candidate'}
    else:
        document = json.loads(evidence); document['allowed_partitions'] = ['construction', 'development_validation']; evidence = json.dumps(document).encode()
    with pytest.raises(ValueError):
        alignment.align_arm_reference(spec, evidence, **kwargs)


def test_budget_exhaustion_retains_complete_candidate_but_never_promotes(monkeypatch):
    spec, evidence, kwargs = inputs()
    monkeypatch.setitem(alignment.CONTRACT, 'maximum_fit_seconds', 0.)
    candidate, audit = alignment.align_arm_reference(spec, evidence, **kwargs)
    assert audit['status'] == 'partial_budget_exhausted'
    assert candidate['reference'] == spec['reference']
    assert len(audit['references']) == 6 and not audit['promoted']


def test_absent_inferred_world_depth_uses_explicit_original_model_fallback():
    spec, evidence, kwargs = inputs()
    document = json.loads(evidence)
    for source in document['sources']:
        for row in source['observations']: row['measurements'].pop('world_landmarks', None)
    _, audit = alignment.align_arm_reference(spec, json.dumps(document).encode(), **kwargs)
    assert all(point['depth_prior_status'] == 'original_articulated_model_fallback'
               for row in audit['references'] for point in row['endpoints'])


def test_included_reference_cannot_point_to_a_different_source_frame():
    spec, evidence, kwargs = inputs()
    document = json.loads(evidence)
    document['sources'][0]['observations'][2]['evidence_refs'][0]['evidence_id'] = f"{'a' * 64}:frame:0"
    with pytest.raises(ValueError, match='reference differs'):
        alignment.align_arm_reference(spec, json.dumps(document).encode(), **kwargs)
