from copy import deepcopy
import json

import mujoco
import numpy as np
import pytest

from weft.body_pixel_refinement import _projection, refine_pixel_reference
from weft.body_refinement import JOINT_LANDMARKS
from weft.human import HumanEnv
from weft.root_motion import build_root_reference
from test_whole_body_feasibility import model


def fixture(*, correction=0., moving=False):
    spec = model()
    angles = dict(spec['reference'][0]['joints'])
    for side in ('left', 'right'):
        angles.update({side + '_shoulder': .4, side + '_shoulder_lateral': .2,
                       side + '_elbow': .8, side + '_hip': .3, side + '_knee': .7})
    spec['reference'] = [{'time_s': i / 30, 'joints': angles.copy()} for i in range(0, 16, 2)]
    spec['root_reference'] = [{'time_s': row['time_s'], 'translation': [0., 0., 0.]} for row in spec['reference']]
    spec['reference_evidence'] = [{'time_s': row['time_s'], 'evidence_id': 'a' * 64 + f':frame:{i * 2}',
                                  'joint_confidence': {name: .9 for name in angles}}
                                 for i, row in enumerate(spec['reference'])]
    evidence = {'schema_version': 1, 'profile': 'exercise-pose-v1', 'source_sha256': spec['source_sha256'],
                'width': 1200, 'height': 600, 'fps': 30., 'frame_count': 16, 'decoded_frames': 16,
                'origin_time_s': 0., 'reference': deepcopy(spec['reference_evidence']),
                'lengths': deepcopy(spec['lengths']), 'frames': [],
                'image_dimension_provenance': {'method': 'authored_analytic_fixture',
                    'source_sha256': spec['source_sha256'], 'width': 1200, 'height': 600}}
    env = HumanEnv(spec)
    try:
        data = mujoco.MjData(env.model)
        target = angles.copy()
        target['left_hip'] += correction
        target['right_elbow'] += correction
        xy, _ = _projection(env, data, np.array([target[name] for name in env.joint_names]))
        for i, meta in enumerate(spec['reference_evidence']):
            origin = np.array([600 + (i * 2 if moving else 0), 300.])
            landmarks = [dict(x=.5, y=.5, z=0., visibility=.9, presence=.9) for _ in range(33)]
            for point, index in zip(xy, JOINT_LANDMARKS.values()):
                pixel = point * 200 + origin
                landmarks[index].update(x=float(pixel[0] / 1200), y=float(pixel[1] / 600))
            evidence['frames'].append({'id': meta['evidence_id'], 'frame': i * 2, 'time_s': meta['time_s'],
                                       'status': 'tracked', 'pose_count': 1, 'landmarks': landmarks})
    finally:
        env.close()
    root = build_root_reference(evidence)
    assert root['status'] == 'ready'
    spec['root_reference'] = [{'time_s': row['time_s'], 'translation': row['translation']} for row in root['reference']]
    return spec, evidence


def mean_error(audit, when):
    return np.mean([point['error_' + when + '_px'] for row in audit['frames'] for point in row['points']
                    if point['effective_fit_weight'] > 0])


def test_exact_nonsquare_geometry_and_moving_root_are_unchanged(monkeypatch):
    spec, evidence = fixture(moving=True)
    before = deepcopy((spec, evidence))
    def forbidden(*args, **kwargs):
        raise AssertionError('No physics, floor refit or support actuation')
    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    monkeypatch.setattr('weft.mobile_human.calibrate_floor', forbidden)
    monkeypatch.setattr('weft.mobile_human.MobileDynamics.apply', forbidden)
    candidate, audit = refine_pixel_reference(spec, evidence)
    assert (spec, evidence) == before
    assert candidate['reference'] == spec['reference']
    assert candidate['root_reference'] == spec['root_reference']
    assert mean_error(audit, 'before') < 1e-10
    assert audit['changed_frames'] == 0
    assert audit['model_units_per_pixel'] == pytest.approx(.005)
    json.dumps(audit, allow_nan=False)


def test_known_position_errors_improve_without_changing_physical_assumptions():
    spec, evidence = fixture(correction=.08)
    candidate, audit = refine_pixel_reference(spec, evidence)
    assert audit['status'] == 'complete' and audit['promoted'] is False
    assert mean_error(audit, 'after') < mean_error(audit, 'before') * .2
    for row in audit['frames']:
        assert row['objective_after'] < row['objective_before']
        assert row['maximum_angle_change_rad'] <= .15 + 1e-12
        assert row['iterations'] <= 12 and row['backtrack_evaluations'] <= 48
        assert row['raw_source_frame'] == evidence['frames'][row['reference_index']]
        assert row['depth_identified'] is False
    for key in ['root_reference', 'mobility', 'lengths', 'body_frame', 'motor_limits', 'dt']:
        assert candidate.get(key) == spec.get(key)
    for before, after in zip(spec['reference_evidence'], candidate['reference_evidence']):
        assert after['joint_confidence'] == before['joint_confidence']
        assert after['pixel_reference_derivation']['depth_identified'] is False


def test_projected_position_jacobian_matches_numeric_derivative():
    spec, _ = fixture()
    env = HumanEnv(spec)
    try:
        data = mujoco.MjData(env.model)
        angles = np.array([spec['reference'][0]['joints'][name] for name in env.joint_names])
        _, analytic = _projection(env, data, angles, True)
        numeric = np.empty_like(analytic)
        for i in range(19):
            delta = np.zeros(19); delta[i] = 1e-6
            numeric[:, i] = ((_projection(env, data, angles + delta)[0] -
                              _projection(env, data, angles - delta)[0]) / 2e-6).ravel()
        np.testing.assert_allclose(analytic, numeric, atol=2e-9)
    finally:
        env.close()


def test_inferred_world_coordinates_are_not_targets():
    spec, evidence = fixture(correction=.08)
    other = deepcopy(evidence)
    for row in other['frames']:
        row['world_landmarks'] = [dict(x=float('nan'), y=1000., z=-1000.)] * 33
    a, first = refine_pixel_reference(spec, evidence)
    b, second = refine_pixel_reference(spec, other)
    assert a['reference'] == b['reference']
    assert first['input_evidence_sha256'] != second['input_evidence_sha256']
    assert 'world_landmarks' in second['loss_excludes']
    json.dumps(second, allow_nan=False)


def test_invalid_cropped_and_low_confidence_endpoints_remain_explicit():
    spec, evidence = fixture(correction=.08)
    for row in evidence['frames']:
        row['landmarks'][15]['x'] = 1.2
        row['landmarks'][16] = None
        row['landmarks'][27]['visibility'] = .1
        row['landmarks'][28]['x'] = 1e308
    _, audit = refine_pixel_reference(spec, evidence)
    for row in audit['frames']:
        points = {point['joint']: point for point in row['points']}
        for name in ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']:
            assert points[name]['effective_fit_weight'] == 0
        assert points['left_wrist']['source_status'] == 'outside_source_image'
        assert points['left_ankle']['source_status'] == 'low_confidence'
        assert points['right_wrist']['error_after_px'] is None
        assert points['right_ankle']['error_after_px'] is None
    json.dumps(audit, allow_nan=False)


def test_interpolated_pelvis_cannot_become_an_observed_target():
    spec, evidence = fixture(correction=.08)
    evidence['frames'][3]['landmarks'][23]['visibility'] = .1
    candidate, audit = refine_pixel_reference(spec, evidence)
    assert candidate['reference'][3] == spec['reference'][3]
    assert audit['frames'][3]['status'] == 'unchanged_unobserved_image_origin'
    assert audit['frames'][4]['temporal_prior_source_time_s'] is None


def test_floor_height_does_not_enter_the_objective():
    spec, evidence = fixture(correction=.08)
    other = deepcopy(spec)
    other['mobility']['base_height'] = .2
    a, _ = refine_pixel_reference(spec, evidence)
    b, audit = refine_pixel_reference(other, evidence)
    for first, second in zip(a['reference'], b['reference']):
        np.testing.assert_allclose(list(first['joints'].values()), list(second['joints'].values()), atol=1e-9)
    assert 'floor_distance' in audit['loss_excludes']


def test_opposite_depth_poses_with_identical_pixels_keep_their_priors():
    spec, evidence = fixture()
    for row in spec['reference']:
        row['joints'] = {name: (.5 if name.endswith(('_shoulder', '_hip')) else 0.) for name in row['joints']}
    other = deepcopy(spec)
    for row in other['reference']:
        row['joints'] = {name: -value for name, value in row['joints'].items()}
    env = HumanEnv(spec)
    try:
        data = mujoco.MjData(env.model)
        xy, _ = _projection(env, data, np.array([spec['reference'][0]['joints'][name] for name in env.joint_names]))
        for frame in evidence['frames']:
            for point, index in zip(xy, JOINT_LANDMARKS.values()):
                frame['landmarks'][index].update(x=float((600 + 200 * point[0]) / 1200),
                                                 y=float((300 + 200 * point[1]) / 600))
    finally:
        env.close()
    a, first = refine_pixel_reference(spec, evidence)
    b, second = refine_pixel_reference(other, evidence)
    assert a['reference'] == spec['reference'] and b['reference'] == other['reference']
    assert a['reference'] != b['reference']
    assert mean_error(first, 'before') < 1e-10 and mean_error(second, 'before') < 1e-10


def test_original_joint_limits_and_correction_bounds_are_preserved():
    spec, evidence = fixture(correction=.08)
    for row in spec['reference']:
        row['joints']['right_elbow'] = 2.6
    candidate, _ = refine_pixel_reference(spec, evidence)
    env = HumanEnv(spec)
    try:
        for original, refined in zip(spec['reference'], candidate['reference']):
            for name, value in refined['joints'].items():
                assert env.joint_limits[name][0] <= value <= env.joint_limits[name][1]
                assert abs(value - original['joints'][name]) <= .15 + 1e-12
    finally:
        env.close()


def test_timeout_preserves_every_original_pose(monkeypatch):
    spec, evidence = fixture(correction=.08)
    ticks = iter([0.] + list(range(61, 100)))
    monkeypatch.setattr('weft.body_pixel_refinement.time.monotonic', lambda: next(ticks))
    candidate, audit = refine_pixel_reference(spec, evidence)
    assert candidate['reference'] == spec['reference']
    assert len(audit['frames']) == 8
    assert audit['status'] == 'partial_budget_exhausted'
    assert all(row['status'] == 'unchanged_compute_budget_exhausted' for row in audit['frames'])


@pytest.mark.parametrize('mutation', [
    lambda s, e: s.update(camera_reference={}),
    lambda s, e: s.update(pixel_reference_refinement={}),
    lambda s, e: s.update(image_reference_refinement={}),
    lambda s, e: e.update(source_sha256='b' * 64),
    lambda s, e: e.pop('image_dimension_provenance'),
    lambda s, e: e.update(frames=e['frames'] * 113),
    lambda s, e: s.update(reference=s['reference'] * 113),
    lambda s, e: s['root_reference'][1].update(translation=[.2, 0., 0.]),
    lambda s, e: s['reference_evidence'][1].update(evidence_id='a' * 64 + ':frame:3'),
    lambda s, e: e['frames'][1].update(time_s=.01),
])
def test_invalid_binding_or_expanded_work_is_rejected_before_model_creation(mutation, monkeypatch):
    spec, evidence = fixture()
    mutation(spec, evidence)
    def forbidden(*args):
        raise AssertionError('Input gate must run before constructing a model')
    monkeypatch.setattr('weft.body_pixel_refinement.HumanEnv', forbidden)
    with pytest.raises(ValueError):
        refine_pixel_reference(spec, evidence)
