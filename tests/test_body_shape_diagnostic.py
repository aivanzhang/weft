from copy import deepcopy
import json

import mujoco
import numpy as np
import pytest

from weft.body_frame import CAMERA_TO_SIM
from weft.body_refinement import JOINT_LANDMARKS
from weft.body_shape_diagnostic import diagnose_shape
from weft.human import HumanEnv
from test_whole_body_feasibility import model


def fixture():
    spec = model()
    spec['reference'] = [{'time_s': i / 10, 'joints': {name: value + (.03 * i if name == 'left_hip' else 0)
                          for name, value in spec['reference'][0]['joints'].items()}} for i in range(6)]
    spec['root_reference'] = [{'time_s': row['time_s'], 'translation': [0., 0., 0.]} for row in spec['reference']]
    spec['reference_evidence'] = [{'time_s': i / 10, 'evidence_id': 'a' * 64 + ':frame:' + str(i)} for i in range(6)]
    env = HumanEnv(spec)
    data = mujoco.MjData(env.model)
    frames = []
    for index, reference in enumerate(spec['reference']):
        data.qpos[env._q_array] = [reference['joints'][name] for name in env.joint_names]
        mujoco.mj_forward(env.model, data)
        pelvis = data.site_xpos[env._site_ids['pelvis']]
        world = [{'x': 0., 'y': 0., 'z': 0.} for _ in range(33)]
        image = [{'x': .5, 'y': .5, 'z': 0., 'visibility': 1., 'presence': 1.} for _ in range(33)]
        for name, landmark in JOINT_LANDMARKS.items():
            position = data.site_xpos[env._site_ids[name]] - pelvis
            camera = CAMERA_TO_SIM.T @ position
            world[landmark] = dict(zip(('x', 'y', 'z'), map(float, camera)))
            image[landmark]['x'] = .5 + position[0] / 2
            image[landmark]['y'] = .5 - position[2] / 2
        frames.append({'id': 'a' * 64 + ':frame:' + str(index), 'frame': index, 'time_s': index / 10,
                       'status': 'tracked', 'pose_count': 1, 'world_landmarks': world, 'landmarks': image})
    return spec, {'source_sha256': 'a' * 64, 'frames': frames}


def run(spec, evidence):
    return diagnose_shape(spec, evidence, width=500, height=500, fps=10, frame_count=6, model_units_per_pixel=.004)


def test_exact_geometry_has_zero_errors_and_no_physics_or_floor_fit(monkeypatch):
    spec, evidence = fixture()
    before = deepcopy((spec, evidence))
    def forbidden(*args, **kwargs):
        raise AssertionError('No floor fitting or dynamics in geometric diagnosis')
    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    monkeypatch.setattr('weft.mobile_human.calibrate_floor', forbidden)
    result = run(spec, evidence)
    assert (spec, evidence) == before
    assert result['counts'] == {'requested': 6, 'compared': 6, 'rejected_retained': 0}
    for partition in result['summary'].values():
        for variant in partition.values():
            assert variant['world']['rmse'] < 1e-14
            assert variant['pixels']['rmse'] < 1e-12
    json.dumps(result, allow_nan=False)


def test_static_asymmetric_length_evidence_improves_without_touching_other_side():
    spec, evidence = fixture()
    for frame in evidence['frames']:
        elbow, wrist = frame['world_landmarks'][13], frame['world_landmarks'][15]
        for axis in ('x', 'y', 'z'):
            wrist[axis] = elbow[axis] + 1.1 * (wrist[axis] - elbow[axis])
        frame['landmarks'][15]['y'] = .5 + wrist['y'] / 2
    result = run(spec, evidence)
    assert result['static_lengths']['left_elbow__left_wrist']['bounded_length'] == pytest.approx(.275)
    assert result['static_lengths']['right_elbow__right_wrist']['bounded_length'] == pytest.approx(.25)
    assert result['summary']['heldout']['lengths']['world']['rmse'] < 1e-14
    assert result['summary']['heldout']['baseline']['world']['rmse'] > .001


def test_static_attachment_evidence_recovers_only_constant_offset():
    spec, evidence = fixture()
    for frame in evidence['frames']:
        for index in (11, 13, 15):
            frame['world_landmarks'][index]['x'] += .02
            frame['landmarks'][index]['x'] += .01
    result = run(spec, evidence)
    assert result['summary']['heldout']['attachments']['world']['rmse'] < 1e-14
    assert result['summary']['heldout']['lengths']['world']['rmse'] > .001
    assert result['static_attachments']['left_shoulder']['bounded_local'][0] == pytest.approx(-.18)


def test_heldout_changes_cannot_alter_fitted_static_parameters():
    spec, evidence = fixture()
    baseline = run(spec, evidence)
    for frame in evidence['frames'][4:]:
        frame['world_landmarks'][27]['y'] += .3
        frame['landmarks'][27]['y'] += .1
    altered = run(spec, evidence)
    assert altered['static_lengths'] == baseline['static_lengths']
    assert altered['static_attachments'] == baseline['static_attachments']
    assert altered['summary']['heldout']['both']['world']['rmse'] > baseline['summary']['heldout']['both']['world']['rmse']
    assert set(altered['training_evidence_ids']).isdisjoint(altered['heldout_evidence_ids'])


def test_crop_and_missing_confidence_remain_raw_but_cannot_fit_shape():
    spec, evidence = fixture()
    for frame in evidence['frames']:
        frame['landmarks'][15]['x'] = 1.01
        frame['world_landmarks'][15]['x'] = 2.
        frame['landmarks'][16]['visibility'] = 0.
    result = run(spec, evidence)
    assert result['static_lengths']['left_elbow__left_wrist']['positive_weight_samples'] == 0
    assert result['static_lengths']['right_elbow__right_wrist']['estimated_length'] is None
    assert result['rows'][0]['raw_landmarks'][15]['x'] == 1.01
    assert result['rows'][0]['outside_image_crop']['left_wrist']
    assert result['rows'][0]['effective_confidence']['left_wrist'] == 0
    json.dumps(result, allow_nan=False)


def test_caps_are_static_and_residual_decomposition_closes():
    spec, evidence = fixture()
    for frame in evidence['frames']:
        frame['world_landmarks'][15]['y'] += 1.
        for index in (11, 13, 15):
            frame['world_landmarks'][index]['x'] += .2
    result = run(spec, evidence)
    length = result['static_lengths']['left_elbow__left_wrist']
    assert length['cap_hit'] and length['bounded_length'] <= length['original_length'] * 1.15
    attachment = result['static_attachments']['left_shoulder']
    assert attachment['cap_hit']
    assert np.linalg.norm(np.array(attachment['bounded_local']) - attachment['original_local']) == pytest.approx(.03)
    for row in result['rows']:
        assert max(value['closure_error'] for value in row['endpoint_decomposition']) < 1e-10


def test_missing_world_frame_is_retained_and_not_interpolated():
    spec, evidence = fixture()
    evidence['frames'][2].pop('world_landmarks')
    result = run(spec, evidence)
    assert result['counts']['rejected_retained'] == 1
    assert result['rows'][2]['status'] == 'missing_world_or_image_landmarks'


@pytest.mark.parametrize('change', [lambda s, e: e.update(source_sha256='b' * 64),
    lambda s, e: e['frames'][0].update(frame=1), lambda s, e: e['frames'][0].update(time_s=.1),
    lambda s, e: s.update(image_reference_refinement={}), lambda s, e: e.update(frames=e['frames'] * 151)])
def test_identity_time_bounds_and_original_reference_required(change):
    spec, evidence = fixture()
    change(spec, evidence)
    with pytest.raises(ValueError):
        run(spec, evidence)
