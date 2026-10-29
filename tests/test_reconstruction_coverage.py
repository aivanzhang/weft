"""Late motion and gaps must remain visible to the fixed replay scorer."""

from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from weft.io import digest
from weft.reconstruction_coverage import plan_reconstruction
from weft.scene_vision import fit_scene, inspect_video, _reconstruction_frames


def measurements(frames=200, objects=('actor', 'other'), missing=()):
    return [{'frame': index, 'time_s': index / 10, 'object_id': name, 'id': f'{index}:{name}'}
            for index in range(frames) for name in objects if (index, name) not in missing]


def test_all_episodes_partial_measurements_and_empty_tail_are_retained():
    missing = {(index, name) for index in range(70, 75) for name in ('actor', 'other')}
    missing |= {(index, 'other') for index in range(130, 135)}
    missing |= {(index, name) for index in range(198, 200) for name in ('actor', 'other')}
    rows = measurements(missing=missing)
    report, generation, development = plan_reconstruction(rows, ['actor', 'other'], [0, 199], 10)
    assert not report['reasons']
    assert report['model_interval'] == [0, 199]
    assert report['generation_end_frame'] == 119
    assert len(report['joint_episodes']) == 3
    assert len(report['frames']) == 200
    assert report['missing_observation_slots'] == 19
    assert {row['id'] for row in generation + development} == {row['id'] for row in rows}
    assert report['per_object']['actor']['development_observations'] == 78
    assert report['per_object']['other']['development_observations'] == 73
    assert report['frames'][-1]['observed_object_ids'] == []
    assert report['frames'][-1]['partition'] == 'development_validation'


def test_short_initial_episode_cannot_be_replaced_with_an_easier_later_one():
    rows = measurements(frames=40, objects=('actor',), missing={(i, 'actor') for i in range(4, 9)})
    report, _, _ = plan_reconstruction(rows, ['actor'], [0, 39], 10)
    assert report['status'] == 'unavailable'
    assert report['model_interval'][0] == 0
    assert 'later episodes cannot replace' in report['reasons'][0]


def test_late_object_without_development_data_cannot_disappear_from_score():
    rows = measurements(missing={(i, 'other') for i in range(120, 200)})
    report, _, _ = plan_reconstruction(rows, ['actor', 'other'], [0, 199], 10)
    assert report['status'] == 'unavailable'
    assert report['per_object']['other']['development_observations'] == 0


@pytest.mark.parametrize('interval,fps', [([0, 1800], 30), ([0, 60], 1), ([2, 1], 30), ([0, 40], float('nan'))])
def test_work_bounds_reject_instead_of_shortening(interval, fps):
    with pytest.raises(ValueError, match='bounded'):
        plan_reconstruction([], ['actor'], interval, fps)


def test_duplicate_measurements_cannot_inflate_coverage():
    rows = measurements(20, ('actor',))
    with pytest.raises(ValueError, match='Duplicate'):
        plan_reconstruction(rows + [rows[0]], ['actor'], [0, 19], 10)


def write_late_motion(path, profile, *, changed=True):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (240, 160))
    assert writer.isOpened()
    texture = np.random.default_rng(8).integers(30, 230, (28, 24), dtype=np.uint8)
    for index in range(200):
        image = np.full((160, 240, 3), 120, np.uint8)
        x, y = 30 + round(index * .3), 40 + (round(max(0, index - 120) * .8) if changed else 0)
        if index not in range(60, 64):
            if profile == 'appearance':
                image[y:y+28, x:x+24] = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
            else:
                cv2.circle(image, (x + 12, y + 14), 12, (0, 230, 230), -1)
        writer.write(image)
    writer.release()


def inspect_fixture(video, out, profile):
    world = {'controlled_id': 'actor', 'target': [.85, .3], 'gravity': [0, 0], 'damping': 1., 'restitution': .5}
    if profile == 'appearance':
        from weft.generic_vision import inspect_generic
        tracking = {'schema_version': 1, 'source_sha256': digest(video), 'start_frame': 0, 'end_frame': 199,
                    'objects': [{'id': 'actor', 'label': 'Synthetic texture', 'box_xywh': [30, 40, 24, 28],
                                 'role': 'active', 'shape': 'box', 'provenance': 'Analytic source rectangle'}]}
        return inspect_generic(video, out, {'tracking': tracking, 'world': world})
    return inspect_video(video, out, {**world, 'objects': [
        {'id': 'actor', 'hue_min': 20, 'hue_max': 40, 'role': 'active', 'shape': 'circle'}]})


@pytest.mark.parametrize('profile', ['appearance', 'planar'])
def test_late_motion_fails_full_replay_instead_of_passing_opening_section(tmp_path, profile):
    video = tmp_path / 'late.mp4'
    write_late_motion(video, profile)
    out = tmp_path / 'run'; out.mkdir()
    evidence, scene = inspect_fixture(video, out, profile)
    coverage = evidence['reconstruction_coverage']
    assert coverage['model_interval'][-1] == 199
    assert len(coverage['joint_episodes']) == 2
    assert evidence['selected_frames'][-1] == 199
    assert evidence['development_validation'][-1]['frame'] == 199
    assert coverage['missing_observation_slots'] >= (5 if profile == 'appearance' else 4)
    assert all(coverage['frames'][index]['missing_object_ids'] == ['actor'] for index in range(60, 64))
    initial = deepcopy(evidence)
    fit_scene(evidence, scene, out)
    assert evidence == initial
    report = json.loads((out / 'reconstruction.json').read_text())
    assert not report['replay_gate']['passes']
    assert len(report['frames']) == 200
    assert report['frames'][61]['measurement_status'] == 'unavailable'
    assert report['frames'][61]['observed'] == []
    assert len(report['frames'][61]['predicted']) == 1
    assert report['frames'][-1]['simulation_time_s'] > 19
    assert report['after']['observations'] == coverage['per_object']['actor']['development_observations']
    assert report['after']['rmse_px'] > 240 * .08
    assert (out / 'reconstruction-contact-sheet.png').exists()


@pytest.mark.parametrize('profile', ['appearance', 'planar'])
def test_constant_motion_with_gap_is_scored_to_the_last_frame(tmp_path, profile):
    video = tmp_path / 'constant.mp4'
    write_late_motion(video, profile, changed=False)
    out = tmp_path / 'run'; out.mkdir()
    evidence, scene = inspect_fixture(video, out, profile)
    fit_scene(evidence, scene, out)
    report = json.loads((out / 'reconstruction.json').read_text())
    assert report['replay_gate']['passes']
    assert report['frames'][-1]['frame'] == 199
    assert report['after']['observations'] >= 79
    broken = deepcopy(evidence)
    broken['reconstruction_coverage']['frames'].pop()
    with pytest.raises(ValueError, match='complete bounded'):
        _reconstruction_frames(broken, scene)
    broken = deepcopy(evidence)
    broken['reconstruction_coverage']['frames'][-1]['partition'] = 'generation'
    with pytest.raises(ValueError, match='scoring partition'):
        _reconstruction_frames(broken, scene)
