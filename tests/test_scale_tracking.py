from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from weft.generic_vision import track_video, validate_config, inspect_generic
from weft.io import digest
from weft.scale_tracking import ScaleAppearanceTrack, SCALE_CONTRACT
from weft.scene_vision import UnsupportedVideo


def settings(**kwargs):
    return {'search_radius_px': 60, 'minimum_score': .7, 'minimum_margin': .06,
            'max_gap_frames': 4, **kwargs}


def pattern():
    rng = np.random.default_rng(123)
    return rng.integers(20, 235, (40, 60), dtype=np.uint8)


def frame(scale=1., center=(140, 100), duplicate=False, blank=False):
    image = np.full((220, 360), 120, np.uint8)
    if blank:
        return image
    patch = cv2.resize(pattern(), (round(60 * scale), round(40 * scale)))
    h, w = patch.shape
    x, y = round(center[0] - w / 2), round(center[1] - h / 2)
    image[y:y+h, x:x+w] = patch
    if duplicate:
        other = cv2.resize(pattern(), (48, 32))
        image[84:116, 226:274] = other
    return image


def test_known_shrink_grow_and_translation_keep_source_template_immutable():
    tracker = ScaleAppearanceTrack(frame(), [110, 80, 60, 40], settings())
    before = tracker.template.copy()
    for i, scale in enumerate([.95, .8, .6, .75, 1., 1.2, 1.4]):
        center = (140 + i, 100 + i)
        result = tracker.locate(frame(scale, center))
        assert result['status'] == 'tracked'
        x, y, w, h = result['box']
        assert [w, h] == [round(60 * scale), round(40 * scale)]
        assert [x+w/2, y+h/2] == pytest.approx(center, abs=.5)
        tracker.commit(result)
        np.testing.assert_array_equal(tracker.template, before)
    assert len(tracker.templates) <= 17
    assert sum(template.nbytes for _, template in tracker.templates) <= SCALE_CONTRACT['maximum_template_bytes']


def test_shrink_is_not_false_translation_velocity():
    tracker = ScaleAppearanceTrack(frame(), [110, 80, 60, 40], settings())
    tracker.commit(tracker.locate(frame(.8)))
    assert tracker.velocity == pytest.approx([0, 0])


def test_different_scale_duplicate_is_identity_ambiguity():
    tracker = ScaleAppearanceTrack(frame(), [110, 80, 60, 40], settings(search_radius_px=150))
    result = tracker.locate(frame(duplicate=True))
    assert result['status'] == 'ambiguous'
    assert result['box'] is None


def test_occlusion_retains_unknowns_and_long_gap_stays_lost():
    tracker = ScaleAppearanceTrack(frame(), [110, 80, 60, 40], settings())
    for _ in range(3):
        result = tracker.locate(frame(blank=True))
        assert result['status'] == 'not_detected' and result['box'] is None
        tracker.commit(result)
    result = tracker.locate(frame(.8))
    assert result['status'] == 'tracked'
    tracker.commit(result)
    for _ in range(5):
        tracker.commit(tracker.locate(frame(blank=True)))
    assert tracker.locate(frame())['status'] == 'lost'


def test_color_mismatch_never_updates_template_or_size():
    gray = frame()
    def color(hue):
        hsv = np.zeros((*gray.shape, 3), np.uint8)
        hsv[:, :, 2] = gray
        hsv[80:120, 110:170, 0] = hue
        hsv[80:120, 110:170, 1] = 220
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    tracker = ScaleAppearanceTrack(gray, [110, 80, 60, 40], settings(), color=color(30))
    signature = deepcopy(tracker.source_color)
    result = tracker.locate(gray, color=color(110))
    assert result['status'] == 'appearance_changed' and result['box'] is None
    tracker.commit(result)
    assert tracker.box == (110, 80, 60, 40) and tracker.source_color == signature


def test_cache_and_search_resource_bounds_before_large_work():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match='memory bound'):
        ScaleAppearanceTrack(rng.integers(0, 255, (960, 960), dtype=np.uint8), [100, 100, 600, 600], settings())
    large = np.full((960, 960), 120, np.uint8)
    large[:220, :360] = frame()
    tracker = ScaleAppearanceTrack(large, [110, 80, 60, 40], settings(search_radius_px=200))
    result = tracker.locate(large)
    assert result['status'] == 'search_budget_exceeded' and result['box'] is None


def test_large_rotation_is_not_claimed_as_scale():
    tracker = ScaleAppearanceTrack(frame(), [110, 80, 60, 40], settings())
    rotated = np.full_like(frame(), 120)
    rotated[70:130, 120:160] = np.rot90(pattern())
    result = tracker.locate(rotated)
    assert result['status'] == 'not_detected' and result['box'] is None


@pytest.mark.parametrize('mode', [None, 'auto', True, [], {}])
def test_invalid_scale_modes_reject(mode):
    with pytest.raises(ValueError, match='Scale mode'):
        validate_config({'schema_version': 1, 'source_sha256': 'a'*64, 'scale_mode': mode, 'objects': []})


def test_joint_scale_combination_rejects_explicitly():
    with pytest.raises(ValueError, match='independent association'):
        validate_config({'schema_version': 1, 'source_sha256': 'a'*64, 'scale_mode': 'pyramid',
                         'association_mode': 'joint', 'objects': []})


def video_fixture(tmp_path):
    video = tmp_path / 'scale.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 20, (360, 220))
    for i in range(16):
        gray = frame(1 - (i // 2) * .05)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()
    tracking = {'schema_version': 1, 'source_sha256': digest(video), 'scale_mode': 'pyramid',
                'objects': [{'id': 'object', 'label': 'Analytic shrinking region', 'box_xywh': [110,80,60,40],
                             'role': 'active', 'shape': 'box', 'provenance': 'Known rendered source pattern'}]}
    return video, tracking


def test_actual_video_preserves_all_slots_scale_provenance_and_centers(tmp_path):
    video, config = video_fixture(tmp_path)
    result = track_video(video, tmp_path/'tracks', config)
    assert result['decoded_frames'] == 16 and len(result['observations']) == 16
    assert all(row['status'] == 'tracked' for row in result['observations'][1:])
    assert max(abs(row['x_px'] - 140) for row in result['observations']) <= .5
    assert result['observations'][-1]['width_px'] < result['observations'][0]['width_px']
    assert result['scale_contract'] == SCALE_CONTRACT
    assert all('not_physical_shape' in row['geometry_status'] for row in result['observations'])
    json.dumps(result, allow_nan=False)


def test_projected_scale_never_becomes_changing_collision_geometry(tmp_path):
    video, tracking = video_fixture(tmp_path)
    out = tmp_path/'scene';out.mkdir()
    config = {'tracking': tracking, 'world': {'controlled_id': 'object', 'target': [.5,.5],
                                            'gravity': [0,0], 'damping': 1., 'restitution': .8}}
    with pytest.raises(UnsupportedVideo, match='projection/rigid-geometry'):
        inspect_generic(video, out, config)
    assert (out/'appearance/appearance-tracks.json').exists()
    assert not (out/'scene.json').exists()
