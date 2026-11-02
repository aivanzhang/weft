import runpy
from copy import deepcopy
from pathlib import Path

import pytest


study = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/studies/assess_aspset_body.py'))


def test_assessment_timestamps_cannot_fit_or_renumber_missing_frames():
    frame = study['frame_index']
    assert frame(3.68, 50., 185) == 184
    assert frame(.08+1e-14, 50., 185) == 4
    for stamp in (1/30, -.02, 3.7, True, float('nan')):
        with pytest.raises(ValueError): frame(stamp, 50., 185)


@pytest.mark.parametrize('damage', ['reference_unit', 'reference_pin', 'camera_source', 'body_source', 'construction_use'])
def test_assessment_rejects_rebound_reference_or_camera(damage):
    body = {'source_sha256': 'a'*64}; camera = dict(body)
    source = {'sha256': 'a'*64, 'subject': 'subject', 'clip': 'clip'}
    reference = {'source_video_unit': 'subject-clip', 'source_video_manifest_sha256': 'b'*64, 'reference_for_construction': False}
    assumptions = {'source_manifest_sha256': 'b'*64}
    validate = study['validate_source_identity']
    validate(body, camera, source, reference, assumptions)
    if damage == 'reference_unit': reference['source_video_unit'] = 'different-clip'
    elif damage == 'reference_pin': reference['source_video_manifest_sha256'] = 'c'*64
    elif damage == 'camera_source': camera['source_sha256'] = 'c'*64
    elif damage == 'body_source': body['source_sha256'] = 'c'*64
    else: reference['reference_for_construction'] = True
    with pytest.raises(ValueError, match='same original source'): validate(body, camera, source, reference, assumptions)


def test_assessment_missing_results_are_unavailable_not_zero_error():
    assert study['stats']([]) == {'count': 0, 'mean': None, 'rmse': None, 'p95': None, 'maximum': None}


def test_assessment_paths_cannot_escape_the_input_ancestry_guard(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    linked = tmp_path/'link'; linked.symlink_to(source, target_is_directory=True)
    for path in (tmp_path/'new'/'..'/'source'/'output', linked/'output'):
        with pytest.raises(ValueError, match='traverse|symlink'): study['safe_path'](path)
    assert study['safe_path'](tmp_path/'new') == tmp_path/'new'
    assert list(source.iterdir()) == []


@pytest.mark.parametrize('damage', ['units', 'translation', 'motion', 'clock', 'slot'])
def test_assessment_requires_fixed_publisher_camera_coordinates(damage):
    frame = {'frame': 0, 'time_s': 0., 'R': [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], 't': [0.,0.,0.]}
    camera = {'frame_count': 2, 'fps': 50., 'units': 'm', 'world_axes': 'right_handed_z_up', 'frames': [frame,deepcopy(frame)]}
    camera['frames'][1].update(frame=1,time_s=.02)
    study['validate_fixed_camera'](camera)
    if damage == 'units': camera['units'] = 'mm'
    elif damage == 'translation': camera['frames'][0]['t'] = [1.,0.,0.]
    elif damage == 'motion': camera['frames'][1]['R'] = [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
    elif damage == 'clock': camera['frames'][1]['time_s'] = .04
    else: camera['frames'][1]['frame'] = 3
    with pytest.raises(ValueError, match='fixed metric'): study['validate_fixed_camera'](camera)
