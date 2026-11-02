from copy import deepcopy
import json
import sys

import numpy as np
import pytest

from test_camera_ground import references
from test_distal_pixel_uncertainty import SETTINGS
from test_multiview_body import setup
from weft.camera_body import make_camera_body
from weft.camera_body_lab import prepare_camera_body_lab
from weft.camera_ground import apply_camera_ground
from weft.human import HumanEnv, build_human, validate_human
from weft.cli import main
from weft.io import canonical_hash, digest, write_json
from weft.lab import CameraHumanSession, Lab
from weft.multiview_body import make_two_view_body
from weft.multiview_depth_branches import METHOD, _temporal
from weft.multiview_root import make_two_view_root
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, verify_recording


def source_inputs(tmp_path, count=2, hidden=None):
    root, predictions, first, second, points, _, camera, camera2, ground = setup(tmp_path,
        hidden={5: [27]} if hidden is None else hidden, flexion=1.4, monocular_flexion=.5)
    evidence = [first, second]; cameras = [camera, camera2]; extra = []
    for index in range(count-2):
        ev, cam = deepcopy(second), deepcopy(camera2)
        ev['source_sha256'] = cam['source_sha256'] = ('c' if index == 0 else 'd')*64
        for i, (row, transform) in enumerate(zip(ev['frames'], cam['frames'], strict=True)):
            row['id'] = f"{ev['source_sha256']}:frame:{i}"
            transform['t'] = [.5+index*.5, .2, 0.]
            q = points[i] @ np.array(transform['R']).T+transform['t']
            pixels = (q/q[:, 2, None]) @ np.array(cam['K']).T
            for landmark, pixel in zip(row['landmarks'], pixels, strict=True):
                landmark.update(x=float(pixel[0]/cam['width']), y=float(pixel[1]/cam['height']))
        evidence.append(ev); cameras.append(cam)
        extra.append({'name': f'extra-{index}', 'evidence': ev, 'camera': cam, 'clock_provenance': 'Analytic simultaneous frames'})
    if extra:
        original = build_human(first, tmp_path/'original-again', body_model='mobile')
        base, _ = make_camera_body(first, camera, original); base = apply_camera_ground(base, ground)
        views = [{'name': name, 'evidence': ev, 'camera': cam, 'clock_provenance': 'Analytic simultaneous frames'}
                 for name, ev, cam in zip(('first', 'second'), evidence[:2], cameras[:2], strict=True)]
        root, _, predictions = make_two_view_root(base, *views, additional_views=extra, world_provenance='Known analytic shared world')
    return root, predictions, evidence, cameras


def construct(root, predictions, evidence, cameras, *, branches):
    return make_two_view_body(root, predictions, *evidence[:2], additional_evidence=evidence[2:],
        limb_fit=True, partial_segments=True, partial_pixel_fit=True, pixel_cameras=cameras,
        pixel_uncertainty=SETTINGS, partial_depth_branches=branches)


@pytest.mark.parametrize('count', [2, 3, 4])
def test_temporal_branch_preserves_upper_bones_and_every_prior_fit(tmp_path, count):
    args = source_inputs(tmp_path, count); before = canonical_hash(args)
    old, old_audit = construct(*args, branches=False); body, audit = construct(*args, branches=True)
    assert canonical_hash(args) == before and body['camera_reference']['method'] == METHOD
    assert audit['depth_branches']['selected_group_poses'] == 1
    assert audit['depth_branches']['before_branch_reference_sha256'] == canonical_hash(old['reference'])
    for field in ('partial_pixels', 'limb_position_fit', 'dimensions'):
        assert audit[field] == old_audit[field]
    for field in ('lengths', 'body_frame', 'root_reference', 'mobility', 'motor_limits'):
        assert body[field] == old[field]
    assert body['camera_reference']['ground'] == old['camera_reference']['ground']
    a, b = references(old), references(body)
    for i, meta in enumerate(body['reference_evidence']):
        if i != 5:
            assert body['reference'][i] == old['reference'][i] and not meta['geometry_depth_branches']
        else:
            branch = meta['geometry_depth_branches']['left_hip']
            assert branch['anchor_reference_indices'] == [4, 6]
            assert branch['pixel_sse_after'] < branch['pixel_sse_before']
            assert branch['temporal_distance_after'] < branch['temporal_distance_before']
            assert any(abs(v) > .3 for v in branch['correction_rad'])
            for name in ('left_hip', 'left_knee'):
                np.testing.assert_allclose(a[i]['human']['joints'][name], b[i]['human']['joints'][name], atol=1e-12)
            assert meta['joint_confidence']['left_knee'] == 0.
            attempt = audit['frames'][i]['groups']['left_hip']['depth_branch_attempt']
            selected = attempt['fit']['candidates'][attempt['fit']['selected_index']]
            offset = np.r_[body['camera_reference']['origin_world_m'][:2], body['camera_reference']['ground']['height_m']]
            np.testing.assert_allclose(np.array(b[i]['human']['joints']['left_ankle'])+offset, selected['world_m'], atol=1e-10)
    env = HumanEnv(body)
    try:
        assert env.scene['human']['depth_branches'] == audit['depth_branches']
        assert not env.scene['learned_policy_available']
    finally: env.close()


def test_missing_bracketing_observations_preserves_every_v2_angle(tmp_path):
    args = source_inputs(tmp_path, hidden={i: [27] for i in range(10)})
    old, _ = construct(*args, branches=False); body, audit = construct(*args, branches=True)
    assert body['reference'] == old['reference']
    assert audit['depth_branches']['selected_group_poses'] == 0
    assert audit['depth_branches']['status_counts'] == {'missing_temporal_anchor': 10}
    for row in audit['frames']:
        attempt = row['groups']['left_hip']['depth_branch_attempt']
        assert attempt['temporal']['status'] == 'missing_bracketing_anchors'
        assert len(attempt['fit']['rays']) == 1


@pytest.mark.parametrize('damage', ['gap', 'direction'])
def test_stale_or_inconsistent_anchors_abstain(damage):
    rows = [{'reference_index': 0, 'time_s': 0., 'direction': [0., 0., -1.]},
            {'reference_index': 2, 'time_s': .2, 'direction': [0., 0., -1.]}]
    if damage == 'gap': rows[1]['time_s'] = 1.
    else: rows[1]['direction'] = [0., 0., 1.]
    result = _temporal({'left_hip': ([r['time_s'] for r in rows], rows)}, 'left_hip', .1)
    assert result['direction'] is None and len(result['anchors']) == 2
    assert result['status'] == ('anchor_gap' if damage == 'gap' else 'inconsistent_anchor_directions')


@pytest.mark.parametrize('damage', ['contract', 'before_hash', 'missing_record', 'input_angles', 'rotation',
    'bend', 'pixel_regression', 'temporal_distance', 'margin', 'anchor_index', 'unsupported_anchor',
    'candidate_index', 'confidence', 'depth_observed', 'method'])
def test_resealed_branch_cannot_rewrite_support_or_hide_the_intermediate_fit(tmp_path, damage):
    body, _ = construct(*source_inputs(tmp_path), branches=True)
    binding = body['camera_reference']['body_refinement']; meta = body['reference_evidence'][5]
    record = meta['geometry_depth_branches']['left_hip']
    if damage == 'contract': binding['depth_branch_contract']['maximum_bend_rad'] = 3.
    elif damage == 'before_hash': binding['depth_branch_reference_sha256'] = 'f'*64
    elif damage == 'missing_record': del meta['geometry_depth_branches']['left_hip']
    elif damage == 'input_angles': record['input_angles_rad'][0] += .1
    elif damage == 'rotation': body['reference'][5]['joints']['left_hip_lateral'] += .01
    elif damage == 'bend': record['correction_rad'][1] = 3.
    elif damage == 'pixel_regression': record['pixel_sse_after'] = record['pixel_sse_before']+1.
    elif damage == 'temporal_distance': record['temporal_distance_after'] += .01
    elif damage == 'margin': record['runner_up_gap_lengths'] = .2
    elif damage == 'anchor_index': record['anchor_reference_indices'] = [3, 6]
    elif damage == 'unsupported_anchor': body['reference_evidence'][4]['geometry_sources']['left_hip'] = 'monocular_prior_missing_points'
    elif damage == 'candidate_index': record['selected_candidate_index'] = 8
    elif damage == 'confidence': meta['joint_confidence']['left_knee'] = .9
    elif damage == 'depth_observed': body['reconstruction_coverage']['two_view_body']['depth_branches']['depth_measured'] = True
    elif damage == 'method': body['camera_reference']['method'] = 'camera-body-partial-pixels-v2'
    binding['output_evidence_sha256'] = canonical_hash(body['reference_evidence'])
    body['camera_reference']['angular_reference_sha256'] = canonical_hash(body['reference'])
    with pytest.raises(ValueError): validate_human(body)


def test_branch_flag_requires_explicit_uncertainty_before_output(tmp_path):
    with pytest.raises(ValueError, match='Depth branches require'):
        prepare_camera_body_lab(tmp_path/'source', tmp_path/'camera', tmp_path/'out', partial_depth_branches=True)
    assert not (tmp_path/'out').exists()


def test_selected_depth_hypothesis_survives_cli_record_export_and_exact_replay(tmp_path, monkeypatch):
    _, _, first, second, _, source, camera, camera2, ground = setup(tmp_path,
        hidden={5: [27]}, flexion=1.4, monocular_flexion=.5)
    write_json(source/'evidence.json', first)
    write_json(source/'source.json', {'artifact': 'source.mp4', 'sha256': first['source_sha256']})
    write_json(source/'task-proposal.json', {'selected_joint': 'left_knee'})
    write_json(source/'manifest.json', {'files': {p.name: digest(p) for p in source.iterdir() if p.is_file()}})
    for name, value in [('first-camera.json', camera), ('second-camera.json', camera2),
                        ('second-evidence.json', second), ('ground.json', ground), ('uncertainty.json', SETTINGS)]:
        write_json(tmp_path/name, value)
    write_json(tmp_path/'config.json', {'schema_version': 1, 'first_name': 'first', 'second_name': 'second',
        'second_evidence': 'second-evidence.json', 'second_camera': 'second-camera.json',
        'clock_provenance': 'Known analytic clock', 'world_provenance': 'Known analytic common world'})
    out = tmp_path/'lab'; original = verify_artifacts(source)
    monkeypatch.setattr(sys, 'argv', ['weft', 'camera-body', '--run', str(source), '--camera', str(tmp_path/'first-camera.json'),
        '--ground', str(tmp_path/'ground.json'), '--root-view', str(tmp_path/'config.json'), '--body-view', 'both',
        '--fit-limbs', '--partial-segments', '--fit-partial-pixels', '--partial-pixel-uncertainty', str(tmp_path/'uncertainty.json'),
        '--partial-depth-branches', '--out', str(out)])
    main(); run = out/'run'; verified = verify_artifacts(run)
    assert json.loads((run/'report.json').read_text())['depth_branches']['selected_group_poses'] == 1
    lab = Lab(out)
    try:
        lab.load_session(); assert lab.session.scene['human']['depth_branches']['selected_group_poses'] == 1
        with pytest.raises(ValueError, match='no learned policy'): lab.control('/api/step', {'policy': True})
        lab.control('/api/record/start', {})
        for action in (1, 2, 7, 9): lab.control('/api/step', {'action': action, 'steps': 2})
        lab.control('/api/reset', {'seed': 4})
        stopped = lab.control('/api/record/stop', {}); lab.export_future.result(timeout=15)
        recording = out/'sessions'/stopped['directory']; replay = CameraHumanSession(run)
        try:
            for event in load_recording(recording)['events']:
                if event['type'] in ('start', 'reset'): replay.reset(event['seed'])
                else:
                    reward, terminated, truncated, _ = replay.step(event['action'])
                    assert (reward, terminated, truncated) == (event['reward'], event['terminated'], event['truncated'])
                assert replay.snapshot() == event['snapshot']
        finally: replay.env.close()
        assert verify_recording(recording)['video_frames'] == 10
    finally: lab.close()
    assert verify_artifacts(source) == original and verify_artifacts(run) == verified
