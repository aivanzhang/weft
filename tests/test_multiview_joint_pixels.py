from copy import deepcopy
import json
import sys

import numpy as np
import pytest

from test_camera_ground import references
from test_joint_pixel_trajectory import SETTINGS
from test_multiview_body import setup
from test_multiview_depth_branches import source_inputs
from weft.camera_body_lab import prepare_camera_body_lab
from weft.cli import main
from weft.human import HumanEnv, validate_human
from weft.io import canonical_hash, digest, write_json
from weft.lab import CameraHumanSession, Lab
from weft.multiview_body import make_two_view_body
from weft.multiview_joint_pixels import METHOD, restore_joint_pixel_input
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, verify_recording


def construct(args, *, fit):
    root, predictions, evidence, cameras = args
    return make_two_view_body(root, predictions, *evidence[:2], additional_evidence=evidence[2:],
        limb_fit=True, partial_segments=True, pixel_cameras=cameras if fit else (),
        joint_pixel_uncertainty=SETTINGS if fit else None)


@pytest.mark.parametrize('views', [2, 3, 4])
def test_joint_trajectory_preserves_sources_and_prior_fits_with_native_geometry(tmp_path, views):
    args = source_inputs(tmp_path, views); pin = canonical_hash(args)
    before, previous = construct(args, fit=False); body, audit = construct(args, fit=True)
    assert canonical_hash(args) == pin and body['camera_reference']['method'] == METHOD
    summary = audit['joint_pixels']
    assert summary['fitted_trajectories'] == 4 and summary['unavailable_trajectories'] == 0
    assert summary['changed_group_poses'] > 0 and summary['image_chi2_after'] < summary['image_chi2_before']
    assert not summary['uncertainty_calibrated'] and not summary['depth_measured']
    restored = restore_joint_pixel_input(body)
    for key in ('reference', 'reference_evidence', 'camera_reference', 'reconstruction_coverage'):
        assert restored[key] == before[key], key
    for key in ('lengths', 'body_frame', 'root_reference', 'mobility', 'motor_limits'):
        assert body[key] == before[key], key
    for key in ('dimensions', 'limb_position_fit', 'partial_segments', 'group_coverage'):
        assert audit[key] == previous[key], key
    assert canonical_hash(audit) == body['camera_reference']['body_refinement']['audit_sha256']
    native = references(body); offset = np.r_[body['camera_reference']['origin_world_m'][:2], body['camera_reference']['ground']['height_m']]
    for group, name in [('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
                        ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow')]:
        distal = name.replace('knee', 'ankle').replace('elbow', 'wrist')
        fit = audit['joint_pixel_trajectories'][group]['fit']
        for frame, row in enumerate(native):
            np.testing.assert_allclose([np.array(row['human']['joints'][n])+offset for n in (name, distal)],
                                       fit['after']['points_world_m'][frame], atol=1e-10)
            detail = audit['frames'][frame]['groups'][group]
            assert {k: v for k, v in detail.items() if k != 'joint_pixel_attempt'} == previous['frames'][frame]['groups'][group]
    assert len(audit['frames'][5]['groups']['left_hip']['joint_pixel_attempt']['views']) == views+1
    env = HumanEnv(body)
    try:
        assert env.scene['human']['joint_pixels'] == summary and not env.scene['learned_policy_available']
    finally: env.close()


@pytest.mark.parametrize('damage', ['contract', 'original_reference', 'original_metadata', 'original_audit', 'method',
    'missing_group', 'missing_endpoint', 'view', 'view_boolean', 'status', 'before', 'correction', 'final_angle',
    'prior_confidence', 'confidence', 'settings', 'uncertainty_label', 'depth', 'changed_count', 'constraint_count'])
def test_resealed_joint_trajectory_cannot_hide_source_or_angle_changes(tmp_path, damage):
    body, _ = construct(source_inputs(tmp_path), fit=True)
    binding = body['camera_reference']['body_refinement']; pin = binding['joint_pixel_fit']
    meta = body['reference_evidence'][5]; record = meta['geometry_joint_pixels']['left_hip']
    summary = body['reconstruction_coverage']['two_view_body']['joint_pixels']
    if damage == 'contract': pin['contract']['maximum_angle_change_rad'] = .5
    elif damage == 'original_reference': pin['input_reference_sha256'] = 'f'*64
    elif damage == 'original_metadata': pin['input_evidence_sha256'] = 'f'*64
    elif damage == 'original_audit': pin['input_audit_sha256'] = 'bad'
    elif damage == 'method': pin['input_method'] = METHOD
    elif damage == 'missing_group': del meta['geometry_joint_pixels']['left_hip']
    elif damage == 'missing_endpoint': del record['observation_statuses']['27']
    elif damage == 'view': record['view_indices']['27'] = [1]
    elif damage == 'view_boolean': record['view_indices']['27'] = [False]
    elif damage == 'status': record['status'] = 'unchanged_no_observations'
    elif damage == 'before': record['input_angles_rad'][0] += .01
    elif damage == 'correction': record['correction_rad'][0] = .31
    elif damage == 'final_angle': body['reference'][5]['joints']['left_knee'] += .01
    elif damage == 'prior_confidence': record['input_joint_confidence'][0] = 2.
    elif damage == 'confidence': meta['joint_confidence']['left_knee'] = .9
    elif damage == 'settings': pin['uncertainty']['image_std_fraction'] = 0.
    elif damage == 'uncertainty_label': summary['uncertainty_calibrated'] = True
    elif damage == 'depth': summary['depth_measured'] = True
    elif damage == 'changed_count': summary['changed_group_poses'] += 1
    elif damage == 'constraint_count': summary['source_image_constraints'] -= 1
    binding['output_evidence_sha256'] = canonical_hash(body['reference_evidence'])
    body['camera_reference']['angular_reference_sha256'] = canonical_hash(body['reference'])
    with pytest.raises(ValueError): validate_human(body)


def test_unavailable_projection_preserves_prior_angles_and_every_original_image(tmp_path, monkeypatch):
    args = source_inputs(tmp_path); before, _ = construct(args, fit=False)
    def reject(*a, **kw): raise ValueError('Projection exceeds the forward depth envelope')
    monkeypatch.setattr('weft.multiview_joint_pixels.fit_joint_trajectory', reject)
    body, audit = construct(args, fit=True)
    assert body['reference'] == before['reference'] and audit['joint_pixels']['unavailable_trajectories'] == 4
    assert audit['joint_pixels']['source_image_constraints'] > 0 and audit['joint_pixels']['changed_group_poses'] == 0
    assert all(v['reason'] == 'Projection exceeds the forward depth envelope' for v in audit['joint_pixel_trajectories'].values())


def test_joint_fit_rejects_stacking_and_mutated_cameras(tmp_path):
    root, predictions, evidence, cameras = source_inputs(tmp_path)
    with pytest.raises(ValueError, match='cannot stack'):
        make_two_view_body(root, predictions, *evidence, limb_fit=True, partial_segments=True, partial_pixel_fit=True,
                           pixel_cameras=cameras, joint_pixel_uncertainty=SETTINGS)
    cameras[0]['frames'][0]['t'][0] += .1
    with pytest.raises(ValueError, match='cameras differ'):
        make_two_view_body(root, predictions, *evidence, limb_fit=True, partial_segments=True,
                           pixel_cameras=cameras, joint_pixel_uncertainty=SETTINGS)


def test_joint_flag_requires_explicit_constructor_before_file_reads(tmp_path):
    with pytest.raises(ValueError, match='Joint images require'):
        prepare_camera_body_lab(tmp_path/'source', tmp_path/'camera', tmp_path/'out', joint_pixel_uncertainty_path=tmp_path/'absent')
    assert not (tmp_path/'out').exists()


def test_joint_image_cli_record_export_replays_exact_states_and_preserves_sources(tmp_path, monkeypatch):
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
        '--fit-limbs', '--partial-segments', '--fit-joint-pixels', str(tmp_path/'uncertainty.json'), '--out', str(out)])
    main(); run = out/'run'; verified = verify_artifacts(run)
    assert (run/'joint-pixel-uncertainty.json').read_bytes() == (tmp_path/'uncertainty.json').read_bytes()
    assert json.loads((run/'report.json').read_text())['joint_pixels']['changed_group_poses'] > 0
    lab = Lab(out)
    try:
        lab.load_session(); assert lab.session.scene['human']['joint_pixels']['changed_group_poses'] > 0
        with pytest.raises(ValueError, match='no learned policy'): lab.control('/api/step', {'policy': True})
        lab.control('/api/record/start', {})
        for action in (1, 2, 7, 9): lab.control('/api/step', {'action': action, 'steps': 2})
        lab.control('/api/reset', {'seed': 4}); stopped = lab.control('/api/record/stop', {})
        lab.export_future.result(timeout=15); recording = out/'sessions'/stopped['directory']; replay = CameraHumanSession(run)
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
