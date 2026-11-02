from copy import deepcopy

import numpy as np
import pytest

import test_multiview_joint_pixels as integration
from test_camera_ground import references
from test_joint_motion_trajectory import MOTION
from test_limb_twist_gauge import inputs
from test_multiview_depth_branches import source_inputs
from weft.human import validate_human
from weft.io import canonical_hash
from weft.joint_pixel_trajectory import fit_joint_trajectory, trajectory_contract
from weft.multiview_body import make_two_view_body
from weft.multiview_joint_pixels import restore_joint_pixel_input


GAUGE = dict(MOTION, schema_version=3)


def construct(args):
    root, predictions, evidence, cameras = args
    return make_two_view_body(root, predictions, *evidence[:2], additional_evidence=evidence[2:],
        limb_fit=True, partial_segments=True, pixel_cameras=cameras, joint_pixel_uncertainty=GAUGE)


@pytest.mark.parametrize('n', [2, 3, 9])
def test_staged_gauge_keeps_exact_images_and_retains_every_original_stage(n):
    args = inputs(n)
    for key in ('image_std_fraction', 'motion_acceleration_std_rad_s2', 'provenance'): args.pop(key)
    fit = fit_joint_trajectory(**args, uncertainty=GAUGE)
    original = fit_joint_trajectory(**args, uncertainty=MOTION)
    assert fit['initial_motion_fit'] == original
    assert fit['method'] == 'time-coupled-joint-pixels-v3'
    assert fit['after']['image_chi2'] <= fit['before']['image_chi2']+1e-10
    assert fit['after']['objective'] <= fit['before']['objective']+1e-10
    np.testing.assert_allclose(fit['before']['points_world_m'], fit['after']['points_world_m'], atol=1e-14)
    if n >= 3:
        assert fit['maximum_angle_change_rad'] > .3
        assert fit['after']['motion_acceleration_chi2'] < fit['before']['motion_acceleration_chi2']*1e-6


@pytest.mark.parametrize('views', [2, 4])
def test_gauged_constructor_restores_source_and_matches_native_geometry(tmp_path, views):
    args = source_inputs(tmp_path, views); pin = canonical_hash(args)
    original, _ = integration.construct(args, fit=False); body, audit = construct(args)
    assert canonical_hash(args) == pin and validate_human(body) == body
    restored = restore_joint_pixel_input(body)
    for key in ('reference', 'reference_evidence', 'root_reference', 'lengths', 'body_frame', 'mobility', 'motor_limits'):
        assert restored[key] == original[key]
    summary = audit['joint_pixels']; assert summary['method'] == 'time-coupled-joint-pixels-v3'
    assert summary['axial_gauge']['force_feasibility'] == 'unverified'
    assert summary['axial_gauge']['maximum_distal_displacement_m'] <= .02+1e-10
    native = references(body)
    offset = np.r_[body['camera_reference']['origin_world_m'][:2], body['camera_reference']['ground']['height_m']]
    for group, first, last in [('left_hip', 'left_knee', 'left_ankle'), ('right_hip', 'right_knee', 'right_ankle'),
                              ('left_shoulder', 'left_elbow', 'left_wrist'), ('right_shoulder', 'right_elbow', 'right_wrist')]:
        fit = audit['joint_pixel_trajectories'][group]['fit']
        for i, row in enumerate(native):
            np.testing.assert_allclose([np.array(row['human']['joints'][name])+offset for name in (first, last)],
                                       fit['after']['points_world_m'][i], atol=1e-10)


@pytest.mark.parametrize('damage', ['base_bound', 'gauge_bound', 'hinge', 'swing', 'record', 'image_guard', 'force_label', 'summary', 'contract'])
def test_resealed_gauge_cannot_escape_source_geometry_or_claim_force_feasibility(tmp_path, damage):
    body, _ = construct(source_inputs(tmp_path)); binding = body['camera_reference']['body_refinement']
    record = body['reference_evidence'][5]['geometry_joint_pixels']['left_hip']
    summary = body['reconstruction_coverage']['two_view_body']['joint_pixels']
    if damage == 'base_bound': record['initial_motion_correction_rad'][0] = .31
    elif damage == 'gauge_bound': record['axial_rotation_rad'] = 1.6
    elif damage == 'hinge': record['correction_rad'][3] += .01
    elif damage == 'swing': record['correction_rad'][1] += .01
    elif damage == 'record': del record['initial_motion_correction_rad']
    elif damage == 'image_guard': summary['axial_gauge']['image_chi2_after'] = summary['axial_gauge']['image_chi2_before']+1
    elif damage == 'force_label': summary['axial_gauge']['force_feasibility'] = 'verified'
    elif damage == 'summary': summary['axial_gauge']['changed_group_poses'] += 1
    else: binding['joint_pixel_fit']['contract']['axial_gauge_contract']['maximum_distal_displacement_m'] = .1
    binding['output_evidence_sha256'] = canonical_hash(body['reference_evidence'])
    body['camera_reference']['angular_reference_sha256'] = canonical_hash(body['reference'])
    with pytest.raises(ValueError): validate_human(body)


def test_v3_is_opt_in_with_explicit_staged_bounds():
    assert trajectory_contract(MOTION)['maximum_angle_change_rad'] == .3
    contract = trajectory_contract(GAUGE)
    assert contract['maximum_angle_change_rad'] is None
    assert contract['initial_fit_maximum_angle_change_rad'] == .3
    assert contract['axial_gauge_contract']['maximum_distal_displacement_m'] == .02


def test_gauge_cli_recording_uses_the_existing_saved_world_path(tmp_path, monkeypatch):
    monkeypatch.setattr(integration, 'SETTINGS', deepcopy(GAUGE))
    integration.test_joint_image_cli_record_export_replays_exact_states_and_preserves_sources(tmp_path, monkeypatch)
