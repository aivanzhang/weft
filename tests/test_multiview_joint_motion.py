"""Acceleration-prior construction uses the existing source, native body and recording path."""

from copy import deepcopy

import numpy as np
import pytest

import test_multiview_joint_pixels as integration
from test_camera_ground import references
from test_joint_motion_trajectory import MOTION
from test_multiview_depth_branches import source_inputs
from weft.human import validate_human
from weft.io import canonical_hash
from weft.multiview_body import make_two_view_body
from weft.multiview_joint_pixels import restore_joint_pixel_input


def construct(args):
    root, predictions, evidence, cameras = args
    return make_two_view_body(root, predictions, *evidence[:2], additional_evidence=evidence[2:],
        limb_fit=True, partial_segments=True, pixel_cameras=cameras, joint_pixel_uncertainty=MOTION)


@pytest.mark.parametrize('views', [2, 4])
def test_motion_prior_preserves_source_binding_and_matches_native_geometry(tmp_path, views):
    args = source_inputs(tmp_path, views); before = canonical_hash(args)
    original, _ = integration.construct(args, fit=False)
    body, audit = construct(args)
    assert canonical_hash(args) == before
    assert validate_human(body) == body
    restored = restore_joint_pixel_input(body)
    for key in ('reference', 'reference_evidence', 'root_reference', 'lengths', 'body_frame', 'mobility', 'motor_limits'):
        assert restored[key] == original[key]
    assert audit['joint_pixels']['method'] == 'time-coupled-joint-pixels-v2'
    assert audit['joint_pixels']['motion_acceleration_chi2_after'] >= 0
    assert body['camera_reference']['body_refinement']['joint_pixel_fit']['uncertainty'] == MOTION
    native = references(body)
    offset = np.r_[body['camera_reference']['origin_world_m'][:2], body['camera_reference']['ground']['height_m']]
    for group, first, last in [('left_hip', 'left_knee', 'left_ankle'), ('right_hip', 'right_knee', 'right_ankle'),
                               ('left_shoulder', 'left_elbow', 'left_wrist'), ('right_shoulder', 'right_elbow', 'right_wrist')]:
        fit = audit['joint_pixel_trajectories'][group]['fit']
        for i, row in enumerate(native):
            np.testing.assert_allclose([np.array(row['human']['joints'][name])+offset for name in (first, last)],
                                       fit['after']['points_world_m'][i], atol=1e-10)


@pytest.mark.parametrize('damage', ['contract', 'version', 'missing_prior', 'summary_method', 'prior_value', 'bounds'])
def test_motion_profile_cannot_be_resealed_as_legacy_or_widened(tmp_path, damage):
    body, _ = construct(source_inputs(tmp_path))
    pin = body['camera_reference']['body_refinement']['joint_pixel_fit']
    if damage == 'contract': pin['contract']['method'] = 'time-coupled-joint-pixels-v1'
    elif damage == 'version': pin['uncertainty']['schema_version'] = 1
    elif damage == 'missing_prior': del pin['uncertainty']['motion_acceleration_std_rad_s2']
    elif damage == 'summary_method': body['reconstruction_coverage']['two_view_body']['joint_pixels']['method'] = 'time-coupled-joint-pixels-v1'
    elif damage == 'prior_value': pin['uncertainty']['motion_acceleration_std_rad_s2'] = 30.
    else: pin['contract']['maximum_angle_change_rad'] = .6
    with pytest.raises(ValueError): validate_human(body)


def test_motion_settings_use_actual_cli_recording_and_exact_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(integration, 'SETTINGS', deepcopy(MOTION))
    integration.test_joint_image_cli_record_export_replays_exact_states_and_preserves_sources(tmp_path, monkeypatch)
