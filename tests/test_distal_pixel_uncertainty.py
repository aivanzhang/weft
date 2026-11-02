from copy import deepcopy
import json

import numpy as np
import pytest

from test_camera_ground import references
from test_distal_pixel_fit import camera
from test_multiview_distal_pixels import inputs
from test_multiview_extra_lab import package
from weft.camera_body_lab import prepare_camera_body_lab
from weft.constrained_stereo import _project
from weft.distal_pixel_fit import distal_geometry, fit_distal_pixels, validate_uncertainty
from weft.human import validate_human
from weft.io import canonical_hash
from weft.multiview_body import make_two_view_body
from weft.spatial_pose import rotation_zyx


SETTINGS = {'schema_version': 1, 'image_std_fraction': .0025, 'angle_prior_std_rad': .3,
            'temporal_prior_std_rad': .45, 'provenance': 'Synthetic known-camera qualification; assumed error scales'}


def controlled(depth=4., count=1, bend=1., sign=-1):
    rotation = rotation_zyx([.3, -.2, .7]); origin = np.array([.1, .2, depth])
    cameras = [camera(.1, .5*i) for i in range(count)]
    target, _ = distal_geometry([.15, -.2] if bend else [0., .2], rotation, bend, .35, origin, sign)
    return dict(rotation=rotation, bend=bend, length=.35, knee=origin, cameras=cameras,
        pixels=[_project(target[None, :], c)[0][0].tolist() for c in cameras], confidence=[.9]*count, sign=sign)


@pytest.mark.parametrize('count', [1, 2, 4])
@pytest.mark.parametrize('sign', [-1, 1])
def test_resized_images_preserve_angles_objective_and_weak_direction_report(count, sign):
    original = controlled(count=count, sign=sign)
    baseline = fit_distal_pixels(**original, uncertainty=SETTINGS)
    for scale in (.5, 2.):
        args = deepcopy(original)
        for view in args['cameras']:
            view['width'] = round(view['width']*scale); view['height'] = round(view['height']*scale)
            view['matrix'][:2] *= scale
        args['pixels'] = (np.array(args['pixels'])*scale).tolist()
        fitted = fit_distal_pixels(**args, uncertainty=SETTINGS)
        np.testing.assert_allclose(fitted['correction_rad'], baseline['correction_rad'], atol=1e-12)
        np.testing.assert_allclose(fitted['initial_image_to_prior_eigenvalues'], baseline['initial_image_to_prior_eigenvalues'], atol=1e-12)
        assert fitted['objective_after'] == pytest.approx(baseline['objective_after'], abs=1e-14)
        assert fitted['weak_image_directions'] == baseline['weak_image_directions']


def test_equivalent_error_ratios_preserve_the_solution_and_every_trial():
    args = controlled(count=2); settings = dict(SETTINGS)
    for key in ('image_std_fraction', 'angle_prior_std_rad', 'temporal_prior_std_rad'): settings[key] *= 2
    a = fit_distal_pixels(**args, previous=[.1, -.1], uncertainty=SETTINGS)
    b = fit_distal_pixels(**args, previous=[.1, -.1], uncertainty=settings)
    for key in ('correction_rad', 'iterations', 'initial_image_to_prior_eigenvalues'):
        assert a[key] == b[key]


def test_weaker_image_assumption_keeps_more_of_the_prior_without_claiming_measured_depth():
    args = controlled()
    precise = fit_distal_pixels(**args, uncertainty=SETTINGS)
    noisy = fit_distal_pixels(**args, uncertainty={**SETTINGS, 'image_std_fraction': .02})
    assert precise['pixel_sse_after'] < noisy['pixel_sse_after']
    assert np.linalg.norm(precise['correction_rad']) > np.linalg.norm(noisy['correction_rad'])
    for result in (precise, noisy):
        assert not result['depth_measured'] and not result['uncertainty_calibrated']
        assert result['maximum_angle_change_rad'] <= .3


def test_distant_small_image_motion_has_less_authority_than_near_motion():
    near = fit_distal_pixels(**controlled(depth=3.), uncertainty=SETTINGS)
    far = fit_distal_pixels(**controlled(depth=30.), uncertainty=SETTINGS)
    assert max(far['initial_image_to_prior_eigenvalues']) < min(near['initial_image_to_prior_eigenvalues'])
    assert np.linalg.norm(far['correction_rad']) < np.linalg.norm(near['correction_rad'])
    assert far['weak_image_directions'] >= near['weak_image_directions']


def test_straight_limb_reports_unobservable_initial_twist_and_exact_targets_ignore_temporal_pull():
    args = controlled(bend=0.)
    fitted = fit_distal_pixels(**args, uncertainty=SETTINGS)
    assert fitted['initial_image_to_prior_eigenvalues'][0] == 0.
    assert fitted['weak_image_directions'] >= 1 and not fitted['depth_measured']
    point, _ = distal_geometry([0., 0.], args['rotation'], args['bend'], args['length'], args['knee'], args['sign'])
    args['pixels'] = [_project(point[None, :], c)[0][0].tolist() for c in args['cameras']]
    unchanged = fit_distal_pixels(**args, uncertainty=SETTINGS, previous=[.2, -.2])
    assert unchanged['correction_rad'] == [0., 0.] and unchanged['iterations'] == []


@pytest.mark.parametrize('damage', ['schema', 'extra', 'missing', 'bool', 'nan', 'image_low', 'image_high',
    'angle_low', 'angle_high', 'temporal_low', 'temporal_high', 'provenance', 'long_provenance', 'not_object'])
def test_invalid_uncertainty_cannot_silently_use_defaults(damage):
    value = dict(SETTINGS)
    if damage == 'schema': value['schema_version'] = True
    elif damage == 'extra': value['calibrated'] = True
    elif damage == 'missing': del value['image_std_fraction']
    elif damage == 'bool': value['angle_prior_std_rad'] = True
    elif damage == 'nan': value['image_std_fraction'] = float('nan')
    elif damage == 'image_low': value['image_std_fraction'] = 0.
    elif damage == 'image_high': value['image_std_fraction'] = .051
    elif damage == 'angle_low': value['angle_prior_std_rad'] = .009
    elif damage == 'angle_high': value['angle_prior_std_rad'] = 1.1
    elif damage == 'temporal_low': value['temporal_prior_std_rad'] = .009
    elif damage == 'temporal_high': value['temporal_prior_std_rad'] = 1.1
    elif damage == 'provenance': value['provenance'] = ' '
    elif damage == 'long_provenance': value['provenance'] = 'x'*2001
    elif damage == 'not_object': value = []
    with pytest.raises(ValueError): validate_uncertainty(value)


@pytest.mark.parametrize('count', [2, 3, 4])
def test_explicit_uncertainty_preserves_supported_native_geometry_and_source_custody(tmp_path, count):
    root, predictions, evidence, cameras = inputs(tmp_path, count)
    bound = canonical_hash([root, predictions, evidence, cameras, SETTINGS])
    options = dict(additional_evidence=evidence[2:], limb_fit=True, partial_segments=True,
        partial_pixel_fit=True, pixel_cameras=cameras)
    old, old_audit = make_two_view_body(root, predictions, *evidence[:2], **options)
    new, audit = make_two_view_body(root, predictions, *evidence[:2], **options, pixel_uncertainty=SETTINGS)
    assert canonical_hash([root, predictions, evidence, cameras, SETTINGS]) == bound
    assert old['camera_reference']['method'] == 'camera-body-partial-pixels-v1'
    assert new['camera_reference']['method'] == 'camera-body-partial-pixels-v2'
    assert new['camera_reference']['body_refinement']['partial_pixel_uncertainty_sha256'] == canonical_hash(SETTINGS)
    assert audit['limb_position_fit'] == old_audit['limb_position_fit']
    assert audit['partial_pixels']['uncertainty'] == SETTINGS and not audit['partial_pixels']['uncertainty_calibrated']
    for name in ('lengths', 'root_reference', 'body_frame', 'mobility', 'motor_limits'):
        assert new[name] == old[name]
    for a, b, meta in zip(references(old), references(new), new['reference_evidence'], strict=True):
        for group in meta['geometry_partial_pixels']:
            distal = group.rsplit('_', 1)[0]+('_knee' if group.endswith('hip') else '_elbow')
            for name in (group, distal):
                np.testing.assert_allclose(a['human']['joints'][name], b['human']['joints'][name], atol=1e-12)


@pytest.mark.parametrize('damage', ['settings', 'hash', 'contract', 'summary', 'calibrated', 'method'])
def test_resealed_body_cannot_change_uncertainty_or_promote_its_status(tmp_path, damage):
    root, predictions, evidence, cameras = inputs(tmp_path, 2)
    body, _ = make_two_view_body(root, predictions, *evidence, limb_fit=True, partial_segments=True,
        partial_pixel_fit=True, pixel_cameras=cameras, pixel_uncertainty=SETTINGS)
    binding = body['camera_reference']['body_refinement']; summary = body['reconstruction_coverage']['two_view_body']['partial_pixels']
    if damage == 'settings': binding['partial_pixel_uncertainty']['image_std_fraction'] = .02
    elif damage == 'hash': binding['partial_pixel_uncertainty_sha256'] = 'e'*64
    elif damage == 'contract': binding['partial_pixel_contract']['uncertainty_calibrated'] = True
    elif damage == 'summary': summary['uncertainty'] = {**SETTINGS, 'angle_prior_std_rad': .9}
    elif damage == 'calibrated': summary['uncertainty_calibrated'] = True
    elif damage == 'method': body['camera_reference']['method'] = 'camera-body-partial-pixels-v1'
    with pytest.raises(ValueError): validate_human(body)


@pytest.mark.parametrize('damage', ['no_pixel_fit', 'bytes', 'during_copy'])
def test_uncertainty_file_rejection_never_publishes_ready_lab(tmp_path, monkeypatch, damage):
    import weft.camera_body_lab as module
    source, camera_path, ground, config, extra = package(tmp_path)
    noise = tmp_path/'uncertainty.json'; noise.write_text(json.dumps(SETTINGS)); out = tmp_path/'lab'
    if damage == 'bytes': noise.write_text(' '*65537)
    if damage == 'during_copy':
        original = module.shutil.copyfile
        def mutate(src, dst):
            result = original(src, dst)
            if str(src).endswith('source.mp4'): noise.write_text('{}')
            return result
        monkeypatch.setattr(module.shutil, 'copyfile', mutate)
    with pytest.raises((ValueError, RuntimeError)):
        prepare_camera_body_lab(source, camera_path, out, ground_path=ground, root_view_path=config,
            additional_view_paths=[extra], body_view='both', partial_segments=True, limb_fit=True,
            partial_pixel_fit=damage != 'no_pixel_fit', pixel_uncertainty_path=noise)
    assert not (out/'status.json').exists()
