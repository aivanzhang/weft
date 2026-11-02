"""Source custody and uncertainty for pixel-constrained partial limbs."""

from collections import Counter
import math
import time

import numpy as np

from weft.camera_pose import validate_camera
from weft.distal_pixel_fit import CONTRACT, UNCERTAINTY_CONTRACT, STATUSES, fit_distal_pixels, validate_uncertainty
from weft.io import canonical_hash
from weft.multiview_pose import _observation
from weft.multiview_segments import COMPONENTS, MULTI_COMPONENTS, PARTIAL_STATUS, MULTI_PARTIAL_STATUS
from weft.spatial_pose import LIMB_ENDPOINTS, SPATIAL_JOINT_NAMES, rotation_to_zyx, rotation_zyx


METHOD = 'camera-body-partial-pixels-v1'
UNCERTAINTY_METHOD = 'camera-body-partial-pixels-v2'
OBSERVATION_STATUSES = frozenset(('pose_not_sampled', 'pose_unavailable_or_competing',
    'unsupported_pose_status', 'invalid_landmarks', 'invalid_landmark', 'outside_image', 'low_confidence', 'available'))


def pixel_status(multi):
    return ('multi' if multi else 'two')+'_view_proximal_image_constrained_distal'


def pixel_components(multi):
    return dict(MULTI_COMPONENTS if multi else COMPONENTS) | {
        'axial_twist': 'image_constrained_prior', 'distal_flexion': 'image_constrained_prior',
        'distal_endpoint': 'image_constrained_depth_unresolved'}


def prepare_pixel_views(cameras, evidence, predictions):
    if not isinstance(cameras, (list, tuple)) or len(cameras) != len(evidence):
        raise ValueError('Partial pixels require every original calibrated source view')
    if [canonical_hash(c) for c in cameras] != [v['camera_sha256'] for v in predictions['views']]:
        raise ValueError('Partial pixel cameras differ from the pinned reconstruction')
    return [(*validate_camera(camera, source), camera['width'], camera['height'])
            for camera, source in zip(cameras, evidence, strict=True)]


def apply_distal_pixels(candidate, original, predictions, sources, audits, prepared, *, multi, uncertainty=None):
    settings = validate_uncertainty(uncertainty) if uncertainty is not None else None
    before = canonical_hash(candidate['reference']); previous = {}; fits = []
    deadline = time.monotonic()+CONTRACT['maximum_body_fit_seconds']
    lengths = candidate['lengths']; anchor = np.asarray(candidate['body_frame']['rotation'])
    partial_status = MULTI_PARTIAL_STATUS if multi else PARTIAL_STATUS
    for i, row in enumerate(audits):
        frame = row['frame']; meta = candidate['reference_evidence'][i]; meta['geometry_partial_pixels'] = {}
        joints = candidate['reference'][i]['joints']; meta['geometry_method'] = UNCERTAINTY_METHOD if settings is not None else METHOD
        torso = anchor @ rotation_zyx([joints[name] for name in SPATIAL_JOINT_NAMES[:3]])
        root = np.asarray(candidate['camera_reference']['origin_world_m'])+original['root_reference'][i]['translation']
        for group, indices in LIMB_ENDPOINTS.items():
            detail = row['groups'][group]
            if detail['status'] != partial_status:
                previous.pop(group, None); continue
            if time.monotonic() >= deadline:
                raise TimeoutError('Partial pixel fitting exceeded its declared deadline')
            observations = [_observation(source.get(frame), indices[-1], view[3], view[4])
                            for source, view in zip(sources, prepared, strict=True)]
            statuses = [v[1] for v in observations]
            if statuses != predictions['frames'][frame]['joints'][indices[-1]]['observation_statuses']:
                raise ValueError('Partial image slots changed after triangulation')
            available = [v for v, observation in enumerate(observations) if observation[0] is not None]
            attempt = {'applied': False, 'observation_statuses': statuses, 'view_indices': available, 'views': [], 'reason': None}
            detail['distal_pixel_attempt'] = attempt
            if not available:
                attempt['reason'] = 'No qualified distal image'; previous.pop(group, None); continue
            cameras, pixels, confidence = [], [], []
            for view in available:
                matrix, radial, transforms, width, height = prepared[view]
                rotation, translation = transforms[frame]
                u, _, v = np.linalg.svd(rotation); proper = u @ v
                change = float(np.max(abs(proper-rotation)))
                if np.linalg.det(proper) <= 0 or change > CONTRACT['rotation_roundoff_limit']:
                    raise ValueError('Partial pixel rotation exceeds the declared precision preparation')
                cameras.append({'matrix': matrix, 'radial': radial, 'rotation': proper,
                                'translation': translation, 'width': width, 'height': height})
                pixels.append(observations[view][0])
                observation = sources[view][frame]['landmarks'][indices[-1]]
                weight = min(observation['visibility'], observation['presence']); confidence.append(weight)
                attempt['views'].append({'view_index': view, 'source_evidence_id': sources[view][frame]['id'],
                    'camera_sha256': predictions['views'][view]['camera_sha256'], 'pixel': pixels[-1].tolist(),
                    'confidence': weight, 'solver_rotation': proper.tolist(), 'maximum_rotation_element_change': change})
            leg = group.endswith('hip'); side = -1 if group.startswith('left_') else 1
            distal = group.rsplit('_', 1)[0]+('_knee' if leg else '_elbow')
            names = (group+'_twist', group+'_lateral', group, distal)
            old = [joints[name] for name in names]; local = rotation_zyx(old[:3]); world = torso @ local
            origin = root+torso @ [side*lengths['hip_width' if leg else 'shoulder_width']/2, 0., 0. if leg else lengths['torso']]
            knee = origin+world @ [0., 0., -lengths['thigh' if leg else 'upper_arm']]
            earlier = previous.get(group)
            correction = earlier[1] if earlier and 0 < meta['time_s']-earlier[0] <= CONTRACT['maximum_temporal_gap_s'] else None
            try:
                fit = fit_distal_pixels(world, old[3], lengths['shin' if leg else 'forearm'], knee,
                    cameras, pixels, confidence, -1 if leg else 1, previous=correction, uncertainty=settings)
            except ValueError as error:
                attempt['reason'] = str(error); previous.pop(group, None); continue
            angles, singular = (rotation_to_zyx(local @ rotation_zyx([fit['correction_rad'][0], 0., 0.]), old[:3])
                                if fit['maximum_angle_change_rad'] > 0 else (np.asarray(old[:3]), False))
            for name, value in zip(names, angles.tolist()+[fit['output_bend_rad']], strict=True):
                joints[name] = value; meta['joint_confidence'][name] = 0.
            attempt.update(applied=True, fit=fit, input_angles_rad=old, chart_singular=singular)
            detail['status'] = meta['geometry_sources'][group] = pixel_status(multi)
            detail['components'] = pixel_components(multi)
            detail['transported_distal_prior_radians'] = detail.pop('retained_distal_flexion_radians')
            detail['pixel_corrected_local_angles'] = [joints[name] for name in names]
            meta['geometry_components'][group] = pixel_components(multi)
            meta['geometry_partial_pixels'][group] = {'input_angles_rad': old, 'correction_rad': fit['correction_rad'],
                'status': fit['status'], 'pixel_sse_before': fit['pixel_sse_before'], 'pixel_sse_after': fit['pixel_sse_after'],
                'view_indices': available, 'observation_statuses': statuses}
            previous[group] = (meta['time_s'], fit['correction_rad']); fits.append(fit)
    report = {'method': (UNCERTAINTY_CONTRACT if settings is not None else CONTRACT)['method'], 'partial_reference_sha256': before, 'fitted_group_poses': len(fits),
        'changed_group_poses': sum(f['maximum_angle_change_rad'] > 0 for f in fits),
        'status_counts': dict(Counter(f['status'] for f in fits)),
        'maximum_angle_change_rad': max((f['maximum_angle_change_rad'] for f in fits), default=0.),
        'source_image_constraints': sum(len(f['source_pixels']) for f in fits),
        'scope': CONTRACT['scope'], 'depth_measured': False}
    if settings is not None:
        report.update(uncertainty=settings, uncertainty_sha256=canonical_hash(settings), uncertainty_calibrated=False,
            weak_image_group_poses=sum(f['weak_image_directions'] > 0 for f in fits))
    return report


def validate_pixel_binding(spec, parent, *, multi):
    binding = spec['camera_reference']['body_refinement']
    views = spec['camera_reference']['root_refinement']['views']
    uncertain = spec['camera_reference']['method'] == UNCERTAINTY_METHOD
    if (binding['partial_pixel_contract'] != dict(UNCERTAINTY_CONTRACT if uncertain else CONTRACT)
            or binding['partial_pixel_camera_sha256'] != [v['camera_sha256'] for v in views]):
        raise ValueError('Partial pixels require their fixed contract and original calibrated view identities')
    if uncertain:
        settings = validate_uncertainty(binding['partial_pixel_uncertainty'])
        summary = spec['reconstruction_coverage']['two_view_body']['partial_pixels']
        if (binding['partial_pixel_uncertainty_sha256'] != canonical_hash(settings)
                or summary.get('uncertainty') != settings or summary.get('uncertainty_sha256') != canonical_hash(settings)
                or summary.get('uncertainty_calibrated') is not False):
            raise ValueError('Partial image uncertainty must preserve its explicit uncalibrated assumptions')
    for meta, reference, prior in zip(spec['reference_evidence'], spec['reference'], parent['reference'], strict=True):
        fits = meta.get('geometry_partial_pixels')
        expected = {name for name, source in meta['geometry_sources'].items() if source == pixel_status(multi)}
        if not isinstance(fits, dict) or set(fits) != expected:
            raise ValueError('Partial pixel provenance must retain every applied image-constrained limb')
        for group, fit in fits.items():
            fields = {'input_angles_rad', 'correction_rad', 'status', 'pixel_sse_before', 'pixel_sse_after',
                      'view_indices', 'observation_statuses'}
            if not isinstance(fit, dict) or set(fit) != fields:
                raise ValueError('Partial pixel provenance differs from its bounded schema')
            old, delta = fit['input_angles_rad'], fit['correction_rad']; costs = [fit['pixel_sse_before'], fit['pixel_sse_after']]
            statuses, indices = fit['observation_statuses'], fit['view_indices']
            if (not isinstance(old, list) or len(old) != 4 or not isinstance(delta, list) or len(delta) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in old+delta+costs)
                    or any(abs(v) > math.tau for v in old) or not 0 <= old[3] <= 2.6
                    or any(abs(v) > .30 for v in delta) or not 0 <= costs[1] <= costs[0]
                    or not isinstance(fit['status'], str) or fit['status'] not in STATUSES
                    or not isinstance(statuses, list) or len(statuses) != len(views)
                    or any(not isinstance(v, str) or v not in OBSERVATION_STATUSES for v in statuses)
                    or not isinstance(indices, list) or not indices or any(type(v) is not int for v in indices)
                    or indices != [i for i, status in enumerate(statuses) if status == 'available']):
                raise ValueError('Partial pixel correction, objective or complete image slots changed')
            names = (group+'_twist', group+'_lateral', group)
            distal = group.rsplit('_', 1)[0]+('_knee' if group.endswith('hip') else '_elbow')
            expected_rotation = rotation_zyx(old[:3]) @ rotation_zyx([delta[0], 0., 0.])
            actual = rotation_zyx([reference['joints'][name] for name in names])
            if (old[3] != prior['joints'][distal] or not np.allclose(actual, expected_rotation, atol=1e-10, rtol=0)
                    or abs(reference['joints'][distal]-old[3]-delta[1]) > 1e-12
                    or meta['joint_confidence'].get(distal) != 0.):
                raise ValueError('Partial pixels cannot change the supported upper axis or promote inferred bend')
