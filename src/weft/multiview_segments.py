"""Use a supported proximal segment without inventing its missing distal endpoint."""

import math
from types import MappingProxyType

import numpy as np

from weft.io import canonical_hash
from weft.spatial_pose import LIMB_ENDPOINTS, SPATIAL_JOINT_NAMES, rotation_to_zyx, rotation_zyx


METHOD = 'two-view-body-v2'
CONTRACT = MappingProxyType({'method': 'two-view-proximal-direction-v1',
    'minimum_segment_length_m': 1e-5, 'antiparallel_cosine_margin': 1e-8,
    'orientation': 'Minimum rotation transports the monocular limb frame to the two-view proximal direction',
    'unobserved': 'Axial twist and distal flexion remain monocular priors; no missing endpoint is filled',
    'complete_limb_angles_unchanged': True, 'dimensions_unchanged': True, 'ground_fitting': False})
PARTIAL_STATUS = 'two_view_proximal_monocular_distal_and_twist'
COMPONENTS = {'proximal_direction': 'two_view_estimate', 'axial_twist': 'transported_monocular_prior',
              'distal_flexion': 'monocular_prior', 'distal_endpoint': 'unavailable'}


def transport_rotation(prior, direction):
    prior, direction = np.asarray(prior, dtype=float), np.asarray(direction, dtype=float)
    if (prior.shape != (3, 3) or direction.shape != (3,) or not np.isfinite(prior).all()
            or not np.isfinite(direction).all() or not np.allclose(prior.T @ prior, np.eye(3), atol=1e-6)
            or not math.isclose(float(np.linalg.det(prior)), 1., abs_tol=1e-6)):
        raise ValueError('Segment transport requires a proper prior rotation and finite direction')
    length = float(np.linalg.norm(direction))
    if length < CONTRACT['minimum_segment_length_m']:
        raise ValueError('Proximal segment is degenerate')
    target, initial = direction/length, -prior[:, 2]
    cosine = float(np.clip(initial @ target, -1., 1.))
    if cosine < -1+CONTRACT['antiparallel_cosine_margin']:
        raise ValueError('Antiparallel direction leaves the transport axis ambiguous')
    axis = np.cross(initial, target)
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    rotation = (np.eye(3)+cross+cross @ cross/(1+cosine)) @ prior
    return rotation, math.acos(cosine)


def apply_partial_segments(candidate, original, predictions, sources, audits):
    from weft.multiview_body import _confidence, _points

    before = canonical_hash(candidate['reference'])
    anchor = np.asarray(candidate['body_frame']['rotation'])
    changed, rejected = 0, 0
    for i, row in enumerate(audits):
        frame, groups = row['frame'], row['groups']
        pose, joints = predictions['frames'][frame], candidate['reference'][i]['joints']
        original_joints = original['reference'][i]['joints']
        meta = candidate['reference_evidence'][i]
        torso = anchor @ rotation_zyx([joints[name] for name in SPATIAL_JOINT_NAMES[:3]])
        components = {}
        for group, required in LIMB_ENDPOINTS.items():
            detail = groups[group]
            if detail['status'] != 'monocular_prior_missing_points':
                continue
            pair = _points(pose, required[:2])
            if pair is None:
                continue
            names = (group+'_twist', group+'_lateral', group)
            prior_angles = [original_joints[name] for name in names]
            distal = group.rsplit('_', 1)[0]+('_knee' if group.endswith('hip') else '_elbow')
            detail['proximal_image_confidence'] = _confidence(frame, required[:2], sources)
            try:
                direction = torso.T @ (pair[1]-pair[0])
                rotation, swing = transport_rotation(rotation_zyx(prior_angles), direction)
                angles, singular = rotation_to_zyx(rotation, prior_angles)
            except ValueError as error:
                detail.update(segment_transport_applied=False, segment_transport_rejection=str(error))
                rejected += 1
                continue
            for name, value in zip(names, angles.tolist(), strict=True):
                joints[name] = value
                # One observed axis does not independently identify any of the three Euler coordinates.
                meta['joint_confidence'][name] = 0.
            if joints[distal] != original_joints[distal]:
                raise ValueError('Partial-segment correction must preserve the original unobserved distal bend')
            detail.update(status=PARTIAL_STATUS, applied=True, segment_transport_applied=True,
                segment_swing_radians=swing, proximal_world_points_m=pair.tolist(),
                original_local_angles=prior_angles, corrected_local_angles=angles.tolist(),
                retained_distal_flexion_radians=joints[distal], chart_singular=singular,
                chart_wrap=bool(np.max(np.abs(angles-prior_angles)) > math.pi),
                assumed_twist=True, components=dict(COMPONENTS))
            components[group] = dict(COMPONENTS)
            meta['geometry_sources'][group] = PARTIAL_STATUS
            changed += 1
        meta['geometry_method'] = METHOD
        if components:
            meta['geometry_components'] = components
    return {'contract': dict(CONTRACT), 'complete_limb_reference_sha256': before,
            'applied_group_poses': changed, 'rejected_group_poses': rejected,
            'scope': 'Supported upper-segment direction only; distal bend/axial orientation retain explicit priors'}
