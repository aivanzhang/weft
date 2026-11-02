"""Retain nominal depth alternatives; select only with image and temporal agreement."""

import math
from types import MappingProxyType

import numpy as np

from weft.camera_pose import _undistort
from weft.constrained_stereo import _camera, _project
from weft.distal_pixel_fit import distal_geometry
from weft.limb_position_fit import _array


CONTRACT = MappingProxyType({
    'method': 'temporal-distal-ray-sphere-v1', 'maximum_views': 4, 'maximum_roots': 8,
    'maximum_body_seconds': 20., 'maximum_anchor_gap_s': .25, 'minimum_anchor_direction_dot': 0.,
    'maximum_axial_change_rad': math.pi, 'maximum_bend_rad': 2.6,
    'maximum_temporal_distance_lengths': .5, 'minimum_branch_gap_lengths': .25,
    'duplicate_endpoint_tolerance_m': 1e-8, 'minimum_pixel_improvement': 1e-12,
    'minimum_temporal_improvement': 1e-10,
    'anchors': 'Nearest complete source-supported fitted limbs on both sides; never partial or selected hypotheses',
    'selection': 'All-view pixel SSE and interpolated-direction distance must improve; reject distant or ambiguous branches',
    'fixed': 'Upper-bone axis/origin, root, lengths, ground, forces, complete fits and source observations',
    'scope': 'Nominal ray/sphere alternatives with an assumed smooth-motion selector, not observed depth or physical confidence',
    'floor_selection': False, 'depth_measured': False,
})


def propose_depth_branches(rotation, bend, length, knee, cameras, pixels, confidence, sign, *, anchor_direction=None):
    orientation = _array(rotation, (3, 3), 1.00001, 'branch orientation')
    origin = _array(knee, (3,), 2000., 'branch knee/elbow')
    if (not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-8, rtol=0)
            or not math.isclose(float(np.linalg.det(orientation)), 1., abs_tol=1e-8)
            or type(bend) not in (int, float) or not math.isfinite(bend) or not 0 <= bend <= 2.6
            or type(length) not in (int, float) or not math.isfinite(length) or not .01 <= length <= 1.
            or type(sign) is not int or sign not in (-1, 1)
            or not isinstance(cameras, (list, tuple)) or not 1 <= len(cameras) <= 4):
        raise ValueError('Depth branches require a proper fixed limb and one to four bounded camera views')
    views = [_camera(c) for c in cameras]
    targets = _array(pixels, (len(views), 2), 16384., 'branch pixels')
    weights = _array(confidence, (len(views),), 1., 'branch confidence')
    if np.min(weights) < .5 or any(np.any(p < 0) or np.any(p > [c['width'], c['height']])
                                 for p, c in zip(targets, views, strict=True)):
        raise ValueError('Depth branches require qualified original image observations')
    anchor = None if anchor_direction is None else _array(anchor_direction, (3,), 1.00001, 'temporal direction')
    if anchor is not None and not math.isclose(float(np.linalg.norm(anchor)), 1., abs_tol=1e-8):
        raise ValueError('Temporal direction must be normalized explicitly')

    def score(point):
        errors = [float(np.linalg.norm(_project(point[None, :], c)[0][0]-p)) for c, p in zip(views, targets, strict=True)]
        sse = float(sum(w*(e/max(c['width'], c['height']))**2 for w, e, c in zip(weights, errors, views, strict=True)))
        distance = None if anchor is None else float(np.linalg.norm((point-origin)/length-anchor))
        return errors, sse, distance

    before, _ = distal_geometry([0., 0.], orientation, bend, length, origin, sign)
    errors, before_sse, before_distance = score(before)
    report = {'method': CONTRACT['method'], 'status': 'no_qualified_branch', 'selected_index': None,
        'input_rotation_world': orientation.tolist(), 'input_bend_rad': bend, 'fixed_knee_world_m': origin.tolist(),
        'fixed_distal_length_m': length, 'bend_sign': sign, 'before_world_m': before.tolist(),
        'pixel_errors_before': errors, 'pixel_sse_before': before_sse,
        'temporal_distance_before': before_distance, 'anchor_direction': None if anchor is None else anchor.tolist(),
        'rays': [], 'candidates': [], 'runner_up_gap_lengths': None, 'depth_measured': False}
    for index, (view, target) in enumerate(zip(views, targets, strict=True)):
        ray_record = {'view_index': index, 'status': 'no_sphere_intersection', 'roots': [], 'discriminant': None}
        report['rays'].append(ray_record)
        normalized = _undistort(target[None, :], view['matrix'], view['radial'])
        if normalized is None:
            ray_record['status'] = 'distortion_not_invertible'; continue
        center = -view['rotation'].T @ view['translation']
        ray = view['rotation'].T @ np.r_[normalized[0], 1.]; ray /= np.linalg.norm(ray)
        offset = origin-center; along = float(offset @ ray)
        perpendicular = offset-along*ray
        discriminant = length**2-float(perpendicular @ perpendicular)
        ray_record['discriminant'] = discriminant
        if discriminant < 0: continue
        ray_record['status'] = 'intersections_retained'
        for depth in (along-math.sqrt(discriminant), along+math.sqrt(discriminant)):
            point = center+depth*ray
            root = {'distance': depth, 'world_m': point.tolist(), 'candidate_index': None, 'rejection': None}
            ray_record['roots'].append(root)
            if depth <= 0:
                root['rejection'] = 'nonforward_root'; continue
            duplicate = next((i for i, c in enumerate(report['candidates']) if np.linalg.norm(
                point-np.array(c['world_m'])) <= CONTRACT['duplicate_endpoint_tolerance_m']), None)
            if duplicate is not None:
                root['candidate_index'] = duplicate; root['rejection'] = 'duplicate_endpoint'
                continue
            local = orientation.T @ (point-origin)/length
            absolute_bend = math.acos(float(np.clip(-local[2], -1., 1.)))
            axial_unknown = math.hypot(local[0], local[1]) <= 1e-8
            twist = 0. if axial_unknown else math.atan2(-sign*local[0], sign*local[1])
            proposal = {'world_m': point.tolist(), 'correction_rad': [twist, absolute_bend-bend],
                'output_bend_rad': absolute_bend, 'axial_twist_unobservable': axial_unknown,
                'pixel_errors': None, 'pixel_sse': None, 'temporal_distance': None, 'rejection': None}
            root['candidate_index'] = len(report['candidates']); report['candidates'].append(proposal)
            if absolute_bend > 2.6:
                proposal['rejection'] = 'hinge_range'; continue
            try: errors, sse, distance = score(point)
            except ValueError:
                proposal['rejection'] = 'projection_outside_envelope'; continue
            proposal.update(pixel_errors=errors, pixel_sse=sse, temporal_distance=distance)
            if sse >= before_sse-CONTRACT['minimum_pixel_improvement']:
                proposal['rejection'] = 'no_all_view_image_improvement'
            elif anchor is None:
                proposal['rejection'] = 'missing_temporal_anchor'
            elif distance > .5 or distance >= before_distance-CONTRACT['minimum_temporal_improvement']:
                proposal['rejection'] = 'insufficient_temporal_agreement'
    if anchor is None:
        report['status'] = 'missing_temporal_anchor'; return report
    # A runner just outside the temporal cutoff still makes a nearby winner ambiguous.
    ranked = sorted((c['temporal_distance'], i) for i, c in enumerate(report['candidates'])
                    if c['rejection'] in (None, 'insufficient_temporal_agreement'))
    if ranked and report['candidates'][ranked[0][1]]['rejection'] is None:
        gap = ranked[1][0]-ranked[0][0] if len(ranked) > 1 else None
        report['runner_up_gap_lengths'] = gap
        if gap is not None and gap < .25:
            report['status'] = 'ambiguous_branches'
        else:
            report.update(status='selected_temporal_hypothesis', selected_index=ranked[0][1])
    return report
