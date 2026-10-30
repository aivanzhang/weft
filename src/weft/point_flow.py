"""Local image-point histories; optical flow does not verify physical identity."""

from copy import deepcopy
from types import MappingProxyType

import cv2
import numpy as np

from weft.feature_tracks import _stamp


CONTRACT = MappingProxyType({
    'method': 'masked-roundtrip-point-flow-v1', 'maximum_points': 128, 'maximum_frames': 500,
    'maximum_image_pixels': 4194304, 'maximum_time_step_s': .2,
    'window_size': 21, 'pyramid_levels': 3, 'iterations': 30, 'epsilon': .01,
    'minimum_eigenvalue': 1e-4, 'maximum_roundtrip_px': 1., 'maximum_step_px': 64.,
    'patch_size': 9, 'mask_inset_px': 16, 'minimum_patch_std': 4., 'minimum_patch_ncc': .9,
    'anchor_patch_size': 31, 'minimum_anchor_ecc': .9, 'maximum_anchor_correction_px': 3.,
    'anchor_template_border_px': 5, 'anchor_gaussian_size': 5,
    'anchor_scale_bounds': (.5, 2.), 'maximum_anchor_axis_ratio': 3.,
    'minimum_point_separation_px': .75,
    'backward_initialization': 'Known previous point, full-resolution local refinement; not a free-search inverse.',
    'reference': 'Original seed identities and patches never change; ended points never resume.',
    'scope': 'Adjacent appearance hypotheses, not verified physical points, source-edit continuity or rigid geometry.',
})


def _images(gray, mask):
    gray, mask = np.asarray(gray), np.asarray(mask)
    if (gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 16
            or gray.size > CONTRACT['maximum_image_pixels']):
        raise ValueError('Use a bounded source-resolution uint8 grayscale image')
    if (mask.shape != gray.shape or mask.dtype != np.uint8
            or not ((mask == 0) | (mask == 255)).all()):
        raise ValueError('Supply a same-size binary uint8 mask with 0/255 values')
    inset = cv2.erode(mask, np.ones((2*CONTRACT['mask_inset_px']+1,)*2, np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return gray, inset


def _inside(point, mask):
    if not np.isfinite(point).all():return False
    x, y = point
    return 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and bool(mask[int(y), int(x)])


def _patch(gray, point, size=CONTRACT['patch_size']):
    # Public coordinates use pixel-cell edges; OpenCV places integer coordinates at centers.
    return cv2.getRectSubPix(gray, (size,)*2,
        (float(point[0]-.5), float(point[1]-.5))).astype(np.float32)


def _correlation(first, second):
    a, b = first-first.mean(), second-second.mean()
    std = [float(np.sqrt(np.mean(p*p))) for p in (a, b)]
    if min(std) < CONTRACT['minimum_patch_std']:return None
    return float(np.clip(np.mean(a*b)/(std[0]*std[1]), -1., 1.))


class OpticalPointHistories:
    """Follow only declared seeds; callers preserve all per-frame outcomes in a journal."""

    def __init__(self, gray, mask, points_px, *, frame, time_s):
        _stamp(frame, time_s); gray, inset = _images(gray, mask)
        points = np.asarray(points_px, dtype=float)
        if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) > CONTRACT['maximum_points']
                or not np.isfinite(points).all()
                or (points < 0).any() or (points >= [gray.shape[1], gray.shape[0]]).any()):
            raise ValueError('Use 0–128 finite original in-image seed points')
        if len(points) > 1:
            distances = np.linalg.norm(points[:, None]-points[None, :], axis=2)
            if (distances+np.eye(len(points))*1e9 < CONTRACT['minimum_point_separation_px']).any():
                raise ValueError('Seed points must have distinct image locations')
        self._frame, self._time, self._count = frame, time_s, 1
        self._size = gray.shape; self._seeds = points.copy(); self._patches = []
        self._anchors = {}; self._warps = {}
        self._anchor_support = np.zeros((CONTRACT['anchor_patch_size'],)*2, np.uint8)
        border = CONTRACT['anchor_template_border_px']; self._anchor_support[border:-border, border:-border] = 255
        self._positions = {}; self._ages = {}; self._ended = {}; rows = []
        for index, point in enumerate(points):
            patch = _patch(gray, point) if _inside(point, inset) else None
            status = ('outside_mask_support' if patch is None else
                'weak_texture' if patch.std() < CONTRACT['minimum_patch_std'] else 'initialization')
            self._patches.append(patch)
            if status == 'initialization':
                self._positions[index] = point.copy(); self._ages[index] = 1
                self._anchors[index] = cv2.GaussianBlur(_patch(gray, point, CONTRACT['anchor_patch_size']),
                    (CONTRACT['anchor_gaussian_size'],)*2, 0)
                self._warps[index] = np.eye(2)
            else:self._ended[index] = {'frame': frame, 'reason': status}
            rows.append(self._row(index, status, point if status == 'initialization' else None))
        self._previous = gray.copy() if self._positions else None
        self._initial = self._result(rows)

    def _row(self, index, status, point=None, **audit):
        return {'seed_id': index, 'original_point_px': self._seeds[index].tolist(),
            'point_px': point.tolist() if point is not None else None, 'status': status,
            'age_frames': self._ages.get(index, 0), 'ended': deepcopy(self._ended.get(index)),
            'proposed_point_px': None, 'backward_point_px': None,
            'forward_valid': None, 'backward_valid': None, 'minimum_eigenvalue': None,
            'roundtrip_error_px': None, 'step_px': None, 'local_patch_ncc': None,
            'original_patch_ncc': None, 'anchor_ecc': None, 'anchor_warp': None,
            'anchor_correction_px': None, **audit}

    def _anchor(self, gray, index, proposal, inset):
        point = proposal['point']; audit = proposal['audit']; size = CONTRACT['anchor_patch_size']
        center = np.full(2, (size-1)/2)
        initial = np.column_stack([self._warps[index], center-self._warps[index] @ center]).astype(np.float32)
        target = cv2.GaussianBlur(_patch(gray, point, size), (CONTRACT['anchor_gaussian_size'],)*2, 0)
        try:
            reported, warp = cv2.findTransformECCWithMask(self._anchors[index], target,
                self._anchor_support, None, initial, cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-5), 1)
        except cv2.error:
            proposal['status'] = 'anchor_alignment_failed'; return
        if not np.isfinite(reported) or not np.isfinite(warp).all():
            proposal['status'] = 'anchor_alignment_failed'; return
        border = CONTRACT['anchor_template_border_px']
        corners = np.array([[border, border], [size-1-border, border],
            [size-1-border, size-1-border], [border, size-1-border]])
        mapped = corners @ warp[:, :2].T+warp[:, 2]
        # A partial overlap must not masquerade as validation of the full fixed template.
        if (mapped < 0).any() or (mapped > size-1).any():
            proposal['status'] = 'anchor_support_outside_patch'; return
        aligned = cv2.warpAffine(target, warp, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        score = _correlation(self._anchors[index][border:-border, border:-border], aligned[border:-border, border:-border])
        corrected = point+warp[:, :2] @ center+warp[:, 2]-center
        singular = np.linalg.svd(warp[:, :2], compute_uv=False)
        audit.update(anchor_ecc=score, anchor_reported_ecc=float(reported), anchor_warp=warp.tolist(),
            anchor_correction_px=float(np.linalg.norm(corrected-point)), unrefined_point_px=point.tolist())
        if score is None or score < CONTRACT['minimum_anchor_ecc']:
            proposal['status'] = 'anchor_appearance_changed'
        elif (np.linalg.det(warp[:, :2]) <= 0 or singular[-1] < CONTRACT['anchor_scale_bounds'][0]
                or singular[0] > CONTRACT['anchor_scale_bounds'][1]
                or singular[0]/singular[-1] > CONTRACT['maximum_anchor_axis_ratio']):
            proposal['status'] = 'anchor_deformation_exceeds_bound'
        elif audit['anchor_correction_px'] > CONTRACT['maximum_anchor_correction_px']:
            proposal['status'] = 'anchor_drift_exceeds_bound'
        elif not _inside(corrected, inset):
            proposal['status'] = 'outside_mask_support'
        else:
            proposal['point'] = corrected; proposal['linear'] = warp[:, :2].copy()
            audit['proposed_point_px'] = corrected.tolist()

    def _result(self, rows):
        return {'method': CONTRACT['method'], 'frame': self._frame, 'time_s': self._time,
            'rows': rows, 'active_points': len(self._positions), 'original_seed_count': len(self._seeds),
            'physical_point_identity_verified': False, 'rigidity_verified': False,
            'source_edit_continuity_verified': False}

    @property
    def initialization(self):
        return deepcopy(self._initial)

    def advance(self, gray, mask, *, frame, time_s):
        _stamp(frame, time_s); gray, inset = _images(gray, mask)
        if gray.shape != self._size:raise ValueError('Source image dimensions changed')
        if frame <= self._frame or time_s <= self._time:raise ValueError('Source frame and clock must increase')
        if self._count >= CONTRACT['maximum_frames']:raise ValueError('Point history frame budget exhausted')
        gap = frame != self._frame+1 or time_s-self._time > CONTRACT['maximum_time_step_s']+1e-12
        ids = list(self._positions); proposals = {}; rows = []
        if ids and not gap and inset.any():
            previous = np.array([self._positions[i]-.5 for i in ids], np.float32).reshape(-1, 1, 2)
            settings = dict(winSize=(CONTRACT['window_size'],)*2, maxLevel=CONTRACT['pyramid_levels'],
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, CONTRACT['iterations'], CONTRACT['epsilon']),
                flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS, minEigThreshold=CONTRACT['minimum_eigenvalue'])
            forward, valid, eigen = cv2.calcOpticalFlowPyrLK(self._previous, gray, previous, None, **settings)
            backward = np.full_like(previous, np.nan); back_valid = np.zeros(len(ids), bool)
            eligible = np.zeros(len(ids), bool)
            if forward is not None:
                eligible = (valid[:, 0] == 1) & np.isfinite(forward[:, 0]).all(axis=1)
                if eligible.any():
                    # The inverse tests a known location; coarse levels can jump to an unrelated patch.
                    backward_settings = settings | {'maxLevel': 0,
                        'flags': settings['flags'] | cv2.OPTFLOW_USE_INITIAL_FLOW}
                    back, checks, _ = cv2.calcOpticalFlowPyrLK(gray, self._previous, forward[eligible],
                        previous[eligible].copy(), **backward_settings)
                    if back is not None:
                        backward[eligible] = back; back_valid[eligible] = checks[:, 0] == 1
            for index, identifier in enumerate(ids):
                point = forward[index, 0].astype(float)+.5 if eligible[index] else None
                returned = backward[index, 0].astype(float)+.5
                audit = {'proposed_point_px': point.tolist() if point is not None else None,
                    'backward_point_px': returned.tolist() if back_valid[index] and np.isfinite(returned).all() else None,
                    'forward_valid': bool(eligible[index]), 'backward_valid': bool(back_valid[index]),
                    'minimum_eigenvalue': float(eigen[index, 0]) if eligible[index] else None}
                status = 'flow_pair'
                if point is None:status = 'forward_flow_failed'
                elif not _inside(point, inset):status = 'outside_mask_support'
                elif not back_valid[index] or not np.isfinite(returned).all():status = 'backward_flow_failed'
                else:
                    old = self._positions[identifier]
                    audit.update(roundtrip_error_px=float(np.linalg.norm(returned-old)), step_px=float(np.linalg.norm(point-old)))
                    local = _patch(gray, point)
                    audit['local_patch_ncc'] = _correlation(_patch(self._previous, old), local)
                    audit['original_patch_ncc'] = _correlation(self._patches[identifier], local)
                    if audit['roundtrip_error_px'] > CONTRACT['maximum_roundtrip_px']:status = 'roundtrip_failed'
                    elif audit['step_px'] > CONTRACT['maximum_step_px']:status = 'step_exceeds_bound'
                    elif audit['local_patch_ncc'] is None:status = 'weak_texture'
                    elif audit['local_patch_ncc'] < CONTRACT['minimum_patch_ncc']:status = 'local_appearance_changed'
                proposals[identifier] = {'point': point, 'status': status, 'audit': audit}
            # Original-patch alignment prevents small translation-only errors accumulating into drift.
            for identifier in ids:
                if proposals[identifier]['status'] == 'flow_pair':
                    self._anchor(gray, identifier, proposals[identifier], inset)
            refined = [i for i in ids if proposals[i]['status'] == 'flow_pair']
            if refined:
                current = np.array([proposals[i]['point']-.5 for i in refined], np.float32).reshape(-1, 1, 2)
                known = np.array([self._positions[i]-.5 for i in refined], np.float32).reshape(-1, 1, 2)
                back, checks, _ = cv2.calcOpticalFlowPyrLK(gray, self._previous, current, known.copy(),
                    **(settings | {'maxLevel': 0, 'flags': settings['flags'] | cv2.OPTFLOW_USE_INITIAL_FLOW}))
                for k, identifier in enumerate(refined):
                    proposal = proposals[identifier]; audit = proposal['audit']
                    audit['unrefined_roundtrip_error_px'] = audit['roundtrip_error_px']
                    valid_back = back is not None and checks[k, 0] == 1 and np.isfinite(back[k]).all()
                    audit['backward_valid'] = bool(valid_back)
                    audit['backward_point_px'] = (back[k, 0].astype(float)+.5).tolist() if valid_back else None
                    audit['roundtrip_error_px'] = float(np.linalg.norm(back[k, 0]+.5-self._positions[identifier])) if valid_back else None
                    audit['step_px'] = float(np.linalg.norm(proposal['point']-self._positions[identifier]))
                    audit['local_patch_ncc'] = _correlation(_patch(self._previous, self._positions[identifier]), _patch(gray, proposal['point']))
                    audit['original_patch_ncc'] = _correlation(self._patches[identifier], _patch(gray, proposal['point']))
                    if not valid_back:proposal['status'] = 'backward_flow_failed'
                    elif audit['roundtrip_error_px'] > CONTRACT['maximum_roundtrip_px']:proposal['status'] = 'roundtrip_failed'
                    elif audit['step_px'] > CONTRACT['maximum_step_px']:proposal['status'] = 'step_exceeds_bound'
                    elif audit['local_patch_ncc'] is None:proposal['status'] = 'weak_texture'
                    elif audit['local_patch_ncc'] < CONTRACT['minimum_patch_ncc']:proposal['status'] = 'local_appearance_changed'
            survivors = [i for i in ids if proposals[i]['status'] == 'flow_pair']
            if len(survivors) > 1:
                points = np.array([proposals[i]['point'] for i in survivors])
                distances = np.linalg.norm(points[:, None]-points[None, :], axis=2)+np.eye(len(points))*1e9
                # Two histories cannot silently collapse into the same observed image point.
                for i in np.where((distances < CONTRACT['minimum_point_separation_px']).any(axis=1))[0]:
                    proposals[survivors[i]]['status'] = 'competing_point_locations'
        active = {}
        for index in range(len(self._seeds)):
            if index in self._ended:
                rows.append(self._row(index, 'ended')); continue
            proposal = proposals.get(index, {'status': 'source_gap' if gap else 'mask_unavailable', 'point': None, 'audit': {}})
            if proposal['status'] == 'flow_pair':
                active[index] = proposal['point']; self._ages[index] += 1
                self._warps[index] = proposal['linear']
            else:self._ended[index] = {'frame': frame, 'reason': proposal['status']}
            rows.append(self._row(index, proposal['status'], active.get(index), **proposal['audit']))
        self._positions = active; self._previous = gray.copy() if active else None
        self._frame, self._time = frame, time_s; self._count += 1
        return self._result(rows)
