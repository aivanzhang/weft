"""Test perspective-normalized carpet matching; the rejected plane is only a search prior."""

import argparse
import json
from pathlib import Path
import shutil
import time
import tomllib

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.regional_features import regional_sift


ROOT = Path(__file__).resolve().parents[2]
MASKS = ROOT/'artifacts/freemocap-carpet-plane-01'
PRIOR = ROOT/'artifacts/freemocap-floor-proxy-02'
SOURCE = ROOT/'artifacts/freemocap-full-source-01'
EVIDENCE = ROOT/'artifacts/freemocap-full-evidence-01'


def read(path):
    return json.loads(path.read_text())


def verified(root, name):
    path = root/name
    if digest(path) != read(root/'manifest.json')['files'][name]:
        raise ValueError('Frozen source changed: '+name)
    return path


def distort(pixel, camera):
    xy = (np.linalg.inv(camera['K']) @ np.r_[pixel, 1.])[:2]
    xyz = np.r_[xy, 1.]
    return cv2.projectPoints(xyz[None, :], np.zeros(3), np.zeros(3), camera['K'], camera['d'])[0].reshape(2)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--regional', action='store_true')
    args = parser.parse_args(); out = args.out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    calibration_path = verified(SOURCE, 'member-04-freemocap_sample_data_camera_calibration.toml')
    calibration = tomllib.loads(calibration_path.read_text())
    prior = read(verified(PRIOR, 'report.json'))
    if prior['status'] != 'floor_cover_proxy_rejected':
        raise ValueError('The declared rejected search prior changed')
    g = prior['selected_candidate']['geometry']; n = np.array(g['up_normal']); origin = np.array(g['origin_m']); d = float(-n @ origin)
    cameras = []
    for i in (0, 1):
        c = calibration[f'cam_{i}']; cameras.append({'K': np.array(c['matrix']), 'd': np.array(c['distortions']),
            'R': cv2.Rodrigues(np.array(c['rotation']))[0], 't': np.array(c['translation'])*.001})
    H = cameras[1]['K'] @ (cameras[1]['R']-np.outer(cameras[1]['t'], n)/d) @ np.linalg.inv(cameras[0]['K'])
    if not np.isfinite(H).all() or abs(d) < .1 or np.linalg.cond(H) > 1e8:
        raise ValueError('The appearance-normalization homography is not bounded')
    # An analytic point on the prior plane checks transform direction, not real calibration.
    point = origin+np.cross(n, [1., 0., 0.])*.1
    left = cameras[0]['K'] @ point; left /= left[2]
    right = cameras[1]['K'] @ (cameras[1]['R'] @ point+cameras[1]['t']); right /= right[2]
    projected = H @ left; projected /= projected[2]
    convention_error = float(np.linalg.norm(projected-right))
    if convention_error > 1e-9:
        raise ValueError('Plane homography direction/convention failed')
    out.mkdir(exist_ok=False); shutil.copyfile(__file__, out/Path(__file__).name)
    if args.regional:
        shutil.copyfile(ROOT/'src/weft/regional_features.py', out/'regional_features.py')
    write_json(out/'pre-run.json', {'created_at': local_time(), 'source_frame': 550,
        'input_manifests': {p.name: digest(p/'manifest.json') for p in (MASKS, PRIOR, SOURCE, EVIDENCE)},
        'hypothesis': 'Wide perspective difference can defeat raw SIFT matching; a coarse plane hypothesis can normalize appearance while match pixels remain traceable to the original images.',
        'prior': {'status': 'rejected_for_floor_placement', 'normal': n.tolist(), 'offset_m': d,
            'use': 'Appearance search only; never substitute prior-plane depths for independently triangulated match geometry.'},
        'cam1_to_cam2_undistorted_homography': H.tolist(), 'analytic_convention_error_px': convention_error,
        'detector': {'maximum_features': 1200, 'contrast_threshold': .02, 'edge_threshold': 10, 'mutual_nearest_ratio': .7,
            'regional_quota_after_mask': args.regional, 'maximum_regional_crop_pixels': 262144 if args.regional else None},
        'opencv_version': cv2.__version__,
        'regional_hypothesis': 'A global feature cap precedes OpenCV mask filtering. Crop original pixels, bound crop area, then select at most 1200 regional features before descriptor computation.' if args.regional else None,
        'masks': 'Original frozen carpet masks, mapped to common perspective and intersected; no wider scene regions.',
        'split': 'Sort mutual feature pairs; every third match is development comparison, all others construction, before geometry grading.',
        'maximum_seconds': 20, 'new_sift_calls': 2, 'new_video_decodes': 0, 'new_body_inference': 0,
        'reserved_camera_opened': False, 'paid_calls': 0,
        'scope': 'Prepared correspondences only; no plane fit, floor admission or changes to failed checks.'})
    started = time.perf_counter(); cv2.setNumThreads(1)
    images, masks = [], []
    for i, c in enumerate(cameras, 1):
        image = cv2.imread(str(verified(EVIDENCE, f'view-{i}-source-0550.png')))
        mask = cv2.imread(str(verified(MASKS, f'view-{i}-carpet-mask.png')), cv2.IMREAD_GRAYSCALE)
        images.append(cv2.undistort(image, c['K'], c['d'], None, c['K']))
        masks.append(cv2.undistort(mask, c['K'], c['d'], None, c['K']))
    images[1] = cv2.warpPerspective(images[1], np.linalg.inv(H), (720, 1280))
    masks[1] = cv2.warpPerspective(masks[1], np.linalg.inv(H), (720, 1280))
    common = ((masks[0] >= 250) & (masks[1] >= 250)).astype(np.uint8)*255
    cv2.imwrite(str(out/'common-floor-mask.png'), common)
    detector = cv2.SIFT_create(nfeatures=1200, contrastThreshold=.02, edgeThreshold=10)
    keypoints, descriptors, audits = [], [], []
    for i, image in enumerate(images, 1):
        if args.regional:
            points, desc, audit = regional_sift(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), common)
            audits.append(audit)
        else:
            points, desc = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), common)
        keypoints.append(points); descriptors.append(desc)
        write_json(out/f'view-{i}-normalized-features.json', [{'index': j, 'pixel': list(p.pt), 'size_px': p.size, 'angle_degrees': p.angle} for j, p in enumerate(points)])
        visible = image.copy(); visible[common == 0] = 0
        cv2.imwrite(str(out/f'view-{i}-normalized-carpet.png'), visible)
        if args.regional:
            cv2.imwrite(str(out/f'view-{i}-normalized-source.png'), image)
    if args.regional:
        write_json(out/'regional-feature-audits.json', audits)
    matches = []
    if all(d is not None and len(d) >= 2 for d in descriptors):
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        forward = {a.queryIdx: (a.trainIdx, float(a.distance), float(b.distance)) for a, b in matcher.knnMatch(*descriptors, k=2) if a.distance < .7*b.distance}
        backward = {a.queryIdx: a.trainIdx for a, b in matcher.knnMatch(descriptors[1], descriptors[0], k=2) if a.distance < .7*b.distance}
        pairs = sorted((i, values) for i, values in forward.items() if backward.get(values[0]) == i)
        for index, (left_id, (right_id, distance, runner)) in enumerate(pairs):
            left = np.array(keypoints[0][left_id].pt); warped_right = np.array(keypoints[1][right_id].pt)
            right = H @ np.r_[warped_right, 1.]; right = right[:2]/right[2]
            pixels = [distort(left, cameras[0]).tolist(), distort(right, cameras[1]).tolist()]
            matches.append({'index': index, 'feature_indices': [left_id, right_id], 'pixels': pixels,
                'normalized_pixels': [left.tolist(), warped_right.tolist()], 'descriptor_distance': distance,
                'runner_up_distance': runner, 'role': 'development_comparison' if index%3 == 0 else 'construction',
                'pixel_derivation': 'Feature center in normalized/interpolated source imagery, mapped back through fixed perspective and distortion transforms; not an annotated physical point.'})
    write_json(out/'matched-pixels-and-split-before-geometry.json', matches)
    canvas = np.hstack(images)
    canvas[np.hstack([common, common]) == 0] = 0
    for row in matches:
        a, b = [tuple(round(v) for v in p) for p in row['normalized_pixels']]; b = (b[0]+720, b[1])
        cv2.line(canvas, a, b, (0, 220, 0), 1)
    cv2.putText(canvas, 'Plane-guided appearance search ONLY; no recovered floor or body', (15, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    cv2.imwrite(str(out/'normalized-carpet-matches.png'), canvas)
    if time.perf_counter()-started > 20:
        raise TimeoutError('Normalized matching exceeded its work envelope')
    report = {'created_at': local_time(), 'status': 'normalized_carpet_matches_prepared',
        'feature_counts': [len(p) for p in keypoints], 'common_mask_pixels': int(np.count_nonzero(common)),
        'mutual_matches': len(matches), 'training_matches': sum(r['role'] == 'construction' for r in matches),
        'regional_features': args.regional, 'regional_detection_audits': audits,
        'comparison_matches': sum(r['role'] == 'development_comparison' for r in matches),
        'runtime_seconds': time.perf_counter()-started, 'analytic_homography_error_px': convention_error,
        'original_floor_prior_still_rejected': True, 'new_plane_fit': False, 'new_floor_admission': False,
        'new_video_decodes': 0, 'new_human_inference': 0, 'reserved_camera_opened': False, 'paid_calls': 0,
        'limitations': ['The rejected plane guides feature search, so geometric agreement with it is not independent proof of that plane.',
            'Correspondences must be checked in original image coordinates and against the unchanged spatial comparison protocol.',
            'No physical floor, anatomy, muscle, contact or changed-action claim follows from feature counts.']}
    write_json(out/'report.json', report)
    print(json.dumps({'feature_counts': report['feature_counts'], 'common_mask_pixels': report['common_mask_pixels'],
        'matches': len(matches), 'runtime_seconds': report['runtime_seconds'], 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
