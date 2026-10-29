"""Fit a floor hypothesis to matched scene texture, with a frozen spatial test split."""

import argparse
import json
from pathlib import Path
import shutil
import time
import tomllib

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.camera_pose import _undistort
from weft.io import digest, local_time, write_json


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'artifacts/freemocap-full-source-01'
EVIDENCE = ROOT/'artifacts/freemocap-full-evidence-01'
POLYGONS = [[[0, 900], [100, 750], [310, 730], [680, 750], [720, 1279], [0, 1279]],
            [[0, 760], [650, 730], [719, 900], [719, 1279], [0, 1279]]]
EXCLUSIONS = [[[90, 630, 330, 940], [300, 740, 420, 830], [50, 1130, 340, 1280], [590, 500, 720, 1250]],
              [[70, 660, 370, 1000], [440, 930, 620, 1090], [400, 1090, 720, 1280]]]


def read(path):
    return json.loads(path.read_text())


def verified(root, name):
    path = root/name
    if digest(path) != read(root/'manifest.json')['files'][name]:
        raise ValueError('Frozen source changed: '+name)
    return path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    calibration_path = verified(SOURCE, 'member-04-freemocap_sample_data_camera_calibration.toml')
    supplied = tomllib.loads(calibration_path.read_text())
    paths = [verified(EVIDENCE, f'view-{i}-source-0550.png') for i in (1, 2)]
    out.mkdir(exist_ok=False); shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'source_frame': 550, 'source_time_s': 550/30,
        'source_images': [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in paths],
        'calibration_sha256': digest(calibration_path), 'carpet_polygons': POLYGONS, 'excluded_rectangles': EXCLUSIONS,
        'hypothesis': 'Wider carpet texture supplies independent scene correspondences and plane extent that the small red cover lacks.',
        'features': {'method': 'SIFT', 'maximum_features_per_view': 1200, 'contrast_threshold': .02, 'edge_threshold': 10,
            'mutual_nearest_ratio': .7, 'maximum_stereo_reprojection_px': 3., 'depth_bounds_m': [.2, 10.]},
        'split_before_geometry': 'Sort mutual descriptor matches by Cam1/Cam2 feature index. Every third match is reserved for development comparison; all others construct the plane.',
        'plane_search': {'seed': 17, 'maximum_hypotheses': 512, 'maximum_triplet_draws': 2048, 'inlier_distance_m': .015},
        'admission': {'minimum_training_inliers': 12, 'minimum_training_inlier_fraction': .8,
            'minimum_validation_matches': 6, 'all_validation_geometry_available': True,
            'maximum_validation_p95_distance_m': .02, 'maximum_validation_distance_m': .04,
            'minimum_training_hull_area_m2': .08, 'maximum_axis_from_image_up_degrees': 45.},
        'maximum_seconds': 20, 'new_video_decodes': 0, 'new_pose_inference': 0, 'human_joints_used': False,
        'reserved_camera_opened': False, 'paid_calls': 0,
        'scope': 'Previously exposed construction images with a new spatial fit/test split, not an untouched final holdout. No calibration, clock or old score changes.'})
    started = time.perf_counter(); cv2.setNumThreads(1)
    images = [cv2.imread(str(p)) for p in paths]; cameras = []; keypoints = []; descriptors = []
    detector = cv2.SIFT_create(nfeatures=1200, contrastThreshold=.02, edgeThreshold=10)
    for i, image in enumerate(images):
        mask = np.zeros(image.shape[:2], np.uint8); cv2.fillPoly(mask, [np.array(POLYGONS[i], np.int32)], 255)
        for x0, y0, x1, y1 in EXCLUSIONS[i]:
            mask[y0:y1, x0:x1] = 0
        cv2.imwrite(str(out/f'view-{i+1}-carpet-mask.png'), mask)
        points, descriptor = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)
        keypoints.append(points); descriptors.append(descriptor)
        write_json(out/f'view-{i+1}-features.json', [{'index': j, 'pixel': list(k.pt), 'size_px': k.size, 'angle_degrees': k.angle, 'response': k.response} for j, k in enumerate(points)])
        c = supplied[f'cam_{i}']; K = np.array(c['matrix']); r = np.array(c['rotation']); R = cv2.Rodrigues(r)[0]; t = np.array(c['translation'])*.001
        cameras.append({'K': K, 'r': r, 'R': R, 't': t, 'd': np.array(c['distortions'])})
    matches = []
    if all(d is not None and len(d) >= 2 for d in descriptors):
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        forward = {a.queryIdx: (a.trainIdx, float(a.distance), float(b.distance)) for a, b in matcher.knnMatch(*descriptors, k=2) if a.distance < .7*b.distance}
        backward = {a.queryIdx: a.trainIdx for a, b in matcher.knnMatch(descriptors[1], descriptors[0], k=2) if a.distance < .7*b.distance}
        pairs = sorted((left, data) for left, data in forward.items() if backward.get(data[0]) == left)
        for index, (left, (right, distance, runner)) in enumerate(pairs):
            matches.append({'index': index, 'feature_indices': [left, right], 'pixels': [list(keypoints[0][left].pt), list(keypoints[1][right].pt)],
                'descriptor_distance': distance, 'runner_up_distance': runner,
                'role': 'development_comparison' if index%3 == 0 else 'construction'})
    write_json(out/'matched-pixels-and-split-before-geometry.json', matches)
    observations = []
    for match in matches:
        rays = [_undistort(np.array([pixel]), c['K'], c['d'][0]) for pixel, c in zip(match['pixels'], cameras)]
        result = dict(match)
        if any(ray is None for ray in rays):
            result.update(status='distortion_unavailable', point_m=None); observations.append(result); continue
        h = cv2.triangulatePoints(*[np.column_stack([c['R'], c['t']]) for c in cameras], rays[0].T, rays[1].T)
        point = (h[:3]/h[3]).ravel(); errors = []; depths = []
        for c, pixel in zip(cameras, match['pixels']):
            prediction = cv2.projectPoints(point[None, :], c['r'], c['t'], c['K'], c['d'])[0].reshape(2)
            errors.append(float(np.linalg.norm(prediction-pixel))); depths.append(float((c['R'] @ point+c['t'])[2]))
        valid = bool(np.isfinite(point).all() and all(.2 <= d <= 10. for d in depths) and max(errors) <= 3.)
        result.update(status='estimated' if valid else 'inconsistent_stereo_match', point_m=point.tolist(), reprojection_px=errors, depths_m=depths)
        observations.append(result)
    write_json(out/'all-stereo-matches.json', observations)
    training = [r for r in observations if r['role'] == 'construction' and r['status'] == 'estimated']
    validation = [r for r in observations if r['role'] == 'development_comparison']
    hypotheses = []; plane = None; rng = np.random.default_rng(17)
    if len(training) >= 12:
        points = np.array([r['point_m'] for r in training]); choices = set()
        for _ in range(2048):
            indices = tuple(sorted(rng.choice(len(points), 3, replace=False).tolist()))
            if indices in choices:
                continue
            choices.add(indices); a, b, c = points[list(indices)]; normal = np.cross(b-a, c-a); length = np.linalg.norm(normal)
            if length <= 1e-8:
                continue
            normal /= length; offset = float(-normal @ a); residual = np.abs(points @ normal+offset); inliers = np.flatnonzero(residual <= .015)
            hypotheses.append({'training_indices': list(indices), 'normal': normal.tolist(), 'offset_m': offset,
                'inlier_indices': inliers.tolist(), 'inlier_rms_m': float(np.sqrt(np.mean(residual[inliers]**2)))})
            if len(hypotheses) >= 512:
                break
        if hypotheses:
            best = min(hypotheses, key=lambda r: (-len(r['inlier_indices']), r['inlier_rms_m']))
            selected = points[best['inlier_indices']]; center = selected.mean(axis=0); _, _, vectors = np.linalg.svd(selected-center)
            normal = vectors[-1]
            if normal[1] > 0:
                normal = -normal
            offset = float(-normal @ center); final_inliers = np.flatnonzero(np.abs(points @ normal+offset) <= .015)
            coordinates = ((points[final_inliers]-center) @ vectors[:2].T).astype(np.float32)
            area = float(cv2.contourArea(cv2.convexHull(coordinates)))
            plane = {'normal_in_camera0': normal.tolist(), 'offset_m': offset, 'training_match_indices': [r['index'] for r in training],
                'training_inlier_indices': final_inliers.tolist(), 'training_inlier_hull_area_m2': area,
                'origin_m': center.tolist(), 'axis_from_image_up_degrees': float(np.degrees(np.arccos(np.clip(-normal[1], -1., 1.))))}
    write_json(out/'all-plane-hypotheses.json', hypotheses)
    write_json(out/'plane-before-comparison.json', {'plane': plane, 'training_matches': len(training), 'comparison_matches': len(validation)})
    comparison = []
    for row in validation:
        distance = None if plane is None or row['status'] != 'estimated' else float(np.dot(row['point_m'], plane['normal_in_camera0'])+plane['offset_m'])
        comparison.append({'match_index': row['index'], 'status': row['status'], 'signed_plane_distance_m': distance})
    distances = [abs(r['signed_plane_distance_m']) for r in comparison if r['signed_plane_distance_m'] is not None]
    training_total = sum(r['role'] == 'construction' for r in observations)
    checks = {'enough_training_inliers': plane is not None and len(plane['training_inlier_indices']) >= 12,
        'training_inlier_fraction': plane is not None and len(plane['training_inlier_indices']) >= .8*training_total,
        'enough_comparison_matches': len(validation) >= 6, 'complete_comparison_geometry': bool(validation) and len(distances) == len(validation),
        'comparison_p95': bool(distances) and float(np.percentile(distances, 95)) <= .02,
        'comparison_maximum': bool(distances) and max(distances) <= .04,
        'spatial_extent': plane is not None and plane['training_inlier_hull_area_m2'] >= .08,
        'upward_axis': plane is not None and plane['axis_from_image_up_degrees'] <= 45.}
    write_json(out/'development-comparison.json', {'checks': checks, 'matches': comparison})
    if all(checks.values()):
        write_json(out/'floor-plane.json', {'schema_version': 1, 'quality': 'reconstructed', **plane,
            'calibration_sha256': digest(calibration_path), 'source_frame': 550, 'physical_validation': 'unverified',
            'meaning': 'Visible carpet plane conditional on descriptor correspondences, camera geometry and static-planar scene assumptions.'})
    inspect = np.hstack(images)
    for r in observations:
        color = (0, 220, 0) if r['status'] == 'estimated' else (0, 0, 255)
        a, b = [tuple(round(v) for v in p) for p in r['pixels']]; b = (b[0]+720, b[1])
        cv2.line(inspect, a, b, color, 1); cv2.circle(inspect, a, 3, color, 1); cv2.circle(inspect, b, 3, color, 1)
    cv2.imwrite(str(out/'source-carpet-matches.png'), inspect)
    if time.perf_counter()-started > 20:
        raise TimeoutError('Floor-feature comparison exceeded its work envelope')
    report = {'created_at': local_time(), 'status': 'carpet_plane_estimated' if all(checks.values()) else 'carpet_plane_rejected',
        'feature_counts': [len(k) for k in keypoints], 'mutual_descriptor_matches': len(matches),
        'estimated_stereo_matches': sum(r['status'] == 'estimated' for r in observations), 'training_matches': len(training),
        'comparison_matches': len(validation), 'plane': plane, 'checks': checks,
        'comparison_p95_distance_m': float(np.percentile(distances, 95)) if distances else None,
        'runtime_seconds': time.perf_counter()-started, 'new_video_decodes': 0, 'human_joints_used': False,
        'reserved_camera_opened': False, 'paid_calls': 0,
        'limitations': ['Repeated carpet texture can produce wrong matches; the saved split and geometric tests do not make provider calibration measured truth.',
            'This is development scene evidence, not anatomy, real muscle forces, body-contact validation or changed-action physical transfer.',
            'All earlier failed floor/vertical checks stay unchanged; no body/default is promoted.']}
    write_json(out/'report.json', report)
    print(json.dumps({'status': report['status'], 'feature_counts': report['feature_counts'], 'matches': len(matches),
        'estimated_stereo_matches': report['estimated_stereo_matches'], 'checks': checks, 'plane': plane,
        'runtime_seconds': report['runtime_seconds'], 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
