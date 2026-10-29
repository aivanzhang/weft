"""Estimate an explicit floor-cover proxy from a visible rectangle, never body feet."""

import argparse
import itertools
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
EVIDENCE = ROOT/'artifacts/freemocap-full-evidence-01'
SOURCE = ROOT/'artifacts/freemocap-full-source-01'
REGIONS = [[290, 740, 435, 840], [430, 930, 620, 1090]]


def read(path):
    return json.loads(path.read_text())


def verified(root, name):
    path = root/name
    if digest(path) != read(root/'manifest.json')['files'][name]:
        raise ValueError('Frozen source changed: '+name)
    return path


def geometry(pixels, cameras):
    rays = [_undistort(p, c['K'], c['k1']) for p, c in zip(pixels, cameras)]
    if any(r is None for r in rays):
        return {'status': 'distortion_unavailable'}
    matrices = [np.column_stack([c['R'], c['t']]) for c in cameras]
    homogeneous = cv2.triangulatePoints(*matrices, rays[0].T, rays[1].T)
    points = (homogeneous[:3]/homogeneous[3]).T
    if not np.isfinite(points).all():
        return {'status': 'nonfinite_intersection'}
    projections, depths = [], []
    for camera, observed in zip(cameras, pixels):
        projection = cv2.projectPoints(points, camera['r'], camera['t'], camera['K'], camera['d'])[0].reshape(-1, 2)
        projections.append(np.linalg.norm(projection-observed, axis=1).tolist())
        depths.extend((points @ camera['R'].T+camera['t'])[:, 2].tolist())
    center = points.mean(axis=0)
    _, singular, vectors = np.linalg.svd(points-center)
    normal = vectors[-1]
    if normal @ -center < 0:
        normal = -normal
    edges = np.roll(points, -1, axis=0)-points
    lengths = np.linalg.norm(edges, axis=1)
    directions = edges/lengths[:, None]
    orthogonal = np.abs(np.sum(directions*np.roll(directions, -1, axis=0), axis=1))
    opposite = [float(abs(lengths[i]-lengths[i+2])/max(lengths[i], lengths[i+2])) for i in (0, 1)]
    return {'status': 'estimated', 'points_m': points.tolist(), 'origin_m': center.tolist(), 'up_normal': normal.tolist(),
        'plane_rms_m': float(singular[-1]/2), 'plane_span_singular_values_m': singular.tolist(),
        'edge_lengths_m': lengths.tolist(), 'opposite_edge_relative_difference': opposite,
        'adjacent_edge_absolute_cosines': orthogonal.tolist(), 'depths_m': depths,
        'reprojection_errors_px': projections, 'reprojection_rmse_px': float(np.sqrt(np.mean(np.square(projections)))),
        'maximum_reprojection_px': float(np.max(projections))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    calibration_path = verified(SOURCE, 'member-04-freemocap_sample_data_camera_calibration.toml')
    calibration = tomllib.loads(calibration_path.read_text())
    images = [verified(EVIDENCE, f'view-{i}-source-0550.png') for i in (1, 2)]
    cameras = []
    for i in (0, 1):
        c = calibration[f'cam_{i}']
        cameras.append({'K': np.array(c['matrix']), 'k1': c['distortions'][0], 'd': np.array(c['distortions']),
            'r': np.array(c['rotation']), 'R': cv2.Rodrigues(np.array(c['rotation']))[0], 't': np.array(c['translation'])*.001})
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {
        'created_at': local_time(), 'frame': 550, 'time_s': 550/30, 'source_regions_xyxy': REGIONS,
        'source_images': [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in images],
        'calibration_sha256': digest(calibration_path),
        'hypothesis': 'A red rectangular cover visibly resting on the carpet constrains a ground-orientation proxy independently of inferred body joints.',
        'detector': 'In each fixed source region: HSV hue <=15 or >=170, saturation >=80, value >=60; largest external contour; approxPolyDP at .02 perimeter; exactly four vertices required.',
        'correspondences': 'Retain all eight cyclic/reversed corner correspondences; rank by stereo reprojection only.',
        'admission': {'maximum_reprojection_px': 3., 'minimum_runner_up_rmse_gap_px': 5., 'maximum_plane_rms_m': .005,
            'maximum_opposite_edge_fraction': .05, 'maximum_adjacent_edge_abs_cosine': .10, 'depth_bounds_m': [.2, 10.]},
        'sensitivity': {'pixel_coordinate_bound_px': 2., 'seed': 17, 'joint_uniform_trials': 256,
            'interpretation': 'Conditional pixel-perturbation sample, not a confidence interval, exhaustive bound, or calibration-uncertainty model.'},
        'maximum_seconds': 15, 'paid_calls': 0, 'reserved_camera_opened': False, 'new_decodes': 0,
        'ground_boundary': 'Top cover plane, not the underlying carpet. Cover thickness and true gravity remain unmeasured; no floor offset is fitted to feet.',
        'original_camera_check': 'Preserved failed monocular check; omitted-corner stereo development consistency passed separately.',
    })
    started = time.perf_counter(); observations = []
    for index, (path, region) in enumerate(zip(images, REGIONS), 1):
        bgr = cv2.imread(str(path)); x0, y0, x1, y1 = region
        crop = bgr[y0:y1, x0:x1]; hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = (((hsv[:, :, 0] <= 15) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 60)).astype(np.uint8)*255
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not contours:
            raise ValueError('No red cover contour in the frozen source region')
        contour = max(contours, key=cv2.contourArea)
        polygon = cv2.approxPolyDP(contour, .02*cv2.arcLength(contour, True), True).reshape(-1, 2)
        write_json(out/f'view-{index}-detected-contour.json', {'region': region, 'contour_area_px': cv2.contourArea(contour), 'vertices': polygon.tolist()})
        cv2.imwrite(str(out/f'view-{index}-mask.png'), mask)
        if len(polygon) != 4 or not 500 <= cv2.contourArea(contour) <= 25000:
            raise ValueError('Frozen cover detector did not find one bounded quadrilateral')
        points = polygon.astype(float)+[x0, y0]; observations.append(points)
        for label, (x, y) in enumerate(points):
            cv2.circle(bgr, (int(x), int(y)), 4, (0, 255, 255), 2)
            cv2.putText(bgr, str(label), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
        cv2.imwrite(str(out/f'view-{index}-corner-overlay.png'), bgr)
    write_json(out/'pixel-observations-before-geometry.json', {'frame': 550, 'views': [p.tolist() for p in observations]})
    alternatives = []
    for reverse, shift in itertools.product((False, True), range(4)):
        permutation = np.roll(np.arange(4)[::-1] if reverse else np.arange(4), shift)
        result = geometry([observations[0], observations[1][permutation]], cameras)
        alternatives.append({'permutation': permutation.tolist(), 'geometry': result})
    ranked = sorted([r for r in alternatives if r['geometry']['status'] == 'estimated'], key=lambda r: r['geometry']['reprojection_rmse_px'])
    if len(ranked) != 8:
        raise ValueError('Correspondence geometry unavailable; preserve alternatives before further use')
    best = ranked[0]; g = best['geometry']; runner_gap = ranked[1]['geometry']['reprojection_rmse_px']-g['reprojection_rmse_px']
    checks = {'stereo_pixel_consistency': g['maximum_reprojection_px'] <= 3., 'correspondence_separation': runner_gap >= 5.,
        'plane_consistency': g['plane_rms_m'] <= .005,
        'rectangle_opposite_edges': max(g['opposite_edge_relative_difference']) <= .05,
        'rectangle_angles': max(g['adjacent_edge_absolute_cosines']) <= .10,
        'forward_bounded_depth': all(.2 <= d <= 10. for d in g['depths_m'])}
    write_json(out/'all-correspondence-alternatives.json', alternatives)
    perturbations = []
    if all(checks.values()):
        rng = np.random.default_rng(17)
        original = [observations[0], observations[1][best['permutation']]]
        for index in range(256):
            candidate = geometry([p+rng.uniform(-2., 2., p.shape) for p in original], cameras)
            if candidate['status'] != 'estimated':
                perturbations.append({'trial': index, 'status': candidate['status']})
            else:
                angle = float(np.degrees(np.arccos(np.clip(np.dot(candidate['up_normal'], g['up_normal']), -1., 1.))))
                offset = float(np.dot(np.subtract(g['origin_m'], candidate['origin_m']), candidate['up_normal']))
                perturbations.append({'trial': index, 'status': 'estimated', 'normal_difference_degrees': angle,
                    'nominal_origin_signed_plane_distance_m': offset, 'geometry': candidate})
            if time.perf_counter()-started > 15:
                raise TimeoutError('Floor-proxy sensitivity exceeds its work envelope')
        normal = np.array(g['up_normal']); right = np.array([1., 0., 0.])-normal[0]*normal; right /= np.linalg.norm(right)
        axes = np.column_stack([right, np.cross(normal, right), normal])
        write_json(out/'floor-proxy.json', {'schema_version': 1, 'quality': 'reconstructed',
            'origin_in_source_camera0_m': g['origin_m'], 'z_up_axes_in_source_camera0': axes.tolist(),
            'plane_height_in_new_world_m': 0., 'meaning': 'Visible red cover top surface, not independently measured carpet height or gravity.',
            'source_frame': 550, 'observed_pixels_sha256': digest(out/'pixel-observations-before-geometry.json'),
            'calibration_sha256': digest(calibration_path), 'selected_correspondence': best['permutation'],
            'assumptions': ['The cover rests flat on the carpet.', 'Camera 0 is above the cover plane.', 'A future z-down gravity direction would be an authored floor-normal assumption, not measured acceleration.'],
            'original_monocular_camera_check_still_failed': True, 'physical_validation': 'unverified'})
    write_json(out/'all-pixel-perturbations.json', perturbations)
    usable = [r for r in perturbations if r['status'] == 'estimated']
    sensitivity = None if not usable else {
        'estimated_trials': len(usable), 'all_trials': len(perturbations),
        'normal_difference_p95_degrees': float(np.percentile([r['normal_difference_degrees'] for r in usable], 95)),
        'normal_difference_max_degrees': max(r['normal_difference_degrees'] for r in usable),
        'nominal_origin_plane_distance_max_abs_m': max(abs(r['nominal_origin_signed_plane_distance_m']) for r in usable)}
    report = {'created_at': local_time(), 'status': 'floor_cover_proxy_estimated' if all(checks.values()) else 'floor_cover_proxy_rejected',
        'checks': checks, 'selected_candidate': best, 'runner_up_rmse_gap_px': runner_gap, 'sensitivity': sensitivity,
        'runtime_seconds': time.perf_counter()-started, 'source_frame': 550, 'reserved_camera_opened': False,
        'new_video_frames_decoded': 0, 'human_poses_used': False, 'paid_calls': 0,
        'limitations': ['A plane from one small visible rectangle is conditional on corner/camera accuracy and flat-cover assumptions.',
            'Pixel perturbations exclude camera uncertainty and unmeasured cover thickness.', 'No body, floor-contact or physical changed-action accuracy is established.']}
    write_json(out/'report.json', report)
    print(json.dumps({'status': report['status'], 'checks': checks, 'best': g, 'sensitivity': sensitivity, 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
