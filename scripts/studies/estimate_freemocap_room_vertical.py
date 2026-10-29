"""Estimate a scene vertical from fixed architectural regions, independently of feet."""

import argparse
import json
from pathlib import Path
import shutil
import time
import tomllib

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.parallel_lines import estimate_parallel_axis


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'artifacts/freemocap-full-evidence-01'
SOURCE = ROOT/'artifacts/freemocap-full-source-01'
REGIONS = [[0, 80, 180, 720], [390, 0, 720, 760]]


def read(path):
    return json.loads(path.read_text())


def verified(root, name):
    path = root/name
    if digest(path) != read(root/'manifest.json')['files'][name]:
        raise ValueError('Frozen source changed: '+name)
    return path


def axis_angle(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0., 1.))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    calibration_path = verified(SOURCE, 'member-04-freemocap_sample_data_camera_calibration.toml')
    calibration = tomllib.loads(calibration_path.read_text())
    paths = [verified(EVIDENCE, f'view-{i}-source-0550.png') for i in (1, 2)]
    out.mkdir(exist_ok=False)
    for path in (Path(__file__), ROOT/'src/weft/parallel_lines.py'):
        shutil.copyfile(path, out/path.name)
    write_json(out/'pre-run.json', {
        'created_at': local_time(), 'source_frame': 550, 'source_time_s': 550/30,
        'source_images': [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in paths],
        'calibration_sha256': digest(calibration_path), 'regions_in_undistorted_pixels_xyxy': REGIONS,
        'hypothesis': 'Long door-frame edges in both construction cameras constrain a shared architectural vertical without using body joints or floor-contact fitting.',
        'line_detection': 'Fixed source K/distortion rectification at original dimensions; OpenCV LSD_REFINE_STD; both endpoints inside region, length >=80 px, abs(dx) <=.6*abs(dy).',
        'deduplication': 'Longest first; same edge if slope differs <=.015 and extrapolated x at y=400 differs <=8 px. Retain every original segment and reason.',
        'axis_method': 'Enumerate plane-pair axes, .5-degree angular consensus, sqrt(line-length) weights; fit each view and both views separately.',
        'admission': {'minimum_distinct_inlier_edges_per_view': 2, 'minimum_combined_inliers': 4,
            'minimum_supported_weight_fraction': .7, 'maximum_cross_view_axis_degrees': 2., 'maximum_axis_from_camera0_image_up_degrees': 45.},
        'assumptions': ['Selected long architectural edges are intended to be vertical.', 'Camera 0 image-up identifies the upward axis sign.', 'Architectural vertical is not an accelerometer measurement or a floor-height observation.'],
        'maximum_lines_per_view': 64, 'maximum_seconds': 20, 'paid_calls': 0,
        'new_video_decodes': 0, 'new_pose_inference': 0, 'human_joints_used': False, 'reserved_camera_opened': False,
        'original_checks': 'Original monocular calibration and red-cover floor failures are not changed or waived.',
    })
    started = time.perf_counter(); view_results = []; all_planes = []; all_weights = []; annotated = []; raw_images = []
    for index, (path, region) in enumerate(zip(paths, REGIONS)):
        c = calibration[f'cam_{index}']; K = np.array(c['matrix']); R = cv2.Rodrigues(np.array(c['rotation']))[0]
        raw = cv2.imread(str(path)); raw_images.append(raw)
        corrected = cv2.undistort(raw, K, np.array(c['distortions']), None, K)
        detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY))[0]
        segments = [] if detected is None else detected.reshape(-1, 4)
        rows = []; candidates = []
        for identifier, segment in enumerate(segments):
            a, b = segment.reshape(2, 2).astype(float); delta = b-a; length = float(np.linalg.norm(delta))
            inside = all(region[0] <= p[0] <= region[2] and region[1] <= p[1] <= region[3] for p in (a, b))
            eligible = inside and length >= 80 and abs(delta[0]) <= .6*abs(delta[1])
            row = {'id': identifier, 'pixels': [a.tolist(), b.tolist()], 'length_px': length,
                'status': 'candidate' if eligible else 'outside_declared_line_envelope'}
            rows.append(row)
            if eligible:
                row['slope_dx_dy'] = float(delta[0]/delta[1])
                row['x_at_y400'] = float(a[0]+(400-a[1])*delta[0]/delta[1]); candidates.append(row)
        selected = []
        for row in sorted(candidates, key=lambda r: -r['length_px']):
            duplicate = next((other for other in selected if abs(row['slope_dx_dy']-other['slope_dx_dy']) <= .015
                and abs(row['x_at_y400']-other['x_at_y400']) <= 8), None)
            if duplicate is not None:
                row.update(status='duplicate_edge', represented_by=duplicate['id'])
            elif len(selected) >= 64:
                row['status'] = 'line_capacity_exceeded'
            else:
                row['status'] = 'distinct_candidate'; selected.append(row)
        planes, weights = [], []
        for row in selected:
            rays = np.column_stack([row['pixels'], np.ones(2)]) @ np.linalg.inv(K).T
            normal = np.cross(rays[0], rays[1]); normal /= np.linalg.norm(normal)
            world_normal = normal @ R; row['normal_in_camera0'] = world_normal.tolist()
            planes.append(world_normal); weights.append(float(np.sqrt(row['length_px'])))
        fit = None; error = None
        if len(selected) >= 2:
            try:
                fit = estimate_parallel_axis(planes, weights)
            except ValueError as failure:
                error = str(failure)
        else:
            error = 'Fewer than two distinct eligible architectural edges'
        write_json(out/f'view-{index+1}-all-line-segments.json', rows)
        write_json(out/f'view-{index+1}-axis-fit.json', {'fit': fit, 'error': error, 'selected_line_ids': [r['id'] for r in selected]})
        for order, row in enumerate(selected):
            color = (0, 220, 0) if fit and order in fit['inlier_indices'] else (0, 0, 255)
            a, b = [tuple(int(round(v)) for v in p) for p in row['pixels']]
            cv2.line(corrected, a, b, color, 2)
            cv2.putText(corrected, str(row['id']), a, cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        cv2.rectangle(corrected, tuple(region[:2]), tuple(region[2:]), (0, 220, 220), 1)
        cv2.imwrite(str(out/f'view-{index+1}-rectified-lines.png'), corrected); annotated.append(corrected)
        view_results.append({'eligible_segments': len(candidates), 'distinct_candidates': len(selected), 'fit': fit, 'error': error})
        all_planes.extend(planes); all_weights.extend(weights)
        if time.perf_counter()-started > 20:
            raise TimeoutError('Architectural-line extraction exceeded its work envelope')
    combined = None; combined_error = None
    if len(all_planes) >= 2:
        try:
            combined = estimate_parallel_axis(all_planes, all_weights)
        except ValueError as error:
            combined_error = str(error)
    angle = axis_angle(*[r['fit']['axis'] for r in view_results]) if all(r['fit'] for r in view_results) else None
    upward = None if combined is None else np.array(combined['axis'])
    if upward is not None and upward @ np.array([0., -1., 0.]) < 0:
        upward = -upward
    image_up_angle = None if upward is None else axis_angle(upward, [0., -1., 0.])
    checks = {'both_views_have_distinct_support': all(r['fit'] is not None and len(r['fit']['inlier_indices']) >= 2 for r in view_results),
        'both_views_have_majority_support': all(r['fit'] is not None and r['fit']['supported_weight_fraction'] >= .7 for r in view_results),
        'combined_support': combined is not None and len(combined['inlier_indices']) >= 4 and combined['supported_weight_fraction'] >= .7,
        'cross_view_direction_agreement': angle is not None and angle <= 2.,
        'upward_direction_envelope': image_up_angle is not None and image_up_angle <= 45.}
    if all(checks.values()):
        write_json(out/'scene-vertical.json', {'schema_version': 1, 'quality': 'reconstructed',
            'up_axis_in_source_camera0': upward.tolist(), 'source_frame': 550,
            'calibration_sha256': digest(calibration_path), 'cross_view_axis_difference_degrees': angle,
            'ground_height': None, 'gravity_magnitude': None, 'physical_validation': 'unverified',
            'meaning': 'Conditional architectural vertical; not measured gravity, ground height or body contact.',
            'original_monocular_and_floor_checks_preserved': True})
    sheet = Image.new('RGB', (1440, 710), 'white'); draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), 'Room vertical diagnostic: original source vs fixed-camera lens correction / detected edges', fill='black')
    draw.text((10, 29), 'Green: within-view axis consensus; red: rejected direction; yellow: declared architectural region. No person or floor fitting.', fill='black')
    for i, bgr in enumerate((raw_images[0], annotated[0], raw_images[1], annotated[1])):
        sheet.paste(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((360, 640)), (i*360, 65))
        draw.text((i*360+8, 49), f'Cam{i//2+1} '+('original' if i%2==0 else 'derived line inspection'), fill='black')
    sheet.save(out/'source-and-vertical-lines.png')
    report = {'created_at': local_time(), 'status': 'architectural_vertical_estimated' if all(checks.values()) else 'architectural_vertical_rejected',
        'checks': checks, 'views': view_results, 'combined': combined, 'combined_error': combined_error,
        'up_axis_in_camera0': None if upward is None else upward.tolist(),
        'cross_view_axis_difference_degrees': angle, 'axis_from_camera0_image_up_degrees': image_up_angle,
        'runtime_seconds': time.perf_counter()-started, 'new_video_decodes': 0, 'new_inference_calls': 0,
        'human_joints_used': False, 'floor_height_estimated': False, 'reserved_camera_opened': False, 'paid_calls': 0,
        'limitations': ['Architectural parallelism/plumb assumptions and supplied camera uncertainty are not independently measured.',
            'A common scene direction does not supply floor height or prove muscle, contact, anatomy or physical transfer.',
            'Original failed camera/floor checks and all rejected detected edges remain visible.']}
    write_json(out/'report.json', report)
    print(json.dumps({'status': report['status'], 'checks': checks, 'distinct_edges': [r['distinct_candidates'] for r in view_results],
        'cross_view_axis_difference_degrees': angle, 'up_axis_in_camera0': report['up_axis_in_camera0'],
        'runtime_seconds': report['runtime_seconds'], 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
