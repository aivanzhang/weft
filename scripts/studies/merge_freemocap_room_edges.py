"""Compare grouped architectural edges using saved detections and unchanged axis gates."""

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
from weft.line_fragments import merge_vertical_fragments
from weft.parallel_lines import estimate_parallel_axis
from estimate_freemocap_room_vertical import axis_angle


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'artifacts/freemocap-room-vertical-01'
EVIDENCE = ROOT/'artifacts/freemocap-full-evidence-01'
SOURCE = ROOT/'artifacts/freemocap-full-source-01'


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
    prior = read(verified(PARENT, 'pre-run.json')); limits = prior['admission']
    calibration_path = verified(SOURCE, 'member-04-freemocap_sample_data_camera_calibration.toml')
    calibration = tomllib.loads(calibration_path.read_text())
    inputs = [read(verified(PARENT, f'view-{i}-all-line-segments.json')) for i in (1, 2)]
    out.mkdir(exist_ok=False)
    for path in (Path(__file__), ROOT/'src/weft/line_fragments.py', ROOT/'src/weft/parallel_lines.py'):
        shutil.copyfile(path, out/path.name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'parent_manifest_sha256': digest(PARENT/'manifest.json'),
        'hypothesis': 'One door edge can be split by a handle or line-detector fragmentation; count its bounded, straight merged trace once.',
        'all_original_eligible_segments': 'Retain both original distinct and duplicate edge candidates; no new detection or wider source region.',
        'grouping': {'maximum_gap_px': 80., 'maximum_separation_px': 3., 'maximum_angle_degrees': 3., 'maximum_fit_residual_px': 2.},
        'unchanged_admission': limits, 'unchanged_axis_tolerance_degrees': .5,
        'nonstraight_groups': 'Unavailable for axis fitting, but their sqrt(length) remains in the support denominator; they cannot be dropped to pass.',
        'maximum_seconds': 20, 'new_source_decodes': 0, 'new_line_detection_calls': 0,
        'new_plot_rectifications': 2, 'human_joints_used': False, 'reserved_camera_opened': False, 'paid_calls': 0})
    started = time.perf_counter(); views = []; all_planes = []; all_weights = []; total_weight = 0.; panels = []
    for i, rows in enumerate(inputs):
        eligible = [r for r in rows if r['status'] in ('distinct_candidate', 'duplicate_edge')]
        grouped = merge_vertical_fragments([np.array(r['pixels']).ravel() for r in eligible])
        c = calibration[f'cam_{i}']; K = np.array(c['matrix']); R = cv2.Rodrigues(np.array(c['rotation']))[0]
        planes, weights, included = [], [], []
        denominator = sum(np.sqrt(g['length_px']) for g in grouped['groups'])
        for index, group in enumerate(grouped['groups']):
            group['source_line_ids'] = [eligible[j]['id'] for j in group['input_indices']]
            if group['status'] != 'straight_group':
                continue
            rays = np.column_stack([group['pixels'], np.ones(2)]) @ np.linalg.inv(K).T
            n = np.cross(rays[0], rays[1]); n /= np.linalg.norm(n)
            normal = n @ R; group['normal_in_camera0'] = normal.tolist()
            included.append(index); planes.append(normal); weights.append(float(np.sqrt(group['length_px'])))
        fit = None; error = None
        if len(planes) >= 2:
            try:
                fit = estimate_parallel_axis(planes, weights)
            except ValueError as failure:
                error = str(failure)
        else:
            error = 'Fewer than two straight edge groups remain'
        fraction = 0. if fit is None else float(sum(weights[j] for j in fit['inlier_indices'])/denominator)
        views.append({'input_segments': len(eligible), 'edge_groups': grouped, 'fit': fit, 'error': error,
            'fit_input_group_indices': included, 'supported_weight_fraction_including_rejected_groups': fraction})
        write_json(out/f'view-{i+1}-grouped-edges.json', views[-1])
        raw = cv2.imread(str(verified(EVIDENCE, f'view-{i+1}-source-0550.png')))
        image = cv2.undistort(raw, K, np.array(c['distortions']), None, K)
        accepted = [] if fit is None else [included[j] for j in fit['inlier_indices']]
        for index, group in enumerate(grouped['groups']):
            color = (0, 220, 0) if index in accepted else (0, 0, 255)
            a, b = [tuple(round(v) for v in point) for point in group['pixels']]
            cv2.line(image, a, b, color, 2); cv2.putText(image, str(index), a, cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        panels.append(image)
        all_planes.extend(planes); all_weights.extend(weights); total_weight += denominator
    combined = None; combined_error = None
    try:
        combined = estimate_parallel_axis(all_planes, all_weights)
    except ValueError as failure:
        combined_error = str(failure)
    fraction = 0. if combined is None else float(sum(all_weights[j] for j in combined['inlier_indices'])/total_weight)
    angle = axis_angle(*[v['fit']['axis'] for v in views]) if all(v['fit'] for v in views) else None
    upward = None if combined is None else np.array(combined['axis'])
    if upward is not None and upward[1] > 0:
        upward = -upward
    up_angle = None if upward is None else axis_angle(upward, [0., -1., 0.])
    checks = {'both_views_have_distinct_support': all(v['fit'] and len(v['fit']['inlier_indices']) >= limits['minimum_distinct_inlier_edges_per_view'] for v in views),
        'both_views_have_majority_support': all(v['supported_weight_fraction_including_rejected_groups'] >= limits['minimum_supported_weight_fraction'] for v in views),
        'combined_support': combined is not None and len(combined['inlier_indices']) >= limits['minimum_combined_inliers'] and fraction >= limits['minimum_supported_weight_fraction'],
        'cross_view_direction_agreement': angle is not None and angle <= limits['maximum_cross_view_axis_degrees'],
        'upward_direction_envelope': up_angle is not None and up_angle <= limits['maximum_axis_from_camera0_image_up_degrees']}
    if all(checks.values()):
        write_json(out/'scene-vertical.json', {'schema_version': 1, 'quality': 'reconstructed', 'up_axis_in_source_camera0': upward.tolist(),
            'source_frame': 550, 'calibration_sha256': digest(calibration_path), 'cross_view_axis_difference_degrees': angle,
            'ground_height': None, 'gravity_magnitude': None, 'physical_validation': 'unverified',
            'meaning': 'Conditional architectural vertical from grouped edges; not measured gravity or floor/body contact.'})
    sheet = Image.new('RGB', (720, 710), 'white'); draw = ImageDraw.Draw(sheet)
    draw.text((10, 12), 'Grouped room edges / source frame 550 / derived lens-corrected imagery', fill='black')
    draw.text((10, 32), 'Green: within-view consensus; red: rejected direction or non-straight group.', fill='black')
    for i, image in enumerate(panels):
        sheet.paste(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((360, 640)), (360*i, 65))
    sheet.save(out/'grouped-edge-inspection.png')
    if time.perf_counter()-started > 20:
        raise TimeoutError('Grouped-edge comparison exceeded its work envelope')
    report = {'created_at': local_time(), 'status': 'architectural_vertical_estimated' if all(checks.values()) else 'architectural_vertical_rejected',
        'checks': checks, 'views': views, 'combined': combined, 'combined_error': combined_error,
        'combined_supported_weight_fraction_including_rejected_groups': fraction, 'cross_view_axis_difference_degrees': angle,
        'up_axis_in_camera0': None if upward is None else upward.tolist(), 'runtime_seconds': time.perf_counter()-started,
        'new_video_decodes': 0, 'new_line_detection_calls': 0, 'human_joints_used': False, 'floor_height_estimated': False,
        'reserved_camera_opened': False, 'paid_calls': 0, 'original_gates_unchanged': True,
        'limitations': ['Grouped image edges remain conditional on architectural and camera assumptions.',
            'Neither a floor-height estimate nor measured gravity, contact, anatomy or physical transfer follows.']}
    write_json(out/'report.json', report)
    print(json.dumps({'status': report['status'], 'checks': checks, 'groups': [len(v['edge_groups']['groups']) for v in views],
        'cross_view_axis_difference_degrees': angle, 'up_axis_in_camera0': report['up_axis_in_camera0'],
        'runtime_seconds': report['runtime_seconds'], 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
