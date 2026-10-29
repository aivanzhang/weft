"""Trace frozen partial-limb errors to image evidence without repairing labels."""

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.camera_reference import JOINTS
from weft.io import local_time, write_json
from weft.multiview_pose import _observation, _triangulate_rays, _view
from weft.pipeline import verify_artifacts
from weft.spatial_pose import LIMB_ENDPOINTS
from check_jog_two_view import arrays, image_point, projected, statistics


def read(path):
    return json.loads(path.read_text())


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b), -1., 1.))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root/'artifacts/multiview-jog-source-01'
    extra = root/'artifacts/multiview-jog-extra-views-02'
    runs = [root/f'artifacts/{prefix}-jog-lab-01-both/run' for prefix in ('two-view-body', 'two-view-segment')]
    poses = [root/f'artifacts/full-pose-comparison-01/jog-drone{i}' for i in (1, 2)]
    camera_paths = [root/'artifacts/camera-jog-root-01/camera.json',
                    root/'artifacts/multiview-jog-full-two-view-01/camera-2.json']
    inputs = [source, extra, *runs, *poses, *(p.parent for p in camera_paths)]
    verification = {str(p): verify_artifacts(p) for p in inputs}
    out = args.out.resolve()
    if any(out.is_relative_to(p) for p in inputs):
        raise ValueError('Use a new output outside sealed inputs')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    shutil.copyfile(Path(__file__).with_name('check_jog_two_view.py'), out/'check_jog_two_view.py')
    write_json(out/'pre-run.json', {'created_at': local_time(), 'input_verification': verification,
        'selected_frames': [104, 148], 'scope': 'Post-hoc diagnosis of known regressions; no fresh holdout',
        'maximum_source_frames': 240, 'paid_requests': 0, 'candidate_or_label_changes': False})
    cameras, evidence = [read(p) for p in camera_paths], [read(p/'evidence.json') for p in poses]
    views = [{'name': f'drone{i+1}', 'camera': cameras[i], 'evidence': evidence[i],
              'clock_provenance': 'Same assumed packaged alignment as the frozen candidates; no offset search'} for i in (0, 1)]
    prepared = [_view(v) for v in views]
    provider = arrays(source/'body-coco17.npz')
    names = provider['names'].tolist()
    audit = read(runs[1]/'two-view-body.json')
    snapshots = [{round(r['source_time_s']*cameras[0]['fps']): r['human']['joints']
                  for r in read(p/'human-reference.json')['frames']} for p in runs]
    directions = []
    for row in audit['frames']:
        if not row['retained_body_pose']:
            continue
        for group, detail in row['groups'].items():
            if not detail.get('segment_transport_applied'):
                continue
            joint_names = [next(name for name, value in JOINTS.items() if value == j) for j in LIMB_ENDPOINTS[group]]
            base, middle, _ = joint_names
            f = row['frame']
            ref = provider['joints'][f, names.index(middle)]-provider['joints'][f, names.index(base)]
            errors = [angle(np.subtract(s[f][middle], s[f][base]), ref) for s in snapshots]
            directions.append({'frame': f, 'group': group, 'v1_direction_error_deg': errors[0],
                'v2_direction_error_deg': errors[1], 'delta_deg': errors[1]-errors[0],
                'image_confidence': detail['proximal_image_confidence'], 'point_statuses': detail['point_statuses']})
    write_json(out/'direction-errors.json', directions)
    hypotheses = []
    for f in range(cameras[0]['frame_count']):
        frame_cameras = [v[1][f] for v in prepared]
        for pair in ((23, 24), (25, 26), (27, 28)):
            points = [[_observation(v[0].get(f), j, 1920, 1080)[0] for j in pair] for v in prepared]
            result = {'frame': f, 'pair': list(pair), 'status': 'unavailable', 'hypotheses': None}
            if all(p is not None for row in points for p in row):
                alternatives = {}
                for label, indices in [('same_labels', (0, 1)), ('second_view_swapped', (1, 0))]:
                    estimates = [_triangulate_rays([points[0][i], points[1][indices[i]]], frame_cameras) for i in (0, 1)]
                    residuals = [e for v in estimates for e in (v['reprojection_px'] or [])]
                    alternatives[label] = {'joints': estimates,
                        'reprojection_rmse_px': statistics(residuals)['rmse'] if len(residuals) == 4 else None}
                result.update(status='compared_no_selection', hypotheses=alternatives)
            hypotheses.append(result)
    write_json(out/'correspondence-hypotheses.json', hypotheses)
    image_errors = []
    for view, (raw, camera, video) in enumerate(zip(evidence, cameras, (source/'drone1.mp4', extra/'drone2.mp4'), strict=True)):
        by_frame = {r['frame']: r for r in raw['frames']}
        capture = cv2.VideoCapture(str(video))
        try:
            for f in range(camera['frame_count']):
                ok, bgr = capture.read()
                if not ok:
                    raise ValueError('Source ended before its declared complete interval')
                if f not in (104, 148):
                    continue
                if abs(capture.get(cv2.CAP_PROP_POS_MSEC)/1000-f/camera['fps']) > 1e-6:
                    raise ValueError('Decoded and retained source clocks disagree')
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                layers = []
                for name in ('left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'):
                    raw_point = image_point(by_frame.get(f), JOINTS[name], camera['width'], camera['height'])
                    ref = projected(provider['joints'][f, names.index(name)], camera, f)
                    image_errors.append({'frame': f, 'view': view+1, 'joint': name,
                        'image_px': None if raw_point is None else raw_point.tolist(), 'reference_px': ref.tolist(),
                        'error_px': None if raw_point is None else float(np.linalg.norm(raw_point-ref))})
                    layers.append((name, raw_point, ref))
                all_points = np.array([p for _, raw_point, ref in layers for p in (raw_point, ref) if p is not None])
                left, top = np.maximum(0, np.floor(all_points.min(axis=0)-70)).astype(int)
                right, bottom = np.minimum([image.width, image.height], np.ceil(all_points.max(axis=0)+70)).astype(int)
                crop = image.crop((left, top, right, bottom)).resize((640, 640))
                draw = ImageDraw.Draw(crop)
                for name, raw_point, ref in layers:
                    for point, color in [(raw_point, '#00d7ff'), (ref, '#ff9e32')]:
                        if point is None: continue
                        x, y = (point-[left, top])*[640/(right-left), 640/(bottom-top)]
                        draw.ellipse((x-5, y-5, x+5, y+5), outline=color, width=3)
                        draw.text((x+8, y-12), name, fill=color, stroke_width=1, stroke_fill='#000000')
                panel = Image.new('RGB', (640, 720), '#F5EADC'); panel.paste(crop, (0, 80))
                title = ImageDraw.Draw(panel)
                title.text((12, 10), f'Drone {view+1}, frame {f} | source-only crop; known diagnostic frame', fill='#214F50')
                title.text((12, 30), 'Cyan: image-model labels | Orange: reconstructed reference projection', fill='#214F50')
                title.text((12, 50), 'Reference is not laboratory truth | Autelica (autelica.com), CC BY 4.0', fill='#214F50')
                panel.save(out/f'view-{view+1}-frame-{f:04d}.png')
        finally:
            capture.release()
    write_json(out/'selected-image-errors.json', image_errors)
    report = {'created_at': local_time(), 'partial_group_poses': len(directions),
        'direction_improved': sum(row['delta_deg'] < -1e-8 for row in directions),
        'direction_worsened': sum(row['delta_deg'] > 1e-8 for row in directions),
        'v1_direction_error_degrees': statistics([row['v1_direction_error_deg'] for row in directions]),
        'v2_direction_error_degrees': statistics([row['v2_direction_error_deg'] for row in directions]),
        'selected_directions': [row for row in directions if row['frame'] in (104, 148)],
        'selected_hypotheses': [row for row in hypotheses if row['frame'] in (104, 148)],
        'correspondence_slots': len(hypotheses), 'input_verification': verification,
        'scope': 'Development diagnosis; labels and geometry are unchanged; no correspondence selected',
        'anatomical_side_verified': False, 'paid_requests': 0}
    if any(verify_artifacts(Path(p)) != v for p, v in verification.items()):
        raise ValueError('A frozen input changed')
    write_json(out/'report.json', report)
    print(json.dumps({'partial_group_poses': len(directions), 'direction_improved': report['direction_improved'],
                      'direction_worsened': report['direction_worsened'], 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
