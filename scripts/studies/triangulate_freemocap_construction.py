"""Inspect complete two-view joints in camera coordinates without inventing a floor."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
import tomllib

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.camera_pose import FIT_JOINTS
from weft.exercise_vision import CONNECTIONS
from weft.io import digest, local_time, write_json
from weft.multiview_pose import CONTRACT, _observation, _triangulate_rays


ROOT = Path(__file__).resolve().parents[2]
POSES = ROOT/'artifacts/freemocap-full-pose-01'
EVIDENCE = ROOT/'artifacts/freemocap-full-evidence-01'
SOURCE = ROOT/'artifacts/freemocap-full-source-01'
SELECTED = [550, 824]
SEGMENTS = {'shoulder_width': (11, 12), 'hip_width': (23, 24),
    'left_upper_arm': (11, 13), 'right_upper_arm': (12, 14), 'left_forearm': (13, 15), 'right_forearm': (14, 16),
    'left_thigh': (23, 25), 'right_thigh': (24, 26), 'left_shin': (25, 27), 'right_shin': (26, 28)}


def read(path):
    return json.loads(path.read_text())


def verified(root, name):
    path = root/name
    if digest(path) != read(root/'manifest.json')['files'][name]:
        raise ValueError('Frozen input changed: '+name)
    return path


def spans(rows, required):
    result = []; active = None
    for row in rows:
        if row['frame'] % 2:
            continue
        present = all(row['joints'][i]['status'] == 'estimated' for i in required)
        if present:
            if active is None:
                active = {'first_frame': row['frame'], 'last_frame': row['frame'], 'sampled_poses': 1}
            else:
                active['last_frame'] = row['frame']; active['sampled_poses'] += 1
        elif active is not None:
            result.append(active); active = None
    if active is not None:
        result.append(active)
    return result


def plot_layout(rows):
    transform = np.array([[.8660254, 0., -.5], [.17101, .939693, .296198]])
    projected = np.array([transform @ np.array(j['world_point_m']) for row in rows
        for j in row['joints'] if j['status'] == 'estimated'])
    if not len(projected):
        raise ValueError('Inspection needs at least one estimated joint')
    low, high = projected.min(axis=0), projected.max(axis=0)
    scale = min(420/max(high[0]-low[0], 1e-6), 510/max(high[1]-low[1], 1e-6))
    return transform, (low+high)/2, scale


def panel(frame, source_images, row, cameras, target, layout):
    canvas = Image.new('RGB', (1200, 710), 'white'); draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), f'Frame {frame} / {frame/30:.3f} s: two-view joint estimates, not a physical body', fill='black')
    draw.text((15, 31), 'Cyan: reprojected estimated joints. Original pixels retained. Camera-space view has no inferred floor or gravity.', fill='black')
    points = {i: np.array(j['world_point_m']) for i, j in enumerate(row['joints']) if j['status'] == 'estimated'}
    for index, (source, camera) in enumerate(zip(source_images, cameras)):
        image = Image.open(source).convert('RGB').resize((360, 640)); ink = ImageDraw.Draw(image)
        projected = {}
        for joint, point in points.items():
            p = point @ camera[2].T+camera[3]; xy = p[:2]/p[2]; xy *= 1+camera[1]*(xy @ xy)
            uv = (camera[0] @ np.r_[xy, 1.])[:2]/2; projected[joint] = tuple(uv)
        for a, b in CONNECTIONS:
            if a in projected and b in projected:
                ink.line([projected[a], projected[b]], fill='#00ffff', width=2)
        for p in projected.values():
            ink.ellipse((p[0]-2, p[1]-2, p[0]+2, p[1]+2), fill='#00ffff')
        canvas.paste(image, (index*370, 65))
        draw.text((index*370+8, 50), f'Construction camera {index+1}', fill='black')
    draw.text((760, 65), 'Camera-space 3D estimate', fill='black')
    draw.text((760, 85), 'x: right; y: down; z: forward', fill='black')
    draw.text((760, 105), f'{len(points)}/33 estimated joints', fill='black')
    if points:
        # Shared bounds preserve relative movement without clipping extended limbs.
        transform, center, scale = layout
        projected = {i: tuple(np.array([975., 420.])+scale*(transform @ p-center)) for i, p in points.items()}
        if any(not (760 <= p[0] <= 1190 and 160 <= p[1] <= 680) for p in projected.values()):
            raise ValueError('Camera-space inspection would clip a joint')
        for a, b in CONNECTIONS:
            if a in projected and b in projected:
                draw.line([projected[a], projected[b]], fill='#214f50', width=3)
        for p in projected.values():
            draw.ellipse((p[0]-3, p[1]-3, p[0]+3, p[1]+3), fill='#965644')
    canvas.save(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    parent = read(verified(POSES, 'report.json'))
    if parent['status'] != 'complete_construction_pose_evidence':
        raise ValueError('Both complete pose streams are required')
    views = [read(verified(POSES, f'view-{i}/evidence.json')) for i in (1, 2)]
    calibration = tomllib.loads(verified(SOURCE, 'member-04-freemocap_sample_data_camera_calibration.toml').read_text())
    sources = read(verified(POSES, 'pre-run.json'))['views']
    cameras, lookup = [], []
    for i, evidence in enumerate(views):
        if (evidence['source_sha256'] != sources[i]['sha256'] or evidence['decoded_frames'] != 1108
                or evidence['fps'] != 30 or evidence['width'] != 720 or evidence['height'] != 1280
                or len(evidence['frames']) != 554):
            raise ValueError('Construction source identity, coverage or dimensions changed')
        c = calibration[f'cam_{i}']; K = np.array(c['matrix']); R = cv2.Rodrigues(np.array(c['rotation']))[0]; t = np.array(c['translation'])*.001
        if c['name']+'.mp4' != Path(sources[i]['member']).name or c['size'] != [720, 1280]:
            raise ValueError('Camera identity differs from source video')
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-10) or abs(np.linalg.det(R)-1) > 1e-10:
            raise ValueError('Camera rotation is not proper')
        cameras.append((K, c['distortions'][0], R, t, 720, 1280, np.linalg.inv(K), -t @ R))
        lookup.append({r['frame']: r for r in evidence['frames']})
    if not np.array_equal(cameras[0][2], np.eye(3)) or not np.array_equal(cameras[0][3], np.zeros(3)):
        raise ValueError('The source world is no longer camera 0')
    out.mkdir(exist_ok=False)
    for path in (Path(__file__), ROOT/'src/weft/multiview_pose.py', ROOT/'src/weft/camera_pose.py'):
        shutil.copyfile(path, out/path.name)
    write_json(out/'pre-run.json', {
        'created_at': local_time(), 'input_manifests': {p.name: digest(p/'manifest.json') for p in (POSES, EVIDENCE, SOURCE)},
        'triangulation_contract': dict(CONTRACT), 'world_axes': 'right_handed_camera0_x_right_y_down_z_forward',
        'units': 'm conditional on supplied board scale and cameras', 'source_frames': 1108, 'joint_slots': 1108*33,
        'sampled_source_frames': list(range(0, 1108, 2)), 'selected_inspection_frames': SELECTED,
        'maximum_seconds': 60, 'maximum_new_inspection_decodes': 2, 'reserved_camera_opened': False,
        'scope': 'Descriptive complete-source geometry using unchanged image confidence/ray gates; no camera-world z-up declaration, floor, body, interpolation or model inference.',
        'physical_validation': 'unverified', 'paid_calls': 0,
    })
    started = time.perf_counter(); rows = []; counts = Counter(); groups = Counter()
    for frame in range(1108):
        joints = []
        for joint in range(33):
            observations = [_observation(view.get(frame), joint, 720, 1280) for view in lookup]
            if all(p[0] is not None for p in observations):
                result = _triangulate_rays([p[0] for p in observations], cameras)
            else:
                result = {'status': 'image_evidence_unavailable', 'world_point_m': None, 'ray_angle_degrees': None, 'ray_separation_m': None, 'reprojection_px': None}
            joints.append(result | {'observation_statuses': [p[1] for p in observations]}); counts[result['status']] += 1
        hips = all(joints[i]['status'] == 'estimated' for i in (23, 24))
        torso = all(joints[i]['status'] == 'estimated' for i in (11, 12, 23, 24))
        complete = all(joints[i]['status'] == 'estimated' for i in FIT_JOINTS)
        group = 'unsampled' if frame % 2 else 'complete_body_joints' if complete else 'partial_with_torso' if torso else 'hips_only_support' if hips else 'hips_unavailable'
        groups[group] += 1
        rows.append({'frame': frame, 'time_s': frame/30, 'coverage': group,
            'source_evidence_ids': [view[frame]['id'] if frame in view else None for view in lookup], 'joints': joints})
        if time.perf_counter()-started > 60:
            raise TimeoutError('Complete two-view triangulation exceeded the work envelope')
    write_json(out/'camera-space-joints.json', {'schema_version': 1, 'world_axes': 'right_handed_camera0_x_right_y_down_z_forward',
        'units': 'm conditional on source calibration', 'frames': rows, 'floor': None, 'physical_validation': 'unverified'})
    spans_by_support = {'hips': spans(rows, (23, 24)), 'torso': spans(rows, (11, 12, 23, 24)), 'complete_body_joints': spans(rows, FIT_JOINTS)}
    write_json(out/'all-supported-sample-spans.json', spans_by_support)
    segment_rows, lengths = [], {name: [] for name in SEGMENTS}
    for row in rows:
        detail = {}
        for name, (a, b) in SEGMENTS.items():
            first, last = row['joints'][a]['world_point_m'], row['joints'][b]['world_point_m']
            value = None if first is None or last is None else float(np.linalg.norm(np.subtract(first, last)))
            detail[name] = value
            if value is not None:
                lengths[name].append(value)
        segment_rows.append({'frame': row['frame'], 'time_s': row['time_s'], 'lengths_m': detail})
    write_json(out/'all-segment-slots.json', segment_rows)
    segment_stats = {name: {'count': len(values), 'median_m': float(np.median(values)), 'minimum_m': min(values), 'maximum_m': max(values),
        'p95_absolute_deviation_from_median_m': float(np.percentile(np.abs(np.array(values)-np.median(values)), 95))} for name, values in lengths.items() if values}
    layout = plot_layout([rows[frame] for frame in SELECTED])
    for frame in SELECTED:
        paths = []
        for i in (0, 1):
            path = out/f'view-{i+1}-source-{frame:04d}.png'
            cached = EVIDENCE/path.name
            if cached.exists():
                shutil.copyfile(verified(EVIDENCE, path.name), path)
            else:
                video = ROOT/sources[i]['path']
                if digest(video) != sources[i]['sha256']:
                    raise ValueError('Inspection video identity changed')
                cap = cv2.VideoCapture(str(video))
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, bgr = cap.read()
                    if not ok or bgr.shape[:2] != (1280, 720) or abs(cap.get(cv2.CAP_PROP_POS_MSEC)/1000-frame/30) > 1e-7:
                        raise ValueError('Exact inspection source frame unavailable')
                    if not cv2.imwrite(str(path), bgr):
                        raise ValueError('Inspection PNG export failed')
                finally:
                    cap.release()
            paths.append(path)
        panel(frame, paths, rows[frame], cameras, out/f'source-and-joints-{frame:04d}.png', layout)
    report = {'created_at': local_time(), 'status': 'complete_camera_space_joint_inspection', 'source_frames': 1108, 'sampled_frames': 554,
        'joint_slots': 1108*33, 'joint_statuses': dict(counts), 'coverage': dict(groups), 'segment_statistics': segment_stats,
        'longest_supported_spans': {name: max(spans, key=lambda r: r['sampled_poses']) if spans else None for name, spans in spans_by_support.items()},
        'runtime_seconds': time.perf_counter()-started, 'new_inspection_frames_decoded': 2, 'cached_source_pngs_reused': 2,
        'new_inference_calls': 0, 'paid_calls': 0, 'reserved_camera_opened': False, 'body_built': False, 'floor': None,
        'limitations': ['Every original sampled, unsampled and unavailable source slot is retained; detected person counts are not ground truth.',
            'Coordinate frame remains the supplied camera-0 frame. No gravity alignment or floor/body admission is inferred.',
            'Segment constancy is a diagnostic, not measured anatomy. Stereo agreement and illustrative joints do not establish physical accuracy.']}
    write_json(out/'report.json', report)
    print(json.dumps({'coverage': dict(groups), 'joint_statuses': dict(counts), 'longest_spans': report['longest_supported_spans'],
        'runtime_seconds': report['runtime_seconds'], 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
