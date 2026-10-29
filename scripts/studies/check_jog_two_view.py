"""Freeze two-image reconstruction before opening descriptive comparison references."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.camera_pose import validate_camera
from weft.camera_reference import JOINTS, project_world
from weft.io import digest, local_time, write_json
from weft.multiview_pose import CONTRACT, triangulate_pair
from weft.pipeline import verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'artifacts/multiview-jog-source-01'
EXTRA = ROOT/'artifacts/multiview-jog-extra-views-02'
POSES = [ROOT/'artifacts/multiview-jog-image-01/run',
         ROOT/'artifacts/multiview-jog-drone2-image-01', ROOT/'artifacts/multiview-jog-drone3-image-01']
BASELINE = ROOT/'artifacts/camera-jog-root-01'
SELECTED = [0, 120, 200, 238, 239]
CLOCK = ('Assumed aligned packaged frame indices. Provider preview viewer seeks all views to equal media time; '
         'all downloaded originals decode 240 frames at 30000/1001 fps. This does not independently prove '
         'cross-view alignment of the originals. Published recovered offsets are retained but not applied; '
         'no clock fitting, resampling or outcome-based offset selection in this experiment.')
EDGES = [(11, 12), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16),
         (11, 23), (12, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > 10 or sum(row.file_size for row in archive.infolist()) > 10_000_000:
            raise ValueError('Array archive exceeds bounded study size')
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def statistics(values):
    if not values: return {'count': 0, 'mean': None, 'rmse': None, 'p95': None, 'maximum': None}
    values = np.asarray(values)
    return {'count': len(values), 'mean': float(values.mean()), 'rmse': float(np.sqrt(np.mean(values**2))),
            'p95': float(np.percentile(values, 95)), 'maximum': float(values.max())}


def image_point(pose, joint, width, height):
    if pose is None or type(pose.get('pose_count')) is not int or pose['pose_count'] != 1: return None
    points = pose.get('landmarks')
    if not isinstance(points, list) or len(points) != 33 or not isinstance(points[joint], dict): return None
    values = [points[joint].get(key) for key in ('x', 'y', 'visibility', 'presence')]
    if any(type(value) not in (int, float) or not np.isfinite(value) for value in values): return None
    x, y, visibility, presence = values
    if not (0 <= x <= 1 and 0 <= y <= 1 and .5 <= visibility <= 1 and .5 <= presence <= 1): return None
    return np.array([x*width, y*height])


def projected(point, camera, frame):
    if point is None: return None
    row = camera['frames'][frame]
    _, pixels, valid = project_world(np.asarray(point)[None, :], row['R'], row['t'], camera['K'], camera['radial_k1'])
    return pixels[0] if valid[0] else None


def torso_up(points):
    if any(points[index] is None for index in (11, 12, 23, 24)): return None
    direction = (np.asarray(points[11])+points[12]-points[23]-points[24])/2
    norm = np.linalg.norm(direction)
    return float(direction[2]/norm) if norm > 1e-9 else None


def panels(out, cameras, poses, predictions, mono):
    videos = [SOURCE/'drone1.mp4', EXTRA/'drone2.mp4', EXTRA/'drone3.mp4']
    paths = []
    for view, video in enumerate(videos):
        capture = cv2.VideoCapture(str(video))
        try:
            for frame in range(240):
                ok, bgr = capture.read()
                if not ok: raise ValueError('Source ended before selected inspection frames')
                if frame not in SELECTED: continue
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((640, 360))
                draw = ImageDraw.Draw(image)
                layers = [('raw image joints', '#00d7ff', {j: image_point(poses[view].get(frame), j, 1920, 1080) for j in JOINTS.values()}),
                          ('single-view depth', '#a15df5', {j: projected(mono[frame][j], cameras[view], frame) for j in JOINTS.values()}),
                          ('two-view 3D', '#ff9e32', {j: projected(predictions['frames'][frame]['joints'][j]['world_point_m'], cameras[view], frame) for j in JOINTS.values()})]
                for _, color, points in layers:
                    pixels = {joint: tuple(point/3) for joint, point in points.items() if point is not None and np.abs(point).max() < 10000}
                    for a, b in EDGES:
                        if a in pixels and b in pixels: draw.line([pixels[a], pixels[b]], fill=color, width=2)
                    for x, y in pixels.values(): draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
                panel = Image.new('RGB', (640, 420), '#F5EADC'); panel.paste(image, (0, 60))
                label = ImageDraw.Draw(panel)
                label.text((10, 5), f'Drone {view+1} / frame {frame} / '+('comparison only' if view == 2 else 'construction view'), fill='#214F50')
                label.text((10, 23), 'Cyan: image joints | Purple: single-view depth | Orange: two-view 3D', fill='#214F50')
                label.text((10, 41), 'Autelica (autelica.com), CC BY 4.0 | Assumed packaged alignment', fill='#214F50')
                path = out/f'view-{view+1}-frame-{frame:04d}.png'; panel.save(path); paths.append(path.name)
        finally:
            capture.release()
    strip = Image.new('RGB', (1920, 420))
    for view in range(3):
        with Image.open(out/f'view-{view+1}-frame-0200.png') as image: strip.paste(image, (640*view, 0))
    strip.save(out/'frame-0200-comparison.png')
    return paths


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--first-pose-run', type=Path, default=POSES[0])
    parser.add_argument('--second-pose-run', type=Path, default=POSES[1]); args = parser.parse_args()
    pose_runs = [args.first_pose_run.resolve(), args.second_pose_run.resolve(), POSES[2]]
    out = args.out.resolve()
    if any(out.is_relative_to(path) for path in (SOURCE, EXTRA, BASELINE, *pose_runs)):
        raise ValueError('Study output must not alter any sealed input')
    out.mkdir(parents=True, exist_ok=False)
    implementation = ['src/weft/multiview_pose.py', 'src/weft/camera_pose.py', 'src/weft/camera_reference.py',
                      'tests/test_multiview_pose.py', 'tests/test_camera_pose.py',
                      'scripts/studies/check_jog_two_view.py', '.local/multiview-pose-tests.xml']
    hashes = {name: digest(ROOT/name) for name in implementation}
    for name in implementation:
        path = out/'implementation'/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, path)
    inputs = {str(path.relative_to(ROOT)): verify_artifacts(path) for path in (SOURCE, EXTRA, BASELINE, *pose_runs)}
    write_json(out/'pre-run.json', {'created_at': local_time(), 'method': dict(CONTRACT), 'implementation': hashes,
        'inputs': inputs, 'construction_views': ['drone1', 'drone2'], 'comparison_view': 'drone3',
        'clock': CLOCK, 'maximum_source_frames': 240, 'maximum_samples_per_view': 120, 'selected_frames': SELECTED,
        'scope': 'Development comparison; calibration itself uses multiple views and is reconstructed, not independent physical truth',
        'criteria': 'Descriptive all-slot availability and paired error; no pass threshold or outcome-based fitting',
        'excluded_from_construction': ['drone3 image poses', 'provider body joints', 'provider UV', 'contacts', 'monocular world landmarks'],
        'frame_200_selection': 'Previously observed inverted single-view pose; not a newly selected holdout',
        'paid_requests': 0, 'policy_training': False})
    started = time.monotonic()
    supplied, transforms = read(SOURCE/'cameras.json'), arrays(SOURCE/'extrinsics.npz')
    if (transforms['R'].shape != (3, 240, 3, 3) or transforms['t'].shape != (3, 240, 3)
            or not np.array_equal(transforms['frame'], np.arange(240))):
        raise ValueError('Supplied cameras do not match the pinned complete interval')
    evidence = [read(path/'evidence.json') for path in pose_runs[:2]]
    videos = [SOURCE/'drone1.mp4', EXTRA/'drone2.mp4']
    if any(raw['source_sha256'] != digest(video) for raw, video in zip(evidence, videos, strict=True)):
        raise ValueError('Construction evidence must belong to its exact original camera video')
    cameras, views = [], []
    base_camera = read(BASELINE/'camera.json')
    for index, raw in enumerate(evidence):
        camera = deepcopy(base_camera)
        camera.update(source_sha256=raw['source_sha256'], K=supplied['cameras'][index]['K'],
                      radial_k1=supplied['cameras'][index]['radial_k1'], provenance='Autelica reconstructed moving camera; '+CLOCK)
        camera['frames'] = [{'frame': frame, 'time_s': frame/camera['fps'],
                            'R': transforms['R'][index, frame].tolist(), 't': transforms['t'][index, frame].tolist()} for frame in range(240)]
        validate_camera(camera, raw); cameras.append(camera)
        views.append({'name': f'drone{index+1}', 'evidence': raw, 'camera': camera, 'clock_provenance': CLOCK})
        write_json(out/f'camera-{index+1}.json', camera)
    compute_start = time.monotonic()
    predictions = triangulate_pair(*views)
    compute_seconds = time.monotonic()-compute_start
    write_json(out/'predictions.json', predictions)
    prediction_hash = digest(out/'predictions.json')
    write_json(out/'construction.json', {'created_at': local_time(), 'prediction_sha256': prediction_hash,
        'runtime_seconds': compute_seconds, 'joint_status_counts': predictions['joint_status_counts'],
        'comparison_inputs_opened': False, 'clock': CLOCK,
        'supplied_sync_offsets_s_not_applied': {c['name']: c['sync_offset_s'] for c in supplied['cameras']}})

    # Comparison inputs are opened only after the candidate is fixed on disk.
    body = arrays(SOURCE/'body-coco17.npz')
    baseline = read(BASELINE/'camera-reference.json')
    third = read(pose_runs[2]/'evidence.json'); evidence.append(third)
    if third['source_sha256'] != digest(EXTRA/'drone3.mp4'):
        raise ValueError('Comparison evidence must belong to its exact third-camera source')
    camera = deepcopy(base_camera)
    camera.update(source_sha256=third['source_sha256'], K=supplied['cameras'][2]['K'],
                  radial_k1=supplied['cameras'][2]['radial_k1'], provenance='Autelica reconstructed comparison camera; '+CLOCK)
    camera['frames'] = [{'frame': frame, 'time_s': frame/camera['fps'],
                        'R': transforms['R'][2, frame].tolist(), 't': transforms['t'][2, frame].tolist()} for frame in range(240)]
    validate_camera(camera, third); cameras.append(camera); write_json(out/'camera-3.json', camera)
    if body['joints'].shape != (240, 17, 3) or not np.isfinite(body['joints']).all() or len(baseline['frames']) != 240:
        raise ValueError('Comparison arrays differ from the pinned complete interval')
    names = {name: index for index, name in enumerate(body['names'].tolist())}
    if not set(JOINTS) <= set(names): raise ValueError('Reference lacks named comparison joints')
    poses = [{row['frame']: row for row in raw['frames']} for raw in evidence]
    mono = [[(np.asarray(row['root_world_m'])+point).tolist() for point in row['world_relative_landmarks_m']]
            if row['status'] == 'estimated' else [None]*33 for row in baseline['frames']]
    rows, pair_3d, pair_third, conditional_3d, conditional_third = [], [], [], [], []
    per_joint, availability = {name: [] for name in JOINTS}, Counter()
    for frame in range(240):
        for name, joint in JOINTS.items():
            estimate = predictions['frames'][frame]['joints'][joint]
            point, old, reference = estimate['world_point_m'], mono[frame][joint], body['joints'][frame, names[name]]
            raw = image_point(poses[2].get(frame), joint, 1920, 1080)
            new_pixel, old_pixel = projected(point, camera, frame), projected(old, camera, frame)
            error = float(np.linalg.norm(np.asarray(point)-reference)) if point is not None else None
            old_error = float(np.linalg.norm(np.asarray(old)-reference)) if old is not None else None
            third_error = float(np.linalg.norm(new_pixel-raw)) if new_pixel is not None and raw is not None else None
            old_third = float(np.linalg.norm(old_pixel-raw)) if old_pixel is not None and raw is not None else None
            availability[estimate['status']] += 1
            if error is not None: conditional_3d.append(error); per_joint[name].append(error)
            if error is not None and old_error is not None: pair_3d.append((old_error, error))
            if third_error is not None: conditional_third.append(third_error)
            if third_error is not None and old_third is not None: pair_third.append((old_third, third_error))
            rows.append({'frame': frame, 'joint': name, 'triangulation_status': estimate['status'],
                'observation_statuses': estimate['observation_statuses'],
                'triangulated_world_m': point, 'monocular_world_m': old, 'reconstructed_reference_m': reference.tolist(),
                'triangulated_reference_error_m': error, 'monocular_reference_error_m': old_error,
                'comparison_image_px': raw.tolist() if raw is not None else None,
                'triangulated_comparison_error_px': third_error, 'monocular_comparison_error_px': old_third})
    write_json(out/'comparison.json', {'scope': 'Descriptive development agreement; no independent truth or untouched holdout', 'rows': rows})
    selected_paths = panels(out, cameras, poses, predictions, mono)
    report = {'created_at': local_time(), 'status': 'descriptive_comparison_complete', 'clock': CLOCK,
        'source_frames': 240, 'all_joint_slots': predictions['joint_slots'], 'all_joint_availability': predictions['joint_status_counts'],
        'major_joint_slots': len(rows), 'major_joint_availability': dict(availability),
        'conditional_reference_error_m': statistics(conditional_3d), 'conditional_comparison_image_error_px': statistics(conditional_third),
        'paired_reference_error_m': {key: statistics([row[index] for row in pair_3d]) for index, key in enumerate(['monocular', 'two_view'])},
        'paired_comparison_image_error_px': {key: statistics([row[index] for row in pair_third]) for index, key in enumerate(['monocular', 'two_view'])},
        'per_joint_conditional_reference_error_m': {name: statistics(values) for name, values in per_joint.items()},
        'frame_200_torso_vertical_direction': {'monocular': torso_up(mono[200]),
            'two_view': torso_up([j['world_point_m'] for j in predictions['frames'][200]['joints']]),
            'reconstructed_reference': torso_up({joint: body['joints'][200, names[name]] for name, joint in JOINTS.items()})},
        'construction_seconds': compute_seconds, 'total_seconds': time.monotonic()-started,
        'prediction_sha256': prediction_hash, 'selected_panels': selected_paths, 'paid_requests': 0,
        'new_body_or_policy': False, 'promoted': False, 'physical_validation': 'unverified'}
    if digest(out/'predictions.json') != prediction_hash or any(digest(ROOT/name) != value for name, value in hashes.items()):
        raise RuntimeError('Frozen predictions or implementation changed during comparison')
    if any(verify_artifacts(ROOT/name) != value for name, value in inputs.items()):
        raise RuntimeError('A sealed input changed during the study')
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
