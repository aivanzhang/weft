"""Grade frozen monocular outputs against an explicitly inferred multiview reference."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.camera_reference import CONTRACT, JOINTS, compare_frame, project_world
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'artifacts/multiview-jog-source-01'
RUNS = {'video': ROOT/'artifacts/multiview-jog-baseline-01', 'image': ROOT/'artifacts/multiview-jog-image-01'}
SELECTED = [0, 120, 238, 239]
EDGES = [('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip')]+[
    (f'{side}_{a}', f'{side}_{b}') for side in ('left', 'right')
    for a, b in [('shoulder', 'elbow'), ('elbow', 'wrist'), ('shoulder', 'hip'), ('hip', 'knee'), ('knee', 'ankle')]]


def _json(path):
    return json.loads(path.read_text())


def _arrays(path):
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > 10 or sum(row.file_size for row in archive.infolist()) > 10_000_000:
            raise ValueError('Reference array archive exceeds bounds')
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _summary(values):
    if not values:
        return {'count': 0, 'mean': None, 'rmse': None, 'p95': None, 'maximum': None}
    values = np.asarray(values)
    return {'count': len(values), 'mean': float(values.mean()), 'rmse': float(np.sqrt(np.mean(values**2))),
            'p95': float(np.percentile(values, 95)), 'maximum': float(values.max())}


def _panel(image, rows, mode, frame):
    image = image.resize((640, 360)).convert('RGB')
    draw = ImageDraw.Draw(image)
    sx, sy = 640/1920, 360/1080
    for key, color in [('reference_image_px', '#00d7ff'), ('predicted_image_px', '#ff9e32')]:
        points = {name: tuple(v*s for v, s in zip(row[key], [sx, sy]))
                  for name, row in rows.items() if row[key] is not None}
        for a, b in EDGES:
            if a in points and b in points:
                draw.line([points[a], points[b]], fill=color, width=2)
        for x, y in points.values():
            draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
    panel = Image.new('RGB', (640, 420), '#F5EADC'); panel.paste(image, (0, 60))
    draw = ImageDraw.Draw(panel)
    counts = Counter(row['image_status'] for row in rows.values())
    draw.text((10, 7), f'{mode.upper()} mode / frame {frame} / {dict(counts)}', fill='#214F50')
    draw.text((10, 24), 'Cyan: provider 3D projected into camera | Orange: Weft raw 2D pose', fill='#214F50')
    draw.text((10, 41), 'Both are estimates; no fitted alignment or physical-accuracy claim.', fill='#214F50')
    return panel


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True); args = parser.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    snapshots = ['src/weft/camera_reference.py', 'tests/test_camera_reference.py',
                 'scripts/studies/check_multiview_pose.py', '.local/camera-reference-tests.xml']
    hashes = {name: digest(ROOT/name) for name in snapshots}
    for name in snapshots:
        target = out/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'contract': CONTRACT, 'selected_frames': SELECTED,
        'scope': 'Development comparison with a third-party reconstructed reference; not marker-based or independent anatomical truth',
        'source': str(SOURCE), 'runs': {k: str(v) for k, v in RUNS.items()}, 'snapshots': hashes,
        'maximum_frames': 240, 'maximum_array_archive_bytes': 10_000_000, 'new_api_calls': 0,
        'prediction_to_reference_time_rule': 'Exact source index with matching decoder timestamp; drone1 offset is zero',
        'source_scale': 'Provider used stated subject height; metric reference scale is not independently measured here',
        'criteria': 'Descriptive errors and complete availability, no fitted alignment or post-hoc pass threshold'})
    source_verification = verify_artifacts(SOURCE)
    for name in ('cameras.json', 'extrinsics.npz', 'body-coco17.npz', 'metadata.json', 'LICENSE.txt'):
        shutil.copyfile(SOURCE/name, out/name)
    cameras = _json(SOURCE/'cameras.json'); metadata = _json(SOURCE/'metadata.json')
    extrinsics, body = _arrays(SOURCE/'extrinsics.npz'), _arrays(SOURCE/'body-coco17.npz')
    assert extrinsics['R'].shape == (3, 240, 3, 3) and extrinsics['t'].shape == (3, 240, 3)
    assert body['joints'].shape == (240, 17, 3) and body['uv'].shape == (3, 240, 17, 2)
    assert np.array_equal(extrinsics['frame'], np.arange(240))
    assert np.isfinite(body['joints']).all() and np.isfinite(body['uv']).all()
    assert len(set(body['names'].tolist())) == 17 and set(JOINTS) <= set(body['names'].tolist())
    assert cameras['world']['units'] == 'm' and cameras['world']['up'] == '+z'
    assert cameras['cameras'][0]['name'] == 'drone1' and cameras['cameras'][0]['sync_offset_s'] == 0
    names = {name: i for i, name in enumerate(body['names'].tolist())}
    reference, projection = [], []
    residuals = {}
    for c, camera in enumerate(cameras['cameras']):
        distances = []
        for frame in range(240):
            xyz, uv, valid = project_world(body['joints'][frame], extrinsics['R'][c, frame], extrinsics['t'][c, frame],
                                            camera['K'], camera['radial_k1'])
            distances.extend(np.linalg.norm(uv[valid]-body['uv'][c, frame, valid], axis=1).tolist())
            if c == 0:
                reference.append({name: xyz[names[name]] for name in JOINTS})
                projection.append({name: uv[names[name]] for name in JOINTS})
        residuals[camera['name']] = _summary(distances)
    capture = cv2.VideoCapture(str(SOURCE/'drone1.mp4'))
    source_frames, images = [], {}
    for frame in range(241):
        ok, bgr = capture.read()
        if not ok: break
        assert frame < 240 and bgr.shape[:2] == (1080, 1920)
        stamp = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
        assert abs(stamp-frame/metadata['fps']) < 1e-6
        source_frames.append({'frame': frame, 'time_s': stamp})
        if frame in SELECTED:
            images[frame] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    assert len(source_frames) == metadata['frames'] == 240
    write_json(out/'source-clock.json', source_frames)
    input_hashes, reports, panels = {}, {}, []
    for mode, directory in RUNS.items():
        state = _json(directory/'status.json')
        assert state['status'] in ('ready', 'failed', 'cancelled'), 'Do not grade live or interrupted work'
        selected = ['status.json', 'lab-config.json', 'run/pose-landmarks.json', 'run/evidence.json',
                    'run/identity-audit.json', 'run/run-config.json']
        if (directory/'failure.json').exists(): selected.append('failure.json')
        input_hashes[mode] = {name: digest(directory/name) for name in selected}
        for name in selected:
            target = out/'inputs'/mode/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory/name, target)
        evidence = _json(directory/'run/evidence.json'); poses = _json(directory/'run/pose-landmarks.json')
        assert evidence['source_sha256'] == digest(SOURCE/'drone1.mp4') == digest(directory/'run/source.mp4')
        assert (evidence['width'], evidence['height'], evidence['frame_count']) == (1920, 1080, 240)
        assert evidence['model']['running_mode'] == mode.upper()
        rows_by_frame = {row['frame']: row for row in poses['frames']}
        assert len(rows_by_frame) == len(poses['frames']) and set(rows_by_frame) <= set(range(240))
        rows, counts2d, counts3d, errors2d, errors3d = [], Counter(), Counter(), [], []
        for frame in source_frames:
            n = frame['frame']; pose = rows_by_frame.get(n)
            if pose is not None:
                assert pose['id'] == f"{evidence['source_sha256']}:frame:{n}"
                assert abs(pose['decoder_timestamp_s']-frame['time_s']) < 1e-6
            comparisons = compare_frame(pose, reference[n], projection[n], width=1920, height=1080)
            rows.append({**frame, 'pose_count': pose['pose_count'] if pose else None, 'joints': comparisons})
            for row in comparisons.values():
                counts2d[row['image_status']] += 1; counts3d[row['relative_3d_status']] += 1
                if row['image_distance_px'] is not None: errors2d.append(row['image_distance_px'])
                if row['relative_3d_distance_mm'] is not None: errors3d.append(row['relative_3d_distance_mm'])
            if n in SELECTED:
                panel = _panel(images[n], comparisons, mode, n); panel.save(out/f'{mode}-{n:04d}.png'); panels.append(panel)
        write_json(out/f'{mode}-comparison.json', {'contract': CONTRACT, 'frames': rows})
        reports[mode] = {'pipeline_status': state['status'], 'failure': _json(directory/'failure.json') if (directory/'failure.json').exists() else None,
                         'source_frames': len(rows), 'sampled_pose_frames': len(rows_by_frame),
                         'pose_statuses': dict(Counter(row['status'] for row in poses['frames'])),
                         'image_statuses': dict(counts2d), 'relative_3d_statuses': dict(counts3d),
                         'conditional_image_distance_px': _summary(errors2d),
                         'conditional_relative_3d_distance_mm': _summary(errors3d),
                         'run_verification': verify_artifacts(directory/'run') if (directory/'run/manifest.json').exists() else None}
    strip = Image.new('RGB', (2560, 840), '#F5EADC')
    for index, panel in enumerate(panels): strip.paste(panel, ((index%4)*640, (index//4)*420))
    strip.save(out/'comparison.png')
    figures = ''.join(f'<h2>{mode.upper()} mode</h2>'+''.join(
        f'<img src="{mode}-{n:04d}.png" alt="{mode} frame {n}: inferred reference and monocular pose">' for n in SELECTED)
        for mode in RUNS)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Weft: moving-camera pose comparison</title><style>body{font:17px system-ui;background:#F5EADC;color:#214F50;'
        'max-width:1100px;margin:2rem auto;padding:1rem}img{max-width:100%;height:auto}a{color:#965644}</style>'
        '<h1>Moving-camera exercise test</h1><p>Complete eight-second clip. Cyan is a third-party multiview reconstruction; '
        'orange is the raw monocular pose. The reference is not laboratory motion capture. No rotation, scale or time offset '
        'is fitted to improve agreement. Unsampled/competing poses remain unavailable.</p>'
        '<p><a href="report.json">Results and failures</a> · <a href="video-comparison.json">Every VIDEO-mode slot</a> · '
        '<a href="image-comparison.json">Every IMAGE-mode slot</a></p>'
        '<p>Source: <a href="https://autelica.com/sample/">Autelica (autelica.com)</a>, CC BY 4.0. '
        'Modified with comparison overlays. Provider calibration, timing and height-based metric scale are estimates.</p>'+figures)
    assert all(digest(ROOT/name) == value for name, value in hashes.items())
    assert all(digest(RUNS[mode]/name) == value for mode, inputs in input_hashes.items() for name, value in inputs.items())
    assert verify_artifacts(SOURCE) == source_verification
    report = {'status': 'completed_development_comparison', 'contract': CONTRACT, 'source_verification': source_verification,
              'input_hashes': input_hashes, 'runs': reports, 'provider_reprojection_to_own_detections_px': residuals,
              'provider_reprojection_scope': 'In-sample consistency, not the provider claimed held-out-camera evaluation',
              'runtime_seconds': time.monotonic()-started, 'paid_requests': 0, 'physics_ready': False}
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
