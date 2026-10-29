"""Reproduce saved root comparisons without fitting to provider body references."""

import argparse
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.body_frame import CAMERA_TO_SIM
from weft.camera_pose import validate_camera
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import verify_artifacts


def read(path):
    return json.loads(path.read_text())


def statistics(values):
    values = np.asarray(values)
    return {'count': len(values), 'mean': float(values.mean()), 'rmse': float(np.sqrt(np.mean(values**2))),
            'p95': float(np.percentile(values, 95)), 'maximum': float(values.max())}


def draw_trajectory(rows, path):
    image = Image.new('RGB', (1000, 760), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), 'Body travel after subtracting each trajectory\'s first root', fill='black')
    draw.text((30, 43), 'Reconstructed reference, not measured physical truth. Missing poses remain missing.', fill='black')
    series = [('reference_displacement_m', '#214F50', 'Provider reference: 240 frames'),
              ('camera_displacement_m', '#965644', 'With supplied cameras: 112 poses'),
              ('baseline_displacement_m', '#777777', 'Original image-motion model: same 112 poses')]
    xy = np.array([row[key][:2] for row in rows for key, _, _ in series if row[key] is not None])
    low, high = xy.min(axis=0)-1, xy.max(axis=0)+1
    scale = min(850/(high[0]-low[0]), 550/(high[1]-low[1]))

    def project(value):
        return (80+(value[0]-low[0])*scale, 650-(value[1]-low[1])*scale)

    for x in range(int(np.ceil(low[0])), int(np.floor(high[0]))+1, 2):
        a, b = project((x, low[1])), project((x, high[1]))
        draw.line([a, b], fill='#eeeeee'); draw.text((a[0]-5, a[1]+8), str(x), fill='black')
    for y in range(int(np.ceil(low[1])), int(np.floor(high[1]))+1, 2):
        a, b = project((low[0], y)), project((high[0], y))
        draw.line([a, b], fill='#eeeeee'); draw.text((a[0]-25, a[1]-5), str(y), fill='black')
    for index, (key, color, label) in enumerate(series):
        draw.text((50, 690+index*20), label, fill=color)
        for row in rows:
            if row[key] is not None:
                x, y = project(row[key]); radius = 2 if index else 1
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    draw.text((30, 80), 'World y (model metres)', fill='black')
    draw.text((790, 666), 'World x (model metres)', fill='black')
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    study, source, run, out = (getattr(args, key).resolve() for key in ('study', 'source', 'run', 'out'))
    if any(out.is_relative_to(path) for path in (study, source, run)):
        raise ValueError('Reproduction must use a separate output directory')
    verified = {'source': verify_artifacts(source), 'baseline_run': verify_artifacts(run)}
    snapshots = {'predictions': study/'camera-reference.json', 'camera': study/'camera.json',
                 'body': source/'body-coco17.npz', 'baseline': run/'human.json'}
    hashes = {key: digest(path) for key, path in snapshots.items()}
    predictions, camera, baseline = (read(snapshots[key]) for key in ('predictions', 'camera', 'baseline'))
    validate_camera(camera, read(run/'evidence.json'))
    if predictions['source_sha256'] != camera['source_sha256'] or baseline['source_sha256'] != camera['source_sha256']:
        raise ValueError('Prediction, camera and body source identities differ')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'input_hashes': hashes,
        'verification': verified, 'scope': 'Post-hoc reproducibility check of frozen predictions, not a new held-out test',
        'alignment': 'Subtract each first root; rotate legacy displacement by first supplied camera only',
        'new_fitting_or_inference': False, 'paid_requests': 0})
    with zipfile.ZipFile(snapshots['body']) as archive:
        if len(archive.infolist()) > 10 or sum(row.file_size for row in archive.infolist()) > 10_000_000:
            raise ValueError('Reference archive exceeds study bounds')
    with np.load(snapshots['body'], allow_pickle=False) as body:
        names, joints = body['names'].tolist(), body['joints'].copy()
    count = camera['frame_count']
    if joints.shape != (count, 17, 3) or not np.isfinite(joints).all():
        raise ValueError('Reference joints differ from the pinned COCO17 interval')
    reference = (joints[:, names.index('left_hip')]+joints[:, names.index('right_hip')])/2
    reference_delta = reference-reference[0]
    original = {int(meta['evidence_id'].rsplit(':', 1)[1]): np.asarray(value['translation'])
                for meta, value in zip(baseline['reference_evidence'], baseline['root_reference'], strict=True)}
    estimated = [row for row in predictions['frames'] if row['status'] == 'estimated']
    if len(predictions['frames']) != count or not estimated or estimated[0]['frame'] != 0 or 0 not in original:
        raise ValueError('Study requires its complete original interval and initial root')
    origin, rotation = np.asarray(estimated[0]['root_world_m']), np.asarray(camera['frames'][0]['R'])
    rows = []
    for frame, prediction in enumerate(predictions['frames']):
        if prediction['frame'] != frame or abs(prediction['time_s']-frame/camera['fps']) > 1e-7:
            raise ValueError('Prediction clock differs from exact source frames')
        row = {key: None for key in ('baseline_displacement_error_m', 'baseline_displacement_m',
            'camera_absolute_error_m', 'camera_displacement_error_m', 'camera_displacement_m')}
        row.update(frame=frame, time_s=prediction['time_s'], status=prediction['status'],
                   reference_displacement_m=reference_delta[frame].tolist(), reference_root_world_m=reference[frame].tolist())
        if prediction['status'] == 'estimated':
            if frame not in original:
                raise ValueError('Compared candidate lacks the paired baseline root')
            root = np.asarray(prediction['root_world_m']); delta = root-origin
            old_delta = (original[frame]-original[0]) @ CAMERA_TO_SIM @ rotation
            row.update(camera_absolute_error_m=float(np.linalg.norm(root-reference[frame])),
                       camera_displacement_error_m=float(np.linalg.norm(delta-reference_delta[frame])),
                       baseline_displacement_error_m=float(np.linalg.norm(old_delta-reference_delta[frame])),
                       camera_displacement_m=delta.tolist(), baseline_displacement_m=old_delta.tolist())
        rows.append(row)
    summary = {key: statistics([row[key] for row in rows if row[key] is not None]) for key in
               ('baseline_displacement_error_m', 'camera_absolute_error_m', 'camera_displacement_error_m')}
    summary.update(paired_frames=len(estimated), source_frames=count,
                   maximum_camera_displacement_component_m=float(np.max(np.abs([row['camera_displacement_m'] for row in rows if row['camera_displacement_m'] is not None]))),
                   reference_final_displacement_m=reference_delta[-1].tolist())
    write_json(out/'comparison.json', {'frames': rows, 'summary': summary})
    expected = read(study/'root-comparison.json')
    if canonical_hash(rows) != canonical_hash(expected['frames']) or canonical_hash(summary) != canonical_hash(expected['summary']):
        raise ValueError('Reproduced rows or metrics differ from the original saved study')
    if any(digest(path) != hashes[key] for key, path in snapshots.items()):
        raise ValueError('Comparison inputs changed')
    draw_trajectory(rows, out/'trajectory.png')
    write_json(out/'report.json', {'status': 'exact_reproduction', 'created_at': local_time(), 'summary': summary,
        'original_comparison_sha256': digest(study/'root-comparison.json'), 'input_hashes': hashes,
        'runtime_seconds': time.monotonic()-started, 'paid_requests': 0,
        'scope': 'Same frozen development data, not new physical evidence or a new generalization test'})
    print(json.dumps({'summary': summary, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
