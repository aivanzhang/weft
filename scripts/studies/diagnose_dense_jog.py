"""Explain the frozen dense-pose rejection without changing evidence or body gates."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def longest_available(rows, pair):
    longest = current = 0
    for row in rows:
        current = current + 1 if row['pairs'][pair] == 'available' else 0
        longest = max(longest, current)
    return longest


def panel(bgr, pose, view, frame, out):
    source = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    landmarks = pose.get('landmarks') or []
    points = np.array([[p['x'] * source.width, p['y'] * source.height] for p in landmarks])
    if len(points) != 33 or not np.isfinite(points).all():
        raise ValueError('Selected diagnostic frame has no complete finite image hypothesis')
    low = np.maximum([0, 0], np.floor(points.min(axis=0) - 50)).astype(int)
    high = np.minimum([source.width, source.height], np.ceil(points.max(axis=0) + 50)).astype(int)
    if np.any(high <= low):
        raise ValueError('Image-model hypothesis does not intersect the source image')
    image = Image.new('RGB', (1200, 730), '#F5EADC')
    image.paste(source.resize((600, 338)), (0, 150))
    crop = source.crop((*low, *high))
    scale = min(580 / crop.width, 570 / crop.height)
    width, height = round(crop.width * scale), round(crop.height * scale)
    left, top = 600 + (600 - width) // 2, 95 + (570 - height) // 2
    image.paste(crop.resize((width, height)), (left, top))
    draw = ImageDraw.Draw(image)
    draw.text((15, 15), f'Drone {view}, frame {frame} | Original image-model labels, not corrected anatomy', fill='#214F50')
    draw.text((15, 38), f'Pose status: {pose["status"]} | No third-camera or reconstructed-body reference used', fill='#214F50')
    for index, name, color in ((23, 'L hip', '#00d7ff'), (24, 'R hip', '#ff9e32'),
                               (25, 'L knee', '#00d7ff'), (26, 'R knee', '#ff9e32'),
                               (27, 'L ankle', '#00d7ff'), (28, 'R ankle', '#ff9e32')):
        x, y = (points[index] - low) * [width / crop.width, height / crop.height] + [left, top]
        confidence = min(landmarks[index]['presence'], landmarks[index]['visibility'])
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
        draw.text((x + 6, y - 11), f'{name} {confidence:.2f}', fill=color, stroke_width=1, stroke_fill='#000000')
    draw.text((15, 525), 'Left: complete source frame | Right: model-defined crop', fill='#214F50')
    draw.text((15, 548), 'Image confidence does not verify identity or 3D consistency.', fill='#214F50')
    draw.text((15, 692), 'Autelica (autelica.com), CC BY 4.0 | Diagnostic only; no labels, camera or thresholds changed', fill='#214F50')
    image.save(out / f'view-{view}-frame-{frame:04d}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    dense = ROOT / 'artifacts/dense-jog-poses-01'
    study = ROOT / 'artifacts/limb-correspondence-dense-jog-01'
    build = ROOT / 'artifacts/dense-jog-body-build-01'
    source = ROOT / 'artifacts/multiview-jog-source-01'
    extra = ROOT / 'artifacts/multiview-jog-extra-views-02'
    originals = [ROOT / f'artifacts/full-pose-comparison-01/jog-drone{i}' for i in (1, 2)]
    inputs = [dense, study, build, source, extra, *originals]
    out = args.out.resolve()
    if any(out.is_relative_to(path) for path in inputs):
        raise ValueError('Use a new diagnostic directory outside sealed inputs')
    verified = {str(path): verify_artifacts(path) for path in inputs}
    raw = read(study / 'construction/raw-predictions.json')
    audit = read(study / 'construction/correspondence-audit.json')
    body = read(build / 'input/human.json')
    retained = {int(row['evidence_id'].rsplit(':', 1)[1]) for row in body['reference_evidence']}
    rows = [{'frame': row['frame'], 'time_s': row['time_s'],
             'body_pose_available': row['frame'] in retained,
             'hips_available': all(row['joints'][j]['status'] == 'estimated' for j in (23, 24)),
             'hips': [row['joints'][j] for j in (23, 24)]} for row in raw['frames']]
    rejected = [row['frame'] for row in rows if row['body_pose_available'] and not row['hips_available']]
    supported = [row['frame'] for row in rows if row['body_pose_available'] and row['hips_available']]
    if len(rows) != 240 or len(rejected) > 4 or not rejected:
        raise ValueError('This bounded diagnosis expects the saved 240-frame rejection')
    selected = set(rejected)
    for frame in rejected:
        selected.update([max(i for i in supported if i < frame), min(i for i in supported if i > frame)])
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / Path(__file__).name)
    write_json(out / 'pre-run.json', {'created_at': local_time(), 'inputs': verified,
        'selection': 'Every rejected retained root plus its nearest supported retained neighbor on each side',
        'selected_frames': sorted(selected), 'maximum_source_frames': 240,
        'scope': 'Post-hoc source-only failure inspection, not candidate fitting or a fresh holdout',
        'paid_requests': 0, 'new_inference': False})
    write_json(out / 'root-coverage.json', {'source_frames': len(rows), 'rows': rows})
    pairs = {pair: {'status_counts': dict(Counter(row['pairs'][pair] for row in audit['frames'])),
                    'longest_available_run': longest_available(audit['frames'], pair)}
             for pair in audit['frames'][0]['pairs']}
    cached, image_rows = [], []
    for view, original, video in zip((1, 2), originals,
            (source / 'drone1.mp4', extra / 'drone2.mp4'), strict=True):
        evidence = read(dense / f'drone{view}/evidence.json')
        previous = read(original / 'evidence.json')
        frames = {row['frame']: row for row in evidence['frames']}
        same = sum(row == frames[row['frame']] for row in previous['frames'])
        if same != len(previous['frames']) or digest(video) != evidence['source_sha256']:
            raise ValueError('Original observations or source identity changed')
        cached.append({'view': view, 'unchanged_cached_frames': same,
                       'new_inferred_frames': len(frames) - same, 'total_frames': len(frames)})
        capture = cv2.VideoCapture(str(video))
        try:
            for frame in range(evidence['frame_count']):
                ok, bgr = capture.read()
                if not ok or abs(capture.get(cv2.CAP_PROP_POS_MSEC)/1000-frame/evidence['fps']) > 1e-6:
                    raise ValueError('Complete source decoding or source clock failed')
                if frame not in selected:
                    continue
                panel(bgr, frames[frame], view, frame, out)
                image_rows.append({'view': view, 'frame': frame, 'pose': frames[frame]})
            if capture.read()[0]:
                raise ValueError('Source interval extends beyond the frozen frame count')
        finally:
            capture.release()
    write_json(out / 'selected-original-poses.json', image_rows)
    if any(verify_artifacts(Path(path)) != value for path, value in verified.items()):
        raise ValueError('A sealed input changed during diagnosis')
    report = {'created_at': local_time(), 'status': 'rejection_explained_no_candidate_created',
        'retained_primary_body_poses': len(retained), 'supported_retained_roots': len(supported),
        'unavailable_primary_body_frames': len(rows) - len(retained),
        'rejected_retained_root_frames': rejected, 'selected_frames': sorted(selected),
        'root_rejections': [row for row in rows if row['frame'] in rejected],
        'correspondence': pairs, 'minimum_required_context_samples': 5, 'cached_observations': cached,
        'original_build_failure': read(build / 'report.json'), 'all_original_inputs_unchanged': True,
        'body_reference_changed': False, 'comparison_references_used': False, 'paid_requests': 0,
        'scope': 'Additional estimates expose an inconsistent root; no skipped frame, interpolation or relaxed ray gate'}
    write_json(out / 'report.json', report)
    print(json.dumps({'retained_poses': len(retained), 'rejected_roots': rejected,
                      'correspondence': pairs, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
