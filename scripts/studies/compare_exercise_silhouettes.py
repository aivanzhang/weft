"""Compare saved 2D poses and SAM silhouettes without fitting either model."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
MASKS = ROOT/'artifacts/meta-exercise-silhouette-masks-02'
POSES = ROOT/'artifacts/fullbody-exercise-01/baseline/run'
JOINTS = {
    'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16, 'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30, 'left_foot_index': 31, 'right_foot_index': 32,
}
EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
         (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32)]
CONTRACT = {
    'method': 'saved-pose-silhouette-agreement-v1', 'minimum_confidence': .5,
    'selected_frames': [0, 34, 66], 'maximum_source_frames': 68, 'joints': JOINTS,
    'coordinates': 'Normalized image coordinates times submitted dimensions; half-open mask cells, no clipping',
    'sampling': 'Exact source frame and decoder timestamp only; retain unsampled frames, no interpolation',
    'scope': 'Cross-model 2D agreement, not independent anatomy, silhouette accuracy, depth or physical validation',
    'new_api_calls': 0, 'model_changes': False, 'manual_labels_used': False,
}


def _json(path):
    return json.loads(path.read_text())


def _cell(point, box, raster):
    left, top, right, bottom = box
    x, y = point
    if not left <= x < right or not top <= y < bottom:
        return None
    return [math.floor((y-top)*raster.shape[0]/(bottom-top)),
            math.floor((x-left)*raster.shape[1]/(right-left))]


def _joint(landmark, dimensions, mask, raster):
    row = {'landmark': landmark, 'source_point_px': None, 'mask_cell_row_col': None}
    if not isinstance(landmark, dict) or any(type(landmark.get(k)) not in (int, float)
            or not math.isfinite(landmark[k]) for k in ('x', 'y', 'visibility', 'presence')):
        return {**row, 'status': 'invalid_landmark'}
    if min(landmark['visibility'], landmark['presence']) < CONTRACT['minimum_confidence']:
        return {**row, 'status': 'low_confidence'}
    point = [landmark['x']*dimensions[0], landmark['y']*dimensions[1]]
    row['source_point_px'] = point
    if not all(0 <= p < n for p, n in zip(point, dimensions)):
        return {**row, 'status': 'outside_image'}
    if mask['mask_status'] != 'decoded' or raster is None:
        return {**row, 'status': 'mask_unavailable'}
    cell = _cell(point, mask['box']['box_xyxy_halfopen'], raster)
    row['mask_cell_row_col'] = cell
    inside = cell is not None and bool(raster[tuple(cell)])
    return {**row, 'status': 'inside_mask' if inside else 'outside_mask'}


def _analytic_checks():
    raster = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    box = [10, 20, 16, 28]
    assert _cell([11, 22], box, raster) == [0, 0]
    assert _cell([15, 26], box, raster) == [1, 2]
    assert _cell([10, 20], box, raster) == [0, 0]
    assert _cell([16, 22], box, raster) is None
    assert _cell([11, 28], box, raster) is None
    mask = {'mask_status': 'decoded', 'box': {'box_xyxy_halfopen': box}}
    point = {'x': .11, 'y': .22, 'presence': .5, 'visibility': .5}
    assert _joint(point, [100, 100], mask, raster)['status'] == 'inside_mask'
    assert _joint({**point, 'x': .13}, [100, 100], mask, raster)['status'] == 'outside_mask'
    assert _joint({**point, 'presence': .499}, [100, 100], mask, raster)['status'] == 'low_confidence'
    assert _joint({**point, 'x': 1}, [100, 100], mask, raster)['status'] == 'outside_image'
    assert _joint(point, [100, 100], mask, None)['status'] == 'mask_unavailable'
    return 10


def _overlay(out, frame, mask, points, dimensions):
    with Image.open(MASKS/frame['preview']) as raw:
        image = raw.convert('RGBA')
    sx, sy = image.width/dimensions[0], image.height/dimensions[1]
    if mask['raster_path']:
        left, top, right, bottom = mask['box']['box_xyxy_halfopen']
        bounds = [round(left*sx), round(top*sy), round(right*sx), round(bottom*sy)]
        with Image.open(MASKS/mask['raster_path']) as raw:
            patch = raw.convert('RGBA').resize((bounds[2]-bounds[0], bounds[3]-bounds[1]), Image.Resampling.NEAREST)
        image.alpha_composite(patch, bounds[:2])
    draw = ImageDraw.Draw(image)
    visible = {JOINTS[name]: row for name, row in points.items() if row['status'] in ('inside_mask', 'outside_mask')}
    for a, b in EDGES:
        if a in visible and b in visible:
            coords = [tuple(v*s for v, s in zip(visible[i]['source_point_px'], [sx, sy])) for i in (a, b)]
            draw.line(coords, fill='#214F50', width=2)
    for row in visible.values():
        x, y = [v*s for v, s in zip(row['source_point_px'], [sx, sy])]
        color = '#ffe348' if row['status'] == 'inside_mask' else '#eb4059'
        draw.ellipse((x-3, y-3, x+3, y+3), fill=color, outline='black', width=1)
    panel = Image.new('RGB', (image.width, image.height+80), '#F5EADC')
    panel.paste(image.convert('RGB'), (0, 80))
    caption = ImageDraw.Draw(panel)
    caption.text((10, 8), f"Frame {frame['frame']} / {frame['time_s']:.2f} seconds", fill='#214F50')
    counts = Counter(p['status'] for p in points.values())
    caption.text((10, 25), f"Yellow: inside {counts['inside_mask']} | Red: outside {counts['outside_mask']}", fill='#214F50')
    caption.text((10, 42), f"Unavailable joints: {len(points)-len(visible)} | Cyan: Meta silhouette", fill='#214F50')
    caption.text((10, 59), 'Model agreement only; not anatomical truth. JPEG preview.', fill='#214F50')
    panel.save(out/f"overlay-{frame['frame']:04d}.png")
    return panel


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True); args = parser.parse_args()
    out = Path(args.out).resolve()
    if any(out == p or p in out.parents for p in (MASKS, POSES)):
        raise ValueError('Study must be separate from its immutable inputs')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    script = Path(__file__).resolve()
    script_hash = digest(script)
    shutil.copyfile(script, out/script.name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'contract': CONTRACT,
        'inputs': {'masks': str(MASKS), 'poses': str(POSES)}, 'script_sha256': script_hash,
        'analytic_mapping_checks_passed': _analytic_checks(), 'maximum_output_bytes': 20_000_000})
    sources = {'masks': verify_artifacts(MASKS), 'poses': verify_artifacts(POSES)}
    for label, directory, names in [
        ('masks', MASKS, ['manifest.json', 'mask-evidence.json', 'original-evidence.json']),
        ('poses', POSES, ['manifest.json', 'pose-landmarks.json', 'source.json', 'run-config.json']),
    ]:
        target = out/'inputs'/label; target.mkdir(parents=True)
        for name in names:
            shutil.copyfile(directory/name, target/name)
    document = _json(MASKS/'mask-evidence.json')
    original = _json(MASKS/'original-evidence.json')
    poses = _json(POSES/'pose-landmarks.json')
    source = document['source']; dimensions = [source['width_px'], source['height_px']]
    assert source['submitted_video_sha256'] == digest(POSES/'source.mp4')
    assert canonical_hash(original) == document['original_evidence_sha256']
    assert document['identity_groups'] == original['identity_groups']
    assert len(document['frames']) == source['frame_count'] <= CONTRACT['maximum_source_frames']
    original_rows = {row['id']: row for row in original['observations']}
    pose_rows = {row['frame']: row for row in poses['frames']}
    assert len(pose_rows) == len(poses['frames'])
    assert set(pose_rows) <= {row['frame'] for row in document['frames']}
    rows, panels = [], []
    counts, by_joint = Counter(), {name: Counter() for name in JOINTS}
    verified_pngs = 0
    for frame in document['frames']:
        assert len(frame['observation_indices']) == 1
        mask = document['observations'][frame['observation_indices'][0]]
        original_row = original_rows[mask['id']]
        assert mask['raw_observation_sha256'] == canonical_hash(original_row)
        for key in ('frame', 'time_s', 'identity', 'phase', 'proposal_status'):
            assert mask[key] == original_row[key]
        assert ('original_source' in mask) == ('original_source' in original_row)
        if 'original_source' in mask:
            assert mask['original_source'] == original_row['original_source']
        raster = None
        if mask['raster_path']:
            with Image.open(MASKS/mask['raster_path']) as image:
                raw = image.tobytes(); raster = np.array(image)
                assert image.mode == 'P' and list(reversed(image.size)) == mask['raster_dimensions']
            assert sha256(raw).hexdigest() == mask['raster_sha256']
            assert sum(raw) == mask['silhouette']['foreground_pixels']
            verified_pngs += 1
        pose = pose_rows.get(frame['frame'])
        if pose:
            assert pose['id'] == f"{source['submitted_video_sha256']}:frame:{frame['frame']}"
            assert abs(pose['decoder_timestamp_s']-frame['time_s']) <= 1e-6
        points = {}
        for name, index in JOINTS.items():
            if pose is None:
                row = {'status': 'pose_not_sampled'}
            elif pose['pose_count'] != 1 or len(pose['landmarks']) != 33:
                row = {'status': 'pose_unavailable'}
            else:
                row = _joint(pose['landmarks'][index], dimensions, mask, raster)
            points[name] = row
            counts[row['status']] += 1
            by_joint[name][row['status']] += 1
        rows.append({'frame': frame['frame'], 'time_s': frame['time_s'], 'mask_id': mask['id'],
                     'pose_status': pose['status'] if pose else 'not_sampled', 'joints': points})
        if frame['frame'] in CONTRACT['selected_frames']:
            panels.append(_overlay(out, frame, mask, points, dimensions))
    write_json(out/'agreement.json', {'contract': CONTRACT, 'pose_model': poses['model'], 'frames': rows})
    strip = Image.new('RGB', (sum(p.width for p in panels), max(p.height for p in panels)), '#F5EADC')
    offset = 0
    for panel in panels:
        strip.paste(panel, (offset, 0)); offset += panel.width
    strip.save(out/'comparison.png')
    images = ''.join(f'<figure><img alt="Frame {n}: model agreement, not anatomy validation" src="overlay-{n:04d}.png"></figure>' for n in CONTRACT['selected_frames'])
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Weft: exercise silhouette comparison</title><style>body{max-width:1000px;margin:2rem auto;padding:1rem;'
        'font:17px system-ui;background:#F5EADC;color:#214F50}img{max-width:100%;height:auto}a{color:#965644}</style>'
        '<h1>Exercise joints and silhouette</h1><p>Yellow joints fall inside the Meta mask; red joints fall outside. '
        'Agreement between two models does not establish anatomical accuracy. These are saved 2D pose estimates, '
        'not a rendering of the 3D simulator. Missing and low-confidence points are not filled in.</p>'
        '<p><a href="agreement.json">Every frame and joint</a> · <a href="report.json">Checks and counts</a></p>'
        '<p>Source: <a href="https://commons.wikimedia.org/wiki/File:Bodyweight_Squats.gif">Bodyweight Squats</a>, '
        'Danielflefil, CC BY-SA 4.0. Lossless MP4 derivative; JPEG inspection previews with model overlays.</p>'+images)
    assert digest(script) == script_hash
    assert verify_artifacts(MASKS) == sources['masks']
    assert verify_artifacts(POSES) == sources['poses']
    output_bytes = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    assert output_bytes < 20_000_000
    report = {'status': 'descriptive_model_comparison', 'scope': CONTRACT['scope'], 'physics_ready': False,
        'frames': len(rows), 'sampled_pose_frames': len(pose_rows), 'joint_slots': sum(counts.values()),
        'counts': dict(counts), 'by_joint': {k: dict(v) for k, v in by_joint.items()},
        'verified_raster_pngs': verified_pngs, 'source_and_identity_unchanged': True,
        'input_verification': sources, 'runtime_seconds': time.monotonic()-started,
        'export_bytes_before_report': output_bytes, 'paid_requests': 0, 'model_changes': False}
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
