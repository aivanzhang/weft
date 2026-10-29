"""Inspect frozen world-coordinate geometry; never fit to the reconstructed reference."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.camera_reference import JOINTS
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import verify_artifacts


def read(path):
    return json.loads(path.read_text())


def stats(values):
    values = np.asarray(values)
    return {'count': len(values), 'mean': float(values.mean()), 'rmse': float(np.sqrt(np.mean(values**2))),
            'minimum': float(values.min()), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'maximum': float(values.max())}


def plot(rows, path):
    image = Image.new('RGB', (1000, 700), 'white'); draw = ImageDraw.Draw(image)
    draw.text((30, 15), 'Body heights in the supplied camera-world coordinates', fill='black')
    draw.text((30, 36), 'Reconstructed comparison only, not independently measured anatomy. Dots retain missing source frames.', fill='black')
    available = [row for row in rows if row['status'] == 'estimated']
    series = [('reference', '#214F50'), ('raw', '#965644'), ('body', '#777777')]
    for panel, field in enumerate(('root_height_m', 'lower_ankle_height_m')):
        top = 90+panel*290
        values = [row[field][key] for row in available for key, _ in series]
        low, high = min(0, min(values))-.05, max(values)+.05
        draw.text((25, top-25), 'Root height (model metres)' if panel == 0 else 'Lower ankle joint height (model metres; not a collision surface)', fill='black')
        for fraction in (0, .5, 1):
            y = top+200*(1-fraction); value = low+(high-low)*fraction
            draw.line((65, y, 950, y), fill='#dddddd'); draw.text((12, y-6), f'{value:.2f}', fill='black')
        for key, color in series:
            for row in available:
                x = 65+885*row['time_s']/rows[-1]['time_s']
                y = top+200*(1-(row[field][key]-low)/(high-low))
                draw.ellipse((x-1.5, y-1.5, x+1.5, y+1.5), fill=color)
        draw.text((70, top+210), '0 s', fill='black'); draw.text((900, top+210), f'{rows[-1]["time_s"]:.2f} s', fill='black')
    for i, (key, color) in enumerate(series):
        label = {'reference':'Provider reconstructed joints', 'raw':'Raw monocular joints + estimated camera root', 'body':'Fixed articulated body + same root'}[key]
        draw.text((30, 625+i*20), label, fill=color)
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    for name in ('baseline', 'candidate', 'source', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    old, new, source, out = (getattr(args, key).resolve() for key in ('baseline', 'candidate', 'source', 'out'))
    if any(out.is_relative_to(path) for path in (old, new, source)):
        raise ValueError('Use a new study directory outside sealed inputs')
    verified = {key: verify_artifacts(path) for key, path in [('baseline', old), ('candidate', new), ('source', source)]}
    previous, current = read(old/'human.json'), read(new/'human.json')
    for key in ('reference', 'root_reference', 'lengths', 'body_frame', 'reference_evidence', 'motor_limits'):
        if canonical_hash(previous[key]) != canonical_hash(current[key]):
            raise ValueError('Ground placement changed body geometry or trajectory')
    if digest(old/'camera-reference.json') != digest(new/'camera-reference.json'):
        raise ValueError('Ground placement changed raw camera-root estimates')
    if current['source_sha256'] != digest(source/'drone1.mp4'):
        raise ValueError('Source identity differs from the completed body')
    out.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'inputs': verified,
        'hypothesis': 'Floor placement changes no body point in camera-world coordinates; remaining height disagreement separates into root, raw shape and authored-geometry components',
        'scope': 'Post-construction descriptive comparison; provider joints are reconstructed, not independent physical truth',
        'criteria': 'Preserve every source slot. World-point invariance and component identities within 1e-10; no accuracy pass threshold or fit.',
        'paid_requests': 0, 'new_fitting': False})
    with zipfile.ZipFile(source/'body-coco17.npz') as archive:
        if len(archive.infolist()) > 10 or sum(row.file_size for row in archive.infolist()) > 10_000_000:
            raise ValueError('Reference archive exceeds study bounds')
    with np.load(source/'body-coco17.npz', allow_pickle=False) as archive:
        joints, names = archive['joints'].astype(float), archive['names'].tolist()
    estimates = read(new/'camera-reference.json')
    if joints.shape != (estimates['source_frames'], 17, 3) or not np.isfinite(joints).all():
        raise ValueError('Reference has the wrong frame interval or geometry')
    evidence = read(new/'evidence.json')
    poses = {row['frame']: row for row in evidence['frames']}
    old_frames = read(old/'human-reference.json')['frames']; new_frames = read(new/'human-reference.json')['frames']
    frames = {row['evidence_id']: (a, b) for row, a, b in zip(current['reference_evidence'], old_frames, new_frames, strict=True)}
    origin = np.array(current['camera_reference']['origin_world_m'])
    old_offset = origin-[0, 0, previous['mobility']['base_height']]
    new_offset = np.array([origin[0], origin[1], current['camera_reference']['ground']['height_m']])
    rows, values, high_confidence, ankle_parts, world_errors = [], [], [], [], []
    for estimate in estimates['frames']:
        frame = estimate['frame']; stamp = estimate['time_s']
        row = {'frame': frame, 'time_s': stamp, 'status': estimate['status'], 'joints': None}
        rows.append(row)
        if estimate['status'] != 'estimated':
            continue
        before, after = frames[estimate['evidence_id']]
        if abs(after['source_time_s']-stamp) > 1e-7 or abs(before['source_time_s']-stamp) > 1e-7:
            raise ValueError('Source and candidate clocks differ')
        native = {name: np.asarray(point)+new_offset for name, point in after['human']['joints'].items()}
        for name, point in before['human']['joints'].items():
            world_errors.append(float(np.max(np.abs(np.asarray(point)+old_offset-native[name]))))
        root = np.asarray(estimate['root_world_m'])
        reference_root = (joints[frame, names.index('left_hip')]+joints[frame, names.index('right_hip')])/2
        raw = np.asarray(estimate['world_relative_landmarks_m'])+root
        compared = {}
        for name, index in JOINTS.items():
            target = joints[frame, names.index(name)]
            image = poses[frame]['landmarks'][index]
            confidence = min(image['visibility'], image['presence'])
            usable = confidence >= .5 and 0 <= image['x'] <= 1 and 0 <= image['y'] <= 1
            entry = {'raw_error_m': float(np.linalg.norm(raw[index]-target)),
                     'body_error_m': float(np.linalg.norm(native[name]-target)),
                     'body_vs_raw_error_m': float(np.linalg.norm(native[name]-raw[index])),
                     'source_confidence': confidence, 'confident_in_image': usable}
            compared[name] = entry; values.append(entry)
            if usable: high_confidence.append(entry)
            if name.endswith('ankle'):
                components = np.array([root[2]-reference_root[2],
                    (raw[index, 2]-root[2])-(target[2]-reference_root[2]), native[name][2]-raw[index, 2]])
                if abs(components.sum()-(native[name][2]-target[2])) > 1e-10:
                    raise ValueError('Height error decomposition is inconsistent')
                entry['vertical_error_components_m'] = components.tolist(); ankle_parts.append(components)
        row.update(joints=compared, root_height_m={'reference': float(reference_root[2]), 'raw': float(root[2]), 'body': float(native['pelvis'][2])},
            lower_ankle_height_m={'reference': float(min(joints[frame, names.index(n), 2] for n in ('left_ankle', 'right_ankle'))),
                                 'raw': float(min(raw[27, 2], raw[28, 2])),
                                 'body': float(min(native['left_ankle'][2], native['right_ankle'][2]))})
    if max(world_errors) > 1e-10:
        raise ValueError('Ground placement changed the underlying body world geometry')
    write_json(out/'geometry.json', {'frames': rows, 'source_frames': len(rows), 'joint_slots': len(rows)*len(JOINTS)})
    plot(rows, out/'height-comparison.png')
    components = np.array(ankle_parts)
    summary = {label: {key: stats([row[key] for row in data]) for key in ('raw_error_m', 'body_error_m', 'body_vs_raw_error_m')}
               for label, data in [('all_available_joints', values), ('confident_in_image_joints', high_confidence)]}
    report = {'created_at': local_time(), 'status': 'coordinate_invariance_verified_height_mismatch_retained',
        'inputs': verified, 'source_statuses': dict(Counter(row['status'] for row in rows)),
        'source_frames': len(rows), 'joint_slots': len(rows)*len(JOINTS), 'summary': summary,
        'ankle_vertical_error_components_m': {name: stats(components[:, i]) for i, name in enumerate(('root', 'raw_relative_shape', 'authored_geometry'))},
        'component_scope': 'Signed vertical components add per endpoint. RMS magnitudes do not add; all available ankle slots retained.',
        'world_points_checked': len(world_errors), 'maximum_world_point_change': max(world_errors),
        'body_model_unchanged_except_floor_offset': True, 'ground_clearance': read(new/'ground-clearance.json')['summary'],
        'reference_sha256': digest(source/'body-coco17.npz'), 'paid_requests': 0, 'runtime_seconds': time.monotonic()-started,
        'scope': 'Reconstructed-reference agreement is descriptive, not accurate anatomy, contact or physical transfer'}
    for key, path in [('baseline', old), ('candidate', new), ('source', source)]:
        if verify_artifacts(path) != verified[key]: raise ValueError('Study input changed')
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
