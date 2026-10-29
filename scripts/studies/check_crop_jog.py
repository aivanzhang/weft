"""Score frozen crop/body candidates on complete slots and matched available subsets."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from weft.benchmark import seal_suite
from weft.camera_reference import JOINTS
from weft.io import local_time, write_json
from weft.pipeline import verify_artifacts
from check_jog_two_view import arrays, image_point, projected, statistics


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'artifacts/two-view-segment-jog-lab-01-both/run'
NEW = ROOT/'artifacts/temporal-crop-jog-body-lab-01/run'
CROP = ROOT/'artifacts/temporal-crop-jog-01'
RAW = ROOT/'artifacts/limb-correspondence-dense-jog-01'
SOURCE = ROOT/'artifacts/multiview-jog-source-01'
THIRD = ROOT/'artifacts/multiview-jog-drone3-image-01'
CAMERAS = ROOT/'artifacts/multiview-jog-full-two-view-01'


def read(path):
    return json.loads(path.read_text())


def body_points(path):
    spec = read(path/'human.json')
    offset = np.array([*spec['camera_reference']['origin_world_m'][:2], spec['camera_reference']['ground']['height_m']])
    rows = {}
    for meta, root, snapshot in zip(spec['reference_evidence'], spec['root_reference'],
                                    read(path/'human-reference.json')['frames'], strict=True):
        frame = int(meta['evidence_id'].rsplit(':', 1)[1])
        if abs(snapshot['source_time_s']-frame/(30000/1001)) > 1e-7:
            raise ValueError('Native body/source clocks differ')
        origin = np.asarray(root['translation'])+spec['camera_reference']['origin_world_m']
        for name in JOINTS:
            rows[(frame, name)] = {'point': np.asarray(snapshot['human']['joints'][name])+offset, 'root': origin}
    return rows, spec


def ray_points(predictions):
    result = {}
    for row in predictions['frames']:
        hips = [row['joints'][j] for j in (23, 24)]
        root = np.mean([p['world_point_m'] for p in hips], axis=0) if all(p['status'] == 'estimated' for p in hips) else None
        for name, joint in JOINTS.items():
            point = row['joints'][joint]
            if point['status'] == 'estimated':
                result[(row['frame'], name)] = {'point': np.asarray(point['world_point_m']), 'root': root}
    return result


def compare(out, label, before, after, provider, third, camera):
    names = provider['names'].tolist()
    roots = provider['joints'][:, [names.index('left_hip'), names.index('right_hip')]].mean(axis=1)
    scores = {group: {metric: [] for metric in ('world_m', 'root_relative_m', 'third_px')}
              for group in ('before', 'after', 'paired_before', 'paired_after', 'added_after', 'lost_before')}
    per_joint = {name: {'before': [], 'after': []} for name in JOINTS}
    rows, coverage, deltas = [], Counter(), []
    for frame in range(240):
        for name, index in JOINTS.items():
            key = (frame, name)
            present = [key in values for values in (before, after)]
            status = 'paired' if all(present) else 'added' if present[1] else 'lost' if present[0] else 'unavailable'
            coverage[status] += 1
            reference = provider['joints'][frame, names.index(name)]
            image = image_point(third.get(frame), index, camera['width'], camera['height'])
            row = {'frame': frame, 'joint': name, 'status': status, 'before': None, 'after': None}
            for mode, values in (('before', before), ('after', after)):
                if key not in values:
                    continue
                point, root = values[key]['point'], values[key]['root']
                pixels = projected(point, camera, frame) if image is not None else None
                errors = {'world_m': float(np.linalg.norm(point-reference)),
                    'root_relative_m': float(np.linalg.norm(point-root-(reference-roots[frame]))) if root is not None else None,
                    'third_px': float(np.linalg.norm(pixels-image)) if pixels is not None else None}
                row[mode] = errors
                groups = [mode]
                if status == 'paired': groups.append('paired_'+mode)
                elif status == 'added': groups.append('added_after')
                elif status == 'lost': groups.append('lost_before')
                for group in groups:
                    for metric, value in errors.items():
                        # Matched image/root scores require both estimates to support that comparison.
                        other = after if mode == 'before' else before
                        if group.startswith('paired') and metric == 'root_relative_m' and other[key]['root'] is None:
                            continue
                        if group.startswith('paired') and metric == 'third_px' and projected(other[key]['point'], camera, frame) is None:
                            continue
                        if value is not None: scores[group][metric].append(value)
                if status == 'paired': per_joint[name][mode].append(errors['world_m'])
            if status == 'paired':
                deltas.append({'frame': frame, 'joint': name, 'world_error_delta_m': row['after']['world_m']-row['before']['world_m']})
            rows.append(row)
    write_json(out/(label+'-slots.json'), rows)
    write_json(out/(label+'-paired-deltas.json'), deltas)
    return {'source_frames': 240, 'joint_slots': len(rows), 'coverage': dict(coverage),
        'metrics': {group: {metric: statistics(values) for metric, values in group_values.items()} for group, group_values in scores.items()},
        'paired_per_joint': {joint: {mode: statistics(values) for mode, values in modes.items()} for joint, modes in per_joint.items()},
        'paired_errors_improved': sum(row['world_error_delta_m'] < -1e-10 for row in deltas),
        'paired_errors_worsened': sum(row['world_error_delta_m'] > 1e-10 for row in deltas),
        'paired_errors_unchanged': sum(abs(row['world_error_delta_m']) <= 1e-10 for row in deltas),
        'largest_paired_regression': max(deltas, key=lambda row: row['world_error_delta_m']) if deltas else None,
        'known_problem_frames': [row for row in rows if row['frame'] in (104, 148, 191)]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    inputs = [OLD, NEW, CROP, RAW, SOURCE, THIRD, CAMERAS]
    if any(out.is_relative_to(path) for path in inputs):
        raise ValueError('Use a new comparison outside immutable inputs')
    verification = {str(path): verify_artifacts(path) for path in inputs}
    out.mkdir(parents=True, exist_ok=False)
    for path in (Path(__file__), Path(__file__).with_name('check_jog_two_view.py')):
        shutil.copyfile(path, out/path.name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'inputs': verification, 'maximum_frames': 240,
        'scope': 'Development comparison after frozen construction; provider reference is reconstructed, not physical truth',
        'scoring': 'All2880 slots retained; paired scores separated from newly recovered estimates and missing slots',
        'body_change': 'New sampling density, image retries and recomputed body parameters; not a crop-only body ablation',
        'new_inference': False, 'new_training': False, 'paid_requests': 0})
    provider, camera = arrays(SOURCE/'body-coco17.npz'), read(CAMERAS/'camera-3.json')
    evidence = read(THIRD/'evidence.json')
    if provider['joints'].shape != (240, 17, 3) or evidence['source_sha256'] != camera['source_sha256']:
        raise ValueError('Comparison domain or image identity changed')
    third = {row['frame']: row for row in evidence['frames']}
    original, repaired = read(RAW/'construction/raw-predictions.json'), read(CROP/'triangulation.json')
    before, old = body_points(OLD); after, new = body_points(NEW)
    unchanged = {field: old[field] == new[field] for field in ('mobility', 'motor_limits', 'controller', 'body_frame', 'lengths')}
    if old['camera_reference']['ground'] != new['camera_reference']['ground'] or not unchanged['motor_limits'] or not unchanged['controller']:
        raise ValueError('Ground or control contract changed')
    refined_frames = [int(row['evidence_id'].rsplit(':', 1)[1]) for row in new['reference_evidence'] if row.get('image_refinement')]
    if refined_frames != read(CROP/'report.json')['accepted_frames']:
        raise ValueError('Native body lost its image re-inference provenance')
    report = {'created_at': local_time(), 'status': 'frozen_crop_body_compared',
        'image_geometry': compare(out, 'image-geometry', ray_points(original), ray_points(repaired), provider, third, camera),
        'native_body': compare(out, 'native-body', before, after, provider, third, camera),
        'original_body_poses': len(before)//12, 'new_body_poses': len(after)//12,
        'unchanged_body_fields': unchanged, 'old_lengths': old['lengths'], 'new_lengths': new['lengths'],
        'refined_body_frames': refined_frames,
        'ground_clearance': {'before': read(OLD/'ground-clearance.json')['summary'], 'after': read(NEW/'ground-clearance.json')['summary']},
        'floor_feasibility': {'before': read(OLD/'reference-feasibility.json')['summary'], 'after': read(NEW/'reference-feasibility.json')['summary']},
        'physical_validation': 'unverified', 'learned_policy_available': False, 'paid_requests': 0,
        'scope': 'Reconstructed-reference development agreement with all losses/regressions retained; not new video diversity or contact validation'}
    if any(verify_artifacts(Path(path)) != value for path, value in verification.items()):
        raise ValueError('A frozen input changed during comparison')
    write_json(out/'report.json', report)
    summary = {name: {key: report[name][key] for key in ('coverage', 'paired_errors_improved', 'paired_errors_worsened', 'paired_errors_unchanged')}
               for name in ('image_geometry', 'native_body')}
    summary['body_pair_rmse_m'] = [report['native_body']['metrics'][mode]['world_m']['rmse'] for mode in ('paired_before', 'paired_after')]
    summary['body_pair_image_rmse_px'] = [report['native_body']['metrics'][mode]['third_px']['rmse'] for mode in ('paired_before', 'paired_after')]
    print(json.dumps({'summary': summary, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
