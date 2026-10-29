"""Freeze image-only correspondence before opening reconstructed comparison joints."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from weft.benchmark import seal_suite
from weft.camera_reference import JOINTS
from weft.io import canonical_hash, digest, local_time, write_json
from weft.limb_correspondence import CONTRACT, triangulate_with_limb_correspondence
from weft.pipeline import _implementation_files, verify_artifacts
from check_jog_two_view import arrays, image_point, projected, statistics


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/two-view-segment-jog-lab-01-both/run'
POSES = [ROOT / f'artifacts/full-pose-comparison-01/jog-drone{i}' for i in (1, 2)]
CAMERAS = [ROOT / 'artifacts/camera-jog-root-01/camera.json',
           ROOT / 'artifacts/multiview-jog-full-two-view-01/camera-2.json']
REFERENCE = ROOT / 'artifacts/multiview-jog-source-01'
THIRD_POSE = ROOT / 'artifacts/multiview-jog-drone3-image-01'
THIRD_CAMERA = ROOT / 'artifacts/multiview-jog-full-two-view-01/camera-3.json'


def read(path):
    return json.loads(path.read_text())


def construct(out):
    inputs = out / 'inputs'
    frozen = verify_artifacts(inputs)
    views = [read(inputs / f'view-{i}.json') for i in (1, 2)]
    started = time.perf_counter()
    predictions, audit, baseline = triangulate_with_limb_correspondence(*views, return_baseline=True)
    elapsed = time.perf_counter() - started
    expected = read(inputs / 'report.json')['raw_predictions_sha256']
    if expected is not None and audit['raw_predictions_sha256'] != expected:
        raise ValueError('Original camera/image reconstruction differs from the frozen baseline')
    if verify_artifacts(inputs) != frozen:
        raise ValueError('Construction inputs changed')
    target = out / 'construction'
    target.mkdir()
    write_json(target / 'raw-predictions.json', baseline)
    write_json(target / 'predictions.json', predictions)
    write_json(target / 'correspondence-audit.json', audit)
    write_json(target / 'report.json', {'created_at': local_time(), 'status': 'construction_frozen',
        'input_verification': frozen, 'wall_seconds': elapsed,
        'changed_pairs': audit['changed_pairs'], 'decisions': dict(Counter(row['status'] for row in audit['decisions'])),
        'window_rejection_counts': audit['window_rejection_counts'], 'source_frames': predictions['source_frames'],
        'joint_slots': predictions['joint_slots'], 'anatomical_side_verified': False,
        'comparison_references_not_opened': True, 'paid_requests': 0, 'new_inference': False})
    seal_suite(target)


def compare(out, baseline, candidate):
    reference = arrays(REFERENCE / 'body-coco17.npz')
    third, camera = read(THIRD_POSE / 'evidence.json'), read(THIRD_CAMERA)
    if third['source_sha256'] != camera['source_sha256']:
        raise ValueError('Third-view source identity differs from calibration')
    poses = {row['frame']: row for row in third['frames']}
    names, rows = reference['names'].tolist(), []
    scores = {mode: {'world_m': [], 'third_px': [], 'changed_world_m': [], 'changed_third_px': []}
              for mode in ('raw', 'candidate')}
    for f in range(candidate['source_frames']):
        for name, index in JOINTS.items():
            before, after = (value['frames'][f]['joints'][index] for value in (baseline, candidate))
            changed = before != after
            row = {'frame': f, 'joint': name, 'changed': changed,
                   'raw_status': before['status'], 'candidate_status': after['status'], 'errors': None}
            if before['status'] == after['status'] == 'estimated':
                expected = reference['joints'][f, names.index(name)]
                uv = image_point(poses.get(f), index, camera['width'], camera['height'])
                errors = {}
                for mode, joint in (('raw', before), ('candidate', after)):
                    point = np.asarray(joint['world_point_m'])
                    world = float(np.linalg.norm(point-expected))
                    image = projected(point, camera, f) if uv is not None else None
                    pixel = float(np.linalg.norm(image-uv)) if image is not None else None
                    errors[mode] = {'world_m': world, 'third_px': pixel}
                    scores[mode]['world_m'].append(world)
                    if pixel is not None: scores[mode]['third_px'].append(pixel)
                    if changed:
                        scores[mode]['changed_world_m'].append(world)
                        if pixel is not None: scores[mode]['changed_third_px'].append(pixel)
                row['errors'] = errors
            rows.append(row)
    deltas = [{'frame': row['frame'], 'joint': row['joint'],
               'world_m': row['errors']['candidate']['world_m']-row['errors']['raw']['world_m'],
               'third_px': (row['errors']['candidate']['third_px']-row['errors']['raw']['third_px']
                            if row['errors']['raw']['third_px'] is not None and row['errors']['candidate']['third_px'] is not None else None)}
              for row in rows if row['changed'] and row['errors']]
    write_json(out / 'comparison.json', {'rows': rows, 'changed_deltas': deltas})
    return {'total_slots': len(rows), 'paired_slots': sum(row['errors'] is not None for row in rows),
        'changed_joint_slots': len(deltas), 'changed_world_improved': sum(row['world_m'] < -1e-10 for row in deltas),
        'changed_world_worsened': sum(row['world_m'] > 1e-10 for row in deltas),
        'changed_third_improved': sum(row['third_px'] is not None and row['third_px'] < -1e-10 for row in deltas),
        'changed_third_worsened': sum(row['third_px'] is not None and row['third_px'] > 1e-10 for row in deltas),
        'metrics': {mode: {name: statistics(value) for name, value in values.items()} for mode, values in scores.items()},
        'largest_world_regression': max(deltas, key=lambda row: row['world_m']) if deltas else None,
        'known_diagnostic_frames': [row for row in rows if row['frame'] in (104, 148)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--first-pose-run', type=Path, default=POSES[0])
    parser.add_argument('--second-pose-run', type=Path, default=POSES[1])
    parser.add_argument('--construct', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.construct:
        construct(out)
        return
    poses = [args.first_pose_run.resolve(), args.second_pose_run.resolve()]
    paths = [BASE, *poses, *(p.parent for p in CAMERAS), REFERENCE, THIRD_POSE]
    if any(out.is_relative_to(path) for path in paths):
        raise ValueError('Output must not change a sealed input')
    verification = {str(path): verify_artifacts(path) for path in paths}
    baseline = read(BASE / 'two-view-triangulation.json')
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    out.mkdir(parents=True, exist_ok=False)
    for path in (Path(__file__), Path(__file__).with_name('check_jog_two_view.py')):
        shutil.copyfile(path, out / path.name)
    for name, path in _implementation_files().items():
        target = out / 'implementation-source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    inputs = out / 'inputs'
    inputs.mkdir()
    for i in range(2):
        evidence, camera, prior = read(poses[i] / 'evidence.json'), read(CAMERAS[i]), baseline['views'][i]
        if evidence['source_sha256'] != prior['source_sha256'] or canonical_hash(camera) != prior['camera_sha256']:
            raise ValueError('Construction videos/cameras do not match the frozen original')
        write_json(inputs / f'view-{i+1}.json', {'name': prior['name'], 'clock_provenance': prior['clock_provenance'],
            'evidence': evidence, 'camera': camera})
    write_json(inputs / 'report.json', {'status': 'construction_inputs_frozen', 'created_at': local_time(),
        'contract': dict(CONTRACT), 'raw_predictions_sha256': canonical_hash(baseline) if poses == POSES else None,
        'excluded': ['provider body joints', 'drone3 poses/camera', 'comparison labels', 'floor/contact fitting'],
        'world_landmarks': 'Retained for source identity but ignored by the algorithm; poisoning invariance tested'})
    seal_suite(inputs)
    write_json(out / 'pre-run.json', {'created_at': local_time(), 'inputs': verification, 'implementation': implementation,
        'hypothesis': 'Short limb-label islands can be corrected from original parent-relative continuity and two image rays',
        'maximum_wall_seconds': 60, 'maximum_source_frames': 240, 'maximum_sampled_poses_per_view': 240,
        'pose_runs': [str(path) for path in poses],
        'scope': 'Development comparison on an already inspected source, not a fresh holdout or independently measured anatomy',
        'selection': 'One fixed analytic-tested contract; no reference-based thresholds, offset search or candidate selection',
        'paid_requests': 0, 'new_training': False, 'body_or_ground_changed': False})
    with (out / 'construction.stdout').open('w') as stdout, (out / 'construction.stderr').open('w') as stderr:
        execution = subprocess.run([sys.executable, str(out / Path(__file__).name), '--out', str(out), '--construct'],
            cwd=ROOT, stdout=stdout, stderr=stderr, timeout=60, check=False)
    if execution.returncode:
        write_json(out / 'report.json', {'status': 'construction_failed', 'exit_code': execution.returncode,
                                       'paid_requests': 0, 'comparison_performed': False})
        print(json.dumps(seal_suite(out)))
        raise SystemExit(1)
    built = verify_artifacts(out / 'construction')
    candidate = read(out / 'construction/predictions.json')
    audit = read(out / 'construction/correspondence-audit.json')
    raw = read(out / 'construction/raw-predictions.json')
    results = compare(out, raw, candidate)
    retained = sum(raw['frames'][row['frame']]['joints'] == row['joints'] for row in baseline['frames']
                   if row['source_evidence_ids'][0] is not None)
    if any(verify_artifacts(Path(path)) != identity for path, identity in verification.items()):
        raise ValueError('An original input changed')
    if verify_artifacts(out / 'construction') != built:
        raise ValueError('Frozen construction changed during comparison')
    if {name: digest(path) for name, path in _implementation_files().items()} != implementation:
        raise ValueError('Implementation changed during the study')
    write_json(out / 'report.json', {'created_at': local_time(), 'status': 'completed_development_comparison',
        'construction': built, 'changed_pairs': audit['changed_pairs'], 'comparison': results,
        'legacy_sampled_frames_with_identical_raw_triangulation': retained,
        'physical_validation': 'unverified', 'body_or_ground_changed': False,
        'paid_requests': 0, 'all_inputs_implementation_and_frozen_construction_unchanged': True})
    print(json.dumps({'changed_pairs': audit['changed_pairs'], 'metrics': results['metrics'],
        'changed_world_improved': results['changed_world_improved'], 'changed_world_worsened': results['changed_world_worsened'],
        'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
