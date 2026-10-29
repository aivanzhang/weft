"""Build a root-only candidate before comparing its frozen geometry with saved references."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from weft.benchmark import seal_suite
from weft.camera_reference import JOINTS
from weft.io import canonical_hash, digest, local_time, write_json
from weft.multiview_root import shape_hash
from weft.pipeline import _implementation_files, verify_artifacts
from check_jog_two_view import CLOCK, arrays, statistics


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'artifacts/full-pose-ground-jog-lab-01/run'
SOURCE = ROOT/'artifacts/multiview-jog-source-01'
INPUT = ROOT/'artifacts/full-pose-body-build-01/input'
SECOND = ROOT/'artifacts/full-pose-comparison-01/jog-drone2/evidence.json'
CAMERA = ROOT/'artifacts/camera-jog-root-01/camera.json'
CAMERA2 = ROOT/'artifacts/multiview-jog-full-two-view-01/camera-2.json'
GROUND = ROOT/'artifacts/camera-ground-jog-plan-01/ground.json'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lab', type=Path, required=True)
    args = parser.parse_args()
    out, lab = args.out.resolve(), args.lab.resolve()
    bundles = [BASE, SOURCE, INPUT, SECOND.parent, CAMERA.parent, CAMERA2.parent, GROUND.parent]
    if any(path.is_relative_to(bundle) for path in (out, lab) for bundle in bundles):
        raise ValueError('Choose outputs outside sealed inputs')
    if out.is_relative_to(lab) or lab.is_relative_to(out) or lab.exists():
        raise ValueError('Study and lab must be separate new directories')
    verified = {str(path): verify_artifacts(path) for path in bundles}
    implementation = _implementation_files()
    hashes = {name: digest(path) for name, path in implementation.items()}
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    shutil.copyfile(Path(__file__).with_name('check_jog_two_view.py'), out/'check_jog_two_view.py')
    write_json(out/'pre-run.json', {'created_at': local_time(), 'inputs': verified,
        'hypothesis': 'Two-view hip midpoint reduces root position error without altering monocular limb shape',
        'scope': 'Development candidate after descriptive diagnosis; not a fresh final holdout',
        'unchanged': ['all original body poses and missing source slots', 'shape and angular trajectory',
                      'ground height', 'motor forces', 'penetration threshold and reference-gap limit'],
        'maximum_frames': 240, 'maximum_body_poses': 112, 'cli_timeout_seconds': 60,
        'new_inference': False, 'paid_requests': 0, 'projected_api_spend_usd': 0,
        'root_diagnosis_already_observed': True, 'reference_body_used_in_construction': False})
    config = out/'root-view.json'
    write_json(config, {'schema_version': 1, 'first_name': 'drone1', 'second_name': 'drone2',
        'second_evidence': os.path.relpath(SECOND, out), 'second_camera': os.path.relpath(CAMERA2, out),
        'clock_provenance': CLOCK,
        'world_provenance': 'Supplied Autelica reconstruction uses one z-up metre world and common subject-height scale. '
                            'Camera accuracy, metric scale and original-file synchronization are not independently measured.'})
    tick = time.monotonic()
    command = [sys.executable, '-m', 'weft', 'camera-body', '--run', str(INPUT), '--camera', str(CAMERA),
               '--ground', str(GROUND), '--root-view', str(config), '--out', str(lab)]
    with (out/'build.stdout').open('w') as stdout, (out/'build.stderr').open('w') as stderr:
        process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=60, check=False)
    wall = time.monotonic()-tick
    if process.returncode:
        write_json(out/'report.json', {'created_at': local_time(), 'status': 'construction_failed',
            'exit_code': process.returncode, 'wall_seconds': wall, 'paid_requests': 0})
        print(json.dumps({'verification': seal_suite(out), 'exit_code': process.returncode}, indent=2))
        raise SystemExit(process.returncode)
    run = lab/'run'
    frozen = verify_artifacts(run)
    write_json(out/'construction-verification.json', {'verification': frozen, 'wall_seconds': wall,
               'reference_body_not_opened_during_construction': True})
    baseline, candidate, retained = (read(path) for path in (BASE/'human.json', run/'human.json',
                                                           run/'inputs/monocular-camera-body.json'))
    if canonical_hash(baseline) != canonical_hash(retained) or shape_hash(candidate) != shape_hash(baseline):
        raise ValueError('Root-only body invariant failed')
    if digest(BASE/'camera-reference.json') != digest(run/'camera-reference.json'):
        raise ValueError('Raw monocular estimates changed')
    provider = arrays(SOURCE/'body-coco17.npz')
    joints, names = provider['joints'], provider['names'].tolist()
    if joints.shape != (240, 17, 3): raise ValueError('Comparison source domain changed')
    reference_roots = joints[:, [names.index('left_hip'), names.index('right_hip')]].mean(axis=1)
    candidates = {}
    for label, path, spec in [('monocular', BASE, baseline), ('two_view_root', run, candidate)]:
        snapshots = read(path/'human-reference.json')['frames']
        offset = np.array([*spec['camera_reference']['origin_world_m'][:2], spec['camera_reference']['ground']['height_m']])
        rows = {}
        for meta, root, snapshot in zip(spec['reference_evidence'], spec['root_reference'], snapshots, strict=True):
            frame = int(meta['evidence_id'].rsplit(':', 1)[1])
            if abs(snapshot['source_time_s']-frame/(30000/1001)) > 1e-7:
                raise ValueError('Body comparison source clock changed')
            rows[frame] = {'root': np.asarray(root['translation'])+spec['camera_reference']['origin_world_m'],
                          'body': {name: np.asarray(point)+offset for name, point in snapshot['human']['joints'].items()}}
        candidates[label] = rows
    if set(candidates['monocular']) != set(candidates['two_view_root']): raise ValueError('Body availability changed')
    scores = {label: {key: [] for key in ('root_distance_m', 'root_height_error_m', 'body_distance_m', 'root_relative_body_distance_m')}
              for label in candidates}
    rows, translations = [], []
    for frame in range(240):
        available = frame in candidates['monocular']
        if available:
            a, b = (candidates[label][frame] for label in candidates)
            delta = b['root']-a['root']
            for name in a['body']:
                translations.append(float(np.linalg.norm(b['body'][name]-a['body'][name]-delta)))
            for label, frames in candidates.items():
                scores[label]['root_distance_m'].append(float(np.linalg.norm(frames[frame]['root']-reference_roots[frame])))
                scores[label]['root_height_error_m'].append(float(frames[frame]['root'][2]-reference_roots[frame, 2]))
        for name in JOINTS:
            row = {'frame': frame, 'joint': name, 'paired_available': available, 'errors_m': None}
            if available:
                errors = {}
                for label, frames in candidates.items():
                    body, root = frames[frame]['body'][name], frames[frame]['root']
                    ref = joints[frame, names.index(name)]
                    absolute = float(np.linalg.norm(body-ref))
                    relative = float(np.linalg.norm(body-root-(ref-reference_roots[frame])))
                    scores[label]['body_distance_m'].append(absolute)
                    scores[label]['root_relative_body_distance_m'].append(relative)
                    errors[label] = {'absolute': absolute, 'root_relative': relative}
                row['errors_m'] = errors
            rows.append(row)
    write_json(out/'comparison.json', {'source_frames': 240, 'joint_slots': len(rows), 'rows': rows})
    if max(translations) > 1e-10: raise ValueError('Native body changed beyond the intended root translation')
    report = {'created_at': local_time(), 'status': 'root_candidate_compared', 'lab': str(lab),
        'construction': frozen, 'construction_wall_seconds': wall,
        'source_frames': 240, 'joint_slots': 2880, 'paired_joints': len(scores['monocular']['body_distance_m']),
        'coverage': read(run/'two-view-root.json')['coverage'],
        'comparison': {label: {key: statistics(values) for key, values in metrics.items()} for label, metrics in scores.items()},
        'maximum_native_translation_invariant_error_m': max(translations),
        'ground_clearance': {label: read(path/'ground-clearance.json')['summary']
                             for label, path in [('monocular', BASE), ('two_view_root', run)]},
        'floor_feasibility': {label: read(path/'reference-feasibility.json')['summary']
                              for label, path in [('monocular', BASE), ('two_view_root', run)]},
        'raw_monocular_estimates_unchanged': True, 'body_shape_unchanged': True, 'accepted': False,
        'physical_validation': 'unverified', 'learned_policy_available': False, 'paid_requests': 0,
        'scope': 'Reconstructed-reference development agreement, not measured anatomy, contact or physical accuracy'}
    if any(verify_artifacts(path) != verification for path, verification in verified.items()):
        raise RuntimeError('An original bundle changed')
    if verify_artifacts(run) != frozen or _implementation_files() != implementation or any(
            digest(path) != hashes[name] for name, path in implementation.items()):
        raise RuntimeError('Candidate or implementation changed after construction')
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
