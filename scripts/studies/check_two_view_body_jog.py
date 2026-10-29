"""Separate length and direction effects before evaluating the combined partial body."""

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from weft.pipeline import _implementation_files, verify_artifacts
from weft.spatial_pose import LIMB_ENDPOINTS
from check_jog_two_view import arrays, image_point, projected, statistics


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'artifacts/two-view-root-jog-lab-01/run'
SOURCE = ROOT/'artifacts/multiview-jog-source-01'
INPUT = ROOT/'artifacts/full-pose-body-build-01/input'
CONFIG = ROOT/'artifacts/two-view-root-jog-study-01/root-view.json'
CAMERA = ROOT/'artifacts/camera-jog-root-01/camera.json'
GROUND = ROOT/'artifacts/camera-ground-jog-plan-01/ground.json'
THIRD_POSE = ROOT/'artifacts/multiview-jog-drone3-image-01'
THIRD_CAMERA = ROOT/'artifacts/multiview-jog-full-two-view-01/camera-3.json'
MODES = ('directions', 'dimensions', 'both')


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lab-prefix', type=Path, required=True)
    parser.add_argument('--partial-segments', action='store_true', help='Compare v2 proximal-only support with frozen v1 candidates')
    args = parser.parse_args()
    out, prefix = args.out.resolve(), args.lab_prefix.resolve()
    modes = ('directions', 'both') if args.partial_segments else MODES
    labs = {mode: Path(str(prefix)+'-'+mode) for mode in modes}
    predecessors = ({mode: ROOT/f'artifacts/two-view-body-jog-lab-01-{mode}/run' for mode in modes}
                    if args.partial_segments else {})
    bundles = [BASE, SOURCE, INPUT, CONFIG.parent, CAMERA.parent, GROUND.parent, THIRD_POSE, THIRD_CAMERA.parent]
    bundles.extend(predecessors.values())
    if any(path.is_relative_to(bundle) for path in (out, *labs.values()) for bundle in bundles):
        raise ValueError('Use new outputs outside sealed inputs')
    if any(out.is_relative_to(path) or path.is_relative_to(out) or path.exists() for path in labs.values()):
        raise ValueError('Candidate labs must be separate new directories')
    verified = {str(path): verify_artifacts(path) for path in bundles}
    implementation = _implementation_files()
    hashes = {name: digest(path) for name, path in implementation.items()}
    out.mkdir(parents=True, exist_ok=False)
    for path in (Path(__file__), Path(__file__).with_name('check_jog_two_view.py')):
        shutil.copyfile(path, out/path.name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'inputs': verified, 'modes': list(modes),
        'partial_segments': args.partial_segments,
        'hypothesis': ('Using supported proximal directions improves missing-endpoint limbs without changing observed complete limbs'
                       if args.partial_segments else 'Two-view supported lengths and directions improve limb agreement beyond root correction'),
        'scope': 'Development comparison; prior coverage/length diagnosis already observed, not a final holdout',
        'unchanged': ['root trajectory', 'ground', 'source body-pose availability', 'motor limits and support gains',
                      'independent penetration and maximum reference-gap limits'],
        'dynamics_change': 'Authored mass/inertia change with dimensions at fixed density; not measured forces',
        'comparison_inputs_excluded_from_construction': ['provider body joints', 'third camera image joints'],
        'maximum_frames': 240, 'body_poses': 112, 'workers': 2, 'cli_timeout_seconds_each': 60,
        'new_inference': False, 'new_training': False, 'paid_requests': 0, 'projected_api_spend_usd': 0})

    def build(mode):
        command = [sys.executable, '-m', 'weft', 'camera-body', '--run', str(INPUT), '--camera', str(CAMERA),
                   '--ground', str(GROUND), '--root-view', str(CONFIG), '--body-view', mode, '--out', str(labs[mode])]
        if args.partial_segments: command.append('--partial-segments')
        started = time.monotonic()
        try:
            with (out/(mode+'.stdout')).open('w') as stdout, (out/(mode+'.stderr')).open('w') as stderr:
                result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=60, check=False)
            return {'exit_code': result.returncode, 'wall_seconds': time.monotonic()-started,
                    'verification': verify_artifacts(labs[mode]/'run') if result.returncode == 0 else None}
        except subprocess.TimeoutExpired:
            return {'exit_code': None, 'timeout': True, 'wall_seconds': time.monotonic()-started, 'verification': None}

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        builds = dict(zip(modes, pool.map(build, modes), strict=True))
    build_seconds = time.monotonic()-started
    write_json(out/'construction-verification.json', {'candidates': builds, 'wall_seconds': build_seconds,
               'comparison_references_not_opened_during_construction': True})
    if any(row['exit_code'] != 0 for row in builds.values()):
        write_json(out/'report.json', {'status': 'construction_failed', 'builds': builds, 'paid_requests': 0})
        print(json.dumps({'verification': seal_suite(out), 'builds': builds}, indent=2))
        raise SystemExit(1)

    paths = {'root_only': BASE, **{'v1_'+mode: path for mode, path in predecessors.items()},
             **{mode: path/'run' for mode, path in labs.items()}}
    base = read(BASE/'human.json')
    candidates = {}
    for mode, path in paths.items():
        spec = read(path/'human.json')
        if mode != 'root_only':
            if canonical_hash(spec['camera_reference']['body_refinement']['input_body']) != canonical_hash(base):
                raise ValueError('Candidates do not share the exact original root-only model')
            for field in ('root_reference', 'mobility', 'body_frame', 'motor_limits'):
                if spec[field] != base[field]: raise ValueError('Fixed field changed: '+field)
        snapshots = read(path/'human-reference.json')['frames']
        offset = np.array([*spec['camera_reference']['origin_world_m'][:2], spec['camera_reference']['ground']['height_m']])
        rows = {}
        for meta, root, snapshot in zip(spec['reference_evidence'], spec['root_reference'], snapshots, strict=True):
            frame = int(meta['evidence_id'].rsplit(':', 1)[1])
            if abs(snapshot['source_time_s']-frame/(30000/1001)) > 1e-7:
                raise ValueError('Body source clock changed')
            rows[frame] = {'root': np.asarray(root['translation'])+spec['camera_reference']['origin_world_m'],
                          'body': {name: np.asarray(point)+offset for name, point in snapshot['human']['joints'].items()}}
        candidates[mode] = rows
    if any(set(value) != set(candidates['root_only']) for value in candidates.values()):
        raise ValueError('A candidate changed body-pose availability')
    invariant_reports = {}
    if args.partial_segments:
        for mode in modes:
            new, old = read(paths[mode]/'human.json'), read(predecessors[mode]/'human.json')
            audit = read(paths[mode]/'two-view-body.json')
            if audit['partial_segments']['complete_limb_reference_sha256'] != canonical_hash(old['reference']):
                raise ValueError('Complete-limb base changed before segment correction')
            if any(new[key] != old[key] for key in ('lengths', 'root_reference', 'mobility', 'body_frame', 'motor_limits')):
                raise ValueError('Partial segments changed geometry outside their direction scope')
            rows = {row['frame']: row for row in audit['frames'] if row['retained_body_pose']}
            maxima = {'untargeted_native_point_difference_m': 0., 'proximal_unit_direction_error': 0.}
            group_count = 0
            for index, meta in enumerate(new['reference_evidence']):
                frame = int(meta['evidence_id'].rsplit(':', 1)[1])
                allowed_angles, allowed_points = set(), set()
                for group, detail in rows[frame]['groups'].items():
                    if not detail.get('segment_transport_applied'): continue
                    group_count += 1
                    allowed_angles.update((group, group+'_twist', group+'_lateral'))
                    base_joint, middle, tip = [next(name for name, value in JOINTS.items() if value == j) for j in LIMB_ENDPOINTS[group]]
                    allowed_points.update((middle, tip))
                    native = candidates[mode][frame]['body'][middle]-candidates[mode][frame]['body'][base_joint]
                    target = np.subtract(*detail['proximal_world_points_m'][::-1])
                    error = float(np.linalg.norm(native/np.linalg.norm(native)-target/np.linalg.norm(target)))
                    maxima['proximal_unit_direction_error'] = max(maxima['proximal_unit_direction_error'], error)
                for name, angle in old['reference'][index]['joints'].items():
                    if name not in allowed_angles and new['reference'][index]['joints'][name] != angle:
                        raise ValueError('Untargeted angle or distal bend changed')
                for name, point in candidates['v1_'+mode][frame]['body'].items():
                    if name not in allowed_points:
                        error = float(np.linalg.norm(candidates[mode][frame]['body'][name]-point))
                        maxima['untargeted_native_point_difference_m'] = max(maxima['untargeted_native_point_difference_m'], error)
            if any(error > 1e-10 for error in maxima.values()):
                raise ValueError('Partial native geometry invariant failed')
            invariant_reports[mode] = {'applied_group_poses': group_count, **maxima,
                                      'complete_limb_base_reference_identical': True, 'distal_bends_unchanged': True}
        write_json(out/'segment-invariants.json', invariant_reports)
    provider = arrays(SOURCE/'body-coco17.npz')
    joints, names = provider['joints'], provider['names'].tolist()
    if joints.shape != (240, 17, 3): raise ValueError('Comparison domain changed')
    reference_roots = joints[:, [names.index('left_hip'), names.index('right_hip')]].mean(axis=1)
    third, camera3 = read(THIRD_POSE/'evidence.json'), read(THIRD_CAMERA)
    if third['source_sha256'] != camera3['source_sha256']: raise ValueError('Third-view identity mismatch')
    image3 = {row['frame']: row for row in third['frames']}
    scores = {mode: {key: [] for key in ('body_distance_m', 'root_relative_body_distance_m', 'third_camera_error_px',
                                        'ankle_height_error_m')} for mode in paths}
    per_joint = {mode: {name: [] for name in JOINTS} for mode in paths}
    rows, image_coverage = [], {'compared': 0, 'unavailable_image_or_body': 0, 'nonforward_candidate': 0}
    for frame in range(240):
        available = frame in candidates['root_only']
        for name, index in JOINTS.items():
            observation = image_point(image3.get(frame), index, camera3['width'], camera3['height'])
            projection = ({mode: projected(value[frame]['body'][name], camera3, frame) for mode, value in candidates.items()}
                          if available else {})
            compare_image = available and observation is not None and all(p is not None for p in projection.values())
            image_coverage['compared' if compare_image else 'nonforward_candidate' if available and observation is not None
                           else 'unavailable_image_or_body'] += 1
            row = {'frame': frame, 'joint': name, 'body_available': available, 'image_comparable': compare_image,
                   'errors': None}
            if available:
                errors = {}
                for mode, value in candidates.items():
                    body, root = value[frame]['body'][name], value[frame]['root']
                    ref = joints[frame, names.index(name)]
                    distance = float(np.linalg.norm(body-ref))
                    relative = float(np.linalg.norm(body-root-(ref-reference_roots[frame])))
                    image_error = float(np.linalg.norm(projection[mode]-observation)) if compare_image else None
                    scores[mode]['body_distance_m'].append(distance)
                    scores[mode]['root_relative_body_distance_m'].append(relative)
                    per_joint[mode][name].append(distance)
                    if image_error is not None: scores[mode]['third_camera_error_px'].append(image_error)
                    if name.endswith('ankle'): scores[mode]['ankle_height_error_m'].append(float(body[2]-ref[2]))
                    errors[mode] = {'absolute_m': distance, 'root_relative_m': relative, 'third_camera_px': image_error}
                row['errors'] = errors
            rows.append(row)
    write_json(out/'comparison.json', {'source_frames': 240, 'joint_slots': 2880, 'rows': rows})
    report = {'created_at': local_time(), 'status': 'partial_body_candidates_compared', 'builds': builds,
        'construction_wall_seconds': build_seconds, 'labs': {mode: str(path) for mode, path in labs.items()},
        'source_frames': 240, 'joint_slots': 2880, 'available_body_joints': len(scores['root_only']['body_distance_m']),
        'third_camera_coverage': image_coverage,
        'comparison': {mode: {key: statistics(values) for key, values in metrics.items()} for mode, metrics in scores.items()},
        'per_joint': {mode: {name: statistics(values) for name, values in group.items()} for mode, group in per_joint.items()},
        'ground_clearance': {mode: read(path/'ground-clearance.json')['summary'] for mode, path in paths.items()},
        'floor_feasibility': {mode: read(path/'reference-feasibility.json')['summary'] for mode, path in paths.items()},
        'lengths': {mode: read(path/'human.json')['lengths'] for mode, path in paths.items()},
        'group_coverage': read(paths['both']/'two-view-body.json')['group_coverage'],
        'accepted': False, 'physical_validation': 'unverified', 'learned_policy_available': False, 'paid_requests': 0,
        'scope': 'Development agreement with reconstructed reference and third-camera image estimates; not measured anatomy or contact'}
    if args.partial_segments:
        report['partial_segments'] = invariant_reports
        report['candidate_methods'] = {mode: read(path/'human.json')['camera_reference']['method'] for mode, path in paths.items()}
    if any(verify_artifacts(path) != verification for path, verification in verified.items()):
        raise RuntimeError('Original bundle changed')
    if any(verify_artifacts(labs[mode]/'run') != builds[mode]['verification'] for mode in modes):
        raise RuntimeError('Candidate changed after construction')
    if _implementation_files() != implementation or any(digest(path) != hashes[name] for name, path in implementation.items()):
        raise RuntimeError('Implementation changed during construction')
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
