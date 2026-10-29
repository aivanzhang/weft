"""Source-only ground orientation, then unchanged full-reference floor checks."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import mujoco

from weft.benchmark import seal_suite
from weft.ground_orientation import GROUND_ORIENTATION_CONTRACT, estimate_ground_orientation
from weft.human import validate_human
from weft.io import canonical_hash, digest, local_time, write_json
from weft.mobile_human import calibrate_floor
from weft.pipeline import verify_artifacts
from weft.whole_body_feasibility import audit_reference_feasibility


CASES = {name: Path('artifacts/mobile-exercise-e2e-10') / name / 'run'
         for name in ('single-leg-squat', 'squat', 'kettlebell-pushup')}
CASES['fullbody-squat'] = Path('artifacts/fullbody-exercise-01/baseline/run')


def read(path):
    return json.loads(path.read_text())


def apply_orientation(spec, estimate):
    if estimate['status'] != 'available' or estimate['normal'] is None:
        raise ValueError('Only source-screened orientations produce candidates')
    if estimate['source_sha256'] != spec['source_sha256']:
        raise ValueError('Orientation source must match the body model')
    result = deepcopy(spec)
    result['mobility'].pop('floor_fit', None)
    result['mobility']['ground_plane'] = {
        'schema_version': 1, 'normal': estimate['normal'],
        'method': 'camera_relative_support_plane_v1', 'source_sha256': spec['source_sha256'],
        'evidence_ids': estimate['construction_evidence_ids'],
        'provenance': 'Source-foot orientation hypothesis from centered pose-estimated heel/toe quads. '
                      'No measured ground contact, camera calibration, sole geometry or gravity. '
                      'Offset is separately calibrated from the first body pose only.',
        'gravity_assumption': 'authored_normal_alignment_not_measured'}
    result['assumptions'] = [text.replace('Camera up is assumed gravity-up',
        'Gravity is authored along the negative inferred ground normal') for text in result['assumptions']]
    result['assumptions'].append('Ground orientation uses construction pose estimates; only its offset '
                                'uses the first body pose. Feet need not be planted; this is not ground truth.')
    return validate_human(calibrate_floor(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = [Path('src/weft') / name for name in (
        'ground_orientation.py', 'mobile_human.py', 'human.py', 'spatial_pose.py', 'body_frame.py',
        'root_motion.py', 'whole_body_feasibility.py', 'whole_body.py', 'recording.py', 'io.py')]
    paths += [Path('tests/test_ground_orientation.py'), Path('tests/test_ground_plane.py'), Path(__file__)]
    hashes = {str(path): digest(path) for path in paths}
    implementation = args.out / 'implementation'
    implementation.mkdir()
    for path in paths:
        shutil.copyfile(path, implementation / path.name)
    inputs = {}
    for name, directory in CASES.items():
        verified = verify_artifacts(directory)
        spec = read(directory / 'human.json')
        media = directory / ('source.mp4' if name == 'fullbody-squat' else 'source.webm')
        if digest(media) != spec['source_sha256']:
            raise ValueError('Source media identity mismatch')
        inputs[name] = {'directory': str(directory), 'verification': verified,
                       'source_sha256': spec['source_sha256'],
                       'pose_sha256': digest(directory / 'pose-landmarks.json'),
                       'model_sha256': canonical_hash(spec)}
    write_json(args.out / 'pre-candidate-freeze.json', {
        'created_at': local_time(), 'implementation_hashes': hashes, 'inputs': inputs,
        'estimator_contract': dict(GROUND_ORIENTATION_CONTRACT), 'maximum_candidates': 4,
        'maximum_seconds': 120, 'api_usd': 0, 'physics_steps': 0,
        'chronology': 'Parent specified screening thresholds after the source agent had computed an '
                      'earlier unnormalized fullbody diagnostic. Not an unseen-source threshold design. '
                      'No thresholds selected against floor grades; no repeated candidate tuning.',
        'candidate_rule': 'Only available source estimates; no fallback normals or per-pose lifting.',
        'offset_rule': 'First pose only; later poses do not affect floor offset.',
        'promotion': 'No automatic default change, training or physical accuracy claim.'})

    def forbidden(*args, **kwargs):
        raise AssertionError('Ground hypothesis study cannot advance physics')

    mujoco.mj_step = forbidden
    rows = []
    for name, directory in CASES.items():
        if time.perf_counter() - started > 120:
            raise TimeoutError('Ground hypothesis study deadline')
        out = args.out / name
        out.mkdir()
        spec = read(directory / 'human.json')
        raw = read(directory / 'pose-landmarks.json')
        if any(not frame['id'].startswith(spec['source_sha256'] + ':frame:') for frame in raw['frames']):
            raise ValueError('Pose source identity mismatch')
        for original, target in [('human.json', 'baseline-human.json'), ('pose-landmarks.json', 'pose-landmarks.json')]:
            shutil.copyfile(directory / original, out / target)
        estimate = estimate_ground_orientation({**raw, 'source_sha256': spec['source_sha256']})
        write_json(out / 'orientation.json', estimate)
        row = {'case': name, 'orientation_status': estimate['status'],
               'orientation_reasons': estimate['reasons'], 'candidate_created': False}
        if estimate['status'] == 'available':
            candidate = apply_orientation(spec, estimate)
            fixed = {key: candidate.get(key) == spec.get(key) for key in (
                'reference', 'root_reference', 'reference_evidence', 'lengths', 'body_frame', 'motor_limits', 'dt')}
            if not all(fixed.values()):
                raise AssertionError('Ground hypothesis changed body/reference fields')
            write_json(out / 'candidate-human.json', candidate)
            baseline = audit_reference_feasibility(spec)
            comparison = audit_reference_feasibility(candidate)
            write_json(out / 'baseline-feasibility.json', baseline)
            write_json(out / 'candidate-feasibility.json', comparison)
            row.update(candidate_created=True, fixed_fields=fixed, normal=estimate['normal'],
                       baseline_feasibility=baseline['summary'], candidate_feasibility=comparison['summary'],
                       candidate_status=comparison['status'], original_base_height=spec['mobility']['base_height'],
                       candidate_base_height=candidate['mobility']['base_height'])
        rows.append(row)
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise RuntimeError('Study implementation changed during execution')
    for name, directory in CASES.items():
        if verify_artifacts(directory) != inputs[name]['verification']:
            raise RuntimeError('Original run changed during study')
    report = {'created_at': local_time(), 'cases': rows, 'seconds': time.perf_counter() - started,
              'api_usd': 0, 'physics_steps': 0, 'promoted': False,
              'scope': 'Construction-only inferred orientation and sampled model consistency. '
                       'Not measured ground, anatomical feet, physical balance, or a final holdout.'}
    write_json(args.out / 'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(args.out)}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
