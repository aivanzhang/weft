"""Compare fixed-length fitting under new noise, biased detections and wrong lengths."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.constrained_stereo import CONTRACT, fit_stereo_lengths
from weft.io import digest, local_time, write_json


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT/'.local/constrained-stereo-failure-01/constrained-stereo-debug-01.json'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compare-prepared', type=Path)
    args = parser.parse_args(); out = args.out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    source = json.loads(FIXTURE.read_text())['cases']
    if len(source) != 24:
        raise ValueError('Known-geometry control coverage changed')
    out.mkdir(exist_ok=False)
    for path in (Path(__file__), ROOT/'src/weft/constrained_stereo.py'):
        shutil.copyfile(path, out/path.name)
    rng = np.random.default_rng(9103); prepared = []
    for case in source:
        truth = np.array(case['truth']); cameras = case['cameras']; pixels = []
        for c in cameras:
            predicted = cv2.projectPoints(truth, cv2.Rodrigues(np.array(c['rotation']))[0], np.array(c['translation']),
                np.array(c['matrix']), np.array([c['radial'], 0., 0., 0., 0.]))[0].reshape(-1, 2)
            pixels.append(predicted)
        noisy = np.array(pixels)+rng.normal(0., 2., (2, 12, 2))
        for cohort in ('new_noise', 'one_hip_bias', 'five_percent_longer_prior'):
            observation = noisy.copy(); constraints = deepcopy(case['constraints'])
            if cohort == 'one_hip_bias':
                observation[1, 7, 0] += 15.
            if cohort == 'five_percent_longer_prior':
                for c in constraints:
                    c['length_m'] *= 1.05
            prepared.append({'index': case['index'], 'cohort': cohort, 'truth': truth.tolist(), 'cameras': cameras,
                'pixels': observation.tolist(), 'constraints': constraints})
    previous = None
    if args.compare_prepared:
        prior = args.compare_prepared.resolve(); path = prior/'prepared-controls.json'
        manifest = json.loads((prior/'manifest.json').read_text())
        if digest(path) != manifest['files'][path.name] or prepared != json.loads(path.read_text()):
            raise ValueError('Solver comparison changed control truths, observations or lengths')
        previous = {'path': str(prior.relative_to(ROOT)), 'manifest_sha256': digest(prior/'manifest.json')}
    write_json(out/'prepared-controls.json', prepared)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'contract': dict(CONTRACT),
        'fixture_sha256': digest(FIXTURE), 'prepared_controls_sha256': digest(out/'prepared-controls.json'),
        'matched_previous_inputs': previous,
        'source_module_sha256': digest(ROOT/'src/weft/constrained_stereo.py'),
        'cases': 72, 'poses_each_cohort': 24, 'noise_seed': 9103, 'image_noise_sigma_px': 2.,
        'bias': 'Cam2 right-hip horizontal observation +15 pixels; all other pixels match new-noise controls.',
        'incorrect_lengths': 'Every target length +5%; image observations identical to new-noise controls.',
        'hypothesis': 'Correct constraints can reduce noisy depth error, but biased image observations or length priors can still corrupt 3D proposals.',
        'unbiased_control_gate': 'All 24 new-noise cases satisfy the unchanged 1e-5 m length tolerance and aggregate 3D RMS improves by at least 5%.',
        'maximum_seconds': 30, 'maximum_output_bytes': 16000000, 'paid_calls': 0,
        'scope': 'Known synthetic geometry, not observed human anatomy, independent real-scene validation or ground/contact evidence.'})
    started = time.perf_counter(); rows = []
    with (out/'all-control-results.jsonl').open('x') as journal:
        for case in prepared:
            truth = np.array(case['truth']); tick = time.perf_counter()
            fit = fit_stereo_lengths(case['pixels'], case['cameras'], case['constraints'])
            errors = [np.linalg.norm(np.array(fit[key])-truth, axis=1) for key in ('initial_points_m', 'point_proposals_m')]
            row = {'index': case['index'], 'cohort': case['cohort'], 'fit': fit,
                'original_3d_errors_m': errors[0].tolist(), 'proposed_3d_errors_m': errors[1].tolist(),
                'runtime_seconds': time.perf_counter()-tick}
            journal.write(json.dumps(row, allow_nan=False)+'\n'); journal.flush(); rows.append(row)
            if time.perf_counter()-started > 30:
                raise TimeoutError('Control study exceeded its work envelope')
    groups = {}
    for cohort in ('new_noise', 'one_hip_bias', 'five_percent_longer_prior'):
        group = [r for r in rows if r['cohort'] == cohort]
        before, after = [np.array([r[key] for r in group]) for key in ('original_3d_errors_m', 'proposed_3d_errors_m')]
        groups[cohort] = {'cases': len(group), 'joint_slots': int(before.size),
            'original_3d_rms_m': float(np.sqrt(np.mean(before**2))), 'proposed_3d_rms_m': float(np.sqrt(np.mean(after**2))),
            'original_p95_3d_error_m': float(np.percentile(before, 95)), 'proposed_p95_3d_error_m': float(np.percentile(after, 95)),
            'joints_improved': int(np.sum(after < before)), 'joints_worsened': int(np.sum(after > before)),
            'constraints_satisfied': sum(r['fit']['length_constraints_satisfied'] for r in group),
            'solver_converged': sum(r['fit']['status'] == 'converged' for r in group),
            'all_joint_image_checks_pass': sum(all(r['fit']['joint_image_limits_satisfied']) for r in group)}
    control = groups['new_noise']
    passed = control['constraints_satisfied'] == 24 and control['proposed_3d_rms_m'] <= .95*control['original_3d_rms_m']
    if digest(ROOT/'src/weft/constrained_stereo.py') != digest(out/'constrained_stereo.py'):
        raise ValueError('Numerical implementation changed during controls')
    times = [r['runtime_seconds'] for r in rows]
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    report = {'created_at': local_time(), 'status': 'known_geometry_controls_complete', 'unbiased_control_passed': passed,
        'groups': groups, 'runtime_seconds': time.perf_counter()-started, 'output_bytes': size,
        'per_fit_seconds': {'median': float(np.median(times)), 'p95': float(np.percentile(times, 95)), 'maximum': max(times)},
        'paid_calls': 0, 'real_source_or_body_used': False, 'physical_validation': 'unverified',
        'limitations': ['Enforced constant lengths do not validate the chosen lengths.',
            'Aggregate synthetic improvement can coexist with individual regressions and does not establish real body or contact accuracy.']}
    write_json(out/'report.json', report)
    print(json.dumps({'control_passed': passed, 'groups': groups, 'runtime_seconds': report['runtime_seconds'],
        'verification': seal_suite(out)}))
    if not passed or size > 16000000:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
