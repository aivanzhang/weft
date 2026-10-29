"""Check saved fits and the frozen third-view comparison without repeating inference."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files


ROOT = Path(__file__).resolve().parents[2]
NAMES = ('stereo-length-controls-01', 'stereo-length-controls-02',
         'freemocap-fixed-length-01', 'freemocap-fixed-length-02',
         'freemocap-third-predictions-01', 'freemocap-third-board-predictions-01',
         'freemocap-third-comparison-01')


def read(path):
    return json.loads(path.read_text())


def project(points, camera):
    pixels, _ = cv2.projectPoints(np.asarray(points, dtype=float),
        cv2.Rodrigues(np.asarray(camera['rotation']))[0], np.asarray(camera['translation']),
        np.asarray(camera['matrix']), np.array([camera['radial'], 0., 0., 0., 0.]))
    return pixels.reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    if out.exists():
        raise ValueError('Verification outputs are append-only')
    seals = {}; files = 0
    for name in NAMES:
        root = ROOT/'artifacts'/name; manifest = read(root/'manifest.json')
        for filename, expected in manifest['files'].items():
            assert digest(root/filename) == expected, (name, filename)
        seals[name] = {'manifest_sha256': digest(root/'manifest.json'), 'files': len(manifest['files'])}
        files += len(manifest['files'])
    previous = ROOT/'artifacts/freemocap-fixed-length-01'
    current = ROOT/'artifacts/freemocap-fixed-length-02'
    for filename in ('prepared-observations.json', 'frozen-lengths.json', 'length-construction-slots.json'):
        assert read(previous/filename) == read(current/filename)
    cameras = read(current/'pre-run.json')['cameras']
    assert cameras == read(previous/'pre-run.json')['cameras']
    controls = read(ROOT/'artifacts/stereo-length-controls-02/prepared-controls.json')
    assert controls == read(ROOT/'artifacts/stereo-length-controls-01/prepared-controls.json')
    assert len(controls) == 72
    fits = [json.loads(line) for i in (1, 2)
            for line in (current/f'worker-{i}/frame-proposals.jsonl').read_text().splitlines()]
    assert sorted(r['frame'] for r in fits) == list(range(0, 1108, 2))
    prepared = {r['frame']: r for r in read(current/'prepared-observations.json')}
    maximum_reprojection_difference = 0.; constraints_checked = 0; proposed_joints = 0
    for row in fits:
        fit = row['fit']
        if fit is None:
            continue
        observation = prepared[row['frame']]; points = np.array(fit['point_proposals_m'])
        errors = np.linalg.norm(np.array([project(points, c) for c in cameras])-observation['pixels'], axis=2)
        maximum_reprojection_difference = max(maximum_reprojection_difference,
            float(np.max(np.abs(errors-fit['proposed_reprojection_px']))))
        lengths = np.array([np.linalg.norm(np.array(c['coefficients']) @ points)-c['length_m']
                           for c in observation['constraints']])
        assert np.allclose(lengths, fit['proposed_length_errors_m'], atol=1e-12, rtol=0)
        assert fit['length_constraints_satisfied'] == bool(np.max(np.abs(lengths)) <= 1e-5)
        assert fit['joint_image_limits_satisfied'] == np.all(errors <= 12.8, axis=0).tolist()
        constraints_checked += len(lengths); proposed_joints += len(points)
    assert maximum_reprojection_difference < 1e-8
    proposal = read(current/'complete-joint-proposals.json')
    assert len(proposal['frames']) == 1108
    assert all(r['frame'] == i and len(r['joint_proposals']) == 33 for i, r in enumerate(proposal['frames']))
    control_errors_checked = 0
    control_lookup = {(r['cohort'], r['index']): r for r in controls}
    assert len(control_lookup) == 72
    for line in (ROOT/'artifacts/stereo-length-controls-02/all-control-results.jsonl').read_text().splitlines():
        row = json.loads(line); case = control_lookup[row['cohort'], row['index']]
        truth = np.array(case['truth'])
        for field, output in [('initial_points_m', 'original_3d_errors_m'),
                              ('point_proposals_m', 'proposed_3d_errors_m')]:
            assert np.allclose(np.linalg.norm(np.array(row['fit'][field])-truth, axis=1), row[output], atol=1e-12, rtol=0)
            control_errors_checked += len(truth)
    frozen = ROOT/'artifacts/freemocap-third-predictions-01'
    protocol = read(frozen/'protocol.json'); predictions = read(frozen/'predictions-before-third-pixels.json')
    camera = protocol['camera'] | {'translation': protocol['camera']['translation_m']}
    projected = 0
    for rows in predictions.values():
        assert len(rows) == 13296 and len({(r['frame'], r['joint']) for r in rows}) == 13296
        active = [r for r in rows if r['pixel'] is not None]
        assert np.allclose(project([r['world_point_m'] for r in active], camera),
                           [r['pixel'] for r in active], atol=1e-8, rtol=0)
        projected += len(active)
    compared = ROOT/'artifacts/freemocap-third-comparison-01'
    observations = read(compared/'all-comparison-image-observations.json')
    report = read(compared/'report.json'); shared_checked = 0
    for cohort, summary in report['human'].items():
        selected = [i for i, r in enumerate(observations)
                    if cohort == 'complete_source' or 400 <= r['frame'] <= 1059]
        for name, stats in summary['paired_comparisons'].items():
            pairs = [(predictions['raw_two_view'][i]['pixel'], predictions[name][i]['pixel'], observations[i]['pixel'])
                     for i in selected if all(x is not None for x in
                        (predictions['raw_two_view'][i]['pixel'], predictions[name][i]['pixel'], observations[i]['pixel']))]
            values = np.array(pairs); a = np.linalg.norm(values[:, 0]-values[:, 2], axis=1)
            b = np.linalg.norm(values[:, 1]-values[:, 2], axis=1)
            assert len(pairs) == stats['shared_slots']
            for errors, key in ((a, 'baseline_on_shared'), (b, 'candidate_on_shared')):
                assert abs(float(np.sqrt(np.mean(errors**2)))-stats[key]['rmse_px']) < 1e-10
                assert abs(float(np.percentile(errors, 95))-stats[key]['p95_px']) < 1e-10
            assert int(np.sum(b < a-1e-9)) == stats['improved_slots']
            assert int(np.sum(b > a+1e-9)) == stats['worsened_slots']
            shared_checked += len(pairs)
    runtime = {name: digest(path) for name, path in _implementation_files().items()}
    assert runtime == protocol['runtime_hashes']
    tests = {}
    for filename, expected in (('constrained-stereo-tests-03.xml', 15), ('projected-joint-evaluation-tests-01.xml', 10)):
        test = ROOT/'.local'/filename; cases = ET.parse(test).findall('.//testcase')
        assert len(cases) == expected and all(not any(c.find(k) is not None for k in ('failure', 'error', 'skipped')) for c in cases)
        tests[filename] = {'passed': expected, 'sha256': digest(test), 'reused_unchanged_result': True}
    write_json(out, {'created_at': local_time(), 'status': 'verified_saved_comparison', 'seals': seals,
        'manifest_files_verified': files, 'source_slots': 1108, 'joint_proposal_slots': 36564,
        'proposed_joints_checked': proposed_joints, 'length_constraints_checked': constraints_checked,
        'maximum_opencv_reprojection_difference_px': maximum_reprojection_difference,
        'known_geometry_errors_checked': control_errors_checked, 'third_view_projections_checked': projected,
        'shared_cohort_pairs_checked': shared_checked, 'runtime_hashes': runtime, 'tests': tests,
        'third_camera_status': 'exposed_development_evidence', 'new_inference_calls': 0, 'paid_calls': 0,
        'body_built': False, 'candidate_promoted': False,
        'verifier_corrections': ['Read the saved truth key, not truth_m.',
            'Join controls by cohort and index; index repeats across cohorts. No fit, scorer or evidence changed.']})
    print(json.dumps({'verified_files': files, 'runtime_files': len(runtime), 'tests': sum(v['passed'] for v in tests.values()),
                      'maximum_reprojection_difference_px': maximum_reprojection_difference}))


if __name__ == '__main__':
    main()
