"""Audit cached multi-ray estimates and grade the two-view-only condition."""

import argparse
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files
from weft.projected_joint_evaluation import grade_projected_joints


ROOT = Path(__file__).resolve().parents[2]
MAJOR = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


def read(path):
    return json.loads(path.read_text())


def verify(root):
    manifest = read(root/'manifest.json')
    for name, expected in manifest['files'].items():
        assert digest(root/name) == expected, name
    return {'files': len(manifest['files']), 'sha256': digest(root/'manifest.json')}


def project(points, c):
    return cv2.projectPoints(np.array(points, dtype=float), cv2.Rodrigues(np.array(c['rotation']))[0],
        np.array(c['translation']), np.array(c['matrix']), np.array([c['radial'], 0., 0., 0., 0.]))[0].reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts': raise ValueError('Use a new artifact directory')
    failed = ROOT/'artifacts/freemocap-three-view-01'; source = ROOT/'artifacts/freemocap-three-view-02'
    seals = {p.name: verify(p) for p in (failed, source)}
    for filename in ('prepared-controls.json', 'prepared-observations.json', 'all-control-results.json',
                     'complete-joint-estimates.json', 'all-support-changes.json', 'all-segment-slots.json'):
        assert read(failed/filename) == read(source/filename), filename
    for i in (1, 2):
        assert digest(failed/f'worker-{i}/all-sample-estimates.jsonl') == digest(source/f'worker-{i}/all-sample-estimates.jsonl')
    pre = read(source/'pre-run.json'); runtime = {n: digest(p) for n, p in _implementation_files().items()}
    assert runtime == pre['runtime_hashes'] and len(runtime) == 121
    earlier = read(ROOT/'artifacts/freemocap-third-predictions-01/protocol.json')['runtime_hashes']
    assert all(runtime[n] == h for n, h in earlier.items()) and set(runtime)-set(earlier) == {'multiray_pose.py'}
    poses = read(source/'complete-joint-estimates.json')['frames']
    prepared = {r['frame']: r for r in read(source/'prepared-observations.json')}
    baseline = read(ROOT/'artifacts/freemocap-camera-space-joints-01/camera-space-joints.json')['frames']
    assert [r['frame'] for r in poses] == list(range(1108))
    maximum_difference = 0.; projections_checked = 0; availability_checked = 0
    for row in poses:
        frame = row['frame']; assert len(row['joints']) == 33 and row['time_s'] == frame/30
        for joint, entry in enumerate(row['joints']):
            assert entry['raw_two_view'] == baseline[frame]['joints'][joint]
            if frame % 2:
                assert all(entry[m]['world_point_m'] is None for m in ('two_view_pixel', 'all_available_views'))
                continue
            observations = prepared[frame]['joints'][joint]
            assert entry['observation_statuses'] == observations['statuses']
            available = [v for v, p in enumerate(observations['pixels']) if p is not None]
            for method in ('two_view_pixel', 'all_available_views'):
                fit = entry[method]; expected = available if method == 'all_available_views' else ([0, 1] if 0 in available and 1 in available else [])
                assert fit['used_views'] == expected
                availability_checked += 1
                if fit.get('point_proposal_m') is None: continue
                point = np.array(fit['point_proposal_m']); errors = []
                for view in fit['used_views']:
                    c = pre['cameras'][view]; position = np.array(c['rotation']) @ point+c['translation']
                    assert .2 <= position[2] <= 100
                    residual = project(point[None, :], c)[0]-observations['pixels'][view]
                    errors.append(float(np.linalg.norm(residual))); projections_checked += 1
                maximum_difference = max(maximum_difference, float(np.max(np.abs(np.array(errors)-fit['reprojection_px']))))
                accepted = fit['optimization_status'] == 'converged' and max(errors) <= 12.8
                assert accepted == (fit['status'] == 'estimated')
                assert (fit['world_point_m'] == fit['point_proposal_m']) if accepted else fit['world_point_m'] is None
    assert maximum_difference < 1e-8
    inputs = read(source/'prepared-controls.json')['cases']; results = read(source/'all-control-results.json')
    assert len(inputs) == len(results) == 288
    for case, result in zip(inputs, results, strict=True):
        assert (case['cohort'], case['index']) == (result['cohort'], result['index'])
        for method in ('baseline', 'candidate'):
            p = result[method]['world_point_m']; error = result['true_3d_errors_m'][method]
            assert error is None if p is None else abs(np.linalg.norm(np.array(p)-case['truth_m'])-error) < 1e-12
    tests = ROOT/'.local/multiray-pose-tests-02.xml'; cases = ET.parse(tests).findall('.//testcase')
    assert len(cases) == 16 and all(not any(c.find(tag) is not None for tag in ('failure', 'error', 'skipped')) for c in cases)
    third = ROOT/'artifacts/freemocap-third-comparison-01'
    observations = read(third/'all-comparison-image-observations.json')
    assert digest(third/'all-comparison-image-observations.json') == read(third/'manifest.json')['files']['all-comparison-image-observations.json']
    predictions = {'raw_two_view': [], 'two_view_pixel': []}
    for row in poses:
        for joint in MAJOR:
            for method in predictions:
                point = row['joints'][joint][method]['world_point_m']; pixel = None
                if point is not None:
                    c = pre['cameras'][2]; position = np.array(c['rotation']) @ point+c['translation']
                    if .2 <= position[2] <= 100:
                        pixel = project([point], c)[0].tolist()
                predictions[method].append({'frame': row['frame'], 'joint': joint, 'pixel': pixel})
    old_predictions = read(ROOT/'artifacts/freemocap-third-predictions-01/predictions-before-third-pixels.json')['raw_two_view']
    assert len(old_predictions) == len(predictions['raw_two_view']) == 13296
    for old, current in zip(old_predictions, predictions['raw_two_view'], strict=True):
        assert old['pixel'] is None if current['pixel'] is None else np.allclose(old['pixel'], current['pixel'], atol=1e-8, rtol=0)
    out.mkdir(exist_ok=False); shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'predictions-before-grading.json', predictions)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'source_manifests': seals,
        'prediction_sha256': digest(out/'predictions-before-grading.json'),
        'comparison_observations_sha256': digest(third/'all-comparison-image-observations.json'),
        'scorer_sha256': digest(ROOT/'src/weft/projected_joint_evaluation.py'),
        'interpretation': 'Previously exposed third-camera development comparison; no independent holdout claim. Only the condition fitted to Cam1/2 is graded, not the condition fitted to all three.',
        'maximum_output_bytes': 16000000, 'new_inference_calls': 0, 'new_fits': 0, 'paid_calls': 0})
    summaries = {}; started = time.perf_counter()
    for name in ('complete_source', 'active_400_through_1059'):
        selected = [i for i, r in enumerate(observations) if name == 'complete_source' or 400 <= r['frame'] <= 1059]
        result = grade_projected_joints({k: [v[i] for i in selected] for k, v in predictions.items()},
                                      [observations[i] for i in selected], baseline='raw_two_view', limit_px=12.8)
        summaries[name] = {k: v for k, v in result.items() if k != 'all_slot_errors'}
        write_json(out/f'{name}-errors.json', result['all_slot_errors'])
    assert runtime == {n: digest(p) for n, p in _implementation_files().items()}
    total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file()); assert total < 16000000
    write_json(out/'report.json', {'created_at': local_time(), 'status': 'saved_three_view_estimates_verified',
        'comparison': summaries, 'copied_fits_unchanged': True, 'complete_source_slots': 1108, 'joint_slots': 36564,
        'availability_decisions_checked': availability_checked, 'opencv_projections_checked': projections_checked,
        'maximum_reprojection_accounting_difference_px': maximum_difference, 'known_geometry_control_cases': len(inputs),
        'tests_reused': {'path': str(tests.relative_to(ROOT)), 'sha256': digest(tests), 'passed': len(cases)},
        'runtime_hashes': runtime, 'output_bytes': total, 'grading_seconds': time.perf_counter()-started,
        'new_fits': 0, 'new_pose_inference_calls': 0, 'new_video_decodes': 0, 'paid_calls': 0,
        'body_built': False, 'candidate_promoted': False, 'floor': None, 'physical_validation': 'unverified'})
    print(json.dumps({'comparison': summaries, 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
