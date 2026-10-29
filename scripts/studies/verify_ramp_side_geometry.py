"""Independent projection/accounting checks; reuse completed physics replay and tests."""

import argparse
from copy import deepcopy
from itertools import product
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image

from weft.affine_camera import fit_affine_camera
from weft.io import digest, local_time, write_json
from weft.orthographic_shape import conditional_axis_scale
from weft.pipeline import _implementation_files, verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
CAMERA = ROOT/'artifacts/ramp-side-camera-01'
DEPTH = ROOT/'artifacts/ramp-conditional-depth-01'


def read(path):
    return json.loads(path.read_text())


def journal(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def independent_vertices(obj):
    quaternion = np.asarray(obj['quaternion'], dtype=float)
    vector_length = np.linalg.norm(quaternion[1:])
    axis_angle = quaternion[1:] / vector_length * (2 * math.atan2(vector_length, quaternion[0])) if vector_length > 1e-15 else np.zeros(3)
    orientation = cv2.Rodrigues(axis_angle)[0]
    local = np.array(list(product((-1., 1.), repeat=3))) * np.asarray(obj['size']) / 2
    return (orientation @ local.T).T + obj['position']


def independent_centroid(pixels):
    hull = cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2).astype(float)
    next_point = np.roll(hull, -1, axis=0)
    cross = hull[:, 0]*next_point[:, 1] - hull[:, 1]*next_point[:, 0]
    return ((hull+next_point)*cross[:, None]).sum(axis=0)/(3*cross.sum())


def independent_overlap(mask, pixels):
    hull = cv2.convexHull((pixels-[400, 350]).astype(np.float32)).reshape(-1, 2)
    low = np.floor(np.minimum(hull.min(axis=0), [0, 0])).astype(int)
    high = np.ceil(np.maximum(hull.max(axis=0), [400, 400])).astype(int)
    predicted = np.fromiter((cv2.pointPolygonTest(hull, (x+.5, y+.5), False) >= 0
        for y in range(low[1], high[1]) for x in range(low[0], high[0])), dtype=bool).reshape(high[1]-low[1], high[0]-low[0])
    source = np.zeros_like(predicted)
    source[-low[1]:400-low[1], -low[0]:400-low[0]] = mask != 0
    return int((source & predicted).sum()) / int((source | predicted).sum())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.exists() or out.parent != ROOT/'.local':
        raise ValueError('Use a new local verification file')
    cv2.setNumThreads(1)
    verification = {str(path.relative_to(ROOT)): verify_artifacts(path) for path in (CAMERA, DEPTH)}
    runtime = {name: digest(path) for name, path in _implementation_files().items()}
    for directory in (CAMERA, DEPTH):
        prior = read(directory/'pre-run.json')['implementation']
        assert all(runtime[name] == value for name, value in prior.items())
    original = read(CAMERA/'inputs/scene.json'); candidate = read(DEPTH/'candidate/scene.json')
    result = read(DEPTH/'report.json'); factor = result['results']['depth_factor']
    expected = deepcopy(original)
    next(obj for obj in expected['objects'] if obj['id'] == 'green-box')['size'][1] *= factor
    for scene in (expected, candidate):
        scene.pop('assumptions')
        for obj in scene['objects']:
            obj.pop('provenance')
    assert expected == candidate
    original_events = journal(CAMERA/'inputs/events.jsonl')
    changed_events = journal(DEPTH/'candidate-coast/events.jsonl')
    assert len(original_events) == len(changed_events) == 71
    observations = read(CAMERA/'inputs/side-image-evidence.json')['rows']
    masks = {}
    for observed in observations:
        with Image.open(CAMERA/'inputs/masks'/Path(observed['mask']).name) as image:
            mask = np.asarray(image).copy()
        masks[observed['frame']] = mask
        y, x = np.nonzero(mask)
        np.testing.assert_allclose([x.mean()+400.5, y.mean()+350.5], observed['centroid_px'], atol=1e-10)
        assert int(np.count_nonzero(mask)) == observed['foreground_pixels']
    conditions = {
        'original': (original_events, read(CAMERA/'camera-fits.json'), read(CAMERA/'all-source-slots.json')),
        'conditional': (changed_events, read(DEPTH/'camera-proposals.json'), read(DEPTH/'all-side-slots.json')),
    }
    maximum_projection_difference = 0.; checks = {}; paired_count = 0
    for name, (events, cameras, hypotheses) in conditions.items():
        checks[name] = {}
        for hypothesis, rows in hypotheses.items():
            assert len(rows) == 37
            camera = np.asarray(cameras[hypothesis]['matrix_px']); errors = []; ious = []
            for frame, row in enumerate(rows):
                assert row['frame'] == frame and abs(row['source_time_s']-frame/25) < 1e-10
                if frame < 7:
                    assert row['status'] == 'before_simulation_initialization' and row['centroid_error_px'] is None
                    continue
                index = row['event_index']; state = events[index]['snapshot']
                actual_clock = 8*1001/30000 + state['time_s']
                assert abs(actual_clock-row['source_time_s']-row['pair_clock_error_s']) < 1e-10
                assert abs(row['pair_clock_error_s']) <= .5*1001/60000 + 1e-9
                obj = next(obj for obj in state['objects'] if obj['id'] == 'green-box')
                pixels = cv2.transform(independent_vertices(obj)[:, None, :], camera).reshape(-1, 2)
                difference = float(np.max(np.abs(pixels-row['projected_corners_px'])))
                maximum_projection_difference = max(maximum_projection_difference, difference)
                assert difference < 1e-8
                center = independent_centroid(pixels)
                np.testing.assert_allclose(center, row['predicted_silhouette_centroid_px'], atol=1e-7)
                error = float(np.linalg.norm(center-observations[frame]['centroid_px']))
                assert abs(error-row['centroid_error_px']) < 1e-7
                if frame == 36:
                    assert abs(independent_overlap(masks[frame], pixels)-row['mask_overlap']['iou']) < 1e-10
                errors.append(error); ious.append(row['mask_overlap']['iou']); paired_count += 1
            expected_result = result['results']['side'][hypothesis][name]
            rmse = float(np.sqrt(np.mean(np.square(errors)))); p95 = float(np.percentile(errors, 95))
            assert abs(rmse-expected_result['centroid_rmse_px']) < 1e-7
            assert abs(p95-expected_result['centroid_p95_px']) < 1e-7
            assert abs(float(np.mean(ious))-expected_result['mean_mask_iou']) < 1e-12
            checks[name][hypothesis] = {'compared': len(errors), 'rmse_px': rmse, 'p95_px': p95}

    front = read(DEPTH/'front-center-comparison.json')
    for name, comparison in front.items():
        errors = [row['error_px'] for row in comparison['rows'] if row['error_px'] is not None]
        assert len(comparison['rows']) == 88 and len(errors) == 72
        assert abs(float(np.sqrt(np.mean(np.square(errors))))-comparison['rmse_px']) < 1e-12
        events = original_events if name == 'original' else changed_events
        matrix = np.asarray(read(DEPTH/'construction-inputs.json')['projection']['matrix_px'])
        for row in comparison['rows']:
            if row['frame'] < 8:
                assert row['predicted_center_px'] is None
                continue
            state = events[2*(row['frame']-8)]['snapshot']
            obj = next(obj for obj in state['objects'] if obj['id'] == row['object_id'])
            predicted = cv2.transform(np.asarray(obj['position'], dtype=float).reshape(1, 1, 3), matrix).reshape(2)
            np.testing.assert_allclose(predicted, row['predicted_center_px'], atol=1e-10)
            assert abs(np.linalg.norm(predicted-row['center_px'])-row['error_px']) < 1e-10

    labels = read(CAMERA/'inputs/source-corners.json')
    construction = read(CAMERA/'construction-world-points.json')
    changed = read(DEPTH/'conditional-world-points.json')
    for name in construction:
        camera = conditions['original'][1][name]
        matrix = np.asarray(camera['matrix_px'])
        predicted = cv2.transform(np.asarray(construction[name])[:, None, :], matrix).reshape(-1, 2)
        np.testing.assert_allclose(predicted, camera['fitted_points_px'], atol=1e-10)
        proposal = conditions['conditional'][1][name]
        after = cv2.transform(np.asarray(changed[name])[:, None, :], np.asarray(proposal['matrix_px'])).reshape(-1, 2)
        np.testing.assert_allclose(after, predicted, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(predicted-labels['image_points_px'], axis=1), camera['errors_px'], atol=1e-10)
    for alternative in read(CAMERA/'depth-ambiguity-controls.json'):
        assert alternative['maximum_fitted_pixel_change'] < 1e-8
    test_count = 0
    for filename in ('affine-camera-tests-01.xml', 'orthographic-shape-tests-01.xml'):
        suites = ET.parse(ROOT/'.local'/filename).getroot().findall('testsuite')
        assert suites and all(int(suite.attrib.get(key, 0)) == 0 for suite in suites for key in ('failures', 'errors', 'skipped'))
        test_count += sum(int(suite.attrib['tests']) for suite in suites)
    assert test_count == 33
    media = {}
    for path, expected_frames in ((CAMERA/'side-comparison.mp4', 37), (DEPTH/'side-comparison.mp4', 37),
            (DEPTH/'candidate-coast/video.mp4', 71), (DEPTH/'candidate-controls/video.mp4', 14)):
        capture = cv2.VideoCapture(str(path)); frames = 0
        try:
            assert capture.isOpened()
            width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                assert frame.shape[:2] == (height, width)
                frames += 1
        finally:
            capture.release()
        assert frames == expected_frames
        media[str(path.relative_to(ROOT))] = {'frames_decoded': frames, 'size': [width, height]}

    points = np.random.default_rng(1984).uniform(.1, .6, (1024, 3))
    matrix = np.array([[1000., 0., -200., 300.], [0., -1000., 0., 900.]])
    pixels = cv2.transform(points[:, None, :], matrix).reshape(-1, 2)
    timings = []
    for _ in range(40):
        started = time.perf_counter()
        camera = fit_affine_camera(points, pixels, image_size=[1920, 1080], scope='construction')
        conditional_axis_scale(camera['matrix_px'], direction=[0., 1., 0.], center=points.mean(axis=0), scope='construction')
        timings.append((time.perf_counter()-started)*1000)
    ledger = read(ROOT/'.local/meta-budget/ledger.json')
    spent = sum(item['charged_microusd'] for item in ledger['reservations'].values()) / 1e6
    report = {'created_at': local_time(), 'status': 'verified', 'artifacts': verification,
        'runtime_files': len(runtime), 'runtime_matches_both_studies': True,
        'scene_change': 'green depth only, plus provenance and assumptions; all other physical settings unchanged',
        'source_side_centroids_verified': 37, 'side_slots_preserved': 148, 'side_projections_verified': paired_count,
        'max_independent_opencv_projection_difference_px': maximum_projection_difference,
        'independent_final_mask_overlap_checks': 4, 'all_front_center_slots_checked': 176,
        'side_metrics': checks, 'saved_passing_tests': test_count, 'passing_tests_repeated': False,
        'reused_completed_exact_replay_states': result['new_coast_states_exact'] + result['new_control_states_exact'],
        'media': media, 'decoded_export_frames': sum(item['frames_decoded'] for item in media.values()),
        'maximum_count_cpu_profile': {'points': 1024, 'repeats': 40,
            'median_ms': float(np.median(timings)), 'p95_ms': float(np.percentile(timings, 95)),
            'scope': 'Numerical fit plus conditional scale only; not UI, decoding or end-to-end latency'},
        'meta_spent_usd': spent, 'meta_calls': len(ledger['reservations']),
        'unsettled_reservations': sum(item['status'] != 'settled' for item in ledger['reservations'].values()),
        'new_paid_calls': 0, 'candidate_promoted': False, 'physical_validation': 'unverified'}
    write_json(out, report); print(json.dumps(report))


if __name__ == '__main__':
    main()
