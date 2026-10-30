"""Verify saved image-point lifetimes, photometric checks and known motion without retracking."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np

from check_surface_feature_diversity import CASES, load_case, source_mask
from verify_track_shape_diversity import verify_fit
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT/'artifacts/surface-feature-diversity-01'
HISTORIES = ROOT/'artifacts/feature-track-diversity-01'


def read(path):
    return json.loads(path.read_text())


def patch(gray, point, size):
    return cv2.getRectSubPix(gray, (size, size), tuple(float(v-.5) for v in point)).astype(np.float32)


def check_lifetime(flow, previous, seeds, frame):
    assert flow['frame'] == frame and flow['original_seed_count'] == len(seeds)
    assert [r['seed_id'] for r in flow['rows']] == list(range(len(seeds)))
    active = 0; maximum_roundtrip_rounding = 0.
    for row in flow['rows']:
        index = row['seed_id']; assert row['original_point_px'] == seeds[index]
        prior = previous['rows'][index] if previous else None
        if row['point_px'] is not None:
            active += 1
            assert row['ended'] is None
            assert row['status'] == ('flow_pair' if prior else 'initialization')
            assert row['age_frames'] == (prior['age_frames']+1 if prior else 1)
            if prior:
                assert prior['point_px'] is not None and row['forward_valid'] and row['backward_valid']
                returned = np.asarray(row['backward_point_px']); old = np.asarray(prior['point_px'])
                distance = np.linalg.norm(returned-old)
                # The journal adds the half-pixel offset in float32 before subtracting the prior point.
                stored = np.linalg.norm((returned-.5).astype(np.float32)+.5-old)
                assert abs(stored-row['roundtrip_error_px']) < 1e-12 and distance <= 1.
                maximum_roundtrip_rounding = max(maximum_roundtrip_rounding, abs(distance-stored))
                assert abs(np.linalg.norm(np.asarray(row['point_px'])-prior['point_px'])-row['step_px']) < 1e-9
                assert row['step_px'] <= 64 and row['anchor_ecc'] >= .9 and row['local_patch_ncc'] >= .9
                assert row['anchor_correction_px'] <= 3
                warp = np.asarray(row['anchor_warp']); singular = np.linalg.svd(warp[:, :2], compute_uv=False)
                assert np.linalg.det(warp[:, :2]) > 0 and singular[-1] >= .5 and singular[0] <= 2
                assert singular[0]/singular[-1] <= 3
                center = np.array([15., 15.])
                reconstructed = np.asarray(row['unrefined_point_px'])+warp[:, :2] @ center+warp[:, 2]-center
                np.testing.assert_allclose(reconstructed, row['point_px'], atol=1e-9)
                assert abs(np.linalg.norm(reconstructed-row['unrefined_point_px'])-row['anchor_correction_px']) < 1e-9
                corners = np.array([[5, 5], [25, 5], [25, 25], [5, 25]]) @ warp[:, :2].T+warp[:, 2]
                assert (corners >= 0).all() and (corners <= 30).all()
        else:
            assert row['ended'] is not None
            if prior and prior['ended']:
                assert row['status'] == 'ended' and row['ended'] == prior['ended']
                assert row['age_frames'] == prior['age_frames']
            else:
                assert row['ended'] == {'frame': frame, 'reason': row['status']}
    assert flow['active_points'] == active
    assert not flow['physical_point_identity_verified'] and not flow['rigidity_verified']
    return maximum_roundtrip_rounding


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--study', type=Path, default=ROOT/'artifacts/point-flow-diversity-01')
    args = parser.parse_args(); study = args.study.resolve(); output = args.out.resolve()
    if output.exists():raise ValueError('Verification output must be new')
    started = time.perf_counter(); cv2.setNumThreads(1)
    checks = {str(p.relative_to(ROOT)): verify_artifacts(p) for p in (study, FEATURES, HISTORIES)}
    pre, report = read(study/'pre-run.json'), read(study/'report.json')
    for name, expected in pre['implementation'].items():
        assert digest(ROOT/name) == expected and digest(study/Path(name).name) == expected
    totals = Counter(); statuses = Counter(); maximum_ncc_difference = 0.; maximum_projection_difference = 0.
    later_disagreements = []; shape_counts = Counter(); maximum_roundtrip_rounding = 0.
    for name, expected in report['cases'].items():
        case = load_case(name, ROOT/CASES[name]); saved = study/name; seeds = read(saved/'seeds.json')
        baseline = [json.loads(line) for line in (HISTORIES/name/'all-frame-tracklets.jsonl').open()]
        capture = cv2.VideoCapture(str(case['source'])); previous_gray = None; previous = {}; anchors = {}; history = []
        case_statuses = Counter(); observed = {identifier: [] for identifier in seeds}
        try:
            for line in (saved/'all-frame-points.jsonl').open():
                frame = json.loads(line); index = frame['frame']; ok, bgr = capture.read()
                assert ok and index == len(history) and frame['time_s'] == case['times'][index]
                assert abs(capture.get(cv2.CAP_PROP_POS_MSEC)/1000-frame['time_s']) < 1e-6
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                for obj in frame['objects']:
                    identifier = obj['object_id']; flow = obj['flow']; seed = seeds[identifier]
                    totals['source_object_slots'] += 1
                    if flow is None:
                        assert index < seed['frame']; continue
                    if identifier not in anchors:
                        locator = seed['features']
                        with np.load(FEATURES/name/locator['file'], allow_pickle=False) as data:
                            selected = data[locator['prefix']+'_selected_indices'].astype(int)
                            assert selected.tolist() == seed['source_indices']
                            np.testing.assert_array_equal(data[locator['prefix']+'_keypoints'][selected, :2],
                                np.asarray(seed['points_px']).reshape(len(selected), 2))
                        anchors[identifier] = [cv2.GaussianBlur(patch(gray, point, 31), (5, 5), 0) for point in seed['points_px']]
                    old = previous.get(identifier)
                    maximum_roundtrip_rounding = max(maximum_roundtrip_rounding,
                        check_lifetime(flow, old, seed['points_px'], index))
                    case_statuses.update(r['status'] for r in flow['rows']); totals['seed_frame_slots'] += len(flow['rows'])
                    observed[identifier].append((index, flow['active_points']))
                    active = [r for r in flow['rows'] if r['point_px'] is not None]
                    if active:
                        region = source_mask(case, case['slots'][(index, identifier)])
                        inset = cv2.erode(region, np.ones((33, 33), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
                        for row in active:
                            x, y = row['point_px']; assert inset[int(y), int(x)]
                            if row['status'] == 'initialization':continue
                            prior = old['rows'][row['seed_id']]
                            local_ncc = cv2.computeECC(patch(previous_gray, prior['point_px'], 9), patch(gray, row['point_px'], 9))
                            maximum_ncc_difference = max(maximum_ncc_difference, abs(local_ncc-row['local_patch_ncc']))
                            target = cv2.GaussianBlur(patch(gray, row['unrefined_point_px'], 31), (5, 5), 0)
                            aligned = cv2.warpAffine(target, np.asarray(row['anchor_warp'], np.float32), (31, 31), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                            value = cv2.computeECC(anchors[identifier][row['seed_id']][5:-5, 5:-5], aligned[5:-5, 5:-5])
                            maximum_ncc_difference = max(maximum_ncc_difference, abs(value-row['anchor_ecc']))
                            totals['photometric_pairs_verified'] += 1
                    prior_obj = next(o for o in baseline[index]['objects'] if o['object_id'] == identifier)
                    opening = {r['reference_id']: r['point_px'] for r in prior_obj['tracks']['rows'] if r['reference_id'] is not None} if prior_obj['tracks'] else {}
                    assert obj['baseline_opening_anchors'] == len(opening)
                    expected_common = [r for r in active if seed['source_indices'][r['seed_id']] in opening]
                    assert len(expected_common) == len(obj['shared_point_differences'])
                    for point, difference in zip(expected_common, obj['shared_point_differences'], strict=True):
                        original = seed['source_indices'][point['seed_id']]
                        assert difference['seed_id'] == point['seed_id'] and difference['baseline_point_px'] == opening[original]
                        distance = float(np.linalg.norm(np.asarray(point['point_px'])-opening[original]))
                        assert abs(distance-difference['distance_px']) < 1e-9
                        if index > seed['frame']:later_disagreements.append(distance)
                    previous[identifier] = flow
                history.append(frame); previous_gray = gray
            assert len(history) == expected['source_frames'] == len(case['times']) and not capture.read()[0]
        finally:capture.release()
        assert dict(case_statuses) == expected['status_counts']; statuses.update(case_statuses)
        totals['source_frames_decoded'] += len(history)
        for identifier, values in observed.items():
            later = [(f, n) for f, n in values if f > seeds[identifier]['frame']]
            assert sum(n > 0 for _, n in later) == expected['by_object'][identifier]['frames_with_original_points']
            assert sum(n >= 6 for _, n in later) == expected['by_object'][identifier]['frames_with_six_original_points']
        coverage = {size: set() for size in (8, 32)}
        for line in (saved/'shape-windows.jsonl').open():
            item = json.loads(line); result = item['result']; length = item['requested_window_frames']; identifier = item['object_id']
            frames = result['frame_ids']; assert frames == list(range(frames[0], min(frames[0]+length, len(history)))) and frames[0] % length == 0
            points = np.full((len(frames), len(result['track_ids']), 2), np.nan)
            for f, index in enumerate(frames):
                assert (identifier, index) not in coverage[length]; coverage[length].add((identifier, index))
                flow = next(o['flow'] for o in history[index]['objects'] if o['object_id'] == identifier)
                if flow:
                    for row in flow['rows']:
                        if row['point_px'] is not None:points[f, row['seed_id']] = row['point_px']
            complete = np.isfinite(points).all(axis=2).all(axis=0)
            assert result['complete_track_ids'] == [i for i, keep in zip(result['track_ids'], complete, strict=True) if keep]
            maximum_projection_difference = max(maximum_projection_difference, verify_fit(result, points[:, complete]))
            shape_counts[str(length)+':'+result['status']] += 1
        assert all(len(slots) == expected['source_object_slots'] for slots in coverage.values())
        video = cv2.VideoCapture(str(saved/'point-histories.mp4')); count = 0
        while True:
            ok, frame = video.read()
            if not ok:break
            assert frame.shape[:2] == (460, 1280); count += 1
        video.release(); assert count == len(history); totals['export_frames_decoded'] += count
    assert maximum_ncc_difference < 1e-5
    inputs = read(study/'synthetic/input-binding.json'); seeds = inputs['seed_points_px']
    errors = {'continuous': [], 'missing_masks': []}; last = {}; clocks = Counter()
    for line in (study/'synthetic/known-warp-points.jsonl').open():
        row = json.loads(line); condition = row['condition']; frame = row['frame']; flow = row['flow']
        maximum_roundtrip_rounding = max(maximum_roundtrip_rounding,
            check_lifetime(flow, last.get(condition), seeds, frame))
        matrix = np.asarray(row['transform']); positions = cv2.transform((np.asarray(seeds)-.5)[None, :, :], matrix[:2])[0]+.5
        for point, error in zip(flow['rows'], row['known_point_errors'], strict=True):
            assert point['seed_id'] == error['seed_id']
            if point['point_px'] is None:assert error['error_px'] is None
            else:
                value = float(np.linalg.norm(np.asarray(point['point_px'])-positions[point['seed_id']]))
                assert abs(value-error['error_px']) < 1e-10; errors[condition].append(value)
        if condition == 'missing_masks' and frame >= 31:assert flow['active_points'] == 0
        last[condition] = flow; clocks[condition] += 1
    assert dict(clocks) == {'continuous': 75, 'missing_masks': 75}
    for name, values in errors.items():
        expected = report['synthetic']['point_error_px'][name]
        assert len(values) == expected['count']
        np.testing.assert_allclose([np.median(values), np.percentile(values, 95), max(values)],
            [expected[k] for k in ('median', 'p95', 'maximum')], atol=1e-10)
    result = {'status': 'verified', 'created_at': local_time(), 'study_manifest_sha256': digest(study/'manifest.json'),
        'checks': checks, 'totals': dict(totals), 'status_counts': dict(statuses), 'shape_windows': dict(shape_counts),
        'maximum_photometric_accounting_difference': maximum_ncc_difference,
        'maximum_projection_accounting_difference_px': maximum_projection_difference,
        'maximum_roundtrip_float32_rounding_px': maximum_roundtrip_rounding,
        'known_point_errors_verified': sum(len(v) for v in errors.values()),
        'later_method_disagreement_px': {'count': len(later_disagreements), 'p95': float(np.percentile(later_disagreements, 95)), 'maximum': max(later_disagreements)},
        'seconds': time.perf_counter()-started, 'new_tracking_fits': 0, 'new_paid_calls': 0,
        'limitations': 'Experimental tracker has worse real-source coverage; no real shape or physical accuracy promotion.'}
    write_json(output, result); print(json.dumps(result))


if __name__ == '__main__':main()
