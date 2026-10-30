"""Audit saved track coverage, low-rank projections and relative shapes without refitting."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np

from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def errors(predicted, observed):
    values = np.sqrt(np.sum((predicted-observed)**2, axis=2))
    return values, [np.sqrt(np.mean(values**2)), np.percentile(values, 95), values.max()]


def verify_fit(result, observed):
    maximum_difference = 0.
    if result['affine_fit']:
        centered = observed-observed.mean(axis=1)[:, None, :]
        matrix = centered.transpose(0, 2, 1).reshape(2*len(observed), -1)
        singular, left, right = cv2.SVDecomp(matrix)
        projected = ((left[:, :3]*singular[:3, 0]) @ right[:3]).reshape(-1, 2, observed.shape[1]).transpose(0, 2, 1)
        projected += observed.mean(axis=1)[:, None, :]
        maximum_difference = float(np.max(np.abs(projected-result['affine_fit']['predicted_points_px'])))
        assert maximum_difference < 1e-7
        values, metrics = errors(projected, observed)
        saved = result['affine_fit']['errors']
        np.testing.assert_allclose(values, saved['per_point_px'], atol=1e-7)
        np.testing.assert_allclose(metrics, [saved[k] for k in ('rms_px', 'p95_px', 'maximum_px')], atol=1e-7)
        np.testing.assert_allclose(singular[2, 0]/singular[0, 0], result['third_singular_ratio'], atol=1e-10)
        if result['metric']:
            motion = left[:, :3]*np.sqrt(singular[:3, 0])
            basis = np.zeros((6, 3, 3))
            for i in range(3):basis[i, i, i] = 1
            for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), 3):basis[k, i, j] = basis[k, j, i] = 1/np.sqrt(2)
            rows = []
            for camera in motion.reshape(-1, 2, 3):
                grams = np.einsum('ai,kij,bj->kab', camera, basis, camera)
                energy = np.sum(camera**2)
                rows.extend([(grams[:, 0, 0]-grams[:, 1, 1])/energy, np.sqrt(2)*grams[:, 0, 1]/energy])
            values = cv2.SVDecomp(np.asarray(rows))[0][:, 0]
            np.testing.assert_allclose(values, result['metric']['constraint_singular_values'], atol=1e-8)
            metric = result['metric']
            if result['status'] == 'unobservable_metric':assert not metric['identifiable']
            if result['status'] == 'inconsistent_metric_constraints':assert metric['null_gap'] < 10
    candidate = result['candidate']
    if candidate:
        points = np.asarray(candidate['points_relative'])
        cameras = np.asarray(candidate['camera_rows_px_per_relative_unit'])
        centers = np.asarray(candidate['feature_centroids_px'])
        projected = np.array([cv2.transform(points[None, :, :], np.column_stack([camera, center]))[0]
            for camera, center in zip(cameras, centers, strict=True)])
        maximum_difference = max(maximum_difference, float(np.max(np.abs(projected-candidate['predicted_points_px']))))
        assert maximum_difference < 1e-7
        values, metrics = errors(projected, observed)
        saved = candidate['errors']
        np.testing.assert_allclose(values, saved['per_point_px'], atol=1e-7)
        np.testing.assert_allclose(metrics, [saved[k] for k in ('rms_px', 'p95_px', 'maximum_px')], atol=1e-7)
        np.testing.assert_allclose(points.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.mean(np.sum(points**2, axis=1)), 1, atol=1e-12)
        for camera in cameras:
            gram = camera @ camera.T
            np.testing.assert_allclose(gram, np.eye(2)*np.trace(gram)/2, atol=1e-8)
        mirrored = points.copy(); mirrored[:, 2] *= -1
        reflected = cameras.copy(); reflected[:, :, 2] *= -1
        reflection = np.einsum('fij,pj->fpi', reflected, mirrored)+centers[:, None, :]
        np.testing.assert_allclose(reflection, projected, atol=1e-10)
    assert not result['rigidity_verified'] and not result['metric_scale_verified']
    return maximum_difference


def normalized_distances(points):
    centered = np.asarray(points)-np.asarray(points).mean(axis=0)
    mean_square_radius = sum(float(point @ point) for point in centered)/len(centered)
    return np.array([np.sqrt(float((a-b) @ (a-b))/mean_square_radius)
        for i, a in enumerate(centered) for b in centered[i+1:]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=ROOT/'artifacts/track-shape-diversity-01')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); study = args.study.resolve(); output = args.out.resolve()
    if output.exists():raise ValueError('Verification output must be new')
    started = time.perf_counter(); cv2.setNumThreads(1)
    report = read(study/'report.json'); pre = read(study/'pre-run.json')
    study_check = verify_artifacts(study)
    original = ROOT/'artifacts/feature-track-diversity-01'
    input_check = verify_artifacts(original)
    assert digest(original/'manifest.json') == pre['input_manifest_sha256']
    for name, expected in pre['implementation'].items():
        assert digest(ROOT/name) == expected
        assert digest(study/Path(name).name) == expected
    totals = Counter(); statuses = Counter(); max_projection_difference = 0.
    for name, expected in report['cases'].items():
        binding = read(original/name/'source-binding.json')
        source = [json.loads(line) for line in (original/name/'all-frame-tracklets.jsonl').open()]
        saved = [json.loads(line) for line in (study/name/'windows.jsonl').open()]
        input_binding = read(study/name/'input-binding.json')
        assert digest(ROOT/input_binding['all_frame_tracklets']) == input_binding['sha256']
        totals['source_frames'] += len(source)
        totals['source_object_slots'] += sum(len(f['objects']) for f in source)
        raw = sum(len(o['tracks']['rows']) if o['tracks'] else 0 for f in source for o in f['objects'])
        assert raw == expected['raw_input_point_slots']; totals['raw_input_point_slots'] += raw
        observed_unique = sum(sum(r['track_id'] is not None for r in o['tracks']['rows'])
            if o['tracks'] else 0 for f in source for o in f['objects'])
        totals['distinct_observed_point_slots'] += observed_unique
        object_counts = {obj['id']: Counter() for obj in binding['objects']}; consumed = set(); observed_total = 0
        assert len(saved) == ((len(source)+7)//8)*len(object_counts)
        for row in saved:
            result = row['result']; identifier = row['object_id']; frames = row['frame_ids']
            assert frames == list(range(frames[0], min(frames[0]+8, len(source)))) and frames[0] % 8 == 0
            items = [next(o for o in source[f]['objects'] if o['object_id'] == identifier) for f in frames]
            rows = [{p['track_id']: p['point_px'] for p in o['tracks']['rows'] if p['track_id'] is not None}
                if o['tracks'] else {} for o in items]
            tracks = sorted(set().union(*(set(r) for r in rows)))
            assert tracks == row['track_ids'] == result['track_ids']
            points = np.full((len(frames), len(tracks), 2), np.nan)
            for f, mapping in enumerate(rows):
                for j, track in enumerate(tracks):
                    if track in mapping:points[f, j] = mapping[track]
            saved_points = np.asarray(row['points_px'], dtype=float).reshape(len(frames), len(tracks), 2)
            np.testing.assert_array_equal(points, saved_points)
            assert row['times_s'] == result['times_s'] == [source[f]['time_s'] for f in frames]
            assert row['source_object_slots'] == [{k: v for k, v in obj.items() if k != 'tracks'} for obj in items]
            present = np.isfinite(points).all(axis=2); complete = present.all(axis=0)
            assert result['complete_track_ids'] == [t for t, keep in zip(tracks, complete, strict=True) if keep]
            assert result['coverage'] == {'source_point_slots': present.size,
                'observed_point_slots': int(present.sum()), 'missing_point_slots': int((~present).sum()),
                'available_per_frame': present.sum(axis=1).tolist(), 'available_per_track': present.sum(axis=0).tolist()}
            observed_total += int(present.sum())
            for frame in frames:
                assert (identifier, frame) not in consumed
                consumed.add((identifier, frame))
            if result['affine_fit']:totals['fitted_point_slots'] += len(frames)*int(complete.sum())
            max_projection_difference = max(max_projection_difference, verify_fit(result, points[:, complete]))
            statuses[result['status']] += 1; object_counts[identifier][result['status']] += 1
        assert len(consumed) == expected['object_slots']
        assert observed_total == observed_unique == expected['distinct_observed_point_slots']
        assert {k: dict(v) for k, v in object_counts.items()} == expected['by_object']
    controls = Counter(); control_errors = []
    expected_status = {'no_depth_motion': 'insufficient_depth_excitation',
        'wrong_correspondence': 'inconsistent_affine_tracks', 'two_view_orientations': 'unobservable_metric',
        'coherent_scale_deformation': 'conditional_shape'}
    for line in (study/'synthetic/controls.jsonl').open():
        row = json.loads(line); result = row['result']; name = row['case']
        max_projection_difference = max(max_projection_difference, verify_fit(result, np.asarray(row['observed_px'])))
        controls[name] += 1
        if name.startswith('rigid-noise'):
            assert result['status'] == 'conditional_shape'
            error = float(np.sqrt(np.mean((normalized_distances(row['truth_points'])-
                normalized_distances(result['candidate']['points_relative']))**2)))
            assert error < pre['contract']['control_distance_rms_limit']
            assert abs(error-row['distance_rms_error']) < 1e-12
            control_errors.append(error)
        else:
            assert result['status'] == expected_status[name]
        if name == 'coherent_scale_deformation':assert row['physical_rigidity_false_by_construction']
    assert sum(controls.values()) == 56 and all(count == 8 for count in controls.values())
    result = {'created_at': local_time(), 'status': 'verified', 'study': str(study.relative_to(ROOT)),
        'study_manifest_sha256': digest(study/'manifest.json'), 'study_check': study_check, 'input_check': input_check,
        'source_totals': dict(totals), 'window_statuses': dict(statuses), 'control_counts': dict(controls),
        'maximum_normalized_control_distance_error': max(control_errors),
        'maximum_independent_projection_difference_px': max_projection_difference,
        'implementation_files_checked': len(pre['implementation']), 'seconds': time.perf_counter()-started,
        'new_fits': 0, 'new_paid_calls': 0, 'limits': 'Real shape accuracy and physical identity remain unverified; zero real windows yield a candidate.'}
    write_json(output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
