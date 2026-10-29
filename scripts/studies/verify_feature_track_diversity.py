"""Independent graph, descriptor and known-warp checks; no tracker or detector rerun."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from verify_surface_feature_diversity import arrays, check_matches
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'artifacts/feature-track-diversity-01'
INPUT = ROOT/'artifacts/surface-feature-diversity-01'


def read(path):
    return json.loads(path.read_text())


def graph_check(features, result, previous, expected_next):
    rows = result['rows']; assert len(rows) == len(features['keypoints'])
    selected = features['selected_indices'].astype(int).tolist()
    adjacent = result['adjacent_matches']; direct = result['original_matches']
    links = {r['current_id']: r['reference_id'] for r in adjacent['rows'] if r['status'] == 'appearance_pair'} if adjacent else {}
    anchors = {r['current_id']: r['reference_id'] for r in direct['rows'] if r['status'] == 'appearance_pair'} if direct else {}
    initialization = result['continuity'] == 'initialization'; candidates = {}; claimed = Counter()
    for index in selected:
        prior = previous.get(links.get(index)); carried = prior['reference_id'] if prior else None
        candidates[index] = sorted({r for r in (carried, anchors.get(index)) if r is not None})
        claimed.update(candidates[index])
    continued = set(); conflicts = set(); next_active = {}; statuses = Counter()
    for index, row in enumerate(rows):
        assert row['current_id'] == index and row['point_px'] == features['keypoints'][index, :2].tolist()
        statuses[row['link_status']] += 1
        if index not in selected:
            assert row['track_id'] is row['reference_id'] is None and row['duplicate_of'] == features['aliases'][index]
            assert row['link_status'] == 'duplicate_location'; continue
        prior = previous.get(links.get(index))
        conflict = not initialization and (len(candidates[index]) > 1 or any(claimed[r] > 1 for r in candidates[index]))
        linked = prior is not None and not conflict
        if initialization:
            assert row['reference_id'] == index and row['anchor_status'] == 'declared_initialization'
        else:
            assert row['reference_candidates'] == candidates[index]
            assert row['previous_current_id'] == links.get(index)
            assert row['previous_track_id'] == (prior['track_id'] if prior else None)
            assert row['reference_id'] == (candidates[index][0] if len(candidates[index]) == 1 and not conflict else None)
        if linked:
            assert row['link_status'] == 'adjacent_pair' and row['track_id'] == prior['track_id']
            assert row['age_frames'] == prior['age_frames']+1; continued.add(prior['track_id'])
        else:
            assert row['track_id'] == expected_next and row['age_frames'] == 1; expected_next += 1
            assert row['link_status'] == ('initialization' if initialization else 'reference_conflict' if conflict else 'new_appearance')
            if prior is not None:conflicts.add(prior['track_id'])
        next_active[index] = row
    expected_ended = {r['track_id']: {'track_id': r['track_id'], 'last_current_id': index,
        'age_frames': r['age_frames'], 'reference_id': r['reference_id'],
        'reason': 'reference_conflict' if r['track_id'] in conflicts else 'no_adjacent_pair'}
        for index, r in previous.items() if r['track_id'] not in continued}
    assert {r['track_id']: r for r in result['ended_tracks']} == expected_ended
    assert result['total_track_ids'] == expected_next and result['active_tracks'] == len(selected)
    assert result['linked_tracks'] == statuses['adjacent_pair'] and result['conflicted_tracks'] == statuses['reference_conflict']
    active_anchors = [r['reference_id'] for r in rows if r['reference_id'] is not None]
    assert result['anchored_tracks'] == len(active_anchors) == len(set(active_anchors))
    assert result['physical_point_identity_verified'] is result['rigidity_verified'] is False
    return next_active, expected_next, statuses


def rotation(angle, center):
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle)); x, y = center
    return np.array([[c, s, (1-c)*x-s*y], [-s, c, s*x+(1-c)*y], [0., 0., 1.]])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.exists() or out.parent != ROOT/'.local':raise ValueError('Use a new local verification file')
    started = time.perf_counter(); seals = {p.name: verify_artifacts(p) for p in (STUDY, INPUT)}
    report, protocol = read(STUDY/'report.json'), read(STUDY/'pre-run.json')
    runtime = {name: digest(path) for name,path in _implementation_files().items()}
    assert runtime == protocol['implementation'] and report['status'] == 'complete'
    assert protocol['input_seal'] == seals[INPUT.name]
    assert digest(ROOT/'tests/test_feature_tracks.py') == digest(STUDY/'test_feature_tracks.py')
    assert digest(ROOT/'.local/feature-track-tests-01.xml') == protocol['test_report_sha256']
    tests = ET.parse(ROOT/'.local/feature-track-tests-01.xml').getroot().find('testsuite')
    assert tests.attrib['tests'] == '12' and tests.attrib['errors'] == tests.attrib['failures'] == tests.attrib['skipped'] == '0'
    cases = {}; all_counts = Counter(); total_frames = total_slots = total_points = total_decoded = 0
    largest_distance_difference = 0.; conflicts = []
    for name, expected in report['cases'].items():
        binding = read(STUDY/name/'source-binding.json')
        assert binding == read(INPUT/name/'source-binding.json') and digest(ROOT/binding['source']) == binding['sha256']
        prior_features = {}; previous = {}; next_id = {}; archive = None; archive_name = None; counts = Counter()
        frames = slots = points = 0; per_object = {obj['id']: [] for obj in binding['objects']}
        try:
            with (INPUT/name/'all-frame-tracks.jsonl').open() as source, (STUDY/name/'all-frame-tracklets.jsonl').open() as stream:
                for frame, (original_line, current_line) in enumerate(zip(source, stream, strict=True)):
                    original, current = json.loads(original_line), json.loads(current_line)
                    assert original['frame'] == current['frame'] == frame
                    assert original['time_s'] == current['time_s'] == binding['times_s'][frame]
                    for old, obj in zip(original['objects'], current['objects'], strict=True):
                        slots += 1
                        for key in ('object_id','source_observation_id','source_identity','mask_status','status','features'):assert old[key] == obj[key]
                        if obj['features'] is None:assert obj['tracks'] is None; continue
                        locator = obj['features']; identifier = obj['object_id']; result = obj['tracks']
                        if locator['file'] != archive_name:
                            if archive is not None:archive.close()
                            archive = np.load(INPUT/name/locator['file'], allow_pickle=False); archive_name = locator['file']
                        features = arrays(archive, locator)
                        if identifier in prior_features:
                            assert result['original_matches'] == old['matches']
                            _, difference = check_matches(prior_features[identifier], features, result['adjacent_matches'])
                            largest_distance_difference = max(largest_distance_difference, difference)
                        else:assert result['continuity'] == 'initialization' and result['original_matches'] is None
                        active, counter, statuses = graph_check(features, result, previous.get(identifier, {}), next_id.get(identifier, 0))
                        previous[identifier], next_id[identifier], prior_features[identifier] = active, counter, features
                        counts.update(statuses); points += len(result['rows'])
                        per_object[identifier].append({'frame': frame, 'time_s': current['time_s'],
                            'direct_reference_pairs': old['matches']['appearance_pairs'],
                            **{key: result[key] for key in ('active_tracks','linked_tracks','anchored_tracks','conflicted_tracks','total_track_ids')},
                            'longest_active_age': max((r['age_frames'] or 0 for r in result['rows']), default=0)})
                        conflicts.extend({'case': name, 'frame': frame, 'object': identifier, 'point': row['current_id'],
                            'reference_candidates': row['reference_candidates'], 'previous_track_id': row['previous_track_id'],
                            'new_track_id': row['track_id']} for row in result['rows'] if row['link_status'] == 'reference_conflict')
                    frames += 1
        finally:
            if archive is not None:archive.close()
        assert frames == expected['frames'] == expected['source_frames'] == len(binding['times_s'])
        assert slots == expected['source_object_slots'] and points == expected['raw_current_point_slots']
        assert dict(counts) == expected['status_counts']
        for identifier, values in per_object.items():
            saved = expected['by_object'][identifier]; later = values[1:]
            assert saved['per_frame'] == values and saved['later_frames'] == len(later)
            assert saved['later_frames_with_any_adjacent_link'] == sum(v['linked_tracks'] > 0 for v in later)
            assert saved['later_frames_with_original_anchor'] == sum(v['anchored_tracks'] > 0 for v in later)
            assert saved['original_reference_conflict_points'] == sum(v['conflicted_tracks'] for v in values)
        video = cv2.VideoCapture(str(STUDY/name/'tracklets.mp4')); decoded = 0
        try:
            assert video.isOpened()
            while True:
                ok, image = video.read()
                if not ok:break
                assert image.shape == (460,1280,3); decoded += 1
        finally:video.release()
        assert decoded == frames
        cases[name] = {'frames': frames, 'object_slots': slots, 'point_slots': points, 'decoded_video_frames': decoded}
        total_frames += frames; total_slots += slots; total_points += points; total_decoded += decoded; all_counts.update(counts)
    synthetic = read(STUDY/'synthetic/report.json'); config = synthetic['config']; previous = {}; original = None
    adjacent_errors = []; anchor_errors = []; max_warp_difference = 0.; seen_ids = set()
    with (STUDY/'synthetic/all-frame-controls.jsonl').open() as stream:
        for frame, line in enumerate(stream):
            row = json.loads(line); assert row['frame'] == frame
            transform = rotation(frame*config['rotation_degrees_per_frame'], np.asarray(config['image_size'])/2)
            np.testing.assert_allclose(row['transform'], transform, atol=1e-12)
            tracks = row['tracks']; point_rows = tracks['rows']
            if original is None:original = {p['current_id']: p['point_px'] for p in point_rows}
            assert len(row['known_point_errors']) == len(point_rows)
            step = rotation(config['rotation_degrees_per_frame'], np.asarray(config['image_size'])/2)
            for point, saved in zip(point_rows, row['known_point_errors'], strict=True):
                assert point['current_id'] == saved['current_id']
                for label, matrix, origin, errors in (
                    ('adjacent_error_px', step, previous.get(point['track_id']) if point['link_status'] == 'adjacent_pair' else None, adjacent_errors),
                    ('original_anchor_error_px', transform, original.get(point['reference_id']), anchor_errors)):
                    if origin is None:assert saved[label] is None; continue
                    predicted = cv2.perspectiveTransform(np.array([[np.asarray(origin)-.5]], dtype=float), matrix)[0,0]+.5
                    error = float(np.linalg.norm(predicted-point['point_px']))
                    assert abs(error-saved[label]) < 1e-10; errors.append(error)
                    max_warp_difference = max(max_warp_difference, abs(error-saved[label]))
                if point['track_id'] is not None and point['link_status'] != 'adjacent_pair':
                    assert point['track_id'] not in seen_ids
                if point['track_id'] is not None:seen_ids.add(point['track_id'])
            if frame in config['missing_mask_frames']:assert tracks['active_tracks'] == tracks['linked_tracks'] == 0
            if frame == max(config['missing_mask_frames'])+1:assert tracks['linked_tracks'] == 0 and tracks['active_tracks'] > 0
            previous = {p['track_id']: p['point_px'] for p in point_rows if p['track_id'] is not None}
    assert frame+1 == config['frames'] == 75
    for label, values in (('adjacent_point_error_px',adjacent_errors),('original_anchor_error_px',anchor_errors)):
        saved = synthetic['accuracy'][label]
        assert len(values) == saved['count'] and abs(np.percentile(values,95)-saved['p95']) < 1e-10
        assert abs(max(values)-saved['maximum']) < 1e-10
    ledger = read(ROOT/'.local/meta-budget/ledger.json'); reservations = list(ledger['reservations'].values())
    result = {'created_at': local_time(), 'status': 'passed', 'seals': seals, 'runtime_files': len(runtime),
        'unchanged_passing_tests_reused': 12, 'cases': cases, 'source_frames': total_frames,
        'source_object_slots': total_slots, 'raw_current_point_slots': total_points, 'decoded_video_frames': total_decoded,
        'status_counts': dict(all_counts), 'reference_conflicts': conflicts,
        'maximum_descriptor_distance_difference': largest_distance_difference,
        'synthetic_frames': 75, 'independent_known_point_errors': len(adjacent_errors)+len(anchor_errors),
        'maximum_known_warp_difference_px': max_warp_difference,
        'meta_budget': {'calls': len(reservations), 'spent_usd': sum(r['charged_microusd'] for r in reservations)/1e6,
            'limit_usd': ledger['total_budget_microusd']/1e6, 'unsettled': sum(r['status'] != 'settled' for r in reservations)},
        'new_inference': False, 'new_paid_calls': 0, 'new_physics': False, 'seconds': time.perf_counter()-started,
        'scope': 'Independent arithmetic and track-graph consistency; real-video physical identity and accuracy unverified'}
    write_json(out, result)
    print(json.dumps({key: value for key,value in result.items() if key != 'reference_conflicts'}))


if __name__ == '__main__':main()
