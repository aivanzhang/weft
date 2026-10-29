"""Audit saved appearance matches without repeating extraction, inference or physics."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'artifacts/surface-feature-diversity-01'


def read(path):
    return json.loads(path.read_text())


def arrays(archive, locator):
    values = {key: archive[f'{locator["prefix"]}_{key}']
        for key in ('keypoints', 'descriptors', 'selected_indices', 'aliases')}
    points, descriptors = values['keypoints'], values['descriptors']
    assert points.shape == (len(points), 6) and len(points) <= 128
    assert descriptors.shape == (len(points), 128) and descriptors.dtype == np.float32
    assert np.isfinite(points).all() and np.isfinite(descriptors).all()
    assert (descriptors >= 0).all() and (descriptors <= 255).all()
    selected, aliases = [], []
    for index, point in enumerate(points):
        nearby = [other for other in selected if np.linalg.norm(point[:2]-points[other, :2]) <= 1.]
        aliases.append(nearby[0] if nearby else index)
        if not nearby:selected.append(index)
    assert values['selected_indices'].tolist() == selected
    assert values['aliases'].tolist() == aliases
    return values


def check_matches(reference, current, result):
    first, second = reference['selected_indices'].astype(int), current['selected_indices'].astype(int)
    assert len(result['rows']) == len(reference['keypoints'])
    assert result['physical_point_identity_verified'] is False and result['rigidity_verified'] is False
    distances = None
    if min(len(first), len(second)) >= 2:
        left, right = reference['descriptors'][first].astype(float), current['descriptors'][second].astype(float)
        # Double-precision distances are independent of OpenCV's saved nearest-neighbor results.
        distances = np.sqrt(np.maximum(0, (left*left).sum(1)[:, None]+(right*right).sum(1)[None, :]-2*left@right.T))
        forward = np.argsort(distances, axis=1, kind='stable')[:, :2]
        reverse = np.argsort(distances, axis=0, kind='stable')[:2, :]
    counts = Counter(); max_distance_difference = 0.
    for index, row in enumerate(result['rows']):
        assert row['reference_id'] == index
        assert row['reference_point_px'] == reference['keypoints'][index, :2].tolist()
        alias = int(reference['aliases'][index]); target = None
        if alias != index:
            expected = 'duplicate_reference_location'; assert row['alias_of'] == alias
        elif distances is None:
            expected = 'insufficient_features'
        else:
            compact = int(np.flatnonzero(first == index)[0]); best, runner_up = forward[compact]
            opposite, competitor = reverse[:, best]
            measured = [distances[compact, best], distances[compact, runner_up],
                distances[opposite, best], distances[competitor, best]]
            reported = [row[key] for key in ('descriptor_distance', 'second_distance', 'reverse_distance', 'reverse_second_distance')]
            difference = float(np.max(np.abs(np.asarray(measured)-reported)))
            max_distance_difference = max(max_distance_difference, difference)
            assert difference < 1e-4 and row['nearest_current_id'] == int(second[best])
            if measured[0] > 220.:expected = 'appearance_distance_rejected'
            elif not measured[0] < .7*measured[1]:expected = 'forward_ambiguity'
            elif opposite != compact:expected = 'not_mutual'
            elif not measured[2] < .7*measured[3]:expected = 'reverse_ambiguity'
            else:expected = 'appearance_pair'; target = int(second[best])
        assert row['status'] == expected and row['current_id'] == target
        assert row['current_point_px'] == (None if target is None else current['keypoints'][target, :2].tolist())
        counts[expected] += 1
    assert result['appearance_pairs'] == counts['appearance_pair']
    return counts, max_distance_difference


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.exists() or out.parent != ROOT/'.local':raise ValueError('Use a new local verification file')
    started = time.perf_counter(); seal = verify_artifacts(STUDY)
    report, protocol = read(STUDY/'report.json'), read(STUDY/'pre-run.json')
    runtime = {name: digest(path) for name, path in _implementation_files().items()}
    assert runtime == protocol['implementation'] and report['status'] == 'complete'
    assert digest(ROOT/'tests/test_surface_features.py') == digest(STUDY/'test_surface_features.py')
    assert digest(ROOT/'.local/surface-features-tests-01.xml') == protocol['test_report_sha256']
    tests = ET.parse(ROOT/'.local/surface-features-tests-01.xml').getroot().find('testsuite')
    assert tests.attrib['tests'] == '15' and tests.attrib['errors'] == tests.attrib['failures'] == tests.attrib['skipped'] == '0'
    results = {}; total_frames = total_objects = total_features = total_pairs = total_video = 0
    for name, expected in report['cases'].items():
        directory = STUDY/name; binding = read(directory/'source-binding.json')
        assert digest(ROOT/binding['source']) == binding['sha256']
        mask_rows = read(directory/'original-mask-slots.json')
        masks = {(row['frame'], row['object_id']): row for row in mask_rows}
        assert len(masks) == len(mask_rows)
        references = {}; counts = Counter(); object_slots = feature_slots = frames = 0
        maximum_difference = 0.; archive = None; archive_name = None; later = {obj['id']: [] for obj in binding['objects']}
        try:
            with (directory/'all-frame-tracks.jsonl').open() as stream:
                for frame, line in enumerate(stream):
                    row = json.loads(line)
                    assert row['frame'] == frame and row['time_s'] == binding['times_s'][frame]
                    assert [obj['object_id'] for obj in row['objects']] == [obj['id'] for obj in binding['objects']]
                    for obj, definition in zip(row['objects'], binding['objects'], strict=True):
                        identifier = obj['object_id']; source = masks[(frame, identifier)]
                        assert obj['source_observation_id'] == source['id'] and obj['source_identity'] == source['identity']
                        assert obj['mask_status'] == source['mask_status']; object_slots += 1
                        initialization = definition['initialization']['frame']
                        if frame < initialization:
                            assert obj['status'] == 'before_initialization' and obj['features'] is obj['matches'] is None
                            continue
                        locator = obj['features']
                        if locator['file'] != archive_name:
                            if archive is not None:archive.close()
                            archive = np.load(directory/locator['file'], allow_pickle=False); archive_name = locator['file']
                        current = arrays(archive, locator)
                        if frame == initialization:references[identifier] = current
                        statuses, difference = check_matches(references[identifier], current, obj['matches'])
                        counts.update(statuses); feature_slots += len(obj['matches']['rows'])
                        maximum_difference = max(maximum_difference, difference)
                        if frame > initialization:later[identifier].append(obj['matches']['appearance_pairs'])
                    frames += 1
        finally:
            if archive is not None:archive.close()
        assert frames == expected['source_frames'] == expected['frames_decoded'] == len(binding['times_s'])
        assert object_slots == len(mask_rows) == expected['source_object_slots']
        assert feature_slots == expected['feature_slots'] and dict(counts) == expected['matching_status_counts']
        for identifier, values in later.items():
            target = expected['by_object'][identifier]
            assert len(values) == target['later_frames'] and sum(values) == target['later_pairs']
            assert sum(v > 0 for v in values) == target['later_frames_with_pairs']
            assert sum(v >= 4 for v in values) == target['later_frames_with_at_least_four_pairs']
        video = cv2.VideoCapture(str(directory/'appearance-tracks.mp4')); decoded = 0
        try:
            assert video.isOpened()
            while True:
                ok, image = video.read()
                if not ok:break
                assert image.shape == (460, 1280, 3); decoded += 1
        finally:video.release()
        assert decoded == frames
        total_frames += frames; total_objects += object_slots; total_features += feature_slots
        total_pairs += counts['appearance_pair']; total_video += decoded
        results[name] = {'frames': frames, 'object_slots': object_slots, 'reference_point_slots': feature_slots,
            'appearance_pairs_including_initialization': counts['appearance_pair'], 'decoded_preview_frames': decoded,
            'maximum_descriptor_distance_difference': maximum_difference}
    ledger = read(ROOT/'.local/meta-budget/ledger.json'); reservations = list(ledger['reservations'].values())
    result = {'created_at': local_time(), 'status': 'passed', 'study': str(STUDY.relative_to(ROOT)), 'seal': seal,
        'runtime_files': len(runtime), 'unchanged_passing_tests_reused': 15, 'cases': results,
        'total_frames': total_frames, 'total_object_slots': total_objects, 'total_reference_point_slots': total_features,
        'total_appearance_pairs_including_initialization': total_pairs, 'decoded_preview_frames': total_video,
        'meta_budget': {'calls': len(reservations), 'spent_usd': sum(r['charged_microusd'] for r in reservations)/1e6,
            'limit_usd': ledger['total_budget_microusd']/1e6, 'unsettled': sum(r['status'] != 'settled' for r in reservations)},
        'seconds': time.perf_counter()-started, 'new_inference': False, 'new_physics': False, 'new_paid_calls': 0,
        'scope': 'Artifact integrity, complete accounting and independent descriptor arithmetic; not verified physical correspondence'}
    write_json(out, result); print(json.dumps(result))


if __name__ == '__main__':main()
