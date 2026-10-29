"""Compact inspection of saved pose coverage; never a reconstruction-accuracy score."""

from collections import Counter
import json
import math
from pathlib import Path
import re


USABLE = {'tracked', 'tracked_partial'}
TERMINAL = {'ready', 'failed', 'cancelled'}


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'Invalid {name}')
    return value


def _time(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 60:
        raise ValueError('Invalid source timestamp')
    return float(value)


def summarize_coverage(evidence, root=None):
    if not isinstance(evidence, dict) or evidence.get('profile') != 'exercise-pose-v1' or type(evidence.get('schema_version')) is not int or evidence['schema_version'] != 1:
        raise ValueError('Expected original exercise pose evidence')
    source = evidence.get('source_sha256')
    if not isinstance(source, str) or not re.fullmatch(r'[0-9a-f]{64}', source):
        raise ValueError('Missing pinned video identity')
    metadata = _integer(evidence.get('frame_count'), 'metadata frame count', 10, 1800)
    decoded = _integer(evidence.get('decoded_frames'), 'decoded frame count', 1, 1800)
    fps = evidence.get('fps')
    if type(fps) not in (int, float) or not math.isfinite(fps) or not 1 <= fps <= 240:
        raise ValueError('Invalid source frame rate')
    frames = evidence.get('frames')
    if not isinstance(frames, list) or not 1 <= len(frames) <= min(decoded, 900):
        raise ValueError('Expected 1–900 retained pose samples')
    markers, issues, counts, identities = [], [], Counter(), {}
    previous, missing = -1, []

    def finish_missing():
        if missing:
            issues.append({'kind': 'unusable_pose', 'first_frame': missing[0]['frame'],
                'last_frame': missing[-1]['frame'], 'seek_s': missing[0]['seek_s'],
                'samples': len(missing)})
            missing.clear()

    for row in frames:
        if not isinstance(row, dict):
            raise ValueError('Pose sample must be a record')
        index = _integer(row.get('frame'), 'source frame', 0, decoded - 1)
        stamp = _time(row.get('time_s'))
        if index <= previous or abs(stamp - index / fps) > 1e-7 or row.get('id') != f'{source}:frame:{index}':
            raise ValueError('Source identities or clocks changed')
        seek = _time(row['decoder_timestamp_s']) if row.get('decoder_timestamp_s') is not None else stamp
        status = row.get('status')
        if not isinstance(status, str) or len(status) > 80:
            raise ValueError('Invalid pose status')
        category = status if status in USABLE else 'competing' if status == 'multiple_people' else 'unusable'
        counts[category] += 1
        markers.append([index, seek, category])
        identities[row['id']] = row
        if status in USABLE:
            finish_missing()
        else:
            missing.append({'frame': index, 'seek_s': seek})
        previous = index
    finish_missing()
    root_summary = {'status': 'not_available', 'long_gaps': None, 'unfilled_gaps': None}
    if root is not None:
        if not isinstance(root, dict) or root.get('profile') != 'image-pelvis-root-reference-v1' or root.get('source_sha256') != source:
            raise ValueError('Movement audit belongs to a different source or method')
        quality = root.get('quality', {})
        if not isinstance(quality, dict):
            raise ValueError('Missing movement-quality record')
        long_gaps, unfilled = quality.get('long_reference_intervals'), quality.get('gaps')
        if (not isinstance(long_gaps, list) or not isinstance(unfilled, list)
                or len(long_gaps) > 899 or len(unfilled) > 900):
            raise ValueError('Movement audit exceeds the pose bound')
        maximum = 0.
        for gap in long_gaps:
            if not isinstance(gap, dict):
                raise ValueError('Movement gap must be a record')
            left = identities.get(gap.get('left_evidence_id'))
            right = identities.get(gap.get('right_evidence_id'))
            if left is None or right is None or left['frame'] >= right['frame']:
                raise ValueError('Movement gap does not match retained source poses')
            span = _time(gap.get('seconds'))
            if abs(right['time_s'] - left['time_s'] - span) > 1e-7:
                raise ValueError('Movement gap clock changed')
            maximum = max(maximum, span)
            issues.append({'kind': 'long_movement_gap', 'first_frame': left['frame'], 'last_frame': right['frame'],
                'seek_s': _time(left.get('decoder_timestamp_s') if left.get('decoder_timestamp_s') is not None else left['time_s']),
                'seconds': span})
        unknown = 0
        for gap in unfilled:
            if not isinstance(gap, dict):
                raise ValueError('Movement gap must be a record')
            if gap.get('status') != 'unfilled':
                continue
            rows = [identities.get(identifier) for identifier in gap.get('evidence_ids', [])]
            if not rows or any(row is None for row in rows) or len(rows) > len(frames):
                raise ValueError('Unfilled movement references do not match source poses')
            unknown += 1
            first, last = rows[0], rows[-1]
            issues.append({'kind': 'unfilled_body_position', 'first_frame': first['frame'], 'last_frame': last['frame'],
                'seek_s': _time(first.get('decoder_timestamp_s') if first.get('decoder_timestamp_s') is not None else first['time_s']),
                'samples': len(rows)})
        root_summary = {'status': root.get('status'), 'long_gaps': len(long_gaps), 'unfilled_gaps': unknown,
            'maximum_gap_s': maximum, 'scope': 'Saved image-based movement check, not physical body approval'}
    quality = evidence.get('quality', {})
    if not isinstance(quality, dict):
        raise ValueError('Invalid saved pose-quality record')
    errors = quality.get('errors', [])
    if not isinstance(errors, list) or len(errors) > 20 or any(not isinstance(value, str) or len(value) > 1000 for value in errors):
        raise ValueError('Invalid saved pose-quality errors')
    return {'schema_version': 1, 'status': 'available', 'source_sha256': source,
        'metadata_frames': metadata, 'decoded_frames': decoded, 'sampled_frames': len(frames),
        'unsampled_frames': decoded - len(frames), 'usable_samples': counts['tracked'] + counts['tracked_partial'],
        'partial_samples': counts['tracked_partial'], 'unusable_samples': counts['unusable'] + counts['competing'],
        'competing_samples': counts['competing'], 'markers': markers,
        'issues': sorted(issues, key=lambda row: (row['first_frame'], row['last_frame'], row['kind'])),
        'movement': root_summary, 'quality_errors': errors,
        'scope': 'Model-reported visibility, not anatomical accuracy, measured muscles or physical contact'}


def load_coverage(run):
    run = Path(run).resolve()

    def read(name, limit, required=True):
        path = (run / name).resolve()
        if not path.is_relative_to(run):
            raise ValueError('Coverage input must stay inside this run')
        if not path.exists() and not required:
            return None
        if not path.is_file() or path.stat().st_size > limit:
            raise ValueError('Missing or oversized pose inspection artifact')
        return json.loads(path.read_text())

    try:
        return summarize_coverage(read('evidence.json', 32 * 1024 * 1024),
                                  read('root-motion.json', 16 * 1024 * 1024, required=False))
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as error:
        return {'status': 'unavailable', 'reason': str(error),
                'scope': 'No completeness or accuracy claim; inspect the preserved source files'}
