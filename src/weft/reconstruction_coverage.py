"""Full-interval reconstruction planning; missing observations are never zero error."""

import math


def plan_reconstruction(rows, object_ids, interval, fps, *, initialization_frame=None, minimum_joint_coverage=0.):
    first, last = interval
    if (any(type(value) is not int for value in interval) or not 0 <= first <= last <= 21799
            or last - first >= 1800 or not math.isfinite(fps) or not 1 <= fps <= 240
            or (last - first + 1) / fps > 60):
        raise ValueError('Reconstruction requires a bounded source interval of at most 1800 frames / 60 seconds')
    if not 1 <= len(object_ids) <= 8 or len(set(object_ids)) != len(object_ids):
        raise ValueError('Reconstruction requires 1–8 unique objects')
    if initialization_frame is not None and not first <= initialization_frame <= last:
        raise ValueError('Tracking initialization must lie in the source interval')
    by_frame, ids = {}, set(object_ids)
    for row in rows:
        index, object_id = row['frame'], row['object_id']
        if (type(index) is not int or not first <= index <= last or object_id not in ids
                or not math.isfinite(row['time_s']) or abs(row['time_s'] - index / fps) > 1e-6):
            raise ValueError('Measurement frame, identity or nominal source time is invalid')
        group = by_frame.setdefault(index, {})
        if object_id in group:
            raise ValueError('Duplicate object measurement in a source frame')
        group[object_id] = row
    joint = sorted(index for index, group in by_frame.items() if set(group) == ids)
    origin = joint[0] if joint else None
    count = last - first + 1
    denominator = count - (initialization_frame is not None)
    coverage = len(joint) / max(1, denominator)
    episodes = []
    for index in joint:
        if not episodes or index - episodes[-1][-1] > max(2, round(fps * .2)):
            episodes.append([])
        episodes[-1].append(index)
    # Time, not successful detection count, determines the evidence split.
    cutoff = origin + max(6, int((last - origin + 1) * .6)) - 1 if origin is not None else None
    generation, development, timeline = [], [], []
    for index in range(first, last + 1):
        observed = by_frame.get(index, {})
        partition = ('before_model_initialization' if origin is None or index < origin else
                     'generation' if index <= cutoff else 'development_validation')
        if partition == 'generation':
            generation.extend(observed.values())
        elif partition == 'development_validation':
            development.extend(observed.values())
        timeline.append({'frame': index, 'time_s': index / fps, 'partition': partition,
                         'observed_object_ids': sorted(observed), 'missing_object_ids': sorted(ids - observed.keys())})
    per_object = {object_id: {'generation_observations': sum(row['object_id'] == object_id for row in generation),
                             'development_observations': sum(row['object_id'] == object_id for row in development)}
                  for object_id in object_ids}
    reasons = []
    if coverage < minimum_joint_coverage:
        reasons.append('Appearance matching covers less than 80% of the authored interval')
    if not episodes or len(episodes[0]) < 10:
        reasons.append('Need at least ten jointly tracked frames in the initial continuous scene; later episodes cannot replace the beginning')
    if any(value['generation_observations'] < 5 or value['development_observations'] < 1 for value in per_object.values()):
        reasons.append('Every object requires five generation and one later development observations')
    report = {'schema_version': 1, 'method': 'full-requested-interval-v1',
              'status': 'unavailable' if reasons else 'planned', 'reasons': reasons,
              'requested_interval': [first, last], 'requested_frames': count,
              'model_interval': [origin, last] if origin is not None else None,
              'simulation_frames': last - origin + 1 if origin is not None else 0,
              'pre_model_frames': origin - first if origin is not None else count,
              'observed_frames': len(by_frame), 'joint_observed_frames': len(joint),
              'joint_tracking_coverage': coverage, 'minimum_required': minimum_joint_coverage,
              'observation_slots': count * len(ids), 'measured_observation_slots': len(rows),
              'missing_observation_slots': count * len(ids) - len(rows),
              'tracking_initialization_frame': initialization_frame,
              'generation_end_frame': cutoff, 'per_object': per_object,
              'joint_episodes': [[episode[0], episode[-1], len(episode)] for episode in episodes],
              'frames': timeline,
              'scope': 'Every requested frame retained; one uninterrupted simulation after common initialization, no reset at gaps. Missing positions are not interpolated or scored as zero error. Nominal frame/fps timing, not metric camera calibration.'}
    return report, generation, development


def coverage_summary(report):
    return {key: value for key, value in report.items() if key not in {'frames', 'per_object', 'joint_episodes'}}
