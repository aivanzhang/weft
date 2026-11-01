"""Bounded pose evidence sampling; this never changes animation curves."""

import math


MAX_SAMPLES = 8
MAX_KEYS = 32
AUTHORED_FPS = 24


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Motion samples require finite numbers')
    return float(value)


def _authored_keys(design):
    keys = design.get('keys') if isinstance(design, dict) else None
    if not isinstance(keys, list) or not 2 <= len(keys) <= MAX_KEYS:
        raise ValueError('Motion sampling requires 2–32 authored keys')
    seconds, poses = [], []
    for key in keys:
        if not isinstance(key, dict) or set(key) != {'second', 'rotations'}:
            raise ValueError('Invalid authored sample key')
        second = _number(key['second'])
        if not 0 <= second <= 10 or (seconds and second <= seconds[-1]):
            raise ValueError('Authored sample seconds must strictly increase within 0–10')
        rotations = key['rotations']
        if not isinstance(rotations, dict) or len(rotations) > 64:
            raise ValueError('Motion sampling supports at most 64 rotations per key')
        pose = {}
        for bone, values in rotations.items():
            if not isinstance(bone, str) or not isinstance(values, (list, tuple)) or len(values) != 3:
                raise ValueError('Sample rotations require named three-angle values')
            for axis, value in enumerate(values):
                angle = _number(value)
                if abs(angle) > 60:
                    raise ValueError('Sample rotations exceed the authored angle limit')
                pose[bone, axis] = angle
        seconds.append(second)
        poses.append(pose)
    return seconds, poses


def _select_keys(seconds, poses):
    if len(seconds) <= MAX_SAMPLES:
        return list(range(len(seconds)))
    selected = [0, len(seconds)-1]
    while len(selected) < MAX_SAMPLES:
        candidates = []
        for left, right in zip(selected, selected[1:]):
            for index in range(left+1, right):
                fraction = (seconds[index]-seconds[left])/(seconds[right]-seconds[left])
                axes = poses[left].keys() | poses[index].keys() | poses[right].keys()
                # Interpolation error exposes brief gestures that uniform samples can miss.
                error = sum((poses[index].get(axis, 0) -
                             ((1-fraction)*poses[left].get(axis, 0) +
                              fraction*poses[right].get(axis, 0)))**2 for axis in sorted(axes))
                coverage = min(seconds[index]-seconds[left], seconds[right]-seconds[index])
                candidates.append((error, coverage, -index, index))
        selected.append(max(candidates)[-1])
        selected.sort()
    return selected


def sample_action(design, frame_range, fps=24):
    """Use authored 24fps key times, or preserve legacy five-fraction sampling."""
    if not isinstance(frame_range, (tuple, list)) or len(frame_range) != 2:
        raise ValueError('Motion samples require a frame range')
    start, end = map(_number, frame_range)
    fps = _number(fps)
    if end <= start or fps <= 0:
        raise ValueError('Static or invalid action cannot provide motion samples')
    if design is None:
        frames = [start+(end-start)*fraction for fraction in (0, .25, .5, .75, 1)]
        return {'frames': frames, 'seconds': [frame/fps for frame in frames],
                'sampling_method': 'legacy_uniform_five'}
    seconds, poses = _authored_keys(design)
    frames = [second*AUTHORED_FPS for second in seconds]
    if not (math.isclose(frames[0], start, rel_tol=1e-6, abs_tol=1e-4) and
            math.isclose(frames[-1], end, rel_tol=1e-6, abs_tol=1e-4)):
        raise ValueError('Authored key times differ from the actual action range')
    selected = _select_keys(seconds, poses)
    return {'frames': [frames[index] for index in selected],
            'seconds': [seconds[index] for index in selected],
            'sampling_method': ('authored_keyframes' if len(seconds) <= MAX_SAMPLES else
                                'authored_keyframes_bounded_pose_error')}
