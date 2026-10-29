"""Static pixel-length hypothesis with frozen pose, projection, root and floor."""

from copy import deepcopy
import math
import time

import mujoco
import numpy as np

from weft.body_shape_diagnostic import CHAINS, INDEX, NAMES, _geometry, _stats
from weft.human import HumanEnv
from weft.io import canonical_hash


CONTRACT = {
    'method': 'shared-static-pixel-length-v1', 'training_fraction': .6,
    'maximum_frames': 900, 'maximum_seconds': 30., 'maximum_length_fraction': .15,
    'minimum_confidence': .5, 'minimum_projection_fraction': .2,
    'minimum_pixel_length': 2., 'minimum_samples': 3, 'prior_weight_fraction': .1,
    'fit': 'confidence times squared projection fraction; ridge toward original ratio 1',
    'scope': 'Shared left/right lengths; fixed angles, scale, attachments, roots and floor; no anatomy claim',
}
LENGTHS = ('upper_arm', 'forearm', 'thigh', 'shin')


def _array(value, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all() or np.max(np.abs(result)) > 1e6:
        raise ValueError('Shape evidence array is invalid or excessive')
    return result


def fit_pixel_shape(spec, diagnosis):
    """Fit only prefix magnitudes and evaluate actual candidate forward kinematics."""
    started = time.monotonic()
    if (spec.get('profile') != 'exercise-v3' or diagnosis.get('model_sha256') != canonical_hash(spec)
            or diagnosis.get('source_sha256') != spec.get('source_sha256')):
        raise ValueError('Pixel shape requires the exact original model and source')
    rows = diagnosis.get('rows')
    if not isinstance(rows, list) or not 2 <= len(rows) <= CONTRACT['maximum_frames']:
        raise ValueError('Pixel shape requires 2–900 retained source poses')
    timing = diagnosis['source_timing']
    fps, count = timing['fps'], timing['frame_count']
    if (type(fps) not in (int, float) or not math.isfinite(fps) or not 1 <= fps <= 240
            or type(count) is not int or not 2 <= count <= 1800 or (count - 1) / fps > 60):
        raise ValueError('Invalid bounded source timing')
    scale = diagnosis['projection']['model_units_per_pixel']
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 1e-5 <= scale <= .1:
        raise ValueError('Invalid frozen projection scale')
    cutoff = CONTRACT['training_fraction'] * (count - 1) / fps
    references = spec['reference']
    if len(rows) != len(references):
        raise ValueError('All source reference slots must be retained')
    prepared, samples = [], {name: [] for name in LENGTHS}
    previous = -1
    for row, ref in zip(rows, references):
        frame, timestamp = row.get('frame'), row.get('time_s')
        if (type(frame) is not int or not previous < frame < count
                or type(timestamp) not in (int, float) or not math.isfinite(timestamp)
                or abs(timestamp - frame / fps) > 1e-6 or timestamp != ref['time_s']
                or row.get('evidence_id') != spec['source_sha256'] + ':frame:' + str(frame)):
            raise ValueError('Source frame identity or timing mismatch')
        previous = frame
        partition = 'training' if timestamp <= cutoff else 'heldout'
        item = {'frame': frame, 'time_s': timestamp, 'evidence_id': row['evidence_id'],
                'partition': partition, 'status': row['status']}
        if row['status'] != 'compared':
            prepared.append((item, None, None, None, None))
            continue
        baseline = _array(row['baseline_model_hip_centered'], (len(NAMES), 3))
        pixels = _array(row['source_pixels_hip_centered'], (len(NAMES), 2))
        world = _array(row['source_world_hip_centered'], (len(NAMES), 3))
        confidence = _array([row['effective_confidence'][name] for name in NAMES], (len(NAMES),))
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError('Confidence must be within [0, 1]')
        prepared.append((item, baseline, pixels, world, confidence))
        if partition != 'training':
            continue
        for base, middle, tip, first, second in CHAINS:
            for a, b, key in ((base, middle, first), (middle, tip, second)):
                ai, bi = INDEX[a], INDEX[b]
                vector = baseline[bi] - baseline[ai]
                length = float(np.linalg.norm(vector))
                if not math.isclose(length, spec['lengths'][key], rel_tol=1e-7, abs_tol=1e-10):
                    raise ValueError('Baseline geometry differs from original model length')
                projected = float(np.linalg.norm(vector[[0, 2]]))
                observed = float(np.linalg.norm(pixels[bi] - pixels[ai]))
                weight = float(min(confidence[ai], confidence[bi]))
                fraction = projected / length
                reason = ('low_confidence' if weight < CONTRACT['minimum_confidence'] else
                          'foreshortened' if fraction < CONTRACT['minimum_projection_fraction'] else
                          'short_image_segment' if min(projected / scale, observed) < CONTRACT['minimum_pixel_length'] else None)
                samples[key].append({'frame': frame, 'segment': a + '__' + b, 'reason': reason,
                                     'weight': weight * fraction ** 2 if reason is None else 0.,
                                     'ratio': observed * scale / projected if reason is None else None})
    candidate, estimates = deepcopy(spec), {}
    for name, observations in samples.items():
        usable = [row for row in observations if row['reason'] is None]
        identified = len(usable) >= CONTRACT['minimum_samples']
        weight = sum(row['weight'] for row in usable)
        prior = CONTRACT['prior_weight_fraction'] * weight
        ratio = ((sum(row['weight'] * row['ratio'] for row in usable) + prior) / (weight + prior)
                 if identified else 1.)
        bounded = float(np.clip(ratio, 1 - CONTRACT['maximum_length_fraction'], 1 + CONTRACT['maximum_length_fraction']))
        candidate['lengths'][name] *= bounded
        estimates[name] = {'status': 'estimated' if identified else 'unidentified_kept_original',
                           'unbounded_ratio': ratio if identified else None, 'bounded_ratio': bounded,
                           'cap_hit': bounded != ratio, 'original_length': spec['lengths'][name],
                           'candidate_length': candidate['lengths'][name], 'samples': observations}
    # The old clearance claim belongs to the old geometry; preserve its offset, not its claim.
    old_floor_fit = candidate['mobility'].pop('floor_fit', None)
    groups = {partition: {variant: {'world': [], 'pixels': []} for variant in ('baseline', 'candidate')}
              for partition in ('training', 'heldout')}
    env = HumanEnv(candidate)
    try:
        data = mujoco.MjData(env.model)
        for (row, baseline, pixels, world, confidence), reference in zip(prepared, references):
            if time.monotonic() - started > CONTRACT['maximum_seconds']:
                raise RuntimeError('Pixel shape wall-time bound exceeded')
            if baseline is None:
                continue
            data.qpos[env._q_array] = [reference['joints'][name] for name in env.joint_names]
            mujoco.mj_forward(env.model, data)
            positions, _ = _geometry(env.model, data, env)
            row['comparisons'] = {}
            for name, geometry in (('baseline', baseline), ('candidate', positions)):
                errors = {'world': np.linalg.norm(geometry - world, axis=1),
                          'pixels': np.linalg.norm(geometry[:, [0, 2]] * [1, -1] / scale - pixels, axis=1)}
                row['comparisons'][name] = {'positions': geometry.tolist(),
                    'endpoint_errors': {key: dict(zip(NAMES, map(float, values))) for key, values in errors.items()}}
                for key, values in errors.items():
                    groups[row['partition']][name][key].extend(values)
    finally:
        env.close()
    return {'schema_version': 1, 'contract': deepcopy(CONTRACT), 'source_sha256': spec['source_sha256'],
            'original_model_sha256': canonical_hash(spec), 'diagnosis_sha256': canonical_hash(diagnosis),
            'candidate': candidate, 'candidate_model_sha256': canonical_hash(candidate),
            'original_floor_fit_not_reapplied': old_floor_fit, 'estimates': estimates,
            'summary': {part: {name: {key: _stats(values) for key, values in errors.items()}
                              for name, errors in variants.items()} for part, variants in groups.items()},
            'rows': [row[0] for row in prepared], 'cutoff_time_s': cutoff,
            'runtime_seconds': time.monotonic() - started, 'physical_validation': 'unverified', 'promoted': False}
