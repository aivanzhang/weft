"""Hip-centered geometry diagnosis; static counterfactuals are not physical body estimates."""

from copy import deepcopy
import math
import time

import mujoco
import numpy as np

from weft.body_refinement import JOINT_LANDMARKS, _json_safe, _targets
from weft.human import HumanEnv
from weft.io import canonical_hash


CONTRACT = {'method': 'first-prefix-static-shape-diagnostic-v1', 'training_fraction': .6,
            'maximum_frames': 900, 'maximum_seconds': 60., 'maximum_length_fraction': .15,
            'maximum_attachment_change': .03, 'crop_bounds': [0., 1.],
            'minimum_projected_length_px': 2., 'fit': 'positive-confidence weighted coordinate medians',
            'scope': 'Inferred model-output geometry; no floor objective, dynamics, camera calibration or physical validation'}
CHAINS = tuple((f'{side}_{base}', f'{side}_{middle}', f'{side}_{tip}', first, second)
               for side in ('left', 'right') for base, middle, tip, first, second in
               [('shoulder', 'elbow', 'wrist', 'upper_arm', 'forearm'), ('hip', 'knee', 'ankle', 'thigh', 'shin')])
NAMES = tuple(JOINT_LANDMARKS)
INDEX = {name: i for i, name in enumerate(NAMES)}


def _stats(values):
    values = np.asarray(values, dtype=float)
    return ({'count': len(values), 'rmse': float(np.sqrt(np.mean(values ** 2))), 'median': float(np.median(values)),
             'minimum': float(values.min()), 'maximum': float(values.max()),
             'p10': float(np.percentile(values, 10)), 'p90': float(np.percentile(values, 90))}
            if values.size else {'count': 0, 'rmse': None, 'median': None, 'minimum': None,
                                 'maximum': None, 'p10': None, 'p90': None})


def _median(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    usable = weights > 0
    if not usable.any():
        return None
    values, weights = values[usable], weights[usable]
    order = np.argsort(values, kind='stable')
    return float(values[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)])


def _geometry(model, data, env):
    center = data.site_xpos[env._site_ids['pelvis']]
    positions = np.array([data.site_xpos[env._site_ids[name]] - center for name in NAMES])
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    return positions, data.xmat[torso].reshape(3, 3).copy()


def _counterfactual(positions, rotation, lengths, attachments, variant):
    result = positions.copy()
    for base, middle, tip, _, _ in CHAINS:
        base_i, middle_i, tip_i = map(INDEX.get, (base, middle, tip))
        if variant in ('attachments', 'both'):
            result[base_i] = rotation @ np.array(attachments[base]['bounded_local'])
        for a, b in ((base_i, middle_i), (middle_i, tip_i)):
            vector = positions[b] - positions[a]
            size = lengths[NAMES[a] + '__' + NAMES[b]]['bounded_length'] if variant in ('lengths', 'both') else np.linalg.norm(vector)
            result[b] = result[a] + vector / np.linalg.norm(vector) * size
    return result


def diagnose_shape(spec, evidence, *, width, height, fps, frame_count, model_units_per_pixel):
    """Return complete bounded comparisons without modifying the model or source evidence."""
    started = time.monotonic()
    if spec.get('profile') != 'exercise-v3' or any(key in spec for key in ('reference_refinement', 'image_reference_refinement')):
        raise ValueError('Shape diagnosis requires an original mobile reference')
    if evidence.get('source_sha256') != spec.get('source_sha256'):
        raise ValueError('Shape diagnosis source identity mismatch')
    if any(type(value) is not int or not 16 <= value <= 8192 for value in (width, height)):
        raise ValueError('Verified image dimensions must be bounded integers')
    if type(frame_count) is not int or not 2 <= frame_count <= 1800:
        raise ValueError('Source frame count exceeds the diagnosis bound')
    if (type(fps) not in (int, float) or not math.isfinite(fps) or not 1 <= fps <= 240
            or (frame_count - 1) / fps > 60):
        raise ValueError('Source timing exceeds the diagnosis bound')
    if type(model_units_per_pixel) not in (int, float) or not math.isfinite(model_units_per_pixel) or not 1e-5 <= model_units_per_pixel <= .1:
        raise ValueError('Projection scale must be finite and explicitly supplied')
    frames, references = evidence.get('frames'), spec.get('reference')
    if not isinstance(frames, list) or not isinstance(references, list) or not 2 <= len(references) <= 900 or len(frames) > 900:
        raise ValueError('Shape diagnosis requires at most 900 retained poses')
    by_id = {row.get('id'): row for row in frames if isinstance(row, dict)}
    provenance = {row.get('time_s'): row for row in spec.get('reference_evidence', [])}
    if len(by_id) != len(frames) or len(provenance) != len(references):
        raise ValueError('Every reference needs unique source provenance')
    cutoff = CONTRACT['training_fraction'] * (frame_count - 1) / fps
    env = HumanEnv(spec)
    data = mujoco.MjData(env.model)
    rows, prepared = [], []
    for reference in references:
        if time.monotonic() - started > CONTRACT['maximum_seconds']:
            raise RuntimeError('Shape diagnosis wall-time bound exceeded')
        source_id = provenance[reference['time_s']]['evidence_id']
        frame = by_id.get(source_id)
        if (frame is None or frame.get('time_s') != reference['time_s'] or not isinstance(frame.get('frame'), int)
                or not 0 <= frame['frame'] < frame_count or abs(frame['time_s'] - frame['frame'] / fps) > 1e-6
                or source_id != spec['source_sha256'] + ':frame:' + str(frame['frame'])):
            raise ValueError('Reference source ID/frame/absolute time mismatch')
        target, _, rejected = _targets(frame)
        row = {'evidence_id': source_id, 'frame': frame['frame'], 'time_s': frame['time_s'],
               'partition': 'training' if frame['time_s'] <= cutoff else 'heldout',
               'source_status': frame.get('status'), 'identity_status': frame.get('identity_status'),
               'raw_landmarks': _json_safe(deepcopy(frame.get('landmarks'))),
               'raw_world_landmarks': _json_safe(deepcopy(frame.get('world_landmarks'))),
               'status': rejected or 'compared', 'clamped_flexion': provenance[reference['time_s']].get('clamped_flexion', [])}
        rows.append(row)
        if rejected:
            continue
        image = np.array([[frame['landmarks'][i][key] for key in ('x', 'y')] for i in JOINT_LANDMARKS.values()])
        if not np.isfinite(image).all():
            row['status'] = 'nonfinite_image_coordinates_retained'
            continue
        confidence = np.array([min(frame['landmarks'][i]['visibility'], frame['landmarks'][i]['presence']) for i in JOINT_LANDMARKS.values()])
        cropped = ~((image >= 0) & (image <= 1) & np.isfinite(image)).all(axis=1)
        weights = np.where(cropped, 0., confidence)
        pixel = image * [width, height]
        pixel -= (pixel[INDEX['left_hip']] + pixel[INDEX['right_hip']]) / 2
        data.qpos[env._q_array] = [reference['joints'][name] for name in env.joint_names]
        mujoco.mj_forward(env.model, data)
        positions, rotation = _geometry(env.model, data, env)
        row.update(source_confidence=dict(zip(NAMES, map(float, confidence))),
                   effective_confidence=dict(zip(NAMES, map(float, weights))),
                   outside_image_crop=dict(zip(NAMES, map(bool, cropped))),
                   source_world_hip_centered=target.tolist(), source_pixels_hip_centered=pixel.tolist(),
                   baseline_model_hip_centered=positions.tolist())
        prepared.append((row, positions, rotation, target, pixel, weights))
    training = [row for row in prepared if row[0]['partition'] == 'training']
    if not training:
        raise ValueError('No usable construction-prefix poses; no shape estimate invented')
    lengths, attachments = {}, {}
    for base, middle, tip, first, second in CHAINS:
        for a, b, key in ((base, middle, first), (middle, tip, second)):
            ai, bi = INDEX[a], INDEX[b]
            values = [float(np.linalg.norm(row[3][bi] - row[3][ai])) for row in training]
            weights = [float(min(row[5][ai], row[5][bi])) if value > 1e-6 else 0.
                       for row, value in zip(training, values)]
            estimated = _median(values, weights)
            original = spec['lengths'][key]
            bounded = float(np.clip(estimated, original * .85, original * 1.15)) if estimated is not None else original
            lengths[a + '__' + b] = {'original_length': original, 'estimated_length': estimated, 'bounded_length': bounded,
                'status': 'estimated' if estimated is not None else 'unidentified_kept_original',
                'cap_hit': estimated is not None and estimated != bounded, 'positive_weight_samples': sum(w > 0 for w in weights),
                'training_source_lengths': _stats(values)}
        base_i = INDEX[base]
        source_local = [row[2].T @ row[3][base_i] for row in training]
        weights = [float(min(row[5][base_i], row[5][INDEX['left_hip']], row[5][INDEX['right_hip']])) for row in training]
        estimate = [_median([local[axis] for local in source_local], weights) for axis in range(3)]
        original = training[0][2].T @ training[0][1][base_i]
        if estimate[0] is None:
            bounded = original
        else:
            delta = np.array(estimate) - original
            bounded = original + delta * min(1., .03 / max(1e-12, np.linalg.norm(delta)))
        attachments[base] = {'original_local': original.tolist(), 'estimated_local': estimate,
            'bounded_local': bounded.tolist(), 'positive_weight_samples': sum(w > 0 for w in weights),
            'cap_hit': bool(estimate[0] is not None and np.linalg.norm(np.array(estimate) - original) > .03)}
    # The pelvis origin remains the midpoint, including after asymmetric estimates.
    attachments['right_hip']['bounded_local'] = (-np.array(attachments['left_hip']['bounded_local'])).tolist()
    aggregate = {partition: {variant: {'world': [], 'pixels': [], 'positive_confidence_world': [], 'positive_confidence_pixels': []}
                             for variant in ('baseline', 'lengths', 'attachments', 'both')} for partition in ('training', 'heldout')}
    segment_rows, decomposition_rows = [], []
    for row, positions, rotation, target, pixel, weights in prepared:
        row['comparisons'], row['segments'], row['endpoint_decomposition'] = {}, [], []
        for variant in ('baseline', 'lengths', 'attachments', 'both'):
            predicted = positions if variant == 'baseline' else _counterfactual(positions, rotation, lengths, attachments, variant)
            world_error = np.linalg.norm(predicted - target, axis=1)
            projected = predicted[:, [0, 2]] * [1, -1] / model_units_per_pixel
            image_error = np.linalg.norm(projected - pixel, axis=1)
            row['comparisons'][variant] = {'positions': predicted.tolist(), 'world_endpoint_errors': dict(zip(NAMES, map(float, world_error))),
                                          'pixel_endpoint_errors': dict(zip(NAMES, map(float, image_error)))}
            group = aggregate[row['partition']][variant]
            group['world'].extend(world_error)
            group['pixels'].extend(image_error)
            group['positive_confidence_world'].extend(world_error[weights > 0])
            group['positive_confidence_pixels'].extend(image_error[weights > 0])
        for base, middle, tip, first, second in CHAINS:
            attachment = positions[INDEX[base]] - target[INDEX[base]]
            length_sum, direction_sum = np.zeros(3), np.zeros(3)
            for a, b, key in ((base, middle, first), (middle, tip, second)):
                ai, bi = INDEX[a], INDEX[b]
                sv, mv = target[bi] - target[ai], positions[bi] - positions[ai]
                size = np.linalg.norm(sv)
                unit = sv / size if size > 1e-10 else np.zeros(3)
                length_sum += (spec['lengths'][key] - size) * unit
                direction_sum += mv - spec['lengths'][key] * unit
                predicted_image = mv[[0, 2]] * [1, -1] / model_units_per_pixel
                source_image = pixel[bi] - pixel[ai]
                pnorm, snorm = np.linalg.norm(predicted_image), np.linalg.norm(source_image)
                denominator = pnorm * snorm
                angle = float(np.degrees(np.arccos(np.clip(predicted_image @ source_image / denominator, -1, 1)))) if min(pnorm, snorm) >= 2 else None
                item = {'segment': a + '__' + b, 'source_world_length': float(size), 'fixed_model_length': spec['lengths'][key],
                        'source_pixel_length': float(snorm), 'baseline_projected_length': float(pnorm),
                        'static_length_projected_length': float(pnorm * lengths[a + '__' + b]['bounded_length'] / spec['lengths'][key]),
                        'image_direction_error_degrees': angle, 'effective_confidence': float(min(weights[ai], weights[bi]))}
                row['segments'].append(item)
                segment_rows.append(item | {'partition': row['partition'], 'frame': row['frame']})
                error = positions[bi] - target[bi]
                closure = float(np.linalg.norm(error - attachment - length_sum - direction_sum))
                assert closure < 1e-10
                item = {'joint': b, 'attachment_vector': attachment.tolist(), 'length_vector': length_sum.tolist(),
                        'direction_vector': direction_sum.tolist(), 'total_vector': error.tolist(), 'closure_error': closure}
                row['endpoint_decomposition'].append(item)
                decomposition_rows.append(item | {'partition': row['partition'], 'frame': row['frame']})
    env.close()
    summaries = {partition: {variant: {key: _stats(value) for key, value in values.items()} for variant, values in variants.items()}
                 for partition, variants in aggregate.items()}
    segment_summary = {name: {partition: {'source_world_length': _stats([r['source_world_length'] for r in segment_rows if r['segment'] == name and r['partition'] == partition]),
                         'image_direction_error_degrees': _stats([r['image_direction_error_degrees'] for r in segment_rows if r['segment'] == name and r['partition'] == partition and r['image_direction_error_degrees'] is not None]),
                         'baseline_pixel_length_error': _stats([r['baseline_projected_length'] - r['source_pixel_length'] for r in segment_rows if r['segment'] == name and r['partition'] == partition]),
                         'static_pixel_length_error': _stats([r['static_length_projected_length'] - r['source_pixel_length'] for r in segment_rows if r['segment'] == name and r['partition'] == partition])}
                       for partition in ('training', 'heldout')} for name in lengths}
    return {'schema_version': 1, 'contract': deepcopy(CONTRACT), 'source_sha256': spec['source_sha256'],
            'model_sha256': canonical_hash(spec), 'source_timing': {'fps': fps, 'frame_count': frame_count},
            'projection': {'width': width, 'height': height, 'model_units_per_pixel': model_units_per_pixel,
                           'scope': 'Frozen uncalibrated baseline scale; hip centering removes root and floor translation'},
            'cutoff_time_s': cutoff, 'training_evidence_ids': [r[0]['evidence_id'] for r in training],
            'heldout_evidence_ids': [r[0]['evidence_id'] for r in prepared if r[0]['partition'] == 'heldout'],
            'static_lengths': lengths, 'static_attachments': attachments, 'summary': summaries,
            'segment_summary': segment_summary,
            'decomposition_summary': {term: _stats([np.linalg.norm(r[term + '_vector']) for r in decomposition_rows])
                                      for term in ('attachment', 'length', 'direction', 'total')},
            'rows': rows, 'counts': {'requested': len(references), 'compared': len(prepared), 'rejected_retained': len(rows) - len(prepared)},
            'runtime_seconds': time.monotonic() - started, 'physical_validation': 'unverified', 'promoted': False}
