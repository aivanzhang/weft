"""Compare frozen image predictions without fitting alignment or hiding missing slots."""

import math

import numpy as np


def _pixel(value):
    if value is None:
        return None
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all() or np.max(np.abs(point)) > 1e5:
        raise ValueError('Pixels must be finite bounded pairs, or explicitly unavailable')
    return point


def _stats(values):
    if not values:
        return {'rmse_px': None, 'median_px': None, 'p95_px': None, 'maximum_px': None}
    a = np.asarray(values)
    return {'rmse_px': float(np.sqrt(np.mean(a*a))), 'median_px': float(np.median(a)),
            'p95_px': float(np.percentile(a, 95)), 'maximum_px': float(a.max())}


def grade_projected_joints(predictions, observations, *, baseline, limit_px):
    if (not isinstance(predictions, dict) or not 1 <= len(predictions) <= 4 or baseline not in predictions
            or not isinstance(observations, list) or not 1 <= len(observations) <= 1800*33
            or type(limit_px) not in (int, float) or not math.isfinite(limit_px) or not 0 < limit_px <= 100):
        raise ValueError('Evaluation requires bounded methods, complete slots and a fixed positive pixel limit')
    keys = [(r['frame'], r['joint']) for r in observations]
    if (len(set(keys)) != len(keys) or keys != sorted(keys)
            or any(type(f) is not int or not 0 <= f < 1800 or type(j) is not int or not 0 <= j < 33 for f, j in keys)):
        raise ValueError('Evaluation frame/joint slots must be unique, ordered and bounded')
    observed = [_pixel(r['pixel']) for r in observations]
    errors, points, summary = {}, {}, {}
    for name, rows in predictions.items():
        if (not isinstance(name, str) or not 1 <= len(name) <= 64 or not isinstance(rows, list)
                or [(r['frame'], r['joint']) for r in rows] != keys):
            raise ValueError('Every method must preserve the same complete frame/joint slots')
        points[name] = [_pixel(r['pixel']) for r in rows]
        errors[name] = [float(np.linalg.norm(p-o)) if p is not None and o is not None else None
                        for p, o in zip(points[name], observed, strict=True)]
        paired = [e for e in errors[name] if e is not None]
        summary[name] = {'total_slots': len(keys), 'observed_slots': sum(o is not None for o in observed),
            'predicted_slots': sum(p is not None for p in points[name]), 'paired_slots': len(paired),
            'observation_without_prediction': sum(o is not None and p is None for o, p in zip(observed, points[name], strict=True)),
            'prediction_without_observation': sum(o is None and p is not None for o, p in zip(observed, points[name], strict=True)),
            'within_limit': sum(e <= limit_px for e in paired), 'outside_limit': sum(e > limit_px for e in paired),
            **_stats(paired)}
    comparisons = {}
    for name in predictions:
        if name == baseline:
            continue
        common = [(a, b) for a, b in zip(errors[baseline], errors[name], strict=True) if a is not None and b is not None]
        comparisons[name] = {'shared_slots': len(common), 'baseline_on_shared': _stats([r[0] for r in common]),
            'candidate_on_shared': _stats([r[1] for r in common]),
            'improved_slots': sum(b < a-1e-9 for a, b in common), 'worsened_slots': sum(b > a+1e-9 for a, b in common),
            'newly_covered_observations': sum(a is None and b is not None for a, b in zip(errors[baseline], errors[name], strict=True)),
            'lost_observed_predictions': sum(a is not None and b is None for a, b in zip(errors[baseline], errors[name], strict=True))}
    rows = [{'frame': f, 'joint': j, 'observation_available': observed[i] is not None,
        'errors_px': {name: values[i] for name, values in errors.items()}} for i, (f, j) in enumerate(keys)]
    return {'limit_px': limit_px, 'baseline': baseline, 'methods': summary, 'paired_comparisons': comparisons,
        'all_slot_errors': rows, 'alignment_fitted': False,
        'scope': 'Image agreement conditional on observation quality; not measured 3D anatomy or physical validation.'}
