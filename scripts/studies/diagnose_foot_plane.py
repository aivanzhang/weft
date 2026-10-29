"""Read-only foot-plane identifiability check, independent of floor or simulation scores."""

from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POSE = ROOT / 'artifacts/fullbody-exercise-01/baseline/run/pose-landmarks.json'
MANUAL = ROOT / 'data/exercise/bodyweight-squats-danielflefil-500.manual.json'
OUT = ROOT / 'artifacts/fullbody-foot-plane-diagnostic-01'
TRANSFORM = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
FEET = [29, 30, 31, 32]
TORSO = [11, 12, 23, 24]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(name, data):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def angle(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1, 1))))


def stats(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return None
    return {'min': float(a.min()), 'median': float(np.median(a)),
            'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


def fit(accepted):
    if not accepted:
        return None
    centered = np.concatenate([r['centered'] for r in accepted])
    _, spectrum, axes = np.linalg.svd(centered, full_matrices=False)
    normal = axes[-1]
    up = np.mean([r['up'] for r in accepted], axis=0)
    up /= np.linalg.norm(up)
    if np.dot(normal, up) < 0:
        normal = -normal
    residual = centered @ normal
    per_frame = [np.sqrt(np.mean((r['centered'] @ normal)**2)) for r in accepted]
    return {'normal_sim': normal.tolist(), 'normal_mediapipe': (TRANSFORM.T @ normal).tolist(),
            'singular_values': spectrum.tolist(), 'covariance_eigenvalues': (spectrum**2/len(centered)).tolist(),
            'second_to_first_singular_ratio': float(spectrum[1]/spectrum[0]),
            'third_to_second_singular_ratio': float(spectrum[2]/spectrum[1]),
            'rms_plane_residual_model_units': float(np.sqrt(np.mean(residual**2))),
            'absolute_residual_model_units': stats(np.abs(residual)),
            'per_frame_rms_model_units': stats(per_frame),
            'per_frame_normal_angle_deg': stats([angle(normal, r['normal']) for r in accepted]),
            'camera_up_tilt_deg': angle(normal, np.array([0., 0., 1.])),
            'torso_up_alignment': float(np.dot(normal, up)), 'normal_z_passes_025': bool(normal[2] >= .25),
            'frames': [r['frame'] for r in accepted]}


def main():
    OUT.mkdir(exist_ok=False)
    inputs = OUT / 'inputs'
    inputs.mkdir()
    for path in [POSE, MANUAL, Path(__file__)]:
        shutil.copyfile(path, inputs / path.name)
    save('pre-analysis-freeze.json', {'created_at': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
          'input_hashes': {p.name: digest(p) for p in inputs.iterdir()},
          'construction_observation_indices': [0, 19], 'construction_source_frames': [0, 38],
          'diagnostic_only_observation_indices': [20, 33], 'diagnostic_source_frames': [40, 66],
          'foot_landmarks': FEET, 'confidence_minimum': .5, 'image_bounds_inclusive': [0, 1],
          'confidence_definition': 'Minimum presence/visibility across image and world records for each endpoint',
          'per_frame_centering': True, 'numeric_second_singular_minimum': 1e-6,
          'numeric_second_to_first_ratio_minimum': 1e-3,
          'rank_scope': 'Numerical noncollinearity only; physical conditioning/spread reported, not assumed',
          'sign': 'Reliable shoulder-midpoint minus hip-midpoint, confidence>=.5; otherwise reject',
          'required_signed_normal_sim_z': .25, 'camera_to_sim': TRANSFORM.tolist(),
          'heldout_and_floor_scores_used_for_construction': False,
          'scope': 'Provisional geometry diagnostic; no calibration or model change'})
    evidence = json.loads(POSE.read_text())
    assert [r['frame'] for r in evidence['frames']] == list(range(0, 68, 2))
    accepted, details, rejects = [], [], Counter()
    for ordinal, frame in enumerate(evidence['frames']):
        record = {'frame': frame['frame'], 'id': frame['id'], 'partition': 'construction' if ordinal < 20 else 'diagnostic_only'}
        image, world = frame.get('landmarks', []), frame.get('world_landmarks', [])
        reason = None
        if len(image) != 33 or len(world) != 33:
            reason = 'missing_landmarks'
        else:
            confidence = {i: min(float(lms[i].get(k, 0)) for lms in [image, world]
                                  for k in ['presence', 'visibility']) for i in FEET + TORSO}
            record['endpoint_confidence'] = confidence
            record['foot_min_confidence'] = min(confidence[i] for i in FEET)
            pixels = np.array([[image[i]['x'], image[i]['y']] for i in FEET])*500
            record['foot_image_pixels'] = pixels.tolist()
            record['min_image_boundary_distance_px'] = float(np.minimum(pixels, 500-pixels).min())
            if record['foot_min_confidence'] < .5:
                reason = 'low_foot_confidence'
            elif not np.isfinite(pixels).all() or np.any(pixels < 0) or np.any(pixels > 500):
                reason = 'image_crop_or_nonfinite'
            elif min(confidence[i] for i in TORSO) < .5:
                reason = 'torso_sign_unreliable'
            else:
                points = np.array([[world[i][k] for k in ['x', 'y', 'z']] for i in FEET]) @ TRANSFORM.T
                torso = np.array([[world[i][k] for k in ['x', 'y', 'z']] for i in TORSO]) @ TRANSFORM.T
                up = (torso[0]+torso[1]-torso[2]-torso[3])/2
                if not np.isfinite(points).all() or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-6:
                    reason = 'nonfinite_or_degenerate_geometry'
                else:
                    up /= np.linalg.norm(up)
                    centroid = points.mean(axis=0)
                    centered = points-centroid
                    _, spectrum, axes = np.linalg.svd(centered, full_matrices=False)
                    normal = axes[-1]
                    if np.dot(normal, up) < 0:
                        normal = -normal
                    record.update(centroid_sim=centroid.tolist(), singular_values=spectrum.tolist(),
                          second_to_first_singular_ratio=float(spectrum[1]/spectrum[0]),
                          third_to_second_singular_ratio=float(spectrum[2]/spectrum[1]),
                          normal_sim=normal.tolist(), torso_up_sim=up.tolist(),
                          torso_sign_dot=float(np.dot(normal, up)),
                          local_plane_rms_model_units=float(spectrum[2]/2),
                          heel_to_toe_lengths_model_units=[float(np.linalg.norm(points[2]-points[0])),
                                                           float(np.linalg.norm(points[3]-points[1]))],
                          heel_spacing_model_units=float(np.linalg.norm(points[0]-points[1])))
                    if spectrum[1] < 1e-6 or spectrum[1]/spectrum[0] < 1e-3:
                        reason = 'rank2_not_supported'
                    elif normal[2] < .25:
                        reason = 'signed_normal_z_below025'
                    else:
                        accepted.append({'ordinal': ordinal, 'frame': frame['frame'], 'centered': centered,
                                         'normal': normal, 'up': up, 'pixels': pixels, 'centroid': centroid})
        record['status'] = 'accepted_for_geometry' if reason is None else 'rejected'
        record['rejection_reason'] = reason
        if reason:
            rejects[record['partition']+':'+reason] += 1
        details.append(record)
    construction = [r for r in accepted if r['ordinal'] < 20]
    diagnostic = [r for r in accepted if r['ordinal'] >= 20]
    generation_fit = fit(construction)
    diagnostic_fit = fit(diagnostic)
    normal = np.array(generation_fit['normal_sim']) if generation_fit else None
    chunks = [fit([r for r in construction if lo <= r['ordinal'] < hi]) for lo, hi in [(0, 10), (10, 20)]]
    image_scatter = {}
    if construction:
        for column, landmark in enumerate(FEET):
            pixels = np.array([r['pixels'][column] for r in construction])
            median = np.median(pixels, axis=0)
            image_scatter[str(landmark)] = {'median_px': median.tolist(),
                  'range_px': np.ptp(pixels, axis=0).tolist(),
                  'distance_from_median_px': stats(np.linalg.norm(pixels-median, axis=1))}
    report = {'construction_count': len(construction), 'diagnostic_count': len(diagnostic),
        'rejection_counts': dict(rejects), 'generation_fit': generation_fit, 'diagnostic_fit_only': diagnostic_fit,
        'generation_half_fits': chunks,
        'generation_halves_angle_deg': angle(np.array(chunks[0]['normal_sim']), np.array(chunks[1]['normal_sim'])) if all(chunks) else None,
        'generation_vs_diagnostic_fit_angle_deg': angle(normal, np.array(diagnostic_fit['normal_sim'])) if generation_fit and diagnostic_fit else None,
        'diagnostic_frame_angles_to_frozen_generation_normal_deg': stats([angle(normal, r['normal']) for r in diagnostic]) if generation_fit else None,
        'construction_leave_one_out_normal_angles_deg': stats([angle(normal, np.array(fit(construction[:i]+construction[i+1:])['normal_sim'])) for i in range(len(construction))]) if len(construction)>2 else None,
        'construction_image_landmark_scatter': image_scatter,
        'construction_centroid_range_sim': np.ptp(np.array([r['centroid'] for r in construction]), axis=0).tolist() if construction else None,
        'frame_details': details, 'floor_or_simulator_scores_read': False, 'model_changes': False,
        'unknowns': ['World landmark depth and metric scale are model estimates, not calibrated3D',
                     'Heel/toe landmarks are not exact sole-plane contact points',
                     'Source-visible feet do not establish measured contact or metricgravity',
                     'Small SVDresidual can coexist with systematically biased or unstable plane orientation',
                     'Perframe centering removes roottranslation; no flooroffset is estimated here']}
    save('report.json', report)
    save('manifest.json', {'schema': 1, 'created_at': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
                          'files': {str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}})
    from weft.pipeline import verify_artifacts
    summary = {k: v for k, v in report.items() if k not in {'frame_details', 'unknowns'}}
    print(json.dumps({'report': summary, 'verification': verify_artifacts(OUT)}, indent=2))


if __name__ == '__main__':
    main()
