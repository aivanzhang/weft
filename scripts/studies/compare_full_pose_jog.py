"""Compare frozen Lite/Full candidates on identical slots; never select or repair poses."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from weft.benchmark import seal_suite
from weft.camera_reference import JOINTS
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
RUNS = {'lite': ROOT/'artifacts/camera-ground-jog-lab-01/run', 'full': ROOT/'artifacts/full-pose-ground-jog-lab-01/run'}
PAIRS = {'lite': ROOT/'artifacts/multiview-jog-two-view-01', 'full': ROOT/'artifacts/multiview-jog-full-two-view-01'}


def read(path):
    return json.loads(path.read_text())


def stats(values):
    if not values: return {'count': 0, 'mean': None, 'rmse': None, 'p95': None, 'maximum': None}
    values = np.asarray(values)
    return {'count': len(values), 'mean': float(values.mean()), 'rmse': float(np.sqrt(np.mean(values**2))),
            'p95': float(np.percentile(values, 95)), 'maximum': float(values.max())}


def torso_up(points):
    vector = (points['left_shoulder']+points['right_shoulder']-points['left_hip']-points['right_hip'])/2
    return float(vector[2]/np.linalg.norm(vector))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    out = args.out.resolve()
    source = ROOT/'artifacts/multiview-jog-source-01'
    if any(out.is_relative_to(p) for p in [*RUNS.values(), *PAIRS.values(), source]): raise ValueError('Use a separate study output')
    verified = {str(p.relative_to(ROOT)): verify_artifacts(p) for p in [*RUNS.values(), *PAIRS.values(), source]}
    out.mkdir(parents=True, exist_ok=False); shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'inputs': verified,
        'scope': 'Post-construction development comparison, not a fresh holdout or parameter fitting',
        'criteria': 'All 240 source frames/twelve-joint slots, paired availability and unchanged numerical comparators',
        'paid_requests': 0, 'new_inference': False, 'new_simulation': False})
    with np.load(source/'body-coco17.npz', allow_pickle=False) as data: reference, names = data['joints'], data['names'].tolist()
    candidates = {}
    for label, run in RUNS.items():
        spec, estimates, evidence = (read(run/name) for name in ('human.json', 'camera-reference.json', 'evidence.json'))
        if spec['source_sha256'] != digest(source/'drone1.mp4') or len(estimates['frames']) != 240:
            raise ValueError('Candidate source or full frame domain differs')
        raw = {row['frame']: row for row in evidence['frames']}
        native = {int(meta['evidence_id'].rsplit(':', 1)[1]): row for meta, row in
                  zip(spec['reference_evidence'], read(run/'human-reference.json')['frames'], strict=True)}
        offset = np.array([*spec['camera_reference']['origin_world_m'][:2], spec['camera_reference']['ground']['height_m']])
        frames = {}
        for frame, estimate in enumerate(estimates['frames']):
            if estimate['frame'] != frame: raise ValueError('Camera estimates changed frame ordering')
            if estimate['status'] != 'estimated': continue
            if abs(native[frame]['source_time_s']-frame/evidence['fps']) > 1e-7: raise ValueError('Body source clock differs')
            frames[frame] = {'body': {name: np.asarray(native[frame]['human']['joints'][name])+offset for name in JOINTS},
                'raw': {name: np.asarray(estimate['root_world_m'])+estimate['world_relative_landmarks_m'][j] for name, j in JOINTS.items()},
                'image': raw[frame]['landmarks']}
        candidates[label] = frames
    rows, errors = [], {scope: {f'{label}_{kind}': [] for label in RUNS for kind in ('raw', 'body')}
                        for scope in ('all_paired_available', 'both_image_confident')}
    for frame in range(240):
        for name, joint in JOINTS.items():
            row = {'frame': frame, 'joint': name, 'paired': all(frame in value for value in candidates.values()), 'errors_m': None}
            if row['paired']:
                distances, confident = {}, True
                for label, frames in candidates.items():
                    image = frames[frame]['image'][joint]
                    confident &= bool(0 <= image['x'] <= 1 and 0 <= image['y'] <= 1 and .5 <= min(image['visibility'], image['presence']) <= 1)
                    for kind in ('raw', 'body'):
                        key = label+'_'+kind
                        distances[key] = float(np.linalg.norm(frames[frame][kind][name]-reference[frame, names.index(name)]))
                        errors['all_paired_available'][key].append(distances[key])
                if confident:
                    for key, value in distances.items(): errors['both_image_confident'][key].append(value)
                row.update(errors_m=distances, both_image_confident=confident)
            rows.append(row)
    write_json(out/'body-comparison.json', {'rows': rows, 'source_frames': 240, 'joint_slots': len(rows)})
    pair_rows = {label: {(row['frame'], row['joint']): row for row in read(path/'comparison.json')['rows']} for label, path in PAIRS.items()}
    if set(pair_rows['lite']) != set(pair_rows['full']) or len(pair_rows['lite']) != 2880:
        raise ValueError('Two-view comparisons do not retain identical joint slots')
    pair_scores = {}
    for field in ('triangulated_reference_error_m', 'triangulated_comparison_error_px'):
        values, coverage = {'lite': [], 'full': []}, Counter()
        for key in pair_rows['lite']:
            a, b = (pair_rows[label][key][field] for label in ('lite', 'full'))
            coverage[('both' if a is not None else 'full_only') if b is not None else ('lite_only' if a is not None else 'neither')] += 1
            if a is not None and b is not None: values['lite'].append(a); values['full'].append(b)
        pair_scores[field] = {'availability': dict(coverage), 'paired': {label: stats(v) for label, v in values.items()}}
    report = {'created_at': local_time(), 'status': 'paired_comparison_complete', 'source_frames': 240, 'joint_slots': len(rows),
        'single_view': {scope: {key: stats(v) for key, v in values.items()} for scope, values in errors.items()},
        'two_view': pair_scores, 'frame_200_body_vertical_direction': {label: torso_up(frames[200]['body']) for label, frames in candidates.items()},
        'ground_clearance': {label: read(path/'ground-clearance.json')['summary'] for label, path in RUNS.items()},
        'physical_validation': 'unverified', 'default_changed': False, 'paid_requests': 0}
    if any(verify_artifacts(ROOT/name) != value for name, value in verified.items()): raise RuntimeError('Frozen candidate changed')
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
