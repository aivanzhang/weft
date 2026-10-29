"""Freeze complete-source predictions and scoring before opening the reserved camera."""

import argparse
import json
from pathlib import Path
import shutil
import tomllib

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.camera_reference import JOINTS, project_world
from weft.exercise_vision import model_provenance
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'artifacts/freemocap-full-source-01'
BASELINE = ROOT/'artifacts/freemocap-camera-space-joints-01'
CANDIDATE = ROOT/'artifacts/freemocap-fixed-length-02'
MODEL = ROOT/'.local/models/pose_landmarker_full.task'


def read(path):
    return json.loads(path.read_text())


def verified(root, name):
    path = root/name
    if digest(path) != read(root/'manifest.json')['files'][name]:
        raise ValueError('Frozen prediction input changed: '+name)
    return path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    raw = read(verified(BASELINE, 'camera-space-joints.json'))['frames']
    proposed = read(verified(CANDIDATE, 'complete-joint-proposals.json'))['frames']
    fits = {}
    for i in (1, 2):
        for line in verified(CANDIDATE, f'worker-{i}/frame-proposals.jsonl').read_text().splitlines():
            row = json.loads(line); fits[row['frame']] = row['fit']
    metadata = read(verified(SOURCE, 'report.json'))
    third = [r for r in metadata['extracted'] if r['role'] == 'reserved_comparison']
    if len(third) != 1 or len(raw) != 1108 or len(proposed) != 1108:
        raise ValueError('Reserved source or complete prediction coverage changed')
    third = third[0]; video = verified(SOURCE, third['file'])
    c = tomllib.loads(verified(SOURCE, 'member-04-freemocap_sample_data_camera_calibration.toml').read_text())['cam_2']
    if Path(third['member']).name != c['name']+'.mp4' or c['size'] != [720, 1280] or any(c['distortions'][j] != 0 for j in range(1, 5)):
        raise ValueError('Reserved video/camera identity or projection model changed')
    R = cv2.Rodrigues(np.array(c['rotation']))[0]; t = np.array(c['translation'])*.001; K = np.array(c['matrix'])
    predictions = {name: [] for name in ('raw_two_view', 'constrained_qualified', 'constrained_all_converged')}
    for frame in range(1108):
        fit = fits.get(frame); converged = fit is not None and fit['status'] == 'converged' and fit['length_constraints_satisfied']
        for joint in sorted(JOINTS.values()):
            original = raw[frame]['joints'][joint]; candidate = proposed[frame]['joint_proposals'][joint]
            points = {'raw_two_view': original['world_point_m'] if original['status'] == 'estimated' else None,
                'constrained_qualified': candidate['point_proposal_m'] if candidate['numerically_supported'] else None,
                'constrained_all_converged': candidate['point_proposal_m'] if converged else None}
            for method, point in points.items():
                pixel = None; status = 'construction_prediction_unavailable'; depth = None
                if point is not None:
                    camera, projected, valid = project_world(np.array(point)[None, :], R, t, K, c['distortions'][0])
                    depth = float(camera[0, 2])
                    if valid[0] and .2 <= depth <= 100. and np.max(np.abs(projected[0])) <= 1e5:
                        pixel = projected[0].tolist()
                        status = 'inside_comparison_image' if 0 <= pixel[0] <= 720 and 0 <= pixel[1] <= 1280 else 'outside_comparison_image'
                    else:
                        status = 'projection_outside_envelope'
                predictions[method].append({'frame': frame, 'joint': joint, 'pixel': pixel,
                    'world_point_m': point, 'depth_m': depth, 'status': status})
    runtime = {n: digest(p) for n, p in _implementation_files().items()}
    out.mkdir(exist_ok=False)
    for path in (Path(__file__), ROOT/'src/weft/projected_joint_evaluation.py', ROOT/'src/weft/camera_reference.py', ROOT/'src/weft/multiview_pose.py'):
        shutil.copyfile(path, out/path.name)
    write_json(out/'predictions-before-third-pixels.json', predictions)
    protocol = {'created_at': local_time(), 'status': 'frozen_before_reserved_camera_decode',
        'input_manifests': {p.name: digest(p/'manifest.json') for p in (SOURCE, BASELINE, CANDIDATE)},
        'reserved_video': {'path': str(video.relative_to(ROOT)), 'sha256': digest(video), 'camera_name': c['name'],
            'fps': 30, 'source_frames': 1108, 'width': 720, 'height': 1280},
        'camera': {'rotation': R.tolist(), 'translation_m': t.tolist(), 'matrix': K.tolist(), 'radial': c['distortions'][0]},
        'model': model_provenance(MODEL, variant='full'), 'mode': 'IMAGE', 'identity': 'raw', 'inference_stride': 2,
        'scorer': {'file': 'projected_joint_evaluation.py', 'sha256': digest(out/'projected_joint_evaluation.py'),
            'baseline': 'raw_two_view', 'limit_px': 12.8, 'observation': 'Original single-pose image observations accepted by unchanged multiview_pose._observation confidence/image bounds. No monocular world coordinates.',
            'cohort': 'All 1108 source frames × 12 major joints; missing, unsampled and unavailable slots remain. Shared-cohort errors and coverage changes are separate.',
            'comparison_sets': ['complete_source', 'active_400_through_1059'],
            'primary_candidate': 'constrained_qualified',
            'secondary_candidate': 'constrained_all_converged includes first-two-view image-gate failures; diagnostic only, never an admission override.',
            'prohibited': ['camera_refitting', 'retiming', 'pose_label_swaps', 'length_refitting', 'source_trimming', 'threshold_changes', 'default_or_body_promotion']},
        'prediction_sha256': digest(out/'predictions-before-third-pixels.json'), 'runtime_hashes': runtime,
        'maximum_inference_calls': 554, 'maximum_video_frames': 1108, 'maximum_seconds': 90,
        'maximum_output_bytes': 128000000, 'projected_paid_spend_usd': 0,
        'reserved_camera_opened': False, 'new_video_decodes': 0,
        'reservation_lifecycle': 'One frozen-method comparison. After execution this camera is exposed development evidence, not a fresh holdout for later repairs.',
        'reference_quality': 'Independent image viewpoint but same model family; not optical-marker joint truth, anatomical measurements, floor/contact evidence or physical transfer.',
        'inspection_frames_after_grading': [416, 550, 824, 966]}
    write_json(out/'protocol.json', protocol)
    write_json(out/'report.json', {'created_at': local_time(), 'status': 'third_camera_predictions_frozen',
        'prediction_slots_per_method': 1108*12, 'methods': {name: {'predicted_slots': sum(r['pixel'] is not None for r in rows),
            'outside_image_slots': sum(r['status'] == 'outside_comparison_image' for r in rows)} for name, rows in predictions.items()},
        'third_camera_opened': False, 'new_inference_calls': 0, 'new_video_decodes': 0, 'paid_calls': 0,
        'body_built': False, 'scope': protocol['reference_quality']})
    print(json.dumps({'methods': read(out/'report.json')['methods'], 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
