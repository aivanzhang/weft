"""Freeze independent board projections before the reserved camera is decoded."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from weft.benchmark import seal_suite
from weft.camera_reference import project_world
from weft.io import digest, local_time, write_json


ROOT = Path(__file__).resolve().parents[2]
GEOMETRY = ROOT/'artifacts/freemocap-full-geometry-01'
PREDICTIONS = ROOT/'artifacts/freemocap-third-predictions-01'


def read(path):
    return json.loads(path.read_text())


def verified(root, name):
    path = root/name
    if digest(path) != read(root/'manifest.json')['files'][name]:
        raise ValueError('Frozen board prediction input changed')
    return path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    body = read(verified(PREDICTIONS, 'protocol.json')); c = body['camera']
    rows = read(verified(GEOMETRY, 'all-prefix-diagnostics.json')); predictions = []
    for row in rows:
        corners = []
        if row['status'] == 'compared':
            points = np.array(row['rigid_fit']['predicted_points'])
            positions, pixels, valid = project_world(points, c['rotation'], c['translation_m'], c['matrix'], c['radial'])
            for index, point, pixel, p, ok in zip(row['source_ids'], points, pixels, positions, valid, strict=True):
                if not ok:
                    raise ValueError('A frozen board corner has no forward projection')
                corners.append({'id': index, 'world_point_m': point.tolist(), 'pixel': pixel.tolist(), 'depth_m': float(p[2])})
        predictions.append({'frame': row['frame'], 'time_s': row['time_s'], 'construction_status': row['status'], 'corners': corners})
    supported = [r['frame'] for r in predictions if r['corners']]
    if len(predictions) != 500 or len(supported) != 18 or sum(len(r['corners']) for r in predictions) != 234:
        raise ValueError('Original board-prediction cohort changed')
    out.mkdir(exist_ok=False); shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'board-predictions-before-third-pixels.json', predictions)
    write_json(out/'protocol.json', {'created_at': local_time(), 'parent_body_predictions_manifest': digest(PREDICTIONS/'manifest.json'),
        'geometry_manifest': digest(GEOMETRY/'manifest.json'), 'reserved_video': body['reserved_video'],
        'supported_frames': supported, 'all_prefix_frame_slots': 500, 'corner_predictions': 234,
        'board': {'squares_xy': [7, 5], 'square_length_m': .058, 'marker_length_m': .0464, 'dictionary': 'DICT_4X4_250'},
        'checks': {'minimum_poses': 10, 'minimum_compared_corners_each_pose': 8, 'maximum_p95_px': 3., 'maximum_error_px': 10.},
        'methods': 'Project previously fixed-size stereo board placements into Cam3; detect its ChArUco corners independently. No camera, clock, scale or pose refit.',
        'missing': 'Retain all 500 prefix slots and all 234 predictions; record unavailable third-view corners, never fabricate zero error.',
        'post_grading_human_inspection_frames': body['inspection_frames_after_grading'],
        'maximum_returned_frame_reads': 22, 'maximum_board_detections': 18, 'maximum_seconds': 30,
        'maximum_output_bytes': 32000000, 'paid_calls': 0, 'third_camera_opened': False,
        'scope': 'Additional camera-consistency evidence, not measured human anatomy, floor height or physical transfer.'})
    write_json(out/'report.json', {'created_at': local_time(), 'status': 'third_board_predictions_frozen',
        'all_prefix_frame_slots': 500, 'supported_poses': 18, 'corner_predictions': 234,
        'third_camera_opened': False, 'new_video_decodes': 0, 'new_inference_calls': 0, 'paid_calls': 0})
    print(json.dumps({'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
