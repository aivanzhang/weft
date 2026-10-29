"""Re-export saved joint panels with shared bounds; never repeat inference or geometry."""

import argparse
import json
from pathlib import Path
import shutil
import tomllib

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from triangulate_freemocap_construction import panel, plot_layout


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'artifacts/freemocap-camera-space-joints-01'
SOURCE = ROOT/'artifacts/freemocap-full-source-01'


def verified(root, name):
    path = root/name
    if digest(path) != json.loads((root/'manifest.json').read_text())['files'][name]:
        raise ValueError('Frozen render input changed: '+name)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    source = verified(PARENT, 'camera-space-joints.json')
    rows = json.loads(source.read_text())['frames']
    if len(rows) != 1108 or [r['frame'] for r in rows] != list(range(1108)):
        raise ValueError('Complete original joint coverage changed')
    calibration = tomllib.loads(verified(SOURCE, 'member-04-freemocap_sample_data_camera_calibration.toml').read_text())
    cameras = []
    for i in (0, 1):
        c = calibration[f'cam_{i}']
        cameras.append((np.array(c['matrix']), c['distortions'][0], cv2.Rodrigues(np.array(c['rotation']))[0], np.array(c['translation'])*.001))
    out.mkdir(exist_ok=False)
    for path in (Path(__file__), Path(__file__).with_name('triangulate_freemocap_construction.py')):
        shutil.copyfile(path, out/path.name)
    selected = [550, 824]
    write_json(out/'pre-run.json', {'created_at': local_time(), 'parent_manifest_sha256': digest(PARENT/'manifest.json'),
        'joint_data_sha256': digest(source), 'selected_frames': selected,
        'scope': 'Correct clipped 3D plot framing only; shared projection/scale across both panels. No source decode, model call, triangulation or body change.'})
    layout = plot_layout([rows[f] for f in selected])
    for frame in selected:
        paths = [verified(PARENT, f'view-{i}-source-{frame:04d}.png') for i in (1, 2)]
        panel(frame, paths, rows[frame], cameras, out/f'source-and-joints-{frame:04d}.png', layout)
    write_json(out/'report.json', {'created_at': local_time(), 'status': 'source_and_joint_panels_reexported',
        'parent_manifest_sha256': digest(PARENT/'manifest.json'), 'joint_data_sha256': digest(source),
        'source_frames_retained_in_parent': 1108, 'joint_slots_retained_in_parent': 36564,
        'rendered_joint_counts': {str(f): sum(j['status'] == 'estimated' for j in rows[f]['joints']) for f in selected},
        'all_rendered_joint_centers_within_plot': True, 'new_source_decode': 0, 'new_inference': 0,
        'new_triangulation': 0, 'paid_calls': 0, 'reserved_camera_opened': False})
    print(json.dumps({'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
