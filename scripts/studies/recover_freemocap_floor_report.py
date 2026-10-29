"""Finalize the saved floor-proxy rejection after a NumPy boolean serialization error."""

import argparse
import json
from pathlib import Path
import shutil

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'artifacts/freemocap-floor-proxy-01'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    manifest = read(PARENT/'manifest.json')
    for name, expected in manifest['files'].items():
        if digest(PARENT/name) != expected:
            raise ValueError('Completed parent evidence changed')
    pre = read(PARENT/'pre-run.json'); limits = pre['admission']
    alternatives = read(PARENT/'all-correspondence-alternatives.json')
    ranked = sorted(alternatives, key=lambda r: r['geometry']['reprojection_rmse_px'])
    best = ranked[0]; g = best['geometry']; gap = ranked[1]['geometry']['reprojection_rmse_px']-g['reprojection_rmse_px']
    checks = {'stereo_pixel_consistency': g['maximum_reprojection_px'] <= limits['maximum_reprojection_px'],
        'correspondence_separation': gap >= limits['minimum_runner_up_rmse_gap_px'],
        'plane_consistency': g['plane_rms_m'] <= limits['maximum_plane_rms_m'],
        'rectangle_opposite_edges': max(g['opposite_edge_relative_difference']) <= limits['maximum_opposite_edge_fraction'],
        'rectangle_angles': max(g['adjacent_edge_absolute_cosines']) <= limits['maximum_adjacent_edge_abs_cosine'],
        'forward_bounded_depth': all(limits['depth_bounds_m'][0] <= d <= limits['depth_bounds_m'][1] for d in g['depths_m'])}
    if all(checks.values()) or read(PARENT/'all-pixel-perturbations.json') or (PARENT/'floor-proxy.json').exists():
        raise ValueError('This recovery handles only the saved pre-sensitivity rejection')
    out.mkdir(exist_ok=False)
    for name in manifest['files']:
        shutil.copyfile(PARENT/name, out/name)
    shutil.copyfile(__file__, out/Path(__file__).name)
    report = {'created_at': local_time(), 'status': 'floor_cover_proxy_rejected', 'checks': checks,
        'selected_candidate': best, 'runner_up_rmse_gap_px': gap, 'sensitivity': None,
        'recovery_parent_manifest_sha256': digest(PARENT/'manifest.json'),
        'recovery_scope': 'Read the saved geometry and summarize with native JSON scalars; no completed source read, contour detection or geometry fit repeated.',
        'source_frame': 550, 'human_poses_used': False, 'reserved_camera_opened': False, 'paid_calls': 0,
        'geometry_recomputed': False, 'floor_proxy_exported': False,
        'limitations': ['The visible cover fails the frozen pixel, plane and opposite-edge requirements; it is not used to place a floor.',
            'Calibration, cover shape and corner-localization errors are not uniquely separated.',
            'The original exception and complete failed alternatives remain preserved.']}
    write_json(out/'report.json', report)
    print(json.dumps({'status': report['status'], 'checks': checks, 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
