"""Compare monocular board placement with stereo geometry without repairing cameras."""

import argparse
import json
from pathlib import Path
import shutil
import time
import tomllib

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.rigid_correspondence import fit_rigid_points


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'artifacts/freemocap-full-evidence-01'
CHECK = ROOT/'artifacts/freemocap-full-board-check-01'
SOURCE = ROOT/'artifacts/freemocap-full-source-01'


def read(path):
    return json.loads(path.read_text())


def verified(root, name):
    path = root/name
    if digest(path) != read(root/'manifest.json')['files'][name]:
        raise ValueError('Frozen evidence changed: '+name)
    return path


def describe(values):
    return {'count': len(values), 'rmse': float(np.sqrt(np.mean(np.square(values)))),
            'p95': float(np.percentile(values, 95)), 'maximum': float(np.max(values))}


def project(points, camera):
    return cv2.projectPoints(points, camera['r'], camera['t'], camera['K'], camera['d'])[0].reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifacts directory')
    baseline = read(verified(CHECK, 'full-supplied-all-prefix-comparisons.json'))
    views = [read(verified(EVIDENCE, f'view-{i}-detections.json')) for i in (1, 2)]
    calibration = tomllib.loads(verified(SOURCE, 'member-04-freemocap_sample_data_camera_calibration.toml').read_text())
    cameras = []
    for i in (0, 1):
        c = calibration[f'cam_{i}']
        cameras.append({'K': np.array(c['matrix']), 'd': np.array(c['distortions']),
                        'r': np.array(c['rotation']), 't': np.array(c['translation'])*.001})
    if len(baseline) != 500 or sum(r['status'] == 'compared' for r in baseline) != 18:
        raise ValueError('Complete prefix or supported cohort changed')
    model = cv2.aruco.CharucoBoard((7, 5), .058, .0464,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)).getChessboardCorners().astype(float)
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    shutil.copyfile(ROOT/'src/weft/rigid_correspondence.py', out/'rigid_correspondence.py')
    write_json(out/'pre-run.json', {
        'created_at': local_time(), 'source_manifests': {p.name: digest(p/'manifest.json') for p in (EVIDENCE, CHECK, SOURCE)},
        'hypothesis': 'Single-view planar depth/orientation sensitivity may amplify cross-view errors; diagnose against fixed stereo geometry.',
        'cohort': 'All 500 prefix slots; 18 original compared poses and 234 shared corners, without filtering.',
        'method': 'Proper scale-one Procrustes placement of known board points onto saved triangulations; projection into both original cameras.',
        'scale_hypothesis': 'Least-squares scalar reported separately, never applied to board or cameras.',
        'monocular_comparison': 'Original all-Cam1-corner solution versus a fresh shared-corner-only Cam1 solution; all alternatives retained.',
        'sensitivity': 'Finite-difference derivative of Cam2 projection with respect to six Cam1 board-pose coordinates, composed with pseudoinverse of Cam1 pixel Jacobian. Descriptive linearized gain, not a confidence interval.',
        'maximum_seconds': 10, 'maximum_poses': 18, 'new_decodes': 0, 'new_detector_calls': 0,
        'reserved_camera_opened': False, 'paid_calls': 0, 'original_gates_unchanged': True,
    })
    started = time.perf_counter()
    rows = []
    for original in baseline:
        frame = original['frame']
        row = {'frame': frame, 'time_s': original['time_s'], 'status': original['status']}
        if original['status'] == 'compared':
            ids = original['source_ids']
            observed = [{p['id']: p['pixel'] for p in view[frame]['corners']} for view in views]
            image = [np.array([data[i] for i in ids]) for data in observed]
            stereo = np.array(original['triangulated_corners_m'])
            fit = fit_rigid_points(model[ids], stereo)
            rigid = np.array(fit['predicted_points'])
            rvec, tvec = np.array(original['board_rotation']), np.array(original['board_translation_m'])
            mono = model[ids] @ cv2.Rodrigues(rvec)[0].T+tvec
            all_ids = sorted(observed[0])
            jacobian = cv2.projectPoints(model[all_ids], rvec, tvec, cameras[0]['K'], cameras[0]['d'])[1][:, :6]
            derivatives = []
            theta = np.r_[rvec, tvec]
            for axis in range(6):
                step = np.zeros(6); step[axis] = 1e-6
                plus, minus = theta+step, theta-step
                p = project(model[ids] @ cv2.Rodrigues(plus[:3])[0].T+plus[3:], cameras[1])
                m = project(model[ids] @ cv2.Rodrigues(minus[:3])[0].T+minus[3:], cameras[1])
                derivatives.append(((p-m)/2e-6).ravel())
            gain = np.linalg.svd(np.column_stack(derivatives) @ np.linalg.pinv(jacobian), compute_uv=False)
            success, shared_r, shared_t = cv2.solvePnP(model[ids], image[0], cameras[0]['K'], cameras[0]['d'], flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                raise ValueError('Shared-corner monocular diagnostic failed')
            shared = model[ids] @ cv2.Rodrigues(shared_r)[0].T+shared_t.ravel()
            row.update({
                'source_ids': ids, 'rigid_fit': fit,
                'stereo_reprojection_errors_px': [np.linalg.norm(project(stereo, c)-pixels, axis=1).tolist() for c, pixels in zip(cameras, image)],
                'rigid_reprojection_errors_px': [np.linalg.norm(project(rigid, c)-pixels, axis=1).tolist() for c, pixels in zip(cameras, image)],
                'original_mono_to_stereo_errors_m': np.linalg.norm(mono-stereo, axis=1).tolist(),
                'original_cross_errors_px': original['cross_view_errors_px'],
                'shared_only_mono_cross_errors_px': np.linalg.norm(project(shared, cameras[1])-image[1], axis=1).tolist(),
                'shared_only_pose': {'rotation': shared_r.ravel().tolist(), 'translation_m': shared_t.ravel().tolist()},
                'camera0_jacobian_singular_values': np.linalg.svd(jacobian, compute_uv=False).tolist(),
                'linearized_cross_camera_worst_pixel_gain': float(gain[0]),
            })
        rows.append(row)
        if time.perf_counter()-started > 10:
            raise TimeoutError('Geometry diagnostic exceeded its work envelope')
    valid = [r for r in rows if r['status'] == 'compared']
    summary = {
        'stereo_reprojection_px': [describe([e for r in valid for e in r['stereo_reprojection_errors_px'][i]]) for i in (0, 1)],
        'rigid_reprojection_px': [describe([e for r in valid for e in r['rigid_reprojection_errors_px'][i]]) for i in (0, 1)],
        'rigid_3d_residual_m': describe([e for r in valid for e in r['rigid_fit']['residuals']]),
        'original_monocular_3d_discrepancy_m': describe([e for r in valid for e in r['original_mono_to_stereo_errors_m']]),
        'original_cross_errors_px': describe([e for r in valid for e in r['original_cross_errors_px']]),
        'shared_only_mono_cross_errors_px': describe([e for r in valid for e in r['shared_only_mono_cross_errors_px']]),
        'unapplied_scale_range': [min(r['rigid_fit']['scale_hypothesis_not_applied'] for r in valid), max(r['rigid_fit']['scale_hypothesis_not_applied'] for r in valid)],
        'linearized_pixel_gain_range': [min(r['linearized_cross_camera_worst_pixel_gain'] for r in valid), max(r['linearized_cross_camera_worst_pixel_gain'] for r in valid)],
    }
    write_json(out/'all-prefix-diagnostics.json', rows)
    report = {'created_at': local_time(), 'status': 'geometry_diagnostic_complete_no_repair', 'summary': summary,
        'supported_poses': len(valid), 'unavailable_poses': len(rows)-len(valid), 'runtime_seconds': time.perf_counter()-started,
        'new_decodes': 0, 'new_detector_calls': 0, 'reserved_camera_opened': False, 'paid_calls': 0,
        'calibration_changed': False, 'scale_changed': False, 'original_gates_unchanged': True,
        'limitations': ['Stereo placement uses both construction views and is not an independent validation.',
            'Near-planarity and spacing agreement cannot establish exact intrinsics, synchronization, ground, anatomy or physical transfer.',
            'Linearized pixel gain is local and assumes small perturbations; it does not identify the cause of the observed discrepancy.']}
    write_json(out/'report.json', report)
    print(json.dumps({'summary': summary, 'runtime_seconds': report['runtime_seconds'], 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
