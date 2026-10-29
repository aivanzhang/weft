"""Test conditional opening-corner cameras against every saved side-view slot."""

import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.affine_camera import fit_affine_camera, project_affine_points
from weft.benchmark import seal_suite
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import _implementation_files
from weft.primitive_geometry import silhouette_overlap
from weft.recording import load_recording
from weft.spatial_rigid import corners, rotation


ROOT = Path(__file__).resolve().parents[2]
MATERIAL = ROOT / 'artifacts/spatial-material-fit-01'
SIDE = ROOT / 'artifacts/spatial-ramp-sandbox-02'
INSPECTION = ROOT / 'artifacts/phys101-constraints-inspection-02'
LABELS = ROOT / 'data/general/phys101-side-corners.json'
CONTRACT = {
    'scope': 'Exposed first-trial development diagnosis, not independent physical accuracy',
    'hypotheses': {'depth-positive': 1, 'depth-negative': -1},
    'fit_frame': 0, 'source_frames': 37, 'source_fps': 25,
    'front_initial_frame': 8, 'front_fps_numerator': 30000, 'front_fps_denominator': 1001,
    'maximum_pair_clock_error_s': .5 * 1001 / 60000 + 1e-9,
    'annotation_max_error_px': 6.,
    'orthographic_axis_scale_ratio_max': 1.05, 'orthographic_abs_row_cosine_max': .05,
    'nominal_later_centroid_p95_px': 12., 'nominal_later_centroid_max_px': 24.,
    'minimum_later_mask_iou': .65,
    'unchanged': ['all scene dimensions', 'materials', 'mass', 'forces', 'initial state', 'simulation states'],
    'depth_gauge_controls': [.5, 1., 2.], 'maximum_seconds': 30, 'maximum_bytes': 32 * 1024**2,
    'new_simulation': False, 'new_inference': False, 'new_paid_calls': 0,
    'repeat_evidence_used': False, 'automatic_point_detection': False, 'selection_or_promotion': False,
}


def read(path):
    return json.loads(path.read_text())


def verified(root, relative):
    path = root / relative
    if digest(path) != read(root / 'manifest.json')['files'][relative]:
        raise ValueError(f'Saved input changed: {relative}')
    return path


def projected_centroid(points):
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2)
    moments = cv2.moments(hull)
    if moments['m00'] <= 0:
        raise ValueError('Projected cuboid has no area')
    return np.array([moments['m10'], moments['m01']]) / moments['m00']


def summarize(rows):
    paired = [row for row in rows if row['status'] == 'compared']
    errors = np.array([row['centroid_error_px'] for row in paired])
    ious = np.array([row['mask_overlap']['iou'] for row in paired])
    return {
        'total_slots': len(rows), 'compared': len(paired), 'unavailable': len(rows) - len(paired),
        'centroid_rmse_px': float(np.sqrt(np.mean(errors**2))),
        'centroid_p95_px': float(np.percentile(errors, 95)), 'centroid_max_px': float(errors.max()),
        'mean_mask_iou': float(ious.mean()), 'minimum_mask_iou': float(ious.min()),
        'nominal_image_check_pass': bool(np.percentile(errors, 95) <= CONTRACT['nominal_later_centroid_p95_px']
            and errors.max() <= CONTRACT['nominal_later_centroid_max_px']
            and ious.min() >= CONTRACT['minimum_later_mask_iou']),
        'maximum_clock_error_s': max(abs(row['pair_clock_error_s']) for row in paired),
        'maximum_corner_annotation_bound_px_per_axis': max(row['annotation_corner_bound_px_per_axis'] for row in paired),
        'final': paired[-1],
    }


def run(out):
    labels = read(LABELS)
    source = ROOT / labels['source']
    initial_image = ROOT / labels['image']
    if digest(source) != labels['source_sha256'] or digest(initial_image) != labels['image_sha256']:
        raise ValueError('Source-only annotation binding changed')
    side = read(verified(SIDE, 'side-image-evidence.json'))
    if len(side['rows']) != 37 or [row['frame'] for row in side['rows']] != list(range(37)):
        raise ValueError('Side evidence must retain every original frame')
    recording = load_recording(MATERIAL / 'selected-coast')
    scene = recording['scene']; events = recording['events']
    if len(events) != 71 or any(event['action'] != 0 for event in events[1:]):
        raise ValueError('Use the unchanged complete saved coast')
    inputs = out / 'inputs'; inputs.mkdir()
    for path, name in [(LABELS, 'source-corners.json'), (source, 'source.mp4'),
                       (initial_image, 'source-initial.png'),
                       (verified(SIDE, 'side-image-evidence.json'), 'side-image-evidence.json'),
                       (verified(MATERIAL, 'selected-coast/scene.json'), 'scene.json'),
                       (verified(MATERIAL, 'selected-coast/events.jsonl'), 'events.jsonl')]:
        shutil.copyfile(path, inputs / name)
    masks = {}; mask_dir = inputs / 'masks'; mask_dir.mkdir()
    for row in side['rows']:
        if row['source_sha256'] != labels['source_sha256'] or abs(row['time_s'] - row['frame']/25) > 1e-9:
            raise ValueError('Side evidence identity or clock changed')
        path = verified(SIDE, row['mask']); shutil.copyfile(path, mask_dir / path.name)
        with Image.open(path) as image:
            masks[row['frame']] = np.asarray(image).copy()
    runtime = _implementation_files()
    used_modules = ['affine_camera.py', 'primitive_geometry.py', 'spatial_rigid.py']
    for name in used_modules:
        shutil.copyfile(runtime[name], out / name)
    shutil.copyfile(ROOT / 'tests/test_affine_camera.py', out / 'test_affine_camera.py')
    write_json(out / 'pre-run.json', {
        'created_at': local_time(), 'contract': CONTRACT, 'labels_sha256': digest(LABELS),
        'material_manifest_sha256': digest(MATERIAL / 'manifest.json'),
        'side_manifest_sha256': digest(SIDE / 'manifest.json'),
        'implementation': {name: digest(path) for name, path in runtime.items()},
        'input_copies': {str(path.relative_to(inputs)): digest(path) for path in sorted(inputs.rglob('*')) if path.is_file()},
        'reused_recording_check': '.local/material-transfer-final-verification-01.json',
        'test_report_sha256': digest(ROOT / '.local/affine-camera-tests-01.xml'),
    })

    started = time.perf_counter(); cv2.setNumThreads(1)
    obj = next(obj for obj in scene['objects'] if obj['id'] == labels['object_id'])
    all_rows = {}; fits = {}; gauge = []; frozen_world_points = {}
    source_times = np.array([8 * 1001 / 30000 + event['snapshot']['time_s'] for event in events])
    size = np.asarray(obj['size']); origin = np.asarray(obj['position'])
    orientation = rotation(obj['quaternion'])
    image_points = np.asarray(labels['image_points_px'], dtype=float)
    for name, depth_sign in CONTRACT['hypotheses'].items():
        local = np.asarray(labels['local_corner_signs'], dtype=float) * size / 2
        local[:, 1] *= depth_sign
        points = local @ orientation.T + origin
        fit = fit_affine_camera(points, image_points, image_size=labels['image_size'], scope='construction')
        fits[name] = fit; frozen_world_points[name] = points.tolist()
        diagnostic = fit['orthographic_diagnostic']
        fit['nominal_corner_check_pass'] = fit['maximum_error_px'] <= CONTRACT['annotation_max_error_px']
        fit['scaled_orthographic_screen_pass'] = bool(
            diagnostic['image_axis_scale_ratio'] <= CONTRACT['orthographic_axis_scale_ratio_max']
            and abs(diagnostic['row_cosine']) <= CONTRACT['orthographic_abs_row_cosine_max'])
        # Exact linear weights propagate the declared corner envelope, not physical uncertainty.
        center = points.mean(axis=0)
        scale = float(np.sqrt(np.mean(np.sum((points-center)**2, axis=1))))
        design = np.column_stack([(points-center)/scale, np.ones(len(points))])
        inverse = np.linalg.pinv(design)
        rows = []
        for observed in side['rows']:
            row = {'frame': observed['frame'], 'source_time_s': observed['time_s'],
                'observed_centroid_px': observed['centroid_px'], 'status': 'before_simulation_initialization',
                'centroid_error_px': None, 'mask_overlap': None}
            if observed['time_s'] + 1e-9 >= source_times[0]:
                index = int(np.argmin(abs(source_times-observed['time_s'])))
                mismatch = source_times[index] - observed['time_s']
                if abs(mismatch) > CONTRACT['maximum_pair_clock_error_s']:
                    raise ValueError('No saved state within the frozen source-clock envelope')
                state = events[index]['snapshot']
                moving = next(obj for obj in state['objects'] if obj['id'] == labels['object_id'])
                vertices = corners(moving); pixels = project_affine_points(vertices, fit['matrix_px'])
                centroid = projected_centroid(pixels)
                weights = np.column_stack([(vertices-center)/scale, np.ones(8)]) @ inverse
                bound = labels['coordinate_uncertainty_px_per_axis'] * np.abs(weights).sum(axis=1)
                row.update(status='compared', event_index=index, simulation_source_time_s=float(source_times[index]),
                    pair_clock_error_s=float(mismatch), projected_corners_px=pixels.tolist(),
                    predicted_silhouette_centroid_px=centroid.tolist(),
                    centroid_error_px=float(np.linalg.norm(centroid-observed['centroid_px'])),
                    mask_overlap=silhouette_overlap(masks[observed['frame']], [400, 350, 800, 750], pixels),
                    annotation_corner_bound_px_per_axis=float(bound.max()))
            rows.append(row)
        all_rows[name] = rows
        for factor in CONTRACT['depth_gauge_controls']:
            changed = local.copy(); changed[:, 1] *= factor
            alternative = fit_affine_camera(changed @ orientation.T + origin, image_points,
                image_size=labels['image_size'], scope='construction')
            gauge.append({'hypothesis': name, 'depth_factor': factor, 'depth_scene_units': float(size[1]*factor),
                'camera': alternative,
                'maximum_fitted_pixel_change': float(np.max(np.abs(np.asarray(alternative['fitted_points_px']) - fit['fitted_points_px']))),
                'geometry_or_dynamics_changed': False})
    write_json(out / 'camera-fits.json', fits)
    write_json(out / 'construction-world-points.json', frozen_world_points)
    write_json(out / 'depth-ambiguity-controls.json', gauge)
    write_json(out / 'all-source-slots.json', all_rows)
    results = {name: summarize(rows) for name, rows in all_rows.items()}

    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(out / 'side-comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (1200, 680))
    if not capture.isOpened() or not writer.isOpened():
        raise RuntimeError('Source decoder or comparison encoder unavailable')
    try:
        for frame in range(37):
            ok, bgr = capture.read()
            if not ok or bgr.shape[:2] != (1080, 1920):
                raise ValueError('Source frame unavailable')
            if abs(capture.get(cv2.CAP_PROP_POS_MSEC)/1000-frame/25) > 1e-7:
                raise ValueError('Source decode clock changed')
            source_crop = Image.fromarray(cv2.cvtColor(bgr[350:750, 400:800], cv2.COLOR_BGR2RGB)).resize((600, 600))
            panel = Image.new('RGB', (1200, 680), '#F5EADC'); draw = ImageDraw.Draw(panel)
            for column, (name, rows) in enumerate(all_rows.items()):
                crop = source_crop.copy(); overlay = ImageDraw.Draw(crop); row = rows[frame]
                if row['status'] == 'compared':
                    pixels = (np.asarray(row['projected_corners_px']) - [400, 350]) * 1.5
                    hull = cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2).tolist()
                    overlay.line([tuple(p) for p in hull + [hull[0]]], fill='#00D7FF', width=3)
                    detail = f'centroid {row["centroid_error_px"]:.1f}px | overlap {row["mask_overlap"]["iou"]:.3f}'
                else:
                    detail = 'No simulated state before front frame 8'
                if frame == 0:
                    for number, point in enumerate(image_points):
                        x, y = (point - [400, 350]) * 1.5
                        overlay.ellipse((x-4, y-4, x+4, y+4), fill='#00D7FF')
                        overlay.text((x+6, y-8), str(number), fill='#FFFFFF')
                panel.paste(crop, (column*600, 32))
                draw.text((column*600+10, 10), f'{name} | frame {frame} | {frame/25:.2f}s', fill='#214F50')
                draw.text((column*600+10, 640), detail, fill='#214F50')
            draw.text((10, 661), 'Cyan: conditional projected box. Same saved simulation; no camera/shape/physics promotion. Source: Physics101.', fill='#214F50')
            writer.write(cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR))
            if frame in (0, 7, 20, 26, 29, 36):
                panel.save(out / f'frame-{frame:04d}.png')
            if time.perf_counter()-started > CONTRACT['maximum_seconds']:
                raise TimeoutError('Camera diagnostic exceeded its bounded runtime')
        if capture.read()[0]:
            raise ValueError('Unexpected source suffix')
    finally:
        capture.release(); writer.release()
    bytes_used = sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
    if bytes_used > CONTRACT['maximum_bytes']:
        raise ValueError('Diagnostic export exceeded its byte bound')
    report = {'created_at': local_time(), 'status': 'complete_conditional_camera_diagnosis',
        'contract': CONTRACT, 'results': results, 'camera_fits': fits,
        'elapsed_seconds': time.perf_counter()-started, 'bytes_before_report_and_manifest': bytes_used,
        'new_source_decodes': 37, 'reused_recording_states': len(events), 'new_paid_calls': 0,
        'physical_validation': 'unverified', 'promotion': False,
        'source_annotations_sha256': digest(LABELS), 'unchanged_scene_hash': canonical_hash(scene),
        'limitations': [
            'Six manual corners and their face assignments are hypotheses; this is not automatic scene reconstruction.',
            'Camera and size are not independently identifiable from the fitted opening corners.',
            'Depth signs are retained, never selected by later image agreement.',
            'Source clocks are provider-aligned assumptions; nearest saved states are not interpolated or retimed.',
            'Silhouette-centroid error is not center-of-mass error or measured 3D pose.',
            'Corner bounds cover only supplied pixel annotations, not geometry, clock, identity, distortion or physics errors.',
            'First-trial evidence is exposed; the failed separate repeat and all sandboxes stay unchanged.',
        ]}
    write_json(out / 'report.json', report)
    summary = ''.join(f'<tr><td>{name}</td><td>{r["centroid_rmse_px"]:.2f}</td><td>{r["centroid_p95_px"]:.2f}</td><td>{r["mean_mask_iou"]:.3f}</td></tr>' for name, r in results.items())
    (out / 'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Weft · second-view camera check</title>
<style>body{{font:17px system-ui;line-height:1.5;max-width:1200px;margin:30px auto;padding:0 20px}}video,img{{max-width:100%}}td,th{{padding:8px;text-align:left}}</style>
<h1>Does the same simulation explain the second view?</h1><p>The camera fits six approximate opening-image corners. Both depth-axis assignments remain visible. Every scene dimension, material, starting state and simulated motion stays unchanged.</p>
<p><strong>This is a conditional projection, not a calibrated physical model.</strong> Different assumed box depths can fit the same opening pixels. A good opening fit alone cannot justify different collision geometry.</p>
<video controls preload="metadata" src="side-comparison.mp4"></video><p>37 source frames retained; pre-initialization frames have no simulated counterpart. Source crop: x=400–800, y=350–750; shown at 1.5×.</p>
<table><tr><th>Depth direction</th><th>Centroid RMS, px</th><th>95th percentile, px</th><th>Mean mask overlap</th></tr>{summary}</table>
<p><a href="report.json">Results and limits</a> · <a href="all-source-slots.json">Every source slot</a> · <a href="camera-fits.json">Camera fits</a> · <a href="depth-ambiguity-controls.json">Depth ambiguity</a> · <a href="inputs/source-corners.json">Source corner hypotheses</a> · <a href="../spatial-material-fit-01/selected-coast/replay.html">Unchanged 3D replay</a></p></html>''')
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT / 'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / Path(__file__).name)
    try:
        report = run(out)
    except Exception as error:
        failure = {'created_at': local_time(), 'status': 'failed', 'type': type(error).__name__, 'error': str(error)}
        write_json(out / 'failure.json', failure)
        write_json(out / 'report.json', failure)
        seal_suite(out)
        raise
    verification = seal_suite(out)
    print(json.dumps({'results': {name: {k: v for k, v in result.items() if k != 'final'} for name, result in report['results'].items()},
        'camera': {name: {'rmse_px': result['rmse_px'], 'orthographic': result['orthographic_diagnostic']} for name, result in report['camera_fits'].items()},
        'seconds': report['elapsed_seconds'], 'verification': verification}))


if __name__ == '__main__':
    main()
