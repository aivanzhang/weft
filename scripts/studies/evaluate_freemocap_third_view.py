"""Execute the frozen third-camera comparison once, retaining all unavailable slots."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.errors import UnsupportedVideo
from weft.exercise_vision import inspect_exercise, model_provenance
from weft.io import digest, local_time, write_json
from weft.multiview_pose import _observation
from weft.pipeline import _implementation_files, verify_artifacts
from weft.projected_joint_evaluation import grade_projected_joints


ROOT = Path(__file__).resolve().parents[2]
BODY = ROOT/'artifacts/freemocap-third-predictions-01'
BOARD = ROOT/'artifacts/freemocap-third-board-predictions-01'
MODEL = ROOT/'.local/models/pose_landmarker_full.task'
EXPECTED_BODY_SEAL = '72706982464be5b677c33452c6b52c85e9cc91de20a82004810b01dce8bd6790'
EXPECTED_BOARD_SEAL = 'cb5c2f064dc47e77ca471f3685f7a3f306975a607444751f9b21bf4c57eb8de2'


def read(path):
    return json.loads(path.read_text())


def decode(cap, frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, image = cap.read()
    if (not ok or image.shape[:2] != (1280, 720)
            or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(frame+1)) > 1e-7
            or abs(cap.get(cv2.CAP_PROP_POS_MSEC)/1000-frame/30) > 1e-7):
        raise ValueError('Exact third-camera source frame unavailable')
    return image


def worker(out, kind):
    body = read(BODY/'protocol.json'); board = read(BOARD/'protocol.json')
    video = ROOT/body['reserved_video']['path']
    if digest(video) != body['reserved_video']['sha256']:
        raise ValueError('Reserved video identity changed')
    out.mkdir(exist_ok=False); started = time.perf_counter()
    write_json(out/'pixel-access-started.json', {'created_at': local_time(), 'kind': kind,
        'body_predictions_seal': digest(BODY/'manifest.json'), 'board_predictions_seal': digest(BOARD/'manifest.json')})
    if kind == 'pose':
        rejection = None
        try:
            evidence = inspect_exercise(video, out, MODEL, pose_mode='image', pose_identity='raw')
        except UnsupportedVideo as error:
            rejection = str(error); evidence = read(out/'evidence.json')
        if (evidence['decoded_frames'] != 1108 or evidence['frame_count'] != 1108 or len(evidence['frames']) != 554
                or evidence['source_sha256'] != body['reserved_video']['sha256'] or evidence['fps'] != 30
                or evidence['width'] != 720 or evidence['height'] != 1280
                or [r['frame'] for r in evidence['frames']] != list(range(0, 1108, 2))):
            raise ValueError('Third-view extraction changed full source coverage')
        for row in evidence['frames']:
            if (abs(row['time_s']-row['frame']/30) > 1e-7 or abs(row['decoder_timestamp_s']-row['time_s']) > 1e-7
                    or row['id'] != evidence['source_sha256']+':frame:'+str(row['frame'])):
                raise ValueError('Third-view clock or evidence identity changed')
        report = {'status': 'complete_third_pose_evidence', 'decoded_frames': 1108, 'inference_calls': 554,
            'monocular_admission_passed': evidence['quality']['passes'], 'monocular_rejection': rejection}
    else:
        cv2.setNumThreads(1)
        target = cv2.aruco.CharucoBoard((7, 5), .058, .0464, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250))
        detector = cv2.aruco.CharucoDetector(target); cap = cv2.VideoCapture(str(video)); detections = []
        try:
            for frame in board['supported_frames']:
                image = decode(cap, frame)
                corners, ids, _, marker_ids = detector.detectBoard(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                row = {'frame': frame, 'time_s': frame/30, 'corners': [] if ids is None else
                    [{'id': int(i), 'pixel': p.tolist()} for i, p in zip(ids.ravel(), corners.reshape(-1, 2), strict=True)],
                    'marker_ids': [] if marker_ids is None else marker_ids.ravel().tolist()}
                detections.append(row)
                if frame in (board['supported_frames'][0], board['supported_frames'][-1]):
                    if not cv2.imwrite(str(out/f'source-{frame:04d}.png'), image):
                        raise ValueError('Board source export failed')
                if time.perf_counter()-started > 30:
                    raise TimeoutError('Third-board detection exceeded its work envelope')
        finally:
            cap.release()
        write_json(out/'all-selected-detections.json', detections)
        report = {'status': 'selected_third_board_evidence', 'returned_frame_reads': len(detections), 'board_detections': len(detections)}
    write_json(out/'worker-report.json', report | {'created_at': local_time(), 'runtime_seconds': time.perf_counter()-started})


def inspect_frame(image, frame, predictions, observations, target):
    canvas = Image.new('RGB', (1440, 705), 'white'); draw = ImageDraw.Draw(canvas)
    labels = ['Original reserved-camera image', 'Independent image-model observations', 'Raw two-view predictions', 'Fixed-length proposals (source failures included)']
    observed = {r['joint']: r['pixel'] for r in observations if r['frame'] == frame and r['pixel'] is not None}
    for column, label in enumerate(labels):
        panel = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((360, 640)); ink = ImageDraw.Draw(panel)
        if column:
            for p in observed.values():
                x, y = np.array(p)*.5; ink.ellipse((x-3, y-3, x+3, y+3), outline='#00ffff', width=2)
        if column >= 2:
            method = 'raw_two_view' if column == 2 else 'constrained_all_converged'
            for row in predictions[method]:
                if row['frame'] != frame or row['pixel'] is None:
                    continue
                x, y = np.array(row['pixel'])*.5
                ink.line((x-4, y, x+4, y), fill='#ffad21', width=2); ink.line((x, y-4, x, y+4), fill='#ffad21', width=2)
        canvas.paste(panel, (column*360, 65)); draw.text((column*360+5, 45), label, fill='black')
    draw.text((10, 8), f'Frame {frame}: cyan circles=image-model observations; orange crosses=frozen two-view predictions', fill='black')
    draw.text((10, 25), 'Comparison-view model agreement is not measured anatomy. No camera, clock, length or prediction was refitted.', fill='black')
    canvas.save(target)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', choices=('pose', 'board')); args = parser.parse_args(); out = args.out.resolve()
    if args.worker:
        worker(out, args.worker); return
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    verified = {p.name: verify_artifacts(p) for p in (BODY, BOARD)}
    if digest(BODY/'manifest.json') != EXPECTED_BODY_SEAL or digest(BOARD/'manifest.json') != EXPECTED_BOARD_SEAL:
        raise ValueError('The preregistered prediction seals changed')
    body, board = read(BODY/'protocol.json'), read(BOARD/'protocol.json')
    runtime = {n: digest(p) for n, p in _implementation_files().items()}
    if (runtime != body['runtime_hashes'] or model_provenance(MODEL, variant='full') != body['model']
            or digest(ROOT/'src/weft/projected_joint_evaluation.py') != body['scorer']['sha256']):
        raise ValueError('Runtime, model or frozen independent scorer changed before third-view access')
    out.mkdir(exist_ok=False); shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'prediction_verifications': verified,
        'runtime_hashes': runtime, 'pose_worker_timeout_seconds': 90, 'board_worker_timeout_seconds': 30,
        'maximum_workers': 2, 'maximum_total_returned_frame_reads': 1130, 'maximum_model_calls': 554,
        'maximum_output_bytes': 160000000, 'paid_calls': 0,
        'scope': 'Execute the two previously sealed protocols once. Independent image-view evidence, not marker ground truth. No refitting or candidate promotion.'})
    started = time.perf_counter()
    def run(kind):
        with (out/f'{kind}.stdout').open('w') as stdout, (out/f'{kind}.stderr').open('w') as stderr:
            try:
                p = subprocess.run([sys.executable, str(out/Path(__file__).name), '--worker', kind, '--out', str(out/kind)],
                    cwd=ROOT, stdout=stdout, stderr=stderr, timeout=90 if kind == 'pose' else 30, check=False)
                return {'exit_code': p.returncode, 'timeout': False}
            except subprocess.TimeoutExpired:
                return {'exit_code': None, 'timeout': True}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = dict(zip(('pose', 'board'), pool.map(run, ('pose', 'board')), strict=True))
    if runtime != {n: digest(p) for n, p in _implementation_files().items()}:
        raise ValueError('Runtime changed while comparison workers ran')
    if any(job['exit_code'] != 0 for job in jobs.values()):
        write_json(out/'report.json', {'created_at': local_time(), 'status': 'third_view_comparison_failed', 'jobs': jobs,
            'third_view_access_started': True, 'paid_calls': 0, 'no_fresh_holdout_claim': True})
        print(json.dumps({'status': 'failed', 'jobs': jobs, 'verification': seal_suite(out)})); raise SystemExit(1)
    evidence = read(out/'pose/evidence.json'); lookup = {r['frame']: r for r in evidence['frames']}
    predictions = read(BODY/'predictions-before-third-pixels.json'); observations = []
    for row in predictions['raw_two_view']:
        point, status = _observation(lookup.get(row['frame']), row['joint'], 720, 1280)
        observations.append({'frame': row['frame'], 'joint': row['joint'], 'pixel': point.tolist() if point is not None else None,
            'status': status, 'evidence_id': lookup[row['frame']]['id'] if row['frame'] in lookup else None})
    write_json(out/'all-comparison-image-observations.json', observations)
    human = {}
    for name, keep in (('complete_source', lambda r: True), ('active_400_through_1059', lambda r: 400 <= r['frame'] <= 1059)):
        result = grade_projected_joints({k: [r for r in rows if keep(r)] for k, rows in predictions.items()},
            [r for r in observations if keep(r)], baseline=body['scorer']['baseline'], limit_px=body['scorer']['limit_px'])
        write_json(out/f'{name}-all-slot-errors.json', result.pop('all_slot_errors')); human[name] = result
    write_json(out/'human-image-comparison.json', human)
    actual = {r['frame']: {p['id']: p['pixel'] for p in r['corners']} for r in read(out/'board/all-selected-detections.json')}
    board_rows = []; errors = []; supported = 0; missing = 0
    for row in read(BOARD/'board-predictions-before-third-pixels.json'):
        corners = []
        for point in row['corners']:
            observation = actual.get(row['frame'], {}).get(point['id'])
            error = None if observation is None else float(np.linalg.norm(np.subtract(point['pixel'], observation)))
            corners.append({'id': point['id'], 'predicted_pixel': point['pixel'], 'observed_pixel': observation, 'error_px': error})
            if error is None:
                missing += 1
            else:
                errors.append(error)
        supported += sum(c['error_px'] is not None for c in corners) >= board['checks']['minimum_compared_corners_each_pose']
        board_rows.append({'frame': row['frame'], 'construction_status': row['construction_status'], 'corners': corners})
    checks = {'enough_supported_poses': supported >= board['checks']['minimum_poses'],
        'p95': bool(errors) and float(np.percentile(errors, 95)) <= board['checks']['maximum_p95_px'],
        'maximum': bool(errors) and max(errors) <= board['checks']['maximum_error_px']}
    board_result = {'checks': checks, 'passed': all(checks.values()), 'all_prefix_slots': len(board_rows),
        'supported_poses': supported, 'compared_corners': len(errors), 'missing_corners': missing,
        'p95_px': float(np.percentile(errors, 95)) if errors else None, 'maximum_px': max(errors) if errors else None}
    write_json(out/'board-all-slot-comparisons.json', board_rows); write_json(out/'board-comparison.json', board_result)
    video = ROOT/body['reserved_video']['path']; cap = cv2.VideoCapture(str(video))
    try:
        for frame in board['post_grading_human_inspection_frames']:
            image = decode(cap, frame)
            if not cv2.imwrite(str(out/f'view-3-source-{frame:04d}.png'), image):
                raise ValueError('Third-view inspection export failed')
            inspect_frame(image, frame, predictions, observations, out/f'third-view-comparison-{frame:04d}.png')
    finally:
        cap.release()
    if digest(video) != body['reserved_video']['sha256'] or any(verify_artifacts(p) != verified[p.name] for p in (BODY, BOARD)):
        raise ValueError('Source or frozen predictions changed during comparison')
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    report = {'created_at': local_time(), 'status': 'frozen_third_view_comparison_complete', 'jobs': jobs,
        'human': human, 'board': board_result, 'runtime_seconds': time.perf_counter()-started, 'output_bytes': size,
        'new_pose_inference_calls': 554, 'full_source_decoded_frames': 1108, 'additional_board_frame_reads': 18,
        'additional_post_grading_inspection_reads': 4, 'paid_calls': 0, 'third_camera_opened': True,
        'reservation_status': 'Exposed by this frozen comparison; development evidence for future work, not a fresh final holdout.',
        'candidate_promoted': False, 'body_built': False, 'floor': None, 'runtime_unchanged': True,
        'limitations': [body['reference_quality'], 'The original construction-view support regression remains; comparison-camera gains cannot waive it.',
            'The original monocular calibration failure remains separate from this stereo-to-third-camera board check.']}
    write_json(out/'report.json', report)
    print(json.dumps({'status': report['status'], 'human': human, 'board': board_result,
        'runtime_seconds': report['runtime_seconds'], 'verification': seal_suite(out)}))
    if size > 160000000:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
