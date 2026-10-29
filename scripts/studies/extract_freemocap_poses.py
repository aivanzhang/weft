"""Extract both complete construction streams once, preserving rejected pose evidence."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from weft.benchmark import seal_suite
from weft.errors import UnsupportedVideo
from weft.exercise_vision import inspect_exercise, model_provenance
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'artifacts/freemocap-full-source-01'
MODEL = ROOT/'.local/models/pose_landmarker_full.task'


def read(path):
    return json.loads(path.read_text())


def worker(out, index):
    parent = read(out.parent/'pre-run.json')
    video = ROOT/parent['views'][index-1]['path']
    if digest(video) != parent['views'][index-1]['sha256']:
        raise ValueError('Source video changed')
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    rejection = None
    try:
        evidence = inspect_exercise(video, out, MODEL, pose_mode='image', pose_identity='raw')
    except UnsupportedVideo as error:
        rejection = str(error)
        evidence = read(out/'evidence.json')
    if (evidence['decoded_frames'] != 1108 or evidence['frame_count'] != 1108
            or len(evidence['frames']) != 554 or evidence['inference_stride'] != 2
            or [r['frame'] for r in evidence['frames']] != list(range(0, 1108, 2))):
        raise ValueError('Complete source/inference coverage changed')
    for row in evidence['frames']:
        if (abs(row['time_s']-row['frame']/30) > 1e-7
                or abs(row['decoder_timestamp_s']-row['time_s']) > 1e-7
                or row['id'] != evidence['source_sha256']+':frame:'+str(row['frame'])):
            raise ValueError('Pose source identity or clock changed')
    result = {'status': 'complete_pose_evidence', 'decoded_frames': 1108, 'inference_calls': 554,
        'pose_statuses': dict(Counter(r['status'] for r in evidence['frames'])),
        'pose_counts': dict(Counter(r['pose_count'] for r in evidence['frames'])),
        'monocular_admission_passed': evidence['quality']['passes'], 'monocular_rejection': rejection,
        'runtime_seconds': time.perf_counter()-started, 'source_sha256': evidence['source_sha256'],
        'model': evidence['model'], 'paid_calls': 0, 'body_built': False}
    write_json(out/'extraction-report.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--worker', type=int, choices=(1, 2))
    args = parser.parse_args(); out = args.out.resolve()
    if args.worker:
        worker(out, args.worker)
        return
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    source = read(SOURCE/'report.json'); manifest = read(SOURCE/'manifest.json')
    videos = [row for row in source['extracted'] if row['role'] == 'construction']
    if len(videos) != 2:
        raise ValueError('Exactly two construction cameras required')
    views = []
    for row in videos:
        path = SOURCE/row['file']
        if digest(path) != manifest['files'][row['file']]:
            raise ValueError('Pinned video changed')
        views.append({'path': str(path.relative_to(ROOT)), 'sha256': row['sha256'], 'member': row['member']})
    provenance = model_provenance(MODEL, variant='full')
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {
        'created_at': local_time(), 'views': views, 'model': provenance, 'runtime_hashes': implementation,
        'source_manifest_sha256': digest(SOURCE/'manifest.json'),
        'source_frames_each': 1108, 'source_fps': 30, 'inference_stride': 2, 'inference_calls_each': 554,
        'mode': 'IMAGE', 'identity': 'raw', 'workers': 2, 'subprocess_timeout_seconds': 120,
        'local_output_budget_bytes': 128000000, 'projected_paid_spend_usd': 0,
        'scope': 'Whole construction videos, including board/absent-person phases; preserve every failed or competing estimate. No cropping, selector, body construction, retiming or third-view access.',
        'calibration_status': 'Original monocular check failed; separate omitted-corner stereo development check passed. Neither is independent physical validation.',
    })
    started = time.perf_counter()
    def run(index):
        with (out/f'view-{index}.stdout').open('w') as stdout, (out/f'view-{index}.stderr').open('w') as stderr:
            try:
                result = subprocess.run([sys.executable, str(out/Path(__file__).name), '--worker', str(index),
                    '--out', str(out/f'view-{index}')], cwd=ROOT, stdout=stdout, stderr=stderr, timeout=120, check=False)
                return {'exit_code': result.returncode, 'timeout': False}
            except subprocess.TimeoutExpired:
                return {'exit_code': None, 'timeout': True}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = dict(zip(('view-1', 'view-2'), pool.map(run, (1, 2)), strict=True))
    if implementation != {name: digest(path) for name, path in _implementation_files().items()}:
        raise ValueError('Runtime changed while pose workers ran')
    reports = {name: read(out/name/'extraction-report.json') for name in jobs if (out/name/'extraction-report.json').exists()}
    total_bytes = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    complete = all(job['exit_code'] == 0 for job in jobs.values()) and len(reports) == 2
    report = {'created_at': local_time(), 'status': 'complete_construction_pose_evidence' if complete else 'pose_extraction_failed',
        'jobs': jobs, 'views': reports, 'runtime_seconds': time.perf_counter()-started, 'output_bytes': total_bytes,
        'within_output_budget': total_bytes <= 128000000, 'paid_calls': 0, 'reserved_camera_opened': False,
        'body_built': False, 'runtime_unchanged': True,
        'limitations': ['Image landmarks and detected person counts are estimates; no anatomy or contact accuracy is established.',
            'Monocular admission failures remain failures. A later explicit episode protocol must retain inactive and unsupported original intervals.']}
    write_json(out/'report.json', report)
    print(json.dumps({'status': report['status'], 'views': {k: {f: v[f] for f in ('pose_statuses', 'pose_counts', 'monocular_admission_passed', 'monocular_rejection')} for k, v in reports.items()},
        'runtime_seconds': report['runtime_seconds'], 'verification': seal_suite(out)}))
    if not complete or total_bytes > 128000000:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
