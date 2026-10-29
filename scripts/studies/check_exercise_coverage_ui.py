"""Package rejected evidence for inspection and measure bounded coverage snapshots."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import numpy as np

from weft.benchmark import seal_suite
from weft.exercise_coverage import summarize_coverage
from weft.exercise_source import describe_source
from weft.io import canonical_hash, digest, local_time, write_json
from weft.lab import Lab
from weft.pipeline import _implementation_files, verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/jacks-gap-samples-01'
VIDEO = ROOT / 'data/exercise/jumping-jacks-burpees.webm'


def read(path):
    return json.loads(path.read_text())


def measure_lab(path):
    lab = Lab(path)
    try:
        before = lab.summary()
        state = canonical_hash(before.get('snapshot'))
        started = time.perf_counter()
        pending = lab.exercise_coverage()
        submit_seconds = time.perf_counter() - started
        if lab.coverage_future is not None:
            lab.coverage_future.result(timeout=10)
        result = lab.exercise_coverage()
        cold_seconds = time.perf_counter() - started
        if result['status'] != 'available':
            raise ValueError(f'Coverage failed for {path.name}: {result}')
        timings = []
        for _ in range(120):
            tick = time.perf_counter()
            payload = json.dumps(lab.exercise_coverage(), allow_nan=False).encode()
            timings.append(time.perf_counter() - tick)
        after = lab.summary()
        if canonical_hash(after.get('snapshot')) != state:
            raise ValueError('Coverage inspection changed the simulation')
        if any(key in after for key in ('markers', 'exercise_coverage', 'pose_coverage')):
            raise ValueError('Coverage must not inflate ordinary status polling')
        return result, {'initial_status': pending['status'], 'background_submit_seconds': submit_seconds,
            'cold_snapshot_seconds': cold_seconds, 'cached_calls': len(timings),
            'cached_json_p50_ms': float(np.percentile(timings, 50) * 1000),
            'cached_json_p95_ms': float(np.percentile(timings, 95) * 1000),
            'cached_json_max_ms': max(timings) * 1000, 'payload_bytes': len(payload),
            'simulation_state_unchanged': True, 'status_payload_excludes_pose_markers': True,
            'scope': 'In-process calls and serialization; not browser, HTTP or sustained-idle performance'}
    finally:
        lab.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.is_relative_to(SOURCE):
        raise ValueError('Use a new output outside the saved study')
    verified = verify_artifacts(SOURCE)
    evidence = read(SOURCE / 'crop/evidence.json')
    if digest(VIDEO) != evidence['source_sha256']:
        raise ValueError('Original video bytes changed')
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / Path(__file__).name)
    write_json(out / 'pre-run.json', {'created_at': local_time(), 'source_verification': verified,
        'implementation': implementation, 'new_inference_calls': 0, 'paid_requests': 0,
        'scope': 'Inspect existing rejected evidence and unchanged playable labs; no model promotion'})
    lab_root = out / 'jacks'
    run = lab_root / 'run'
    run.mkdir(parents=True)
    for name, checksum in read(SOURCE / 'crop/manifest.json')['files'].items():
        destination = run / ('input-report.json' if name == 'report.json' else name)
        shutil.copyfile(SOURCE / 'crop' / name, destination)
        if digest(destination) != checksum:
            raise ValueError('Copied candidate input changed')
    shutil.copyfile(SOURCE / 'crop/manifest.json', run / 'input-manifest.json')
    shutil.copyfile(SOURCE / 'new-observations.jsonl', run / 'gap-observations.jsonl')
    shutil.copyfile(SOURCE / 'pre-run.json', run / 'gap-study-plan.json')
    shutil.copyfile(VIDEO, run / 'source.webm')
    if digest(run / 'source.webm') != evidence['source_sha256']:
        raise ValueError('Source video copy changed')
    source = describe_source(evidence['source_sha256'])
    if source is None:
        raise ValueError('Original exercise attribution is required')
    write_json(run / 'source.json', source | {'artifact': 'source.webm', 'sha256': evidence['source_sha256']})
    for name, path in _implementation_files().items():
        target = run / 'implementation-source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(run / 'implementation.json', implementation)
    audit = read(run / 'root-motion.json')
    failure = {'status': 'rejected_mobile_body', 'created_at': local_time(), 'body_constructed': False,
        'errors': audit['quality']['errors'], 'source_study': str(SOURCE),
        'scope': 'Original video and failed estimates remain inspectable; no replacement simulation'}
    write_json(run / 'report.json', failure)
    seal_suite(run)
    settings = {'schema_version': 1, 'profile': 'exercise', 'video': str(VIDEO),
        'pose_mode': 'image', 'pose_identity': 'raw', 'body_model': 'mobile',
        'scope': 'Inspection-only rejected candidate; no worker or training started'}
    write_json(lab_root / 'lab-config.json', settings)
    events = [{'index': index, 'time': local_time(), 'stage': stage, 'message': message}
        for index, (stage, message) in enumerate([
            ('source', 'Preserved the original 1,356-frame source video and attribution'),
            ('observe', 'Reused 717 saved samples: 634 usable estimates, including partial poses'),
            ('world', 'Movement audit retains four long gaps and one missing body position'),
            ('failed', 'No continuous mobile body; inspect tracking issues in Video')])]
    write_json(lab_root / 'status.json', {'status': 'failed', 'events': events,
        'error': {'message': '; '.join(audit['quality']['errors']), 'time': local_time()}})
    measurements = {}
    cases = {'jacks_rejected': lab_root,
        'pushup_playable': ROOT / 'artifacts/selective-crop-bodies-01/kettlebell-pushup/selected_lab',
        'squat_playable': ROOT / 'artifacts/selective-crop-bodies-01/squat/selected_lab'}
    for name, path in cases.items():
        coverage, measurement = measure_lab(path)
        write_json(out / (name + '-coverage.json'), coverage)
        measurements[name] = measurement | {key: coverage[key] for key in (
            'decoded_frames', 'sampled_frames', 'usable_samples', 'partial_samples',
            'unusable_samples', 'competing_samples', 'unsampled_frames', 'movement')}
    analytic = deepcopy(evidence)
    analytic.update(frame_count=1800, decoded_frames=1800, fps=30, frames=[], quality={'errors': []})
    for index in range(900):
        frame = index * 2
        analytic['frames'].append({'id': f"{analytic['source_sha256']}:frame:{frame}",
            'frame': frame, 'time_s': frame / 30, 'decoder_timestamp_s': frame / 30,
            'status': 'tracked' if index % 2 else 'no_pose'})
    tick = time.perf_counter()
    maximum = summarize_coverage(analytic)
    maximum_payload = json.dumps(maximum, allow_nan=False).encode()
    capacity = {'scope': 'Analytic maximum sample count; no video or accuracy claim',
        'samples': len(maximum['markers']), 'issues': len(maximum['issues']),
        'summary_and_json_seconds': time.perf_counter() - tick, 'payload_bytes': len(maximum_payload)}
    if capacity['samples'] != 900 or capacity['issues'] != 450 or capacity['payload_bytes'] > 256 * 1024:
        raise ValueError('Maximum-count summary is incomplete or unexpectedly large')
    if implementation != {name: digest(path) for name, path in _implementation_files().items()}:
        raise ValueError('Runtime changed during inspection verification')
    report = {'created_at': local_time(), 'status': 'inspection_and_snapshot_verified',
        'measurements': measurements, 'capacity': capacity, 'paid_requests': 0,
        'new_inference_calls': 0, 'new_body_models': 0, 'physical_accuracy_verified': False,
        'launch': '.venv-lab/bin/python -m weft lab --resume ' + str(lab_root.relative_to(ROOT)),
        'browser_or_server_started': False}
    write_json(out / 'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
