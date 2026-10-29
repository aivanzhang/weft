"""Source-pinned full-interval toy-van development check; prepare never runs tracking."""

import argparse
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/general/corgi-hand-crank-van.webm'
CONFIG = ROOT / 'data/general/corgi-hand-crank-van.scene.json'
MANUAL = ROOT / 'data/general/corgi-hand-crank-van.manual.json'
PROVENANCE = ROOT / 'data/general/corgi-hand-crank-van.source.json'
EXPECTED_SOURCE = 'caad84aeda62860cbd682caac2a8a1e21e6f12f715615c9c56193dc5926aebc8'


def now():
    return datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def seal(directory):
    manifest = {'schema': 1, 'created_at': now(), 'files': {
        str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*'))
        if p.is_file()}}
    write(directory / 'manifest.json', manifest)
    return verify(directory)


def verify(directory):
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, wanted in manifest['files'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or digest(path) != wanted:
            raise ValueError(f'Frozen evidence changed: {name}')
    return {'verified_files': len(manifest['files']), 'manifest_sha256': digest(manifest_path)}


def prepare(out):
    if digest(SOURCE) != EXPECTED_SOURCE:
        raise ValueError('Original source identity changed')
    original = json.loads(CONFIG.read_text())
    manual = json.loads(MANUAL.read_text())
    tracking = original['tracking']
    if (tracking['source_sha256'] != EXPECTED_SOURCE or manual['source_sha256'] != EXPECTED_SOURCE
            or [tracking['start_frame'], tracking['end_frame']] != [0, 353]
            or manual['frame_interval_inclusive'] != [0, 353] or tracking['initialization_frame'] != 0
            or original['world']['target'] != [.5, .7]):
        raise ValueError('Prepared source/interval/target differs from the declared study')
    derived = deepcopy(original)
    derived['world']['target'][1] = .3
    check = deepcopy(derived)
    check['world']['target'][1] = .7
    if check != original:
        raise ValueError('Only the authored target y coordinate may change')
    if len(manual['observations']) != 6 or sum(r['center_px'] is not None for r in manual['observations']) != 5:
        raise ValueError('Preserve all five centers and the ambiguous crossfade slot')
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / 'inputs'
    inputs.mkdir()
    copies = {SOURCE: 'source.webm', CONFIG: 'original-scene.json', MANUAL: 'manual.json',
              PROVENANCE: 'source-provenance.json', Path(__file__).resolve(): 'check_toy_van.py'}
    for source, name in copies.items():
        shutil.copyfile(source, inputs / name)
    write(inputs / 'derived-scene.json', derived)
    write(inputs / 'pre-tracking-freeze.json', {
        'created_at': now(), 'original_paths': {name: str(p.relative_to(ROOT)) for p, name in copies.items()},
        'original_hashes': {name: digest(p) for p, name in copies.items()},
        'source_sha256': EXPECTED_SOURCE, 'source_interval_inclusive': [0, 353],
        'only_config_change': {'path': 'world.target[1]', 'before': .7, 'after': .3,
                               'reason': 'Authored target must fit 16:9 image-width height .5625'},
        'source_video_trimmed_or_resized': False, 'tracking_mode': 'unchanged default independent',
        'labels_exposed_to_tracker': False, 'manual_slots': 6, 'manual_centers': 5,
        'ambiguous_null_slots': 1, 'center_uncertainty_radius_source_px': 16,
        'episodes': 100, 'eval_episodes': 20, 'seed': 1729, 'worker_wall_seconds': 360,
        'outer_wall_seconds': 420, 'external_api_usd': 0, 'workers': 1,
        'expected_outcome': 'Moving-camera or coverage rejection, not a promised outcome',
        'limitations': ['Original crossfades and 3D hand manipulation retained',
                       'No dedicated crossfade or hand-manipulation detector',
                       'Sparse assistant visual development labels, not physical ground truth',
                       'No promotion or default changes']})
    frozen = seal(inputs)
    write(out / 'prepared.json', {'created_at': now(), 'status': 'prepared_not_executed',
                                  'inputs': frozen, 'authorization': 'Wait for root GO before run mode'})
    return frozen


def grade(trace, manual):
    if trace['source_sha256'] != manual['source_sha256']:
        raise ValueError('Trace source differs from frozen manual observations')
    if trace['selected_interval'] != [0, 353]:
        raise ValueError('Trace does not preserve the complete selected source interval')
    index = {(r['frame'], r['object_id']): r for r in trace['observations']}
    scale = [trace['source_width'] / trace['width'], trace['source_height'] / trace['height']]
    rows, errors = [], []
    for label in manual['observations']:
        prediction = index.get((label['frame'], manual['object_id']))
        status = prediction['status'] if prediction else 'missing_trace_slot'
        accepted = status == 'tracked'
        initialized = status == 'initialized'
        center = ([prediction['x_px'] * scale[0], prediction['y_px'] * scale[1]]
                  if accepted or initialized else None)
        observed = label['center_px']
        error = math.dist(center, observed) if center is not None and observed is not None else None
        if accepted and error is not None:
            errors.append(error)
        rows.append({**label, 'tracker_status': status, 'tracker_center_source_px': center,
                     'accepted_measurement': accepted, 'authored_initialization': initialized,
                     'error_source_px': error, 'scored_tracking_error': error if accepted else None,
                     'uncertainty_radius_source_px': 16,
                     'within_manual_uncertainty': error <= 16 if error is not None else None,
                     'excess_over_uncertainty_px': max(0., error-16) if error is not None else None,
                     'ambiguous_slot_acceptance': accepted and observed is None,
                     'uncertainty_note': 'Null crossfade is unknown, not physical absence; initialization is not tracker accuracy.'})
    return {'rows': rows, 'manual_slots': 6, 'manual_centers': 5, 'ambiguous_null_slots': 1,
            'noninitialization_center_slots': sum(r['center_px'] is not None and r['frame'] != 0 for r in rows),
            'accepted_noninitialization_centers': len(errors),
            'accepted_error_rmse_source_px': math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None,
            'accepted_max_error_source_px': max(errors) if errors else None,
            'accepted_within_manual_uncertainty': sum(e <= 16 for e in errors),
            'ambiguous_slot_accepted': any(r['ambiguous_slot_acceptance'] for r in rows),
            'missing_tracks_are_not_zero_error': True,
            'scope': 'All frozen slots retained; sparse source-only development checks, not physical validation'}


def run(out):
    inputs = out / 'inputs'
    prepared = json.loads((out / 'prepared.json').read_text())
    if verify(inputs) != prepared['inputs'] or digest(Path(__file__)) != digest(inputs / 'check_toy_van.py'):
        raise ValueError('Prepared inputs or study script changed; do not rerun an altered study')
    if (out / 'execution-freeze.json').exists() or (out / 'lab').exists():
        raise FileExistsError('Execution evidence already exists; preserve it without rerun')
    paths = [p for p in sorted((ROOT / 'src/weft').rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts]
    paths += [ROOT / p for p in ('pyproject.toml', 'requirements.lock', 'requirements-lab.lock')]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for p in paths:
        target = out / 'implementation' / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    command = [str(ROOT / '.venv-lab/bin/python'), '-m', 'weft', 'lab', '--video', str(inputs / 'source.webm'),
               '--profile', 'appearance', '--config', str(inputs / 'derived-scene.json'), '--out', str(out / 'lab'),
               '--episodes', '100', '--eval-episodes', '20', '--seed', '1729', '--no-serve', '--source-note',
               'OrangeLark64, CC BY-SA4.0; complete original toy-van handling with source crossfades retained']
    write(out / 'execution-freeze.json', {'created_at': now(), 'command': command, 'hashes': hashes,
                                         'input_verification': verify(inputs), 'maximum_seconds': 420})
    started = time.monotonic()
    timed_out = False
    with (out / 'cli-output.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=420)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
    lab = out / 'lab'
    report = {'created_at': now(), 'returncode': returncode, 'outer_timeout': timed_out,
              'seconds': time.monotonic()-started, 'source_sha256': EXPECTED_SOURCE,
              'pipeline_status': 'unknown', 'failure': None, 'manual_grade': None,
              'source_implementation_unchanged': hashes == {str(p.relative_to(ROOT)): digest(p) for p in paths},
              'input_verification_after': verify(inputs), 'source_crossfade_dedicated_detector': False,
              'physical_validation': False, 'default_or_promotion_changed': False}
    for filename, field in [('status.json', 'status_record'), ('failure.json', 'failure')]:
        if (lab / filename).is_file():
            report[field] = json.loads((lab / filename).read_text())
    report['pipeline_status'] = report.get('status_record', {}).get('status', 'unknown')
    trace_path = lab / 'run/appearance/appearance-tracks.json'
    if trace_path.is_file():
        trace = json.loads(trace_path.read_text())
        report['manual_grade'] = grade(trace, json.loads((inputs / 'manual.json').read_text()))
        report['trace'] = {'path': str(trace_path.relative_to(out)), 'sha256': digest(trace_path),
                           'selected_interval': trace['selected_interval'], 'decoded_frames': trace['decoded_frames'],
                           'camera_check': trace['camera_check'], 'physics_ready': trace['physics_ready']}
    else:
        report['manual_grade_unavailable_reason'] = 'No saved tracking trace; failure preserved without invented predictions'
    write(out / 'report.json', report)
    verification = seal(out)
    return {'verification': verification, 'returncode': returncode,
            'pipeline_status': report['pipeline_status'], 'failure': report['failure'],
            'manual_grade': report['manual_grade'], 'seconds': report['seconds']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    print(json.dumps(prepare(output) if args.mode == 'prepare' else run(output), indent=2))
