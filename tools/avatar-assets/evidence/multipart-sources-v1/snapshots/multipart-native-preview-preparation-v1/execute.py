"""One bounded native preview of an explicitly pinned local assembly."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'sources'))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, write_json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(prepared):
    for row in prepared['inputs']:
        path = ROOT / row['path']
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Pinned input changed: ' + row['path'])
    for row in prepared['snapshots']:
        path = HERE / row['path']
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Pinned preview source changed: ' + row['path'])
    if digest(prepared['blender']) != prepared['blender_sha256']:
        raise ValueError('Pinned Blender binary changed')


def main():
    if sys.argv[1:] != ['--execute']:
        raise ValueError('Use --execute only after the parent releases this pinned preparation')
    prepared = json.loads((HERE / 'prepared.json').read_text())
    check(prepared)
    out = ROOT / prepared['output']
    out.mkdir(mode=0o700)
    started = time.monotonic()
    report = {'schema_version': 1, 'kind': 'unrigged_assembly_native_preview',
              'status': 'unsupported', 'rig_created': False, 'animation_verified': False,
              'quality_approved': False, 'new_provider_calls': 0, 'new_pipeline_role_calls': 0,
              'prepared_sha256': digest(HERE / 'prepared.json'), 'limits': prepared['limits'],
              'measurement_scope': 'Native wall time and sampled direct-child RSS; controller overhead is separate'}
    try:
        source = ROOT / prepared['model']
        target = out / 'original.glb'
        remaining = source.stat().st_size
        if remaining > 64 * 1024**2:
            raise ValueError('Model exceeds 64 MiB')
        with source.open('rb') as incoming, target.open('xb') as saved:
            while remaining:
                block = incoming.read(min(1024**2, remaining))
                if not block:
                    raise ValueError('Source shrank during copy')
                saved.write(block)
                remaining -= len(block)
            if incoming.read(1):
                raise ValueError('Source grew during copy')
        if digest(target) != prepared['model_sha256']:
            raise ValueError('Copied model differs from the released input')
        (out / 'sources').mkdir()
        for name in ('blender_provider_preview.py', 'blender_verify.py'):
            shutil.copyfile(HERE / 'sources/scripts' / name, out / 'sources' / name)
        (out / 'tmp').mkdir()
        os.environ.clear()
        os.environ.update(PATH='/usr/bin:/bin', TMPDIR=str(out / 'tmp'), LANG='C.UTF-8',
                          PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                          VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [prepared['blender'], '--background', '--factory-startup', '--disable-autoexec',
                   '--python-exit-code', '1', '--threads', '4', '--python',
                   str(out / 'sources/blender_provider_preview.py'), '--', str(target), str(out),
                   '720', '16', 'false']
        write_json(out / 'started.json', {'command': command, 'prepared_sha256': report['prepared_sha256'],
                                        'model_sha256': prepared['model_sha256'], 'limits': prepared['limits']})
        job = run_bounded(command, out / 'native.log', out, timeout=120, memory_mb=2048, output_mb=128)
        write_json(out / 'job.json', job)
        report['job'] = job
        if job['exit_code'] != 0 or job['failure'] or not job['process_exit_verified']:
            raise ValueError('Native preview failed; preserve the exact resource receipt and partial files')
        check(prepared)
        if digest(target) != prepared['model_sha256']:
            raise ValueError('Native input changed')
        native = json.loads((out / 'native-inspection.json').read_text())
        if native['rigs'] or native['actions'] or native['animation_verified'] or len(native['renders']) != 4:
            raise ValueError('Unexpected native rig/action/render result')
        report.update(status='needs_visual_review', source_sha256=prepared['model_sha256'], native=native)
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
    report['controller_elapsed_seconds'] = round(time.monotonic() - started, 3)
    report['artifacts'] = artifacts(out, enforce_budget=False)
    total = sum(row['bytes'] for row in report['artifacts']) + len(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode()) + 1
    if total > 128 * 1024**2:
        report.update(status='unsupported', error={'type': 'ValueError', 'message': 'Final pack exceeds128MiB'})
    write_json(out / 'manifest.json', report)
    print(json.dumps({'out': str(out), 'status': report['status'], 'job': report.get('job'), 'error': report.get('error')}))
    return 0 if report['status'] == 'needs_visual_review' else 2


if __name__ == '__main__':
    raise SystemExit(main())
