"""Root-released single native read-only color probe; no retries or render."""
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'sources/host'))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, read_json, require, sha256, write_json


def main():
    require(not (HERE/'execution-started.json').exists(), 'Probe already started; preserve unknown/failed attempts')
    prepared = read_json(HERE/'prepared.json')
    for row in prepared['frozen_files']:
        require(sha256(HERE/row['path']) == row['sha256'], 'Frozen probe input changed')
    require(sha256(prepared['source_model']) == prepared['source_model_sha256'], 'Original model changed')
    require(sha256(prepared['blender_executable']) == prepared['blender_executable_sha256'], 'Blender changed')
    command = [prepared['blender_executable'], '--background', '--factory-startup', '--disable-autoexec',
               '--python-exit-code', '1', '--threads', '4', '--python', str(HERE/'probe_pixels.py')]
    require(command == prepared['native_command'], 'Native command differs from prepared command')
    write_json(HERE/'execution-started.json', {'command': command, 'prepared_sha256': sha256(HERE/'prepared.json')})
    minimal = {name: os.environ[name] for name in ('PATH', 'TMPDIR', 'LANG') if name in os.environ}
    minimal['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.clear()
    os.environ.update(minimal)
    started = time.monotonic()
    try:
        job = run_bounded(command, HERE/'native.log', HERE, timeout=60, memory_mb=512, output_mb=32)
        write_json(HERE/'native-job.json', job)
        require(job['exit_code'] == 0 and not job['failure'], 'Color probe failed; inspect preserved native evidence')
        report = read_json(HERE/'pixel-observations.json')
        require(report['status'] == 'pixel_measurements_complete_interpretation_pending', 'Missing actual pixel measurements')
        for row in prepared['frozen_files']:
            require(sha256(HERE/row['path']) == row['sha256'], 'Probe changed a frozen input')
        require(sha256(prepared['source_model']) == prepared['source_model_sha256'], 'Original model changed during probe')
        write_json(HERE/'completion.json', {'status': 'measurements_complete_interpretation_pending',
            'elapsed_seconds': time.monotonic()-started, 'native_jobs': 1, 'renders': 0,
            'source_inputs_unchanged': True, 'provider_calls': 0, 'quality_approved': False})
        write_json(HERE/'manifest.json', {'status': 'measurements_complete_interpretation_pending', 'artifacts': artifacts(HERE)})
    except Exception as error:
        write_json(HERE/'execution-failure.json', {'status': 'failed_preserved', 'error': type(error).__name__+': '+str(error),
            'automatic_retry': False, 'renders': 0, 'provider_calls': 0, 'quality_approved': False})
        write_json(HERE/'failure-manifest.json', {'status': 'failed_preserved', 'artifacts': artifacts(HERE)})
        raise


if __name__ == '__main__':
    main()
