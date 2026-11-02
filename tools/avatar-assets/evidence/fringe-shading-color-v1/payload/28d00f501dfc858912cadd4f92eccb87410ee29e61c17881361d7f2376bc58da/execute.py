"""Run the one released renderer control under the unchanged fixed limits."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'sources/host'))
from avatar_assets.runner import run_bounded


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(name, data):
    with (HERE/name).open('x') as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write('\n')


def main():
    assert not (HERE/'execution-started.json').exists(), 'One attempt only'
    prepared = json.loads((HERE/'prepared.json').read_text())
    for row in prepared['frozen_files']:
        assert sha(HERE/row['path']) == row['sha256'], row['path']
    assert sha(prepared['blender']) == prepared['blender_sha256']
    write('execution-started.json', {'command': prepared['command'], 'prepared_sha256': sha(HERE/'prepared.json')})
    allowed = {key: os.environ[key] for key in ('PATH', 'TMPDIR', 'LANG') if key in os.environ}
    allowed['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.clear(); os.environ.update(allowed)
    job = run_bounded(prepared['command'], HERE/'native.log', HERE, timeout=60, memory_mb=1024, output_mb=16)
    write('native-job.json', job)
    unchanged = all(sha(HERE/row['path']) == row['sha256'] for row in prepared['frozen_files'])
    result_path = HERE/'native-result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    status = result['status'] if job['exit_code'] == 0 and not job['failure'] and result else 'native_failure_preserved'
    write('completion.json', {'status': status, 'frozen_inputs_unchanged': unchanged,
        'process_exit_verified': job.get('process_exit_verified'), 'automatic_retry': False,
        'new_generation_calls': 0, 'quality_approved': False})
    artifacts = [{'path': str(p.relative_to(HERE)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in sorted(HERE.rglob('*')) if p.is_file()]
    write('manifest.json', {'status': status, 'artifacts': artifacts, 'native_invocations': 1,
        'provider_ai_role_calls': 0, 'quality_approved': False})
    print(json.dumps({'status': status, 'pid': job['pid'], 'exit_code': job['exit_code'],
        'elapsed_seconds': job['elapsed_seconds'], 'sampled_peak_rss_bytes': job['sampled_peak_rss_bytes'],
        'final_output_bytes': job['final_output_bytes'], 'process_exit_verified': job.get('process_exit_verified')}))
    assert unchanged and job['exit_code'] == 0 and not job['failure']


if __name__ == '__main__':
    main()
