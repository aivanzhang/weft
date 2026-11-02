"""One explicitly released native build, then bounded read-only CPU export verification."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE/'sources/host'))
from avatar_assets.runner import run_bounded


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    prepared = json.loads((HERE/'prepared.json').read_text())
    assert prepared['status'] == 'source_frozen_native_held', 'Unexpected preparation state'
    release = json.loads((HERE/'native-release.json').read_text())
    assert release['status'] == 'root_released_single_native_component' and release['prepared_sha256'] == sha(HERE/'prepared.json'), 'Native lane has not been released'
    assert not (HERE/'execution-started.json').exists(), 'One native attempt only'
    for row in prepared['files']:
        assert sha(HERE/row['path']) == row['sha256'], row['path']
    for row in prepared['inputs']:
        assert sha(ROOT/row['path']) == row['sha256'], row['path']
    assert sha(prepared['blender']) == prepared['blender_sha256']
    config = json.loads((HERE/'config.json').read_text())
    out = ROOT/config['output']
    out.mkdir(exist_ok=False)
    save(HERE/'execution-started.json', {'prepared_sha256': sha(HERE/'prepared.json'), 'command': prepared['native_command']})
    allowed = {key: os.environ[key] for key in ('PATH', 'TMPDIR', 'LANG') if key in os.environ}
    allowed.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    os.environ.clear(); os.environ.update(allowed)
    limits = config['native_limits']
    native = run_bounded(prepared['native_command'], HERE/'native.log', out,
                         timeout=limits['seconds'], memory_mb=limits['memory_mib'], output_mb=limits['output_mib'])
    save(HERE/'native-job.json', native)
    postprocess, audit = None, None
    if native['exit_code'] == 0 and not native['failure'] and native['process_exit_verified']:
        postprocess = run_bounded(prepared['postprocess_command'], HERE/'postprocess.log', out,
                                  timeout=30, memory_mb=512, output_mb=128)
        save(HERE/'postprocess-job.json', postprocess)
    if postprocess and postprocess['exit_code'] == 0 and not postprocess['failure'] and postprocess['process_exit_verified']:
        folder = out/'audit'; folder.mkdir(exist_ok=False)
        limits = config['verification_limits']
        audit = run_bounded(prepared['audit_command'], HERE/'audit.log', folder,
                            timeout=limits['seconds'], memory_mb=limits['memory_mib'], output_mb=limits['output_mib'])
        save(HERE/'audit-job.json', audit)
    status = ('component_numerics_passed_pending_visual_review' if audit and audit['exit_code'] == 0
              and not audit['failure'] and audit['process_exit_verified'] else 'failure_preserved_no_retry')
    unchanged = all(sha(HERE/row['path']) == row['sha256'] for row in prepared['files'])
    assert unchanged
    save(out/'completion.json', {'status': status, 'frozen_inputs_unchanged': unchanged,
         'native_process_exit_verified': native['process_exit_verified'],
         'audit_process_exit_verified': audit['process_exit_verified'] if audit else None,
         'native_job': str((HERE/'native-job.json').relative_to(ROOT)),
         'postprocess_job': str((HERE/'postprocess-job.json').relative_to(ROOT)) if postprocess else None,
         'audit_job': str((HERE/'audit-job.json').relative_to(ROOT)) if audit else None,
         'automatic_retry': False, 'quality_approved': False, 'provider_ai_role_calls': 0})
    files = [{'path': str(path.relative_to(out)), 'bytes': path.stat().st_size, 'sha256': sha(path)}
             for path in sorted(out.rglob('*')) if path.is_file()]
    save(out/'manifest.json', {'schema_version': 1, 'status': status, 'source_sha256': config['source_sha256'],
         'preparation': str(HERE.relative_to(ROOT)), 'prepared_sha256': sha(HERE/'prepared.json'),
         'artifacts': files, 'quality_approved': False, 'provider_ai_role_calls': 0})
    print(json.dumps({'status': status, 'native_pid': native['pid'], 'native_exit_code': native['exit_code'],
        'native_seconds': native['elapsed_seconds'], 'sampled_native_rss': native['sampled_peak_rss_bytes'],
        'native_output_bytes': native['final_output_bytes'], 'audit_exit_code': audit['exit_code'] if audit else None}))
    assert status == 'component_numerics_passed_pending_visual_review'


if __name__ == '__main__':
    main()
