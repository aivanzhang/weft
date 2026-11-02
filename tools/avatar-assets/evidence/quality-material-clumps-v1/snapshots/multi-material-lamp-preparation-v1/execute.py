"""One public finish call, with direct native ownership and bounded CPU phases."""
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.runner import _output_bytes, resident_bytes, run_bounded


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value, *, replace=False):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    target = path.with_suffix(path.suffix + '.tmp') if replace else path
    with target.open('x') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(target, path)


def check_pins(prepared):
    for row in prepared['inputs']:
        path = ROOT / row['path']
        if path.is_symlink() or not path.is_file() or sha(path) != row['sha256']:
            raise ValueError('Frozen input changed: ' + row['path'])
    for row in prepared['snapshots']:
        path = HERE / row['path']
        if path.is_symlink() or not path.is_file() or sha(path) != row['sha256']:
            raise ValueError('Frozen preparation changed: ' + row['path'])
    if sha(prepared['blender']) != prepared['blender_sha256']:
        raise ValueError('Frozen Blender binary changed')


def main():
    if sys.argv[1:] != ['--execute']:
        raise ValueError('Prepared only; parent must explicitly run execute.py --execute once')
    os.chdir(ROOT)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Controller received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    prepared = json.loads((HERE / 'prepared.json').read_text())
    check_pins(prepared)
    out = ROOT / prepared['output']
    if out.exists() or out.is_symlink():
        raise ValueError('Immutable output already exists; no automatic retry')
    work = HERE / 'execution'
    work.mkdir()
    write(work / 'started.json', {'prepared_sha256': sha(HERE / 'prepared.json'),
                                'source_sha256': prepared['source_sha256']})
    command = [str(ROOT / '.venv/bin/python'), str(HERE / 'finish_worker.py')]
    env = {k: os.environ[k] for k in ('PATH', 'TMPDIR', 'LANG') if k in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    os.environ.clear()
    os.environ.update(env)
    report = {'schema_version': 1, 'status': 'running', 'prepared_sha256': sha(HERE / 'prepared.json'),
              'source_sha256': prepared['source_sha256'], 'output': prepared['output'],
              'native_calls': 0, 'cpu_phases': {}, 'cpu_exit_code': None, 'cpu_exit_verified': False,
              'native_job': None, 'worker_result': None, 'failure': None, 'cleanup_events': [],
              'new_provider_calls': 0, 'new_pipeline_role_calls': 0, 'quality_approved': False}
    started = time.monotonic()
    phase, phase_started, buffer, result_seen = 'preflight', started, b'', False
    cpu_log = work / 'cpu.log'
    with cpu_log.open('xb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                   cwd=ROOT, env=env, start_new_session=True)
        report['cpu_pid'] = process.pid
        write(work / 'cpu-handle.json', {'pid': process.pid, 'command': command, 'cwd': str(ROOT)})
        try:
            while True:
                elapsed = time.monotonic() - phase_started
                measured = report['cpu_phases'].setdefault(phase, {'elapsed_seconds': 0, 'sampled_peak_rss_bytes': None,
                                                                 'rss_samples': 0, 'limit_seconds': 30, 'limit_mib': 512})
                measured['elapsed_seconds'] = elapsed
                if process.poll() is None:
                    try:
                        rss = resident_bytes(process.pid)
                    except ProcessLookupError:
                        process.wait(timeout=.25)
                    else:
                        measured['sampled_peak_rss_bytes'] = max(measured['sampled_peak_rss_bytes'] or 0, rss)
                        measured['rss_samples'] += 1
                        if rss > 512 * 1024**2:
                            raise ValueError('CPU resident-memory limit exceeded')
                if elapsed > 30:
                    raise ValueError('CPU ' + phase + ' wall-time limit exceeded')
                if cpu_log.stat().st_size > 16 * 1024**2:
                    raise ValueError('CPU log bound exceeded')
                size = (_output_bytes(out) if out.exists() else 0) + _output_bytes(work)
                report['combined_output_bytes'] = size
                if size > 128 * 1024**2:
                    raise ValueError('Combined output bound exceeded')
                write(work / 'controller-monitor.json', report, replace=True)
                ready, _, _ = select.select([process.stdout], [], [], .1)
                if ready:
                    data = os.read(process.stdout.fileno(), 65536)
                    buffer += data
                    if len(buffer) > 131072:
                        raise ValueError('IPC response bound exceeded')
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        msg = json.loads(line)
                        write(work / ('ipc-' + str(len(list(work.glob('ipc-*.json')))).zfill(2) + '.json'), msg)
                        if msg.get('kind') == 'native':
                            if phase != 'preflight' or report['native_calls'] != 0:
                                raise ValueError('Repeated/unexpected native request')
                            expected = [prepared['blender'], '--background', '--factory-startup', '--disable-autoexec',
                                        '--python-exit-code', '1', '--threads', '4', '--python',
                                        str(out / 'sources/blender_finish_generated.py'), '--',
                                        str(out / 'finish-config.json'), str(out)]
                            if (set(msg) != {'kind', 'command', 'log', 'output', 'cwd', 'requested_limits'}
                                or msg['command'] != expected or msg['cwd'] != str(ROOT)
                                or msg['output'] != str(out) or msg['log'] != str(out / 'finish.log')
                                or msg['requested_limits'] != {'timeout': 600, 'memory_mb': 2048, 'output_mb': 512}):
                                raise ValueError('Native request differs from the exact public finish command')
                            check_pins(prepared)
                            copied = out / 'sources/blender_finish_generated.py'
                            if sha(copied) != prepared['native_finisher_sha256']:
                                raise ValueError('Copied native source changed')
                            report['native_calls'] = 1
                            write(work / 'native-started.json', {'command': expected, 'cwd': str(ROOT),
                                                               'limits': prepared['native_bounds']})
                            # The worker blocks on stdin; only this parent launches/cleans native work.
                            job = run_bounded(expected, out / 'finish.log', out,
                                              timeout=120, memory_mb=2048, output_mb=128)
                            report['native_job'] = job
                            write(work / 'native-job.json', job)
                            if not job.get('process_exit_verified') or job.get('manual_stop_required'):
                                raise ValueError('Native custody is unresolved; stop without further processing')
                            check_pins(prepared)
                            process.stdin.write((json.dumps({'kind': 'native_result', 'job': job}) + '\n').encode())
                            process.stdin.flush()
                            phase, phase_started = 'postprocess', time.monotonic()
                        elif msg.get('kind') in ('complete', 'error'):
                            if result_seen:
                                raise ValueError('Duplicate worker result')
                            report['worker_result'], result_seen = msg, True
                        else:
                            raise ValueError('Unknown worker protocol message')
                    if not data and process.poll() is not None:
                        break
                if process.poll() is not None and not ready:
                    break
            if buffer:
                raise ValueError('Incomplete final IPC message')
            if process.wait(timeout=1) != 0 or not result_seen or report['worker_result']['kind'] != 'complete':
                raise ValueError('Public finish worker failed; preserve exact logs and result')
            if report['native_calls'] != 1 or report['native_job']['exit_code'] != 0 or report['native_job']['failure']:
                raise ValueError('Public finish did not complete a successful real native call')
            manifest = out / 'manifest.json'
            if json.loads(manifest.read_text()).get('status') != 'needs_visual_review':
                raise ValueError('Unexpected public finish terminal manifest')
            check_pins(prepared)
            report.update(status='finished_needs_independent_audit', model_sha256=sha(out / 'model.glb'),
                          manifest_sha256=sha(manifest), frozen_inputs_unchanged=True)
        except BaseException as error:
            report.update(status='failed', failure={'type': type(error).__name__, 'message': str(error)[:4000]})
            raise
        finally:
            for sig, grace in ((signal.SIGTERM, 5), (signal.SIGKILL, 2)):
                if process.poll() is not None:
                    break
                try:
                    os.killpg(process.pid, sig)
                    report['cleanup_events'].append({'signal': int(sig), 'scope': 'owned_CPU_group', 'sent': True})
                    process.wait(timeout=grace)
                except (OSError, subprocess.TimeoutExpired) as error:
                    report['cleanup_events'].append({'signal': int(sig), 'error': str(error)[:500]})
            report.update(cpu_exit_code=process.poll(), cpu_exit_verified=process.poll() is not None,
                          elapsed_seconds=time.monotonic() - started,
                          measurement_scope='Per-phase sampled CPU-child RSS; separate native direct-child receipt; not whole-tree RSS')
            if not report['cpu_exit_verified']:
                report['status'] = 'needs_inspection'
            write(work / 'receipt.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'elapsed_seconds', 'cpu_exit_verified', 'native_calls')}))


if __name__ == '__main__':
    main()
