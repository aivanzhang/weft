"""Trusted child import paths and inherited limits; no video processing or physics."""

import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from types import SimpleNamespace

import pytest

from weft import lab, lab_worker


@pytest.mark.parametrize('cpu,size,expected_cpu,expected_size', [
    ((90, 95), (1 << 20, 2 << 20), (90, 95), (1 << 20, 2 << 20)),
    ((resource.RLIM_INFINITY, resource.RLIM_INFINITY), (resource.RLIM_INFINITY, resource.RLIM_INFINITY),
     (300, 305), (512 << 20, 512 << 20)),
    ((20, resource.RLIM_INFINITY), (4096, resource.RLIM_INFINITY), (20, 305), (4096, 512 << 20)),
    ((400, 500), (1 << 30, 2 << 30), (300, 305), (512 << 20, 512 << 20)),
    ((0, 0), (0, 0), (0, 0), (0, 0)),
    ((305, 305), (1024, 4096), (300, 305), (1024, 4096)),
])
def test_main_only_tightens_inherited_soft_and_hard_caps(monkeypatch, cpu, size, expected_cpu, expected_size):
    limits = {resource.RLIMIT_CPU: cpu, resource.RLIMIT_FSIZE: size}
    applied, executed = [], []
    monkeypatch.setattr(lab_worker.resource, 'getrlimit', lambda kind: limits[kind])

    def apply(kind, value):
        old = limits[kind]
        assert value[0] <= value[1]
        assert all(previous == resource.RLIM_INFINITY or current <= previous for current, previous in zip(value, old))
        limits[kind] = value; applied.append((kind, value))

    monkeypatch.setattr(lab_worker.resource, 'setrlimit', apply)
    monkeypatch.setattr(lab_worker.sys, 'argv', ['lab_worker.py', 'fixture-root'])
    monkeypatch.setattr(lab_worker, 'execute', lambda root: executed.append((root, dict(limits))) or 0)
    with pytest.raises(SystemExit) as result:
        lab_worker.main()
    assert result.value.code == 0
    assert applied == [(resource.RLIMIT_CPU, expected_cpu), (resource.RLIMIT_FSIZE, expected_size)]
    assert executed == [('fixture-root', {resource.RLIMIT_CPU: expected_cpu, resource.RLIMIT_FSIZE: expected_size})]


def frozen_package(tmp_path):
    runtime = tmp_path / 'frozen-runtime'; package = runtime / 'weft'; package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'lab.py').write_text('# Trusted frozen package entry path\n')
    return runtime, package


def launch_fixture(tmp_path, monkeypatch, *, failure=None):
    runtime, package = frozen_package(tmp_path)
    root = tmp_path / 'new-lab'; root.mkdir()
    machine = lab.Lab.__new__(lab.Lab)
    machine.root, machine.worker, machine.worker_log = root, None, None
    calls, watchers = [], []

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        if failure is not None:
            raise failure
        return SimpleNamespace(pid=123)

    class Thread:
        def __init__(self, **kwargs): watchers.append(kwargs)
        def start(self): watchers[-1]['started'] = True

    monkeypatch.setattr(lab, '__file__', str(package / 'lab.py'))
    monkeypatch.setattr(lab.subprocess, 'Popen', popen)
    monkeypatch.setattr(lab.threading, 'Thread', Thread)
    return machine, runtime, package, calls, watchers


def test_start_worker_pins_current_package_without_caller_paths_or_credentials(tmp_path, monkeypatch):
    machine, runtime, _, calls, watchers = launch_fixture(tmp_path, monkeypatch)
    for name in ('META_API_KEY', 'AWS_SECRET_ACCESS_KEY', 'PYTHONHOME', 'PYTHONUSERBASE', 'PYTHONSTARTUP'):
        monkeypatch.setenv(name, 'fixture-do-not-forward')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path / 'caller-imports'))
    machine.start_worker()
    try:
        argv, options = calls[0]
        assert argv == [sys.executable, '-P', '-B', '-m', 'weft.lab_worker', str(machine.root)]
        assert options['env'] == {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'PYTHONUNBUFFERED': '1',
            'PYTHONPATH': str(runtime), 'PYTHONNOUSERSITE': '1', 'PYTHONSAFEPATH': '1',
            'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
            'MPLCONFIGDIR': str(machine.root / 'runtime/matplotlib')}
        assert options['stdout'] is machine.worker_log and options['stderr'] == subprocess.STDOUT
        assert machine.worker.pid == 123
        assert watchers == [{'target': machine._watch_worker, 'daemon': True, 'name': 'weft-pipeline-watch', 'started': True}]
    finally:
        machine.worker_log.close()


def test_launch_failure_closes_its_log_without_starting_a_watcher(tmp_path, monkeypatch):
    machine, _, _, calls, watchers = launch_fixture(tmp_path, monkeypatch, failure=OSError('fixture launch failure'))
    with pytest.raises(OSError, match='fixture launch failure'):
        machine.start_worker()
    assert machine.worker_log.closed and len(calls) == 1 and not watchers
    assert machine.worker is None


def test_existing_log_is_preserved_before_any_launch(tmp_path, monkeypatch):
    machine, _, _, calls, watchers = launch_fixture(tmp_path, monkeypatch)
    path = machine.root / 'worker.log'; path.write_bytes(b'preserved earlier failure')
    with pytest.raises(FileExistsError):
        machine.start_worker()
    assert path.read_bytes() == b'preserved earlier failure' and not calls and not watchers


def test_real_child_selects_frozen_package_not_working_directory_or_editable_checkout(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    machine, runtime, package, calls, _ = launch_fixture(tmp_path, monkeypatch)
    (package / 'lab_worker.py').write_text("""
import json, os, sys
print(json.dumps({'file':__file__,'safe_path':sys.flags.safe_path,'no_user_site':sys.flags.no_user_site,
 'no_bytecode':sys.dont_write_bytecode,'pythonpath':os.environ.get('PYTHONPATH'),
 'secret_present':'META_API_KEY' in os.environ,'path':sys.path}))
""")
    hostile = tmp_path / 'caller-directory'; (hostile / 'weft').mkdir(parents=True)
    (hostile / 'weft/__init__.py').write_text("raise AssertionError('Caller directory shadowed trusted package')\n")
    monkeypatch.setenv('PYTHONPATH', str(hostile)); monkeypatch.setenv('META_API_KEY', 'fixture-do-not-forward')
    machine.start_worker(); argv, options = calls[0]; machine.worker_log.close()
    monkeypatch.setattr(subprocess, 'Popen', real_popen)
    child = subprocess.run(argv, env=options['env'], cwd=hostile, capture_output=True, text=True, timeout=5)
    assert child.returncode == 0, child.stderr
    assert len(child.stdout.encode()) <= 4096
    result = json.loads(child.stdout)
    assert Path(result['file']) == package / 'lab_worker.py'
    assert result['safe_path'] and result['no_user_site'] and result['no_bytecode']
    assert result['pythonpath'] == str(runtime) and str(hostile) not in result['path'] and not result['secret_present']
    assert not list(runtime.rglob('__pycache__')) and not list(hostile.rglob('__pycache__'))


def test_actual_worker_main_respects_smaller_isolated_os_limits(tmp_path):
    runtime, package = frozen_package(tmp_path)
    (package / 'lab_worker.py').write_bytes(Path(lab_worker.__file__).read_bytes())
    # Dependency stubs prevent this startup-only smoke from importing processing code.
    (package / 'io.py').write_text("def forbidden(*a, **k): raise AssertionError('Processing was not authorized')\ndigest=local_time=write_json=forbidden\n")
    (package / 'pipeline.py').write_text("def forbidden(*a, **k): raise AssertionError('Processing was not authorized')\n_implementation_files=_runtime_packages=run=verify_artifacts=forbidden\n")
    script = """
import json, resource, sys
sys.path.insert(0, sys.argv[1])
resource.setrlimit(resource.RLIMIT_CPU, (2, 3))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 2097152))
from weft import lab_worker
def startup_only(root):
    print(json.dumps({'cpu':resource.getrlimit(resource.RLIMIT_CPU),
        'size':resource.getrlimit(resource.RLIMIT_FSIZE),'file':lab_worker.__file__,'root':root}))
    return 0
lab_worker.execute = startup_only
sys.argv = ['lab_worker.py','startup-only-no-video']
lab_worker.main()
"""
    child = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(runtime)],
        capture_output=True, text=True, timeout=5, env={'PATH': os.defpath})
    assert child.returncode == 0, child.stderr
    assert len(child.stdout.encode()) <= 4096
    result = json.loads(child.stdout)
    assert result == {'cpu': [2, 3], 'size': [1048576, 2097152],
                      'file': str(package / 'lab_worker.py'), 'root': 'startup-only-no-video'}
    assert not list(runtime.rglob('__pycache__'))
