"""Offline trusted-export and isolated-pipe tests; no Git process or physical simulation."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
import json
from pathlib import Path
import textwrap
import threading

import pytest

from weft import portal_lab_runtime as runtime


LAB = textwrap.dedent('''
    import json, os, threading, time
    from pathlib import Path
    class Lab:
        def __init__(self, root):
            self.root = Path(root)
            self.lock = threading.Lock()
            self.recorder = None
            self.export_future = None
            self.count = 0
        def load_session(self):
            if (self.root / 'reject').exists():
                raise ValueError('fixture pinned runtime rejected')
        def summary(self):
            return {'count': self.count, 'root': str(self.root),
                    'credentials': {key: value for key, value in os.environ.items() if 'KEY' in key},
                    'module': __file__}
        def control(self, path, data):
            if path == '/fail': raise ValueError('fixture action rejected')
            if path == '/sleep': time.sleep(.5)
            if path == '/large': return {'large': 'x' * (9 << 20)}
            self.count += 1
            if path == '/record': self.recorder = True
            return {'count': self.count}
        def inspection(self): return {'inspection': True}
        def exercise_coverage(self): return {'coverage': True}
        def recording_inspection(self, name): return {'name': name}
        def recording_evidence(self, name): return {'name': name}
        def close(self):
            (self.root / 'closed.json').write_text(json.dumps({'recording': self.recorder is not None}))
''').encode()


def fake_exporter(monkeypatch, files):
    project = Path(__file__).parents[1]
    exporter = runtime._exporter(project)
    blobs, entries = {}, []
    for name, content in files.items():
        oid = sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        blobs[oid] = content
        entries.append(f'100644 blob {oid} {len(content)}\t{name}'.encode() + b'\0')
    calls = []
    def git(project, *args, limit, data=None):
        calls.append(args)
        if args[0] == 'ls-tree':
            assert args[3] in runtime.TRUSTED_RELEASES
            return b''.join(entries)
        assert args == ('cat-file', '--batch') and len(data) <= 12 << 10
        return b''.join(oid + b' blob ' + str(len(blobs[oid.decode()])).encode() + b'\n'
                       + blobs[oid.decode()] + b'\n' for oid in data.splitlines())
    monkeypatch.setattr(exporter, '_git', git)
    monkeypatch.setattr(runtime, '_exporter', lambda project: exporter)
    return calls


@pytest.fixture
def archive(tmp_path, monkeypatch):
    files = {'src/weft/__init__.py': b'', 'src/weft/lab.py': LAB,
             'src/weft/shared_exercise_lab.py': b'# Synthetic fixture only\n',
             'pyproject.toml': b'', 'requirements-lab.lock': b''}
    calls = fake_exporter(monkeypatch, files)
    root = tmp_path / 'lab'
    root.mkdir()
    return root, runtime.prepare_runtime(tmp_path), calls


def test_export_cache_rechecks_full_trusted_tree_and_rejects_extra_code(archive):
    root, saved, calls = archive
    assert (saved / 'src/weft/lab.py').read_bytes() == LAB
    assert runtime.prepare_runtime(root.parent) == saved
    assert len([call for call in calls if call[0] == 'cat-file']) == 1
    (saved / 'src/weft/extra.py').write_text('raise RuntimeError("not trusted")')
    with pytest.raises(ValueError, match='unexpected file'):
        runtime.prepare_runtime(root.parent)


def test_export_preserves_changed_cache_and_refuses_execution(archive):
    root, saved, _ = archive
    path = saved / 'src/weft/lab.py'
    path.write_bytes(b'# changed')
    with pytest.raises(ValueError, match='differs from trusted Git'):
        runtime.prepare_runtime(root.parent)
    assert path.read_bytes() == b'# changed'


def test_export_batches_requests_within_existing_git_helper_bound(tmp_path, monkeypatch):
    files = {'src/weft/__init__.py': b'', 'src/weft/lab.py': LAB,
             'src/weft/shared_exercise_lab.py': b'', 'pyproject.toml': b'', 'requirements-lab.lock': b''}
    files.update({f'src/weft/fixture_{i}.py': str(i).encode() for i in range(320)})
    calls = fake_exporter(monkeypatch, files)
    runtime.prepare_runtime(tmp_path)
    assert len([call for call in calls if call[0] == 'cat-file']) == 2


def test_proxy_routes_same_root_serially_without_credentials_and_closes_recording(archive, monkeypatch):
    root, saved, _ = archive
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-reach-child')
    monkeypatch.setenv('META_API_KEY', 'must-not-reach-child')
    lab = runtime.ArchivedLab(root, saved)
    try:
        info = lab.summary()
        assert info['root'] == str(root) and info['credentials'] == {}
        assert info['module'] == str(saved / 'src/weft/lab.py')
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: lab.control('/step', {}), range(12)))
        assert sorted(item['count'] for item in results) == list(range(1, 13))
        assert lab.inspection()['status'] in {'loading', 'unavailable'}
        assert lab._inspection_future.result()['status'] == 'unavailable'
        assert lab.exercise_coverage() == {'coverage': True}
        assert lab.recording_inspection('a') == {'name': 'a'}
        assert lab.recording_evidence('a') == {'name': 'a'}
        assert lab.recorder is None and lab.export_future.done()
        lab.control('/record', {})
        assert lab.recorder is True
        with pytest.raises(ValueError, match='no configured avatar'):
            lab.avatars.resource('a', 'b')
    finally:
        lab.close()
    assert (root / 'closed.json').read_text() == '{"recording": true}'
    assert lab._process.poll() is not None


def test_inspection_uses_one_current_offthread_reader_not_archived_rpc(archive, monkeypatch):
    from weft import lab_inspection

    root, saved, _ = archive
    entered, release = threading.Event(), threading.Event()
    calls = []
    def inspect(run, *, profile):
        calls.append((run, profile, threading.current_thread().ident))
        entered.set()
        assert release.wait(3)
        return {'status': 'available', 'actors': [{'id': 'authored_support'}]}
    monkeypatch.setattr(lab_inspection, 'load_inspection', inspect)
    lab = runtime.ArchivedLab(root, saved)
    try:
        assert lab.inspection() == {'status': 'loading'}
        assert entered.wait(3)
        for _ in range(5):
            assert lab.inspection() == {'status': 'loading'}
        assert len(calls) == 1 and calls[0] == (root / 'run', 'shared-world', calls[0][2])
        assert calls[0][2] != threading.current_thread().ident
        assert lab.summary()['count'] == 0
        release.set()
        assert lab._inspection_future.result(timeout=3)['status'] == 'available'
        assert lab.inspection() == {'status': 'available', 'actors': [{'id': 'authored_support'}]}
        assert len(calls) == 1
    finally:
        release.set()
        lab.close()
    assert lab._inspection_executor is None
    with pytest.raises(RuntimeError, match='closed'):
        lab.inspection()


def test_operation_errors_do_not_discard_worker_but_unknown_rpc_is_blocked(archive):
    root, saved, _ = archive
    lab = runtime.ArchivedLab(root, saved)
    try:
        with pytest.raises(ValueError, match='action rejected'):
            lab.control('/fail', {})
        with pytest.raises(ValueError, match='Unsupported'):
            lab._rpc('__getattribute__', 'root')
        assert lab.summary()['count'] == 0
    finally:
        lab.close()


def test_timeout_terminates_child_without_retrying_action(archive):
    root, saved, _ = archive
    lab = runtime.ArchivedLab(root, saved, request_seconds=.03)
    with pytest.raises(TimeoutError, match='timed out'):
        lab.control('/sleep', {})
    assert lab._process.poll() is not None
    with pytest.raises(RuntimeError, match='closed'):
        lab.summary()
    lab.close()


def test_admission_failure_surfaces_before_proxy_becomes_available(archive):
    root, saved, _ = archive
    (root / 'reject').touch()
    with pytest.raises(ValueError, match='pinned runtime rejected'):
        runtime.ArchivedLab(root, saved)


def test_request_and_response_bounds_are_enforced(archive):
    root, saved, _ = archive
    lab = runtime.ArchivedLab(root, saved)
    try:
        with pytest.raises(ValueError, match='request exceeds'):
            lab.control('/step', {'huge': 'x' * runtime.MAX_REQUEST})
        with pytest.raises(ValueError, match='response exceeds'):
            lab.control('/large', {})
        assert lab.summary()['count'] == 0
    finally:
        lab.close()


@pytest.mark.parametrize('error', [None, 'corrupt evidence', *runtime.COMPATIBILITY_ERRORS])
def test_factory_falls_back_only_for_exact_runtime_mismatch(tmp_path, monkeypatch, error):
    from weft import lab as current
    calls = []
    class Local:
        def __init__(self, root): calls.append('current')
        def load_session(self):
            if error: raise ValueError(error)
        def close(self): calls.append('closed')
    monkeypatch.setattr(current, 'Lab', Local)
    monkeypatch.setattr(runtime, 'compatible_runtime', lambda root, project: calls.append('trusted') or project)
    monkeypatch.setattr(runtime, 'ArchivedLab', lambda root, saved: calls.append('archived') or 'proxy')
    if error == 'corrupt evidence':
        with pytest.raises(ValueError, match=error): runtime.open_portal_lab(tmp_path, project=tmp_path)
        assert calls == ['current', 'closed']
    elif error:
        assert runtime.open_portal_lab(tmp_path, project=tmp_path) == 'proxy'
        assert calls == ['current', 'closed', 'trusted', 'archived']
    else:
        assert isinstance(runtime.open_portal_lab(tmp_path, project=tmp_path), Local)
        assert calls == ['current']


def sealed_selection(root, physics, policy):
    def encode(value): return (json.dumps(value, sort_keys=True) + '\n').encode()
    (root / 'run').mkdir(parents=True, exist_ok=True)
    contract = {'runtime_sha256': physics, 'implementation_sha256': physics,
                'learned_policy_available': policy is not None, 'scene_sha256': 'a' * 64, 'task_sha256': 'b' * 64}
    files = {'lab-implementation/' + name: pin for name, pin in physics.items()}
    config = {'control_profile': 'shared_exercise', **{key: contract[key] for key in
              ('scene_sha256', 'task_sha256', 'learned_policy_available')}}
    if policy is not None:
        raw = encode({'runtime_sha256': policy})
        (root / 'run/policy.json').write_bytes(raw)
        files['policy.json'] = sha256(raw).hexdigest()
        contract.update(policy_runtime_sha256=policy, policy_sha256=files['policy.json'])
        config['policy_sha256'] = files['policy.json']
    raw = encode(contract)
    (root / 'run/exercise-lab.json').write_bytes(raw)
    files['exercise-lab.json'] = sha256(raw).hexdigest()
    raw = encode({'status': 'ready', 'control_profile': 'shared_exercise', 'files': files})
    (root / 'run/manifest.json').write_bytes(raw)
    config['manifest_sha256'] = sha256(raw).hexdigest()
    (root / 'lab-config.json').write_bytes(encode(config))


def test_policy_only_runtime_difference_selects_matching_release(tmp_path, monkeypatch):
    exporter = runtime._exporter(Path(__file__).parents[1])
    physics_raw = (Path(runtime.__file__).parent / 'human.py').read_bytes()
    physics = {'human.py': sha256(physics_raw).hexdigest()}
    policy = {**physics, 'shared_exercise_learning.py': sha256(b'old policy runtime').hexdigest()}
    root = tmp_path / 'lab'
    sealed_selection(root, physics, policy)
    saved = {}
    for index, commit in enumerate(runtime.TRUSTED_RELEASES):
        package = tmp_path / commit / 'src/weft'
        package.mkdir(parents=True)
        (package / 'human.py').write_bytes(physics_raw)
        (package / 'shared_exercise_learning.py').write_bytes(
            b'old policy runtime' if index == len(runtime.TRUSTED_RELEASES) - 1 else b'wrong policy runtime')
        saved[commit] = package.parent.parent
    monkeypatch.setattr(runtime, '_exporter', lambda project: exporter)
    calls = []
    monkeypatch.setattr(runtime, 'prepare_runtime', lambda project, trusted_commit: calls.append(trusted_commit) or saved[trusted_commit])
    assert runtime.compatible_runtime(root, tmp_path) == saved[runtime.TRUSTED_RELEASES[-1]]
    assert calls == list(runtime.TRUSTED_RELEASES)


@pytest.mark.parametrize('change', ['unsealed_policy', 'manifest', 'overlap', 'path', 'unknown'])
def test_invalid_or_unknown_sealed_runtime_cannot_select_code(tmp_path, monkeypatch, change):
    physics = {'human.py': 'a' * 64}
    policy = {**physics, 'shared_exercise_learning.py': 'b' * 64}
    if change == 'overlap': policy['human.py'] = 'c' * 64
    if change == 'path': policy['../escape.py'] = 'b' * 64
    root = tmp_path / 'lab'
    sealed_selection(root, physics, policy)
    if change == 'unsealed_policy': (root / 'run/policy.json').write_text('{}')
    if change == 'manifest': (root / 'run/manifest.json').write_text('{}')
    calls = []
    def export(project, trusted_commit):
        calls.append(trusted_commit)
        package = tmp_path / trusted_commit / 'src/weft'
        package.mkdir(parents=True)
        (package / 'human.py').write_bytes(b'unknown runtime')
        return package.parent.parent
    monkeypatch.setattr(runtime, 'prepare_runtime', export)
    with pytest.raises(ValueError):
        runtime.compatible_runtime(root, Path(__file__).parents[1])
    assert bool(calls) is (change == 'unknown')


def test_matching_current_runtime_cannot_hide_invalid_evidence(tmp_path, monkeypatch):
    raw = (Path(runtime.__file__).parent / 'human.py').read_bytes()
    sealed_selection(tmp_path / 'lab', {'human.py': sha256(raw).hexdigest()}, None)
    monkeypatch.setattr(runtime, 'prepare_runtime', lambda *args, **kw: pytest.fail('No archived export needed'))
    with pytest.raises(ValueError, match='already match current'):
        runtime.compatible_runtime(tmp_path / 'lab', Path(__file__).parents[1])


def test_second_fixed_release_exports_and_reports_selected_commit(tmp_path, monkeypatch):
    files = {'src/weft/__init__.py': b'', 'src/weft/lab.py': LAB,
             'src/weft/shared_exercise_lab.py': b'', 'pyproject.toml': b'', 'requirements-lab.lock': b''}
    fake_exporter(monkeypatch, files)
    selected = runtime.TRUSTED_RELEASES[0]
    saved = runtime.prepare_runtime(tmp_path, trusted_commit=selected)
    root = tmp_path / 'lab'
    root.mkdir()
    lab = runtime.ArchivedLab(root, saved)
    try:
        assert lab.summary()['execution_runtime']['git_commit'] == selected
    finally:
        lab.close()
    with pytest.raises(ValueError, match='fixed trusted'):
        runtime.prepare_runtime(tmp_path, trusted_commit='a' * 40)
