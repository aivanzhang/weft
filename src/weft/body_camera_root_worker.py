"""Isolated original-body extraction and pure proposal; no body admission or dynamics."""

import importlib.metadata
import importlib.util
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import ssl
import stat
import subprocess
import sys
import time
import urllib.request


PROFILE = 'body-camera-root-worker-v1'
PLAN_PROFILE = 'body-camera-root-plan-v1'
RUNTIME_MODULES = frozenset({'__init__', 'body_frame', 'body_refinement', 'constrained_qp', 'errors', 'human', 'io',
    'mobile_human', 'root_motion', 'spatial_pose', 'whole_body_reference_inputs', 'whole_body_reference_refinement', 'whole_body_reference_worker',
    'camera_reference', 'body_camera_root_refinement', 'body_camera_root_inputs', 'body_camera_root_geometry',
    'body_camera_root_worker'})
RUNTIME_FILES = frozenset(f'runtime/weft/{name}.py' for name in RUNTIME_MODULES)
REQUIRED_INPUTS = frozenset({'inputs/human.json', 'inputs/evidence-derived.json',
                           'inputs/input-receipt.json', 'inputs/camera-config.json'})
OUTPUT_LIMITS = {'geometry.json': 8 << 20, 'camera-root-input.json': 8 << 20,
    'target-provenance.json': 8 << 20, 'source-slots.json': 8 << 20,
    'proposal.json': 16 << 20, 'geometry-check.json': 8 << 20}
NATIVE_LIMITS = {'model_compilations': 2, 'data_allocations': 4, 'forward_calls': 5402}
MAX_SECONDS, MAX_INPUT_BYTES, MAX_OUTPUT_BYTES = 30., 32 << 20, 16 << 20
SOURCE_LIMIT, PLAN_LIMIT, RECEIPT_LIMIT = 2 << 20, 128 << 10, 64 << 10
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _require(value, message):
    if not value:
        raise ValueError('Camera/root worker: ' + message)


def _json(raw):
    def unique(rows):
        result = {}
        for key, value in rows:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    try:
        result = json.loads(raw, object_pairs_hook=unique)
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, RecursionError, OverflowError) as error:
        raise ValueError('Camera/root worker requires finite strict JSON') from error
    _require(type(result) is dict, 'JSON object required')
    return result


def _encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def _parts(name):
    _require(type(name) is str and 0 < len(name) <= 256 and not name.startswith('/') and '\\' not in name
             and ':' not in name and all(ord(c) >= 32 for c in name), 'unsafe member path')
    parts = name.split('/')
    _require(len(parts) <= 4 and all(part not in ('', '.', '..') for part in parts), 'unsafe member path')
    return parts


def _read(root_fd, name, bound, expected=None):
    parts = _parts(name); parent = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent); parent = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= bound, 'nonregular or oversized input')
        raw = stream.read(bound + 1); after = os.fstat(stream.fileno())
    _require(len(raw) == before.st_size and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
             == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'input changed during read')
    if expected is not None:
        _require(sha256(raw).hexdigest() == expected, 'file differs from frozen pin: ' + name)
    return raw


def _write(root_fd, name, raw, bound):
    _require(name in OUTPUT_LIMITS or name == 'worker.json', 'unexpected output')
    _require(type(raw) is bytes and 0 < len(raw) <= bound, 'output byte bound')
    fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root_fd)
    with os.fdopen(fd, 'w+b') as stream:
        stream.write(raw); stream.flush(); stream.seek(0)
        _require(stream.read(bound + 1) == raw, 'output readback differs')


def _plan(raw):
    plan = _json(raw)
    _require(type(plan.get('schema_version')) is int and plan['schema_version'] == 1
             and plan.get('profile') == PLAN_PROFILE and plan.get('status') == 'frozen', 'frozen plan required')
    inputs, runtime = plan.get('input_files'), plan.get('runtime_files')
    _require(type(inputs) is dict and REQUIRED_INPUTS <= inputs.keys() and len(inputs) <= 64, 'bounded selected metadata required')
    _require(type(runtime) is dict and set(runtime) == RUNTIME_FILES, 'exact runtime closure required')
    for name, digest in {**inputs, **runtime}.items():
        _parts(name)
        _require(type(digest) is str and _SHA.fullmatch(digest), 'exact file hash required')
    _require(all(name.startswith('inputs/') and name.endswith(('.json', '.jsonl', '.py')) for name in inputs),
             'worker reads metadata only, never source media')
    versions = plan.get('runtime_versions')
    _require(type(versions) is dict and set(versions) == {'python', 'numpy', 'mujoco', 'gymnasium'}
             and all(type(value) is str and 0 < len(value) <= 100 for value in versions.values()), 'exact dependency versions required')
    settings = plan.get('settings')
    bounds = {'maximum_seconds': 10., 'maximum_iterations': 8, 'maximum_backtracks': 8,
              'maximum_camera_rotation_rad': .15, 'maximum_root_translation_model_units': .08,
              'maximum_root_correction_speed_model_units_s': .3}
    _require(type(settings) is dict and set(settings) <= bounds.keys() and 'maximum_seconds' in settings, 'tighter pure fit settings required')
    for name, value in settings.items():
        integer = name in ('maximum_iterations', 'maximum_backtracks')
        _require(type(value) in ((int,) if integer else (int, float)) and math.isfinite(value)
                 and (1 if integer else .001) <= value <= bounds[name], 'unsupported fit bound')
    return plan


def _install_limits():
    for kind, wanted in ((resource.RLIMIT_CPU, (30, 31)), (resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES)),
                         (resource.RLIMIT_CORE, (0, 0))):
        soft, hard = resource.getrlimit(kind)
        new_hard = wanted[1] if hard == resource.RLIM_INFINITY else min(hard, wanted[1])
        new_soft = min(wanted[0], new_hard) if soft == resource.RLIM_INFINITY else min(soft, wanted[0], new_hard)
        resource.setrlimit(kind, (new_soft, new_hard))


def _guards(base, started):
    class Guards(base._Guards):
        def __init__(self, started):
            super().__init__(started)
            self.native_limits = NATIVE_LIMITS.copy()
            self.startup_os_probes = []

        def check(self):
            if time.monotonic() - self.started >= MAX_SECONDS:
                raise TimeoutError('Camera/root worker exceeded its 30-second bound')

        def counted(self, name, function):
            def call(*args, **kwargs):
                self.check()
                if self.counts[name] >= self.native_limits[name]:
                    return self.forbid('native call cap: ' + name)()
                self.counts[name] += 1
                value = function(*args, **kwargs)
                self.check()
                return value
            return call

        def _import_check(self, name):
            root = name.split('.')[0]
            if root in base._MEDIA or root in base._INFERENCE:
                self.forbid('media or inference import')()
            if name.startswith('weft.') and name.split('.')[1] not in RUNTIME_MODULES:
                self.forbid('unlisted runtime import')()

        def install(self):
            super().install()
            signal.setitimer(signal.ITIMER_REAL, max(.001, MAX_SECONDS - (time.monotonic() - self.started)))

    return Guards(started)


def _input_native_limits(human_raw):
    references = _json(human_raw).get('reference')
    _require(type(references) is list and 3 <= len(references) <= 900, 'bounded original references required')
    times = []
    for row in references:
        stamp = row.get('time_s') if type(row) is dict else None
        _require(type(stamp) in (int, float) and math.isfinite(stamp) and 0 <= stamp <= 60
                 and (not times or times[-1] < stamp <= times[-1] + .25 + 1e-9), 'bounded original source clocks required')
        times.append(stamp)
    grid = math.floor((times[-1] - times[0]) * 30 + 1e-9) + 1
    _require(grid <= 1801, 'bounded original grid required')
    return {**NATIVE_LIMITS, 'forward_calls': min(NATIVE_LIMITS['forward_calls'], 2 * (len(references) + grid))}


def _runtime_files(root_fd, plan, check):
    for name, expected in plan['runtime_files'].items():
        check(); _read(root_fd, name, SOURCE_LIMIT, expected)
    fd = os.open('runtime/weft', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
    try:
        names = set()
        with os.scandir(fd) as entries:
            for entry in entries:
                check()
                _require(entry.name in {name + '.py' for name in RUNTIME_MODULES}
                         and entry.is_file(follow_symlinks=False), 'unlisted runtime file or bytecode')
                names.add(entry.name)
        _require(names == {name + '.py' for name in RUNTIME_MODULES}, 'runtime inventory differs')
    finally:
        os.close(fd)


def _inputs(root_fd, plan, check):
    values, total = {}, 0
    for name, expected in plan['input_files'].items():
        check()
        bound = 128 << 10 if name in {'inputs/input-receipt.json', 'inputs/camera-config.json'} else 8 << 20
        raw = _read(root_fd, name, min(bound, MAX_INPUT_BYTES - total), expected)
        total += len(raw); values[name] = raw
    return values


def _load_guard(path):
    spec = importlib.util.spec_from_file_location('_weft_camera_guard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _versions():
    return {'python': sys.version.split()[0], **{name: importlib.metadata.version(name) for name in ('numpy', 'mujoco', 'gymnasium')}}


def _architecture_probe(guard, argv, *args, **kwargs):
    allowed = ['/usr/sbin/sysctl', '-n', 'sysctl.proc_translated']
    if sys.platform != 'darwin' or argv != allowed or args or kwargs != {'capture_output': True} or guard.startup_os_probes:
        return guard.forbid('unexpected runtime startup process')()
    guard.check()
    popen = next(original for owner, name, original in guard.originals if owner is subprocess and name == 'Popen')
    started = time.monotonic()
    row = {'command': allowed, 'returncode': None, 'timeout_seconds': 2.}
    guard.startup_os_probes.append(row)
    process = popen(allowed, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    close_fds=True, env={'PATH': '/usr/bin:/bin'})
    try:
        stdout, stderr = process.communicate(timeout=2)
        _require(len(stdout) <= 4096 and len(stderr) <= 4096, 'architecture probe output exceeds bound')
        row.update(returncode=process.returncode, stdout_bytes=len(stdout), stderr_bytes=len(stderr))
        guard.check()
        return subprocess.CompletedProcess(allowed, process.returncode, stdout, stderr)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)
        row['seconds'] = time.monotonic() - started


def _import_mujoco(guard):
    # MuJoCo's exact Darwin architecture check is the sole read-only startup subprocess.
    original = subprocess.run
    subprocess.run = lambda argv, *args, **kwargs: _architecture_probe(guard, argv, *args, **kwargs)
    try:
        return importlib.import_module('mujoco')
    finally:
        subprocess.run = original


def _load_runtime(runtime, guard, versions):
    _require(not any(name == 'weft' or name.startswith('weft.') for name in sys.modules), 'fresh isolated Weft interpreter required')
    sys.path.insert(0, str(runtime))
    import numpy
    mujoco = _import_mujoco(guard)
    import gymnasium
    from weft.human import HumanEnv
    _require(all(module.__version__ == versions[name] for name, module in
                 (('numpy', numpy), ('mujoco', mujoco), ('gymnasium', gymnasium))), 'imported dependency version differs')
    guard.native(mujoco, HumanEnv)
    from weft.body_camera_root_geometry import extract_original_geometry, check_proposed_geometry
    from weft.body_camera_root_inputs import prepare_camera_root_document
    from weft.body_camera_root_refinement import propose_camera_root_refinement
    return extract_original_geometry, prepare_camera_root_document, propose_camera_root_refinement, check_proposed_geometry


def _proposal(value, input_raw):
    _require(type(value) is dict and value.get('status') in ('proposed', 'unchanged', 'unavailable', 'partial'), 'unsupported proposal result')
    report = value.get('report')
    _require(type(report) is dict and report.get('input_sha256') == sha256(input_raw).hexdigest()
             and report.get('admitted') is False and report.get('promoted') is False
             and report.get('physical_validation') == 'unverified' and report.get('counts', {}).get('physics_steps') == 0,
             'proposal cannot claim admission or dynamics')
    return _encode(value)


def run_worker(study, expected_plan_sha256):
    """Child-only entry point with irreversible resource limits and owned output files."""
    started, root_fd, guard = time.monotonic(), None, None
    error, plan_sha, versions, loaded, output_files = None, None, {}, {}, {}
    effective_limits, limits_from_input = NATIVE_LIMITS.copy(), False
    old_path, output_bytes, proposal_status = sys.path[:], 0, None

    def check():
        if time.monotonic() - started >= MAX_SECONDS:
            raise TimeoutError('Camera/root worker exceeded its 30-second bound')

    def save(name, raw):
        nonlocal output_bytes
        check(); _json(raw)
        _require(output_bytes + len(raw) <= MAX_OUTPUT_BYTES, 'combined outputs exceed 16 MiB')
        _write(root_fd, name, raw, OUTPUT_LIMITS[name])
        output_bytes += len(raw); output_files[name] = sha256(raw).hexdigest()

    try:
        path = Path(study).absolute()
        _require('..' not in path.parts and len(path.parts) <= 64
                 and not any(p.is_symlink() for p in (path, *path.parents)), 'safe study path required')
        root_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for name in (*OUTPUT_LIMITS, 'worker.json'):
            try: os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError: continue
            raise FileExistsError('Worker output already exists: ' + name)
        _install_limits()
        _require(type(expected_plan_sha256) is str and _SHA.fullmatch(expected_plan_sha256), 'exact plan hash required')
        plan_raw = _read(root_fd, 'plan.json', PLAN_LIMIT, expected_plan_sha256)
        plan_sha = sha256(plan_raw).hexdigest(); plan = _plan(plan_raw)
        worker = Path(__file__).resolve(strict=True)
        _require(worker == path / 'runtime/weft/body_camera_root_worker.py', 'execute the frozen worker only')
        _runtime_files(root_fd, plan, check)
        inputs = _inputs(root_fd, plan, check)
        effective_limits = _input_native_limits(inputs['inputs/human.json']); limits_from_input = True
        base = _load_guard(path / 'runtime/weft/whole_body_reference_worker.py')
        guard = _guards(base, started); guard.native_limits = effective_limits.copy(); guard.install()
        versions = _versions(); _require(versions == plan['runtime_versions'], 'installed dependency versions differ')
        extract, adapt, propose, diagnose = _load_runtime(worker.parent.parent, guard, versions)
        runtime = {'implementation_sha256': {Path(name).name: digest for name, digest in plan['runtime_files'].items()},
                   'versions': versions}
        human, evidence = inputs['inputs/human.json'], inputs['inputs/evidence-derived.json']
        identity = _json(inputs['inputs/input-receipt.json'])
        if 'source' in plan:
            _require(plan['source'] == identity, 'input receipt differs from the frozen source identity')
        geometry = extract(human, evidence, runtime=runtime, deadline=started + MAX_SECONDS)
        save('geometry.json', geometry)
        prepared = {'files': {name: raw for name, raw in inputs.items() if name not in
                    {'inputs/input-receipt.json', 'inputs/camera-config.json'}}, 'identity': identity}
        adapted = adapt(prepared, inputs['inputs/camera-config.json'], geometry, deadline=started + MAX_SECONDS)
        for key, name in (('input_document_bytes', 'camera-root-input.json'), ('provenance_bytes', 'target-provenance.json'),
                          ('source_slots_bytes', 'source-slots.json')):
            save(name, adapted[key])
        check()
        _require(MAX_SECONDS - (time.monotonic() - started) > plan['settings']['maximum_seconds'] + 3,
                 'insufficient reserved time for pure proposal and final diagnostic')
        guard.counts['fitter_calls'] += 1
        proposal = propose(adapted['input_document_bytes'], settings=plan['settings'])
        proposal_raw = _proposal(proposal, adapted['input_document_bytes'])
        proposal_status = proposal['status']; save('proposal.json', proposal_raw)
        diagnostic = diagnose(human, evidence, geometry, proposal_raw, deadline=started + MAX_SECONDS)
        save('geometry-check.json', diagnostic)
        _runtime_files(root_fd, plan, check)
        _inputs(root_fd, plan, check)
        _read(root_fd, 'plan.json', PLAN_LIMIT, plan_sha)
        for name, module in tuple(sys.modules.items()):
            if name == 'weft' or name.startswith('weft.'):
                member = Path(module.__file__).resolve(strict=True).relative_to(path).as_posix()
                _require(member in RUNTIME_FILES, 'imported Weft code is outside the frozen closure')
                loaded[name] = member
        check()
    except (Exception, KeyboardInterrupt) as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
    finally:
        if guard is not None: guard.restore()
        sys.path[:] = old_path
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = {'schema_version': 1, 'profile': PROFILE, 'status': 'failed' if error else 'completed',
        'error': error, 'plan_sha256': plan_sha, 'proposal_status': proposal_status,
        'output_files': output_files, 'output_bytes': output_bytes,
        'counts': guard.counts if guard else {**dict.fromkeys(NATIVE_LIMITS, 0), 'fitter_calls': 0,
            'physics_steps': 0, 'reset_calls': 0, 'media_calls': 0, 'inference_calls': 0, 'forbidden_calls': 0},
        'denied_calls': guard.denied if guard else {}, 'native_call_limits': effective_limits,
        'global_native_call_limits': NATIVE_LIMITS, 'native_limits_derived_from_input': limits_from_input,
        'startup_os_probes': guard.startup_os_probes if guard else [],
        'counts_scope': 'Started allowed calls; denied attempts separate. No dynamics, rendering or source-media reads.',
        'runtime_versions': versions, 'loaded_weft': loaded, 'seconds': time.monotonic() - started,
        'peak_rss_bytes': int(rss if sys.platform == 'darwin' else rss * 1024),
        'memory_scope': 'Observed peak RSS, not a hard memory cap',
        'isolated_mode': bool(sys.flags.isolated), 'bytecode_disabled': sys.dont_write_bytecode,
        'external_api_usd': 0, 'admitted': False, 'promoted': False, 'physical_validation': 'unverified',
        'scope': 'Pinned trusted geometry and pure fit; only one exact Darwin architecture subprocess during import. Not an OS sandbox or simulator admission'}
    try:
        _require(root_fd is not None, 'no safe receipt directory')
        _write(root_fd, 'worker.json', _encode(receipt), RECEIPT_LIMIT)
    except Exception as exc:
        error = error or {'type': type(exc).__name__, 'message': str(exc)[:2000]}
        print(json.dumps({'status': 'failed', 'receipt_written': False, 'error': error}), file=sys.stderr)
    finally:
        if root_fd is not None: os.close(root_fd)
    return 1 if error else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit('Expected frozen study directory and plan SHA256')
    return run_worker(argv[0], argv[1])


if __name__ == '__main__':
    raise SystemExit(main())
