"""Isolated child for pinned, trusted fitting code; not an OS security sandbox.

Launch with Python -I -B. Guards bound this fixed code path; the parent owns hard
termination, missing receipts, source custody and final report validation.
"""

import builtins
from hashlib import sha256
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import socket
import stat
import subprocess
import sys
import time


PROFILE = 'whole-body-reference-candidate-worker-v1'
INPUT_FILES = {'inputs/human.json', 'inputs/evidence-derived.json'}
RUNTIME_MODULES = {'__init__', 'body_frame', 'body_refinement', 'constrained_qp', 'errors', 'human', 'io',
                   'mobile_human', 'root_motion', 'spatial_pose', 'whole_body_reference_refinement', 'whole_body_reference_worker'}
RUNTIME_FILES = {f'runtime/weft/{name}.py' for name in RUNTIME_MODULES}
SURFACE_MODULES = {'source_surface_fit_inputs', 'source_surface_envelopes', 'source_surface_interval_construction',
                   'source_surface_anchors', 'camera_pose', 'camera_ground', 'world_alignment', 'world_evidence', 'observations'}
CAMERA_MODULES = {'camera_body_fit', 'camera_shared_contract', 'camera_body', 'camera_reference',
                  'multiview_root', 'multiview_body', 'multiview_pose', 'multiview_segments'}
SURFACE_ROLES = {'camera': 'inputs/surface-camera.json', 'surface_intervals': 'inputs/surface-intervals.json',
                 'annotations': 'inputs/surface-annotations.json', 'body_binding': 'inputs/surface-body-binding.json'}
SETTINGS = {'finalization_seconds': 8, 'maximum_geometry_calls': 120000, 'maximum_sweeps': 2}
NATIVE_LIMITS = {'model_compilations': 2, 'data_allocations': 6, 'forward_calls': 120010}
DENIAL_TRACE_LIMITS = {'maximum_events': 8, 'maximum_frames': 6, 'maximum_symbol_chars': 64, 'maximum_bytes': 16384}
INPUT_LIMIT, SOURCE_LIMIT, OUTPUT_LIMIT, PLAN_LIMIT, RECEIPT_LIMIT = 8 << 20, 2 << 20, 16 << 20, 128 << 10, 64 << 10
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_MEDIA = {'cv2', 'PIL', 'imageio', 'imageio_ffmpeg', 'av', 'ffmpeg', 'moviepy', 'pygame', 'matplotlib'}
_INFERENCE = {'mediapipe', 'torch', 'torchvision', 'tensorflow', 'transformers', 'diffusers', 'ultralytics', 'sam2'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique)
        json.dumps(value, allow_nan=False)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError('Expected bounded finite strict JSON') from error
    _require(type(value) is dict, 'Expected a JSON object')
    return value


def _encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _canonical(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _parts(name):
    _require(isinstance(name, str) and 0 < len(name) <= 256 and not name.startswith('/')
             and '\\' not in name and ':' not in name and all(ord(c) >= 32 for c in name), 'Unsafe worker member path')
    parts = name.split('/')
    _require(len(parts) <= 4 and all(p not in ('', '.', '..') for p in parts), 'Unsafe worker member path')
    return parts


def _parent(root_fd, name):
    parts = _parts(name)
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        return fd, parts[-1]
    except BaseException:
        os.close(fd)
        raise


def _read(root_fd, name, bound, expected=None):
    parent, leaf = _parent(root_fd, name)
    try:
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= bound, 'Worker input is not a bounded regular file')
        raw = stream.read(bound + 1)
        after = os.fstat(stream.fileno())
        _require(len(raw) <= bound and (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                 (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'Worker input changed while reading')
    if expected is not None:
        _require(sha256(raw).hexdigest() == expected, 'Worker file differs from its exact pin: ' + name)
    return raw


def _write(root_fd, name, raw, bound):
    _require(type(raw) is bytes and 0 < len(raw) <= bound, 'Worker output exceeds its byte bound')
    _require(name in {'candidate-human.json', 'angle-proposal.json', 'report.json', 'worker.json'}, 'Unexpected worker output')
    fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root_fd)
    with os.fdopen(fd, 'w+b') as stream:
        stream.write(raw); stream.flush(); stream.seek(0)
        _require(sha256(stream.read(bound + 1)).digest() == sha256(raw).digest(), 'Worker output readback differs')


def _plan(raw):
    plan = _json(raw)
    _require(type(plan.get('schema_version')) is int and plan['schema_version'] == 1
             and plan.get('profile') == 'whole-body-reference-candidate-plan-v1' and plan.get('status') == 'frozen', 'Frozen worker plan required')
    _require(plan.get('proposal_objective') in {'image_v1', 'floor_first_v1', 'surface_interval_v1', 'camera_surface_interval_v1'}, 'Unsupported proposal objective')
    camera_mode = plan['proposal_objective'] == 'camera_surface_interval_v1'
    surface_mode = plan['proposal_objective'] in {'surface_interval_v1', 'camera_surface_interval_v1'}
    input_files, runtime_files = INPUT_FILES, RUNTIME_FILES
    if surface_mode:
        surface = plan.get('surface_constraints')
        _require(type(surface) is dict and set(surface) == {'profile', 'config_sha256', 'source_sha256',
                 'weight', 'observation_count', 'optional_inputs'} and surface['profile'] == 'source-surface-fit-config-v1',
                 'Frozen surface input descriptor required')
        count, roles, weight = surface['observation_count'], surface['optional_inputs'], surface['weight']
        _require(type(count) is int and 1 <= count <= 8 and type(roles) is list
                 and all(type(role) is str and role in SURFACE_ROLES for role in roles)
                 and roles == [role for role in SURFACE_ROLES if role in roles], 'Invalid frozen surface input inventory')
        _require(type(weight) in (int, float) and math.isfinite(weight) and 0 < weight <= 1000,
                 'Explicit bounded surface weight required')
        _require(all(type(surface[key]) is str and _SHA.fullmatch(surface[key])
                     for key in ('config_sha256', 'source_sha256')), 'Exact surface config/source identities required')
        input_files = INPUT_FILES | {'inputs/surface-config.json',
            *(f'inputs/surface-observation-{i:02d}.json' for i in range(count)), *(SURFACE_ROLES[role] for role in roles)}
        runtime_files = RUNTIME_FILES | {f'runtime/weft/{name}.py' for name in SURFACE_MODULES}
    else:
        _require('surface_constraints' not in plan, 'Default fitting cannot silently consume surface constraints')
    if camera_mode:
        scale = plan.get('camera_image_scale')
        _require(type(scale) in (int, float) and math.isfinite(scale) and 1e-6 <= scale <= 1,
                 'Explicit camera image scale must be within [1e-6,1]')
        _require('camera' in plan['surface_constraints']['optional_inputs'], 'Camera fitting requires its exact supplied camera')
        runtime_files = runtime_files | {f'runtime/weft/{name}.py' for name in CAMERA_MODULES}
    else:
        _require('camera_image_scale' not in plan, 'Other objectives cannot silently consume a camera image scale')
    for key, names in (('input_files', input_files), ('runtime_files', runtime_files)):
        values = plan.get(key)
        _require(type(values) is dict and set(values) == names, 'Worker requires the exact bounded ' + key)
        for name, value in values.items():
            _parts(name)
            _require(isinstance(value, str) and _SHA.fullmatch(value), 'Worker file identity must be an exact SHA256')
    if surface_mode:
        _require(plan['surface_constraints']['config_sha256'] == plan['input_files']['inputs/surface-config.json'],
                 'Frozen surface config identity differs')
    settings = plan.get('settings')
    _require(type(settings) is dict and set(settings) == set(SETTINGS) | {'maximum_seconds'}, 'Unexpected fit settings')
    seconds = settings['maximum_seconds']
    _require(type(seconds) in (int, float) and math.isfinite(seconds) and .001 <= seconds <= 30, 'Fit seconds must be .001–30')
    for name, expected in SETTINGS.items():
        _require(type(settings[name]) is int and settings[name] == expected, 'Fixed finalization/call/sweep settings differ')
    versions = plan.get('runtime_versions')
    _require(type(versions) is dict and set(versions) == {'python', 'numpy', 'mujoco', 'gymnasium'}
             and all(isinstance(v, str) and 0 < len(v) <= 100 for v in versions.values()), 'Exact runtime versions required')
    return plan


def _install_limits():
    for kind, wanted in ((resource.RLIMIT_CPU, (40, 41)), (resource.RLIMIT_FSIZE, (OUTPUT_LIMIT, OUTPUT_LIMIT)),
                         (resource.RLIMIT_CORE, (0, 0))):
        old_soft, old_hard = resource.getrlimit(kind)
        hard = min(wanted[1], old_hard) if old_hard != resource.RLIM_INFINITY else wanted[1]
        soft = min(wanted[0], hard, old_soft) if old_soft != resource.RLIM_INFINITY else min(wanted[0], hard)
        resource.setrlimit(kind, (soft, hard))


class _Guards:
    def __init__(self, started):
        self.started, self.originals = started, []
        self.counts = {name: 0 for name in (*NATIVE_LIMITS, 'physics_steps', 'reset_calls', 'media_calls', 'inference_calls',
                                          'fitter_calls', 'forbidden_calls')}
        self.denied = {}
        self.denied_callers, self.denied_caller_bytes, self.denied_callers_omitted = [], 0, 0
        self.runtime_modules = RUNTIME_MODULES

    def check(self):
        if time.monotonic() - self.started >= 40:
            raise TimeoutError('Trusted fitting worker exceeded its 40-second guard')

    def patch(self, owner, name, value):
        self.originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def _denied_caller(self, category):
        if len(self.denied_callers) >= DENIAL_TRACE_LIMITS['maximum_events']:
            self.denied_callers_omitted += 1
            return
        def symbol(value):
            value = value if type(value) is str else type(value).__name__
            return ''.join(c if 32 <= ord(c) <= 126 else '?' for c in value[:DENIAL_TRACE_LIMITS['maximum_symbol_chars']])
        event, frame = {'category': symbol(category), 'frames': []}, None
        try:
            # Capture code identities only; never locals, arguments, source text or full paths.
            frame = sys._getframe(2)
            for _ in range(DENIAL_TRACE_LIMITS['maximum_frames']):
                if frame is None:
                    break
                code = frame.f_code
                event['frames'].append({'module': symbol(frame.f_globals.get('__name__', 'unknown')),
                    'file': symbol(code.co_filename.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]),
                    'function': symbol(code.co_name), 'line': frame.f_lineno})
                frame = frame.f_back
        except Exception:
            event['unavailable'] = 'caller_stack_unavailable'
        finally:
            del frame
        size = len(json.dumps(event, separators=(',', ':'), allow_nan=False).encode())
        if self.denied_caller_bytes + size > DENIAL_TRACE_LIMITS['maximum_bytes']:
            self.denied_callers_omitted += 1
            return
        self.denied_callers.append(event)
        self.denied_caller_bytes += size

    def denial_diagnostics(self):
        return {'schema_version': 1, 'profile': 'bounded-denied-callers-v1', 'events': self.denied_callers,
                'event_bytes': self.denied_caller_bytes, 'omitted_events': self.denied_callers_omitted,
                'limits': DENIAL_TRACE_LIMITS,
                'scope': 'First bounded blocked-call stacks; code names only, no arguments or locals. All attempt counts remain in denied_calls.'}

    def forbid(self, category):
        def denied(*args, **kwargs):
            self.counts['forbidden_calls'] += 1
            self.denied[category] = self.denied.get(category, 0) + 1
            try:
                self._denied_caller(category)
            except Exception:
                # Diagnostics cannot change whether a forbidden operation is blocked.
                self.denied_callers_omitted += 1
            raise PermissionError('Trusted fitting worker forbids ' + category)
        return denied

    def counted(self, name, function):
        def call(*args, **kwargs):
            self.check()
            if self.counts[name] >= NATIVE_LIMITS[name]:
                return self.forbid('native call cap: ' + name)()
            self.counts[name] += 1
            result = function(*args, **kwargs)
            self.check()
            return result
        return call

    def _import_check(self, name):
        root = name.split('.')[0]
        if root in _MEDIA or root in _INFERENCE:
            self.forbid('media import' if root in _MEDIA else 'inference import')()
        if name.startswith('weft.') and name.split('.')[1] not in self.runtime_modules:
            self.forbid('unlisted runtime import')()

    def find_spec(self, fullname, path=None, target=None):
        self._import_check(fullname)
        return None

    def install(self):
        self.check()
        for name in ('connect', 'connect_ex', 'bind', 'listen', 'sendto', 'sendmsg'):
            if hasattr(socket.socket, name): self.patch(socket.socket, name, self.forbid('network'))
        for name in ('create_connection', 'create_server', 'getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr'):
            if hasattr(socket, name): self.patch(socket, name, self.forbid('network'))
        self.patch(subprocess, 'Popen', self.forbid('nested process'))
        for name in dir(os):
            if name in {'system', 'fork', 'forkpty', 'posix_spawn', 'posix_spawnp'} or name.startswith(('spawn', 'exec')):
                if callable(getattr(os, name)): self.patch(os, name, self.forbid('nested process'))
        original_import = builtins.__import__
        def importing(name, *args, **kwargs):
            self._import_check(name)
            return original_import(name, *args, **kwargs)
        self.patch(builtins, '__import__', importing)
        sys.meta_path.insert(0, self)
        def expired(*args):
            raise TimeoutError('Trusted fitting worker exceeded its wall/CPU guard')
        self.signal_handlers = {kind: signal.getsignal(kind) for kind in (signal.SIGALRM, signal.SIGXCPU)}
        for kind in self.signal_handlers: signal.signal(kind, expired)
        signal.setitimer(signal.ITIMER_REAL, max(.001, 40 - (time.monotonic() - self.started)))

    def native(self, mujoco, human):
        original_model = mujoco.MjModel
        class ModelProxy:
            from_xml_string = staticmethod(self.counted('model_compilations', original_model.from_xml_string))
            from_xml_path = staticmethod(self.forbid('path model'))
            from_binary_path = staticmethod(self.forbid('path model'))
        self.patch(mujoco, 'MjModel', ModelProxy)
        self.patch(mujoco, 'MjData', self.counted('data_allocations', mujoco.MjData))
        self.patch(mujoco, 'mj_forward', self.counted('forward_calls', mujoco.mj_forward))
        for name in dir(mujoco):
            if name.startswith(('mj_step', 'mj_resetData')):
                self.patch(mujoco, name, self.forbid('physics step' if name.startswith('mj_step') else 'reset'))
            elif name in {'Renderer', 'GLContext', 'MjrContext'} or name.startswith('mjr_'):
                if callable(getattr(mujoco, name)): self.patch(mujoco, name, self.forbid('render'))
        for name, category in (('reset', 'reset'), ('step', 'physics step'), ('render', 'render')):
            self.patch(human, name, self.forbid(category))

    def restore(self):
        if hasattr(self, 'signal_handlers'):
            signal.setitimer(signal.ITIMER_REAL, 0)
            for kind, handler in self.signal_handlers.items(): signal.signal(kind, handler)
        if self in sys.meta_path: sys.meta_path.remove(self)
        for owner, name, value in reversed(self.originals): setattr(owner, name, value)


def _versions():
    return {'python': sys.version.split()[0], **{name: importlib.metadata.version(name) for name in ('numpy', 'mujoco', 'gymnasium')}}


def _load_runtime(runtime, guards, versions):
    _require(not any(name == 'weft' or name.startswith('weft.') for name in sys.modules), 'Frozen worker must start without an imported Weft package')
    sys.path.insert(0, str(runtime))
    import numpy
    import mujoco
    import gymnasium
    from weft.human import HumanEnv
    _require(all(module.__version__ == versions[name] for name, module in
                 (('numpy', numpy), ('mujoco', mujoco), ('gymnasium', gymnasium))), 'Imported native dependency version differs')
    guards.native(mujoco, HumanEnv)
    from weft.whole_body_reference_refinement import propose_whole_body_reference_refinement
    return propose_whole_body_reference_refinement


def _verify_files(root_fd, plan, check):
    for key, bound in (('input_files', INPUT_LIMIT), ('runtime_files', SOURCE_LIMIT)):
        for name, expected in plan[key].items():
            check(); _read(root_fd, name, bound, expected)
    directory = os.open('runtime/weft', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
    try:
        expected_names = {name.rsplit('/', 1)[-1] for name in plan['runtime_files']}
        names = set()
        with os.scandir(directory) as entries:
            for entry in entries:
                check()
                _require(entry.name in expected_names and entry.is_file(follow_symlinks=False),
                         'Frozen runtime contains unpinned files or bytecode')
                names.add(entry.name)
        _require(names == expected_names, 'Frozen runtime inventory differs')
    finally:
        os.close(directory)


def _result(result, plan):
    camera_mode = plan['proposal_objective'] == 'camera_surface_interval_v1'
    keys = {'status', 'human_candidate', 'report'} | ({'angle_proposal'} if camera_mode else set())
    _require(type(result) is dict and set(result) == keys, 'Unexpected fitter result shape')
    if camera_mode:
        _require(result['human_candidate'] is None, 'Camera angle proposals cannot replace the original human body')
    candidate, report = result['angle_proposal'] if camera_mode else result['human_candidate'], result['report']
    method = {'image_v1': 'blocked-whole-body-floor-reference-v1',
              'floor_first_v1': 'blocked-whole-body-floor-first-reference-v1',
              'surface_interval_v1': 'blocked-whole-body-surface-interval-reference-v1',
              'camera_surface_interval_v1': 'blocked-camera-body-surface-interval-reference-v1'}[plan['proposal_objective']]
    _require(type(candidate) is dict and type(report) is dict and result['status'] in
             {'partial_budget_exhausted', 'rejected_construction_proposal', 'unadmitted_construction_proposal'}, 'Unsupported proposal result')
    _require(report.get('status') == result['status'] and report.get('profile') == method and report.get('promoted') is False
             and report.get('physical_validation') == 'unverified' and type(report.get('physics_steps')) is int
             and report['physics_steps'] == 0 and report.get('contract', {}).get('floor_penetration_limit_model_units') == .025,
             'Proposal cannot change admission, physics or floor limit')
    marker = candidate.get('whole_body_reference_refinement')
    _require(type(marker) is dict and marker.get('method') == method and marker.get('status') == result['status']
             and marker.get('promoted') is False, 'Candidate must remain explicitly unpromoted')
    for key, name in (('input_human_sha256', 'inputs/human.json'), ('input_evidence_sha256', 'inputs/evidence-derived.json')):
        _require(report.get(key) == marker.get(key) == plan['input_files'][name], 'Returned input identity differs')
    if camera_mode:
        scope = 'unsupported_angle_overlay_not_a_human_spec'
        _require(type(candidate.get('schema_version')) is int and candidate['schema_version'] == 1
                 and candidate.get('profile') == report.get('candidate_kind') == 'camera-body-angle-proposal-v1'
                 and candidate.get('installation') == report.get('geometry_admission') == scope
                 and all(candidate.get(key) is False for key in ('body_installed', 'raw_camera_changed', 'root_or_floor_changed', 'promoted'))
                 and candidate.get('contact') == 'unknown' and candidate.get('physical_validation') == 'unverified',
                 'Camera angle overlay must remain uninstalled with original camera/root/floor unchanged')
        bindings, camera = candidate.get('input_bindings'), report.get('camera_fit')
        _require(type(bindings) is dict and type(camera) is dict and camera.get('input_bindings') == bindings
                 and type(camera.get('image_loss_scale_per_pixel')) in (int, float)
                 and camera.get('image_loss_scale_per_pixel') == plan['camera_image_scale']
                 and camera.get('image_loss_scale_status') == 'authored', 'Camera fitting identity or authored scale differs')
        for key, name in (('body_raw_sha256', 'inputs/human.json'), ('evidence_raw_sha256', 'inputs/evidence-derived.json'),
                          ('camera_raw_sha256', 'inputs/surface-camera.json')):
            _require(bindings.get(key) == plan['input_files'][name], 'Returned raw camera/body/evidence identity differs')
    _require(report.get('candidate_sha256') == _canonical(candidate), 'Returned canonical candidate identity differs')
    output = {'angle-proposal.json' if camera_mode else 'candidate-human.json': _encode(candidate), 'report.json': _encode(report)}
    _require(sum(len(raw) for raw in output.values()) <= OUTPUT_LIMIT, 'Combined proposal output exceeds 16 MiB')
    return output


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value) if sys.platform == 'darwin' else int(value * 1024) if sys.platform.startswith('linux') else None


def run_worker(study, expected_plan_sha256):
    """Child-only entry point: installs irreversible resource limits."""
    started = time.monotonic()
    guards = _Guards(started)
    versions, loaded, result, plan_sha, root_fd, error = {}, {}, None, None, None, None
    old_path = sys.path[:]
    try:
        path = Path(study).absolute()
        _require('..' not in path.parts and not path.is_symlink(), 'Unsafe study directory')
        path = path.resolve(strict=True)
        _require(len(path.parts) <= 64, 'Study directory exceeds path depth bound')
        root_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for name in ('worker.json', 'candidate-human.json', 'angle-proposal.json', 'report.json'):
            try: os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError: continue
            raise FileExistsError('Worker output already exists: ' + name)
        _install_limits()
        guards.install()
        _require(isinstance(expected_plan_sha256, str) and _SHA.fullmatch(expected_plan_sha256), 'Expected plan must be an exact SHA256')
        plan_raw = _read(root_fd, 'plan.json', PLAN_LIMIT, expected_plan_sha256)
        plan_sha = sha256(plan_raw).hexdigest()
        plan = _plan(plan_raw)
        if plan['proposal_objective'] in {'surface_interval_v1', 'camera_surface_interval_v1'}:
            guards.runtime_modules = RUNTIME_MODULES | SURFACE_MODULES
        if plan['proposal_objective'] == 'camera_surface_interval_v1':
            guards.runtime_modules = guards.runtime_modules | CAMERA_MODULES
        worker = Path(__file__).resolve(strict=True)
        _require(worker == path / 'runtime/weft/whole_body_reference_worker.py', 'Worker must execute from this frozen runtime')
        _verify_files(root_fd, plan, guards.check)
        inputs = {name: _read(root_fd, name, INPUT_LIMIT, expected) for name, expected in plan['input_files'].items()}
        for raw in inputs.values(): _json(raw)
        versions = _versions()
        _require(versions == plan['runtime_versions'], 'Installed runtime versions differ')
        guards.check()
        fit = _load_runtime(worker.parent.parent, guards, versions)
        surface_options = {}
        if plan['proposal_objective'] in {'surface_interval_v1', 'camera_surface_interval_v1'}:
            from weft.source_surface_fit_inputs import prepare_surface_fit_objective
            surface_options = prepare_surface_fit_objective(inputs, plan['surface_constraints'],
                **({'camera_image_scale': plan['camera_image_scale']} if plan['proposal_objective'] == 'camera_surface_interval_v1' else {}))
        guards.check(); guards.counts['fitter_calls'] += 1
        result = fit(inputs['inputs/human.json'], inputs['inputs/evidence-derived.json'],
                     proposal_objective=plan['proposal_objective'], settings=plan['settings'], **surface_options)
        guards.check()
        outputs = _result(result, plan)
        for name, raw in outputs.items():
            guards.check(); _write(root_fd, name, raw, OUTPUT_LIMIT)
        _verify_files(root_fd, plan, guards.check)
        guards.check(); _read(root_fd, 'plan.json', PLAN_LIMIT, plan_sha)
        for name, module in tuple(sys.modules.items()):
            if name == 'weft' or name.startswith('weft.'):
                member = Path(module.__file__).resolve(strict=True).relative_to(path).as_posix()
                _require(member in plan['runtime_files'], 'Imported Weft module is outside the frozen closure')
                loaded[name] = member
        guards.check()
    except (Exception, KeyboardInterrupt) as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
    finally:
        guards.restore()
        sys.path[:] = old_path
    proposal_status = result.get('status') if isinstance(result, dict) else None
    if not isinstance(proposal_status, str) or proposal_status not in {
            'partial_budget_exhausted', 'rejected_construction_proposal', 'unadmitted_construction_proposal'}:
        proposal_status = None
    receipt = {'schema_version': 1, 'profile': PROFILE, 'status': 'failed' if error else 'completed', 'error': error,
        'proposal_status': proposal_status,
        'plan_sha256': plan_sha, 'counts': guards.counts, 'denied_calls': guards.denied,
        'denied_callers': guards.denial_diagnostics(), 'native_call_limits': NATIVE_LIMITS,
        'counts_scope': 'Native counters count started allowed calls, not completed operations; denied attempts are separate.',
        'runtime_versions': versions, 'loaded_weft': loaded, 'seconds': time.monotonic() - started,
        'peak_rss_bytes': _rss_bytes(), 'memory_scope': 'Observed peak RSS in bytes, not a hard RSS cap.',
        'isolated_mode': bool(sys.flags.isolated), 'bytecode_disabled': sys.dont_write_bytecode,
        'video_decodes': 0, 'inference_calls': 0, 'training_episodes': 0, 'paid_calls': 0, 'external_api_usd': 0,
        'promoted': False, 'scope': 'Pinned trusted fitter only; guards are not an OS security sandbox. No task/policy or physical validation.'}
    try:
        _require(root_fd is not None, 'No safe study directory for a worker receipt')
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
        print('Usage: whole_body_reference_worker.py STUDY EXPECTED_PLAN_SHA256', file=sys.stderr)
        return 1
    return run_worker(argv[0], argv[1])


if __name__ == '__main__':
    raise SystemExit(main())
