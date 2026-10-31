import builtins
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from weft import whole_body_reference_worker as worker


def raw(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def digest(value):
    return sha256(value).hexdigest()


def study_fixture(tmp_path, monkeypatch, *, objective='image_v1', fit_seconds=30):
    for name in tuple(sys.modules):
        if name.startswith('weft.') and name != worker.__name__:
            monkeypatch.delitem(sys.modules, name)
    study = tmp_path / 'study'; study.mkdir()
    package = study / 'runtime/weft'; package.mkdir(parents=True)
    (study / 'inputs').mkdir()
    inputs = {'inputs/human.json': raw({'reference': []}), 'inputs/evidence-derived.json': raw({'frames': []})}
    for name, value in inputs.items(): (study / name).write_bytes(value)
    runtime = {}
    original_worker = Path(worker.__file__).read_bytes()
    for name in worker.RUNTIME_FILES:
        value = original_worker if name.endswith('whole_body_reference_worker.py') else b'"""Pure worker test fixture."""\n'
        (study / name).write_bytes(value); runtime[name] = digest(value)
    versions = {'python': sys.version.split()[0], 'numpy': 'fixture', 'mujoco': 'fixture', 'gymnasium': 'fixture'}
    plan = {'schema_version': 1, 'profile': 'whole-body-reference-candidate-plan-v1', 'status': 'frozen',
            'proposal_objective': objective, 'input_files': {name: digest(value) for name, value in inputs.items()},
            'runtime_files': runtime, 'runtime_versions': versions,
            'settings': {'maximum_seconds': fit_seconds, 'finalization_seconds': 8, 'maximum_geometry_calls': 120000, 'maximum_sweeps': 2}}
    (study / 'plan.json').write_bytes(raw(plan))
    monkeypatch.setattr(worker, '__file__', str(package / 'whole_body_reference_worker.py'))
    monkeypatch.setattr(sys.modules['weft'], '__file__', str(package / '__init__.py'))
    monkeypatch.setattr(worker, '_install_limits', lambda: None)
    monkeypatch.setattr(worker.signal, 'setitimer', lambda *args: None)
    monkeypatch.setattr(worker, '_versions', lambda: versions.copy())
    return study, plan, digest(raw(plan)), inputs


def fake_result(plan, *, status='rejected_construction_proposal'):
    method = ('blocked-whole-body-floor-first-reference-v1' if plan['proposal_objective'] == 'floor_first_v1'
              else 'blocked-whole-body-floor-reference-v1')
    identities = {'input_human_sha256': plan['input_files']['inputs/human.json'],
                  'input_evidence_sha256': plan['input_files']['inputs/evidence-derived.json']}
    candidate = {'reference': [], 'whole_body_reference_refinement': {
        'method': method, 'status': status, 'promoted': False, **identities}}
    report = {'schema_version': 1, 'profile': method, 'status': status, 'promoted': False, 'physical_validation': 'unverified',
              'physics_steps': 0, 'contract': {'floor_penetration_limit_model_units': .025},
              'candidate_sha256': digest(json.dumps(candidate, sort_keys=True, allow_nan=False).encode()), **identities}
    return {'status': status, 'human_candidate': candidate, 'report': report}


def fake_native():
    calls = []
    def call(name):
        def invoke(*args, **kwargs):
            calls.append(name)
            return {'fake': name}
        return invoke
    native = SimpleNamespace(MjModel=SimpleNamespace(from_xml_string=call('model')),
        MjData=call('data'), mj_forward=call('forward'), mj_step=call('step'), mj_step1=call('step1'),
        mj_step2=call('step2'), mj_resetData=call('reset'), mj_resetDataKeyframe=call('keyframe'),
        Renderer=call('renderer'), GLContext=call('gl'), MjrContext=call('render-context'), mjr_render=call('render'))
    class Human:
        reset = call('human-reset')
        step = call('human-step')
        render = call('human-render')
    return native, Human, calls


def fake_loader(monkeypatch, plan, callback=None):
    received = []
    def load(runtime, guards, versions):
        native, human, calls = fake_native()
        guards.native(native, human)
        def fit(*args, **kwargs):
            received.append((args, kwargs))
            model = native.MjModel.from_xml_string('<mujoco/>')
            data = native.MjData(model)
            native.mj_forward(model, data)
            result = fake_result(plan)
            if callback: callback(result)
            return result
        return fit
    monkeypatch.setattr(worker, '_load_runtime', load)
    return received


@pytest.mark.parametrize('objective, seconds', [('image_v1', .001), ('floor_first_v1', 30)])
def test_child_helper_calls_fitter_once_and_seals_only_unpromoted_outputs(tmp_path, monkeypatch, objective, seconds):
    study, plan, pin, inputs = study_fixture(tmp_path, monkeypatch, objective=objective, fit_seconds=seconds)
    received = fake_loader(monkeypatch, plan)
    assert worker.run_worker(study, pin) == 0
    receipt = json.loads((study / 'worker.json').read_bytes())
    assert receipt['status'] == 'completed' and receipt['error'] is None
    assert receipt['plan_sha256'] == pin and receipt['proposal_status'] == 'rejected_construction_proposal'
    assert receipt['runtime_versions'] == plan['runtime_versions'] and receipt['promoted'] is False
    assert receipt['counts'] == {'model_compilations': 1, 'data_allocations': 1, 'forward_calls': 1, 'fitter_calls': 1,
                                 'physics_steps': 0, 'reset_calls': 0, 'media_calls': 0, 'inference_calls': 0, 'forbidden_calls': 0}
    assert receipt['denied_calls'] == {} and receipt['paid_calls'] == receipt['external_api_usd'] == 0
    assert receipt['peak_rss_bytes'] > 0 and 'not a hard RSS cap' in receipt['memory_scope']
    assert received == [((inputs['inputs/human.json'], inputs['inputs/evidence-derived.json']),
                         {'proposal_objective': objective, 'settings': plan['settings']})]
    assert (study / 'candidate-human.json').stat().st_size + (study / 'report.json').stat().st_size <= worker.OUTPUT_LIMIT
    assert all((study / name).read_bytes() == value for name, value in inputs.items())
    assert not (study / 'manifest.json').exists(), 'The parent, not this worker, seals the study'


@pytest.mark.parametrize('mutate, fragment', [
    (lambda p: p.update(schema_version=True), 'Frozen worker plan'),
    (lambda p: p.update(proposal_objective='tune-again'), 'Unsupported proposal'),
    (lambda p: p['settings'].update(maximum_seconds=True), 'Fit seconds'),
    (lambda p: p['settings'].update(maximum_seconds=30.1), 'Fit seconds'),
    (lambda p: p['settings'].update(maximum_seconds=0), 'Fit seconds'),
    (lambda p: p['settings'].update(finalization_seconds=9), 'Fixed'),
    (lambda p: p['settings'].update(maximum_sweeps=3), 'Fixed'),
    (lambda p: p['runtime_files'].update({'runtime/weft/extra.py': 'a' * 64}), 'exact bounded runtime_files'),
    (lambda p: p['input_files'].update({'../escape': 'a' * 64}), 'exact bounded input_files'),
])
def test_plan_rejects_unbounded_or_ambiguous_settings_before_import(tmp_path, monkeypatch, mutate, fragment):
    study, plan, _, _ = study_fixture(tmp_path, monkeypatch)
    mutate(plan)
    encoded = raw(plan); (study / 'plan.json').write_bytes(encoded)
    monkeypatch.setattr(worker, '_load_runtime', lambda *a: pytest.fail('Native import reached'))
    assert worker.run_worker(study, digest(encoded)) == 1
    receipt = json.loads((study / 'worker.json').read_bytes())
    assert receipt['status'] == 'failed' and fragment in receipt['error']['message']
    assert receipt['counts']['fitter_calls'] == 0 and receipt['counts']['model_compilations'] == 0
    assert not (study / 'candidate-human.json').exists()


@pytest.mark.parametrize('document', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'[]'])
def test_strict_json_keeps_duplicate_nonfinite_and_nonobject_inputs_out(document):
    with pytest.raises(ValueError): worker._json(document)


def test_import_network_and_process_guards_are_installed_without_importing_media(monkeypatch):
    monkeypatch.setattr(worker.signal, 'setitimer', lambda *args: None)
    guard = worker._Guards(worker.time.monotonic())
    before = set(sys.modules)
    original = subprocess.Popen
    try:
        guard.install()
        blocked = [lambda: socket.create_connection(('example.invalid', 1)), lambda: socket.socket.connect(None, None),
                   lambda: socket.socket.bind(None, None), lambda: subprocess.Popen(['never-run']), lambda: os.system('never-run'),
                   lambda: builtins.__import__('cv2'), lambda: importlib.import_module('torch'),
                   lambda: importlib.import_module('weft.meta')]
        if hasattr(os, 'posix_spawn'): blocked.append(lambda: os.posix_spawn('/never-run', [], {}))
        for invoke in blocked:
            with pytest.raises(PermissionError): invoke()
        assert guard.counts['forbidden_calls'] == len(blocked)
        assert not ({'cv2', 'torch'} & (set(sys.modules) - before))
        assert guard.counts['media_calls'] == guard.counts['inference_calls'] == 0
    finally:
        guard.restore()
    assert subprocess.Popen is original and guard not in sys.meta_path


def test_native_caps_count_started_calls_and_block_steps_resets_paths_and_rendering():
    guard = worker._Guards(worker.time.monotonic())
    native, human, calls = fake_native()
    original = native.MjModel
    try:
        guard.native(native, human)
        for name, invoke in [('model_compilations', lambda: native.MjModel.from_xml_string('<mujoco/>')),
                             ('data_allocations', lambda: native.MjData(None)), ('forward_calls', lambda: native.mj_forward(None, None))]:
            guard.counts[name] = worker.NATIVE_LIMITS[name] - 1
            invoke()
            assert guard.counts[name] == worker.NATIVE_LIMITS[name]
            with pytest.raises(PermissionError, match='native call cap'): invoke()
        for invoke in (lambda: native.MjModel.from_xml_path('/never-read'), lambda: native.MjModel.from_binary_path('/never-read'),
                       lambda: native.mj_step(None, None), lambda: native.mj_resetData(None, None),
                       lambda: native.mj_resetDataKeyframe(None, None, 0), lambda: native.Renderer(None),
                       lambda: native.mjr_render(None), lambda: human().step(0), lambda: human().reset(), lambda: human().render()):
            with pytest.raises(PermissionError): invoke()
        assert calls == ['model', 'data', 'forward']
        assert guard.counts['physics_steps'] == guard.counts['reset_calls'] == guard.counts['media_calls'] == 0
    finally:
        guard.restore()
    assert native.MjModel is original


def test_resource_and_deadline_bounds_do_not_weaken_stricter_parent_limits(monkeypatch):
    installed = []
    monkeypatch.setattr(resource, 'getrlimit', lambda kind: (20, 25) if kind == resource.RLIMIT_CPU else (resource.RLIM_INFINITY,) * 2)
    monkeypatch.setattr(resource, 'setrlimit', lambda kind, pair: installed.append((kind, pair)))
    worker._install_limits()
    assert installed == [(resource.RLIMIT_CPU, (20, 25)), (resource.RLIMIT_FSIZE, (16 << 20, 16 << 20)), (resource.RLIMIT_CORE, (0, 0))]
    monkeypatch.setattr(worker.time, 'monotonic', lambda: 40.)
    guard = worker._Guards(0.)
    invoked = []
    with pytest.raises(TimeoutError): guard.counted('forward_calls', lambda: invoked.append(1))()
    assert invoked == [] and guard.counts['forward_calls'] == 0


@pytest.mark.parametrize('platform, expected', [('darwin', 123), ('linux', 125952), ('unknown', None)])
def test_rss_is_normalized_without_a_hard_cap_claim(monkeypatch, platform, expected):
    monkeypatch.setattr(worker.sys, 'platform', platform)
    monkeypatch.setattr(resource, 'getrusage', lambda _: SimpleNamespace(ru_maxrss=123))
    assert worker._rss_bytes() == expected


@pytest.mark.parametrize('changed', ['inputs/human.json', 'runtime/weft/root_motion.py', 'plan.json'])
def test_changed_pins_after_fitting_preserve_failure_not_completion(tmp_path, monkeypatch, changed):
    study, plan, pin, _ = study_fixture(tmp_path, monkeypatch)
    def alter(_):
        path = study / changed
        data = path.read_bytes(); path.write_bytes(data[:-1] + b' ')
    fake_loader(monkeypatch, plan, alter)
    assert worker.run_worker(study, pin) == 1
    receipt = json.loads((study / 'worker.json').read_bytes())
    assert receipt['status'] == 'failed' and 'exact pin' in receipt['error']['message']
    assert receipt['counts']['fitter_calls'] == 1 and receipt['promoted'] is False


@pytest.mark.parametrize('kind', ['plan', 'input', 'runtime', 'symlink', 'bytecode'])
def test_file_bounds_links_and_unpinned_runtime_files_fail_before_import(tmp_path, monkeypatch, kind):
    study, plan, pin, _ = study_fixture(tmp_path, monkeypatch)
    if kind in {'plan', 'input', 'runtime'}:
        name, bound = {'plan': ('plan.json', worker.PLAN_LIMIT), 'input': ('inputs/human.json', worker.INPUT_LIMIT),
                       'runtime': ('runtime/weft/human.py', worker.SOURCE_LIMIT)}[kind]
        with (study / name).open('r+b') as stream: stream.truncate(bound + 1)
    elif kind == 'symlink':
        member = study / 'inputs/human.json'
        outside = tmp_path / 'outside.json'; member.rename(outside); member.symlink_to(outside)
    else:
        (study / 'runtime/weft/__pycache__').mkdir()
    monkeypatch.setattr(worker, '_load_runtime', lambda *a: pytest.fail('Native import reached'))
    assert worker.run_worker(study, pin) == 1
    receipt = json.loads((study / 'worker.json').read_bytes())
    assert receipt['status'] == 'failed' and receipt['counts']['fitter_calls'] == 0


@pytest.mark.parametrize('change', [lambda r: r['report'].update(promoted=True), lambda r: r['report'].update(physics_steps=1),
                                   lambda r: r['report'].update(candidate_sha256='a' * 64), lambda r: r.update(extra='unknown')])
def test_returned_result_cannot_promote_or_change_identity(tmp_path, monkeypatch, change):
    study, plan, pin, _ = study_fixture(tmp_path, monkeypatch)
    fake_loader(monkeypatch, plan, change)
    assert worker.run_worker(study, pin) == 1
    assert not (study / 'candidate-human.json').exists() and not (study / 'report.json').exists()
    assert json.loads((study / 'worker.json').read_bytes())['status'] == 'failed'


def test_invalid_returned_status_cannot_poison_the_failure_receipt(tmp_path, monkeypatch):
    study, plan, pin, _ = study_fixture(tmp_path, monkeypatch)
    fake_loader(monkeypatch, plan, lambda value: value.update(status=['invalid', float('nan')]))
    assert worker.run_worker(study, pin) == 1
    receipt = json.loads((study / 'worker.json').read_bytes())
    assert receipt['status'] == 'failed' and receipt['proposal_status'] is None
    assert receipt['error'] is not None and receipt['counts']['fitter_calls'] == 1


def test_combined_output_bound_and_existing_output_are_not_overwritten(tmp_path, monkeypatch, capsys):
    study, plan, pin, _ = study_fixture(tmp_path, monkeypatch)
    fake_loader(monkeypatch, plan)
    monkeypatch.setattr(worker, 'OUTPUT_LIMIT', 64)
    assert worker.run_worker(study, pin) == 1
    assert not (study / 'candidate-human.json').exists()
    receipt = (study / 'worker.json').read_bytes()
    assert 'Combined proposal output' in json.loads(receipt)['error']['message']
    assert worker.run_worker(study, pin) == 1
    assert (study / 'worker.json').read_bytes() == receipt
    assert 'receipt_written' in capsys.readouterr().err


def test_bad_plan_hash_has_failure_receipt_and_cli_helper_forwards_only_arguments(tmp_path, monkeypatch, capsys):
    study, plan, _, _ = study_fixture(tmp_path, monkeypatch)
    assert worker.run_worker(study, '0' * 64) == 1
    receipt = json.loads((study / 'worker.json').read_bytes())
    assert receipt['status'] == 'failed' and receipt['plan_sha256'] is None
    calls = []
    monkeypatch.setattr(worker, 'run_worker', lambda *args: calls.append(args) or 0)
    assert worker.main([str(study), 'a' * 64]) == 0
    assert calls == [(str(study), 'a' * 64)]
    assert worker.main([]) == 1 and 'Usage:' in capsys.readouterr().err
