"""Worker custody and lifecycle checks; models/geometry are mocked except import-only startup."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from weft import body_camera_root_worker as worker
from weft import whole_body_reference_worker as legacy
from test_whole_body_reference_worker import fake_native


def raw(value):
    return worker._encode(value)


def digest(value):
    return sha256(value).hexdigest()


def setup_study(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    for name in tuple(sys.modules):
        if name.startswith('weft.') and name != worker.__name__:
            monkeypatch.delitem(sys.modules, name)
    study = tmp_path / 'study'; study.mkdir()
    package = study / 'runtime/weft'; package.mkdir(parents=True)
    (study / 'inputs').mkdir()
    identity = {'profile': 'fixture-source', 'source_sha256': 'a' * 64}
    inputs = {'inputs/human.json': raw({'reference': [{'time_s': i / 30} for i in range(3)]}), 'inputs/evidence-derived.json': raw({'frames': []}),
              'inputs/input-receipt.json': raw(identity), 'inputs/camera-config.json': raw({'quality': 'authored'})}
    for name, value in inputs.items(): (study / name).write_bytes(value)
    source = Path(worker.__file__).parent
    pins = {}
    for name in worker.RUNTIME_FILES:
        value = (source / Path(name).name).read_bytes() if Path(name).name in {
            'body_camera_root_worker.py', 'whole_body_reference_worker.py'} else b'"""Pure fixture module."""\n'
        (study / name).write_bytes(value); pins[name] = digest(value)
    versions = {'python': sys.version.split()[0], 'numpy': 'fixture', 'mujoco': 'fixture', 'gymnasium': 'fixture'}
    plan = {'schema_version': 1, 'profile': worker.PLAN_PROFILE, 'status': 'frozen', 'source': identity,
            'input_files': {name: digest(value) for name, value in inputs.items()}, 'runtime_files': pins,
            'runtime_versions': versions, 'settings': {'maximum_seconds': .1}}
    (study / 'plan.json').write_bytes(raw(plan))
    monkeypatch.setattr(worker, '__file__', str(package / 'body_camera_root_worker.py'))
    monkeypatch.setattr(sys.modules['weft'], '__file__', str(package / '__init__.py'))
    monkeypatch.setattr(worker, '_install_limits', lambda: None)
    monkeypatch.setattr(worker.signal, 'setitimer', lambda *args: None)
    monkeypatch.setattr(worker, '_versions', lambda: versions.copy())
    return study, plan, inputs


def fake_pipeline(monkeypatch, *, change=True, after_propose=None, fail_stage=None):
    calls = []
    def load(runtime, guards, versions):
        native, human, native_calls = fake_native(); guards.native(native, human)
        def model():
            made = native.MjModel.from_xml_string('<original/>')
            native.MjData(made); data = native.MjData(made)
            for _ in range(5): native.mj_forward(made, data)
        def extract(human_raw, evidence_raw, **kwargs):
            calls.append('extract'); model()
            assert kwargs['runtime']['versions'] == versions
            return raw({'profile': 'fixed-original-body-geometry-v1', 'status': 'completed'})
        def adapt(prepared, camera, geometry, **kwargs):
            calls.append('adapt')
            assert set(prepared['files']) == {'inputs/human.json', 'inputs/evidence-derived.json'}
            if fail_stage == 'adapt': raise ValueError('malformed bound camera')
            return {'input_document_bytes': raw({'profile': 'fixture-input'}),
                    'provenance_bytes': raw({'targets': []}), 'source_slots_bytes': raw({'slots': []}), 'pins': {}}
        def propose(input_raw, **kwargs):
            calls.append('propose')
            if after_propose: after_propose()
            return {'status': 'proposed' if change else 'unchanged', 'camera_candidate': {}, 'root_reference_candidate': [],
                    'report': {'input_sha256': digest(input_raw), 'admitted': False, 'promoted': False,
                               'physical_validation': 'unverified', 'counts': {'physics_steps': 0}}}
        def diagnose(human_raw, evidence_raw, geometry, proposal, **kwargs):
            calls.append('diagnose')
            if change: model()
            return raw({'profile': 'fixed-original-body-geometry-check-v1', 'status': 'completed' if change else 'not_required'})
        return extract, adapt, propose, diagnose
    monkeypatch.setattr(worker, '_load_runtime', load)
    return calls


@pytest.mark.parametrize('change', [False, True])
def test_single_proposal_pipeline_pins_outputs_and_preserves_original_inputs(tmp_path, monkeypatch, change):
    study, plan, inputs = setup_study(tmp_path, monkeypatch)
    calls = fake_pipeline(monkeypatch, change=change)
    assert worker.run_worker(study, digest(raw(plan))) == 0
    receipt = json.loads((study / 'worker.json').read_bytes())
    assert receipt['status'] == 'completed' and calls == ['extract', 'adapt', 'propose', 'diagnose']
    assert receipt['counts']['model_compilations'] == (2 if change else 1)
    assert receipt['counts']['data_allocations'] == (4 if change else 2)
    assert receipt['counts']['forward_calls'] == (10 if change else 5)
    assert receipt['counts']['fitter_calls'] == 1 and receipt['counts']['forbidden_calls'] == 0
    assert receipt['native_call_limits'] == {'model_compilations': 2, 'data_allocations': 4, 'forward_calls': 12}
    assert receipt['global_native_call_limits'] == worker.NATIVE_LIMITS and receipt['native_limits_derived_from_input']
    assert set(receipt['output_files']) == set(worker.OUTPUT_LIMITS)
    assert all(digest((study / name).read_bytes()) == pin for name, pin in receipt['output_files'].items())
    assert receipt['output_bytes'] == sum((study / name).stat().st_size for name in receipt['output_files'])
    assert all((study / name).read_bytes() == value for name, value in inputs.items())
    assert receipt['runtime_versions'] == plan['runtime_versions'] and not receipt['admitted'] and not receipt['promoted']
    assert not (study / 'candidate-human.json').exists() and not (study / 'manifest.json').exists()


@pytest.mark.parametrize('mutate, fragment', [
    (lambda plan: plan.update(schema_version=True), 'frozen plan'),
    (lambda plan: plan['runtime_files'].pop('runtime/weft/whole_body_reference_inputs.py'), 'exact runtime'),
    (lambda plan: plan['input_files'].update({'inputs/source.mp4': 'a' * 64}), 'metadata only'),
    (lambda plan: plan['input_files'].update({'inputs/../secret.json': 'a' * 64}), 'unsafe'),
    (lambda plan: plan['settings'].update(maximum_seconds=10.1), 'unsupported fit'),
    (lambda plan: plan['settings'].update(maximum_iterations=True), 'unsupported fit'),
    (lambda plan: plan['settings'].update(new_gain=1), 'tighter'),
    (lambda plan: plan['runtime_versions'].update(torch='fixture'), 'exact dependency'),
])
def test_invalid_plans_reject_before_import_or_geometry(tmp_path, monkeypatch, mutate, fragment):
    study, plan, _ = setup_study(tmp_path, monkeypatch)
    mutate(plan); (study / 'plan.json').write_bytes(raw(plan))
    monkeypatch.setattr(worker, '_load_runtime', lambda *args: pytest.fail('Runtime import reached'))
    assert worker.run_worker(study, digest(raw(plan))) == 1
    receipt = json.loads((study / 'worker.json').read_bytes())
    assert fragment in receipt['error']['message'] and receipt['counts']['model_compilations'] == 0


def test_adapter_failure_preserves_original_geometry_and_exact_failure_receipt(tmp_path, monkeypatch):
    study, plan, _ = setup_study(tmp_path, monkeypatch)
    fake_pipeline(monkeypatch, fail_stage='adapt')
    assert worker.run_worker(study, digest(raw(plan))) == 1
    receipt = json.loads((study / 'worker.json').read_bytes())
    assert receipt['status'] == 'failed' and set(receipt['output_files']) == {'geometry.json'}
    assert receipt['counts']['fitter_calls'] == 0 and not (study / 'proposal.json').exists()


@pytest.mark.parametrize('member', ['inputs/human.json', 'plan.json', 'runtime/weft/io.py'])
def test_changed_inputs_plan_or_code_cannot_complete(tmp_path, monkeypatch, member):
    study, plan, _ = setup_study(tmp_path, monkeypatch)
    fake_pipeline(monkeypatch, after_propose=lambda: (study / member).write_bytes(b'{}'))
    assert worker.run_worker(study, digest(raw(plan))) == 1
    receipt = json.loads((study / 'worker.json').read_bytes())
    assert 'frozen pin' in receipt['error']['message'] and (study / 'proposal.json').exists()


def test_combined_output_limit_stops_before_fitting_and_preserves_completed_outputs(tmp_path, monkeypatch):
    study, plan, _ = setup_study(tmp_path, monkeypatch)
    calls = fake_pipeline(monkeypatch)
    monkeypatch.setattr(worker, 'MAX_OUTPUT_BYTES', 95)
    assert worker.run_worker(study, digest(raw(plan))) == 1
    receipt = json.loads((study / 'worker.json').read_bytes())
    assert 'combined outputs' in receipt['error']['message'] and 'propose' not in calls
    assert set(receipt['output_files']) == {'geometry.json'}


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    study, plan, _ = setup_study(tmp_path, monkeypatch)
    (study / 'proposal.json').write_bytes(b'original saved output')
    monkeypatch.setattr(worker, '_load_runtime', lambda *args: pytest.fail('Runtime import reached'))
    assert worker.run_worker(study, digest(raw(plan))) == 1
    assert (study / 'proposal.json').read_bytes() == b'original saved output'


def test_narrow_guard_caps_do_not_change_legacy_caps(monkeypatch):
    guard = worker._guards(legacy, time.monotonic())
    native, human, calls = fake_native()
    try:
        guard.native(native, human)
        for _ in range(2): native.MjModel.from_xml_string('<original/>')
        for _ in range(4): native.MjData(None)
        guard.counts['forward_calls'] = 5401
        native.mj_forward(None, None)
        for invoke in (lambda: native.MjModel.from_xml_string('<extra/>'), lambda: native.MjData(None),
                       lambda: native.mj_forward(None, None), lambda: native.mj_step(None, None),
                       lambda: native.mj_resetData(None, None), lambda: native.Renderer(None)):
            with pytest.raises(PermissionError): invoke()
        assert guard.counts['forbidden_calls'] == 6
        assert legacy.NATIVE_LIMITS == {'model_compilations': 2, 'data_allocations': 6, 'forward_calls': 120010}
    finally:
        guard.restore()


def test_resource_limits_only_tighten_inherited_caps(monkeypatch):
    seen = []
    monkeypatch.setattr(worker.resource, 'getrlimit', lambda kind: (2, 3))
    monkeypatch.setattr(worker.resource, 'setrlimit', lambda kind, limits: seen.append(limits))
    worker._install_limits()
    assert seen == [(2, 3), (2, 3), (0, 0)]


def test_effective_forward_cap_uses_exact_source_span_not_fixed_maximum():
    human = raw({'reference': [{'time_s': i / 15} for i in range(118)]})
    assert worker._input_native_limits(human) == {'model_compilations': 2, 'data_allocations': 4, 'forward_calls': 706}
    with pytest.raises(ValueError, match='source clocks'):
        worker._input_native_limits(raw({'reference': [{'time_s': 0.}, {'time_s': True}, {'time_s': 1.}]}))


def test_startup_allowance_rejects_other_commands_options_and_repeats(monkeypatch):
    monkeypatch.setattr(worker.sys, 'platform', 'darwin')
    guard = worker._guards(legacy, time.monotonic())
    command = ['/usr/sbin/sysctl', '-n', 'sysctl.proc_translated']
    for argv, kwargs in ((['/bin/sh'], {'capture_output': True}), (command, {'capture_output': True, 'shell': True}),
                         (command, {})):
        with pytest.raises(PermissionError): worker._architecture_probe(guard, argv, **kwargs)
    guard.startup_os_probes.append({'command': command})
    with pytest.raises(PermissionError): worker._architecture_probe(guard, command, capture_output=True)
    assert guard.counts['forbidden_calls'] == 4


def test_runtime_inventory_rejects_bytecode_without_importing_it(tmp_path, monkeypatch):
    study, plan, _ = setup_study(tmp_path, monkeypatch)
    (study / 'runtime/weft/__pycache__').mkdir()
    monkeypatch.setattr(worker, '_load_runtime', lambda *args: pytest.fail('Runtime import reached'))
    assert worker.run_worker(study, digest(raw(plan))) == 1
    assert 'unlisted runtime' in json.loads((study / 'worker.json').read_bytes())['error']['message']


def test_fresh_isolated_runtime_imports_only_frozen_closure_without_models(tmp_path):
    source = Path(worker.__file__).parent
    package = tmp_path / 'runtime/weft'; package.mkdir(parents=True)
    for name in worker.RUNTIME_MODULES:
        (package / (name + '.py')).write_bytes((source / (name + '.py')).read_bytes())
    code = '''import importlib.util,json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); spec=importlib.util.spec_from_file_location('standalone_camera_worker',p/'weft/body_camera_root_worker.py')
w=importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
base=w._load_guard(p/'weft/whole_body_reference_worker.py'); guard=w._guards(base,time.monotonic())
try:
 guard.install(); api=w._load_runtime(p,guard,w._versions())
 loaded={name:str(pathlib.Path(m.__file__).resolve()) for name,m in sys.modules.items() if name=='weft' or name.startswith('weft.')}
 assert all(pathlib.Path(value).is_relative_to(p) for value in loaded.values())
 assert all(value==0 for value in guard.counts.values())
 print(json.dumps({'functions':len(api),'counts':guard.counts,'loaded':sorted(loaded),'startup_os_probes':guard.startup_os_probes,'isolated':sys.flags.isolated,'bytecode_disabled':sys.dont_write_bytecode}))
finally: guard.restore()
'''
    env = {'PATH': '/usr/bin:/bin', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(package.parent)],
                            env=env, cwd=tmp_path, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
    value = json.loads(result.stdout)
    assert value['functions'] == 4 and value['isolated'] and value['bytecode_disabled']
    assert all(count == 0 for count in value['counts'].values())
    assert 'weft.whole_body_reference_inputs' in value['loaded']
    assert len(value['startup_os_probes']) == (1 if sys.platform == 'darwin' else 0)
