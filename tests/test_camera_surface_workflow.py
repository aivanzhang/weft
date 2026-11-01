"""Opt-in camera custody reaches the existing worker without admitting an angle overlay."""

from copy import deepcopy
from hashlib import sha256
import json
import subprocess
import sys
from types import ModuleType

import pytest

from weft import source_surface_fit_inputs as inputs
from weft import whole_body_reference_pipeline as pipeline
from weft import whole_body_reference_worker as worker
from test_surface_fit_workflow import bundle, snapshot
from test_whole_body_reference_pipeline import workflow, verify_seal
from test_whole_body_reference_worker import study_fixture


def raw(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def camera_plan():
    files = worker.INPUT_FILES | {'inputs/surface-config.json', 'inputs/surface-observation-00.json',
                                  'inputs/surface-camera.json'}
    runtime = worker.RUNTIME_MODULES | worker.SURFACE_MODULES | worker.CAMERA_MODULES
    return {'schema_version': 1, 'profile': 'whole-body-reference-candidate-plan-v1', 'status': 'frozen',
        'proposal_objective': 'camera_surface_interval_v1', 'camera_image_scale': .001,
        'surface_constraints': {'profile': inputs.PROFILE, 'config_sha256': 'a' * 64, 'source_sha256': 'b' * 64,
                               'weight': 2., 'observation_count': 1, 'optional_inputs': ['camera']},
        'input_files': dict.fromkeys(files, 'a' * 64),
        'runtime_files': {f'runtime/weft/{name}.py': 'c' * 64 for name in runtime},
        'runtime_versions': dict.fromkeys(('python', 'numpy', 'mujoco', 'gymnasium'), 'fixture'),
        'settings': worker.SETTINGS | {'maximum_seconds': .1}}


def camera_result(plan):
    method, status = 'blocked-camera-body-surface-interval-reference-v1', 'rejected_construction_proposal'
    identity = {'input_human_sha256': plan['input_files']['inputs/human.json'],
                'input_evidence_sha256': plan['input_files']['inputs/evidence-derived.json']}
    bindings = {'body_raw_sha256': identity['input_human_sha256'], 'evidence_raw_sha256': identity['input_evidence_sha256'],
                'camera_raw_sha256': plan['input_files']['inputs/surface-camera.json']}
    candidate = {'schema_version': 1, 'profile': 'camera-body-angle-proposal-v1', 'input_bindings': bindings,
        'reference_candidate': [], 'whole_body_reference_refinement': {'method': method, 'status': status, 'promoted': False, **identity},
        'installation': 'unsupported_angle_overlay_not_a_human_spec', 'body_installed': False,
        'raw_camera_changed': False, 'root_or_floor_changed': False, 'promoted': False,
        'contact': 'unknown', 'physical_validation': 'unverified'}
    report = {'schema_version': 1, 'profile': method, 'status': status, 'promoted': False,
        'physical_validation': 'unverified', 'physics_steps': 0, 'contract': {'floor_penetration_limit_model_units': .025},
        'candidate_kind': candidate['profile'], 'geometry_admission': candidate['installation'],
        'camera_fit': {'input_bindings': deepcopy(bindings), 'image_loss_scale_per_pixel': plan['camera_image_scale'],
                       'image_loss_scale_status': 'authored'}, 'candidate_sha256': worker._canonical(candidate), **identity}
    return {'status': status, 'human_candidate': None, 'angle_proposal': candidate, 'report': report}


@pytest.mark.parametrize('scale', [1e-6, 1.])
def test_camera_worker_plan_accepts_explicit_bounded_scale_and_exact_runtime(scale):
    plan = camera_plan(); plan['camera_image_scale'] = scale
    assert worker._plan(raw(plan)) == plan
    assert set(inputs.CAMERA_RUNTIME_MODULES) == worker.CAMERA_MODULES


@pytest.mark.parametrize('change', [
    lambda p: p.pop('camera_image_scale'), lambda p: p.update(camera_image_scale=True),
    lambda p: p.update(camera_image_scale=0), lambda p: p.update(camera_image_scale=1e-7),
    lambda p: p.update(camera_image_scale=1.001), lambda p: p.update(camera_image_scale=float('nan')),
    lambda p: p['surface_constraints'].update(optional_inputs=[]),
    lambda p: p['runtime_files'].pop('runtime/weft/camera_body_fit.py'),
    lambda p: p['runtime_files'].update({'runtime/weft/unrequested.py': 'a' * 64}),
    lambda p: p.update(proposal_objective='surface_interval_v1'),
])
def test_camera_worker_plan_rejects_missing_ambiguous_or_cross_mode_options(change):
    plan = camera_plan(); change(plan)
    with pytest.raises(ValueError):
        worker._plan(json.dumps(plan).encode())


def test_surface_preparation_passes_exact_camera_without_changing_interval_objective(tmp_path):
    path, config, body = bundle(tmp_path / 'config')
    files, descriptor = snapshot(path, config['source_sha256'])
    result = inputs.prepare_surface_fit_objective(files | {'inputs/human.json': body}, descriptor, camera_image_scale=.002)
    assert result['camera_document'] == files['inputs/surface-camera.json']
    assert result['camera_image_scale'] == .002
    assert result['surface_interval_objective'].describe()['body_sha256'] == sha256(body).hexdigest()
    assert result['surface_interval_objective'].image_admission(0)['source_frame'] == 0


@pytest.mark.parametrize('scale', [True, 0., 1e-7, 1.01, float('inf')])
def test_surface_camera_scale_is_validated_before_geometry(tmp_path, scale):
    path, config, body = bundle(tmp_path / 'config')
    files, descriptor = snapshot(path, config['source_sha256'])
    with pytest.raises(ValueError, match='camera image scale'):
        inputs.prepare_surface_fit_objective(files | {'inputs/human.json': body}, descriptor, camera_image_scale=scale)


def test_camera_scale_cannot_supply_a_missing_calibration(tmp_path):
    path, config, body = bundle(tmp_path / 'config', available=False)
    files, descriptor = snapshot(path, config['source_sha256'])
    with pytest.raises(ValueError, match='exact supplied camera'):
        inputs.prepare_surface_fit_objective(files | {'inputs/human.json': body}, descriptor, camera_image_scale=.001)


def test_worker_writes_only_separate_uninstalled_overlay():
    plan = camera_plan(); result = camera_result(plan)
    files = worker._result(result, plan)
    assert set(files) == {'angle-proposal.json', 'report.json'}
    assert json.loads(files['angle-proposal.json'])['installation'] == 'unsupported_angle_overlay_not_a_human_spec'
    assert result['human_candidate'] is None


@pytest.mark.parametrize('change', [
    lambda r: r.update(human_candidate={}), lambda r: r['angle_proposal'].update(body_installed=True),
    lambda r: r['angle_proposal'].update(raw_camera_changed=True),
    lambda r: r['angle_proposal'].update(root_or_floor_changed=True),
    lambda r: r['angle_proposal'].update(contact='verified'),
    lambda r: r['angle_proposal']['input_bindings'].update(camera_raw_sha256='f' * 64),
    lambda r: r['report']['camera_fit'].update(image_loss_scale_per_pixel=.002),
    lambda r: r['report']['camera_fit'].update(image_loss_scale_status='measured'),
    lambda r: r['report'].update(geometry_admission='admitted'),
    lambda r: r['report']['contract'].update(floor_penetration_limit_model_units=.03),
])
def test_worker_rejects_overlay_promotion_changed_inputs_and_relaxed_contract(change):
    plan = camera_plan(); result = camera_result(plan); change(result)
    with pytest.raises(ValueError):
        worker._result(result, plan)


def test_worker_routes_camera_options_and_retains_separate_output(tmp_path, monkeypatch):
    study, plan, _, files = study_fixture(tmp_path, monkeypatch, objective='camera_surface_interval_v1')
    contract = camera_plan()
    plan.update(camera_image_scale=.001, surface_constraints=contract['surface_constraints'])
    for name in contract['input_files'].keys() - files.keys():
        files[name] = b'{}'; (study / name).write_bytes(files[name])
    plan['input_files'] = {name: sha256(data).hexdigest() for name, data in files.items()}
    plan['surface_constraints']['config_sha256'] = plan['input_files']['inputs/surface-config.json']
    for name in contract['runtime_files'].keys() - plan['runtime_files'].keys():
        data = b'# pure worker routing fixture\n'; (study / name).write_bytes(data)
        plan['runtime_files'][name] = sha256(data).hexdigest()
    (study / 'plan.json').write_bytes(raw(plan))
    calls = []
    helper = ModuleType('weft.source_surface_fit_inputs')
    helper.__file__ = str(study / 'runtime/weft/source_surface_fit_inputs.py')
    def options(values, descriptor, **kwargs):
        calls.append(('prepare', values, descriptor, kwargs))
        return {'camera_document': values['inputs/surface-camera.json'], **kwargs}
    helper.prepare_surface_fit_objective = options
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    def load(*args):
        def fit(*args, **kwargs):
            calls.append(('fit', args, kwargs))
            return camera_result(plan)
        return fit
    monkeypatch.setattr(worker, '_load_runtime', load)
    assert worker.run_worker(study, sha256(raw(plan)).hexdigest()) == 0
    assert calls[0][3] == {'camera_image_scale': .001}
    assert calls[1][1] == (files['inputs/human.json'], files['inputs/evidence-derived.json'])
    assert calls[1][2]['camera_document'] == files['inputs/surface-camera.json']
    assert calls[1][2]['proposal_objective'] == 'camera_surface_interval_v1'
    assert (study / 'angle-proposal.json').exists() and not (study / 'candidate-human.json').exists()
    receipt = json.loads((study / 'worker.json').read_bytes())
    assert receipt['status'] == 'completed' and receipt['counts']['fitter_calls'] == 1
    assert receipt['counts']['model_compilations'] == receipt['counts']['physics_steps'] == receipt['paid_calls'] == 0


@pytest.mark.parametrize('mismatch', [False, True])
def test_parent_uses_separate_camera_custody_and_rejects_different_camera_bytes(workflow, tmp_path, monkeypatch, mismatch):
    source, out, source_snapshot, calls = workflow
    path, config, body = bundle(tmp_path / 'config')
    camera_bytes = (path.parent / config['camera']['path']).read_bytes()
    source_snapshot['identity']['source_sha256'] = config['source_sha256']
    source_snapshot['files'].update({'inputs/human.json': body, 'inputs/source-camera.json': camera_bytes + (b' ' if mismatch else b'')})
    module = ModuleType('weft.camera_body_reference_inputs')
    module.prepare_camera_reference_inputs = lambda *a, **kw: source_snapshot
    def recheck(*args, **kwargs): calls['rechecks'] += 1
    module.recheck_camera_reference_inputs = recheck
    monkeypatch.setitem(sys.modules, module.__name__, module)
    def runtime(*, surface=False, camera=False):
        assert surface and camera
        names = worker.RUNTIME_MODULES | worker.SURFACE_MODULES | worker.CAMERA_MODULES
        return ({f'runtime/weft/{name}.py': b'# frozen fixture\n' for name in names},
                dict.fromkeys(('python', 'numpy', 'mujoco', 'gymnasium'), 'fixture'))
    monkeypatch.setattr(pipeline, '_runtime', runtime)
    def run(study, pin, seconds):
        plan = worker._plan((study / 'plan.json').read_bytes())
        for name, data in worker._result(camera_result(plan), plan).items(): (study / name).write_bytes(data)
        (study / 'worker.json').write_bytes(raw({'status': 'completed', 'plan_sha256': pin}))
        return {'returncode': 0, 'error': None, 'interrupted': False, 'seconds': .01}
    monkeypatch.setattr(pipeline, '_run_worker', run)
    kwargs = {'proposal_objective': 'camera_surface_interval_v1', 'surface_config': path, 'camera_image_scale': .001}
    if mismatch:
        with pytest.raises(ValueError, match='exact original camera-body source bytes'):
            pipeline.prepare_whole_body_reference_proposal(source, out, **kwargs)
        assert not out.exists() and calls == {'reports': 0, 'rechecks': 0}
    else:
        result = pipeline.prepare_whole_body_reference_proposal(source, out, **kwargs)
        assert result['status'] == 'completed' and result['simulation_ready'] is False
        assert result['candidate_kind'] == 'camera-body-angle-proposal-v1'
        assert calls == {'reports': 1, 'rechecks': 1}
        assert verify_seal(out)['status'] == verify_seal(out / 'proposal')['status'] == 'completed'
        assert not (out / 'proposal/candidate-human.json').exists()


@pytest.mark.parametrize('kwargs', [
    {'proposal_objective': 'camera_surface_interval_v1', 'surface_config': 'unused'},
    {'proposal_objective': 'camera_surface_interval_v1', 'surface_config': 'unused', 'camera_image_scale': True},
    {'proposal_objective': 'camera_surface_interval_v1', 'surface_config': 'unused', 'camera_image_scale': 1.01},
    {'proposal_objective': 'image_v1', 'camera_image_scale': .001},
])
def test_parent_rejects_camera_mode_mismatch_before_source_reads(workflow, kwargs):
    source, out, _, _ = workflow
    with pytest.raises(ValueError, match='[Cc]amera image scale'):
        pipeline.prepare_whole_body_reference_proposal(source, out, **kwargs)
    assert not out.exists()


def test_cli_routes_camera_surface_options_without_starting_a_server(tmp_path, monkeypatch, capsys):
    from weft.cli import main
    calls = []
    monkeypatch.setattr(pipeline, 'prepare_whole_body_reference_proposal',
                        lambda *a, **kw: calls.append((a, kw)) or {'status': 'completed'})
    monkeypatch.setattr(sys, 'argv', ['weft', 'body-reference-propose', '--run', 'camera-body', '--out', str(tmp_path / 'out'),
        '--objective', 'camera-surface-intervals', '--surface-config', 'config.json', '--camera-image-scale', '.001'])
    main()
    assert calls[0][1]['proposal_objective'] == 'camera_surface_interval_v1'
    assert calls[0][1]['surface_config'] == 'config.json' and calls[0][1]['camera_image_scale'] == .001
    assert json.loads(capsys.readouterr().out)['status'] == 'completed'


def test_cold_archived_camera_helpers_import_without_native_or_parent_only_dependencies(tmp_path):
    files, _ = pipeline._runtime(surface=True, camera=True)
    assert len(files) == 29 and sum(map(len, files.values())) < 8 << 20
    for name, data in files.items():
        target = tmp_path / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    script = r'''
import builtins,json,os,socket,subprocess,sys
def denied(*args,**kwargs):raise AssertionError('Unexpected process or network call')
subprocess.Popen=subprocess.run=subprocess.check_call=subprocess.check_output=denied
socket.socket=socket.create_connection=socket.getaddrinfo=denied
for name in ('system','fork','forkpty','posix_spawn','posix_spawnp'):
    if hasattr(os,name):setattr(os,name,denied)
original=builtins.__import__
def importing(name,*args,**kwargs):
    assert name.split('.')[0] not in {'mujoco','gymnasium','cv2','mediapipe','torch','tensorflow','imageio','PIL'}, name
    assert name not in {'weft.recording','weft.shared_world_packaging'}, name
    return original(name,*args,**kwargs)
builtins.__import__=importing
sys.path.insert(0,sys.argv[1])
from weft.camera_body_fit import prepare_camera_body_fit
from weft.whole_body_reference_refinement import propose_whole_body_reference_refinement
assert callable(prepare_camera_body_fit) and callable(propose_whole_body_reference_refinement)
assert 'mujoco' not in sys.modules and 'gymnasium' not in sys.modules
print(json.dumps({'cold_camera_import':True,'native_engine_imported':False}))
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(tmp_path / 'runtime')],
        capture_output=True, text=True, timeout=5,
        env={'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1',
             'OMP_NUM_THREADS': '1', 'TZ': 'America/Los_Angeles'})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'cold_camera_import': True, 'native_engine_imported': False}
