"""Mocked state isolation and exact invocation sequence; no native parity claim."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from weft import shared_exercise_action_prediction as prediction
from weft.spatial_pose import SPATIAL_JOINT_NAMES


def fixture(monkeypatch, *, roots=True):
    model = SimpleNamespace(nu=22, na=0, nplugin=0, nsensor=0, nv=28, nq=29, nbody=3, ngeom=3,
        opt=SimpleNamespace(integrator=prediction.mujoco.mjtIntegrator.mjINT_IMPLICITFAST, timestep=1/480))
    def data():
        d = SimpleNamespace(time=0., qpos=np.zeros(29), qvel=np.zeros(28), act=np.zeros(0), ctrl=np.zeros(22),
            qacc=np.zeros(28), qacc_warmstart=np.full(28, .125), qfrc_applied=np.full(28, 9.),
            xfrc_applied=np.full((3, 6), 9.), xpos=np.ones((3, 3)), xquat=np.zeros((3, 4)),
            geom_xpos=np.ones((3, 3)), geom_xmat=np.zeros((3, 9)), userdata=np.zeros(0),
            mocap_pos=np.zeros((0, 3)), mocap_quat=np.zeros((0, 4)), plugin_state=np.zeros(0),
            ncon=1, cached_time=0.)
        for name in ('qfrc_bias', 'qfrc_passive', 'qfrc_actuator', 'qfrc_constraint', 'qfrc_smooth', 'qacc_smooth'):
            setattr(d, name, np.zeros(28))
        return d
    live = data(); calls = []; states = []
    names = SPATIAL_JOINT_NAMES + ('root_x', 'root_y', 'root_z')
    engine = SimpleNamespace(model=model, data=live, human_action_names=names, human_motor_count=22,
        human_limits=np.arange(1., 23.), _actuators=np.arange(21, -1, -1), ankle_names=(),
        _q=np.arange(22), _v=np.arange(22), scale=1., _dynamic_objects=[{'id': 'bell'}],
        _object_body_ids={'bell': 2}, substeps=16, dt=1/30, _initialized=True, _done=False, steps=0,
        scene={'profile': 'shared-world-v2', 'bounds': [3., 3., 3.]})
    def contacts(*, data):
        assert data is not live
        calls.append('contacts')
        return {'cached_time': data.cached_time, 'details': [{'distance': -.001}], 'counts': {'object_floor': 1}}
    engine._current_contacts = contacts
    count = 22 if roots else 19
    env = SimpleNamespace(_engine=engine, action_count=count, action_names=names[:count], _done=False,
        horizon_steps=254, scene=engine.scene, task={'profile': 'shared-exercise-task-v1'})
    def allocate(m):
        assert m is model; calls.append('allocate')
        scratch = data(); states.append(scratch); return scratch
    def copy(dest, m, src):
        assert src is live and dest is not live and m is model
        calls.append('copy')
        for key, value in vars(src).items(): setattr(dest, key, deepcopy(value))
    def step(m, d):
        assert d is not live and m is model
        assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
        calls.append('step'); d.cached_time = d.time
        d.qacc[:22] = d.ctrl + d.qacc_warmstart[:22]
        d.qvel[:22] += d.qacc[:22] * model.opt.timestep
        d.qpos[:22] += d.qvel[:22] * model.opt.timestep
        d.qacc_warmstart[:] = d.qacc
        d.time += model.opt.timestep
    def forward(m, d):
        assert d is not live and m is model
        calls.append('forward'); d.cached_time = d.time
    def object_velocity(m, d, kind, body, output, local):
        output[:] = 0.
    monkeypatch.setattr(prediction.mujoco, 'MjData', allocate)
    monkeypatch.setattr(prediction.mujoco, 'mj_copyData', copy)
    monkeypatch.setattr(prediction.mujoco, 'mj_step', step)
    monkeypatch.setattr(prediction.mujoco, 'mj_forward', forward)
    monkeypatch.setattr(prediction.mujoco, 'mj_objectVelocity', object_velocity)
    for name in dir(prediction.mujoco):
        if name.startswith('get_mjcb_'): monkeypatch.setattr(prediction.mujoco, name, lambda: None)
    return SimpleNamespace(env=env, engine=engine, model=model, live=live, calls=calls, states=states)


@pytest.mark.parametrize('roots', [False, True])
def test_exact_substep_sequence_motor_scaling_and_live_isolation(monkeypatch, roots):
    item = fixture(monkeypatch, roots=roots)
    before = prediction._identity(item.live)
    predictor = prediction.prepare_action_predictor(item.env)
    action = np.linspace(-.1, .1, item.env.action_count)
    result = predictor.predict(action)
    assert result['status'] == 'predicted' and result['native_parity_claim'] is False
    assert result['native_substeps'] == 16 and result['boundary_forwards'] == result['copies'] == 1
    assert item.calls == ['allocate', 'copy'] + ['step', 'contacts'] * 16 + ['forward', 'contacts']
    full = np.zeros(22); full[:len(action)] = action
    expected = np.zeros(22); expected[item.engine._actuators] = full * item.engine.human_limits
    np.testing.assert_array_equal(result['endpoint']['ctrl'], expected)
    assert result['human_action'] == full.tolist()
    assert prediction._identity(item.live) == before
    assert all(row['contacts']['cached_time'] == row['solver_time_s'] for row in result['substep_contacts'])
    assert result['endpoint']['contacts']['cached_time'] == result['endpoint']['time_s']
    assert result['endpoint']['time_s'] == pytest.approx(1/30)


def test_repeat_prediction_recopies_full_warmstart_and_output_is_detached(monkeypatch):
    item = fixture(monkeypatch)
    predictor = prediction.prepare_action_predictor(item.env)
    first = predictor.predict(np.zeros(22)); second = predictor.predict(np.zeros(22))
    assert first['endpoint'] == second['endpoint'] and first['substep_contacts'] == second['substep_contacts']
    assert item.calls.count('allocate') == 1 and item.calls.count('copy') == 2
    assert any(second['endpoint']['qvel'])
    first['endpoint']['qpos'][0] = 999.
    assert second['endpoint']['qpos'][0] != 999. and item.states[0].qpos[0] != 999.
    assert item.live.qpos[0] == 0.


@pytest.mark.parametrize('case', ['callback', 'done', 'replaced', 'timing', 'motor_limits'])
def test_unsupported_state_does_not_advance_any_physics(monkeypatch, case):
    item = fixture(monkeypatch); predictor = prediction.prepare_action_predictor(item.env)
    if case == 'callback': monkeypatch.setattr(prediction.mujoco, 'get_mjcb_control', lambda: object())
    elif case == 'done': item.env._done = True
    elif case == 'replaced': item.engine.data = SimpleNamespace()
    elif case == 'timing': item.model.opt.timestep *= 2
    else: item.engine.human_limits[0] *= 2
    result = predictor.predict(np.zeros(22))
    assert result['status'] == 'unavailable' and result['reason']
    assert result['native_substeps'] == result['copies'] == 0 and not item.calls


@pytest.mark.parametrize('value', [[True] * 22, [2.] * 22, [float('nan')] * 22, [0.] * 19])
def test_invalid_actions_reject_without_allocating(monkeypatch, value):
    item = fixture(monkeypatch); predictor = prediction.prepare_action_predictor(item.env)
    with pytest.raises(ValueError, match='action'): predictor.predict(value)
    assert not item.calls and predictor.describe()['calls_used'] == 0


def test_prediction_call_budget_stops_before_copy_or_step(monkeypatch):
    item = fixture(monkeypatch); predictor = prediction.prepare_action_predictor(item.env, maximum_predictions=1)
    assert predictor.predict(np.zeros(22))['status'] == 'predicted'
    before = list(item.calls)
    result = predictor.predict(np.zeros(22))
    assert result['reason'] == 'prediction_call_budget' and item.calls == before
    assert predictor.describe()['calls_used'] == 1


def test_instability_preserves_partial_count_without_endpoint_or_extra_forward(monkeypatch):
    item = fixture(monkeypatch); original = prediction.mujoco.mj_step
    def unstable(model, data):
        original(model, data)
        if item.calls.count('step') == 3: data.qvel[0] = float('nan')
    monkeypatch.setattr(prediction.mujoco, 'mj_step', unstable)
    before = prediction._identity(item.live)
    result = prediction.prepare_action_predictor(item.env).predict(np.zeros(22))
    assert result['status'] == 'unavailable' and result['reason'] == 'predicted_instability:nonfinite_dynamics'
    assert result['native_substeps'] == 3 and result['boundary_forwards'] == 0 and result['endpoint'] is None
    assert prediction._identity(item.live) == before


def test_live_change_is_reported_not_overwritten(monkeypatch):
    item = fixture(monkeypatch); original = prediction.mujoco.mj_step
    def mutation(model, data):
        original(model, data); item.live.qpos[0] = .2
    monkeypatch.setattr(prediction.mujoco, 'mj_step', mutation)
    result = prediction.prepare_action_predictor(item.env).predict(np.zeros(22))
    assert result['reason'] == 'live_state_changed_during_prediction' and result['endpoint'] is None
    assert item.live.qpos[0] == .2


def test_callback_added_mid_prediction_stops_at_next_explicit_check(monkeypatch):
    item = fixture(monkeypatch); original = prediction.mujoco.mj_step
    def step(model, data):
        original(model, data)
        monkeypatch.setattr(prediction.mujoco, 'get_mjcb_control', lambda: object())
    monkeypatch.setattr(prediction.mujoco, 'mj_step', step)
    result = prediction.prepare_action_predictor(item.env).predict(np.zeros(22))
    assert result['native_substeps'] == 1 and result['reason'].startswith('unexpected_callback:')
    assert result['endpoint'] is None


def test_callback_added_at_boundary_cannot_publish_completed_endpoint(monkeypatch):
    item = fixture(monkeypatch); original = prediction.mujoco.mj_forward
    def forward(model, data):
        original(model, data)
        monkeypatch.setattr(prediction.mujoco, 'get_mjcb_control', lambda: object())
    monkeypatch.setattr(prediction.mujoco, 'mj_forward', forward)
    result = prediction.prepare_action_predictor(item.env).predict(np.zeros(22))
    assert result['status'] == 'unavailable' and result['reason'].startswith('unexpected_callback:')
    assert result['native_substeps'] == 16 and result['boundary_forwards'] == 1
    assert result['endpoint'] is None


def test_nonfinite_initial_state_and_missing_copy_abstain_before_step(monkeypatch):
    item = fixture(monkeypatch); item.live.qpos[0] = float('nan')
    result = prediction.prepare_action_predictor(item.env).predict(np.zeros(22))
    assert result['reason'] == 'initial_state:nonfinite_dynamics' and result['native_substeps'] == 0
    item.live.qpos[0] = .1
    monkeypatch.setattr(prediction.mujoco, 'mj_copyData', lambda *args: None)
    result = prediction.prepare_action_predictor(item.env).predict(np.zeros(22))
    assert result['reason'] == 'scratch_copy_identity_mismatch'


def test_time_budget_includes_final_encoding(monkeypatch):
    item = fixture(monkeypatch)
    clock = [0.]
    monkeypatch.setattr(prediction.time, 'monotonic', lambda: clock[0])
    original = prediction.json.dumps
    def encode(*args, **kwargs):
        clock[0] += .01
        return original(*args, **kwargs)
    monkeypatch.setattr(prediction.json, 'dumps', encode)
    result = prediction.prepare_action_predictor(item.env, maximum_seconds=.005).predict(np.zeros(22))
    assert result['status'] == 'unavailable' and result['reason'] == 'prediction_active_compute_budget'
    assert result['seconds'] == .01 and result['endpoint'] is None


def test_overlapping_calls_and_oversized_outputs_are_explicit(monkeypatch):
    item = fixture(monkeypatch); predictor = prediction.prepare_action_predictor(item.env)
    predictor._lock.acquire()
    try: assert predictor.predict(np.zeros(22))['reason'] == 'prediction_already_running'
    finally: predictor._lock.release()
    monkeypatch.setattr(prediction, 'MAX_RESULT_BYTES', 1500)
    result = predictor.predict(np.zeros(22))
    assert result['reason'] == 'prediction_result_byte_budget' and result['endpoint'] is None


def test_native_exception_preserves_attempted_and_completed_counts(monkeypatch):
    item = fixture(monkeypatch)
    def fail(model, data):
        data.time += model.opt.timestep
        raise RuntimeError('native step failed after starting')
    monkeypatch.setattr(prediction.mujoco, 'mj_step', fail)
    before = prediction._identity(item.live)
    result = prediction.prepare_action_predictor(item.env).predict(np.zeros(22))
    assert result['native_substeps_attempted'] == 1 and result['native_substeps'] == 0
    assert result['boundary_forwards_attempted'] == result['boundary_forwards'] == 0
    assert result['copies_attempted'] == result['copies'] == 1
    assert result['data_allocations_attempted'] == result['data_allocations'] == 1
    assert result['status'] == 'unavailable' and result['endpoint'] is None
    assert prediction._identity(item.live) == before


@pytest.mark.parametrize('change', ['profile', 'plugin', 'sensors', 'integrator', 'substeps'])
def test_unsupported_models_reject_before_native_allocation(monkeypatch, change):
    item = fixture(monkeypatch)
    if change == 'profile': item.env.scene['profile'] = 'shared-world-v5'
    elif change == 'plugin': item.model.nplugin = 1
    elif change == 'sensors': item.model.nsensor = 1
    elif change == 'integrator': item.model.opt.integrator = prediction.mujoco.mjtIntegrator.mjINT_EULER
    else: item.engine.substeps = 17
    with pytest.raises(ValueError): prediction.prepare_action_predictor(item.env)
    assert not item.calls
