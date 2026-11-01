"""Fixed-action continuation and failure accounting with mocked native operations."""

import json

import numpy as np
import pytest

from test_shared_exercise_action_prediction import fixture as single_fixture
from weft import shared_exercise_action_prediction as single
from weft import shared_exercise_sequence_prediction as sequence


def fixture(monkeypatch, **kwargs):
    item = single_fixture(monkeypatch, **kwargs)
    item.engine.horizon_steps = item.env.horizon_steps
    return item


def prepare(item, **kwargs):
    return sequence.prepare_action_sequence_predictor(item.env, **kwargs)


@pytest.mark.parametrize('roots', [False, True])
def test_continuation_keeps_full_boundary_state_and_matches_first_single_prediction(monkeypatch, roots):
    item = fixture(monkeypatch, roots=roots)
    before = single._identity(item.live)
    item.env._accumulator = {'sentinel': [1, 2, 3]}
    actions = [np.linspace(-.1, .1, item.env.action_count), np.zeros(item.env.action_count)]
    predictor = prepare(item)
    report = predictor.predict(actions)
    assert report['status'] == 'predicted' and report['sequence_complete']
    assert not report['native_parity_claim'] and report['live_state_unchanged']
    assert report['copies'] == report['data_allocations'] == 1
    assert report['native_substeps'] == report['native_substeps_attempted'] == 32
    assert report['boundary_forwards'] == report['boundary_forwards_attempted'] == 2
    assert report['controls_reserved'] == report['controls_attempted'] == report['controls_completed'] == 2
    assert item.calls == ['allocate', 'copy'] + (['step', 'contacts'] * 16 + ['forward', 'contacts']) * 2
    first, second = report['controls']
    assert first['complete'] and second['complete']
    assert second['initial_time_s'] == first['endpoint']['time_s']
    assert second['substep_contacts'][0]['solver_time_s'] == first['endpoint']['time_s']
    assert report['endpoint'] == second['endpoint']
    assert set(report['endpoint']) == {*single.STATE_FIELDS, 'time_s', 'contacts'}
    assert report['endpoint']['time_s'] == pytest.approx(2/30)
    assert first['endpoint']['qvel'] != second['endpoint']['qvel']
    assert any(second['endpoint']['qacc_warmstart'])
    for row in report['controls']:
        assert row['endpoint']['contacts']['cached_time'] == row['endpoint']['time_s']
        assert all(c['contacts']['cached_time'] == c['solver_time_s'] for c in row['substep_contacts'])
        if not roots:
            assert row['human_action'][19:] == [0., 0., 0.]
    original = single.prepare_action_predictor(item.env).predict(actions[0])
    assert first['endpoint'] == original['endpoint'] and first['substep_contacts'] == original['substep_contacts']
    assert single._identity(item.live) == before and item.engine.steps == 0
    assert item.env._accumulator == {'sentinel': [1, 2, 3]}
    assert predictor.describe()['maximum_native_substeps'] == 128


def test_inputs_are_copied_before_physics_and_outputs_detach_from_reused_scratch(monkeypatch):
    item = fixture(monkeypatch)
    actions = np.zeros((2, 22), dtype=np.float32)
    original = sequence.mujoco.mj_step
    def mutate_input(model, data):
        actions[1, 0] = .9
        original(model, data)
    monkeypatch.setattr(sequence.mujoco, 'mj_step', mutate_input)
    predictor = prepare(item)
    first = predictor.predict(actions)
    assert first['actions'][1][0] == first['controls'][1]['human_action'][0] == 0.
    actions[:] = 0.
    second = predictor.predict(actions)
    assert second['endpoint'] == first['endpoint']
    assert item.calls.count('allocate') == 1 and item.calls.count('copy') == 2
    first['endpoint']['qpos'][0] = 999.
    first['controls'][0]['substep_contacts'][0]['contacts']['details'].clear()
    assert second['endpoint']['qpos'][0] != 999. and item.states[0].qpos[0] != 999.
    assert second['controls'][0]['substep_contacts'][0]['contacts']['details']


@pytest.mark.parametrize('actions', [[], [[0.] * 22] * 5, [0.] * 22, [[0.] * 19],
    [[True] * 22], [[float('nan')] * 22], [[float('inf')] * 22], [[2.] * 22],
    [[complex(0, 1)] * 22], np.zeros((1, 22), dtype=bool), np.zeros((1, 22), dtype=object),
    [np.full(22, np.iinfo(np.int64).min, dtype=np.int64)], [np.array(0.)]])
def test_invalid_fixed_actions_reject_before_any_native_work(monkeypatch, actions):
    item = fixture(monkeypatch); predictor = prepare(item)
    with pytest.raises(ValueError):
        predictor.predict(actions)
    assert not item.calls and predictor.describe()['sequences_used'] == predictor.describe()['controls_reserved'] == 0


def test_no_generators_callbacks_or_nested_coercion(monkeypatch):
    item = fixture(monkeypatch); predictor = prepare(item)
    def callback():
        pytest.fail('No callback can run')
    def generator():
        pytest.fail('No generator can be consumed')
        yield np.zeros(22)
    class Coercion:
        def __array__(self, *args, **kwargs):
            pytest.fail('Nested input must reject before NumPy coercion')
    for actions in (callback, generator(), [[Coercion()] * 22], [[[]] * 22]):
        with pytest.raises(ValueError): predictor.predict(actions)
    assert not item.calls


def test_late_invalid_row_rejects_entire_request_before_allocation(monkeypatch):
    item = fixture(monkeypatch); predictor = prepare(item)
    with pytest.raises(ValueError): predictor.predict([np.zeros(22), [True] * 22])
    assert not item.calls and predictor.describe()['controls_reserved'] == 0


@pytest.mark.parametrize('kwargs', [dict(maximum_sequences=0), dict(maximum_sequences=65),
    dict(maximum_sequences=True), dict(maximum_controls=65), dict(maximum_controls=False),
    dict(maximum_sequence_controls=0), dict(maximum_sequence_controls=5),
    dict(maximum_seconds=0), dict(maximum_seconds=float('nan')), dict(maximum_seconds=11)])
def test_configuration_bounds_reject_without_native_allocation(monkeypatch, kwargs):
    item = fixture(monkeypatch)
    with pytest.raises(ValueError): prepare(item, **kwargs)
    assert not item.calls


def test_frozen_model_validator_is_reused(monkeypatch):
    item = fixture(monkeypatch); item.model.nplugin = 1
    with pytest.raises(ValueError, match='rigid-motor envelope'): prepare(item)
    assert not item.calls


def test_sequence_and_aggregate_control_budgets_reject_without_partial_admission(monkeypatch):
    item = fixture(monkeypatch); predictor = prepare(item, maximum_sequences=2, maximum_controls=3)
    assert predictor.predict(np.zeros((2, 22)))['status'] == 'predicted'
    before = list(item.calls)
    rejected = predictor.predict(np.zeros((2, 22)))
    assert rejected['reason'] == 'sequence_control_budget' and not rejected['controls']
    assert rejected['controls_reserved'] == 0 and item.calls == before
    assert predictor.predict(np.zeros((1, 22)))['status'] == 'predicted'
    before = list(item.calls)
    assert predictor.predict(np.zeros((1, 22)))['reason'] == 'sequence_call_budget'
    assert predictor.describe()['controls_reserved'] == 3 and item.calls == before


def test_request_longer_than_remaining_live_horizon_is_not_shortened(monkeypatch):
    item = fixture(monkeypatch); item.engine.steps = item.env.horizon_steps - 1
    result = prepare(item).predict(np.zeros((2, 22)))
    assert result['reason'] == 'insufficient_live_control_horizon'
    assert result['requested_controls'] == 2 and result['controls_reserved'] == 0
    assert not result['sequence_complete'] and result['endpoint'] is None and not item.calls


def test_shorter_underlying_engine_horizon_rejects_entire_sequence(monkeypatch):
    item = fixture(monkeypatch); item.engine.horizon_steps = 1
    result = prepare(item).predict(np.zeros((2, 22)))
    assert result['reason'] == 'insufficient_live_control_horizon'
    assert result['controls_reserved'] == 0 and not item.calls


def test_partial_native_failure_preserves_prefix_and_nonrefundable_reservation(monkeypatch):
    item = fixture(monkeypatch); predictor = prepare(item, maximum_controls=2)
    original = sequence.mujoco.mj_step
    def fail(model, data):
        original(model, data)
        if item.calls.count('step') == 19:
            raise RuntimeError('native step failed after advancing scratch')
    monkeypatch.setattr(sequence.mujoco, 'mj_step', fail)
    before = single._identity(item.live)
    result = predictor.predict(np.zeros((2, 22)))
    assert result['status'] == 'unavailable' and not result['sequence_complete'] and result['endpoint'] is None
    assert result['controls_reserved'] == result['controls_attempted'] == 2 and result['controls_completed'] == 1
    assert result['native_substeps_attempted'] == 19 and result['native_substeps'] == 18
    assert result['boundary_forwards_attempted'] == result['boundary_forwards'] == 1
    first, second = result['controls']
    assert first['complete'] and first['endpoint'] and len(first['substep_contacts']) == 16
    assert not second['complete'] and second['endpoint'] is None
    assert second['native_substeps_attempted'] == 3 and second['native_substeps'] == 2
    assert len(second['substep_contacts']) == 2
    assert result['live_state_unchanged'] and single._identity(item.live) == before
    calls = list(item.calls)
    assert predictor.predict(np.zeros((1, 22)))['reason'] == 'sequence_control_budget'
    assert item.calls == calls


@pytest.mark.parametrize('operation,counter', [('MjData', 'data_allocations'), ('mj_copyData', 'copies'),
    ('mj_forward', 'boundary_forwards')])
def test_native_exceptions_keep_attempted_and_returned_counts(monkeypatch, operation, counter):
    item = fixture(monkeypatch)
    def fail(*args): raise OSError('native operation failed')
    monkeypatch.setattr(sequence.mujoco, operation, fail)
    result = prepare(item).predict(np.zeros((2, 22)))
    assert result['status'] == 'unavailable' and result['failure_type'] == 'OSError'
    assert result[counter + '_attempted'] == 1 and result[counter] == 0
    assert result['controls_reserved'] == 2 and result['endpoint'] is None


@pytest.mark.parametrize('change', ['callback', 'engine', 'model', 'data', 'timing', 'limits', 'actuators', 'steps', 'done', 'horizon'])
def test_mid_sequence_contract_changes_abstain_before_next_control(monkeypatch, change):
    item = fixture(monkeypatch); predictor = prepare(item)
    original = sequence.mujoco.mj_forward
    def forward(model, data):
        original(model, data)
        if change == 'callback': monkeypatch.setattr(sequence.mujoco, 'get_mjcb_control', lambda: object())
        elif change == 'engine': item.env._engine = object()
        elif change == 'model': item.engine.model = object()
        elif change == 'data': item.engine.data = object()
        elif change == 'timing': item.model.opt.timestep *= 2
        elif change == 'limits': item.engine.human_limits[0] *= 2
        elif change == 'actuators': item.engine._actuators[:] = 0
        elif change == 'steps': item.engine.steps += 1
        elif change == 'horizon': item.engine.horizon_steps = 1
        else: item.env._done = True
    monkeypatch.setattr(sequence.mujoco, 'mj_forward', forward)
    result = predictor.predict(np.zeros((2, 22)))
    assert result['status'] == 'unavailable' and result['reason']
    assert result['native_substeps'] == 16 and result['boundary_forwards'] == 1
    assert result['controls_attempted'] == 1 and result['controls_completed'] == 0
    assert not result['sequence_complete'] and result['endpoint'] is None


def test_timeout_retains_partial_work_and_consumes_lifetime_budget(monkeypatch):
    item = fixture(monkeypatch); clock = [0.]
    monkeypatch.setattr(sequence.time, 'monotonic', lambda: clock[0])
    predictor = prepare(item, maximum_seconds=1.)
    original = sequence.mujoco.mj_step
    def step(model, data):
        original(model, data)
        if item.calls.count('step') == 17: clock[0] = 2.
    monkeypatch.setattr(sequence.mujoco, 'mj_step', step)
    result = predictor.predict(np.zeros((2, 22)))
    assert result['reason'] == 'sequence_active_compute_budget' and result['native_substeps'] == 17
    assert result['controls_completed'] == 1 and result['controls_reserved'] == 2
    assert result['seconds'] == predictor.describe()['active_compute_seconds_used'] == 2.
    before = list(item.calls)
    assert predictor.predict(np.zeros((1, 22)))['reason'] == 'sequence_active_compute_budget'
    assert item.calls == before


def test_final_encoding_time_and_live_mutation_are_checked(monkeypatch):
    item = fixture(monkeypatch); clock = [0.]
    monkeypatch.setattr(sequence.time, 'monotonic', lambda: clock[0])
    original = sequence.json.dumps
    def encode(value, *args, **kwargs):
        if isinstance(value, dict) and 'profile' in value:
            clock[0] += 2.
            item.live.qfrc_applied[0] = 5.
        return original(value, *args, **kwargs)
    monkeypatch.setattr(sequence.json, 'dumps', encode)
    result = prepare(item, maximum_seconds=1.).predict(np.zeros((1, 22)))
    assert result['status'] == 'unavailable' and not result['sequence_complete'] and result['endpoint'] is None
    assert result['live_state_unchanged'] is False and result['seconds'] == 2.
    assert item.live.qfrc_applied[0] == 5.


def test_full_live_state_change_is_reported_without_restoring_external_work(monkeypatch):
    item = fixture(monkeypatch); original = sequence.mujoco.mj_step
    def step(model, data):
        original(model, data); item.live.qacc_warmstart[0] = .75
    monkeypatch.setattr(sequence.mujoco, 'mj_step', step)
    result = prepare(item).predict(np.zeros((2, 22)))
    assert result['reason'] == 'live_state_changed_during_sequence' and result['live_state_unchanged'] is False
    assert item.live.qacc_warmstart[0] == .75 and result['endpoint'] is None


def test_aggregate_report_cap_retains_explicit_omission_counts_and_stops(monkeypatch):
    item = fixture(monkeypatch)
    item.engine._current_contacts = lambda *, data: {'payload': 'x' * 4500, 'cached_time': data.cached_time}
    monkeypatch.setattr(sequence, 'MAX_RESULT_BYTES', 128 << 10)
    result = prepare(item).predict(np.zeros((3, 22)))
    assert result['status'] == 'unavailable' and result['reason'] == 'sequence_result_byte_budget'
    assert result['controls_reserved'] == 3 and result['controls_completed'] == 2
    assert result['native_substeps'] == 32 and result['boundary_forwards'] == 2
    assert result['omitted_detail']['endpoints'] == 2 and result['omitted_detail']['substep_contacts'] == 32
    assert result['omitted_detail']['pre_omission_sha256']
    assert all(row['endpoint'] is None and row['substep_contacts'] == [] for row in result['controls'])
    assert len(json.dumps(result, allow_nan=False).encode()) < sequence.MAX_RESULT_BYTES


def test_nonfinite_contact_output_abstains_with_bounded_failed_detail_record(monkeypatch):
    item = fixture(monkeypatch)
    item.engine._current_contacts = lambda *, data: {'nonfinite': float('nan')}
    result = prepare(item).predict(np.zeros((2, 22)))
    assert result['status'] == 'unavailable' and result['endpoint'] is None
    assert result['omitted_detail']['substep_contacts'] == 16
    assert result['omitted_detail']['pre_omission_sha256'] is None
    json.dumps(result, allow_nan=False)


def test_overlapping_prediction_does_not_allocate_or_reserve(monkeypatch):
    item = fixture(monkeypatch); predictor = prepare(item)
    predictor._lock.acquire()
    try:
        result = predictor.predict(np.zeros((1, 22)))
    finally:
        predictor._lock.release()
    assert result['reason'] == 'sequence_prediction_already_running' and result['endpoint'] is None
    assert predictor.describe()['sequences_used'] == 0 and not item.calls


@pytest.mark.parametrize('case', ['alias', 'copy', 'nonfinite'])
def test_scratch_identity_and_initial_state_guards_stop_before_integration(monkeypatch, case):
    item = fixture(monkeypatch)
    if case == 'alias': monkeypatch.setattr(sequence.mujoco, 'MjData', lambda model: item.live)
    elif case == 'copy':
        item.live.qpos[0] = .1
        monkeypatch.setattr(sequence.mujoco, 'mj_copyData', lambda *args: None)
    else: item.live.qpos[0] = float('nan')
    result = prepare(item).predict(np.zeros((2, 22)))
    assert result['status'] == 'unavailable' and result['native_substeps_attempted'] == 0
    assert result['controls_reserved'] == 2 and result['endpoint'] is None


def test_hard_aggregate_control_limit_is_exact_and_scratch_is_reused(monkeypatch):
    item = fixture(monkeypatch)
    predictor = prepare(item, maximum_sequences=64, maximum_controls=64, maximum_seconds=10.)
    for _ in range(16):
        assert predictor.predict(np.zeros((4, 22)))['status'] == 'predicted'
    before = list(item.calls)
    assert predictor.predict(np.zeros((1, 22)))['reason'] == 'sequence_control_budget'
    assert predictor.describe()['maximum_native_substeps'] == 1024
    assert predictor.describe()['controls_reserved'] == 64 and item.calls == before
    assert item.calls.count('copy') == 16 and item.calls.count('allocate') == 1
