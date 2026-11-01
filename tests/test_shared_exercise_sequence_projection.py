"""Two-action search using explicit fake forecasts; no native or scorer execution."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from test_shared_exercise_action_projection import fixture as measurement_fixture
from weft import shared_exercise_sequence_projection as projection
from weft.shared_exercise_action_prediction import STATE_FIELDS, _identity


def fixture(monkeypatch, *, roots=True, effect=None):
    item = measurement_fixture(monkeypatch, roots=roots)
    item.single_calls = item.calls
    item.calls, item.sequence_budgets = [], []
    item.second = np.linspace(-.15, .15, item.env.action_count)

    class Predictor:
        def predict(self, actions):
            first, second = (a.copy() for a in actions)
            item.calls.append((first, second))
            rows = []
            for index, action in enumerate((first, second)):
                start = item.live.time + index * item.engine.dt
                endpoint = {name: np.asarray(getattr(item.live, name)).tolist() for name in STATE_FIELDS}
                endpoint.update(time_s=start + item.engine.dt, contacts={'max_human_floor_penetration_model_units': 0.})
                endpoint['qvel'][22] = 0. if index == 0 else .3 + 2 * float(first[0])
                human = np.zeros(22); human[:len(action)] = action
                rows.append({'control': index + 1, 'initial_time_s': start, 'human_action': human.tolist(),
                    'complete': True, 'native_substeps': 16, 'native_substeps_attempted': 16,
                    'boundary_forwards': 1, 'boundary_forwards_attempted': 1,
                    'endpoint': endpoint, 'substep_contacts': [
                        {'substep': substep + 1, 'solver_time_s': start + substep * item.engine.dt / 16,
                         'integrated_time_s': start + (substep + 1) * item.engine.dt / 16,
                         'contacts': {'max_human_floor_penetration_model_units': 0.}}
                        for substep in range(16)]})
            result = {'status': 'predicted', 'reason': None, 'sequence_complete': True, 'controls': rows,
                'endpoint': rows[-1]['endpoint'], 'requested_controls': 2, 'controls_reserved': 2,
                'controls_attempted': 2, 'controls_completed': 2, 'native_substeps': 32,
                'native_substeps_attempted': 32, 'boundary_forwards': 2, 'boundary_forwards_attempted': 2,
                'copies': 1, 'copies_attempted': 1, 'data_allocations': int(len(item.calls) == 1),
                'data_allocations_attempted': int(len(item.calls) == 1)}
            if effect is not None:
                effect(first, second, result, len(item.calls))
            return result

    def prepare(env, **kwargs):
        assert env is item.env
        item.sequence_budgets.append(kwargs)
        return Predictor()
    monkeypatch.setattr(projection, 'prepare_action_sequence_predictor', prepare)
    return item


@pytest.mark.parametrize('first_dtype,second_dtype', [(np.float32, np.float64), (np.float64, np.float32)])
def test_future_boundary_drives_correction_with_exact_fixed_bytes_and_counted_repeat(monkeypatch, first_dtype, second_dtype):
    item = fixture(monkeypatch)
    base, fixed = item.base.astype(first_dtype), item.second.astype(second_dtype)
    before = _identity(item.live)
    projector = projection.prepare_action_sequence_projector(item.env)
    pair, report = projector.project(base, fixed)
    assert report['status'] == 'selected' and -.076 < pair[0][0] < -.074
    assert pair[0].dtype == base.dtype and pair[0][19:].tobytes() == base[19:].tobytes()
    assert pair[1].dtype == fixed.dtype and pair[1].tobytes() == fixed.tobytes()
    assert all(first[19:].tobytes() == base[19:].tobytes() and second.tobytes() == fixed.tobytes()
        for first, second in item.calls)
    assert len(item.calls) == report['sequence_calls'] == 24
    assert report['native_counts']['native_substeps'] == 768
    assert report['native_counts']['copies'] == 24 and report['native_counts']['boundary_forwards'] == 48
    assert report['native_counts']['data_allocations'] == 1 and report['native_counts_complete']
    assert report['forecast_control_counts'] == dict.fromkeys(('controls_reserved', 'controls_attempted', 'controls_completed'), 48)
    assert [row['joint'] for row in report['evaluations'] if row['kind'] == 'finite_difference'] == list(range(19))
    assert report['evaluations'][-1]['kind'] == 'selected_repeat'
    assert len(report['verified_sequence']['controls']) == 2 and len(report['constraints']) == 16
    assert report['measured_boundary_proposal_only'] and not report['task_metrics_forecast']
    assert all(row['sampling'] == 'integrated_control_endpoint' for row in report['constraints'] if 'speed' in row['name'])
    assert not item.single_calls and item.budgets == [{'maximum_predictions': 1, 'maximum_seconds': 5.}]
    assert item.sequence_budgets == [{'maximum_sequences': 24, 'maximum_controls': 48,
        'maximum_sequence_controls': 2, 'maximum_seconds': 5.}]
    assert _identity(item.live) == before and item.engine.steps == 48
    pair[0][0] = 1.; pair[1][0] = 1.
    assert base[0] == 0. and fixed[0] != 1.
    again, unavailable = projector.project(base, fixed)
    assert again is None and unavailable['reason'] == 'decision_lifetime_budget' and len(item.calls) == 24


def test_unchanged_pair_still_repeats_both_controls(monkeypatch):
    def effect(first, second, result, number):
        result['controls'][1]['endpoint']['qvel'][22] = 0.
    item = fixture(monkeypatch, effect=effect)
    pair, report = projection.prepare_action_sequence_projector(item.env).project(item.base, item.second)
    assert report['status'] == 'unchanged' and len(item.calls) == 2
    assert pair[0].tobytes() == item.base.tobytes() and pair[1].tobytes() == item.second.tobytes()
    assert [row['kind'] for row in report['evaluations']] == ['base', 'selected_repeat']


def test_two_reference_targets_are_frozen_once_and_measured_at_distinct_times(monkeypatch):
    seen = []
    def effect(first, second, result, number):
        for index, row in enumerate(result['controls']):
            row['endpoint']['qpos'][:22] = [.1 * (index + 1)] * 22
            row['endpoint']['qvel'][22] = 0.
    item = fixture(monkeypatch, effect=effect)
    def target(when):
        seen.append(when)
        return np.full(22, .1 * round((when - item.live.time) / item.engine.dt)), np.zeros(22)
    item.engine._target_at = target
    pair, report = projection.prepare_action_sequence_projector(item.env).project(item.base, item.second)
    assert pair is not None and seen == [item.live.time + item.engine.dt, item.live.time + 2 * item.engine.dt]
    for row in report['evaluations']:
        assert row['values'][4:6] == [0., 0.] and row['values'][12:14] == [0., 0.]
    assert [row['control'] for row in report['constraints']] == [1] * 8 + [2] * 8


@pytest.mark.parametrize('boundary', [0, 1])
def test_failure_at_either_boundary_is_not_discarded(monkeypatch, boundary):
    def effect(first, second, result, number):
        for row in result['controls']: row['endpoint']['qvel'][22] = 0.
        result['controls'][boundary]['endpoint']['qvel'][22] = .3
    item = fixture(monkeypatch, effect=effect)
    pair, report = projection.prepare_action_sequence_projector(item.env).project(item.base, item.second)
    assert pair is None and report['status'] == 'no_verified_candidate'
    assert report['reason'] == 'linearized_row_box_conflict' and len(item.calls) == 20
    assert all(not row['passes_measured_constraints'] for row in report['evaluations'])


def test_passing_difference_is_kept_when_linear_solver_has_no_proposal(monkeypatch):
    def effect(first, second, result, number):
        result['controls'][1]['endpoint']['qvel'][22] = .3 - 2 * first[0]
    item = fixture(monkeypatch, effect=effect)
    monkeypatch.setattr(projection, '_linear_projection', lambda *args: (None, {'status': 'row_box_conflict', 'sweeps': 0}))
    pair, report = projection.prepare_action_sequence_projector(item.env).project(item.base, item.second)
    assert report['status'] == 'selected' and pair[0][0] == .1
    assert len(item.calls) == 21 and report['linear_model']['status'] == 'row_box_conflict'
    assert report['evaluations'][-1]['kind'] == 'selected_repeat'


@pytest.mark.parametrize('boundary,field', [(0, 'endpoint'), (1, 'endpoint'), (0, 'substep_contacts'), (1, 'substep_contacts')])
def test_repeat_compares_full_state_and_substep_contacts_at_both_boundaries(monkeypatch, boundary, field):
    def effect(first, second, result, number):
        if number == 24:
            if field == 'endpoint': result['controls'][boundary]['endpoint']['ctrl'][0] = .12
            else: result['controls'][boundary]['substep_contacts'][0]['contacts']['extra'] = 'changed'
    item = fixture(monkeypatch, effect=effect)
    pair, report = projection.prepare_action_sequence_projector(item.env).project(item.base, item.second)
    assert pair is None and report['reason'] == 'selected_sequence_not_repeatable'
    assert report['verified_sequence'] is None and len(item.calls) == 24


def test_partial_failed_forecast_keeps_both_counters_and_available_control_evidence(monkeypatch):
    def effect(first, second, result, number):
        result.update(status='unavailable', reason='second-control native failure', sequence_complete=False,
            endpoint=None, controls_completed=1, native_substeps=17, native_substeps_attempted=18,
            boundary_forwards=1, boundary_forwards_attempted=1)
        result['controls'][1].update(complete=False, endpoint=None, substep_contacts=[], native_substeps=1,
            native_substeps_attempted=2, boundary_forwards=0, boundary_forwards_attempted=0)
    item = fixture(monkeypatch, effect=effect)
    pair, report = projection.prepare_action_sequence_projector(item.env).project(item.base, item.second)
    assert pair is None and report['status'] == 'unavailable' and len(item.calls) == 1
    assert report['failed_sequence']['controls'][0]['endpoint']
    assert report['failed_sequence']['controls'][1]['endpoint'] is None
    assert report['native_counts']['native_substeps'] == 17 and report['native_counts']['native_substeps_attempted'] == 18
    assert report['forecast_control_counts']['controls_reserved'] == 2 and report['forecast_control_counts']['controls_completed'] == 1
    assert report['failed_evaluation']['prediction_attempted'] is True


def test_unreported_predictor_exception_marks_native_counts_unknown(monkeypatch):
    item = fixture(monkeypatch)
    projector = projection.prepare_action_sequence_projector(item.env)
    def fail(actions): raise RuntimeError('outside sequence report')
    monkeypatch.setattr(projector._predictor, 'predict', fail)
    pair, report = projector.project(item.base, item.second)
    assert pair is None and not report['native_counts_complete'] and report['sequence_calls'] == 1
    assert report['reason'] == 'sequence_predictor_raised:RuntimeError'
    assert report['failed_evaluation']['kind'] == 'base'


@pytest.mark.parametrize('damage', ['missing_count', 'partial_reserved', 'omitted', 'row_time'])
def test_incomplete_forecast_protocol_cannot_be_accepted(monkeypatch, damage):
    def effect(first, second, result, number):
        if damage == 'missing_count': result.pop('copies')
        elif damage == 'partial_reserved': result['controls_reserved'] = 1
        elif damage == 'omitted': result['omitted_detail'] = {'endpoints': 1}
        else: result['controls'][1]['initial_time_s'] += .01
    item = fixture(monkeypatch, effect=effect)
    pair, report = projection.prepare_action_sequence_projector(item.env).project(item.base, item.second)
    assert pair is None and report['status'] == 'unavailable' and len(item.calls) == 1
    assert report['failed_sequence'] and report['failed_evaluation']['kind'] == 'base'


@pytest.mark.parametrize('setting', [dict(maximum_sequences=23), dict(maximum_controls=47)])
def test_full_search_and_repeat_reservation_rejects_before_finite_differences(monkeypatch, setting):
    item = fixture(monkeypatch)
    config = projection.SequenceProjectionConfig(**setting)
    pair, report = projection.prepare_action_sequence_projector(item.env, config=config).project(item.base, item.second)
    assert pair is None and report['reason'] == 'insufficient_budget_for_complete_sequence_search'
    assert len(item.calls) == 1 and report['forecast_control_counts']['controls_reserved'] == 2


def test_base_pass_cannot_skip_repeat_when_budget_is_exhausted(monkeypatch):
    def effect(first, second, result, number): result['controls'][1]['endpoint']['qvel'][22] = 0.
    item = fixture(monkeypatch, effect=effect)
    config = projection.SequenceProjectionConfig(maximum_sequences=1, maximum_controls=2)
    pair, report = projection.prepare_action_sequence_projector(item.env, config=config).project(item.base, item.second)
    assert pair is None and report['reason'] == 'sequence_projection_prediction_budget' and len(item.calls) == 1


def test_solver_sweeps_share_one_counted_budget(monkeypatch):
    item = fixture(monkeypatch)
    def solver(matrix, bound, lower, upper, maximum, check):
        for _ in range(maximum + 1): check()
        pytest.fail('Exceeded solver must abstain')
    monkeypatch.setattr(projection, '_linear_projection', solver)
    config = projection.SequenceProjectionConfig(maximum_sweeps=3)
    pair, report = projection.prepare_action_sequence_projector(item.env, config=config).project(item.base, item.second)
    assert pair is None and report['reason'] == 'sequence_projection_solver_budget'
    assert report['solver_sweeps'] == 3 and len(item.calls) == 20


def test_one_deadline_includes_measurement_and_final_encoding(monkeypatch):
    item = fixture(monkeypatch); clock = [0.]
    monkeypatch.setattr(projection.time, 'monotonic', lambda: clock[0])
    original = projection.json.dumps
    def encode(value, *args, **kwargs):
        if isinstance(value, dict) and 'evaluations' in value: clock[0] = 6.
        return original(value, *args, **kwargs)
    monkeypatch.setattr(projection.json, 'dumps', encode)
    pair, report = projection.prepare_action_sequence_projector(item.env).project(item.base, item.second)
    assert pair is None and report['reason'] == 'sequence_projection_active_budget'
    assert report['active_seconds'] == 6. and report['verified_sequence'] is None


@pytest.mark.parametrize('change', ['live', 'accumulator', 'engine'])
def test_live_or_task_contract_changes_are_not_restored_or_accepted(monkeypatch, change):
    def effect(first, second, result, number):
        if change == 'live': item.live.qpos[0] = .3
        elif change == 'accumulator': item.env._accumulator = object()
        else: item.env._engine = object()
    item = fixture(monkeypatch, effect=effect)
    pair, report = projection.prepare_action_sequence_projector(item.env).project(item.base, item.second)
    assert pair is None and report['status'] == 'unavailable' and len(item.calls) == 1
    if change == 'live': assert item.live.qpos[0] == .3


def test_caller_mutation_cannot_change_copied_fixed_second(monkeypatch):
    def effect(first, second, result, number): item.second[:] = .8
    item = fixture(monkeypatch, effect=effect); before = item.second.copy()
    pair, report = projection.prepare_action_sequence_projector(item.env).project(item.base, item.second)
    assert pair is not None and pair[1].tobytes() == before.tobytes()
    assert all(second.tobytes() == before.tobytes() for first, second in item.calls)


def test_near_cap_proof_abstains_without_losing_measured_evaluations(monkeypatch):
    def effect(first, second, result, number):
        for row in result['controls']: row['endpoint']['proof_padding'] = 'x' * 14000
    item = fixture(monkeypatch, effect=effect)
    monkeypatch.setattr(projection, 'MAX_RESULT_BYTES', 48 << 10)
    pair, report = projection.prepare_action_sequence_projector(item.env).project(item.base, item.second)
    assert pair is None and report['reason'] == 'sequence_projection_result_byte_budget'
    assert report['omitted_verified_controls'] == 2 and report['verified_sequence'] is None
    assert len(report['evaluations']) == 24 and report['native_counts']['native_substeps'] == 768
    assert len(json.dumps(report, allow_nan=False).encode()) < projection.MAX_RESULT_BYTES


def test_oversized_failed_sequence_reports_explicit_omission_without_acceptance(monkeypatch):
    def effect(first, second, result, number):
        result.update(status='unavailable', reason='failed after starting', sequence_complete=False)
        result['controls'][0]['endpoint']['padding'] = 'x' * 65536
    item = fixture(monkeypatch, effect=effect)
    monkeypatch.setattr(projection, 'MAX_RESULT_BYTES', 32 << 10)
    pair, report = projection.prepare_action_sequence_projector(item.env).project(item.base, item.second)
    assert pair is None and report['status'] == 'unavailable'
    assert 'failed_sequence' not in report and report['omitted_failed_sequence']['controls'] == 2
    assert report['failed_evaluation']['kind'] == 'base' and report['native_counts']['copies'] == 1
    assert len(json.dumps(report, allow_nan=False).encode()) < projection.MAX_RESULT_BYTES


def test_invalid_actions_and_locked_calls_never_predict(monkeypatch):
    item = fixture(monkeypatch)
    for bad in ([True] * 22, [0.] * 19, np.zeros(22, dtype=np.int64), np.full(22, np.nan)):
        pair, report = projection.prepare_action_sequence_projector(item.env).project(item.base, bad)
        assert pair is None and report['status'] == 'unavailable'
    projector = projection.prepare_action_sequence_projector(item.env)
    projector._lock.acquire()
    try:
        pair, report = projector.project(item.base, item.second)
    finally: projector._lock.release()
    assert pair is None and report['reason'] == 'sequence_projection_already_running' and not item.calls


@pytest.mark.parametrize('kwargs', [dict(maximum_sequences=25), dict(maximum_controls=49), dict(maximum_sweeps=65),
    dict(maximum_seconds=6), dict(maximum_sequences=True), dict(difference=.025), dict(trust_radius=.3)])
def test_configuration_is_bounded_and_has_no_scale_sweep_setting(kwargs):
    with pytest.raises(ValueError): projection.SequenceProjectionConfig(**kwargs)


def test_immutable_configuration_and_nineteen_joint_contract(monkeypatch):
    config = projection.SequenceProjectionConfig()
    with pytest.raises(FrozenInstanceError): config.maximum_sequences = 25
    item = fixture(monkeypatch, roots=False)
    pair, report = projection.prepare_action_sequence_projector(item.env, config=config).project(item.base, item.second)
    assert pair[0].shape == pair[1].shape == (19,) and report['status'] == 'selected'
