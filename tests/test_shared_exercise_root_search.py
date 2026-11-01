"""Root polling against explicit fake nonlinear responses; no native or scorer execution."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from test_shared_exercise_action_projection import fixture as measurement_fixture
from weft import shared_exercise_root_search as search
from weft.shared_exercise_action_prediction import STATE_FIELDS, _identity
from weft.spatial_pose import SPATIAL_JOINT_NAMES


def fixture(monkeypatch, *, effect=None, response=None):
    item = measurement_fixture(monkeypatch)
    item.single_calls = item.calls
    item.calls, item.sequence_budgets = [], []
    item.base = np.zeros(22)
    item.second = np.zeros(22)
    item.base[:19] = np.linspace(-.1, .1, 19)
    item.second[:19] = np.linspace(.2, -.2, 19)
    item.env.action_names = (*SPATIAL_JOINT_NAMES, 'root_x', 'root_y', 'root_z')
    item.env.task['control']['root_assistance'] = 'explicit_3_axis'
    item.env.horizon_steps = item.engine.horizon_steps = 254
    item.metrics = {'completed_interval': False, 'failure': None, 'observed_substeps': 768}
    item.env.metrics = lambda: deepcopy(item.metrics)

    class Predictor:
        def predict(self, actions):
            actions = tuple(action.copy() for action in actions)
            item.calls.append(actions)
            rows = []
            for index, action in enumerate(actions):
                start = item.live.time + index * item.engine.dt
                endpoint = {name: np.asarray(getattr(item.live, name)).tolist() for name in STATE_FIELDS}
                endpoint.update(time_s=start + item.engine.dt, contacts={'max_human_floor_penetration_model_units': 0.})
                endpoint['qvel'][22] = (response(actions, index) if response else
                    0. if index == 0 else .4 + .6 * actions[0][19] + .6 * actions[1][20])
                rows.append({'control': index + 1, 'initial_time_s': start, 'human_action': action.astype(float).tolist(),
                    'complete': True, 'native_substeps': 16, 'native_substeps_attempted': 16,
                    'boundary_forwards': 1, 'boundary_forwards_attempted': 1,
                    'endpoint': endpoint, 'substep_contacts': [
                        {'substep': substep + 1, 'solver_time_s': start + substep * item.engine.dt / 16,
                         'integrated_time_s': start + (substep + 1) * item.engine.dt / 16,
                         'contacts': {'max_human_floor_penetration_model_units': 0.}}
                        for substep in range(16)]})
            count = len(actions)
            result = {'status': 'predicted', 'reason': None, 'sequence_complete': True, 'controls': rows,
                'endpoint': rows[-1]['endpoint'], 'requested_controls': count, 'controls_reserved': count,
                'controls_attempted': count, 'controls_completed': count, 'native_substeps': 16 * count,
                'native_substeps_attempted': 16 * count, 'boundary_forwards': count, 'boundary_forwards_attempted': count,
                'copies': 1, 'copies_attempted': 1, 'data_allocations': int(len(item.calls) == 1),
                'data_allocations_attempted': int(len(item.calls) == 1)}
            if effect:
                effect(actions, result, len(item.calls))
            return result

    def prepare(env, **kwargs):
        assert env is item.env
        item.sequence_budgets.append(kwargs)
        return Predictor()
    monkeypatch.setattr(search, 'prepare_action_sequence_predictor', prepare)
    return item


@pytest.mark.parametrize('first_dtype,second_dtype', [(np.float32, np.float64), (np.float64, np.float32)])
def test_measured_two_round_correction_changes_both_roots_not_any_joint(monkeypatch, first_dtype, second_dtype):
    item = fixture(monkeypatch)
    bases = (item.base.astype(first_dtype), item.second.astype(second_dtype))
    before = _identity(item.live)
    controller = search.prepare_root_assistance_search(item.env)
    actions, report = controller.search(bases)
    assert report['status'] == 'selected' and actions[0][19] == -.25 and actions[1][20] == -.25
    assert len(item.calls) == report['sequence_calls'] == 24
    assert [row['completed_candidates'] for row in report['rounds']] == [12, 10]
    assert all(row['strictly_improved_measured_violation'] for row in report['rounds'])
    assert report['evaluations'][-1]['kind'] == 'selected_repeat'
    for attempted in item.calls:
        for action, base in zip(attempted, bases):
            assert action.dtype == base.dtype and action[:19].tobytes() == base[:19].tobytes()
            assert np.all(abs(action[19:].astype(float) - base[19:]) <= .25) and np.all(abs(action) <= 1)
    counts = report['native_counts']
    assert counts['native_substeps'] == 768 and counts['copies'] == 24 and counts['boundary_forwards'] == 48
    assert counts['data_allocations'] == 1 and report['native_counts_complete']
    assert report['requested_forecast_controls_reserved'] == 48
    assert report['forecast_control_counts'] == dict.fromkeys(('controls_reserved', 'controls_attempted', 'controls_completed'), 48)
    assert report['live_state_unchanged'] and report['task_metrics_unchanged'] and _identity(item.live) == before
    assert report['measured_boundary_proposal_only'] and not report['task_metrics_forecast']
    assert not item.single_calls and item.budgets == [{'maximum_predictions': 1, 'maximum_seconds': 5.}]
    assert item.sequence_budgets == [{'maximum_sequences': 26, 'maximum_controls': 52,
        'maximum_sequence_controls': 2, 'maximum_seconds': 5.}]
    actions[0][0] = 1.; actions[1][0] = 1.
    assert bases[0][0] == -.1 and bases[1][0] != 1.
    again, stopped = controller.search(bases)
    assert again is None and stopped['reason'] == 'decision_lifetime_budget' and len(item.calls) == 24


def test_discontinuous_nonlinear_response_needs_no_affine_fit(monkeypatch):
    def response(actions, index):
        if index == 0: return 0.
        first, second = actions[0][19] < 0, actions[1][20] > 0
        return .1 if first and second else .25 if first or second else .4
    item = fixture(monkeypatch, response=response)
    actions, report = search.prepare_root_assistance_search(item.env).search((item.base, item.second))
    assert report['status'] == 'selected' and actions[0][19] == -.25 and actions[1][20] == .25
    assert 'linear_model' not in report and 'solver_sweeps' not in report
    assert all(min(row['values']) >= 0 for row in report['evaluations'])


def test_unchanged_base_still_repeats_complete_sequence(monkeypatch):
    item = fixture(monkeypatch, response=lambda actions, index: .1)
    actions, report = search.prepare_root_assistance_search(item.env).search((item.base, item.second))
    assert report['status'] == 'unchanged' and len(item.calls) == 2 and report['rounds'] == []
    assert [row['kind'] for row in report['evaluations']] == ['base', 'selected_repeat']
    assert actions[0].tobytes() == item.base.tobytes() and actions[1].tobytes() == item.second.tobytes()


def test_final_control_uses_explicit_one_action_without_shortening(monkeypatch):
    item = fixture(monkeypatch, response=lambda actions, index: .3 + .8 * actions[0][19])
    item.env.horizon_steps = item.engine.horizon_steps = 49
    actions, report = search.prepare_root_assistance_search(item.env).search([item.base])
    assert report['status'] == 'selected' and len(actions) == 1 and actions[0][19] == -.25
    assert all(len(actions) == 1 for actions in item.calls) and report['supplied_controls'] == 1
    assert len(report['verified_sequence']['controls']) == 1 and len(report['constraints']) == 8
    assert report['forecast_control_counts']['controls_completed'] == len(item.calls)
    assert len(item.calls) == 12


@pytest.mark.parametrize('shorter', ['task', 'engine'])
def test_supplied_two_actions_exceeding_either_horizon_refuse_before_prediction(monkeypatch, shorter):
    item = fixture(monkeypatch)
    (item.env if shorter == 'task' else item.engine).horizon_steps = 49
    actions, report = search.prepare_root_assistance_search(item.env).search([item.base, item.second])
    assert actions is None and report['reason'] == 'explicit_sequence_exceeds_remaining_horizon' and not item.calls


def test_no_strict_violation_improvement_stops_after_one_poll(monkeypatch):
    item = fixture(monkeypatch, response=lambda actions, index: .3)
    actions, report = search.prepare_root_assistance_search(item.env).search([item.base, item.second])
    assert actions is None and report['status'] == 'no_verified_candidate' and len(item.calls) == 13
    assert len(report['rounds']) == 1 and not report['rounds'][0]['strictly_improved_measured_violation']
    assert len(report['evaluations']) == 13 and report['verified_sequence'] is None


def test_mean_violation_breaks_maximum_tie_using_measured_values(monkeypatch):
    def effect(actions, result, number):
        for row in result['controls']:
            row['endpoint']['qvel'][22] = .3
            row['endpoint']['qvel'][25] = 1.5 + (actions[0][19] if row['control'] == 1 else 0.)
    item = fixture(monkeypatch, effect=effect)
    _, report = search.prepare_root_assistance_search(item.env).search([item.base, item.second])
    assert len(report['rounds']) == 2 and report['rounds'][0]['best_evaluation'] == 1
    base, improved = report['evaluations'][:2]
    assert base['rank']['maximum_normalized_excess'] == improved['rank']['maximum_normalized_excess'] == 1.
    assert improved['rank']['mean_normalized_excess'] < base['rank']['mean_normalized_excess']


def test_minimum_change_wins_among_passing_candidates_and_clipped_polls_deduplicate(monkeypatch):
    def response(actions, index):
        return .1 if actions[0][19] == 1. or actions[1][20] < 0 else .3
    item = fixture(monkeypatch, response=response)
    item.base[19] = .9
    actions, report = search.prepare_root_assistance_search(item.env).search((item.base, item.second))
    assert report['status'] == 'selected' and actions[0][19] == 1. and actions[1][20] == 0.
    assert report['correction_squared_norm'] == pytest.approx(.01)
    encoded = [search._key(pair) for pair in item.calls[:-1]]
    assert len(encoded) == len(set(encoded)) and report['skipped_candidates']
    assert len(item.calls) <= 26


@pytest.mark.parametrize('direction', [-1, 1])
def test_float32_original_root_box_rounds_inward_even_at_second_poll(monkeypatch, direction):
    item = fixture(monkeypatch, response=lambda actions, index: .3)
    base = item.base.astype(np.float32)
    base[19] = -.1 * direction
    center = (base.copy(), item.second.astype(np.float64))
    trial = search._root_trial(center, center, 0, 0, direction, .25)
    delta = float(trial[0][19]) - float(base[19])
    assert 0 < direction * delta <= .25
    assert abs(float(np.float32(float(base[19]) + direction * .25)) - float(base[19])) > .25
    repeated = search._root_trial(center, trial, 0, 0, direction, .25)
    assert repeated[0].tobytes() == trial[0].tobytes()
    actions, report = search.prepare_root_assistance_search(item.env).search(center)
    assert actions is None and report['status'] == 'no_verified_candidate'
    assert all(np.all(abs(pair[0][19:].astype(float) - base[19:]) <= .25) for pair in item.calls)


@pytest.mark.parametrize('boundary,field', [(0, 'endpoint'), (1, 'endpoint'), (0, 'substep_contacts'), (1, 'substep_contacts')])
def test_full_repeat_detects_value_changes_at_either_boundary(monkeypatch, boundary, field):
    def effect(actions, result, number):
        if number == 24:
            if field == 'endpoint': result['controls'][boundary][field]['ctrl'][0] = .123
            else: result['controls'][boundary][field][0]['contacts']['changed'] = True
    item = fixture(monkeypatch, effect=effect)
    actions, report = search.prepare_root_assistance_search(item.env).search((item.base, item.second))
    assert actions is None and report['reason'] == 'selected_sequence_not_repeatable'
    assert report['verified_sequence'] is None and len(report['evaluations']) == 24


def test_partial_second_control_keeps_attempt_counts_action_and_completed_row(monkeypatch):
    def effect(actions, result, number):
        if number == 3:
            result.update(status='unavailable', reason='second-control failure', sequence_complete=False,
                endpoint=None, controls_completed=1, native_substeps=17, native_substeps_attempted=18,
                boundary_forwards=1, boundary_forwards_attempted=1)
            result['controls'][1].update(complete=False, endpoint=None, substep_contacts=[], native_substeps=1,
                native_substeps_attempted=2, boundary_forwards=0, boundary_forwards_attempted=0)
    item = fixture(monkeypatch, effect=effect)
    actions, report = search.prepare_root_assistance_search(item.env).search((item.base, item.second))
    assert actions is None and report['status'] == 'unavailable' and len(item.calls) == 3
    assert len(report['evaluations']) == 2 and report['failed_sequence']['controls'][0]['endpoint']
    assert report['failed_sequence']['controls'][1]['endpoint'] is None
    assert report['failed_evaluation']['kind'] == 'root_poll' and report['failed_evaluation']['prediction_attempted']
    assert report['rounds'][0]['status'] == 'interrupted' and report['rounds'][0]['completed_candidates'] == 1
    assert report['native_counts']['native_substeps'] == 81 and report['native_counts']['native_substeps_attempted'] == 82
    assert report['requested_forecast_controls_reserved'] == 6 and report['forecast_control_counts']['controls_completed'] == 5


def test_unreported_prediction_exception_consumes_reservation_and_marks_counts_unknown(monkeypatch):
    item = fixture(monkeypatch)
    controller = search.prepare_root_assistance_search(item.env)
    def fail(actions): raise RuntimeError('outside report')
    monkeypatch.setattr(controller._predictor, 'predict', fail)
    actions, report = controller.search((item.base, item.second))
    assert actions is None and report['sequence_calls'] == 1 and report['requested_forecast_controls_reserved'] == 2
    assert not report['native_counts_complete'] and report['failed_evaluation']['kind'] == 'base'


@pytest.mark.parametrize('damage', ['missing_count', 'incomplete_reservation', 'omitted', 'time', 'action'])
def test_incomplete_protocol_never_ranks_or_accepts_candidate(monkeypatch, damage):
    def effect(actions, result, number):
        if damage == 'missing_count': result.pop('copies')
        elif damage == 'incomplete_reservation': result['controls_reserved'] = 1
        elif damage == 'omitted': result['omitted_detail'] = {'endpoints': 1}
        elif damage == 'time': result['controls'][1]['initial_time_s'] += .01
        else: result['controls'][1]['human_action'][0] += .1
    item = fixture(monkeypatch, effect=effect)
    actions, report = search.prepare_root_assistance_search(item.env).search((item.base, item.second))
    assert actions is None and report['status'] == 'unavailable' and len(item.calls) == 1
    assert report['evaluations'] == [] and report['failed_evaluation']['kind'] == 'base'


@pytest.mark.parametrize('setting', [dict(maximum_sequences=13), dict(maximum_controls=27)])
def test_entire_poll_plus_repeat_must_fit_before_poll_starts(monkeypatch, setting):
    item = fixture(monkeypatch)
    config = search.RootSearchConfig(**setting)
    actions, report = search.prepare_root_assistance_search(item.env, config=config).search((item.base, item.second))
    assert actions is None and report['reason'] == 'root_search_prediction_budget' and len(item.calls) == 1
    assert report['rounds'][0]['completed_candidates'] == 0 and report['rounds'][0]['status'] == 'interrupted'


def test_base_pass_cannot_omit_required_repeat(monkeypatch):
    item = fixture(monkeypatch, response=lambda actions, index: 0.)
    config = search.RootSearchConfig(maximum_sequences=1, maximum_controls=2)
    actions, report = search.prepare_root_assistance_search(item.env, config=config).search((item.base, item.second))
    assert actions is None and report['reason'] == 'root_search_prediction_budget' and len(item.calls) == 1
    assert report['failed_evaluation']['kind'] == 'selected_repeat' and not report['failed_evaluation']['prediction_attempted']


def test_shared_deadline_includes_report_encoding(monkeypatch):
    item = fixture(monkeypatch); clock = [0.]
    monkeypatch.setattr(search.time, 'monotonic', lambda: clock[0])
    original = search.json.dumps
    def encode(value, *args, **kwargs):
        if isinstance(value, dict) and 'evaluations' in value: clock[0] = 6.
        return original(value, *args, **kwargs)
    monkeypatch.setattr(search.json, 'dumps', encode)
    actions, report = search.prepare_root_assistance_search(item.env).search((item.base, item.second))
    assert actions is None and report['reason'] == 'root_search_active_budget' and report['active_seconds'] == 6.


@pytest.mark.parametrize('change', ['live', 'accumulator', 'engine', 'metrics'])
def test_live_or_task_changes_are_reported_not_restored(monkeypatch, change):
    def effect(actions, result, number):
        if change == 'live': item.live.qpos[0] = .3
        elif change == 'accumulator': item.env._accumulator = object()
        elif change == 'engine': item.env._engine = object()
        else: item.metrics['observed_substeps'] += 1
    item = fixture(monkeypatch, effect=effect)
    actions, report = search.prepare_root_assistance_search(item.env).search((item.base, item.second))
    assert actions is None and report['status'] == 'unavailable'
    if change == 'live': assert item.live.qpos[0] == .3 and not report['live_state_unchanged']
    if change == 'metrics': assert not report['task_metrics_unchanged']


def test_original_actions_are_copied_before_any_prediction(monkeypatch):
    def effect(actions, result, number): item.base[:] = .8; item.second[:] = -.8
    item = fixture(monkeypatch, effect=effect)
    expected = (item.base.copy(), item.second.copy())
    actions, report = search.prepare_root_assistance_search(item.env).search((item.base, item.second))
    assert report['status'] == 'selected'
    assert all(pair[0][:19].tobytes() == expected[0][:19].tobytes()
        and pair[1][:19].tobytes() == expected[1][:19].tobytes() for pair in item.calls)


def test_near_cap_verified_proof_abstains_but_preserves_every_measurement(monkeypatch):
    def effect(actions, result, number):
        for row in result['controls']: row['endpoint']['proof_padding'] = 'x' * 16000
    item = fixture(monkeypatch, effect=effect)
    monkeypatch.setattr(search, 'MAX_RESULT_BYTES', 48 << 10)
    actions, report = search.prepare_root_assistance_search(item.env).search((item.base, item.second))
    assert actions is None and report['reason'] == 'root_search_result_byte_budget'
    assert report['omitted_verified_controls'] == 2 and len(report['evaluations']) == 24
    assert len(json.dumps(report, allow_nan=False, separators=(',', ':')).encode()) <= search.MAX_RESULT_BYTES


def test_large_partial_proof_reports_explicit_omission_and_attempt(monkeypatch):
    def effect(actions, result, number):
        result.update(status='unavailable', reason='partial proof', sequence_complete=False)
        result['controls'][0]['endpoint']['padding'] = 'x' * 65536
    item = fixture(monkeypatch, effect=effect)
    monkeypatch.setattr(search, 'MAX_RESULT_BYTES', 32 << 10)
    actions, report = search.prepare_root_assistance_search(item.env).search((item.base, item.second))
    assert actions is None and report['status'] == 'unavailable' and 'failed_sequence' not in report
    assert report['omitted_failed_sequence']['controls'] == 2 and report['failed_evaluation']['kind'] == 'base'
    assert report['native_counts']['copies'] == 1
    assert len(json.dumps(report, allow_nan=False, separators=(',', ':')).encode()) <= search.MAX_RESULT_BYTES


@pytest.mark.parametrize('bad', [[], [np.zeros(22)] * 3, np.zeros(22), [[True] * 22], [[0.] * 23],
    [np.zeros(22, dtype=int)], [np.full(22, np.nan)], [[[0.]] * 22], (x for x in ())])
def test_all_input_shape_and_dtype_validation_precedes_prediction(monkeypatch, bad):
    item = fixture(monkeypatch)
    actions, report = search.prepare_root_assistance_search(item.env).search(bad)
    assert actions is None and report['status'] == 'unavailable' and not item.calls


def test_overlapping_decision_does_not_consume_or_predict(monkeypatch):
    item = fixture(monkeypatch)
    controller = search.prepare_root_assistance_search(item.env)
    controller._lock.acquire()
    try: actions, report = controller.search((item.base, item.second))
    finally: controller._lock.release()
    assert actions is None and report['reason'] == 'root_search_already_running' and not item.calls and not controller._used


@pytest.mark.parametrize('kwargs', [dict(maximum_sequences=27), dict(maximum_controls=53), dict(maximum_seconds=6),
    dict(maximum_sequences=True), dict(root_residual_bound=.5)])
def test_configuration_remains_within_fixed_authority_and_work_bounds(kwargs):
    with pytest.raises(ValueError): search.RootSearchConfig(**kwargs)


def test_immutable_config_and_explicit_assistance_required(monkeypatch):
    config = search.RootSearchConfig()
    with pytest.raises(FrozenInstanceError): config.maximum_sequences = 100
    item = fixture(monkeypatch)
    item.env.task['control']['root_assistance'] = 'disabled'
    with pytest.raises(ValueError, match='existing explicit'): search.prepare_root_assistance_search(item.env)
    assert not item.calls and not item.sequence_budgets
