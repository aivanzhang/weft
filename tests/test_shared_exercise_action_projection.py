"""Bounded search and measurement tests with a fake predictor, never native physics."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from weft import shared_exercise_action_projection as projection
from weft.shared_exercise_action_prediction import STATE_FIELDS, _identity


def fixture(monkeypatch, *, roots=True, scale=1., effect=None):
    live = SimpleNamespace(time=1.6, **{name: np.zeros(0) for name in STATE_FIELDS})
    live.qpos, live.qvel = np.zeros(29), np.zeros(28)
    live.qpos[25] = 1.
    live.ctrl, live.xfrc_applied = np.zeros(22), np.zeros((3, 6))
    model = SimpleNamespace(nq=29, nv=28)
    initial = SimpleNamespace(position=(0., 0., 0.), quaternion=(1., 0., 0., 0.))
    accumulator = SimpleNamespace(initial={"bell": initial})
    engine = SimpleNamespace(model=model, data=live, _initialized=True, _done=False, scale=scale,
        _q=np.arange(22), _object_free_addresses={"bell": (22, 22)}, dt=1 / 30, substeps=16, steps=48,
        _target_at=lambda elapsed: (np.zeros(22), np.zeros(22)))
    task = {"profile": "shared-exercise-task-v1", "gates_sha256": projection.GATES_SHA256,
            "equipment_ids": ["bell"], "contact_requirements": [], "control": {"object_forces": "forbidden"}}
    env = SimpleNamespace(_engine=engine, _done=False, _accumulator=accumulator, task=task, action_count=22 if roots else 19)
    calls, budgets = [], []

    class FakePredictor:
        def predict(self, action):
            calls.append(action.copy())
            endpoint = {name: np.asarray(getattr(live, name)).tolist() for name in STATE_FIELDS}
            endpoint.update(time_s=live.time + engine.dt, contacts={"max_human_floor_penetration_model_units": 0.})
            endpoint["qvel"][22] = .3 + 2 * float(action[0])
            result = {"status": "predicted", "endpoint": endpoint,
                "native_substeps": 16, "native_substeps_attempted": 16, "boundary_forwards": 1,
                "boundary_forwards_attempted": 1, "copies": 1, "copies_attempted": 1,
                "data_allocations": int(len(calls) == 1), "data_allocations_attempted": int(len(calls) == 1),
                "substep_contacts": [{"substep": index + 1, "solver_time_s": live.time + index * engine.dt / 16,
                    "integrated_time_s": live.time + (index + 1) * engine.dt / 16,
                    "contacts": {"max_human_floor_penetration_model_units": 0.}} for index in range(16)]}
            if effect is not None:
                effect(action, result, len(calls))
            return result

    def prepare(env_arg, **kwargs):
        assert env_arg is env
        budgets.append(kwargs)
        return FakePredictor()
    monkeypatch.setattr(projection, "prepare_action_predictor", prepare)
    base = np.zeros(env.action_count)
    if roots:
        base[19:] = (.125, -.0, -.25)
    return SimpleNamespace(env=env, engine=engine, live=live, base=base, calls=calls, budgets=budgets)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_selected_action_uses_all_joints_repeat_and_exact_root_bytes(monkeypatch, dtype):
    item = fixture(monkeypatch)
    base = item.base.astype(dtype)
    before = _identity(item.live)
    projector = projection.prepare_action_projector(item.env)
    action, report = projector.project(base)
    assert report["status"] == "selected" and action.dtype == dtype
    assert action[19:].tobytes() == base[19:].tobytes()
    assert -.076 < action[0] < -.074 and np.all(abs(action) <= 1)
    assert np.all(abs(action[:19].astype(float) - base[:19]) <= .25)
    assert _identity(item.live) == before
    assert len(item.calls) == report["prediction_calls"] == 24
    assert report["native_counts"]["native_substeps"] == 384
    assert report["native_counts"]["copies"] == report["native_counts"]["boundary_forwards"] == 24
    assert report["native_counts"]["data_allocations"] == 1
    assert [row["joint"] for row in report["evaluations"] if row["kind"] == "finite_difference"] == list(range(19))
    assert report["evaluations"][-1]["kind"] == "selected_repeat"
    assert report["verified_endpoint"] and item.budgets == [{"maximum_predictions": 24, "maximum_seconds": 5.}]
    for attempted in item.calls:
        assert attempted[19:].tobytes() == base[19:].tobytes()


def test_unchanged_base_only_predicts_once_and_never_resets_lifetime_budget(monkeypatch):
    item = fixture(monkeypatch, scale=2.)
    base = item.base.astype(np.float32)
    projector = projection.prepare_action_projector(item.env)
    action, report = projector.project(base)
    assert report["status"] == "unchanged" and action.tobytes() == base.tobytes()
    action[0] = .9
    assert base[0] == 0
    assert report["prediction_calls"] == 1 and report["verified_endpoint"]
    action, again = projector.project(base)
    assert action is None and again["reason"] == "decision_lifetime_budget" and len(item.calls) == 1


def test_disabled_roots_remain_nineteen_joint_actions(monkeypatch):
    item = fixture(monkeypatch, roots=False)
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert report["status"] == "selected" and action.shape == (19,)
    assert all(call.shape == (19,) for call in item.calls)


def test_one_sided_differences_point_inward_at_action_bounds(monkeypatch):
    def effect(action, result, _):
        result["endpoint"]["qvel"][22] = .3 + 2 * (action[0] - 1)
    item = fixture(monkeypatch, effect=effect)
    item.base[:19] = np.tile([1., -1.], 10)[:19]
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert report["status"] == "selected" and .924 < action[0] < .926
    for index, trial in enumerate(item.calls[1:20]):
        differences = trial[:19] - item.base[:19]
        assert np.flatnonzero(differences).tolist() == [index]
        assert differences[index] * item.base[index] < 0 and np.all(abs(trial) <= 1)


def test_float32_selected_at_trust_bound_rounds_inward_and_preserves_roots(monkeypatch):
    def effect(action, result, _):
        distance = float(action[0]) - float(item.base[0])
        result["endpoint"]["qvel"][22] = .1 if distance >= .24999998 else .3
    item = fixture(monkeypatch, effect=effect)
    item.base = item.base.astype(np.float32)
    item.base[0] = -.1
    delta = np.zeros(19); delta[0] = .25
    monkeypatch.setattr(projection, "_linear_projection", lambda *args: (delta, {"status": "converged"}))
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert report["status"] == "selected" and action.dtype == np.float32
    assert action[19:].tobytes() == item.base[19:].tobytes()
    assert 0 < float(action[0]) - float(item.base[0]) <= .25
    rounded_outside = np.float32(float(item.base[0]) + .25)
    assert float(rounded_outside) - float(item.base[0]) > .25
    assert action[0] == np.nextafter(rounded_outside, np.float32(-np.inf))
    assert np.all(abs(action) <= 1)


def test_fixed_response_reports_only_local_box_conflict_not_global_infeasibility(monkeypatch):
    def effect(_, result, __):
        result["endpoint"]["qvel"][22] = .3
    item = fixture(monkeypatch, effect=effect)
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert action is None and report["status"] == "no_verified_candidate"
    assert report["reason"] == "linearized_row_box_conflict"
    assert report["linearized_feasibility_is_not_physical_feasibility"]
    assert report["verified_endpoint"] is None and len(item.calls) == 20


def test_measured_finite_difference_survives_local_box_conflict(monkeypatch):
    def effect(action, result, _):
        result["endpoint"]["qvel"][22] = .3 - 2 * action[0]
    item = fixture(monkeypatch, effect=effect)
    monkeypatch.setattr(projection, "_linear_projection", lambda *args: (None, {"status": "row_box_conflict"}))
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert report["status"] == "selected" and action[0] == .1
    assert report["linear_model"]["status"] == "row_box_conflict" and len(item.calls) == 21
    assert report["evaluations"][-1]["kind"] == "selected_repeat"


def test_smaller_measured_projected_candidate_wins_over_passing_difference(monkeypatch):
    def effect(action, result, _):
        result["endpoint"]["qvel"][22] = .3 - 2 * action[0]
    item = fixture(monkeypatch, effect=effect)
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert report["status"] == "selected" and .074 < action[0] < .076
    assert report["correction_squared_norm"] < .1 ** 2


def test_nonlinear_candidate_failures_do_not_expand_search(monkeypatch):
    def effect(action, result, _):
        result["endpoint"]["qvel"][22] = .3 + 2 * max(0., action[0])
    item = fixture(monkeypatch, effect=effect)
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert action is None and report["reason"] == "nonlinear_candidates_fail_measured_constraints"
    assert len(item.calls) == 23 and report["verified_endpoint"] is None


def test_repeat_checks_complete_endpoint_not_only_optimized_metrics(monkeypatch):
    def effect(_, result, number):
        if number == 24:
            result["endpoint"]["ctrl"][0] = .123
    item = fixture(monkeypatch, effect=effect)
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert action is None and report["reason"] == "selected_prediction_not_repeatable"
    assert len(item.calls) == 24 and report["verified_endpoint"] is None


def test_equipment_metrics_use_reset_pose_scale_and_angular_norm(monkeypatch):
    def effect(_, result, __):
        q, v = result["endpoint"]["qpos"], result["endpoint"]["qvel"]
        q[22] = .02
        q[25:29] = [np.cos(.05), 0., 0., np.sin(.05)]
        v[22:28] = [.1, .2, .2, .1, .2, .2]
    item = fixture(monkeypatch, scale=2., effect=effect)
    _, report = projection.prepare_action_projector(item.env).project(item.base)
    assert report["status"] == "unchanged"
    assert report["evaluations"][0]["values"][:4] == pytest.approx([.01, .1, .15, .3])
    assert [row["limit"] for row in report["constraints"][:4]] == [.03, .15, .15, .75]
    assert all(not row["true_substep_equivalent"] for row in report["constraints"][:6])


def test_floor_samples_keep_solver_clock_and_distinguish_boundary(monkeypatch):
    def effect(_, result, __):
        result["endpoint"]["qvel"][22] = 0.
        result["substep_contacts"][4]["contacts"]["max_human_floor_penetration_model_units"] = .024
        result["endpoint"]["contacts"]["max_human_floor_penetration_model_units"] = .01
    item = fixture(monkeypatch, effect=effect)
    _, report = projection.prepare_action_projector(item.env).project(item.base)
    assert report["status"] == "unchanged"
    assert report["evaluations"][0]["values"][-2:] == [.024, .01]
    assert report["constraints"][-2]["sampling"] == "solver_start_cached_contact_samples"
    assert report["constraints"][-2]["true_substep_equivalent"] is True
    assert report["constraints"][-1]["true_substep_equivalent"] is False


@pytest.mark.parametrize("damage", ["missing_floor", "clock", "nonfinite", "quaternion"])
def test_unsupported_measurements_abstain_without_guessing(monkeypatch, damage):
    def effect(_, result, __):
        if damage == "missing_floor":
            result["substep_contacts"][0]["contacts"].clear()
        elif damage == "clock":
            result["substep_contacts"][0]["solver_time_s"] += .01
        elif damage == "nonfinite":
            result["endpoint"]["qvel"][22] = 1e308
        else:
            result["endpoint"]["qpos"][25] = 2.
    item = fixture(monkeypatch, effect=effect)
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert action is None and report["status"] == "unavailable" and len(item.calls) == 1


@pytest.mark.parametrize("value", [np.zeros(22, dtype=int), np.zeros(22, dtype=np.float16), [True] + [0.] * 21,
                                  [float("nan")] * 22, [2.] * 22, [0.] * 19])
def test_invalid_actions_have_no_prediction_work(monkeypatch, value):
    item = fixture(monkeypatch)
    action, report = projection.prepare_action_projector(item.env).project(value)
    assert action is None and report["status"] == "unavailable" and not item.calls


def test_required_contacts_reject_before_predictor_preparation(monkeypatch):
    item = fixture(monkeypatch)
    item.env.task["contact_requirements"] = [{"kind": "upward_support"}]
    with pytest.raises(ValueError, match="required-contact"):
        projection.prepare_action_projector(item.env)
    assert not item.budgets


def test_reset_after_preparation_rejects_without_reusing_stale_reference(monkeypatch):
    item = fixture(monkeypatch)
    projector = projection.prepare_action_projector(item.env)
    item.env._accumulator = deepcopy(item.env._accumulator)
    action, report = projector.project(item.base)
    assert action is None and report["reason"] == "environment_contract_changed" and not item.calls


def test_live_mutation_is_detected_without_restoring_over_external_state(monkeypatch):
    item = fixture(monkeypatch, effect=lambda *_: setattr(item.live, "time", 2.))
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert action is None and report["reason"] == "live_state_changed_during_projection"
    assert item.live.time == 2. and len(item.calls) == 1


def test_partial_prediction_counters_are_preserved_on_unavailable(monkeypatch):
    def effect(_, result, __):
        result.update(status="unavailable", reason="native_exception", endpoint=None,
                      native_substeps=2, native_substeps_attempted=3, boundary_forwards=0,
                      boundary_forwards_attempted=0)
    item = fixture(monkeypatch, effect=effect)
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert action is None and report["reason"] == "prediction_unavailable:native_exception"
    assert report["prediction_calls"] == 1
    assert report["native_counts"]["native_substeps"] == 2
    assert report["native_counts"]["native_substeps_attempted"] == 3


def test_small_aggregate_budget_does_not_start_partial_difference_search(monkeypatch):
    item = fixture(monkeypatch)
    config = projection.ProjectionConfig(maximum_predictions=20)
    action, report = projection.prepare_action_projector(item.env, config=config).project(item.base)
    assert action is None and report["reason"] == "insufficient_budget_for_complete_search"
    assert len(item.calls) == 1 and report["maximum_predictions"] == 20


def test_final_encoding_time_is_charged_and_clears_selected_endpoint(monkeypatch):
    item = fixture(monkeypatch, scale=2.)
    clock = [0.]
    monkeypatch.setattr(projection.time, "monotonic", lambda: clock[0])
    encode = projection.json.dumps
    def slow_encode(*args, **kwargs):
        clock[0] += .02
        return encode(*args, **kwargs)
    monkeypatch.setattr(projection.json, "dumps", slow_encode)
    action, report = projection.prepare_action_projector(item.env,
        config=projection.ProjectionConfig(maximum_seconds=.01)).project(item.base)
    assert action is None and report["reason"] == "projection_active_compute_budget"
    assert report["verified_endpoint"] is None and report["active_seconds"] == .02


def test_result_byte_cap_abstains_without_returning_accepted_action(monkeypatch):
    item = fixture(monkeypatch, scale=2.)
    monkeypatch.setattr(projection, "MAX_RESULT_BYTES", 1024)
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert action is None and report["reason"] == "projection_result_byte_budget"
    assert report["verified_endpoint"] is None and report["evaluations"] == []


@pytest.mark.parametrize("settings", [{"maximum_predictions": 25}, {"maximum_predictions": True},
    {"maximum_seconds": float("nan")}, {"maximum_sweeps": 65}, {"trust_radius": .5}, {"difference": 0.}])
def test_invalid_configuration_rejects(settings):
    with pytest.raises(ValueError):
        projection.ProjectionConfig(**settings)


def test_configuration_is_immutable():
    config = projection.ProjectionConfig()
    with pytest.raises(FrozenInstanceError):
        config.maximum_predictions = 64


def test_minimum_change_uses_first_and_nineteenth_joint_not_an_eight_axis_subset(monkeypatch):
    def effect(action, result, _):
        result["endpoint"]["qvel"][22] = .3 + action[0] + action[18]
    item = fixture(monkeypatch, effect=effect)
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert report["status"] == "selected"
    assert action[0] == pytest.approx(action[18])
    assert -.076 < action[0] < -.074 and np.count_nonzero(action[:19]) == 2


def test_linear_iteration_limit_is_bounded_and_not_an_infeasibility_certificate():
    matrix = np.zeros((2, 19)); matrix[:, 0] = [1., -1.]
    calls = []
    delta, report = projection._linear_projection(matrix, np.array([-.2, -.2]),
        np.full(19, -.25), np.full(19, .25), 64, lambda: calls.append(1))
    assert report["status"] == "iteration_limit" and report["proposal_only"] is True
    assert np.isfinite(delta).all() and np.all(abs(delta) <= .25)
    assert np.max(matrix @ delta + .2) > 0
    assert report["sweeps"] == len(calls) == 64


@pytest.mark.parametrize("passes", [True, False])
def test_iteration_limited_proposal_requires_actual_nonlinear_pass_and_repeat(monkeypatch, passes):
    item = fixture(monkeypatch)
    delta = np.zeros(19); delta[0] = -.1 if passes else -.025
    monkeypatch.setattr(projection, "_linear_projection", lambda *args: (delta,
        {"status": "iteration_limit", "proposal_only": True, "sweeps": 64, "maximum_linear_violation": .001}))
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert report["linear_model"]["status"] == "iteration_limit"
    assert len([row for row in report["evaluations"] if row["kind"] == "candidate"]) == 3
    if passes:
        assert report["status"] == "selected" and action[0] == -.1
        assert action[19:].tobytes() == item.base[19:].tobytes()
        assert len(item.calls) == 24 and report["evaluations"][-1]["kind"] == "selected_repeat"
        assert report["evaluations"][-1]["passes_measured_constraints"] is True
    else:
        assert action is None and report["status"] == "no_verified_candidate"
        assert report["reason"] == "nonlinear_candidates_fail_measured_constraints"
        assert len(item.calls) == 23 and report["verified_endpoint"] is None


def test_endpoint_human_peak_limits_are_checked_without_claiming_interval_means(monkeypatch):
    def effect(_, result, __):
        result["endpoint"]["qvel"][22] = 0.
        result["endpoint"]["qpos"][0:19] = [.71] * 19
        result["endpoint"]["qpos"][19:22] = [.31] * 3
    item = fixture(monkeypatch, effect=effect)
    action, report = projection.prepare_action_projector(item.env).project(item.base)
    assert action is None and report["status"] == "no_verified_candidate"
    assert report["evaluations"][0]["values"][4:6] == pytest.approx([.71, .31])
    assert [row["limit"] for row in report["constraints"][4:6]] == [.7, .3]
    assert all(not row["true_substep_equivalent"] for row in report["constraints"][4:6])


def test_unexpected_predictor_exception_marks_native_counts_incomplete(monkeypatch):
    item = fixture(monkeypatch)
    projector = projection.prepare_action_projector(item.env)
    def fail(_):
        raise RuntimeError("unexpected wrapper failure")
    projector._predictor.predict = fail
    action, report = projector.project(item.base)
    assert action is None and report["reason"] == "predictor_raised:RuntimeError"
    assert report["prediction_calls"] == 1 and report["native_counts_complete"] is False


def test_active_budget_stops_after_prediction_without_partial_search(monkeypatch):
    clock = [0.]
    item = fixture(monkeypatch, effect=lambda *_: clock.__setitem__(0, .02))
    monkeypatch.setattr(projection.time, "monotonic", lambda: clock[0])
    action, report = projection.prepare_action_projector(item.env,
        config=projection.ProjectionConfig(maximum_seconds=.01)).project(item.base)
    assert action is None and report["reason"] == "projection_active_compute_budget"
    assert len(item.calls) == 1 and report["prediction_calls"] == 1
