"""Saved body errors distinguish recoverable averages, cumulative peaks and unknown clocks."""

from copy import deepcopy

import mujoco
import pytest

from weft import shared_exercise_diagnostics as diagnostics
from weft.io import canonical_hash
from weft.shared_exercise_recording import recording_scene
from weft.shared_exercise_task import GATES, SharedExerciseTaskEnv, build_shared_exercise_task
from test_shared_exercise_diagnostics import event_step, seal_recording
from test_shared_exercise_recording import short_task
from test_shared_world_human_only import person_scene


@pytest.fixture(scope="module", params=("mixed", "person"))
def boundaries(request):
    if request.param == "person":
        scene = person_scene()
        task = build_shared_exercise_task(scene)
    else:
        scene, task = short_task()
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=0)
        rows = [{"index": 0, "type": "start", "seed": 0, "snapshot": env.snapshot()}]
        rows.extend(event_step(env, index) for index in range(1, 4))
        return recording_scene(scene, task), rows
    finally:
        env.close()


def with_errors(events, changes):
    # Synthetic summaries exercise display logic, not physical trajectories or replay.
    events = deepcopy(events)
    for index, event in enumerate(events):
        state = event["snapshot"]
        metrics = state["task_metrics"]
        for name, aggregation in diagnostics.HUMAN_METRICS.items():
            metrics[name] = None if not metrics["valid_substeps"] and aggregation == "running_rms" else 0.
        metrics.update(changes.get(index, {}))
        failures = [name for name in diagnostics.HUMAN_METRICS
                    if metrics[name] is None or metrics[name] > GATES[name] + 1e-12]
        if metrics["failure"]:
            failures.insert(0, metrics["failure"])
        failures.extend(name for name in metrics["failed_gates"]
                        if name not in diagnostics.HUMAN_METRICS and name != metrics["failure"])
        metrics["failed_gates"] = failures
        metrics["success"] = metrics["completed_interval"] and not failures
        state["terminated"] = bool(metrics["failure"] or metrics["success"])
        state["truncated"] = metrics["completed_interval"] and not state["terminated"]
        if event["type"] == "step":
            event.update(terminated=state["terminated"], truncated=state["truncated"])
    return events


def no_physics(monkeypatch):
    monkeypatch.setattr(mujoco, "mj_step", lambda *_a, **_k: pytest.fail("Saved diagnosis cannot run physics"))


def test_peak_failure_has_bounded_window_and_original_task_outcome(boundaries, monkeypatch):
    scene, saved = boundaries
    events = with_errors(saved, {1: {"peak_angular_rmse_rad": .6},
                                2: {"peak_angular_rmse_rad": .9}, 3: {"peak_angular_rmse_rad": .9}})
    before = canonical_hash([scene, events])
    no_physics(monkeypatch)
    result = diagnostics.recording_failure_timeline(scene, events)
    episode = result["episodes"][0]
    human = episode["human_tracking"]
    first = human["first_failure"]
    assert first["event_index"] == first["control_step"] == 2
    assert first["previous_safe_elapsed_s"] == pytest.approx(1 / 30)
    assert first["first_failed_elapsed_s"] == pytest.approx(2 / 30)
    crossing, = human["first_crossings"]
    assert crossing["metric"] == "peak_angular_rmse_rad" and crossing["previous_value"] == .6
    assert crossing["value"] == .9 and crossing["threshold"] == .7
    assert crossing["aggregation"] == "cumulative_peak"
    assert first["source_frame_bracket"]["native_video_time_available"] is False
    assert episode["terminal"]["failed_gates"] == events[-1]["snapshot"]["task_metrics"]["failed_gates"]
    assert not episode["terminal"]["success"] and episode["terminal"]["completed_interval"]
    assert canonical_hash([scene, events]) == before


def test_running_average_recovery_is_not_a_permanent_failure(boundaries):
    scene, saved = boundaries
    events = with_errors(saved, {1: {"angular_rmse_rad": .1},
                                2: {"angular_rmse_rad": .4}, 3: {"angular_rmse_rad": .2}})
    episode = diagnostics.recording_failure_timeline(scene, events)["episodes"][0]
    human = episode["human_tracking"]
    first, = human["first_crossings"]
    assert first["metric"] == "angular_rmse_rad" and first["aggregation"] == "running_rms"
    assert first["value"] == .4 and first["previous_value"] == .1
    assert human["terminal"][0] == {"metric": "angular_rmse_rad", "value": .2,
        "threshold": .35, "status": "within_limit", "aggregation": "running_rms"}
    assert "angular_rmse_rad" not in episode["terminal"]["failed_gates"]


def test_unobserved_reset_is_not_a_crossing_or_known_safe_boundary(boundaries):
    scene, saved = boundaries
    initial = diagnostics.recording_failure_timeline(scene, saved[:1])["episodes"][0]["human_tracking"]
    assert initial["first_failure"] is None and initial["first_crossings"] == []
    assert [row["status"] for row in initial["terminal"][:2]] == ["unavailable", "unavailable"]
    events = with_errors(saved, {index: {"angular_rmse_rad": .4} for index in (1, 2, 3)})
    human = diagnostics.recording_failure_timeline(scene, events)["episodes"][0]["human_tracking"]
    crossing, = human["first_crossings"]
    assert crossing["status"] == "first_observed_boundary" and crossing["previous_value"] is None
    assert crossing["previous_safe_elapsed_s"] is human["first_failure"]["previous_safe_elapsed_s"] is None
    assert crossing["first_failed_elapsed_s"] == pytest.approx(1 / 30)


@pytest.mark.parametrize("metric", [name for name, kind in diagnostics.HUMAN_METRICS.items() if kind == "cumulative_peak"])
def test_decreasing_body_peaks_reject_inconsistent_history(boundaries, metric):
    scene, saved = boundaries
    events = with_errors(saved, {1: {metric: GATES[metric] * 2}})
    with pytest.raises(ValueError, match="human maxima decreased"):
        diagnostics.recording_failure_timeline(scene, events)


def test_exact_limits_and_tolerance_do_not_create_crossings(boundaries):
    scene, saved = boundaries
    events = with_errors(saved, {index: {name: GATES[name] + 1e-12 for name in diagnostics.HUMAN_METRICS}
                                for index in (1, 2, 3)})
    human = diagnostics.recording_failure_timeline(scene, events)["episodes"][0]["human_tracking"]
    assert human["first_failure"] is None and human["first_crossings"] == []
    assert all(row["status"] == "within_limit" for row in human["terminal"])


def test_reset_clears_peak_history_and_retains_each_episode(boundaries):
    scene, saved = boundaries
    first = with_errors(saved, {index: {"peak_angular_rmse_rad": .9} for index in (1, 2, 3)})
    second = with_errors(saved, {3: {"peak_root_rmse_model_units": .4}})
    second[0].update(type="reset", seed=1)
    events = first + second
    for index, row in enumerate(events):
        row["index"] = index
    a, b = diagnostics.recording_failure_timeline(scene, events)["episodes"]
    assert a["human_tracking"]["first_failure"]["event_index"] == 1
    assert b["human_tracking"]["first_failure"]["event_index"] == 7
    assert b["human_tracking"]["first_failure"]["reset_event_index"] == 4
    assert b["human_tracking"]["first_crossings"][0]["metric"] == "peak_root_rmse_model_units"
    assert b["human_tracking"]["terminal"][2]["status"] == "within_limit"


def test_reset_peak_has_no_invented_preceding_safe_time(boundaries):
    scene, saved = boundaries
    events = with_errors(saved[:1], {0: {"max_floor_penetration_model_units": .03}})
    human = diagnostics.recording_failure_timeline(scene, events)["episodes"][0]["human_tracking"]
    row, = human["first_crossings"]
    assert row["status"] == "failed_at_reset" and row["control_step"] == 0
    assert row["previous_value"] is row["previous_safe_elapsed_s"] is None
    assert human["first_failure"]["previous_safe_elapsed_s"] is None


def test_body_crossing_does_not_label_stale_failure_geometry_current(boundaries):
    scene, saved = boundaries
    events = deepcopy(saved[:2])
    state = events[1]["snapshot"]
    state["engine_state"] = deepcopy(events[0]["snapshot"]["engine_state"])
    state["engine_state_status"] = "last_valid_control_boundary"
    state["task_metrics"]["failure"] = "engine_failure:analytical_failure_state"
    events = with_errors(events, {1: {"peak_angular_rmse_rad": .8}})
    first = diagnostics.recording_failure_timeline(scene, events)["episodes"][0]["human_tracking"]["first_failure"]
    assert first["engine_state_status"] == "last_valid_control_boundary"
    assert first["last_valid_geometry_elapsed_s"] == 0
    assert first["first_failed_elapsed_s"] == pytest.approx(1 / 30)
    assert first["source_frame_bracket"]["native_video_time_available"] is False


def test_all_body_gates_retained_with_truthful_equipment_applicability(boundaries):
    scene, saved = boundaries
    events = with_errors(saved, {index: {name: GATES[name] * 2 for name in diagnostics.HUMAN_METRICS}
                                for index in (1, 2, 3)})
    result = diagnostics.recording_failure_timeline(scene, events)
    human = result["episodes"][0]["human_tracking"]
    assert [row["metric"] for row in human["first_crossings"]] == list(diagnostics.HUMAN_METRICS)
    assert len(human["first_failure"]["crossings"]) == 5
    assert all(row["status"] == "above_limit" for row in human["terminal"])
    expected = "not_applicable" if not scene["engine_scene"]["objects"] else "applied"
    assert result["equipment_stability_applicability"] == expected
    if expected == "not_applicable":
        assert result["first_equipment_failure"] is None
        assert result["episodes"][0]["first_crossings"] == []


def test_sealed_body_summary_requires_no_media_or_replay(tmp_path, boundaries, monkeypatch):
    scene, saved = boundaries
    events = with_errors(saved, {3: {"max_floor_penetration_model_units": .03}})
    root = seal_recording(tmp_path / "trial", scene, events)
    original = diagnostics._read
    def read(fd, name, *args):
        assert name not in {"video.mp4", "preview.gif", "replay.html"}
        return original(fd, name, *args)
    monkeypatch.setattr(diagnostics, "_read", read)
    no_physics(monkeypatch)
    result = diagnostics.inspect_recording_failure(root)
    assert result["episodes"][0]["human_tracking"]["first_failure"]["event_index"] == 3
    assert result["integrity"]["action_replay_performed"] is result["integrity"]["media_verified"] is False
