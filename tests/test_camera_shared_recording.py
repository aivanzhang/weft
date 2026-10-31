"""Real saved-state/metric validation with an analytical engine and no native or media work."""

from copy import deepcopy
import json

import numpy as np
import pytest

from test_camera_shared_task import AnalyticalEngine, analytical_only, scene_fixture, _deny
from weft import camera_shared_task as task_api
from weft import camera_shared_state as state_api
from weft import camera_shared_recording as adapter
from weft import shared_world
from weft.camera_shared_receipt import IDENTITIES, PROTOCOL
from weft.io import canonical_hash


class RecordingEngine(AnalyticalEngine):
    def snapshot(self):
        source = super().snapshot()
        action = [float(value) for value in self.actions[-1]] if self.steps else [0.] * 22
        failed = self.mode == "finite_failure" and self.steps > 0
        state = {**source, "bounds": list(self.scene["bounds"]),
                 "coordinate_frame": deepcopy(self.scene["coordinate_frame"]), "controlled_id": "human", "target": None,
                 "episode_steps": self.steps, "substeps": self.scene["substeps"], "success": False,
                 "terminated": bool(failed), "truncated": bool(not failed and self.steps >= self.scene["task"]["horizon_steps"]),
                 "normalized_action": action, "virtual_force": [0., 0., 0.], "physical_validation": "unverified",
                 "contacts": [], "contact_details": [], "contact_count": 0,
                 "contact_summary": {key: 0 for key in shared_world.CONTACT_KINDS},
                 "coupled_metrics": {"target_object_id": None, "object_distance_world": None, "object_speed_world_s": None,
                    "object_parked": None, "human_object_contact_satisfied": None, "object_goal_status": "not_applicable",
                    "human_object_contact_status": "not_applicable", "completed_horizon": False, "success": False,
                    "steps": self.steps, "failure": "human_position_envelope_exceeded" if failed else None}}
        state["human"].update(control_mode="direct_motor_actions", normalized_action=action,
            motor_forces=[a * limit for a, limit in zip(action, self.scene["action_contract"]["physical_limits"], strict=True)],
            segments=[{"name": row[0], "a": [0., 0., 1.], "b": [0., 0., 1.2], "radius": .04}
                      for row in shared_world.HUMAN_SEGMENT_DEFS],
            muscles=[{"status": "illustrative", "activation": 0., "a": [0., 0., 1.], "b": [0., 0., 1.2]} for _ in range(38)])
        return state


@pytest.fixture(autouse=True)
def recording_boundary(analytical_only, monkeypatch):
    monkeypatch.setattr(task_api, "SharedWorldEnv", RecordingEngine)


def setup(*, root=False):
    scene = scene_fixture()
    scene["bounds"] = [6., 6., 6.]
    scene["substeps"] = scene["coordinate_frame"]["clock_binding"]["physics_substeps"] = 9
    task = task_api.build_camera_shared_task(scene, root_assistance=root)
    envelope = adapter.recording_scene(scene, task)
    env = task_api.CameraSharedTaskEnv(scene, task)
    return scene, task, envelope, env


def _reset(env, index=0, *, seed=7, kind="start"):
    env.reset(seed=seed)
    return {"type": kind, "index": index, "seed": seed, "snapshot": env.snapshot()}


def _step(env, index, *, controller="manual", action=None):
    action = (env.authored_action() if controller == "authored" else np.zeros(env.action_count)) if action is None else action
    _, reward, terminated, truncated, _ = env.step(action)
    return {"type": "step", "index": index, "controller": controller, "action": np.asarray(action, dtype=float).tolist(),
            "reward": reward, "terminated": terminated, "truncated": truncated, "snapshot": env.snapshot()}


def sequence(*, root=False):
    scene, task, envelope, env = setup(root=root)
    events = [_reset(env)]
    manual = np.zeros(env.action_count)
    manual[0] = .1
    if root:
        manual[-1] = -.1
    events.append(_step(env, 1, action=manual))
    events.append(_reset(env, 2, seed=19, kind="reset"))
    events.extend(_step(env, index, controller="authored" if index == 3 else "coast") for index in (3, 4, 5))
    return scene, task, envelope, env, events


@pytest.mark.parametrize("root", [False, True])
def test_actual_v6_scene_full_state_and_public_replay_identity(root):
    scene, task, envelope, env, events = sequence(root=root)
    validate = adapter.prepare_snapshot_validator(envelope)
    original = deepcopy((scene, task, envelope, events))
    for event in events:
        checked = validate(event["snapshot"])
        assert checked == event["snapshot"] and checked is not event["snapshot"]
        adapter.validate_event(event, envelope)
    assert adapter.validate_recording_scene(envelope) == envelope
    view = adapter.replay_view(envelope, events)
    assert view["scene"]["profile"] == "shared-world-v6" and "camera_reference" in view["scene"]["human_spec"]
    assert view["exercise_task"] == task
    assert view["camera_identity"] == {"scene_sha256": task["scene_sha256"], "task_sha256": canonical_hash(task),
                                       **{key: task[key] for key in IDENTITIES}}
    for projected, original_event in zip(view["events"], events, strict=True):
        assert projected["snapshot"] == original_event["snapshot"]["engine_state"]
        assert projected["exercise_metrics"] == original_event["snapshot"]["task_metrics"]
    assert (scene, task, envelope, events) == original
    assert events[-1]["snapshot"]["task_metrics"]["success"]
    env.close()


def test_policy_avatar_alias_and_extra_fields_cannot_enter_recording(monkeypatch):
    scene, task, envelope, _ = setup()
    for mutate in (lambda value: value.update(avatar={"asset": "invented"}),
                   lambda value: value.update(learned_policy={}), lambda value: value.update(dt=1 / 30),
                   lambda value: value.update(objects=[{"id": "invented"}])):
        changed = deepcopy(envelope)
        mutate(changed)
        with pytest.raises(ValueError):
            adapter.validate_recording_scene(changed)
    monkeypatch.setattr(adapter, "prepare_camera_shared_task", _deny)
    with pytest.raises(ValueError, match="policy"):
        adapter.recording_scene(scene, task, policy_document=b"{}")


def test_every_actual_camera_pin_is_independently_checked_without_legacy_relabeling():
    _, task, envelope, env = setup()
    event = _reset(env)
    validate = adapter.prepare_snapshot_validator(envelope)
    for key in ("scene_sha256", "task_sha256", *IDENTITIES):
        changed = deepcopy(event["snapshot"])
        changed[key] = "0" * 64
        with pytest.raises(ValueError, match="identities"):
            validate(changed)
    for target, key, value in (("task_metrics", "profile", "shared-exercise-task-v2"),
                               ("task_metrics", "task_sha256", task["metric_context"]["kernel_arguments_sha256"]),
                               ("task_metrics", "evaluator_sha256", task["kernel_evaluator_sha256"]),
                               ("engine_state", "profile", "shared-world-v3"),
                               ("engine_state", "camera_coupling_sha256", "0" * 64)):
        changed = deepcopy(event["snapshot"])
        changed[target][key] = value
        with pytest.raises(ValueError):
            validate(changed)


def test_camera_action_clock_reset_gate_and_reward_tampering_is_rejected():
    _, _, envelope, env = setup()
    _reset(env)
    snapshot = _step(env, 1)["snapshot"]
    validate = adapter.prepare_snapshot_validator(envelope)
    edits = [lambda state: state["last_action"].append(0.),
             lambda state: state["last_action"].__setitem__(0, -(2**63)),
             lambda state: state["engine_state"]["normalized_action"].__setitem__(19, .1),
             lambda state: state["task_metrics"].update(observed_source_time_s=10.),
             lambda state: state["task_metrics"].update(gates_sha256="0" * 64),
             lambda state: state["task_metrics"].update(failure_reward_penalty=-1.),
             lambda state: state["task_metrics"].update(reward_integral=3., mean_reward=3. / (env.dt * env.horizon_steps)),
             lambda state: state["task_metrics"].update(observed_substeps=7),
             lambda state: state["task_metrics"].update(angular_rmse_rad=.1),
             lambda state: state["task_metrics"].update(completed_interval=True, success=True),
             lambda state: state["engine_state"].update(episode_steps=0),
             lambda state: state["resolved_reset_options"].pop("profile"),
             lambda state: state["task_metrics"].update(reset_protocol={}),
             lambda state: state["engine_state"]["contact_summary"].update(human_object=1)]
    for edit in edits:
        changed = deepcopy(snapshot)
        edit(changed)
        with pytest.raises(ValueError):
            validate(changed)
    assert validate(snapshot) == snapshot


@pytest.mark.parametrize("mode", ["finite_failure", "nonfinite", "bad_clock"])
def test_partial_stale_and_invalid_clock_failures_remain_inspectable(mode):
    _, _, envelope, env = setup()
    start = _reset(env)
    validate = adapter.prepare_snapshot_validator(envelope)
    env._engine.mode = mode
    event = _step(env, 1)
    assert validate(event["snapshot"]) == event["snapshot"]
    adapter.validate_event(event, envelope)
    metrics = event["snapshot"]["task_metrics"]
    assert metrics["failure"] and not metrics["success"] and metrics["failure_reward_penalty"] == -1.
    if mode == "nonfinite":
        assert event["snapshot"]["engine_state"] == start["snapshot"]["engine_state"]
        assert event["snapshot"]["engine_state_status"] == "last_valid_control_boundary"
        assert metrics["observed_source_time_s"] > start["snapshot"]["task_metrics"]["observed_source_time_s"]
    if mode == "finite_failure":
        assert event["snapshot"]["engine_state"]["partial_action"]["completed_substeps"] == 3
    changed = deepcopy(event["snapshot"])
    changed["engine_state_status"] = "current" if mode == "nonfinite" else "last_valid_control_boundary"
    if mode == "nonfinite":
        with pytest.raises(ValueError):
            validate(changed)


def test_late_recovery_cannot_remove_failed_gate_and_cancel_never_becomes_step():
    _, _, envelope, env = setup()
    start = _reset(env)
    validate = adapter.prepare_snapshot_validator(envelope)
    cancelled = env.cancel()
    assert validate(cancelled) == cancelled
    for event in ({**start, "snapshot": cancelled},
                  {"type": "step", "action": [0.] * 19, "reward": -1., "terminated": True,
                   "truncated": False, "snapshot": cancelled},
                  {"type": "cancel", "snapshot": cancelled}):
        with pytest.raises(ValueError, match="cancel"):
            adapter.validate_event(event, envelope)
    _reset(env)
    env._engine.penetration = .03
    _step(env, 1)
    env._engine.penetration = 0.
    _step(env, 2)
    final = _step(env, 3)
    assert validate(final["snapshot"]) == final["snapshot"]
    assert final["snapshot"]["task_metrics"]["failed_gates"] == ["max_floor_penetration_model_units"]
    changed = deepcopy(final["snapshot"])
    changed["task_metrics"].update(failure="cancelled", failure_reward_penalty=-1., completed_interval=False)
    changed["terminated"], changed["truncated"] = True, False
    with pytest.raises(ValueError):
        validate(changed)


def test_only_genuine_resets_and_supported_controller_events_are_recordable():
    _, _, envelope, _, events = sequence()
    step, start = events[1], events[0]
    edits = [lambda value: value.update(controller="learned"),
             lambda value: value.update(controller="reference_proposal"),
             lambda value: value.update(controller="coast"),
             lambda value: value.update(policy_sha256="0" * 64),
             lambda value: value.update(reference_controller_sha256="0" * 64),
             lambda value: value.update(avatar_sha256="0" * 64),
             lambda value: value.update(reward=float("nan")),
             lambda value: value.update(terminated=True),
             lambda value: value["action"].__setitem__(0, True)]
    for edit in edits:
        changed = deepcopy(step)
        edit(changed)
        with pytest.raises(ValueError):
            adapter.validate_event(changed, envelope)
    advanced = {"type": "reset", "seed": 1, "snapshot": step["snapshot"]}
    with pytest.raises(ValueError, match="initial"):
        adapter.validate_event(advanced, envelope)
    with pytest.raises(ValueError, match="fields"):
        adapter.validate_event({**start, "controller": "manual"}, envelope)


@pytest.mark.parametrize("root", [False, True])
def test_full_action_replay_returns_exact_camera_identity_without_policies(root, monkeypatch):
    _, task, envelope, _, events = sequence(root=root)
    from weft import recording

    monkeypatch.setattr(recording, "load_recording", lambda *_args, **_kwargs: deepcopy({"scene": envelope, "events": events}))
    result = adapter.verify_action_replay("analytical-no-files", max_seconds=2)
    assert result["protocol"] == PROTOCOL and result["exact_state_replay"]
    assert result["events"] == 6 and result["steps"] == 4
    assert result["scene_sha256"] == task["scene_sha256"] and result["task_sha256"] == canonical_hash(task)
    assert {key: result[key] for key in IDENTITIES} == {key: task[key] for key in IDENTITIES}
    assert result["policy_sha256"] is result["learned_actions_verified"] is None
    assert RecordingEngine.instances[-1].closed


@pytest.mark.parametrize("change", ["state", "reward", "authored"])
def test_replay_mismatch_closes_engine_and_never_reports_exact(change, monkeypatch):
    _, _, envelope, _, events = sequence()
    from weft import recording

    if change == "state":
        events[1]["snapshot"]["task_metrics"]["observed_source_time_s"] += .1
    elif change == "reward":
        events[1]["reward"] += .1
    else:
        events[3]["action"][0] = .123
        events[3]["snapshot"]["last_action"][0] = .123
    monkeypatch.setattr(recording, "load_recording", lambda *_args, **_kwargs: deepcopy({"scene": envelope, "events": events}))
    with pytest.raises(ValueError, match="differs"):
        adapter.verify_action_replay("analytical-no-files", max_seconds=2)
    assert RecordingEngine.instances[-1].closed


def test_replay_load_deadline_and_unsupported_budget_cannot_create_engine(monkeypatch):
    _, _, envelope, _, events = sequence()
    from weft import recording

    count = len(RecordingEngine.instances)
    ticks = iter((0., 3.))
    monkeypatch.setattr(adapter.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(recording, "load_recording", lambda *_args, **_kwargs: {"scene": envelope, "events": events})
    with pytest.raises(TimeoutError, match="before engine"):
        adapter.verify_action_replay("analytical-no-files", max_seconds=2)
    assert len(RecordingEngine.instances) == count
    for budget in (True, 0, 181, float("nan")):
        with pytest.raises(ValueError, match="positive bound"):
            adapter.verify_action_replay("analytical-no-files", max_seconds=budget)


def test_prepared_validation_and_rendering_reuse_pins_without_native_hash_or_model_work(monkeypatch):
    _, _, envelope, env = setup()
    snapshot = _reset(env)["snapshot"]
    validate = adapter.prepare_snapshot_validator(envelope)
    rendered, annotated, prepared = [], [], []

    def prepare(scene, *, width, show_target):
        assert scene["profile"] == "shared-world-v6" and show_target is False
        prepared.append(scene)

        def render(state):
            rendered.append(deepcopy(state))
            return np.zeros((4, width, 3), dtype=np.uint8)

        return render

    def annotate(frame, state):
        annotated.append(state["task_metrics"]["profile"])
        return frame

    monkeypatch.setattr(shared_world, "prepare_shared_renderer", prepare)
    monkeypatch.setattr(adapter, "_annotate_frame", annotate)
    render = adapter.prepare_renderer(envelope, width=64)
    for owner, names in ((state_api, ("canonical_hash",)), (shared_world, ("canonical_hash", "validate_shared_world")),
                         (task_api, ("implementation_identity", "validate_shared_world", "canonical_hash"))):
        for name in names:
            monkeypatch.setattr(owner, name, _deny)
    before = deepcopy(snapshot)
    assert validate(snapshot) == snapshot
    assert np.array_equal(render(snapshot), render(snapshot))
    assert snapshot == before and len(prepared) == 1 and len(rendered) == 2
    assert annotated == [task_api.METRICS_PROFILE] * 2


def test_snapshot_boundedness_rejects_before_engine_validation(monkeypatch):
    _, _, envelope, env = setup()
    snapshot = _reset(env)["snapshot"]
    monkeypatch.setattr(state_api, "_prepare_engine_validator", lambda *_args: _deny)
    validate = adapter.prepare_snapshot_validator(envelope)
    snapshot["padding"] = "x" * state_api.MAX_SNAPSHOT_BYTES
    with pytest.raises(ValueError, match="bound"):
        validate(snapshot)
