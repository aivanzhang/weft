"""Mock recording boundaries; only the pure action scatter implementation is real."""

import builtins
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from weft.io import canonical_hash


ROOT = Path(__file__).parents[1]
EVALUATOR = "e" * 64


def scene(root_assistance=False):
    engine = {"profile": "shared-world-v5", "dt": .02,
              "objects": [{"id": "equipment", "position": [1., 2., 3.]}],
              "action_contract": {"names": [f"action_{index}" for index in range(64)]}}
    task = {"profile": "shared-exercise-feet-task-v1",
            "base_task": {"control": {"root_assistance": "explicit_3_axis" if root_assistance else "disabled"}},
            "base_scene_sha256": "a" * 64, "base_task_sha256": "b" * 64,
            "projection_contract_sha256": "c" * 64, "adapter_implementation_sha256": "d" * 64,
            "foot_reference_sha256": "f" * 64}
    return {"schema_version": 1, "profile": "shared-exercise-feet-recording-v1", "engine_scene": engine,
            "exercise_task": task, "dt": engine["dt"], "objects": deepcopy(engine["objects"])}


def snapshot(action=None, *, step=0):
    action = [0.] * 25 if action is None else list(action)
    normalized = [0.] * 64
    for index, value in zip(tuple(range(19)) + tuple(range(22, 28)), action):
        normalized[index] = value
    return {"last_action": action, "engine_state_status": "current", "terminated": False, "truncated": False,
            "resolved_reset_options": {"jitter": False, "jitter_scale": 0.},
            "engine_state": {"normalized_action": normalized, "steps": step, "episode_steps": step,
                             "time_s": step * .02, "terminated": False, "truncated": False},
            "task_metrics": {"evaluator_sha256": EVALUATOR, "observed_substeps": step * 8,
                             "valid_substeps": step * 8, "simulated_duration_s": step * .02,
                             "valid_observed_duration_s": step * .02, "reward_integral": step * .01,
                             "mean_reward": .5 if step else 0., "failure_reward_penalty": 0.,
                             "failure": None, "success": False, "completed_interval": False},
            "ankle_diagnostics": {"observed_substeps": step * 8,
                                  "latest_substep": {"unscored_effort": .7} if step else None}}


def step_event(action=None, controller="manual"):
    action = [index / 50 for index in range(25)] if action is None else list(action)
    return {"index": 1, "type": "step", "action": action, "controller": controller, "reward": .5,
            "terminated": False, "truncated": False, "snapshot": snapshot(action, step=1)}


def replace_at(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


@pytest.fixture
def adapter(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Feet recording unit checks cannot invoke native, model, network or subprocess work")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"mujoco", "mediapipe", "tensorflow", "torch", "cv2", "requests", "httpx"}:
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for owner, names in ((socket, ("socket", "create_connection")),
                         (subprocess, ("Popen", "run", "check_call", "check_output")),
                         (os, ("system", "popen", "fork", "posix_spawn", "posix_spawnp"))):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, forbidden)

    def install(name, **members):
        module = ModuleType("weft." + name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        return module

    calls = SimpleNamespace(preparations=[], validator_preparations=[], validations=[],
                            renderer_preparations=[], renderings=[], annotations=[], order=[])

    def prepare(task, engine):
        calls.preparations.append((deepcopy(task), deepcopy(engine)))
        return deepcopy(engine), {"unused_base_scoring_scene": True}, deepcopy(task)

    def prepare_validator(engine, task):
        calls.validator_preparations.append((engine, task))

        def validate(state):
            calls.validations.append(state)
            calls.order.append("validate")
            if state.get("invalid"):
                raise ValueError("Mock full-state validator rejected the frame")

        return validate

    def prepare_renderer(engine, **options):
        calls.renderer_preparations.append((engine, options))

        def render(state):
            calls.renderings.append(state)
            calls.order.append("render")
            return ("mock_image", len(calls.renderings))

        return render

    def annotate(image, state):
        calls.annotations.append((image, state))
        calls.order.append("annotate")
        return ("annotated", image, state["engine_state_status"])

    install("shared_support_task", MAX_CONTACTS=64)
    install("shared_world_observation", **{name: type(name, (), {})
            for name in ("EquipmentState", "HumanObjectContact", "SubstepSample")})
    install("shared_world", _HUMAN_SEGMENT_BY_GEOM={"left_foot_geom": "left_foot", "right_foot_geom": "right_foot"},
            prepare_shared_renderer=prepare_renderer, SharedWorldEnv=forbidden, validate_shared_world=forbidden)
    install("shared_exercise_feet_task", prepare_shared_exercise_feet_task=prepare, SharedExerciseFeetTaskEnv=forbidden)
    install("shared_exercise_feet_state", prepare_shared_exercise_feet_snapshot_validator=prepare_validator)
    install("shared_exercise_recording", _annotate_frame=annotate)
    loader = install("recording", load_recording=forbidden)

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, ROOT / "src/weft" / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    projection = load("weft.shared_exercise_feet_projection", "shared_exercise_feet_projection.py")
    module = load("_feet_recording_unit", "shared_exercise_feet_recording.py")
    return SimpleNamespace(module=module, calls=calls, projection=projection, loader=loader)


def test_envelope_prepares_exact_world_and_detaches_object_alias(adapter):
    value = scene()
    before = deepcopy(value)
    result = adapter.module.recording_scene(value["engine_scene"], value["exercise_task"])
    assert result == value and value == before
    assert adapter.calls.preparations == [(value["exercise_task"], value["engine_scene"])]
    result["objects"][0]["position"][0] = 99.
    assert result["engine_scene"]["objects"] == value["engine_scene"]["objects"]
    assert adapter.module.validate_recording_scene(value) == value


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 1.), ("schema_version", 2),
    ("profile", "shared-exercise-recording-v1"), ("policy", {}), ("policy_sha256", "f" * 64),
    ("reference_controller", {}), ("avatar", {}), ("avatars", []),
])
def test_versioned_envelope_rejects_legacy_policy_avatar_and_extra_fields(adapter, field, value):
    document = scene()
    document[field] = value
    with pytest.raises(ValueError, match="versioned contract"):
        adapter.module.validate_recording_scene(document)
    assert not adapter.calls.preparations


def test_missing_envelope_field_is_rejected_before_task_preparation(adapter):
    value = scene()
    del value["objects"]
    with pytest.raises(ValueError, match="versioned contract"):
        adapter.module.validate_recording_scene(value)
    assert not adapter.calls.preparations


@pytest.mark.parametrize("change", ["dt", "objects", "numeric_alias"])
def test_aliases_preserve_exact_world_bytes_not_only_python_numeric_equality(adapter, change):
    value = scene()
    if change == "dt":
        value["dt"] *= 2
    elif change == "objects":
        value["objects"][0]["position"][0] += .1
    else:
        value["objects"][0]["position"][0] = 1
        assert value["objects"] == value["engine_scene"]["objects"]
    with pytest.raises(ValueError, match="aliases"):
        adapter.module.validate_recording_scene(value)


@pytest.mark.parametrize("root_assistance", [False, True])
def test_action_route_keeps_25_rotations_then_optional_three_roots_and_zero_object_forces(adapter, monkeypatch, root_assistance):
    actual_scatter = adapter.projection.scatter_task_action
    routed = []

    def capture(action, **options):
        result = actual_scatter(action, **options)
        routed.append((options, result))
        return result

    monkeypatch.setattr(adapter.module, "scatter_task_action", capture)
    action = tuple(index / 40 for index in range(28 if root_assistance else 25))
    assert adapter.module.validate_action(action, scene(root_assistance)) is None
    options, engine = routed[0]
    assert options == {"engine_action_count": 64, "root_assistance": root_assistance}
    assert engine[:19] == action[:19] and engine[22:28] == action[19:25]
    assert engine[19:22] == (action[25:28] if root_assistance else (0., 0., 0.))
    assert engine[28:] == (0.,) * 36


@pytest.mark.parametrize("root_assistance,size", [(False, 24), (False, 28), (True, 25), (True, 29)])
def test_wrong_action_route_size_is_not_padded_or_clipped(adapter, root_assistance, size):
    with pytest.raises(ValueError, match="exactly"):
        adapter.module.validate_action([0.] * size, scene(root_assistance))


@pytest.mark.parametrize("value", [True, "0", float("nan"), float("inf"), 1.01])
def test_actions_reject_coercion_nonfinite_or_out_of_range_values(adapter, value):
    action = [0.] * 25
    action[22] = value
    with pytest.raises(ValueError):
        adapter.module.validate_action(action, scene())


@pytest.mark.parametrize("action", [None, {"action": [0.] * 25}, "0" * 25])
def test_action_requires_an_explicit_vector(adapter, action):
    with pytest.raises(ValueError, match="explicit numeric vector"):
        adapter.module.validate_action(action, scene())


@pytest.mark.parametrize("control", [None, True, "implicit"])
def test_action_requires_explicit_root_assistance(adapter, control):
    value = scene()
    value["exercise_task"]["base_task"]["control"]["root_assistance"] = control
    with pytest.raises(ValueError, match="explicit root-assistance"):
        adapter.module.validate_action([0.] * 25, value)


@pytest.mark.parametrize("controller", ["manual", "coast", "authored", "omitted"])
def test_supported_step_controllers_keep_exact_action_metadata(adapter, controller):
    event = step_event([0.] * 25 if controller == "coast" else None, controller)
    if controller == "omitted":
        del event["controller"]
    before = deepcopy(event)
    assert adapter.module.validate_event(event, scene()) is None
    assert event == before


@pytest.mark.parametrize("controller", [None, True, {}, "learned", "reference"])
def test_unqualified_step_controller_is_rejected(adapter, controller):
    with pytest.raises(ValueError, match="manual, coast or authored"):
        adapter.module.validate_event(step_event(controller=controller), scene())


@pytest.mark.parametrize("kind", ["start", "step"])
@pytest.mark.parametrize("field", ["policy_sha256", "reference_controller_sha256"])
def test_controller_hash_claims_are_forbidden_even_if_empty(adapter, kind, field):
    event = step_event() if kind == "step" else {"type": "start", "seed": 0, "snapshot": snapshot()}
    event[field] = None
    with pytest.raises(ValueError, match="unqualified"):
        adapter.module.validate_event(event, scene())


def test_nonzero_coast_and_action_snapshot_mismatch_are_rejected(adapter):
    with pytest.raises(ValueError, match="coast actions must be zero"):
        adapter.module.validate_event(step_event(controller="coast"), scene())
    event = step_event()
    event["snapshot"]["last_action"][24] += .01
    with pytest.raises(ValueError, match="action differs"):
        adapter.module.validate_event(event, scene())


@pytest.mark.parametrize("reward", [None, True, "0.5", float("nan"), float("inf")])
def test_reward_metadata_must_be_finite_numeric_not_coerced(adapter, reward):
    event = step_event()
    event["reward"] = reward
    with pytest.raises(ValueError, match="reward must be finite"):
        adapter.module.validate_event(event, scene())


@pytest.mark.parametrize("field", ["terminated", "truncated"])
@pytest.mark.parametrize("flag", [True, 0, None])
def test_termination_flags_are_boolean_and_agree_with_full_state(adapter, field, flag):
    event = step_event()
    event[field] = flag
    with pytest.raises(ValueError, match="termination"):
        adapter.module.validate_event(event, scene())


@pytest.mark.parametrize("kind,seed", [("start", 0), ("reset", 2**32 - 1)])
def test_start_and_reset_accept_only_genuine_empty_control_boundaries(adapter, kind, seed):
    assert adapter.module.validate_event({"type": kind, "seed": seed, "snapshot": snapshot()}, scene()) is None


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1., None])
def test_reset_seed_requires_bounded_integer(adapter, seed):
    with pytest.raises(ValueError, match="seed"):
        adapter.module.validate_event({"type": "reset", "seed": seed, "snapshot": snapshot()}, scene())


@pytest.mark.parametrize("path,value", [
    (("engine_state_status",), "last_valid"), (("last_action", 24), .1),
    (("engine_state", "normalized_action", 28), .1),
    *((('engine_state', key), 1) for key in ("steps", "episode_steps", "time_s")),
    *((('task_metrics', key), 1) for key in ("observed_substeps", "valid_substeps", "simulated_duration_s",
                                           "valid_observed_duration_s", "reward_integral", "mean_reward", "failure_reward_penalty")),
    (("terminated",), True), (("truncated",), True), (("engine_state", "terminated"), True),
    (("engine_state", "truncated"), True), (("task_metrics", "failure"), "failed_step"),
    (("task_metrics", "success"), True), (("task_metrics", "completed_interval"), True),
    (("ankle_diagnostics", "observed_substeps"), 1), (("ankle_diagnostics", "latest_substep"), {}),
])
def test_reset_cannot_relabel_advanced_failed_or_nonempty_state(adapter, path, value):
    state = snapshot()
    replace_at(state, path, value)
    with pytest.raises(ValueError, match="genuine initial control boundary"):
        adapter.module.validate_event({"type": "reset", "seed": 17, "snapshot": state}, scene())


def test_reset_rejects_controller_and_unknown_event_type(adapter):
    event = {"type": "reset", "seed": 17, "controller": "manual", "snapshot": snapshot()}
    with pytest.raises(ValueError, match="only to action events"):
        adapter.module.validate_event(event, scene())
    event.pop("controller")
    event["type"] = "stop"
    with pytest.raises(ValueError, match="start, reset or step"):
        adapter.module.validate_event(event, scene())


@pytest.mark.parametrize("kind", ["start", "reset", "step"])
@pytest.mark.parametrize("field", ["learned_policy", "reference_controller", "avatar"])
def test_events_reject_unsupported_metadata_before_it_can_enter_exports(adapter, kind, field):
    event = step_event() if kind == "step" else {"type": kind, "seed": 17, "snapshot": snapshot()}
    event[field] = {"claimed": True}
    with pytest.raises(ValueError, match="versioned event contract"):
        adapter.module.validate_event(event, scene())


@pytest.mark.parametrize("kind,field", [
    ("step", "type"), ("step", "action"), ("step", "reward"), ("step", "terminated"),
    ("step", "truncated"), ("step", "snapshot"), ("reset", "seed"), ("reset", "snapshot"),
])
def test_events_require_their_complete_shape(adapter, kind, field):
    event = step_event() if kind == "step" else {"type": kind, "seed": 17, "snapshot": snapshot()}
    del event[field]
    with pytest.raises(ValueError):
        adapter.module.validate_event(event, scene())


@pytest.mark.parametrize("index", [True, -1, 1., None])
def test_optional_journal_index_requires_nonnegative_integer(adapter, index):
    event = step_event()
    event["index"] = index
    with pytest.raises(ValueError, match="journal index"):
        adapter.module.validate_event(event, scene())


def test_prepared_renderer_validates_every_frame_without_repreparing(adapter):
    value = scene()
    render = adapter.module.prepare_renderer(value, width=800)
    first, second = snapshot(), snapshot(step=1)
    second["engine_state_status"] = "last_valid"
    assert render(first) == ("annotated", ("mock_image", 1), "current")
    assert render(second) == ("annotated", ("mock_image", 2), "last_valid")
    with pytest.raises(ValueError, match="full-state validator"):
        render({"invalid": True})
    calls = adapter.calls
    assert len(calls.preparations) == len(calls.validator_preparations) == len(calls.renderer_preparations) == 1
    assert calls.renderer_preparations == [(value["engine_scene"], {"width": 800, "show_target": False})]
    assert calls.validations == [first, second, {"invalid": True}]
    assert calls.renderings == [first["engine_state"], second["engine_state"]]
    assert [state for _, state in calls.annotations] == [first, second]
    assert calls.order == ["validate", "render", "annotate", "validate", "render", "annotate", "validate"]


def test_snapshot_convenience_functions_delegate_to_prepared_validators(adapter):
    state, value = snapshot(), scene()
    assert adapter.module.validate_snapshot(state, value) is None
    assert adapter.calls.validations == [state]
    assert adapter.module.render_snapshot(state, value, width=320)[0] == "annotated"
    assert adapter.calls.renderer_preparations[0][1] == {"width": 320, "show_target": False}


def test_replay_view_retains_metrics_ankle_diagnostics_and_last_valid_label_without_mutation(adapter):
    value, event = scene(), step_event()
    event["snapshot"]["engine_state_status"] = "last_valid"
    before = deepcopy(event)
    result = adapter.module.replay_view(value, [event])
    assert result["scene"] == value["engine_scene"] and result["exercise_task"] == value["exercise_task"]
    view = result["events"][0]
    assert view["snapshot"] == event["snapshot"]["engine_state"]
    assert view["exercise_metrics"] == event["snapshot"]["task_metrics"]
    assert view["ankle_diagnostics"] == event["snapshot"]["ankle_diagnostics"]
    assert view["engine_state_status"] == "last_valid" and event == before
    assert "unscored ankle" in result["feet_recording_scope"] and "not physical validation" in result["feet_recording_scope"]


@pytest.fixture
def replay(adapter, monkeypatch):
    events = [{"index": 0, "type": "start", "seed": 17, "snapshot": snapshot()}, step_event(),
              {"index": 2, "type": "reset", "seed": 29, "snapshot": snapshot()},
              {**step_event([.1] * 25, "authored"), "index": 3}]
    events[2]["snapshot"]["resolved_reset_options"] = {"jitter": True, "jitter_scale": .25}
    loaded = {"scene": scene(), "events": events}
    calls, created, load_calls = [], [], []

    class Env:
        def __init__(self, engine, task):
            self.evaluator_sha256 = EVALUATOR
            self.index, self.closed = -1, 0
            self.state = None
            self.raise_on = self.wrong_state = self.wrong_outcome = self.wrong_authored = None
            created.append((engine, task, self))

        def maybe_raise(self, name):
            if self.raise_on == name:
                raise RuntimeError("mock " + name + " failure")

        def reset(self, *, seed, options):
            self.maybe_raise("reset")
            self.index += 1
            calls.append(("reset", seed, deepcopy(options)))
            self.state = deepcopy(events[self.index]["snapshot"])

        def authored_action(self):
            self.maybe_raise("authored")
            calls.append(("authored",))
            action = events[self.index + 1]["action"] if self.wrong_authored is None else self.wrong_authored
            return SimpleNamespace(tolist=lambda: deepcopy(action))

        def step(self, action):
            self.maybe_raise("step")
            self.index += 1
            calls.append(("step", deepcopy(action)))
            event = events[self.index]
            self.state = deepcopy(event["snapshot"])
            outcome = tuple(event[key] for key in ("reward", "terminated", "truncated"))
            if self.wrong_outcome is not None:
                outcome = self.wrong_outcome
            return ("mock_observation", *outcome, {})

        def snapshot(self):
            self.maybe_raise("snapshot")
            state = deepcopy(self.state)
            if self.wrong_state is not None and self.index == 1:
                path, value = self.wrong_state
                replace_at(state, path, value)
            return state

        def close(self):
            self.closed += 1
            calls.append(("close",))

    def load(directory, *, deadline):
        load_calls.append((directory, deadline))
        return loaded

    monkeypatch.setattr(adapter.module, "SharedExerciseFeetTaskEnv", Env)
    monkeypatch.setattr(adapter.loader, "load_recording", load)
    monkeypatch.setattr(adapter.module, "time", SimpleNamespace(monotonic=lambda: 100.))

    def configure(**changes):
        def build(engine, task):
            env = Env(engine, task)
            for name, value in changes.items():
                setattr(env, name, value)
            return env
        monkeypatch.setattr(adapter.module, "SharedExerciseFeetTaskEnv", build)

    return SimpleNamespace(module=adapter.module, loader=adapter.loader, loaded=loaded, events=events,
                           calls=calls, created=created, load_calls=load_calls, configure=configure)


def test_exact_mock_replay_uses_reset_options_actions_scores_and_frozen_identity_receipt(replay):
    result = replay.module.verify_action_replay("mock-only-recording", max_seconds=5)
    value = replay.loaded["scene"]
    assert replay.load_calls == [("mock-only-recording", 105.)]
    assert replay.calls == [("reset", 17, {"jitter": False, "jitter_scale": 0.}),
                            ("step", replay.events[1]["action"]),
                            ("reset", 29, {"jitter": True, "jitter_scale": .25}), ("authored",),
                            ("step", replay.events[3]["action"]), ("close",)]
    assert replay.created[0][:2] == (value["engine_scene"], value["exercise_task"])
    assert replay.created[0][2].closed == 1
    assert result["events"] == 4 and result["steps"] == 2 and result["exact_state_replay"] is True
    assert result["scene_sha256"] == canonical_hash(value["engine_scene"])
    assert result["task_sha256"] == canonical_hash(value["exercise_task"])
    for field in ("base_scene_sha256", "base_task_sha256", "projection_contract_sha256", "adapter_implementation_sha256",
                  "foot_reference_sha256"):
        assert result[field] == value["exercise_task"][field]
    assert "not physical validation" in result["scope"]


@pytest.mark.parametrize("bound", [True, 0, -1, 181, float("nan"), float("inf"), "5"])
def test_invalid_replay_deadline_is_rejected_before_load_or_env(replay, bound):
    with pytest.raises(ValueError, match="positive bound"):
        replay.module.verify_action_replay("mock", max_seconds=bound)
    assert replay.load_calls == replay.created == []


@pytest.mark.parametrize("change", ["profile", "empty"])
def test_wrong_profile_or_empty_recording_never_constructs_env(replay, change):
    if change == "profile":
        replay.loaded["scene"]["profile"] = "shared-exercise-recording-v1"
    else:
        replay.loaded["events"] = []
    with pytest.raises(ValueError, match="nonempty feet"):
        replay.module.verify_action_replay("mock")
    assert not replay.created


def test_loader_failure_never_constructs_env(replay, monkeypatch):
    def failed_load(*_args, **_kwargs):
        raise ValueError("mock pinned journal rejected")
    monkeypatch.setattr(replay.loader, "load_recording", failed_load)
    with pytest.raises(ValueError, match="journal rejected"):
        replay.module.verify_action_replay("mock")
    assert not replay.created


def test_changed_base_evaluator_fails_before_reset_and_closes_env(replay):
    replay.configure(evaluator_sha256="f" * 64)
    with pytest.raises(ValueError, match="base evaluator differs"):
        replay.module.verify_action_replay("mock")
    assert replay.calls == [("close",)] and replay.created[0][2].closed == 1


def test_authored_label_cannot_replay_a_different_action(replay):
    replay.configure(wrong_authored=[.2] * 25)
    with pytest.raises(ValueError, match="authored action differs at event 3"):
        replay.module.verify_action_replay("mock")
    assert sum(call[0] == "step" for call in replay.calls) == 1
    assert replay.calls[-2:] == [("authored",), ("close",)] and replay.created[0][2].closed == 1


@pytest.mark.parametrize("outcome", [(.4, False, False), (.5, True, False), (.5, False, True)])
def test_replay_reward_and_termination_mismatch_close_env(replay, outcome):
    replay.configure(wrong_outcome=outcome)
    with pytest.raises(ValueError, match="outcome differs at event 1"):
        replay.module.verify_action_replay("mock")
    assert replay.calls[-1] == ("close",) and replay.created[0][2].closed == 1


@pytest.mark.parametrize("path,value", [
    (("engine_state", "time_s"), .03), (("task_metrics", "reward_integral"), .99),
    (("ankle_diagnostics", "latest_substep", "unscored_effort"), .99), (("engine_state_status",), "last_valid"),
])
def test_full_state_replay_includes_base_scores_unscored_ankles_and_failure_label(replay, path, value):
    replay.configure(wrong_state=(path, value))
    with pytest.raises(ValueError, match="full state/base score replay differs at event 1"):
        replay.module.verify_action_replay("mock")
    assert replay.calls[-1] == ("close",) and replay.created[0][2].closed == 1


@pytest.mark.parametrize("operation", ["reset", "step", "authored", "snapshot"])
def test_replay_always_closes_constructed_env_after_dependency_failure(replay, operation):
    replay.configure(raise_on=operation)
    with pytest.raises(RuntimeError, match="mock " + operation + " failure"):
        replay.module.verify_action_replay("mock")
    assert replay.calls[-1] == ("close",) and replay.created[0][2].closed == 1


@pytest.mark.parametrize("terminal", [False, True])
def test_deadline_before_work_and_after_final_snapshot_fails_closed(replay, monkeypatch, terminal):
    ticks = iter([100.] + ([100.] * len(replay.events) if terminal else []) + [105.])
    monkeypatch.setattr(replay.module, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(TimeoutError, match="wall-time bound"):
        replay.module.verify_action_replay("mock", max_seconds=5)
    assert replay.created[0][2].closed == 1 and replay.calls[-1] == ("close",)
    assert sum(call[0] == "step" for call in replay.calls) == (2 if terminal else 0)
