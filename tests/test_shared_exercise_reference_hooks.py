"""Recording-hook checks use stub environments; collection may import native libraries."""

from copy import deepcopy
import json

import numpy as np
import pytest

from test_shared_exercise_reference_controller import controller_fixture, encode, prepare, refresh_hashes, velocity_index
from weft import recording as recordings
from weft import shared_exercise_policies as policies
from weft import shared_exercise_recording as hooks
from weft import shared_exercise_reference_recording as reference_recording
from weft.io import canonical_hash


@pytest.fixture
def case(monkeypatch, velocity_index):
    bundle = controller_fixture()
    bundle["scene"]["objects"] = []
    refresh_hashes(bundle)
    monkeypatch.setattr(hooks, "validate_shared_world", deepcopy)
    def validate_task(task, scene):
        assert task["scene_sha256"] == canonical_hash(scene)
        return deepcopy(task)
    monkeypatch.setattr(hooks, "validate_shared_exercise_task", validate_task)
    monkeypatch.setattr(reference_recording, "_trusted_velocity_index", lambda: velocity_index)
    provider = prepare(bundle, velocity_index)
    raw = encode(provider.recipe())
    envelope = hooks.recording_scene(bundle["scene"], bundle["task"], reference_controller_document=raw)
    return bundle, raw, envelope


def test_optional_envelope_roundtrip_preserves_original_scene_and_task(case):
    bundle, raw, envelope = case
    before = deepcopy(bundle)
    validated = hooks.validate_recording_scene(envelope)
    assert validated == envelope and validated is not envelope
    assert validated["reference_controller"]["document_utf8"].encode() == raw
    assert validated["engine_scene"] == bundle["scene"]
    assert validated["exercise_task"] == bundle["task"]
    assert bundle == before


def test_legacy_omission_does_not_load_reference_helper(case, monkeypatch):
    bundle, _, _ = case
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Legacy recordings must not require a reference recipe")
    monkeypatch.setattr(reference_recording, "reference_controller_envelope", forbidden)
    monkeypatch.setattr(reference_recording, "validate_reference_controller_envelope", forbidden)
    scene = hooks.recording_scene(bundle["scene"], bundle["task"])
    assert set(scene) == {"schema_version", "profile", "engine_scene", "exercise_task", "dt", "objects"}
    assert hooks.validate_recording_scene(scene) == scene


def test_optional_recipe_and_legacy_learned_policy_can_coexist(case, monkeypatch):
    bundle, raw, _ = case
    seen = []
    def validate_policy(document, scene, task):
        seen.append(document)
        assert scene == bundle["scene"] and task == bundle["task"]
    monkeypatch.setattr(policies, "validate_policy", validate_policy)
    policy_raw = b'{"profile":"test-policy"}'
    scene = hooks.recording_scene(bundle["scene"], bundle["task"], policy_document=policy_raw,
                                  reference_controller_document=raw)
    assert hooks.validate_recording_scene(scene) == scene
    assert scene["learned_policy"]["document_utf8"].encode() == policy_raw
    assert len(seen) == 2


@pytest.mark.parametrize("kind", ["unknown_field", "missing_field", "alias", "recipe_hash"])
def test_recording_envelope_rejects_unknown_fields_and_changed_aliases(case, kind):
    scene = deepcopy(case[2])
    if kind == "unknown_field": scene["another_controller"] = {}
    elif kind == "missing_field": del scene["objects"]
    elif kind == "alias": scene["dt"] = .5
    else: scene["reference_controller"]["sha256"] = "d" * 64
    with pytest.raises(ValueError):
        hooks.validate_recording_scene(scene)


def step_event(scene):
    action = [0.] * 22
    return {"index": 1, "type": "step", "controller": "reference_proposal",
            "reference_controller_sha256": scene["reference_controller"]["sha256"],
            "action": action, "reward": .5, "terminated": False, "truncated": False,
            "snapshot": {"last_action": list(action), "terminated": False, "truncated": False}}


def test_reference_event_accepts_exact_recipe_hash(case):
    scene = case[2]
    event = step_event(scene)
    before = deepcopy(event)
    hooks.validate_event(event, scene)
    assert event == before


@pytest.mark.parametrize("kind", ["missing_hash", "wrong_hash", "hash_list", "hash_none", "hash_on_manual",
    "hash_on_start", "no_envelope", "policy_hash_on_reference", "hash_without_controller", "unknown_controller"])
def test_reference_event_requires_step_source_and_exact_hash(case, kind):
    scene = deepcopy(case[2])
    event = step_event(scene)
    if kind == "missing_hash": del event["reference_controller_sha256"]
    elif kind == "wrong_hash": event["reference_controller_sha256"] = "d" * 64
    elif kind == "hash_list": event["reference_controller_sha256"] = [scene["reference_controller"]["sha256"]]
    elif kind == "hash_none": event["reference_controller_sha256"] = None
    elif kind == "hash_on_manual": event["controller"] = "manual"
    elif kind == "hash_on_start": event["type"] = "start"
    elif kind == "no_envelope": del scene["reference_controller"]
    elif kind == "policy_hash_on_reference": event["policy_sha256"] = "d" * 64
    elif kind == "hash_without_controller": del event["controller"]
    else: event["controller"] = "proposal-ish"; del event["reference_controller_sha256"]
    with pytest.raises(ValueError):
        hooks.validate_event(event, scene)


def snapshot(tick, action):
    return {"tick": tick, "last_action": list(action), "seed": 17,
            "engine_state": {"mock_tick": tick}, "engine_state_status": "current",
            "task_metrics": {"evaluator_sha256": "e" * 64},
            "resolved_reset_options": {"jitter": False, "jitter_scale": 0.},
            "terminated": tick == 2, "truncated": False}


def replay_case(case, monkeypatch, *, kind="reference_proposal"):
    scene = deepcopy(case[2])
    if kind != "reference_proposal":
        del scene["reference_controller"]
    if kind == "learned":
        scene["learned_policy"] = {"document_utf8": '{"profile":"stub-policy"}', "sha256": "f" * 64}
    actions = [[0.] * 22, [0.] * 22] if kind == "coast" else [[.01] * 22, [-.02] * 22]
    events = [{"index": 0, "type": "start", "seed": 17, "snapshot": snapshot(0, [0.] * 22)}]
    for tick, action in enumerate(actions, 1):
        event = {"index": tick, "type": "step", "controller": kind, "action": action,
                 "reward": sum(action), "terminated": tick == 2, "truncated": False, "snapshot": snapshot(tick, action)}
        if kind == "reference_proposal": event["reference_controller_sha256"] = scene["reference_controller"]["sha256"]
        if kind == "learned": event["policy_sha256"] = scene["learned_policy"]["sha256"]
        events.append(event)
    log = []
    class Environment:
        evaluator_sha256 = "e" * 64
        def __init__(self, engine_scene, task):
            assert engine_scene == case[0]["scene"] and task == case[0]["task"]
            self.scene, self.task = deepcopy(engine_scene), deepcopy(task)
            self.tick, self.closed, self.step_calls = 0, False, 0
            self.current = None
            log.append("constructed")

        def reset(self, *, seed, options):
            assert seed == 17 and options == {"jitter": False, "jitter_scale": 0.}
            self.tick = 0
            self.current = snapshot(0, [0.] * 22)
            log.append("reset")

        def step(self, action):
            self.step_calls += 1
            self.tick += 1
            self.current = snapshot(self.tick, action)
            log.append("step")
            return None, sum(action), self.tick == 2, False, {}

        def authored_action(self):
            log.append("authored_action")
            return np.asarray(actions[self.tick], dtype=float)

        def snapshot(self):
            return deepcopy(self.current)

        def close(self):
            self.closed = True
            log.append("closed")

    instances = []
    def construct(engine_scene, task):
        env = Environment(engine_scene, task)
        instances.append(env)
        return env
    def load(_directory, *, deadline):
        assert type(deadline) is float
        log.append("loaded")
        return {"scene": scene, "events": events}
    def prepare_reference(envelope, env):
        assert envelope == case[2]["reference_controller"]
        assert env.scene == case[0]["scene"] and env.task == case[0]["task"]
        log.append("reference_bound")
        def action():
            log.append("reference_action")
            return np.asarray(actions[env.tick], dtype=float)
        return action
    def prepare_policy(_document, env):
        def action():
            log.append("learned_action")
            return np.asarray(actions[env.tick], dtype=float)
        return action
    monkeypatch.setattr(recordings, "load_recording", load)
    monkeypatch.setattr(hooks, "SharedExerciseTaskEnv", construct)
    monkeypatch.setattr(reference_recording, "prepare_recorded_reference_controller", prepare_reference)
    monkeypatch.setattr(policies, "prepare_policy", prepare_policy)
    return scene, events, log, instances


def test_replay_verifies_action_before_step_and_preserves_exact_counts(case, monkeypatch):
    scene, events, log, instances = replay_case(case, monkeypatch)
    before = deepcopy((scene, events))
    result = hooks.verify_action_replay("unused-recording-path")
    assert log == ["loaded", "constructed", "reference_bound", "reset", "reference_action", "step",
                   "reference_action", "step", "closed"]
    assert result["events"] == 3 and result["steps"] == result["reference_actions_verified"] == 2
    assert result["exact_state_replay"] is True
    assert result["reference_controller_sha256"] == scene["reference_controller"]["sha256"]
    assert result["scene_sha256"] == canonical_hash(scene["engine_scene"])
    assert result["task_sha256"] == canonical_hash(scene["exercise_task"])
    assert instances[0].closed and (scene, events) == before


def test_tampered_reference_action_rejects_before_step_and_closes(case, monkeypatch):
    _, events, log, instances = replay_case(case, monkeypatch)
    # Replace the saved vector, without changing the independently generated callback vector.
    events[1]["action"] = [.2] * 22
    events[1]["snapshot"]["last_action"] = [.2] * 22
    with pytest.raises(ValueError, match="reference controller replay differs"):
        hooks.verify_action_replay("unused-recording-path")
    assert instances[0].step_calls == 0 and instances[0].closed
    assert log[-2:] == ["reference_action", "closed"]


@pytest.mark.parametrize("failure", ["missing_runtime", "evaluator", "reward", "state"])
def test_replay_failure_closes_environment_without_fallback(case, monkeypatch, failure):
    _, events, log, instances = replay_case(case, monkeypatch)
    if failure == "missing_runtime":
        def reject(*_args, **_kwargs):
            raise ValueError("missing or changed controller runtime")
        monkeypatch.setattr(reference_recording, "prepare_recorded_reference_controller", reject)
    elif failure == "evaluator": events[0]["snapshot"]["task_metrics"]["evaluator_sha256"] = "d" * 64
    elif failure == "reward": events[1]["reward"] += 1.
    else: events[1]["snapshot"]["tick"] += 10
    with pytest.raises(ValueError):
        hooks.verify_action_replay("unused-recording-path")
    assert instances[0].closed and log[-1] == "closed"
    assert instances[0].step_calls == (0 if failure in {"missing_runtime", "evaluator"} else 1)


@pytest.mark.parametrize("kind", ["manual", "coast", "authored", "learned"])
def test_legacy_replay_controllers_do_not_bind_new_provider(case, monkeypatch, kind):
    _, _, log, instances = replay_case(case, monkeypatch, kind=kind)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Legacy replay must not bind the new provider")
    monkeypatch.setattr(reference_recording, "prepare_recorded_reference_controller", forbidden)
    result = hooks.verify_action_replay("unused-recording-path")
    assert result["steps"] == 2 and result["exact_state_replay"] is True
    assert "reference_controller_sha256" not in result and "reference_actions_verified" not in result
    assert "reference_action" not in log and instances[0].closed
    if kind == "learned": assert result["learned_actions_verified"] == 2
    if kind == "authored": assert log.count("authored_action") == 2


def test_replay_view_retains_recipe_identity_and_honest_scope(case, monkeypatch):
    scene, events, _, _ = replay_case(case, monkeypatch)
    view = hooks.replay_view(scene, events)
    assert view["scene"] == scene["engine_scene"] and view["exercise_task"] == scene["exercise_task"]
    assert view["reference_controller"]["sha256"] == scene["reference_controller"]["sha256"]
    assert "not a learned policy or repaired geometry" in view["reference_controller"]["scope"]
    assert view["events"][1]["reference_controller_sha256"] == scene["reference_controller"]["sha256"]
    assert view["events"][1]["snapshot"] == events[1]["snapshot"]["engine_state"]
    assert "reference_controller" not in hooks.replay_view({k: v for k, v in scene.items() if k != "reference_controller"}, events)
