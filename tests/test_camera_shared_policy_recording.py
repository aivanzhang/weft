"""Exact external-policy decisions and journal replay with analytical dynamics only."""

from copy import deepcopy

import numpy as np
import pytest

from test_camera_shared_task import analytical_only, _deny
from test_camera_shared_policy import environment
from test_camera_shared_learning import boundary
from test_camera_shared_recording import _reset, _step
from test_camera_shared_policy_admission import fixture_admission
from weft import camera_shared_policy_admission as admission
from weft import camera_shared_policy_recording as adapter
from weft import camera_shared_recording as original
from weft import camera_shared_task as task_api
from weft import recording, shared_world
from weft.io import canonical_hash


def setup(roots=False):
    fixture = environment(roots)
    env = task_api.CameraSharedTaskEnv(fixture.scene, fixture.task)
    document, admitted = fixture_admission(env)
    envelope = adapter.recording_scene(env.scene, env.task, policy_document=document, policy_admission=admitted)
    sampler = admission.prepare_camera_policy_sampler(document, admitted, env)
    return env, envelope, sampler


def learned(env, scene, sampler, index):
    action, sample = sampler()
    event = _step(env, index, action=action)
    event.update(controller="learned", policy_sha256=scene["learned_policy"]["sha256"],
                 policy_admission_sha256=canonical_hash(scene["policy_admission"]), policy_sample=sample)
    return event


def sequence(roots=False):
    env, scene, sample = setup(roots)
    events = [_reset(env)]
    manual = np.zeros(env.action_count)
    manual[0] = .1
    events.extend((_step(env, 1, action=manual), _step(env, 2, controller="coast")))
    events.append(learned(env, scene, sample, 3))
    events.append(_reset(env, 4, seed=19, kind="reset"))
    events.append(_step(env, 5, controller="authored"))
    events.extend(learned(env, scene, sample, index) for index in (6, 7))
    return env, scene, events


@pytest.mark.parametrize("roots", [False, True])
def test_real_journal_preserves_transitions_policy_deltas_reset_and_exact_replay(tmp_path, roots):
    env, scene, events = sequence(roots)
    validate = adapter.prepare_snapshot_validator(scene)
    for event in events:
        validate(event["snapshot"])
        adapter.validate_event(event, scene)
    recorder = recording.Recorder(tmp_path / "session", scene, "shared-world")
    recorder.start(events[0]["snapshot"], seed=events[0]["seed"])
    for event in events[1:]:
        recorder.append({key: value for key, value in event.items() if key != "index"})
    recorder.close()
    loaded = recording.load_recording(tmp_path / "session")
    assert loaded["events"] == events and loaded["scene"] == scene
    proof = adapter.verify_action_replay(tmp_path / "session")
    assert proof["protocol"] == "camera-shared-policy-full-state-replay-v1"
    assert proof["exact_state_replay"] is True and proof["events"] == 8 and proof["steps"] == 6
    assert proof["learned_actions_verified"] == 3
    assert proof["policy_sha256"] == scene["learned_policy"]["sha256"]
    assert proof["policy_admission_sha256"] == canonical_hash(scene["policy_admission"])
    view = adapter.replay_view(scene, events)
    assert view["camera_policy_admission"] == scene["policy_admission"]
    assert view["camera_policy_summary"]["learning_qualification"] == "unqualified_research"
    assert view["camera_identity"] == original.replay_view(scene, events)["camera_identity"]
    assert view["events"][3]["policy_sample"] == events[3]["policy_sample"]
    assert scene["engine_scene"]["learned_policy_available"] is False
    assert scene["exercise_task"]["scope"]["learning"] == "unqualified"
    env.close()


def test_old_recording_stays_policy_free_and_new_envelope_rejects_aliases():
    env, scene, _ = setup()
    raw = scene["learned_policy"]["document_utf8"].encode()
    with pytest.raises(ValueError, match="policy"):
        original.recording_scene(env.scene, env.task, policy_document=raw)
    with pytest.raises(ValueError):
        original.validate_recording_scene(scene)
    for field, value in (("profile", original.PROFILE), ("avatar", {}), ("dt", 1 / 30),
                         ("objects", [{"id": "invented"}])):
        changed = deepcopy(scene)
        changed[field] = value
        with pytest.raises(ValueError):
            adapter.validate_recording_scene(changed)
    changed = deepcopy(scene)
    changed["learned_policy"]["document_utf8"] += " "
    with pytest.raises(ValueError, match="bytes"):
        adapter.validate_recording_scene(changed)
    env.close()


def test_learned_metadata_cannot_leak_to_manual_or_rewrite_roots_delta_or_identity():
    env, scene, events = sequence(True)
    source = events[3]
    mutations = (
        lambda row: row.update(controller="manual"),
        lambda row: row.update(controller="reference_proposal"),
        lambda row: row.update(policy_sha256="0" * 64),
        lambda row: row.update(policy_admission_sha256="0" * 64),
        lambda row: row["policy_sample"].update(phase=float("nan")),
        lambda row: row["policy_sample"]["applied_joint_delta"].__setitem__(0, .1),
        lambda row: row["policy_sample"]["authored_action"].__setitem__(-1, .1),
        lambda row: row.pop("policy_sample"),
    )
    for mutation in mutations:
        changed = deepcopy(source)
        mutation(changed)
        with pytest.raises(ValueError):
            adapter.validate_event(changed, scene)
    manual = deepcopy(events[1])
    manual["policy_sample"] = deepcopy(source["policy_sample"])
    with pytest.raises(ValueError):
        adapter.validate_event(manual, scene)
    env.close()


@pytest.mark.parametrize("field", ["phase", "action", "reward", "snapshot"])
def test_replay_rejects_valid_shape_forged_decisions_and_full_state_and_closes(monkeypatch, field):
    env, scene, original_events = sequence()
    events = deepcopy(original_events)
    event = events[3]
    if field == "phase":
        event["policy_sample"]["phase"] = .5
    elif field == "action":
        event["action"][0] += .001
        event["snapshot"]["last_action"][0] += .001
        event["policy_sample"]["applied_joint_delta"][0] = event["action"][0] - event["policy_sample"]["authored_action"][0]
    elif field == "reward":
        event["reward"] += .001
    else:
        event["snapshot"]["task_metrics"]["mean_reward"] += .001
    monkeypatch.setattr(recording, "load_recording", lambda *args, **kwargs: {"scene": scene, "events": events})
    instances = []

    def create(*args):
        actual = task_api.CameraSharedTaskEnv(*args)
        instances.append(actual)
        return actual

    monkeypatch.setattr(adapter, "CameraSharedTaskEnv", create)
    with pytest.raises(ValueError, match="differs"):
        adapter.verify_action_replay("unused")
    assert len(instances) == 1 and instances[0]._engine.closed
    env.close()


def test_manual_terminal_failure_is_retained_with_zero_learned_actions(monkeypatch):
    env, scene, _ = setup()
    events = [_reset(env), _step(env, 1, action=np.full(env.action_count, .9))]
    final = events[-1]["snapshot"]
    assert final["task_metrics"]["failure"] == "human_position_envelope_exceeded"
    assert final["engine_state"]["partial_action"]["completed_substeps"] == 3
    adapter.prepare_snapshot_validator(scene)(final)
    monkeypatch.setattr(recording, "load_recording", lambda *args, **kwargs: {"scene": scene, "events": events})
    result = adapter.verify_action_replay("unused")
    assert result["steps"] == 1 and result["learned_actions_verified"] == 0 and result["exact_state_replay"]
    env.close()


def test_renderer_prepares_once_and_hot_frames_do_not_hash_or_prepare(monkeypatch):
    env, scene, _ = setup()
    snapshot = _reset(env)["snapshot"]
    calls = []

    def prepare(actual, *, width, show_target):
        calls.append((actual, width, show_target))
        return lambda state: np.zeros((10, width, 3), dtype=np.uint8)

    monkeypatch.setattr(shared_world, "prepare_shared_renderer", prepare)
    render = adapter.prepare_renderer(scene, width=64)
    monkeypatch.setattr(adapter, "validate_recording_scene", _deny)
    monkeypatch.setattr(adapter, "prepare_snapshot_validator", _deny)
    monkeypatch.setattr(admission, "_runtime_sources", _deny)
    monkeypatch.setattr(shared_world, "canonical_hash", _deny)
    assert np.array_equal(render(snapshot), render(snapshot))
    assert len(calls) == 1 and calls[0] == (scene["engine_scene"], 64, False)
    env.close()


def test_replay_deadline_prevents_engine_creation(monkeypatch):
    env, scene, events = sequence()
    env.close()
    monkeypatch.setattr(recording, "load_recording", lambda *args, **kwargs: {"scene": scene, "events": events})
    clock = iter((1., 3.))
    monkeypatch.setattr(adapter.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(adapter, "CameraSharedTaskEnv", _deny)
    with pytest.raises(TimeoutError, match="before engine"):
        adapter.verify_action_replay("unused", max_seconds=1)
