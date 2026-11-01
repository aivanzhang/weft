"""Real journals and camera scoring with analytical dynamics, never native models."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from test_camera_reference_controller import api, runtime, fixture, prepare, encode
from test_camera_shared_recording import RecordingEngine, _reset, _step
from test_camera_shared_task import analytical_only, _deny
from weft import camera_reference_recording as adapter
from weft import camera_shared_recording as base
from weft import camera_shared_task as task_api
from weft import recording, shared_world


class ReferenceEngine(RecordingEngine):
    def authored_human_action(self):
        return np.clip(self.feature_matrix().sum(axis=1), -1., 1.)


@pytest.fixture
def build(api, runtime, analytical_only, monkeypatch):
    monkeypatch.setattr(task_api, "SharedWorldEnv", ReferenceEngine)
    monkeypatch.setattr(adapter, "_trusted_velocity_index", lambda: runtime[1])
    package = Path(adapter.__file__).parent
    hashes = {name: sha256((package / name).read_bytes()).hexdigest() for name in adapter.REQUIRED_RUNTIME_FILES}
    extended = (hashes, *runtime[1:])
    live = []

    def context(*, roots=False, changed=True):
        data = fixture(api, extended, root=roots, corrections=(0., .02, .01, .03) if changed else (0.,) * 4)
        data["scene"]["bounds"] = [6., 6., 6.]
        data["scene"]["gravity"] = [0., 0., -9.81]
        data["task"] = task_api.build_camera_shared_task(data["scene"], root_assistance=roots)
        data["action"] = api.camera_reference_action_contract(data["scene"], data["task"])
        provider = prepare(api, extended, data)
        document = encode(provider.recipe())
        scene = adapter.recording_scene(data["scene"], data["task"], reference_controller_document=document)
        env = task_api.CameraSharedTaskEnv(data["scene"], data["task"])
        live.append(env)
        sampler = adapter.prepare_recorded_reference_controller(scene["reference_controller"], env)
        return env, scene, sampler, document

    yield context
    for env in live:
        env.close()


def proposal(env, scene, sampler, index):
    action = sampler()
    assert action.tolist()[19:] == env.authored_action().tolist()[19:]
    row = _step(env, index, action=action, controller="reference_proposal")
    row["reference_controller_sha256"] = scene["reference_controller"]["sha256"]
    return row


def sequence(build, roots=False):
    env, scene, sampler, document = build(roots=roots)
    rows = [_reset(env)]
    rows.append(_step(env, 1, action=np.zeros(env.action_count)))
    rows.append(proposal(env, scene, sampler, 2))
    rows.append(_reset(env, 3, seed=19, kind="reset"))
    rows.append(_step(env, 4, controller="authored"))
    rows.append(proposal(env, scene, sampler, 5))
    rows.append(_step(env, 6, controller="coast"))
    return env, scene, rows, document


def save(tmp_path, scene, rows):
    root = tmp_path / "session"
    recorder = recording.Recorder(root, scene, "shared_exercise")
    recorder.start(rows[0]["snapshot"], seed=rows[0]["seed"])
    for row in rows[1:]:
        recorder.append({k: v for k, v in row.items() if k != "index"})
    recorder.close()
    return root


@pytest.mark.parametrize("roots", [False, True])
def test_real_journal_replays_exact_proposal_actions_state_reset_and_roots(build, tmp_path, roots):
    env, scene, rows, document = sequence(build, roots)
    original = deepcopy((scene, rows))
    root = save(tmp_path, scene, rows)
    loaded = recording.load_recording(root)
    assert loaded["scene"] == scene and loaded["events"] == rows
    assert adapter.validate_recording_scene(scene) == scene
    proof = adapter.verify_action_replay(root)
    assert proof["protocol"] == adapter.PROTOCOL and proof["exact_state_replay"] is True
    assert (proof["events"], proof["steps"], proof["reference_actions_verified"]) == (7, 5, 2)
    assert proof["reference_controller_sha256"] == sha256(document).hexdigest()
    assert proof["learned_actions_verified"] is None and proof["policy_sha256"] is None
    view = adapter.replay_view(scene, rows)
    assert view["camera_identity"] == base.replay_view(scene, rows)["camera_identity"]
    assert view["events"][2]["controller"] == "reference_proposal"
    assert view["events"][2]["reference_controller_sha256"] == sha256(document).hexdigest()
    assert view["reference_controller"]["kind"] == "source_fit_authored_controller"
    assert (scene, rows) == original and scene["engine_scene"]["learned_policy_available"] is False


@pytest.mark.parametrize("roots", [False, True])
def test_unchanged_overlay_matches_actual_camera_authored_control(build, roots):
    env, _, sampler, _ = build(roots=roots, changed=False)
    env.reset(seed=17)
    for _ in range(3):
        action = sampler()
        assert action.tolist() == env.authored_action().tolist()
        env.step(action)


@pytest.mark.parametrize("change", ["mode", "pin", "missing_pin", "policy", "extra", "start", "reset"])
def test_proposal_metadata_cannot_change_mode_or_bypass_event_allowlists(build, change):
    _, scene, rows, _ = sequence(build)
    row = deepcopy(rows[2])
    if change == "mode": row["controller"] = "manual"
    elif change == "pin": row["reference_controller_sha256"] = "0" * 64
    elif change == "missing_pin": row.pop("reference_controller_sha256")
    elif change == "policy": row["controller"] = "learned"
    elif change == "extra": row["policy_sample"] = {}
    else:
        row = deepcopy(rows[0 if change == "start" else 3])
        row.update(controller="reference_proposal", reference_controller_sha256=scene["reference_controller"]["sha256"])
    with pytest.raises(ValueError):
        adapter.validate_event(row, scene)


@pytest.mark.parametrize("change", ["runtime", "missing_runtime", "path", "sha", "body", "task", "profile", "policy"])
def test_envelope_rejects_changed_inputs_before_native_admission(build, monkeypatch, change):
    _, scene, _, _ = build()
    mutated = deepcopy(scene)
    embedded = mutated["reference_controller"]
    if change == "runtime": embedded["implementation_sha256"]["human.py"] = "f" * 64
    elif change == "missing_runtime":
        recipe = json.loads(embedded["document_utf8"])
        recipe["runtime_sha256"].pop("shared_exercise_reset.py")
        document = encode(recipe)
        embedded.update(document_utf8=document.decode(), sha256=sha256(document).hexdigest())
    elif change == "path":
        recipe = json.loads(embedded["document_utf8"])
        recipe["runtime_sha256"]["../outside.py"] = "f" * 64
        document = encode(recipe)
        embedded.update(document_utf8=document.decode(), sha256=sha256(document).hexdigest())
    elif change == "sha": embedded["sha256"] = "f" * 64
    elif change == "body": mutated["engine_scene"]["human_spec"]["dt"] *= 2
    elif change == "task": mutated["exercise_task"]["gates_sha256"] = "f" * 64
    elif change == "profile": mutated["profile"] = base.PROFILE
    else: mutated["learned_policy"] = {}
    monkeypatch.setattr(adapter, "prepare_camera_shared_task", _deny)
    with pytest.raises(ValueError):
        adapter.validate_recording_scene(mutated)


@pytest.mark.parametrize("field", ["action", "reward", "snapshot"])
def test_replay_detects_shape_valid_forgery_and_closes_environment(build, tmp_path, monkeypatch, field):
    _, scene, rows, _ = sequence(build)
    root = save(tmp_path, scene, rows)
    altered = deepcopy(rows)
    row = altered[2]
    if field == "action":
        row["action"][0] += .001
        row["snapshot"]["last_action"][0] += .001
    elif field == "reward": row["reward"] += .001
    else: row["snapshot"]["task_metrics"]["mean_reward"] += .001
    monkeypatch.setattr(recording, "load_recording", lambda *args, **kwargs: {"scene": scene, "events": altered})
    created = []
    def create(*args):
        env = task_api.CameraSharedTaskEnv(*args)
        created.append(env)
        return env
    monkeypatch.setattr(adapter, "CameraSharedTaskEnv", create)
    with pytest.raises(ValueError, match="differs"):
        adapter.verify_action_replay(root)
    assert len(created) == 1 and created[0]._engine.closed


def test_original_camera_recordings_remain_distinct_and_dispatch(build, tmp_path):
    env, scene, _, _ = build()
    ordinary = adapter.recording_scene(env.scene, env.task)
    assert ordinary["profile"] == base.PROFILE and "reference_controller" not in ordinary
    with pytest.raises(ValueError): base.validate_recording_scene(scene)
    with pytest.raises(ValueError): adapter.validate_recording_scene(ordinary)
    rows = [_reset(env), _step(env, 1, controller="authored")]
    root = save(tmp_path, ordinary, rows)
    assert adapter.verify_action_replay(root)["protocol"] == base.PROTOCOL


def test_prepared_action_has_no_runtime_reads_or_json_preparation(build, monkeypatch):
    env, _, sampler, _ = build(roots=True)
    env.reset(seed=7)
    monkeypatch.setattr(adapter, "_prepare", _deny)
    monkeypatch.setattr(adapter, "read_package_file", _deny)
    for _ in range(20):
        assert len(sampler()) == 22


def test_runtime_read_budget_rejects_before_oversized_next_read(build, monkeypatch):
    _, _, _, document = build()
    declarations = json.loads(document)["runtime_sha256"]
    calls = []
    def read(_root, name, *, max_bytes):
        calls.append(max_bytes)
        if len(calls) == 1: return b"0" * 17
        assert max_bytes == 15
        raise ValueError("source exceeds remaining budget")
    monkeypatch.setattr(adapter, "MAX_RUNTIME_BYTES", 32)
    monkeypatch.setattr(adapter, "read_package_file", read)
    with pytest.raises(ValueError, match="remaining budget"):
        adapter._runtime_hashes(declarations)
    assert calls == [32, 15]


def test_renderer_prepares_once_and_keeps_exact_camera_metrics(build, monkeypatch):
    env, scene, _, _ = build()
    snapshot = _reset(env)["snapshot"]
    compiled, rendered = [], []
    def prepare_render(actual, **kwargs):
        compiled.append((actual, kwargs))
        def render(state):
            rendered.append(state)
            return np.zeros((4, 4, 3), dtype=np.uint8)
        return render
    monkeypatch.setattr(shared_world, "prepare_shared_renderer", prepare_render)
    monkeypatch.setattr(base, "_annotate_frame", lambda frame, state: (frame, state["task_metrics"]))
    render = adapter.prepare_renderer(scene)
    for _ in range(3):
        frame, metrics = render(snapshot)
        assert frame.shape == (4, 4, 3) and metrics == snapshot["task_metrics"]
    assert len(compiled) == 1 and len(rendered) == 3
    assert compiled[0][0] == env.scene and compiled[0][1]["show_target"] is False
