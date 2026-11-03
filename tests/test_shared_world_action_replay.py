"""Analytic simulated journals, not qualified source reconstruction or physical motion."""

from copy import deepcopy
import json

import pytest

from weft.io import canonical_hash
from weft.recording import Recorder, load_recording
from weft.shared_world import SharedWorldEnv
from weft.shared_world_lab import authored_shared_world_scene
from weft.shared_world_action_replay import METADATA_KEY, recording_contract, validate_contract, verify_action_replay


def journal(tmp_path, *, native=False, resets=False, actor="human"):
    if native:
        from test_dynamic_native_interval import dynamic_scene
        scene = dynamic_scene()
    else:
        scene = authored_shared_world_scene()
    scene["learned_policy_available"] = False
    env = SharedWorldEnv(scene)
    try:
        env.select_target(actor)
        scene = env.scene
        contract = recording_contract(scene, source_run_manifest_sha256="a" * 64,
            motion_ledger_sha256="b" * 64, seed=1729)
        env.reset(seed=1729, options={"jitter": False})
        recorder = Recorder(tmp_path / "session", scene, profile="shared-world", shared_world_action_replay_contract=contract)
        recorder.start(env.snapshot(), seed=1729)
        for index in range(3):
            action = [0.] * len(env.action_names)
            action[-3] = .01 * index
            _, reward, terminated, truncated, _ = env.step(action)
            recorder.append({"type": "step", "action": action, "reward": reward,
                "terminated": terminated, "truncated": truncated, "snapshot": env.snapshot(), "controller": "manual"})
        if resets:
            env.reset(seed=19, options={"jitter": False})
            recorder.append({"type": "reset", "seed": 19, "snapshot": env.snapshot()})
        recorder.close()
        return recorder.root, scene, contract
    finally:
        env.close()


def rewrite(root, events=None, metadata=None):
    if events is not None:
        data = b"".join((json.dumps(event) + "\n").encode() for event in events)
        (root / "events.jsonl").write_bytes(data)
        meta = json.loads((root / "recording.json").read_bytes())
        meta.update(events=len(events), journal_bytes=len(data))
        (root / "recording.json").write_text(json.dumps(meta))
    if metadata is not None:
        (root / "recording.json").write_text(json.dumps(metadata))


@pytest.mark.parametrize("native,actor,resets", [(False, "human", True), (False, "ball", False), (True, "human", False)])
def test_real_recorder_manual_reset_and_partial_native_replay(tmp_path, native, actor, resets):
    root, scene, contract = journal(tmp_path, native=native, resets=resets, actor=actor)
    proof = verify_action_replay(root, expected_contract_sha256=canonical_hash(contract))
    assert proof["exact_state_replay"] and proof["steps"] == 3 and proof["resets"] == resets
    assert proof["controlled_id"] == actor and proof["expected_contract_pin_supplied"]
    assert proof["source_admission_verified"] is False and proof["task_success_available"] is False
    assert proof["qualification"] == "caller_pinned_simulation_replay"
    if native:
        final = json.loads((root / "events.jsonl").read_text().splitlines()[-1])["snapshot"]
        assert final["time_s"] == pytest.approx(.075) and final["truncated"]
    assert load_recording(root)["metadata"][METADATA_KEY] == contract


def test_unpinned_replay_is_only_self_consistency(tmp_path):
    root, _, _ = journal(tmp_path)
    proof = verify_action_replay(root)
    assert proof["qualification"] == "unbound_self_consistency" and not proof["source_admission_verified"]


@pytest.mark.parametrize("field", ["scene_sha256", "source_binding_sha256", "native_interval_sha256", "controlled_id",
    "motion_ledger_sha256", "source_run_manifest_sha256", "runtime_sha256", "runtime_versions", "initial_seed"])
def test_contract_damage_rejected(tmp_path, field):
    root, _, contract = journal(tmp_path)
    metadata = json.loads((root / "recording.json").read_bytes())
    damaged = metadata[METADATA_KEY]
    if field == "runtime_sha256":
        damaged[field]["shared_world.py"] = "f" * 64
    elif field == "runtime_versions":
        damaged[field]["numpy"] = "wrong"
    elif field == "initial_seed":
        damaged[field] += 1
    else:
        damaged[field] = "f" * 64
    rewrite(root, metadata=metadata)
    with pytest.raises(ValueError):
        verify_action_replay(root, expected_contract_sha256=canonical_hash(contract))


@pytest.mark.parametrize("change", ["action", "state", "reward", "flag", "controller", "actor", "order", "bool_action", "extra"])
def test_resealed_journal_damage_rejected(tmp_path, change):
    root, _, contract = journal(tmp_path)
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    row = events[1]
    if change == "action": row["action"][-3] = .1
    elif change == "state": row["snapshot"]["time_s"] += .001
    elif change == "reward": row["reward"] += .01
    elif change == "flag": row["truncated"] = not row["truncated"]
    elif change == "controller": row["controller"] = "reference"
    elif change == "actor": row["snapshot"]["controlled_id"] = "ball"
    elif change == "order": row["index"] = 2
    elif change == "bool_action": row["action"][0] = False
    else: row["policy_sha256"] = "a" * 64
    rewrite(root, events=events)
    with pytest.raises(ValueError):
        verify_action_replay(root, expected_contract_sha256=canonical_hash(contract))


def test_default_recorder_has_no_opt_in_and_midrun_start_refused(tmp_path):
    scene = authored_shared_world_scene()
    scene["learned_policy_available"] = False
    contract = recording_contract(scene, source_run_manifest_sha256="a" * 64, motion_ledger_sha256="b" * 64)
    scene["learned_policy_available"] = False
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0, options={"jitter": False})
        legacy = Recorder(tmp_path / "legacy", scene, profile="shared-world")
        legacy.start(env.snapshot()); legacy.close()
        assert METADATA_KEY not in json.loads((legacy.root / "recording.json").read_bytes())
        env.step([0.] * len(env.action_names))
        recorder = Recorder(tmp_path / "midrun", scene, profile="shared-world", shared_world_action_replay_contract=contract)
        with pytest.raises(ValueError, match="reset-bound"):
            recorder.start(env.snapshot())
        assert not recorder.root.exists()
    finally:
        env.close()


def test_manual_recorder_refuses_implicit_controller_before_write(tmp_path):
    scene = authored_shared_world_scene()
    scene["learned_policy_available"] = False
    contract = recording_contract(scene, source_run_manifest_sha256="a" * 64, motion_ledger_sha256="b" * 64)
    scene["learned_policy_available"] = False
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0, options={"jitter": False})
        recorder = Recorder(tmp_path / "manual", scene, profile="shared-world", shared_world_action_replay_contract=contract)
        recorder.start(env.snapshot())
        action = [0.] * len(env.action_names)
        _, reward, terminated, truncated, _ = env.step(action)
        with pytest.raises(ValueError, match="exact manual event fields"):
            recorder.append({"type": "step", "action": action, "reward": reward, "terminated": terminated,
                "truncated": truncated, "snapshot": env.snapshot()})
        assert recorder.summary()["events"] == 1
        recorder.close()
    finally:
        env.close()


def test_closed_contract_rejects_policy_jitter_and_unknown_fields():
    scene = authored_shared_world_scene()
    scene["learned_policy_available"] = False
    contract = recording_contract(scene, source_run_manifest_sha256="a" * 64, motion_ledger_sha256="b" * 64)
    for field, value in [("controller", "learned"), ("reset_options", {"jitter": 0}), ("extra", None)]:
        changed = deepcopy(contract); changed[field] = value
        with pytest.raises(ValueError): validate_contract(changed, scene)


def test_bounds_and_stale_runtime_fail_before_env(tmp_path, monkeypatch):
    root, _, contract = journal(tmp_path)
    import weft.shared_world as model
    monkeypatch.setattr(model, "SharedWorldEnv", lambda *_: pytest.fail("must not construct an environment"))
    with pytest.raises(ValueError, match="step limit"):
        verify_action_replay(root, max_steps=0)
    import weft.shared_world_action_replay as replay
    monkeypatch.setattr(replay, "runtime_identity", lambda _: {})
    with pytest.raises(ValueError, match="runtime differs"):
        verify_action_replay(root, expected_contract_sha256=canonical_hash(contract))


def test_contract_scene_mutation_refused_before_initial_write(tmp_path):
    scene = authored_shared_world_scene(); scene["learned_policy_available"] = False
    contract = recording_contract(scene, source_run_manifest_sha256="a" * 64, motion_ledger_sha256="b" * 64)
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0, options={"jitter": False})
        recorder = Recorder(tmp_path / "changed", scene, profile="shared-world", shared_world_action_replay_contract=contract)
        recorder.scene["target"][0] += .01
        with pytest.raises(ValueError, match="identity changed"):
            recorder.start(env.snapshot())
        assert not recorder.root.exists()
    finally:
        env.close()


@pytest.mark.parametrize("bad", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_duplicate_nonfinite_metadata_rejected_before_engine(tmp_path, monkeypatch, bad):
    root, _, _ = journal(tmp_path)
    (root / "recording.json").write_text(bad)
    import weft.shared_world as model
    monkeypatch.setattr(model, "SharedWorldEnv", lambda *_: pytest.fail("must not construct an environment"))
    with pytest.raises(ValueError): verify_action_replay(root)


def test_grid_aligned_native_replay_pins_clock_without_changing_legacy_closure():
    from test_dynamic_native_interval import dynamic_scene
    from weft.shared_world_lab import _lab_runtime_modules
    from weft.shared_world_action_replay import runtime_identity
    scene = dynamic_scene(); scene["learned_policy_available"] = False
    scene["task"].pop("native_interval")
    for name in ("reference", "root_reference"):
        for index, row in enumerate(scene["human_spec"][name]):
            row["time_s"] = index / 30
    from weft.shared_world import validate_shared_world
    scene = validate_shared_world(scene)
    assert "source_clock.py" not in _lab_runtime_modules(scene)
    pins = runtime_identity(scene)["runtime_sha256"]
    assert "source_clock.py" in pins and "dynamic_native_interval.py" not in pins
    contract = recording_contract(scene, source_run_manifest_sha256="a" * 64, motion_ledger_sha256="b" * 64)
    contract["runtime_sha256"].pop("source_clock.py")
    with pytest.raises(ValueError, match="closure"):
        validate_contract(contract, scene)


def test_pure_replay_module_closure_matches_generic_producer_plus_explicit_dependencies():
    from test_dynamic_native_interval import dynamic_scene
    from weft.shared_world_lab import _lab_runtime_modules
    from weft.shared_world_action_replay import _runtime_modules
    for scene in (authored_shared_world_scene(), dynamic_scene()):
        expected = set(_lab_runtime_modules(scene)) | {"shared_world_action_replay.py", "recording.py", "io.py"}
        if "native_clock" in scene["human_spec"]: expected.add("source_clock.py")
        assert _runtime_modules(scene) == expected
