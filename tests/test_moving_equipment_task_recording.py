"""Analytic task journals: real simulation replay, not source reconstruction qualification."""

from copy import deepcopy
from hashlib import sha256
import json
import subprocess
import sys
import time

import pytest

from test_moving_equipment_task import example
from weft.io import canonical_hash
from weft.moving_equipment_task import MovingEquipmentTaskEnv
from weft.moving_equipment_task_lab import compose_task_snapshot
from weft.moving_equipment_task_recording import (TaskRecorder, recording_contract, verify_task_replay,
    export_task_recording, load_task_recording_evidence, METADATA_KEY, RECEIPT)


def state(env):
    value = env.snapshot()
    return compose_task_snapshot(value["engine_diagnostics"], value["task"])


def journal(tmp_path, example, *, steps=3, reset=False):
    scene, bundle, task = example
    contract = recording_contract(scene, task, bundle, task_lab_manifest_sha256="a"*64,
        parent_manifest_sha256="b"*64, task_lab_contract_sha256="c"*64, seed=19)
    recorder = TaskRecorder(tmp_path / "recording", scene, task, bundle, contract)
    env = MovingEquipmentTaskEnv(scene, task, bundle)
    try:
        env.reset(seed=19, options={"jitter": False})
        recorder.start(state(env), seed=19)
        for index in range(steps):
            action = [0.] * len(env.action_names)
            action[-3] = index * .0001
            _, reward, terminated, truncated, _ = env.step(action)
            recorder.append({"type": "step", "snapshot": state(env), "action": action, "reward": reward,
                "terminated": terminated, "truncated": truncated, "controller": "manual"})
            if terminated or truncated: break
        if reset:
            env.reset(seed=29, options={"jitter": False})
            recorder.append({"type": "reset", "snapshot": state(env), "seed": 29})
        recorder.close()
    finally:
        env.close()
    return recorder.root, contract


def mutate_events(root, change):
    events = [json.loads(line) for line in (root / "events.jsonl").read_bytes().splitlines()]
    change(events)
    raw = b"".join((json.dumps(event) + "\n").encode() for event in events)
    (root / "events.jsonl").write_bytes(raw)
    metadata = json.loads((root / "recording.json").read_bytes())
    metadata.update(events=len(events), journal_bytes=len(raw))
    (root / "recording.json").write_text(json.dumps(metadata))


@pytest.mark.parametrize("steps,reset", [(3, False), (14, False), (14, True)])
def test_exact_real_task_replay_partial_terminal_and_reset(tmp_path, example, steps, reset):
    root, contract = journal(tmp_path, example, steps=steps, reset=reset)
    proof = verify_task_replay(root, expected_contract_sha256=canonical_hash(contract))
    assert proof["exact_state_replay"] and proof["exact_task_reward_replay"]
    assert proof["steps"] == steps and proof["resets"] == reset
    assert proof["source_admission_verified"] is False
    assert proof["qualification"] == "caller_pinned_simulation_replay"
    assert proof["completed_interval"] == (steps == 14 and not reset)
    if steps == 14 and not reset:
        assert proof["final_episode_metrics"]["elapsed_s"] == pytest.approx(.45)
    else:
        assert not proof["success"] and proof["outcome"] == "stopped_or_terminated_partial"
    metadata = json.loads((root / "recording.json").read_bytes())
    assert "shared_world_action_replay_contract" not in metadata
    assert metadata[METADATA_KEY]["contract_sha256"] == canonical_hash(contract)


@pytest.mark.parametrize("damage", ["reward", "task_metrics", "engine_state", "action", "controller", "index", "seed", "flags", "fake_success"])
def test_changed_journal_refuses_before_or_during_exact_replay(tmp_path, example, damage):
    root, contract = journal(tmp_path, example)
    def change(events):
        event = events[1]
        if damage == "reward": event["reward"] += .1
        elif damage == "task_metrics": event["snapshot"]["task"]["mean_reward"] += .01
        elif damage == "engine_state": event["snapshot"]["engine_diagnostics"]["time_s"] += .001
        elif damage == "action": event["action"][-1] += .05
        elif damage == "controller": event["controller"] = "learned"
        elif damage == "index": event["index"] = 0
        elif damage == "seed": events[0]["seed"] += 1
        elif damage == "flags": event["truncated"] = not event["truncated"]
        else: event["snapshot"]["success"] = True
    mutate_events(root, change)
    with pytest.raises(ValueError): verify_task_replay(root, expected_contract_sha256=canonical_hash(contract))


@pytest.mark.parametrize("damage", ["task", "bundle", "family", "child", "runtime", "duplicate_json", "failed_attempt", "unclosed", "generic_contract"])
def test_contract_custody_failure_and_runtime_refusal(tmp_path, example, monkeypatch, damage):
    root, contract = journal(tmp_path, example)
    if damage in {"failed_attempt", "unclosed", "generic_contract"}:
        path = root / "recording.json"; value = json.loads(path.read_bytes())
        if damage == "failed_attempt": value["failed_attempt"] = {"reason": "unrecorded failed action"}
        elif damage == "unclosed": value["status"] = "recording"
        else: value["shared_world_action_replay_contract"] = {}
        path.write_text(json.dumps(value))
    elif damage == "duplicate_json":
        path = root / "task-recording-contract.json"
        path.write_bytes(path.read_bytes().rstrip()[:-1] + b',"profile":"moving-equipment-task-recording-v1"}\n')
    else:
        name = "placement-task.json" if damage == "task" else "moving-proposals.json" if damage in {"bundle", "family"} else "task-recording-contract.json"
        path = root / name; value = json.loads(path.read_bytes())
        if damage == "task": value["goal"]["speed_tolerance_world_s"] *= 2
        elif damage == "bundle": value["candidates"] = []
        elif damage == "family": value["depth_family"]["scope"] = "verified physical depth"
        elif damage == "child": value["task_lab_manifest_sha256"] = "d"*64
        else:
            # A changed non-scorer runtime remains structurally valid but must fail before a model exists.
            value["runtime_sha256"]["recording.py"] = "0"*64
        path.write_text(json.dumps(value))
        if damage == "runtime":
            contract = value
            meta_path = root / "recording.json"; meta = json.loads(meta_path.read_bytes())
            meta[METADATA_KEY]["contract_sha256"] = canonical_hash(value); meta_path.write_text(json.dumps(meta))
    def forbidden(*args, **kwargs): raise AssertionError("Native environment must not be created for invalid custody")
    monkeypatch.setattr("weft.moving_equipment_task.MovingEquipmentTaskEnv", forbidden)
    with pytest.raises(ValueError): verify_task_replay(root, expected_contract_sha256=canonical_hash(contract))


def test_failed_attempt_preserved_and_refused(tmp_path, example):
    scene, bundle, task = example
    contract = recording_contract(scene, task, bundle, task_lab_manifest_sha256="a"*64,
        parent_manifest_sha256="b"*64, task_lab_contract_sha256="c"*64)
    env = MovingEquipmentTaskEnv(scene, task, bundle)
    try:
        env.reset(seed=0)
        recorder = TaskRecorder(tmp_path / "failed", scene, task, bundle, contract)
        recorder.start(state(env))
        recorder.mark_failed({"reason": "runtime failed", "attempted_action": [0.] * len(env.action_names)})
        with pytest.raises(ValueError, match="failed"): recorder.check_capacity()
        recorder.close(); recorder.close()
        assert json.loads((recorder.root / "recording.json").read_bytes())["failed_attempt"]["reason"] == "runtime failed"
        with pytest.raises(ValueError, match="failed attempt"): verify_task_replay(recorder.root)
    finally:
        env.close()


def test_no_mid_episode_start_or_task_free_contract(tmp_path, example):
    scene, bundle, task = example
    root, contract = journal(tmp_path, example)
    last = json.loads((root / "events.jsonl").read_bytes().splitlines()[-1])["snapshot"]
    recorder = TaskRecorder(tmp_path / "late", scene, task, bundle, contract)
    with pytest.raises(ValueError, match="reset"): recorder.start(last, 19)
    assert not recorder.root.exists()
    broken = deepcopy(contract); broken["scope"] = "unscored exploration"
    with pytest.raises(ValueError): TaskRecorder(tmp_path / "wrong", scene, task, bundle, broken)


def test_deadline_before_native_and_symlink_refusal(tmp_path, example, monkeypatch):
    root, contract = journal(tmp_path, example)
    with pytest.raises(TimeoutError): verify_task_replay(root, deadline=time.monotonic()-1)
    path = root / "placement-task.json"
    raw = path.read_bytes(); path.unlink(); other = tmp_path / "other.json"; other.write_bytes(raw); path.symlink_to(other)
    with pytest.raises(ValueError): verify_task_replay(root)


def test_original_event_limit_and_new_replay_cap_remain_bounded(tmp_path, example, monkeypatch):
    scene, bundle, task = example
    root, contract = journal(tmp_path, example)
    monkeypatch.setattr("weft.moving_equipment_task_recording.MAX_STEPS", 2)
    with pytest.raises(ValueError, match="step bound"): verify_task_replay(root)
    env = MovingEquipmentTaskEnv(scene, task, bundle)
    try:
        env.reset(seed=19)
        snapshot = state(env)
        snapshot["task"]["oversized"] = "x" * 65536
        recorder = TaskRecorder(tmp_path / "large", scene, task, bundle, contract)
        with pytest.raises(ValueError, match="size limit"): recorder.start(snapshot, seed=19)
        assert not recorder.root.exists()
    finally: env.close()


def test_real_media_export_saved_pure_receipt_and_mutation(tmp_path, example):
    root, contract = journal(tmp_path, example)
    pin = canonical_hash(contract)
    result = export_task_recording(root, expected_contract_sha256=pin)
    assert result["proof"]["exact_task_reward_replay"] and not result["proof"]["success"]
    assert result["verification"] == "trusted_saved_receipt_not_a_new_physics_replay" and result["exact_replay_verified"]
    assert (root / "video.mp4").is_file() and (root / "replay.html").is_file()
    script = '''
import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'mujoco','numpy','gymnasium','cv2','PIL'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Deny())
from weft.moving_equipment_task_recording import load_task_recording_evidence
result=load_task_recording_evidence(sys.argv[1],expected_contract_sha256=sys.argv[2],expected_receipt_sha256=sys.argv[3])
assert result['proof']['exact_task_reward_replay']
'''
    child = subprocess.run([sys.executable, "-c", script, str(root), pin, result["receipt_sha256"]], capture_output=True, text=True, timeout=20)
    assert child.returncode == 0, child.stderr
    assert export_task_recording(root, expected_contract_sha256=pin,
        expected_receipt_sha256=result["receipt_sha256"])["receipt_sha256"] == result["receipt_sha256"]
    unbound = load_task_recording_evidence(root, expected_contract_sha256=pin)
    assert unbound["verification"] == "self_consistent_saved_claim_only" and not unbound["exact_replay_verified"]
    assert "proof" not in unbound and unbound["claimed_proof"] == result["proof"]
    receipt = json.loads((root / RECEIPT).read_bytes()); receipt["proof"]["success"] = True
    (root / RECEIPT).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="proof differs"):
        load_task_recording_evidence(root, expected_contract_sha256=pin)


def test_coordinated_saved_claim_rewrite_requires_trusted_receipt_pin(tmp_path, example):
    import weft.moving_equipment_task_recording as recording
    root, contract = journal(tmp_path, example)
    pin = canonical_hash(contract)
    result = export_task_recording(root, expected_contract_sha256=pin)
    # A journal/manifest/proof rewrite is self-consistent, but has not executed a physics replay.
    mutate_events(root, lambda events: events[-1].update(reward=123.))
    manifest = json.loads((root / "manifest.json").read_bytes())
    for name in ("events.jsonl", "recording.json"):
        manifest["files"][name] = sha256((root / name).read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    receipt = json.loads((root / RECEIPT).read_bytes())
    for name in ("events.jsonl", "recording.json", "manifest.json"):
        raw = (root / name).read_bytes()
        receipt["files"][name] = {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
    raw, _, events, steps, resets = recording._read(root, time.monotonic()+10)
    receipt["proof"] = recording._proof(contract, {k: sha256(v).hexdigest() for k,v in raw.items()}, events, steps, resets, True)
    (root / RECEIPT).write_text(json.dumps(receipt))
    unbound = load_task_recording_evidence(root, expected_contract_sha256=pin)
    assert not unbound["exact_replay_verified"] and "proof" not in unbound
    with pytest.raises(ValueError):
        load_task_recording_evidence(root, expected_contract_sha256=pin, expected_receipt_sha256=result["receipt_sha256"])
    with pytest.raises(ValueError, match="reward"):
        verify_task_replay(root, expected_contract_sha256=pin)
