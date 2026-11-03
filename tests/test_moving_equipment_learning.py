"""Bounded analytic policy search; only sealed-task admission is explicitly fixture-stubbed."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_moving_equipment_task import example
import weft.moving_equipment_learning as learning
from weft.moving_equipment_task import MovingEquipmentTaskEnv


@pytest.fixture
def admitted(example, monkeypatch):
    scene, bundle, task = example
    value = SimpleNamespace(scene=deepcopy(scene), task=deepcopy(task), bundle=deepcopy(bundle),
        scene_bytes=learning._bytes(scene), task_bytes=learning._bytes(task), bundle_bytes=learning._bytes(bundle, 32 << 20),
        manifest_sha256="a"*64, parent_manifest_sha256="b"*64, runtime_sha256={})
    def validate(candidate):
        if candidate is not value:
            raise ValueError("Original sealed task admission required")
        return candidate
    monkeypatch.setattr(learning, "_admitted", validate)
    return value


def test_one_bounded_analytic_search_freezes_policy_before_same_start_comparison(admitted, tmp_path, monkeypatch):
    original = {name: sha256(getattr(admitted, name)).hexdigest() for name in ("scene_bytes", "task_bytes", "bundle_bytes")}
    created = []
    class Watched(MovingEquipmentTaskEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs);created.append(self)
        def close(self):
            self.closed = True;super().close()
    monkeypatch.setattr(learning, "MovingEquipmentTaskEnv", Watched)
    out = tmp_path / "learning"
    result = learning.train_moving_equipment_policy(admitted, out, generations=1, population=4, max_seconds=10)
    assert len(created) == 1 and created[0].closed
    assert result["status"] == "complete" and result["control_steps"] <= result["maximum_control_steps"] <= 20000
    assert 0 < result["control_steps"] <= 6*admitted.task["interval"]["horizon_steps"]
    assert result["independent_changed_start_evaluation"] == "unavailable"
    assert result["api_calls"] == result["inference_calls"] == 0
    policy_raw = (out/"policy.json").read_bytes();policy = json.loads(policy_raw)
    assert learning.validate_moving_equipment_policy(policy, admitted) == policy
    evaluated = json.loads((out/"evaluation.json").read_bytes())
    assert evaluated["policy_sha256"] == sha256(policy_raw).hexdigest() == result["policy_sha256"]
    assert evaluated["frozen_nominal_replay_matches_training"] is True
    assert [row["controller"] for row in evaluated["comparisons"]] == ["zero", "authored_feedback", "random", "frozen_policy"]
    assert [row["reused_training_trial"] for row in evaluated["comparisons"]] == [True, True, False, False]
    assert all(row["trial"]["reset"] == {"jitter": False} for row in evaluated["comparisons"])
    assert all(row["trial"]["metrics"]["elapsed_s"] == pytest.approx(.45)
               for row in evaluated["comparisons"] if row["trial"]["metrics"]["completed_interval"])
    training = json.loads((out/"generation-1.json").read_bytes())
    actual_steps = sum(row["trial"]["control_steps"] for row in training["candidates"] if not row["cached"])
    actual_steps += sum(row["trial"]["control_steps"] for row in evaluated["comparisons"] if not row["reused_training_trial"])
    assert result["control_steps"] == actual_steps
    manifest = json.loads((out/"manifest.json").read_bytes())
    for name, digest in manifest["files"].items():
        assert sha256((out/name).read_bytes()).hexdigest() == digest
    assert sum(path.stat().st_size for path in out.rglob("*") if path.is_file()) <= learning.MAX_OUTPUT_BYTES
    assert {name: sha256(getattr(admitted, name)).hexdigest() for name in original} == original
    with pytest.raises(ValueError, match="new moving learning output"):
        learning.train_moving_equipment_policy(admitted, out, generations=1, population=4)


def policy_fixture(admission):
    hashes, versions, _ = learning._runtime(admission, admission.scene)
    dim = 12+3*len(admission.scene["action_contract"]["dynamic_object_ids"])
    return {"schema_version": 1, "profile": learning.PROFILE, "algorithm": learning.ALGORITHM,
        "source": learning._pins(admission, admission.scene, admission.task), "runtime_sha256": hashes,
        "runtime_versions": versions, "action_names": admission.scene["action_contract"]["names"],
        "weights": [1.]*dim, "features": deepcopy(learning.FEATURES), "training_seed": 1,
        "training_score": 0., "scope": learning.SCOPE}


def test_feedback_has_all_explicit_channels_and_never_rehashes_per_action(admitted, monkeypatch):
    env = MovingEquipmentTaskEnv(admitted.scene, admitted.task, admitted.bundle)
    try:
        policy = policy_fixture(admitted)
        action = learning.prepare_moving_equipment_policy(policy, env, admitted)
        env.reset(seed=1)
        monkeypatch.setattr(learning, "read_package_file", lambda *_a, **_k: pytest.fail("No policy reads during actions"))
        monkeypatch.setattr(learning, "canonical_hash", lambda *_a, **_k: pytest.fail("No policy hashes during actions"))
        first = action()
        assert first.shape == (len(env.action_names),) and np.isfinite(first).all() and np.max(abs(first)) <= 1
        assert first[-1] > 0  # Declared gravity-compensation gain creates an explicit upward virtual force.
        env.step(first)
        second = action()
        assert not np.array_equal(first, second)
        assert np.max(abs(second)) <= 1
        assert not action.weights.flags.writeable
    finally:
        env.close()


@pytest.mark.parametrize("damage", ["source", "task", "runtime", "actions", "features", "bool", "nan", "length", "scope"])
def test_policy_source_task_runtime_and_gain_tampering_refuse(admitted, damage):
    policy = policy_fixture(admitted)
    if damage == "source": policy["source"]["task_manifest_sha256"] = "f"*64
    elif damage == "task": policy["source"]["task_sha256"] = "f"*64
    elif damage == "runtime": policy["runtime_sha256"]["moving_equipment_learning.py"] = "f"*64
    elif damage == "actions": policy["action_names"] = policy["action_names"][:-1]
    elif damage == "features": policy["features"]["privileged_simulator_state"] = False
    elif damage == "bool": policy["weights"][0] = True
    elif damage == "nan": policy["weights"][0] = float("nan")
    elif damage == "length": policy["weights"].pop()
    else: policy["scope"] = "independent physical success"
    with pytest.raises(ValueError): learning.validate_moving_equipment_policy(policy, admitted)


@pytest.mark.parametrize("parameters", [{"generations": 0}, {"population": 13}, {"seed": True}, {"max_seconds": 61}])
def test_bounds_refuse_before_creating_output(admitted, tmp_path, parameters):
    out = tmp_path/"refused"
    with pytest.raises(ValueError): learning.train_moving_equipment_policy(admitted, out, **parameters)
    assert not out.exists()


def test_aggregate_control_budget_counts_baselines_and_freeze_check(admitted, tmp_path):
    admitted.task["interval"]["horizon_steps"] = 501
    with pytest.raises(ValueError, match="20000"):
        learning.train_moving_equipment_policy(admitted, tmp_path/"refused", generations=4, population=12)
    assert not (tmp_path/"refused").exists()


def test_failed_reset_is_preserved_once_without_unchanged_retry(admitted, tmp_path, monkeypatch):
    resets = []
    class Failed:
        _metrics = None
        def __init__(self,*args): pass
        def reset(self,**kwargs):
            resets.append(1);raise ValueError("initial source floor refusal")
        def close(self): resets.append("closed")
    monkeypatch.setattr(learning,"MovingEquipmentTaskEnv",Failed)
    out=tmp_path/"failed"
    with pytest.raises(ValueError,match="initial source floor"):
        learning.train_moving_equipment_policy(admitted,out,generations=1,population=4,max_seconds=5)
    assert resets == [1,"closed"]
    failure=json.loads((out/"failure.json").read_bytes())
    assert failure["control_steps"] == 0 and failure["partial_trial"]["status"] == "failed"
    assert not (out/"policy.json").exists()


def test_output_exhaustion_retains_failure_reservation(tmp_path):
    writer=learning._Output(tmp_path/"bounded")
    writer.bytes=learning.MAX_OUTPUT_BYTES-(256<<10)
    with pytest.raises(ValueError,match="reservation"): writer.save("too-large.json",b"x")
    writer.json("failure.json",{"reason":"bounded"})
    assert (writer.path/"failure.json").exists()


def test_deadline_preserves_partial_trial_without_policy(admitted,tmp_path,monkeypatch):
    class Empty:
        def __init__(self,*args): pass
        def close(self): pass
    monkeypatch.setattr(learning,"MovingEquipmentTaskEnv",Empty)
    def stopped(*args,**kwargs):
        error=TimeoutError("aggregate deadline")
        error.partial_trial={"control_steps":0,"status":"interrupted"}
        raise error
    monkeypatch.setattr(learning,"_trial",stopped)
    out=tmp_path/"interrupted"
    with pytest.raises(TimeoutError):learning.train_moving_equipment_policy(admitted,out,generations=1,population=4)
    assert json.loads((out/"failure.json").read_bytes())["partial_trial"]["status"] == "interrupted"
    assert not (out/"policy.json").exists()


def test_admitted_runtime_map_is_parsed_once_per_check(admitted):
    reads = []
    class Proxy:
        @property
        def runtime_sha256(self):
            reads.append(1)
            return admitted.runtime_sha256
    learning._runtime(Proxy(), admitted.scene)
    assert reads == [1]


def test_saved_wrapper_loads_once_and_passes_only_remaining_budget(tmp_path, monkeypatch):
    import weft.moving_equipment_task_lab as labs
    source=tmp_path/"source";source.mkdir();(source/"lab-config.json").write_text('{}')
    calls=[];clock=[0.]
    monkeypatch.setattr(learning.time,"monotonic",lambda:clock[0])
    admission=object()
    def load(run,config,*,max_seconds):
        calls.append(("load",run,config,max_seconds));clock[0]=20.;return admission
    def train(value,out,**kwargs):
        assert value is admission
        calls.append(("train",out,kwargs));clock[0]=60.;return {"status":"complete"}
    monkeypatch.setattr(labs,"load_moving_equipment_task_lab",load)
    monkeypatch.setattr(learning,"train_moving_equipment_policy",train)
    result=learning.train_saved_moving_equipment_policy(source,tmp_path/"new",max_seconds=75)
    assert calls[0][0] == "load" and calls[0][-1] == 30
    assert calls[1][2]["max_seconds"] == 55
    assert result["source_admissions"] == 1 and result["aggregate_seconds"] == 60


@pytest.mark.parametrize("bad",["inside","same","symlink","traversal"])
def test_saved_wrapper_refuses_bad_output_before_source_admission(tmp_path,monkeypatch,bad):
    import weft.moving_equipment_task_lab as labs
    source=tmp_path/"source";source.mkdir()
    output=source/"child" if bad=="inside" else source if bad=="same" else tmp_path/"new"
    if bad=="symlink": output.symlink_to(source,target_is_directory=True)
    if bad=="traversal": output=tmp_path/"directory"/".."/"new"
    monkeypatch.setattr(labs,"load_moving_equipment_task_lab",lambda *_a,**_k:pytest.fail("No admission"))
    monkeypatch.setattr(learning,"read_package_file",lambda *_a,**_k:pytest.fail("No reads"))
    with pytest.raises(ValueError): learning.train_saved_moving_equipment_policy(source,output)
