"""V5 policy custody and control dispatch; all simulator entrypoints stay blocked."""
from types import SimpleNamespace
import time

import mujoco
import numpy as np
import pytest

from weft import shared_exercise_feedback_learning as learner
from weft import shared_exercise_feedback_residual as actor
from weft import shared_exercise_lab as lab
from weft import shared_exercise_policies as policies
from weft import shared_exercise_recording as recording
from weft import shared_world
from weft import recording as common_recording
from test_shared_exercise_feedback_lab import analytical_package, no_simulation, save_package, sealed
from test_shared_exercise_selection import NoPhysicsEnv


@pytest.fixture(autouse=True)
def metadata_only(monkeypatch, no_simulation):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Split-policy admission checks cannot execute native/model/media/network work")

    monkeypatch.setattr(mujoco, "MjModel", SimpleNamespace(from_xml_string=forbidden, from_binary_path=forbidden))
    for name in ("MjData", "Renderer", "mj_step", "mj_forward", "mj_resetData"):
        monkeypatch.setattr(mujoco, name, forbidden)
    # The existing metadata fixtures need authored mass, not a model compilation.
    monkeypatch.setattr(shared_world, "human_subtree_mass", lambda _spec, scale=1.: 42. * scale ** 3)
    import cv2
    import socket
    import subprocess
    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(cv2, "VideoWriter", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def package(*, person=True, roots=True, correction=True, feasible=False):
    data = analytical_package(person=person, assistance=roots, correction=correction, feasible=feasible)
    env = NoPhysicsEnv(data.scene, data.task)
    weights = np.zeros(actor.contract(actor.SPLIT_POLICY_PROFILE)["coefficient_shape"])
    if correction:
        weights[1, 0], weights[3, 1] = .01, -.02
    data.policy = actor.make_policy(env, weights, 1729, data.policy["training_score"],
        data.policy["optimization_rank"], actor.runtime_hashes(data.scene), profile=actor.SPLIT_POLICY_PROFILE)
    data.files["policy.json"] = lab._bytes(data.policy)
    data.training["policy_sha256"] = lab._sha(data.files["policy.json"])
    data.files["training.json"] = lab._bytes(data.training)
    data.manifest = sealed(data.files)
    return data


@pytest.mark.parametrize("person", [False, True])
@pytest.mark.parametrize("roots", [False, True])
def test_v5_admission_keeps_complete_saved_proof_and_training_scope(monkeypatch, person, roots):
    data = package(person=person, roots=roots)
    deadline, calls = time.monotonic() + 5, []
    def verify(manifest, files, scene, task, *, deadline):
        assert manifest == data.manifest and files is data.files
        assert scene == data.scene and task == data.task
        calls.append(deadline)
    monkeypatch.setattr(learner, "verify_shared_exercise_feedback_package", verify)
    policy, identity = lab._validate_policy_package(data.manifest, data.files, data.scene, data.task, deadline=deadline)
    assert calls == [deadline] and policy == data.policy
    assert identity["policy_sha256"] == lab._sha(data.files["policy.json"])
    assert identity["learning_files_verified"] == len(data.files)
    assert policy["groups"] == list(actor.contract(actor.SPLIT_POLICY_PROFILE)["groups"])
    assert len(policy["action_names"]) == (22 if roots else 19)
    status = lab._policy_status(policy, data.files["training.json"])
    assert status["training_feasible"] is False
    assert status["selected_feedback_kind"] == "searched_feedback_residuals"
    assert status["evaluation_status"] == "not_attached" and status["evaluation_successes"] is None
    assert status["training_score_improved_over_authored"] is False


def test_zero_v5_remains_authored_and_cannot_skip_saved_search_proof(monkeypatch):
    data = package(correction=False, feasible=True)
    assert policies.selection_kind(data.policy) == ("authored_unchanged", True, False)
    status = lab._policy_status(data.policy, data.files["training.json"])
    assert status["training_feasible"] is True and status["selected_feedback_kind"] == "authored_unchanged"
    def reject(*_args, **_kwargs):
        raise ValueError("saved split-policy proof rejected")
    monkeypatch.setattr(learner, "verify_shared_exercise_feedback_package", reject)
    with pytest.raises(ValueError, match="saved split-policy proof"):
        lab._validate_policy_package(data.manifest, data.files, data.scene, data.task)


def test_v5_real_saved_verifier_rejects_labels_without_search_evidence():
    data = package()
    with pytest.raises(ValueError, match="generations"):
        lab._validate_policy_package(data.manifest, data.files, data.scene, data.task, deadline=time.monotonic() + 5)


@pytest.mark.parametrize("legacy", [False, True])
def test_existing_v4_and_v2_admission_keep_their_original_summary_and_proof_route(monkeypatch, legacy):
    data = analytical_package(legacy=legacy)
    calls = []
    monkeypatch.setattr(learner, "verify_shared_exercise_feedback_package", lambda *_a, **_k: calls.append(True))
    policy, _ = lab._validate_policy_package(data.manifest, data.files, data.scene, data.task)
    assert policy == data.policy and calls == ([] if legacy else [True])
    assert policy["profile"] == ("shared-exercise-policy-v2" if legacy else "shared-exercise-policy-v4")
    summary = lab._policy_status(policy, data.files["training.json"])
    assert ("training_feasible" in summary) is not legacy
    assert ("selection_objective" in summary) is not legacy
    assert summary["selected_feedback_kind"] == "authored_unchanged"


@pytest.mark.parametrize("change", ["legacy_shape", "group_alias", "false_baseline", "runtime_bytes"])
def test_v5_resealed_metadata_cannot_change_policy_meaning(monkeypatch, change):
    data = package()
    if change == "legacy_shape": data.policy["feedback_coefficients"] = [[0., 0.]] * 3
    elif change == "group_alias": data.policy["groups"][1] = "proximal"
    elif change == "false_baseline":
        data.training.update(selected_feedback_kind="authored_unchanged", selected_authored_baseline=True)
    else: data.files["implementation/shared_exercise_feedback_residual.py"] += b"\n# changed\n"
    data.files["policy.json"] = lab._bytes(data.policy)
    data.training["policy_sha256"] = lab._sha(data.files["policy.json"])
    data.files["training.json"] = lab._bytes(data.training); data.manifest = sealed(data.files)
    monkeypatch.setattr(learner, "verify_shared_exercise_feedback_package",
                        lambda *_a, **_k: pytest.fail("Invalid metadata reached expensive saved proof"))
    with pytest.raises(ValueError):
        lab._validate_policy_package(data.manifest, data.files, data.scene, data.task)


@pytest.mark.parametrize("person", [False, True])
def test_v5_prepare_and_resume_retain_exact_policy_and_full_proof(tmp_path, monkeypatch, person):
    data = package(person=person)
    source, trained, out = (tmp_path / name for name in ("source", "trained", "lab"))
    save_package(source, data.source_files, data.files["input-manifest.json"])
    save_package(trained, data.files, data.manifest)
    calls = []
    monkeypatch.setattr(learner, "verify_shared_exercise_feedback_package", lambda *_a, deadline: calls.append(deadline))
    monkeypatch.setattr(lab, "_reference_frames", lambda _scene, _deadline: {"frames": [], "scope": "metadata-only fixture"})
    result = lab.prepare_shared_exercise_lab(source, out, policy_run=trained, max_seconds=5)
    assert result["learned_policy_available"] is True and len(calls) == 2
    deadline = time.monotonic() + 5
    loaded = lab.load_shared_exercise_lab(out / "run", deadline=deadline)
    assert calls[-1] == deadline and len(calls) == 3
    assert loaded["policy_document"] == data.files["policy.json"]
    assert loaded["contract"]["exercise_policy"]["profile"] == actor.SPLIT_POLICY_PROFILE
    assert loaded["contract"]["exercise_policy"]["training_feasible"] is False
    for name, raw in data.files.items(): assert (out / "run/learning-run" / name).read_bytes() == raw


def test_v5_select_evaluate_use_existing_feedback_route_only(tmp_path, monkeypatch):
    data = package(); (tmp_path / "policy.json").write_bytes(data.files["policy.json"])
    calls = []
    monkeypatch.setattr(learner, "select_shared_exercise_feedback_policy", lambda *a, **k: calls.append(("select", a, k)) or "selected")
    monkeypatch.setattr(learner, "evaluate_shared_exercise_feedback_policy", lambda *a, **k: calls.append(("evaluate", a, k)) or "evaluated")
    out = tmp_path / "new"
    assert policies.select_policy(tmp_path, out, max_seconds=4) == "selected"
    assert policies.evaluate_policy(tmp_path, out, trials=1) == "evaluated"
    assert calls == [("select", (tmp_path, out), {"max_seconds": 4}), ("evaluate", (tmp_path, out), {"trials": 1})]
    for profile in ("shared-world-v4", "shared-world-v5", "shared-world-v6"):
        with pytest.raises(ValueError): policies.validate_policy(data.policy, {"profile": profile}, data.task)
    (tmp_path / "policy.json").write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(ValueError): policies.select_policy(tmp_path, out)
    assert len(calls) == 2


@pytest.mark.parametrize("roots", [False, True])
def test_v5_recorded_policy_routes_exact_actor_and_rejects_action_tampering(monkeypatch, roots):
    data = package(roots=roots)
    envelope = recording.recording_scene(data.scene, data.task, policy_document=data.files["policy.json"])
    assert recording.validate_recording_scene(envelope) == envelope
    assert envelope["learned_policy"]["document_utf8"].encode() == data.files["policy.json"]
    env = NoPhysicsEnv(data.scene, data.task)
    env.observation_contract = lambda: {"profile": "shared-exercise-observation-v1"}
    features = np.full((env.action_count, 3), .1)
    env.feature_matrix = lambda: features.copy()
    expected = policies.prepare_policy(data.policy, env)().tolist()
    initial = {"task_metrics": {"evaluator_sha256": env.evaluator_sha256}, "resolved_reset_options": {"jitter": False, "jitter_scale": 0.}}
    final = {"task_metrics": {"evaluator_sha256": env.evaluator_sha256}, "last_action": expected, "terminated": True, "truncated": False}
    closed = []
    class ReplayEnv(NoPhysicsEnv):
        def __init__(self, scene, task):
            super().__init__(scene, task); self.current = initial
        def observation_contract(self): return {"profile": "shared-exercise-observation-v1"}
        def feature_matrix(self): return features.copy()
        def reset(self, **_kwargs): self.current = initial
        def step(self, action):
            assert action == expected; self.current = final
            return None, .7, True, False, {}
        def snapshot(self): return self.current
        def close(self): closed.append(True)
    step = {"type": "step", "index": 1, "controller": "learned", "policy_sha256": lab._sha(data.files["policy.json"]),
            "action": expected, "reward": .7, "terminated": True, "truncated": False, "snapshot": final}
    recording.validate_event(step, envelope)
    events = [{"type": "start", "index": 0, "seed": 11, "snapshot": initial}, step]
    monkeypatch.setattr(common_recording, "load_recording", lambda *_a, **_k: {"scene": envelope, "events": events})
    monkeypatch.setattr(recording, "SharedExerciseTaskEnv", ReplayEnv)
    proof = recording.verify_action_replay("analytical-no-directory")
    assert proof["learned_actions_verified"] == 1 and proof["policy_sha256"] == step["policy_sha256"]
    assert closed == [True]
    step["action"] = expected.copy(); step["action"][0] += .001
    with pytest.raises(ValueError, match="learned controller replay differs"):
        recording.verify_action_replay("analytical-no-directory")
    assert closed == [True, True]
    step["action"] = expected; step["policy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned policy"):
        recording.validate_event(step, envelope)
