"""V4 lab admission boundaries use analytical documents, never physics or training."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time

import mujoco
import numpy as np
import pytest

from weft import shared_exercise_feedback_learning as learner
from weft import shared_exercise_feedback_residual as actor
from weft import shared_exercise_lab as lab
from weft import shared_exercise_residual_learning as residual
from weft import shared_exercise_task as tasks
from weft import shared_world_lab as world_lab
from weft.shared_exercise_feasibility_learning import optimization_rank
from weft.shared_exercise_recording import recording_scene, validate_recording_scene
from weft.shared_world_observation import EquipmentState
from test_shared_exercise_selection import NoPhysicsEnv
from test_shared_world_fullspan import fullspan_scene
from test_shared_world_human_only import person_scene


@pytest.fixture(autouse=True)
def no_simulation(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Admission tests must not construct, advance or render a simulation")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(mujoco, "mj_forward", forbidden)
    monkeypatch.setattr(lab, "SharedWorldEnv", forbidden)
    monkeypatch.setattr(lab, "SharedExerciseTaskEnv", forbidden)
    monkeypatch.setattr(learner, "SharedExerciseTaskEnv", forbidden)

    def analytical_floor(spec):
        # Admission needs authored metadata, not a fresh geometry calibration.
        result = deepcopy(spec)
        mobility = result["mobility"]
        mobility["floor_fit"] = {"method": "first_pose_collision_surface_clearance_v1",
            "reference_time_s": result["reference"][0]["time_s"], "future_poses_used": False,
            "original_base_height": mobility["base_height"], "base_height_shift": 0.,
            "lowest_surface_before": mobility["clearance"], "lowest_surface_after": mobility["clearance"]}
        result["assumptions"].append("Analytical admission fixture: floor metadata is authored, not measured.")
        return result

    monkeypatch.setattr(world_lab, "calibrate_floor", analytical_floor)


def sealed(files, status="trained"):
    return lab._bytes({"schema_version": 1, "status": status,
                       "files": {name: lab._sha(raw) for name, raw in files.items()}})


def analytical_package(*, person=True, assistance=True, correction=False, feasible=False, legacy=False):
    scene = person_scene(steps=2) if person else fullspan_scene(steps=2)
    task = tasks.build_shared_exercise_task(scene, root_assistance=assistance)
    env = NoPhysicsEnv(scene, task)
    equipment = tuple(EquipmentState(row["id"], tuple(row["position"]), tuple(row["quaternion"]),
        tuple(row["velocity"]), tuple(row["angular_velocity"]), (0., 0., 0.)) for row in scene["objects"])
    measured = tasks._ExerciseMetrics(scene, task, equipment, 0., 1.)
    duration = scene["dt"] * env.horizon_steps
    measured.samples = measured.valid_samples = env.horizon_steps * scene["substeps"]
    measured.elapsed = measured.valid_duration = duration
    measured.angular_squared, measured.root_squared = .2 ** 2 * duration, .03 ** 2 * duration
    measured.angular_peak, measured.root_peak = (.2 if feasible else .9), .03
    measured.reward_integral = .7 * duration
    rank = optimization_rank([{"status": "completed", "metrics": measured.metrics()}] * 2)
    policy = residual._policy(env, np.zeros((2, 19)), 1729, .7, residual._runtime_hashes(scene)) if legacy else actor.make_policy(
        env, np.full((3, 2), .01 if correction else 0.), 1729, rank["reward_score"], rank, actor.runtime_hashes(scene))
    policy_raw = lab._bytes(policy)
    source_files = {"scene.json": lab._bytes(scene), "exercise-task.json": lab._bytes(task)}
    if person:
        source_files["inventory.json"] = lab._bytes(scene["inventory"])
    input_manifest = sealed(source_files, "ready")
    training = {"status": "trained", "policy_sha256": lab._sha(policy_raw),
        "selected_feedback_kind": "searched_feedback_residuals" if correction else "authored_unchanged",
        "selected_authored_baseline": not correction, "selected_coast_baseline": False,
        "improvement_over_authored_training": 0., "training_score_improved_over_authored": False,
        "scope": "Analytical admission fixture, not a search result"}
    settings = {key: policy[key] for key in ("scene_sha256", "task_sha256", "gates_sha256", "runtime_sha256")}
    settings["source_manifest_sha256"] = lab._sha(input_manifest)
    files = {**source_files, "policy.json": policy_raw, "training.json": lab._bytes(training),
        "settings.json": lab._bytes(settings), "input-manifest.json": input_manifest,
        "diagnostics/fixture.jsonl": b'{"scope":"Analytical proof-retention fixture; no actions"}\n'}
    files.update({"implementation/" + name: Path(actor.__file__).with_name(name).read_bytes()
                  for name in policy["runtime_sha256"]})
    return SimpleNamespace(scene=scene, task=task, policy=policy, training=training,
        files=files, manifest=sealed(files), source_files=source_files)


def save_package(root, files, manifest):
    root.mkdir()
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (root / "manifest.json").write_bytes(manifest)


@pytest.mark.parametrize("person", [False, True])
@pytest.mark.parametrize("assistance", [False, True])
def test_v4_admission_calls_complete_saved_proof_once_with_inherited_deadline(monkeypatch, person, assistance):
    data = analytical_package(person=person, assistance=assistance, correction=True)
    deadline, calls = time.monotonic() + 5, []

    def verify(manifest, files, scene, task, *, deadline):
        calls.append(deadline)
        assert manifest == data.manifest and files is data.files
        assert scene == data.scene and task == data.task

    monkeypatch.setattr(learner, "verify_shared_exercise_feedback_package", verify)
    policy, identity, baseline = lab._validate_policy_package(data.manifest, data.files, data.scene, data.task,
        deadline=deadline, include_baseline=True)
    assert calls == [deadline] and baseline is None
    assert policy == data.policy and identity["policy_sha256"] == lab._sha(data.files["policy.json"])
    assert len(identity["policy_runtime_sha256"]) == (26 if person else 25)
    assert identity["learning_files_verified"] == len(data.files)
    assert "previous_reward_policy_sha256" not in identity


@pytest.mark.parametrize("person", [False, True])
def test_v4_failed_proof_blocks_lab_before_output_and_preserves_inputs(tmp_path, monkeypatch, person):
    data = analytical_package(person=person)
    source, learning, out = (tmp_path / name for name in ("source", "learning", "lab"))
    save_package(source, data.source_files, data.files["input-manifest.json"])
    save_package(learning, data.files, data.manifest)
    before = {path: path.read_bytes() for root in (source, learning) for path in root.rglob("*") if path.is_file()}

    def reject(*_args, **_kwargs):
        raise ValueError("Injected inconsistent saved candidate/selection proof")

    monkeypatch.setattr(learner, "verify_shared_exercise_feedback_package", reject)
    monkeypatch.setattr(lab, "_implementation_snapshot", lambda: pytest.fail("Preparation continued after rejected proof"))
    with pytest.raises(ValueError, match="saved candidate/selection proof"):
        lab.prepare_shared_exercise_lab(source, out, policy_run=learning, max_seconds=5)
    assert not out.exists() and {path: path.read_bytes() for path in before} == before


def test_real_strict_verifier_rejects_analytic_labels_without_a_search_proof():
    data = analytical_package()
    with pytest.raises(ValueError, match="generations"):
        lab._validate_policy_package(data.manifest, data.files, data.scene, data.task, deadline=time.monotonic() + 5)


@pytest.mark.parametrize("mutation", ["label", "policy_hash", "runtime", "inventory_pin", "scene_binding"])
def test_v4_resealed_contradictions_reject_before_proof(monkeypatch, mutation):
    data = analytical_package(correction=True)
    if mutation in {"label", "policy_hash"}:
        training = lab._json(data.files["training.json"])
        if mutation == "label":
            training.update(selected_feedback_kind="authored_unchanged", selected_authored_baseline=True)
        else:
            training["policy_sha256"] = "0" * 64
        data.files["training.json"] = lab._bytes(training)
    elif mutation == "runtime":
        data.files["implementation/shared_exercise_feedback_residual.py"] += b"\n# tampered\n"
    else:
        policy = lab._json(data.files["policy.json"])
        if mutation == "inventory_pin":
            policy["runtime_sha256"].pop("person_inventory.py")
        else:
            policy["scene_sha256"] = "0" * 64
        data.files["policy.json"] = lab._bytes(policy)
    monkeypatch.setattr(learner, "verify_shared_exercise_feedback_package",
                        lambda *_a, **_k: pytest.fail("Contradictory identity reached the proof verifier"))
    with pytest.raises(ValueError):
        lab._validate_policy_package(sealed(data.files), data.files, data.scene, data.task)


@pytest.mark.parametrize("mode", ["expired_before", "expired_after", "standalone"])
def test_v4_proof_cannot_extend_its_callers_budget(monkeypatch, mode):
    data = analytical_package()
    now, calls = [100.], []
    monkeypatch.setattr(lab, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def verify(*_args, deadline):
        calls.append(deadline)
        if mode == "expired_after":
            now[0] = deadline + .1

    monkeypatch.setattr(learner, "verify_shared_exercise_feedback_package", verify)
    options = {} if mode == "standalone" else {"deadline": 99. if mode == "expired_before" else 101.}
    if mode == "standalone":
        lab._validate_policy_package(data.manifest, data.files, data.scene, data.task, **options)
        assert calls == [130.]
    else:
        with pytest.raises(TimeoutError):
            lab._validate_policy_package(data.manifest, data.files, data.scene, data.task, **options)
        assert calls == ([] if mode == "expired_before" else [101.])


@pytest.mark.parametrize("feasible", [False, True])
@pytest.mark.parametrize("correction", [False, True])
def test_v4_status_separates_training_feasibility_baseline_and_unattached_evaluation(feasible, correction):
    data = analytical_package(feasible=feasible, correction=correction)
    status = lab._policy_status(data.policy, data.files["training.json"])
    assert status["training_feasible"] is feasible
    assert status["selection_objective"] == data.policy["selection_objective"]
    assert status["selected_feedback_kind"] == ("searched_feedback_residuals" if correction else "authored_unchanged")
    assert status["training_score_improved_over_authored"] is False
    assert status["evaluation_status"] == "not_attached" and status["evaluation_successes"] is None


@pytest.mark.parametrize("person", [False, True])
def test_legacy_v2_admission_does_not_invoke_new_proof_or_add_feasibility_fields(monkeypatch, person):
    data = analytical_package(person=person, legacy=True)
    monkeypatch.setattr(learner, "verify_shared_exercise_feedback_package",
                        lambda *_a, **_k: pytest.fail("Legacy package reinterpreted by v4 proof"))
    policy, _ = lab._validate_policy_package(data.manifest, data.files, data.scene, data.task)
    assert policy == data.policy
    status = lab._policy_status(policy, data.files["training.json"])
    assert "training_feasible" not in status and "selection_objective" not in status


@pytest.mark.parametrize("person", [False, True])
def test_lab_prepare_and_resume_preserve_complete_proof_with_original_deadlines(tmp_path, monkeypatch, person):
    data = analytical_package(person=person)
    source, learning, out = (tmp_path / name for name in ("source", "learning", "lab"))
    save_package(source, data.source_files, data.files["input-manifest.json"])
    save_package(learning, data.files, data.manifest)
    proof_deadlines, reference_deadlines = [], []
    monkeypatch.setattr(learner, "verify_shared_exercise_feedback_package",
                        lambda *_args, deadline: proof_deadlines.append(deadline))

    def reference(_scene, deadline):
        reference_deadlines.append(deadline)
        return {"scope": "Analytical packaging test; no reference rendering", "frames": []}

    monkeypatch.setattr(lab, "_reference_frames", reference)
    result = lab.prepare_shared_exercise_lab(source, out, policy_run=learning, max_seconds=5)
    assert result["learned_policy_available"]
    # Preparation verifies the written package through the normal resume reader.
    assert proof_deadlines == reference_deadlines * 2 and len(reference_deadlines) == 1
    deadline = time.monotonic() + 5
    loaded = lab.load_shared_exercise_lab(out / "run", deadline=deadline)
    assert proof_deadlines[-1] == deadline and len(proof_deadlines) == 3
    assert loaded["policy_document"] == data.files["policy.json"]
    for name, raw in data.files.items():
        assert (out / "run/learning-run" / name).read_bytes() == raw
    assert loaded["contract"]["exercise_policy"]["training_feasible"] is False


@pytest.mark.parametrize("person", [False, True])
@pytest.mark.parametrize("assistance", [False, True])
def test_recording_envelope_retains_v4_exact_bytes_and_rejects_tampering(person, assistance):
    data = analytical_package(person=person, assistance=assistance, correction=True)
    bundle = recording_scene(data.scene, data.task, policy_document=data.files["policy.json"])
    assert validate_recording_scene(bundle) == bundle
    assert bundle["learned_policy"]["document_utf8"].encode() == data.files["policy.json"]
    assert bundle["learned_policy"]["sha256"] == lab._sha(data.files["policy.json"])
    changed = deepcopy(bundle)
    changed["learned_policy"]["document_utf8"] += " "
    with pytest.raises(ValueError, match="hash differs"):
        validate_recording_scene(changed)
