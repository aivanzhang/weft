"""Real driver/CEM/journal arithmetic with a synthetic environment; no native physics or training study."""

from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from weft import shared_exercise_equipment_feedback as actor
from weft import shared_exercise_equipment_feedback_learning as learner
from weft import shared_exercise_equipment_feedback_policy as policy_module
from weft import shared_exercise_feedback_residual as v5
from weft import shared_exercise_policies as dispatch
from weft import shared_exercise_task as tasks
from weft.io import canonical_hash, digest
from weft.shared_exercise_feasibility_learning import optimization_rank
from weft.shared_exercise_reset import describe_reset_protocol
from weft.shared_world_observation import EquipmentState
from test_shared_exercise_equipment_feedback import fixture as actor_fixture, moving_body


def build_context(directory, patch, *, horizon=3, modes=None):
    modes = {} if modes is None else modes
    initializing = True
    template, _ = actor_fixture()
    scene, task = template.scene, template.task
    scene.update(dt=1. / 30., substeps=16, gravity=[0., 0., -9.81], task={"start_offset_s": 0., "horizon_steps": horizon})
    scene["human_spec"]["reference"] = [{"time_s": 0.}, {"time_s": horizon / 30.}]
    scene["objects"] = [{"id": name, "role": "passive", "position": [.5 + 2. * i, 0., 1.],
                          "quaternion": [1., 0., 0., 0.], "velocity": [0., 0., 0.], "angular_velocity": [0., 0., 0.]}
                         for i, name in enumerate(task["equipment_ids"])]
    task.update(interval=dict(scene["task"]), source_scope={"scope": "authored fixture, no source video or native dynamics"})
    calls = SimpleNamespace(source=0, constructs=0, resets=0, issued=0, returned=0, features=0, snapshots=0, closes=0)
    evaluator = digest(tasks.__file__)

    class Env:
        def __init__(self, scene_value, task_value):
            calls.constructs += 1
            self.scene, self.task = scene_value, task_value
            self.scene_sha256, self.task_sha256, self.evaluator_sha256 = canonical_hash(scene_value), canonical_hash(task_value), evaluator
            self._scene, self._task, self._static_support = self.scene, self.task, False
            self.action_count, self.action_names, self.observation_size = 22, tuple(scene_value["action_contract"]["names"]), 190
            self._equipment_ids = tuple(task_value["equipment_ids"])
            self.horizon_steps = task_value["interval"]["horizon_steps"]
            self._engine = object()
            self._done, self.terminated, self.truncated, self._state_status = True, False, False, "current"
            self.steps = 0
            self.feature_error = False

        action_contract = tasks.SharedExerciseTaskEnv.action_contract
        observation_contract = tasks.SharedExerciseTaskEnv.observation_contract

        def _features(self):
            result = np.zeros((22, 3))
            result[:, 2] = .013456789 + self.steps * .00001
            return result

        def feature_matrix(self):
            calls.features += 1
            result = self._features()
            if self.feature_error:
                result[0, 0] += .5
            return result

        def _update_observation(self):
            body = moving_body()
            body[88:154] = self._features().ravel()
            body[154] = self.steps / self.horizon_steps
            self._accumulator = self.metric
            self.observation_space = SimpleNamespace(contains=lambda value: value.shape == (190,) and np.isfinite(value).all())
            self._last_observation = tasks.SharedExerciseTaskEnv._observation(self, body)

        def reset(self, *, seed, options):
            calls.resets += 1
            if modes.get("reset_failure") and not initializing:
                raise RuntimeError("authored reset failure")
            self.seed, self.options, self.steps = seed, deepcopy(options), 0
            self._done = self.terminated = self.truncated = False
            self._state_status = "current"
            items = tuple(EquipmentState(row["id"], tuple(row["position"]), tuple(row["quaternion"]),
                tuple(row["velocity"]), tuple(row["angular_velocity"]), (0., 0., 0.)) for row in self.scene["objects"])
            self.metric = tasks._ExerciseMetrics(self.scene, self.task, items, 0., 1.)
            self.candidate = (calls.resets - 1) // 2
            self.score = (.7, .9, .8, .65)[self.candidate % 4]
            self._update_observation()
            return self.observation(), self.metrics()

        def observation(self):
            return self._last_observation.copy()

        def metrics(self):
            return self.metric.metrics() | {"scene_sha256": self.scene_sha256, "task_sha256": self.task_sha256,
                "evaluator_sha256": self.evaluator_sha256, "reset_protocol": describe_reset_protocol(self.options)}

        def snapshot(self):
            return {"engine_state": {"seed": self.seed, "steps": self.steps}, "task_metrics": self.metrics()}

        def step(self, action):
            calls.issued += 1
            if modes.get("step_failure") == calls.issued:
                raise RuntimeError("authored step failure after admission")
            if modes.get("returned_engine_failure") and calls.resets == 1 and self.steps == 1:
                self.metric.fail("engine_failure:authored returned failure")
                self.metric.penalize_failure()
                self._done, self.terminated, self.truncated = True, True, False
                self._state_status = "last_valid_control_boundary"
                calls.returned += 1
                return self.observation(), -1., True, False, {**self.metrics(), "observation_status": self._state_status}
            self.steps += 1
            self.metric.samples = self.metric.valid_samples = self.steps * 16
            self.metric.elapsed = self.metric.valid_duration = self.steps * self.scene["dt"]
            self.metric.angular_squared, self.metric.root_squared = .2**2 * self.metric.elapsed, .03**2 * self.metric.elapsed
            self.metric.angular_peak, self.metric.root_peak = .2, .03
            self.metric.reward_integral = self.score * self.metric.elapsed
            # Outcomes are authored to test feasibility-before-reward, not caused by simulated actions.
            if self.candidate % 4 != 2:
                self.metric.equipment[self.task["equipment_ids"][0]]["speed_model_units_s"] = .25
            final = self.metrics()
            self.terminated = final["success"]
            self.truncated = final["completed_interval"] and not self.terminated
            self._done = self.terminated or self.truncated
            self._update_observation()
            calls.returned += 1
            return self.observation(), self.score, self.terminated, self.truncated, {
                **self.metrics(), "observation_status": "current", "legacy_coupled_score_used": False}

        def close(self):
            calls.closes += 1

    patch.setattr(v5, "_context", lambda scene, task: (deepcopy(scene), deepcopy(task), 22))
    versions = {"python": "mocked", "numpy": "test-runtime", "mujoco": "not executed", "gymnasium": "mocked"}
    patch.setattr(v5, "_runtime_versions", lambda: dict(versions))
    patch.setattr(policy_module, "_runtime_versions", lambda: dict(versions))
    metadata = Env(scene, task)
    metadata.reset(seed=1, options=learner.TRAIN_RESETS[0])
    metadata.steps = horizon
    metadata.metric.samples = metadata.metric.valid_samples = horizon * 16
    metadata.metric.elapsed = metadata.metric.valid_duration = horizon / 30.
    metadata.metric.angular_squared, metadata.metric.root_squared = .2**2 * horizon / 30., .03**2 * horizon / 30.
    metadata.metric.angular_peak, metadata.metric.root_peak = .2, .03
    metadata.metric.reward_integral = .7 * horizon / 30.
    metric = metadata.metrics()
    rank = optimization_rank([{"status": "completed", "metrics": metric}] * 2)
    base = v5.make_policy(metadata, np.full((5, 2), .1), 1729, rank["reward_score"], rank, v5.runtime_hashes(scene),
                          profile=v5.SPLIT_POLICY_PROFILE)
    initializing = False
    for name in vars(calls): setattr(calls, name, 0)
    source = directory / "source"; source.mkdir(parents=True)
    scene_raw, task_raw, base_raw = map(learner._bytes, (scene, task, base))
    (source / "scene.json").write_bytes(scene_raw); (source / "exercise-task.json").write_bytes(task_raw)
    manifest = learner._bytes({"schema_version": 1, "status": "ready", "files": {
        "scene.json": learner._sha(scene_raw), "exercise-task.json": learner._sha(task_raw)}})
    (source / "manifest.json").write_bytes(manifest)
    base_path = directory / "base.json"; base_path.write_bytes(base_raw)

    def admitted(run, task_path, assistance, deadline):
        calls.source += 1
        assert Path(run) == source and task_path is None and assistance is False
        proof = SimpleNamespace(report={"scope": "mocked admission, no physical evidence"})
        return source, scene, task, scene_raw, task_raw, manifest, proof

    def initialize(out, scene_bytes, task_bytes, source_manifest, proof, extra):
        out.mkdir()
        files = {"scene.json": scene_bytes, "exercise-task.json": task_bytes, "input-manifest.json": source_manifest,
                 "source-verification.json": learner._bytes(proof.report), **extra,
                 **{"implementation/" + name: Path(policy_module.__file__).with_name(name).read_bytes()
                    for name in policy_module.runtime_hashes(scene)}}
        for name, raw in files.items():
            destination = out / name; destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(raw)

    def validator(_scene, _task):
        def validate(snapshot):
            calls.snapshots += 1
            assert set(snapshot) == {"engine_state", "task_metrics"}
        return validate

    patch.setattr(learner, "_source", admitted)
    patch.setattr(learner.residual, "_initialize", initialize)
    patch.setattr(learner.residual, "_source_views", lambda *_: {})
    patch.setattr(learner, "SharedExerciseTaskEnv", Env)
    patch.setattr(learner, "prepare_shared_exercise_snapshot_validator", validator)
    return SimpleNamespace(scene=scene, task=task, base=base, base_raw=base_raw, base_path=base_path, source=source,
        source_manifest=manifest, calls=calls, Env=Env, directory=directory, rank=rank, modes=modes)


def train(context, name="trained", **kwargs):
    return learner.train_shared_exercise_equipment_feedback_policy(context.source, context.directory / name,
        base_policy_path=context.base_path, base_policy_sha256=learner._sha(context.base_raw), **kwargs)


def read_package(path):
    manifest = (path / "manifest.json").read_bytes()
    return manifest, {name: (path / name).read_bytes() for name in json.loads(manifest)["files"]}


def verify(context, manifest, files):
    return learner.verify_equipment_feedback_training(manifest, files, context.scene, context.task,
        base_policy_sha256=learner._sha(context.base_raw), source_manifest_sha256=learner._sha(context.source_manifest),
        deadline=time.monotonic() + 10.)


def reseal(files):
    return learner._bytes({"schema_version": 1, "status": "trained", "files": {name: learner._sha(raw) for name, raw in files.items()}})


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        context = build_context(tmp_path_factory.mktemp("tiny"), patch)
        result = train(context)
        manifest, files = read_package(Path(result["directory"]))
        yield context, result, manifest, files


def test_real_cem_journal_and_original_rank_select_feasible_not_highest_reward(tiny):
    context, result, manifest, files = tiny
    proof = verify(context, manifest, files)
    assert result["selected_candidate"] == {"generation": 0, "candidate": 2}
    assert result["selected_optimization_rank"]["feasible"] is True
    assert result["selected_optimization_rank"]["reward_score"] < result["history"][0]["candidates"][1]["score"]
    assert context.calls.issued == context.calls.returned == context.calls.features == 24
    assert context.calls.resets == 8 and context.calls.constructs == context.calls.closes == 1
    assert proof["control_steps"] == proof["returned_controls"] == 24 and proof["diagnostic_files"] == 8
    assert proof["actual_raw_journal_bytes"] <= proof["projected_raw_journal_bytes"]
    assert files["base-policy.json"] == context.base_raw
    assert not any("evaluation" in name for name in files)
    with pytest.raises(ValueError, match="Unsupported"):
        dispatch._profile(json.loads(files["policy.json"]))


def test_mocked_full_254_control_workflow_measures_journal_without_native(tmp_path, monkeypatch):
    context = build_context(tmp_path, monkeypatch, horizon=254)
    result = train(context)
    manifest, files = read_package(Path(result["directory"]))
    proof = verify(context, manifest, files)
    assert context.calls.issued == context.calls.returned == context.calls.features == 4 * 2 * 254
    assert proof["control_steps"] == 2032 and proof["evaluated_episodes"] == 8
    assert result["diagnostic_bytes"] == proof["actual_raw_journal_bytes"]
    assert result["diagnostic_bytes"] <= result["projected_raw_journal_bytes"] <= 24 * 1024 * 1024
    zero = result["history"][0]["candidates"][0]
    for trial in zero["trials"]:
        rows = [json.loads(line) for line in files[trial["diagnostics"]["artifact"]].splitlines()]
        assert len(rows) == 508
        for row in rows[::2]:
            assert row["base_action"] == row["action"]
    (tmp_path / "mock-full-horizon-result.json").write_bytes(learner._bytes({
        "scope": "Authored environment outcomes, no native simulation or training experiment", **proof,
        "maximum_file_bytes_observed": max(len(raw) for name, raw in files.items() if name.startswith("diagnostics/")),
        "issued_controls": context.calls.issued, "returned_controls": context.calls.returned}))


def test_previous_best_and_zero_anchor_cache_do_not_issue_extra_controls(tmp_path, monkeypatch):
    context = build_context(tmp_path, monkeypatch)
    result = train(context, generations=2)
    manifest, files = read_package(Path(result["directory"]))
    proof = verify(context, manifest, files)
    assert result["cache_hits"] == 2 and proof["unique_candidates"] == 6
    assert context.calls.issued == 6 * 2 * 3
    assert result["history"][1]["candidates"][0]["cached"] is True
    assert result["history"][1]["candidates"][1]["cached"] is True


def test_projected_overflow_rejects_before_any_native_admission_or_environment(tmp_path, monkeypatch):
    context = build_context(tmp_path, monkeypatch, horizon=254)
    with pytest.raises(ValueError, match="Projected complete-interval"):
        train(context, generations=2, population=8)
    assert context.calls.source == context.calls.constructs == context.calls.resets == context.calls.issued == 0
    assert not (tmp_path / "trained").exists()


@pytest.mark.parametrize("failure", ["reset_failure", "step_failure"])
def test_failed_trials_preserve_attempted_returned_counts_and_never_publish_policy(tmp_path, monkeypatch, failure):
    context = build_context(tmp_path, monkeypatch, modes={failure: True if failure == "reset_failure" else 2})
    with pytest.raises((ValueError, RuntimeError)):
        train(context)
    manifest, files = read_package(tmp_path / "trained")
    state = json.loads(files["training.json"])
    assert json.loads(manifest)["status"] == "failed" and "failure.json" in files and "policy.json" not in files
    assert state["control_steps"] == context.calls.issued and state["returned_controls"] == context.calls.returned
    if failure == "step_failure":
        assert context.calls.issued == 2 and context.calls.returned == 1
        trial = state["active_candidates"][0]["trials"][0]
        assert trial["diagnostics"]["action_intents"] == 2 and trial["diagnostics"]["result_records"] == 1
        assert trial["status"] == "interrupted"


def test_prepared_policy_reads_exact_current_boundary_and_no_cold_dependencies(tiny, monkeypatch):
    context, _, _, files = tiny
    env = context.Env(context.scene, context.task)
    observation, _ = env.reset(seed=5, options=learner.TRAIN_RESETS[0])
    policy = json.loads(files["policy.json"])
    prepared = policy_module.prepare_equipment_feedback_policy(policy, env, base_policy_bytes=context.base_raw,
        source_manifest_sha256=learner._sha(context.source_manifest))
    before = context.calls.features
    monkeypatch.setattr(policy_module, "runtime_hashes", lambda *_: pytest.fail("Hot-path runtime hashing"))
    monkeypatch.setattr(v5, "validate_shared_exercise_feedback_policy", lambda *_: pytest.fail("Hot-path policy admission"))
    action, sample = prepared.sample(observation)
    assert context.calls.features == before + 1 and sample["observation"].tobytes() == observation.tobytes()
    next_observation, *_ = env.step(action)
    with pytest.raises(ValueError, match="stale"):
        prepared.sample(observation)
    prepared.sample(next_observation)
    env.feature_error = True
    with pytest.raises(ValueError, match="v5 features"):
        prepared.sample(next_observation)
    env.feature_error = False; env._engine = object()
    with pytest.raises(ValueError, match="unchanged active"):
        prepared.sample(next_observation)


@pytest.mark.parametrize("field,value", [("profile", "shared-exercise-policy-v5"), ("base_profile", "shared-exercise-policy-v4"),
    ("source_manifest_sha256", "0" * 64), ("base_policy_file_sha256", "0" * 64), ("base_policy_sha256", "0" * 64),
    ("gates_sha256", "0" * 64), ("runtime_sha256", {}), ("root_assistance", "disabled"),
    ("equipment_feedback_coefficients", [[0., 0.]] * 5), ("training_score", True)])
def test_new_policy_cannot_relabel_old_artifacts_or_change_pinned_context(tiny, field, value):
    context, _, _, files = tiny
    policy = json.loads(files["policy.json"]); policy[field] = value
    with pytest.raises(ValueError):
        policy_module.validate_equipment_feedback_policy(policy, context.scene, context.task,
            base_policy_bytes=context.base_raw, source_manifest_sha256=learner._sha(context.source_manifest))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lossless_arrays_preserve_signed_zero_subnormals_and_reject_oversized_payload(dtype):
    values = np.asarray([-0., 0., np.nextafter(dtype(0), dtype(1)), 1. / 3., -1e6], dtype=dtype)
    packed = learner._pack(values)
    assert learner._unpack(packed, (5,)).tobytes() == values.tobytes()
    packed["data"] += "AAAA"
    with pytest.raises(ValueError, match="byte bound"):
        learner._unpack(packed, (5,))


@pytest.mark.parametrize("mutation", ["coefficient", "rank", "selected", "settings", "source_pin", "base_pin",
    "observation", "base_features", "action", "actor_values", "result_metrics", "missing_result", "dependency",
    "control_clock", "cancelling_rewards"])
def test_resealed_saved_training_rejects_changed_math_provenance_and_accounting(tiny, mutation):
    context, result, _, original = tiny
    files = dict(original)
    if mutation in ("coefficient", "rank", "selected"):
        doc = json.loads(files["training.json"])
        if mutation == "coefficient": doc["history"][0]["candidates"][1]["equipment_feedback_coefficients"][0][0] += .001
        elif mutation == "rank": doc["history"][0]["candidates"][0]["optimization_rank"]["feasible"] = True
        else: doc["selected_candidate"]["candidate"] = 1
        files["training.json"] = learner._bytes(doc)
    elif mutation == "settings":
        doc = json.loads(files["settings.json"]); doc["training_seeds"][0] += 1
        files["settings.json"] = learner._bytes(doc)
    elif mutation in ("source_pin", "base_pin"):
        files["input-manifest.json" if mutation == "source_pin" else "base-policy.json"] += b"\n"
    elif mutation == "dependency":
        files["implementation/shared_exercise_equipment_feedback.py"] += b"\n"
    else:
        training = json.loads(files["training.json"])
        receipt = training["history"][0]["candidates"][0]["trials"][0]["diagnostics"]
        name = receipt["artifact"]
        lines = [json.loads(line) for line in files[name].splitlines()]
        if mutation == "missing_result": lines.pop()
        elif mutation == "result_metrics": lines[1]["info"]["equipment"][0]["speed_model_units_s"] += .1
        elif mutation == "control_clock":
            lines[1]["info"].update(observed_substeps=32, valid_substeps=32,
                                   simulated_duration_s=2 / 30., valid_observed_duration_s=2 / 30.)
        elif mutation == "cancelling_rewards":
            lines[1]["reward"] += .1; lines[3]["reward"] -= .1
        else:
            shape = json.loads(files["settings.json"])["diagnostic_schema"]["arrays"][mutation]
            array = learner._unpack(lines[0][mutation], shape).copy()
            array.flat[22 if mutation == "observation" else 0] += .001
            lines[0][mutation] = learner._pack(array)
        files[name] = b"".join(learner._compact(line) for line in lines)
        receipt.update(sha256=learner._sha(files[name]), raw_bytes=len(files[name]))
        files["training.json"] = learner._bytes(training)
    with pytest.raises(ValueError):
        verify(context, reseal(files), files)


def test_package_source_proof_precedes_saved_policy_admission(tiny, monkeypatch):
    context, result, manifest, files = tiny
    calls = []
    monkeypatch.setattr(learner, "_verified_inputs", lambda *_args, **_kwargs: (manifest, files))

    def custody(scene, directory, **options):
        calls.append("source_proof")
        assert scene == context.scene and options["expected_manifest_sha256"] == learner._sha(manifest)
        raise ValueError("authored source custody rejection")

    monkeypatch.setattr(learner, "collect_shared_source_files", custody)
    monkeypatch.setattr(learner, "verify_equipment_feedback_training", lambda *_args, **_kwargs: pytest.fail("Proof bypass"))
    with pytest.raises(ValueError, match="source custody"):
        learner.verify_equipment_feedback_package(Path(result["directory"]), manifest_sha256=learner._sha(manifest),
            base_policy_sha256=learner._sha(context.base_raw), source_manifest_sha256=learner._sha(context.source_manifest),
            deadline=time.monotonic() + 10.)
    assert calls == ["source_proof"]


def test_deadline_after_journaling_keeps_intent_without_counting_an_unissued_step(tmp_path, monkeypatch):
    context = build_context(tmp_path, monkeypatch)
    env = context.Env(context.scene, context.task)
    prepared = policy_module.PreparedEquipmentFeedbackPolicy(env, context.base, np.zeros((8, 2)))
    out = tmp_path / "journal"; out.mkdir()
    writer = learner._Writer(out, time.monotonic() + 10.)
    journal = learner._Journal(writer, "diagnostics/one.jsonl", context.scene, context.task, np.zeros((8, 2)),
        learner._static_metrics(context.scene, context.task, env.evaluator_sha256, learner.TRAIN_RESETS[0]))
    budget = learner.residual._Budget(3); budget.returned = 0
    expired = [False]
    original_attempt = journal.attempt

    def attempt(*args):
        original_attempt(*args)
        expired[0] = True

    def check(_deadline):
        if expired[0]: raise TimeoutError("authored elapsed deadline")

    monkeypatch.setattr(journal, "attempt", attempt)
    monkeypatch.setattr(learner, "_check", check)
    with pytest.raises(TimeoutError) as caught:
        learner._trial(env, prepared, 7, learner.TRAIN_RESETS[0], time.monotonic() + 10., budget, journal, lambda _: None)
    row = caught.value.partial_trial
    assert row["control_steps"] == row["returned_controls"] == budget.used == context.calls.issued == 0
    assert row["diagnostics"]["action_intents"] == 1 and row["diagnostics"]["result_records"] == 0


def test_complete_journal_file_count_is_reserved_before_environment_construction(tmp_path, monkeypatch):
    context = build_context(tmp_path, monkeypatch)
    monkeypatch.setattr(learner.residual, "MAX_OUTPUT_FILES", 40)
    with pytest.raises(ValueError, match="output ceiling"):
        train(context)
    assert context.calls.constructs == context.calls.resets == context.calls.issued == 0


def test_returned_engine_failure_preserves_original_absent_legacy_flag_and_is_ineligible(tmp_path, monkeypatch):
    context = build_context(tmp_path, monkeypatch, modes={"returned_engine_failure": True})
    result = train(context)
    manifest, files = read_package(Path(result["directory"]))
    proof = verify(context, manifest, files)
    failed_candidate = result["history"][0]["candidates"][0]
    trial = failed_candidate["trials"][0]
    assert trial["status"] == "failed" and trial["control_steps"] == trial["returned_controls"] == 2
    assert failed_candidate["optimization_rank"]["complete"] is False
    rows = [json.loads(line) for line in files[trial["diagnostics"]["artifact"]].splitlines()]
    assert rows[-1]["info"]["observation_status"] == "last_valid_control_boundary"
    assert "legacy_coupled_score_used" not in rows[-1]["info"]
    assert proof["control_steps"] == context.calls.issued == context.calls.returned == 23
    assert proof["selected_candidate"] == {"generation": 0, "candidate": 2}
