"""One bounded continuation keeps independent comparison and a recordable task together."""

import json
from pathlib import Path
import shutil

import pytest

from test_shared_exercise_run import source_fixture
from test_shared_world_fullspan import fullspan_scene
from weft import exercise_learning_pipeline as pipeline
from weft.io import digest, write_json
from weft.lab import Lab
from weft.pipeline import verify_artifacts
from weft.shared_exercise_task import SharedExerciseTaskEnv, build_shared_exercise_task
from weft.shared_world import validate_shared_world


SETTINGS = {"training_seed": 4151, "evaluation_seed": 6173, "generations": 1, "population": 4, "trials": 2}


def _json(path):
    return json.loads(path.read_bytes())


def _files(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _seal(root):
    write_json(root / "manifest.json", {"schema_version": 1, "status": "ready",
        "files": {name: digest(root / name) for name in _files(root) if name != "manifest.json"}})


@pytest.fixture(scope="module")
def completed_workflow(tmp_path_factory):
    root = tmp_path_factory.mktemp("exercise_learning_workflow")
    source = root / "source"
    source.mkdir()
    scene = fullspan_scene(steps=3)
    names = {"ball": "training-weight", "marker_box": "foam-pad", "table": "support-plinth"}
    for row in scene["objects"]:
        row["id"] = names[row["id"]]
    scene["task"]["object_goal"]["target_object_id"] = "training-weight"
    scene.pop("action_contract")
    scene = validate_shared_world(scene)
    task = build_shared_exercise_task(scene, root_assistance=True, contact_requirements=[
        {"object_id": "foam-pad", "wrist": "left", "kind": "wrist_region_contact", "start_step": 0, "end_step": 3}])
    write_json(source / "scene.json", scene)
    write_json(source / "exercise-task.json", task)
    write_json(source / "source.json", {"scope": "Analytical fixture only; not a measured exercise"})
    (source / "source.webm").write_bytes(b"Preserved analytical bytes; no video decoding in this test.\n")
    _seal(source)
    before = _files(source)
    out, events, steps, labs = root / "workflow", [], [], []
    native_step, prepare = SharedExerciseTaskEnv.step, pipeline.prepare_shared_exercise_lab

    def step(self, action):
        frozen = _json(out / "plan.json")
        assert frozen["scene_file_sha256"] == digest(source / "scene.json")
        assert frozen["task_file_sha256"] == digest(source / "exercise-task.json")
        assert frozen["maximum_control_steps"] == 48
        assert events[-1]["stage"] in {"training", "independent_evaluation"}
        steps.append(events[-1]["stage"])
        return native_step(self, action)

    def prepare_one(run, destination, **options):
        labs.append((run, destination, options))
        return prepare(run, destination, **options)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SharedExerciseTaskEnv, "step", step)
        patch.setattr(pipeline, "prepare_shared_exercise_lab", prepare_one)
        result = pipeline.prepare_exercise_learning_pipeline(source, out, train_settings=SETTINGS,
            max_seconds=60, progress=events.append)
    return source, out, result, events, steps, labs, before


def test_one_frozen_workflow_keeps_failed_task_visible_and_records_exact_policy(completed_workflow):
    source, out, result, events, steps, labs, before = completed_workflow
    assert result["status"] == result["lab_status"] == "ready"
    assert [path.name for path in sorted(out.iterdir()) if path.is_dir()] == ["evaluation", "lab", "selected", "training"]
    assert len(labs) == 1 and labs[0][:2] == (out / "selected", out / "lab")
    assert labs[0][2]["task_path"] == out / "exercise-task.json"
    assert labs[0][2]["policy_run"] == out / "selected"
    assert labs[0][2]["evaluation_run"] == out / "evaluation"
    assert [(row["stage"], row["status"]) for row in events] == [
        ("training", "running"), ("training", "trained"),
        ("training_selection", "running"), ("training_selection", "selected"),
        ("independent_evaluation", "running"), ("independent_evaluation", "evaluated"),
        ("interactive_lab", "running"), ("interactive_lab", "ready")]
    assert events[-1]["lab_status"] == "ready" and events[-1]["lab_directory"] == str(out / "lab")
    assert events[-1]["scene_sha256"] == result["scene_sha256"]
    assert events[-1]["task_sha256"] == result["task_sha256"]
    assert len(steps) == result["training"]["control_steps"] + result["evaluation"]["control_steps"] == 48
    assert steps.count("training") == steps.count("independent_evaluation") == 24
    plan = _json(out / "plan.json")
    assert plan["training_settings"] == pipeline.validate_train_settings(SETTINGS)
    assert set(plan["training_seeds"]).isdisjoint(plan["evaluation_seeds"])
    assert plan["evaluation_feedback_to_optimizer"] is False
    assert plan["policy_profile"] == "shared-exercise-policy-v2"
    assert plan["maximum_physics_substeps"] == 48 * _json(source / "scene.json")["substeps"]
    assert plan["workers"] == 1 and plan["external_api_usd"] == 0
    assert result["selection"]["selection_status"] == "retained_original_no_feasible_candidate"
    assert result["selection"]["feasible_candidates"] == 0
    assert result["evaluation"]["comparison_complete"] is True
    for row in result["evaluation"]["controllers"].values():
        assert row["successes"] == 0 and row["completed_intervals"] == 2 and row["reset_failures"] == 0
    assert result["physical_validation"] == "unverified"
    policy_sha = digest(out / "selected/policy.json")
    assert policy_sha == digest(out / "evaluation/policy.json") == digest(out / "lab/run/policy.json")
    for directory in ("training", "selected", "evaluation", "lab/run"):
        assert (out / directory / "scene.json").read_bytes() == before["scene.json"]
        assert (out / directory / "exercise-task.json").read_bytes() == before["exercise-task.json"]
    manifest = _json(out / "manifest.json")
    assert not any(name.startswith("lab/") and not name.startswith("lab/run/") for name in manifest["files"])
    run_before = _files(out / "lab/run")
    lab = Lab(out / "lab")
    try:
        status = lab.summary()
        assert status["learned_policy_available"] is True
        assert status["exercise_policy"]["evaluation_status"] == "evaluated"
        lab.control("/api/record/start", {})
        state = lab.control("/api/step", {"policy": True})["snapshot"]
        assert not any(state["engine_state"]["normalized_action"][22:])
        assert all(row["virtual_force"] == [0., 0., 0.] for row in state["engine_state"]["objects"]
                   if row["role"] != "static")
        lab.control("/api/record/stop", {})
        recording = lab.export_future.result(timeout=30)
        assert recording["action_replay"]["exact_state_replay"] is True
        assert recording["action_replay"]["learned_actions_verified"] == 1
        assert recording["action_replay"]["policy_sha256"] == policy_sha
    finally:
        lab.close()
    assert _files(source) == before and _files(out / "lab/run") == run_before
    assert verify_artifacts(out)["manifest_sha256"] == result["manifest_sha256"]
    assert result["bytes"] < pipeline.MAX_BYTES and result["files"] < pipeline.MAX_FILES


@pytest.mark.parametrize("settings", [None, {}, [], {"evaluation_seed": 1729}, {"evaluation_seed": True},
    {"evaluation_seed": -1}, {"evaluation_seed": 2**31}, {"evaluation_seed": 2, "generations": 0},
    {"evaluation_seed": 2, "generations": 7}, {"evaluation_seed": 2, "population": 3},
    {"evaluation_seed": 2, "population": 13}, {"evaluation_seed": 2, "trials": 0},
    {"evaluation_seed": 2, "trials": 4}, {"evaluation_seed": 2, "jitter_scale": 0},
    {"evaluation_seed": 2, "jitter_scale": float("nan")}, {"evaluation_seed": 2, "jitter_scale": True},
    {"evaluation_seed": 2, "horizon_steps": 1}, {"evaluation_seed": 2, "policy_family": "v3"}])
def test_invalid_settings_reject_before_source_io_or_output(tmp_path, monkeypatch, settings):
    monkeypatch.setattr(pipeline, "_plain_path", lambda *_: pytest.fail("Invalid settings reached IO"))
    with pytest.raises(ValueError):
        pipeline.prepare_exercise_learning_pipeline("missing", tmp_path / "out", train_settings=settings)
    assert not (tmp_path / "out").exists()


def test_settings_defaults_are_detached_and_evaluation_seed_is_explicit():
    supplied = {"evaluation_seed": 6173}
    result = pipeline.validate_train_settings(supplied)
    supplied["evaluation_seed"] = 1
    assert result == {"generations": 2, "population": 8, "training_seed": 1729,
                      "evaluation_seed": 6173, "trials": 3, "jitter_scale": .1}


def _stage_package(path, scene_raw, task_raw, status, **documents):
    path.mkdir()
    (path / "scene.json").write_bytes(scene_raw)
    (path / "exercise-task.json").write_bytes(task_raw)
    for name, value in documents.items():
        write_json(path / name, value)
    write_json(path / "manifest.json", {"schema_version": 1, "status": status,
        "files": {name: digest(path / name) for name in _files(path)}})
    return digest(path / "manifest.json")


def _stub_stages(monkeypatch):
    """Inject completed stage boundaries to test failures without repeating the native search."""
    calls = []
    monkeypatch.setattr(SharedExerciseTaskEnv, "step", lambda *_: pytest.fail("Fault-boundary tests cannot advance physics"))

    def training(run, out, **options):
        calls.append("training")
        scene, task = (run / "scene.json").read_bytes(), Path(options["task_path"]).read_bytes()
        horizon = json.loads(task)["interval"]["horizon_steps"]
        settings = {"training_seeds": pipeline.residual._seed_plan(options["seed"], 2),
                    "maximum_control_steps": 2 * options["generations"] * options["population"] * horizon}
        seal = _stage_package(out, scene, task, "trained", **{"settings.json": settings, "policy.json": {"fault_fixture": True}})
        return {"manifest_sha256": seal, "policy_sha256": digest(out / "policy.json"), "control_steps": 0,
                "evaluated_episodes": 0, "completed_generations": options["generations"]}

    def selection(run, out, **_options):
        calls.append("training_selection")
        shutil.copytree(run, out)
        return {"manifest_sha256": digest(out / "manifest.json"), "policy_sha256": digest(out / "policy.json"),
                "selection_status": "retained_original_no_feasible_candidate", "feasible_candidates": 0,
                "unique_candidates": 1, "selected_candidate": {"generation": 0, "candidate": 0}}

    def evaluation(run, out, **options):
        calls.append("independent_evaluation")
        scene, task = (run / "scene.json").read_bytes(), (run / "exercise-task.json").read_bytes()
        seeds = pipeline.residual._seed_plan(options["seed"], options["trials"], _json(run / "settings.json")["training_seeds"])
        seal = _stage_package(out, scene, task, "evaluated", **{"policy.json": _json(run / "policy.json")})
        return {"manifest_sha256": seal, "policy_sha256": digest(out / "policy.json"),
            "evaluation_seeds": seeds, "maximum_control_steps": 4 * options["trials"] * json.loads(task)["interval"]["horizon_steps"],
            "control_steps": 0, "comparison_complete": False, "independent_changed_states": 0,
            "controllers": {name: {"successes": 0, "completed_intervals": 0, "reset_failures": options["trials"],
                                   "mean_score": -2.} for name in pipeline.residual.CONTROLLERS}}

    def lab(run, out, **options):
        calls.append("interactive_lab")
        out.mkdir()
        scene, task = (run / "scene.json").read_bytes(), Path(options["task_path"]).read_bytes()
        seal = _stage_package(out / "run", scene, task, "ready")
        write_json(out / "status.json", {"status": "ready", "scope": "Fault-injection fixture only"})
        return {"manifest_sha256": seal, "lab_directory": str(out),
                "scene_sha256": pipeline.canonical_hash(json.loads(scene)), "task_sha256": pipeline.canonical_hash(json.loads(task))}

    monkeypatch.setattr(pipeline.residual, "train_shared_exercise_residual_policy", training)
    monkeypatch.setattr(pipeline, "select_feasible_shared_exercise_policy", selection)
    monkeypatch.setattr(pipeline.residual, "evaluate_shared_exercise_residual_policy", evaluation)
    monkeypatch.setattr(pipeline, "prepare_shared_exercise_lab", lab)
    return calls


@pytest.mark.parametrize("stage,owner,name,preserved", [
    ("training", "residual", "train_shared_exercise_residual_policy", []),
    ("training_selection", "pipeline", "select_feasible_shared_exercise_policy", ["training"]),
    ("independent_evaluation", "residual", "evaluate_shared_exercise_residual_policy", ["training", "selected"]),
    ("interactive_lab", "pipeline", "prepare_shared_exercise_lab", ["training", "selected", "evaluation"]),
])
def test_stage_failure_preserves_completed_prefix_and_never_starts_later_work(tmp_path, monkeypatch, stage, owner, name, preserved):
    source, _ = source_fixture(tmp_path)
    original = _files(source)
    calls = _stub_stages(monkeypatch)

    def fail(_run, out, **_options):
        calls.append(stage)
        out.mkdir()
        (out / "partial.txt").write_text("Owned partial evidence retained")
        raise RuntimeError(f"injected {stage} failure")

    monkeypatch.setattr(pipeline.residual if owner == "residual" else pipeline, name, fail)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match=f"injected {stage}"):
        pipeline.prepare_exercise_learning_pipeline(source, out, train_settings=SETTINGS)
    report = _json(out / "result.json")
    assert report["status"] == "failed" and report["lab_status"] == "not_ready"
    assert report["stages"][-1]["stage"] == stage and report["stages"][-1]["status"] == "running"
    assert calls == list(pipeline.STAGE_LIMITS)[:list(pipeline.STAGE_LIMITS).index(stage) + 1]
    for name in preserved:
        assert (out / name / "manifest.json").is_file()
    assert _files(source) == original
    assert verify_artifacts(out)["verified_files"] > 0


@pytest.mark.parametrize("change", ["scene", "failure_marker", "runtime", "plan", "task"])
def test_changed_frozen_input_stops_before_training(tmp_path, monkeypatch, change):
    source, _ = source_fixture(tmp_path)
    out = tmp_path / "changed"
    calls = _stub_stages(monkeypatch)
    native_runtime = pipeline._runtime

    def tamper(event):
        if event["stage"] != "training" or event["status"] != "running":
            return
        if change == "scene":
            (source / "scene.json").write_text("{}")
        elif change == "failure_marker":
            write_json(source / "late-failure.json", {"status": "failed"})
        elif change == "runtime":
            def changed():
                files, versions = native_runtime()
                return {**files, "changed.py": "0" * 64}, versions
            monkeypatch.setattr(pipeline, "_runtime", changed)
        else:
            (out / ("plan.json" if change == "plan" else "exercise-task.json")).write_text("{}")

    with pytest.raises((ValueError, RuntimeError), match="SHA256|failure marker|runtime"):
        pipeline.prepare_exercise_learning_pipeline(source, out, train_settings=SETTINGS, progress=tamper)
    assert calls == []
    assert _json(out / "result.json")["status"] == "failed"
    assert not (out / "training").exists()


@pytest.mark.parametrize("mode", ["training", "legacy_evaluation"])
def test_oversized_whole_interval_rejects_before_output_without_clipping(tmp_path, monkeypatch, mode):
    source = tmp_path / "source"
    source.mkdir()
    scene = fullspan_scene(steps=300 if mode == "training" else 61)
    if mode == "legacy_evaluation":
        scene["profile"] = "shared-world-v1"
    write_json(source / "scene.json", scene)
    _seal(source)
    before = _files(source)
    calls = _stub_stages(monkeypatch)
    with pytest.raises(ValueError, match="full-interval action budget"):
        pipeline.prepare_exercise_learning_pipeline(source, tmp_path / "out", train_settings={"evaluation_seed": 6173})
    assert calls == [] and not (tmp_path / "out").exists() and _files(source) == before


@pytest.mark.parametrize("name,listed", [("failure.json", False), ("study-failure.json", False),
                                        ("policy.json", True), ("evaluation.json", True), ("construction-seed.json", True)])
def test_failed_or_previous_learning_input_is_rejected_before_output(tmp_path, monkeypatch, name, listed):
    source, _ = source_fixture(tmp_path)
    write_json(source / name, {"status": "failed"})
    if listed:
        _seal(source)
    calls = _stub_stages(monkeypatch)
    with pytest.raises(ValueError):
        pipeline.prepare_exercise_learning_pipeline(source, tmp_path / "out", train_settings=SETTINGS)
    assert calls == [] and not (tmp_path / "out").exists()


def test_sealed_task_cannot_be_overridden_by_an_external_file(tmp_path, monkeypatch):
    source, scene = source_fixture(tmp_path)
    write_json(source / "exercise-task.json", build_shared_exercise_task(scene))
    _seal(source)
    replacement = tmp_path / "replacement.json"
    write_json(replacement, build_shared_exercise_task(scene, root_assistance=True))
    calls = _stub_stages(monkeypatch)
    with pytest.raises(ValueError, match="SHA256"):
        pipeline.prepare_exercise_learning_pipeline(source, tmp_path / "out", train_settings=SETTINGS, task_path=replacement)
    assert calls == [] and not (tmp_path / "out").exists()


def test_complete_stage_capacity_is_reserved_before_training(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path)
    calls = _stub_stages(monkeypatch)
    monkeypatch.setattr(pipeline, "MAX_BYTES", pipeline.STAGE_LIMITS["training"][1])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="reserve the complete training"):
        pipeline.prepare_exercise_learning_pipeline(source, out, train_settings=SETTINGS)
    assert calls == [] and not (out / "training").exists()
    assert _json(out / "result.json")["status"] == "failed"


def test_admission_deadline_creates_no_orphan_output(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path)
    calls = _stub_stages(monkeypatch)
    monkeypatch.setattr(pipeline, "_check", lambda *_: (_ for _ in ()).throw(TimeoutError("admission deadline")))
    with pytest.raises(TimeoutError, match="admission deadline"):
        pipeline.prepare_exercise_learning_pipeline(source, tmp_path / "out", train_settings=SETTINGS)
    assert calls == [] and not (tmp_path / "out").exists()


@pytest.mark.parametrize("late_failure", ["callback", "seal"])
def test_ready_lab_is_reported_immediately_and_survives_later_failure(tmp_path, monkeypatch, late_failure):
    source, _ = source_fixture(tmp_path)
    calls = _stub_stages(monkeypatch)
    events = []

    def progress(event):
        events.append(event)
        if event["stage"] == "interactive_lab" and event["status"] == "ready" and late_failure == "callback":
            raise RuntimeError("late callback failed")

    if late_failure == "seal":
        monkeypatch.setattr(pipeline, "_seal", lambda *_: (_ for _ in ()).throw(TimeoutError("late seal deadline")))
    out = tmp_path / "out"
    with pytest.raises((RuntimeError, TimeoutError), match="late"):
        pipeline.prepare_exercise_learning_pipeline(source, out, train_settings=SETTINGS, progress=progress)
    assert calls == list(pipeline.STAGE_LIMITS)
    assert events[-1]["lab_status"] == "ready" and events[-1]["lab_directory"] == str(out / "lab")
    assert len([row for row in events if row["stage"] == "interactive_lab"]) == 2
    result = _json(out / "result.json")
    assert result["status"] == "failed" and result["lab_status"] == "ready"
    assert result["scene_sha256"] == events[-1]["scene_sha256"]
    assert result["task_sha256"] == events[-1]["task_sha256"]
    assert (out / "lab/run/manifest.json").is_file()
    if late_failure == "seal":
        assert result["seal_scope"] == "failure_receipt_only" and result["unsealed_stage_outputs_preserved"] is True
    verify_artifacts(out)


@pytest.mark.parametrize("prefix,stage,status", [("training", "training", "trained"),
    ("selected", "training_selection", "selected"), ("evaluation", "independent_evaluation", "evaluated"),
    ("lab/run", "interactive_lab", "ready")])
@pytest.mark.parametrize("mutation", ["reseal", "marker"])
def test_completed_child_identity_cannot_be_replaced_between_stages(tmp_path, monkeypatch, prefix, stage, status, mutation):
    source, _ = source_fixture(tmp_path)
    calls = _stub_stages(monkeypatch)
    out = tmp_path / "out"

    def tamper(event):
        if (event["stage"], event["status"]) != (stage, status):
            return
        child = out / prefix
        if mutation == "marker":
            write_json(child / "late-failure.json", {"status": "failed"})
        else:
            manifest = _json(child / "manifest.json")
            manifest["replaced_after_consumption"] = True
            write_json(child / "manifest.json", manifest)

    with pytest.raises(ValueError, match="SHA256|failure marker"):
        pipeline.prepare_exercise_learning_pipeline(source, out, train_settings=SETTINGS, progress=tamper)
    assert calls == list(pipeline.STAGE_LIMITS)[:list(pipeline.STAGE_LIMITS).index(stage) + 1]
    report = _json(out / "result.json")
    assert report["status"] == "failed"
    if mutation == "reseal":
        assert report["stage_manifests"][prefix] != digest(out / prefix / "manifest.json")
        assert report["seal_scope"] == "failure_receipt_only"
    verify_artifacts(out)


@pytest.mark.parametrize("prefix", ["training", "selected", "evaluation", "lab/run"])
def test_final_single_hash_pass_rejects_changed_consumed_child_bytes(tmp_path, monkeypatch, prefix):
    source, _ = source_fixture(tmp_path)
    _stub_stages(monkeypatch)
    out = tmp_path / "out"

    def tamper(event):
        if (event["stage"], event["status"]) == ("interactive_lab", "ready"):
            (out / prefix / "scene.json").write_text("{}")

    with pytest.raises(ValueError, match="completed workflow stage changed"):
        pipeline.prepare_exercise_learning_pipeline(source, out, train_settings=SETTINGS, progress=tamper)
    report = _json(out / "result.json")
    assert report["status"] == "failed" and report["seal_scope"] == "failure_receipt_only"
    assert report["unsealed_stage_outputs_preserved"] is True
    verify_artifacts(out)


def test_progress_cannot_mutate_retained_summary_objects(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path)
    _stub_stages(monkeypatch)
    out = tmp_path / "out"

    def mutate(event):
        if event["stage"] == "training_selection" and event["status"] == "selected":
            event["selected_candidate"]["candidate"] = 100
        if event["stage"] == "independent_evaluation" and event["status"] == "evaluated":
            event["controllers"]["learned"]["successes"] = 100

    result = pipeline.prepare_exercise_learning_pipeline(source, out, train_settings=SETTINGS, progress=mutate)
    assert result["selection"]["selected_candidate"]["candidate"] == 0
    assert result["evaluation"]["controllers"]["learned"]["successes"] == 0
    assert _json(out / "stages.json")["stages"] == result["stages"]


@pytest.mark.parametrize("via_other", [False, True])
def test_parent_path_alias_cannot_create_output_inside_immutable_input(tmp_path, monkeypatch, via_other):
    source, _ = source_fixture(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    before = _files(source)
    target = (other if via_other else source) / ".." / "source" / "new"
    calls = _stub_stages(monkeypatch)
    with pytest.raises(ValueError, match="parent-directory traversal"):
        pipeline.prepare_exercise_learning_pipeline(source, target, train_settings=SETTINGS)
    assert calls == [] and not (source / "new").exists() and _files(source) == before


@pytest.fixture(scope="module")
def untrained_lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("workflow_untrained_input")
    source, _ = source_fixture(root)
    lab = root / "lab"
    pipeline.prepare_shared_exercise_lab(source, lab)
    return lab


@pytest.mark.parametrize("changed", ["config", "status", "failure_marker"])
def test_consumed_outer_lab_metadata_is_pinned_before_training(tmp_path, monkeypatch, untrained_lab, changed):
    source = tmp_path / "input-lab"
    shutil.copytree(untrained_lab, source)
    calls = _stub_stages(monkeypatch)

    def tamper(event):
        if (event["stage"], event["status"]) == ("training", "running"):
            if changed == "failure_marker":
                write_json(source / "late-failure.json", {"status": "failed"})
            else:
                path = source / ("lab-config.json" if changed == "config" else "status.json")
                value = _json(path)
                value["changed_after_admission"] = True
                write_json(path, value)

    out = tmp_path / "out"
    with pytest.raises(ValueError, match="SHA256|failure marker"):
        pipeline.prepare_exercise_learning_pipeline(source, out, train_settings=SETTINGS, progress=tamper)
    assert calls == [] and not (out / "training").exists()
    assert set(_json(out / "plan.json")["lab_metadata"]) == {"lab-config.json", "status.json"}
