"""Failure timing must preserve reset identity, sparse clocks, custody and unchanged dynamics."""

from copy import deepcopy
from dataclasses import replace
import json
import shutil
import time

import mujoco
import numpy as np
import pytest

from weft import shared_exercise_diagnostics as module
from weft.io import canonical_hash, digest, write_json
from weft.pipeline import verify_artifacts
from weft.shared_exercise_recording import recording_scene
from weft.shared_exercise_task import GATES, SharedExerciseTaskEnv
from test_shared_exercise_recording import short_task
from test_shared_exercise_task import setup, sample, wrist_contact


def event_step(env, index, action=None):
    action = env.authored_action() if action is None else action
    _, reward, terminated, truncated, _ = env.step(action)
    return {"index": index, "type": "step", "action": action.tolist(), "reward": reward,
            "terminated": terminated, "truncated": truncated, "snapshot": env.snapshot(), "controller": "authored"}


@pytest.fixture(scope="module")
def saved_boundaries():
    scene, task = short_task()
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=0)
        events = [{"index": 0, "type": "start", "seed": 0, "snapshot": env.snapshot()}]
        for index in range(1, 4):
            events.append(event_step(env, index))
        return recording_scene(scene, task), events
    finally:
        env.close()


def artificial_peaks(events, changes):
    """Synthetic saved maxima isolate timeline logic without claiming physical replay."""
    events = deepcopy(events)
    for index, event in enumerate(events):
        metrics = event["snapshot"]["task_metrics"]
        for row in metrics["equipment"]:
            row.update({key: 0. for key in module.METRICS})
        metrics["equipment"][0].update(changes.get(index, {}))
        failures = [name for name in metrics["failed_gates"] if not name.startswith("equipment_stability:")]
        for row in metrics["equipment"]:
            row["within_limits"] = all(row[key] <= GATES["equipment_" + key] + 1e-12 for key in module.METRICS)
            if not row["within_limits"]:
                failures.append("equipment_stability:" + row["object_id"])
        metrics["failed_gates"] = failures
        metrics["success"] = metrics["completed_interval"] and not failures
        snapshot = event["snapshot"]
        snapshot["terminated"] = bool(metrics["failure"] or metrics["success"])
        snapshot["truncated"] = metrics["completed_interval"] and not snapshot["terminated"]
        if event["type"] == "step":
            event.update(terminated=snapshot["terminated"], truncated=snapshot["truncated"])
    return events


def seal_recording(root, scene, events):
    root.mkdir()
    files = {"scene.json": module._bytes(scene), "events.jsonl": b"".join(module._bytes(row) for row in events),
             "video.mp4": b"not decoded", "preview.gif": b"not decoded", "replay.html": b"not inspected"}
    files["recording.json"] = module._bytes({"schema_version": 1, "profile": "shared-exercise", "status": "closed",
        "scene_sha256": canonical_hash(scene), "events": len(events), "journal_bytes": len(files["events.jsonl"]),
        "seed": events[0]["seed"]})
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    write_json(root / "manifest.json", {"schema_version": 1, "video_frames": len(events),
        "files": {name: module._sha(raw) for name, raw in files.items()}})
    return root


def reseal(root):
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["files"] = {name: digest(root / name) for name in manifest["files"]}
    write_json(root / "manifest.json", manifest)


def test_first_boundary_all_ties_later_consequence_and_episode_resets(saved_boundaries, monkeypatch):
    scene, saved = saved_boundaries
    peaks = {"speed_model_units_s": .2, "angular_speed_rad_s": .8}
    first = artificial_peaks(saved, {1: peaks, 2: peaks, 3: peaks | {"displacement_model_units": .04}})
    second = artificial_peaks(saved, {3: {"orientation_rad": .2}})
    second[0].update(type="reset", seed=4)
    events = first + second
    for index, event in enumerate(events):
        event["index"] = index
    before = canonical_hash(events)
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("Saved timeline must not run physics"))
    result = module.recording_failure_timeline(scene, events)
    assert canonical_hash(events) == before
    assert [row["reset_event_index"] for row in result["episodes"]] == [0, 4]
    a, b = result["episodes"]
    assert a["first_equipment_failure"]["event_index"] == 1
    assert [row["metric"] for row in a["first_equipment_failure"]["crossings"]] == ["speed_model_units_s", "angular_speed_rad_s"]
    assert a["first_crossings"][-1]["event_index"] == 3
    assert a["first_crossings"][-1]["previous_safe_elapsed_s"] == pytest.approx(2 / 30)
    assert a["first_crossings"][-1]["value_scope"] == "cumulative_peak_not_boundary_instantaneous_value"
    assert b["first_equipment_failure"]["event_index"] == 7 and b["first_equipment_failure"]["control_step"] == 3
    assert b["first_crossings"][0]["previous_maximum"] == 0
    assert result["first_equipment_failure"] == a["first_equipment_failure"]


def test_tolerance_equality_is_not_a_crossing(saved_boundaries):
    scene, events = saved_boundaries
    maximum = GATES["equipment_speed_model_units_s"] + 1e-12
    events = artificial_peaks(events, {index: {"speed_model_units_s": maximum} for index in (1, 2, 3)})
    assert module.recording_failure_timeline(scene, events)["first_equipment_failure"] is None


def test_reset_failure_has_no_invented_safe_time(saved_boundaries):
    scene, events = saved_boundaries
    events = artificial_peaks(events[:1], {0: {"speed_model_units_s": .2}})
    first = module.recording_failure_timeline(scene, events)["first_equipment_failure"]["crossings"][0]
    assert first["status"] == "failed_at_reset"
    assert first["previous_safe_elapsed_s"] is first["previous_maximum"] is None


@pytest.mark.parametrize("kind", ["decreasing_peak", "missing_event", "advanced_reset", "evaluator", "action"])
def test_inconsistent_recording_cannot_publish_partial_timing(saved_boundaries, kind):
    scene, events = saved_boundaries
    events = artificial_peaks(events, {1: {"speed_model_units_s": .2}, 2: {"speed_model_units_s": .2},
                                       3: {"speed_model_units_s": .2}})
    if kind == "decreasing_peak":
        events = artificial_peaks(events, {1: {"speed_model_units_s": .2}})
    elif kind == "missing_event":
        events.pop(1)
        for i, row in enumerate(events):
            row["index"] = i
    elif kind == "advanced_reset":
        events[1].update(type="reset", seed=0)
    elif kind == "evaluator":
        events[2]["snapshot"]["task_metrics"]["evaluator_sha256"] = "0" * 64
    else:
        events[1]["action"][0] = -.999
    with pytest.raises(ValueError):
        module.recording_failure_timeline(scene, events)


def test_failed_engine_geometry_clock_is_not_labeled_current(monkeypatch):
    scene, task = short_task()
    env = SharedExerciseTaskEnv(scene, task)
    original = mujoco.mj_step

    def break_step(model, data):
        original(model, data)
        data.qpos[0] = np.nan

    try:
        env.reset(seed=0)
        events = [{"index": 0, "type": "start", "seed": 0, "snapshot": env.snapshot()}]
        monkeypatch.setattr(mujoco, "mj_step", break_step)
        events.append(event_step(env, 1))
        terminal = module.recording_failure_timeline(recording_scene(scene, task), events)["episodes"][0]["terminal"]
        assert terminal["engine_state_status"] == "last_valid_control_boundary"
        assert terminal["last_valid_geometry_elapsed_s"] == 0 < terminal["elapsed_s"]
        assert terminal["failure"] and not terminal["completed_interval"]
    finally:
        env.close()


def clock_fixture(a=58, b=62, total=257):
    return {"human_spec": {"reference": [{"time_s": 58 / 30}, {"time_s": 62 / 30}],
        "reference_evidence": [{"time_s": 58 / 30, "evidence_id": "a" * 64 + f":frame:{a}"},
                               {"time_s": 62 / 30, "evidence_id": "a" * 64 + f":frame:{b}"}]},
        "task": {"start_offset_s": 0.}, "source_binding": {"total_source_frames": total}}


def test_sparse_reference_bracket_does_not_fill_missing_slots_or_native_clock():
    clock = module._ReferenceClock(clock_fixture())
    row = clock.at(1 / 30)
    assert row["reference_clock_s"] == pytest.approx(59 / 30)
    assert row["source_frame_bracket"]["frames_without_reference_between"] == [59, 60, 61]
    assert row["source_frame_bracket"]["native_video_time_available"] is False
    exact = clock.at(4 / 30)["source_frame_bracket"]
    assert exact["status"] == "exact_reference" and exact["lower"] == exact["upper"]
    assert clock.at(1)["source_frame_bracket"]["status"] == "outside_reference"


@pytest.mark.parametrize("a,b,total", [(58, 1000000000, 1000000001), (58, 257, 257), (58, 1, 257), (58, 58, 257), (0, 2000, 2001)])
def test_frame_ids_and_range_materialization_are_bounded(a, b, total):
    with pytest.raises(ValueError):
        module._ReferenceClock(clock_fixture(a, b, total)).at(1 / 30)


def test_boundary_human_geom_is_labeled_as_its_collision_segment():
    result = module._contacts({"contact_details": [{"objects": ["right_elbow_geom", "bell"], "kind": "human_object",
        "normal_force": 2, "distance": -.01, "position": [0, 0, 0], "parts": [None, None]}]}, "bell")
    assert result[0]["human_geom"] == "right_elbow_geom" and result[0]["human_segment"] == "right_forearm"


def test_inspection_verifies_only_declared_bytes_no_media_or_physics(tmp_path, saved_boundaries, monkeypatch):
    scene, events = saved_boundaries
    root = seal_recording(tmp_path / "session", scene, events)
    (root / "video.mp4").write_bytes(b"Deliberately different media, not attested by this helper")
    original = module._read
    reads = []

    def read(fd, name, *args):
        assert name not in {"video.mp4", "preview.gif", "replay.html"}
        reads.append(name)
        return original(fd, name, *args)

    monkeypatch.setattr(module, "_read", read)
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("No physics in recording inspection"))
    result = module.inspect_recording_failure(root)
    proof = result["integrity"]
    assert proof["manifest_sha256"] == digest(root / "manifest.json")
    assert set(proof["required_files_sha256"]) == {"scene.json", "recording.json", "events.jsonl"}
    assert proof["media_verified"] is proof["action_replay_performed"] is False
    assert len(reads) == 8


@pytest.mark.parametrize("kind", ["tamper", "metadata", "partial", "unsealed", "duplicate", "profile", "count"])
def test_bad_sealed_inputs_reject_without_partial_summary(tmp_path, saved_boundaries, kind):
    scene, events = saved_boundaries
    root = seal_recording(tmp_path / "session", scene, events)
    if kind == "tamper":
        (root / "events.jsonl").write_bytes(b"{}\n")
    elif kind in {"metadata", "profile", "count"}:
        row = json.loads((root / "recording.json").read_bytes())
        row[{"metadata": "status", "profile": "profile", "count": "events"}[kind]] = "running" if kind == "metadata" else "wrong"
        write_json(root / "recording.json", row)
        reseal(root)
    elif kind == "partial":
        journal = root / "events.jsonl"
        journal.write_bytes(journal.read_bytes()[:-1])
        reseal(root)
    elif kind == "unsealed":
        (root / "manifest.json").unlink()
    else:
        (root / "recording.json").write_text('{"status":"closed","status":"closed"}')
        reseal(root)
    with pytest.raises(ValueError):
        module.inspect_recording_failure(root)


@pytest.mark.parametrize("kind", ["root", "ancestor", "file"])
def test_symlinks_are_rejected_at_every_path_level(tmp_path, saved_boundaries, kind):
    scene, events = saved_boundaries
    root = seal_recording(tmp_path / "session", scene, events)
    if kind == "file":
        path = root / "scene.json"
        path.rename(root / "original-scene.json")
        path.symlink_to(root / "original-scene.json")
    elif kind == "root":
        (tmp_path / "linked").symlink_to(root, target_is_directory=True)
        root = tmp_path / "linked"
    else:
        (tmp_path / "linked").symlink_to(tmp_path, target_is_directory=True)
        root = tmp_path / "linked" / "session"
    with pytest.raises(ValueError, match="unsafe|linked"):
        module.inspect_recording_failure(root)


def test_changed_input_during_reduction_is_rejected(tmp_path, saved_boundaries, monkeypatch):
    scene, events = saved_boundaries
    root = seal_recording(tmp_path / "session", scene, events)
    original = module.recording_failure_timeline

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        (root / "events.jsonl").write_bytes(b"{}\n")
        return result

    monkeypatch.setattr(module, "recording_failure_timeline", changed)
    with pytest.raises(ValueError, match="SHA256"):
        module.inspect_recording_failure(root)


def test_summary_event_episode_time_and_byte_bounds(saved_boundaries, monkeypatch):
    scene, events = saved_boundaries
    with pytest.raises(ValueError, match="1801"):
        module.recording_failure_timeline(scene, [events[0]] * 1802)
    repeated = [dict(deepcopy(events[0]), index=index, type="start" if index == 0 else "reset") for index in range(65)]
    with pytest.raises(ValueError, match="64-episode"):
        module.recording_failure_timeline(scene, repeated)
    with pytest.raises(ValueError, match="wall-time"):
        module.recording_failure_timeline(scene, events, max_seconds=float("nan"))
    monkeypatch.setattr(module, "MAX_SUMMARY_BYTES", 50)
    with pytest.raises(ValueError, match="summary"):
        module.recording_failure_timeline(scene, events)


def test_substep_first_gate_precedes_later_displacement_and_force_sign_is_explicit():
    scene, task, acc = setup(scale=2.)
    raw = []
    diagnostic = module._SubstepDiagnosis(scene, task, raw.append, time.monotonic() + 5)
    diagnostic.reset(acc.initial)
    contacts = (wrist_contact(force_on_human_world=(0., 0., -4.)),)
    for index in range(3):
        equipment = tuple(replace(row, velocity=(.4, 0., 0.) if index else (0., 0., 0.),
            position=(row.position[0] + (.08 if index == 2 else 0.), *row.position[1:]))
            if row.object_id == "working_bell" else row for row in acc.initial.values())
        record = sample(acc, index, equipment=equipment, contacts=contacts if index else ())
        acc.consume(record)
        diagnostic.consume(record)
    result = diagnostic.summary()
    assert result["first_equipment_failure"]["substep_index"] == 2
    assert [row["metric"] for row in result["first_crossings"]] == ["speed_model_units_s", "displacement_model_units"]
    first = result["first_crossings"][0]
    assert first["value"] == .2 and first["previous_maximum"] == 0
    assert first["solver_contact_time_s"] < first["first_failed_elapsed_s"]
    force = first["dynamic_contacts"][0]
    assert force["force_on_object_world"] == [-0., -0., 4.]
    assert force["impulse_on_object_world"][2] == pytest.approx(4 * scene["dt"] / scene["substeps"])
    assert result["equipment_maxima"] == acc.equipment
    assert len(raw) == 3


def test_substep_invalid_clock_has_no_invented_impulse_and_overflow_rejects():
    scene, task, acc = setup()
    rows = []
    diagnostic = module._SubstepDiagnosis(scene, task, rows.append, time.monotonic() + 5)
    diagnostic.reset(acc.initial)
    diagnostic.consume(sample(acc, 0, valid=False, interval_end_s=None, contacts=(wrist_contact(),), failure="unknown_clock"))
    row = json.loads(rows[0])
    assert row["dynamic_contacts"][0]["impulse_on_object_world"] is None
    assert diagnostic.summary()["first_equipment_failure"] is None
    diagnostic.contact_rows = 10000
    with pytest.raises(ValueError, match="10000"):
        diagnostic.consume(sample(acc, 1, contacts=(wrist_contact(),)))
    assert len(rows) == 1


@pytest.fixture(scope="module")
def analytic_training(tmp_path_factory):
    from test_shared_exercise_run import source_fixture
    from weft.shared_exercise_residual_learning import train_shared_exercise_residual_policy, prepare_shared_exercise_residual_policy

    root = tmp_path_factory.mktemp("diagnostic-training")
    source, _ = source_fixture(root)
    learning = root / "learning"
    train_shared_exercise_residual_policy(source, learning, generations=1, population=4, root_assistance=True)
    scene = json.loads((learning / "scene.json").read_bytes())
    task = json.loads((learning / "exercise-task.json").read_bytes())
    policy = json.loads((learning / "policy.json").read_bytes())
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=0)
        events = [{"index": 0, "type": "start", "seed": 0, "snapshot": env.snapshot()}]
        action = prepare_shared_exercise_residual_policy(policy, env)
        for index in range(1, 4):
            event = event_step(env, index, action())
            event.update(controller="learned", policy_sha256=digest(learning / "policy.json"))
            events.append(event)
        envelope = recording_scene(scene, task, policy_document=(learning / "policy.json").read_bytes())
        recording = seal_recording(root / "recording", envelope, events)
        return learning, recording
    finally:
        env.close()


def test_analytic_study_exactly_matches_training_and_preserves_inputs(tmp_path, analytic_training, monkeypatch):
    learning, recording = analytic_training
    before = {str(path): digest(path) for root in (learning, recording) for path in root.rglob("*") if path.is_file()}
    out = tmp_path / "diagnosis"
    scene = json.loads((learning / "scene.json").read_bytes())
    task = json.loads((learning / "exercise-task.json").read_bytes())
    reference = reference_fixture(tmp_path / "reference", scene, task)
    original = SharedExerciseTaskEnv.step
    steps = []

    def step(self, action):
        assert (out / "plan.json").is_file()
        assert json.loads((out / "plan.json").read_bytes())["maximum_training_rollouts"] == 1
        steps.append(1)
        return original(self, action)

    monkeypatch.setattr(SharedExerciseTaskEnv, "step", step)
    result = module.diagnose_shared_exercise_failure(recording, learning, out, reference_run=reference)
    assert result["status"] == "diagnosed" and result["exact_saved_training_match"]
    assert len(steps) == result["actual_control_steps"] == 3
    assert result["actual_physics_substeps"] == 48
    assert result["bytes"] == sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    assert result["bytes"] < module.MAX_STUDY_BYTES
    report = json.loads((out / "diagnosis.json").read_bytes())
    assert report["evaluation_used"] is False and report["source_video_decodes"] == 0
    assert report["actual_training_trial"]["final_snapshot_sha256"] == report["expected_final_snapshot_sha256"]
    assert report["training_substeps"]["observed_substeps"] == 48
    assert report["saved_kinematic_reference"]["manifest_sha256"] == digest(reference / "manifest.json")
    assert report["reference_overlap_artifact"] == "reference-overlaps.json"
    assert verify_artifacts(out)["manifest_sha256"] == result["manifest_sha256"]
    assert before == {str(path): digest(path) for root in (learning, recording) for path in root.rglob("*") if path.is_file()}


def test_nonmatching_training_terminal_seals_failure_without_retry(tmp_path, analytic_training, monkeypatch):
    learning, recording = analytic_training
    original = module._selected_training_trial

    def wrong(*args):
        candidate, trial = original(*args)
        return candidate, dict(trial, final_snapshot_sha256="0" * 64)

    monkeypatch.setattr(module, "_selected_training_trial", wrong)
    out = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="exact saved training"):
        module.diagnose_shared_exercise_failure(recording, learning, out)
    assert json.loads((out / "manifest.json").read_bytes())["status"] == "failed"
    failure = json.loads((out / "failure.json").read_bytes())
    assert failure["no_retry_performed"] and failure["observed_substeps"] == 48
    assert not (out / "diagnosis.json").exists()
    verify_artifacts(out)


def test_output_bytes_reserved_before_write_and_existing_output_preserved(tmp_path):
    output = module._StudyOutput(tmp_path / "bounded")
    output.write("evidence.json", b"{}")
    before = output.used
    output.used = module.MAX_STUDY_BYTES - 128 * 1024
    with pytest.raises(ValueError, match="before write"):
        output.write("too-big.json", b"x")
    assert not (output.root / "too-big.json").exists()
    with pytest.raises(FileExistsError):
        module._StudyOutput(output.root)
    assert (output.root / "evidence.json").read_bytes() == b"{}" and before == 2


def reference_fixture(root, scene, task):
    env = SharedExerciseTaskEnv(scene, task)
    try:
        frames = [env._engine.reference_snapshot(index) for index in range(len(scene["human_spec"]["reference"]))]
    finally:
        env.close()
    first = frames[0]
    first["contact_details"].append({"objects": ["right_elbow_geom", scene["objects"][0]["id"]],
        "kind": "human_object", "position": [0., 0., 0.], "distance": -.02, "normal_force": 123456.})
    first["contact_count"] = len(first["contact_details"])
    first["contacts"] = [list(pair) for pair in sorted({tuple(sorted(row["objects"])) for row in first["contact_details"]})]
    root.mkdir()
    files = {"scene.json": scene, "exercise-task.json": task,
        "exercise-lab.json": {"scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task)},
        "human-reference.json": {"schema_version": 1, "profile": "shared-world-kinematic-reference-v1",
            "frames": frames, "reference_poses_total": len(frames), "reference_poses_included": len(frames)}}
    for name, data in files.items():
        write_json(root / name, data)
    write_json(root / "manifest.json", {"schema_version": 1, "status": "ready", "control_profile": "shared_exercise",
        "files": {name: digest(root / name) for name in files}})
    return root


def test_saved_kinematic_overlaps_never_infer_solver_force_or_dynamics(tmp_path, saved_boundaries, monkeypatch):
    envelope, _ = saved_boundaries
    scene, task = envelope["engine_scene"], envelope["exercise_task"]
    root = reference_fixture(tmp_path / "reference", scene, task)
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("No dynamics when reading saved references"))
    _, _, identity, summary = module._reference_inputs(root, scene, task, time.monotonic() + 5)
    assert identity["manifest_sha256"] == digest(root / "manifest.json")
    overlap = summary["rows"][0]["overlaps"][-1]
    assert overlap["human_segment"] == "right_forearm" and overlap["distance_world"] == -.02
    assert set(overlap) == {"object_id", "human_geom", "human_segment", "distance_world", "penetration_model_units"}
    assert "123456" not in json.dumps(summary)
    assert "No dynamic contact" in summary["scope"]


@pytest.mark.parametrize("kind", ["scene", "dynamic", "coverage", "moved_equipment"])
def test_reference_cannot_substitute_scene_or_dynamic_state(tmp_path, saved_boundaries, kind):
    envelope, _ = saved_boundaries
    scene, task = envelope["engine_scene"], envelope["exercise_task"]
    root = reference_fixture(tmp_path / "reference", scene, task)
    reference = json.loads((root / "human-reference.json").read_bytes())
    if kind == "scene":
        changed = deepcopy(scene)
        changed["gravity"][2] = -8
        write_json(root / "scene.json", changed)
    elif kind == "dynamic":
        reference["frames"][0]["substeps"] = 16
    elif kind == "coverage":
        reference["frames"].pop()
    else:
        reference["frames"][0]["objects"][0]["position"][0] += .1
    write_json(root / "human-reference.json", reference)
    reseal(root)
    with pytest.raises(ValueError):
        module._reference_inputs(root, scene, task, time.monotonic() + 5)


@pytest.mark.parametrize("name,raw", [("manifest.json", b"[]"), ("events.jsonl", b"[]\n")])
def test_non_object_sealed_inputs_are_explicit_errors(tmp_path, saved_boundaries, name, raw):
    scene, events = saved_boundaries
    root = seal_recording(tmp_path / "session", scene, events)
    (root / name).write_bytes(raw)
    if name != "manifest.json":
        reseal(root)
    with pytest.raises(ValueError):
        module.inspect_recording_failure(root)


def test_replaced_recording_between_inspection_and_plan_is_rejected_before_physics(tmp_path, analytic_training, monkeypatch):
    learning, original_recording = analytic_training
    recording = tmp_path / "recording"
    shutil.copytree(original_recording, recording)
    original = module.inspect_recording_failure

    def inspect(*args, **kwargs):
        result = original(*args, **kwargs)
        metadata = json.loads((recording / "recording.json").read_bytes())
        metadata["changed_after_inspection"] = True
        write_json(recording / "recording.json", metadata)
        reseal(recording)
        return result

    monkeypatch.setattr(module, "inspect_recording_failure", inspect)
    monkeypatch.setattr(SharedExerciseTaskEnv, "step", lambda *args: pytest.fail("Changed inputs cannot reach dynamics"))
    out = tmp_path / "unused"
    with pytest.raises(ValueError, match="changed after"):
        module.diagnose_shared_exercise_failure(recording, learning, out)
    assert not out.exists()


@pytest.mark.parametrize("kind", ["seed", "reset", "control_bound", "incomplete"])
def test_only_original_first_training_start_is_eligible(analytic_training, kind):
    learning, _ = analytic_training
    policy, training, settings = [json.loads((learning / name).read_bytes()) for name in ("policy.json", "training.json", "settings.json")]
    _, trial = module._selected_training_trial(policy, training, settings)
    if kind == "seed":
        trial["seed"] += 1
    elif kind == "reset":
        trial["reset_options"] = {"jitter": True}
    elif kind == "control_bound":
        trial["control_steps"] = settings["horizon_steps"] = 255
    else:
        trial["status"] = "interrupted"
    with pytest.raises(ValueError, match="first original"):
        module._selected_training_trial(policy, training, settings)
