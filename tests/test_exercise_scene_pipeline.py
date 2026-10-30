import json
from types import SimpleNamespace

import pytest

from test_equipment_proposal import packages
from weft import exercise_scene_pipeline as pipeline
from weft.io import digest, write_json
from weft.lab import Lab


def test_full_reference_assembly_uses_one_candidate_and_one_lab(tmp_path):
    from test_equipment_proposal import seal

    body, equipment, config_path, original, _ = packages(tmp_path)
    original["task"]["horizon_steps"] = 3
    write_json(body / "scene.json", original)
    seal(body)
    out = tmp_path / "full-reference"
    result = pipeline.prepare_exercise_scene(body, equipment, config_path, out,
                                            root_assistance=True, full_reference=True)
    assert result["status"] == "ready"
    ready_world = next(row for row in result["stages"] if row["stage"] == "shared_world" and row["status"] == "ready")
    assert ready_world["profile"] == "shared-world-v2"
    assert ready_world["reference_interval"]["selected"]["horizon_steps"] == 60
    assert sorted(path.name for path in out.iterdir() if path.is_dir()) == ["candidate", "lab", "proposal"]
    assert json.loads((out / "settings.json").read_bytes())["full_reference"] is True
    scene = json.loads((out / "candidate/scene.json").read_bytes())
    task = json.loads((out / "lab/run/exercise-task.json").read_bytes())
    assert scene["profile"] == "shared-world-v2" and task["interval"] == {"start_offset_s": 0., "horizon_steps": 60}
    assert scene["human_spec"] == original["human_spec"]
    assert task["control"]["root_assistance"] == "explicit_3_axis"
    assert task["contact_requirements"] == []
    lab = Lab(out / "lab")
    try:
        assert lab.summary()["learned_policy_available"] is False
        state = lab.control("/api/step", {"controller": "reference"})["snapshot"]
        assert state["engine_state"]["steps"] == 1
        assert not any(state["engine_state"]["normalized_action"][22:])
    finally:
        lab.close()
    manifest = json.loads((out / "manifest.json").read_bytes())
    assert all(digest(out / name) == sha for name, sha in manifest["files"].items())


@pytest.mark.parametrize("value", [1, "true", None])
def test_full_reference_assembly_choice_requires_boolean_before_io(tmp_path, value):
    with pytest.raises(ValueError, match="explicitly enabled"):
        pipeline.prepare_exercise_scene("missing", "missing", "missing", tmp_path / "out", full_reference=value)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("command", ["shared-exercise-build", "shared-world-reconstruct"])
def test_full_reference_cli_passes_choice_to_existing_pipeline(tmp_path, monkeypatch, capsys, command):
    import sys
    from weft import cli, shared_world_pipeline

    called = {}

    def prepare(*args, **kwargs):
        called.update(kwargs)
        return {"status": "ready"}

    argv = ["weft", command, "--run", "body", "--evidence-run", "evidence", "--out", str(tmp_path / "new"),
            "--full-reference"]
    if command == "shared-exercise-build":
        argv += ["--config", "config"]
        monkeypatch.setattr(pipeline, "prepare_exercise_scene", prepare)
    else:
        argv += ["--proposal-run", "proposal", "--target-object", "weight"]
        monkeypatch.setattr(shared_world_pipeline, "construct_source_equipment_run", prepare)
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert called["full_reference"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "ready"


def test_one_command_assembles_canonical_evidence_into_a_human_only_lab(tmp_path):
    body, equipment, config_path, original, config = packages(tmp_path)
    inputs = {path: path.read_bytes() for directory in (body, equipment)
              for path in directory.rglob("*") if path.is_file()}
    out = tmp_path / "assembled"
    result = pipeline.prepare_exercise_scene(body, equipment, config_path, out, root_assistance=True)
    assert result["status"] == "ready"
    assert result["lab_status"] == "ready"
    assert result["seal_scope"] == "complete_immutable_outputs"
    assert result["seconds"] >= result["sealing_seconds"] > 0
    assert result["launch_arguments"] == ["lab", "--resume", str(out / "lab")]
    assert [(row["stage"], row["status"]) for row in result["stages"]] == [
        ("equipment_priors", "running"), ("equipment_priors", "proposal_ready"),
        ("shared_world", "running"), ("shared_world", "ready"),
        ("interactive_lab", "running"), ("interactive_lab", "ready")]
    scene = json.loads((out / "candidate/scene.json").read_text())
    assert scene["human_spec"] == original["human_spec"]
    assert scene["task"]["object_goal"]["target_object_id"] == config["target_object_id"]
    assert {obj["id"] for obj in scene["objects"]} == {"foam-pad", "training-weight"}
    manifest = json.loads((out / "manifest.json").read_text())
    assert "lab/run/scene.json" in manifest["files"]
    assert not any(name.startswith("lab/") and not name.startswith("lab/run/") for name in manifest["files"])
    assert all(digest(out / name) == sha for name, sha in manifest["files"].items())
    lab = Lab(out / "lab")
    try:
        status = lab.summary()
        assert status["control_profile"] == "shared_exercise"
        assert status["learned_policy_available"] is False
        assert len(status["action_contract"]["names"]) == 22
        step = lab.control("/api/step", {"controller": "coast"})["snapshot"]
        assert step["engine_state"]["steps"] == 1
        assert not any(step["engine_state"]["normalized_action"][22:])
    finally:
        lab.close()
    assert all(path.read_bytes() == raw for path, raw in inputs.items())
    assert all(digest(out / name) == sha for name, sha in manifest["files"].items())


def test_unsupported_configuration_is_inspectable_without_a_world_or_lab(tmp_path, monkeypatch):
    body, equipment, config_path, _, config = packages(tmp_path)
    config["camera"]["mode"] = "unknown_calibration"
    write_json(config_path, config)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported evidence must not advance to construction")

    monkeypatch.setattr(pipeline, "construct_source_equipment_run", forbidden)
    out = tmp_path / "unsupported"
    result = pipeline.prepare_exercise_scene(body, equipment, config_path, out)
    assert result["status"] == "unsupported_input" and result["lab_directory"] is None
    assert (out / "proposal/input-config.json").read_bytes() == config_path.read_bytes()
    assert not (out / "candidate").exists() and not (out / "lab").exists()
    assert json.loads((out / "manifest.json").read_text())["status"] == "unsupported_input"


def test_failure_before_lab_keeps_completed_proposal_without_a_ready_lab(tmp_path, monkeypatch):
    body, equipment, config_path, _, _ = packages(tmp_path)

    def rejected(*args, **kwargs):
        raise ValueError("Changed construction input")

    monkeypatch.setattr(pipeline, "construct_source_equipment_run", rejected)
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="Changed construction input"):
        pipeline.prepare_exercise_scene(body, equipment, config_path, out)
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "failed" and result["lab_directory"] is None
    assert (out / "proposal/geometry-proposal.json").is_file()
    assert json.loads((out / "manifest.json").read_text())["status"] == "failed"
    assert not (out / "lab").exists()


def test_late_progress_failure_preserves_the_legitimately_ready_inner_lab(tmp_path):
    body, equipment, config_path, _, _ = packages(tmp_path)
    out = tmp_path / "late-failure"

    def progress(row):
        if row["stage"] == "interactive_lab" and row["status"] == "ready":
            raise RuntimeError("Progress destination closed after lab completion")

    with pytest.raises(RuntimeError, match="Progress destination closed"):
        pipeline.prepare_exercise_scene(body, equipment, config_path, out, progress=progress)
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "failed" and result["lab_status"] == "ready"
    assert result["lab_directory"] == str(out / "lab")
    assert result["launch_arguments"] == ["lab", "--resume", str(out / "lab")]
    assert json.loads((out / "lab/status.json").read_text())["status"] == "ready"
    lab = Lab(out / "lab")
    try:
        assert lab.summary()["control_profile"] == "shared_exercise"
        assert lab.control("/api/step", {"controller": "coast"})["snapshot"]["engine_state"]["steps"] == 1
    finally:
        lab.close()


def _fake_stages(monkeypatch):
    from weft import shared_world_lab

    calls = []

    def body_seed(source, out, **kwargs):
        calls.append(("body_seed", kwargs))
        out.mkdir()
        write_json(out / "manifest.json", {"files": {}})
        return out

    def proposal(base, evidence, config, out, **kwargs):
        calls.append(("equipment_priors", kwargs))
        out.mkdir()
        (out / "proposal.json").write_bytes(b"{}")
        return {"status": "proposal_ready", "manifest_sha256": "1" * 64,
                "construction_ready_arguments": {"target_object_id": "explicit-target", "frame_translation": [.2, 0.]}}

    def world(base, proposal, evidence, out, **kwargs):
        calls.append(("shared_world", kwargs))
        out.mkdir()
        (out / "scene.json").write_bytes(b"{}")
        return {"manifest_sha256": "2" * 64}

    def lab(run, out, **kwargs):
        calls.append(("interactive_lab", kwargs))
        (out / "run").mkdir(parents=True)
        (out / "run/scene.json").write_bytes(b"{}")
        write_json(out / "status.json", {"status": "ready"})
        return {"manifest_sha256": "3" * 64, "scene_sha256": "4" * 64,
                "task_sha256": "5" * 64, "lab_directory": str(out)}

    monkeypatch.setattr(pipeline, "prepare_equipment_proposal", proposal)
    monkeypatch.setattr(pipeline, "construct_source_equipment_run", world)
    monkeypatch.setattr(pipeline, "prepare_shared_exercise_lab", lab)
    monkeypatch.setattr(shared_world_lab, "prepare_shared_world_seed", body_seed)
    return calls


@pytest.mark.parametrize("stage", list(pipeline.STAGE_LIMITS))
@pytest.mark.parametrize("bound", ["files", "bytes"])
def test_reserve_aggregate_capacity_before_starting_each_stage(tmp_path, monkeypatch, stage, bound):
    calls = _fake_stages(monkeypatch)
    limits = {name: (1, 2) for name in pipeline.STAGE_LIMITS}
    limits[stage] = (pipeline.MAX_FILES, 2) if bound == "files" else (1, pipeline.MAX_BYTES)
    monkeypatch.setattr(pipeline, "STAGE_LIMITS", limits)
    out = tmp_path / "bounded"
    with pytest.raises(ValueError, match="cannot reserve the complete " + stage):
        pipeline.prepare_exercise_scene(None if stage == "body_seed" else tmp_path / "base",
            tmp_path / "evidence", tmp_path / "config", out,
            human_run=tmp_path / "human" if stage == "body_seed" else None)
    assert stage not in [name for name, _ in calls]
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "failed" and result["lab_status"] == "not_ready"


@pytest.mark.parametrize("base,human,scale,message", [
    (None, None, None, "exactly one"), ("base", "human", None, "exactly one"),
    ("base", None, 1., "only to a mobile"),
    (None, "human", True, "finite prior"), (None, "human", float("nan"), "finite prior"),
    (None, "human", .49, "finite prior"), (None, "human", 2.01, "finite prior"),
])
def test_mobile_source_choices_reject_before_output(tmp_path, base, human, scale, message):
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match=message):
        pipeline.prepare_exercise_scene(base, "evidence", "config", out, human_run=human, body_scale=scale)
    assert not out.exists()


def test_mobile_source_preparation_keeps_one_bounded_assembly(tmp_path, monkeypatch):
    calls = _fake_stages(monkeypatch)
    out = tmp_path / "mobile"
    result = pipeline.prepare_exercise_scene(None, tmp_path / "evidence", tmp_path / "config", out,
        human_run=tmp_path / "human", body_scale=1.2, full_reference=True)
    assert result["status"] == "ready"
    assert [name for name, _ in calls] == ["body_seed", "equipment_priors", "shared_world", "interactive_lab"]
    assert dict(calls)["body_seed"]["scale"] == 1.2
    assert dict(calls)["body_seed"]["deadline"] == dict(calls)["shared_world"]["deadline"]
    settings = json.loads((out / "settings.json").read_bytes())
    assert settings["base_run"] is None and settings["human_run"] == str(tmp_path / "human")
    assert settings["body_scale"] == 1.2 and settings["inference_calls"] == settings["api_calls"] == 0
    assert sum(row["stage"] == "interactive_lab" and row["status"] == "ready" for row in result["stages"]) == 1
    assert "body_seed/manifest.json" in json.loads((out / "manifest.json").read_bytes())["files"]


def test_mobile_source_build_preserves_unknown_slots_and_exposes_only_selected_equipment(tmp_path):
    from test_shared_world_seed import mobile_source_packages

    human, equipment, config, original, _ = mobile_source_packages(tmp_path, as_lab=True)
    before = {path: path.read_bytes() for directory in (human, equipment)
              for path in directory.rglob("*") if path.is_file()}
    out = tmp_path / "mobile-full-reference"
    result = pipeline.prepare_exercise_scene(None, equipment, config, out,
        human_run=human, full_reference=True, root_assistance=True)
    assert result["status"] == "ready" and result["lab_status"] == "ready"
    assert sorted(path.name for path in out.iterdir() if path.is_dir()) == ["body_seed", "candidate", "lab", "proposal"]
    scene = json.loads((out / "candidate/scene.json").read_bytes())
    assert scene["profile"] == "shared-world-v2" and scene["human_spec"] == original
    assert {row["id"] for row in scene["objects"]} == {"foam-pad", "training-weight"}
    assert (out / "body_seed/human.json").read_bytes() == (human / "run/human.json").read_bytes()
    seed = json.loads((out / "body_seed/construction-seed.json").read_bytes())
    assert seed["simulation_ready"] is False and seed["physics_steps"] == seed["physics_models"] == 0
    assert not (out / "candidate/construction-seed.json").exists()
    assert not (out / "lab/run/construction-seed.json").exists()
    binding = scene["source_binding"]
    assert binding["total_source_frames"] == 63
    body_binding = binding["entity_bindings"][0]
    assert body_binding["observed_slots"] == 60 and body_binding["missing_slots"] == 3
    observation_path = binding["evidence_documents"][body_binding["observation_document_sha256"]]
    observations = json.loads((out / "candidate" / observation_path).read_bytes())
    assert [row["frame"] for row in observations["observations"] if row["visibility"] == "not_sampled"] == [0, 1, 31]
    lab = Lab(out / "lab")
    try:
        snapshot = lab.control("/api/step", {"controller": "reference"})["snapshot"]
        assert snapshot["engine_state"]["steps"] == 1
        assert not any(snapshot["engine_state"]["normalized_action"][22:])
    finally:
        lab.close()
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_mobile_source_cli_passes_explicit_scale_without_a_shared_base(tmp_path, monkeypatch, capsys):
    import sys
    from weft import cli

    called = {}

    def prepare(base, evidence, config, out, **kwargs):
        called.update(base=base, **kwargs)
        kwargs["progress"]({"stage": "body_seed", "status": "running"})
        return {"status": "ready"}

    monkeypatch.setattr(pipeline, "prepare_exercise_scene", prepare)
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-build", "--human-run", "mobile",
        "--body-scale", "1.2", "--evidence-run", "evidence", "--config", "config", "--out", str(tmp_path / "out")])
    cli.main()
    assert called["base"] is None and called["human_run"] == "mobile" and called["body_scale"] == 1.2
    output = capsys.readouterr()
    assert "Prepare source body: running" in output.err
    assert json.loads(output.out)["status"] == "ready"


def test_deadline_after_lab_keeps_ready_path_and_only_seals_small_failure_receipt(tmp_path, monkeypatch):
    calls = _fake_stages(monkeypatch)
    clock = [10.]
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def progress(row):
        if row["stage"] == "interactive_lab" and row["status"] == "ready":
            clock[0] = 16.

    out = tmp_path / "expired"
    with pytest.raises(TimeoutError):
        pipeline.prepare_exercise_scene(tmp_path / "base", tmp_path / "evidence", tmp_path / "config", out,
                                        root_assistance=True, max_seconds=5, progress=progress)
    assert dict(calls)["shared_world"] == {"target_object_id": "explicit-target", "source_mappings": (),
                                            "frame_translation": [.2, 0.], "full_reference": False, "deadline": 15.}
    assert dict(calls)["interactive_lab"]["root_assistance"] is True
    result = json.loads((out / "result.json").read_text())
    assert result["lab_status"] == "ready" and result["lab_directory"] == str(out / "lab")
    assert result["seal_scope"] == "failure_receipt_only"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["files"] == {"result.json": digest(out / "result.json")}
    assert json.loads((out / "lab/status.json").read_text())["status"] == "ready"


def test_final_seal_failure_is_not_retried_or_allowed_to_mask_the_failure(tmp_path, monkeypatch):
    _fake_stages(monkeypatch)
    original = pipeline.read_package_file
    attempts = []

    def damaged(root, name, **kwargs):
        attempts.append(name)
        if name == "candidate/scene.json":
            raise ValueError("Final hash read rejected changed file")
        return original(root, name, **kwargs)

    monkeypatch.setattr(pipeline, "read_package_file", damaged)
    out = tmp_path / "hash-failed"
    with pytest.raises(ValueError, match="Final hash read rejected changed file"):
        pipeline.prepare_exercise_scene(tmp_path / "base", tmp_path / "evidence", tmp_path / "config", out)
    assert attempts.count("candidate/scene.json") == 1
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "failed" and result["lab_status"] == "ready"
    assert result["seal_scope"] == "failure_receipt_only"
    assert result["error"]["message"] == "Final hash read rejected changed file"


def test_oversized_partial_stage_preserves_original_error_and_independent_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_BYTES", 8192)
    monkeypatch.setattr(pipeline, "METADATA_RESERVE_BYTES", 1024)
    monkeypatch.setattr(pipeline, "STAGE_LIMITS", {"equipment_priors": (1, 2)})

    def failed(base, evidence, config, out, **kwargs):
        out.mkdir()
        (out / "partial.bin").write_bytes(b"x" * 8193)
        raise RuntimeError("Primary stage failure")

    monkeypatch.setattr(pipeline, "prepare_equipment_proposal", failed)
    out = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="Primary stage failure"):
        pipeline.prepare_exercise_scene(tmp_path / "base", tmp_path / "evidence", tmp_path / "config", out)
    result = json.loads((out / "result.json").read_text())
    assert result["error"]["message"] == "Primary stage failure"
    assert "output file/byte bound" in result["sealing_error"]["message"]
    assert result["seal_scope"] == "failure_receipt_only" and result["unsealed_stage_outputs_preserved"]
    assert (out / "proposal/partial.bin").stat().st_size == 8193
    assert json.loads((out / "manifest.json").read_text())["files"] == {"result.json": digest(out / "result.json")}


@pytest.mark.parametrize("expires", [False, True])
def test_final_hashing_is_deadline_checked_and_included_in_elapsed_time(tmp_path, monkeypatch, expires):
    _fake_stages(monkeypatch)
    clock = [0.]
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original = pipeline.read_package_file

    def delayed(*args, **kwargs):
        raw = original(*args, **kwargs)
        clock[0] += 1.
        return raw

    monkeypatch.setattr(pipeline, "read_package_file", delayed)
    out = tmp_path / "timed"
    if expires:
        with pytest.raises(TimeoutError):
            pipeline.prepare_exercise_scene(tmp_path / "base", tmp_path / "evidence", tmp_path / "config", out,
                                            max_seconds=2)
        assert json.loads((out / "result.json").read_text())["status"] == "failed"
    else:
        result = pipeline.prepare_exercise_scene(tmp_path / "base", tmp_path / "evidence", tmp_path / "config", out,
                                                 max_seconds=20)
        assert result["seconds"] == result["sealing_seconds"] == clock[0] > 0
        assert json.loads((out / "result.json").read_text())["seconds"] == clock[0]
