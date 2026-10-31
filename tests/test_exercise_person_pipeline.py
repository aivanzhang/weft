"""Person-only orchestration checks use fake producers, never model or physics work."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import pytest

from test_exercise_build_pipeline import fake_learning
from test_exercise_build_report import fixture as report_fixture, put, seal as report_seal
from test_exercise_scene_inputs import inputs, sha, write
from test_exercise_scene_native_recovery import child, full_inputs, stages as recovery_stages
from weft import cli, exercise_build_report, exercise_scene_inputs as admission, exercise_scene_pipeline as pipeline
from weft import shared_world_human, shared_world_lab
from weft.person_inventory import make_person_inventory


def test_person_preflight_pins_original_and_declaration_without_equipment(tmp_path, monkeypatch):
    video, equipment, config, model = inputs(tmp_path, monkeypatch)
    ready = admission.preflight_human_scene_inputs(video, pose_model=model)
    receipt = ready.receipt()
    assert receipt["profile"] == "exercise-person-inputs-v1"
    assert receipt["scene_scope"] == "human_only"
    assert receipt["inventory"] == make_person_inventory(sha(video.read_bytes()))
    assert receipt["pose_options"] == {"pose_mode": "video", "pose_identity": "raw"}
    assert ready.evidence_run is ready.config_path is None and ready.source_mappings == ()
    assert all(value == 0 for value in receipt["work"].values())
    assert not any(Path(pin["path"]).is_relative_to(equipment) or Path(pin["path"]) == config for pin in receipt["input_pins"])
    receipt["inventory"]["absence_verified"] = True
    assert ready.receipt()["inventory"]["absence_verified"] is False
    admission.recheck_exercise_scene_inputs(ready, deadline=time.monotonic() + 10)


@pytest.mark.parametrize("kind", ["source", "model", "empty", "extension", "symlink", "recipe"])
def test_person_input_mistakes_never_reach_body_preparation(tmp_path, monkeypatch, kind):
    video, _, _, model = inputs(tmp_path, monkeypatch)
    if kind in {"source", "model"}:
        ready = admission.preflight_human_scene_inputs(video, pose_model=model)
        (video if kind == "source" else model).write_bytes(b"changed")
        with pytest.raises(ValueError):
            admission.prepare_exercise_scene_body(ready, tmp_path / "body")
    else:
        options = {}
        if kind == "empty":
            video.write_bytes(b"")
        elif kind == "extension":
            video = tmp_path / "source.txt"
            video.write_bytes(b"wrong container")
        elif kind == "symlink":
            alias = tmp_path / "alias.webm"
            alias.symlink_to(video)
            video = alias
        else:
            options = {"pose_recovery": "selective-crop", "pose_mode": "video"}
        with pytest.raises(ValueError):
            admission.preflight_human_scene_inputs(video, pose_model=model, **options)
    assert not (tmp_path / "body").exists()


def fake_world_and_lab(monkeypatch, calls, *, fail_lab=False):
    def forbidden(*args, **options):
        pytest.fail("Person-only scope cannot create scaffolding or equipment")

    monkeypatch.setattr(shared_world_lab, "prepare_shared_world_seed", forbidden)
    monkeypatch.setattr(pipeline, "prepare_equipment_proposal", forbidden)
    monkeypatch.setattr(pipeline, "construct_source_equipment_run", forbidden)

    def construct(source, inventory, out, **options):
        calls.append(("shared_world", {"source": source, "inventory": deepcopy(inventory), **options}))
        child(out, "shared-world-v3", "ready", inventory["source_sha256"])
        return {"status": "ready", "profile": "shared-world-v3", "manifest_sha256": sha((out / "manifest.json").read_bytes()),
                "readiness": "candidate_constructed_not_lab_admitted", "reference_interval": {"complete": True}}

    def lab(source, out, **options):
        calls.append(("interactive_lab", {"source": source, **options}))
        if fail_lab:
            raise ValueError("Unchanged initial geometry gate rejects fixture")
        (out / "run").mkdir(parents=True)
        write(out / "run/scene.json", {"profile": "shared-world-v3", "objects": []})
        write(out / "status.json", {"status": "ready"})
        return {"lab_directory": str(out), "scene_sha256": "4" * 64, "task_sha256": "5" * 64, "manifest_sha256": "6" * 64}

    monkeypatch.setattr(shared_world_human, "construct_human_only_run", construct)
    monkeypatch.setattr(pipeline, "prepare_shared_exercise_lab", lab)


def setup_build(tmp_path, monkeypatch, mode, *, train=False, fail_lab=False):
    calls, options = [], {"scene_scope": "human_only", "root_assistance": True}
    fake_world_and_lab(monkeypatch, calls, fail_lab=fail_lab)
    if mode == "saved":
        source = tmp_path / "source-body"
        source.mkdir()
        options["human_run"] = source
        monkeypatch.setattr(shared_world_human, "read_person_body_source_identity",
                            lambda *args, **kwargs: {"source_sha256": "a" * 64})
    else:
        paths = full_inputs(tmp_path, monkeypatch)[0] if mode == "crop" else inputs(tmp_path, monkeypatch)
        options.update(video=paths[0], pose_model=paths[3])
        if mode == "crop":
            options["pose_recovery"] = "selective-crop"
            ready = admission.preflight_human_scene_inputs(paths[0], pose_model=paths[3], pose_recovery="selective-crop")
            recovery_stages(monkeypatch, ready)
        else:
            def body(video, out, **kwargs):
                calls.append(("body_source", kwargs))
                return child(out, "exercise-body-source-v1", "construction_only", sha(video.read_bytes()))
            monkeypatch.setattr(admission.exercise_body_source, "prepare_exercise_body_source", body)
    if train:
        options["train_settings"] = {"evaluation_seed": 51489, "training_seed": 7216, "generations": 1, "population": 4, "trials": 1}
        fake_learning(monkeypatch, calls)
    return calls, options


@pytest.mark.parametrize("mode", ["saved", "raw", "crop"])
@pytest.mark.parametrize("train", [False, True])
def test_declared_person_uses_one_direct_candidate_and_one_final_lab(tmp_path, monkeypatch, mode, train):
    calls, options = setup_build(tmp_path, monkeypatch, mode, train=train)
    out = tmp_path / "build"
    result = pipeline.prepare_exercise_scene(None, None, None, out, **options)
    assert result["status"] == result["lab_status"] == "ready"
    expected_lab = out / ("learning_workflow/lab" if train else "lab")
    assert result["lab_directory"] == str(expected_lab)
    assert sum(name in {"interactive_lab", "learning_workflow"} for name, _ in calls) == 1
    assert not (out / "body_seed").exists() and not (out / "proposal").exists()
    settings = json.loads((out / "settings.json").read_bytes())
    inventory = json.loads((out / "person-inventory.json").read_bytes())
    assert settings["full_reference"] is True and settings["scene_scope"] == "human_only"
    assert settings["evidence_run"] is settings["config"] is None
    assert settings["inventory"] == dict(calls)["shared_world"]["inventory"] == inventory
    assert inventory["absence_verified"] is False and inventory["support"] == "ground_only"
    assert {"equipment_priors", "body_seed"}.isdisjoint(settings["stage_output_reservations"])
    assert settings["stage_output_reservations"]["shared_world"] == {"files": 512, "bytes": 128 * 1024 * 1024}
    manifest = json.loads((out / "manifest.json").read_bytes())
    assert all(sha((out / name).read_bytes()) == digest for name, digest in manifest["files"].items())
    assert not any(name.endswith("status.json") for name in manifest["files"])
    if mode == "crop":
        assert (out / "pose_proposals/manifest.json").is_file() and (out / "recovered_poses/manifest.json").is_file()
        assert json.loads((out / "input-preflight.json").read_bytes())["pose_options"] == {"pose_mode": "image", "pose_identity": "raw"}


@pytest.mark.parametrize("contradiction", ["evidence", "config", "mapping", "base", "unknown_scope"])
def test_conflicting_scope_rejects_before_io_or_output(tmp_path, contradiction):
    options = {"video": "source.webm", "pose_model": "pose.task", "scene_scope": "human_only"}
    base = evidence = config = None
    if contradiction == "evidence": evidence = "equipment"
    if contradiction == "config": config = "equipment.json"
    if contradiction == "mapping": options["source_mappings"] = ["mapping"]
    if contradiction == "base": base, options = "base", {"scene_scope": "human_only"}
    if contradiction == "unknown_scope": options["scene_scope"] = "automatic"
    with pytest.raises(ValueError, match="scope"):
        pipeline.prepare_exercise_scene(base, evidence, config, tmp_path / "out", **options)
    assert not (tmp_path / "out").exists()


def test_construction_ready_does_not_bypass_lab_geometry_gate(tmp_path, monkeypatch):
    calls, options = setup_build(tmp_path, monkeypatch, "saved", fail_lab=True)
    out = tmp_path / "build"
    with pytest.raises(ValueError, match="geometry gate"):
        pipeline.prepare_exercise_scene(None, None, None, out, **options)
    result = json.loads((out / "result.json").read_bytes())
    assert result["status"] == "failed" and result["lab_status"] == "not_ready"
    assert (out / "candidate/manifest.json").is_file()
    assert [name for name, _ in calls] == ["shared_world", "interactive_lab"]


@pytest.mark.parametrize("mutation", ["inventory", "candidate_manifest", "candidate_member", "body_member"])
def test_completed_person_evidence_cannot_be_replaced_by_progress_observer(tmp_path, monkeypatch, mutation):
    _, options = setup_build(tmp_path, monkeypatch, "raw")
    out = tmp_path / "build"
    def progress(row):
        if row["stage"] == "shared_world" and row["status"] == "ready":
            path = {"inventory": "person-inventory.json", "candidate_manifest": "candidate/manifest.json",
                    "candidate_member": "candidate/data.json", "body_member": "body_source/data.json"}[mutation]
            (out / path).write_bytes(b"{}\n")
    with pytest.raises(ValueError):
        pipeline.prepare_exercise_scene(None, None, None, out, progress=progress, **options)
    result = json.loads((out / "result.json").read_bytes())
    assert result["status"] == "failed" and result["seal_scope"] == "failure_receipt_only"


@pytest.mark.parametrize("mode", ["video", "human-run"])
def test_cli_person_declaration_needs_no_equipment_files(tmp_path, monkeypatch, capsys, mode):
    seen = {}
    monkeypatch.setattr(pipeline, "prepare_exercise_scene", lambda *args, **kwargs: seen.update(args=args, options=kwargs) or {"status": "ready"})
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-build", "--person-only", "--" + mode, "source", "--out", str(tmp_path / "out")])
    cli.main()
    assert seen["args"][1:3] == (None, None)
    assert seen["options"]["scene_scope"] == "human_only"
    assert json.loads(capsys.readouterr().out)["status"] == "ready"


@pytest.mark.parametrize("extra", [["--config", "config"], ["--evidence-run", "equipment"], ["--source-mapping", "mapping"]])
def test_cli_conflicts_do_not_call_builder(tmp_path, monkeypatch, capsys, extra):
    monkeypatch.setattr(pipeline, "prepare_exercise_scene", lambda *args, **kwargs: pytest.fail("Builder called"))
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-build", "--person-only", "--video", "source", "--out", str(tmp_path / "out"), *extra])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code != 0
    assert "without" in capsys.readouterr().err


@pytest.mark.parametrize("scene_state", ["constructed", "not_constructed", "contradictory"])
def test_offline_report_distinguishes_declared_absence_from_passed_equipment_gates(tmp_path, scene_state):
    template = tmp_path / "template"
    report_fixture(template, ready=True)
    root = tmp_path / "source"
    root.mkdir()
    for path in template.rglob("*"):
        name = path.relative_to(template).as_posix()
        if path.is_file() and not name.startswith(("proposal/", "body_seed/")):
            put(root, name, path.read_bytes())
    settings = json.loads((root / "settings.json").read_bytes())
    source_sha = json.loads((root / "body_source/source.json").read_bytes())["sha256"]
    inventory = make_person_inventory(source_sha)
    settings.update(scene_scope="human_only", inventory=inventory)
    put(root, "settings.json", settings)
    put(root, "person-inventory.json", inventory)
    put(root, "candidate/inventory.json", inventory)
    if scene_state != "not_constructed":
        put(root, "candidate/scene.json", {"profile": "shared-world-v3", "objects": [] if scene_state == "constructed" else [{"id": "hidden-weight"}],
            "controlled_id": "human", "target": None, "inventory": inventory,
            "task": {"type": "human_motion_v1", "object_goal": None}, "source_binding": {"source_sha256": source_sha}})
    result = json.loads((root / "result.json").read_bytes())
    result["stages"] = [row for row in result["stages"] if row["stage"] not in {"body_seed", "equipment_priors"}]
    put(root, "result.json", result)
    put(root, "stages.json", {"stages": result["stages"]})
    report_seal(root)
    out = tmp_path / "report"
    if scene_state == "contradictory":
        with pytest.raises(ValueError, match="contradicts.*inventory"):
            exercise_build_report.prepare_exercise_build_report(root, out)
        return
    exercise_build_report.prepare_exercise_build_report(root, out)
    report = json.loads((out / "report.json.txt").read_bytes())
    assert report["equipment"]["status"] == ("not_applicable" if scene_state == "constructed" else "declared_scope_only")
    assert report["equipment"]["inventory"] == inventory
    states = {row["stage"]: row["state"] for row in report["stages"]}
    assert states["body_seed"] == states["equipment_priors"] == "not_requested"
    html = (out / "index.html").read_text()
    assert ("not applicable, not passed" if scene_state == "constructed" else "no constructed scene is available") in html
    assert "not detector-verified" in html
    assert any(row["source_name"] == "candidate/inventory.json" for row in report["artifacts"])


def test_person_report_rejects_contradictory_equipment_instead_of_hiding_it(tmp_path):
    root = tmp_path / "source"
    report_fixture(root, ready=True)
    settings = json.loads((root / "settings.json").read_bytes())
    source_sha = json.loads((root / "body_source/source.json").read_bytes())["sha256"]
    inventory = make_person_inventory(source_sha)
    settings.update(scene_scope="human_only", inventory=inventory)
    put(root, "settings.json", settings)
    put(root, "person-inventory.json", inventory)
    report_seal(root)
    with pytest.raises(ValueError, match="contradictory equipment"):
        exercise_build_report.prepare_exercise_build_report(root, tmp_path / "report")


def test_person_inspector_keeps_full_interval_and_does_not_claim_native_sync():
    from weft.lab_inspection import _timeline

    human = {"source_sha256": "a" * 64, "reference": [{"time_s": 4 / 30}, {"time_s": 62 / 30}],
        "reference_evidence": [{"time_s": frame / 30, "evidence_id": "a" * 64 + f":frame:{frame}"} for frame in (4, 62)]}
    scene = {"profile": "shared-world-v3", "dt": 1 / 30, "task": {"start_offset_s": 0., "horizon_steps": 58}}
    timeline = _timeline(scene, human, 63, {})
    assert timeline["task_steps"] == 58
    assert timeline["task_window_s"] == pytest.approx([4 / 30, 62 / 30])
    assert timeline["source_slots"]["leading_without_reference"] == 4
    assert timeline["native_source_clock"] == "not_read_here" and timeline["media_sync_verified"] is False
