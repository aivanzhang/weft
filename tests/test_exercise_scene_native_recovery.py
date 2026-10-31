"""Opt-in raw-build composition; detector and body producers are never executed."""

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from test_exercise_scene_inputs import inputs, preflight, sha, write
from weft import exercise_scene_inputs as module
from weft import pose_crop_pipeline, pose_source_recovery


def full_inputs(tmp_path, monkeypatch):
    paths = inputs(tmp_path, monkeypatch)
    pinned = {"sha256": sha(paths[3].read_bytes()), "bytes": paths[3].stat().st_size}
    # Analytical model bytes replace only the pinned test identity; no detector is invoked.
    monkeypatch.setitem(module.exercise_vision.MODEL_VARIANTS, "full", pinned)
    model = {**pinned, "variant": "full", "model_version": "float16/1", "delegate": "CPU", "network_inference": False}
    monkeypatch.setattr(module.exercise_vision, "model_provenance", lambda _path: deepcopy(model))
    return paths, model


def outputs(tmp_path):
    return {"out": tmp_path / "body_source", "pose_proposals_out": tmp_path / "pose_proposals",
            "recovered_pose_out": tmp_path / "recovered_poses"}


def child(root, profile, status, source_sha):
    root.mkdir()
    write(root / "data.json", {"source_sha256": source_sha, "scope": "Analytical orchestration fixture"})
    write(root / "manifest.json", {"schema_version": 1, "profile": profile, "status": status,
        "simulation_ready": False, "files": {"data.json": sha((root / "data.json").read_bytes())}})
    return root


def stages(monkeypatch, ready, *, fail_at=None, clock=None):
    calls = []

    def finish(name, source, target, profile, status, options):
        calls.append((name, source, target, options))
        child(target, profile, status, ready.source_sha256)
        if fail_at == name:
            write(target / "failure.json", {"status": "failed", "reason": "Unchanged stage gate rejected fixture"})
            raise ValueError("Unchanged stage gate rejected fixture")
        if options.get("progress"):
            detail = {"message": "Analytical progress", "counts": {"sampled_frames": 32}}
            options["progress"]("tracking", detail)
            assert detail["counts"] == {"sampled_frames": 32}
        if clock is not None:
            clock[0] += {"proposals": 130, "recovery": 20, "body": 10}[name]
        return target

    def proposals(video, target, **options):
        return finish("proposals", video, target, pose_crop_pipeline.PROFILE, "source_evidence_only", options)

    def recovery(source, target, **options):
        return finish("recovery", source, target, pose_source_recovery.PROFILE, "source_evidence_only", options)

    def body(source, target, **options):
        return finish("body", source, target, module.exercise_body_source.PROFILE, "construction_only", options)

    monkeypatch.setattr(pose_crop_pipeline, "prepare_pose_crop_proposals", proposals)
    monkeypatch.setattr(pose_source_recovery, "prepare_pose_source_recovery", recovery)
    monkeypatch.setattr(module.exercise_body_source, "prepare_exercise_body_source_from_evidence", body)
    return calls


def test_opt_in_resolves_only_full_image_raw_and_pins_optional_sources(tmp_path, monkeypatch):
    paths, _ = full_inputs(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    default = preflight(paths)
    ready = preflight(paths, pose_recovery="selective-crop")
    receipt = ready.receipt()
    assert default.pose_recovery is None
    assert default.receipt()["pose_options"] == {"pose_mode": "video", "pose_identity": "raw"}
    assert set(default.receipt()["implementation_sha256"]) == set(module.IMPLEMENTATION_FILES)
    assert "pose_crop_pipeline.py" not in default.receipt()["implementation_sha256"]
    assert receipt["pose_options"] == {"pose_mode": "image", "pose_identity": "raw"}
    assert receipt["pose_recovery"] == ready.pose_recovery == "selective-crop"
    assert receipt["pose_model"]["variant"] == "full"
    assert receipt["recovery_contract"]["stage_seconds"] == {"pose_proposals": 180, "pose_recovery": 30, "body_source": 30}
    assert receipt["recovery_contract"]["retries"] == 0
    expected = set(module.IMPLEMENTATION_FILES) | set(pose_crop_pipeline.IMPLEMENTATION_FILES) | set(pose_source_recovery.IMPLEMENTATION_FILES)
    assert set(receipt["implementation_sha256"]) == expected
    for name, checksum in receipt["implementation_sha256"].items():
        assert checksum == sha(Path(module.__file__).with_name(name).read_bytes())
    assert receipt["selected_equipment"] == default.receipt()["selected_equipment"]
    assert receipt["source_sha256"] == default.receipt()["source_sha256"]
    assert all(value == 0 for value in receipt["work"].values())
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("options", [{"pose_mode": "video"}, {"pose_identity": "temporal"},
                                      {"pose_recovery": "automatic"}, {"pose_recovery": True}])
def test_conflicting_recovery_options_reject_before_input_reads(tmp_path, monkeypatch, options):
    paths, _ = full_inputs(tmp_path, monkeypatch)
    options = {"pose_recovery": "selective-crop", **options}
    monkeypatch.setattr(module._Reader, "read", lambda *a, **k: pytest.fail("Conflicting recipe read an input"))
    with pytest.raises(ValueError, match="Pose recovery|Selective-crop"):
        preflight(paths, **options)


def test_nonfull_model_cannot_enter_optional_native_pipeline(tmp_path, monkeypatch):
    paths = inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="exact pinned official Full"):
        preflight(paths, pose_recovery="selective-crop")


@pytest.mark.parametrize("field,value", [("variant", "lite"), ("model_version", "future"),
                                        ("delegate", "GPU"), ("network_inference", True)])
def test_full_model_metadata_must_match_fixed_recipe(tmp_path, monkeypatch, field, value):
    paths, model = full_inputs(tmp_path, monkeypatch)
    model[field] = value
    with pytest.raises(ValueError, match="exact pinned official Full"):
        preflight(paths, pose_recovery="selective-crop")


def test_one_proposal_recovery_body_chain_keeps_exact_inputs_and_no_extra_labs(tmp_path, monkeypatch):
    paths, _ = full_inputs(tmp_path, monkeypatch)
    ready = preflight(paths, pose_recovery="selective-crop", pose_mode="image", pose_identity="raw",
                      source_note="Fixture source note must be retained")
    calls, updates = stages(monkeypatch, ready), []

    def progress(stage, detail):
        updates.append((stage, deepcopy(detail)))
        if "counts" in detail:
            detail["counts"]["sampled_frames"] = -1

    target = outputs(tmp_path)
    assert module.prepare_exercise_scene_body(ready, **target, max_seconds=240, progress=progress) == target["out"]
    assert [row[0] for row in calls] == ["proposals", "recovery", "body"]
    proposal, recovered, body = calls
    assert proposal[1:3] == (paths[0], target["pose_proposals_out"])
    assert proposal[3]["pose_model"] == paths[3] and proposal[3]["source_note"] == ready.source_note
    assert recovered[1:3] == (target["pose_proposals_out"], target["recovered_pose_out"])
    assert recovered[3]["source_video"] == paths[0]
    assert body[1:3] == (target["recovered_pose_out"], target["out"])
    assert all(1 <= row[3]["max_seconds"] <= maximum for row, maximum in zip(calls, (180, 30, 30)))
    assert [(stage, detail["status"]) for stage, detail in updates if "status" in detail] == [
        (stage, status) for stage in ("pose_proposals", "pose_recovery", "body_source") for status in ("running", "ready")]
    assert all(len(detail["manifest_sha256"]) == 64 for _, detail in updates if detail.get("status") == "ready")
    assert not any(path.name in {"scene.json", "task.json", "exercise-task.json", "lab"} for path in tmp_path.rglob("*"))


@pytest.mark.parametrize("fail_at", ["proposals", "recovery", "body"])
def test_stage_rejection_preserves_completed_and_stopped_evidence_without_retry(tmp_path, monkeypatch, fail_at):
    paths, _ = full_inputs(tmp_path, monkeypatch)
    ready = preflight(paths, pose_recovery="selective-crop")
    calls = stages(monkeypatch, ready, fail_at=fail_at)
    target = outputs(tmp_path)
    with pytest.raises(ValueError, match="Unchanged stage gate"):
        module.prepare_exercise_scene_body(ready, **target, max_seconds=240)
    order = ["proposals", "recovery", "body"]
    assert [row[0] for row in calls] == order[:order.index(fail_at) + 1]
    for name, _, path, _ in calls:
        assert (path / "data.json").is_file()
        assert (path / "failure.json").exists() is (name == fail_at)
    if fail_at != "body":
        assert not target["out"].exists()


@pytest.mark.parametrize("case", ["missing", "same", "nested", "source", "different_parent", "parent_alias", "symlink", "existing"])
def test_output_plan_rejects_before_any_producer(tmp_path, monkeypatch, case):
    paths, _ = full_inputs(tmp_path, monkeypatch)
    ready = preflight(paths, pose_recovery="selective-crop")
    calls = stages(monkeypatch, ready)
    target = outputs(tmp_path)
    if case == "missing":
        del target["recovered_pose_out"]
    elif case == "same":
        target["pose_proposals_out"] = target["out"]
    elif case == "nested":
        target["pose_proposals_out"] = target["out"] / "proposals"
    elif case == "source":
        target["pose_proposals_out"] = paths[1]
    elif case == "different_parent":
        target["pose_proposals_out"] = tmp_path / "other/proposals"
    elif case == "parent_alias":
        target["pose_proposals_out"] = tmp_path / "other/../pose_proposals"
    elif case == "symlink":
        link = tmp_path / "alias"
        link.symlink_to(tmp_path, target_is_directory=True)
        target = {key: link / path.name for key, path in target.items()}
    else:
        target["pose_proposals_out"].mkdir()
    with pytest.raises(ValueError):
        module.prepare_exercise_scene_body(ready, **target, max_seconds=240)
    assert not calls and not (tmp_path / "body_source").exists()


def test_default_branch_rejects_hidden_recovery_paths_or_expanded_time(tmp_path, monkeypatch):
    paths = inputs(tmp_path, monkeypatch)
    ready = preflight(paths)
    with pytest.raises(ValueError, match="explicit selective-crop"):
        module.prepare_exercise_scene_body(ready, **outputs(tmp_path))
    with pytest.raises(ValueError, match="1–180"):
        module.prepare_exercise_scene_body(ready, tmp_path / "body", max_seconds=181)


@pytest.mark.parametrize("case", ["input", "reseal", "failure_marker"])
def test_ready_callbacks_cannot_change_inputs_before_next_producer(tmp_path, monkeypatch, case):
    paths, _ = full_inputs(tmp_path, monkeypatch)
    ready = preflight(paths, pose_recovery="selective-crop")
    calls, target = stages(monkeypatch, ready), outputs(tmp_path)

    def progress(stage, detail):
        if stage != "pose_proposals" or detail["status"] != "ready":
            return
        if case == "input":
            paths[2].write_bytes(b"different config")
        elif case == "reseal":
            path = target["pose_proposals_out"] / "manifest.json"
            value = json.loads(path.read_bytes()); value["changed"] = True; write(path, value)
        else:
            write(target["pose_proposals_out"] / "failure.json", {"status": "failed"})

    with pytest.raises(ValueError):
        module.prepare_exercise_scene_body(ready, **target, max_seconds=240, progress=progress)
    assert [row[0] for row in calls] == ["proposals"]
    assert not target["recovered_pose_out"].exists() and not target["out"].exists()


@pytest.mark.parametrize("changed", ["pose_proposals_out", "recovered_pose_out", "out"])
def test_final_guard_detects_unresealed_retained_child_mutation(tmp_path, monkeypatch, changed):
    paths, _ = full_inputs(tmp_path, monkeypatch)
    ready = preflight(paths, pose_recovery="selective-crop")
    calls, target = stages(monkeypatch, ready), outputs(tmp_path)

    def progress(stage, detail):
        if stage == "body_source" and detail["status"] == "ready":
            (target[changed] / "data.json").write_bytes(b"changed after consumption")

    with pytest.raises(ValueError, match="pinned SHA256"):
        module.prepare_exercise_scene_body(ready, **target, max_seconds=240, progress=progress)
    assert [row[0] for row in calls] == ["proposals", "recovery", "body"]
    assert (target["out"] / "data.json").exists()


@pytest.mark.parametrize("limit", [60, 241])
def test_time_envelope_or_missing_final_stage_reserve_rejects_before_inference(tmp_path, monkeypatch, limit):
    paths, _ = full_inputs(tmp_path, monkeypatch)
    ready = preflight(paths, pose_recovery="selective-crop")
    calls = stages(monkeypatch, ready)
    with pytest.raises((ValueError, TimeoutError)):
        module.prepare_exercise_scene_body(ready, **outputs(tmp_path), max_seconds=limit)
    assert calls == []


def test_shared_deadline_reserves_recovery_and_body_without_resetting_budget(tmp_path, monkeypatch):
    paths, _ = full_inputs(tmp_path, monkeypatch)
    ready = preflight(paths, pose_recovery="selective-crop")
    clock = [0.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    calls = stages(monkeypatch, ready, clock=clock)
    module.prepare_exercise_scene_body(ready, **outputs(tmp_path), max_seconds=240, deadline=210)
    assert calls[0][3]["max_seconds"] == 150 and calls[0][3]["deadline"] == 210
    assert [row[3]["max_seconds"] for row in calls[1:]] == [30, 30]
    assert clock[0] == 160


def test_optional_implementation_change_after_preflight_stops_before_producer(tmp_path, monkeypatch):
    paths, _ = full_inputs(tmp_path, monkeypatch)
    ready = preflight(paths, pose_recovery="selective-crop")
    calls = stages(monkeypatch, ready)
    original = module.read_package_file

    def changed(directory, name, **options):
        if name == "pose_crop_pipeline.py":
            raise ValueError("Optional implementation changed")
        return original(directory, name, **options)

    monkeypatch.setattr(module, "read_package_file", changed)
    with pytest.raises(ValueError, match="Optional implementation changed"):
        module.prepare_exercise_scene_body(ready, **outputs(tmp_path), max_seconds=240)
    assert calls == []
