"""Complete-reference promotion preserves source custody and never hides rejected preparation."""

from copy import deepcopy
import json
import sys

import mujoco
import pytest

from weft import shared_exercise_fullspan as module
from weft.io import canonical_hash, digest, write_json
from weft.pipeline import verify_artifacts
from weft.shared_exercise_task import GATES_SHA256, build_shared_exercise_task, validate_shared_exercise_task
from weft.shared_world import validate_shared_world
from weft.shared_world_packaging import collect_shared_source_files
from test_shared_world_fullspan import fullspan_scene
from test_shared_world_packaging import candidate as mapped_candidate, seal
from test_shared_world_reconstruction import fixture as body_fixture
from weft.shared_world_reconstruction import construct_equipment_world


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def source_fixture(tmp_path, *, scene=None, task=None, extras=None):
    source = tmp_path / "source"
    source.mkdir()
    scene = deepcopy(scene) if scene is not None else fullspan_scene()
    scene["profile"] = "shared-world-v1"
    scene["task"].update(horizon_steps=3, start_offset_s=0.)
    scene["learned_policy_available"] = False
    scene = validate_shared_world(scene)
    task = deepcopy(task) if task is not None else build_shared_exercise_task(scene, root_assistance=True)
    write_json(source / "scene.json", scene)
    write_json(source / "exercise-task.json", task)
    for name, raw in (extras or {}).items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    seal(source)
    return source, scene, task


def test_complete_promotion_preserves_every_original_byte_without_physics(tmp_path, monkeypatch):
    old = {"policy.json": b'{"old_policy":true}', "evaluation.json": b'{"status":"evaluated","old_successes":3}',
           "reconstruction.json": b'{"status":"ready","old_image_fit":0.2}',
           "source.json": b'{"attribution":"analytic example; not measured"}'}
    source, original, task = source_fixture(tmp_path, extras=old)
    before = snapshot(source)
    monkeypatch.setattr(mujoco, "mj_step", lambda *a, **k: pytest.fail("Promotion must not step physics"))
    monkeypatch.setattr(mujoco, "mj_resetData", lambda *a, **k: pytest.fail("Promotion must not reset physics"))
    target = tmp_path / "fullspan"
    result = module.prepare_fullspan_shared_exercise(source, target)
    assert result["status"] == "ready", result
    assert snapshot(source) == before
    assert result["physics_steps"] == result["resets"] == result["training_episodes"] == result["api_calls"] == 0
    assert result["old_policy_or_evaluation_attached"] is False
    new = validate_shared_world(json.loads((target / "scene.json").read_bytes()))
    updated = validate_shared_exercise_task(json.loads((target / "exercise-task.json").read_bytes()), new)
    assert new["profile"] == "shared-world-v2" and new["task"]["horizon_steps"] == 254
    assert new["task"]["start_offset_s"] == 0 and new["dt"] == original["dt"]
    restored = deepcopy(new)
    restored["profile"], restored["task"] = original["profile"], original["task"]
    assert restored == original
    assert updated["scene_sha256"] == canonical_hash(new)
    assert updated["gates_sha256"] == task["gates_sha256"] == GATES_SHA256
    assert {k: v for k, v in updated.items() if k not in {"scene_sha256", "interval"}} == {
        k: v for k, v in task.items() if k not in {"scene_sha256", "interval"}}
    for name, raw in before.items():
        assert (target / "source-run" / name).read_bytes() == raw
    assert (target / "source.json").read_bytes() == old["source.json"]
    for name in ("policy.json", "evaluation.json", "reconstruction.json"):
        assert not (target / name).exists()
    assert result["reference_interval"]["reference_poses"] == 255
    assert all(result["preserved"].values())
    assert verify_artifacts(target)["manifest_sha256"] == result["manifest_sha256"]


def test_source_prefix_missing_slots_and_original_clock_binding_survive(tmp_path):
    base, proposal, kwargs, documents = body_fixture(include_documents=True)
    scene = construct_equipment_world(base, proposal, **kwargs)
    evidence = {path: documents[sha] for sha, path in scene["source_binding"]["evidence_documents"].items()}
    source, original, task = source_fixture(tmp_path, scene=scene, extras=evidence)
    target = tmp_path / "fullspan"
    result = module.prepare_fullspan_shared_exercise(source, target)
    assert result["status"] == "ready", result
    new = json.loads((target / "scene.json").read_bytes())
    updated = json.loads((target / "exercise-task.json").read_bytes())
    assert new["source_binding"] == original["source_binding"]
    assert new["coordinate_frame"] == original["coordinate_frame"]
    assert new["human_spec"] == original["human_spec"]
    assert updated["source_scope"] == task["source_scope"]
    for row in updated["source_scope"]["equipment_initialization"]:
        assert (row["body_start_frame"], row["equipment_observation_frame"], row["unobserved_prefix_frames"]) == (2, 8, 6)
    assert dict(collect_shared_source_files(new, target).files) == evidence
    verify_artifacts(target)


@pytest.mark.parametrize("root_video", [False, True])
def test_derivative_mapping_and_exact_original_video_remain_self_contained(tmp_path, root_video):
    source = tmp_path / "mapped"
    scene, _ = mapped_candidate(source, equivalent=True)
    task = build_shared_exercise_task(scene)
    write_json(source / "exercise-task.json", task)
    original = next(raw for name, raw in collect_shared_source_files(scene, source).files.items()
                    if name.endswith("/source.webm"))
    if root_video:
        (source / "source.webm").write_bytes(original)
    seal(source)
    before = snapshot(source)
    target = tmp_path / "fullspan"
    result = module.prepare_fullspan_shared_exercise(source, target)
    assert result["status"] == "ready", result
    assert snapshot(source) == before
    assert (target / "source.webm").read_bytes() == original
    new = json.loads((target / "scene.json").read_bytes())
    assert new["source_binding"] == scene["source_binding"]
    assert dict(collect_shared_source_files(new, target).files) == dict(collect_shared_source_files(scene, source).files)
    verify_artifacts(target)


@pytest.mark.parametrize("mutation", ["contact", "gate", "learned", "v2", "missing_task", "changed_source", "fractional", "running"])
def test_unsupported_or_changed_inputs_publish_only_explicit_failure(tmp_path, mutation):
    source, scene, task = source_fixture(tmp_path)
    if mutation == "contact":
        task["contact_requirements"] = [{"object_id": "ball", "wrist": "right", "kind": "upward_support",
                                         "start_step": 0, "end_step": 3}]
    elif mutation == "gate":
        task["gates_sha256"] = "a" * 64
    elif mutation == "learned":
        scene["learned_policy_available"] = True
        task = build_shared_exercise_task(scene)
    elif mutation == "v2":
        scene["profile"] = "shared-world-v2"
        scene["task"]["horizon_steps"] = 254
        task = build_shared_exercise_task(scene)
    elif mutation == "fractional":
        for key in ("reference", "root_reference"):
            scene["human_spec"][key][-1]["time_s"] += .001
        task = build_shared_exercise_task(scene)
    write_json(source / "scene.json", scene)
    write_json(source / "exercise-task.json", task)
    seal(source)
    if mutation == "missing_task":
        manifest = json.loads((source / "manifest.json").read_bytes())
        del manifest["files"]["exercise-task.json"]
        write_json(source / "manifest.json", manifest)
    elif mutation == "running":
        manifest = json.loads((source / "manifest.json").read_bytes())
        manifest["status"] = "running"
        write_json(source / "manifest.json", manifest)
    elif mutation == "changed_source":
        (source / "scene.json").write_bytes(b'{}')
    before = snapshot(source)
    target = tmp_path / "failed"
    result = module.prepare_fullspan_shared_exercise(source, target)
    assert result["status"] == "failed" and result["runnable_scene_published"] is False
    assert not (target / "scene.json").exists() and not (target / "exercise-task.json").exists()
    assert snapshot(source) == before
    assert json.loads((target / "manifest.json").read_bytes())["status"] == "failed"
    verify_artifacts(target)


@pytest.mark.parametrize("budget", [True, 0., 31., float("nan")])
def test_invalid_execution_bounds_create_nothing(tmp_path, budget):
    source, _, _ = source_fixture(tmp_path)
    target = tmp_path / "unused"
    with pytest.raises(ValueError, match="bound"):
        module.prepare_fullspan_shared_exercise(source, target, max_seconds=budget)
    assert not target.exists()


@pytest.mark.parametrize("relationship", ["inside", "outside_parent", "existing", "source_symlink", "output_symlink"])
def test_output_paths_never_overwrite_or_follow_aliases(tmp_path, relationship):
    source, _, _ = source_fixture(tmp_path)
    before = snapshot(source)
    target = tmp_path / "unused"
    if relationship == "inside":
        target = source / "nested"
    elif relationship == "outside_parent":
        target = tmp_path
    elif relationship == "existing":
        target.mkdir()
        (target / "keep").write_bytes(b"saved evidence")
    elif relationship == "source_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        source = alias
    else:
        target.symlink_to(source, target_is_directory=True)
    with pytest.raises((ValueError, FileExistsError)):
        module.prepare_fullspan_shared_exercise(source, target)
    assert snapshot(source) == before
    if relationship == "existing":
        assert (target / "keep").read_bytes() == b"saved evidence"


def test_output_bound_failure_keeps_receipt_without_ready_artifacts(tmp_path, monkeypatch):
    source, _, _ = source_fixture(tmp_path)
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 1024)
    out = tmp_path / "bounded"
    result = module.prepare_fullspan_shared_exercise(source, out)
    assert result["status"] == "failed"
    assert not (out / "scene.json").exists()
    manifest = json.loads((out / "manifest.json").read_bytes())
    assert manifest["status"] == "failed" and manifest["custody"] == "receipt_only"
    assert manifest["unsealed_outputs_preserved"]


def test_failed_finalization_preserves_written_scene_and_cannot_leave_ready_manifest(tmp_path, monkeypatch):
    source, _, _ = source_fixture(tmp_path)
    out = tmp_path / "failed"
    original = module._unchanged
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise TimeoutError("Synthetic post-write expiry")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_unchanged", fail_second)
    result = module.prepare_fullspan_shared_exercise(source, out)
    assert result["status"] == "failed"
    assert not (out / "scene.json").exists() and not (out / "exercise-task.json").exists()
    assert (out / "scene.publication-failed.json").is_file()
    assert (out / "exercise-task.publication-failed.json").is_file()
    assert (out / "fullspan.publication-failed.json").is_file()
    assert json.loads((out / "manifest.json").read_bytes())["status"] == "failed"
    verify_artifacts(out)


def test_cli_reports_failed_preparation_with_nonzero_exit(tmp_path, monkeypatch, capsys):
    from weft.cli import main

    source, _, _ = source_fixture(tmp_path)
    manifest = json.loads((source / "manifest.json").read_bytes())
    manifest["status"] = "running"
    write_json(source / "manifest.json", manifest)
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-fullspan", "--run", str(source), "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
