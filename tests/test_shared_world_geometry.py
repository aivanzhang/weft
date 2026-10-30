"""Authored shape alternatives preserve evidence, framing and task meaning."""

from copy import deepcopy
import hashlib
import json

import pytest

from weft import shared_world_geometry as module
from weft.io import canonical_hash, write_json
from weft.shared_exercise_task import build_shared_exercise_task
from weft.shared_exercise_lab import prepare_shared_exercise_lab
from weft.shared_world import validate_shared_world
from weft.shared_world_lab import authored_shared_world_scene
from weft.shared_world_packaging import collect_shared_source_files, read_package_file, read_package_manifest
from weft.shared_world_reconstruction import construct_equipment_world
from test_shared_world_packaging import candidate as mapped_candidate, seal
from test_shared_world_reconstruction import fixture as body_fixture


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def geometry(obj):
    part = {"id": "material", "shape": obj["shape"], "position": [0., 0., 0.], "quaternion": [1., 0., 0., 0.]}
    part.update({key: deepcopy(obj[key]) for key in ("size", "radius") if key in obj})
    return {"profile": "rigid-compound-v1", "origin": "authored_hypothesis",
            "assumptions": ["Analytic explicit geometry prior; no observed cavity or grasp."],
            "inertia": {"mode": "authored_centered_diagonal", "diagonal": [.004, .004, .004]}, "parts": [part]}


def inputs(tmp_path, *, scene=None, object_id="marker_box", extras=None):
    source = tmp_path / "source"
    source.mkdir()
    scene = deepcopy(scene) if scene is not None else authored_shared_world_scene()
    scene["task"]["horizon_steps"] = 3
    scene = validate_shared_world(scene)
    write_json(source / "scene.json", scene)
    for name, raw in (extras or {}).items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    seal(source)
    config = {"schema_version": 1, "profile": module.CONFIG_PROFILE, "base_scene_sha256": canonical_hash(scene),
              "assumptions": ["Explicit alternative geometry, not inferred from segmentation absence."],
              "replacements": [{"object_id": object_id,
                                "geometry": geometry(next(obj for obj in scene["objects"] if obj["id"] == object_id))}]}
    config_path = tmp_path / "shape.json"
    write_json(config_path, config)
    return source, scene, config_path, config


def verify(root):
    raw, hashes = read_package_manifest(root)
    for name, digest in hashes.items():
        read_package_file(root, name, max_bytes=64 * 1024 * 1024, expected_sha256=digest)
    return json.loads(raw), hashes


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_ready_alternative_preserves_physical_contract_and_archives_old_fit_and_policies(tmp_path):
    old = {"policy.json": b'{"old_policy":true}', "evaluation.json": b'{"status":"evaluated","successes":3}',
           "geometry-proposal.json": b'{"prior_fit":0.1}', "reconstruction.json": b'{"status":"ready","prior_box_fit":0.1}',
           "inspection/box.png": b"prior box panel"}
    source, scene, config_path, config = inputs(tmp_path, extras=old)
    before, config_raw = snapshot(source), config_path.read_bytes()
    obj = config["replacements"][0]["geometry"]
    obj["parts"] = [
        {"id": "block", "shape": "box", "position": [-.02, 0., 0.], "quaternion": [1., 0., 0., 0.], "size": [.02, .04, .04]},
        {"id": "round", "shape": "sphere", "position": [.02, 0., 0.], "quaternion": [1., 0., 0., 0.], "radius": .01},
        {"id": "bar", "shape": "capsule", "position": [0., 0., 0.], "quaternion": [1., 0., 0., 0.], "radius": .01, "length": .03},
    ]
    write_json(config_path, config)
    config_raw = config_path.read_bytes()
    out = tmp_path / "hypothesis"
    result = module.prepare_shared_world_geometry(source, config_path, out)
    assert result["status"] == "ready", result["reasons"]
    assert snapshot(source) == before and config_path.read_bytes() == config_raw
    manifest, hashes = verify(out)
    assert manifest["status"] == "ready" and result["manifest_sha256"] == sha((out / "manifest.json").read_bytes())
    assert (out / "input-config.json").read_bytes() == config_raw
    changed = validate_shared_world(json.loads((out / "scene.json").read_bytes()))
    assert all(changed[key] == scene[key] for key in scene if key not in {"objects", "assumptions", "learned_policy_available"})
    assert changed["objects"][0] == scene["objects"][0] and changed["objects"][2] == scene["objects"][2]
    before_object, after_object = scene["objects"][1], changed["objects"][1]
    assert after_object["geometry"] == obj
    for key in before_object.keys() - {"shape", "size", "provenance"}:
        assert after_object[key] == before_object[key]
    assert changed["assumptions"][:-1] == [module.BASE_CONTEXT + item for item in scene["assumptions"]]
    assert changed["learned_policy_available"] is False
    assert all(result["preserved"].values()) and result["physics_steps"] == result["fits"] == result["api_calls"] == 0
    assert result["source_fit"] == result["physical_validation"] == "unverified"
    assert result["full_reference_clearance_checked"] is False
    for name, raw in old.items():
        assert (out / "source-run" / name).read_bytes() == raw
        if name != "reconstruction.json":
            assert name not in hashes
    reconstruction = json.loads((out / "reconstruction.json").read_bytes())
    assert reconstruction["mode"] == "authored_compound_geometry_hypothesis" and "prior_box_fit" not in reconstruction
    assert reconstruction["source_fit"] == reconstruction["physical_validation"] == "unverified"
    previous = json.loads((out / "previous-geometry.json").read_bytes())
    assert previous["base_assumptions"] == scene["assumptions"]
    assert previous["objects"][0]["previous_geometry"] == {"shape": "box", "size": [.07] * 3}
    initial = json.loads((out / "initial-geometry-check.json").read_bytes())
    assert initial["physics_steps"] == 0 and initial["status"] == "passed_existing_initial_clearance_checks"


def test_source_prefix_observation_clock_and_uncertainty_survive_without_old_shape_fit(tmp_path):
    base, proposal, kwargs, documents = body_fixture(include_documents=True)
    scene = construct_equipment_world(base, proposal, **kwargs)
    evidence = {path: documents[digest] for digest, path in scene["source_binding"]["evidence_documents"].items()}
    source, scene, config_path, _ = inputs(tmp_path, scene=scene, object_id="equipment-a", extras=evidence)
    original_scope = build_shared_exercise_task(scene)["source_scope"]
    result = module.prepare_shared_world_geometry(source, config_path, tmp_path / "hypothesis")
    assert result["status"] == "ready", result["reasons"]
    changed = json.loads((tmp_path / "hypothesis/scene.json").read_bytes())
    assert build_shared_exercise_task(changed)["source_scope"] == original_scope
    for row in original_scope["equipment_initialization"]:
        assert (row["body_start_frame"], row["equipment_observation_frame"], row["unobserved_prefix_frames"]) == (2, 8, 6)
        assert row["initial_state_scope"] == "authored_static_extrapolation"
    old, new = scene["objects"][0]["provenance"], changed["objects"][0]["provenance"]
    for key in ("source_prefix", "source_reference", "source_bounds_px", "camera_hypothesis", "reference_object_force"):
        assert new[key] == old[key]
    assert new["source_prefix"]["source_slots_filled"] == 0
    for key in ("projected_bounds_px", "geometry_model_units", "initial_geometry_checks"):
        assert key not in new
    assert new["previous_geometry_provenance"] == old and new["previous_geometry_validation_inherited"] is False
    box_claim = next(value for value in scene["assumptions"] if "solid-box" in value)
    assert box_claim not in changed["assumptions"] and module.BASE_CONTEXT + box_claim in changed["assumptions"]
    previous = json.loads((tmp_path / "hypothesis/previous-geometry.json").read_bytes())
    assert previous["base_assumptions"] == scene["assumptions"]
    assert changed["human_spec"] == scene["human_spec"] and changed["source_binding"] == scene["source_binding"]
    proof = collect_shared_source_files(changed, tmp_path / "hypothesis")
    assert proof.report["source_evidence_verified"] and dict(proof.files) == evidence
    verify(tmp_path / "hypothesis")


def test_sealed_custom_exercise_task_rebinds_only_scene_identity(tmp_path):
    scene = authored_shared_world_scene()
    scene["task"]["horizon_steps"] = 3
    task = build_shared_exercise_task(scene, root_assistance=True, contact_requirements=[
        {"object_id": "ball", "wrist": "right", "kind": "wrist_region_contact", "start_step": 0, "end_step": 3}])
    raw = (json.dumps(task, indent=4) + "\n").encode()
    source, scene, config_path, _ = inputs(tmp_path, scene=scene, extras={"exercise-task.json": raw})
    out = tmp_path / "hypothesis"
    result = module.prepare_shared_world_geometry(source, config_path, out)
    assert result["status"] == "ready", result["reasons"]
    updated_raw = (out / "exercise-task.json").read_bytes()
    updated = json.loads(updated_raw)
    assert updated["scene_sha256"] == result["candidate_scene_sha256"] != task["scene_sha256"]
    assert {key: value for key, value in updated.items() if key != "scene_sha256"} == {
        key: value for key, value in task.items() if key != "scene_sha256"}
    assert (out / "source-run/exercise-task.json").read_bytes() == raw
    receipt = result["exercise_task"]
    assert receipt["changed_fields"] == ["scene_sha256"] and receipt["parameters_source_scope_and_gates_unchanged"]
    assert receipt["previous_file_sha256"] == sha(raw) and receipt["new_file_sha256"] == sha(updated_raw)
    assert receipt["previous_task_sha256"] == canonical_hash(task) and receipt["new_task_sha256"] == canonical_hash(updated)
    assert {"shared_exercise_task.py", "shared_exercise_reset.py"} <= result["implementation_sha256"].keys()
    verify(out)


def test_invalid_sealed_exercise_task_is_not_silently_rebuilt(tmp_path, monkeypatch):
    scene = authored_shared_world_scene()
    scene["task"]["horizon_steps"] = 3
    task = build_shared_exercise_task(scene)
    task["gates_sha256"] = "f" * 64
    source, _, config_path, _ = inputs(tmp_path, scene=scene, extras={"exercise-task.json": json.dumps(task).encode()})
    monkeypatch.setattr(module, "SharedWorldEnv", lambda *a, **k: pytest.fail("Invalid task reached the simulator"))
    out = tmp_path / "unsupported"
    result = module.prepare_shared_world_geometry(source, config_path, out)
    assert result["status"] == "unsupported_input" and not (out / "scene.json").exists()
    assert "gate" in result["reasons"][0]["message"]
    assert not (out / "exercise-task.json").exists()
    verify(out)


@pytest.mark.parametrize("root_video", [False, True])
def test_exact_source_media_views_and_complete_equivalence_bundle_remain_loadable(tmp_path, root_video):
    source = tmp_path / "source"
    scene, _ = mapped_candidate(source, equivalent=True)
    proof = collect_shared_source_files(scene, source)
    original = next(raw for name, raw in proof.files.items() if name.endswith("/source.webm"))
    views = {"source.json": b'{"attribution":"saved analytic producer fixture"}\n',
             "pose-quality.json": b'{"scope":"unchanged source pose confidence"}\n'}
    for name, raw in views.items():
        (source / name).write_bytes(raw)
    if root_video:
        (source / "source.webm").write_bytes(original)
    seal(source)
    config = {"schema_version": 1, "profile": module.CONFIG_PROFILE, "base_scene_sha256": canonical_hash(scene),
              "assumptions": ["An authored shape prior, not a source fit."],
              "replacements": [{"object_id": "marker_box", "geometry": geometry(scene["objects"][1])}]}
    config_path = tmp_path / "shape.json"
    write_json(config_path, config)
    out = tmp_path / "hypothesis"
    result = module.prepare_shared_world_geometry(source, config_path, out)
    assert result["status"] == "ready", result["reasons"]
    for name, raw in {**views, "source.webm": original}.items():
        assert (out / name).read_bytes() == raw and result["source_view_files"][name] == sha(raw)
    changed = json.loads((out / "scene.json").read_bytes())
    assert changed["source_binding"] == scene["source_binding"]
    assert dict(collect_shared_source_files(changed, out).files) == dict(proof.files)
    verify(out)


@pytest.mark.parametrize("mutation", ["stale", "unknown", "duplicate", "version", "pose", "inertia", "parts", "nonfinite", "duplicate_json"])
def test_unsupported_or_invalid_configs_preserve_sealed_nonready_attempts_before_environment(tmp_path, monkeypatch, mutation):
    source, _, config_path, config = inputs(tmp_path)
    if mutation == "stale": config["base_scene_sha256"] = "f" * 64
    elif mutation == "unknown": config["replacements"][0]["object_id"] = "unknown-object"
    elif mutation == "duplicate": config["replacements"] *= 2
    elif mutation == "version": config["schema_version"] = 2
    elif mutation == "pose": config["replacements"][0]["position"] = [0., 0., 1.]
    elif mutation == "inertia": config["replacements"][0]["geometry"]["inertia"]["diagonal"] = [.01, .01, 1.]
    elif mutation == "parts": config["replacements"][0]["geometry"]["parts"] *= 9
    write_json(config_path, config)
    if mutation == "nonfinite": config_path.write_bytes(config_path.read_bytes().replace(b'0.004', b'NaN', 1))
    elif mutation == "duplicate_json": config_path.write_bytes(config_path.read_bytes().replace(b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1'))
    before = snapshot(source)
    def no_environment(*args, **kwargs):
        pytest.fail("Invalid proposals must fail before creating an environment")
    monkeypatch.setattr(module, "SharedWorldEnv", no_environment)
    out = tmp_path / "rejected"
    result = module.prepare_shared_world_geometry(source, config_path, out)
    assert result["status"] in {"unsupported_input", "rejected"} and result["reasons"]
    assert snapshot(source) == before and not (out / "scene.json").exists()
    assert (out / "input-config.json").read_bytes() == config_path.read_bytes()
    assert verify(out)[0]["status"] == result["status"]


def test_overlap_in_nonfirst_part_is_rejected_with_candidate_and_initial_failure(tmp_path):
    source, _, config_path, config = inputs(tmp_path)
    config["replacements"][0]["geometry"]["parts"].append({"id": "buried", "shape": "sphere",
        "position": [0., 0., -.05], "quaternion": [1., 0., 0., 0.], "radius": .02})
    write_json(config_path, config)
    out = tmp_path / "rejected"
    result = module.prepare_shared_world_geometry(source, config_path, out)
    assert result["status"] == "rejected" and result["reasons"][0]["stage"] == "initial_geometry"
    assert not result["initial_geometry_admitted"] and not (out / "scene.json").exists()
    assert (out / "scene.candidate.rejected.json").is_file() and (out / "candidate.xml").is_file()
    initial = json.loads((out / "initial-geometry-check.json").read_bytes())
    assert initial["status"] == "rejected" and initial["physics_steps"] == 0
    assert "intersect" in initial["reason"]["message"].lower()
    verify(out)


@pytest.mark.parametrize("mutation", ["hash", "missing_proof", "source_video", "two_videos", "bad_status"])
def test_unverified_incomplete_or_ambiguous_source_custody_cannot_be_promoted(tmp_path, monkeypatch, mutation):
    source = tmp_path / "source"
    scene, doc = mapped_candidate(source, equivalent=True)
    config = {"schema_version": 1, "profile": module.CONFIG_PROFILE, "base_scene_sha256": canonical_hash(scene),
              "assumptions": ["Analytic authored alternative."],
              "replacements": [{"object_id": "marker_box", "geometry": geometry(scene["objects"][1])}]}
    config_path = tmp_path / "shape.json"
    write_json(config_path, config)
    if mutation == "hash":
        (source / doc).write_bytes(b"changed")
    elif mutation == "missing_proof":
        manifest = json.loads((source / "manifest.json").read_bytes())
        del manifest["files"][doc]
        write_json(source / "manifest.json", manifest)
    else:
        if mutation == "bad_status":
            write_json(source / "status.json", {"status": "failed"})
        else:
            (source / "source.webm").write_bytes(b"not original")
            if mutation == "two_videos": (source / "source.mp4").write_bytes(b"ambiguous")
        seal(source)
    monkeypatch.setattr(module, "SharedWorldEnv", lambda *a, **k: pytest.fail("Unverified sources reached the simulator"))
    out = tmp_path / "rejected"
    result = module.prepare_shared_world_geometry(source, config_path, out)
    assert result["status"] == "unsupported_input" and not (out / "scene.json").exists()
    assert (out / "source-run/manifest.json").read_bytes() == (source / "manifest.json").read_bytes()
    verify(out)


@pytest.mark.parametrize("path_kind", ["source", "config", "output_parent", "overlap", "occupied"])
def test_unsafe_paths_reject_without_creating_or_overwriting_outputs(tmp_path, path_kind):
    source, _, config_path, _ = inputs(tmp_path)
    out = tmp_path / "out"
    if path_kind == "source":
        link = tmp_path / "linked-source"
        link.symlink_to(source, target_is_directory=True)
        source = link
    elif path_kind == "config":
        link = tmp_path / "linked-config.json"
        link.symlink_to(config_path)
        config_path = link
    elif path_kind == "output_parent":
        parent = tmp_path / "linked-output"
        parent.symlink_to(tmp_path / "source", target_is_directory=True)
        out = parent / "child"
    elif path_kind == "overlap":
        out = source / "child"
    else:
        out.mkdir()
        (out / "sentinel").write_bytes(b"keep")
    before = snapshot(tmp_path / "source")
    with pytest.raises(ValueError):
        module.prepare_shared_world_geometry(source, config_path, out)
    assert snapshot(tmp_path / "source") == before
    assert (out / "sentinel").read_bytes() == b"keep" if path_kind == "occupied" else not out.exists()


@pytest.mark.parametrize("mutation", ["config", "source"])
def test_changed_inputs_after_initial_check_fail_without_replacing_frozen_bytes(tmp_path, monkeypatch, mutation):
    source, _, config_path, _ = inputs(tmp_path)
    initial = module._initial_check
    original = config_path.read_bytes() if mutation == "config" else (source / "scene.json").read_bytes()
    def changed(scene):
        result = initial(scene)
        target = config_path if mutation == "config" else source / "scene.json"
        target.write_bytes(original + b" ")
        return result
    monkeypatch.setattr(module, "_initial_check", changed)
    out = tmp_path / "rejected"
    result = module.prepare_shared_world_geometry(source, config_path, out)
    assert result["status"] == "unsupported_input" and result["reasons"][0]["stage"] == "input_reverification"
    assert not (out / "scene.json").exists()
    frozen = out / ("input-config.json" if mutation == "config" else "source-run/scene.json")
    assert frozen.read_bytes() == original
    verify(out)


@pytest.mark.parametrize("bound", ["config_bytes", "input_files", "deadline"])
def test_bounded_failures_leave_a_nonready_receipt(tmp_path, monkeypatch, bound):
    source, _, config_path, _ = inputs(tmp_path, extras={"extra.json": b"{}"})
    if bound == "config_bytes": monkeypatch.setattr(module, "MAX_CONFIG_BYTES", 32)
    elif bound == "input_files": monkeypatch.setattr(module, "MAX_INPUT_FILES", 1)
    else:
        original = module._initial_check
        def timeout(scene):
            original(scene)
            raise TimeoutError("Injected bounded native-call overrun")
        monkeypatch.setattr(module, "_initial_check", timeout)
    out = tmp_path / "rejected"
    result = module.prepare_shared_world_geometry(source, config_path, out)
    assert result["status"] == ("failed" if bound == "deadline" else "unsupported_input")
    assert not (out / "scene.json").exists()
    verify(out)


@pytest.mark.parametrize("failure", ["scene_write", "partial_manifest"])
def test_interrupted_publication_preserves_failed_custody_without_ready_seal(tmp_path, monkeypatch, failure):
    source, _, config_path, _ = inputs(tmp_path)
    writer, failed = module.write_package_files, False
    def interrupted(directory, files):
        nonlocal failed
        name = next(iter(files))
        if not failed and name == ("scene.json" if failure == "scene_write" else "manifest.json"):
            failed = True
            if failure == "partial_manifest": writer(directory, {"manifest.json": b'{"interrupted":'})
            raise OSError("Injected publication failure")
        return writer(directory, files)
    monkeypatch.setattr(module, "write_package_files", interrupted)
    out = tmp_path / "failed"
    result = module.prepare_shared_world_geometry(source, config_path, out)
    assert result["status"] == "failed" and json.loads((out / "status.json").read_bytes())["status"] == "failed"
    assert json.loads((out / result["manifest_name"]).read_bytes())["status"] == "failed"
    if failure == "scene_write":
        assert not (out / "scene.json").exists()
        verify(out)
    else:
        assert result["manifest_name"] == "manifest.json"
        assert (out / "manifest.publication-failed.json").read_bytes() == b'{"interrupted":'
        assert result["publication_failure"]["preserved_publication_artifacts"]["manifest.json"]["sha256"] == sha(b'{"interrupted":')
        assert (out / "scene.candidate.publication-failed.json").is_file() and not (out / "scene.json").exists()
        verify(out)
    with pytest.raises(ValueError, match="unsuccessful"):
        prepare_shared_exercise_lab(out, tmp_path / "forbidden-lab")


def test_complete_manifest_then_error_preserves_bytes_but_cannot_leave_a_ready_scene(tmp_path, monkeypatch):
    source, _, config_path, _ = inputs(tmp_path)
    writer, failed = module.write_package_files, False
    def postwrite_error(directory, files):
        nonlocal failed
        writer(directory, files)
        if "manifest.json" in files and not failed:
            failed = True
            raise OSError("Injected error after successful final-manifest publication")
    monkeypatch.setattr(module, "write_package_files", postwrite_error)
    out = tmp_path / "committed"
    result = module.prepare_shared_world_geometry(source, config_path, out)
    assert result["status"] == "failed" and verify(out)[0]["status"] == "failed"
    preserved_manifest = json.loads((out / "manifest.publication-failed.json").read_bytes())
    assert preserved_manifest["status"] == "ready"
    preserved_scene = (out / "scene.candidate.publication-failed.json").read_bytes()
    assert sha(preserved_scene) == preserved_manifest["files"]["scene.json"] and not (out / "scene.json").exists()
    assert json.loads((out / "geometry-hypothesis.json").read_bytes())["status"] == "failed"
    assert json.loads((out / "reconstruction.json").read_bytes())["status"] == "failed"
    with pytest.raises(ValueError, match="unsuccessful"):
        prepare_shared_exercise_lab(out, tmp_path / "forbidden-lab")
    retry = module.prepare_shared_world_geometry(out, config_path, tmp_path / "forbidden-geometry")
    assert retry["status"] == "unsupported_input" and not (tmp_path / "forbidden-geometry/scene.json").exists()


def test_config_validation_returns_a_detached_canonical_identity_contract(tmp_path):
    source, scene, config_path, config = inputs(tmp_path)
    detached = module.validate_geometry_hypothesis_config(config)
    detached["replacements"][0]["geometry"]["parts"][0]["position"][0] = 1
    assert config["replacements"][0]["geometry"]["parts"][0]["position"] == [0., 0., 0.]
    original_sha = sha((source / "scene.json").read_bytes())
    (source / "scene.json").write_bytes(json.dumps(scene, separators=(",", ":")).encode())
    seal(source)
    assert sha((source / "scene.json").read_bytes()) != original_sha
    result = module.prepare_shared_world_geometry(source, config_path, tmp_path / "hypothesis")
    assert result["status"] == "ready" and result["base_scene_sha256"] == config["base_scene_sha256"]


@pytest.mark.parametrize("seconds", [False, 0, 61, float("inf")])
def test_invalid_execution_bounds_reject_before_output(tmp_path, seconds):
    with pytest.raises(ValueError, match="1–60"):
        module.prepare_shared_world_geometry(tmp_path / "source", tmp_path / "config", tmp_path / "out", max_seconds=seconds)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name,status", [("manifest.json", "running"), ("manifest.json", "queued"),
    ("manifest.json", "unrecognized"), ("manifest.json", None), ("status.json", "running"),
    ("report.json", "queued"), ("training.json", "running"), ("evaluation.json", "queued")])
def test_nonterminal_or_unknown_statuses_reject_before_environment(tmp_path, monkeypatch, name, status):
    source, _, config_path, _ = inputs(tmp_path)
    document = json.loads((source / name).read_bytes()) if name == "manifest.json" else {}
    document["status"] = status
    write_json(source / name, document)
    if name != "manifest.json": seal(source)
    monkeypatch.setattr(module, "SharedWorldEnv", lambda *a, **k: pytest.fail("Incomplete run reached the simulator"))
    out = tmp_path / "unsupported"
    result = module.prepare_shared_world_geometry(source, config_path, out)
    assert result["status"] == "unsupported_input" and "terminal status" in result["reasons"][0]["message"]
    assert not (out / "scene.json").exists()
    verify(out)


@pytest.mark.parametrize("status", ["ready", "trained", "evaluated"])
def test_supported_terminal_statuses_remain_compatible(tmp_path, status):
    source, _, config_path, _ = inputs(tmp_path)
    manifest = json.loads((source / "manifest.json").read_bytes())
    manifest["status"] = status
    write_json(source / "manifest.json", manifest)
    result = module.prepare_shared_world_geometry(source, config_path, tmp_path / "hypothesis")
    assert result["status"] == "ready", result["reasons"]


@pytest.mark.parametrize("bound", ["bytes", "files"])
def test_oversized_retained_plan_emits_bounded_receipt_without_claiming_full_custody(tmp_path, monkeypatch, bound):
    source, _, config_path, _ = inputs(tmp_path)
    before = snapshot(source)
    if bound == "bytes": monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 16 * 1024)
    else: monkeypatch.setattr(module, "MAX_OUTPUT_FILES", 8)
    out = tmp_path / "receipt"
    result = module.prepare_shared_world_geometry(source, config_path, out)
    assert result["status"] == "rejected" and result["custody"]["mode"] == "receipt_only"
    assert not result["custody"]["complete_input_archive_preserved"] and not result["custody"]["candidate_document_preserved"]
    assert result["base_manifest_sha256"] == sha(before["manifest.json"])
    assert result["input_config_sha256"] == sha(config_path.read_bytes())
    assert result["implementation_sha256"]["shared_world_geometry.py"] == sha(module.Path(module.__file__).read_bytes())
    assert result["input_paths"] == {"base_run": str(source), "config": str(config_path)}
    assert snapshot(source) == before and not (out / "scene.json").exists() and not (out / "source-run").exists()
    manifest, hashes = verify(out)
    assert manifest["status"] == "rejected" and set(hashes) == {"geometry-hypothesis.json", "reconstruction.json"}
    assert result["bytes"] <= min(module.MAX_OUTPUT_BYTES, module.MAX_FAILURE_RECEIPT_BYTES)
    assert result["files"] <= module.MAX_OUTPUT_FILES
    reconstruction = json.loads((out / "reconstruction.json").read_bytes())
    assert reconstruction["custody"] == result["custody"]
    assert reconstruction["details"] == {"full_report": "geometry-hypothesis.json"}
