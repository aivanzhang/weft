"""Exercise build routing and source custody; detector/physics producers are replaced explicitly."""

import json
from pathlib import Path
import sys
import time

import pytest

from test_exercise_build_pipeline import fake_learning
from test_exercise_scene_inputs import inputs, sha, write
from test_exercise_scene_native_recovery import child, full_inputs, stages as recovery_stages
from weft import cli, exercise_scene_inputs as admission, exercise_scene_pipeline as pipeline
from weft import exercise_support_inputs as support_inputs, shared_world_human, shared_world_lab, shared_world_support


def priors(source_sha):
    return {"schema_version": 1, "profile": support_inputs.PROFILE, "source_sha256": source_sha,
        "scale": 1., "root_assistance": False,
        "objects": [{"id": "bench", "label": "Authored bench", "role": "static", "shape": "box",
            "position": [1., .75, .8], "quaternion": [1., 0., 0., 0.], "size": [.6, .5, .04],
            "velocity": [0., 0., 0.], "angular_velocity": [0., 0., 0.], "mass": 10.,
            "friction": [.8, .01, .001], "color": "#707070",
            "provenance": {"mode": "authored_addition", "calibration_status": "estimated_prior",
                "physical_validation": "unverified", "source_contact_verified": False,
                "assumptions": ["Authored geometry; not observed contact"]}}],
        "support_requirements": [{"object_id": "bench", "human_segment": "torso", "direction_world": [0., 0., 1.],
            "minimum_load_body_weight_fraction": .2, "start_step": 6, "end_step": 60}]}


def prior_file(tmp_path, source_sha):
    path = tmp_path / "supports.json"
    path.write_text(json.dumps(priors(source_sha), indent=3) + "\n\n")
    return path


def test_raw_support_preflight_has_no_equipment_absence_claim(tmp_path, monkeypatch):
    video, _, _, model = inputs(tmp_path, monkeypatch)
    path = prior_file(tmp_path, sha(video.read_bytes()))
    ready = admission.preflight_support_scene_inputs(video, path, pose_model=model)
    receipt = ready.receipt()
    assert receipt["profile"] == "exercise-static-support-inputs-v1"
    assert receipt["scene_scope"] == "human_static_supports" and "inventory" not in receipt
    assert receipt["support_priors"] == priors(sha(video.read_bytes()))
    assert receipt["support_priors_sha256"] == sha(path.read_bytes())
    assert ready.evidence_run is None and ready.config_path == path and ready.source_mappings == ()
    assert all(count == 0 for count in receipt["work"].values())
    assert any(row["path"] == str(path) for row in receipt["input_pins"])
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        admission.recheck_exercise_scene_inputs(ready, deadline=time.monotonic() + 10)


def test_wrong_video_rejects_before_model_or_output(tmp_path, monkeypatch):
    video, _, _, model = inputs(tmp_path, monkeypatch)
    path = prior_file(tmp_path, "f" * 64)
    monkeypatch.setattr(admission, "_model_pins", lambda *_args: pytest.fail("Wrong video reached model preflight"))
    with pytest.raises(ValueError, match="different original video"):
        admission.preflight_support_scene_inputs(video, path, pose_model=model)


@pytest.mark.parametrize("kind", ["body_pin", "unknown", "source", "scale", "root", "observed", "moving", "segment", "window", "oversize"])
def test_bad_support_prior_rejects_before_inference(tmp_path, kind):
    value = priors("a" * 64)
    if kind == "body_pin": value["body_source_manifest_sha256"] = "b" * 64
    elif kind == "unknown": value["calibrated"] = True
    elif kind == "source": value["source_sha256"] = "unknown"
    elif kind == "scale": value["scale"] = .01
    elif kind == "root": value["root_assistance"] = 1
    elif kind == "observed": value["objects"][0]["provenance"]["source_contact_verified"] = True
    elif kind == "moving": value["objects"][0]["role"] = "passive"
    elif kind == "segment": value["support_requirements"][0]["human_segment"] = "palm"
    elif kind == "window": value["support_requirements"][0]["end_step"] = 0
    else: value["padding"] = "x" * (128 << 10)
    with pytest.raises(ValueError):
        support_inputs.validate_support_priors(json.dumps(value).encode())


@pytest.mark.parametrize("kind", ["direction_length", "direction_unit", "direction_boolean", "load_zero",
                                 "load_large", "load_boolean", "maximum_horizon", "overlap"])
def test_invalid_load_window_rejects_before_model_preflight(tmp_path, monkeypatch, kind):
    video, _, _, model = inputs(tmp_path, monkeypatch)
    value = priors(sha(video.read_bytes()))
    row = value["support_requirements"][0]
    if kind == "direction_length": row["direction_world"] = [0., 1.]
    elif kind == "direction_unit": row["direction_world"] = [0., 0., 2.]
    elif kind == "direction_boolean": row["direction_world"] = [0., 0., True]
    elif kind == "load_zero": row["minimum_load_body_weight_fraction"] = 0.
    elif kind == "load_large": row["minimum_load_body_weight_fraction"] = 2.1
    elif kind == "load_boolean": row["minimum_load_body_weight_fraction"] = True
    elif kind == "maximum_horizon": row["end_step"] = 1801
    else: value["support_requirements"].append(dict(row))
    path = tmp_path / "bad-load.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(admission, "_model_pins", lambda *_args: pytest.fail("Invalid load reached model preflight"))
    with pytest.raises(ValueError):
        admission.preflight_support_scene_inputs(video, path, pose_model=model)


def setup_build(tmp_path, monkeypatch, mode, *, train=False, fail_lab=False):
    calls = []
    source_sha = "a" * 64
    options = {"scene_scope": "human_static_supports"}
    if mode == "saved":
        source = child(tmp_path / "body", "exercise-body-source-v1", "construction_only", source_sha)
        options["human_run"] = source
    else:
        paths = full_inputs(tmp_path, monkeypatch)[0] if mode == "crop" else inputs(tmp_path, monkeypatch)
        source_sha = sha(paths[0].read_bytes())
        options.update(video=paths[0], pose_model=paths[3])
        if mode == "crop":
            options["pose_recovery"] = "selective-crop"
        else:
            def body(video, out, **kwargs):
                calls.append(("body_source", kwargs))
                return child(out, "exercise-body-source-v1", "construction_only", source_sha)
            monkeypatch.setattr(admission.exercise_body_source, "prepare_exercise_body_source", body)
    path = prior_file(tmp_path, source_sha)
    options["static_supports"] = path
    if mode == "crop":
        ready = admission.preflight_support_scene_inputs(paths[0], path, pose_model=paths[3], pose_recovery="selective-crop")
        recovery_stages(monkeypatch, ready)

    def identity(source, **kwargs):
        return {"source_sha256": source_sha, "manifest_sha256": sha((Path(source) / "manifest.json").read_bytes())}
    monkeypatch.setattr(shared_world_human, "read_person_body_source_identity", identity)

    def forbidden(*args, **kwargs):
        pytest.fail("Authored supports cannot create a ground-only inventory or equipment scaffold")
    monkeypatch.setattr(shared_world_lab, "prepare_shared_world_seed", forbidden)
    monkeypatch.setattr(shared_world_human, "construct_human_only_run", forbidden)
    monkeypatch.setattr(pipeline, "prepare_equipment_proposal", forbidden)
    monkeypatch.setattr(pipeline, "construct_source_equipment_run", forbidden)

    def construct(source, config, out, **kwargs):
        resolved = json.loads(Path(config).read_bytes())
        assert resolved["source_sha256"] == source_sha
        assert resolved["body_source_manifest_sha256"] == sha((Path(source) / "manifest.json").read_bytes())
        assert resolved["objects"] == priors(source_sha)["objects"]
        calls.append(("shared_world", {"source": source, "config": resolved, **kwargs}))
        child(out, "shared-world-v4", "ready", source_sha)
        return {"status": "ready", "profile": "shared-world-v4", "manifest_sha256": sha((out / "manifest.json").read_bytes()),
                "readiness": "candidate_constructed_not_lab_admitted", "reference_interval": {"complete": True}}
    monkeypatch.setattr(shared_world_support, "construct_static_support_run", construct)

    def lab(source, out, **kwargs):
        declared = json.loads((Path(source) / "manifest.json").read_bytes())["files"]
        if any(sha((Path(source) / name).read_bytes()) != expected for name, expected in declared.items()):
            raise ValueError("Lab admission rejects changed candidate bytes")
        calls.append(("interactive_lab", kwargs))
        if fail_lab:
            raise ValueError("Initial geometry remains inadmissible")
        (out / "run").mkdir(parents=True)
        write(out / "run/scene.json", {"profile": "shared-world-v4", "objects": priors(source_sha)["objects"]})
        write(out / "status.json", {"status": "ready"})
        return {"lab_directory": str(out), "scene_sha256": "4" * 64, "task_sha256": "5" * 64, "manifest_sha256": "6" * 64}
    monkeypatch.setattr(pipeline, "prepare_shared_exercise_lab", lab)
    if train:
        options["train_settings"] = {"evaluation_seed": 81741, "generations": 1, "population": 4, "trials": 1}
        fake_learning(monkeypatch, calls)
    return calls, options


@pytest.mark.parametrize("mode", ["saved", "raw", "crop"])
@pytest.mark.parametrize("train", [False, True])
def test_support_build_keeps_one_body_world_and_final_lab(tmp_path, monkeypatch, mode, train):
    calls, options = setup_build(tmp_path, monkeypatch, mode, train=train)
    raw = options["static_supports"].read_bytes()
    out = tmp_path / "build"
    result = pipeline.prepare_exercise_scene(None, None, None, out, **options)
    assert result["status"] == result["lab_status"] == "ready"
    assert result["lab_directory"] == str(out / ("learning_workflow/lab" if train else "lab"))
    assert sum(name in {"interactive_lab", "learning_workflow"} for name, _ in calls) == 1
    assert not any((out / name).exists() for name in ("body_seed", "proposal", "person-inventory.json"))
    assert (out / "support-priors.json").read_bytes() == raw
    settings = json.loads((out / "settings.json").read_bytes())
    assert settings["scene_scope"] == "human_static_supports" and settings["full_reference"] is True
    assert settings["support_priors_sha256"] == sha(raw) and "inventory" not in settings
    assert settings["stage_output_reservations"]["support_priors"] == {"files": 2, "bytes": 256 << 10}
    resolved = json.loads((out / "support-config.json").read_bytes())
    assert resolved["profile"] == shared_world_support.CONFIG_PROFILE
    assert {k: v for k, v in resolved.items() if k not in {"profile", "body_source_manifest_sha256"}} == {
        k: v for k, v in json.loads(raw).items() if k != "profile"}
    assert any(row["stage"] == "support_priors" and row["status"] == "ready" for row in result["stages"])
    manifest = json.loads((out / "manifest.json").read_bytes())
    assert all(sha((out / name).read_bytes()) == expected for name, expected in manifest["files"].items())


def test_support_lab_failure_keeps_body_candidate_and_assumptions(tmp_path, monkeypatch):
    _, options = setup_build(tmp_path, monkeypatch, "raw", fail_lab=True)
    out = tmp_path / "build"
    with pytest.raises(ValueError, match="geometry remains"):
        pipeline.prepare_exercise_scene(None, None, None, out, **options)
    result = json.loads((out / "result.json").read_bytes())
    assert result["status"] == "failed" and result["lab_status"] == "not_ready"
    assert (out / "body_source/manifest.json").is_file() and (out / "candidate/manifest.json").is_file()
    assert (out / "support-priors.json").read_bytes() == options["static_supports"].read_bytes()


@pytest.mark.parametrize("target", ["prior", "resolved", "candidate", "source"])
def test_progress_cannot_replace_pinned_support_inputs_or_outputs(tmp_path, monkeypatch, target):
    _, options = setup_build(tmp_path, monkeypatch, "saved")
    out = tmp_path / "build"
    def progress(row):
        if row["stage"] == "support_priors" and row["status"] == "ready":
            path = {"prior": out / "support-priors.json", "resolved": out / "support-config.json",
                    "source": options["human_run"] / "manifest.json"}.get(target)
            if path is not None:
                path.write_bytes(path.read_bytes() + b" ")
        if target == "candidate" and row["stage"] == "shared_world" and row["status"] == "ready":
            path = out / "candidate/data.json"
            path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        pipeline.prepare_exercise_scene(None, None, None, out, progress=progress, **options)
    result = json.loads((out / "result.json").read_bytes())
    assert result["status"] == "failed"
    assert not (out / "lab").exists()


@pytest.mark.parametrize("overrides", [{"body_scale": 1.}, {"root_assistance": True},
    {"train_settings": {"evaluation_seed": 7342, "policy_family": "feedback-residual"}}])
def test_conflicting_support_options_stop_before_output(tmp_path, monkeypatch, overrides):
    _, options = setup_build(tmp_path, monkeypatch, "saved")
    options.update(overrides)
    out = tmp_path / "build"
    with pytest.raises(ValueError):
        pipeline.prepare_exercise_scene(None, None, None, out, **options)
    assert not out.exists()


def test_cli_forwards_explicit_support_scope_without_absence_claim(tmp_path, monkeypatch, capsys):
    observed = {}
    def prepare(*args, **kwargs):
        observed.update(kwargs)
        return {"status": "ready", "lab_status": "ready"}
    monkeypatch.setattr(pipeline, "prepare_exercise_scene", prepare)
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-build", "--human-run", "body",
        "--static-supports", "supports.json", "--out", str(tmp_path / "build")])
    assert cli.main() in (None, 0)
    assert observed["scene_scope"] == "human_static_supports" and observed["static_supports"] == "supports.json"
    assert "inventory" not in observed and observed["train_settings"] is None
