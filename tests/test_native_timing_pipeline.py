"""Native timing propagation uses analytical stage substitutes, without inference or physics."""

import json
import sys

import pytest

from test_exercise_person_pipeline import setup_build
from test_exercise_scene_inputs import inputs
from weft import cli, exercise_scene_inputs as admission, exercise_scene_pipeline as scene
from weft.io import digest, write_json
from weft.portal_pipeline import execute_job


@pytest.mark.parametrize("enabled", [False, True])
def test_person_raw_pipeline_passes_native_capture_to_body_only(tmp_path, monkeypatch, enabled):
    calls, options = setup_build(tmp_path, monkeypatch, "raw")
    result = scene.prepare_exercise_scene(None, None, None, tmp_path / "build", native_timing=enabled, **options)
    assert result["status"] == "ready"
    body = next(options for name, options in calls if name == "body_source")
    assert body.get("native_timing", False) is enabled
    assert ("native_timing" in body) is enabled
    assert all("native_timing" not in options for name, options in calls if name != "body_source")
    settings = json.loads((tmp_path / "build/settings.json").read_bytes())
    assert ("native_timing" in settings) is enabled
    if enabled:
        assert settings["reconstruction_clock"] == "source-native-clock-v1"


@pytest.mark.parametrize("route", ["base", "human", "geometry", "crop"])
def test_native_capture_rejects_saved_or_recovery_routes_before_io(tmp_path, route):
    options = {"native_timing": True}
    base = None
    if route == "base": base = "missing"
    elif route == "human": options["human_run"] = "missing"
    elif route == "geometry": options["geometry_run"] = "missing"
    else: options.update(video="missing", pose_model="missing", pose_recovery="selective-crop")
    with pytest.raises(ValueError, match="Native timing"):
        scene.prepare_exercise_scene(base, None, None, tmp_path / "out", **options)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_native_capture_requires_boolean_before_io(tmp_path, value):
    with pytest.raises(ValueError, match="explicitly enabled"):
        scene.prepare_exercise_scene(None, None, None, tmp_path / "out", native_timing=value)
    assert not (tmp_path / "out").exists()


def test_body_adapter_rejects_recovery_even_when_called_without_pipeline(tmp_path, monkeypatch):
    video, _, _, model = inputs(tmp_path, monkeypatch)
    ready = admission.preflight_human_scene_inputs(video, pose_model=model)
    # Only the construction-mode conflict is exercised; no recovery producer is invoked.
    object.__setattr__(ready, "pose_recovery", "selective-crop")
    with pytest.raises(ValueError, match="selective-crop"):
        admission.prepare_exercise_scene_body(ready, tmp_path / "body", native_timing=True)
    assert not (tmp_path / "body").exists()


@pytest.mark.parametrize("route", ["raw", "saved", "geometry", "crop"])
def test_scene_cli_passes_native_capture_only_for_supported_raw_route(tmp_path, monkeypatch, route):
    captured = []
    monkeypatch.setattr(scene, "prepare_exercise_scene", lambda *a, **kw: captured.append(kw) or {"status": "ready"})
    argv = ["weft", "shared-exercise-build", "--native-timing", "--out", str(tmp_path / "out")]
    if route == "raw": argv += ["--video", "source.mp4", "--pose-model", "model.task", "--person-only"]
    elif route == "saved": argv += ["--human-run", "saved", "--person-only"]
    elif route == "geometry": argv += ["--geometry-run", "saved"]
    else: argv += ["--video", "source.mp4", "--pose-model", "model.task", "--person-only", "--pose-recovery", "selective-crop"]
    monkeypatch.setattr(sys, "argv", argv)
    if route == "raw":
        cli.main()
        assert captured[0]["native_timing"] is True
    else:
        with pytest.raises(SystemExit) as error: cli.main()
        assert error.value.code == 1 and not captured


def test_portal_raw_exercise_enables_native_capture_and_surfaces_clock_progress(tmp_path, monkeypatch):
    import weft.portal_pipeline as portal
    video = tmp_path / "source.webm"
    video.write_bytes(b"analytical source only")
    write_json(tmp_path / "job.json", {"id": "a" * 32, "source_file": video.name,
                                      "source_sha256": digest(video), "ai": False, "mode": "exercise"})
    def sample(source, out):
        out.mkdir()
        return {}
    monkeypatch.setattr(portal, "sample_video", sample)
    captured = []
    def build(*args, **options):
        captured.append(options)
        options["progress"]({"stage": "body_source_progress", "status": "native_clock",
                             "detail": {"message": "Verified native presentation timing"}})
        return {"status": "ready", "lab_directory": str(tmp_path / "build/lab")}
    monkeypatch.setattr(scene, "prepare_exercise_scene", build)
    assert execute_job(tmp_path, project=tmp_path) == 0
    assert captured[0]["native_timing"] is True
    assert captured[0]["scene_scope"] == "human_only"
    status = json.loads((tmp_path / "status.json").read_bytes())
    assert status["lab"] == "build/lab"
    assert any(row["message"] == "Verified native presentation timing" for row in status["events"])
