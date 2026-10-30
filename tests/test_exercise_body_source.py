"""Analytical pose fixtures test custody and construction, not physical reconstruction."""

import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image
import pytest

from test_exercise_vision import inference_result, stub_detector, video as write_video
from weft import exercise_body_source as source
from weft import exercise_vision, human, shared_world_lab
from weft.io import digest, write_json
from weft.pipeline import verify_artifacts


VIDEO = b"Analytical source fixture, not decoded video or physical evidence.\n"


def model_metadata(mode="video"):
    return {"name": "MediaPipe Pose Landmarker Lite", "model_version": "float16/1",
            **exercise_vision.MODEL_VARIANTS["lite"], "delegate": "CPU", "network_inference": False,
            "maximum_pose_hypotheses": 2, "mediapipe_version": "analytical-fixture",
            "running_mode": mode.upper(), "temporal_tracking": mode == "video"}


def write_pose_files(out, *, mode="video", identity="raw", model=None, dropped=(0, 2, 30)):
    source_sha = digest(out / "source.webm")
    evidence = {"schema_version": 1, "profile": "exercise-pose-v1", "source_sha256": source_sha,
                "width": 640, "height": 480, "fps": 30., "frame_count": 63, "decoded_frames": 63,
                "inference_stride": 2, "sample_rate_fps": 15., "model": model or model_metadata(mode),
                "assumptions": exercise_vision.ASSUMPTIONS[:], "scope": "Analytical fixture only",
                "timestamp_basis": "Frame index divided by declared constant fps; per-frame decoder timestamps also preserved",
                "frames": []}
    for index in range(0, 63, 2):
        row = exercise_vision.summarize_result(inference_result(index * .005, count=0 if index in dropped else 1))
        row.update(id=f"{source_sha}:frame:{index}", frame=index, time_s=index / 30,
                   decoder_timestamp_s=None if index == 14 else index / 30 + .003,
                   inference_seconds=0.)
        evidence["frames"].append(row)
    write_json(out / "pose-landmarks.json", {"frames": evidence["frames"], "model": evidence["model"],
                                             "scope": evidence["scope"]})
    exercise_vision.resolve_pose_identity(evidence, out, identity)
    return exercise_vision.finalize_evidence(evidence, out)


def seal(root, **extra):
    write_json(root / "manifest.json", {"schema_version": 1, **extra,
               "files": {path.relative_to(root).as_posix(): digest(path)
                         for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"}})


def pose_run(root, *, mode="video", identity="raw"):
    root.mkdir()
    (root / "source.webm").write_bytes(VIDEO)
    write_json(root / "source.json", {"artifact": "source.webm", "sha256": digest(root / "source.webm"),
                                      "rights": "Analytical fixture, no real person"})
    write_json(root / "run-config.json", {"profile": "exercise", "body_model": "hinge",
                                          "pose_mode": mode, "pose_identity": identity})
    evidence = write_pose_files(root, mode=mode, identity=identity)
    write_json(root / "model.json", evidence["model"])
    (root / "unrelated.txt").write_text("The complete input seal is checked, not only copied evidence.\n")
    seal(root)
    return root


def forbid_runtime(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Construction body must not decode/infer, train, advance physics or render simulation")

    monkeypatch.setattr(exercise_vision, "inspect_exercise", forbidden)
    monkeypatch.setattr(exercise_vision.cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(human, "HumanEnv", forbidden)
    from weft import human_learning, recording
    monkeypatch.setattr(human_learning, "train_human", forbidden)
    monkeypatch.setattr(recording, "render_snapshot", forbidden)


def test_saved_pose_constructs_only_body_with_exact_unknown_slots_and_clocks(tmp_path, monkeypatch):
    original = pose_run(tmp_path / "poses", identity="temporal")
    before = {path: path.read_bytes() for path in original.rglob("*") if path.is_file()}
    forward, calls = mujoco.mj_forward, []
    monkeypatch.setattr(mujoco, "mj_forward", lambda *args: calls.append(True) or forward(*args))
    forbid_runtime(monkeypatch)
    out = tmp_path / "body"
    assert source.prepare_exercise_body_source_from_evidence(original, out) == out
    assert len(calls) == 1
    assert all(path.read_bytes() == raw for path, raw in before.items())
    for name in source.POSE_FILES | {"source.webm", "joint-angles.json"}:
        assert (out / name).read_bytes() == before[original / name]
    marker = json.loads((out / "body-source.json").read_text())
    assert marker["profile"] == source.PROFILE and marker["status"] == "construction_only"
    assert marker["simulation_ready"] is False
    assert marker["pose_options"] == {"pose_mode": "video", "pose_identity": "temporal"}
    assert json.loads((out / "run-config.json").read_text())["body_model"] == "hinge"
    assert marker["construction"]["body_model"] == "mobile"
    assert marker["construction"]["original_pose_config_sha256"] == digest(original / "run-config.json")
    assert marker["work"] == {"video_decodes": 0, "inference_calls": 0, "floor_forward_calls": 1,
                               "physics_steps": 0, "training_episodes": 0, "simulation_renders": 0,
                               "api_calls": 0, "observation_media": "saved bytes reused"}
    observations = json.loads((out / "observations.json").read_text())
    assert len(observations["observations"]) == 63
    assert observations["relationships"] == []
    assert observations["observations"][30]["visibility"] == "not_detected"
    assert observations["observations"][31]["visibility"] == "not_sampled"
    clocks = json.loads((out / "source-clock.json").read_text())
    assert clocks["samples"][7] == {"frame": 14, "time_s": 14 / 30, "decoder_timestamp_s": None}
    assert clocks["samples"][0]["decoder_timestamp_s"] == .003
    assert not any((out / name).exists() for name in ("scene.json", "task.json", "lab-config.json",
                                                     "interventions.json", "policy.json", "reconstruction.gif"))
    files = {path.relative_to(out).as_posix(): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    body, _ = source.validate_body_source_files(files)
    assert body["profile"] == "exercise-v3" and body["reference"][0]["time_s"] == 4 / 30
    assert verify_artifacts(out)["verified_files"] == len(files) - 1
    assert sum(map(len, files.values())) < source.MAX_BYTES


@pytest.mark.parametrize("mode,identity", [("video", "raw"), ("image", "temporal")])
def test_raw_input_uses_explicit_options_and_one_local_inference_stage(tmp_path, monkeypatch, mode, identity):
    video, model = tmp_path / "input.webm", tmp_path / "model.task"
    video.write_bytes(VIDEO); model.write_bytes(b"analytical model bytes")
    entry = {"sha256": digest(model), "bytes": model.stat().st_size}
    monkeypatch.setattr(exercise_vision, "MODEL_VARIANTS", {"lite": entry})
    monkeypatch.setattr(exercise_vision, "model_provenance", lambda path: entry)
    calls = []

    def inspect(path, out, model_path, progress, artifact_sink, **options):
        calls.append((Path(path).read_bytes(), model_path, options))
        progress("tracking", {"sampled_frames": 1})
        return write_pose_files(out, mode=options["pose_mode"], identity=options["pose_identity"])

    monkeypatch.setattr(exercise_vision, "inspect_exercise", inspect)
    out = tmp_path / "body"
    kwargs = {} if identity == "raw" else {"pose_mode": mode, "pose_identity": identity}
    source.prepare_exercise_body_source(video, out, pose_model=model, **kwargs)
    assert calls == [(VIDEO, model, {"pose_mode": mode, "pose_identity": identity})]
    marker = json.loads((out / "body-source.json").read_text())
    assert marker["work"]["video_decodes"] == 1 and marker["work"]["inference_calls"] == 32
    assert marker["pose_options"]["pose_identity"] == identity
    assert (out / "source.webm").read_bytes() == video.read_bytes()


def test_new_profile_feeds_seed_without_fabricating_legacy_scene(tmp_path, monkeypatch):
    original = pose_run(tmp_path / "poses")
    out, seed = tmp_path / "body", tmp_path / "seed"
    source.prepare_exercise_body_source_from_evidence(original, out)
    forbid_runtime(monkeypatch)
    monkeypatch.setattr(shared_world_lab, "SharedWorldEnv", lambda *args: pytest.fail("No intermediate simulator"))
    shared_world_lab.prepare_shared_world_seed(out, seed)
    assert not (out / "scene.json").exists()
    assert (seed / "human.json").read_bytes() == (out / "human.json").read_bytes()
    receipt = json.loads((seed / "construction-seed.json").read_text())
    assert receipt["input_profile"] == source.PROFILE
    assert receipt["body_source_validator_sha256"] == digest(Path(source.__file__))
    assert (seed / "implementation/exercise_body_source.py").read_bytes() == Path(source.__file__).read_bytes()
    assert receipt["source_binding"]["total_source_frames"] == 63
    assert receipt["clock_binding"]["source_time_offset_s"] == 4 / 30
    assert (seed / "inputs/human-source/body-source.json").read_bytes() == (out / "body-source.json").read_bytes()
    assert not (seed / "inputs/human-source/scene.json").exists()
    assert verify_artifacts(seed)["verified_files"] > 10


@pytest.mark.parametrize("change", ["source", "model", "mode", "identity", "native", "slots", "quality", "raw_hash"])
def test_saved_identity_conflicts_fail_without_a_successful_package(tmp_path, monkeypatch, change):
    original = pose_run(tmp_path / "poses")
    evidence = json.loads((original / "evidence.json").read_text())
    if change == "source": evidence["source_sha256"] = "0" * 64
    elif change == "model": evidence["model"]["sha256"] = "0" * 64
    elif change == "mode":
        write_json(original / "run-config.json", {"profile": "exercise", "pose_mode": "image", "pose_identity": "raw"})
    elif change == "identity": evidence["pose_identity"]["mode"] = "temporal"
    elif change == "native": evidence["frames"][5]["decoder_timestamp_s"] += .002
    elif change == "slots": evidence["frames"].pop(5)
    elif change == "quality": evidence["quality"].update(passes=False, errors=["Original source rejected"])
    else: evidence["pose_identity"]["raw_artifact_sha256"] = "0" * 64
    write_json(original / "evidence.json", evidence); seal(original)
    forbid_runtime(monkeypatch)
    out = tmp_path / "body"
    with pytest.raises(ValueError):
        source.prepare_exercise_body_source_from_evidence(original, out)
    assert json.loads((out / "failure.json").read_text())["status"] == "failed"
    assert not (out / "body-source.json").exists()
    assert json.loads((out / "manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("status_target", ["manifest", "lifecycle", "report"])
def test_rejected_source_status_is_not_laundered(tmp_path, status_target):
    lab = tmp_path / "lab"; lab.mkdir()
    original = pose_run(lab / "run")
    if status_target == "manifest": seal(original, status="failed")
    elif status_target == "lifecycle": write_json(lab / "status.json", {"status": "failed"})
    else:
        write_json(original / "report.json", {"status": "rejected"}); seal(original)
    with pytest.raises(ValueError):
        source.prepare_exercise_body_source_from_evidence(lab, tmp_path / "body")


def test_original_uncopied_input_is_rechecked_after_body_preparation(tmp_path):
    original = pose_run(tmp_path / "poses")

    def mutate(stage, detail):
        (original / "unrelated.txt").write_text("Changed during construction")

    with pytest.raises(ValueError, match="SHA256"):
        source.prepare_exercise_body_source_from_evidence(original, tmp_path / "body", progress=mutate)
    assert json.loads((tmp_path / "body/manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("kind", ["input", "component", "output", "traversal"])
def test_symlink_or_traversal_paths_are_rejected(tmp_path, kind):
    original = pose_run(tmp_path / "poses")
    selected, out = original, tmp_path / "body"
    if kind == "input":
        selected = tmp_path / "linked"; selected.symlink_to(original, target_is_directory=True)
    elif kind == "component":
        parent = tmp_path / "linked"; parent.symlink_to(tmp_path, target_is_directory=True); selected = parent / "poses"
    elif kind == "output":
        parent = tmp_path / "linked"; parent.symlink_to(tmp_path, target_is_directory=True); out = parent / "body"
    else: selected = original / ".." / "poses"
    with pytest.raises(ValueError): source.prepare_exercise_body_source_from_evidence(selected, out)


@pytest.mark.parametrize("limit", ["bytes", "files", "deadline"])
def test_stage_bounds_leave_inspectable_failure_not_a_ready_source(tmp_path, monkeypatch, limit):
    original = pose_run(tmp_path / "poses")
    if limit == "bytes": monkeypatch.setattr(source, "MAX_BYTES", 1024)
    elif limit == "files": monkeypatch.setattr(source, "MAX_FILES", 4)
    else:
        counter = iter([0., 0., 200.] + [200.] * 100)
        monkeypatch.setattr(source.time, "monotonic", lambda: next(counter))
    with pytest.raises((ValueError, TimeoutError)):
        source.prepare_exercise_body_source_from_evidence(original, tmp_path / "body")
    assert (tmp_path / "body/failure.json").is_file()
    assert not (tmp_path / "body/body-source.json").exists()


@pytest.mark.parametrize("seconds", [0, 181, True, float("nan")])
def test_invalid_deadline_rejects_before_creating_output(tmp_path, seconds):
    with pytest.raises(ValueError):
        source.prepare_exercise_body_source_from_evidence(tmp_path / "missing", tmp_path / "body", max_seconds=seconds)
    assert not (tmp_path / "body").exists()


def test_oversized_raw_input_rejects_before_inference_or_output(tmp_path):
    video, model = tmp_path / "source.webm", tmp_path / "model.task"
    with video.open("wb") as stream: stream.truncate(source.MAX_FILE_BYTES + 1)
    model.write_bytes(b"unused")
    with pytest.raises(ValueError, match="64-MiB"):
        source.prepare_exercise_body_source(video, tmp_path / "body", pose_model=model)
    assert not (tmp_path / "body").exists()


def test_partial_raw_failure_preserves_original_pose_artifacts(tmp_path, monkeypatch):
    from weft.errors import UnsupportedVideo

    video, model = tmp_path / "input.webm", tmp_path / "model.task"
    video.write_bytes(VIDEO); model.write_bytes(b"model fixture")
    entry = {"sha256": digest(model), "bytes": model.stat().st_size}
    monkeypatch.setattr(exercise_vision, "model_provenance", lambda path: entry)

    def rejected(path, out, model_path, **kwargs):
        write_json(out / "pose-landmarks.json", {"frames": [], "failure": "competing poses"})
        raise UnsupportedVideo("Competing poses remain unresolved")

    monkeypatch.setattr(exercise_vision, "inspect_exercise", rejected)
    out = tmp_path / "body"
    with pytest.raises(UnsupportedVideo): source.prepare_exercise_body_source(video, out, pose_model=model)
    assert (out / "source.webm").read_bytes() == VIDEO
    assert json.loads((out / "pose-landmarks.json").read_text())["failure"] == "competing poses"
    assert json.loads((out / "manifest.json").read_text())["status"] == "failed"
    assert not (out / "human.json").exists()


def test_raw_stage_rejects_a_returned_model_different_from_pinned_input(tmp_path, monkeypatch):
    video, model = tmp_path / "input.webm", tmp_path / "model.task"
    video.write_bytes(VIDEO); model.write_bytes(b"model fixture")
    entry = {"sha256": digest(model), "bytes": model.stat().st_size}
    monkeypatch.setattr(exercise_vision, "model_provenance", lambda path: entry)
    monkeypatch.setattr(exercise_vision, "inspect_exercise", lambda *args, **kwargs: {"model": {"sha256": "0" * 64}})
    with pytest.raises(ValueError, match="different model"):
        source.prepare_exercise_body_source(video, tmp_path / "body", pose_model=model)


def test_saved_quality_flags_cannot_override_current_source_quality_gates(tmp_path):
    original = pose_run(tmp_path / "poses")
    evidence = json.loads((original / "evidence.json").read_text())
    raw = json.loads((original / "pose-landmarks.json").read_text())
    for row in evidence["frames"]:
        if row["status"] in exercise_vision.USABLE_STATUSES:
            row["joints"] = dict.fromkeys(exercise_vision.JOINT_NAMES, 0.)
    raw["frames"] = evidence["frames"]
    write_json(original / "pose-landmarks.json", raw)
    raw_sha = digest(original / "pose-landmarks.json")
    evidence["pose_identity"]["raw_artifact_sha256"] = raw_sha
    audit = json.loads((original / "identity-audit.json").read_text()); audit["raw_artifact_sha256"] = raw_sha
    write_json(original / "identity-audit.json", audit)
    write_json(original / "evidence.json", evidence); seal(original)
    with pytest.raises(ValueError, match="Rejected/unsupported"):
        source.prepare_exercise_body_source_from_evidence(original, tmp_path / "body")


@pytest.mark.parametrize("change", ["marker", "observations", "clock", "human_source", "new_status"])
def test_new_seed_profile_rejects_resealed_conflicting_source_contract(tmp_path, change):
    original = pose_run(tmp_path / "poses")
    out = tmp_path / "body"
    source.prepare_exercise_body_source_from_evidence(original, out)
    marker = json.loads((out / "body-source.json").read_text())
    if change == "marker": marker["profile"] = "unknown-body-v1"
    elif change == "new_status": marker["simulation_ready"] = True
    else:
        name = {"observations": "observations.json", "clock": "source-clock.json", "human_source": "human.json"}[change]
        value = json.loads((out / name).read_text())
        if change == "observations": value["observations"][31]["visibility"] = "not_detected"
        elif change == "clock": value["samples"][0]["decoder_timestamp_s"] = 0.
        else: value["source_sha256"] = "0" * 64
        write_json(out / name, value)
        marker["artifacts"][name] = digest(out / name)
    write_json(out / "body-source.json", marker)
    seal(out, profile=source.PROFILE, status="construction_only", simulation_ready=False)
    with pytest.raises(ValueError): shared_world_lab.prepare_shared_world_seed(out, tmp_path / "seed")
    assert not (tmp_path / "seed").exists()


def test_current_body_source_validator_change_prevents_seed_seal(tmp_path, monkeypatch):
    original = pose_run(tmp_path / "poses")
    out = tmp_path / "body"
    source.prepare_exercise_body_source_from_evidence(original, out)
    read, calls = shared_world_lab.read_package_file, []

    def changing(directory, name, **kwargs):
        raw = read(directory, name, **kwargs)
        if name == "exercise_body_source.py":
            calls.append(True)
            if len(calls) > 1:
                return raw + b"# changed validator\n"
        return raw

    monkeypatch.setattr(shared_world_lab, "read_package_file", changing)
    with pytest.raises(ValueError, match="admission validator changed"):
        shared_world_lab.prepare_shared_world_seed(out, tmp_path / "seed")
    assert not (tmp_path / "seed/manifest.json").exists()


def test_terminal_manifest_readback_failure_cannot_be_admitted(tmp_path, monkeypatch):
    original = pose_run(tmp_path / "poses")
    out = tmp_path / "body"
    write = source.write_package_files

    def fail_after_manifest(directory, files):
        write(directory, files)
        if "manifest.json" in files:
            raise OSError("Injected post-write manifest readback failure")

    monkeypatch.setattr(source, "write_package_files", fail_after_manifest)
    with pytest.raises(OSError, match="readback failure"):
        source.prepare_exercise_body_source_from_evidence(original, out)
    assert (out / "failure.json").exists()
    assert json.loads((out / "manifest.json").read_text())["status"] == "construction_only"
    with pytest.raises(ValueError, match="failure marker"):
        shared_world_lab.prepare_shared_world_seed(out, tmp_path / "seed")
    with pytest.raises(ValueError, match="unrejected"):
        source.prepare_exercise_body_source_from_evidence(out, tmp_path / "reused")
    assert not (tmp_path / "seed").exists()


def test_sink_replacement_reserves_final_size_and_cannot_overwrite_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "MAX_BYTES", 4096)
    monkeypatch.setattr(source, "MAX_FILE_BYTES", 2048)
    monkeypatch.setattr(source, "RESERVE_BYTES", 1024)
    (tmp_path / "source.webm").write_bytes(VIDEO)
    sink = source.BodyArtifactSink(tmp_path, deadline=time.monotonic() + 5)
    sink.write_json("evidence.json", {"value": "a" * 800})
    sink.write_json("evidence.json", {"value": "b" * 1800})
    preserved = (tmp_path / "evidence.json").read_bytes()
    with pytest.raises(ValueError, match="reserved output capacity"):
        sink.write_json("evidence.json", {"value": "c" * 2050})
    assert (tmp_path / "evidence.json").read_bytes() == preserved
    with pytest.raises(ValueError, match="reserved output capacity"):
        sink.write_text("other.txt", "x" * 1400)
    with pytest.raises(ValueError, match="preserved input"):
        sink.write_text("source.webm", "changed")
    assert not (tmp_path / "other.txt").exists()
    assert sum(path.stat().st_size for path in tmp_path.iterdir()) < source.MAX_BYTES - source.RESERVE_BYTES


def test_sink_bounds_preview_encoding_before_any_oversized_file(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "MAX_FILE_BYTES", 16 * 1024)
    image = Image.fromarray(np.random.default_rng(219).integers(0, 256, (256, 256, 3), dtype=np.uint8))
    sink = source.BodyArtifactSink(tmp_path, deadline=time.monotonic() + 5)
    with pytest.raises(ValueError, match="reserved output capacity"):
        sink.write_image("preview.png", image)
    assert not (tmp_path / "preview.png").exists()


def test_observation_previews_keep_legacy_bytes_with_optional_sink(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(); new.mkdir()
    images = [Image.new("RGB", (160, 120), color) for color in ("red", "blue", "green")]
    exercise_vision._save_previews(images, [0., .1, .2], old)
    exercise_vision._save_previews(images, [0., .1, .2], new,
        artifact_sink=source.BodyArtifactSink(new, deadline=time.monotonic() + 5))
    for name in ("tracks.gif", "tracks-contact-sheet.png"):
        assert (old / name).read_bytes() == (new / name).read_bytes()


def test_mobile_body_default_and_bounded_writer_outputs_are_byte_identical(tmp_path, monkeypatch):
    original = pose_run(tmp_path / "poses")
    evidence = json.loads((original / "evidence.json").read_text())
    old, new = tmp_path / "old", tmp_path / "new"
    new.mkdir()
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("No dynamics"))
    old_body = human.build_human(evidence, old, body_model="mobile")
    new_body = human.build_human(evidence, new, body_model="mobile",
        artifact_sink=source.BodyArtifactSink(new, deadline=time.monotonic() + 5))
    assert old_body == new_body
    assert {path.name for path in old.iterdir()} == {path.name for path in new.iterdir()}
    assert all(path.read_bytes() == (new / path.name).read_bytes() for path in old.iterdir())


def test_actual_pose_producer_growth_cannot_exceed_retained_output_bound(tmp_path, monkeypatch):
    video, model = tmp_path / "input.mp4", tmp_path / "model.task"
    write_video(video); model.write_bytes(b"model fixture")
    detector = stub_detector(monkeypatch, lambda index: inference_result(index * .04))
    monkeypatch.setattr(exercise_vision, "model_provenance", lambda path: {"sha256": digest(model), "bytes": model.stat().st_size})
    monkeypatch.setattr(source, "_implementation", lambda deadline: ({}, {"files": {}, "runtime_versions": {}}))
    monkeypatch.setattr(source, "MAX_BYTES", 32 * 1024)
    monkeypatch.setattr(source, "RESERVE_BYTES", 8 * 1024)
    out = tmp_path / "body"
    with pytest.raises(ValueError, match="reserved output capacity"):
        source.prepare_exercise_body_source(video, out, pose_model=model)
    assert detector.closed
    assert sum(path.stat().st_size for path in out.rglob("*") if path.is_file()) <= source.MAX_BYTES
    assert json.loads((out / "failure.json").read_text())["status"] == "failed"
    assert not (out / "human.json").exists()


def test_actual_body_producer_growth_preserves_failure_metadata_reserve(tmp_path, monkeypatch):
    original = pose_run(tmp_path / "poses")
    out = tmp_path / "body"

    def restrict(stage, detail):
        if stage == "body":
            used = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
            monkeypatch.setattr(source, "MAX_BYTES", used + source.RESERVE_BYTES + 1024)

    with pytest.raises(ValueError, match="reserved output capacity"):
        source.prepare_exercise_body_source_from_evidence(original, out, progress=restrict)
    assert (out / "failure.json").exists()
    assert json.loads((out / "manifest.json").read_text())["status"] == "failed"
    assert sum(path.stat().st_size for path in out.rglob("*") if path.is_file()) <= source.MAX_BYTES


def test_late_unlisted_failure_marker_stops_saved_body_publication(tmp_path):
    original = pose_run(tmp_path / "poses")

    def fail_source(stage, detail):
        write_json(original / "failure.json", {"status": "failed", "reason": "late source failure"})

    out = tmp_path / "body"
    with pytest.raises(ValueError, match="gained a failure marker"):
        source.prepare_exercise_body_source_from_evidence(original, out, progress=fail_source)
    assert json.loads((out / "manifest.json").read_text())["status"] == "failed"


def test_late_unlisted_failure_marker_stops_new_profile_seed(tmp_path, monkeypatch):
    original = pose_run(tmp_path / "poses")
    body, seed = tmp_path / "body", tmp_path / "seed"
    source.prepare_exercise_body_source_from_evidence(original, body)
    prepare = shared_world_lab._adapted_shared_world

    def fail_source(*args, **kwargs):
        result = prepare(*args, **kwargs)
        write_json(body / "failure.json", {"status": "failed", "reason": "late source failure"})
        return result

    monkeypatch.setattr(shared_world_lab, "_adapted_shared_world", fail_source)
    with pytest.raises(ValueError, match="gained a failure marker"):
        shared_world_lab.prepare_shared_world_seed(body, seed)
    assert not (seed / "manifest.json").exists()
