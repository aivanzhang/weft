"""Analytical detached native views; real source custody is separately qualified without decoding."""
from copy import deepcopy
import builtins
import importlib
import json
from pathlib import Path
import time

import pytest

from test_native_clock_runtime import native_binding
from test_whole_body_reference_inputs import make_body_source, encode, sha, mutate, prepare
from weft import native_reference_inputs as native
from weft import whole_body_reference_inputs as legacy
from weft.source_clock import derive_evidence, clock_summary


def views(tmp_path):
    root = make_body_source(tmp_path / "body")
    human = json.loads((root / "human.json").read_bytes())
    original = json.loads((root / "evidence.json").read_bytes())
    ticks = [2000 + 30 * i + 10 * (i // 3) for i in range(10)]
    for row in original["frames"]:
        row["decoder_timestamp_s"] = ticks[row["frame"]] / 1000 - 1.99
    binding = native_binding(original, ticks)
    derived = derive_evidence(original, binding)
    human["native_clock"] = clock_summary(binding)
    for index, row in enumerate(derived["frames"]):
        for key in ("reference", "reference_evidence", "root_reference"):
            human[key][index]["time_s"] = row["time_s"]
    producer = sha((Path(native.__file__).parent / "exercise_vision.py").read_bytes())
    h, o, d = map(encode, (human, original, derived))
    dim = {"schema_version": 1, "profile": native.PROFILE, "body_manifest_sha256": "a" * 64,
        "source_sha256": original["source_sha256"], "source_artifact": "source.webm",
        "human_sha256": sha(h), "original_evidence_sha256": sha(o), "derived_evidence_sha256": sha(d),
        "width": original["width"], "height": original["height"], "decoded_frames": original["decoded_frames"],
        "metadata_frames": original["frame_count"], "decoder_producer_sha256": producer,
        "decoder_profile": legacy._DECODERS[producer], "producer_manifest_member": legacy._PRODUCER,
        "body_constructor_decoder_sha256": producer, "original_decoder_attachment": None,
        "clock": clock_summary(binding), "capture_artifacts": {},
        "pose_options": {"pose_mode": "video", "pose_identity": "raw"},
        "coordinate_scope": native.COORDINATE_SCOPE, "physical_validation": "unverified"}
    return h, o, d, encode(dim)


def test_native_view_preserves_exact_native_and_original_bytes(tmp_path):
    args = views(tmp_path)
    before = tuple(args)
    human, derived, dimensions = native.validate_native_reference_view(*args)
    assert args == before
    assert [r["time_s"] for r in derived["frames"]] != [r["time_s"] for r in json.loads(args[1])["frames"]]
    assert human["native_clock"] == dimensions["clock"]
    assert "image_dimension_provenance" not in derived
    derived["frames"][0]["time_s"] = 99
    assert native.validate_native_reference_view(*args)[1]["frames"][0]["time_s"] == 0


@pytest.mark.parametrize("field,value", [
    ("width", 1199), ("width", True), ("metadata_frames", 11),
    ("body_manifest_sha256", "unknown"), ("source_artifact", "../source.webm"),
    ("decoder_producer_sha256", "f" * 64), ("physical_validation", "verified"),
    ("capture_artifacts", {"decoder-clock.json": "a" * 64}),
    ("pose_options", {"pose_mode": "video", "pose_identity": "raw", "max_pose_hypotheses": True}),
])
def test_native_view_rejects_malformed_sidecar(tmp_path, field, value):
    h, o, d, dim = views(tmp_path)
    value_dim = json.loads(dim); value_dim[field] = value
    with pytest.raises(ValueError):
        native.validate_native_reference_view(h, o, d, encode(value_dim))


@pytest.mark.parametrize("kind", ["landmark", "time", "original", "dropped_reference", "dimension_injection"])
def test_native_view_rejects_coordinated_byte_hash_tampering(tmp_path, kind):
    h, o, d, dim = views(tmp_path)
    human, original, derived, dimensions = map(json.loads, (h, o, d, dim))
    if kind == "landmark": derived["frames"][1]["landmarks"][11]["x"] += .01
    elif kind == "time": derived["frames"][1]["time_s"] += .001
    elif kind == "original": original["frames"][1]["landmarks"][11]["x"] += .01
    elif kind == "dropped_reference": human["reference"].pop()
    else: derived["image_dimension_provenance"] = {"width": 1200, "height": 600}
    h, o, d = map(encode, (human, original, derived))
    dimensions.update(human_sha256=sha(h), original_evidence_sha256=sha(o), derived_evidence_sha256=sha(d))
    with pytest.raises(ValueError):
        native.validate_native_reference_view(h, o, d, encode(dimensions))


def test_detached_worker_import_and_validation_need_no_decoder_or_physics(tmp_path, monkeypatch):
    args = views(tmp_path)
    original_import = builtins.__import__
    def guarded(name, *arguments, **kwargs):
        assert name.split(".")[0] not in {"numpy", "mujoco", "cv2", "mediapipe", "torch", "socket", "urllib"}
        return original_import(name, *arguments, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    importlib.reload(native)
    native.validate_native_reference_view(*args)
    assert set(native.VIEW_RUNTIME_FILES) == {"native_reference_inputs.py", "whole_body_reference_inputs.py", "source_clock.py", "io.py"}


@pytest.mark.parametrize("hidden", ["region", "single", "region_raw"])
def test_new_reviewed_producer_does_not_admit_hidden_legacy_routes(tmp_path, hidden):
    root = make_body_source(tmp_path / "body")
    if hidden == "region":
        mutate(root, "run-config.json", lambda value: value.update(person_region={"x": 0}))
    elif hidden == "single":
        mutate(root, "run-config.json", lambda value: value.update(max_pose_hypotheses=1))
    else:
        mutate(root, "pose-landmarks.json", lambda value: value["frames"][0].update(region_raw={}))
    with pytest.raises(ValueError, match="whole-image two-hypothesis"):
        prepare(root)


def test_unsampled_resolution_change_rejects_current_reviewed_decoder(monkeypatch, tmp_path):
    from test_exercise_decoder_clock import setup
    from weft import exercise_vision as vision
    source, capture, detector = setup(monkeypatch, tmp_path)
    read = capture.read
    def changed():
        ok, frame = read()
        return ok, frame[:-1] if ok and capture.frame == 1 else frame
    monkeypatch.setattr(capture, "read", changed)
    with pytest.raises(ValueError, match="dimensions|resolution"):
        vision.inspect_exercise(source, tmp_path, "mocked-model")
    evidence = json.loads((tmp_path / "evidence.json").read_bytes())
    assert evidence["decoded_frames"] == 2 and len(evidence["frames"]) == 1
    assert "quality" not in evidence and capture.released and detector.closed


def sealed_native_envelope(tmp_path, monkeypatch, *, attachment=False):
    """Synthetic envelope tests stub the separately tested full body/pose validator, not native reversibility."""
    h, o, d, dim = views(tmp_path)
    root = tmp_path / "body"
    source = b"analytical source; no media execution"
    source_sha = sha(source)
    human, original = (json.loads(raw.replace(b'a' * 64, source_sha.encode())) for raw in (h, o))
    binding = native_binding(original, [2000 + 30 * i + 10 * (i // 3) for i in range(10)])
    derived = derive_evidence(original, binding)
    human["native_clock"] = clock_summary(binding)
    for name, value in (("human.json", human), ("evidence.json", original), ("reconstruction-evidence.json", derived),
                        ("native-clock.json", binding)):
        (root / name).write_bytes(encode(value))
    (root / "source.webm").write_bytes(source)
    marker = json.loads((root / "body-source.json").read_bytes())
    marker.update(source_sha256=source_sha)
    original_root = None
    if attachment:
        original_root = tmp_path / "original"
        original_root.mkdir()
        original_producer = b"# Separately pinned analytical original decoder fixture\n"
        digest = sha(original_producer)
        monkeypatch.setitem(legacy._DECODERS, digest, "analytical-decoder-for-attachment-tests")
        path = original_root / legacy._PRODUCER
        path.parent.mkdir(); path.write_bytes(original_producer)
        parent_files = {name: sha((root / name).read_bytes()) for name in (*legacy._POSE, "source.webm")}
        parent_files[legacy._PRODUCER] = digest
        parent = encode({"files": parent_files, "status": "source_evidence_only"})
        (original_root / "manifest.json").write_bytes(parent)
        (root / "inputs").mkdir()
        (root / "inputs/source-manifest.json").write_bytes(parent)
        marker.update(input_kind="sealed_pose_evidence", input_manifest_sha256=sha(parent))
    (root / "body-source.json").write_bytes(encode(marker))
    def seal_envelope():
        files = {path.relative_to(root).as_posix(): sha(path.read_bytes()) for path in root.rglob("*")
                 if path.is_file() and path.name != "manifest.json"}
        (root / "manifest.json").write_bytes(encode({"profile": "exercise-body-source-v1", "status": "construction_only",
                                                   "simulation_ready": False, "files": files}))
    seal_envelope()
    from weft import exercise_body_source
    calls = []
    def validate(files, *, deadline):
        calls.append(frozenset(files))
        return json.loads(files["human.json"]), {}
    monkeypatch.setattr(exercise_body_source, "validate_body_source_files", validate)
    return root, original_root, calls, seal_envelope


def test_prepare_keeps_immutable_original_native_bytes_and_calls_full_validator(tmp_path, monkeypatch):
    root, _, calls, _ = sealed_native_envelope(tmp_path, monkeypatch)
    before = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    packet = native.prepare_native_reference_inputs(root, deadline=time.monotonic() + 10)
    assert len(calls) == 1 and packet.original_evidence_bytes == before["evidence.json"]
    assert packet.derived_evidence_bytes == before["reconstruction-evidence.json"]
    assert dict(packet.files) == before
    with pytest.raises(TypeError): packet.files["x"] = b"new"
    detached = packet.binding; detached["rows"][0]["pts"] = 999
    assert packet.binding["rows"][0]["pts"] == 2000


def test_original_producer_attachment_is_explicit_and_keeps_body_untouched(tmp_path, monkeypatch):
    root, source, _, _ = sealed_native_envelope(tmp_path, monkeypatch, attachment=True)
    before = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="provide original_pose_source"):
        native.prepare_native_reference_inputs(root, deadline=time.monotonic() + 10)
    packet = native.prepare_native_reference_inputs(root, original_pose_source=source, deadline=time.monotonic() + 10)
    dim = packet.dimensions
    assert dim["decoder_producer_sha256"] != dim["body_constructor_decoder_sha256"]
    assert packet.files["reference-inputs/original-decoder.py"] == (source / legacy._PRODUCER).read_bytes()
    assert dim["original_decoder_attachment"]["manifest_sha256"] == sha((root / "inputs/source-manifest.json").read_bytes())
    assert before == {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("mutation", ["producer", "manifest", "symlink", "failure", "wrong_source_join"])
def test_attachment_rejects_malformed_or_wrong_source_proof(tmp_path, monkeypatch, mutation):
    root, source, _, seal = sealed_native_envelope(tmp_path, monkeypatch, attachment=True)
    if mutation == "producer": (source / legacy._PRODUCER).write_bytes(b"changed")
    elif mutation == "manifest": (source / "manifest.json").write_bytes(b"{}")
    elif mutation == "symlink":
        (source / legacy._PRODUCER).rename(source / "original.py")
        (source / legacy._PRODUCER).symlink_to(source / "original.py")
    elif mutation == "failure": (source / "failure.json").write_bytes(b"{}")
    else:
        parent = json.loads((source / "manifest.json").read_bytes()); parent["files"]["evidence.json"] = "f" * 64
        raw = encode(parent)
        (source / "manifest.json").write_bytes(raw)
        (root / "inputs/source-manifest.json").write_bytes(raw)
        marker = json.loads((root / "body-source.json").read_bytes()); marker["input_manifest_sha256"] = sha(raw)
        (root / "body-source.json").write_bytes(encode(marker)); seal()
    with pytest.raises((ValueError, OSError)):
        native.prepare_native_reference_inputs(root, original_pose_source=source, deadline=time.monotonic() + 10)


def test_original_attachment_reserves_aggregate_allowance_before_read(tmp_path, monkeypatch):
    root, source, _, _ = sealed_native_envelope(tmp_path, monkeypatch, attachment=True)
    monkeypatch.setattr(native, "MAX_INPUT_BYTES", 1)
    with pytest.raises(ValueError, match="aggregate input bound"):
        native.prepare_native_reference_inputs(root, original_pose_source=source, deadline=time.monotonic() + 10)


def test_changed_seal_during_full_validation_is_not_published(tmp_path, monkeypatch):
    root, _, _, _ = sealed_native_envelope(tmp_path, monkeypatch)
    from weft import exercise_body_source
    def changed(files, *, deadline):
        (root / "manifest.json").write_bytes(b"{}")
        return json.loads(files["human.json"]), {}
    monkeypatch.setattr(exercise_body_source, "validate_body_source_files", changed)
    with pytest.raises(ValueError, match="sealed SHA256"):
        native.prepare_native_reference_inputs(root, deadline=time.monotonic() + 10)
