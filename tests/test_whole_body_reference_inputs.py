"""Synthetic sealed metadata tests; fixtures do not claim actual video acquisition."""

from copy import deepcopy
import builtins
import hashlib
import importlib
import json
from pathlib import Path
import time

import pytest

from weft import whole_body_reference_inputs as inputs


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def make_body_source(root, *, route="local_video", producer_name=inputs._PRODUCER, temporal=False):
    """Return an analytical package reusable by public-command tests, without native work."""
    from test_whole_body_reference_refinement import fixture
    human, evidence = fixture(5)
    evidence.pop("image_dimension_provenance")
    model = {"sha256": "b" * 64, "scope": "Synthetic metadata; no detector executed"}
    evidence.update(model=model, decoded_frames=10, inference_stride=2, quality={"passes": True, "errors": []})
    original = {"frames": deepcopy(evidence["frames"]), "model": model, "scope": "Analytical fixture"}
    original_raw = encode(original)
    repaired = []
    if temporal:
        frame = evidence["frames"][2]
        frame["identity_raw"] = deepcopy(frame)
        frame["identity_status"] = "temporally_reassigned_not_verified_anatomy"
        repaired.append(frame["id"])
    mode = "temporal" if temporal else "raw"
    evidence["pose_identity"] = {"mode": mode, "audit_artifact": "identity-audit.json",
                                  "raw_artifact_sha256": sha(original_raw), "repaired_frames": len(repaired)}
    source_sha = evidence["source_sha256"]
    config = {"profile": "exercise", "pose_mode": "video", "pose_identity": mode}
    decoder = (Path(inputs.__file__).parent / "exercise_vision.py").read_bytes()
    assert sha(decoder) in inputs._DECODERS, "Fixture needs a reviewed producer version"
    files = {
        "human.json": encode(human), "evidence.json": encode(evidence), "pose-landmarks.json": original_raw,
        "source.json": encode({"artifact": "source.webm", "sha256": source_sha}), "model.json": encode(model),
        "run-config.json": encode(config), "observations.json": encode({"scope": "Unconsumed analytical observations"}),
        "identity-audit.json": encode({"schema_version": 1, "mode": mode, "source_sha256": source_sha,
            "raw_artifact": "pose-landmarks.json", "raw_artifact_sha256": sha(original_raw), "repaired_evidence_ids": repaired}),
        "source-clock.json": encode({"schema_version": 1, "source_sha256": source_sha, "fps": 30,
            "metadata_frames": 10, "decoded_frames": 10, "nominal_basis": "frame_index_divided_by_fps",
            "samples": [{key: frame[key] for key in ("frame", "time_s", "decoder_timestamp_s")}
                        for frame in evidence["frames"]]}),
        "implementation.json": encode({"files": {"exercise_vision.py": sha(decoder)}}),
        inputs._PRODUCER: decoder,
    }
    source_manifest_sha = None
    if route == "sealed_pose_evidence":
        source_manifest = {"schema_version": 1, "status": "source_evidence_only",
                           "files": {name: sha(files[name]) for name in inputs._POSE}}
        source_manifest["files"].update({"source.webm": source_sha, producer_name: sha(decoder)})
        files["inputs/source-manifest.json"] = encode(source_manifest)
        source_manifest_sha = sha(files["inputs/source-manifest.json"])
    marker = {"schema_version": 1, "profile": inputs.BODY_PROFILE, "status": "construction_only",
        "simulation_ready": False, "source_artifact": "source.webm", "source_sha256": source_sha,
        "pose_options": {"pose_mode": "video", "pose_identity": mode}, "input_kind": route,
        "input_manifest_sha256": source_manifest_sha,
        "construction": {"body_model": "mobile", "body_profile": "exercise-v3",
            "original_pose_config": "run-config.json", "original_pose_config_sha256": sha(files["run-config.json"])},
        "artifacts": {name: sha(files[name]) for name in inputs._REQUIRED}}
    files["body-source.json"] = encode(marker)
    root.mkdir()
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    seal(root)
    return root


def seal(root):
    marker_path = root / "body-source.json"
    marker = json.loads(marker_path.read_bytes())
    marker["artifacts"] = {name: sha((root / name).read_bytes()) for name in inputs._REQUIRED}
    marker["construction"]["original_pose_config_sha256"] = marker["artifacts"]["run-config.json"]
    if (root / "inputs/source-manifest.json").exists():
        marker["input_manifest_sha256"] = sha((root / "inputs/source-manifest.json").read_bytes())
    marker_path.write_bytes(encode(marker))
    files = {path.relative_to(root).as_posix(): sha(path.read_bytes())
             for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"}
    files["source.webm"] = marker["source_sha256"]
    (root / "manifest.json").write_bytes(encode({"schema_version": 1, "profile": inputs.BODY_PROFILE,
        "status": "construction_only", "simulation_ready": False, "files": files}))


def mutate(root, name, change, *, reseal=True):
    path = root / name
    document = json.loads(path.read_bytes())
    change(document)
    path.write_bytes(encode(document))
    if reseal:
        seal(root)


def prepare(root, **kwargs):
    return inputs.prepare_reference_inputs(root, deadline=time.monotonic() + 10, **kwargs)


def test_import_and_reads_never_import_native_inference_or_network(monkeypatch, tmp_path):
    root = make_body_source(tmp_path / "body")
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"numpy", "mujoco", "gymnasium", "cv2", "mediapipe", "torch", "urllib", "socket"}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    importlib.reload(inputs)
    snapshot = prepare(root)
    inputs.recheck_reference_inputs(snapshot, deadline=time.monotonic() + 10)


@pytest.mark.parametrize("route,producer_name", [("local_video", inputs._PRODUCER),
    ("sealed_pose_evidence", inputs._PRODUCER), ("sealed_pose_evidence", "implementation-src--weft--exercise_vision.py")])
def test_standalone_routes_preserve_exact_bytes_without_media_or_siblings(tmp_path, route, producer_name):
    root = make_body_source(tmp_path / "body", route=route, producer_name=producer_name)
    original = (root / "evidence.json").read_bytes()
    snapshot = prepare(root, expected_manifest_sha256=sha((root / "manifest.json").read_bytes()))
    assert snapshot["files"]["inputs/evidence-original.json"] == original
    assert snapshot["files"]["inputs/human.json"] == (root / "human.json").read_bytes()
    derived = json.loads(snapshot["files"]["inputs/evidence-derived.json"])
    provenance = derived.pop("image_dimension_provenance")
    assert derived == json.loads(original)
    assert provenance["source_evidence_sha256"] == sha(original)
    assert provenance["source_evidence_canonical_sha256"] == sha(json.dumps(derived, sort_keys=True, allow_nan=False).encode())
    assert provenance["source_manifest_sha256"] == snapshot["identity"]["source_manifest_sha256"]
    assert "media bytes were not read" in snapshot["identity"]["media_verification"]
    assert not (root / "source.webm").exists()
    assert all(path.name != "source.webm" for path, _ in snapshot["pins"])
    assert len(snapshot["pins"]) == (14 if route == "sealed_pose_evidence" else 13)
    inputs.recheck_reference_inputs(snapshot, deadline=time.monotonic() + 10)


def test_temporal_raw_identity_and_distinct_native_clock_remain_exact(tmp_path):
    root = make_body_source(tmp_path / "body", temporal=True)
    snapshot = prepare(root)
    evidence = json.loads(snapshot["files"]["inputs/evidence-derived.json"])
    assert evidence["frames"][2]["identity_raw"] == json.loads((root / "pose-landmarks.json").read_bytes())["frames"][2]
    assert evidence["frames"][2]["decoder_timestamp_s"] != evidence["frames"][2]["time_s"]


@pytest.mark.parametrize("field", ["human.json", "evidence.json", "pose-landmarks.json", "body-source.json",
                                   "implementation-source/exercise_vision.py"])
def test_tampered_selected_member_is_rejected_before_derivation(tmp_path, field):
    root = make_body_source(tmp_path / "body")
    with (root / field).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="SHA256"):
        prepare(root)


@pytest.mark.parametrize("name,change,reason", [
    ("source.json", lambda d: d.update(sha256="c" * 64), "media identity"),
    ("human.json", lambda d: d.update(source_sha256="c" * 64), "body/source"),
    ("human.json", lambda d: d.update(whole_body_refinement={}), "body/source"),
    ("human.json", lambda d: d["reference_evidence"][1].update(evidence_id="other"), "reference"),
    ("evidence.json", lambda d: d.update(decoded_frames=12), "Incomplete decoder"),
    ("evidence.json", lambda d: d["frames"].pop(), "sampled pose"),
    ("evidence.json", lambda d: d["frames"][1].update(decoder_timestamp_s=0), "clocks"),
    ("evidence.json", lambda d: d["frames"][1]["landmarks"][0].update(x=.9), "raw record"),
    ("evidence.json", lambda d: d["model"].update(sha256="d" * 64), "model identities"),
    ("evidence.json", lambda d: d.update(image_dimension_provenance={"method": "verified_source_video"}), "dimension claims"),
    ("evidence.json", lambda d: d.update(image_dimension_provenance={"method": "authored_analytic_fixture"}), "dimension claims"),
    ("evidence.json", lambda d: d["quality"].update(passes=False), "quality"),
    ("identity-audit.json", lambda d: d.update(raw_artifact_sha256="c" * 64), "Raw pose identity"),
    ("source-clock.json", lambda d: d["samples"][1].update(time_s=.2), "decoder clock"),
    ("implementation.json", lambda d: d["files"].update({"exercise_vision.py": "d" * 64}), "producer version"),
    ("body-source.json", lambda d: d.update(simulation_ready=True), "construction marker"),
])
def test_conflicting_sealed_metadata_fails(tmp_path, name, change, reason):
    root = make_body_source(tmp_path / "body")
    mutate(root, name, change)
    with pytest.raises(ValueError, match=reason):
        prepare(root)


@pytest.mark.parametrize("mutation", [
    lambda d: d["files"].update({"evidence.json": "c" * 64}),
    lambda d: d["files"].pop(inputs._PRODUCER),
    lambda d: d["files"].update({inputs._PRODUCER: "c" * 64}),
    lambda d: d.update(status="failed"),
])
def test_saved_source_cannot_assert_unbound_or_failed_decoder_provenance(tmp_path, mutation):
    root = make_body_source(tmp_path / "body", route="sealed_pose_evidence")
    mutate(root, "inputs/source-manifest.json", mutation)
    with pytest.raises(ValueError):
        prepare(root)


def test_unknown_producer_version_is_not_inferred_from_filename(tmp_path):
    root = make_body_source(tmp_path / "body")
    (root / inputs._PRODUCER).write_bytes(b"# not a reviewed producer\n")
    mutate(root, "implementation.json", lambda d: d["files"].update({"exercise_vision.py": sha((root / inputs._PRODUCER).read_bytes())}))
    with pytest.raises(ValueError, match="producer version"):
        prepare(root)


@pytest.mark.parametrize("name", ["../escape", "absolute/../escape", "a\\b", "/absolute", "manifest.json/x", "human.json/child"])
def test_manifest_paths_and_collisions_fail(tmp_path, name):
    root = make_body_source(tmp_path / "body")
    mutate(root, "manifest.json", lambda d: d["files"].update({name: "c" * 64}), reseal=False)
    with pytest.raises(ValueError):
        prepare(root)


@pytest.mark.parametrize("mode", ["duplicate", "nan", "infinite", "depth", "oversized"])
def test_strict_json_and_per_document_bound(tmp_path, mode):
    root = make_body_source(tmp_path / "body")
    bad = {"duplicate": b'{"key":1,"key":2}', "nan": b'{"key":NaN}', "infinite": b'{"key":1e999}',
           "depth": b'{"key":' + b'[' * 66 + b'0' + b']' * 66 + b'}',
           "oversized": b' ' * (inputs.MAX_JSON_BYTES + 1)}[mode]
    (root / "observations.json").write_bytes(bad)
    seal(root)
    with pytest.raises(ValueError):
        prepare(root)


def test_manifest_and_selected_aggregate_bounds(tmp_path, monkeypatch):
    root = make_body_source(tmp_path / "body")
    monkeypatch.setattr(inputs, "MAX_INPUT_BYTES", 64 * 1024)
    with pytest.raises(ValueError, match="bound"):
        prepare(root)


@pytest.mark.parametrize("kind", ["root", "member", "parent"])
def test_symlink_roots_members_and_ancestors_are_never_followed(tmp_path, kind):
    root = make_body_source(tmp_path / "body")
    if kind == "root":
        link = tmp_path / "linked"
        link.symlink_to(root, target_is_directory=True)
        root = link
    elif kind == "parent":
        link = tmp_path / "linked"
        link.symlink_to(tmp_path, target_is_directory=True)
        root = link / "body"
    else:
        original = root / "human-original.json"
        (root / "human.json").rename(original)
        (root / "human.json").symlink_to(original)
    with pytest.raises((ValueError, OSError)):
        prepare(root)


@pytest.mark.parametrize("change", ["member", "manifest", "failure", "root"])
def test_recheck_detects_changed_bytes_failure_and_directory_replacement(tmp_path, change):
    root = make_body_source(tmp_path / "body")
    snapshot = prepare(root)
    if change in {"member", "manifest"}:
        path = root / ("human.json" if change == "member" else "manifest.json")
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "failure":
        (root / "failure.json").symlink_to(root / "missing")
    else:
        root.rename(tmp_path / "prior")
        root.mkdir()
    with pytest.raises((ValueError, OSError)):
        inputs.recheck_reference_inputs(snapshot, deadline=time.monotonic() + 10)


def test_deadline_and_wrong_expected_seal_fail_before_more_reads(tmp_path):
    root = make_body_source(tmp_path / "body")
    with pytest.raises(TimeoutError):
        inputs.prepare_reference_inputs(root, deadline=time.monotonic() - 1)
    with pytest.raises(ValueError, match="SHA256"):
        prepare(root, expected_manifest_sha256="c" * 64)


def test_saved_body_metadata_compatibility_preserves_lowconfidence_and_clock(tmp_path):
    root = Path(__file__).parents[1] / "artifacts/heavy-video-body-01"
    if not root.exists():
        pytest.skip("Optional local saved body is not present")
    snapshot = prepare(root, expected_manifest_sha256="b38ed6570f2fd67ba8bba6eae97192ef43c8bcb742f2c721d649bcb3bfd76232")
    evidence = json.loads(snapshot["files"]["inputs/evidence-derived.json"])
    assert (len(evidence["frames"]), evidence["decoded_frames"], evidence["frame_count"]) == (118, 235, 236)
    low = next(frame for frame in evidence["frames"] if frame["frame"] == 44)
    assert low["status"] == "low_confidence" and low["decoder_timestamp_s"] == 1.481
    assert len(json.loads(snapshot["files"]["inputs/human.json"])["reference"]) == 117
    assert all(path.suffix not in {".webm", ".mp4"} for path, _ in snapshot["pins"])
