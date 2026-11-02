"""Analytical capture receipts remain exact through wrappers; no inference or training."""

from copy import deepcopy
import hashlib
import json
import time

import pytest

from test_source_clock_capture import analytical_capture, native_log
from weft import exercise_body_source as body
from weft import shared_exercise_lab as lab
from weft import shared_exercise_learning as gains
from weft import shared_exercise_residual_learning as residual
from weft import source_clock_capture as capture
from weft.io import digest, write_json
from weft.observations import adapt_exercise
from weft.source_clock import clock_summary, derive_evidence


def encoded(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def forbidden(*args, **kwargs):
    pytest.fail("Custody validation must not decode, resolve an executable, infer or train")


@pytest.fixture
def captured(tmp_path, monkeypatch):
    video, original, rows, _ = analytical_capture(tmp_path, monkeypatch)
    binding, receipt, log = capture.collect_native_clock(video, original, rows)
    derived = derive_evidence(json.loads(original), binding)
    files = {"evidence.json": original, "native-clock.json": encoded(binding),
        "source-clock.json": encoded({"samples": []}), "reconstruction-evidence.json": encoded(derived),
        "decoder-clock.json": encoded({"schema_version": 1, "source_sha256": binding["source_sha256"], "complete": True, "rows": rows}),
        "native-clock-capture.json": encoded(receipt), "native-clock-decode.log": log}
    pins = {name: hashlib.sha256(files[name]).hexdigest() for name in body.CAPTURE_FILES}
    marker = {"construction": {"native_clock_capture": "native-clock-capture.json"}, "artifacts": pins}
    files["body-source.json"] = encoded(marker)
    observations = adapt_exercise(derived)
    observations["provenance"]["input_artifacts"] = pins
    proof = encoded(observations)
    sha = hashlib.sha256(proof).hexdigest()
    files[f"evidence/{sha}.json"] = proof
    scene = {"profile": "shared-world-v3", "human_spec": {"native_clock": clock_summary(binding)},
             "source_binding": {"evidence_documents": {sha: f"evidence/{sha}.json"}}}
    monkeypatch.setattr(capture, "_run", forbidden)
    monkeypatch.setattr(capture, "_bundled_ffmpeg", forbidden)
    return files, scene


def package(root, files):
    root.mkdir(exist_ok=True)
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(raw)
    manifest = encoded({"files": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}})
    (root / "manifest.json").write_bytes(manifest)
    return manifest


@pytest.mark.parametrize("reader", [gains._native_source_views, residual._source_views])
def test_learning_readers_retain_all_exact_receipts_without_decode(tmp_path, captured, reader):
    files, scene = captured
    root = tmp_path / "package"
    manifest = package(root, files)
    views = reader(root, manifest, scene)
    assert set(views) == lab.NATIVE_SOURCE_VIEWS | lab.NATIVE_CAPTURE_VIEWS | {"body-source.json"}
    assert all(raw == files[name] for name, raw in views.items())


@pytest.mark.parametrize("missing", [*sorted(body.CAPTURE_FILES), "all", "all_and_marker", "marker", "proof"])
@pytest.mark.parametrize("reader", [gains._native_source_views, residual._source_views])
def test_resealed_missing_capture_chain_cannot_become_legacy(tmp_path, captured, reader, missing):
    files, scene = captured
    if missing in {"all", "all_and_marker"}:
        for name in body.CAPTURE_FILES: files.pop(name)
        if missing == "all_and_marker": files.pop("body-source.json")
    elif missing == "marker": files.pop("body-source.json")
    elif missing == "proof": files.pop(next(iter(scene["source_binding"]["evidence_documents"].values())))
    else: files.pop(missing)
    root = tmp_path / "package"
    with pytest.raises(ValueError, match="capture"):
        reader(root, package(root, files), scene)


@pytest.mark.parametrize("changed", sorted(body.CAPTURE_FILES))
def test_even_resealed_marker_cannot_override_pinned_observation_hashes(captured, changed):
    files, scene = captured
    files[changed] += b" "
    marker = json.loads(files["body-source.json"])
    marker["artifacts"][changed] = hashlib.sha256(files[changed]).hexdigest()
    files["body-source.json"] = encoded(marker)
    with pytest.raises(ValueError, match="pinned identities"):
        lab.native_source_views(files, scene)


def test_receipt_semantics_revalidated_even_without_observation_claim(captured):
    files, scene = captured
    scene = deepcopy(scene)
    scene.pop("source_binding")
    receipt = json.loads(files["native-clock-capture.json"])
    receipt["native_reached_eof"] = False
    files["native-clock-capture.json"] = encoded(receipt)
    marker = json.loads(files["body-source.json"])
    marker["artifacts"]["native-clock-capture.json"] = hashlib.sha256(files["native-clock-capture.json"]).hexdigest()
    files["body-source.json"] = encoded(marker)
    with pytest.raises(ValueError, match="completion claims"):
        lab.native_source_views(files, scene)


def test_only_exact_capture_log_names_enter_bounded_input_inventory(tmp_path, captured):
    files, _ = captured
    allowed = {"native-clock-decode.log": files["native-clock-decode.log"],
               "source-run/native-clock-decode.log": files["native-clock-decode.log"]}
    root = tmp_path / "allowed"
    package(root, allowed)
    assert lab._verified_inputs(root, deadline=time.monotonic() + 10)[1] == allowed
    allowed["unrelated.log"] = b"not a capture receipt"
    package(root, allowed)
    with pytest.raises(ValueError, match="logs"):
        lab._verified_inputs(root, deadline=time.monotonic() + 10)


def test_person_world_lab_and_learning_wrappers_keep_capture_files(tmp_path, monkeypatch):
    from test_exercise_body_source import forbid_runtime, pose_run, seal
    from test_native_clock_runtime import native_binding
    from weft.person_inventory import make_person_inventory
    from weft.shared_world_human import construct_human_only_run

    poses = pose_run(tmp_path / "poses")
    evidence = json.loads((poses / "evidence.json").read_bytes())
    binding = native_binding(evidence, [60000 + i * 1000 for i in range(63)],
                             time_base="1/30000", offset=-1.997)
    clock = tmp_path / "clock.json"
    binary = tmp_path / "synthetic-ffmpeg"
    binary.write_bytes(b"Analytical executable identity; never executed")
    log = native_log([row["pts"] for row in binding["rows"]], time_base="1/30000", width=640, height=480)
    rows = [{"frame": row["frame"], "decoder_timestamp_s": row["decoder_timestamp_s"]} for row in binding["rows"]]
    monkeypatch.setattr(capture, "_bundled_ffmpeg", lambda: binary)
    monkeypatch.setattr(capture, "_run", lambda *_: log)
    binding, receipt, log = capture.collect_native_clock(poses / "source.webm", (poses / "evidence.json").read_bytes(), rows)
    write_json(clock, binding)
    write_json(poses / "decoder-clock.json", {"schema_version": 1, "source_sha256": binding["source_sha256"], "complete": True, "rows": rows})
    write_json(poses / "native-clock-capture.json", receipt)
    (poses / "native-clock-decode.log").write_bytes(log)
    seal(poses)
    monkeypatch.setattr(capture, "_run", forbidden)
    monkeypatch.setattr(capture, "_bundled_ffmpeg", forbidden)
    forbid_runtime(monkeypatch)
    output = tmp_path / "body"
    body.prepare_exercise_body_source_from_evidence(poses, output, native_clock=clock)
    kept = {name: (output / name).read_bytes() for name in body.CAPTURE_FILES}
    candidate = tmp_path / "candidate"
    construct_human_only_run(output, make_person_inventory(binding["source_sha256"]), candidate)
    scene = json.loads((candidate / "scene.json").read_bytes())
    manifest = (candidate / "manifest.json").read_bytes()
    assert all((candidate / name).read_bytes() == raw for name, raw in kept.items())
    for reader in (gains._native_source_views, residual._source_views):
        views = reader(candidate, manifest, scene)
        assert all(views[name] == raw for name, raw in kept.items())
    result = tmp_path / "lab"
    lab.prepare_shared_exercise_lab(candidate, result)
    assert all((result / "run" / name).read_bytes() == raw for name, raw in kept.items())
    lab.load_shared_exercise_lab(result / "run")
