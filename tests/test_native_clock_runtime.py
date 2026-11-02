"""Synthetic native-clock runtime checks; no media inference or physical claims."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time

import pytest

from test_root_motion import evidence
from test_exercise_body_source import forbid_runtime, pose_run
from test_whole_body_reference_inputs import make_body_source
from weft import whole_body_reference_inputs as inputs
from weft.human import build_spatial_human, validate_human
from weft.io import canonical_hash
from weft.root_motion import build_root_reference
from weft.shared_world import validate_shared_world
from weft.shared_world_human import _scene
from weft.shared_world_lab import authored_shared_world_scene
from weft.source_clock import clock_summary, derive_evidence


def native_binding(source, ticks, *, time_base="1/1000", offset=-1.99):
    numerator, denominator = map(int, time_base.split("/"))
    raw = inputs._bytes(source)
    return {"schema_version": 1, "profile": "source-native-clock-v1",
        "source_sha256": source["source_sha256"], "evidence_sha256": sha256(raw).hexdigest(),
        "evidence_canonical_sha256": canonical_hash(source), "metadata_frames": source["frame_count"],
        "decoded_frames": source["decoded_frames"], "nominal_fps": source["fps"],
        "time_base": time_base, "native_origin_pts": ticks[0],
        "native_to_reconstruction_offset_s": -ticks[0] * numerator / denominator,
        "decoder_minus_native_s": offset, "capture_clock_status": "unknown",
        "detector_clock_status": "preserved_nominal_frame_index_divided_by_fps",
        "rows": [{"frame": i, "pts": tick, "decoder_timestamp_s": tick * numerator / denominator + offset}
                 for i, tick in enumerate(ticks)]}


def root_fixture(*, gap=False, missing=False):
    source = evidence(dx=2)
    source["decoded_frames"] = 10
    ticks = [2000 + value for value in (0, 40, 100, 140, 200, 240, 300, 340, 400, 440)]
    if gap:
        ticks[5:] = [tick + 300 for tick in ticks[5:]]
    for row, tick in zip(source["frames"], ticks):
        row["decoder_timestamp_s"] = tick / 1000 - 1.99
    if missing:
        source["frames"][4]["landmarks"][23]["visibility"] = .1
    binding = native_binding(source, ticks)
    return source, derive_evidence(source, binding), binding


def test_root_interpolation_uses_native_elapsed_time_and_preserves_original():
    original, view, _ = root_fixture(missing=True)
    saved = deepcopy(original)
    result = build_root_reference(view)
    assert result["status"] == "ready"
    assert [row["time_s"] for row in result["reference"]] == [0, .04, .1, .14, .2, .24, .3, .34, .4, .44]
    assert result["reference"][4]["status"] == "interpolated_reference_only"
    assert result["reference"][4]["translation"][0] == pytest.approx(.042)
    assert build_root_reference(original)["reference"][4]["translation"][0] == pytest.approx(.04)
    assert original == saved


def test_native_gap_fails_existing_gate_before_interpolation():
    original, view, _ = root_fixture(gap=True, missing=True)
    assert build_root_reference(original)["status"] == "ready"
    result = build_root_reference(view)
    assert result["status"] == "insufficient_reference"
    assert result["reference"][4]["translation"] is None
    assert result["quality"]["long_reference_intervals"]


def test_verified_signed_decoder_origin_is_preserved_in_native_root():
    source, _, _ = root_fixture()
    ticks = [2000 + 50 * i for i in range(10)]
    for row, tick in zip(source["frames"], ticks):
        row["decoder_timestamp_s"] = tick / 1000 - 2.1
    view = derive_evidence(source, native_binding(source, ticks, offset=-2.1))
    assert build_root_reference(view)["reference"][0]["decoder_timestamp_s"] == pytest.approx(-.1)
    with pytest.raises(ValueError, match="decoder timestamp"):
        build_root_reference(source)


def test_spatial_mobile_builder_keeps_native_knots_summary_and_fixed_dt(tmp_path, monkeypatch):
    original = pose_run(tmp_path / "poses")
    source = json.loads((original / "evidence.json").read_bytes())
    ticks = [2000 + 30 * i + (i // 3) * 10 for i in range(source["decoded_frames"])]
    for row in source["frames"]:
        if row["decoder_timestamp_s"] is not None:
            row["decoder_timestamp_s"] = ticks[row["frame"]] / 1000 - 1.99
    binding = native_binding(source, ticks)
    view = derive_evidence(source, binding)
    forbid_runtime(monkeypatch)
    build_spatial_human(view, tmp_path / "human", mobile=True)
    human = json.loads((tmp_path / "human" / "human.json").read_bytes())
    assert human["native_clock"] == clock_summary(binding)
    assert human["dt"] == 1 / 30
    expected = [row["time_s"] for row in view["reference"]]
    assert [row["time_s"] for row in human["reference"]] == expected
    assert [row["time_s"] for row in human["root_reference"]] == expected


@pytest.mark.parametrize("field", ["time_s", "landmarks"])
def test_root_rejects_changed_native_derived_evidence(field):
    _, view, _ = root_fixture()
    if field == "time_s":
        view["frames"][2][field] += .01
    else:
        view["frames"][2][field][11]["x"] += .01
    with pytest.raises(ValueError, match="Native source clock"):
        build_root_reference(view)


def clocked_scene():
    _, _, binding = root_fixture()
    scene = authored_shared_world_scene()
    human = scene["human_spec"]
    human["source_sha256"] = binding["source_sha256"]
    human["native_clock"] = clock_summary(binding)
    clock = scene["coordinate_frame"]["clock_binding"]
    clock.update(native_clock=deepcopy(human["native_clock"]), alignment_method="native_pts_elapsed_time_v1",
                 source_fps=binding["nominal_fps"], source_time_offset_s=human["reference"][0]["time_s"])
    return scene


def test_native_summary_is_explicit_in_body_and_world_without_changing_dt():
    scene = clocked_scene()
    result = validate_shared_world(scene)
    assert result["dt"] == result["human_spec"]["dt"] == 1 / 30
    assert result["substeps"] == 16
    assert result["human_spec"]["native_clock"] == result["coordinate_frame"]["clock_binding"]["native_clock"]
    legacy = authored_shared_world_scene()
    assert validate_shared_world(legacy) == legacy


@pytest.mark.parametrize("mutation", ["missing", "hash", "fps", "offset", "method"])
def test_native_shared_clock_rejects_mismatch(mutation):
    scene = clocked_scene()
    clock = scene["coordinate_frame"]["clock_binding"]
    if mutation == "missing": clock.pop("native_clock")
    elif mutation == "hash": clock["native_clock"]["binding_sha256"] = "f" * 64
    elif mutation == "fps": clock["source_fps"] = 40
    elif mutation == "offset": clock["source_time_offset_s"] += .01
    else: clock["alignment_method"] = "pose_frame_index_over_fps_prior_v1"
    with pytest.raises(ValueError, match="clock"):
        validate_shared_world(scene)


def test_native_body_summary_requires_matching_source_and_observations():
    human = clocked_scene()["human_spec"]
    bad = deepcopy(human)
    bad["source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="summary source"):
        validate_human(bad)
    with pytest.raises(ValueError, match="same native source clock"):
        _scene(human, {"source": {}}, {}, {}, 1.)


def reference_documents(tmp_path, monkeypatch):
    # Synthetic custody exercises the protocol, not qualification of the current decoder.
    producer = (Path(inputs.__file__).parent / "exercise_vision.py").read_bytes()
    monkeypatch.setitem(inputs._DECODERS, sha256(producer).hexdigest(), "analytical-fixture-decoder-only")
    root = make_body_source(tmp_path / "body")
    files = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    documents = {name: json.loads(raw) for name, raw in files.items() if name.endswith(".json")}
    source = documents["evidence.json"]
    binding = native_binding(source, [60000 + i * 1000 for i in range(10)], time_base="1/30000", offset=-1.986)
    documents["native-clock.json"] = binding
    documents["reconstruction-evidence.json"] = derive_evidence(source, binding)
    documents["human.json"]["native_clock"] = clock_summary(binding)
    for name in ("native-clock.json", "reconstruction-evidence.json", "human.json"):
        files[name] = inputs._bytes(documents[name])
    return root, documents, files


def test_reference_input_validator_independently_checks_original_and_native_custody(tmp_path, monkeypatch):
    _, docs, files = reference_documents(tmp_path, monkeypatch)
    inputs._pose_bindings(docs, files, docs["body-source.json"])
    docs["reconstruction-evidence.json"]["frames"][1]["time_s"] += .001
    files["reconstruction-evidence.json"] = inputs._bytes(docs["reconstruction-evidence.json"])
    with pytest.raises(ValueError, match="Native source clock"):
        inputs._pose_bindings(docs, files, docs["body-source.json"])


def test_native_refinement_rejects_without_silently_using_nominal_snapshot(tmp_path, monkeypatch):
    root, docs, files = reference_documents(tmp_path, monkeypatch)
    marker = docs["body-source.json"]
    marker["construction"]["clock_profile"] = "source-native-clock-v1"
    marker["artifacts"] = {name: sha256(files[name]).hexdigest() for name in inputs._REQUIRED | inputs._NATIVE_CLOCK}
    files["body-source.json"] = inputs._bytes(marker)
    manifest = docs["manifest.json"]
    manifest["files"].update({name: sha256(raw).hexdigest() for name, raw in files.items() if name != "manifest.json"})
    files["manifest.json"] = inputs._bytes(manifest)
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    with pytest.raises(ValueError, match="explicit native dimension-provenance adapter"):
        inputs.prepare_reference_inputs(root, deadline=time.monotonic() + 10)
