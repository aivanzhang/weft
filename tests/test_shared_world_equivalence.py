"""Adversarial source-binding checks using sealed recorded-producer fixtures."""

from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import os
import shutil

import pytest

from weft import shared_world_evidence as shared
from weft import source_equivalence
from weft.observations import validate_observations
from weft.source_equivalence import load_source_equivalence
from test_source_equivalence import fixture as make_preparation, seal
from test_world_evidence import SOURCE, document, inputs


PARTITIONS = ("construction", "audit", "unassigned")


def _read(root, name):
    return json.loads((root / name).read_bytes())


def _write(root, name, value):
    (root / name).write_bytes(json.dumps(value, sort_keys=True).encode())


def _framehash(root, side, mutate):
    path = root / f"{side}-frames.sha256"
    lines = []
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            values = [part.strip() for part in line.split(",")]
            mutate(values)
            line = ", ".join(values)
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")


def _equivalence(path, *, derivative_offset=3, conflict=None):
    make_preparation(path, derivative_offset=derivative_offset)
    derivative = (path / "lossless.mp4").read_bytes() + path.name.encode()
    (path / "lossless.mp4").write_bytes(derivative)
    derivative_sha = hashlib.sha256(derivative).hexdigest()
    report = _read(path, "report.json")
    mapping = _read(path, "frame-mapping-and-pixels.json")
    derivative_clock = _read(path, "derivative-decoder-clock.json")
    report.update(derivative_sha256=derivative_sha, derivative_bytes=len(derivative))
    mapping["derivative_sha256"] = derivative_sha
    derivative_clock["video_sha256"] = derivative_sha
    _write(path, "derivative-decoder-clock.json", derivative_clock)
    origins = _read(path, "clock-origins.json")

    if conflict == "pixel":
        for side in ("source", "derivative"):
            _framehash(path, side, lambda row: row.__setitem__(5, "a" * 64))
        for row in mapping["frames"]:
            row.update(source_yuv_sha256="a" * 64, derivative_yuv_sha256="a" * 64)
    elif conflict == "native_clock":
        for side in ("source", "derivative"):
            _framehash(path, side, lambda row: [row.__setitem__(index, str(int(row[index]) + 1)) for index in (1, 2)])
            for key in ("native_minus_decoder_pts", "native_first_pts", "native_last_pts"):
                origins[side][key] += 1
        for row in mapping["frames"]:
            for key in ("source_pts", "source_native_pts", "derivative_native_pts"):
                row[key] += 1
        mapping["source_pts_offset"] += 1
        report["first_pts"] += 1
        report["last_pts"] += 1
    elif conflict == "decoder_clock":
        source_clock = _read(path, "source-decoder-clock.json")
        for row in source_clock["frames"]:
            row["decoder_time_s"] = (round(row["decoder_time_s"] * 1000) + 1) / 1000
        _write(path, "source-decoder-clock.json", source_clock)
        origins["source"]["native_minus_decoder_pts"] -= 1
        origins["source"]["decoder_first_pts"] += 1
        origins["source"]["decoder_last_pts"] += 1
        origins["source"]["decoder_pts"] = [value + 1 for value in origins["source"]["decoder_pts"]]
        for row in mapping["frames"]:
            row["source_decoder_pts"] += 1
        report["source_derivative_decoder_times_identical"] = False
    elif conflict == "duration":
        _framehash(path, "source", lambda row: row.__setitem__(3, str(int(row[3]) + 1)))
        for row in mapping["frames"]:
            row["source_duration_pts"] += 1
        report["source_duration_counts_pts"] = {str(key): value for key, value in
            Counter(row["source_duration_pts"] for row in mapping["frames"]).items()}
        report["duration_mismatch_frames"] = sum(row["source_duration_pts"] != row["derivative_duration_pts"]
                                                  for row in mapping["frames"])
    elif conflict == "dimensions":
        for side in ("source", "derivative"):
            _framehash(path, side, lambda row: row.__setitem__(4, "12"))
            for name in (f"{side}-frames.sha256", f"{side}-frames.log"):
                target = path / name
                target.write_text(target.read_text().replace("2x2", "2x4"))
        report["dimensions"] = [2, 4]
        pre_run = _read(path, "pre-run.json")
        pre_run["dimensions"] = [2, 4]
        _write(path, "pre-run.json", pre_run)
    elif conflict is not None:
        raise AssertionError("Unknown fixture conflict")

    report["decoder_clock_origin"] = {side: {key: value for key, value in clock.items() if key != "decoder_pts"}
                                      for side, clock in origins.items()}
    _write(path, "report.json", report)
    _write(path, "frame-mapping-and-pixels.json", mapping)
    _write(path, "clock-origins.json", origins)
    seal(path)
    return load_source_equivalence(path)


def _document(equivalence, *, derivative=False, version=1, partitions=PARTITIONS, tag="view"):
    result = document(version=version, partitions=partitions)
    media_sha = equivalence.derivative_sha256 if derivative else equivalence.original_sha256
    width, height = equivalence.dimensions
    result["source"].update(sha256=media_sha, width_px=width, height_px=height)
    result["image"] = {"width_px": width, "height_px": height}
    result["provenance"]["test_view"] = tag
    if version == 3:
        scale = Fraction(equivalence.time_base)
        times = [float((row.derivative_decoder_pts if derivative else row.source_decoder_pts) * scale)
                 for row in equivalence.frames[:len(partitions)]]
        result["source"]["frame_times_s"] = times
    else:
        times = [index / result["source"]["fps"] for index in range(len(partitions))]
    for row in result["evidence"]:
        row["id"] = row["id"].replace(SOURCE, media_sha)
        row["time_s"] = times[row["frame"]]
    for row in result["observations"]:
        row["id"] = row["id"].replace(SOURCE, media_sha)
        row["evidence_ids"] = [value.replace(SOURCE, media_sha) for value in row["evidence_ids"]]
        row["time_s"] = times[row["frame"]]
    validate_observations(result)
    return result


def _arguments(documents, equivalents=()):
    pinned = [inputs(doc) for doc in documents]
    world_ids = ["human", *(f"track_{index}" for index in range(1, len(documents))), "authored"]
    refs = {world_id: (sha, "person") for world_id, (sha, _, _) in zip(world_ids, pinned)}
    refs["authored"] = None
    return {sha: raw for sha, raw, _ in pinned}, {
        "source_sha256": documents[0]["source"]["sha256"], "world_ids": world_ids,
        "entity_refs": refs, "source_equivalences": equivalents}


def _scene(binding):
    return {"source_binding": binding, "objects": [{"id": row["world_entity_id"]}
            for row in binding["entity_bindings"] if row["world_entity_id"] != "human"]}


def _save(root, binding, documents, equivalents):
    root.mkdir()
    for sha, name in binding["evidence_documents"].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(documents[sha])
    by_manifest = {item.manifest_sha256: item for item in equivalents}
    for row in binding.get("source_equivalences", []):
        item = by_manifest[row["bundle_manifest_sha256"]]
        for name in item.file_names:
            path = root / row["bundle_path"] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item.raw_bytes(name))


def _retime(document, times):
    document["source"]["frame_times_s"] = times
    for row in document["observations"] + document["evidence"]:
        row["time_s"] = times[row["frame"]]
    validate_observations(document)


def test_verified_alias_keeps_raw_documents_qualified_ids_and_original_clock_assumptions(tmp_path):
    equivalent = _equivalence(tmp_path / "preparation")
    original = _document(equivalent)
    derived = _document(equivalent, derivative=True, version=3)
    documents, arguments = _arguments([original, derived], (equivalent,))
    before = dict(documents)
    binding, construction = shared.build_shared_world_binding(documents, **arguments)
    assert documents == before
    assert binding["source_sha256"] == equivalent.original_sha256
    assert binding["total_source_frames"] == 3
    assert binding["source_equivalences"] == [{"original_source_sha256": equivalent.original_sha256,
        "derivative_source_sha256": equivalent.derivative_sha256, "bundle_manifest_sha256": equivalent.manifest_sha256,
        "bundle_path": f"source-mappings/{equivalent.manifest_sha256}"}]
    assert binding["entity_bindings"][-1]["provenance_mode"] == "authored_addition"
    assert binding["entity_bindings"][-1]["observation_document_sha256"] is None
    assert [(row["observed_slots"], row["missing_slots"]) for row in binding["entity_bindings"]] == [(1, 2), (1, 2), (0, 3)]
    exported = json.loads(construction)
    assert b"SECRET" not in construction and exported["physical_validation"] == "unverified"
    assert len(exported["sources"]) == 2
    for source in exported["sources"]:
        raw = json.loads(documents[source["document_sha256"]])
        assert source["source"] == raw["source"] and source["time"] == raw["time"]
        for actual, expected in zip(source["observations"], raw["observations"]):
            assert all(actual[key] == expected[key] for key in ("id", "frame", "time_s", "object_id", "partition"))
            assert actual["entity_ref"] == {"document_sha256": source["document_sha256"], "object_id": expected["object_id"]}
    run = tmp_path / "run"
    _save(run, binding, documents, (equivalent,))
    checked = shared.verify_shared_world_sources(_scene(binding), run)
    assert checked["source_evidence_verified"] is True
    assert checked["construction_sha256"] == hashlib.sha256(construction).hexdigest()
    assert {row["time"]["status"] for row in checked["source_clocks"]} == {"assumed", "observed"}
    assert all((run / name).read_bytes() == documents[sha] for sha, name in binding["evidence_documents"].items())


def test_alias_is_not_admitted_without_its_explicit_verified_bundle(tmp_path):
    equivalent = _equivalence(tmp_path / "preparation")
    documents, arguments = _arguments([_document(equivalent), _document(equivalent, derivative=True, version=3)])
    with pytest.raises(ValueError, match="same original|verified derivative"):
        shared.build_shared_world_binding(documents, **arguments)


@pytest.mark.parametrize("derivative", [False, True])
@pytest.mark.parametrize("mistake", ["native_pts", "assumed_fps", "scaled_time"])
def test_v3_times_must_match_the_selected_source_decoder_not_native_pts_or_fps(tmp_path, derivative, mistake):
    equivalent = _equivalence(tmp_path / "preparation")
    original = _document(equivalent, version=3)
    derived = _document(equivalent, derivative=True, version=3)
    target = derived if derivative else original
    times = ([row.source_native_pts / 1000 for row in equivalent.frames] if mistake == "native_pts"
             else [index / 30 for index in range(3)] if mistake == "assumed_fps"
             else [value * 2 for value in target["source"]["frame_times_s"]])
    _retime(target, times)
    documents, arguments = _arguments([original, derived], (equivalent,))
    with pytest.raises(ValueError, match="clock"):
        shared.build_shared_world_binding(documents, **arguments)


def test_unique_bundle_and_document_order_canonicalizes_but_derivative_origins_remain_distinct(tmp_path):
    first = _equivalence(tmp_path / "first")
    second = _equivalence(tmp_path / "second", derivative_offset=1)
    documents, arguments = _arguments([_document(first), _document(first, derivative=True, version=3),
                                       _document(second, derivative=True, version=3)], (first, second))
    binding, construction = shared.build_shared_world_binding(documents, **arguments)
    reversed_arguments = {**arguments, "source_equivalences": (second, first)}
    reordered, reordered_construction = shared.build_shared_world_binding(dict(reversed(list(documents.items()))), **reversed_arguments)
    assert reordered == binding and reordered_construction == construction
    assert first.source_clock == second.source_clock
    assert first.derivative_clock.native_minus_decoder_pts == 3
    assert second.derivative_clock.native_minus_decoder_pts == 1
    assert [row["derivative_source_sha256"] for row in binding["source_equivalences"]] == sorted((first.derivative_sha256, second.derivative_sha256))
    run = tmp_path / "run"
    _save(run, binding, documents, (first, second))
    reordered_scene = _scene(deepcopy(binding))
    reordered_scene["source_binding"]["source_equivalences"].reverse()
    assert shared.verify_shared_world_sources(reordered_scene, run)["construction_sha256"] == hashlib.sha256(construction).hexdigest()


@pytest.mark.parametrize("clone", [False, True])
def test_duplicate_derivative_rejects_even_if_reloaded_as_an_independent_snapshot(tmp_path, clone):
    first = _equivalence(tmp_path / "preparation")
    duplicate = first
    if clone:
        copy = tmp_path / "copy"
        copy.mkdir()
        for name in first.file_names:
            (copy / name).write_bytes(first.raw_bytes(name))
        duplicate = load_source_equivalence(copy)
        assert duplicate is not first
    documents, arguments = _arguments([_document(first), _document(first, derivative=True, version=3)], (first, duplicate))
    with pytest.raises(ValueError, match="Duplicate|duplicate|redundant"):
        shared.build_shared_world_binding(documents, **arguments)


@pytest.mark.parametrize("conflict", ["pixel", "native_clock", "decoder_clock", "duration", "dimensions"])
def test_individually_valid_bundles_cannot_disagree_about_the_same_original(tmp_path, conflict):
    first = _equivalence(tmp_path / "first")
    conflicting = _equivalence(tmp_path / "conflicting", conflict=conflict)
    assert first.original_sha256 == conflicting.original_sha256
    assert first.derivative_sha256 != conflicting.derivative_sha256
    documents, arguments = _arguments([_document(first), _document(first, derivative=True, version=3),
                                       _document(conflicting, derivative=True, version=3)], (first, conflicting))
    with pytest.raises(ValueError, match="original|canonical|conflict|disagree"):
        shared.build_shared_world_binding(documents, **arguments)


@pytest.mark.parametrize("mismatch", ["dimensions", "count"])
def test_original_document_dimensions_and_full_count_must_match_the_verified_media(tmp_path, mismatch):
    equivalent = _equivalence(tmp_path / "preparation")
    original = _document(equivalent, partitions=PARTITIONS[:2] if mismatch == "count" else PARTITIONS)
    if mismatch == "dimensions":
        original["source"]["width_px"] = original["image"]["width_px"] = 4
    validate_observations(original)
    documents, arguments = _arguments([original, _document(equivalent, derivative=True, version=3)], (equivalent,))
    with pytest.raises(ValueError, match="dimensions|coverage|frame count"):
        shared.build_shared_world_binding(documents, **arguments)


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("frame,replacement", [(1, "construction"), (1, "generation"),
                                               (1, "development_validation"), (1, "unassigned"), (2, "construction")])
def test_same_frame_reservations_cannot_change_across_documents_or_aliases(tmp_path, alias, frame, replacement):
    equivalent = _equivalence(tmp_path / "preparation")
    original = _document(equivalent, tag="original")
    other = _document(equivalent, derivative=alias, version=3 if alias else 1, tag="other")
    for row in other["observations"]:
        if row["frame"] == frame:
            row["partition"] = replacement
    validate_observations(other)
    documents, arguments = _arguments([original, other], (equivalent,) if alias else ())
    with pytest.raises(ValueError, match="partition|reserved"):
        shared.build_shared_world_binding(documents, **arguments)


def test_unbound_object_cannot_hide_a_conflicting_frame_partition(tmp_path):
    equivalent = _equivalence(tmp_path / "preparation")
    original = _document(equivalent)
    original["observations"][1]["partition"] = "audit"
    assert original["observations"][1]["object_id"] == "ball"
    validate_observations(original)
    documents, arguments = _arguments([original])
    with pytest.raises(ValueError, match="partition|reserved"):
        shared.build_shared_world_binding(documents, **arguments)


def test_legitimate_unsampled_slots_coexist_with_sampled_aliases_but_stay_withheld(tmp_path):
    equivalent = _equivalence(tmp_path / "preparation")
    original = _document(equivalent, partitions=("unsampled", "audit", "unassigned"))
    derived = _document(equivalent, derivative=True, version=3)
    documents, arguments = _arguments([original, derived], (equivalent,))
    _, construction = shared.build_shared_world_binding(documents, **arguments)
    exported = json.loads(construction)["sources"]
    source = next(row for row in exported if row["source"]["sha256"] == equivalent.original_sha256)
    alias = next(row for row in exported if row["source"]["sha256"] == equivalent.derivative_sha256)
    assert len(source["observations"]) == len(alias["observations"]) == 6
    for row in source["observations"][:2]:
        assert row["partition"] == "unsampled" and row["admission"] == "withheld"
        assert row["visibility"] is None and row["measurements"] == {} and row["evidence_refs"] == []
    assert all(row["admission"] == "included" for row in alias["observations"][:2])


@pytest.mark.parametrize("visibility", ["detected", "unknown"])
def test_unsampled_partition_cannot_launder_slots_with_another_visibility(tmp_path, visibility):
    equivalent = _equivalence(tmp_path / "preparation")
    original = _document(equivalent, partitions=("unsampled", "audit", "unassigned"))
    original["observations"][0]["visibility"] = visibility
    validate_observations(original)
    documents, arguments = _arguments([original, _document(equivalent, derivative=True, version=3)], (equivalent,))
    with pytest.raises(ValueError, match="unsampled|not_sampled"):
        shared.build_shared_world_binding(documents, **arguments)


@pytest.mark.parametrize("operation", ["build", "verify"])
def test_documents_and_mapping_snapshots_share_one_aggregate_byte_budget(tmp_path, monkeypatch, operation):
    equivalent = _equivalence(tmp_path / "preparation")
    documents, arguments = _arguments([_document(equivalent), _document(equivalent, derivative=True, version=3)], (equivalent,))
    binding, _ = shared.build_shared_world_binding(documents, **arguments)
    run = tmp_path / "run"
    _save(run, binding, documents, (equivalent,))
    cap = sum(map(len, documents.values())) + equivalent.total_input_bytes - 1
    assert sum(map(len, documents.values())) < cap and equivalent.total_input_bytes < cap
    monkeypatch.setattr(shared, "MAX_INPUT_BYTES", cap)
    with pytest.raises(ValueError, match="byte|MiB|bound|budget"):
        if operation == "build":
            shared.build_shared_world_binding(documents, **arguments)
        else:
            shared.verify_shared_world_sources(_scene(binding), run)


@pytest.mark.parametrize("operation", ["build", "verify"])
def test_mapping_file_budget_is_shared_across_bundles(tmp_path, monkeypatch, operation):
    first, second = (_equivalence(tmp_path / name) for name in ("first", "second"))
    documents, arguments = _arguments([_document(first), _document(first, derivative=True, version=3),
                                       _document(second, derivative=True, version=3)], (first, second))
    binding, _ = shared.build_shared_world_binding(documents, **arguments)
    run = tmp_path / "run"
    _save(run, binding, documents, (first, second))
    cap = len(first.file_names) + len(second.file_names) - 1
    monkeypatch.setattr(shared, "MAX_SOURCE_FILES", cap)
    with pytest.raises(ValueError, match="file|bound|budget"):
        if operation == "build":
            shared.build_shared_world_binding(documents, **arguments)
        else:
            shared.verify_shared_world_sources(_scene(binding), run)


@pytest.mark.parametrize("value", [None, {}, ""])
def test_falsey_noncollection_equivalence_values_do_not_bypass_validation(tmp_path, value):
    equivalent = _equivalence(tmp_path / "preparation")
    documents, arguments = _arguments([_document(equivalent)], value)
    with pytest.raises(ValueError):
        shared.build_shared_world_binding(documents, **arguments)


def test_more_than_seven_equivalence_bundles_rejects_before_unused_mapping_accounting(tmp_path):
    equivalents = tuple(_equivalence(tmp_path / f"preparation-{index}") for index in range(8))
    documents, arguments = _arguments([_document(equivalents[0])], equivalents)
    with pytest.raises(ValueError, match="seven|7"):
        shared.build_shared_world_binding(documents, **arguments)


@pytest.mark.parametrize("budget", [0., -1., float("nan"), float("inf")])
def test_preflight_deadline_budget_must_be_positive_and_finite(tmp_path, monkeypatch, budget):
    equivalent = _equivalence(tmp_path / "preparation")
    documents, arguments = _arguments([_document(equivalent), _document(equivalent, derivative=True, version=3)], (equivalent,))
    binding, _ = shared.build_shared_world_binding(documents, **arguments)
    run = tmp_path / "run"
    _save(run, binding, documents, (equivalent,))
    monkeypatch.setattr(shared, "MAX_SOURCE_SECONDS", budget)
    with pytest.raises(ValueError, match="time|deadline|budget|finite"):
        shared.verify_shared_world_sources(_scene(binding), run)


def test_verified_mapping_requires_its_dedicated_manifest_named_path(tmp_path):
    equivalent = _equivalence(tmp_path / "preparation")
    documents, arguments = _arguments([_document(equivalent), _document(equivalent, derivative=True, version=3)], (equivalent,))
    binding, _ = shared.build_shared_world_binding(documents, **arguments)
    binding["source_equivalences"][0]["bundle_path"] = f"other-mappings/{equivalent.manifest_sha256}"
    run = tmp_path / "run"
    _save(run, binding, documents, (equivalent,))
    with pytest.raises(ValueError, match="path|source-mappings"):
        shared.verify_shared_world_sources(_scene(binding), run)


@pytest.mark.parametrize("field", ["original_source_sha256", "derivative_source_sha256", "bundle_manifest_sha256"])
def test_verifier_rejects_changed_equivalence_binding_hashes(tmp_path, field):
    equivalent = _equivalence(tmp_path / "preparation")
    documents, arguments = _arguments([_document(equivalent), _document(equivalent, derivative=True, version=3)], (equivalent,))
    binding, _ = shared.build_shared_world_binding(documents, **arguments)
    run = tmp_path / "run"
    _save(run, binding, documents, (equivalent,))
    binding["source_equivalences"][0][field] = "b" * 64
    with pytest.raises(ValueError):
        shared.verify_shared_world_sources(_scene(binding), run)


def test_mapping_parent_replaced_by_symlink_after_validation_cannot_escape_run(tmp_path, monkeypatch):
    equivalent = _equivalence(tmp_path / "preparation")
    documents, arguments = _arguments([_document(equivalent), _document(equivalent, derivative=True, version=3)], (equivalent,))
    binding, _ = shared.build_shared_world_binding(documents, **arguments)
    run = tmp_path / "run"
    _save(run, binding, documents, (equivalent,))
    outside = tmp_path / "outside-mappings"
    shutil.copytree(run / "source-mappings", outside)
    original_loader = source_equivalence.load_source_equivalence
    seen_fds = []

    def replace_parent_before_open(path, **kwargs):
        seen_fds.append(kwargs["dir_fd"])
        assert os.fstat(kwargs["dir_fd"]).st_ino == run.stat().st_ino
        (run / "source-mappings").rename(tmp_path / "moved-original-mappings")
        (run / "source-mappings").symlink_to(outside, target_is_directory=True)
        return original_loader(path, **kwargs)

    monkeypatch.setattr(source_equivalence, "load_source_equivalence", replace_parent_before_open)
    with pytest.raises(ValueError, match="unsafe"):
        shared.verify_shared_world_sources(_scene(binding), run)
    assert len(seen_fds) == 1
    with pytest.raises(OSError):
        os.fstat(seen_fds[0])
