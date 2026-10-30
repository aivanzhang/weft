"""Recorded-producer fixtures test binding integrity, not independent media decoding."""

from collections import Counter
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from weft import source_equivalence as module
from weft.source_equivalence import SourceEquivalence, load_source_equivalence


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_bytes(json.dumps(value, sort_keys=True).encode())


def seal(root):
    write(root / "manifest.json", {"schema": 1, "files": {
        path.name: sha(path.read_bytes()) for path in root.iterdir()
        if path.is_file() and path.name != "manifest.json"}})


def edit(root, name, mutate):
    value = json.loads((root / name).read_bytes())
    mutate(value)
    write(root / name, value)
    seal(root)


def fixture(root, *, derivative_offset=3):
    root.mkdir()
    source, derivative = b"original video producer fixture", b"lossless video producer fixture"
    source_sha, derivative_sha = sha(source), sha(derivative)
    (root / "source.webm").write_bytes(source)
    (root / "lossless.mp4").write_bytes(derivative)
    native = [3, 36, 70]
    source_ticks = [value - 3 for value in native]
    derivative_ticks = [value - derivative_offset for value in native]
    durations = {"source": [33, 33, 33], "derivative": [33, 34, 30]}
    pixels = [sha(bytes([index]) * 6) for index in range(3)]
    clocks = {}
    for side, media, digest, ticks, offset in (
        ("source", "source.webm", source_sha, source_ticks, 3),
        ("derivative", "lossless.mp4", derivative_sha, derivative_ticks, derivative_offset),
    ):
        clock = {"time_base": "1/1000", "native_minus_decoder_pts": offset, "native_first_pts": native[0],
                 "native_last_pts": native[-1], "decoder_first_pts": ticks[0], "decoder_last_pts": ticks[-1],
                 "all_frames_same_offset": True, "decoder_pts": ticks}
        clocks[side] = clock
        write(root / f"{side}-decoder-clock.json", {"video_sha256": digest, "decoded_to_eof": True,
            "metadata": {"reported_frames": 99},
            "frames": [{"frame": index, "decoder_time_s": value / 1000} for index, value in enumerate(ticks)]})
        header = ("#format: frame checksums\n#version: 2\n#hash: SHA256\n#software: recorded-fixture\n"
                  "#tb 0: 1/1000\n#media_type 0: video\n#codec_id 0: rawvideo\n#dimensions 0: 2x2\n"
                  "#sar 0: 1/1\n#stream#, dts,        pts, duration,     size, hash\n")
        (root / f"{side}-frames.sha256").write_text(header + "\n".join(
            f"0, {pts}, {pts}, {durations[side][index]}, 6, {pixels[index]}" for index, pts in enumerate(native)) + "\n")
        write(root / f"{side}-frames-command.json", ["/recorded/ffmpeg", "-hide_banner", "-nostdin", "-n",
            "-threads", "2", "-copyts", "-i", f"/recorded/{media}", "-map", "0:v:0", "-an", "-sn", "-dn",
            "-c:v", "rawvideo", "-pix_fmt", "yuv420p", "-threads:v", "2", "-fps_mode", "passthrough",
            "-enc_time_base:v", "demux", "-f", "framehash", "-hash", "sha256", f"/recorded/{side}-frames.sha256"])
        write(root / f"{side}-frames-result.json", {"returncode": 0})
        (root / f"{side}-frames.log").write_text("Input #0\n Stream #0:0: Video: recorded, yuv420p, 2x2\n"
                                               "Output #0\nframe=    3 fps=0.0 Lsize= 1KiB\n")
    write(root / "clock-origins.json", clocks)
    mapping = {"time_base": "1/1000", "source_sha256": source_sha, "derivative_sha256": derivative_sha,
               "source_interval": [0, 2], "source_pts_offset": derivative_offset, "media_native_pts_shifted": False,
               "frames": [{"source_frame": index, "relative_frame": index, "source_pts": pts,
                           "relative_pts": derivative_ticks[index], "source_native_pts": pts, "derivative_native_pts": pts,
                           "source_decoder_pts": source_ticks[index], "derivative_decoder_pts": derivative_ticks[index],
                           "source_yuv_sha256": pixels[index], "derivative_yuv_sha256": pixels[index],
                           "source_duration_pts": durations["source"][index],
                           "derivative_duration_pts": durations["derivative"][index]} for index, pts in enumerate(native)]}
    write(root / "frame-mapping-and-pixels.json", mapping)
    report = {"source_sha256": source_sha, "derivative_sha256": derivative_sha, "derivative_file": "lossless.mp4",
              "derivative_bytes": len(derivative), "dimensions": [2, 2], "time_base": "1/1000", "frame_count": 3,
              "first_pts": 3, "last_pts": 70, "mapping_file": "frame-mapping-and-pixels.json", "failures": [],
              "source_and_derivative_decoded_to_eof": True, "trailing_frames": 0, "media_native_pts_shifted": False,
              "all_frame_mapping_verified": True, "all_native_pts_identical": True, "all_native_yuv_pixels_identical": True,
              "source_derivative_decoder_times_identical": source_ticks == derivative_ticks,
              "source_duration_counts_pts": {str(k): v for k, v in Counter(durations["source"]).items()},
              "derivative_duration_counts_pts": {str(k): v for k, v in Counter(durations["derivative"]).items()},
              "duration_mismatch_frames": 2,
              "decoder_clock_origin": {side: {key: value for key, value in clock.items() if key != "decoder_pts"}
                                       for side, clock in clocks.items()},
              "encoding": {"pixel_format": "yuv420p", "bit_depth": 8, "codec": "libx264", "crf": 0,
                           "crop": False, "trim": False, "resize": False, "fps_change": False, "video_only": True}}
    write(root / "report.json", report)
    write(root / "pre-run.json", {"source_sha256": source_sha, "selection": "entire_original_video",
        "source_interval": [0, 2], "dimensions": [2, 2], "native_time_base": "1/1000", "expected_source_frames": 3,
        "ffmpeg_path": "/recorded/ffmpeg", "ffmpeg_sha256": "f" * 64})
    write(root / "encode-command.json", ["/recorded/ffmpeg", "-hide_banner", "-nostdin", "-n", "-threads", "2",
        "-copyts", "-i", "/recorded/source.webm", "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1",
        "-map_chapters", "-1", "-c:v", "libx264", "-crf", "0", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-threads:v", "2", "-bf", "0", "-fps_mode", "passthrough", "-enc_time_base:v", "demux",
        "-video_track_timescale", "1000", "-movflags", "+faststart", "/recorded/lossless.mp4"])
    write(root / "encode-result.json", {"returncode": 0})
    seal(root)
    return root


@pytest.fixture
def bundle(tmp_path):
    return fixture(tmp_path / "bundle")


def test_complete_pairing_keeps_origins_durations_and_exact_bytes(bundle):
    result = load_source_equivalence(bundle)
    assert result.original_sha256 == sha((bundle / "source.webm").read_bytes())
    assert result.derivative_sha256 == sha((bundle / "lossless.mp4").read_bytes())
    assert result.original_sha256 != result.derivative_sha256
    assert result.dimensions == (2, 2) and result.time_base == "1/1000"
    assert result.source_clock.native_minus_decoder_pts == result.derivative_clock.native_minus_decoder_pts == 3
    assert [(row.source_frame, row.derivative_frame) for row in result.frames] == [(0, 0), (1, 1), (2, 2)]
    assert result.frames[1].source_native_pts == result.frames[1].derivative_native_pts == 36
    assert result.frames[1].source_decoder_pts == result.frames[1].derivative_decoder_pts == 33
    assert result.frames[1].source_duration_pts == 33 and result.frames[1].derivative_duration_pts == 34
    assert result.inspection()["duration_mismatch_frames"] == 2
    assert "manifest.json" in result.file_names
    assert result.total_input_bytes == sum((bundle / name).stat().st_size for name in result.file_names)
    assert all(result.raw_bytes(name) == (bundle / name).read_bytes() for name in result.file_names)
    assert result.manifest_sha256 == sha(result.raw_bytes("manifest.json"))
    assert "not independently decoded" in result.limitations[0]


def test_different_decoder_origins_stay_explicit(tmp_path):
    result = load_source_equivalence(fixture(tmp_path / "bundle", derivative_offset=1))
    assert result.source_clock.native_minus_decoder_pts == 3
    assert result.derivative_clock.native_minus_decoder_pts == 1
    assert result.frames[0].source_native_pts == result.frames[0].derivative_native_pts == 3
    assert result.frames[0].source_decoder_pts == 0 and result.frames[0].derivative_decoder_pts == 2


def test_result_and_snapshot_are_immutable_after_live_file_changes(bundle):
    result = load_source_equivalence(bundle)
    saved = result.raw_bytes("source.webm")
    (bundle / "source.webm").write_bytes(b"changed after verified snapshot")
    assert result.raw_bytes("source.webm") == saved
    with pytest.raises(FrozenInstanceError):
        result.original_sha256 = "a" * 64
    with pytest.raises(FrozenInstanceError):
        result.frames[0].source_frame = 100
    with pytest.raises(TypeError):
        result._raw["source.webm"] = b"changed"
    with pytest.raises(TypeError, match="load_source_equivalence"):
        SourceEquivalence()
    summary = result.inspection()
    summary["dimensions"][0] = 100
    summary["source_clock"]["native_minus_decoder_pts"] = 0
    assert result.dimensions == (2, 2) and result.source_clock.native_minus_decoder_pts == 3


@pytest.mark.parametrize("file,key,value", [
    ("frame-mapping-and-pixels.json", "source_pts_offset", 0),
    ("frame-mapping-and-pixels.json", "source_interval", [1, 2]),
    ("frame-mapping-and-pixels.json", "source_sha256", "a" * 64),
    ("frame-mapping-and-pixels.json", "time_base", "1/500"),
    ("frame-mapping-and-pixels.json", "media_native_pts_shifted", True),
    ("source-decoder-clock.json", "decoded_to_eof", False),
    ("derivative-decoder-clock.json", "video_sha256", "a" * 64),
    ("report.json", "derivative_bytes", 1),
    ("report.json", "frame_count", True),
    ("report.json", "trailing_frames", 1),
    ("report.json", "all_native_yuv_pixels_identical", False),
    ("report.json", "duration_mismatch_frames", 0),
    ("pre-run.json", "selection", "cropped_interval"),
    ("source-frames-result.json", "returncode", 1),
    ("encode-result.json", "returncode", False),
])
def test_resealed_contradictory_metadata_is_rejected(bundle, file, key, value):
    edit(bundle, file, lambda doc: doc.__setitem__(key, value))
    with pytest.raises(ValueError):
        load_source_equivalence(bundle)


@pytest.mark.parametrize("name", ["frame-mapping-and-pixels.json", "source-decoder-clock.json", "derivative-decoder-clock.json"])
@pytest.mark.parametrize("change", ["missing", "duplicate", "reorder"])
def test_complete_ordered_frame_coverage_is_required_after_resealing(bundle, name, change):
    def mutate(doc):
        rows = doc["frames"]
        if change == "missing":
            rows.pop()
        elif change == "duplicate":
            rows[1] = rows[0]
        else:
            rows[0], rows[1] = rows[1], rows[0]
    edit(bundle, name, mutate)
    with pytest.raises(ValueError):
        load_source_equivalence(bundle)


@pytest.mark.parametrize("value", [33.5, 33.0, True, -1, 66])
def test_mapping_ticks_are_exact_integers_not_fractions_or_scaled_time(bundle, value):
    edit(bundle, "frame-mapping-and-pixels.json", lambda doc: doc["frames"][1].__setitem__("relative_pts", value))
    with pytest.raises(ValueError):
        load_source_equivalence(bundle)


@pytest.mark.parametrize("times", [[0, .0335, .067], [0, .066, .134], [0, .034, .067], [0, True, .067]])
def test_decoder_times_cannot_be_fractional_scaled_or_repaired_per_frame(bundle, times):
    edit(bundle, "derivative-decoder-clock.json", lambda doc: [row.__setitem__("decoder_time_s", value)
        for row, value in zip(doc["frames"], times)])
    with pytest.raises(ValueError):
        load_source_equivalence(bundle)


def test_serialization_roundoff_is_not_a_new_clock_origin(bundle):
    edit(bundle, "source-decoder-clock.json", lambda doc: doc["frames"][1].__setitem__("decoder_time_s", .03300000000000001))
    result = load_source_equivalence(bundle)
    assert result.frames[1].source_decoder_pts == 33 and result.source_clock.native_minus_decoder_pts == 3


@pytest.mark.parametrize("name,field", [("clock-origins.json", "native_minus_decoder_pts"),
                                       ("clock-origins.json", "decoder_first_pts"),
                                       ("clock-origins.json", "native_last_pts")])
def test_recorded_origin_claims_must_match_each_frame(bundle, name, field):
    edit(bundle, name, lambda doc: doc["derivative"].__setitem__(field, 900))
    with pytest.raises(ValueError, match="origin mismatch"):
        load_source_equivalence(bundle)


@pytest.mark.parametrize("reseal", [False, True])
@pytest.mark.parametrize("media", ["source.webm", "lossless.mp4"])
def test_changed_media_is_rejected_even_if_manifest_is_resealed(bundle, media, reseal):
    (bundle / media).write_bytes(b"different media")
    if reseal:
        seal(bundle)
    with pytest.raises(ValueError, match="SHA256 mismatch|media hashes"):
        load_source_equivalence(bundle)


def test_matching_mapping_hash_cannot_hide_different_native_pixels(bundle):
    path = bundle / "derivative-frames.sha256"
    raw = path.read_text()
    old = sha(bytes([1]) * 6)
    path.write_text(raw.replace(old, "a" * 64))
    edit(bundle, "frame-mapping-and-pixels.json", lambda doc: doc["frames"][1].__setitem__("derivative_yuv_sha256", "a" * 64))
    with pytest.raises(ValueError, match="decoded pixels changed"):
        load_source_equivalence(bundle)


@pytest.mark.parametrize("replacement", [
    ("#dimensions 0: 2x2", "#dimensions 0: 2x4"),
    ("#tb 0: 1/1000", "#tb 0: 2/2000"),
    ("#hash: SHA256", "#hash: MD5"),
    ("#sar 0: 1/1", "#sar 0: 2/1"),
    ("0, 36, 36, 34, 6,", "0, 36, 37, 34, 6,"),
    ("0, 36, 36, 34, 6,", "0, 3, 3, 34, 6,"),
])
def test_framehash_format_order_dimensions_and_clock_are_not_assumed(bundle, replacement):
    path = bundle / "derivative-frames.sha256"
    path.write_text(path.read_text().replace(*replacement))
    seal(bundle)
    with pytest.raises(ValueError):
        load_source_equivalence(bundle)


@pytest.mark.parametrize("key", ["crop", "trim", "resize", "fps_change"])
def test_declared_transformations_are_not_admitted(bundle, key):
    edit(bundle, "report.json", lambda doc: doc["encoding"].__setitem__(key, True))
    with pytest.raises(ValueError, match="transformation"):
        load_source_equivalence(bundle)


@pytest.mark.parametrize("name,change", [
    ("encode-command.json", lambda command: command.__setitem__(31, "cfr")),
    ("encode-command.json", lambda command: command.extend(["-vf", "crop=2:2"])),
    ("source-frames-command.json", lambda command: command.__setitem__(10, "0:v:1")),
    ("derivative-frames-command.json", lambda command: command.__setitem__(17, "yuv444p")),
])
def test_success_booleans_cannot_hide_transforming_recorded_commands(bundle, name, change):
    edit(bundle, name, change)
    with pytest.raises(ValueError):
        load_source_equivalence(bundle)


@pytest.mark.parametrize("tail", ["", "frame= 4 fps=0.0 Lsize= 1KiB", "frame= 3 fps=0.0 size= 1KiB",
                                  "frame= 3 Lsize= 1KiB\nframe= 3 Lsize= 1KiB"])
def test_framehash_completion_must_independently_agree_with_frame_coverage(bundle, tail):
    path = bundle / "source-frames.log"
    path.write_text(path.read_text().split("frame=", 1)[0] + tail)
    seal(bundle)
    with pytest.raises(ValueError, match="process count"):
        load_source_equivalence(bundle)


def test_framehash_trailing_frames_are_not_dropped(bundle):
    path = bundle / "derivative-frames.sha256"
    with path.open("a") as stream:
        stream.write(f"0, 103, 103, 33, 6, {'a' * 64}\n")
    seal(bundle)
    with pytest.raises(ValueError, match="coverage differs"):
        load_source_equivalence(bundle)


def test_root_and_nested_directory_symlinks_are_rejected(bundle, tmp_path):
    link = tmp_path / "bundle-link"
    link.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        load_source_equivalence(link)
    (bundle / "directory-link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="regular files"):
        load_source_equivalence(bundle)


@pytest.mark.parametrize("bad_path", ["../outside", "/outside", "a/../../outside", "a//b", "./source.webm", "a\\b", "C:/media", "."])
def test_manifest_paths_cannot_escape_or_alias(bundle, bad_path):
    doc = json.loads((bundle / "manifest.json").read_bytes())
    doc["files"][bad_path] = "a" * 64
    write(bundle / "manifest.json", doc)
    with pytest.raises(ValueError, match="path"):
        load_source_equivalence(bundle)


def test_unsealed_files_missing_files_and_symlinks_are_rejected(bundle):
    extra = bundle / "extra.txt"
    extra.write_text("not sealed")
    with pytest.raises(ValueError, match="complete seal"):
        load_source_equivalence(bundle)
    extra.unlink()
    saved = (bundle / "source.webm").read_bytes()
    (bundle / "source.webm").unlink()
    with pytest.raises(ValueError, match="complete seal"):
        load_source_equivalence(bundle)
    (bundle / "other.webm").write_bytes(saved)
    (bundle / "source.webm").symlink_to("other.webm")
    with pytest.raises(ValueError, match="regular files"):
        load_source_equivalence(bundle)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":', b'\xff'])
def test_consumed_json_rejects_duplicates_nonfinite_and_malformed_bytes(bundle, raw):
    (bundle / "report.json").write_bytes(raw)
    seal(bundle)
    with pytest.raises(ValueError):
        load_source_equivalence(bundle)


@pytest.mark.parametrize("constant,value", [("MAX_FILES", 2), ("MAX_FILE_BYTES", 10), ("MAX_BUNDLE_BYTES", 200),
                                           ("MAX_JSON_BYTES", 10), ("MAX_JSON_NODES", 3), ("MAX_JSON_DEPTH", 1),
                                           ("MAX_FRAMES", 2), ("MAX_PIXELS", 2)])
def test_input_resource_caps_are_enforced_before_unbounded_work(bundle, monkeypatch, constant, value):
    monkeypatch.setattr(module, constant, value)
    with pytest.raises(ValueError):
        load_source_equivalence(bundle)


def test_bundle_snapshot_can_be_copied_and_reverified_without_original_paths(bundle, tmp_path):
    original = load_source_equivalence(bundle)
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in original.file_names:
        (copy / name).write_bytes(original.raw_bytes(name))
    restored = load_source_equivalence(copy)
    assert restored.frames == original.frames and restored.manifest_sha256 == original.manifest_sha256
    assert restored.inspection() == original.inspection()


def test_large_malformed_blank_line_log_rejects_within_a_bounded_process(bundle):
    path = bundle / "source-frames.log"
    path.write_text(path.read_text().split("frame=", 1)[0] + "\n" * 1_000_000 + "frame= 3 fps=0.0 size= 1KiB\n")
    seal(bundle)
    # A subprocess deadline also bounds the test if a non-interruptible regex regresses.
    script = """
import json, sys, time
from weft.source_equivalence import load_source_equivalence
start = time.monotonic()
try:
    load_source_equivalence(sys.argv[1], deadline=start + 2)
except ValueError as error:
    print(json.dumps({'error': str(error), 'seconds': time.monotonic() - start}))
else:
    raise AssertionError('Malformed completion was accepted')
"""
    result = subprocess.run([sys.executable, "-c", script, str(bundle)], capture_output=True,
                            text=True, check=True, timeout=5)
    outcome = json.loads(result.stdout)
    assert "process count" in outcome["error"]
    assert outcome["seconds"] < 2


@pytest.mark.parametrize("separator", ["\n", "\r", "\r\n"])
def test_ffmpeg_progress_records_preserve_carriage_return_and_terminal_count_support(bundle, separator):
    path = bundle / "source-frames.log"
    prefix = path.read_text().split("frame=", 1)[0]
    progress = ["frame= 1 fps=0.0 size= 1KiB", "\tframe= 2 fps=0.0 size= 1KiB",
                "  frame= 3 fps=0.0 Lsize= 1KiB"]
    path.write_text(prefix + separator.join(progress) + separator)
    seal(bundle)
    assert len(load_source_equivalence(bundle).frames) == 3


def test_relative_bundle_uses_the_held_directory_after_its_path_is_replaced(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    bundle = fixture(parent / "bundle")
    expected = sha((bundle / "manifest.json").read_bytes())
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        parent.rename(tmp_path / "moved-parent")
        outside = tmp_path / "outside"
        outside.mkdir()
        parent.symlink_to(outside, target_is_directory=True)
        loaded = load_source_equivalence("bundle", dir_fd=fd)
        assert loaded.manifest_sha256 == expected
        assert os.fstat(fd).st_ino == (tmp_path / "moved-parent").stat().st_ino
    finally:
        os.close(fd)


@pytest.mark.parametrize("relative", ["../bundle", "/bundle", "nested//bundle", ".", ""])
def test_directory_anchored_loader_requires_a_strict_relative_bundle_path(bundle, relative):
    fd = os.open(bundle.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError, match="path"):
            load_source_equivalence(relative, dir_fd=fd)
        assert os.fstat(fd).st_ino == bundle.parent.stat().st_ino
    finally:
        os.close(fd)


def test_directory_anchored_loader_rejects_symlinked_parent_components(bundle):
    (bundle.parent / "link").symlink_to(bundle.parent, target_is_directory=True)
    fd = os.open(bundle.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError, match="unsafe"):
            load_source_equivalence("link/bundle", dir_fd=fd)
    finally:
        os.close(fd)


@pytest.mark.parametrize("descriptor", [-1, True, "directory"])
def test_directory_anchored_loader_rejects_invalid_descriptor_types(bundle, descriptor):
    with pytest.raises(ValueError, match="descriptor"):
        load_source_equivalence("bundle", dir_fd=descriptor)


def test_directory_anchored_loader_rejects_closed_and_nondirectory_descriptors(bundle):
    closed = os.open(bundle.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.close(closed)
    with pytest.raises(ValueError, match="unsafe"):
        load_source_equivalence("bundle", dir_fd=closed)
    file_fd = os.open(bundle / "source.webm", os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="directory"):
            load_source_equivalence("bundle", dir_fd=file_fd)
        assert os.fstat(file_fd).st_size == (bundle / "source.webm").stat().st_size
    finally:
        os.close(file_fd)
