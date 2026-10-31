from io import BytesIO
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

from weft import pose_pixel_support_package as package
from weft import source_frame_pixels as extract
from test_pose_pixel_support_package import inputs, put, raw, seal, sha


def fixture(root, *, video=b"Synthetic source bytes; fake decoder only", suffix=".avi"):
    pose, _ = inputs(root, video=video)
    (pose / "source.webm").rename(pose / ("source" + suffix))
    evidence = json.loads((pose / "evidence.json").read_bytes())
    evidence.update(width=96, height=64, fps=30., frame_count=12, decoded_frames=12)
    for row in evidence["frames"]:
        row["decoder_timestamp_s"] = row["frame"] / 30
    put(pose, "evidence.json", evidence)
    put(pose, "pose-landmarks.json", {"frames": evidence["frames"], "source_sha256": evidence["source_sha256"]})
    identity = json.loads((pose / "identity-audit.json").read_bytes())
    identity["raw_artifact_sha256"] = sha((pose / "pose-landmarks.json").read_bytes())
    put(pose, "identity-audit.json", identity)
    put(pose, "source-clock.json", {"source_sha256": evidence["source_sha256"], "decoded_frames": 12,
        "samples": [{key: row[key] for key in ("frame", "time_s", "decoder_timestamp_s")} for row in evidence["frames"]]})
    marker = json.loads((pose / "body-source.json").read_bytes())
    marker.update(schema_version=1, source_artifact="source" + suffix,
        artifacts={name: sha((pose / name).read_bytes()) for name in package.POSE_FILES if name != "body-source.json"})
    put(pose, "body-source.json", marker)
    seal(pose, "exercise-body-source-v1", "construction_only")
    return pose


def bgr(frame):
    y, x = np.indices((64, 96), dtype=np.uint8)
    return np.stack(((x * 3 + frame * 7) % 251, (y * 4) % 251, (x + y + frame) % 251), axis=2)


class Capture:
    def __init__(self, *, short=None, mutation=None):
        self.count, self.short, self.mutation, self.closed = 0, short, mutation, False

    def isOpened(self):
        return True

    def getBackendName(self):
        return "analytical_fake_decoder"

    def get(self, prop):
        values = {cv2.CAP_PROP_FRAME_WIDTH: 96, cv2.CAP_PROP_FRAME_HEIGHT: 64,
            cv2.CAP_PROP_FPS: 30., cv2.CAP_PROP_FRAME_COUNT: 12,
            cv2.CAP_PROP_POS_FRAMES: self.count,
            cv2.CAP_PROP_POS_MSEC: (self.count - 1) / 30 * 1000}
        value = values[prop]
        return self.mutation(self.count, prop, value) if self.mutation else value

    def read(self):
        index = self.count
        self.count += 1
        return (False, None) if self.short is not None and index >= self.short else (True, bgr(index))

    def release(self):
        self.closed = True


def local_decoder(monkeypatch, capture=None, after=None):
    capture = capture or Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)

    def run(video, output, plan, deadline):
        assert Path(video).is_file() and Path(video).stat().st_mode & 0o777 == 0o600
        assert sha(Path(video).read_bytes()) == plan["source_sha256"]
        work = {"decoder_opens": 0, "read_calls": 0, "decoded_frames": 0}
        try:
            result = {"status": "complete", **extract._decode(str(video), str(output), plan, deadline, work)}
        except Exception as error:
            result = {"status": "failed", "work": work, "error": {"type": type(error).__name__, "message": str(error)}}
        if after:
            after()
        return result

    monkeypatch.setattr(extract, "_run_bounded", run)
    return capture


def check_manifest(out):
    manifest = json.loads((out / "manifest.json").read_bytes())
    assert set(manifest["files"]) == {p.name for p in out.iterdir()} - {"manifest.json"}
    assert all(sha((out / name).read_bytes()) == expected for name, expected in manifest["files"].items())
    assert sum(p.stat().st_size for p in out.iterdir()) <= extract.MAX_BYTES
    assert len(list(out.iterdir())) <= extract.MAX_FILES
    return manifest


def test_native_lossless_prefix_and_existing_companion_acceptance(tmp_path):
    video = tmp_path / "analytic.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 30., (96, 64))
    assert writer.isOpened(), "The pinned FFmpeg runtime must support the analytical lossless fixture"
    try:
        for frame in range(12):
            writer.write(bgr(frame))
    finally:
        writer.release()
    pose = fixture(tmp_path, video=video.read_bytes())
    before = {p.name: sha(p.read_bytes()) for p in pose.iterdir()}
    out = tmp_path / "extracted"
    receipt = extract.prepare_source_frame_pixels(pose, out, frames=[0, 2, 4, 6])
    report = json.loads((out / "report.json").read_bytes())
    assert receipt["status"] == "pixels_extracted" and receipt["read_calls"] == receipt["decoded_frames"] == 7
    assert report["work"]["decoder_opens"] == 1 and report["full_video_decode_verified"] is False
    assert report["saved_source_decoded_frames"] == 12
    assert report["prefix_clocks"] == [[n, n / 30, n + 1] for n in range(7)]
    assert report["backend"] == "FFMPEG"
    for row in report["rows"]:
        rgb = np.array(Image.open(BytesIO((out / row["file"]).read_bytes())))
        expected = bgr(row["frame"])[:, :, ::-1]
        assert np.array_equal(rgb, expected)
        assert row["decoded_rgb_sha256"] == sha(expected.tobytes())
        assert row["sha256"] == sha((out / row["file"]).read_bytes())
    assert check_manifest(out)["profile"] == "source-frame-pixels-v1"
    assert before == {p.name: sha(p.read_bytes()) for p in pose.iterdir()}
    companion = package.prepare_pose_pixel_support(pose, out, tmp_path / "companion")
    assert companion["status"] == "complete" and companion["slot_count"] == 36


@pytest.mark.parametrize("frames", [[], [0], [0, 2, 4, 6, 8], [2, 0], [0, 0], [-1, 2], [0, 1800], [False, 2], [0., 2], [1, 2]])
def test_bad_or_unsampled_ids_reject_before_decode(tmp_path, monkeypatch, frames):
    pose = fixture(tmp_path)
    monkeypatch.setattr(extract, "_run_bounded", lambda *a: pytest.fail("No decode for invalid IDs"))
    with pytest.raises(ValueError):
        extract.prepare_source_frame_pixels(pose, tmp_path / "out", frames=frames)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("seconds", [0, 31, False, float("nan"), float("inf"), "3"])
def test_invalid_deadline_rejects(tmp_path, seconds):
    with pytest.raises(ValueError):
        extract.prepare_source_frame_pixels(tmp_path / "missing", tmp_path / "out", frames=[0, 2], max_seconds=seconds)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["video", "clock", "null_clock", "frame_id", "dimensions", "model", "identity", "failure", "profile", "refined"])
def test_conflicting_sealed_receipts_reject_before_decode(tmp_path, monkeypatch, mutation):
    pose = fixture(tmp_path)
    if mutation == "video":
        (pose / "source.avi").write_bytes(b"changed original")
    elif mutation == "failure":
        put(pose, "failure.json", {"status": "failed"})
    elif mutation == "profile":
        seal(pose, "unknown", "construction_only")
    else:
        evidence = json.loads((pose / "evidence.json").read_bytes())
        if mutation == "clock":
            evidence["frames"][1]["decoder_timestamp_s"] += .001
        elif mutation == "null_clock":
            evidence["frames"][1]["decoder_timestamp_s"] = None
            clock = json.loads((pose / "source-clock.json").read_bytes())
            clock["samples"][1]["decoder_timestamp_s"] = None
            put(pose, "source-clock.json", clock)
        elif mutation == "frame_id":
            evidence["frames"][1]["id"] = "changed"
        elif mutation == "dimensions":
            evidence["width"] = 4096
            evidence["height"] = 4096
        elif mutation == "model":
            evidence["model"] = {"name": "unbound"}
        elif mutation == "identity":
            evidence["pose_identity"] = {"mode": "temporal"}
        else:
            evidence["frames"][1]["image_refinement"] = {}
        put(pose, "evidence.json", evidence)
        marker = json.loads((pose / "body-source.json").read_bytes())
        marker["artifacts"] = {name: sha((pose / name).read_bytes()) for name in package.POSE_FILES if name != "body-source.json"}
        put(pose, "body-source.json", marker)
        seal(pose, "exercise-body-source-v1", "construction_only")
    monkeypatch.setattr(extract, "_run_bounded", lambda *a: pytest.fail("Conflicting source must never decode"))
    with pytest.raises(ValueError):
        extract.prepare_source_frame_pixels(pose, tmp_path / "out", frames=[0, 2])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("where", ["root", "ancestor", "video", "evidence", "output_parent"])
def test_symlinks_never_reach_native_decoder(tmp_path, monkeypatch, where):
    pose = fixture(tmp_path)
    output = tmp_path / "out"
    if where in {"root", "ancestor"}:
        link = tmp_path / "linked"
        link.symlink_to(pose if where == "root" else tmp_path, target_is_directory=True)
        pose = link if where == "root" else link / "pose"
    elif where == "output_parent":
        link = tmp_path / "linked"
        link.symlink_to(tmp_path, target_is_directory=True)
        output = link / "out"
    else:
        name = "source.avi" if where == "video" else "evidence.json"
        outside = tmp_path / "original"
        (pose / name).rename(outside)
        (pose / name).symlink_to(outside)
    monkeypatch.setattr(extract, "_run_bounded", lambda *a: pytest.fail("Linked paths must not decode"))
    with pytest.raises((OSError, ValueError)):
        extract.prepare_source_frame_pixels(pose, output, frames=[0, 2])
    assert not (tmp_path / "out").exists()


def test_fifo_source_is_rejected_without_blocking(tmp_path):
    import os
    pose = fixture(tmp_path)
    (pose / "source.avi").unlink()
    os.mkfifo(pose / "source.avi")
    with pytest.raises(ValueError, match="regular file"):
        extract.prepare_source_frame_pixels(pose, tmp_path / "out", frames=[0, 2])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("location", ["contained", "alias", "existing"])
def test_output_never_overwrites_or_enters_input(tmp_path, location):
    pose = fixture(tmp_path)
    out = pose / "out" if location == "contained" else tmp_path / "other" / ".." / "pose" / "out"
    if location == "existing":
        out = tmp_path / "out"
        out.mkdir()
        (out / "keep").write_bytes(b"preserve")
    with pytest.raises((ValueError, FileExistsError)):
        extract.prepare_source_frame_pixels(pose, out, frames=[0, 2])
    assert not (pose / "out").exists()
    if location == "existing":
        assert (out / "keep").read_bytes() == b"preserve"


@pytest.mark.parametrize("fault", ["early_eof", "timestamp", "nonmonotone", "position", "metadata", "unknown"])
def test_decoder_failures_preserve_counts_and_partial_pixels(tmp_path, monkeypatch, fault):
    pose = fixture(tmp_path)
    def mutate(index, prop, value):
        if fault == "metadata" and prop == cv2.CAP_PROP_FRAME_WIDTH:
            return value + 1
        if index == 3 and prop == cv2.CAP_PROP_POS_MSEC:
            return {"timestamp": value + 1, "nonmonotone": 0, "unknown": float("nan")}.get(fault, value)
        if fault == "position" and index == 3 and prop == cv2.CAP_PROP_POS_FRAMES:
            return 10
        return value
    cap = local_decoder(monkeypatch, Capture(short=2 if fault == "early_eof" else None, mutation=mutate))
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        extract.prepare_source_frame_pixels(pose, out, frames=[0, 2, 4, 6])
    failure = json.loads((out / "failure.json").read_bytes())
    assert failure["work"]["decoder_opens"] == 1 and cap.closed
    assert failure["work"]["read_calls"] == (0 if fault == "metadata" else 3)
    assert failure["work"]["decoded_frames"] == (0 if fault == "metadata" else 2 if fault == "early_eof" else 3)
    assert check_manifest(out)["status"] == "failed"
    assert (out / "frame-0000.png").exists() is (fault != "metadata")
    with pytest.raises(ValueError, match="failed"):
        package.prepare_pose_pixel_support(pose, out, tmp_path / "companion")


def test_oversized_png_fails_without_resize_or_oversized_write(tmp_path, monkeypatch):
    pose = fixture(tmp_path)
    local_decoder(monkeypatch)
    monkeypatch.setattr(cv2, "imencode", lambda *a: (True, np.zeros(extract.MAX_BYTES, np.uint8)))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="PNG"):
        extract.prepare_source_frame_pixels(pose, out, frames=[0, 2])
    assert not list(out.glob("*.png"))
    assert check_manifest(out)["status"] == "failed"


@pytest.mark.parametrize("mutation", ["video", "manifest", "failure", "evidence"])
def test_late_input_mutation_cannot_publish_success(tmp_path, monkeypatch, mutation):
    pose = fixture(tmp_path)
    def after():
        if mutation == "video":
            (pose / "source.avi").write_bytes(b"later change")
        elif mutation == "manifest":
            seal(pose, "changed", "construction_only")
        elif mutation == "failure":
            put(pose, "failure.json", {"status": "failed"})
        else:
            (pose / "evidence.json").write_bytes(b"{}")
    local_decoder(monkeypatch, after=after)
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        extract.prepare_source_frame_pixels(pose, out, frames=[0, 2])
    assert check_manifest(out)["status"] == "failed"
    assert len(list(out.glob("*.png"))) == 2


def test_consumer_budget_preflight_prevents_native_work(tmp_path, monkeypatch):
    pose = fixture(tmp_path)
    monkeypatch.setattr(package, "RESERVE", package.MAX_BYTES)
    monkeypatch.setattr(extract, "_run_bounded", lambda *a: pytest.fail("No work without consumer capacity"))
    with pytest.raises(ValueError, match="reservation"):
        extract.prepare_source_frame_pixels(pose, tmp_path / "out", frames=[0, 2])
    assert not (tmp_path / "out").exists()


def test_replaced_output_directory_cannot_receive_success_claim(tmp_path, monkeypatch):
    pose = fixture(tmp_path)
    out, moved = tmp_path / "out", tmp_path / "retained-partial"
    def after():
        out.rename(moved)
        out.mkdir()
        (out / "unrelated").write_bytes(b"untouched")
    local_decoder(monkeypatch, after=after)
    with pytest.raises(ValueError, match="destination identity"):
        extract.prepare_source_frame_pixels(pose, out, frames=[0, 2])
    assert (out / "unrelated").read_bytes() == b"untouched"
    assert list(out.iterdir()) == [out / "unrelated"]
    assert check_manifest(moved)["status"] == "failed"


def test_supervisor_timeout_terminates_then_kills_only_owned_worker(monkeypatch, tmp_path):
    class Pipe:
        def poll(self, seconds):
            assert 0 <= seconds <= 1
            return False
        def close(self):
            pass
    class Process:
        pid = None
        def __init__(self):
            self.events = []
        def start(self):
            self.pid = 123
        def is_alive(self):
            return True
        def terminate(self):
            self.events.append("terminate")
        def kill(self):
            self.events.append("kill")
        def join(self, timeout):
            assert timeout == .5
            self.events.append("join")
        def close(self):
            self.events.append("close")
    owned = Process()
    class Context:
        def Pipe(self, duplex):
            assert duplex is False
            return Pipe(), Pipe()
        def Process(self, *, target, args, daemon):
            assert target == extract._worker and daemon is True
            return owned
    monkeypatch.setattr(extract.multiprocessing, "get_context", lambda mode: Context() if mode == "spawn" else pytest.fail(mode))
    with pytest.raises(TimeoutError, match="wall timeout"):
        extract._run_bounded(tmp_path / "video", tmp_path, {}, time.monotonic() + 1)
    assert owned.events == ["terminate", "join", "kill", "join", "close"]


def test_interrupted_native_call_keeps_unknown_counts_and_failure_seal(tmp_path, monkeypatch):
    pose = fixture(tmp_path)
    def timeout(*args):
        raise TimeoutError("Native decoder wall timeout")
    monkeypatch.setattr(extract, "_run_bounded", timeout)
    out = tmp_path / "out"
    with pytest.raises(TimeoutError):
        extract.prepare_source_frame_pixels(pose, out, frames=[0, 2])
    failure = json.loads((out / "failure.json").read_bytes())
    assert failure["work"] is None and "Unknown" in failure["count_scope"]
    assert check_manifest(out)["status"] == "failed"


def test_native_worker_removes_credentials_and_capture_hooks(monkeypatch, tmp_path):
    environment = {"PATH": "/usr/bin", "TMPDIR": str(tmp_path), "API_KEY": "must-not-pass",
                   "OPENCV_FFMPEG_CAPTURE_OPTIONS": "protocol_whitelist;https", "DYLD_INSERT_LIBRARIES": "untrusted"}
    redirected = []
    monkeypatch.setattr(extract, "os", SimpleNamespace(environ=environment, devnull="/dev/null",
                        dup2=lambda source, destination: redirected.append(destination)))
    def decode(*args):
        assert environment == {"PATH": "/usr/bin", "TMPDIR": str(tmp_path),
                               "OPENCV_FFMPEG_CAPTURE_OPTIONS": extract.CAPTURE_OPTIONS}
        assert "protocol_whitelist;file" in extract.CAPTURE_OPTIONS
        assert "format_whitelist;matroska,webm,mov,avi" in extract.CAPTURE_OPTIONS
        assert "enable_drefs;0" in extract.CAPTURE_OPTIONS
        return {"rows": []}
    monkeypatch.setattr(extract, "_decode", decode)
    class Pipe:
        closed = False
        def send_bytes(self, value):
            self.value = value
        def close(self):
            self.closed = True
    pipe = Pipe()
    extract._worker(pipe, "unused", "unused", {}, time.monotonic() + 1)
    assert pipe.closed and json.loads(pipe.value)["status"] == "complete"
    assert redirected == [1, 2]
