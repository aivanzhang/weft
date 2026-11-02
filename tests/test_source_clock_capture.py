"""Bounded native decode custody; synthetic clocks are not physical timing evidence."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import pytest

from weft import source_clock_capture as capture
from weft.io import canonical_hash, digest


def evidence_for(video, decoder_rows, *, fps=30., width=32, height=24):
    source = digest(video)
    samples = [{"frame": row["frame"], "id": f"{source}:frame:{row['frame']}",
                "time_s": row["frame"] / fps, "decoder_timestamp_s": row["decoder_timestamp_s"]}
               for row in decoder_rows[::2]]
    document = {"schema_version": 1, "profile": "exercise-pose-v1", "source_sha256": source,
                "frame_count": len(decoder_rows), "decoded_frames": len(decoder_rows), "fps": fps,
                "width": width, "height": height, "frames": samples,
                "reference": [{"evidence_id": row["id"], "time_s": row["time_s"]} for row in samples]}
    return (json.dumps(document, indent=2) + "\n").encode()


def native_log(ticks, *, time_base="1/1000", width=32, height=24):
    lines = ["ffmpeg version synthetic-clock-fixture", f"[Parsed_showinfo_0 @ fixture] config in time_base: {time_base}, frame_rate: 30/1"]
    lines += [f"[Parsed_showinfo_0 @ fixture] n: {i} pts: {pts} pts_time:0 fmt:yuv420p cl:left sar:1/1 s:{width}x{height} i:P"
              for i, pts in enumerate(ticks)]
    return ("\n".join(lines) + "\n").encode()


def analytical_capture(tmp_path, monkeypatch, *, origin=-1500, offset=1.7):
    video = tmp_path / "source.webm"
    video.write_bytes(b"Analytical source; no physical video evidence")
    binary = tmp_path / "ffmpeg"
    binary.write_bytes(b"Analytical executable identity; not executed")
    ticks = [origin + value for value in (0, 33, 67, 110, 143, 176)]
    rows = [{"frame": i, "decoder_timestamp_s": tick / 1000 + offset} for i, tick in enumerate(ticks)]
    evidence = evidence_for(video, rows)
    log = native_log(ticks)
    monkeypatch.setattr(capture, "_bundled_ffmpeg", lambda: binary)
    monkeypatch.setattr(capture, "_run", lambda command, deadline: log)
    return video, evidence, rows, log


def test_actual_video_uses_original_decoder_rows_and_preserves_raw_bytes(tmp_path):
    video = tmp_path / "fixture.mp4"
    binary = capture._bundled_ffmpeg()
    subprocess.run([str(binary), "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=32x24:rate=30", "-frames:v", "10", "-pix_fmt", "yuv420p", "-c:v", "libx264", str(video)],
                   check=True, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                   env={"PATH": "/usr/bin:/bin", "LANG": "C"})
    decoder = cv2.VideoCapture(str(video))
    rows = []
    try:
        while True:
            ok, frame = decoder.read()
            if not ok:
                break
            rows.append({"frame": len(rows), "decoder_timestamp_s": decoder.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                         "width": frame.shape[1], "height": frame.shape[0]})
    finally:
        decoder.release()
    assert len(rows) == 10
    raw = evidence_for(video, rows)
    before = video.read_bytes(), bytes(raw), deepcopy(rows)
    binding, receipt, log = capture.collect_native_clock(video, raw, rows)
    assert before == (video.read_bytes(), raw, rows)
    assert binding["decoded_frames"] == 10
    assert receipt["native_reached_eof"] is True
    assert receipt["work"] == {"native_decodes": 1, "additional_opencv_decodes": 0, "inference_calls": 0, "paid_calls": 0}
    assert "-copyts" in receipt["command"] and "protocol_whitelist" in " ".join(receipt["command"])
    assert capture.validate_capture(binding, receipt, log, rows, source_sha256=digest(video),
                                    evidence_sha256=hashlib.sha256(raw).hexdigest()) == receipt


@pytest.mark.parametrize("origin,offset", [(-1500, 1.7), (3000, -3.2), (0, -.007)])
def test_signed_origins_and_irregular_native_ticks_are_exact(tmp_path, monkeypatch, origin, offset):
    video, raw, rows, log = analytical_capture(tmp_path, monkeypatch, origin=origin, offset=offset)
    binding, receipt, saved = capture.collect_native_clock(video, raw, rows)
    assert saved == log
    assert binding["native_origin_pts"] == origin
    assert binding["decoder_minus_native_s"] == pytest.approx(offset)
    assert [row["pts"] for row in binding["rows"]] == [origin + x for x in (0, 33, 67, 110, 143, 176)]
    assert receipt["capture_clock_status"] == "unknown"
    assert receipt["evidence_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("mutation", ["missing", "order", "none", "drift", "dimensions", "count", "source"])
def test_mismatched_original_decode_is_rejected(tmp_path, monkeypatch, mutation):
    video, raw, rows, log = analytical_capture(tmp_path, monkeypatch)
    if mutation == "missing": rows.pop()
    elif mutation == "order": rows[2]["frame"] = 4
    elif mutation == "none": rows[2]["decoder_timestamp_s"] = None
    elif mutation == "drift": rows[1]["decoder_timestamp_s"] += .001
    elif mutation == "dimensions": rows[1].update(width=33, height=24)
    elif mutation == "count": monkeypatch.setattr(capture, "_run", lambda *_: native_log([-1500, -1467]))
    else: video.write_bytes(b"Replaced source")
    with pytest.raises(ValueError): capture.collect_native_clock(video, raw, rows)


@pytest.mark.parametrize("log", [native_log([]), native_log([0, 0]), native_log([2, 1]),
    native_log([0], width=8), native_log(list(range(1801))),
    native_log([0]).replace(b"pts: 0", b"pts: NOPTS"),
    native_log([0]).replace(b"n: 0", b"n: 2"), native_log([0]) + native_log([1])])
def test_native_parser_rejects_incomplete_reordered_or_capped_stream(log):
    with pytest.raises(ValueError): capture._parse_native_log(log)


@pytest.mark.parametrize("field", ["native_log_sha256", "binding_sha256", "decoder_rows_sha256", "ffmpeg_sha256",
                                    "source_sha256", "evidence_sha256", "native_reached_eof", "command", "bounds", "work"])
def test_receipt_tampering_rejects_without_decoder(tmp_path, monkeypatch, field):
    video, raw, rows, _ = analytical_capture(tmp_path, monkeypatch)
    binding, receipt, log = capture.collect_native_clock(video, raw, rows)
    def forbidden(*args): pytest.fail("Receipt validation must not decode or resolve an executable")
    monkeypatch.setattr(capture, "_run", forbidden)
    monkeypatch.setattr(capture, "_bundled_ffmpeg", forbidden)
    bad = deepcopy(receipt)
    if field == "native_reached_eof": bad[field] = False
    elif field == "command": bad[field][bad[field].index("-copyts")] = "-start_at_zero"
    elif field == "bounds": bad[field]["native_frame_cap"] = 1800
    elif field == "work": bad[field]["inference_calls"] = 1
    elif field == "ffmpeg_sha256": bad[field] = "not-a-sha"
    else: bad[field] = "0" * 64
    with pytest.raises(ValueError):
        capture.validate_capture(binding, bad, log, rows, source_sha256=digest(video), evidence_sha256=hashlib.sha256(raw).hexdigest())


def test_resealed_log_cannot_change_pinned_native_ticks(tmp_path, monkeypatch):
    video, raw, rows, _ = analytical_capture(tmp_path, monkeypatch)
    binding, receipt, log = capture.collect_native_clock(video, raw, rows)
    log = log.replace(b"pts: -1467", b"pts: -1466")
    receipt["native_log_sha256"] = hashlib.sha256(log).hexdigest()
    with pytest.raises(ValueError, match="ticks differ"):
        capture.validate_capture(binding, receipt, log, rows, source_sha256=digest(video), evidence_sha256=hashlib.sha256(raw).hexdigest())


def test_native_process_has_no_credentials_and_is_reaped(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child")
    log = capture._run([sys.executable, "-c", "import os,sys;sys.stderr.write(str('OPENAI_API_KEY' in os.environ))"], time.monotonic() + 3)
    assert log == b"False"


@pytest.mark.parametrize("failure", ["timeout", "overflow", "exit"])
def test_failed_native_process_is_killed_reaped_and_pipe_closed(monkeypatch, failure):
    processes = []
    original = subprocess.Popen
    def start(*args, **kwargs):
        child = original(*args, **kwargs)
        processes.append(child)
        return child
    monkeypatch.setattr(capture.subprocess, "Popen", start)
    code = {"timeout": "import time;time.sleep(5)",
            "overflow": f"import sys;sys.stderr.write('x'*{capture.MAX_LOG_BYTES + 1});sys.stderr.flush()",
            "exit": "raise SystemExit(2)"}[failure]
    with pytest.raises((ValueError, TimeoutError)):
        capture._run([sys.executable, "-c", code], time.monotonic() + (.1 if failure == "timeout" else 3))
    assert len(processes) == 1 and processes[0].poll() is not None and processes[0].stderr.closed


def test_source_change_during_capture_cannot_publish_receipt(tmp_path, monkeypatch):
    video, raw, rows, log = analytical_capture(tmp_path, monkeypatch)
    def changed(*args):
        video.write_bytes(b"source replaced during native decode")
        return log
    monkeypatch.setattr(capture, "_run", changed)
    with pytest.raises(ValueError, match="source changed"):
        capture.collect_native_clock(video, raw, rows)
