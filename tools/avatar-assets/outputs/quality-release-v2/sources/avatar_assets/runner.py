"""Bound wall time, resident memory, log bytes and output while local jobs run."""

import os
import ctypes
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from avatar_assets.safety import require


def resident_bytes(pid):
    if sys.platform == "darwin":
        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        buffer = ctypes.create_string_buffer(96)
        size = lib.proc_pidinfo(pid, 4, 0, buffer, len(buffer))
        if size <= 0:
            raise OSError(ctypes.get_errno(), "Cannot inspect child memory")
        return int.from_bytes(buffer.raw[8:16], "little")
    return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def run_bounded(command, log, output, timeout=180, memory_mb=1536, output_mb=512):
    require(not Path(log).exists(), "Log already exists")
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                "BLENDER_USER_CONFIG": str(Path(output) / "blender-config"),
                "BLENDER_USER_SCRIPTS": str(Path(output) / "blender-scripts"),
                "BLENDER_USER_EXTENSIONS": str(Path(output) / "blender-extensions")})
    start, peak, last_size_check, max_output = time.monotonic(), 0, 0, 0
    failure = None
    with Path(log).open("xb") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            (Path(log).parent / (Path(log).stem + "-handle.json")).write_text(
                json.dumps({"pid": process.pid, "command": command}) + "\n")
            while process.poll() is None:
                time.sleep(.25)
                elapsed = time.monotonic() - start
                try:
                    peak = max(peak, resident_bytes(process.pid))
                except OSError:
                    if process.poll() is None:
                        failure = "resident-memory monitor unavailable"
                if elapsed - last_size_check > 2:
                    max_output = sum(p.stat().st_size for p in Path(output).rglob("*") if p.is_file() and not p.is_symlink())
                    last_size_check = elapsed
                if elapsed > timeout:
                    failure = "wall-time budget exceeded"
                elif peak > memory_mb * 1024 * 1024:
                    failure = "resident-memory budget exceeded"
                elif max_output > output_mb * 1024 * 1024:
                    failure = "output byte budget exceeded"
                elif Path(log).stat().st_size > 16 * 1024 * 1024:
                    failure = "log byte budget exceeded"
                if failure:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
            code = process.wait()
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    max_output = sum(p.stat().st_size for p in Path(output).rglob("*") if p.is_file() and not p.is_symlink())
    if max_output > output_mb * 1024 * 1024:
        failure = failure or "output byte budget exceeded"
    if Path(log).stat().st_size > 16 * 1024 * 1024:
        failure = failure or "log byte budget exceeded"
    return {"command": command, "pid": process.pid, "exit_code": code,
            "elapsed_seconds": round(time.monotonic() - start, 3), "sampled_peak_rss_bytes": peak,
            "sampling_interval_seconds": .25, "failure": failure, "final_output_bytes": max_output,
            "limits": {"wall_seconds": timeout, "rss_mb": memory_mb, "output_mb": output_mb}}
