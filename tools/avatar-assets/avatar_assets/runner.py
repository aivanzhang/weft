"""Bound wall time, resident memory, log bytes and output while local jobs run."""

import os
import ctypes
import json
import math
import signal
import stat
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
        if size < 16:
            raise OSError(ctypes.get_errno(), "Cannot inspect child memory")
        return int.from_bytes(buffer.raw[8:16], "little")
    return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _output_bytes(root):
    # Do not follow links or silently ignore permission failures during accounting.
    pending, total, entries = [Path(root)], 0, 0
    while pending:
        with os.scandir(pending.pop()) as children:
            for child in children:
                entries += 1
                if entries > 100_000:
                    raise OSError("Output inventory exceeds 100000 entries")
                try:
                    info = child.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue  # A temporary child file disappeared during the scan.
                if stat.S_ISDIR(info.st_mode):
                    pending.append(child.path)
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
    return total


def _atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as handle:
        json.dump(data, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _stop_owned(process, report, checkpoint, fail, poll):
    """Bound cleanup even when the OS refuses signals; never invent an exit code."""
    events = report["cleanup_events"]
    report["cleanup_status"] = "requested"
    for signum, grace in ((signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        if poll() is not None:
            break
        report["status"] = "stop_requested"
        checkpoint()  # The resource violation survives a signal/cleanup exception.
        group_sent = False
        try:
            os.killpg(process.pid, signum)
            group_sent = True
            events.append({"scope": "owned_process_group", "signal": int(signum), "result": "sent"})
        except OSError as error:
            events.append({"scope": "owned_process_group", "signal": int(signum),
                           "result": "missing" if isinstance(error, ProcessLookupError) else "error",
                           "errno": error.errno, "error": str(error)})
        # A denied group signal does not imply that signaling our direct child is denied.
        if not group_sent and poll() is None:
            checkpoint()
            try:
                process.send_signal(signum)
                events.append({"scope": "owned_child", "signal": int(signum), "result": "sent"})
            except OSError as error:
                events.append({"scope": "owned_child", "signal": int(signum),
                               "result": "missing" if isinstance(error, ProcessLookupError) else "error",
                               "errno": error.errno, "error": str(error)})
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            events.append({"scope": "owned_child", "result": "wait_timeout", "seconds": grace})
        except OSError as error:
            events.append({"scope": "owned_child", "result": "wait_error",
                           "errno": error.errno, "error": str(error)})
    exited = poll() is not None
    report["cleanup_status"] = "child_exit_verified" if exited else "unresolved_owned_child"
    report["manual_stop_required"] = not exited
    if not exited:
        fail("owned process cleanup incomplete; inspect preserved PID before further jobs")
    checkpoint()


def run_bounded(command, log, output, timeout=180, memory_mb=1536, output_mb=512):
    """Run one owned process; checkpoint monitor evidence and allow at most 7s cleanup."""
    log, output = Path(log), Path(output)
    monitor = log.with_name(log.stem + "-monitor.json")
    handle_path = log.with_name(log.stem + "-handle.json")
    for path in (log, monitor, monitor.with_name(monitor.name + ".tmp"), handle_path):
        require(not path.exists() and not path.is_symlink(), f"Job artifact already exists: {path.name}")
    require(output.is_dir() and log.parent.is_dir(), "Job output directories must exist")
    for value in (timeout, memory_mb, output_mb):
        require(isinstance(value, (int, float)) and math.isfinite(value) and value > 0,
                "Job limits must be finite positive numbers")
    command = [os.fspath(item) for item in command]
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                "BLENDER_USER_CONFIG": str(Path(output) / "blender-config"),
                "BLENDER_USER_SCRIPTS": str(Path(output) / "blender-scripts"),
                "BLENDER_USER_EXTENSIONS": str(Path(output) / "blender-extensions")})
    start, last_size_check = time.monotonic(), 0
    report = {"schema_version": 1, "status": "starting", "command": command, "pid": None,
              "exit_code": None, "elapsed_seconds": 0, "sampled_peak_rss_bytes": None,
              "rss_samples": 0, "sampling_interval_seconds": .25, "failure": None,
              "failure_reasons": [], "monitor_errors": [], "cleanup_events": [],
              "cleanup_status": "not_required", "manual_stop_required": False,
              "final_output_bytes": None, "sampled_max_output_bytes": 0,
              "monitor_checkpoint": monitor.name,
              "measurement_scope": "Observed wall time and sampled direct-child RSS; descendant RSS is not measured",
              "descendant_exit_verified": False,
              "limits": {"wall_seconds": timeout, "rss_mb": memory_mb, "output_mb": output_mb,
                         "log_bytes": 16 * 1024 * 1024, "cleanup_grace_seconds": 7,
                         "output_inventory_entries": 100_000}}

    def fail(reason):
        if reason not in report["failure_reasons"]:
            report["failure_reasons"].append(reason)
        report["failure"] = report["failure"] or reason

    def error_record(operation, error):
        record = {"operation": operation, "type": type(error).__name__,
                  "errno": getattr(error, "errno", None), "error": str(error)}
        if record not in report["monitor_errors"]:
            report["monitor_errors"].append(record)

    def checkpoint():
        report["elapsed_seconds"] = round(time.monotonic() - start, 3)
        try:
            _atomic_json(monitor, report)
        except OSError as error:
            error_record("checkpoint", error)
            fail("monitor checkpoint unavailable")

    def check_sizes(final=False):
        try:
            size = _output_bytes(output)
            report["sampled_max_output_bytes"] = max(report["sampled_max_output_bytes"], size)
            if final:
                report["final_output_bytes"] = size
            if size > output_mb * 1024 * 1024:
                fail("output byte budget exceeded")
        except OSError as error:
            error_record("output_inventory", error)
            fail("output byte monitor unavailable")
        try:
            if log.stat().st_size > 16 * 1024 * 1024:
                fail("log byte budget exceeded")
        except OSError as error:
            error_record("log_size", error)
            fail("log byte monitor unavailable")

    def poll():
        try:
            return process.poll()
        except OSError as error:
            error_record("process_status", error)
            fail("process status monitor unavailable")
            return process.returncode  # Only a previously reaped exit code is evidence.

    pending_error = None
    with log.open("xb") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        report.update(pid=process.pid, status="running")
        try:
            with handle_path.open("x") as handle_record:
                json.dump({"pid": process.pid, "command": command, "monitor_checkpoint": monitor.name}, handle_record)
                handle_record.write("\n")
            checkpoint()
            while poll() is None:
                if report["failure"]:
                    break
                time.sleep(.25)
                elapsed = time.monotonic() - start
                try:
                    rss = resident_bytes(process.pid)
                    report["sampled_peak_rss_bytes"] = max(report["sampled_peak_rss_bytes"] or 0, rss)
                    report["rss_samples"] += 1
                    report["last_rss_sample_elapsed_seconds"] = round(elapsed, 3)
                except OSError as error:
                    # The child can exit between poll() and proc_pidinfo()/statm.
                    if isinstance(error, ProcessLookupError) and poll() is None:
                        # macOS can drop proc_pidinfo before waitpid observes the exit.
                        try:
                            process.wait(timeout=.25)
                        except (subprocess.TimeoutExpired, OSError):
                            pass
                    if poll() is None:
                        error_record("resident_memory", error)
                        fail("resident-memory monitor unavailable")
                    else:
                        report["rss_exit_race_observed"] = True
                if elapsed > timeout:
                    fail("wall-time budget exceeded")
                if (report["sampled_peak_rss_bytes"] or 0) > memory_mb * 1024 * 1024:
                    fail("resident-memory budget exceeded")
                if elapsed - last_size_check > 2:
                    check_sizes()
                    last_size_check = elapsed
                    checkpoint()
                else:
                    try:
                        if log.stat().st_size > 16 * 1024 * 1024:
                            fail("log byte budget exceeded")
                    except OSError as error:
                        error_record("log_size", error)
                        fail("log byte monitor unavailable")
                if report["failure"]:
                    break
        except BaseException as error:
            pending_error = error
            error_record("runner", error)
            fail(f"runner exception: {type(error).__name__}")
        finally:
            if poll() is None:
                _stop_owned(process, report, checkpoint, fail, poll)
            report["exit_code"] = poll()
            report["process_exit_verified"] = report["exit_code"] is not None
            check_sizes(final=True)
            report["status"] = "finished" if report["process_exit_verified"] else "cleanup_incomplete"
            checkpoint()
    if pending_error is not None:
        pending_error.add_note(f"Runner evidence: {monitor}")
        raise pending_error
    return report
