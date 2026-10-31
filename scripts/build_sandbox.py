"""Build a video sandbox and its step-by-step report using the existing Weft CLI."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo


PROJECT = Path(__file__).resolve().parents[1]
MAX_LOG_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024


def _path(value):
    path = Path(value).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Inputs and outputs must not pass through symlinks or '..'")
    return path


def _read(path, limit=MAX_METADATA_BYTES):
    path = _path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError(f"Expected a bounded regular file: {path}")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if (len(raw) != before.st_size or len(raw) > limit
            or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
               (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise ValueError(f"Input changed while reading: {path}")
    return raw


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate metadata key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("Non-finite metadata value")

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid, parse_float=finite)
    if not isinstance(value, dict):
        raise ValueError("Expected a metadata object")
    return value


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _environment(root):
    # The local workflow needs neither provider credentials nor the caller's Python startup hooks.
    return {"PATH": str(PROJECT / ".venv-lab/bin") + os.pathsep + "/usr/bin:/bin",
            "PYTHONPATH": str(PROJECT / "src"), "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "TZ": "America/Los_Angeles", "TMPDIR": "/private/tmp" if sys.platform == "darwin" else "/tmp",
            "MPLCONFIGDIR": str(root / "plot-cache")}


def _run(command, root, name, seconds):
    started, process = time.monotonic(), None
    sizes = {"stdout": 0, "stderr": 0}
    error, interrupted = None, False
    with (root / (name + ".stdout.log")).open("xb") as out, (root / (name + ".stderr.log")).open("xb") as err:
        with selectors.DefaultSelector() as selector:
            try:
                process = subprocess.Popen(command, cwd=PROJECT, env=_environment(root), stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, bufsize=0)
                selector.register(process.stdout, selectors.EVENT_READ, (out, "stdout"))
                selector.register(process.stderr, selectors.EVENT_READ, (err, "stderr"))
                while selector.get_map() or process.poll() is None:
                    if time.monotonic() - started >= seconds:
                        raise TimeoutError(f"{name} exceeded its {seconds:g}-second process limit")
                    for key, _ in selector.select(.1):
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        stream, channel = key.data
                        remaining = MAX_LOG_BYTES - sizes[channel]
                        stream.write(chunk[:remaining])
                        sizes[channel] += min(len(chunk), remaining)
                        if len(chunk) > remaining:
                            raise ValueError(f"{name} exceeded its log limit")
                        if channel == "stderr":
                            # Native logs stay inspectable without allowing terminal control sequences.
                            text = chunk.decode("utf-8", errors="replace")
                            print("".join(c for c in text if c in "\n\t" or ord(c) >= 32 and ord(c) != 127), end="", file=sys.stderr, flush=True)
            except (Exception, KeyboardInterrupt) as exc:
                interrupted = isinstance(exc, KeyboardInterrupt)
                error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            finally:
                if process is not None:
                    # A native call may ignore Python deadlines; only this owned process group is stopped.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
                    process.stdout.close()
                    process.stderr.close()
    return {"returncode": process.returncode if process else None, "error": error,
            "interrupted": interrupted, "seconds": time.monotonic() - started, "log_bytes": sizes}


def _terminal(build):
    manifest_raw, result_raw = _read(build / "manifest.json"), _read(build / "result.json")
    manifest, result = _json(manifest_raw), _json(result_raw)
    if (not isinstance(manifest.get("files"), dict)
            or manifest["files"].get("result.json") != hashlib.sha256(result_raw).hexdigest()
            or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or type(result.get("schema_version")) is not int or result["schema_version"] != 1
            or manifest.get("seal_scope") not in ("complete_immutable_outputs", "failure_receipt_only")
            or manifest.get("seal_scope") != result.get("seal_scope")
            or manifest.get("status") != result.get("status")):
        raise ValueError("Build terminal receipt is missing or differs from its seal")
    return result, hashlib.sha256(manifest_raw).hexdigest()


def _report_identity(root, source_sha, build_status):
    raw = _read(root / "manifest.json")
    manifest = _json(raw)
    if (manifest.get("profile") != "shared-exercise-build-report-v1"
            or manifest.get("status") != "report_generated"
            or manifest.get("source_manifest_sha256") != source_sha
            or not isinstance(manifest.get("files"), dict)):
        raise ValueError("Report is missing or refers to a different build")
    for name in ("index.html", "report.json.txt"):
        contents = _read(root / name)
        if hashlib.sha256(contents).hexdigest() != manifest["files"].get(name):
            raise ValueError("Report summary or page differs from its seal")
        if name == "report.json.txt":
            summary = _json(contents)
            if summary.get("source_manifest_sha256") != source_sha or summary.get("build_status") != build_status:
                raise ValueError("Report summary contradicts its build")
    return hashlib.sha256(raw).hexdigest()


def _input_identity(build, command, pins, source_sha):
    raw = _read(build / "manifest.json")
    if hashlib.sha256(raw).hexdigest() != source_sha:
        raise ValueError("Build seal changed before input-identity verification")
    manifest = _json(raw)
    expected = manifest["files"].get("input-preflight.json")
    if manifest["seal_scope"] == "failure_receipt_only" or expected is None:
        return "unavailable_before_sealed_preflight"
    contents = _read(build / "input-preflight.json")
    if hashlib.sha256(contents).hexdigest() != expected:
        raise ValueError("Build input preflight differs from its seal")
    preflight = _json(contents)
    video = command[command.index("--video") + 1]
    model = command[command.index("--pose-model") + 1]
    rows = preflight.get("input_pins")
    if (not isinstance(rows, list) or len(rows) > 320 or not all(isinstance(row, dict) for row in rows)
            or preflight.get("source_sha256") != pins[video] or not isinstance(preflight.get("pose_model"), dict)
            or preflight["pose_model"].get("sha256") != pins[model]):
        raise ValueError("Build consumed different source/model bytes than the workflow plan")
    for path, sha in pins.items():
        matches = [row for row in rows if row.get("path") == path]
        if len(matches) != 1 or matches[0].get("sha256") != sha:
            raise ValueError("Build source/model paths or identities differ from the workflow plan")
    return "sealed_source_and_model_match_plan"


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Local exercise video; never uploaded")
    source.add_argument("--inspect-build", help="Saved build to report without rerunning video processing")
    parser.add_argument("--person-only", action="store_true", help="Declare one person, ground-only support and no task-relevant equipment")
    parser.add_argument("--evidence-run", help="Otherwise, sealed equipment observations")
    parser.add_argument("--config", help="Otherwise, explicit camera/equipment configuration")
    parser.add_argument("--source-mapping", action="append", default=[])
    parser.add_argument("--pose-model", help="Local model; default .local/models/pose_landmarker_lite.task")
    parser.add_argument("--pose-recovery", choices=("selective-crop",), help="Explicit Full/IMAGE/raw recipe; no automatic model fallback")
    parser.add_argument("--root-assistance", action="store_true", help="Enable three virtual root controls; not recovered human balance")
    parser.add_argument("--train", action="store_true", help="Eight training episodes, then three separate starts per controller")
    parser.add_argument("--source-note", help="Video attribution or provenance")
    parser.add_argument("--out", help="New directory; default artifacts/video-<local timestamp>-<random suffix>")
    parser.add_argument("--serve", action="store_true", help="After success, start the local sandbox; never opens a browser")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, help="Build deadline, within the existing pipeline's limits")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.inspect_build:
        if any((args.person_only, args.evidence_run, args.config, args.source_mapping, args.pose_model,
                args.pose_recovery, args.root_assistance, args.train, args.source_note, args.max_seconds is not None)):
            parser.error("--inspect-build cannot change or rerun saved build settings")
    elif args.person_only:
        if args.evidence_run or args.config or args.source_mapping:
            parser.error("--person-only cannot include equipment or source mappings")
    elif not (args.evidence_run and args.config):
        parser.error("Declare --person-only, or supply both --evidence-run and --config")
    if len(args.source_mapping) > 7:
        parser.error("At most seven source mappings are supported")
    return args


def main(argv=None):
    args = _arguments(argv)
    python = PROJECT / ".venv-lab/bin/python"
    if not python.is_file():
        raise ValueError("Missing .venv-lab/bin/python; this launcher does not install dependencies")
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    root = _path(args.out or PROJECT / "artifacts" / (now.strftime("video-%Y%m%d-%H%M%S-") + os.urandom(3).hex()))
    if root.exists():
        saved = root / "build" if (root / "build").is_dir() else root
        command = [str(python), str(Path(__file__).absolute()), "--inspect-build", str(saved)]
        raise ValueError("Output already exists. Inspect without rerunning: " + shlex.join(command))
    prefix = [str(python), "-P", "-B", "-m", "weft"]
    command, pins = None, {}
    if args.inspect_build:
        build = _path(args.inspect_build)
        if root.is_relative_to(build) or build.is_relative_to(root):
            raise ValueError("Report output must be outside the saved build")
        _terminal(build)
        maximum = 0
    else:
        video = _path(args.video)
        variant = "full" if args.pose_recovery else "lite"
        model = _path(args.pose_model or PROJECT / ".local/models" / f"pose_landmarker_{variant}.task")
        maximum = args.max_seconds
        if maximum is None:
            maximum = (600 if args.train else 450) if args.pose_recovery else (450 if args.train else 270)
        if not math.isfinite(maximum) or not 1 <= maximum <= (660 if args.pose_recovery else 480):
            raise ValueError("Build deadline is outside the pipeline's allowed interval")
        inputs = [video, model] + [_path(p) for p in (args.evidence_run, args.config, *args.source_mapping) if p]
        if any(root.is_relative_to(p) or p.is_relative_to(root) for p in inputs):
            raise ValueError("Output must be outside every input")
        for path in (video, model):
            pins[str(path)] = hashlib.sha256(_read(path, 64 * 1024 * 1024)).hexdigest()
        build = root / "build"
        command = prefix + ["shared-exercise-build", "--video", str(video), "--pose-model", str(model),
                            "--full-reference", "--max-seconds", str(maximum), "--out", str(build)]
        if args.person_only:
            command += ["--person-only"]
        else:
            command += ["--evidence-run", str(_path(args.evidence_run)), "--config", str(_path(args.config))]
            for mapping in args.source_mapping:
                command += ["--source-mapping", str(_path(mapping))]
        if args.pose_recovery:
            command += ["--pose-recovery", args.pose_recovery]
        if args.source_note:
            command += ["--source-note", args.source_note]
        if args.root_assistance:
            command += ["--root-assistance"]
        if args.train:
            command += ["--train", "--training-generations", "1", "--training-population", "4",
                        "--training-seed", "1729", "--evaluation-seed", "938401", "--evaluation-trials", "3"]
    root.mkdir(parents=True, mode=0o700)
    report_command = prefix + ["shared-exercise-inspect", "--run", str(build), "--out", str(root / "report")]
    _write(root / "workflow-plan.json", {"profile": "local-video-sandbox-workflow-v1", "created_at": now.isoformat(),
        "build_command": command, "report_command": report_command, "input_sha256": pins,
        "build_process_seconds": maximum + 25 if command else 0, "report_process_seconds": 40,
        "log_bytes_per_stream": MAX_LOG_BYTES, "api_calls": 0, "retries": 0,
        "training_evidence": "exposed development; fixed separate seeds, not a final holdout",
        "native_memory_limit": "not hard-capped; underlying frame/output buffers are bounded"})
    receipt = {"status": "stopped", "build": str(build), "report": None, "lab": None,
               "api_calls": 0, "retries": 0, "runtime_admission": "not_checked_by_launcher"}
    print(f"Artifacts: {root}\nLocal processing only; no paid API calls.", flush=True)
    if command:
        print("1/3 Build the world (stage details below)", flush=True)
        receipt["build_process"] = _run(command, root, "build", maximum + 25)
    try:
        terminal, receipt["build_manifest_sha256"] = _terminal(build)
        receipt["build_status"] = terminal.get("status")
        if receipt.get("build_process", {}).get("interrupted"):
            raise ValueError("Interrupted; preserved outputs were not rerun or reported automatically")
        print("2/3 Save the step-by-step report", flush=True)
        receipt["report_process"] = _run(report_command, root, "report", 40)
        if receipt["report_process"]["returncode"] != 0 or receipt["report_process"]["error"]:
            raise ValueError("Report generation stopped; inspect report.stderr.log")
        receipt["report_manifest_sha256"] = _report_identity(root / "report", receipt["build_manifest_sha256"], receipt["build_status"])
        if _terminal(build)[1] != receipt["build_manifest_sha256"]:
            raise ValueError("Saved build changed while reporting")
        receipt["report"] = str(root / "report/index.html")
        print(f"Report: {receipt['report']}", flush=True)
        receipt["input_binding"] = _input_identity(build, command, pins, receipt["build_manifest_sha256"]) if command else "saved_build_not_reprocessed"
        build_process = receipt.get("build_process", {"returncode": 0, "error": None})
        if (build_process["returncode"] == 0 and build_process["error"] is None
                and terminal.get("status") == "ready" and terminal.get("lab_status") == "ready"):
            if command and receipt["input_binding"] != "sealed_source_and_model_match_plan":
                raise ValueError("Ready build has no sealed input identities matching the workflow plan")
            lab = _path(terminal.get("lab_directory", ""))
            if lab not in (build / "lab", build / "learning_workflow/lab"):
                raise ValueError("Saved lab path is outside the expected build layout")
            receipt.update(status="ready", lab=str(lab))
            print("3/3 Sandbox ready; task success and physical accuracy are separate checks.", flush=True)
        else:
            print("3/3 No playable sandbox: the build stopped. The report retains the reason and evidence.", flush=True)
    except (ValueError, OSError, TypeError, RecursionError) as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        print(f"Stopped: {exc}. Preserved artifacts: {root}", file=sys.stderr)
    _write(root / "workflow-result.json", receipt)
    if receipt["status"] != "ready":
        return 1
    lab_command = prefix + ["lab", "--resume", receipt["lab"], "--port", str(args.port)]
    print("Sandbox → choose a control mode → Start. Record & start → Stop & save exports replay/video/actions.")
    print("Start or reopen (requires this build's compatible runtime):\n" + shlex.join(lab_command), flush=True)
    if args.serve:
        # Serving is explicitly user-controlled and separate from the bounded construction process.
        return subprocess.call(lab_command, cwd=PROJECT, env=_environment(root))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, TypeError, RecursionError) as error:
        print(f"weft: {error}", file=sys.stderr)
        sys.exit(1)
