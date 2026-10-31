"""Run one saved-body proposal in an owned process and publish an offline report."""

from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import time

from weft.io import local_time
from weft.shared_world_packaging import read_package_file, write_package_files


PROFILE = "whole-body-reference-workflow-v1"
STUDY_PROFILE = "whole-body-reference-candidate-study-v1"
MAX_BYTES = 64 * 1024 * 1024
MAX_RECOVERY_BYTES = 160 * 1024 * 1024
MAX_RECOVERY_INPUT_BYTES = 96 * 1024 * 1024
MAX_MEDIA_BYTES = 64 * 1024 * 1024
MAX_FILES = 96
MAX_LOG_BYTES = 256 * 1024
RUNTIME_MODULES = (
    "__init__", "whole_body_reference_worker", "whole_body_reference_refinement", "constrained_qp",
    "io", "root_motion", "spatial_pose", "body_frame", "body_refinement", "human", "mobile_human", "errors",
)


def _json(raw):
    def unique(pairs):
        value = {}
        for name, item in pairs:
            if name in value:
                raise ValueError("Duplicate proposal metadata key")
            value[name] = item
        return value
    try:
        result = json.loads(raw, object_pairs_hook=unique)
        json.dumps(result, allow_nan=False)
    except (ValueError, RecursionError) as error:
        raise ValueError("Proposal metadata must be finite strict JSON") from error
    if not isinstance(result, dict):
        raise ValueError("Proposal metadata must be an object")
    return result


def _bytes(value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > 128 * 1024:
        raise ValueError("Proposal metadata exceeds 128 KiB")
    return raw


def _path(value):
    path = Path(value).absolute()
    if len(path.parts) > 64 or ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Proposal paths must be bounded and contain no traversal or symlinks")
    return path


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Body proposal workflow exceeded its time limit")


def _environment(study):
    # No caller credentials, Python hooks, proxy settings or user configuration reach the child.
    return {"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "TZ": "America/Los_Angeles", "TMPDIR": str(study / "temporary"),
            "MPLCONFIGDIR": str(study / "temporary")}


def _run_worker(study, plan_sha, seconds, *, script_name="whole_body_reference_worker.py"):
    if script_name not in {"whole_body_reference_worker.py", "body_camera_root_worker.py"}:
        raise ValueError("Unknown trusted body proposal worker")
    started, process = time.monotonic(), None
    sizes, error, interrupted = {"stdout": 0, "stderr": 0}, None, False
    script = study / "runtime/weft" / script_name
    command = [sys.executable, "-I", "-B", str(script), str(study), plan_sha]
    with (study / "stdout.log").open("xb") as out, (study / "stderr.log").open("xb") as err:
        with selectors.DefaultSelector() as selector:
            try:
                process = subprocess.Popen(command, cwd=study, env=_environment(study), stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, bufsize=0)
                selector.register(process.stdout, selectors.EVENT_READ, (out, "stdout"))
                selector.register(process.stderr, selectors.EVENT_READ, (err, "stderr"))
                while selector.get_map() or process.poll() is None:
                    if time.monotonic() - started >= seconds:
                        raise TimeoutError("Body proposal worker exceeded its hard wall limit")
                    for key, _ in selector.select(min(.1, max(.001, seconds - (time.monotonic() - started)))):
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        stream, channel = key.data
                        remaining = MAX_LOG_BYTES - sizes[channel]
                        stream.write(chunk[:remaining])
                        sizes[channel] += min(len(chunk), remaining)
                        if len(chunk) > remaining:
                            raise ValueError("Body proposal worker exceeded its log limit")
            except (Exception, KeyboardInterrupt) as exc:
                interrupted = isinstance(exc, KeyboardInterrupt)
                error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            finally:
                if process is not None:
                    # Native calls cannot outlive their owned process group after cancellation or timeout.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
                    process.stdout.close()
                    process.stderr.close()
    return {"returncode": process.returncode if process else None, "error": error,
            "interrupted": interrupted, "seconds": time.monotonic() - started, "log_bytes": sizes,
            "command": command, "environment_keys": sorted(_environment(study))}


def _inventory(root, deadline, *, max_bytes=MAX_BYTES, media_name=None):
    files, total, directories = {}, 0, [root]
    visits = 0
    while directories:
        with os.scandir(directories.pop()) as entries:
            for entry in entries:
                _check(deadline)
                visits += 1
                if visits > MAX_FILES * 2:
                    raise ValueError("Proposal output exceeds its directory bound")
                info, path = entry.stat(follow_symlinks=False), Path(entry.path)
                if stat.S_ISDIR(info.st_mode):
                    directories.append(path)
                    continue
                name = path.relative_to(root).as_posix()
                limit = MAX_MEDIA_BYTES if name == media_name else 16 * 1024 * 1024
                if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                    raise ValueError("Proposal output must contain bounded regular files")
                total += info.st_size
                if total > max_bytes or len(files) >= MAX_FILES - 1:
                    raise ValueError("Proposal output exceeds its package bound")
                raw = read_package_file(root, name, max_bytes=max(1, info.st_size))
                files[name] = sha256(raw).hexdigest()
    return files, total


def _seal(root, profile, status, deadline, *, max_bytes=MAX_BYTES, media_name=None, expected_files=None):
    files, total = _inventory(root, deadline, max_bytes=max_bytes, media_name=media_name)
    if expected_files is not None and any(files.get(name) != digest for name, digest in expected_files.items()):
        raise ValueError("Proposal output changed before its terminal seal")
    raw = _bytes({"schema_version": 1, "profile": profile, "status": status, "files": files})
    if total + len(raw) > max_bytes:
        raise ValueError("Proposal seal exceeds its output reservation")
    write_package_files(root, {"manifest.json": raw})
    return sha256(raw).hexdigest()


def _runtime():
    source = Path(__file__).parent
    files = {f"runtime/weft/{name}.py": read_package_file(source, name + ".py", max_bytes=2 * 1024 * 1024)
             for name in RUNTIME_MODULES}
    return files, {"python": sys.version.split()[0], **{name: version(name) for name in ("numpy", "mujoco", "gymnasium")}}


def prepare_whole_body_reference_proposal(source, out, *, expected_manifest_sha256=None,
                                          proposal_objective="image_v1", fit_seconds=30., max_seconds=60.):
    """Deliver a proposal and report, never install a body or change any acceptance gate."""
    from weft.whole_body_reference_inputs import prepare_reference_inputs, recheck_reference_inputs
    from weft.whole_body_reference_report import prepare_whole_body_reference_report

    if proposal_objective not in {"image_v1", "floor_first_v1"}:
        raise ValueError("Unknown body proposal objective")
    if (type(fit_seconds) not in (int, float) or not math.isfinite(fit_seconds) or not .001 <= fit_seconds <= 30
            or type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 20 <= max_seconds <= 60):
        raise ValueError("Fit time must be 0.001–30 seconds; workflow time must be 20–60 seconds")
    source, out = _path(source), _path(out)
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Proposal output must be new and outside its source")
    started = time.monotonic()
    deadline = started + max_seconds
    try:
        snapshot = prepare_reference_inputs(source, expected_manifest_sha256=expected_manifest_sha256,
                                            deadline=min(deadline, started + 10))
    except (TypeError, KeyError, AttributeError) as error:
        raise ValueError("Malformed body-source metadata; expected the sealed body-source schema") from error
    runtime, versions = _runtime()
    initial = {**snapshot["files"], **runtime}
    recovery = snapshot["identity"].get("recovery")
    media_name, output_limit = None, MAX_BYTES
    input_limit = 32 * 1024 * 1024
    if recovery is not None:
        media = snapshot["identity"].get("source_artifact")
        if media not in {"source" + suffix for suffix in (".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv")}:
            raise ValueError("Recovery requires its exact original source container")
        media_name, output_limit = "inputs/" + media, MAX_RECOVERY_BYTES
        if recovery.get("media", {}).get("artifact") != media_name:
            raise ValueError("Recovery media copy differs from its verified source artifact")
        if media_name not in initial or len(initial[media_name]) > MAX_MEDIA_BYTES:
            raise ValueError("Recovery source exceeds its existing 64-MiB media bound")
        input_limit = MAX_RECOVERY_INPUT_BYTES + sum(map(len, runtime.values()))
    if len(initial) > 64 or sum(map(len, initial.values())) > input_limit:
        raise ValueError("Proposal input snapshot exceeds its reservation")
    _check(deadline)
    out.mkdir(parents=True, mode=0o700)
    study = out / "proposal"
    study.mkdir(mode=0o700)
    (study / "temporary").mkdir(mode=0o700)
    result = {"schema_version": 1, "profile": PROFILE, "status": "failed", "created_at": local_time(),
              "directory": str(out), "proposal_status": None, "promoted": False, "simulation_ready": False,
              "physical_validation": "unverified", "external_api_usd": 0,
              "source": snapshot["identity"], "report_path": None, "worker": None, "error": None}
    stage = "snapshot"
    try:
        write_package_files(study, initial)
        plan = {"schema_version": 1, "profile": "whole-body-reference-candidate-plan-v1", "status": "frozen",
                "proposal_objective": proposal_objective,
                "input_files": {name: sha256(initial[name]).hexdigest() for name in
                                ("inputs/human.json", "inputs/evidence-derived.json")},
                "runtime_files": {name: sha256(raw).hexdigest() for name, raw in runtime.items()},
                "runtime_versions": versions,
                "settings": {"maximum_seconds": fit_seconds, "finalization_seconds": 8,
                             "maximum_geometry_calls": 120000, "maximum_sweeps": 2},
                "source": snapshot["identity"],
                "bounds": {"worker_wall_seconds": 45, "workflow_seconds": max_seconds,
                           "output_bytes": output_limit, "output_files": MAX_FILES, "log_bytes_per_stream": MAX_LOG_BYTES,
                           "copied_input_bytes": sum(map(len, snapshot["files"].values())),
                           "source_media_bytes": len(initial[media_name]) if media_name else 0,
                           "memory_scope": "Bounded input, geometry calls and output; observed RSS, not a hard RSS cap",
                           "execution_scope": "Reviewed trusted fitter in a restricted child, not an OS sandbox for generated code"},
                "physical_validation": "unverified", "promoted": False, "external_api_usd": 0}
        plan_raw = _bytes(plan)
        write_package_files(study, {"plan.json": plan_raw})
        stage = "proposal"
        remaining = min(45., deadline - time.monotonic() - 10.)
        if remaining <= 0:
            raise TimeoutError("No time remains for a bounded body proposal")
        process = _run_worker(study, sha256(plan_raw).hexdigest(), remaining)
        result["process"] = process
        write_package_files(study, {"supervision.json": _bytes(process)})
        worker_path = study / "worker.json"
        if worker_path.exists():
            result["worker"] = _json(read_package_file(study, "worker.json", max_bytes=128 * 1024))
        if process["error"] is not None or process["returncode"] != 0:
            raise RuntimeError("Body proposal worker failed; preserved logs and receipts contain the reason")
        worker = result["worker"]
        if not isinstance(worker, dict) or worker.get("status") != "completed" or worker.get("plan_sha256") != sha256(plan_raw).hexdigest():
            raise ValueError("Worker completion is missing or differs from the frozen plan")
        recheck_reference_inputs(snapshot, deadline=deadline)
        for name, raw in initial.items():
            _check(deadline)
            read_package_file(study, name, max_bytes=max(1, len(raw)), expected_sha256=sha256(raw).hexdigest())
        read_package_file(study, "plan.json", max_bytes=len(plan_raw), expected_sha256=sha256(plan_raw).hexdigest())
        study_sha = _seal(study, STUDY_PROFILE, "completed", deadline, max_bytes=output_limit, media_name=media_name)
        result["proposal_manifest_sha256"] = study_sha
        stage = "inspection"
        inspection = prepare_whole_body_reference_report(study, out / "inspection", expected_manifest_sha256=study_sha,
                                                         max_seconds=min(10., deadline - time.monotonic()))
        result.update(status="completed", proposal_status=inspection["proposal_status"],
                      report_path=str(out / "inspection/index.html"), inspection=inspection)
    except (Exception, KeyboardInterrupt) as error:
        result["error"] = {"stage": stage, "type": type(error).__name__, "message": str(error)[:2000]}
        # A killed worker has unknown in-flight counts, not an invented zero-work result.
        if not (study / "manifest.json").exists():
            try:
                result["proposal_manifest_sha256"] = _seal(study, STUDY_PROFILE, "failed", time.monotonic() + 5,
                                                          max_bytes=output_limit, media_name=media_name)
            except Exception as seal_error:
                result["proposal_seal_error"] = {"type": type(seal_error).__name__, "message": str(seal_error)[:1000]}
    try:
        files, total = _inventory(out, time.monotonic() + 5, max_bytes=output_limit,
                                  media_name="proposal/" + media_name if media_name else None)
        result["seal_scope"] = "complete_immutable_outputs"
    except Exception as error:
        files, total = {}, 0
        result.update(status="failed", seal_scope="failure_receipt_only",
                      seal_error={"type": type(error).__name__, "message": str(error)[:1000]})
    result["seconds"] = time.monotonic() - started
    result_raw = _bytes(result)
    if len(files) + 2 > MAX_FILES or total + len(result_raw) + 128 * 1024 > output_limit:
        files = {}
        result.update(status="failed", seal_scope="failure_receipt_only",
                      seal_error={"type": "ValueError", "message": "No space reserved for the terminal seal"})
        result_raw = _bytes(result)
    write_package_files(out, {"result.json": result_raw})
    files["result.json"] = sha256(result_raw).hexdigest()
    manifest = _bytes({"schema_version": 1, "profile": PROFILE, "status": result["status"],
                       "seal_scope": result["seal_scope"], "files": files})
    write_package_files(out, {"manifest.json": manifest})
    result["manifest_sha256"] = sha256(manifest).hexdigest()
    return result
