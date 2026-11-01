"""Isolate one pinned calibration call; trusted-code guards are not an OS sandbox."""

from hashlib import sha256
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import selectors
import signal
import stat
import subprocess
import sys
import time


PROFILE = "floor-target-calibration-worker-v1"
PLAN_PROFILE = "floor-target-calibration-worker-plan-v1"
PROCESS_PROFILE = "floor-target-calibration-process-v1"
RUNTIME_MODULES = frozenset({"__init__", "floor_target_calibration_worker", "floor_target_calibration",
    "camera_pose", "camera_ground", "source_surface_anchors", "world_evidence", "observations",
    "world_alignment", "io"})
RUNTIME_FILES = frozenset(f"runtime/weft/{name}.py" for name in RUNTIME_MODULES)
MAX_SECONDS, MAX_CORE_SECONDS, CPU_SECONDS = 10., 5., 8
MAX_INPUT_BYTES, MAX_RUNTIME_BYTES, MAX_OUTPUT_BYTES = 4 << 20, 4 << 20, 4 << 20
PLAN_LIMIT, REQUEST_LIMIT, SOURCE_LIMIT = 128 << 10, 64 << 10, 2 << 20
RESULT_LIMIT, RECEIPT_LIMIT, LOG_LIMIT, RSS_LIMIT = 1 << 20, 64 << 10, 256 << 10, 512 << 20
CORE_FILES = frozenset({"calibration.json", "camera.json", "ground.json"})
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_DOCUMENT = re.compile(r"inputs/(observations|measurements)/([0-9a-f]{64})\.json\Z")
_DENIED_IMPORTS = frozenset({"mujoco", "gymnasium", "pymunk", "mediapipe", "torch", "torchvision",
    "tensorflow", "transformers", "diffusers", "ultralytics", "sam2", "PIL", "imageio",
    "imageio_ffmpeg", "av", "ffmpeg", "moviepy", "pygame", "matplotlib", "requests", "httpx"})


def _require(condition, message):
    if not condition:
        raise ValueError("Floor calibration worker: " + message)


def _encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _json(raw):
    def unique(items):
        value = {}
        for key, item in items:
            _require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    try:
        value = json.loads(raw, object_pairs_hook=unique)
        json.dumps(value, allow_nan=False)
    except (ValueError, UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError("Floor calibration worker requires finite strict JSON") from error
    _require(type(value) is dict, "JSON object required")
    return value


def _number(value, maximum):
    return type(value) in (int, float) and math.isfinite(value) and 0 < value <= maximum


def _path(value):
    path = Path(value).absolute()
    _require(len(path.parts) <= 64 and ".." not in path.parts
             and not any(p.is_symlink() for p in (path, *path.parents)), "unsafe package path")
    return path


def _parent(root_fd, name):
    _require(type(name) is str and len(name) <= 256 and "\\" not in name and ":" not in name,
             "unsafe member path")
    parts = name.split("/")
    _require(1 <= len(parts) <= 4 and all(p not in ("", ".", "..") for p in parts)
             and all(ord(c) >= 32 for c in name), "unsafe member path")
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        return fd, parts[-1]
    except BaseException:
        os.close(fd)
        raise


def _read(root_fd, name, bound, expected=None, *, allow_empty=False):
    parent, leaf = _parent(root_fd, name)
    try:
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and (0 if allow_empty else 1) <= before.st_size <= bound,
                 "nonregular or oversized input: " + name)
        raw = stream.read(bound + 1)
        after = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    _require(len(raw) == before.st_size and identity(before) == identity(after), "input changed during read: " + name)
    _require(expected is None or sha256(raw).hexdigest() == expected, "file differs from frozen pin: " + name)
    return raw


def _write(root_fd, name, raw, bound):
    _require(type(raw) is bytes and 0 < len(raw) <= bound, "output byte bound")
    parent, leaf = _parent(root_fd, name)
    try:
        fd = os.open(leaf, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
    finally:
        os.close(parent)
    with os.fdopen(fd, "w+b") as stream:
        stream.write(raw); stream.flush(); stream.seek(0)
        _require(stream.read(bound + 1) == raw, "output readback differs")


def _log_stream(root_fd, name):
    parent, leaf = _parent(root_fd, name)
    try:
        fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
    finally:
        os.close(parent)
    return os.fdopen(fd, "wb")


def _plan(raw):
    plan = _json(raw)
    _require(set(plan) == {"schema_version", "profile", "status", "source_sha256", "input_files",
        "runtime_files", "runtime_versions", "settings"} and type(plan["schema_version"]) is int
        and plan["schema_version"] == 1 and plan["profile"] == PLAN_PROFILE and plan["status"] == "frozen",
        "exact frozen worker plan required")
    _require(type(plan["source_sha256"]) is str and _SHA.fullmatch(plan["source_sha256"]), "source SHA required")
    inputs, runtime = plan["input_files"], plan["runtime_files"]
    _require(type(inputs) is dict and "inputs/request.json" in inputs and 2 <= len(inputs) <= 19,
             "bounded exact input inventory required")
    groups = {"observations": [], "measurements": []}
    for name, pin in inputs.items():
        _require(type(pin) is str and _SHA.fullmatch(pin), "raw input SHA required")
        match = _DOCUMENT.fullmatch(name) if type(name) is str else None
        if match:
            _require(match[2] == pin, "document filename must equal its raw SHA")
            groups[match[1]].append(pin)
        else:
            _require(name in {"inputs/request.json", "inputs/camera.json", "inputs/ground.json"}, "unexpected input alias")
    _require(1 <= len(groups["observations"]) <= 8 and len(groups["measurements"]) <= 8,
             "bounded observation/measurement groups required")
    _require(type(runtime) is dict and set(runtime) == RUNTIME_FILES
             and all(type(pin) is str and _SHA.fullmatch(pin) for pin in runtime.values()), "exact child runtime closure required")
    versions = plan["runtime_versions"]
    _require(type(versions) is dict and set(versions) == {"python", "numpy", "opencv"}
             and all(type(v) is str and 0 < len(v) <= 100 for v in versions.values()), "dependency versions required")
    _require(type(plan["settings"]) is dict and set(plan["settings"]) == {"max_seconds"}
             and _number(plan["settings"]["max_seconds"], MAX_CORE_SECONDS), "core deadline must be positive and at most five seconds")
    return plan


def _snapshot(root_fd, pins, total_bound, per_file, check):
    values, total = {}, 0
    for name, pin in sorted(pins.items()):
        check()
        limit = REQUEST_LIMIT if name == "inputs/request.json" else per_file
        raw = _read(root_fd, name, min(limit, total_bound - total), pin)
        values[name] = raw; total += len(raw)
    check()
    return values


def _versions():
    return {"python": sys.version.split()[0], "numpy": importlib.metadata.version("numpy"),
            "opencv": importlib.metadata.version("opencv-contrib-python")}


def _install_limits():
    for kind, wanted in ((resource.RLIMIT_CPU, CPU_SECONDS), (resource.RLIMIT_FSIZE, MAX_OUTPUT_BYTES),
                         (resource.RLIMIT_CORE, 0)):
        soft, hard = resource.getrlimit(kind)
        hard = wanted if hard == resource.RLIM_INFINITY else min(hard, wanted)
        soft = hard if soft == resource.RLIM_INFINITY else min(soft, hard)
        resource.setrlimit(kind, (soft, hard))


class _FrozenFinder:
    def __init__(self, package, files):
        self.package, self.files = package, files

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "weft" and not fullname.startswith("weft."):
            return None
        name = "__init__" if fullname == "weft" else fullname.removeprefix("weft.")
        _require(name in RUNTIME_MODULES, "unlisted Weft import: " + fullname)
        return importlib.util.spec_from_loader(fullname, self, is_package=fullname == "weft")

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = "__init__" if module.__name__ == "weft" else module.__name__.removeprefix("weft.")
        member = f"runtime/weft/{name}.py"
        module.__file__ = str(self.package / member)
        if module.__name__ == "weft":
            module.__path__ = [str(self.package / "runtime/weft")]
        # Execute the verified snapshot, not a second path lookup after admission.
        exec(compile(self.files[member], module.__file__, "exec"), module.__dict__)


class _Guards:
    def __init__(self, deadline):
        self.deadline = deadline
        self.allow_cv2 = False
        self.counts = {"core_calls": 0, "solver_calls": 0, "rodrigues_calls": 0, "forbidden_calls": 0}
        self.denied = {}

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Floor calibration worker exceeded its deadline")

    def forbid(self, category):
        self.counts["forbidden_calls"] += 1
        self.denied[category] = self.denied.get(category, 0) + 1
        raise PermissionError("Floor calibration worker forbids " + category)

    def audit(self, event, args):
        if event.startswith("socket."):
            self.forbid("network")
        if event == "subprocess.Popen" or event in {"os.system", "os.fork", "os.forkpty", "os.posix_spawn", "os.exec", "pty.spawn"}:
            self.forbid("nested process")
        if event == "import" and args:
            self._import_check(args[0])

    def _import_check(self, fullname):
        root = fullname.split(".")[0]
        if root in _DENIED_IMPORTS or root == "cv2" and not self.allow_cv2:
            self.forbid("media/model/physics/provider import" if root != "cv2" else "OpenCV outside planar-target mode")

    def find_spec(self, fullname, path=None, target=None):
        self._import_check(fullname)
        return None

    def numerical(self, cv2):
        cv2.setNumThreads(1)
        for name in ("VideoCapture", "VideoWriter", "imread", "imreadmulti", "imdecode", "imdecodemulti",
                     "imwrite", "imwritemulti", "imencode", "imencodemulti", "imshow", "namedWindow"):
            if hasattr(cv2, name):
                setattr(cv2, name, lambda *a, **k: self.forbid("media processing"))
        original_solve, original_rodrigues = cv2.solvePnPGeneric, cv2.Rodrigues

        def solve(*args, **kwargs):
            self.check()
            if self.counts["solver_calls"] >= 1 or len(args) != 4 or kwargs != {"flags": cv2.SOLVEPNP_IPPE}:
                self.forbid("solver contract or call cap")
            self.counts["solver_calls"] += 1
            result = original_solve(*args, **kwargs)
            self.check()
            return result

        def rodrigues(*args, **kwargs):
            self.check()
            if self.counts["rodrigues_calls"] >= 2:
                self.forbid("Rodrigues call cap")
            self.counts["rodrigues_calls"] += 1
            result = original_rodrigues(*args, **kwargs)
            self.check()
            return result

        cv2.solvePnPGeneric, cv2.Rodrigues = solve, rodrigues


def _load_core(package, runtime, plan, inputs, guard):
    _require(not any(n == "weft" or n.startswith("weft.") for n in sys.modules), "fresh isolated Weft process required")
    _require("cv2" not in sys.modules, "fresh isolated OpenCV state required")
    _require(_versions() == plan["runtime_versions"], "installed dependency versions differ")
    guard.allow_cv2 = _json(inputs["inputs/request.json"]).get("mode") == "planar_target"
    sys.addaudithook(guard.audit)
    sys.meta_path.insert(0, _FrozenFinder(package, runtime))
    sys.meta_path.insert(0, guard)
    if guard.allow_cv2:
        guard.numerical(importlib.import_module("cv2"))
    module = importlib.import_module("weft.floor_target_calibration")
    return module.prepare_floor_target_calibration


def _child_context(package):
    _require(sys.flags.isolated and sys.dont_write_bytecode, "child must use Python -I -B")
    _require(Path(__file__).resolve(strict=True) == package / "runtime/weft/floor_target_calibration_worker.py",
             "execute the frozen worker only")


def _peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _loaded_runtime(package):
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        if name == "weft" or name.startswith("weft."):
            member = Path(module.__file__).relative_to(package).as_posix()
            _require(member in RUNTIME_FILES, "loaded Weft code is outside frozen closure")
            loaded[name] = member
    return loaded


def _child(package_dir, expected_plan_sha256):
    started, root_fd, error = time.monotonic(), None, None
    guard = _Guards(started + MAX_SECONDS)
    output_files, output_bytes, calibration_status, plan, loaded = {}, 0, None, None, {}
    try:
        _install_limits()
        package = _path(package_dir)
        root_fd = os.open(package, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        _child_context(package)
        _require(type(expected_plan_sha256) is str and _SHA.fullmatch(expected_plan_sha256), "exact plan SHA required")
        plan_raw = _read(root_fd, "worker-plan.json", PLAN_LIMIT, expected_plan_sha256)
        plan = _plan(plan_raw)
        for name in CORE_FILES | {"worker.json"}:
            try: os.stat("worker/" + name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError: continue
            raise FileExistsError("Worker output already exists: " + name)
        inputs = _snapshot(root_fd, plan["input_files"], MAX_INPUT_BYTES, MAX_INPUT_BYTES, guard.check)
        runtime = _snapshot(root_fd, plan["runtime_files"], MAX_RUNTIME_BYTES, SOURCE_LIMIT, guard.check)
        core = _load_core(package, runtime, plan, inputs, guard)
        documents = {group: {pin: inputs[name] for name, pin in plan["input_files"].items()
                            if name.startswith("inputs/" + group + "/")} for group in ("observations", "measurements")}
        guard.check(); guard.counts["core_calls"] += 1
        result = core(documents["observations"], source_sha256=plan["source_sha256"],
            request_document=inputs["inputs/request.json"], measurement_documents=documents["measurements"],
            supplied_camera_document=inputs.get("inputs/camera.json"), supplied_ground_document=inputs.get("inputs/ground.json"),
            max_seconds=plan["settings"]["max_seconds"])
        guard.check()
        files = result.files()
        _require(type(files) is dict and set(files) in ({"calibration.json"}, set(CORE_FILES)), "unexpected core output inventory")
        _require(all(type(raw) is bytes and 0 < len(raw) <= MAX_OUTPUT_BYTES for raw in files.values())
                 and sum(map(len, files.values())) <= MAX_OUTPUT_BYTES, "combined core outputs exceed four MiB")
        _require(len(files["calibration.json"]) <= RESULT_LIMIT, "calibration result exceeds one MiB")
        description = _json(files["calibration.json"])
        calibration_status = description.get("status")
        _require(description.get("profile") == "floor-target-calibration-result-v1" and calibration_status in
                 {"qualified_nominal", "unavailable", "rejected", "ambiguous"}, "unknown calibration outcome")
        _require((calibration_status == "qualified_nominal") == (set(files) == CORE_FILES), "qualified-only artifacts required")
        _require(guard.counts["forbidden_calls"] == 0, "forbidden attempt occurred")
        _require(_peak_rss() <= RSS_LIMIT, "observed peak RSS exceeds acceptance bound")
        for name in sorted(files):
            guard.check(); _json(files[name])
            _write(root_fd, "worker/" + name, files[name], RESULT_LIMIT if name == "calibration.json" else MAX_OUTPUT_BYTES)
            output_files["worker/" + name] = sha256(files[name]).hexdigest(); output_bytes += len(files[name])
        # Final byte checks retain custody even though execution used cached exact code and data.
        _snapshot(root_fd, plan["input_files"], MAX_INPUT_BYTES, MAX_INPUT_BYTES, guard.check)
        _snapshot(root_fd, plan["runtime_files"], MAX_RUNTIME_BYTES, SOURCE_LIMIT, guard.check)
        _read(root_fd, "worker-plan.json", PLAN_LIMIT, expected_plan_sha256)
        loaded = _loaded_runtime(package)
        guard.check()
    except (Exception, KeyboardInterrupt) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:1200]}
    receipt = {"schema_version": 1, "profile": PROFILE, "status": "failed" if error else "completed",
        "plan_sha256": expected_plan_sha256, "calibration_status": calibration_status, "error": error,
        "output_files": output_files, "output_bytes": output_bytes, "counts": guard.counts,
        "denied_calls": guard.denied, "seconds": time.monotonic() - started, "peak_rss_bytes": _peak_rss(),
        "runtime_versions": plan["runtime_versions"] if plan else None, "loaded_weft": loaded,
        "limits": {"cpu_seconds_hard": CPU_SECONDS, "core_seconds_cooperative": plan["settings"]["max_seconds"] if plan else None,
            "core_output_bytes": MAX_OUTPUT_BYTES, "observed_peak_rss_bytes": RSS_LIMIT},
        "memory_scope": "Observed process peak RSS, not a hard memory cap",
        "scope": "One pinned trusted numerical call; not an OS sandbox, source authentication or body admission",
        "paid_calls": 0, "external_api_usd": 0, "package_sealed": False}
    try:
        _require(root_fd is not None, "no safe receipt directory")
        _write(root_fd, "worker/worker.json", _encode(receipt), RECEIPT_LIMIT)
    except Exception as exc:
        error = error or {"type": type(exc).__name__, "message": str(exc)[:1200]}
        print(json.dumps({"status": "failed", "worker_receipt_written": False, "error": error}), file=sys.stderr)
    finally:
        if root_fd is not None:
            os.close(root_fd)
    return 1 if error else 0


def _environment(package):
    return {"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "TZ": "America/Los_Angeles"}


def _supervise(command, package, stdout, stderr, seconds):
    started, process, error = time.monotonic(), None, None
    sizes, cleanup_error, reaped = {"stdout": 0, "stderr": 0}, None, True
    with selectors.DefaultSelector() as selector:
        try:
            process = subprocess.Popen(command, cwd=package, env=_environment(package), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, bufsize=0)
            reaped = False
            selector.register(process.stdout, selectors.EVENT_READ, (stdout, "stdout"))
            selector.register(process.stderr, selectors.EVENT_READ, (stderr, "stderr"))
            while selector.get_map() or process.poll() is None:
                remaining = seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Floor calibration worker exceeded its hard wall limit")
                for key, _ in selector.select(min(.05, remaining)):
                    chunk = os.read(key.fileobj.fileno(), 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    stream, channel = key.data
                    capacity = LOG_LIMIT - sum(sizes.values())
                    stream.write(chunk[:capacity]); sizes[channel] += min(capacity, len(chunk))
                    if len(chunk) > capacity:
                        raise ValueError("Floor calibration worker exceeded its combined log bound")
        except (Exception, KeyboardInterrupt) as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:1200]}
        finally:
            if process is not None:
                try:
                    if process.poll() is None:
                        try: os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                        except OSError: process.kill()
                    process.wait(timeout=2)
                    reaped = True
                except (OSError, subprocess.TimeoutExpired) as exc:
                    cleanup_error = {"type": type(exc).__name__, "message": str(exc)[:600]}
                finally:
                    for stream in (process.stdout, process.stderr):
                        if stream is not None: stream.close()
    return {"returncode": process.returncode if process else None, "error": error, "cleanup_error": cleanup_error,
            "pid": process.pid if process else None, "process_reaped": reaped,
            "seconds": time.monotonic() - started, "log_bytes": sizes, "command": command,
            "environment_keys": sorted(_environment(package)), "hard_wall_seconds": seconds,
            "cleanup_seconds_bound": 2, "log_bytes_bound": LOG_LIMIT}


def run_floor_target_worker(package_dir, *, expected_plan_sha256, max_seconds=MAX_SECONDS):
    """Run once in a new worker/ directory; the caller alone seals the outer package."""
    _require(_number(max_seconds, MAX_SECONDS), "parent wall limit must be positive and at most ten seconds")
    _require(type(expected_plan_sha256) is str and _SHA.fullmatch(expected_plan_sha256), "exact plan SHA required")
    package = _path(package_dir)
    root_fd = os.open(package, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        plan = _plan(_read(root_fd, "worker-plan.json", PLAN_LIMIT, expected_plan_sha256))
        script_name = "runtime/weft/floor_target_calibration_worker.py"
        _read(root_fd, script_name, SOURCE_LIMIT, plan["runtime_files"][script_name])
        os.mkdir("worker", mode=0o700, dir_fd=root_fd)
        command = [sys.executable, "-I", "-B", str(package / script_name), str(package), expected_plan_sha256]
        with _log_stream(root_fd, "worker/stdout.log") as stdout, _log_stream(root_fd, "worker/stderr.log") as stderr:
            process = _supervise(command, package, stdout, stderr, max_seconds)
        receipt, output_files, validation_error = None, {}, None
        try:
            receipt_raw = _read(root_fd, "worker/worker.json", RECEIPT_LIMIT)
            output_files["worker/worker.json"] = sha256(receipt_raw).hexdigest()
            receipt = _json(receipt_raw)
            _require(receipt.get("profile") == PROFILE and receipt.get("plan_sha256") == expected_plan_sha256,
                     "worker receipt identity differs")
            _require(receipt.get("status") in {"completed", "failed"}, "worker receipt status missing")
            core_files = receipt.get("output_files")
            _require(type(core_files) is dict and set(core_files) <= {"worker/" + name for name in CORE_FILES},
                     "worker receipt output inventory differs")
            total = 0
            for name, pin in core_files.items():
                _require(type(pin) is str and _SHA.fullmatch(pin), "worker output SHA required")
                raw = _read(root_fd, name, min(RESULT_LIMIT if name.endswith("calibration.json") else MAX_OUTPUT_BYTES,
                                             MAX_OUTPUT_BYTES - total), pin)
                if name == "worker/calibration.json":
                    calibration = _json(raw)
                    _require(calibration.get("profile") == "floor-target-calibration-result-v1"
                             and calibration.get("status") == receipt.get("calibration_status"), "calibration outcome differs from receipt")
                total += len(raw); output_files[name] = pin
            _require(total == receipt.get("output_bytes"), "worker output byte count differs")
            if receipt["status"] == "completed":
                _require(receipt.get("error") is None and receipt.get("counts", {}).get("core_calls") == 1
                         and receipt.get("counts", {}).get("forbidden_calls") == 0
                         and receipt.get("runtime_versions") == plan["runtime_versions"]
                         and type(receipt.get("peak_rss_bytes")) is int and 0 <= receipt["peak_rss_bytes"] <= RSS_LIMIT,
                         "completed worker receipt has failure or resource mismatch")
                expected = CORE_FILES if receipt.get("calibration_status") == "qualified_nominal" else {"calibration.json"}
                _require(receipt.get("calibration_status") in {"qualified_nominal", "unavailable", "rejected", "ambiguous"}
                         and set(core_files) == {"worker/" + name for name in expected}, "completed output status differs")
            process["worker_receipt_sha256"] = sha256(receipt_raw).hexdigest()
        except (OSError, ValueError) as exc:
            validation_error = {"type": type(exc).__name__, "message": str(exc)[:1200]}
        try:
            for channel in ("stdout", "stderr"):
                name = "worker/" + channel + ".log"
                raw = _read(root_fd, name, LOG_LIMIT, allow_empty=True)
                _require(len(raw) == process["log_bytes"][channel], "captured log byte count differs")
                output_files[name] = sha256(raw).hexdigest()
            directory_fd = os.open("worker", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
            try:
                with os.scandir(directory_fd) as entries:
                    seen = set()
                    for entry in entries:
                        _require(len(seen) < 6 and entry.is_file(follow_symlinks=False), "unexpected worker output")
                        seen.add("worker/" + entry.name)
            finally:
                os.close(directory_fd)
            _require(seen == set(output_files), "worker output inventory differs from inspected files")
        except (OSError, ValueError) as exc:
            validation_error = validation_error or {"type": type(exc).__name__, "message": str(exc)[:1200]}
        success = (process["returncode"] == 0 and process["error"] is None and process["cleanup_error"] is None
                   and validation_error is None and receipt["status"] == "completed")
        process.update(schema_version=1, profile=PROCESS_PROFILE, status="completed" if success else "failed",
                       plan_sha256=expected_plan_sha256, validation_error=validation_error, package_sealed=False)
        process_raw = _encode(process)
        _write(root_fd, "worker/process.json", process_raw, RECEIPT_LIMIT)
        output_files["worker/process.json"] = sha256(process_raw).hexdigest()
        if not process["process_reaped"]:
            raise RuntimeError("Floor calibration child could not be reaped; preserve worker/process.json")
        return {"status": process["status"], "calibration_status": receipt.get("calibration_status") if receipt else None,
                "output_files": output_files, "worker_receipt": receipt, "process": process}
    finally:
        os.close(root_fd)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit("Expected frozen package directory and worker-plan SHA256")
    return _child(argv[0], argv[1])


if __name__ == "__main__":
    raise SystemExit(main())
