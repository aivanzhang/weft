"""Run incompatible saved labs in fixed trusted releases, never artifact-supplied code."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import threading
import time
import uuid


TRUSTED_COMMIT = "5819ba452aa65b492cdf586b53dc827045d0d5ba"
TRUSTED_RELEASES = ("64f4bef5d721100edc2a3a347d386dfbbea35c28", TRUSTED_COMMIT)
MAX_REQUEST = 8192
MAX_RESPONSE = 8 << 20
METHODS = frozenset({"summary", "control", "exercise_coverage", "inspection", "recording_inspection",
                     "recording_evidence", "state", "close"})
RUNTIME_MISMATCH = "Exercise physics/evaluator runtime differs; use the pinned implementation or prepare a new lab"
COMPATIBILITY_ERRORS = frozenset({RUNTIME_MISMATCH,
    "Exercise residual policy task, scene, evaluator or runtime differs",
    "Feedback residual source, task, gates or runtime differs"})


def _exporter(project):
    path = Path(project) / "scripts/play_sandbox.py"
    spec = importlib.util.spec_from_file_location("weft_trusted_runtime_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_runtime(project, *, trusted_commit=TRUSTED_COMMIT):
    """Reuse the trusted export's no-follow reads, Git isolation and exclusive publication."""
    project = Path(project).absolute()
    if trusted_commit not in TRUSTED_RELEASES:
        raise ValueError("Portal runtime must be a fixed trusted release")
    exporter = _exporter(project)
    raw = exporter._git(project, "ls-tree", "-rz", "--long", trusted_commit, "--",
                        "src/weft", "pyproject.toml", "requirements-lab.lock", limit=256 << 10)
    rows, total = {}, 0
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        header, name = entry.decode("ascii").split("\t", 1)
        mode, kind, oid, size = header.split()
        exporter._relative(name)
        size = int(size)
        if (mode not in {"100644", "100755"} or kind != "blob" or not re.fullmatch(r"[a-f0-9]{40}", oid)
                or not (name.startswith("src/weft/") or name in {"pyproject.toml", "requirements-lab.lock"})
                or name in rows or not 0 <= size <= 2 << 20):
            raise ValueError("Unsupported trusted portal runtime member")
        rows[name] = (oid, size)
        total += size
        if len(rows) > 1024 or total > 32 << 20:
            raise ValueError("Trusted portal runtime exceeds its export bound")
    if not {"src/weft/__init__.py", "src/weft/lab.py", "src/weft/shared_exercise_lab.py",
            "pyproject.toml", "requirements-lab.lock"} <= rows.keys():
        raise ValueError("Trusted portal runtime lacks required files")
    parent = project / ".local/portal-lab-runtime"
    directory = exporter._directory(parent, create=True)
    runtime = parent / trusted_commit
    try:
        try:
            os.stat(trusted_commit, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            entries = list(rows.values())
            data = b"".join(exporter._git(project, "cat-file", "--batch",
                data=b"".join(oid.encode() + b"\n" for oid, _ in entries[start:start + 250]),
                limit=sum(size for _, size in entries[start:start + 250]) + 250 * 80)
                for start in range(0, len(entries), 250))
            staged = ".stage-" + uuid.uuid4().hex
            os.mkdir(staged, mode=0o700, dir_fd=directory)
            offset = 0
            for name, (oid, size) in rows.items():
                end = data.find(b"\n", offset)
                content = data[end + 1:end + size + 1]
                if (end < 0 or data[offset:end] != f"{oid} blob {size}".encode()
                        or len(content) != size or exporter._blob_hash(content) != oid
                        or data[end + size + 1:end + size + 2] != b"\n"):
                    raise ValueError("Trusted portal runtime blob identity differs")
                offset = end + size + 2
                path = parent / staged / name
                path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(content)
                path.chmod(0o600)
            if offset != len(data):
                raise ValueError("Trusted portal runtime export contains trailing bytes")
            exporter._verify_cache(parent / staged, rows)
            exporter._publish(directory, staged, trusted_commit)
        exporter._verify_cache(runtime, rows)
    finally:
        os.close(directory)
    return runtime


def _sealed_runtime_pins(root, exporter):
    """Metadata selects a release; its unchanged validator still admits the complete lab."""
    root = Path(root).absolute()
    config = exporter._json(exporter._read(root, "lab-config.json", 64 << 10))
    manifest_raw = exporter._read(root, "run/manifest.json", 512 << 10)
    manifest = exporter._json(manifest_raw)
    sealed = manifest.get("files")
    if (config.get("manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest()
            or config.get("control_profile") != "shared_exercise"
            or manifest.get("status") != "ready" or manifest.get("control_profile") != "shared_exercise"
            or not isinstance(sealed, dict) or not 1 <= len(sealed) <= 2048):
        raise ValueError("Archived runtime selection requires an exact sealed exercise lab")

    def document(name, limit):
        raw = exporter._read(root, "run/" + name, limit)
        if hashlib.sha256(raw).hexdigest() != sealed.get(name):
            raise ValueError("Archived runtime selection metadata differs from its seal")
        return exporter._json(raw)

    def pins(value):
        if (not isinstance(value, dict) or not 1 <= len(value) <= 128
                or any(not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*\.py", name)
                       or not isinstance(pin, str) or not re.fullmatch(r"[a-f0-9]{64}", pin)
                       for name, pin in value.items())):
            raise ValueError("Archived runtime selection requires bounded module hashes")
        return value

    contract = document("exercise-lab.json", 128 << 10)
    selected = dict(pins(contract.get("runtime_sha256")))
    implementation = contract.get("implementation_sha256")
    if (not isinstance(implementation, dict) or any(implementation.get(name) != pin
            or sealed.get("lab-implementation/" + name) != pin for name, pin in selected.items())
            or any(config.get(key) != contract.get(key) for key in ("scene_sha256", "task_sha256", "learned_policy_available"))
            or type(contract.get("learned_policy_available")) is not bool):
        raise ValueError("Archived runtime selection contract pins conflict")
    if contract["learned_policy_available"]:
        policy = document("policy.json", 64 << 10)
        policy_pins = pins(policy.get("runtime_sha256"))
        if (policy_pins != contract.get("policy_runtime_sha256")
                or contract.get("policy_sha256") != sealed.get("policy.json")
                or config.get("policy_sha256") != sealed.get("policy.json")
                or any(name in selected and selected[name] != pin for name, pin in policy_pins.items())):
            raise ValueError("Archived runtime selection policy pins conflict")
        selected.update(policy_pins)
    elif "policy.json" in sealed or "policy_runtime_sha256" in contract:
        raise ValueError("Untrained archived lab cannot carry policy pins")
    return selected


def _matching_runtime(package, pins, exporter):
    total = 0
    for name, expected in pins.items():
        try:
            raw = exporter._read(package, name, 2 << 20)
        except FileNotFoundError:
            return False
        total += len(raw)
        if total > 16 << 20:
            raise ValueError("Archived runtime comparison exceeds its byte bound")
        if hashlib.sha256(raw).hexdigest() != expected:
            return False
    return True


def compatible_runtime(root, project):
    exporter = _exporter(project)
    pins = _sealed_runtime_pins(root, exporter)
    if _matching_runtime(Path(__file__).parent, pins, exporter):
        raise ValueError("Saved runtime pins already match current code; compatibility cannot repair invalid evidence")
    for commit in TRUSTED_RELEASES:
        saved = prepare_runtime(project, trusted_commit=commit)
        if _matching_runtime(saved / "src/weft", pins, exporter):
            return saved
    raise ValueError("Saved lab runtime and policy pins match no supported trusted release")


def child_environment(runtime, root):
    return {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(runtime) / "src"), "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MPLCONFIGDIR": str(Path(root) / "runtime/matplotlib")}


class _ExportState:
    def __init__(self, owner):
        self.owner = owner

    def done(self):
        return not self.owner._rpc("state")["exporting"]


class _NoAvatars:
    def resource(self, *args):
        raise ValueError("Archived portal labs have no configured avatar resources")


class ArchivedLab:
    """The HTTP server retains authentication; only closed lab operations cross the pipe."""

    def __init__(self, root, runtime, *, python=sys.executable, startup_seconds=45, request_seconds=15):
        self.root = Path(root).resolve()
        self.runtime_commit = Path(runtime).name
        if self.runtime_commit not in TRUSTED_RELEASES:
            raise ValueError("Archived lab requires a fixed trusted release cache")
        self.lock = threading.RLock()
        self.token = ""
        self.avatars = _NoAvatars()
        self.export_future = _ExportState(self)
        self._request_seconds = request_seconds
        self._sequence = 0
        self._closed = False
        self._buffer = bytearray()
        self._process = subprocess.Popen([str(python), "-P", "-B", str(Path(__file__).absolute()),
            "--worker", str(self.root), "--runtime", str(Path(runtime).absolute())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=runtime, env=child_environment(runtime, self.root), close_fds=True)
        os.set_blocking(self._process.stdout.fileno(), False)
        os.set_blocking(self._process.stdin.fileno(), False)
        try:
            reply = self._receive(time.monotonic() + startup_seconds)
            if reply.get("ready") is not True:
                raise ValueError(reply.get("error", "Archived lab admission failed"))
        except BaseException:
            self._terminate()
            raise

    def _receive(self, deadline):
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            while True:
                end = self._buffer.find(b"\n")
                if end >= 0:
                    raw = bytes(self._buffer[:end])
                    del self._buffer[:end + 1]
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise ValueError("Invalid archived lab response")
                    return value
                if len(self._buffer) > MAX_RESPONSE:
                    raise ValueError("Archived lab response exceeds its bound")
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("Archived lab response timed out; worker stopped")
                block = os.read(self._process.stdout.fileno(), min(65536, MAX_RESPONSE + 1 - len(self._buffer)))
                if not block:
                    raise RuntimeError("Archived lab worker stopped unexpectedly")
                self._buffer.extend(block)

    def _rpc(self, method, *args):
        if method not in METHODS:
            raise ValueError("Unsupported archived lab operation")
        with self.lock:
            if self._closed:
                raise RuntimeError("Archived lab worker is closed")
            self._sequence += 1
            raw = (json.dumps({"id": self._sequence, "method": method, "args": args}, allow_nan=False) + "\n").encode()
            if len(raw) > MAX_REQUEST:
                raise ValueError("Archived lab request exceeds its bound")
            deadline = time.monotonic() + self._request_seconds
            try:
                offset = 0
                with selectors.DefaultSelector() as selector:
                    selector.register(self._process.stdin, selectors.EVENT_WRITE)
                    while offset < len(raw):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not selector.select(remaining):
                            raise TimeoutError("Archived lab request timed out; worker stopped")
                        offset += os.write(self._process.stdin.fileno(), raw[offset:])
                response = self._receive(deadline)
                if response.get("id") != self._sequence:
                    raise ValueError("Archived lab response identity differs")
            except BaseException:
                self._terminate()
                raise
            if "error" in response:
                raise ValueError(response["error"])
            return response["result"]

    @property
    def recorder(self):
        return True if self._rpc("state")["recording"] else None

    def summary(self):
        return {**self._rpc("summary"), "execution_runtime": {
            "mode": "preserved_release", "git_commit": self.runtime_commit,
            "scope": "Original simulation and scoring runtime; saved results are not migrated."}}
    def control(self, path, data): return self._rpc("control", path, data)
    def exercise_coverage(self): return self._rpc("exercise_coverage")
    def inspection(self): return self._rpc("inspection")
    def recording_inspection(self, name): return self._rpc("recording_inspection", name)
    def recording_evidence(self, name): return self._rpc("recording_evidence", name)

    def _terminate(self):
        self._closed = True
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        self._process.stdin.close()
        self._process.stdout.close()

    def close(self):
        with self.lock:
            if not self._closed:
                try:
                    self._rpc("close")
                finally:
                    self._terminate()


def open_portal_lab(root, *, project):
    from weft.lab import Lab
    current = Lab(root)
    try:
        current.load_session()
        return current
    except ValueError as error:
        current.close()
        if str(error) not in COMPATIBILITY_ERRORS:
            raise
    except BaseException:
        current.close()
        raise
    return ArchivedLab(root, compatible_runtime(root, project))


def _worker(root, runtime):
    protocol = sys.stdout
    sys.stdout = sys.stderr
    lab = None

    def emit(value):
        raw = json.dumps(value, allow_nan=False, separators=(",", ":"))
        if len(raw.encode()) > MAX_RESPONSE:
            raise ValueError("Archived lab response exceeds its bound")
        protocol.write(raw + "\n")
        protocol.flush()

    try:
        import weft.lab
        if Path(weft.lab.__file__).absolute() != Path(runtime).absolute() / "src/weft/lab.py":
            raise ValueError("Archived lab imported outside its trusted runtime")
        lab = weft.lab.Lab(root)
        lab.load_session()
        emit({"ready": True})
        while True:
            raw = sys.stdin.buffer.readline(MAX_REQUEST + 1)
            if not raw:
                break
            if len(raw) > MAX_REQUEST or not raw.endswith(b"\n"):
                raise ValueError("Archived lab request exceeds its bound")
            request = json.loads(raw)
            if (not isinstance(request, dict) or set(request) != {"id", "method", "args"}
                    or type(request["id"]) is not int or request["method"] not in METHODS
                    or not isinstance(request["args"], list)):
                raise ValueError("Unsupported archived lab request")
            method = request["method"]
            try:
                if method == "close":
                    lab.close()
                    lab = None
                    emit({"id": request["id"], "result": None})
                    break
                with lab.lock:
                    if method == "state":
                        result = {"recording": lab.recorder is not None,
                                  "exporting": lab.export_future is not None and not lab.export_future.done()}
                    else:
                        result = getattr(lab, method)(*request["args"])
                emit({"id": request["id"], "result": result})
            except (ValueError, OSError, RuntimeError, TypeError) as error:
                emit({"id": request["id"], "error": str(error)[:4096]})
    except Exception as error:
        emit({"ready": False, "error": str(error)[:4096]})
    finally:
        if lab is not None:
            lab.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--runtime", required=True)
    args = parser.parse_args()
    _worker(args.worker, args.runtime)
