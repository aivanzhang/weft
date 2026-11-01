"""Authored bytes and fake children only; no OpenCV, source data or process runs."""

from copy import deepcopy
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

from weft import floor_target_calibration_worker as worker


def raw(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def pin(value):
    return sha256(value).hexdigest()


@pytest.fixture(autouse=True)
def no_real_work(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("A real subprocess or numerical core was reached")
    monkeypatch.setattr(worker.subprocess, "Popen", blocked)
    monkeypatch.setattr(worker, "_load_core", blocked)
    monkeypatch.setattr(worker, "_install_limits", lambda: None)
    monkeypatch.setattr(worker, "_child_context", lambda package: None)
    monkeypatch.setattr(worker, "_loaded_runtime", lambda package: {})
    monkeypatch.setattr(worker, "_peak_rss", lambda: 48 << 20)


def package_fixture(tmp_path, *, child=True, mode="supplied"):
    package = tmp_path / "package"; package.mkdir()
    observations = [raw({"authored_observation": index}) for index in (2, 1)]
    measurements = [raw({"authored_measurement": True})]
    request = raw({"mode": mode, "observation_sha256s": [pin(v) for v in observations],
                   "measurement_sha256s": [pin(v) for v in measurements]})
    inputs = {"inputs/request.json": request, "inputs/camera.json": raw({"authored_camera": True}),
              "inputs/ground.json": raw({"authored_ground": True}),
              **{f"inputs/observations/{pin(v)}.json": v for v in observations},
              **{f"inputs/measurements/{pin(v)}.json": v for v in measurements}}
    runtime = {name: b'"""Authored worker dependency stand-in."""\n' for name in worker.RUNTIME_FILES}
    runtime["runtime/weft/floor_target_calibration_worker.py"] = Path(worker.__file__).read_bytes()
    for name, value in {**inputs, **runtime}.items():
        destination = package / name; destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(value)
    versions = {"python": sys.version.split()[0], "numpy": "authored", "opencv": "authored"}
    plan = {"schema_version": 1, "profile": worker.PLAN_PROFILE, "status": "frozen", "source_sha256": "a" * 64,
            "input_files": {name: pin(v) for name, v in inputs.items()},
            "runtime_files": {name: pin(v) for name, v in runtime.items()}, "runtime_versions": versions,
            "settings": {"max_seconds": 5.}}
    (package / "worker-plan.json").write_bytes(raw(plan))
    if child: (package / "worker").mkdir()
    return package, plan, pin(raw(plan)), inputs


def fake_core(monkeypatch, *, status="qualified_nominal", callback=None):
    calls = []
    files = {"calibration.json": raw({"schema_version": 1, "profile": "floor-target-calibration-result-v1", "status": status})}
    if status == "qualified_nominal":
        files.update({"camera.json": raw({"raw_authored_camera": True}), "ground.json": raw({"raw_authored_ground": True})})
    def invoke(*args, **kwargs):
        calls.append((args, kwargs))
        if callback: callback(files)
        return SimpleNamespace(files=lambda: files)
    monkeypatch.setattr(worker, "_load_core", lambda *a: invoke)
    return calls, files


@pytest.mark.parametrize("status", ["qualified_nominal", "unavailable", "rejected", "ambiguous"])
def test_one_core_call_preserves_exact_bytes_and_non_success_outcomes(tmp_path, monkeypatch, status):
    package, plan, plan_pin, inputs = package_fixture(tmp_path)
    calls, files = fake_core(monkeypatch, status=status)
    assert worker._child(package, plan_pin) == 0
    receipt = json.loads((package / "worker/worker.json").read_bytes())
    assert receipt["status"] == "completed" and receipt["calibration_status"] == status
    assert receipt["counts"] == {"core_calls": 1, "solver_calls": 0, "rodrigues_calls": 0, "forbidden_calls": 0}
    assert receipt["package_sealed"] is False and receipt["external_api_usd"] == 0
    assert receipt["runtime_versions"] == plan["runtime_versions"]
    assert len(calls) == 1
    assert calls[0][0] == ({pin(value): value for name, value in inputs.items() if "/observations/" in name},)
    assert calls[0][1] == {"source_sha256": plan["source_sha256"], "request_document": inputs["inputs/request.json"],
        "measurement_documents": {pin(value): value for name, value in inputs.items() if "/measurements/" in name},
        "supplied_camera_document": inputs["inputs/camera.json"], "supplied_ground_document": inputs["inputs/ground.json"],
        "max_seconds": 5.}
    assert receipt["output_files"] == {"worker/" + name: pin(value) for name, value in files.items()}
    assert all((package / "worker" / name).read_bytes() == value for name, value in files.items())
    assert all((package / name).read_bytes() == value for name, value in inputs.items())
    assert not (package / "manifest.json").exists()


@pytest.mark.parametrize("change", [
    lambda p: p.update(schema_version=True),
    lambda p: p["settings"].update(max_seconds=5.01),
    lambda p: p["settings"].update(max_seconds=True),
    lambda p: p["runtime_files"].update({"runtime/weft/human.py": "b" * 64}),
    lambda p: p["input_files"].update({"inputs/observations/" + "b" * 64 + ".json": "c" * 64}),
    lambda p: p["input_files"].update({"inputs/../source.mp4": "b" * 64}),
])
def test_plan_rejects_ambiguous_or_unbounded_inputs_before_core(tmp_path, change):
    package, plan, _, _ = package_fixture(tmp_path)
    change(plan); (package / "worker-plan.json").write_bytes(raw(plan))
    assert worker._child(package, pin(raw(plan))) == 1
    receipt = json.loads((package / "worker/worker.json").read_bytes())
    assert receipt["counts"]["core_calls"] == 0 and receipt["status"] == "failed"


@pytest.mark.parametrize("name", ["inputs/camera.json", "runtime/weft/camera_pose.py"])
def test_preparation_hash_mismatch_never_reaches_core(tmp_path, name):
    package, _, plan_pin, _ = package_fixture(tmp_path)
    (package / name).write_bytes(b"changed")
    assert worker._child(package, plan_pin) == 1
    receipt = json.loads((package / "worker/worker.json").read_bytes())
    assert "frozen pin" in receipt["error"]["message"] and receipt["counts"]["core_calls"] == 0


def test_aggregate_budget_applies_before_oversized_observation_read(tmp_path, monkeypatch):
    package, plan, _, _ = package_fixture(tmp_path)
    oversized = b" " * worker.MAX_INPUT_BYTES
    old = next(name for name in plan["input_files"] if "/observations/" in name)
    del plan["input_files"][old]
    name = f"inputs/observations/{pin(oversized)}.json"
    (package / name).write_bytes(oversized); plan["input_files"][name] = pin(oversized)
    (package / "worker-plan.json").write_bytes(raw(plan))
    original_read, calls = worker._read, []
    def observed(*args, **kwargs):
        if args[1] == name: calls.append(args[2])
        return original_read(*args, **kwargs)
    monkeypatch.setattr(worker, "_read", observed)
    assert worker._child(package, pin(raw(plan))) == 1
    assert len(calls) == 1 and calls[0] < len(oversized)
    assert "oversized input" in json.loads((package / "worker/worker.json").read_bytes())["error"]["message"]


def test_symlink_and_existing_outputs_never_get_overwritten(tmp_path):
    package, _, plan_pin, _ = package_fixture(tmp_path)
    sentinel = tmp_path / "sentinel.json"; sentinel.write_bytes(b"preserve")
    (package / "worker/calibration.json").symlink_to(sentinel)
    assert worker._child(package, plan_pin) == 1
    assert sentinel.read_bytes() == b"preserve" and (package / "worker/calibration.json").is_symlink()
    receipt = (package / "worker/worker.json").read_bytes()
    assert worker._child(package, plan_pin) == 1
    assert (package / "worker/worker.json").read_bytes() == receipt


@pytest.mark.parametrize("member", ["inputs/ground.json", "runtime/weft/io.py", "worker-plan.json"])
def test_final_recheck_retains_failure_and_partial_core_outputs(tmp_path, monkeypatch, member):
    package, _, plan_pin, _ = package_fixture(tmp_path)
    calls, _ = fake_core(monkeypatch, callback=lambda _: (package / member).write_bytes(b"changed after preparation"))
    assert worker._child(package, plan_pin) == 1
    receipt = json.loads((package / "worker/worker.json").read_bytes())
    assert len(calls) == 1 and receipt["status"] == "failed" and "frozen pin" in receipt["error"]["message"]
    assert len(receipt["output_files"]) == 3 and not (package / "manifest.json").exists()


def test_output_bounds_and_observed_rss_fail_without_publishing_success(tmp_path, monkeypatch):
    package, _, plan_pin, _ = package_fixture(tmp_path)
    fake_core(monkeypatch, callback=lambda files: files.update({"calibration.json": b" " * (worker.RESULT_LIMIT + 1)}))
    assert worker._child(package, plan_pin) == 1
    receipt = json.loads((package / "worker/worker.json").read_bytes())
    assert "one MiB" in receipt["error"]["message"] and not receipt["output_files"]
    other = tmp_path / "rss"; other.mkdir()
    package, _, plan_pin, _ = package_fixture(other)
    fake_core(monkeypatch); monkeypatch.setattr(worker, "_peak_rss", lambda: worker.RSS_LIMIT + 1)
    assert worker._child(package, plan_pin) == 1
    receipt = json.loads((package / "worker/worker.json").read_bytes())
    assert "RSS" in receipt["error"]["message"] and "not a hard memory" in receipt["memory_scope"]


def test_frozen_loader_executes_cached_code_and_denies_unlisted_weft(tmp_path):
    member = "runtime/weft/io.py"
    finder = worker._FrozenFinder(tmp_path, {member: b"VALUE = 'cached'\n"})
    module = importlib.util.module_from_spec(finder.find_spec("weft.io"))
    finder.exec_module(module)
    assert module.VALUE == "cached"
    with pytest.raises(ValueError, match="unlisted Weft"):
        finder.find_spec("weft.human")


def test_guards_deny_process_network_media_and_count_only_one_ippe():
    guard = worker._Guards(worker.time.monotonic() + 10)
    for event, args in (("subprocess.Popen", ()), ("socket.__new__", ()), ("os.exec", ()),
                        ("import", ("mujoco",)), ("import", ("torch",))):
        with pytest.raises(PermissionError): guard.audit(event, args)
    called = []
    cv2 = SimpleNamespace(SOLVEPNP_IPPE=6, setNumThreads=lambda value: called.append(("threads", value)),
        solvePnPGeneric=lambda *a, **k: called.append("solve"), Rodrigues=lambda *a, **k: called.append("rotation"),
        imread=lambda *a: pytest.fail("Decoder reached"))
    guard.numerical(cv2)
    with pytest.raises(PermissionError): cv2.imread("never-read.png")
    cv2.solvePnPGeneric(1, 2, 3, 4, flags=6)
    with pytest.raises(PermissionError): cv2.solvePnPGeneric(1, 2, 3, 4, flags=6)
    assert called == [("threads", 1), "solve"] and guard.counts["solver_calls"] == 1
    assert guard.counts["forbidden_calls"] == 7


def test_wrong_solver_flag_is_rejected_before_numerical_call():
    guard = worker._Guards(worker.time.monotonic() + 10)
    cv2 = SimpleNamespace(SOLVEPNP_IPPE=6, setNumThreads=lambda value: None,
        solvePnPGeneric=lambda *a, **k: pytest.fail("Wrong solver reached"), Rodrigues=lambda *a, **k: None)
    guard.numerical(cv2)
    with pytest.raises(PermissionError): cv2.solvePnPGeneric(1, 2, 3, 4, flags=0)
    assert guard.counts["solver_calls"] == 0


def test_supplied_mode_denies_even_an_available_fake_cv2_import(monkeypatch):
    guard = worker._Guards(worker.time.monotonic() + 10)
    reached = []
    class FakeOpenCV:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "cv2":
                reached.append(fullname)
                return importlib.util.spec_from_loader(fullname, self)
        def create_module(self, spec): return None
        def exec_module(self, module): module.authored_fake = True
    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    monkeypatch.setattr(sys, "meta_path", [guard, FakeOpenCV(), *sys.meta_path])
    with pytest.raises(PermissionError, match="outside planar-target"):
        importlib.import_module("cv2")
    assert reached == [] and guard.counts["solver_calls"] == 0
    guard.allow_cv2 = True
    try:
        assert importlib.import_module("cv2").authored_fake is True
        assert reached == ["cv2"]
    finally:
        sys.modules.pop("cv2", None)


class Pipe:
    def __init__(self, descriptor): self.descriptor, self.closed = descriptor, False
    def fileno(self): return self.descriptor
    def close(self): self.closed = True


class FakeSelector:
    def __init__(self): self.mapping = {}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def register(self, stream, events, data): self.mapping[stream] = SimpleNamespace(fileobj=stream, data=data)
    def unregister(self, stream): del self.mapping[stream]
    def get_map(self): return self.mapping
    def select(self, seconds): return [(key, 1) for key in list(self.mapping.values())]


def fake_process(monkeypatch, package, plan, plan_pin, *, chunks=None, active=False, tamper=False, no_receipt=False):
    process = SimpleNamespace(pid=876543, stdout=Pipe(61), stderr=Pipe(62), returncode=None if active else 0)
    process.waits, process.kills, launches = [], [], []
    process.poll = lambda: process.returncode
    def wait(timeout):
        process.waits.append(timeout)
        return process.returncode
    process.wait = wait
    def killed(pid, sig):
        process.kills.append((pid, sig)); process.returncode = -9
    monkeypatch.setattr(worker.os, "killpg", killed)
    def start(command, **kwargs):
        launches.append((command, kwargs))
        if not no_receipt:
            output = raw({"profile": "floor-target-calibration-result-v1", "status": "unavailable"})
            (package / "worker/calibration.json").write_bytes(output)
            receipt = {"profile": worker.PROFILE, "plan_sha256": plan_pin, "status": "completed", "error": None,
                "calibration_status": "unavailable", "output_files": {"worker/calibration.json": pin(output)},
                "output_bytes": len(output), "counts": {"core_calls": 1, "forbidden_calls": 0},
                "peak_rss_bytes": 48 << 20, "runtime_versions": plan["runtime_versions"]}
            (package / "worker/worker.json").write_bytes(raw(receipt))
            if tamper: (package / "worker/calibration.json").write_bytes(b"changed")
        return process
    queues = {61: list((chunks or {}).get(61, [b"authored fake output\n", b""])), 62: list((chunks or {}).get(62, [b""]))}
    monkeypatch.setattr(worker.subprocess, "Popen", start)
    monkeypatch.setattr(worker.selectors, "DefaultSelector", FakeSelector)
    monkeypatch.setattr(worker.os, "read", lambda fd, bound: queues[fd].pop(0) if queues[fd] else b"")
    return process, launches


def test_parent_returns_all_written_pins_after_reap_and_never_seals_package(tmp_path, monkeypatch):
    package, plan, plan_pin, _ = package_fixture(tmp_path, child=False)
    process, launches = fake_process(monkeypatch, package, plan, plan_pin)
    monkeypatch.setenv("SECRET_TEST_KEY", "never-forward")
    result = worker.run_floor_target_worker(package, expected_plan_sha256=plan_pin)
    assert result["status"] == "completed" and result["calibration_status"] == "unavailable"
    expected = {"worker/" + path.name: pin(path.read_bytes()) for path in (package / "worker").iterdir()}
    assert result["output_files"] == expected and len(expected) == 5
    assert process.waits == [2] and not process.kills and process.stdout.closed and process.stderr.closed
    command, options = launches[0]
    assert command[1:3] == ["-I", "-B"] and command[-1] == plan_pin and options["start_new_session"] is True
    assert "SECRET_TEST_KEY" not in options["env"] and "HOME" not in options["env"] and "PYTHONPATH" not in options["env"]
    assert not (package / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        worker.run_floor_target_worker(package, expected_plan_sha256=plan_pin)
    assert len(launches) == 1


def test_parent_bad_plan_or_worker_pin_fails_before_spawn(tmp_path):
    package, _, plan_pin, _ = package_fixture(tmp_path, child=False)
    with pytest.raises(ValueError, match="frozen pin"):
        worker.run_floor_target_worker(package, expected_plan_sha256="b" * 64)
    (package / "runtime/weft/floor_target_calibration_worker.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen pin"):
        worker.run_floor_target_worker(package, expected_plan_sha256=plan_pin)
    assert not (package / "worker").exists()


def test_parent_timeout_kills_reaps_and_retains_missing_receipt_failure(tmp_path, monkeypatch):
    package, plan, plan_pin, _ = package_fixture(tmp_path, child=False)
    process, launches = fake_process(monkeypatch, package, plan, plan_pin, active=True, no_receipt=True)
    now = [0.]
    def clock(): now[0] += .1; return now[0]
    monkeypatch.setattr(worker.time, "monotonic", clock)
    result = worker.run_floor_target_worker(package, expected_plan_sha256=plan_pin, max_seconds=.5)
    assert result["status"] == "failed" and result["worker_receipt"] is None
    assert result["process"]["error"]["type"] == "TimeoutError"
    assert process.kills == [(process.pid, signal.SIGKILL)] and process.waits == [2] and len(launches) == 1
    assert (package / "worker/process.json").exists()


def test_parent_log_overflow_preserves_only_bounded_prefix_and_reaps(tmp_path, monkeypatch):
    package, plan, plan_pin, _ = package_fixture(tmp_path, child=False)
    chunks = {61: [b"x" * 8192] * (worker.LOG_LIMIT // 8192 + 1), 62: [b""]}
    process, _ = fake_process(monkeypatch, package, plan, plan_pin, chunks=chunks, active=True, no_receipt=True)
    result = worker.run_floor_target_worker(package, expected_plan_sha256=plan_pin)
    assert result["status"] == "failed" and "log bound" in result["process"]["error"]["message"]
    assert (package / "worker/stdout.log").stat().st_size == worker.LOG_LIMIT
    assert sum(result["process"]["log_bytes"].values()) == worker.LOG_LIMIT and process.kills and process.waits == [2]


def test_parent_rejects_tampered_child_output_without_changing_it(tmp_path, monkeypatch):
    package, plan, plan_pin, _ = package_fixture(tmp_path, child=False)
    fake_process(monkeypatch, package, plan, plan_pin, tamper=True)
    result = worker.run_floor_target_worker(package, expected_plan_sha256=plan_pin)
    assert result["status"] == "failed" and "frozen pin" in result["process"]["validation_error"]["message"]
    assert (package / "worker/calibration.json").read_bytes() == b"changed"


def test_capture_read_failure_still_kills_and_reaps(tmp_path, monkeypatch):
    package, plan, plan_pin, _ = package_fixture(tmp_path, child=False)
    process, _ = fake_process(monkeypatch, package, plan, plan_pin, active=True, no_receipt=True)
    def broken(*args): raise OSError("Authored read failure")
    monkeypatch.setattr(worker.os, "read", broken)
    result = worker.run_floor_target_worker(package, expected_plan_sha256=plan_pin)
    assert result["status"] == "failed" and result["process"]["error"]["type"] == "OSError"
    assert process.kills and process.waits == [2] and process.stdout.closed and process.stderr.closed


def test_log_parent_symlink_cannot_escape_package(tmp_path):
    package = tmp_path / "package"; package.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (package / "worker").symlink_to(outside, target_is_directory=True)
    fd = os.open(package, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(OSError): worker._log_stream(fd, "worker/stdout.log")
    finally:
        os.close(fd)
    assert not list(outside.iterdir())


def test_group_kill_failure_falls_back_to_owned_child_and_reaps(tmp_path, monkeypatch):
    package, plan, plan_pin, _ = package_fixture(tmp_path, child=False)
    process, _ = fake_process(monkeypatch, package, plan, plan_pin, active=True, no_receipt=True)
    def broken(*args): raise OSError("Authored group lookup failure")
    def kill(): process.kills.append("owned child"); process.returncode = -9
    process.kill = kill
    monkeypatch.setattr(worker.os, "read", broken)
    monkeypatch.setattr(worker.os, "killpg", broken)
    result = worker.run_floor_target_worker(package, expected_plan_sha256=plan_pin)
    assert result["status"] == "failed" and result["process"]["process_reaped"] is True
    assert process.kills == ["owned child"] and process.waits == [2]


def test_failed_reaping_is_preserved_and_never_returns_as_terminal_success(tmp_path, monkeypatch):
    package, plan, plan_pin, _ = package_fixture(tmp_path, child=False)
    process, _ = fake_process(monkeypatch, package, plan, plan_pin, no_receipt=True)
    def unavailable(timeout): raise worker.subprocess.TimeoutExpired("authored fake child", timeout)
    process.wait = unavailable
    with pytest.raises(RuntimeError, match="could not be reaped"):
        worker.run_floor_target_worker(package, expected_plan_sha256=plan_pin)
    saved = json.loads((package / "worker/process.json").read_bytes())
    assert saved["status"] == "failed" and saved["process_reaped"] is False
    assert saved["pid"] == process.pid and process.stdout.closed and process.stderr.closed
