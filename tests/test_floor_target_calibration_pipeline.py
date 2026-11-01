"""Package custody and terminal behavior; no numerical solver is run here."""

from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


def encoded(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    boundary = ModuleType("weft.floor_target_calibration_worker")
    boundary.RUNTIME_FILES = frozenset({"runtime/weft/__init__.py", "runtime/weft/camera_pose.py"})
    boundary.run_floor_target_worker = lambda *a, **k: pytest.fail("Unexpected numerical worker")
    monkeypatch.setitem(sys.modules, boundary.__name__, boundary)
    from weft import floor_target_calibration_pipeline as pipeline
    source = tmp_path / "source"; source.mkdir()
    observation = source / "observations.json"; observation.write_bytes(b'{"authored_observation":true}\n')
    measurement = source / "measurements.json"; measurement.write_bytes(b'{"authored_measurement":true}\n')
    request = source / "request.json"; request.write_bytes(b'{"authored_request":true}\n')
    runtime = tmp_path / "runtime-source"; runtime.mkdir()
    names = {name.removeprefix("runtime/weft/") for name in pipeline.RUNTIME_FILES} | set(pipeline.EXTRA_RUNTIME_FILES)
    for name in names:
        path = runtime / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"/* retained style */\n" if name == "assets/brand.css" else b"# exact stub runtime\n")
    monkeypatch.setattr(pipeline, "RUNTIME_ROOT", runtime)
    monkeypatch.setattr(pipeline, "version", lambda name: "fixture-version")
    calls, report_calls = [], []
    controls = {"status": "qualified_nominal", "worker_status": "completed", "omit_ground": False,
                "wrong_hash": False, "extra_file": False, "exception": False, "report_css": None}

    def worker(out, **kwargs):
        calls.append((out, kwargs))
        worker_dir = out / "worker"; worker_dir.mkdir()
        metadata = {"schema_version": 1, "profile": "floor-target-calibration-result-v1",
                    "status": controls["status"], "counts": {"retained_frames": 12}, "fixture": "mocked numerical boundary"}
        files = {"calibration.json": encoded(metadata), "worker.json": b'{"fixture":true}\n',
                 "stdout.log": b"", "stderr.log": b"", "process.json": b'{"returncode":0}\n'}
        if controls["status"] == "qualified_nominal":
            files["camera.json"] = b'{"fixture_camera":true}\n'
            if not controls["omit_ground"]:
                files["ground.json"] = b'{"fixture_ground":true}\n'
        for name, raw in files.items(): (worker_dir / name).write_bytes(raw)
        if controls["extra_file"]: (worker_dir / "unreported.json").write_bytes(b"{}")
        if controls["exception"]: raise TimeoutError("authored worker timeout")
        pins = {"worker/" + name: sha256(raw).hexdigest() for name, raw in files.items()}
        if controls["wrong_hash"]: pins["worker/calibration.json"] = "f" * 64
        return {"status": controls["worker_status"], "calibration_status": controls["status"], "output_files": pins,
                "worker_receipt": {}, "process": {"returncode": 0}}

    def report(files, *, max_seconds):
        report_calls.append((files, max_seconds))
        css = controls["report_css"] or b"/* retained style */\n"
        return {"index.html": b"<!doctype html><title>Authored report boundary</title>", "brand.css": css}

    monkeypatch.setattr(pipeline, "run_floor_target_worker", worker)
    reporter = ModuleType("weft.floor_target_calibration_report")
    reporter.build_floor_target_calibration_report = report
    monkeypatch.setitem(sys.modules, reporter.__name__, reporter)
    target = tmp_path / "output"

    def invoke(**options):
        defaults = {"source_sha256": "a" * 64, "request": request, "measurements": [measurement]}
        defaults.update(options)
        return pipeline.prepare_floor_target_calibration_package([observation], target, **defaults)

    return pipeline, invoke, target, source, runtime, calls, report_calls, controls


def test_one_worker_retains_inputs_runtime_evidence_and_exact_nominal_downloads(setup):
    pipeline, invoke, out, source, runtime, calls, reports, _ = setup
    result = invoke()
    assert result["status"] == "qualified_nominal" and result["sealed"] is True
    assert len(calls) == len(reports) == 1
    plan_raw = (out / "worker-plan.json").read_bytes()
    assert calls[0][1]["expected_plan_sha256"] == sha256(plan_raw).hexdigest()
    assert 0 < calls[0][1]["max_seconds"] <= 10
    plan = json.loads(plan_raw)
    assert plan["settings"] == {"max_seconds": 5.}
    assert set(plan["runtime_files"]) == set(pipeline.RUNTIME_FILES)
    assert plan["source_sha256"] == "a" * 64
    inventory = json.loads((out / "input-files.json").read_bytes())
    assert [row["role"] for row in inventory["files"]] == ["observation", "measurement", "request"]
    for row, original in zip(inventory["files"], ("observations.json", "measurements.json", "request.json"), strict=True):
        assert (out / row["path"]).read_bytes() == (source / original).read_bytes()
        assert row["sha256"] == sha256((source / original).read_bytes()).hexdigest()
    for name in ("calibration.json", "camera.json", "ground.json"):
        assert (out / name).read_bytes() == (out / "worker" / name).read_bytes()
    assert set(reports[0][0]) == {"calibration.json", "camera.json", "ground.json", "inputs/request.json", "input-files.json"}
    assert 0 < reports[0][1] <= 5
    assert json.loads((out / "result.json").read_bytes())["sealed"] is False
    manifest_raw = (out / "manifest.json").read_bytes(); manifest = json.loads(manifest_raw)
    assert result["manifest_sha256"] == sha256(manifest_raw).hexdigest()
    assert manifest["status"] == "qualified_nominal"
    actual = {p.relative_to(out).as_posix(): sha256(p.read_bytes()).hexdigest()
              for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
    assert manifest["files"] == actual
    assert "worker/stdout.log" in actual and "worker/stderr.log" in actual
    assert not (out / "failure.json").exists()


@pytest.mark.parametrize("status", ["unavailable", "rejected", "ambiguous"])
def test_no_qualification_still_seals_inspection_without_nominal_pair(setup, status):
    _, invoke, out, _, _, calls, reports, controls = setup
    controls["status"] = status
    result = invoke()
    assert result["status"] == status and result["sealed"] is True
    assert result["camera_path"] is result["ground_path"] is None
    assert len(calls) == len(reports) == 1
    assert not (out / "camera.json").exists() and not (out / "ground.json").exists()
    assert "camera.json" not in reports[0][0]


@pytest.mark.parametrize("change", ["omit_ground", "wrong_hash", "extra_file", "exception"])
def test_worker_failure_or_foreign_output_is_preserved_without_retry_or_seal(setup, change):
    _, invoke, out, _, _, calls, _, controls = setup
    controls[change] = True
    result = invoke()
    assert result["status"] == "failed" and result["sealed"] is False
    assert len(calls) == 1 and not (out / "manifest.json").exists()
    assert (out / "worker/calibration.json").is_file() and (out / "failure.json").is_file()
    assert (out / "inputs/request.json").is_file()


def test_failed_process_cannot_be_promoted_by_a_present_calibration(setup):
    _, invoke, out, _, _, calls, reports, controls = setup
    controls["worker_status"] = "failed"
    result = invoke()
    assert result["status"] == "failed" and len(calls) == 1 and not reports
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize("which", ["source", "runtime"])
def test_changed_original_input_or_runtime_prevents_terminal_publication(setup, which):
    _, invoke, out, source, runtime, _, _, _ = setup
    def change(stage, detail):
        if stage == "seal":
            path = source / "observations.json" if which == "source" else runtime / "camera_pose.py"
            path.write_bytes(b"changed during preparation")
    result = invoke(progress=change)
    assert result["status"] == "failed" and not (out / "manifest.json").exists()
    assert "changed before publication" in result["error"]["message"]


def test_report_must_use_exact_archived_brand_asset(setup):
    _, invoke, out, _, _, _, _, controls = setup
    controls["report_css"] = b"unrecorded style"
    result = invoke()
    assert result["status"] == "failed" and "stylesheet" in result["error"]["message"]
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize("role,size", [("observations.json", (4 << 20) + 1), ("request.json", (64 << 10) + 1)])
def test_input_bounds_stop_before_worker_without_trimming(setup, role, size):
    _, invoke, out, source, _, calls, _, _ = setup
    (source / role).write_bytes(b" " * size)
    result = invoke()
    assert result["status"] == "failed" and not calls
    assert not (out / "manifest.json").exists()


def test_terminal_publication_is_not_relabelled_after_late_writer_failure(setup, monkeypatch):
    pipeline, invoke, out, _, _, _, _, _ = setup
    write = pipeline.write_package_files
    def late(directory, files):
        write(directory, files)
        if "manifest.json" in files: raise OSError("late caller I/O failure")
    monkeypatch.setattr(pipeline, "write_package_files", late)
    with pytest.raises(OSError, match="late caller"):
        invoke()
    assert json.loads((out / "manifest.json").read_bytes())["status"] == "qualified_nominal"
    assert not (out / "failure.json").exists()
    sibling = json.loads(out.with_name(out.name + ".caller-failure.json").read_bytes())
    assert sibling["artifact_unchanged"] and sibling["status"] == "caller_failed_after_terminal_publication"


def test_partial_terminal_attempt_remains_failed_and_preserved(setup, monkeypatch):
    pipeline, invoke, out, _, _, _, _, _ = setup
    write = pipeline.write_package_files
    def partial(directory, files):
        if "manifest.json" in files:
            (directory / "manifest.json").write_bytes(b'{"incomplete":')
            raise OSError("partial seal")
        return write(directory, files)
    monkeypatch.setattr(pipeline, "write_package_files", partial)
    result = invoke()
    assert result["status"] == "failed" and result["sealed"] is False
    assert (out / "manifest.json").read_bytes() == b'{"incomplete":'
    assert (out / "failure.json").exists()


@pytest.mark.parametrize("seconds", [9, 31, float("nan"), True])
def test_invalid_bounds_have_no_output_or_worker_side_effect(setup, seconds):
    _, invoke, out, _, _, calls, _, _ = setup
    with pytest.raises(ValueError, match="between ten and thirty"):
        invoke(max_seconds=seconds)
    assert not out.exists() and not calls


def test_existing_output_is_not_modified(setup):
    _, invoke, out, _, _, calls, _, _ = setup
    out.mkdir(); (out / "owner.txt").write_text("preserve")
    with pytest.raises(ValueError, match="output must be new"):
        invoke()
    assert (out / "owner.txt").read_text() == "preserve" and not calls
