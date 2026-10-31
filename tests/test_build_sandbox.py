"""A failed build must remain inspectable without starting a server or repeating inference."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SPEC = importlib.util.spec_from_file_location("build_sandbox", Path(__file__).parents[1] / "scripts/build_sandbox.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def raw(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def saved_build(path, *, ready=False, trained=False, lab=None, preflight=None):
    path.mkdir(parents=True)
    result = {"schema_version": 1, "status": "ready" if ready else "failed",
              "lab_status": "ready" if ready else "not_ready", "seal_scope": "complete_immutable_outputs",
              "lab_directory": lab or str(path / ("learning_workflow/lab" if trained else "lab")) if ready else None}
    result_raw = raw(result)
    (path / "result.json").write_bytes(result_raw)
    manifest = {"schema_version": 1, "status": result["status"], "seal_scope": result["seal_scope"],
                "files": {"result.json": hashlib.sha256(result_raw).hexdigest()}}
    if preflight is not None:
        contents = raw(preflight)
        (path / "input-preflight.json").write_bytes(contents)
        manifest["files"]["input-preflight.json"] = hashlib.sha256(contents).hexdigest()
    (path / "manifest.json").write_bytes(raw(manifest))
    return path


def saved_report(path, build):
    result, source = launcher._terminal(build)
    path.mkdir()
    contents = {"index.html": b"<!doctype html><title>Saved evidence</title>",
                "report.json.txt": raw({"source_manifest_sha256": source, "build_status": result["status"]})}
    for name, content in contents.items():
        (path / name).write_bytes(content)
    (path / "manifest.json").write_bytes(raw({"schema_version": 1, "profile": "shared-exercise-build-report-v1",
        "status": "report_generated", "source_manifest_sha256": source,
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}}))


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / ".venv-lab/bin").mkdir(parents=True)
    (root / ".venv-lab/bin/python").symlink_to(sys.executable)
    (root / ".local/models").mkdir(parents=True)
    for variant in ("lite", "full"):
        (root / f".local/models/pose_landmarker_{variant}.task").write_bytes(b"test model " + variant.encode())
    (root / "video.webm").write_bytes(b"test source; no actual inference in launcher orchestration checks")
    monkeypatch.setattr(launcher, "PROJECT", root)
    monkeypatch.setattr(launcher.subprocess, "call", lambda *_a, **_k: pytest.fail("Unexpected server"))
    return root


def mocked_worker(monkeypatch, *, ready=False, report_error=False, lab=None, interrupted=False, child_error=None, input_change=None):
    calls = []

    def run(command, root, name, seconds):
        calls.append((command, name, seconds))
        if name == "build":
            pins = json.loads((root / "workflow-plan.json").read_bytes())["input_sha256"]
            video, model = (command[command.index(flag) + 1] for flag in ("--video", "--pose-model"))
            preflight = {"source_sha256": pins[video], "pose_model": {"sha256": pins[model]},
                         "input_pins": [{"path": p, "sha256": sha} for p, sha in pins.items()]}
            if input_change:
                input_change(preflight)
            saved_build(root / "build", ready=ready, trained="--train" in command, lab=lab, preflight=preflight)
        elif not report_error:
            saved_report(root / "report", Path(command[command.index("--run") + 1]))
        return {"returncode": (0 if ready else 1) if name == "build" else (1 if report_error else 0),
                "error": child_error if name == "build" else None, "interrupted": interrupted if name == "build" else False,
                "seconds": 0, "log_bytes": {"stdout": 0, "stderr": 0}}

    monkeypatch.setattr(launcher, "_run", run)
    return calls


@pytest.mark.parametrize("flags", [[], ["--person-only", "--config", "x"], ["--evidence-run", "x"],
    ["--person-only", "--source-mapping", "x"], ["--person-only", "--port", "65536"]])
def test_scope_conflicts_before_output(project, flags):
    with pytest.raises(SystemExit):
        launcher.main(["--video", str(project / "video.webm"), *flags])
    assert not (project / "artifacts").exists()


@pytest.mark.parametrize("flag", ["--train", "--person-only", "--root-assistance"])
def test_inspect_never_changes_saved_settings(project, flag):
    with pytest.raises(SystemExit):
        launcher.main(["--inspect-build", "unused", flag])
    assert not (project / "artifacts").exists()


@pytest.mark.parametrize("deadline", ["nan", "inf", "0", "481"])
def test_deadline_rejects_before_output(project, deadline):
    with pytest.raises(ValueError, match="deadline"):
        launcher.main(["--video", str(project / "video.webm"), "--person-only", "--max-seconds", deadline])
    assert not (project / "artifacts").exists()


def test_failed_build_gets_report_but_never_server(project, monkeypatch, capsys):
    calls = mocked_worker(monkeypatch)
    out = project / "attempt"
    assert launcher.main(["--video", str(project / "video.webm"), "--person-only", "--serve", "--out", str(out)]) == 1
    assert [row[1] for row in calls] == ["build", "report"]
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["status"] == "stopped" and receipt["lab"] is None
    assert receipt["report"] == str(out / "report/index.html")
    assert "No playable sandbox" in capsys.readouterr().out


def test_ready_training_uses_same_pipeline_and_prints_controls(project, monkeypatch, capsys):
    calls = mocked_worker(monkeypatch, ready=True)
    out = project / "attempt"
    assert launcher.main(["--video", str(project / "video.webm"), "--person-only", "--train",
                          "--root-assistance", "--pose-recovery", "selective-crop", "--out", str(out)]) == 0
    command, _, seconds = calls[0]
    assert "shared-exercise-build" in command and "--full-reference" in command
    assert command[command.index("--pose-model") + 1].endswith("pose_landmarker_full.task")
    assert command[command.index("--evaluation-seed") + 1] != command[command.index("--training-seed") + 1]
    assert seconds == 625
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["lab"] == str(out / "build/learning_workflow/lab")
    assert "Record & start → Stop & save" in capsys.readouterr().out


def test_equipment_arguments_remain_explicit(project, monkeypatch):
    calls = mocked_worker(monkeypatch)
    assert launcher.main(["--video", str(project / "video.webm"), "--evidence-run", str(project / "evidence"),
                          "--config", str(project / "config.json"), "--source-mapping", str(project / "mapping")]) == 1
    command = calls[0][0]
    assert "--person-only" not in command
    assert "--evidence-run" in command and "--config" in command and "--source-mapping" in command
    assert "--root-assistance" not in command and "--train" not in command


def test_report_only_retains_failed_source_bytes(project, monkeypatch):
    build = saved_build(project / "existing")
    original = {p.name: p.read_bytes() for p in build.iterdir()}
    calls = mocked_worker(monkeypatch)
    assert launcher.main(["--inspect-build", str(build), "--out", str(project / "report-attempt")]) == 1
    assert [row[1] for row in calls] == ["report"]
    assert original == {p.name: p.read_bytes() for p in build.iterdir()}


def test_report_failure_and_interruption_do_not_launch(project, monkeypatch):
    for index, settings in enumerate((dict(ready=True, report_error=True), dict(ready=True, interrupted=True),
                                      dict(ready=True, child_error={"type": "TimeoutError"}))):
        calls = mocked_worker(monkeypatch, **settings)
        out = project / f"attempt-{index}"
        assert launcher.main(["--video", str(project / "video.webm"), "--person-only", "--serve", "--out", str(out)]) == 1
        assert json.loads((out / "workflow-result.json").read_bytes())["lab"] is None
        if settings.get("interrupted"):
            assert len(calls) == 1


def test_outside_lab_is_not_launched(project, monkeypatch):
    mocked_worker(monkeypatch, ready=True, lab=str(project / "unrelated/lab"))
    assert launcher.main(["--video", str(project / "video.webm"), "--person-only", "--serve"]) == 1


def test_explicit_serve_only_after_ready_report(project, monkeypatch):
    mocked_worker(monkeypatch, ready=True)
    served = []
    monkeypatch.setattr(launcher.subprocess, "call", lambda *a, **kw: served.append((a, kw)) or 0)
    out = project / "attempt"
    assert launcher.main(["--video", str(project / "video.webm"), "--person-only", "--serve", "--port", "8123", "--out", str(out)]) == 0
    assert len(served) == 1
    assert served[0][0][0][-2:] == ["--port", "8123"]
    assert json.loads((out / "workflow-result.json").read_bytes())["status"] == "ready"


def test_existing_outputs_and_symlink_inputs_are_preserved(project, tmp_path):
    out = project / "existing"
    out.mkdir()
    (out / "mine").write_text("retained")
    with pytest.raises(ValueError, match="already exists"):
        launcher.main(["--video", str(project / "video.webm"), "--person-only", "--out", str(out)])
    linked = project / "linked.webm"
    linked.symlink_to(project / "video.webm")
    with pytest.raises(ValueError, match="symlink"):
        launcher.main(["--video", str(linked), "--person-only"])
    assert (out / "mine").read_text() == "retained"


def test_named_pipe_is_rejected_without_waiting_for_a_writer(tmp_path):
    fifo = tmp_path / "source.webm"
    os.mkfifo(fifo)
    code = ("import importlib.util,sys; "
            "s=importlib.util.spec_from_file_location('launcher',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m._read(sys.argv[2])")
    result = subprocess.run([sys.executable, "-c", code, str(Path(launcher.__file__)), str(fifo)],
                            capture_output=True, timeout=2)
    assert result.returncode != 0
    assert b"Expected a bounded regular file" in result.stderr


@pytest.mark.parametrize("content", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]'])
def test_strict_receipts(content):
    with pytest.raises(ValueError):
        launcher._json(content)


def test_terminal_hash_conflict_rejected(project):
    build = saved_build(project / "existing")
    (build / "result.json").write_bytes(b'{}')
    with pytest.raises(ValueError, match="seal"):
        launcher._terminal(build)


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(source_sha256="a" * 64),
    lambda d: d["pose_model"].update(sha256="b" * 64),
    lambda d: d["input_pins"][0].update(path="/different/source.webm"),
    lambda d: d["input_pins"].append(d["input_pins"][0]),
])
def test_changed_child_input_keeps_report_but_blocks_serve(project, monkeypatch, mutation):
    mocked_worker(monkeypatch, ready=True, input_change=mutation)
    out = project / "attempt"
    assert launcher.main(["--video", str(project / "video.webm"), "--person-only", "--serve", "--out", str(out)]) == 1
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["report"] and receipt["lab"] is None and receipt["status"] == "stopped"
    assert "workflow plan" in receipt["error"]["message"]


def test_report_hash_conflict_rejected(project):
    build = saved_build(project / "existing")
    report = project / "report"
    saved_report(report, build)
    (report / "index.html").write_bytes(b"changed")
    with pytest.raises(ValueError, match="seal"):
        launcher._report_identity(report, launcher._terminal(build)[1], "failed")


def test_input_identity_uses_the_terminal_and_report_seal(project):
    build = saved_build(project / "existing")
    with pytest.raises(ValueError, match="seal changed"):
        launcher._input_identity(build, [], {}, "a" * 64)


def test_worker_environment_excludes_keys_and_startup_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "not-a-real-key")
    monkeypatch.setenv("PYTHONSTARTUP", "/untrusted")
    env = launcher._environment(tmp_path)
    assert "META_API_KEY" not in env and "PYTHONSTARTUP" not in env
    result = launcher._run([sys.executable, "-P", "-c", "import os; print('META_API_KEY' in os.environ)"], tmp_path, "worker", 2)
    assert result["returncode"] == 0 and result["error"] is None
    assert (tmp_path / "worker.stdout.log").read_text().strip() == "False"


def test_worker_timeout_is_external_even_after_streams_close(tmp_path):
    code = "import os, time; os.close(1); os.close(2); time.sleep(5)"
    result = launcher._run([sys.executable, "-c", code], tmp_path, "worker", .2)
    assert result["error"]["type"] == "TimeoutError"
    assert result["returncode"] != 0 and result["seconds"] < 2


def test_worker_output_is_bounded_and_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "MAX_LOG_BYTES", 1024)
    result = launcher._run([sys.executable, "-c", "print('x' * 8192)"], tmp_path, "worker", 2)
    assert result["error"]["type"] == "ValueError"
    assert (tmp_path / "worker.stdout.log").stat().st_size == 1024
    assert len(list(tmp_path.glob("*.log"))) == 2
