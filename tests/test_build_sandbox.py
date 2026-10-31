"""A failed build must remain inspectable without starting a server or repeating inference."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


SPEC = importlib.util.spec_from_file_location("build_sandbox", Path(__file__).parents[1] / "scripts/build_sandbox.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def raw(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def saved_build(path, *, ready=False, trained=False, lab=None, preflight=None, members=None):
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
    for name, contents in (members or {}).items():
        member = path / name
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(contents)
        manifest["files"][name] = hashlib.sha256(contents).hexdigest()
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


def test_console_and_server_are_mutually_exclusive(project):
    with pytest.raises(SystemExit):
        launcher.main(["--video", str(project / "video.webm"), "--person-only", "--console", "--serve"])
    assert not (project / "artifacts").exists()


@pytest.mark.parametrize("ready", [False, True])
def test_console_starts_only_after_ready_build_and_report(project, monkeypatch, ready):
    mocked_worker(monkeypatch, ready=ready)
    calls = []
    monkeypatch.setattr(launcher.subprocess, "call", lambda cmd, **kwargs: calls.append((cmd, kwargs)) or 0)
    out = project / "console-build"
    assert launcher.main(["--video", str(project / "video.webm"), "--person-only", "--console", "--out", str(out)]) == (0 if ready else 1)
    assert len(calls) == int(ready)
    if ready:
        assert calls[0][0][-3:] == ["lab-console", "--resume", str(out / "build/lab")]
        assert "stdin" not in calls[0][1]
        assert "META_API_KEY" not in calls[0][1]["env"]


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


def body_source(path):
    """Analytical metadata only; constructor validation is never stubbed as physical evidence."""
    path.mkdir()
    source_raw, human_raw = b"unchanged source bytes", raw({"fixture": "body"})
    source_sha, human_sha = hashlib.sha256(source_raw).hexdigest(), hashlib.sha256(human_raw).hexdigest()
    source = raw({"artifact": "source.webm", "sha256": source_sha, "license": "fixture only"})
    marker = raw({"schema_version": 1, "profile": launcher.BODY_SOURCE_PROFILE, "status": "construction_only",
                  "simulation_ready": False, "source_artifact": "source.webm", "source_sha256": source_sha,
                  "artifacts": {"source.json": hashlib.sha256(source).hexdigest(), "human.json": human_sha},
                  "pose_options": {"pose_mode": "video", "pose_identity": "raw"}})
    members = {"source.webm": source_raw, "human.json": human_raw, "source.json": source, "body-source.json": marker}
    for name, contents in members.items():
        (path / name).write_bytes(contents)
    (path / "manifest.json").write_bytes(raw({"schema_version": 1, "profile": launcher.BODY_SOURCE_PROFILE,
        "status": "construction_only", "simulation_ready": False,
        "files": {name: hashlib.sha256(contents).hexdigest() for name, contents in members.items()}}))
    return path


def body_consumption(body, *, person=True):
    identity = launcher._body_identity(body)
    hashes = identity["metadata_sha256"]
    prefix = "candidate/" if person else "body_seed/inputs/human-source/"
    archive = "candidate/inputs/body-source-manifest.json" if person else "body_seed/inputs/human-source/source-manifest.json"
    report = "candidate/reconstruction.json" if person else "body_seed/construction-seed.json"
    proof = {"source_sha256": identity["source_sha256"], "source_verification": {"verified_files": 4}}
    if person:
        proof.update(profile="shared-world-v3", status="ready", source_manifest_sha256=hashes["manifest.json"],
                     human_file_sha256=identity["human_sha256"])
    else:
        proof.update(profile="shared-world-construction-seed-v1", status="construction_only", simulation_ready=False,
                     input_profile=launcher.BODY_SOURCE_PROFILE, source_directory=str(body),
                     human_json_sha256=identity["human_sha256"])
        proof["source_verification"]["manifest_sha256"] = hashes["manifest.json"]
    return {"settings.json": raw({"human_run": str(body), "video": None, "base_run": None,
                "inference_calls": 0, **({"scene_scope": "human_only"} if person else {})}),
            archive: (body / "manifest.json").read_bytes(), report: raw(proof),
            **{prefix + name: (body / name).read_bytes() for name in ("body-source.json", "source.json", "human.json")}}


def mocked_body_worker(monkeypatch, *, ready=True, mutation=None, original_mutation=None, partial=False):
    calls = []

    def run(command, root, name, seconds):
        calls.append((command, name, seconds))
        if name == "build":
            body = Path(command[command.index("--human-run") + 1])
            members = body_consumption(body, person="--person-only" in command)
            if partial:
                members = {"settings.json": members["settings.json"]}
            if mutation:
                mutation(members)
            saved_build(root / "build", ready=ready, trained="--train" in command, members=members)
            if original_mutation:
                original_mutation(body)
        else:
            saved_report(root / "report", Path(command[command.index("--run") + 1]))
        return {"returncode": (0 if ready else 1) if name == "build" else 0,
                "error": None, "interrupted": False, "seconds": 0, "log_bytes": {"stdout": 0, "stderr": 0}}

    monkeypatch.setattr(launcher, "_run", run)
    return calls


def test_saved_body_scope_matches_public_assembler_contract(project):
    from weft.exercise_scene_pipeline import prepare_exercise_scene

    body = body_source(project / "body")
    scope = json.loads(body_consumption(body)["settings.json"])["scene_scope"]
    # A base input stops at the public scope guard, before any source reads or native stage.
    with pytest.raises(ValueError, match="Person-only scope accepts"):
        prepare_exercise_scene(project / "unused-base", None, None, project / "unused-out", scene_scope=scope)
    with pytest.raises(ValueError, match="Scene scope requires"):
        prepare_exercise_scene(project / "unused-base", None, None, project / "unused-out", scene_scope="person_only")
    assert not (project / "unused-out").exists()


@pytest.mark.parametrize("person", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_saved_body_ready_uses_common_pipeline_without_raw_inputs(project, monkeypatch, person, train):
    body = body_source(project / "body")
    original = {p.name: p.read_bytes() for p in body.iterdir()}
    original_read = launcher._read

    def metadata_only(path, limit=launcher.MAX_METADATA_BYTES):
        if path.is_relative_to(body):
            assert path.name in launcher.BODY_METADATA, "Launcher must not duplicate the constructor's body/media scan"
        return original_read(path, limit)

    monkeypatch.setattr(launcher, "_read", metadata_only)
    calls = mocked_body_worker(monkeypatch)
    out = project / "saved-attempt"
    flags = ["--person-only"] if person else ["--evidence-run", str(project / "observations"),
        "--config", str(project / "camera.json"), "--source-mapping", str(project / "mapping")]
    if train:
        flags += ["--train", "--policy-family", "feedback-residual"]
    assert launcher.main(["--human-run", str(body), "--root-assistance", *flags, "--out", str(out)]) == 0
    assert [row[1] for row in calls] == ["build", "report"]
    command = calls[0][0]
    assert calls[0][2] == (295 if train else 115)
    assert "--full-reference" in command and "--root-assistance" in command
    assert not {"--video", "--pose-model", "--pose-recovery", "--source-note"} & set(command)
    assert ("--train" in command) == train and ("--person-only" in command) == person
    if train:
        assert command[command.index("--training-policy-family") + 1] == "feedback-residual"
    plan = json.loads((out / "workflow-plan.json").read_bytes())
    assert set(plan["input_sha256"]) == {str(body / name) for name in launcher.BODY_METADATA}
    assert plan["body_source"]["status"] == "construction_only"
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["input_binding"] == "sealed_body_source_match_plan" and receipt["status"] == "ready"
    assert receipt["lab"] == str(out / "build" / ("learning_workflow/lab" if train else "lab"))
    assert original == {p.name: p.read_bytes() for p in body.iterdir()}


@pytest.mark.parametrize("flag,value", [("--pose-model", "unused"), ("--pose-recovery", "selective-crop"),
    ("--source-note", "changed attribution"), ("--source-note", ""), ("--pose-mode", "image"), ("--pose-identity", "temporal")])
def test_saved_body_rejects_raw_only_options_before_output(project, flag, value):
    with pytest.raises(SystemExit):
        launcher.main(["--human-run", "unused", "--person-only", flag, value])
    assert not (project / "artifacts").exists()


@pytest.mark.parametrize("other", ["--video", "--inspect-build"])
def test_saved_body_input_is_mutually_exclusive(project, other):
    with pytest.raises(SystemExit):
        launcher.main(["--human-run", "unused", other, "unused", "--person-only"])
    assert not (project / "artifacts").exists()


@pytest.mark.parametrize("train,deadline", [(False, "181"), (True, "481"), (False, "nan"), (False, "0")])
def test_saved_body_deadline_rejects_before_output(project, train, deadline):
    with pytest.raises(ValueError, match="deadline"):
        launcher.main(["--human-run", "unused", "--person-only", "--max-seconds", deadline, *(["--train"] if train else [])])
    assert not (project / "artifacts").exists()


@pytest.mark.parametrize("person", [True, False])
@pytest.mark.parametrize("partial", [True, False])
def test_saved_body_failed_build_still_reports_without_interaction(project, monkeypatch, person, partial):
    body = body_source(project / "body")
    calls = mocked_body_worker(monkeypatch, ready=False, partial=partial)
    out = project / "saved-attempt"
    scope = ["--person-only"] if person else ["--evidence-run", "unused", "--config", "unused"]
    assert launcher.main(["--human-run", str(body), *scope, "--console", "--out", str(out)]) == 1
    assert [row[1] for row in calls] == ["build", "report"]
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["report"] and receipt["lab"] is None and receipt["status"] == "stopped"
    assert receipt["input_binding"] == ("unavailable_before_sealed_construction" if partial else "sealed_body_source_match_plan")


@pytest.mark.parametrize("person", [True, False])
def test_saved_body_ready_without_consumption_proof_is_blocked(project, monkeypatch, person):
    body = body_source(project / "body")
    mocked_body_worker(monkeypatch, partial=True)
    out = project / "saved-attempt"
    scope = ["--person-only"] if person else ["--evidence-run", "unused", "--config", "unused"]
    assert launcher.main(["--human-run", str(body), *scope, "--serve", "--out", str(out)]) == 1
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["report"] and receipt["lab"] is None
    assert "no sealed input identities" in receipt["error"]["message"]


@pytest.mark.parametrize("person", [True, False])
@pytest.mark.parametrize("kind", ["settings", "manifest", "metadata", "human", "report"])
def test_saved_body_child_substitution_preserves_report_but_blocks_lab(project, monkeypatch, person, kind):
    body = body_source(project / "body")

    def mutate(members):
        if kind == "settings":
            doc = json.loads(members["settings.json"]); doc["human_run"] += "-other"
            members["settings.json"] = raw(doc)
        elif kind == "report":
            name = "candidate/reconstruction.json" if person else "body_seed/construction-seed.json"
            doc = json.loads(members[name]); doc["source_sha256"] = "a" * 64
            members[name] = raw(doc)
        else:
            prefix = "candidate/" if person else "body_seed/inputs/human-source/"
            name = ("candidate/inputs/body-source-manifest.json" if person else prefix + "source-manifest.json") if kind == "manifest" else prefix + ("source.json" if kind == "metadata" else "human.json")
            members[name] += b" "

    mocked_body_worker(monkeypatch, mutation=mutate)
    out = project / "saved-attempt"
    scope = ["--person-only"] if person else ["--evidence-run", "unused", "--config", "unused"]
    assert launcher.main(["--human-run", str(body), *scope, "--serve", "--out", str(out)]) == 1
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["report"] and receipt["lab"] is None and "workflow plan" in receipt["error"]["message"]


@pytest.mark.parametrize("kind", ["manifest", "metadata", "failure"])
def test_saved_body_original_change_blocks_ready_result(project, monkeypatch, kind):
    body = body_source(project / "body")

    def mutate(original):
        name = {"manifest": "manifest.json", "metadata": "source.json", "failure": "failure.json"}[kind]
        with (original / name).open("ab") as stream:
            stream.write(b" ")

    mocked_body_worker(monkeypatch, original_mutation=mutate)
    out = project / "saved-attempt"
    assert launcher.main(["--human-run", str(body), "--person-only", "--serve", "--out", str(out)]) == 1
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["report"] and receipt["lab"] is None and receipt["error"]


@pytest.mark.parametrize("kind", ["failed", "legacy", "unsealed", "conflicting", "failure_marker", "oversize"])
def test_saved_body_bad_metadata_rejects_before_output(project, kind):
    body = body_source(project / "body")
    manifest = json.loads((body / "manifest.json").read_bytes())
    if kind in ("failed", "legacy"):
        manifest.update({"status": "failed"} if kind == "failed" else {"profile": "exercise-v3"})
    elif kind == "unsealed":
        (body / "source.json").write_bytes(raw({"artifact": "source.webm", "sha256": "b" * 64}))
    elif kind == "conflicting":
        contents = raw({"artifact": "source.webm", "sha256": "b" * 64})
        (body / "source.json").write_bytes(contents)
        manifest["files"]["source.json"] = hashlib.sha256(contents).hexdigest()
    elif kind == "failure_marker":
        (body / "failure.json").write_text("{}")
    else:
        (body / "source.json").write_bytes(b" " * (launcher.MAX_METADATA_BYTES + 1))
    (body / "manifest.json").write_bytes(raw(manifest))
    with pytest.raises(ValueError):
        launcher.main(["--human-run", str(body), "--person-only"])
    assert not (project / "artifacts").exists()


def test_saved_body_output_overlap_and_symlink_reject_without_mutation(project):
    body = body_source(project / "body")
    before = {p.name: p.read_bytes() for p in body.iterdir()}
    with pytest.raises(ValueError, match="outside every input"):
        launcher.main(["--human-run", str(body), "--person-only", "--out", str(body / "output")])
    linked = project / "body-link"
    linked.symlink_to(body)
    with pytest.raises(ValueError, match="symlink"):
        launcher.main(["--human-run", str(linked), "--person-only"])
    assert before == {p.name: p.read_bytes() for p in body.iterdir()}


def preserved_build(path, *, trained=False, lifecycle="ready", preflight=None):
    """Real packaging metadata shape; no simulator or physical-validity claim."""
    sha = lambda data: hashlib.sha256(data).hexdigest()
    canonical = lambda doc: sha(json.dumps(doc, sort_keys=True, allow_nan=False).encode())
    lab = path / ("learning_workflow/lab" if trained else "lab")
    source_prefix = "learning_workflow/selected/" if trained else "candidate/"
    scene = {"schema_version": 1, "profile": "shared-world-v1",
             "source_binding": {"mode": "source_qualified", "source_sha256":
                                preflight["source_sha256"] if preflight else "a" * 64}}
    task = {"schema_version": 1, "profile": "shared-exercise-task-v1", "scene_sha256": canonical(scene)}
    policy = raw({"profile": "shared-exercise-policy-v2", "scene_sha256": canonical(scene),
                  "task_sha256": canonical(task)})
    source_files = {"scene.json": raw(scene), "source.webm": b"fixture media; never read by handoff"}
    candidate_manifest = raw({"schema_version": 1, "status": "ready",
                              "files": {name: sha(data) for name, data in source_files.items()}})
    members = {"candidate/manifest.json": candidate_manifest,
               **{"candidate/" + name: data for name, data in source_files.items()}}
    if trained:
        source_files.update({"exercise-task.json": raw(task), "policy.json": policy})
        members["learning_workflow/exercise-task.json"] = raw(task)
    source_manifest = raw({"schema_version": 1, "status": "trained" if trained else "ready",
                           "files": {name: sha(data) for name, data in source_files.items()}})
    members.update({source_prefix + "manifest.json": source_manifest,
                    **{source_prefix + name: data for name, data in source_files.items()}})
    contract = {"schema_version": 1, "profile": "shared-world", "control_profile": "shared_exercise",
                "scene_sha256": canonical(scene), "task_sha256": canonical(task),
                "reset_options": {"jitter": False, "jitter_scale": 0.0}, "learned_policy_available": trained,
                "source_manifest_sha256": sha(source_manifest), "source_files_verified": len(source_files),
                "implementation_sha256": {"runtime.py": sha(b"fixture runtime; never imported")},
                "runtime_sha256": {"runtime.py": sha(b"fixture runtime; never imported")},
                "runtime_versions": {"python": "fixture", "mujoco": "fixture"}}
    run = {"scene.json": raw(scene), "exercise-task.json": raw(task), "source-run/manifest.json": source_manifest,
           **{"source-run/" + name: data for name, data in source_files.items()},
           "lab-implementation/runtime.py": b"fixture runtime; never imported"}
    if trained:
        run.update({"policy.json": policy, "learning-run/manifest.json": source_manifest,
                    **{"learning-run/" + name: data for name, data in source_files.items()},
                    "policy-evaluation.json": raw({"status": "evaluated", "physical_validation": "unverified"})})
        run["policy-evaluation-run/evaluation.json"] = run["policy-evaluation.json"]
        run["policy-evaluation-run/manifest.json"] = raw({"schema_version": 1, "status": "evaluated",
            "files": {"evaluation.json": sha(run["policy-evaluation.json"])}})
        contract.update({"policy_sha256": sha(policy), "learning_manifest_sha256": sha(source_manifest),
            "policy_evaluation_manifest_sha256": sha(run["policy-evaluation-run/manifest.json"]),
            "policy_evaluation_sha256": sha(run["policy-evaluation.json"])})
    run["exercise-lab.json"] = raw(contract)
    manifest = raw({"schema_version": 1, "status": "ready", "control_profile": "shared_exercise",
                    "files": {name: sha(data) for name, data in run.items()}})
    prefix = lab.relative_to(path).as_posix() + "/run/"
    members.update({prefix + "manifest.json": manifest, **{prefix + name: data for name, data in run.items()}})
    saved_build(path, ready=True, trained=trained, preflight=preflight, members=members)
    config = {key: contract[key] for key in ("schema_version", "profile", "control_profile", "scene_sha256",
        "task_sha256", "reset_options", "learned_policy_available", *(["policy_sha256", "learning_manifest_sha256",
        "policy_evaluation_manifest_sha256", "policy_evaluation_sha256"] if trained else []))}
    config.update(source_run=str(path / source_prefix.rstrip("/")), manifest_sha256=sha(manifest))
    (lab / "lab-config.json").write_bytes(raw(config))
    (lab / "status.json").write_bytes(raw({"status": lifecycle, "events": [{"stage": "shared_exercise_import"}]}))
    result = json.loads((path / "result.json").read_bytes())
    result.update(status="failed", scene_sha256=contract["scene_sha256"], task_sha256=contract["task_sha256"],
        launch_arguments=["lab", "--resume", str(lab)], physical_validation="unverified",
        error={"type": "TimeoutError", "message": "Finalization stopped after lab preparation"},
        stages=[{"stage": "learning_interactive_lab" if trained else "interactive_lab", "status": "ready",
                 "manifest_sha256": sha(manifest)}])
    (path / "result.json").write_bytes(raw(result))
    reseal_preserved(path)
    return path, lab


def reseal_preserved(build):
    """Reseal fixture mutations so semantic conflicts are tested beyond byte tampering."""
    result = json.loads((build / "result.json").read_bytes())
    lab = build / ("learning_workflow/lab" if (build / "learning_workflow/lab").exists() else "lab")
    manifest = json.loads((lab / "run/manifest.json").read_bytes())
    manifest["files"] = {name: hashlib.sha256((lab / "run" / name).read_bytes()).hexdigest() for name in manifest["files"]}
    manifest_raw = raw(manifest)
    (lab / "run/manifest.json").write_bytes(manifest_raw)
    run_sha = hashlib.sha256(manifest_raw).hexdigest()
    config = json.loads((lab / "lab-config.json").read_bytes())
    config["manifest_sha256"] = run_sha
    (lab / "lab-config.json").write_bytes(raw(config))
    result["stages"][0]["manifest_sha256"] = run_sha
    (build / "result.json").write_bytes(raw(result))
    parent = json.loads((build / "manifest.json").read_bytes())
    parent.update(status=result["status"], seal_scope=result["seal_scope"],
        files={name: hashlib.sha256((build / name).read_bytes()).hexdigest() for name in parent["files"]})
    (build / "manifest.json").write_bytes(raw(parent))


@pytest.mark.parametrize("trained", [False, True])
@pytest.mark.parametrize("lifecycle", ["ready", "closed"])
def test_preserved_lab_metadata_accepts_real_shapes_without_media_or_runtime_reads(project, monkeypatch, trained, lifecycle):
    build, lab = preserved_build(project / "existing", trained=trained, lifecycle=lifecycle)
    before = {str(p.relative_to(build)): p.read_bytes() for p in build.rglob("*") if p.is_file()}
    original_read, reads = launcher._read, []

    def metadata_only(path, limit=launcher.MAX_METADATA_BYTES):
        assert path.suffix == ".json", "Recovery must not scan media or execute saved runtime code"
        reads.append(path)
        return original_read(path, limit)

    terminal, seal = launcher._terminal(build)
    monkeypatch.setattr(launcher, "_read", metadata_only)
    info = launcher._preserved_lab(build, terminal, seal)
    assert info["status"] == "identity_metadata_verified" and info["directory"] == str(lab)
    assert info["lifecycle_status"] == lifecycle and info["runtime_admission"] == "not_checked_by_launcher"
    assert 0 < info["metadata_bytes_read"] <= launcher.MAX_HANDOFF_BYTES
    assert all(reads.count(path) == 2 for path in set(reads)), "Read each selected member once plus the final recheck"
    assert "full artifact and runtime checks" in info["scope"]
    assert before == {str(p.relative_to(build)): p.read_bytes() for p in build.rglob("*") if p.is_file()}


@pytest.mark.parametrize("trained", [False, True])
@pytest.mark.parametrize("interaction", ["--serve", "--console"])
def test_preserved_lab_failure_prints_manual_commands_without_launch(project, monkeypatch, capsys, trained, interaction):
    calls = []

    def run(command, root, name, seconds):
        calls.append(name)
        if name == "build":
            pins = json.loads((root / "workflow-plan.json").read_bytes())["input_sha256"]
            video, model = (command[command.index(flag) + 1] for flag in ("--video", "--pose-model"))
            preflight = {"source_sha256": pins[video], "pose_model": {"sha256": pins[model]},
                         "input_pins": [{"path": p, "sha256": sha} for p, sha in pins.items()]}
            preserved_build(root / "build", trained=trained, preflight=preflight)
        else:
            saved_report(root / "report", root / "build")
        return {"returncode": 1 if name == "build" else 0, "error": None, "interrupted": False}

    monkeypatch.setattr(launcher, "_run", run)
    out = project / "attempt $(not-a-command)"
    args = ["--video", str(project / "video.webm"), "--person-only", interaction, "--port", "8123", "--out", str(out)]
    assert launcher.main(args + (["--train"] if trained else [])) == 1
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["build_status"] == "failed" and receipt["status"] == "stopped" and receipt["lab"] is None
    assert receipt["preserved_lab"]["status"] == "identity_metadata_verified" and receipt["report"]
    assert calls == ["build", "report"]
    output = capsys.readouterr().out
    assert "Build stopped, but a prepared lab was preserved" in output
    assert "compatible-runtime checks still apply" in output and "No playable sandbox" not in output
    commands = [shlex.split(line) for line in output.splitlines() if " --resume " in line]
    lab = receipt["preserved_lab"]["directory"]
    assert len(commands) == 2 and commands[0][-5:] == ["lab", "--resume", lab, "--port", "8123"]
    assert commands[1][-3:] == ["lab-console", "--resume", lab]


def test_preserved_lab_inspect_closed_child_does_not_rerun_build(project, monkeypatch, capsys):
    build, _ = preserved_build(project / "existing", trained=True, lifecycle="closed")
    original = {str(p.relative_to(build)): p.read_bytes() for p in build.rglob("*") if p.is_file()}
    calls = mocked_worker(monkeypatch)
    out = project / "inspection"
    assert launcher.main(["--inspect-build", str(build), "--console", "--out", str(out)]) == 1
    assert [row[1] for row in calls] == ["report"]
    assert json.loads((out / "workflow-result.json").read_bytes())["preserved_lab"]["lifecycle_status"] == "closed"
    assert "lab-console --resume" in capsys.readouterr().out
    assert original == {str(p.relative_to(build)): p.read_bytes() for p in build.rglob("*") if p.is_file()}


@pytest.mark.parametrize("matches", [False, True])
def test_preserved_lab_matches_fresh_saved_body_source_without_video_preflight(project, matches):
    build, _ = preserved_build(project / "existing")
    terminal, seal = launcher._terminal(build)
    expected = "a" * 64 if matches else "b" * 64
    if matches:
        assert launcher._preserved_lab(build, terminal, seal, expected_source_sha=expected)["source_sha256"] == expected
    else:
        with pytest.raises(ValueError, match="workflow plan"):
            launcher._preserved_lab(build, terminal, seal, expected_source_sha=expected)


@pytest.mark.parametrize("kind", ["failed_run", "failure_marker", "missing_member", "tampered_contract",
    "source_copy", "wrong_task", "terminal_scene", "config_scene", "config_source", "policy_pin", "source_video"])
def test_preserved_lab_rejects_failed_tampered_or_mismatched_child(project, monkeypatch, capsys, kind):
    build, lab = preserved_build(project / "existing", trained=True,
                                 preflight={"source_sha256": "a" * 64})
    paths = {"failed_run": lab / "run/manifest.json", "source_copy": lab / "run/source-run/manifest.json",
             "wrong_task": lab / "run/exercise-task.json", "terminal_scene": build / "result.json",
             "config_scene": lab / "lab-config.json", "config_source": lab / "lab-config.json",
             "policy_pin": lab / "lab-config.json", "source_video": build / "input-preflight.json"}
    if kind == "failure_marker":
        (lab / "failure.json").write_bytes(raw({"status": "failed"}))
    elif kind == "missing_member":
        (lab / "run/lab-implementation/runtime.py").unlink()
    elif kind == "tampered_contract":
        with (lab / "run/exercise-lab.json").open("ab") as stream:
            stream.write(b" ")
    else:
        document = json.loads(paths[kind].read_bytes())
        if kind == "failed_run":
            document["status"] = "failed"
        elif kind == "source_copy":
            document["files"]["scene.json"] = "b" * 64
        elif kind in ("wrong_task", "terminal_scene", "config_scene"):
            document["scene_sha256"] = "b" * 64
        elif kind == "config_source":
            document["source_run"] = str(project / "different-source")
        elif kind == "policy_pin":
            document["policy_sha256"] = "b" * 64
        else:
            document["source_sha256"] = "b" * 64
        paths[kind].write_bytes(raw(document))
        reseal_preserved(build)
    mocked_worker(monkeypatch)
    out = project / "inspection"
    assert launcher.main(["--inspect-build", str(build), "--serve", "--out", str(out)]) == 1
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["report"] and receipt["lab"] is None and receipt["status"] == "stopped"
    assert receipt["error"] and "preserved_lab" not in receipt
    assert " --resume " not in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["outside_lab", "parent_traversal", "member_escape", "symlink_member",
    "missing_stage", "parent_member_pin", "receipt_only"])
def test_preserved_lab_requires_safe_parent_linkage(project, kind):
    build, lab = preserved_build(project / "existing")
    parent = json.loads((build / "manifest.json").read_bytes())
    result = json.loads((build / "result.json").read_bytes())
    if kind in ("outside_lab", "parent_traversal"):
        result["lab_directory"] = str(project / "unrelated/lab") if kind == "outside_lab" else str(build / "x/../lab")
    elif kind == "member_escape":
        parent["files"]["../escape.json"] = "a" * 64
    elif kind == "symlink_member":
        target = lab / "run/lab-implementation/runtime.py"
        target.unlink()
        target.symlink_to(project / "video.webm")
    elif kind == "missing_stage":
        result["stages"] = []
    elif kind == "parent_member_pin":
        parent["files"]["lab/run/scene.json"] = "b" * 64
    else:
        parent["seal_scope"] = result["seal_scope"] = "failure_receipt_only"
        parent["files"] = {}
    (build / "result.json").write_bytes(raw(result))
    parent["files"]["result.json"] = hashlib.sha256(raw(result)).hexdigest()
    (build / "manifest.json").write_bytes(raw(parent))
    terminal, seal = launcher._terminal(build)
    with pytest.raises((ValueError, OSError)):
        launcher._preserved_lab(build, terminal, seal)


@pytest.mark.parametrize("kind", ["metadata_bytes", "deadline", "changed_metadata"])
def test_preserved_lab_checks_are_bounded_and_rechecked(project, monkeypatch, kind):
    build, lab = preserved_build(project / "existing")
    terminal, seal = launcher._terminal(build)
    if kind == "metadata_bytes":
        monkeypatch.setattr(launcher, "MAX_HANDOFF_BYTES", 20)
    elif kind == "deadline":
        clock = iter((0, 6))
        monkeypatch.setattr(launcher.time, "monotonic", lambda: next(clock))
    else:
        original_read, count = launcher._read, 0

        def changing_read(path, limit=launcher.MAX_METADATA_BYTES):
            nonlocal count
            data = original_read(path, limit)
            if path == lab / "lab-config.json":
                count += 1
                if count == 2:
                    return data + b" "
            return data

        monkeypatch.setattr(launcher, "_read", changing_read)
    with pytest.raises(ValueError, match="bounded|bound|changed"):
        launcher._preserved_lab(build, terminal, seal)
