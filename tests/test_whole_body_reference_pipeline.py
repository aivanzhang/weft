from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import pytest

from weft import whole_body_reference_pipeline as pipeline


def encode(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    from weft import whole_body_reference_inputs as inputs
    from weft import whole_body_reference_report as report
    source = tmp_path / "source"
    source.mkdir()
    snapshot = {"files": {"inputs/human.json": b"{}", "inputs/evidence-derived.json": b"{}"},
                "identity": {"source_sha256": "a" * 64}, "pins": []}
    calls = {"rechecks": 0, "reports": 0}
    monkeypatch.setattr(inputs, "prepare_reference_inputs", lambda *a, **kw: snapshot)
    def recheck(*args, **kwargs):
        calls["rechecks"] += 1
    monkeypatch.setattr(inputs, "recheck_reference_inputs", recheck)
    monkeypatch.setattr(pipeline, "_runtime", lambda: ({"runtime/weft/__init__.py": b""}, {"python": "fixture"}))
    def run(study, digest, seconds):
        assert 0 < seconds <= 45
        (study / "worker.json").write_bytes(encode({"status": "completed", "plan_sha256": digest}))
        (study / "candidate-human.json").write_bytes(b"{}")
        (study / "report.json").write_bytes(b"{}")
        return {"returncode": 0, "error": None, "interrupted": False, "seconds": .01}
    monkeypatch.setattr(pipeline, "_run_worker", run)
    def inspect(study, out, **kwargs):
        calls["reports"] += 1
        assert sha256((study / "manifest.json").read_bytes()).hexdigest() == kwargs["expected_manifest_sha256"]
        assert json.loads((study / "manifest.json").read_bytes())["status"] == "completed"
        out.mkdir()
        (out / "index.html").write_text("Rejected — not promoted")
        return {"proposal_status": "rejected_construction_proposal", "directory": str(out)}
    monkeypatch.setattr(report, "prepare_whole_body_reference_report", inspect)
    return source, tmp_path / "output", snapshot, calls


def verify_seal(root):
    manifest = json.loads((root / "manifest.json").read_bytes())
    for name, expected in manifest["files"].items():
        assert sha256((root / name).read_bytes()).hexdigest() == expected
    return manifest


def test_completed_delivery_does_not_admit_rejected_proposal(workflow):
    source, out, snapshot, calls = workflow
    result = pipeline.prepare_whole_body_reference_proposal(source, out)
    assert result["status"] == "completed"
    assert result["proposal_status"] == "rejected_construction_proposal"
    assert result["simulation_ready"] is result["promoted"] is False
    assert result["external_api_usd"] == 0
    assert result["report_path"] == str(out / "inspection/index.html")
    assert calls == {"reports": 1, "rechecks": 1}
    assert not any(source.iterdir())
    assert verify_seal(out)["status"] == "completed"
    assert verify_seal(out / "proposal")["profile"] == pipeline.STUDY_PROFILE


def test_explicit_objective_and_tighter_fit_only_change_plan(workflow):
    source, out, _, _ = workflow
    pipeline.prepare_whole_body_reference_proposal(source, out, proposal_objective="floor_first_v1", fit_seconds=.5)
    plan = json.loads((out / "proposal/plan.json").read_bytes())
    assert plan["proposal_objective"] == "floor_first_v1"
    assert plan["settings"] == {"maximum_seconds": .5, "finalization_seconds": 8.,
                                 "maximum_geometry_calls": 120000, "maximum_sweeps": 2}
    assert plan["promoted"] is False


def test_parent_plan_satisfies_actual_worker_contract(workflow, monkeypatch):
    from weft.whole_body_reference_worker import _plan
    source, out, _, _ = workflow
    monkeypatch.setattr(pipeline, "_runtime", lambda: (
        {f"runtime/weft/{name}.py": b"# frozen fixture\n" for name in pipeline.RUNTIME_MODULES},
        dict.fromkeys(("python", "numpy", "mujoco", "gymnasium"), "fixture")))
    pipeline.prepare_whole_body_reference_proposal(source, out, fit_seconds=.1)
    plan = _plan((out / "proposal/plan.json").read_bytes())
    assert plan["settings"]["maximum_seconds"] == .1


@pytest.mark.parametrize("options", [{"fit_seconds": 0}, {"fit_seconds": 31}, {"fit_seconds": True},
                                    {"fit_seconds": float("nan")}, {"max_seconds": 19}, {"max_seconds": 61},
                                    {"max_seconds": float("inf")}, {"proposal_objective": "automatic"}])
def test_invalid_bounds_do_not_create_outputs(workflow, options):
    source, out, _, _ = workflow
    with pytest.raises(ValueError):
        pipeline.prepare_whole_body_reference_proposal(source, out, **options)
    assert not out.exists()


@pytest.mark.parametrize("error_type", [TypeError, KeyError, AttributeError])
def test_malformed_metadata_is_a_bounded_input_error(workflow, monkeypatch, error_type):
    from weft import whole_body_reference_inputs as inputs
    source, out, _, _ = workflow
    def invalid(*args, **kwargs):
        raise error_type("untrusted metadata detail")
    monkeypatch.setattr(inputs, "prepare_reference_inputs", invalid)
    with pytest.raises(ValueError, match="Malformed body-source metadata") as caught:
        pipeline.prepare_whole_body_reference_proposal(source, out)
    assert "untrusted metadata detail" not in str(caught.value)
    assert not out.exists()


def test_cli_malformed_sealed_source_exits_cleanly_before_worker(tmp_path, monkeypatch, capsys):
    from test_whole_body_reference_inputs import make_body_source, mutate
    from weft.cli import main
    source = make_body_source(tmp_path / "body")
    mutate(source, "body-source.json", lambda marker: marker.update(source_artifact=[]))
    out = tmp_path / "proposal"
    monkeypatch.setattr(sys, "argv", ["weft", "body-reference-propose", "--run", str(source), "--out", str(out)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1 and not out.exists()
    message = capsys.readouterr().err
    assert "Malformed body-source metadata" in message and "Traceback" not in message


def test_never_overwrites_or_nests_in_source(workflow):
    source, out, _, _ = workflow
    for target in (source, source / "candidate", source.parent):
        with pytest.raises(ValueError, match="new and outside"):
            pipeline.prepare_whole_body_reference_proposal(source, target)


def test_source_recheck_failure_preserves_worker_without_report(workflow, monkeypatch):
    from weft import whole_body_reference_inputs as inputs
    source, out, _, calls = workflow
    def changed(*args, **kwargs):
        raise ValueError("Pinned input changed")
    monkeypatch.setattr(inputs, "recheck_reference_inputs", changed)
    result = pipeline.prepare_whole_body_reference_proposal(source, out)
    assert result["status"] == "failed" and result["worker"]["status"] == "completed"
    assert result["error"]["message"] == "Pinned input changed"
    assert calls["reports"] == 0
    assert verify_seal(out / "proposal")["status"] == "failed"
    assert verify_seal(out)["status"] == "failed"


def test_timeout_has_unknown_worker_counts_and_failure_seal(workflow, monkeypatch):
    source, out, _, calls = workflow
    monkeypatch.setattr(pipeline, "_run_worker", lambda *a: {"returncode": -9,
                        "error": {"type": "TimeoutError", "message": "Deadline"}, "seconds": .01})
    result = pipeline.prepare_whole_body_reference_proposal(source, out)
    assert result["status"] == "failed" and result["worker"] is None
    assert calls["reports"] == 0
    assert verify_seal(out / "proposal")["status"] == "failed"
    assert verify_seal(out)["status"] == "failed"


def test_worker_cannot_change_copied_inputs(workflow, monkeypatch):
    source, out, _, calls = workflow
    old = pipeline._run_worker
    def changed(study, *args):
        result = old(study, *args)
        (study / "inputs/human.json").write_bytes(b'{"changed":true}')
        return result
    monkeypatch.setattr(pipeline, "_run_worker", changed)
    result = pipeline.prepare_whole_body_reference_proposal(source, out)
    assert result["status"] == "failed" and calls["reports"] == 0


def test_worker_cannot_rewrite_the_frozen_plan(workflow, monkeypatch):
    source, out, _, calls = workflow
    old = pipeline._run_worker
    def changed(study, *args):
        result = old(study, *args)
        plan = json.loads((study / "plan.json").read_bytes())
        plan["settings"]["maximum_seconds"] = .001
        (study / "plan.json").write_bytes(encode(plan))
        return result
    monkeypatch.setattr(pipeline, "_run_worker", changed)
    result = pipeline.prepare_whole_body_reference_proposal(source, out)
    assert result["status"] == "failed" and calls["reports"] == 0
    assert verify_seal(out / "proposal")["status"] == "failed"


def test_report_failure_does_not_relabel_completed_study(workflow, monkeypatch):
    from weft import whole_body_reference_report as report
    source, out, _, _ = workflow
    def broken(*args, **kwargs):
        raise ValueError("Invalid proposal report")
    monkeypatch.setattr(report, "prepare_whole_body_reference_report", broken)
    result = pipeline.prepare_whole_body_reference_proposal(source, out)
    assert result["status"] == "failed" and result["error"]["stage"] == "inspection"
    assert verify_seal(out / "proposal")["status"] == "completed"
    assert verify_seal(out)["status"] == "failed"


def test_oversized_partial_output_gets_truthful_receipt_only_seal(workflow, monkeypatch):
    source, out, _, _ = workflow
    old = pipeline._run_worker
    def bloated(study, *args):
        result = old(study, *args)
        with (study / "unexpected.bin").open("wb") as stream:
            stream.truncate(16 * 1024 * 1024 + 1)
        return result
    monkeypatch.setattr(pipeline, "_run_worker", bloated)
    result = pipeline.prepare_whole_body_reference_proposal(source, out)
    assert result["status"] == "failed" and result["seal_scope"] == "failure_receipt_only"
    saved = json.loads((out / "result.json").read_bytes())
    assert saved["status"] == "failed" and saved["seal_scope"] == "failure_receipt_only"
    assert set(verify_seal(out)["files"]) == {"result.json"}


def make_process(tmp_path, body):
    study = tmp_path / "process"
    script = study / "runtime/weft/whole_body_reference_worker.py"
    script.parent.mkdir(parents=True)
    (study / "temporary").mkdir()
    script.write_text(body)
    return study


def test_real_child_receives_no_caller_keys_or_python_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-placeholder-not-a-key")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "caller-hook"))
    study = make_process(tmp_path, "import json, os, sys\nprint(json.dumps({'key': 'META_API_KEY' in os.environ, 'pythonpath': 'PYTHONPATH' in os.environ, 'isolated': sys.flags.isolated}))\n")
    result = pipeline._run_worker(study, "a" * 64, 3)
    assert result["returncode"] == 0 and result["error"] is None
    assert json.loads((study / "stdout.log").read_bytes()) == {"key": False, "pythonpath": False, "isolated": 1}


def test_real_child_excessive_logs_are_truncated_and_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_LOG_BYTES", 64)
    study = make_process(tmp_path, "import os\nos.write(1, b'x' * 100000)\n")
    result = pipeline._run_worker(study, "a" * 64, 3)
    assert result["error"]["type"] == "ValueError"
    assert (study / "stdout.log").stat().st_size == 64


def test_real_child_deadline_stops_owned_process(tmp_path):
    study = make_process(tmp_path, "import time\ntime.sleep(10)\n")
    result = pipeline._run_worker(study, "a" * 64, .05)
    assert result["error"]["type"] == "TimeoutError"
    assert result["returncode"] is not None and result["seconds"] < 3


def test_cli_passes_explicit_options_and_reports_failed_delivery(tmp_path, monkeypatch, capsys):
    from weft.cli import main
    calls = []
    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "failed", "promoted": False}
    monkeypatch.setattr(pipeline, "prepare_whole_body_reference_proposal", fake)
    monkeypatch.setattr(sys, "argv", ["weft", "body-reference-propose", "--run", "body", "--out", str(tmp_path / "out"),
                                     "--objective", "floor-first", "--fit-seconds", "2", "--manifest-sha256", "a" * 64])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert calls[0][1] == {"expected_manifest_sha256": "a" * 64, "proposal_objective": "floor_first_v1",
                           "fit_seconds": 2., "max_seconds": 60.}
    assert "preserved result" in capsys.readouterr().err
