import json
import sys

import pytest

from weft import cli, runtime_qualification


def invoke(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["weft", "qualify", *args])
    cli.main()


def test_cli_preserves_explicit_input_seal_and_bounds(monkeypatch, capsys):
    captured = {}

    def qualify(cases, out, **bounds):
        captured.update(cases=cases, out=out, bounds=bounds)
        return {"status": "completed", "cases": [{"task_success": False}], "operationally_qualified": 1}

    monkeypatch.setattr(runtime_qualification, "qualify_runtimes", qualify)
    invoke(monkeypatch, ["--run", "sealed", "--out", "result", "--manifest-sha256", "a" * 64,
        "--controller", "authored", "--seed", "41", "--probe-channel", "7", "--probe-steps", "4",
        "--max-seconds", "30", "--max-steps", "1000", "--max-output-mib", "8"])
    assert captured == {"cases": [{"id": "environment", "input": "sealed", "manifest_sha256": "a" * 64,
        "controller": "authored", "seed": 41, "probe_channel": 7, "probe_steps": 4}],
        "out": "result", "bounds": {"max_seconds": 30, "max_steps": 1000, "max_output_bytes": 8 * 1024 * 1024}}
    assert json.loads(capsys.readouterr().out)["cases"][0]["task_success"] is False


def test_cli_case_paths_use_plan_directory_without_hiding_symlinks(tmp_path, monkeypatch):
    link = tmp_path / "linked-input"
    link.symlink_to(tmp_path / "actual-input", target_is_directory=True)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "named", "input": "linked-input", "manifest_sha256": "b" * 64}]))
    captured = []

    def qualify(cases, out, **bounds):
        captured.extend(cases)
        return {"status": "completed", "cases": [{}], "operationally_qualified": 1}

    monkeypatch.setattr(runtime_qualification, "qualify_runtimes", qualify)
    invoke(monkeypatch, ["--cases", str(path), "--out", str(tmp_path / "out")])
    assert captured == [{"id": "named", "input": str(link), "manifest_sha256": "b" * 64}]


@pytest.mark.parametrize("raw", [b'[{"id":"a","input":"one","input":"two"}]', b'[{"seed":NaN}]', b' ' * (1024 * 1024 + 1)])
def test_cli_rejects_ambiguous_or_excessive_plan_before_execution(tmp_path, monkeypatch, raw):
    path = tmp_path / "cases.json"
    path.write_bytes(raw)
    monkeypatch.setattr(runtime_qualification, "qualify_runtimes", lambda *_args, **_kwargs: pytest.fail("Execution must not start"))
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, ["--cases", str(path), "--out", str(tmp_path / "out")])
    assert error.value.code == 1


def test_cli_rejects_case_overrides_before_opening_plan(monkeypatch):
    monkeypatch.setattr(runtime_qualification, "qualify_runtimes", lambda *_args, **_kwargs: pytest.fail("Execution must not start"))
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, ["--cases", "not-opened.json", "--out", "result", "--seed", "41"])
    assert error.value.code == 1


@pytest.mark.parametrize("status,qualified", [("failed", 0), ("completed", 0)])
def test_cli_retains_result_and_nonzero_exit_for_unqualified_case(monkeypatch, capsys, status, qualified):
    result = {"status": status, "cases": [{"complete_interval": False}], "operationally_qualified": qualified, "directory": "saved"}
    monkeypatch.setattr(runtime_qualification, "qualify_runtimes", lambda *_args, **_kwargs: result)
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, ["--run", "sealed", "--out", "saved"])
    captured = capsys.readouterr()
    assert error.value.code == 1
    assert json.loads(captured.out) == result
    assert "saved/report.json" in captured.err
