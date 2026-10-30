"""Study finalization faults use fake cases and never call source constructors."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def study(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / "scripts/studies/check_exercise_body_source.py"
    spec = importlib.util.spec_from_file_location("exercise_body_study_fixture", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    plan = {"profile": "exercise-body-source-equivalence-study-v1", "output": "study",
            "cases": [{"id": name, "input": "source-" + name, "method": "api"} for name in ("a", "b", "c")]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(module.sys, "argv", [str(script), "--plan", "plan.json", "--out", "study"])
    monkeypatch.setattr(module, "preflight", lambda plan, deadline: {})
    for name in ("prepare_exercise_body_source_from_evidence", "prepare_shared_world_seed", "_adapted_shared_world"):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: pytest.fail("No actual source helper call in driver fault checks"))

    def fake_case(case, out, deadline):
        root = out / case["id"]
        root.mkdir()
        (root / "proof.json").write_text(json.dumps({"id": case["id"], "fixture": True}))
        return {"id": case["id"], "status": "passed", "scope": "fake case without source work"}

    monkeypatch.setattr(module, "run_case", fake_case)
    return module, tmp_path, fake_case


def test_fake_completed_study_has_one_matching_terminal_seal(study):
    module, root, _ = study
    assert module.main() == 0
    report = json.loads((root / "study/report.json").read_text())
    manifest = json.loads((root / "study/manifest.json").read_text())
    assert report["status"] == manifest["status"] == "passed"
    assert manifest["files"]["report.json"] == module.sha((root / "study/report.json").read_bytes())
    assert not (root / "study/study-failure.json").exists()


def test_later_case_deadline_retains_completed_case_and_failed_study(study, monkeypatch):
    module, root, fake_case = study
    now = [0.]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])

    def slow_case(case, out, deadline):
        result = fake_case(case, out, deadline)
        now[0] = 87.
        return result

    monkeypatch.setattr(module, "run_case", slow_case)
    assert module.main() == 1
    failure = json.loads((root / "study/study-failure.json").read_text())
    assert failure["status"] == "failed" and failure["phase"] == "case:b"
    assert failure["completed_cases"][0]["id"] == "a"
    assert (root / "study/a/proof.json").exists()
    assert not (root / "study/b").exists()
    assert not (root / "study/report.json").exists()


def test_final_pin_failure_retains_cases_but_never_a_passed_study(study, monkeypatch):
    module, root, _ = study
    calls = []

    def preflight(plan, deadline):
        calls.append(True)
        if len(calls) > 1:
            raise ValueError("Frozen implementation changed")
        return {}

    monkeypatch.setattr(module, "preflight", preflight)
    assert module.main() == 1
    failure = json.loads((root / "study/study-failure.json").read_text())
    assert failure["phase"] == "recheck_inputs" and len(failure["completed_cases"]) == 3
    assert all((root / "study" / name / "proof.json").exists() for name in ("a", "b", "c"))
    assert not (root / "study/report.json").exists()


def test_final_hash_read_failure_uses_reserved_failure_receipt(study, monkeypatch):
    module, root, _ = study
    read = module.read_package_file

    def fail_read(directory, name, **kwargs):
        if name == "a/proof.json":
            raise OSError("Injected final hashing failure")
        return read(directory, name, **kwargs)

    monkeypatch.setattr(module, "read_package_file", fail_read)
    assert module.main() == 1
    failure = json.loads((root / "study/study-failure.json").read_text())
    assert failure["phase"] == "finalize" and failure["error"]["type"] == "OSError"
    assert not (root / "study/report.json").exists()
    assert not (root / "study/manifest.json").exists()


def test_terminal_readback_failure_has_authoritative_failure_even_if_manifest_exists(study, monkeypatch):
    module, root, _ = study
    write = module.write_package_files

    def fail_terminal(directory, files):
        write(directory, files)
        if "manifest.json" in files:
            raise OSError("Injected terminal readback failure")

    monkeypatch.setattr(module, "write_package_files", fail_terminal)
    assert module.main() == 1
    failure = json.loads((root / "study/study-failure.json").read_text())
    assert failure["status"] == "failed" and failure["phase"] == "finalize"
    assert (root / "study/manifest.json").exists()
    report = json.loads((root / "study/report.json").read_text())
    assert "absence of study-failure.json" in report["terminal_rule"]
