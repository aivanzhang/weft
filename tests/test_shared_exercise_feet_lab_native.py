"""Bounded lab integration, not recovery accuracy or successful physical support."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

from test_shared_exercise_feet_candidate import inputs, seal
from test_shared_exercise_feet_native import local_native_only
from weft import cli, lab_console
from weft.capabilities import describe_run_capabilities
from weft.io import canonical_hash
from weft.lab import Lab
from weft.recording import load_recording
from weft.recording_receipt import load_recording_evidence
from weft.shared_exercise_feet_task import SharedExerciseFeetTaskEnv
from weft.shared_exercise_task import build_shared_exercise_task
from weft.shared_support_task import evaluator_identity


def invoke(monkeypatch, capsys, *arguments):
    monkeypatch.setattr(sys, "argv", ["weft", *map(str, arguments)])
    cli.main()
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("assistance", [False, True], ids=["roots-off", "roots-on"])
def test_native_feet_candidate_lab_console_receipt(tmp_path, monkeypatch, capsys, local_native_only, assistance):
    source, reference, original, task, files, _ = inputs(tmp_path, "shared-world-v4")
    if assistance:
        task = build_shared_exercise_task(original, root_assistance=True,
                                          support_requirements=task["support_requirements"])
        files["exercise-task.json"] = (json.dumps(task, separators=(",", ":")) + "\n\n").encode()
        seal(source, files)
    original_bytes = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    candidate, target = tmp_path / "candidate", tmp_path / "lab"
    built = invoke(monkeypatch, capsys, "shared-feet-build", "--run", source,
                   "--foot-reference", reference, "--out", candidate, "--max-seconds", 20)
    assert built["status"] == "constructed"
    actual = json.loads((candidate / "scene.json").read_bytes())
    outer = json.loads((candidate / "exercise-task.json").read_bytes())
    expected_task = deepcopy(task)
    expected_task["scene_sha256"] = outer["base_scene_sha256"]
    assert outer["base_task"] == expected_task
    assert (candidate / "input-task.json").read_bytes() == files["exercise-task.json"]
    caps = describe_run_capabilities(candidate)
    assert caps["action_contract"]["dimension"] == (28 if assistance else 25)
    assert caps["task_contract"]["base_scene_profile"] == "shared-world-v4"
    prepared = invoke(monkeypatch, capsys, "shared-exercise-lab", "--run", candidate,
                      "--out", target, "--max-seconds", 20)
    assert prepared["learned_policy_available"] is False
    run = target / "run"
    before = {p.relative_to(run).as_posix(): sha256(p.read_bytes()).hexdigest()
              for p in run.rglob("*") if p.is_file()}
    assert (run / "foot-reference.json").read_bytes() == actual["foot_extension"]["reference_json"].encode()
    controls = 0
    real_step = SharedExerciseFeetTaskEnv.step

    def bounded_step(instance, action):
        nonlocal controls
        assert controls < 8, "At most four live and four replay controls per fixture"
        controls += 1
        return real_step(instance, action)

    monkeypatch.setattr(SharedExerciseFeetTaskEnv, "step", bounded_step)
    commands = iter(["controls", "record start", "motor left_ankle_z 0.1", "step 2",
                     "reset 29", "mode reference", "step 2", "status", "record stop", "quit"])
    transcript = []
    original_console = lab_console.console

    def scripted_console(root, *, seed):
        return original_console(root, seed=seed, input_fn=lambda _: next(commands), output_fn=transcript.append)

    monkeypatch.setattr(lab_console, "console", scripted_console)
    played = invoke(monkeypatch, capsys, "lab-console", "--resume", target, "--seed", 19)
    (tmp_path / "console.txt").write_text("\n".join(transcript) + "\n")
    (tmp_path / "console-result.json").write_text(json.dumps(played, indent=2) + "\n")
    recording = played["recording"]
    assert recording["export"] == "ready", recording
    assert recording["verified_action_replay"]["exact_state_replay"] is True
    sessions = [p for p in (target / "sessions").iterdir() if p.is_dir()]
    assert len(sessions) == 1
    journal = load_recording(sessions[0])
    steps = [row for row in journal["events"] if row["type"] == "step"]
    assert 2 <= len(steps) <= 4 and controls == 2 * len(steps)
    assert {row["controller"] for row in steps} == {"manual", "authored"}
    assert len([row for row in journal["events"] if row["type"] == "reset"]) == 1
    expected_evaluator = evaluator_identity({name: sha256((run / "lab-implementation" / name).read_bytes()).hexdigest()
                                             for name in ("shared_exercise_task.py", "shared_support_task.py")})
    for event in journal["events"]:
        state = event["snapshot"]
        assert state["task_metrics"]["evaluator_sha256"] == expected_evaluator
        assert state["task_metrics"]["task_sha256"] == outer["base_task_sha256"]
        assert state["ankle_diagnostics"]["scored"] is False
        assert state["ankle_diagnostics"]["foot_support_credit"] is False
    before_inspection = controls
    evidence = load_recording_evidence(target, sessions[0].name)
    assert evidence["status"] == "available", evidence
    assert evidence["replay"]["protocol"] == "shared-exercise-feet-full-state-replay-v1"
    assert evidence["replay"]["status"] == "passed"
    assert evidence["recorded"]["actions"] == len(steps)
    assert evidence["identities"]["task_sha256"] == canonical_hash(outer)
    assert evidence["identities"]["foot_reference_sha256"] == outer["foot_reference_sha256"]
    assert controls == before_inspection
    resumed = Lab(target)
    try:
        assert resumed.summary()["snapshot"]["engine_state"]["steps"] == 0
    finally:
        resumed.close()
    assert before == {p.relative_to(run).as_posix(): sha256(p.read_bytes()).hexdigest()
                      for p in run.rglob("*") if p.is_file()}
    assert original_bytes == {p.relative_to(source).as_posix(): p.read_bytes()
                              for p in source.rglob("*") if p.is_file()}
    (tmp_path / "acceptance.json").write_text(json.dumps({
        "scope": "Authored fixture integration; distant support remains unsuccessful, not physical qualification",
        "candidate": built, "lab": prepared, "live_actions": len(steps), "total_controls": controls,
        "receipt": evidence, "final_metrics": steps[-1]["snapshot"]["task_metrics"],
        "root_assistance": outer["control"]["root_assistance"], "evaluator_sha256": expected_evaluator,
    }, indent=2, allow_nan=False) + "\n")
