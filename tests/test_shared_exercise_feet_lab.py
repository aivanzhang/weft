"""Model-free lab plumbing; native task/recording behavior has separate acceptance."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import mujoco
import pytest

from weft import shared_exercise_feet_recording as feet_recording
from weft import shared_exercise_feet_task as feet_task
from weft import shared_exercise_lab as lab
from weft.io import canonical_hash


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Lab plumbing checks cannot run physics, media, network or subprocesses")

    for owner, names in ((mujoco, ("MjData", "mj_step", "mj_forward", "Renderer")),
                         (mujoco.MjModel, ("from_xml_string", "from_xml_path")),
                         (socket, ("socket", "create_connection")), (subprocess, ("Popen",))):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)


def write(path, value):
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return raw


def seal(root):
    write(root / "manifest.json", {"schema_version": 1, "status": "constructed",
          "files": {p.name: sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.name != "manifest.json"}})


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    scene = {"profile": "shared-world-v5", "base_profile": "shared-world-v3", "objects": [],
             "source_binding": {"mode": "analytic_fixture"},
             "foot_extension": {"reference_json": "{\"fixture\": true}",
                                "reference_sha256": sha256(b'{"fixture": true}').hexdigest()}}
    task = {"profile": "shared-exercise-feet-task-v1", "scene_sha256": canonical_hash(scene),
            "control": {"root_assistance": "disabled"}, "base_task": {"preserved_custom_requirement": True}}
    write(source / "scene.json", scene)
    write(source / "exercise-task.json", task)
    write(source / "inventory.json", {"scope": "mock input"})
    seal(source)
    calls = {"tasks": [], "proof": [], "references": []}

    def validate(task_value, actual):
        calls["tasks"].append((deepcopy(task_value), deepcopy(actual)))
        if task_value.get("profile") != task["profile"] or task_value["scene_sha256"] != canonical_hash(actual):
            raise ValueError("Mock outer task identity differs")
        return deepcopy(task_value)

    def proof(actual, root, **_kwargs):
        calls["proof"].append((actual, root))
        return SimpleNamespace(report={"mode": "analytic_fixture", "source_evidence_verified": False},
                               files={}, validate_layout=lambda **_kwargs: None)

    def reference(actual, _deadline):
        calls["references"].append(actual)
        return {"scope": "mock reference, no physics", "frames": []}

    package = Path(lab.__file__).parent
    monkeypatch.setattr(lab, "validate_shared_world", deepcopy)
    monkeypatch.setattr(feet_task, "validate_shared_exercise_feet_task", validate)
    monkeypatch.setattr(lab, "collect_shared_source_files", proof)
    monkeypatch.setattr(lab, "_reference_frames", reference)
    monkeypatch.setattr(lab, "_implementation_snapshot", lambda: {
        name: (package / name).read_bytes() for name in lab.runtime_files(scene)})
    return SimpleNamespace(source=source, scene=scene, task=task, calls=calls)


@pytest.mark.parametrize("external", [False, True])
def test_feet_lab_seals_exact_task_source_and_new_runtime_closure(inputs, tmp_path, external):
    original = {p.name: p.read_bytes() for p in inputs.source.iterdir()}
    task_path = tmp_path / "explicit-task.json" if external else None
    if external:
        task_path.write_bytes(original["exercise-task.json"])
    target = tmp_path / "lab"
    lab.prepare_shared_exercise_lab(inputs.source, target, task_path=task_path)
    loaded = lab.load_shared_exercise_lab(target / "run")
    assert loaded["scene"] == inputs.scene and loaded["task"] == inputs.task
    assert loaded["contract"]["control_profile"] == "shared_exercise"
    assert loaded["contract"]["learned_policy_available"] is False
    assert "unscored ankle" in loaded["contract"]["scope"]
    assert set(loaded["contract"]["runtime_sha256"]) == set(lab.runtime_files(inputs.scene))
    assert {"shared_exercise_feet_task.py", "shared_exercise_feet_state.py", "shared_exercise_feet_recording.py",
            "shared_world_feet.py", "foot_body.py", "foot_reference.py", "shared_support_task.py",
            "recording.py", "person_inventory.py"} <= set(loaded["contract"]["runtime_sha256"])
    assert (target / "run/exercise-task.json").read_bytes() == original["exercise-task.json"]
    for name, raw in original.items():
        assert (inputs.source / name).read_bytes() == (target / "run/source-run" / name).read_bytes() == raw
    assert inputs.calls["references"] == [inputs.scene]
    assert inputs.calls["tasks"] and all(value == (inputs.task, inputs.scene) for value in inputs.calls["tasks"])


@pytest.mark.parametrize("choice", ["policy", "evaluation", "missing_task", "stale_task", "oversized_task", "roots"])
def test_feet_lab_rejects_unsupported_or_conflicting_admission_without_output(inputs, tmp_path, choice):
    options = {}
    if choice == "policy":
        options["policy_run"] = tmp_path / "unused-policy"
    elif choice == "evaluation":
        options.update(policy_run=tmp_path / "unused-policy", evaluation_run=tmp_path / "unused-evaluation")
    elif choice == "missing_task":
        # Remove only a test-created input, not any project evidence.
        (inputs.source / "exercise-task.json").unlink()
        seal(inputs.source)
    elif choice == "roots":
        options["root_assistance"] = True
    else:
        value = deepcopy(inputs.task)
        if choice == "stale_task":
            value["scene_sha256"] = "f" * 64
        else:
            value["extra"] = "x" * (128 * 1024)
        write(inputs.source / "exercise-task.json", value)
        seal(inputs.source)
    target = tmp_path / "lab"
    with pytest.raises(ValueError):
        lab.prepare_shared_exercise_lab(inputs.source, target, **options)
    assert not target.exists() and not inputs.calls["references"]
    if choice in {"policy", "evaluation", "oversized_task"}:
        assert not inputs.calls["tasks"]


@pytest.mark.parametrize("mode", ["prepare", "resume"])
def test_displayed_reference_cannot_differ_from_actual_simulated_reference(inputs, tmp_path, mode):
    target = tmp_path / "lab"
    if mode == "resume":
        lab.prepare_shared_exercise_lab(inputs.source, target)
        source = target / "run"
        manifest = json.loads((source / "manifest.json").read_bytes())
    else:
        source = inputs.source
        manifest = json.loads((source / "manifest.json").read_bytes())
    raw = write(source / "foot-reference.json", {"different": "but pinned by outer manifest"})
    manifest["files"]["foot-reference.json"] = sha256(raw).hexdigest()
    write(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Displayed foot reference differs"):
        if mode == "prepare":
            lab.prepare_shared_exercise_lab(source, target)
        else:
            lab.load_shared_exercise_lab(source)


@pytest.mark.parametrize("choice", ["missing", "oversized", "policy"])
def test_feet_metadata_preflight_stops_before_native_admission(inputs, tmp_path, monkeypatch, choice):
    if choice == "missing":
        (inputs.source / "exercise-task.json").unlink()
    elif choice == "oversized":
        write(inputs.source / "exercise-task.json", {"oversized": "x" * (128 << 10)})
    seal(inputs.source)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Rejected feet metadata must stop before native scene or source verification")

    monkeypatch.setattr(lab, "validate_shared_world", forbidden)
    monkeypatch.setattr(lab, "collect_shared_source_files", forbidden)
    with pytest.raises(ValueError):
        lab.prepare_shared_exercise_lab(inputs.source, tmp_path / "unused", **(
            {"policy_run": tmp_path / "unqualified"} if choice == "policy" else {}))


@pytest.mark.parametrize("base", ["shared-world-v2", "shared-world-v3", "shared-world-v4"])
def test_runtime_extensions_are_feet_only_and_pin_imported_support_scorer(base):
    original = lab.runtime_files({"profile": base})
    expected = (*lab.RUNTIME_FILES, *(("person_inventory.py",) if base == "shared-world-v3" else
                                     ("shared_support_task.py",) if base == "shared-world-v4" else ()))
    assert original == expected
    feet = lab.runtime_files({"profile": "shared-world-v5", "base_profile": base})
    assert len(feet) == len(set(feet)) and "shared_support_task.py" in feet
    assert ("person_inventory.py" in feet) is (base == "shared-world-v3")
    assert set(original) <= set(feet)


@pytest.mark.parametrize("feet", [False, True])
def test_action_replay_dispatches_by_saved_envelope_without_reading_journal(monkeypatch, tmp_path, feet):
    calls = []
    profile = "shared-exercise-feet-recording-v1" if feet else "shared-exercise-recording-v1"
    write(tmp_path / "scene.json", {"profile": profile})
    monkeypatch.setattr(lab, "verify_action_replay", lambda root, **kw: calls.append(("legacy", root, kw)) or "legacy")
    monkeypatch.setattr(feet_recording, "verify_action_replay", lambda root, **kw: calls.append(("feet", root, kw)) or "feet")
    expected = "feet" if feet else "legacy"
    assert lab.verify_shared_exercise_action_replay(tmp_path, max_seconds=7) == expected
    assert calls == [(expected, tmp_path, {"max_seconds": 7})]


def test_session_uses_one_feet_environment_and_dedicated_action_validator(monkeypatch, tmp_path):
    from weft import capabilities

    scene = {"profile": "shared-world-v5"}
    task = {"profile": "shared-exercise-feet-task-v1"}
    envelope = {"profile": "shared-exercise-feet-recording-v1"}
    calls = []

    class Env:
        def __init__(self, actual, outer):
            calls.append(("env", actual, outer))

        def reset(self, **options):
            calls.append(("reset", options))
            return [1], {}

        def snapshot(self):
            return {"profile": "shared-exercise-feet-state-v1"}

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(lab, "load_shared_exercise_lab", lambda *a, **k: {
        "scene": scene, "task": task, "contract": {}, "artifacts": [], "source_verification": {}})
    monkeypatch.setattr(feet_task, "SharedExerciseFeetTaskEnv", Env)
    monkeypatch.setattr(feet_recording, "recording_scene", lambda actual, outer: envelope)
    monkeypatch.setattr(feet_recording, "validate_action", lambda action, saved: calls.append(("action", action, saved)))
    monkeypatch.setattr(capabilities, "describe_capabilities", lambda *a, **k: {"scene_profile": scene["profile"]})
    session = lab.SharedExerciseSession(tmp_path)
    assert session.recording_scene() == envelope
    assert session.resolve_action([0, .5]) == [0., .5]
    assert calls == [("env", scene, task), ("reset", {"seed": 0, "options": lab.RESET_OPTIONS}),
                     ("action", [0, .5], envelope)]
    assert session.capabilities() == {"scene_profile": "shared-world-v5"}
    with pytest.raises(ValueError, match="no learned policy"):
        session.policy_action()


def test_feet_avatar_options_stop_before_recording(monkeypatch, tmp_path):
    from weft.lab import Lab

    closed = []
    session = SimpleNamespace(scene_profile="shared-world-v5", env=SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(lab, "SharedExerciseSession", lambda *a, **k: session)
    instance = Lab.__new__(Lab)
    instance.session, instance.root = None, tmp_path
    instance.config = {"profile": "shared-world", "control_profile": "shared_exercise"}
    instance.avatars = SimpleNamespace(packages={"avatar": object()})
    instance.status = lambda: {"status": "ready"}
    with pytest.raises(ValueError, match="without avatar options"):
        instance.load_session()
    assert closed == [True] and instance.session is None and not list(tmp_path.iterdir())


def test_cli_builds_one_candidate_without_starting_a_server(monkeypatch, capsys):
    from weft.cli import main

    calls = []
    module = ModuleType("weft.shared_exercise_feet_candidate")
    module.prepare_shared_feet_candidate = lambda *a, **k: calls.append((a, k)) or {"status": "constructed"}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "argv", ["weft", "shared-feet-build", "--run", "base", "--foot-reference", "feet.json",
                                    "--task", "task.json", "--out", "candidate", "--max-seconds", "12"])
    main()
    assert calls == [(("base", "feet.json", "candidate"), {"task_path": "task.json", "max_seconds": 12.})]
    assert json.loads(capsys.readouterr().out)["status"] == "constructed"


def test_cli_verifies_actions_through_profile_dispatch(monkeypatch, capsys):
    from weft.cli import main
    from weft import recording

    calls = []
    monkeypatch.setattr(recording, "verify_recording", lambda path: {"directory": path, "verified_files": 6})
    monkeypatch.setattr(lab, "verify_shared_exercise_action_replay", lambda path: calls.append(path) or {"exact_state_replay": True})
    monkeypatch.setattr(sys, "argv", ["weft", "replay", "saved-recording", "--verify-actions"])
    main()
    assert calls == ["saved-recording"]
    assert json.loads(capsys.readouterr().out)["action_replay"] == {"exact_state_replay": True}
