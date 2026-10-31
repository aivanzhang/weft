"""Guided object stages compose existing commands without silently changing evidence."""

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("spatial_builder_test", PROJECT / "scripts/build_sandbox.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def raw(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def seal(root, members, *, legacy=False):
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, value in members.items():
        data = value if isinstance(value, bytes) else raw(value)
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[name] = sha(data)
    (root / "manifest.json").write_bytes(raw({"schema" if legacy else "schema_version": 1, "files": files}))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / ".venv-lab/bin").mkdir(parents=True)
    (root / ".venv-lab/bin/python").symlink_to(sys.executable)
    (root / "scripts").mkdir()
    for name in ("build_sandbox.py", "spatial_sandbox_workflow.py"):
        (root / "scripts" / name).write_bytes((PROJECT / "scripts" / name).read_bytes())
    scene = {"profile": "planar-rigid-v1", "objects": [{"id": "ball"}], "dt": 1 / 30}
    seal(root / "source", {"scene.json": scene}, legacy=True)
    config = {"profile": "planar-to-spatial-hypothesis-v1", "source_scene_sha256": canonical(scene),
              "assumptions": ["Authored depth, gravity, materials and target; not measured 3D physics."]}
    (root / "spatial.json").write_bytes(raw(config))
    (root / "tracking.json").write_bytes(raw({"explicit": "tracking regions and planar priors"}))
    (root / "video.avi").write_bytes(b"fixture bytes; never decoded")
    monkeypatch.setattr(launcher, "PROJECT", root)
    monkeypatch.setattr(launcher.subprocess, "call", lambda *_a, **_k: pytest.fail("Unrequested interaction"))
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"mujoco", "cv2", "gymnasium", "mediapipe", "numpy", "weft"}:
            pytest.fail("Launcher must not load native, media or inference code")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    return root


def args(root, *extra, direct=False):
    source = ["--profile", "spatial", "--video", str(root / "video.avi"), "--tracking-profile", "appearance",
              "--tracking-config", str(root / "tracking.json")] if direct else ["--planar-run", str(root / "source")]
    return source + ["--config", str(root / "spatial.json"), "--out", str(root / "result"), *extra]


def spatial_scene():
    return {"schema_version": 1, "profile": "spatial-rigid-v1", "objects": [{"id": "ball"}], "task": {"max_steps": 300}}


def learning(root, scene, *, episodes=12, mismatch=False, legacy=False):
    scene_sha = "f" * 64 if mismatch else canonical(scene)
    policy = {"profile": "spatial-policy-v2", "scene_sha256": scene_sha, "physical_validation": "unverified"}
    regimes = {"nominal_exact": {name: {"episodes": 1, "successes": 0} for name in ("learned", "random", "coast", "brake")},
               "changed_initial_conditions": {"policies": {name: {"episodes": episodes, "successes": 0}
                                                           for name in ("learned", "random", "coast", "brake")}}}
    evaluation = {"scene_sha256": scene_sha, "policy_sha256": sha(raw(policy)), "physical_validation": "unverified"}
    if not legacy:
        evaluation["regimes"] = regimes
    seal(root, {"scene.json": scene, "policy.json": policy, "rl-evaluation.json": evaluation,
                "training.json": {"completed_generations": 8}})


def worker(root, monkeypatch, *, corrupt=None, fail=None, mutate=None):
    calls = []

    def run(command, output, stage, seconds):
        calls.append((stage, command, seconds))
        for suffix in (".stdout.log", ".stderr.log"):
            (output / (stage + suffix)).write_bytes(b"bounded mocked output")
        if fail == stage:
            (output / (stage + "-partial.json")).write_bytes(raw({"failed": True}))
            return {"returncode": 1, "error": {"type": "TimeoutError"}, "interrupted": False}
        if stage == "tracking":
            scene = json.loads((root / "source/scene.json").read_bytes())
            if corrupt == "source_pin":
                scene["dt"] = .05
            report = {"source_sha256": sha((root / "video.avi").read_bytes()), "learning": {"status": "not_requested"}}
            if corrupt == "video_pin":
                report["source_sha256"] = "f" * 64
            seal(output / "planar-lab/run", {"scene.json": scene, "report.json": report}, legacy=True)
            (output / "planar-lab/status.json").write_bytes(raw({"status": "ready"}))
        elif stage == "construction":
            source = Path(command[command.index("--run") + 1])
            scene_name = command[command.index("--scene-artifact") + 1]
            manifest = (source / "manifest.json").read_bytes()
            config_raw = (output / "inputs/spatial-config.json").read_bytes()
            scene = spatial_scene()
            report = {"status": "constructed_hypothesis", "physical_validation": "unverified",
                      "source_manifest_sha256": sha(manifest), "scene_sha256": canonical(scene)}
            if corrupt == "construction_binding":
                report["source_manifest_sha256"] = "f" * 64
            seal(output / "candidate", {"scene.json": scene, "report.json": report,
                "inputs/planar/manifest.json": manifest, "inputs/original-scene.json": (source / scene_name).read_bytes(),
                "inputs/authored-config.json": config_raw})
            if corrupt == "construction_tamper":
                (output / "candidate/scene.json").write_bytes(raw({"substituted": True}))
        elif stage == "training":
            learning(output / "learning", spatial_scene(), mismatch=corrupt == "policy_scene")
            if corrupt == "policy_missing":
                (output / "learning/manifest.json").unlink()
            if corrupt == "evaluation_counts":
                path = output / "learning"
                docs = {name: (path / name).read_bytes() for name in ("scene.json", "policy.json", "training.json")}
                evaluation = json.loads((path / "rl-evaluation.json").read_bytes())
                evaluation["regimes"]["nominal_exact"]["learned"]["successes"] = 2
                seal(path, {**docs, "rl-evaluation.json": evaluation})
        elif stage == "lab":
            candidate = output / "candidate"
            requested = "--learning" in command
            members = {"scene.json": spatial_scene(), "source-input-manifest.json": (candidate / "manifest.json").read_bytes(),
                       "report.json": {"status": "ready_spatial_lab", "physical_validation": "unverified",
                                       "learned_policy_available": requested}}
            if requested:
                source = Path(command[command.index("--learning") + 1])
                members.update({"learning-input-manifest.json": (source / "manifest.json").read_bytes(),
                                "policy.json": (source / "policy.json").read_bytes()})
            if corrupt == "lab_scene":
                members["scene.json"] = {**spatial_scene(), "task": {"max_steps": 2}}
            if corrupt == "unexpected_policy":
                members["policy.json"] = {"unexpected": True}
            seal(output / "lab/run", members)
            (output / "lab/status.json").write_bytes(raw({"status": "ready"}))
            if corrupt == "lab_failure_marker":
                (output / "lab/run/failure.json").write_bytes(b"{}")
            if mutate:
                mutate(root, output)
        return {"returncode": 0, "error": None, "interrupted": False, "seconds": .01}

    monkeypatch.setattr(launcher, "_run", run)
    return calls


def result(root):
    return json.loads((root / "result/workflow-result.json").read_bytes())


def test_saved_manual_chain_preserves_inputs_and_has_no_implicit_training(workspace, monkeypatch):
    source_before = (workspace / "source/manifest.json").read_bytes()
    calls = worker(workspace, monkeypatch)
    assert launcher.main(args(workspace)) == 0
    assert [c[0] for c in calls] == ["construction", "lab"]
    assert all(c[2] > 0 for c in calls)
    assert "--manifest-sha256" in calls[0][1] and "--learning" not in calls[-1][1]
    receipt = result(workspace)
    assert receipt["status"] == "ready" and receipt["learning"] == {"status": "not_requested"}
    assert receipt["physical_validation"] == "unverified" and not receipt["promoted"]
    assert receipt["api_calls"] == receipt["retries"] == 0
    assert set(receipt["resume_commands"]) == {"browser", "console"}
    assert (workspace / "source/manifest.json").read_bytes() == source_before
    assert "workflow-result.json" in (workspace / "result/index.html").read_text()


def test_training_settings_and_unsuccessful_task_are_reported_honestly(workspace, monkeypatch):
    calls = worker(workspace, monkeypatch)
    assert launcher.main(args(workspace, "--train", "--training-max-seconds", "17")) == 0
    assert [c[0] for c in calls] == ["construction", "training", "lab"]
    command = calls[1][1]
    for flag, value in {"--seed": "1729", "--eval-seed": "20261031", "--generations": "3", "--population": "8",
                        "--eval-episodes": "12", "--max-seconds": "17.0"}.items():
        assert command[command.index(flag) + 1] == value
    assert calls[1][2] <= 22
    learned = result(workspace)["learning"]
    assert learned["mode"] == "trained" and learned["evidence_partition"] == "exposed_development"
    assert learned["regimes"]["changed_initial_conditions"]["learned"] == {"successes": 0, "episodes": 12}
    assert "0/12 successful trials" in (workspace / "result/index.html").read_text()


@pytest.mark.parametrize("legacy", [False, True])
def test_reuse_does_not_train_and_allows_other_saved_evaluation_counts(workspace, monkeypatch, legacy):
    learning(workspace / "existing-policy", spatial_scene(), episodes=4, legacy=legacy)
    calls = worker(workspace, monkeypatch)
    assert launcher.main(args(workspace, "--learning", str(workspace / "existing-policy"))) == 0
    assert [c[0] for c in calls] == ["construction", "lab"]
    assert calls[-1][1][-2:] == ["--learning", str(workspace / "existing-policy")]
    saved = result(workspace)["learning"]
    assert saved["mode"] == "reused"
    if legacy:
        assert saved["regimes"] == {} and "unavailable" in saved["reason"]
    else:
        assert saved["regimes"]["changed_initial_conditions"]["learned"]["episodes"] == 4


def test_direct_video_explicit_config_and_source_binding(workspace, monkeypatch):
    calls = worker(workspace, monkeypatch)
    assert launcher.main(args(workspace, direct=True)) == 0
    assert [c[0] for c in calls] == ["tracking", "construction", "lab"]
    command = calls[0][1]
    assert "--no-train" in command and "--no-serve" in command
    assert command[command.index("--profile") + 1] == "appearance"
    assert result(workspace)["video_hash_bytes"] == (workspace / "video.avi").stat().st_size


@pytest.mark.parametrize("corrupt", ["source_pin", "video_pin"])
def test_direct_video_mismatch_keeps_planar_output_without_constructing(workspace, monkeypatch, corrupt):
    config_before = (workspace / "spatial.json").read_bytes()
    calls = worker(workspace, monkeypatch, corrupt=corrupt)
    assert launcher.main(args(workspace, "--serve", direct=True)) == 1
    assert [c[0] for c in calls] == ["tracking"]
    assert (workspace / "result/planar-lab/run/scene.json").exists()
    assert (workspace / "spatial.json").read_bytes() == config_before
    assert result(workspace)["lab"] is None


@pytest.mark.parametrize("failure", ["tracking", "construction", "training", "lab"])
def test_failed_process_preserves_receipt_and_never_continues_or_launches(workspace, monkeypatch, failure):
    calls = worker(workspace, monkeypatch, fail=failure)
    assert launcher.main(args(workspace, "--train", "--console", direct=True)) == 1
    assert calls[-1][0] == failure
    assert result(workspace)["stages"][-1]["status"] == "stopped"
    assert (workspace / "result" / (failure + "-partial.json")).exists()
    assert (workspace / "result/index.html").exists()


@pytest.mark.parametrize("corrupt", ["construction_binding", "construction_tamper", "policy_scene", "policy_missing",
                                     "evaluation_counts", "lab_scene", "lab_failure_marker"])
def test_metadata_tampering_and_mismatches_stop_before_handoff(workspace, monkeypatch, corrupt):
    worker(workspace, monkeypatch, corrupt=corrupt)
    assert launcher.main(args(workspace, "--train", "--console")) == 1
    assert result(workspace)["lab"] is None and "resume_commands" not in result(workspace)


def test_manual_does_not_accept_unrequested_policy(workspace, monkeypatch):
    worker(workspace, monkeypatch, corrupt="unexpected_policy")
    assert launcher.main(args(workspace)) == 1


@pytest.mark.parametrize("flag", ["--serve", "--console"])
def test_interaction_only_after_ready_when_explicit(workspace, monkeypatch, flag):
    worker(workspace, monkeypatch)
    launched = []
    monkeypatch.setattr(launcher.subprocess, "call", lambda command, **kwargs: launched.append((command, kwargs)) or 0)
    assert launcher.main(args(workspace, flag)) == 0
    assert len(launched) == 1
    assert ("lab-console" in launched[0][0]) is (flag == "--console")
    assert not any("KEY" in key for key in launched[0][1]["env"])


@pytest.mark.parametrize("extra", [["--train", "--learning", "policy"], ["--person-only"], ["--root-assistance"],
    ["--pose-model", "model"], ["--profile", "exercise"], ["--policy-family", "phase-residual"],
    ["--tracking-config", "tracking"], ["--training-max-seconds", "10"]])
def test_mixed_scope_rejected_before_output(workspace, extra):
    with pytest.raises(SystemExit):
        launcher.main(args(workspace, *extra))
    assert not (workspace / "result").exists()


@pytest.mark.parametrize("missing", ["--config", "--tracking-profile", "--tracking-config"])
def test_direct_video_requires_explicit_inputs(workspace, missing):
    command = args(workspace, direct=True)
    index = command.index(missing)
    del command[index:index + 2]
    with pytest.raises(SystemExit):
        launcher.main(command)
    assert not (workspace / "result").exists()


@pytest.mark.parametrize("flag,value", [("--max-seconds", "nan"), ("--max-seconds", "361"),
                                       ("--training-max-seconds", "inf"), ("--training-max-seconds", "0")])
def test_deadlines_invalid_before_output(workspace, flag, value):
    with pytest.raises(ValueError, match="deadline"):
        launcher.main(args(workspace, "--train", flag, value))
    assert not (workspace / "result").exists()


def test_nested_scene_selection_and_html_escape(workspace, monkeypatch):
    original = (workspace / "source/scene.json").read_bytes()
    seal(workspace / "nested", {"hypothesis/scene.json": original})
    config = json.loads((workspace / "spatial.json").read_bytes())
    config["assumptions"] = ["<script>alert('not executable')</script>"]
    (workspace / "spatial.json").write_bytes(raw(config))
    calls = worker(workspace, monkeypatch)
    command = args(workspace, "--scene-artifact", "hypothesis/scene.json")
    command[command.index("--planar-run") + 1] = str(workspace / "nested")
    assert launcher.main(command) == 0
    assert "hypothesis/scene.json" in calls[0][1]
    page = (workspace / "result/index.html").read_text()
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_consumed_input_changes_block_ready_without_repeating_stages(workspace, monkeypatch):
    calls = worker(workspace, monkeypatch, mutate=lambda root, out: (root / "spatial.json").write_bytes(b"{}"))
    assert launcher.main(args(workspace)) == 1
    assert [c[0] for c in calls] == ["construction", "lab"]
    assert "changed during execution" in result(workspace)["error"]["message"]


def test_output_collision_and_overlap_are_not_overwritten(workspace):
    (workspace / "result").mkdir()
    (workspace / "result/owned.txt").write_text("preserved")
    with pytest.raises(ValueError, match="already exists"):
        launcher.main(args(workspace))
    assert (workspace / "result/owned.txt").read_text() == "preserved"
    command = args(workspace)
    command[command.index("--out") + 1] = str(workspace / "source/child")
    with pytest.raises(ValueError, match="outside"):
        launcher.main(command)


def test_parent_traversal_scene_rejects_before_output(workspace):
    with pytest.raises(ValueError, match="Unsafe"):
        launcher.main(args(workspace, "--scene-artifact", "../scene.json"))
    assert not (workspace / "result").exists()


def test_failed_source_manifest_status_never_starts_construction(workspace, monkeypatch):
    path = workspace / "source/manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["status"] = "failed"
    path.write_bytes(raw(manifest))
    calls = worker(workspace, monkeypatch)
    assert launcher.main(args(workspace)) == 1 and not calls
    assert "unsuccessful" in result(workspace)["error"]["message"]


def test_reused_policy_scene_mismatch_never_prepares_lab(workspace, monkeypatch):
    learning(workspace / "existing-policy", spatial_scene(), episodes=4, mismatch=True)
    calls = worker(workspace, monkeypatch)
    assert launcher.main(args(workspace, "--learning", str(workspace / "existing-policy"))) == 1
    assert [c[0] for c in calls] == ["construction"]
    assert result(workspace)["learning"]["status"] == "requested_not_completed"


def test_interrupted_child_has_no_handoff_or_retry(workspace, monkeypatch):
    calls = []

    def interrupt(command, *rest):
        calls.append(command)
        return {"returncode": 0, "error": None, "interrupted": True}

    monkeypatch.setattr(launcher, "_run", interrupt)
    assert launcher.main(args(workspace, "--serve")) == 1 and len(calls) == 1
    assert result(workspace)["stages"][-1]["status"] == "stopped"


@pytest.mark.parametrize("kind", ["deadline", "config_bytes"])
def test_workflow_resource_limits_stop_without_a_child(workspace, monkeypatch, kind):
    calls = worker(workspace, monkeypatch)
    extra = []
    if kind == "deadline":
        extra = ["--max-seconds", "1"]
    else:
        (workspace / "spatial.json").write_bytes(b" " * (65 << 10))
    assert launcher.main(args(workspace, *extra)) == 1 and not calls
    assert (workspace / "result/index.html").exists()
    assert result(workspace)["status"] == "stopped"
