"""Manual video labs retain reconstruction/admission gates without running a learner."""

from copy import deepcopy
import json
import sys

import pytest

from weft import cli, lab as lab_module, lab_worker
from weft.capabilities import describe_run_capabilities
from weft.io import digest, write_json
from weft.lab import Lab, prepare_lab
from weft.lab_console import Controls
from weft.recording import load_recording


def scene_fixture():
    return {
        "schema_version": 1, "profile": "planar-rigid-v1", "width": 1, "height": .6,
        "dt": 1 / 30, "gravity": [0, 0], "damping": .9, "restitution": .65,
        "objects": [{"id": "actor", "label": "Actor", "role": "active", "shape": "circle",
                     "position": [.2, .3], "velocity": [0, 0], "radius": .03,
                     "color": "#DDCC00", "mass": 1, "provenance": {"geometry": "synthetic"}}],
        "controlled_id": "actor", "target": [.8, .3],
        "task": {"position_tolerance": .06, "speed_tolerance": .15, "max_steps": 300, "acceleration": 2},
        "relationships": [], "assumptions": ["Synthetic software fixture; not physical video evidence"],
    }


def seal(run):
    write_json(run / "manifest.json", {"files": {
        p.name: digest(p) for p in run.iterdir() if p.is_file() and p.name != "manifest.json"}})


def ready_lab(root, profile="planar", *, manual=True, legacy=False):
    run = root / "run"
    run.mkdir(parents=True)
    settings = {"profile": profile}
    if manual:
        settings.update(training_enabled=False, episodes=None, eval_episodes=None)
    write_json(root / "lab-config.json", settings)
    write_json(root / "status.json", {"status": "ready", "events": []})
    write_json(run / "scene.json", scene_fixture())
    if not legacy:
        write_json(run / "run-config.json", settings)
    if not manual:
        write_json(run / "policy.json", {"schema_version": 1, "algorithm": "tabular_q_learning",
            "error_edges": [-.1, .1], "velocity_edges": [-.1, .1], "q_values": {}})
    seal(run)
    return root


@pytest.mark.parametrize("profile", ["planar", "appearance"])
def test_manual_session_steps_records_resets_and_rejects_learned_without_mutation(tmp_path, profile):
    root = ready_lab(tmp_path / "lab", profile)
    original = {p.name: p.read_bytes() for p in (root / "run").iterdir()}
    instance = Lab(root)
    try:
        summary = instance.summary()
        controllers = summary["capabilities"]["controllers"]
        assert summary["learned_policy_available"] is False
        assert controllers["manual"] and not controllers["learned_policy_available"]
        assert controllers["trained_target_id"] is None
        assert summary["capabilities"]["evidence_separation"]["physical_validation"] == "unverified"
        assert describe_run_capabilities(root / "run")["controllers"] == controllers
        controls = Controls(summary["capabilities"])
        with pytest.raises(ValueError, match="learned controller"):
            controls.select_mode("learned")
        before = instance.session.snapshot()
        assert instance.control("/api/step", {"action": 2, "steps": 2})["snapshot"] != before
        started = instance.control("/api/record/start", {})
        assert started["snapshot"]["steps"] == 0
        instance.control("/api/step", {"action": 2, "steps": 2})
        saved = instance.session.snapshot()
        journal = root / "sessions" / started["directory"] / "events.jsonl"
        prior_journal = journal.read_bytes()
        for done in (False, True):
            instance.session.done = done
            with pytest.raises(ValueError, match="no learned policy"):
                instance.control("/api/step", {"policy": True, "steps": 2})
            assert instance.session.snapshot() == saved and journal.read_bytes() == prior_journal
        instance.control("/api/reset", {"seed": 7})
        instance.control("/api/step", {"action": 1})
    finally:
        # Closing preserves the real journal without invoking rendering or a replay export.
        instance.close()
    recording = load_recording(root / "sessions" / started["directory"])
    assert [event["type"] for event in recording["events"]] == ["start", "step", "step", "reset", "step"]
    assert [event["action"] for event in recording["events"] if event["type"] == "step"] == [2, 2, 1]
    assert recording["metadata"]["status"] == "closed"
    assert {p.name: p.read_bytes() for p in (root / "run").iterdir()} == original
    resumed = Lab(root)
    try:
        assert resumed.summary()["learned_policy_available"] is False
        assert resumed.session.snapshot()["steps"] == 0
    finally:
        resumed.close()


@pytest.mark.parametrize("mutation", ["mutable_mode", "mutable_profile", "unsealed_marker", "tampered_marker",
                                       "unexpected_policy", "unexpected_sealed_policy", "missing_marker", "episode_request"])
def test_manual_admission_rejects_mismatched_or_unsealed_opt_out(tmp_path, mutation):
    root = ready_lab(tmp_path / "lab")
    run = root / "run"
    settings = json.loads((run / "run-config.json").read_text())
    if mutation in ("mutable_mode", "mutable_profile"):
        changed = dict(settings, **({"training_enabled": True} if mutation == "mutable_mode" else {"profile": "appearance"}))
        write_json(root / "lab-config.json", changed)
    elif mutation == "unsealed_marker":
        manifest = json.loads((run / "manifest.json").read_text())
        del manifest["files"]["run-config.json"]
        write_json(run / "manifest.json", manifest)
    elif mutation == "missing_marker":
        (run / "run-config.json").unlink()
        seal(run)
    elif mutation in ("unexpected_policy", "unexpected_sealed_policy"):
        write_json(run / "policy.json", {})
        if mutation == "unexpected_sealed_policy":
            seal(run)
    else:
        settings.update(training_enabled=True) if mutation == "tampered_marker" else settings.update(episodes=1)
        write_json(run / "run-config.json", settings)
        if mutation == "episode_request":
            seal(run)
    instance = Lab(root)
    try:
        with pytest.raises(ValueError):
            instance.load_session()
        assert instance.session is None
    finally:
        instance.close()


@pytest.mark.parametrize("failure", ["missing", "tampered", "removed_from_seal", "mutable_opt_out"])
def test_old_trained_labs_never_turn_policy_failures_into_manual_mode(tmp_path, failure):
    root = ready_lab(tmp_path / "lab", manual=False, legacy=True)
    policy = root / "run" / "policy.json"
    if failure in ("missing", "removed_from_seal"):
        policy.unlink()
        if failure == "removed_from_seal":
            seal(root / "run")
    elif failure == "tampered":
        policy.write_text("{}\n")
    else:
        write_json(root / "lab-config.json", {"profile": "planar", "training_enabled": False})
    instance = Lab(root)
    try:
        with pytest.raises(ValueError):
            instance.load_session()
    finally:
        instance.close()


def test_legacy_trained_planar_control_is_unchanged(tmp_path):
    instance = Lab(ready_lab(tmp_path / "lab", manual=False, legacy=True))
    try:
        assert instance.summary()["learned_policy_available"] is True
        assert instance.capabilities(instance.session)["controllers"]["trained_target_id"] == "actor"
        assert len(instance.control("/api/step", {"policy": True})["transitions"]) == 1
    finally:
        instance.close()


@pytest.mark.parametrize("profile", ["planar", "appearance"])
def test_manual_http_controls_record_and_reject_learned(tmp_path, profile):
    from test_lab_http import request, serving

    root = ready_lab(tmp_path / "lab", profile)
    with serving(Lab(root)) as server:
        code, _, body = request(server, "/api/status")
        assert code == 200 and json.loads(body)["learned_policy_available"] is False
        code, _, body = request(server, "/api/record/start", method="POST", data={})
        assert code == 200
        name = json.loads(body)["directory"]
        journal = root / "sessions" / name / "events.jsonl"
        before = journal.read_bytes()
        code, _, body = request(server, "/api/step", method="POST", data={"policy": True})
        assert code == 400 and "no learned policy" in json.loads(body)["error"]
        assert journal.read_bytes() == before and server.lab.session.snapshot()["steps"] == 0
        code, _, body = request(server, "/api/step", method="POST", data={"action": 2, "steps": 2})
        assert code == 200 and json.loads(body)["snapshot"]["steps"] == 2
        code, _, body = request(server, "/api/reset", method="POST", data={"seed": 7})
        assert code == 200 and json.loads(body)["snapshot"]["steps"] == 0
    recording = load_recording(root / "sessions" / name)
    assert [event["type"] for event in recording["events"]] == ["start", "step", "step", "reset"]
    assert recording["metadata"]["status"] == "closed"


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.mp4"
    path.write_bytes(b"Not decoded by these focused orchestration checks")
    return path


@pytest.mark.parametrize("options", [{"profile": "rolling"}, {"profile": "exercise"}, {"episodes": 1},
                                    {"eval_episodes": 1}, {"no_train": 1}])
def test_invalid_no_train_options_fail_before_creating_output(source, tmp_path, options):
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="no-train"):
        prepare_lab(source, out, **({"no_train": True} | options))
    assert not out.exists()


@pytest.mark.parametrize("mode", ["resume", "scene_run", "shared_world_run", "learning"])
def test_cli_no_train_does_not_relabel_or_import_saved_runs(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(sys, "argv", ["weft", "lab", "--no-train", "--" + mode.replace("_", "-"), str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("manual", [False, True])
def test_public_cli_resolves_training_defaults_and_opt_out(source, tmp_path, monkeypatch, manual):
    calls = []

    class ReadyWorker:
        def __init__(self, root, **_options):
            self.root = root
        def start_worker(self):
            calls.append(json.loads((self.root / "lab-config.json").read_text()))
        def status(self):
            return {"status": "ready"}
        def close(self):
            pass

    monkeypatch.setattr(lab_module, "Lab", ReadyWorker)
    out = tmp_path / "new"
    argv = ["weft", "lab", "--profile", "planar", "--video", str(source), "--out", str(out), "--no-serve"]
    monkeypatch.setattr(sys, "argv", argv + (["--no-train"] if manual else []))
    cli.main()
    assert len(calls) == 1
    settings = calls[0]
    assert settings["episodes"] == (None if manual else 1500)
    assert settings["eval_episodes"] == (None if manual else 100)
    assert settings.get("training_enabled", True) is not manual
    if not manual:
        assert "training_enabled" not in settings


@pytest.fixture
def pipeline_stubs(monkeypatch):
    from weft import generic_vision, scene, scene_learning, scene_vision
    calls = {"inspect": [], "fit": 0, "steps": 0, "train": []}

    def inspect(_video, _out, config=None, progress=None):
        calls["inspect"].append(deepcopy(config))
        if calls.get("tracking_failure"):
            raise ValueError("Incomplete source coverage")
        return {"observations": "fixture"}, scene_fixture()

    def fit(evidence, initial, out, progress=None):
        assert evidence == {"observations": "fixture"}
        calls["fit"] += 1
        write_json(out / "reconstruction.json", {"replay_gate": {"passes": not calls.get("gate_failure", False)}})
        return initial

    class Probe:
        def __init__(self, _scene):
            self.steps = 0
        def reset(self, **_kwargs):
            pass
        def snapshot(self):
            return {"steps": self.steps}
        def step(self, action):
            calls["steps"] += 1
            self.steps += 1
            return None, 0., False, False, {}
        def close(self):
            pass

    def train(_scene, out, **options):
        calls["train"].append(options)
        write_json(out / "policy.json", {"synthetic_training_stub": True})
        return {"synthetic_training_stub": True}

    monkeypatch.setattr(scene_vision, "inspect_video", inspect)
    monkeypatch.setattr(generic_vision, "inspect_generic", inspect)
    monkeypatch.setattr(scene_vision, "fit_scene", fit)
    monkeypatch.setattr(scene, "SceneEnv", Probe)
    monkeypatch.setattr(scene_learning, "train_scene", train)
    monkeypatch.setattr(lab_worker, "_implementation_files", lambda: {})
    monkeypatch.setattr(lab_worker, "_runtime_packages", lambda: {})
    return calls


@pytest.mark.parametrize("profile", ["planar", "appearance"])
@pytest.mark.parametrize("manual", [False, True])
def test_worker_keeps_source_and_action_gates_but_skips_only_requested_learning(source, tmp_path, pipeline_stubs, profile, manual):
    config = tmp_path / "tracking.json"
    write_json(config, {"source_sha256": digest(source), "fixture": "passed intact to inspector"})
    root = prepare_lab(source, tmp_path / "build", profile=profile, config=config, no_train=manual)
    assert lab_worker.execute(root) == 0
    run = root / "run"
    status = json.loads((root / "status.json").read_text())
    report = json.loads((run / "report.json").read_text())
    settings = json.loads((run / "run-config.json").read_text())
    assert status["status"] == "ready"
    assert pipeline_stubs["inspect"] == [json.loads(config.read_text())]
    assert pipeline_stubs["fit"] == 1 and pipeline_stubs["steps"] == 180
    assert (run / "source.mp4").read_bytes() == source.read_bytes()
    assert settings == json.loads((root / "lab-config.json").read_text())
    assert len(pipeline_stubs["train"]) == (0 if manual else 1)
    assert (run / "policy.json").exists() is not manual
    assert not (run / "training.json").exists() and not (run / "rl-evaluation.json").exists()
    if manual:
        assert report["learning"] == {
            "status": "not_requested", "learned_policy_available": False, "evaluation": None,
            "scope": "Manual controls only; no policy training or policy evaluation was requested"}
        assert "train" not in [event["stage"] for event in status["events"]]
    else:
        options = pipeline_stubs["train"][0]
        assert options["episodes"] == 1500 and options["eval_episodes"] == 100 and options["seed"] == 1729
        assert report["learning"] == {"synthetic_training_stub": True}


@pytest.mark.parametrize("failure", ["tracking_failure", "gate_failure"])
def test_manual_build_still_refuses_source_or_reconstruction_failure(source, tmp_path, pipeline_stubs, failure):
    root = prepare_lab(source, tmp_path / "build", no_train=True)
    pipeline_stubs[failure] = True
    assert lab_worker.execute(root) == 1
    assert pipeline_stubs["train"] == [] and pipeline_stubs["steps"] == 0
    assert (root / "run" / "source.mp4").read_bytes() == source.read_bytes()
    assert (root / "failure.json").is_file()
    assert not (root / "run" / "manifest.json").exists()
    assert json.loads((root / "status.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("changes", [{"profile": "rolling"}, {"training_enabled": 0}, {"episodes": 1}])
def test_worker_rechecks_no_train_mode_before_source_work(source, tmp_path, pipeline_stubs, changes):
    root = prepare_lab(source, tmp_path / "build", no_train=True)
    settings = json.loads((root / "lab-config.json").read_text())
    write_json(root / "lab-config.json", settings | changes)
    assert lab_worker.execute(root) == 1
    assert pipeline_stubs["inspect"] == [] and pipeline_stubs["train"] == [] and pipeline_stubs["steps"] == 0
    assert not (root / "run").exists()
