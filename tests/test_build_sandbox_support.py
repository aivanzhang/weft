"""Static-support launch identity checks use sealed metadata, never native execution."""

import builtins
import hashlib
import json
from pathlib import Path
import socket

import pytest

from test_build_sandbox import (
    body_consumption,
    body_source,
    launcher,
    preserved_build,
    project,
    raw,
    reseal_preserved,
    saved_build,
    saved_report,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def no_external_execution(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"cv2", "mediapipe", "mujoco"}, "Unexpected native import"
        return original_import(name, *args, **kwargs)

    def blocked(*_args, **_kwargs):
        pytest.fail("Unexpected worker, subprocess, or network execution")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(launcher, "_run", blocked)
    monkeypatch.setattr(launcher.subprocess, "Popen", blocked)
    monkeypatch.setattr(launcher.subprocess, "call", blocked)
    monkeypatch.setattr(socket, "socket", blocked)


@pytest.fixture
def support_inputs(project):
    body = body_source(project / "body")
    video = project / "video.webm"
    video.write_bytes((body / "source.webm").read_bytes())
    prior = project / "support priors.json"
    prior.write_bytes(raw({
        "schema_version": 1, "profile": "shared-static-support-priors-v1",
        "source_sha256": sha(video.read_bytes()), "scale": 1.25, "root_assistance": True,
        "objects": [{"id": "bench", "role": "static", "shape": "box", "position": [1., 1., .3],
            "quaternion": [1., 0., 0., 0.], "velocity": [0., 0., 0.], "angular_velocity": [0., 0., 0.],
            "mass": 10., "friction": [.8, .01, .001], "color": "#707070", "size": [.6, .4, .2],
            "provenance": {"mode": "authored_addition", "calibration_status": "estimated_prior",
                "physical_validation": "unverified", "source_contact_verified": False,
                "assumptions": ["Analytical geometry; no observed contact identity."]}}],
        "support_requirements": [{"object_id": "bench", "human_segment": "left_forearm",
            "direction_world": [0., 0., 1.], "minimum_load_body_weight_fraction": .05,
            "start_step": 0, "end_step": 6}],
    }) + b"  \n")
    return body, video, prior


def support_members(body, video, prior, *, saved):
    members = body_consumption(body)
    assumptions = json.loads(prior.read_bytes())
    settings = json.loads(members["settings.json"])
    settings.update(scene_scope="human_static_supports", static_supports=str(prior),
        support_priors_sha256=sha(prior.read_bytes()), body_scale=assumptions["scale"],
        root_assistance=assumptions["root_assistance"], full_reference=True,
        evidence_run=None, config=None, source_mappings=[])
    if not saved:
        settings.update(human_run=None, video=str(video), inference_calls=1)
        members.update({"body_source/" + name: (body / name).read_bytes()
                        for name in (*launcher.BODY_METADATA, "human.json", "source.webm")})
    members["settings.json"] = raw(settings)
    reconstruction = json.loads(members["candidate/reconstruction.json"])
    reconstruction["profile"] = "shared-world-v4"
    members["candidate/reconstruction.json"] = raw(reconstruction)
    config = raw({**assumptions, "profile": "shared-static-support-construction-v1",
                  "body_source_manifest_sha256": sha((body / "manifest.json").read_bytes())})
    members.update({"support-priors.json": prior.read_bytes(), "support-config.json": config,
                    "candidate/inputs/support-config.json": config})
    return members


def support_worker(monkeypatch, inputs, *, ready=True, mutation=None, original_mutation=False, partial=False):
    body, video, prior = inputs
    calls = []

    def run(command, root, name, seconds):
        calls.append((command, name, seconds))
        if name == "build":
            saved = "--human-run" in command
            members = support_members(body, video, prior, saved=saved)
            if partial:
                members = {key: members[key] for key in ("settings.json", "support-priors.json")}
            if mutation:
                mutation(members)
            pins = json.loads((root / "workflow-plan.json").read_bytes())["input_sha256"]
            preflight = None
            if not saved:
                model = command[command.index("--pose-model") + 1]
                preflight = {"source_sha256": pins[str(video)], "pose_model": {"sha256": pins[model]},
                             "input_pins": [{"path": p, "sha256": value} for p, value in pins.items()]}
            saved_build(root / "build", ready=ready, trained="--train" in command,
                        preflight=preflight, members=members)
            if original_mutation:
                prior.write_bytes(prior.read_bytes() + b" ")
        else:
            saved_report(root / "report", Path(command[command.index("--run") + 1]))
        return {"returncode": (0 if ready else 1) if name == "build" else 0,
                "error": None, "interrupted": False, "seconds": 0,
                "log_bytes": {"stdout": 0, "stderr": 0}}

    monkeypatch.setattr(launcher, "_run", run)
    return calls


@pytest.mark.parametrize("flags", [
    ["--person-only"], ["--evidence-run", "unused"], ["--config", "unused"],
    ["--source-mapping", "unused"], ["--root-assistance"],
    ["--train", "--policy-family", "feedback-residual"], ["--profile", "spatial"],
])
def test_static_support_conflicts_reject_before_output(project, flags):
    out = project / "attempt"
    with pytest.raises(SystemExit):
        launcher.main(["--video", "unused", "--static-supports", "unused", *flags, "--out", str(out)])
    assert not out.exists()


def test_static_support_inspect_override_rejects_before_output(project):
    out = project / "inspection"
    with pytest.raises(SystemExit):
        launcher.main(["--inspect-build", "unused", "--static-supports", "unused", "--out", str(out)])
    assert not out.exists()


@pytest.mark.parametrize("kind", ["oversize", "wrong_source"])
def test_static_support_prior_preflight_rejects_before_output(project, support_inputs, kind):
    _, video, prior = support_inputs
    if kind == "oversize":
        prior.write_bytes(prior.read_bytes().ljust((128 << 10) + 1, b" "))
    else:
        doc = json.loads(prior.read_bytes())
        doc["source_sha256"] = "f" * 64
        prior.write_bytes(raw(doc))
    out = project / "attempt"
    with pytest.raises(ValueError, match="bounded regular file|exact original video"):
        launcher.main(["--video", str(video), "--static-supports", str(prior), "--out", str(out)])
    assert not out.exists()


@pytest.mark.parametrize("saved,trained", [(False, False), (False, True), (True, False), (True, True)])
def test_static_support_raw_and_saved_inputs_keep_prior_owned_settings(project, monkeypatch, support_inputs, saved, trained):
    body, video, prior = support_inputs
    if not saved and not trained:
        prior.write_bytes(prior.read_bytes().ljust(128 << 10, b" "))
    original = prior.read_bytes()
    calls = support_worker(monkeypatch, support_inputs)
    out = project / "attempt"
    source = ["--human-run", str(body)] if saved else ["--video", str(video)]
    flags = ["--train", "--policy-family", "phase-residual"] if trained else []
    assert launcher.main([*source, "--static-supports", str(prior), *flags, "--out", str(out)]) == 0
    assert [item[1] for item in calls] == ["build", "report"]
    command = calls[0][0]
    assert "shared-exercise-build" in command and "--full-reference" in command
    assert command[command.index("--static-supports") + 1] == str(prior)
    assert not {"--person-only", "--root-assistance", "--body-scale", "--evidence-run", "--config"} & set(command)
    assert ("--train" in command) is trained
    if saved:
        assert not {"--video", "--pose-model", "--pose-recovery"} & set(command)
    plan = json.loads((out / "workflow-plan.json").read_bytes())
    expected = {str(body / name) for name in launcher.BODY_METADATA} if saved else {
        str(video), str(project / ".local/models/pose_landmarker_lite.task")}
    assert set(plan["input_sha256"]) == expected | {str(prior)}
    assert plan["input_sha256"][str(prior)] == sha(original)
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["status"] == "ready" and receipt["report"]
    assert receipt["input_binding"] == ("sealed_body_source_match_plan" if saved else "sealed_source_and_model_match_plan")
    assert receipt["lab"] == str(out / "build" / ("learning_workflow/lab" if trained else "lab"))
    assert (out / "build/support-priors.json").read_bytes() == prior.read_bytes() == original


@pytest.mark.parametrize("kind", ["original_prior", "copied_prior", "settings_source", "settings_prior_path",
    "settings_scope", "settings_scale", "settings_root", "settings_full_reference", "resolved_body_pin",
    "candidate_config", "candidate_body_manifest"])
def test_static_support_substitution_keeps_report_but_blocks_lab(project, monkeypatch, support_inputs, kind):
    _, video, prior = support_inputs

    def mutate(members):
        if kind == "copied_prior":
            members["support-priors.json"] += b" "
        elif kind.startswith("settings_"):
            doc = json.loads(members["settings.json"])
            key, value = {
                "settings_source": ("video", str(project / "other.webm")),
                "settings_prior_path": ("static_supports", str(project / "other.json")),
                "settings_scope": ("scene_scope", "human_only"),
                "settings_scale": ("body_scale", 1.0),
                "settings_root": ("root_assistance", False),
                "settings_full_reference": ("full_reference", False),
            }[kind]
            doc[key] = value
            members["settings.json"] = raw(doc)
        elif kind == "resolved_body_pin":
            doc = json.loads(members["support-config.json"])
            doc["body_source_manifest_sha256"] = "f" * 64
            members["support-config.json"] = raw(doc)
        elif kind in ("candidate_config", "candidate_body_manifest"):
            name = "support-config.json" if kind == "candidate_config" else "body-source-manifest.json"
            members["candidate/inputs/" + name] += b" "

    calls = support_worker(monkeypatch, support_inputs, mutation=mutate, original_mutation=kind == "original_prior")
    out = project / "attempt"
    assert launcher.main(["--video", str(video), "--static-supports", str(prior), "--serve", "--out", str(out)]) == 1
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["report"] and receipt["lab"] is None and receipt["status"] == "stopped"
    assert receipt["error"]["type"] == "ValueError"
    assert [item[1] for item in calls] == ["build", "report"]


@pytest.mark.parametrize("missing", ["support-config.json", "candidate/inputs/support-config.json",
                                      "candidate/inputs/body-source-manifest.json"])
def test_static_support_ready_requires_resolved_candidate_proof(project, monkeypatch, support_inputs, missing):
    _, video, prior = support_inputs
    support_worker(monkeypatch, support_inputs, mutation=lambda members: members.pop(missing))
    out = project / "attempt"
    assert launcher.main(["--video", str(video), "--static-supports", str(prior), "--console", "--out", str(out)]) == 1
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["input_binding"] == "unavailable_before_sealed_support_assumptions"
    assert receipt["report"] and receipt["lab"] is None
    assert "no sealed input identities" in receipt["error"]["message"]


def test_static_support_preconstruction_failure_still_reports(project, monkeypatch, support_inputs):
    body, _, prior = support_inputs
    calls = support_worker(monkeypatch, support_inputs, ready=False, partial=True)
    out = project / "attempt"
    assert launcher.main(["--human-run", str(body), "--static-supports", str(prior), "--console", "--out", str(out)]) == 1
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["input_binding"] == "unavailable_before_sealed_support_assumptions"
    assert receipt["report"] and receipt["lab"] is None and receipt["build_status"] == "failed"
    assert [item[1] for item in calls] == ["build", "report"]


def preserved_support(path):
    build, lab = preserved_build(path, trained=True, lifecycle="closed")
    parent = json.loads((build / "manifest.json").read_bytes())
    members = {name.replace("learning_workflow/selected/", "learning_workflow/training/"): (build / name).read_bytes()
               for name in parent["files"]}
    (build / "learning_workflow/selected").rename(build / "learning_workflow/training")
    canonical = lambda doc: sha(json.dumps(doc, sort_keys=True, allow_nan=False).encode())
    scene = json.loads(members["candidate/scene.json"])
    scene["profile"] = "shared-world-v4"
    task = {"schema_version": 1, "profile": "shared-support-task-v1", "scene_sha256": canonical(scene)}
    policy = raw({"profile": "shared-support-policy-v1", "scene_sha256": canonical(scene), "task_sha256": canonical(task)})
    for name in members:
        if name.endswith("/scene.json"):
            members[name] = raw(scene)
        elif name.endswith("/exercise-task.json"):
            members[name] = raw(task)
        elif name.endswith("/policy.json"):
            members[name] = policy
    training_prefix = "learning_workflow/training/"
    training_files = {name[len(training_prefix):]: sha(data) for name, data in members.items()
                      if name.startswith(training_prefix) and name != training_prefix + "manifest.json"}
    training_manifest = raw({"schema_version": 1, "status": "trained", "files": training_files})
    prefix = "learning_workflow/lab/run/"
    for name in (training_prefix + "manifest.json", prefix + "source-run/manifest.json", prefix + "learning-run/manifest.json"):
        members[name] = training_manifest
    members["candidate/manifest.json"] = raw({"schema_version": 1, "status": "ready", "files": {
        name[len("candidate/"):]: sha(data) for name, data in members.items()
        if name.startswith("candidate/") and name != "candidate/manifest.json"}})
    identity = {"scene_sha256": canonical(scene), "task_sha256": canonical(task),
                "policy_sha256": sha(policy), "learning_manifest_sha256": sha(training_manifest)}
    contract = json.loads(members[prefix + "exercise-lab.json"])
    contract.update(**identity, source_manifest_sha256=sha(training_manifest))
    members[prefix + "exercise-lab.json"] = raw(contract)
    members["settings.json"] = raw({"scene_scope": "human_static_supports"})
    members["learning_workflow/result.json"] = raw({"profile": "shared-support-learning-workflow-v1",
        "training": {"policy_sha256": sha(policy), "selected_feedback_kind": "authored_unchanged"},
        "selection": {"selection_status": "not_applicable", "performed": False, "evaluation_used": False,
            "directory": str(build / "learning_workflow/training"), "policy_sha256": sha(policy),
            "manifest_sha256": sha(training_manifest), "reason": "Optimizer-frozen policy; no feasibility selection."}})
    result = json.loads(members["result.json"])
    result.update(scene_sha256=identity["scene_sha256"], task_sha256=identity["task_sha256"])
    members["result.json"] = raw(result)
    for name, data in members.items():
        (build / name).parent.mkdir(parents=True, exist_ok=True)
        (build / name).write_bytes(data)
    config = json.loads((lab / "lab-config.json").read_bytes())
    config.update(**identity, source_run=str(build / "learning_workflow/training"))
    (lab / "lab-config.json").write_bytes(raw(config))
    parent["files"] = {name: sha(data) for name, data in members.items()}
    (build / "manifest.json").write_bytes(raw(parent))
    reseal_preserved(build)
    return build, lab


@pytest.mark.parametrize("mutation", [None, "profile", "selection_status", "performed", "evaluation_used", "policy_sha256"])
def test_static_support_preserved_training_policy_requires_explicit_selection_contract(project, monkeypatch, capsys, mutation):
    build, lab = preserved_support(project / "existing")
    if mutation:
        path = build / "learning_workflow/result.json"
        doc = json.loads(path.read_bytes())
        if mutation == "profile":
            doc["profile"] = "shared-exercise-learning-workflow-v1"
        else:
            doc["selection"][mutation] = {"selection_status": "selected", "performed": True,
                                          "evaluation_used": True, "policy_sha256": "f" * 64}[mutation]
        path.write_bytes(raw(doc))
        reseal_preserved(build)
    original = {str(path.relative_to(build)): path.read_bytes() for path in build.rglob("*") if path.is_file()}
    calls, reads = [], []
    original_read = launcher._read

    def metadata_only(path, limit=launcher.MAX_METADATA_BYTES):
        if path.is_relative_to(build):
            assert path.suffix == ".json", "Saved lab recovery must not decode media or read runtime code"
        reads.append(path)
        return original_read(path, limit)

    def report_only(command, root, name, seconds):
        assert name == "report"
        calls.append(name)
        saved_report(root / "report", build)
        return {"returncode": 0, "error": None, "interrupted": False}

    monkeypatch.setattr(launcher, "_read", metadata_only)
    monkeypatch.setattr(launcher, "_run", report_only)
    out = project / "inspection"
    assert launcher.main(["--inspect-build", str(build), "--console", "--out", str(out)]) == 1
    receipt = json.loads((out / "workflow-result.json").read_bytes())
    assert receipt["report"] and receipt["lab"] is None and calls == ["report"]
    if mutation:
        assert "preserved_lab" not in receipt and "optimizer/selection provenance" in receipt["error"]["message"]
    else:
        info = receipt["preserved_lab"]
        assert info["directory"] == str(lab) and info["status"] == "identity_metadata_verified"
        assert info["lifecycle_status"] == "closed" and info["runtime_admission"] == "not_checked_by_launcher"
        assert 0 < info["metadata_bytes_read"] <= launcher.MAX_HANDOFF_BYTES
        assert build / "learning_workflow/training/manifest.json" in reads
        assert not any("selected" in path.parts for path in reads)
        assert "lab-console --resume" in capsys.readouterr().out
    assert original == {str(path.relative_to(build)): path.read_bytes() for path in build.rglob("*") if path.is_file()}


@pytest.mark.parametrize("kind", ["policy", "learning_manifest"])
def test_static_support_preserved_lab_rejects_substituted_training_identity(project, kind):
    build, lab = preserved_support(project / "existing")
    member, pin = (("policy.json", "policy_sha256") if kind == "policy" else
                   ("learning-run/manifest.json", "learning_manifest_sha256"))
    path = lab / "run" / member
    doc = json.loads(path.read_bytes())
    doc["fixture_note"] = "Different lab-local identity; optimizer source remains unchanged."
    contents = raw(doc)
    path.write_bytes(contents)
    # A consistent local reseal must not replace the frozen optimizer identity.
    for metadata in (lab / "run/exercise-lab.json", lab / "lab-config.json"):
        doc = json.loads(metadata.read_bytes())
        doc[pin] = sha(contents)
        metadata.write_bytes(raw(doc))
    reseal_preserved(build)
    terminal, seal = launcher._terminal(build)
    with pytest.raises(ValueError, match="optimizer|training|policy|learning"):
        launcher._preserved_lab(build, terminal, seal)
