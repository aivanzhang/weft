"""Mode-derived body inputs retain original modes, complete decisions and rejection gates."""

import json
import sys
import time

import mujoco
import pytest

from test_exercise_body_source import VIDEO, forbid_runtime, model_metadata, seal, write_pose_files
from weft import cli, exercise_body_source as body_source, exercise_vision, pose_mode_source as recovery, shared_world_lab
from weft.io import digest, write_json


def files(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def original_pair(root, *, collapse=True):
    video = root / "source.webm"
    video.write_bytes(VIDEO)
    primary, alternative = root / "video", root / "image"
    primary.mkdir(); alternative.mkdir()
    for directory, mode, capacity in ((primary, "video", 1), (alternative, "image", 2)):
        (directory / "source.webm").write_bytes(VIDEO)
        model = model_metadata(mode)
        if capacity == 1:
            model.update(maximum_pose_hypotheses=1, person_count_assumption="single_person_unverified")
        evidence = write_pose_files(directory, mode=mode, model=model)
        if collapse and capacity == 1:
            # Move the inferred knee toward its hip in both domains; no floor score defines the fixture.
            row = evidence["frames"][12]
            for name in ("landmarks", "world_landmarks"):
                hip, knee = row[name][24], row[name][26]
                for axis in ("x", "y", "z"):
                    knee[axis] = hip[axis] + .1 * (knee[axis] - hip[axis])
            row.update(exercise_vision.summarize_landmarks(row["landmarks"], row["world_landmarks"]))
            write_json(directory / "pose-landmarks.json", {"frames": evidence["frames"], "model": model, "scope": evidence["scope"]})
            exercise_vision.resolve_pose_identity(evidence, directory, "raw")
            exercise_vision.finalize_evidence(evidence, directory)
        (directory / "unrelated.txt").write_text("Uncopied inputs still require integrity checks.\n")
        seal(directory)
    return primary, alternative, video


def prepare(root, **options):
    primary, alternative, video = original_pair(root, **options)
    out = root / "recovered"
    recovery.prepare_pose_mode_recovery(primary, alternative, out, source_video=video)
    return out, primary, alternative, video


def reseal_proof(payload):
    marker = json.loads(payload["pose-mode-recovery.json"])
    marker["artifacts"] = {name: body_source._sha(payload[name]) for name in marker["artifacts"]}
    payload["pose-mode-recovery.json"] = body_source._bytes(marker)


def test_verified_mode_package_reaches_body_with_exact_provenance_and_no_inference(tmp_path, monkeypatch):
    forbid_runtime(monkeypatch)
    forward, calls = mujoco.mj_forward, []
    monkeypatch.setattr(mujoco, "mj_forward", lambda *args: calls.append(True) or forward(*args))
    out, primary, alternative, video = prepare(tmp_path)
    assert not calls
    payload = files(out)
    chosen = recovery.validate_pose_mode_files(payload, source_sha256=digest(video))
    assert chosen["pose_mode_recovery"]["accepted_frames"] == [24]
    assert payload["pose-landmarks.json"] == (primary / "pose-landmarks.json").read_bytes()
    assert payload["mode-recovery/alternative-evidence.json"] == (alternative / "evidence.json").read_bytes()
    row = chosen["frames"][12]
    assert row["mode_refinement"]["model"]["running_mode"] == "IMAGE"
    assert row["mode_refinement_raw"] == json.loads(payload["pose-landmarks.json"])["frames"][12]
    decisions = json.loads(payload["mode-recovery/decisions.json"])
    assert len(decisions["rows"]) == 32 and sum(len(row["segments"]) for row in decisions["rows"]) == 256
    assert all(row["accepted"] is False for row in decisions["rows"] if row["frame"] != 24)
    body = tmp_path / "body"
    body_source.prepare_exercise_body_source_from_evidence(out, body)
    assert len(calls) == 1
    for name in recovery.PROOF_FILES | body_source.POSE_FILES:
        assert (body / name).read_bytes() == payload[name]
    human, observations = body_source.validate_body_source_files(files(body))
    audit = next(row for row in human["reference_evidence"] if row["evidence_id"].endswith(":frame:24"))
    assert audit["mode_refinement"] == chosen["frames"][12]["mode_refinement"]
    assert len(observations["observations"]) == 63
    assert observations["observations"][30]["visibility"] == "not_detected"
    assert observations["observations"][31]["visibility"] == "not_sampled"
    assert json.loads((body / "source-clock.json").read_bytes())["samples"][7]["decoder_timestamp_s"] is None
    assert not any((body / name).exists() for name in ("task.json", "scene.json", "policy.json"))


def test_unchanged_mode_proposal_does_not_invent_replacements(tmp_path, monkeypatch):
    forbid_runtime(monkeypatch)
    out, primary, _, _ = prepare(tmp_path, collapse=False)
    chosen, original = [json.loads((folder / "evidence.json").read_bytes()) for folder in (out, primary)]
    assert chosen["frames"] == original["frames"]
    assert chosen["pose_mode_recovery"]["accepted_frames"] == []


def test_mode_proof_survives_shared_world_seed_without_dynamics_or_extra_forward(tmp_path, monkeypatch):
    forbid_runtime(monkeypatch)
    out, _, _, _ = prepare(tmp_path)
    body = body_source.prepare_exercise_body_source_from_evidence(out, tmp_path / "body")
    before = files(body)
    monkeypatch.setattr(mujoco, "mj_forward", lambda *_a: pytest.fail("Seed must not re-evaluate geometry"))
    monkeypatch.setattr(mujoco, "MjData", lambda *_a: pytest.fail("Seed must not allocate simulation state"))
    monkeypatch.setattr(shared_world_lab, "adapt_to_shared_world", lambda *_a, **_k: pytest.fail("No disposable body lab"))
    from weft import shared_world
    mass, mass_calls = shared_world.human_subtree_mass, []
    monkeypatch.setattr(shared_world, "human_subtree_mass", lambda *a, **kw: mass_calls.append(True) or mass(*a, **kw))
    seed = shared_world_lab.prepare_shared_world_seed(body, tmp_path / "seed", deadline=time.monotonic() + 10)
    marker = json.loads((seed / "construction-seed.json").read_bytes())
    assert marker["physics_models"] == marker["physics_steps"] == marker["inference_calls"] == 0
    assert marker["simulation_ready"] is False
    assert len(mass_calls) == 1 and "static mass" in marker["physics_models_scope"]
    assert set(marker["pose_recovery_validator_sha256"]) == set(recovery.IMPLEMENTATION_FILES)
    assert (seed / "human.json").read_bytes() == before["human.json"]
    for name in recovery.PROOF_FILES:
        assert (seed / "inputs/human-source" / name).read_bytes() == before[name]
    assert files(body) == before


@pytest.mark.parametrize("kind", ["missing_proof", "decision", "selected", "original", "clock", "recipe", "model", "source", "raw", "failed_original", "implementation"])
def test_resealed_changes_cannot_bypass_mode_derivation(tmp_path, monkeypatch, kind):
    forbid_runtime(monkeypatch)
    out, _, _, video = prepare(tmp_path)
    payload = files(out)
    if kind == "missing_proof":
        payload.pop("mode-recovery/alternative-manifest.json")
    elif kind == "raw":
        payload["pose-landmarks.json"] = payload["mode-recovery/alternative-pose-landmarks.json"]
    elif kind == "implementation":
        payload["mode-recovery/implementation-source/pose_mode_recovery.py"] += b"\n"
    else:
        filename = {"decision": "mode-recovery/decisions.json", "selected": "evidence.json",
            "original": "evidence.json", "clock": "mode-recovery/alternative-evidence.json",
            "recipe": "pose-mode-recovery.json", "model": "model.json", "source": "source.json",
            "failed_original": "mode-recovery/alternative-manifest.json"}[kind]
        value = json.loads(payload[filename])
        if kind == "decision":
            value["rows"][12]["accepted"] = False
        elif kind == "selected":
            value["frames"][12]["world_landmarks"][26]["x"] += .1
        elif kind == "original":
            value["frames"][12]["mode_refinement_raw"]["time_s"] += .1
        elif kind == "clock":
            value["frames"][12]["decoder_timestamp_s"] += .1
        elif kind == "recipe":
            value["contract"]["collapse_ratio"] = .9
        elif kind == "model":
            value["running_mode"] = "IMAGE"
        elif kind == "source":
            value["sha256"] = "a" * 64
        else:
            value["status"] = "failed"
        payload[filename] = body_source._bytes(value)
    if kind != "missing_proof":
        reseal_proof(payload)
    with pytest.raises(ValueError):
        recovery.validate_pose_mode_files(payload, source_sha256=digest(video))


@pytest.mark.parametrize("kind", ["uncopied_corruption", "wrong_video", "failed_input", "link", "output_collision", "invalid_bound"])
def test_input_failures_cannot_become_successful_sealed_evidence(tmp_path, monkeypatch, kind):
    forbid_runtime(monkeypatch)
    primary, alternative, video = original_pair(tmp_path)
    out = tmp_path / "result"
    kwargs = {}
    if kind == "uncopied_corruption":
        (alternative / "unrelated.txt").write_text("changed")
    elif kind == "wrong_video":
        video.write_bytes(b"another source")
    elif kind == "failed_input":
        write_json(alternative / "failure.json", {"status": "failed"})
    elif kind == "link":
        link = tmp_path / "alias"
        link.symlink_to(alternative, target_is_directory=True)
        alternative = link
    elif kind == "output_collision":
        out = primary
    else:
        kwargs["max_seconds"] = 0
    with pytest.raises(ValueError):
        recovery.prepare_pose_mode_recovery(primary, alternative, out, source_video=video, **kwargs)
    if out != primary and out.exists():
        assert not (out / "manifest.json").exists()
        assert json.loads((out / "failure.json").read_bytes())["profile"] == recovery.PROFILE


def test_missing_proof_cannot_enter_body_even_with_resealed_manifest(tmp_path, monkeypatch):
    forbid_runtime(monkeypatch)
    out, _, _, _ = prepare(tmp_path)
    manifest = json.loads((out / "manifest.json").read_bytes())
    manifest["profile"] = "original_pose_evidence"
    manifest["files"].pop("pose-mode-recovery.json")
    write_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match="explicit complete recovery proof"):
        body_source.prepare_exercise_body_source_from_evidence(out, tmp_path / "body")


def test_mode_recovery_cli_is_explicit_and_preserves_source_only_result(tmp_path, monkeypatch, capsys):
    primary, alternative, video = original_pair(tmp_path)
    forbid_runtime(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["weft", "exercise-pose-recover-modes", "--primary-run", str(primary),
        "--alternative-run", str(alternative), "--video", str(video), "--out", str(tmp_path / "cli"), "--max-seconds", "10"])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["profile"] == recovery.PROFILE and result["simulation_ready"] is False
    assert result["new_inference"] is False
