"""Saved crop proofs preserve source observations and fail closed on altered derivations."""

from copy import deepcopy
import json
from pathlib import Path
import time

import mujoco
import pytest

from test_exercise_body_source import VIDEO, model_metadata, write_pose_files
from test_exercise_vision import inference_result
from weft import exercise_body_source as body_source, exercise_vision, human, shared_world_lab
from weft import pose_source_recovery as recovery
from weft.errors import UnsupportedVideo
from weft.io import digest, write_json
from weft.pose_crop import map_crop_pose
from weft.pose_crop_selection import apply_selective_crop, assess_selective_crop, plan_selective_crop


def seal(root, **extra):
    write_json(root / "manifest.json", {"schema_version": 1, **extra, "files": {
        path.relative_to(root).as_posix(): digest(path) for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"}})


def files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def finish(evidence, directory, original_raw):
    (directory / "pose-landmarks.json").write_bytes(original_raw)
    exercise_vision.resolve_pose_identity(evidence, directory, "raw")
    try:
        exercise_vision.finalize_evidence(evidence, directory)
    except UnsupportedVideo:
        pass


def proposal_case(root, *, outcome="recover"):
    """One analytical source; no media decode, detector or physical observation."""
    root.mkdir()
    video = root.parent / (root.name + ".webm")
    video.write_bytes(VIDEO)
    whole, selected = root / "whole_image", root / "selected"
    whole.mkdir(); selected.mkdir()
    (whole / "source.webm").write_bytes(VIDEO)
    model = {**model_metadata("image"), **exercise_vision.MODEL_VARIANTS["full"],
             "name": "MediaPipe Pose Landmarker Full", "variant": "full"}
    baseline = write_pose_files(whole, mode="image", model=model, dropped=(30,) if outcome != "unchanged" else ())
    if outcome == "competing":
        row = baseline["frames"][15]
        row.update(exercise_vision.summarize_result(inference_result(.15, count=2)))
        raw = recovery._bytes({"frames": baseline["frames"], "model": model, "scope": baseline["scope"]})
        finish(baseline, whole, raw)
    raw = (whole / "pose-landmarks.json").read_bytes()
    plans = {row["frame"]: plan_selective_crop(baseline, row) for row in baseline["frames"]}
    eligible = [frame for frame, plan in plans.items() if plan["eligible"]]
    proposals, decisions, journal = {}, {}, []
    indexed = {row["frame"]: row for row in baseline["frames"]}
    for frame in eligible:
        plan = plans[frame]["crop"]
        raw_crop = exercise_vision.summarize_result(inference_result(frame * .005))
        left, top, right, bottom = plan["box_xyxy"]
        for point in raw_crop["landmarks"]:
            point["x"] = (point["x"] * baseline["width"] - left) / (right - left)
            point["y"] = (point["y"] * baseline["height"] - top) / (bottom - top)
            point["z"] *= baseline["width"] / (right - left)
        if outcome == "reject_crop":
            raw_crop = exercise_vision.summarize_result(inference_result(count=2))
        proposed = map_crop_pose(raw_crop, plan, baseline, indexed[frame], .001)
        proposed["crop_proposal"] = plan
        decision = assess_selective_crop(baseline, indexed[frame], proposed, plans[frame])
        proposals[frame], decisions[frame] = proposed, decision
        journal.append({"frame": frame, "raw_crop_result": raw_crop, "proposed": proposed, "decision": decision})
    chosen = apply_selective_crop(baseline, proposals, plans, decisions)
    finish(chosen, selected, raw)
    write_json(root / "crop-plan.json", {"plans": list(plans.values()), "eligible_frames": eligible,
                                         "bound_exceeded": False})
    (root / "crop-observations.jsonl").write_bytes(b"".join(
        json.dumps(row, sort_keys=True, allow_nan=False).encode() + b"\n" for row in journal))
    write_json(root / "decisions.json", list(decisions.values()))
    write_json(root / "report.json", {"status": "complete_video_selection_frozen", "scope": "Analytical fixture"})
    seal(root)
    return root, video, baseline, chosen


def forbid_source_work(monkeypatch, *, allow_forward=False):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Saved recovery must not decode/infer, simulate or train")
    monkeypatch.setattr(exercise_vision.cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(exercise_vision, "_detector", forbidden)
    monkeypatch.setattr(exercise_vision, "inspect_exercise", forbidden)
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(human, "HumanEnv", forbidden)
    if not allow_forward:
        monkeypatch.setattr(mujoco, "mj_forward", forbidden)


@pytest.mark.parametrize("outcome,accepted", [("recover", [30]), ("unchanged", []), ("reject_crop", []), ("competing", [])])
def test_sealed_source_recovery_preserves_originals_unknown_clocks_and_abstentions(tmp_path, monkeypatch, outcome, accepted):
    source, video, baseline, selected = proposal_case(tmp_path / "proposals", outcome=outcome)
    original = files(source); video_raw = video.read_bytes()
    forbid_source_work(monkeypatch)
    out = tmp_path / "recovered"
    assert recovery.prepare_pose_source_recovery(source, out, source_video=video) == out
    result = files(out); marker = json.loads(result["pose-recovery.json"])
    assert files(source) == original and video.read_bytes() == video_raw
    assert marker["profile"] == recovery.PROFILE and marker["status"] == "source_evidence_only"
    assert marker["simulation_ready"] is False and marker["result"]["accepted_frames"] == accepted
    assert marker["result"]["source_slots"] == 63 and marker["result"]["sampled_slots"] == 32
    assert marker["result"]["pose_quality_passes"] is (outcome != "competing")
    assert marker["work"] == {"video_decodes": 0, "inference_calls": 0, "physics_steps": 0, "body_models": 0, "api_calls": 0}
    assert result["evidence.json"] == original["selected/evidence.json"]
    assert result["pose-landmarks.json"] == original["whole_image/pose-landmarks.json"]
    assert result["recovery/candidates.jsonl"] == original["crop-observations.jsonl"]
    assert result["source.webm"] == video_raw
    verified = recovery.validate_pose_recovery_files(result, source_sha256=digest(video))
    assert verified == selected
    for before, after in zip(baseline["frames"], verified["frames"], strict=True):
        assert all(after[key] == before[key] for key in ("id", "frame", "time_s", "decoder_timestamp_s"))
        if before["frame"] not in accepted:
            assert before == after
    assert verified["frames"][7]["decoder_timestamp_s"] is None
    assert not any(name in result for name in ("human.json", "scene.json", "task.json"))


def mutate_case(source, kind):
    def mutate(name, callback):
        value = json.loads((source / name).read_bytes())
        callback(value)
        write_json(source / name, value)

    if kind == "plan":
        mutate("crop-plan.json", lambda value: value["plans"][15]["crop"]["box_xyxy"].__setitem__(0, 1))
    elif kind == "decision":
        mutate("decisions.json", lambda value: value[0].update(accepted=False))
    elif kind in {"candidate_clock", "raw_mapping", "candidate_decision", "missing_candidate", "extra_candidate", "rejected_candidate"}:
        rows = [json.loads(line) for line in (source / "crop-observations.jsonl").read_bytes().splitlines()]
        if kind == "candidate_clock": rows[0]["proposed"]["decoder_timestamp_s"] = .123
        elif kind == "raw_mapping": rows[0]["raw_crop_result"]["landmarks"][11]["x"] += .1
        elif kind == "candidate_decision": rows[0]["decision"]["accepted"] = False
        elif kind == "missing_candidate": rows = []
        elif kind == "extra_candidate": rows.append(deepcopy(rows[0]))
        else: rows[0]["proposed"]["pose_count"] = 1
        (source / "crop-observations.jsonl").write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    elif kind == "selected":
        mutate("selected/evidence.json", lambda value: value["frames"][0].update(inference_seconds=2.))
    elif kind == "original_native_clock":
        mutate("whole_image/pose-landmarks.json", lambda value: value["frames"][0].update(decoder_timestamp_s=.123))
    elif kind == "model":
        mutate("whole_image/evidence.json", lambda value: value["model"].update(variant="lite"))
    elif kind == "identity":
        mutate("whole_image/evidence.json", lambda value: value["pose_identity"].update(mode="temporal"))
    elif kind == "chain":
        mutate("whole_image/evidence.json", lambda value: value.update(pose_reinference={"method": "prior"}))
    elif kind == "raw_identity":
        mutate("selected/identity-audit.json", lambda value: value.update(raw_artifact_sha256="0" * 64))
    elif kind == "incomplete_slots":
        mutate("whole_image/evidence.json", lambda value: value["frames"].pop())
    elif kind == "failed":
        mutate("report.json", lambda value: value.update(status="failed"))
    seal(source)


@pytest.mark.parametrize("kind", ["plan", "decision", "candidate_clock", "raw_mapping", "candidate_decision",
    "missing_candidate", "extra_candidate", "selected", "original_native_clock", "model", "identity", "chain",
    "raw_identity", "incomplete_slots", "failed"])
def test_changed_or_incomplete_derivation_is_retained_as_failure_not_evidence(tmp_path, monkeypatch, kind):
    source, video, _, _ = proposal_case(tmp_path / "proposals")
    mutate_case(source, kind)
    forbid_source_work(monkeypatch)
    out = tmp_path / "recovered"
    with pytest.raises(ValueError):
        recovery.prepare_pose_source_recovery(source, out, source_video=video)
    assert json.loads((out / "failure.json").read_bytes())["status"] == "failed"
    assert not (out / "manifest.json").exists()


def test_rejected_candidates_are_recomputed_instead_of_ignored(tmp_path, monkeypatch):
    source, video, _, _ = proposal_case(tmp_path / "proposals", outcome="reject_crop")
    mutate_case(source, "rejected_candidate")
    forbid_source_work(monkeypatch)
    with pytest.raises(ValueError, match="raw model"):
        recovery.prepare_pose_source_recovery(source, tmp_path / "recovered", source_video=video)


@pytest.mark.parametrize("problem", ["source", "unsealed", "failure_marker", "input_bytes", "output_bytes", "file_count",
                                     "journal", "row", "deadline", "symlink"])
def test_source_integrity_and_resource_bounds(tmp_path, monkeypatch, problem):
    source, video, _, _ = proposal_case(tmp_path / "proposals")
    forbid_source_work(monkeypatch)
    options = {}
    if problem == "source": video.write_bytes(b"Different original video")
    elif problem == "unsealed": (source / "manifest.json").unlink()
    elif problem == "failure_marker": write_json(source / "failure.json", {"status": "failed"})
    elif problem == "input_bytes": monkeypatch.setattr(recovery, "MAX_BYTES", 300_000)
    elif problem == "output_bytes": monkeypatch.setattr(recovery, "RESERVE_BYTES", recovery.MAX_BYTES - 10)
    elif problem == "file_count": monkeypatch.setattr(recovery, "MAX_FILES", 2)
    elif problem == "journal": monkeypatch.setattr(recovery, "MAX_JOURNAL_BYTES", 2)
    elif problem == "row": monkeypatch.setattr(recovery, "MAX_LINE_BYTES", 2)
    elif problem == "deadline": options["max_seconds"] = 0
    else:
        link = tmp_path / "linked.webm"; link.symlink_to(video); video = link
    with pytest.raises((ValueError, OSError)):
        recovery.prepare_pose_source_recovery(source, tmp_path / "recovered", source_video=video, **options)


def test_portable_recovery_proof_is_required_and_covers_all_recorded_bytes(tmp_path, monkeypatch):
    source, video, _, _ = proposal_case(tmp_path / "proposals")
    forbid_source_work(monkeypatch)
    out = recovery.prepare_pose_source_recovery(source, tmp_path / "recovered", source_video=video)
    original = files(out)
    for name in ("recovery/baseline-evidence.json", "recovery/candidates.jsonl", "recovery/decisions.json",
                 "recovery/implementation-source/pose_crop_selection.py", "source.webm"):
        changed = dict(original); changed[name] += b" "
        with pytest.raises(ValueError):
            recovery.validate_pose_recovery_files(changed, source_sha256=digest(video))
    changed = dict(original); changed.pop("recovery/plans.json")
    with pytest.raises(ValueError):
        recovery.validate_pose_recovery_files(changed, source_sha256=digest(video))
    with pytest.raises(TimeoutError):
        recovery.validate_pose_recovery_files(original, source_sha256=digest(video), deadline=time.monotonic() - 1)


@pytest.mark.parametrize("change", ["failed_status", "failure_member"])
def test_resealed_portable_proof_cannot_relabel_failed_original_manifest(tmp_path, monkeypatch, change):
    source, video, _, _ = proposal_case(tmp_path / "proposals")
    forbid_source_work(monkeypatch)
    out = recovery.prepare_pose_source_recovery(source, tmp_path / "recovered", source_video=video)
    changed = files(out)
    manifest = json.loads(changed["recovery/input-manifest.json"])
    if change == "failed_status":
        manifest["status"] = "failed"
    else:
        manifest["files"]["failure.json"] = "a" * 64
    changed["recovery/input-manifest.json"] = recovery._bytes(manifest)
    marker = json.loads(changed["pose-recovery.json"])
    marker["input_manifest_sha256"] = recovery._sha(changed["recovery/input-manifest.json"])
    marker["artifacts"]["recovery/input-manifest.json"] = marker["input_manifest_sha256"]
    changed["pose-recovery.json"] = recovery._bytes(marker)
    with pytest.raises(ValueError, match="failed or oversized"):
        recovery.validate_pose_recovery_files(changed, source_sha256=digest(video))


def test_verified_recovery_builds_one_body_with_exact_proof_and_unknown_slots(tmp_path, monkeypatch):
    source, video, _, selected = proposal_case(tmp_path / "proposals")
    recovered = recovery.prepare_pose_source_recovery(source, tmp_path / "recovered", source_video=video)
    original = files(recovered)
    forward, calls = mujoco.mj_forward, []
    forbid_source_work(monkeypatch, allow_forward=True)
    monkeypatch.setattr(mujoco, "mj_forward", lambda *args: calls.append(True) or forward(*args))
    out = tmp_path / "body"
    assert body_source.prepare_exercise_body_source_from_evidence(recovered, out) == out
    assert len(calls) == 1 and files(recovered) == original
    produced = files(out)
    assert all(produced[name] == original[name] for name in recovery.PROOF_FILES | recovery.POSE_FILES | {"source.webm"})
    marker = json.loads(produced["body-source.json"])
    assert recovery.PROOF_FILES <= marker["artifacts"].keys()
    assert set(recovery.IMPLEMENTATION_FILES) <= json.loads(produced["implementation.json"])["files"].keys()
    assert marker["status"] == "construction_only" and marker["simulation_ready"] is False
    assert marker["work"]["inference_calls"] == marker["work"]["physics_steps"] == 0
    human_spec, observations = body_source.validate_body_source_files(produced)
    assert len(human_spec["reference"]) == len(selected["reference"])
    assert len(observations["observations"]) == 63
    assert json.loads(produced["source-clock.json"])["samples"][7]["decoder_timestamp_s"] is None
    assert not (out / "scene.json").exists() and not (out / "task.json").exists()


def test_recovery_body_reaches_construction_seed_with_exact_proof_and_no_extra_physics(tmp_path, monkeypatch):
    source, video, _, _ = proposal_case(tmp_path / "proposals")
    recovered = recovery.prepare_pose_source_recovery(source, tmp_path / "recovered", source_video=video)
    body = body_source.prepare_exercise_body_source_from_evidence(recovered, tmp_path / "body")
    original = files(body)
    forbid_source_work(monkeypatch)
    monkeypatch.setattr(shared_world_lab, "adapt_to_shared_world", lambda *_a, **_k: pytest.fail("No legacy body lab"))
    seed = shared_world_lab.prepare_shared_world_seed(body, tmp_path / "seed", deadline=time.monotonic() + 10)
    receipt = json.loads((seed / "construction-seed.json").read_bytes())
    assert receipt["physics_models"] == receipt["physics_steps"] == receipt["inference_calls"] == 0
    assert receipt["simulation_ready"] is False
    assert set(receipt["pose_recovery_validator_sha256"]) == set(recovery.IMPLEMENTATION_FILES)
    for name in recovery.PROOF_FILES:
        assert (seed / "inputs/human-source" / name).read_bytes() == original[name]
    assert (seed / "human.json").read_bytes() == original["human.json"]
    assert files(body) == original


def test_competing_source_is_preserved_but_body_admission_remains_rejected(tmp_path, monkeypatch):
    source, video, _, _ = proposal_case(tmp_path / "proposals", outcome="competing")
    recovered = recovery.prepare_pose_source_recovery(source, tmp_path / "recovered", source_video=video)
    forbid_source_work(monkeypatch)
    monkeypatch.setattr(body_source, "build_human", lambda *_args, **_kwargs: pytest.fail("Quality rejection precedes body construction"))
    with pytest.raises(ValueError, match="Rejected/unsupported"):
        body_source.prepare_exercise_body_source_from_evidence(recovered, tmp_path / "body")
    assert (recovered / "manifest.json").exists()
    assert not (tmp_path / "body/human.json").exists()


def test_crop_rows_without_explicit_proof_cannot_bypass_body_custody(tmp_path, monkeypatch):
    source, video, _, _ = proposal_case(tmp_path / "proposals")
    recovered = recovery.prepare_pose_source_recovery(source, tmp_path / "recovered", source_video=video)
    raw = {name: (recovered / name).read_bytes() for name in body_source.POSE_FILES}
    forbid_source_work(monkeypatch)
    with pytest.raises(ValueError, match="explicit complete recovery proof"):
        body_source._validated_pose(raw, {"pose_mode": "image", "pose_identity": "raw"}, digest(video))
