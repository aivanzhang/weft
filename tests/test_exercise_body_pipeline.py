"""Construction-only pose bodies reach one coupled lab without an intermediate body simulation."""

import json

import mujoco
import pytest

from test_equipment_proposal import seal
from test_exercise_body_source import pose_run
from test_shared_world_seed import mobile_source_packages
from weft import exercise_body_source, exercise_scene_pipeline, exercise_vision, shared_world_lab
from weft.io import digest, write_json
from weft.lab import Lab
from weft.pipeline import verify_artifacts


def _files(root):
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _packages(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("This saved-evidence integration must not decode, infer, train or run hidden interventions")

    monkeypatch.setattr(exercise_vision, "inspect_exercise", forbidden)
    monkeypatch.setattr(exercise_vision.cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(shared_world_lab, "adapt_to_shared_world", forbidden)
    monkeypatch.setattr(shared_world_lab, "_probe_shared_interventions", forbidden)
    from weft import human_learning, shared_exercise_learning, shared_world_learning
    monkeypatch.setattr(human_learning, "train_human", forbidden)
    monkeypatch.setattr(shared_exercise_learning, "train_shared_exercise_policy", forbidden)
    monkeypatch.setattr(shared_world_learning, "train_shared_world_policy", forbidden)

    legacy = tmp_path / "legacy-fixture"
    legacy.mkdir()
    _, equipment, config_path, _, config = mobile_source_packages(legacy)
    prior = _files(legacy)
    poses = pose_run(tmp_path / "poses")
    source_sha = digest(poses / "source.webm")
    old_sha = config["source_sha256"]
    assert source_sha != old_sha

    # Keep the unfamiliar equipment observations, but never relabel the incompatible legacy body rows.
    document = json.loads((equipment / "scene-observations.json").read_text().replace(old_sha, source_sha))
    names = {"foam-pad", "training-weight"}
    document["objects"] = [row for row in document["objects"] if row["id"] in names]
    for key in ("observations", "evidence"):
        document[key] = [row for row in document[key] if row["object_id"] in names]
    document["provenance"] = {"scope": "Analytical equipment fixture rebound to the legal sampled pose source",
        "original_fixture_document_sha256": digest(equipment / "scene-observations.json"),
        "derivation": "Only source IDs changed and unrelated legacy-person rows removed; equipment frames and boxes retained"}
    derived = tmp_path / "equipment"
    derived.mkdir()
    write_json(derived / "scene-observations.json", document)
    write_json(derived / "report.json", {"status": "evidence_only", "physics_ready": False})
    seal(derived)
    config = json.loads(config_path.read_text())
    config["source_sha256"] = source_sha
    selected = tmp_path / "equipment-config.json"
    write_json(selected, config)

    body = tmp_path / "body-source"
    exercise_body_source.prepare_exercise_body_source_from_evidence(poses, body)
    assert not (body / "scene.json").exists() and not (body / "task.json").exists()
    assert all(path.read_bytes() == raw for path, raw in prior.items())
    retained = {**prior, **_files(poses), **_files(body), **_files(derived), selected: selected.read_bytes()}
    return body, derived, selected, retained


def test_construction_body_preserves_source_gaps_through_one_world_task_and_lab(tmp_path, monkeypatch):
    native_step = mujoco.mj_step
    monkeypatch.setattr(mujoco, "mj_step", lambda *_args, **_kwargs: pytest.fail("No pre-lab physics actions"))
    body, equipment, config, retained = _packages(tmp_path, monkeypatch)
    human_raw = (body / "human.json").read_bytes()
    human = json.loads(human_raw)
    observations_raw = (body / "observations.json").read_bytes()
    observations = json.loads(observations_raw)
    clocks_raw = (body / "source-clock.json").read_bytes()
    clocks = json.loads(clocks_raw)
    marker = json.loads((body / "body-source.json").read_bytes())
    assert marker["status"] == "construction_only" and marker["simulation_ready"] is False
    assert marker["work"]["inference_calls"] == marker["work"]["video_decodes"] == 0
    assert marker["work"]["physics_steps"] == marker["work"]["training_episodes"] == 0

    lab_calls = []
    prepare_lab = exercise_scene_pipeline.prepare_shared_exercise_lab

    def one_lab(source, destination, **options):
        lab_calls.append((source, destination))
        return prepare_lab(source, destination, **options)

    monkeypatch.setattr(exercise_scene_pipeline, "prepare_shared_exercise_lab", one_lab)
    out = tmp_path / "assembled"
    result = exercise_scene_pipeline.prepare_exercise_scene(None, equipment, config, out,
        human_run=body, full_reference=True, root_assistance=True)
    assert result["status"] == result["lab_status"] == "ready"
    assert lab_calls == [(out / "candidate", out / "lab")]
    assert sorted(path.name for path in out.iterdir() if path.is_dir()) == ["body_seed", "candidate", "lab", "proposal"]
    assert [(row["stage"], row["status"]) for row in result["stages"]] == [
        ("body_seed", "running"), ("body_seed", "ready"),
        ("equipment_priors", "running"), ("equipment_priors", "proposal_ready"),
        ("shared_world", "running"), ("shared_world", "ready"),
        ("interactive_lab", "running"), ("interactive_lab", "ready")]

    seed = out / "body_seed"
    assert (seed / "human.json").read_bytes() == human_raw
    assert (seed / "inputs/human-source/source-clock.json").read_bytes() == clocks_raw
    assert (seed / "inputs/human-source/observations.json").read_bytes() == observations_raw
    seed_receipt = json.loads((seed / "construction-seed.json").read_bytes())
    assert seed_receipt["input_profile"] == exercise_body_source.PROFILE
    assert seed_receipt["physics_models"] == seed_receipt["physics_steps"] == 0

    scene = json.loads((out / "candidate/scene.json").read_bytes())
    assert scene["profile"] == "shared-world-v2" and scene["human_spec"] == human
    assert (out / "lab/run/scene.json").read_bytes() == (out / "candidate/scene.json").read_bytes()
    assert {row["id"] for row in scene["objects"]} == {"foam-pad", "training-weight"}
    assert all(obj["role"] == "passive" and obj["velocity"] == obj["angular_velocity"] == [0., 0., 0.]
               and obj["provenance"]["reference_object_force"] == [0., 0., 0.] for obj in scene["objects"])
    assert scene["coordinate_frame"]["clock_binding"]["source_time_offset_s"] == 4 / 30
    assert scene["task"]["start_offset_s"] == 0.
    assert scene["task"]["horizon_steps"] == 58
    binding = scene["source_binding"]
    assert binding["total_source_frames"] == 63
    person = next(row for row in binding["entity_bindings"] if row["world_entity_id"] == "human")
    assert person["observed_slots"] == 29 and person["missing_slots"] == 34
    linked = json.loads((out / "candidate" / binding["evidence_documents"][person["observation_document_sha256"]]).read_bytes())
    assert linked["observations"] == observations["observations"]
    rows = linked["observations"]
    assert [row["frame"] for row in rows] == list(range(63))
    assert [row["frame"] for row in rows if row["visibility"] == "not_detected"] == [0, 2, 30]
    assert [row["frame"] for row in rows if row["visibility"] == "not_sampled"] == list(range(1, 63, 2))
    assert clocks["samples"][7]["decoder_timestamp_s"] is None
    assert len(human["reference"]) == len(human["reference_evidence"]) == 29

    task = json.loads((out / "lab/run/exercise-task.json").read_bytes())
    assert task["interval"] == {"start_offset_s": 0., "horizon_steps": 58}
    assert task["control"]["object_forces"] == "forbidden"
    assert task["contact_requirements"] == []
    assert not (out / "lab/run/policy.json").exists()
    run_before = _files(out / "lab/run")
    lab = Lab(out / "lab")
    try:
        session = lab.load_session()
        before = session.snapshot()
        with pytest.raises(ValueError, match="22"):
            lab.control("/api/step", {"action": [0.] * 28})
        assert session.snapshot() == before
        steps = []

        def counted_step(*args, **kwargs):
            steps.append(1)
            return native_step(*args, **kwargs)

        monkeypatch.setattr(mujoco, "mj_step", counted_step)
        after = lab.control("/api/step", {"controller": "reference"})["snapshot"]
        assert len(steps) == scene["substeps"]
        assert after["engine_state"]["steps"] == 1
        assert not any(after["engine_state"]["normalized_action"][22:])
        assert all(obj["virtual_force"] == [0., 0., 0.] for obj in after["engine_state"]["objects"])
    finally:
        lab.close()
    assert all(path.read_bytes() == raw for path, raw in {**retained, **run_before}.items())
    assert verify_artifacts(out)["verified_files"] > 0
    assert verify_artifacts(body)["verified_files"] > 0


def test_constructor_rejection_retains_body_and_proposal_without_publishing_lab(tmp_path, monkeypatch):
    monkeypatch.setattr(mujoco, "mj_step", lambda *_args, **_kwargs: pytest.fail("Rejected construction cannot advance physics"))
    body, equipment, config, retained = _packages(tmp_path, monkeypatch)
    constructor = exercise_scene_pipeline.construct_source_equipment_run
    calls = []

    def rejected_target(*args, **options):
        calls.append(options["target_object_id"])
        # Exercise the real constructor's rejection after a valid proposal, not a substituted validator.
        return constructor(*args, **{**options, "target_object_id": "missing-from-source"})

    monkeypatch.setattr(exercise_scene_pipeline, "construct_source_equipment_run", rejected_target)
    monkeypatch.setattr(exercise_scene_pipeline, "prepare_shared_exercise_lab",
                        lambda *_args, **_kwargs: pytest.fail("Rejected world reached lab preparation"))
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="target|Target"):
        exercise_scene_pipeline.prepare_exercise_scene(None, equipment, config, out,
            human_run=body, full_reference=True, root_assistance=True)
    assert calls == ["foam-pad"]
    report = json.loads((out / "result.json").read_bytes())
    assert report["status"] == "failed" and report["lab_status"] == "not_ready"
    assert report["stages"][-1]["stage"] == "shared_world"
    assert report["stages"][-1]["status"] == "running"
    assert not (out / "lab").exists()
    assert (out / "body_seed/construction-seed.json").is_file()
    assert (out / "proposal/geometry-proposal.json").is_file()
    assert verify_artifacts(out)["verified_files"] > 0
    assert all(path.read_bytes() == raw for path, raw in retained.items())
