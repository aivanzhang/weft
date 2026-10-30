"""Mobile-source preparation preserves evidence without running a throwaway world."""

import hashlib
import json
import time

import pytest

from test_equipment_proposal import inputs, seal
from weft.io import write_json
from weft.observations import adapt_run
from weft.pipeline import verify_artifacts
from weft import shared_world_lab as lab
from weft.shared_world_evidence import verify_shared_world_sources


def mobile_source_packages(root, *, as_lab=False):
    """Return mobile input, equipment export, config path, original human and config."""
    base, config, _, documents, _ = inputs()
    video = b"Unfamiliar analytical fixture source; not a decodable or physical video.\n"
    source_sha = hashlib.sha256(video).hexdigest()
    old_sha = base["human_spec"]["source_sha256"]
    human = json.loads(json.dumps(base["human_spec"]).replace(old_sha, source_sha))
    config = json.loads(json.dumps(config).replace(old_sha, source_sha))
    equipment_document = json.loads(next(iter(documents.values())).replace(old_sha.encode(), source_sha.encode()))
    missing = {0, 1, 31}
    keep = [index for index, row in enumerate(human["reference_evidence"])
            if int(row["evidence_id"].rsplit(":", 1)[1]) not in missing]
    for name in ("reference", "root_reference", "reference_evidence"):
        human[name] = [human[name][index] for index in keep]
    frames = []
    for row in equipment_document["observations"]:
        if row["object_id"] != "person" or row["frame"] in missing:
            continue
        points = [{"x": 0., "y": 0., "z": 0., "visibility": 0., "presence": 0.} for _ in range(33)]
        for point in row["measurements"]["landmarks"]["value"]:
            points[point["index"]] = {"x": point["position"][0] / 640, "y": point["position"][1] / 480,
                                       "z": 0., "visibility": 1., "presence": 1.}
        frames.append({"id": f"{source_sha}:frame:{row['frame']}", "frame": row["frame"],
                       "time_s": row["time_s"], "status": "tracked", "landmarks": points})
    evidence = {"profile": "exercise-pose-v1", "source_sha256": source_sha,
                "width": 640, "height": 480, "fps": 30., "frame_count": 65, "decoded_frames": 63,
                "frames": frames}
    mobile = root / "unfamiliar-mobile"
    run = mobile / "run" if as_lab else mobile
    run.mkdir(parents=True)
    (run / "human.json").write_text(json.dumps(human, indent=1) + "\n\n")
    write_json(run / "scene.json", {"schema_version": 1, "profile": "exercise-v3", "source_sha256": source_sha})
    write_json(run / "evidence.json", evidence)
    write_json(run / "source.json", {"sha256": source_sha, "scope": "Analytical fixture bytes, not physical evidence"})
    (run / "source.webm").write_bytes(video)
    (run / "unrelated-proof.txt").write_text("This sealed input must also be rechecked.\n")
    seal(run)
    if as_lab:
        write_json(mobile / "status.json", {"status": "ready"})
    equipment = root / "unfamiliar-equipment"
    equipment.mkdir()
    write_json(equipment / "scene-observations.json", equipment_document)
    write_json(equipment / "report.json", {"status": "evidence_only", "physics_ready": False})
    seal(equipment)
    config_path = root / "unfamiliar-config.json"
    write_json(config_path, config)
    return mobile, equipment, config_path, human, config


def forbid_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Body preparation must not create or inspect a simulator")

    for name in ("SharedWorldEnv", "audit_shared_world_feasibility", "_probe_shared_interventions", "_reference_frames_artifact"):
        monkeypatch.setattr(lab, name, forbidden)


@pytest.mark.parametrize("as_lab", [False, True])
def test_seed_preserves_exact_source_slots_clocks_and_bytes_without_physics(tmp_path, monkeypatch, as_lab):
    mobile, _, _, human, _ = mobile_source_packages(tmp_path, as_lab=as_lab)
    run = mobile / "run" if as_lab else mobile
    before = {path: path.read_bytes() for path in mobile.rglob("*") if path.is_file()}
    observations = adapt_run(mobile)
    forbid_physics(monkeypatch)
    out = tmp_path / "seed"
    assert lab.prepare_shared_world_seed(mobile, out, scale=1.0) == out
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert (out / "human.json").read_bytes() == before[run / "human.json"]
    assert (out / "source.webm").read_bytes() == before[run / "source.webm"]
    for name in ("human.json", "scene.json", "evidence.json"):
        assert (out / "inputs/human-source" / name).read_bytes() == before[run / name]
    scene = json.loads((out / "scene.json").read_bytes())
    assert scene["human_spec"] == human
    binding = scene["source_binding"]
    document_path = next(iter(binding["evidence_documents"].values()))
    assert (out / document_path).read_bytes() == lab._seed_bytes(observations)
    assert binding["total_source_frames"] == 63
    assert binding["entity_bindings"][0]["observed_slots"] == 60
    assert binding["entity_bindings"][0]["missing_slots"] == 3
    assert all(row["observed_slots"] == 0 for row in binding["entity_bindings"][1:])
    assert observations["source"]["metadata_frames"] == 65
    assert [row["frame"] for row in observations["observations"] if row["visibility"] == "not_sampled"] == [0, 1, 31]
    assert scene["coordinate_frame"]["clock_binding"]["source_time_offset_s"] == 2 / 30
    assert verify_shared_world_sources(scene, out)["source_evidence_verified"]
    report = json.loads((out / "construction-seed.json").read_bytes())
    assert report["status"] == "construction_only" and report["simulation_ready"] is False
    assert report["scaffold_entity_ids"] == ["ball", "table"]
    assert all(report[name] == 0 for name in ("physics_steps", "physics_models", "reference_renders",
                                             "video_decodes", "inference_calls", "api_calls"))
    assert report["source_clock"]["basis"] == "frame_index_divided_by_fps"
    verified = verify_artifacts(out)
    files = [path for path in out.rglob("*") if path.is_file()]
    assert verified["verified_files"] == len(files) - 1 < lab.SEED_MAX_FILES
    assert sum(path.stat().st_size for path in files) < lab.SEED_MAX_BYTES
    assert not any((out / name).exists() for name in ("lab", "status.json", "interventions.json", "human-reference.json"))


def test_legacy_adapter_uses_identical_scene_and_observation_factory(tmp_path, monkeypatch):
    mobile, _, _, _, _ = mobile_source_packages(tmp_path)
    calls = []
    monkeypatch.setattr(lab, "audit_shared_world_feasibility", lambda scene: calls.append("audit") or {"status": "fixture", "summary": {}})
    monkeypatch.setattr(lab, "_probe_shared_interventions", lambda scene: calls.append("probe") or {})
    monkeypatch.setattr(lab, "_reference_frames_artifact", lambda scene: calls.append("references") or {})
    legacy, seed = tmp_path / "legacy", tmp_path / "seed"
    lab.adapt_to_shared_world(mobile, legacy, scale=1.25)
    lab.prepare_shared_world_seed(mobile, seed, scale=1.25)
    assert calls == ["audit", "probe", "references"]
    for name in ("scene.json", "task.json", "construction-evidence.json"):
        assert (seed / name).read_bytes() == (legacy / name).read_bytes()
    scene = json.loads((seed / "scene.json").read_bytes())
    for name in scene["source_binding"]["evidence_documents"].values():
        assert (seed / name).read_bytes() == (legacy / name).read_bytes()


@pytest.mark.parametrize("document,change", [
    ("human.json", {"profile": "exercise-v2"}),
    ("human.json", {"dt": 1 / 60}),
    ("human.json", {"camera_reference": {}}),
    ("scene.json", {"profile": "exercise-v1"}),
    ("scene.json", {"source_sha256": "b" * 64}),
    ("evidence.json", {"source_sha256": "b" * 64}),
    ("evidence.json", {"profile": "appearance-tracks-v1"}),
])
def test_unsupported_or_conflicting_inputs_never_publish_a_seed(tmp_path, document, change):
    mobile, _, _, _, _ = mobile_source_packages(tmp_path)
    path = mobile / document
    value = json.loads(path.read_bytes()); value.update(change); write_json(path, value); seal(mobile)
    out = tmp_path / "seed"
    with pytest.raises(ValueError):
        lab.prepare_shared_world_seed(mobile, out)
    assert not out.exists()


@pytest.mark.parametrize("target", ["manifest.json", "status.json", "report.json", "reconstruction.json", "lifecycle"])
def test_failed_inputs_are_not_promoted(tmp_path, target):
    mobile, _, _, _, _ = mobile_source_packages(tmp_path, as_lab=target == "lifecycle")
    run = mobile / "run" if target == "lifecycle" else mobile
    if target == "manifest.json":
        value = json.loads((run / target).read_bytes()); value["status"] = "failed"; write_json(run / target, value)
    else:
        write_json(mobile / "status.json" if target == "lifecycle" else run / target, {"status": "failed"})
        if target != "lifecycle":
            seal(run)
    with pytest.raises(ValueError, match="failed or rejected"):
        lab.prepare_shared_world_seed(mobile, tmp_path / "seed")
    assert not (tmp_path / "seed").exists()


@pytest.mark.parametrize("target", ["human.json", "manifest.json", "unrelated-proof.txt", "lifecycle"])
def test_input_change_after_preparation_cannot_receive_a_terminal_seal(tmp_path, monkeypatch, target):
    mobile, _, _, _, _ = mobile_source_packages(tmp_path, as_lab=target == "lifecycle")
    run = mobile / "run" if target == "lifecycle" else mobile
    original = lab._adapted_shared_world

    def change_after_snapshot(*args, **kwargs):
        result = original(*args, **kwargs)
        path = mobile / "status.json" if target == "lifecycle" else run / target
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(lab, "_adapted_shared_world", change_after_snapshot)
    with pytest.raises(ValueError, match="changed|pinned|bounded regular"):
        lab.prepare_shared_world_seed(mobile, tmp_path / "seed")
    assert not (tmp_path / "seed/manifest.json").exists()
    assert not (tmp_path / "seed/lab").exists()


@pytest.mark.parametrize("scale", [True, 0.49, 2.01, float("nan"), "1"])
def test_invalid_scale_rejects_before_io(tmp_path, scale):
    with pytest.raises(ValueError, match="scale prior"):
        lab.prepare_shared_world_seed("missing", tmp_path / "seed", scale=scale)
    assert not (tmp_path / "seed").exists()


@pytest.mark.parametrize("deadline", [True, float("inf"), "tomorrow", 0.])
def test_invalid_or_expired_deadline_rejects_before_io(tmp_path, deadline):
    with pytest.raises((ValueError, TimeoutError), match="deadline|wall-time"):
        lab.prepare_shared_world_seed("missing", tmp_path / "seed", deadline=deadline)
    assert not (tmp_path / "seed").exists()


def test_input_count_byte_bounds_and_missing_required_document(tmp_path, monkeypatch):
    mobile, _, _, _, _ = mobile_source_packages(tmp_path)
    out = tmp_path / "seed"
    with monkeypatch.context() as patch:
        patch.setattr(lab, "SEED_MAX_FILES", 3)
        with pytest.raises(ValueError, match="512-file"):
            lab.prepare_shared_world_seed(mobile, out)
    with monkeypatch.context() as patch:
        patch.setattr(lab, "SEED_MAX_BYTES", 100)
        with pytest.raises(ValueError, match="128-MiB"):
            lab.prepare_shared_world_seed(mobile, out)
    manifest = json.loads((mobile / "manifest.json").read_bytes())
    manifest["files"].pop("evidence.json"); write_json(mobile / "manifest.json", manifest)
    with pytest.raises(ValueError, match="sealed human, scene and evidence"):
        lab.prepare_shared_world_seed(mobile, out)
    assert not out.exists()


def test_expired_after_factory_preserves_no_ready_output(tmp_path, monkeypatch):
    mobile, _, _, _, _ = mobile_source_packages(tmp_path)
    original = lab._adapted_shared_world

    def expire(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(lab.time, "monotonic", lambda: kwargs["deadline"] + 1)
        return result

    monkeypatch.setattr(lab, "_adapted_shared_world", expire)
    with pytest.raises(TimeoutError, match="wall-time"):
        lab.prepare_shared_world_seed(mobile, tmp_path / "seed", deadline=time.monotonic() + 10)
    assert not (tmp_path / "seed").exists()


@pytest.mark.parametrize("kind", ["root", "parent", "member"])
def test_symlink_inputs_cannot_be_laundered_into_a_sealed_seed(tmp_path, kind):
    mobile, _, _, _, _ = mobile_source_packages(tmp_path)
    if kind == "root":
        source = tmp_path / "alias"; source.symlink_to(mobile, target_is_directory=True)
    elif kind == "parent":
        alias = tmp_path / "alias"; alias.symlink_to(tmp_path, target_is_directory=True)
        source = alias / mobile.name
    else:
        (mobile / "linked.txt").symlink_to(mobile / "unrelated-proof.txt")
        seal(mobile); source = mobile
    with pytest.raises(ValueError, match="symlinks|unsafe"):
        lab.prepare_shared_world_seed(source, tmp_path / "seed")
    assert not (tmp_path / "seed").exists()


def test_scaffolding_cannot_be_packaged_as_a_playable_lab(tmp_path, monkeypatch):
    from weft.shared_exercise_lab import prepare_shared_exercise_lab

    mobile, _, _, _, _ = mobile_source_packages(tmp_path)
    seed = lab.prepare_shared_world_seed(mobile, tmp_path / "seed")
    forbid_physics(monkeypatch)
    with pytest.raises(ValueError, match="Construction-only"):
        lab.prepare_shared_world_lab(seed, tmp_path / "lab")
    assert not (tmp_path / "lab").exists()
    with pytest.raises(ValueError, match="[Cc]onstruction"):
        prepare_shared_exercise_lab(seed, tmp_path / "exercise-lab")
    assert not (tmp_path / "exercise-lab").exists()


def test_source_mutation_and_output_overlap_are_rejected(tmp_path):
    mobile, _, _, _, _ = mobile_source_packages(tmp_path)
    with pytest.raises(ValueError, match="outside its immutable"):
        lab.prepare_shared_world_seed(mobile, mobile / "seed")
    (mobile / "evidence.json").write_bytes((mobile / "evidence.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="pinned SHA256"):
        lab.prepare_shared_world_seed(mobile, tmp_path / "seed")
    assert not (tmp_path / "seed").exists()


def test_unusable_pose_stays_distinct_from_an_unsampled_slot(tmp_path):
    mobile, _, _, human, config = mobile_source_packages(tmp_path)
    value = json.loads((mobile / "evidence.json").read_bytes())
    value["frames"].append({"id": f"{human['source_sha256']}:frame:31", "frame": 31,
                            "time_s": 31 / 30, "status": "no_pose", "landmarks": []})
    value["frames"].sort(key=lambda row: row["frame"])
    write_json(mobile / "evidence.json", value); seal(mobile)
    out = lab.prepare_shared_world_seed(mobile, tmp_path / "seed")
    scene = json.loads((out / "scene.json").read_bytes())
    observation_path = next(iter(scene["source_binding"]["evidence_documents"].values()))
    observations = json.loads((out / observation_path).read_bytes())
    slots = {row["frame"]: row for row in observations["observations"]}
    assert [slots[index]["visibility"] for index in (0, 1, 31)] == ["not_sampled", "not_sampled", "not_detected"]
    assert slots[31]["measurements"] == {} and len(slots[31]["evidence_ids"]) == 1
    assert scene["human_spec"] == human
    assert [(row["object_id"], row["source_frame"]) for row in config["equipment"]] == [("foam-pad", 12), ("training-weight", 18)]


def test_clock_outside_shared_world_range_is_not_silently_clamped(tmp_path):
    mobile, _, _, _, _ = mobile_source_packages(tmp_path)
    value = json.loads((mobile / "evidence.json").read_bytes())
    value["fps"] = 4.
    for row in value["frames"]:
        row["time_s"] = row["frame"] / value["fps"]
    write_json(mobile / "evidence.json", value); seal(mobile)
    with pytest.raises(ValueError, match="clamped or retimed"):
        lab.prepare_shared_world_seed(mobile, tmp_path / "seed")
    assert not (tmp_path / "seed").exists()


def test_partial_interval_is_not_filled_to_make_a_complete_source(tmp_path):
    mobile, _, _, _, _ = mobile_source_packages(tmp_path)
    value = json.loads((mobile / "evidence.json").read_bytes())
    value["decode_audit"] = {"scope": "selected_interval", "selected_interval": [2, 62],
                              "selected_frames_complete": True, "source_complete": False, "prefix_advanced_frames": 2}
    value["decoded_frames"] = 61
    write_json(mobile / "evidence.json", value); seal(mobile)
    with pytest.raises(ValueError, match="complete original frame interval"):
        lab.prepare_shared_world_seed(mobile, tmp_path / "seed")
    assert not (tmp_path / "seed").exists()


def test_output_reservation_rejects_before_any_write(tmp_path, monkeypatch):
    mobile, _, _, _, _ = mobile_source_packages(tmp_path)
    input_bytes = sum(path.stat().st_size for path in mobile.iterdir() if path.is_file())
    monkeypatch.setattr(lab, "SEED_MAX_BYTES", input_bytes + 1)
    with pytest.raises(ValueError, match="reserve terminal metadata"):
        lab.prepare_shared_world_seed(mobile, tmp_path / "seed")
    assert not (tmp_path / "seed").exists()
