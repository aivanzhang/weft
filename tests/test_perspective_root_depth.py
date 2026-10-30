"""Known camera-ray motions test depth fitting without using contacts or changed-action outcomes."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from weft import perspective_root_depth as depth
from weft.arm_reference_alignment import perspective_points
from weft.body_refinement import JOINT_LANDMARKS, _kinematics
from weft.human import HumanEnv, validate_human
from test_arm_reference_alignment import inputs


def fixture(*, camera_distance=2., withheld=False, secret_shift=0., weak_frame=None, correction=True):
    spec, evidence, kwargs = inputs(withheld=withheld, secret_shift=secret_shift)
    camera = kwargs["camera_hypothesis"]
    camera["distance_model_units"] = camera_distance
    camera["focal_px"] = camera_distance / camera["original_scale_at_anchor_model_units_per_pixel"]
    document = json.loads(evidence)
    observations = {row["frame"]: row for row in document["sources"][0]["observations"]}
    env = HumanEnv(spec)
    changes = [0., .03, .06, .08, .07, .04] if correction else [0.] * 6
    try:
        data = mujoco.MjData(env.model)
        for index, (reference, meta) in enumerate(zip(spec["reference"], spec["reference_evidence"])):
            frame = int(meta["evidence_id"].rsplit(":", 1)[1])
            row = observations[frame]
            if row["admission"] != "included":
                continue
            env.mobility.reference(data, index)
            relative, _ = _kinematics(env, data, [reference["joints"][name] for name in env.joint_names])
            pelvis = data.site_xpos[env._site_ids["pelvis"]].copy()
            direction = np.array([pelvis[0] / (camera_distance + pelvis[1]), 1.,
                                  (pelvis[2] - camera["world_anchor_model_units"][2]) / (camera_distance + pelvis[1])])
            target, _ = perspective_points(relative + pelvis + direction * changes[index], camera)
            points = {point["index"]: point for point in row["measurements"]["landmarks"]["value"]}
            for name, pixel in zip(JOINT_LANDMARKS, target):
                point = points[JOINT_LANDMARKS[name]]
                point["position"] = pixel.tolist()
                point["visibility"] = .2 if frame == weak_frame else 1.
    finally:
        env.close()
    raw = json.dumps(spec, sort_keys=True).encode()
    return raw, json.dumps(document, sort_keys=True).encode(), kwargs


def test_known_depth_change_improves_images_without_changing_ray_first_pose_or_inputs(monkeypatch):
    raw, evidence, kwargs = fixture()
    original = deepcopy((raw, evidence, kwargs))
    monkeypatch.setattr(mujoco, "mj_step", lambda *args, **kw: pytest.fail("No dynamics in a construction proposal"))
    result = depth.propose_perspective_root_depth(raw, evidence, **kwargs)
    assert (raw, evidence, kwargs) == original
    report = result["report"]
    assert result["status"] == "image_hypothesis_unadmitted"
    assert not report["promoted"] and report["geometry_admission"] == "unverified_not_executed"
    assert report["physics_steps"] == 0
    assert report["changed_reference_poses"] == 5
    assert 0 < report["maximum_depth_change_model_units"] < .15
    assert report["maximum_pelvis_ray_error_px"] < 1e-10
    assert report["first_root_unchanged"]
    assert len(result["root_reference_candidate"]) == len(report["references"]) == 6
    assert len(report["source_slots"]) == 11
    image = report["image_summary"]
    assert image["confident_samples"] == 72
    assert image["after_rmse_px"] < image["before_rmse_px"]
    assert all(row["after_rmse_px"] <= row["before_rmse_px"] + 1e-8 for row in image["per_joint"].values())
    for original_root, candidate in zip(json.loads(raw)["root_reference"], result["root_reference_candidate"]):
        assert original_root["time_s"] == candidate["time_s"]
    candidate = json.loads(raw)
    candidate["root_reference"] = result["root_reference_candidate"]
    with pytest.raises(ValueError, match="depth must remain authored zero"):
        validate_human(candidate)


def test_projected_ray_derivative_matches_numeric_difference():
    raw, _, kwargs = fixture()
    camera = kwargs["camera_hypothesis"]
    world = np.array([[[.2, .1, .5], [-.1, -.2, .9]], [[.3, .2, .4], [.1, .1, .8]]])
    direction = np.array([[.2, 1., .1], [.1, 1., -.2]])
    delta = np.array([.03, -.04])
    _, derivative = depth._project(world, direction, delta, camera)
    high, _ = depth._project(world, direction, delta + 1e-6, camera)
    low, _ = depth._project(world, direction, delta - 1e-6, camera)
    np.testing.assert_allclose((high - low) / 2e-6, derivative, atol=1e-7, rtol=0)


def test_weak_and_withheld_rows_remain_unmodified_and_cannot_change_the_fit():
    first = fixture(withheld=True, secret_shift=0., weak_frame=4)
    other = fixture(withheld=True, secret_shift=90., weak_frame=4)
    result = depth.propose_perspective_root_depth(first[0], first[1], **first[2])
    second = depth.propose_perspective_root_depth(other[0], other[1], **other[2])
    assert result["root_reference_candidate"] == second["root_reference_candidate"]
    original = json.loads(first[0])
    for index in (0, 2, 3):
        assert result["root_reference_candidate"][index] == original["root_reference"][index]
    assert result["report"]["references"][3]["source_admission"] == "withheld"
    assert len(result["report"]["source_slots"]) == 11
    assert result["report"]["image_summary"]["confident_samples"] == 48
    assert result["report"]["image_summary"]["unchanged_first_and_unsupported_rows_included"]


def test_far_camera_abstains_on_flat_likelihood_despite_regularization():
    raw, evidence, kwargs = fixture(camera_distance=50.)
    result = depth.propose_perspective_root_depth(raw, evidence, **kwargs)
    assert result["status"] == "unsupported"
    assert "no_locally_identifiable_supported_depth_rows" in result["report"]["failures"]
    assert result["root_reference_candidate"] == json.loads(raw)["root_reference"]
    assert result["report"]["iterations"] == 0
    assert all(not row["fit_enabled"] for row in result["report"]["references"])


def test_unchanged_exact_images_never_claim_an_improvement():
    raw, evidence, kwargs = fixture(correction=False)
    result = depth.propose_perspective_root_depth(raw, evidence, **kwargs)
    assert result["status"] == "rejected_image_hypothesis"
    assert "no_confident_image_improvement" in result["report"]["failures"]
    assert result["root_reference_candidate"] == json.loads(raw)["root_reference"]


def test_banded_solver_matches_an_independent_dense_solution():
    diagonal = np.array([3., 4., 5., 3.])
    off = np.array([-.5, -1., -.25])
    rhs = np.array([2., -1., 4., .5])
    dense = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)
    np.testing.assert_allclose(depth._tridiagonal(diagonal, off, rhs), np.linalg.solve(dense, rhs), atol=1e-12)


@pytest.mark.parametrize("change", ["source", "reference_id", "clock", "anchor", "scale", "partition", "raw_observation", "stacked"])
def test_source_clock_camera_and_construction_boundaries_reject(change):
    raw, evidence, kwargs = fixture()
    model, document = json.loads(raw), json.loads(evidence)
    if change == "source":
        model["source_sha256"] = "b" * 64
    elif change == "reference_id":
        model["reference_evidence"][1]["evidence_id"] = model["reference_evidence"][0]["evidence_id"]
    elif change == "clock":
        model["reference_evidence"][1]["time_s"] += .001
    elif change == "anchor":
        kwargs["camera_hypothesis"]["world_anchor_model_units"][2] += .01
    elif change == "scale":
        kwargs["camera_hypothesis"]["focal_px"] *= 2
    elif change == "partition":
        document["allowed_partitions"] = ["construction", "audit"]
    elif change == "raw_observation":
        document["profile"] = "scene-observations-v1"
    else:
        model["arm_equipment_refinement"] = {"status": "candidate"}
    with pytest.raises(ValueError):
        depth.propose_perspective_root_depth(json.dumps(model).encode(), json.dumps(document).encode(), **kwargs)


@pytest.mark.parametrize("settings", [{"maximum_depth_change_model_units": .16}, {"minimum_confidence": .4},
    {"maximum_seconds": 31}, {"maximum_iterations": True}, {"maximum_iterations": 13}, {"unknown": 1},
    {"image_sigma_px": float("nan")}, {"maximum_backtracks": 1.5}])
def test_bounds_reject_without_kinematics(settings, monkeypatch):
    raw, evidence, kwargs = fixture()
    monkeypatch.setattr(depth, "HumanEnv", lambda *args: pytest.fail("Invalid settings reached compilation"))
    with pytest.raises(ValueError):
        depth.propose_perspective_root_depth(raw, evidence, settings=settings, **kwargs)


def test_budget_exhaustion_preserves_all_inputs_slots_and_unchanged_roots(monkeypatch):
    raw, evidence, kwargs = fixture()
    def exhausted(_):
        raise TimeoutError("test budget exhausted")
    monkeypatch.setattr(depth, "_check", exhausted)
    result = depth.propose_perspective_root_depth(raw, evidence, **kwargs)
    assert result["status"] == "partial_budget_exhausted"
    assert len(result["report"]["source_slots"]) == 11
    assert len(result["report"]["references"]) == 6
    assert result["root_reference_candidate"] == json.loads(raw)["root_reference"]
    assert result["report"]["image_summary"] is None


def test_near_camera_envelope_rejects_instead_of_changing_camera_or_bounds():
    raw, evidence, kwargs = fixture(camera_distance=.1)
    with pytest.raises(ValueError, match="camera"):
        depth.propose_perspective_root_depth(raw, evidence, **kwargs)


def test_subresolution_reference_intervals_fail_before_temporal_overflow():
    raw, evidence, kwargs = fixture()
    model = json.loads(raw)
    for key in ("reference", "root_reference", "reference_evidence"):
        model[key][1]["time_s"] = 1e-8
    with pytest.raises(ValueError, match="intervals are too small"):
        depth.propose_perspective_root_depth(json.dumps(model).encode(), evidence, **kwargs)


def test_malformed_construction_fails_with_an_explicit_input_error():
    raw, evidence, kwargs = fixture()
    document = json.loads(evidence)
    del document["sources"][0]["observations"]
    with pytest.raises(ValueError, match="Malformed qualified construction"):
        depth.propose_perspective_root_depth(raw, json.dumps(document).encode(), **kwargs)


def study_fixture(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/studies/check_perspective_root_depth.py"
    spec = importlib.util.spec_from_file_location("root_depth_study", path)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    raw, evidence, kwargs = fixture()
    source, out = tmp_path / "source", tmp_path / "study"
    source.mkdir()
    entity, camera = kwargs["human_entity_ref"], kwargs["camera_hypothesis"]
    files = {"human.json": raw, "construction-evidence.json": evidence,
             "scene.json": json.dumps({"human_spec": json.loads(raw), "objects": [{"provenance": {"camera_hypothesis": camera}}],
                 "source_binding": {"entity_bindings": [{"world_entity_id": "human", "observation_entity_id": entity["object_id"],
                     "observation_document_sha256": entity["document_sha256"]}]}}).encode(),
             "geometry-proposal.json": json.dumps({"camera_hypothesis": camera,
                 "body_binding": {"human_json_sha256": hashlib.sha256(raw).hexdigest()}}).encode(),
             "source.json": b'{"license":"authored analytical fixture"}'}
    for name, value in files.items():
        (source / name).write_bytes(value)
    (source / "manifest.json").write_text(json.dumps({"files": {
        name: hashlib.sha256(value).hexdigest() for name, value in files.items()}}))
    return study, source, out, files


def test_study_preserves_one_fit_and_caps_individual_reads_below_aggregate_budget(tmp_path, monkeypatch):
    study, source, out, files = study_fixture(tmp_path)
    calls, bounds = [], []
    read, propose = study.read_package_file, study.propose_perspective_root_depth
    def bounded(*args, **kwargs):
        bounds.append(kwargs["max_bytes"])
        assert kwargs["max_bytes"] <= 64 * 1024 * 1024
        return read(*args, **kwargs)
    def fitted(*args, **kwargs):
        frozen = json.loads((out / "pre-fit.json").read_text())
        assert frozen["candidate_count"] == 1 and frozen["settings"] == depth.DEFAULTS
        calls.append(1)
        return propose(*args, **kwargs)
    monkeypatch.setattr(study, "read_package_file", bounded)
    monkeypatch.setattr(study, "propose_perspective_root_depth", fitted)
    report = study.run(source, out)
    assert report["status"] == "image_hypothesis_unadmitted"
    assert calls == [1] and max(bounds) == 64 * 1024 * 1024
    manifest = json.loads((out / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    assert all((out / "inputs" / name).read_bytes() == value for name, value in files.items())


def test_study_reserves_complete_archive_proposal_and_metadata_before_output_creation(tmp_path, monkeypatch):
    study, source, out, _ = study_fixture(tmp_path)
    monkeypatch.setattr(study, "MAX_OUTPUT_BYTES", study.MAX_PROPOSAL_BYTES + study.METADATA_RESERVE_BYTES)
    monkeypatch.setattr(study, "propose_perspective_root_depth", lambda *args, **kwargs: pytest.fail("Oversized plan ran a fit"))
    with pytest.raises(ValueError, match="archive plus proposal/metadata reserves"):
        study.run(source, out)
    assert not out.exists()


@pytest.mark.parametrize("failure", ["postwrite_bound", "partial_manifest", "project_lock_changed"])
def test_study_failure_seals_only_failed_receipts_and_preserves_unqualified_outputs(tmp_path, monkeypatch, failure):
    study, source, out, files = study_fixture(tmp_path)
    changed = False
    original_propose, original_write, original_read = study.propose_perspective_root_depth, study.write_package_files, Path.read_bytes
    def fitted(*args, **kwargs):
        nonlocal changed
        result = original_propose(*args, **kwargs)
        if failure == "postwrite_bound":
            monkeypatch.setattr(study, "MAX_OUTPUT_BYTES", 1024)
        if failure == "project_lock_changed":
            changed = True
        return result
    def write(directory, payloads):
        if failure == "partial_manifest" and "manifest.pending.json" in payloads:
            (Path(directory) / "manifest.pending.json").write_bytes(b'{"status":"unfinished')
            raise OSError("injected manifest write failure")
        return original_write(directory, payloads)
    def read(path):
        raw = original_read(path)
        return raw + b"\n" if changed and path.name == "requirements-lab.lock" and path.parent.name == "weft" else raw
    monkeypatch.setattr(study, "propose_perspective_root_depth", fitted)
    monkeypatch.setattr(study, "write_package_files", write)
    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises((ValueError, OSError)):
        study.run(source, out)
    manifest = json.loads((out / "manifest.json").read_text())
    report = json.loads((out / "report.json").read_text())
    assert manifest["status"] == report["status"] == "failed"
    assert manifest["custody"] == report["custody"] == "failure_receipt_only"
    assert manifest["unsealed_outputs_preserved"]
    assert set(manifest["files"]) == {"report.json", "failure.json"}
    assert sum((out / name).stat().st_size for name in ("manifest.json", "report.json", "failure.json")) <= study.MAX_FAILURE_RECEIPT_BYTES
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    assert all((out / "inputs" / name).read_bytes() == value for name, value in files.items())
    assert (out / "proposal.json").is_file()
    if failure == "partial_manifest":
        assert (out / "manifest.pending.json").read_bytes() == b'{"status":"unfinished'
        assert (out / "report.publication-failed.json").is_file()
    if failure == "project_lock_changed":
        assert "project/requirements-lab.lock" in report["error"]
