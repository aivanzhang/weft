from copy import deepcopy
import json
import subprocess
import sys

import pytest

from weft.io import digest, write_json
from weft.lab import Lab, SpatialSession
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, verify_recording
from weft.spatial_lab import authored_spatial_scene, prepare_spatial_lab, write_sealed_spatial_fixture
from test_lab_http import request, serving


def _seal_dir(root):
    write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "files": {
                str(p.relative_to(root)): digest(p)
                for p in sorted(root.rglob("*"))
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )


def test_prepare_spatial_lab_preserves_source_immutability_and_provenance(tmp_path):
    source = tmp_path / "source-3d"
    fixture_info = write_sealed_spatial_fixture(source)
    (source / "unsealed-note.txt").write_text("must not enter packaged lab")
    before = verify_artifacts(source)
    assert before == fixture_info["verification"]

    out = tmp_path / "spatial-lab"
    result = prepare_spatial_lab(source, out)
    assert verify_artifacts(source) == before
    assert result["learned_policy_available"] is False
    assert result["physical_validation"] == "unverified"
    assert verify_artifacts(out / "run") == result["verification"]
    assert not (out / "run" / "unsealed-note.txt").exists()
    assert not (out / "run" / "inputs" / "source-run" / "unsealed-note.txt").exists()
    assert not (out / "run" / "policy.json").exists()

    report = json.loads((out / "run" / "report.json").read_text())
    assert report["physical_validation"] == "unverified"
    assert report["physics_ready"] is False
    assert report["learned_policy_available"] is False
    assert report["paid_requests"] == 0

    interventions = json.loads((out / "run" / "interventions.json").read_text())
    assert [trial["name"] for trial in interventions["trials"]] == [
        "coast",
        "left",
        "right",
        "forward",
        "back",
        "up",
        "down",
        "brake",
    ]
    brake_check = interventions["moving_brake_check"]
    assert brake_check["speed_after_brake_mps"] < brake_check["speed_after_coast_mps"]

    with pytest.raises(FileExistsError):
        prepare_spatial_lab(source, out)
    with pytest.raises(ValueError, match="outside the immutable source run"):
        prepare_spatial_lab(source, source / "nested-lab")


@pytest.mark.parametrize(
    ("mutator", "expected_pattern"),
    [
        (lambda root: write_json(root / "report.json", {"physical_validation": "verified"}), "physically unverified"),
        (lambda root: write_json(root / "report.json", {"physical_validation": "unverified", "physics_ready": True}), "physically unverified"),
        (lambda root: write_json(root / "report.json", {"physical_validation": "unverified", "status": "failed"}), "failed or rejected"),
        (lambda root: write_json(root / "report.json", {"physical_validation": "unverified", "candidate_sha256": "abc"}), "missing its sealed candidate.json"),
    ],
)
def test_prepare_spatial_lab_rejects_false_physical_claims_and_failed_exports(tmp_path, mutator, expected_pattern):
    source = tmp_path / "bad-source"
    write_sealed_spatial_fixture(source)
    mutator(source)
    _seal_dir(source)
    out = tmp_path / "rejected-out"
    with pytest.raises(ValueError, match=expected_pattern):
        prepare_spatial_lab(source, out)
    assert not out.exists()


def test_prepare_spatial_lab_rejects_unsealed_sidecars_bad_paths_and_initial_overlap(tmp_path):
    source = tmp_path / "unsealed-candidate"
    write_sealed_spatial_fixture(source)
    write_json(source / "candidate.json", {"physical_validation": "unverified"})
    with pytest.raises(ValueError, match="Unsealed candidate.json"):
        prepare_spatial_lab(source, tmp_path / "out-1")

    overlapping = tmp_path / "overlapping"
    overlapping.mkdir()
    bad_scene = authored_spatial_scene()
    bad_scene["objects"][0]["position"] = [0.50, 0.42, 0.075]  # overlaps companion sphere
    write_json(overlapping / "scene.json", bad_scene)
    _seal_dir(overlapping)
    with pytest.raises(ValueError, match="intersect"):
        prepare_spatial_lab(overlapping, tmp_path / "out-2")
    assert not (tmp_path / "out-2").exists()

    bad_manifest_dir = tmp_path / "bad-manifest"
    write_sealed_spatial_fixture(bad_manifest_dir)
    manifest = json.loads((bad_manifest_dir / "manifest.json").read_text())
    scene_path = bad_manifest_dir / "scene.json"
    manifest["files"][str(scene_path.resolve())] = digest(scene_path)
    write_json(bad_manifest_dir / "manifest.json", manifest)
    with pytest.raises(ValueError, match="canonical relative"):
        prepare_spatial_lab(bad_manifest_dir, tmp_path / "out-3")


def test_spatial_lab_controls_alter_intended_3d_axes_and_reject_invalid_actions(tmp_path):
    source = tmp_path / "source"
    out = tmp_path / "lab"
    write_sealed_spatial_fixture(source)
    prepare_spatial_lab(source, out)

    lab = Lab(out)
    try:
        summary = lab.summary()
        assert isinstance(lab.session, SpatialSession)
        assert summary["profile"] == "spatial-rigid"
        assert summary["learned_policy_available"] is False
        assert summary["scene"]["profile"] == "spatial-rigid-v1"

        def run_from_reset(actions):
            lab.control("/api/reset", {"seed": 11})
            for code in actions:
                res = lab.control("/api/step", {"action": code, "steps": 2})
            return next(o for o in res["snapshot"]["objects"] if o["id"] == "block")

        coast = run_from_reset([0])
        left = run_from_reset([1])
        right = run_from_reset([2])
        forward = run_from_reset([3])
        back = run_from_reset([4])
        up = run_from_reset([5])
        down = run_from_reset([6])
        coast_after_push = run_from_reset([2, 0])
        brake_after_push = run_from_reset([2, 7])

        assert left["position"][0] < coast["position"][0] < right["position"][0]
        assert back["position"][1] < coast["position"][1] < forward["position"][1]
        assert down["position"][2] < coast["position"][2] < up["position"][2]
        assert abs(brake_after_push["velocity"][0]) < abs(coast_after_push["velocity"][0])

        snap_before = deepcopy(lab.session.snapshot())
        for invalid in (
            {"action": -1},
            {"action": 8},
            {"action": [0.0] * 22},
            {"policy": True},
            {"controller": "reference"},
            {"action": 1, "steps": 0},
            {"action": 1, "steps": 7},
        ):
            with pytest.raises(ValueError):
                lab.control("/api/step", invalid)
            assert lab.session.snapshot() == snap_before
        with pytest.raises(ValueError):
            lab.control("/api/joint", {"joint": "left_knee"})
    finally:
        lab.close()


def test_http_spatial_lab_auth_recording_export_reopen_and_exact_replay_with_resets(tmp_path):
    source = tmp_path / "source"
    out = tmp_path / "lab"
    write_sealed_spatial_fixture(source)
    prepare_spatial_lab(source, out)
    run_seal_before = verify_artifacts(out / "run")

    with serving(Lab(out)) as server:
        assert request(server, "/api/status", token=False)[0] == 401
        assert request(server, "/api/status", headers={"Host": "localhost"})[0] == 403
        assert request(server, "/api/status", headers={"Origin": "https://evil.example"})[0] == 403

        code, _, body = request(server, "/api/status")
        assert code == 200
        status_doc = json.loads(body)
        assert status_doc["profile"] == "spatial-rigid"
        assert status_doc["learned_policy_available"] is False

        code, _, body = request(server, "/api/record/start", method="POST", data={})
        assert code == 200
        session_name = json.loads(body)["directory"]

        recorded_steps = []
        for action in (2, 3, 5, 7):
            code, _, body = request(server, "/api/step", method="POST", data={"action": action, "steps": 2})
            assert code == 200
            recorded_steps.extend(json.loads(body)["transitions"])

        code, _, _ = request(server, "/api/reset", method="POST", data={"seed": 23})
        assert code == 200

        for action in (1, 4, 6, 0):
            code, _, body = request(server, "/api/step", method="POST", data={"action": action, "steps": 1})
            assert code == 200
            recorded_steps.extend(json.loads(body)["transitions"])

        code, _, _ = request(server, "/api/record/stop", method="POST", data={})
        assert code == 200
        export_result = server.lab.export_future.result(timeout=15)

        session_dir = out / "sessions" / session_name
        assert export_result["steps"] == 12
        assert (session_dir / "replay.html").is_file()
        assert (session_dir / "video.mp4").is_file()
        assert (session_dir / "preview.gif").is_file()
        assert "3D recorded simulation" in (session_dir / "replay.html").read_text()

        verified_rec = verify_recording(session_dir)
        assert verified_rec["steps"] == 12

        code, headers, _ = request(server, f"/files/sessions/{session_name}/replay.html?token={server.lab.token}", token=False)
        assert code == 200
        assert "attachment" in headers.get("Content-Disposition", "")

    assert verify_artifacts(out / "run") == run_seal_before

    # Reopen the saved lab and verify exact state/reward/reset replay.
    reopened = Lab(out)
    try:
        reopened_summary = reopened.summary()
        assert reopened_summary["sessions"] == [{"name": session_name, "complete": True}]
    finally:
        reopened.close()

    saved = load_recording(out / "sessions" / session_name)
    step_rows = [row for row in saved["events"] if row["type"] == "step"]
    assert [{k: v for k, v in row.items() if k != "index"} for row in step_rows] == recorded_steps

    replay_session = SpatialSession(out / "run")
    try:
        reset_events = 0
        for row in saved["events"]:
            if row["type"] in ("start", "reset"):
                if row["type"] == "reset":
                    reset_events += 1
                replay_session.reset(row["seed"])
            else:
                reward, terminated, truncated, _ = replay_session.step(row["action"])
                assert (reward, terminated, truncated) == (row["reward"], row["terminated"], row["truncated"])
            assert replay_session.snapshot() == row["snapshot"]
        assert reset_events == 1
    finally:
        replay_session.env.close()


def test_spatial_runtime_failure_preserves_last_valid_state_and_exports_prefix(tmp_path, monkeypatch):
    source = tmp_path / "source"
    out = tmp_path / "lab"
    write_sealed_spatial_fixture(source)
    prepare_spatial_lab(source, out)

    lab = Lab(out)
    try:
        started = lab.control("/api/record/start", {})
        stepped = lab.control("/api/step", {"action": 2, "steps": 1})
        last_valid_objects = deepcopy(stepped["snapshot"]["objects"])

        def fail_step(action):
            raise RuntimeError("Spatial dynamics exceeded finite envelope")

        monkeypatch.setattr(lab.session.env, "step", fail_step)
        response = lab.control("/api/step", {"action": 5, "steps": 1})
        assert response["recording"] is False
        assert response["transitions"] == []
        assert response["snapshot"]["objects"] == last_valid_objects
        assert response["snapshot"]["terminated"] is True
        assert response["snapshot"]["simulation_failure"]["attempted_action"] == 5

        export_result = lab.export_future.result(timeout=15)
        assert export_result["steps"] == 1
        saved = load_recording(out / "sessions" / started["directory"])
        assert saved["metadata"]["failed_attempt"]["reason"] == "Spatial dynamics exceeded finite envelope"

        reset_res = lab.control("/api/reset", {"seed": 5})
        assert "simulation_failure" not in reset_res["snapshot"]
        assert reset_res["snapshot"]["terminated"] is False
    finally:
        lab.close()


def test_cli_spatial_fixture_spatial_lab_and_lab_scene_run_no_serve(tmp_path):
    fixture_dir = tmp_path / "cli-fixture"
    lab_one = tmp_path / "cli-spatial-lab"
    lab_two = tmp_path / "cli-lab-scene-run"

    res_fixture = subprocess.run(
        [sys.executable, "-m", "weft", "spatial-fixture", "--out", str(fixture_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(res_fixture.stdout)["directory"] == str(fixture_dir.resolve())

    res_pkg = subprocess.run(
        [sys.executable, "-m", "weft", "spatial-lab", "--run", str(fixture_dir), "--out", str(lab_one)],
        check=True,
        capture_output=True,
        text=True,
    )
    pkg_doc = json.loads(res_pkg.stdout)
    assert pkg_doc["learned_policy_available"] is False
    assert (lab_one / "lab-config.json").is_file()

    res_lab = subprocess.run(
        [sys.executable, "-m", "weft", "lab", "--scene-run", str(fixture_dir), "--out", str(lab_two), "--no-serve"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"status": "ready"' in res_lab.stdout
    assert (lab_two / "run" / "manifest.json").is_file()

    bad = subprocess.run(
        [sys.executable, "-m", "weft", "lab", "--resume", str(lab_one), "--scene-run", str(fixture_dir)],
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0
    assert "--resume cannot be combined" in (bad.stdout + bad.stderr)
