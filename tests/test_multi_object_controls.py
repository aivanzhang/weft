"""Multi-object actor switching, policy guard, HTTP /api/actor, and exact replay checks."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from weft.io import digest, write_json
from weft.lab import Lab
from weft.recording import load_recording, verify_recording
from weft.scene import SceneEnv
from weft.spatial_lab import authored_spatial_scene, prepare_spatial_lab, write_sealed_spatial_fixture
from weft.spatial_rigid import SpatialRigidEnv
from test_capabilities import make_planar_scene
from test_lab_http import request, serving


def make_multi_object_planar_lab(root):
    root = Path(root)
    run = root / "run"
    run.mkdir(parents=True)
    scene = make_planar_scene(profile="planar-rigid-v2", multi_object=True)
    write_json(root / "lab-config.json", {"profile": "planar"})
    write_json(root / "status.json", {"status": "ready", "events": []})
    write_json(run / "scene.json", scene)
    write_json(
        run / "policy.json",
        {
            "schema_version": 1,
            "algorithm": "tabular_q_learning",
            "controlled_id": "actor-a",
            "error_edges": [-0.1, 0.1],
            "velocity_edges": [-0.1, 0.1],
            "q_values": {},
        },
    )
    write_json(run / "manifest.json", {"files": {p.name: digest(p) for p in sorted(run.iterdir())}})
    return Lab(root)


def test_planar_select_actor_preserves_live_physics_and_rejects_static_or_truncated_overflow():
    env = SceneEnv(make_planar_scene(profile="planar-rigid-v2", multi_object=True))
    try:
        env.reset(seed=3, options={"jitter": False})
        for _ in range(4):
            env.step(2)
        before_switch = env.snapshot()
        pos_a_before = before_switch["objects"][0]["position"]
        vel_a_before = before_switch["objects"][0]["velocity"]
        pos_b_before = before_switch["objects"][1]["position"]

        assert pos_a_before[0] > 0.2
        with pytest.raises(ValueError, match="Static object"):
            env.select_actor("wall-0")
        with pytest.raises(ValueError, match="Unknown scene object"):
            env.select_actor("missing-id")
        assert env.scene["controlled_id"] == "actor-a"

        obs = env.select_actor("actor-b")
        assert env.scene["controlled_id"] == "actor-b"
        np.testing.assert_allclose(obs[:2], np.array(env.scene["target"]) - np.array(pos_b_before), atol=1e-6)
        after_switch = env.snapshot()
        assert after_switch["objects"][0]["position"] == pos_a_before
        assert after_switch["objects"][0]["velocity"] == vel_a_before
        assert after_switch["objects"][1]["position"] == pos_b_before

        for _ in range(4):
            env.step(4)
        moved = env.snapshot()
        assert moved["objects"][1]["position"][1] > pos_b_before[1]
        assert moved["objects"][0]["position"][0] > pos_a_before[0], "Actor A continues coasting while Actor B is driven"
    finally:
        env.close()

    short_scene = make_planar_scene(profile="planar-rigid-v1", multi_object=True)
    short_scene["task"]["max_steps"] = 20
    short_env = SceneEnv(short_scene)
    try:
        short_env.reset(seed=0, options={"jitter": False})
        for _ in range(20):
            _, _, terminated, truncated, _ = short_env.step(0)
        assert truncated and not terminated
        short_env.select_actor("actor-b")
        assert short_env.truncated and short_env._done
        with pytest.raises(RuntimeError, match="Reset before stepping"):
            short_env.step(2)
    finally:
        short_env.close()


def test_spatial_select_actor_preserves_3d_state_and_drives_switched_object():
    env = SpatialRigidEnv(authored_spatial_scene())
    try:
        env.reset(seed=5)
        for _ in range(6):
            env.step(2)
        before = env.snapshot()
        block_before = next(obj for obj in before["objects"] if obj["id"] == "block")
        comp_before = next(obj for obj in before["objects"] if obj["id"] == "companion")
        assert block_before["position"][0] > 0.24

        with pytest.raises(ValueError, match="Static object"):
            env.select_actor("floor")
        with pytest.raises(ValueError, match="Unknown spatial object"):
            env.select_actor("nonexistent")
        assert env.scene["controlled_id"] == "block"

        obs = env.select_actor("companion")
        assert env.scene["controlled_id"] == "companion"
        np.testing.assert_allclose(obs[-4:-1], np.array(env.scene["target"]) - np.array(comp_before["position"]), atol=1e-6)
        switched = env.snapshot()
        assert switched["controlled_id"] == "companion"
        assert switched["virtual_force"] == [0.0, 0.0, 0.0]
        assert next(obj for obj in switched["objects"] if obj["id"] == "block")["position"] == block_before["position"]

        for _ in range(8):
            env.step(5)
        after = env.snapshot()
        comp_after = next(obj for obj in after["objects"] if obj["id"] == "companion")
        assert comp_after["position"][2] > comp_before["position"][2], "Upward (+z) action lifts switched companion sphere"
    finally:
        env.close()


def test_lab_actor_control_trained_policy_guard_immutability_and_exact_replay(tmp_path):
    planar_lab = make_multi_object_planar_lab(tmp_path / "planar-lab")
    scene_hash_before = digest(planar_lab.root / "run" / "scene.json")
    manifest_hash_before = digest(planar_lab.root / "run" / "manifest.json")
    try:
        summary = planar_lab.summary()
        assert summary["capabilities"]["entities"]["active_control_target"] == "actor-a"
        assert summary["capabilities"]["controllers"]["trained_target_id"] == "actor-a"

        res = planar_lab.control("/api/actor", {"actor": "actor-b"})
        assert res["scene"]["controlled_id"] == "actor-b"
        assert res["capabilities"]["entities"]["active_control_target"] == "actor-b"

        with pytest.raises(ValueError, match="trained for actor-a"):
            planar_lab.control("/api/step", {"steps": 1, "policy": True})

        planar_lab.control("/api/actor", {"actor": "actor-a"})
        step_ok = planar_lab.control("/api/step", {"steps": 1, "policy": True})
        assert len(step_ok["transitions"]) == 1
        assert digest(planar_lab.root / "run" / "scene.json") == scene_hash_before
        assert digest(planar_lab.root / "run" / "manifest.json") == manifest_hash_before
    finally:
        planar_lab.close()

    spatial_src = tmp_path / "spatial-src"
    write_sealed_spatial_fixture(spatial_src)
    src_manifest_before = digest(spatial_src / "manifest.json")
    spatial_lab_dir = tmp_path / "spatial-lab"
    prepare_spatial_lab(spatial_src, spatial_lab_dir)
    run_scene_before = digest(spatial_lab_dir / "run" / "scene.json")

    spatial_lab = Lab(spatial_lab_dir)
    try:
        switched = spatial_lab.control("/api/actor", {"actor": "companion"})
        assert switched["snapshot"]["controlled_id"] == "companion"
        assert switched["capabilities"]["entities"]["active_control_target"] == "companion"

        rec_start = spatial_lab.control("/api/record/start", {})
        session_dir = spatial_lab_dir / "sessions" / rec_start["directory"]
        with pytest.raises(ValueError, match="Stop and export the recording"):
            spatial_lab.control("/api/actor", {"actor": "block"})

        spatial_lab.control("/api/step", {"action": 2, "steps": 4})
        spatial_lab.control("/api/reset", {"seed": 11})
        spatial_lab.control("/api/step", {"action": 5, "steps": 4})
        spatial_lab.control("/api/record/stop", {})
        export_result = spatial_lab.export_future.result(timeout=30)
        assert Path(export_result["replay"]).is_file()
        assert Path(export_result["video"]).is_file()

        verify_recording(session_dir)
        loaded = load_recording(session_dir)
        assert loaded["interrupted"] is False and loaded["incomplete_journal"] is False
        assert loaded["scene"]["controlled_id"] == "companion"
        assert len(loaded["events"]) == 10
        assert loaded["events"][-1]["snapshot"]["controlled_id"] == "companion"
        assert digest(spatial_src / "manifest.json") == src_manifest_before
        assert digest(spatial_lab_dir / "run" / "scene.json") == run_scene_before
    finally:
        spatial_lab.close()


def test_http_actor_endpoint_enforces_auth_and_profile_scope(tmp_path):
    with serving(make_multi_object_planar_lab(tmp_path / "http-planar")) as server:
        assert request(server, "/api/actor", method="POST", data={"actor": "actor-b"}, token=False)[0] == 401
        code, _, body = request(server, "/api/actor", method="POST", data={"actor": "actor-b"})
        assert code == 200
        payload = json.loads(body)
        assert payload["scene"]["controlled_id"] == "actor-b"
        assert payload["capabilities"]["entities"]["active_control_target"] == "actor-b"
        assert request(server, "/api/actor", method="POST", data={"actor": "wall-0"})[0] == 400
