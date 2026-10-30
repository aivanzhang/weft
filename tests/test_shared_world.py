"""End-to-end tests for coupled articulated-human + 3D rigid-object shared-world-v1 runtime."""

from copy import deepcopy
import json
from pathlib import Path
import urllib.error
import urllib.request

import numpy as np
import pytest

from weft.capabilities import describe_capabilities, describe_run_capabilities
from weft.io import canonical_hash, digest, local_time, write_json
from weft.lab import Lab, LabServer
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, recover_incomplete_recording, verify_recording
from weft.shared_world import (
    OBJECT_ACTIONS,
    PROFILE,
    SharedWorldEnv,
    audit_shared_world_feasibility,
    human_subtree_mass,
    render_snapshot,
    validate_shared_snapshot,
    validate_shared_world,
)
from weft.shared_world_lab import (
    _analytic_mobile_human_spec,
    adapt_to_shared_world,
    authored_shared_world_scene,
    prepare_shared_world_lab,
    write_sealed_shared_world_fixture,
)
from weft.shared_world_learning import policy_action, train_shared_world_policy, validate_shared_policy
from weft.whole_body import SUCCESS_LIMITS


def _http_json(port, token, method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Origin": f"http://127.0.0.1:{port}",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_validate_shared_world_enforces_schema_and_cross_track_invariants():
    scene = authored_shared_world_scene()
    validated = validate_shared_world(scene)
    assert validated["profile"] == PROFILE
    assert validated["physical_validation"] == "unverified"
    assert validated["action_contract"]["human_motor_count"] == 22
    assert validated["action_contract"]["object_force_channels"] == 6
    assert validated["action_contract"]["dynamic_object_ids"] == ["ball", "marker_box"]

    bad_phys = deepcopy(scene)
    bad_phys["physical_validation"] = "verified"
    with pytest.raises(ValueError, match="unverified"):
        validate_shared_world(bad_phys)

    bad_limits = deepcopy(scene)
    bad_limits["task"]["human_limits"]["angular_rmse_rad"] = 0.50
    with pytest.raises(ValueError, match="SUCCESS_LIMITS"):
        validate_shared_world(bad_limits)

    bad_id = deepcopy(scene)
    bad_id["objects"][0]["id"] = "torso"
    with pytest.raises(ValueError, match="disjoint from human identifiers"):
        validate_shared_world(bad_id)

    bad_role = deepcopy(scene)
    bad_role["objects"][0]["role"] = "active"  # controlled_id is 'human', so dynamic objects must be passive
    with pytest.raises(ValueError, match="role must be 'passive'"):
        validate_shared_world(bad_role)

    bad_dt = deepcopy(scene)
    bad_dt["dt"] = 1.0 / 60.0
    with pytest.raises(ValueError, match="dt"):
        validate_shared_world(bad_dt)

    # Source-qualified slot conservation & unique observation_entity_id guards.
    sha = "a" * 64
    sq = deepcopy(scene)
    sq["human_spec"]["source_sha256"] = sha
    sq["source_binding"] = {
        "mode": "source_qualified",
        "source_sha256": sha,
        "total_source_frames": 61,
        "evidence_documents": {"d" * 64: "evidence/poses.json", "e" * 64: "evidence/objects.json"},
        "entity_bindings": [
            {
                "world_entity_id": "human",
                "entity_kind": "articulated_human",
                "observation_entity_id": f"{sha}:human:0",
                "observation_document_sha256": "d" * 64,
                "provenance_mode": "observed_track",
                "observed_slots": 58,
                "missing_slots": 3,
                "pre_init_slots": 0,
            },
            {
                "world_entity_id": "ball",
                "entity_kind": "rigid_object",
                "observation_entity_id": f"{sha}:obj:ball",
                "observation_document_sha256": "e" * 64,
                "provenance_mode": "observed_track",
                "observed_slots": 50,
                "missing_slots": 6,
                "pre_init_slots": 5,
            },
            {
                "world_entity_id": "marker_box",
                "entity_kind": "rigid_object",
                "observation_entity_id": f"{sha}:obj:marker_box",
                "observation_document_sha256": "e" * 64,
                "provenance_mode": "inferred_prior",
                "observed_slots": 0,
                "missing_slots": 61,
                "pre_init_slots": 0,
            },
            {
                "world_entity_id": "table",
                "entity_kind": "rigid_object",
                "observation_entity_id": None,
                "observation_document_sha256": None,
                "provenance_mode": "authored_addition",
                "observed_slots": 0,
                "missing_slots": 61,
                "pre_init_slots": 0,
            },
        ],
    }
    assert validate_shared_world(sq)["source_binding"]["mode"] == "source_qualified"

    ambiguous_calibration = deepcopy(scene)
    ambiguous_calibration["coordinate_frame"]["calibration_status"] = "calibrated"
    with pytest.raises(ValueError, match="verified calibration binding"):
        validate_shared_world(ambiguous_calibration)

    same_bare_id = deepcopy(sq)
    same_bare_id["source_binding"]["entity_bindings"][0]["observation_entity_id"] = f"{sha}:obj:ball"
    validate_shared_world(same_bare_id)

    absent_document = deepcopy(sq)
    absent_document["source_binding"]["entity_bindings"][0]["observation_document_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="supplied evidence document"):
        validate_shared_world(absent_document)

    fabricated_observation = deepcopy(sq)
    fabricated_observation["source_binding"]["entity_bindings"][-1]["observation_entity_id"] = "invented-table"
    with pytest.raises(ValueError, match="must not invent observation identities"):
        validate_shared_world(fabricated_observation)

    for bad_path in ("../observations.json", "/tmp/observations.json", "./evidence.json", "evidence//a.json", "evidence\\a.json"):
        unsafe_document = deepcopy(sq)
        unsafe_document["source_binding"]["evidence_documents"]["d" * 64] = bad_path
        with pytest.raises(ValueError, match="safe relative JSON paths"):
            validate_shared_world(unsafe_document)

    dropped_slots = deepcopy(sq)
    dropped_slots["source_binding"]["entity_bindings"][1]["missing_slots"] = 5
    with pytest.raises(ValueError, match="Slot conservation failed"):
        validate_shared_world(dropped_slots)

    dup_obs = deepcopy(sq)
    dup_obs["source_binding"]["entity_bindings"][2]["observation_entity_id"] = f"{sha}:obj:ball"
    with pytest.raises(ValueError, match="Duplicate observation identity"):
        validate_shared_world(dup_obs)

    sha_mismatch = deepcopy(sq)
    sha_mismatch["human_spec"]["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="source_sha256 differs"):
        validate_shared_world(sha_mismatch)


def test_human_subtree_mass_isolation_and_scale_bridge():
    scene = authored_shared_world_scene()
    heavy = deepcopy(scene)
    heavy.pop("action_contract", None)
    heavy["objects"][0]["mass"] = 40.0
    heavy["objects"][2]["mass"] = 180.0
    heavy = validate_shared_world(heavy)

    # Human motor/root force limits must depend strictly on human mass, never on rigid object masses.
    assert np.allclose(
        scene["action_contract"]["physical_limits"][:22],
        heavy["action_contract"]["physical_limits"][:22],
        atol=1e-9,
        rtol=0,
    )
    assert heavy["action_contract"]["physical_limits"][22] == pytest.approx(40.0 * 6.0)

    m1 = human_subtree_mass(scene["human_spec"], scale=1.0)
    m_scaled = human_subtree_mass(scene["human_spec"], scale=1.2)
    assert m_scaled == pytest.approx(m1 * (1.2**3), rel=1e-5)

    scaled = deepcopy(scene)
    scaled.pop("action_contract", None)
    scaled["bounds"] = [2.4, 2.6, 3.0]
    scaled["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"] = 1.2
    scaled["objects"][0]["position"][1] += 0.25
    scaled["objects"][1]["position"][1] += 0.25
    scaled["objects"][2]["position"][1] += 0.25
    scaled["objects"][0]["position"][2] += 0.20
    scaled["objects"][1]["position"][2] += 0.20
    scaled["objects"][2]["position"][2] += 0.20
    scaled["target"][1] += 0.25
    scaled["target"][2] += 0.20
    env = SharedWorldEnv(scaled)
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == env.observation_space.shape
        assert info["physical_validation"] == "unverified"
        snap = env.snapshot()
        validate_shared_snapshot(snap, env.scene)
        assert snap["human"]["height"] > 1.8
    finally:
        env.close()


def test_bidirectional_newtons_third_law_contact_in_single_mujoco_model():
    scene = authored_shared_world_scene()
    no_contact_scene = deepcopy(scene)
    no_contact_scene.pop("action_contract", None)
    # Move the ball sideways out of the right arm's sweep while keeping everything else identical.
    no_contact_scene["objects"][0]["position"][0] = round(scene["objects"][0]["position"][0] - 0.34, 4)

    env_coupled = SharedWorldEnv(scene)
    env_free = SharedWorldEnv(no_contact_scene)
    try:
        # Both human and rigid objects live in one single MjModel with 22 explicit human motors.
        assert env_coupled.model.nu == 22
        assert env_coupled.model.body("ball").id > 0
        assert env_coupled.model.body("right_elbow_body").id > 0

        env_coupled.reset(seed=0, options={"jitter": False})
        env_free.reset(seed=0, options={"jitter": False})
        y0 = next(o["position"][1] for o in env_coupled.snapshot()["objects"] if o["id"] == "ball")

        max_elbow_diff = 0.0
        max_elbow_vel_diff = 0.0
        for _ in range(env_coupled.horizon_steps):
            act_coupled = env_coupled.authored_action()
            act_free = env_free.authored_action()
            # Zero virtual object force is applied: all ball motion comes purely from human limb contact.
            assert np.all(act_coupled[22:] == 0.0)
            env_coupled.step(act_coupled)
            env_free.step(act_free)
            h_c = env_coupled.snapshot()["human"]
            h_f = env_free.snapshot()["human"]
            max_elbow_diff = max(
                max_elbow_diff,
                abs(h_c["angles"]["right_elbow"] - h_f["angles"]["right_elbow"]),
                abs(h_c["angles"]["right_shoulder"] - h_f["angles"]["right_shoulder"]),
            )
            max_elbow_vel_diff = max(
                max_elbow_vel_diff,
                abs(h_c["angular_velocities"]["right_elbow"] - h_f["angular_velocities"]["right_elbow"]),
            )

        metrics = env_coupled.metrics()
        y_end = next(o["position"][1] for o in env_coupled.snapshot()["objects"] if o["id"] == "ball")
        assert metrics["human_object_contacts_total"] >= 1
        assert metrics["max_human_object_normal_force"] > 0.5
        assert y_end - y0 > 0.12, f"Expected human arm push to move ball >12cm forward, got {y_end - y0:.4f}m"
        # Equal-and-opposite contact impulse from the 0.45 kg ball measurably perturbs the human arm joints.
        assert max_elbow_diff > 1e-3
        assert max_elbow_vel_diff > 1e-2
        assert metrics["human_tracking_passed"] is True
    finally:
        env_coupled.close()
        env_free.close()

    # Reverse test: launch the 3D ball into an unactuated (zero-torque) human arm and verify both rebound & arm motion.
    impact_scene = deepcopy(scene)
    impact_scene.pop("action_contract", None)
    impact_scene["objects"][0]["velocity"] = [0.0, -1.8, 0.0]
    env_impact = SharedWorldEnv(impact_scene)
    env_no_ball = SharedWorldEnv(no_contact_scene)
    try:
        env_impact.reset(seed=0, options={"jitter": False})
        env_no_ball.reset(seed=0, options={"jitter": False})
        zeros = np.zeros(len(env_impact.action_names), dtype=float)
        for _ in range(8):
            env_impact.step(zeros)
            env_no_ball.step(zeros)
        snap_imp = env_impact.snapshot()
        snap_nb = env_no_ball.snapshot()
        assert env_impact.metrics()["human_object_contacts_total"] >= 1
        arm_vel_delta = abs(
            snap_imp["human"]["angular_velocities"]["right_elbow"]
            - snap_nb["human"]["angular_velocities"]["right_elbow"]
        ) + abs(
            snap_imp["human"]["angular_velocities"]["right_shoulder"]
            - snap_nb["human"]["angular_velocities"]["right_shoulder"]
        )
        assert arm_vel_delta > 0.05
    finally:
        env_impact.close()
        env_no_ball.close()


def test_feasibility_preflight_reset_jitter_and_target_switching():
    scene = authored_shared_world_scene()
    feasibility = audit_shared_world_feasibility(scene)
    assert feasibility["status"] == "feasible"
    assert feasibility["summary"]["physics_steps"] == 0
    assert feasibility["summary"]["initial_cross_entity_intersections"] == 0

    overlapping = deepcopy(scene)
    overlapping.pop("action_contract", None)
    # Place ball directly inside the static table box.
    overlapping["objects"][0]["position"] = list(overlapping["objects"][2]["position"])
    bad_feas = audit_shared_world_feasibility(overlapping)
    assert bad_feas["status"] == "infeasible"
    assert bad_feas["summary"]["initial_cross_entity_intersections"] >= 1
    env_bad = SharedWorldEnv(overlapping)
    try:
        with pytest.raises(ValueError, match="intersect"):
            env_bad.reset(seed=0)
    finally:
        env_bad.close()

    env = SharedWorldEnv(scene)
    try:
        with pytest.raises(ValueError, match="require jitter=True"):
            env.reset(seed=1, options={"jitter": False, "position_jitter": 0.02})

        for s in range(5):
            _, info = env.reset(
                seed=s,
                options={"jitter": True, "jitter_scale": 1.0, "position_jitter": 0.02, "velocity_jitter": 0.08},
            )
            assert info["reset_jitter"] is True
            assert info["initial_angular_offset_rmse_rad"] > 0.0
            assert set(info["object_perturbations"]) == {"ball", "marker_box"}

        # Dynamic target switching across human and dynamic rigid objects.
        env.select_target("ball")
        assert env.scene["controlled_id"] == "ball"
        assert next(o["role"] for o in env.scene["objects"] if o["id"] == "ball") == "active"
        assert next(o["role"] for o in env.scene["objects"] if o["id"] == "marker_box") == "passive"

        env.select_target("marker_box")
        assert env.scene["controlled_id"] == "marker_box"
        assert next(o["role"] for o in env.scene["objects"] if o["id"] == "ball") == "passive"
        assert next(o["role"] for o in env.scene["objects"] if o["id"] == "marker_box") == "active"

        with pytest.raises(ValueError, match="Static object"):
            env.select_target("table")
        with pytest.raises(ValueError, match="Unknown shared-world control target"):
            env.select_target("ghost")

        env.select_target("human")
        frame = render_snapshot(env.snapshot(), env.scene)
        assert frame.ndim == 3 and frame.shape[2] == 3
        ref_snap = env.reference_snapshot(0)
        assert ref_snap["human"]["reference"] is True
        ref_frame = render_snapshot(ref_snap, env.scene)
        assert ref_frame.shape == frame.shape
    finally:
        env.close()

    caps = describe_capabilities(scene)
    assert caps["representation_family"] == "mixed_rigid_articulated_3d"
    assert caps["entities"]["mixed_rigid_articulated"] is True
    assert [a["id"] for a in caps["entities"]["controllable_actors"]] == ["human", "ball", "marker_box"]
    assert caps["evidence_separation"]["physical_validation"] == "unverified"


def test_adapt_to_shared_world_rejects_source_without_observation_evidence(tmp_path):
    human_src = tmp_path / "mobile_human_run"
    human_src.mkdir()
    sha = "c" * 64
    human_spec = _analytic_mobile_human_spec()
    human_spec["source_sha256"] = sha
    write_json(human_src / "human.json", human_spec)
    write_json(human_src / "source.json", {"sha256": sha, "fps": 30.0, "frames": 65})
    write_json(
        human_src / "pose-quality.json",
        {"usable_frames": 61, "partial_frames": 2, "unusable_frames": 2, "status": "ok"},
    )
    manifest = {
        "schema_version": 1,
        "created_at": local_time(),
        "files": {name: digest(human_src / name) for name in ("human.json", "source.json", "pose-quality.json")},
    }
    write_json(human_src / "manifest.json", manifest)
    before_manifest_sha = digest(human_src / "manifest.json")

    adapted_dir = tmp_path / "adapted_shared_world"
    with pytest.raises(ValueError, match="[Oo]bservation|[Ee]vidence"):
        adapt_to_shared_world(human_src, adapted_dir, scale=1.1)
    assert digest(human_src / "manifest.json") == before_manifest_sha
    assert not adapted_dir.exists()


def test_shared_world_training_lab_control_recording_and_recovery(tmp_path):
    fixture_dir = tmp_path / "fixture_run"
    fixture_sealed = write_sealed_shared_world_fixture(fixture_dir)
    fixture_manifest_before = digest(fixture_dir / "manifest.json")
    assert fixture_sealed["manifest_sha256"] == fixture_manifest_before

    policy_dir = tmp_path / "policy_run"
    evaluation = train_shared_world_policy(
        fixture_dir,
        policy_dir,
        seed=42,
        generations=5,
        population=12,
        eval_episodes=6,
        max_seconds=90,
    )
    verify_artifacts(policy_dir)
    assert digest(fixture_dir / "manifest.json") == fixture_manifest_before
    assert evaluation["human_limits"] == SUCCESS_LIMITS
    assert evaluation["human_limits_modified"] is False
    assert evaluation["physical_validation"] == "unverified"

    changed = evaluation["regimes"]["changed_initial_conditions"]
    assert changed["learned"]["mean_return"] > changed["zero"]["mean_return"]
    assert changed["learned"]["mean_return"] > changed["random"]["mean_return"]
    assert changed["learned"]["human_object_contact_rate"] >= 0.8
    assert changed["learned"]["success_rate"] > changed["zero"]["success_rate"]

    pol_doc = json.loads((policy_dir / "policy.json").read_text())
    pol_scene = json.loads((policy_dir / "scene.json").read_text())
    validate_shared_policy(pol_doc, pol_scene)
    env_check = SharedWorldEnv(pol_scene)
    try:
        env_check.reset(seed=0)
        act = policy_action(pol_doc, env_check)
        assert act.shape == (len(env_check.action_names),)
        manual_vec = env_check.manual_action_vector(selected_motor="right_elbow", motor_nudge=0.5).tolist()
    finally:
        env_check.close()

    lab_dir = tmp_path / "shared_lab"
    packaged = prepare_shared_world_lab(fixture_dir, lab_dir, policy_run=policy_dir)
    assert packaged["learned_policy_available"] is True
    assert digest(fixture_dir / "manifest.json") == fixture_manifest_before
    run_manifest_before = digest(lab_dir / "run" / "manifest.json")

    lab = Lab(lab_dir)
    try:
        status = lab.summary()
        assert status["status"] == "ready"
        assert status["profile"] == "shared-world"
        assert status["capabilities"]["entities"]["mixed_rigid_articulated"] is True
        assert status["learned_policy_available"] is True
        assert status["evidence_provenance"]["physical_validation"] == "unverified"

        rec_start = lab.control("/api/record/start", {})
        assert rec_start["recording"] is True
        assert rec_start["snapshot"]["steps"] == 0

        with pytest.raises(ValueError, match="Stop and export the recording"):
            lab.control("/api/actor", {"actor": "ball"})

        s_ref = lab.control("/api/step", {"controller": "reference", "steps": 6})
        assert s_ref["snapshot"]["steps"] == 6

        s_pol = lab.control("/api/step", {"policy": True, "steps": 6})
        assert s_pol["snapshot"]["steps"] == 12

        s_man = lab.control("/api/step", {"action": manual_vec, "steps": 2})
        assert s_man["snapshot"]["steps"] == 14

        s_rst = lab.control("/api/reset", {"seed": 11})
        assert s_rst["snapshot"]["steps"] == 0

        s_pol2 = lab.control("/api/step", {"policy": True, "steps": 3})
        assert s_pol2["snapshot"]["steps"] == 3

        rec_stop = lab.control("/api/record/stop", {})
        assert rec_stop["recording"] is False
        exported = lab.export_future.result(timeout=15)
        session_dir = Path(exported["directory"])
        verified_rec = verify_recording(session_dir)
        assert verified_rec["steps"] == 17
        assert verified_rec["resets"] == 1
        assert (session_dir / "replay.html").is_file()
        assert (session_dir / "video.mp4").is_file()
        assert "shared-world-v1" in (session_dir / "replay.html").read_text(encoding="utf-8")

        switched = lab.control("/api/actor", {"actor": "ball"})
        assert switched["scene"]["controlled_id"] == "ball"
        assert switched["snapshot"]["controlled_id"] == "ball"

        y_before = next(o["position"][1] for o in switched["snapshot"]["objects"] if o["id"] == "ball")
        pushed = lab.control("/api/step", {"action": OBJECT_ACTIONS.index("forward"), "steps": 4})
        y_after = next(o["position"][1] for o in pushed["snapshot"]["objects"] if o["id"] == "ball")
        assert y_after > y_before + 0.005

        assert digest(lab_dir / "run" / "manifest.json") == run_manifest_before
    finally:
        lab.close()

    # Non-destructive crash recovery on an interrupted coupled recording with a truncated trailing JSON line.
    crashed_dir = tmp_path / "crashed_shared_recording"
    crashed_dir.mkdir()
    loaded_ok = load_recording(session_dir)
    write_json(crashed_dir / "scene.json", loaded_ok["scene"])
    write_json(
        crashed_dir / "recording.json",
        {**loaded_ok["metadata"], "status": "recording"},
    )
    raw_lines = (session_dir / "events.jsonl").read_bytes().splitlines(keepends=True)
    with (crashed_dir / "events.jsonl").open("wb") as stream:
        for line in raw_lines[:6]:
            stream.write(line)
        stream.write(b'{"index":6,"type":"step","truncated_crash":')

    crashed_events_sha = digest(crashed_dir / "events.jsonl")
    recovered_dir = tmp_path / "recovered_shared_recording"
    recovered = recover_incomplete_recording(crashed_dir, recovered_dir, export=True)
    assert recovered["events"] == 6
    assert recovered["recovery"]["dropped_trailing_partial_event"] is True
    assert digest(crashed_dir / "events.jsonl") == crashed_events_sha
    assert verify_recording(recovered_dir)["events"] == 6


def test_shared_world_lab_http_server_loopback(tmp_path):
    fixture_dir = tmp_path / "fixture_run"
    write_sealed_shared_world_fixture(fixture_dir)
    lab_dir = tmp_path / "http_lab"
    prepare_shared_world_lab(fixture_dir, lab_dir)

    lab = Lab(lab_dir)
    server = LabServer(("127.0.0.1", 0), lab)
    import threading

    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    port = server.server_port
    try:
        code, status = _http_json(port, lab.token, "GET", "/api/status")
        assert code == 200
        assert status["profile"] == "shared-world"
        assert status["capabilities"]["entities"]["mixed_rigid_articulated"] is True

        code, stepped = _http_json(port, lab.token, "POST", "/api/step", {"controller": "reference", "steps": 3})
        assert code == 200 and stepped["snapshot"]["steps"] == 3

        code, switched = _http_json(port, lab.token, "POST", "/api/actor", {"actor": "ball"})
        assert code == 200 and switched["snapshot"]["controlled_id"] == "ball"

        code, obj_step = _http_json(
            port,
            lab.token,
            "POST",
            "/api/step",
            {"action": OBJECT_ACTIONS.index("forward"), "steps": 2},
        )
        assert code == 200 and obj_step["snapshot"]["steps"] == 5
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        lab.close()
