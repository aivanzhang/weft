"""Telemetry must account for every solve without perturbing simulation."""

from dataclasses import FrozenInstanceError

import mujoco
import numpy as np
import pytest

from weft.shared_world import SharedWorldEnv, prepare_shared_snapshot_validator
from weft.shared_world_lab import authored_shared_world_scene


def test_observer_preserves_exact_legacy_trajectory_and_detaches_samples():
    samples = []
    scene = authored_shared_world_scene()
    observed = SharedWorldEnv(scene, substep_observer=samples.append)
    legacy = SharedWorldEnv(scene)
    try:
        observed.reset(seed=7)
        legacy.reset(seed=7)
        assert not samples
        for _ in range(scene["task"]["horizon_steps"]):
            action = legacy.authored_action()
            actual, expected = observed.step(action), legacy.step(action)
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1:] == expected[1:]
            assert observed.snapshot() == legacy.snapshot()
            for field in ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "qfrc_applied", "xfrc_applied"):
                np.testing.assert_array_equal(getattr(observed.data, field), getattr(legacy.data, field))
        assert len(samples) == observed.steps * observed.substeps
        assert any(sample.contacts for sample in samples)
        for index, sample in enumerate(samples):
            assert sample.valid and sample.failure is None
            assert sample.step_index == index // observed.substeps + 1
            assert sample.substep_index == index % observed.substeps + 1
            assert sample.interval_start_s == pytest.approx(index * observed.dt / observed.substeps)
            assert sample.contact_time_s == sample.interval_start_s
            assert sample.interval_end_s == pytest.approx((index + 1) * observed.dt / observed.substeps)
        first = samples[0]
        before = repr(first)
        with pytest.raises(FrozenInstanceError):
            first.valid = False
        with pytest.raises(TypeError):
            first.wrist_positions["left"] = (0.0, 0.0, 0.0)
        with pytest.raises(TypeError):
            first.equipment[0].position[0] = 1.0
        observed.reset(seed=19)
        assert repr(first) == before
    finally:
        observed.close()
        legacy.close()


def test_integrated_equipment_and_solver_wrist_clocks_are_distinct():
    samples, differences = [], []
    scene = authored_shared_world_scene()
    box = next(obj for obj in scene["objects"] if obj["id"] == "marker_box")
    box["velocity"] = [0.2, 0.0, 0.0]
    box["quaternion"] = [0.9238795325112867, 0.0, 0.0, 0.3826834323650898]
    box["angular_velocity"] = [0.3, 0.1, 0.2]

    def inspect(sample):
        samples.append(sample)
        scratch = mujoco.MjData(env.model)
        mujoco.mj_copyData(scratch, env.model, env.data)
        mujoco.mj_forward(env.model, scratch)
        for obj in sample.equipment:
            body = env.model.body(obj.object_id).id
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(env.model, scratch, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            np.testing.assert_allclose(obj.position, scratch.xpos[body], atol=1e-14, rtol=0)
            np.testing.assert_allclose(obj.quaternion, scratch.xquat[body], atol=1e-14, rtol=0)
            np.testing.assert_allclose(obj.velocity, velocity[3:], atol=1e-14, rtol=0)
            np.testing.assert_allclose(obj.angular_velocity, velocity[:3], atol=1e-14, rtol=0)
        for side, point in sample.wrist_positions.items():
            site = env.model.site(f"{side}_wrist_site").id
            np.testing.assert_array_equal(point, env.data.site_xpos[site])
            differences.append(np.linalg.norm(np.array(point) - scratch.site_xpos[site]))
        np.testing.assert_array_equal(sample.human_qpos, env.data.qpos[env._q])
        np.testing.assert_array_equal(sample.human_qvel, env.data.qvel[env._v])
        np.testing.assert_array_equal(sample.target_qpos, env.target_state()[0])

    env = SharedWorldEnv(scene, substep_observer=inspect)
    try:
        env.reset(seed=0)
        env.step(env.authored_action())
        assert len(samples) == env.substeps
        assert max(differences) > 1e-7
    finally:
        env.close()


def test_contact_force_direction_matches_free_body_constraint_force():
    comparisons = []

    def inspect(sample):
        for obj in env._dynamic_objects:
            contacts = [contact for contact in sample.contacts if contact.object_id == obj["id"]]
            if not contacts:
                continue
            gid = env.model.geom(obj["id"]).id
            other_contacts = [con for con in env.data.contact
                              if gid in (con.geom1, con.geom2)
                              and not ({int(con.geom1), int(con.geom2)} & set(env._human_geom_to_name))]
            if other_contacts:
                continue
            joint = env.model.joint(f"{obj['id']}_joint").id
            vadr = int(env.model.jnt_dofadr[joint])
            on_human = np.sum([contact.force_on_human_world for contact in contacts], axis=0)
            np.testing.assert_allclose(-on_human, env.data.qfrc_constraint[vadr:vadr + 3], atol=1e-10, rtol=0)
            for contact in contacts:
                assert contact.human_segment in {"right_forearm", "right_upper_arm"}
                assert np.dot(contact.force_on_human_world, contact.normal_on_human_world) == pytest.approx(
                    contact.normal_force, abs=1e-10)
            comparisons.append(float(np.linalg.norm(on_human)))

    env = SharedWorldEnv(authored_shared_world_scene(), substep_observer=inspect)
    try:
        env.reset(seed=0)
        for _ in range(env.horizon_steps):
            env.step(env.authored_action())
        assert comparisons and max(comparisons) > 0.5
    finally:
        env.close()


def test_partial_failure_emits_last_finite_substep():
    samples = []
    scene = authored_shared_world_scene()
    box = next(obj for obj in scene["objects"] if obj["id"] == "marker_box")
    box["position"] = [scene["bounds"][0] - 0.036, 0.15, 1.7]
    box["velocity"] = [3.0, 0.0, 0.0]
    env = SharedWorldEnv(scene, substep_observer=samples.append)
    try:
        env.reset(seed=5)
        for _ in range(8):
            _, _, terminated, truncated, _ = env.step(env.authored_action())
            if terminated or truncated:
                break
        assert samples[-1].valid
        assert samples[-1].failure == "object_left_bounds"
        assert samples[-1].interval_end_s == env.data.time
        assert samples[-1].substep_index < env.substeps
        assert len(samples) == (env.steps - 1) * env.substeps + env._last_action_substeps
    finally:
        env.close()


def test_nonfinite_step_emits_invalid_marker_without_geometry(monkeypatch):
    samples = []
    env = SharedWorldEnv(authored_shared_world_scene(), substep_observer=samples.append)
    original_step = mujoco.mj_step

    def broken_step(model, data):
        original_step(model, data)
        if len(samples) == 2:
            data.qpos[0] = np.nan

    try:
        env.reset(seed=0)
        monkeypatch.setattr(mujoco, "mj_step", broken_step)
        with pytest.raises(RuntimeError, match="nonfinite_dynamics"):
            env.step(env.authored_action())
        assert len(samples) == 3
        assert samples[-1].valid is False
        assert samples[-1].failure == "nonfinite_dynamics"
        assert samples[-1].human_qpos == samples[-1].equipment == samples[-1].contacts == ()
        assert not samples[-1].wrist_positions
        assert samples[-1].max_human_floor_penetration_model_units is None
        assert samples[-1].interval_end_s == env.data.time
    finally:
        env.close()


def test_failed_observer_cannot_retry_an_advanced_episode():
    samples = []

    def reject(sample):
        samples.append(sample)
        raise ValueError("test observer failure")

    env = SharedWorldEnv(authored_shared_world_scene(), substep_observer=reject)
    try:
        env.reset(seed=0)
        with pytest.raises(RuntimeError, match="observer failed"):
            env.step(env.authored_action())
        assert env.steps == 1 and env._last_action_substeps == 1
        assert env.terminated and env._done
        assert env.data.time == samples[0].interval_end_s
        with pytest.raises(RuntimeError, match="Reset before stepping"):
            env.step(np.zeros(env.action_space.shape))
        env.reset(seed=0)
        assert env.data.time == 0
    finally:
        env.close()


def test_contact_limit_failure_still_accounts_for_the_completed_substep(monkeypatch):
    samples = []
    env = SharedWorldEnv(authored_shared_world_scene(), substep_observer=samples.append)
    original = env._current_contacts

    def limited():
        if len(samples) == 1:
            raise RuntimeError("test contact bound")
        return original()

    try:
        env.reset(seed=0)
        monkeypatch.setattr(env, "_current_contacts", limited)
        with pytest.raises(RuntimeError, match="contact observation failed"):
            env.step(env.authored_action())
        assert len(samples) == env._last_action_substeps == 2
        assert samples[-1].valid is False
        assert samples[-1].failure == "contact_observation_failed"
        assert samples[-1].contacts == ()
        assert samples[-1].interval_end_s == env.data.time
        assert env.steps == 1 and env.terminated
    finally:
        env.close()


def test_observer_requires_callable():
    with pytest.raises(TypeError, match="callable"):
        SharedWorldEnv(authored_shared_world_scene(), substep_observer=[])


def test_prepared_state_validation_is_pinned_but_not_cached():
    scene = authored_shared_world_scene()
    validate = prepare_shared_snapshot_validator(scene)
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0)
        state = env.snapshot()
        validate(state)
        scene["objects"][0]["mass"] += 1
        validate(state)
        state["objects"][0]["mass"] += 1
        with pytest.raises(ValueError, match="mass"):
            validate(state)
    finally:
        env.close()
