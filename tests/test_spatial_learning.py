"""Bounded 3D rigid RL training, collision-free reset jitter, and lab integration checks."""

import json
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from weft.io import digest
from weft.lab import Lab
from weft.pipeline import verify_artifacts
from weft.spatial_lab import authored_spatial_scene, prepare_spatial_lab, write_sealed_spatial_fixture
from weft.spatial_learning import (
    POLICY_PROFILE,
    WEIGHT_DIM,
    policy_action,
    train_spatial_policy,
    validate_spatial_policy,
)
from weft.spatial_rigid import SpatialRigidEnv


def test_spatial_reset_jitter_is_opt_in_seeded_collision_free_and_bounded():
    scene = authored_spatial_scene()
    env = SpatialRigidEnv(scene)
    try:
        exact_obs, exact_info = env.reset(seed=19)
        assert exact_info["reset_jitter"] is False
        assert "perturbation" not in exact_info

        opts = {"jitter": True, "position_jitter": 0.04, "velocity_jitter": 0.2}
        obs_a, info_a = env.reset(seed=19, options=opts)
        state_a = env.snapshot()
        obs_a_repeat, info_a_repeat = env.reset(seed=19, options=opts)
        np.testing.assert_array_equal(obs_a, obs_a_repeat)
        assert info_a == info_a_repeat
        assert info_a["reset_jitter"] is True
        assert not np.allclose(obs_a, exact_obs)

        obs_b, _ = env.reset(seed=23, options=opts)
        assert not np.allclose(obs_a, obs_b)

        for i, obj in enumerate(scene["objects"]):
            for other in scene["objects"][i + 1 :]:
                if obj["role"] == other["role"] == "static":
                    continue
                dist = mujoco.mj_geomDistance(
                    env.model,
                    env.data,
                    env.model.geom(obj["id"]).id,
                    env.model.geom(other["id"]).id,
                    0.01,
                    None,
                )
                assert dist >= 1e-4
        assert state_a["physical_validation"] == "unverified"

        for bad_opts in (
            {"jitter": "yes"},
            {"jitter": False, "position_jitter": 0.02},
            {"jitter": True, "position_jitter": 0.2},
            {"jitter": True, "velocity_jitter": 1.5},
            {"jitter": True, "position_jitter": True},
            {"jitter": True, "unknown": 1},
        ):
            with pytest.raises(ValueError):
                env.reset(seed=1, options=bad_opts)
    finally:
        env.close()


def test_train_spatial_policy_seals_reproducible_bundle_and_drives_spatial_lab(tmp_path):
    source_dir = tmp_path / "spatial-source"
    write_sealed_spatial_fixture(source_dir)
    source_manifest_before = digest(source_dir / "manifest.json")

    learn_a = tmp_path / "learn-a"
    learn_b = tmp_path / "learn-b"
    eval_a = train_spatial_policy(
        source_dir,
        learn_a,
        seed=17,
        eval_seed=901,
        generations=4,
        population=10,
        eval_episodes=6,
    )
    eval_b = train_spatial_policy(
        source_dir,
        learn_b,
        seed=17,
        eval_seed=901,
        generations=4,
        population=10,
        eval_episodes=6,
    )
    assert digest(source_dir / "manifest.json") == source_manifest_before
    verify_artifacts(learn_a)
    assert json.loads((learn_a / "policy.json").read_text()) == json.loads((learn_b / "policy.json").read_text())
    assert eval_a["policies"] == eval_b["policies"]
    assert eval_a["physical_validation"] == "unverified"
    assert set(eval_a["policies"]) == {"learned", "random", "coast", "brake"}
    assert set(eval_a["regimes"]) == {"nominal_exact", "changed_initial_conditions"}
    assert eval_a["policies"]["learned"]["mean_final_distance_m"] < eval_a["policies"]["coast"]["mean_final_distance_m"]

    policy = json.loads((learn_a / "policy.json").read_text())
    assert policy["profile"] == POLICY_PROFILE
    assert len(policy["weights"]) == WEIGHT_DIM
    validate_spatial_policy(policy, json.loads((source_dir / "scene.json").read_text()))

    tampered = dict(policy, weights=[99.0] * WEIGHT_DIM)
    with pytest.raises(ValueError, match="policy weight"):
        validate_spatial_policy(tampered)

    lab_dir = tmp_path / "trained-lab"
    prepare_spatial_lab(source_dir, lab_dir, policy_run=learn_a)
    lab = Lab(lab_dir)
    try:
        summary = lab.summary()
        assert summary["learned_policy_available"] is True
        assert summary["capabilities"]["controllers"]["learned_policy_available"] is True
        assert summary["capabilities"]["controllers"]["trained_target_id"] == "block"

        step_res = lab.control("/api/step", {"steps": 6, "policy": True})
        assert len(step_res["transitions"]) == 6
        assert all(0 <= t["action"] <= 7 for t in step_res["transitions"])

        lab.control("/api/actor", {"actor": "companion"})
        with pytest.raises(ValueError, match="trained for block"):
            lab.control("/api/step", {"steps": 1, "policy": True})
    finally:
        lab.close()


def test_spatial_train_cli_and_timeout_preservation(tmp_path, monkeypatch):
    source_dir = tmp_path / "cli-source"
    write_sealed_spatial_fixture(source_dir)
    cli_out = tmp_path / "cli-learn"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "weft",
            "spatial-train",
            "--run",
            str(source_dir),
            "--out",
            str(cli_out),
            "--generations",
            "2",
            "--population",
            "6",
            "--eval-episodes",
            "4",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(completed.stdout)
    assert report["profile"] == POLICY_PROFILE
    verify_artifacts(cli_out)

    from weft import spatial_learning

    ticks = iter([0.0, 5.0, 5.0])
    monkeypatch.setattr(spatial_learning.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(TimeoutError):
        train_spatial_policy(
            source_dir,
            tmp_path / "timed-out",
            generations=2,
            population=4,
            eval_episodes=4,
            max_seconds=1.0,
        )
    saved_training = json.loads((tmp_path / "timed-out" / "training.json").read_text())
    assert saved_training["budget_exhausted"] is True
    assert not (tmp_path / "timed-out" / "manifest.json").exists()
