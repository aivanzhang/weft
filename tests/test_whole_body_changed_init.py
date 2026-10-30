"""Whole-body changed-initial-condition reset perturbations and matched-baseline evaluation checks."""

import json
import subprocess
import sys

import numpy as np
import pytest

from weft.human import build_human
from weft.io import digest
from weft.pipeline import verify_artifacts
from weft.whole_body import SUCCESS_LIMITS, WholeBodyEnv
from weft.whole_body_learning import evaluate_changed_conditions, train_whole_body
from test_mobile_integration import evidence


@pytest.fixture
def mobile_spec(tmp_path):
    return build_human(evidence(), tmp_path / "body", body_model="mobile")


def test_whole_body_reset_jitter_scale_and_explicit_offsets_preserve_default_and_joint_limits(mobile_spec):
    env = WholeBodyEnv(mobile_spec, horizon_steps=6)
    try:
        default_obs, default_info = env.reset(seed=42)
        default_q = env.data.qpos[env._q].copy()
        default_v = env.data.qvel[env._v].copy()

        scale1_obs, scale1_info = env.reset(seed=42, options={"jitter": True, "jitter_scale": 1.0})
        np.testing.assert_array_equal(default_obs, scale1_obs)
        np.testing.assert_array_equal(default_q, env.data.qpos[env._q])
        np.testing.assert_array_equal(default_v, env.data.qvel[env._v])
        assert default_info["jitter_scale"] == 1.0 and scale1_info["jitter_scale"] == 1.0

        exact_obs, exact_info = env.reset(seed=42, options={"jitter": False})
        zero_scale_obs, zero_info = env.reset(seed=42, options={"jitter": True, "jitter_scale": 0.0})
        np.testing.assert_array_equal(exact_obs, zero_scale_obs)
        assert exact_info["initial_angular_offset_rmse_rad"] == 0.0
        assert zero_info["initial_root_speed_rmse_model_units_s"] == 0.0

        _, info_3x = env.reset(seed=42, options={"jitter": True, "jitter_scale": 3.0})
        _, info_5x = env.reset(seed=42, options={"jitter": True, "jitter_scale": 5.0})
        assert info_5x["initial_angular_speed_rmse_rad_s"] > info_3x["initial_angular_speed_rmse_rad_s"] > default_info["initial_angular_speed_rmse_rad_s"]
        for idx, name in enumerate(env.joint_names):
            low, high = env.base.joint_limits[name]
            assert low <= env.data.qpos[env._q[idx]] <= high

        qpos_off = [0.05] * 19 + [0.01, -0.01, 0.02]
        qvel_off = [0.1] * 19 + [0.15, 0.0, 0.0]
        _, pert_info = env.reset(seed=7, options={"jitter": False, "qpos_offset": qpos_off, "qvel_offset": qvel_off})
        assert pert_info["initial_root_speed_rmse_model_units_s"] == pytest.approx(np.sqrt(0.15**2 / 3), rel=1e-6)
        assert pert_info["initial_root_offset_rmse_model_units"] == pytest.approx(np.sqrt((0.01**2 + 0.01**2 + 0.02**2) / 3), rel=1e-6)

        for invalid in (
            {"jitter": False, "jitter_scale": 1.0},
            {"jitter": True, "jitter_scale": -0.1},
            {"jitter": True, "jitter_scale": 5.1},
            {"jitter": True, "jitter_scale": True},
            {"jitter": False, "qpos_offset": [0.5] * 22},
            {"jitter": False, "qvel_offset": [0.0] * 21},
            {"jitter": False, "qvel_offset": [True] + [0.0] * 21},
        ):
            with pytest.raises(ValueError):
                env.reset(seed=1, options=invalid)
    finally:
        env.close()


def test_evaluate_changed_conditions_compares_matched_baselines_with_unmodified_gates(tmp_path, mobile_spec):
    learning_dir = tmp_path / "wb-learning"
    train_whole_body(mobile_spec, learning_dir, generations=2, population=4, horizon_steps=3, seed=11)
    manifest_before = digest(learning_dir / "manifest.json")

    eval_out = tmp_path / "wb-changed-eval"
    report = evaluate_changed_conditions(
        learning_dir,
        eval_out,
        seed=938401,
        jitter_scales=(0.0, 1.0, 3.0, 5.0),
        trials_per_window=2,
    )
    assert digest(learning_dir / "manifest.json") == manifest_before
    verify_artifacts(eval_out)
    assert report["success_limits"] == SUCCESS_LIMITS
    assert report["success_limits_modified"] is False
    assert report["physical_validation"] == "unverified"
    assert set(report["jitter_regimes"]) == {"scale_0.0x", "scale_1.0x", "scale_3.0x", "scale_5.0x"}
    assert set(report["explicit_perturbations"]) == {
        "lateral_root_push_plus_x",
        "lateral_root_push_minus_x",
        "joint_flexion_offset_0p08rad",
        "upward_root_offset_0p03",
    }
    for regime in (*report["jitter_regimes"].values(), *report["explicit_perturbations"].values()):
        assert set(regime["policies"]) == {"learned", "zero", "random", "authored_feedback"}
        seeds = [row["seed"] for row in regime["policies"]["learned"]["trials"]]
        for ctrl_summary in regime["policies"].values():
            assert [row["seed"] for row in ctrl_summary["trials"]] == seeds
            assert "floor_penetration_failures" in ctrl_summary

    cli_out = tmp_path / "wb-changed-cli"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "weft",
            "body-evaluate",
            "--learning",
            str(learning_dir),
            "--out",
            str(cli_out),
            "--trials-per-window",
            "2",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cli_report = json.loads(completed.stdout)
    assert cli_report["jitter_regimes"] == report["jitter_regimes"]
    verify_artifacts(cli_out)
