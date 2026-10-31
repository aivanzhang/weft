"""Capability/console integration with analytical samples; no native engine or media."""

from copy import deepcopy
import json
import os
from hashlib import sha256
from pathlib import Path

import pytest

from test_shared_exercise_feet_task import AnalyticalEngine, foot_contact, pure_orchestration_only, setup
from weft.capabilities import describe_capabilities, describe_run_capabilities
from weft.io import canonical_hash
from weft.lab_console import Controls, _brief
from weft.shared_exercise_feet_task import SharedExerciseFeetTaskEnv


UI_CASES = {}


@pytest.fixture(autouse=True)
def public_engine_snapshot_shape(monkeypatch):
    original_snapshot = AnalyticalEngine.snapshot

    def snapshot(engine):
        value = original_snapshot(engine)
        # Real engine snapshots carry their profile; hashes belong to the outer task snapshot.
        value["profile"] = engine.scene["profile"]
        value.pop("scene_profile")
        value.pop("scene_sha256")
        return value

    monkeypatch.setattr(AnalyticalEngine, "snapshot", snapshot)


@pytest.fixture(scope="session", autouse=True)
def save_requested_ui_cases():
    yield
    destination = os.environ.get("WEFT_FEET_UI_FIXTURES")
    if destination:
        assert len(UI_CASES) == 6
        Path(destination).write_text(json.dumps(list(UI_CASES.values()), allow_nan=False) + "\n")


@pytest.mark.parametrize("profile", ["shared-world-v2", "shared-world-v3", "shared-world-v4"])
@pytest.mark.parametrize("root", [False, True])
def test_feet_capabilities_and_console_preserve_base_score_and_explicit_controls(profile, root):
    base, scene, original, task = setup(profile, root=root)
    caps = describe_capabilities(scene, lab_profile="shared_exercise", exercise_task=task)
    env = SharedExerciseFeetTaskEnv(scene, task)
    try:
        env.reset(seed=1729, options={"jitter": False, "jitter_scale": 0.})
        initial = env.snapshot()
        if scene["objects"]:
            env._engine.contacts_at = lambda _: (foot_contact(object_id=scene["objects"][0]["id"]),)
        env.step([0.] * env.action_count)
        active = env.snapshot()
        env._engine.mode = "nonfinite"
        env.step([0.] * env.action_count)
        failed = env.snapshot()
        UI_CASES[(profile, root)] = {"scene": scene, "task": task, "capabilities": caps,
            "action_contract": env.action_contract(), "states": {"initial": initial, "active": active, "failed": failed}}
        contract = caps["action_contract"]
        assert contract["dimension"] == 25 + 3 * root
        assert contract["actuator_names"] == env.action_contract()["names"]
        assert contract["physical_units"] == env.action_contract()["units"]
        for name in ("engine_motor_indices", "joint_channels", "body_joint_channels", "ankle_channels", "root_assistance_channels", "object_force_channels"):
            assert contract[name] == env.action_contract()[name]
        assert caps["observation_contract"]["dimension"] == env.observation_size
        assert caps["entities"]["mixed_rigid_articulated"] is (profile != "shared-world-v3")
        assert caps["entities"]["multi_target_selection_endpoint"] is None
        assert [row["id"] for row in caps["entities"]["controllable_actors"]] == ["human"]
        base_caps = describe_capabilities(base, exercise_task=original)
        assert caps["task_contract"]["success_criteria"] == base_caps["task_contract"]["success_criteria"]
        identity = caps["task_contract"]["identity"]
        assert identity["scene_sha256"] == canonical_hash(scene)
        assert identity["base_scene_sha256"] == canonical_hash(base)
        assert identity["task_sha256"] == canonical_hash(task)
        assert identity["base_task_sha256"] == canonical_hash(original)
        assert caps["task_contract"]["diagnostics"]["foot_support_credit"] is False
        assert caps["controllers"]["learned_policy_available"] is False
        controls = Controls(caps)
        controls.select_motor(contract["actuator_names"][19], ".25")
        assert controls.step(1) == {"steps": 1, "action": [0.] * 19 + [.25] + [0.] * (env.action_count - 20)}
        controls.select_mode("coast")
        assert controls.step(1) == {"steps": 1, "controller": "coast"}
        controls.select_mode("reference")
        assert controls.step(1) == {"steps": 1, "controller": "reference"}
        with pytest.raises(ValueError, match="learned"):
            controls.select_mode("learned")
        summary = _brief(failed)
        assert summary["angular_rmse_rad"] == failed["task_metrics"]["angular_rmse_rad"]
        assert summary["unscored_ankle_diagnostics"]["latest_sample"]["interval_end_s"] is None
        assert summary["engine_state_status"] == "last_valid_control_boundary"
        assert summary["unscored_ankle_diagnostics"]["observed_substeps"] > active["task_metrics"]["observed_substeps"]
        assert "excluded_foot_contacts" not in summary["unscored_ankle_diagnostics"]["latest_sample"]
        assert _brief(initial)["unscored_ankle_diagnostics"]["ankle_hold_error_rmse_rad"] is None
    finally:
        env.close()


@pytest.mark.parametrize("flags", [{"exercise_task": None}, {"learned_policy_available": True}, {"trained_target_id": "human"}])
def test_feet_capabilities_reject_generic_or_unqualified_controls(flags):
    _, scene, _, task = setup()
    with pytest.raises(ValueError, match="explicit feet task"):
        describe_capabilities(scene, **{"exercise_task": task, **flags})


def test_feet_capabilities_revalidate_base_identity_instead_of_trusting_labels():
    _, scene, _, task = setup()
    changed = deepcopy(task)
    changed["base_task"]["control"]["root_assistance"] = "explicit_3_axis"
    with pytest.raises(ValueError):
        describe_capabilities(scene, exercise_task=changed)


@pytest.mark.parametrize("change", ["none", "unsealed_task", "oversized_task", "changed_bytes", "foreign_task"])
def test_raw_feet_candidate_inspection_requires_a_bounded_manifest_pinned_task(tmp_path, monkeypatch, change):
    _, scene, _, task = setup()
    if change == "foreign_task":
        task["base_task_sha256"] = "0" * 64
    files = {"scene.json": json.dumps(scene).encode(), "exercise-task.json": json.dumps(task).encode()}
    if change == "oversized_task":
        files["exercise-task.json"] += b" " * (128 * 1024 + 1 - len(files["exercise-task.json"]))
    for name, raw in files.items():
        (tmp_path / name).write_bytes(raw)
    members = {name: sha256(raw).hexdigest() for name, raw in files.items()
               if name != "exercise-task.json" or change != "unsealed_task"}
    (tmp_path / "manifest.json").write_text(json.dumps({"files": members}))
    if change == "changed_bytes":
        (tmp_path / "exercise-task.json").write_bytes(files["exercise-task.json"] + b" ")
    if change == "oversized_task":
        def unbounded_verification(_):
            pytest.fail("Oversized task must reject before the full artifact verifier reads it")

        monkeypatch.setattr("weft.capabilities.verify_artifacts", unbounded_verification)
    if change == "none":
        result = describe_run_capabilities(tmp_path)
        assert result["scene_profile"] == "shared-world-v5"
        assert result["lab_profile"] == "shared_exercise"
        assert result["task_contract"]["identity"]["task_sha256"] == canonical_hash(task)
    else:
        with pytest.raises(ValueError):
            describe_run_capabilities(tmp_path)
