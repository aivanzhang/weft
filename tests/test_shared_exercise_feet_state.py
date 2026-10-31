"""Analytical snapshot validation; native engine validation is injected only in tests."""

from copy import deepcopy
import json
import math
import socket
import subprocess
from time import perf_counter, process_time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from test_shared_exercise_feet_task import AnalyticalEngine, foot_contact, setup
from weft import shared_exercise_feet_state as states
from weft import shared_exercise_feet_task as feet
from weft import shared_exercise_task as original
from weft import shared_world
from weft.io import canonical_hash


class SnapshotEngine(AnalyticalEngine):
    """Supply minimal snapshot fields for orchestration, not native-geometry validation."""

    def snapshot(self):
        state = super().snapshot()
        state.update(profile="shared-world-v5", controlled_id="human", target=self.scene["target"],
                     normalized_action=list(self.actions[-1]) if self.steps else [0.] * len(self.action_names),
                     episode_steps=self.steps, terminated=False, truncated=False, engine_fixture_valid=True)
        state["human"]["reference"] = False
        if self.steps and not math.isclose(self.time, self.steps * self.scene["dt"], abs_tol=1e-10):
            count = round(self.time / (self.scene["dt"] / self.scene["substeps"])) - (self.steps - 1) * self.scene["substeps"]
            state["partial_action"] = {"completed_substeps": count, "requested_substeps": self.scene["substeps"]}
            state["terminated"] = True
        return state


@pytest.fixture(autouse=True)
def pure_validation_only(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Snapshot tests cannot compile models, run physics, use network or start workers")

    monkeypatch.setattr(mujoco, "MjModel", SimpleNamespace(
        from_xml_string=forbidden, from_xml_path=forbidden, from_binary_path=forbidden))
    for name in ("MjData", "Renderer", "mj_forward", "mj_step", "mj_resetData", "mj_contactForce", "mj_kinematics"):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(shared_world, "SharedWorldEnv", forbidden)
    monkeypatch.setattr(original, "SharedWorldEnv", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(shared_world, "human_subtree_mass", lambda _spec, scale=1.: 70. * scale**3)
    monkeypatch.setattr(feet, "SharedWorldEnv", SnapshotEngine)
    calls = {"prepare": [], "snapshots": 0}

    def prepare(scene):
        assert scene["profile"] == "shared-world-v5"
        pinned = canonical_hash(scene)
        calls["prepare"].append(pinned)

        def validate(state):
            calls["snapshots"] += 1
            if (state.get("profile") != "shared-world-v5" or state.get("scene_sha256") != pinned
                    or state.get("engine_fixture_valid") is not True):
                raise ValueError("Analytical actual-v5 engine validator rejected the snapshot")

        return validate

    monkeypatch.setattr(states, "prepare_shared_snapshot_validator", prepare)
    # The differential comparator gets an independent legacy engine stand-in, never relabeled v5 data.
    monkeypatch.setattr(original, "prepare_shared_snapshot_validator", lambda _scene: deepcopy)
    return calls


def opened(profile="shared-world-v3", *, root=False, objects=1):
    base, scene, base_task, task = setup(profile, root=root)
    if objects > 1:
        assert profile == "shared-world-v4"
        for target in (base, scene):
            prototype = target["objects"][0]
            binding = target["source_binding"]["entity_bindings"][1]
            target["objects"] = [{**deepcopy(prototype), "id": f"support_{i}",
                                  "position": [.3 + .6 * (i % 4), .3 + .6 * (i // 4), .2]} for i in range(objects)]
            target["source_binding"]["entity_bindings"] = [target["source_binding"]["entity_bindings"][0],
                *[{**deepcopy(binding), "world_entity_id": obj["id"]} for obj in target["objects"]]]
        requirement = {**base_task["support_requirements"][0], "object_id": "support_0"}
        base_task = original.build_shared_exercise_task(base, support_requirements=[requirement], root_assistance=root)
        task = feet.build_shared_exercise_feet_task(scene, base_task)
    env = feet.SharedExerciseFeetTaskEnv(scene, task)
    env.reset(seed=17)
    return base, scene, base_task, task, env


def legacy_snapshot(current, base, base_task):
    steps = current["engine_state"]["steps"]
    root = base_task["control"]["root_assistance"] == "explicit_3_axis"
    return {"schema_version": 1, "profile": original.STATE_PROFILE,
        "scene_sha256": canonical_hash(base), "task_sha256": canonical_hash(base_task),
        "engine_state": {"profile": base["profile"], "controlled_id": "human", "target": base["target"],
            "steps": steps, "time_s": current["task_metrics"]["simulated_duration_s"],
            "normalized_action": [0.] * len(base["action_contract"]["names"]), "objects": deepcopy(base["objects"])},
        "task_metrics": deepcopy(current["task_metrics"]), "last_action": [0.] * (22 if root else 19),
        "engine_state_status": "current", "resolved_reset_options": deepcopy(current["resolved_reset_options"]),
        "terminated": current["terminated"], "truncated": current["truncated"]}


@pytest.mark.parametrize("profile", ["shared-world-v2", "shared-world-v3", "shared-world-v4"])
def test_prepared_validator_preserves_initial_complete_and_one_shot_snapshots(profile, pure_validation_only):
    _, scene, _, task, env = opened(profile)
    validate = states.prepare_shared_exercise_feet_snapshot_validator(scene, task)
    first = env.snapshot()
    assert validate(first) == first
    for _ in range(2):
        env.step([0.] * 25)
        current = env.snapshot()
        assert validate(current) == current
    assert current["task_metrics"]["completed_interval"]
    assert pure_validation_only == {"prepare": [canonical_hash(scene)], "snapshots": 3}
    assert states.validate_shared_exercise_feet_snapshot(current, scene, task) == current
    detached = validate(current)
    detached["ankle_diagnostics"]["valid_substeps"] = 0
    assert current["ankle_diagnostics"]["valid_substeps"] == 16
    env.close()


@pytest.mark.parametrize("profile", ["shared-world-v2", "shared-world-v3", "shared-world-v4"])
def test_legacy_metric_validation_matches_existing_accept_reject_rules(profile):
    base, scene, base_task, task, env = opened(profile)
    old = original.prepare_shared_exercise_snapshot_validator(base, base_task)
    helper = (states._prepare_support_metrics_validator(base, base_task, env.evaluator_sha256)
              if profile == "shared-world-v4" else states._prepare_legacy_metrics_validator(base, base_task))
    for _ in range(2):
        env.step([0.] * 25)
    current = legacy_snapshot(env.snapshot(), base, base_task)
    assert old(current) == current and helper(current) is None
    changes = [lambda m: m.update(observed_substeps=True), lambda m: m.update(valid_substeps=17),
        lambda m: m.update(valid_observed_duration_s=0.), lambda m: m.update(gates_sha256="0" * 64),
        lambda m: m.update(gates_provenance="changed"), lambda m: m.update(angular_rmse_rad=.8),
        lambda m: m.update(joint_effort_mean_squared=1.1), lambda m: m.update(root_assistance_mean_squared=.1),
        lambda m: m.update(mean_reward=50.), lambda m: m.update(success=not m["success"]),
        lambda m: m.update(failed_gates=[] if m["failed_gates"] else ["invented"])]
    if profile == "shared-world-v4":
        changes += [lambda m: m["support_requirements"][0].update(observed_duration_s=0.),
                    lambda m: m["support_requirements"][0].update(qualifying_duration_s=.5),
                    lambda m: m["support_requirements"][0].update(peak_load_body_weight_fraction=None)]
    elif profile == "shared-world-v2":
        changes += [lambda m: m["equipment"][0].update(displacement_model_units=.5)]
    for mutate in changes:
        wrong = deepcopy(current)
        mutate(wrong["task_metrics"])
        for check in (old, helper):
            with pytest.raises(ValueError):
                check(wrong)
    env.close()


@pytest.mark.parametrize("field", ["scene_sha256", "base_scene_sha256", "base_task_sha256", "task_sha256",
                                  "projection_profile", "projection_contract_sha256", "foot_reference_sha256",
                                  "adapter_implementation_sha256"])
def test_every_outer_identity_is_pinned_before_engine_validation(field, pure_validation_only):
    _, scene, _, task, env = opened()
    validate = states.prepare_shared_exercise_feet_snapshot_validator(scene, task)
    snapshot = env.snapshot()
    snapshot[field] = "0" * 64
    with pytest.raises(ValueError, match="pinned"):
        validate(snapshot)
    assert pure_validation_only["snapshots"] == 0
    env.close()


def test_private_preparation_is_reused_and_actual_engine_validation_cannot_be_skipped(monkeypatch, pure_validation_only):
    _, scene, _, task, env = opened()
    validate = states.prepare_shared_exercise_feet_snapshot_validator(scene, task)
    snapshot = env.snapshot()
    scene["human_spec"]["root_reference"][0]["translation"][0] = 1.
    task["base_task"]["interval"]["horizon_steps"] = 1

    def forbidden(*_args, **_kwargs):
        pytest.fail("Per-frame validation repeated preparation, implementation reads or identity hashing")

    monkeypatch.setattr(feet, "prepare_shared_exercise_feet_task", forbidden)
    monkeypatch.setattr(feet, "adapter_implementation_identity", forbidden)
    monkeypatch.setattr(states, "prepare_shared_snapshot_validator", forbidden)
    monkeypatch.setattr(states, "canonical_hash", forbidden)
    monkeypatch.setattr(states.Path, "read_bytes", forbidden)
    assert validate(snapshot) == snapshot
    wrong = deepcopy(snapshot)
    wrong["engine_state"]["engine_fixture_valid"] = False
    with pytest.raises(ValueError, match="actual-v5"):
        validate(wrong)
    assert len(pure_validation_only["prepare"]) == 1 and pure_validation_only["snapshots"] == 2
    env.close()


@pytest.mark.parametrize("root", [False, True])
def test_public_action_mapping_and_original_evaluator_provenance_remain_strict(root):
    _, scene, _, task, env = opened("shared-world-v2", root=root)
    validate = states.prepare_shared_exercise_feet_snapshot_validator(scene, task)
    action = [.1] * 19 + [.2] * 6 + ([.3] * 3 if root else [])
    env.step(action)
    snapshot = env.snapshot()
    assert validate(snapshot) == snapshot
    mutations = [lambda s: s["last_action"].__setitem__(19, True),
        lambda s: s["engine_state"]["normalized_action"].__setitem__(28, .1),
        lambda s: s["engine_state"]["normalized_action"].__setitem__(22, .4),
        lambda s: s["task_metrics"].update(evaluator_sha256="0" * 64),
        lambda s: s["task_metrics"]["source_scope"].update(grasp_supported=0),
        lambda s: s["engine_state"]["human"].update(reference=True)]
    if not root:
        mutations.append(lambda s: s["engine_state"]["normalized_action"].__setitem__(19, .1))
    for mutate in mutations:
        wrong = deepcopy(snapshot)
        mutate(wrong)
        with pytest.raises(ValueError):
            validate(wrong)
    env.close()


@pytest.mark.parametrize("mode", ["finite_failure", "nonfinite"])
def test_partial_and_stale_failure_clocks_keep_real_observed_coverage(mode):
    _, scene, _, task, env = opened("shared-world-v4")
    validate = states.prepare_shared_exercise_feet_snapshot_validator(scene, task)
    env._engine.mode = mode
    env._engine.contacts_at = lambda _: (foot_contact(),)
    env.step([0.] * 25)
    snapshot = env.snapshot()
    assert validate(snapshot) == snapshot
    assert snapshot["terminated"] and not snapshot["truncated"]
    assert snapshot["ankle_diagnostics"]["observed_substeps"] == (2 if mode == "finite_failure" else 4)
    assert snapshot["ankle_diagnostics"]["valid_substeps"] == (2 if mode == "finite_failure" else 3)
    if mode == "nonfinite":
        assert snapshot["engine_state_status"] == "last_valid_control_boundary"
        latest = snapshot["ankle_diagnostics"]["latest_substep"]
        assert latest["interval_end_s"] is None and latest["valid"] is False
        raw_rejected = deepcopy(snapshot)
        raw_rejected["ankle_diagnostics"]["latest_substep"].update(interval_start_s=-.01, contact_time_s=-.01)
        assert validate(raw_rejected) == raw_rejected
    wrong = deepcopy(snapshot)
    wrong["engine_state_status"] = "current" if mode == "nonfinite" else "last_valid_control_boundary"
    with pytest.raises(ValueError):
        validate(wrong)
    env.close()


def test_diagnostic_claims_coverage_and_latest_aggregate_algebra_cannot_be_forged():
    _, scene, _, task, env = opened("shared-world-v4")
    validate = states.prepare_shared_exercise_feet_snapshot_validator(scene, task)
    env._engine.contacts_at = lambda _: (foot_contact(),)
    env.step([0.] * 25)
    snapshot = env.snapshot()
    assert validate(snapshot) == snapshot
    changes = [lambda d: d.update(scored=True), lambda d: d.update(foot_support_credit=0),
        lambda d: d.update(source="recorded_physical_observation"),
        lambda d: d.update(retention="complete_force_trace"), lambda d: d.update(ground_load_requirements="qualified"),
        lambda d: d.update(observed_substeps=7), lambda d: d.update(valid_observed_duration_s=0.),
        lambda d: d.update(ankle_effort_mean_squared=None), lambda d: d.update(ankle_hold_error_rmse_rad=1.),
        lambda d: d.update(excluded_foot_contact_records=7),
        lambda d: d["contact_aggregates"][1].update(peak_normal_force=0.),
        lambda d: d["contact_aggregates"][0].update(last_contact_time_s=0.),
        lambda d: d["contact_aggregates"][0].update(peak_normal_force=99.),
        lambda d: d["latest_substep"].update(contact_time_s=0.),
        lambda d: d["latest_substep"]["excluded_foot_contacts"][0].update(human_geom="unknown_alias"),
        lambda d: d["latest_substep"]["excluded_foot_contacts"][0].update(human_segment="left_forearm"),
        lambda d: d["latest_substep"]["excluded_foot_contacts"][0].update(force_on_human_world=[0., True, 1.])]
    for mutate in changes:
        wrong = deepcopy(snapshot)
        mutate(wrong["ankle_diagnostics"])
        with pytest.raises(ValueError):
            validate(wrong)
    env.close()


def test_finite_shape_and_utf8_byte_bounds_reject_before_native_validation(pure_validation_only):
    _, scene, _, task, env = opened()
    validate = states.prepare_shared_exercise_feet_snapshot_validator(scene, task)
    snapshot = env.snapshot()
    for value in (float("nan"), "☃" * (states.MAX_SNAPSHOT_BYTES // 2), [0] * (states.MAX_SNAPSHOT_NODES + 1)):
        wrong = deepcopy(snapshot)
        wrong["extra"] = value
        with pytest.raises(ValueError):
            validate(wrong)
    wrong = deepcopy(snapshot)
    wrong["extra"] = wrong
    with pytest.raises(ValueError, match="bounded JSON structure"):
        validate(wrong)
    assert pure_validation_only["snapshots"] == 0
    env.close()


def test_repeated_maximum_latest_contacts_and_scene_pairs_stay_bounded(pure_validation_only):
    _, scene, _, task, env = opened("shared-world-v4", objects=12)
    validate = states.prepare_shared_exercise_feet_snapshot_validator(scene, task)
    contacts = tuple(foot_contact(object_id=f"support_{i % 12}") for i in range(256))
    env._engine.contacts_at = lambda _: contacts
    env.step([0.] * 25)
    snapshot = env.snapshot()
    encoded_bytes = len(json.dumps(snapshot, separators=(",", ":"), allow_nan=False).encode())
    assert encoded_bytes < states.MAX_SNAPSHOT_BYTES
    assert len(snapshot["ankle_diagnostics"]["contact_aggregates"]) == 24
    assert len(snapshot["ankle_diagnostics"]["latest_substep"]["excluded_foot_contacts"]) == 256
    assert snapshot["ankle_diagnostics"]["excluded_foot_contact_records"] == 8 * 256
    start, cpu_start = perf_counter(), process_time()
    for _ in range(32):
        assert validate(snapshot) == snapshot
    elapsed, cpu = perf_counter() - start, process_time() - cpu_start
    assert len(pure_validation_only["prepare"]) == 1 and pure_validation_only["snapshots"] == 32
    print(json.dumps({"feet_state_measurement": {"iterations": 32, "latest_contacts": 256, "aggregate_pairs": 24,
          "snapshot_bytes": encoded_bytes, "wall_seconds": elapsed, "cpu_seconds": cpu,
          "native_engine_validation": "injected_analytical_guard_only"}}))
    env.close()
