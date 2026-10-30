"""Independent exercise gates use all substeps, not legacy terminal parking/contact scores."""

from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from weft.io import canonical_hash
from weft.shared_exercise_task import (
    GATES, GATES_SHA256, SharedExerciseTaskEnv, _ExerciseMetrics, build_shared_exercise_task,
    prepare_shared_exercise_snapshot_validator, validate_shared_exercise_snapshot, validate_shared_exercise_task,
)
from weft.shared_world import validate_shared_world
from weft.shared_world_observation import EquipmentState, HumanObjectContact, SubstepSample
from test_shared_world_equipment_contract import proxy_scene
from test_shared_world_reconstruction import fixture as body_fixture


def setup(*, horizon=2, root=False, contact=None, scale=1.):
    scene = proxy_scene()
    scene.pop("action_contract")
    scene["task"]["horizon_steps"] = horizon
    # Longer gap tests need an explicitly extended analytic reference, never end holding.
    while scene["human_spec"]["reference"][-1]["time_s"] < horizon / 30 - 1e-9:
        stamp = scene["human_spec"]["reference"][-1]["time_s"] + 1 / 30
        for key in ("reference", "root_reference"):
            row = deepcopy(scene["human_spec"][key][-1])
            row["time_s"] = stamp
            scene["human_spec"][key].append(row)
    scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"] = scale
    for obj in scene["objects"]:
        obj["position"][2] = obj["size"][2] / 2
    scene["target"] = list(scene["objects"][0]["position"])
    scene = validate_shared_world(scene)
    requirements = [] if contact is None else [{"object_id": "working_bell", "wrist": "left", "kind": contact,
                                               "start_step": 0, "end_step": horizon}]
    task = build_shared_exercise_task(scene, root_assistance=root, contact_requirements=requirements)
    equipment = tuple(EquipmentState(obj["id"], tuple(obj["position"]), tuple(obj["quaternion"]),
                                     (0.,) * 3, (0.,) * 3, (0.,) * 3) for obj in scene["objects"])
    accumulator = _ExerciseMetrics(scene, task, equipment, 0., 1.)
    return scene, task, accumulator


def sample(acc, index, *, contacts=(), equipment=None, **changes):
    tick = acc.dt / acc.substeps
    zero = (0.,) * 22
    record = SubstepSample(index // acc.substeps + 1, index % acc.substeps + 1,
        index * tick, (index + 1) * tick, index * tick, True, None,
        zero, zero, zero, zero, tuple(acc.initial.values()) if equipment is None else equipment, tuple(contacts),
        MappingProxyType({"left": (0., 0., 0.), "right": (1., 0., 0.)}), 0.)
    return replace(record, **changes)


def wrist_contact(**changes):
    return replace(HumanObjectContact("left_forearm_geom", "left_forearm", "working_bell", (0., 0., 0.),
                                     0., 1., (0., 0., 1.), (0., 0., 1.)), **changes)


def finish(acc, mutate=None):
    for index in range(acc.horizon * acc.substeps):
        record = sample(acc, index, contacts=(wrist_contact(),))
        acc.consume(mutate(record, index) if mutate else record)
    return acc.metrics()


def test_task_pins_scene_gates_interval_and_returns_detached_contracts():
    scene, task, _ = setup()
    before = canonical_hash(scene)
    assert task["gates_sha256"] == GATES_SHA256
    assert GATES["angular_rmse_rad"] == .35 and GATES["root_rmse_model_units"] == .15
    assert validate_shared_exercise_task(task, scene) == task
    assert canonical_hash(scene) == before
    changed = deepcopy(task)
    changed["interval"]["horizon_steps"] -= 1
    with pytest.raises(ValueError, match="interval"):
        validate_shared_exercise_task(changed, scene)
    changed = deepcopy(task)
    changed["gates_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen gate"):
        validate_shared_exercise_task(changed, scene)
    with pytest.raises(TypeError):
        GATES["equipment_displacement_model_units"] = 1


@pytest.mark.parametrize("change", [
    {"kind": "grasp"}, {"wrist": "hand"}, {"object_id": "invented"}, {"start_step": True}, {"end_step": 3},
])
def test_unsupported_grasp_or_invalid_contact_contract_rejects(change):
    scene, task, _ = setup(contact="wrist_region_contact")
    task["contact_requirements"][0].update(change)
    with pytest.raises(ValueError):
        validate_shared_exercise_task(task, scene)


def test_every_dynamic_equipment_object_remains_in_stability_gates():
    scene, task, _ = setup()
    task["equipment_ids"].pop()
    with pytest.raises(ValueError, match="every unique"):
        validate_shared_exercise_task(task, scene)


def test_source_frame_two_to_eight_gap_is_not_trimmed_or_filled():
    scene, _, _ = body_fixture()
    for obj in scene["objects"]:
        if obj["role"] != "static":
            obj["provenance"]["source_prefix"] = {"body_start_frame": 2, "equipment_observation_frame": 8,
                                                     "static_extrapolation_assumed": True, "source_slots_filled": 0}
    task = build_shared_exercise_task(scene)
    assert task["interval"]["start_offset_s"] == 0
    for row in task["source_scope"]["equipment_initialization"]:
        assert row["body_start_frame"] == 2 and row["equipment_observation_frame"] == 8
        assert row["unobserved_prefix_frames"] == 6 and row["initial_state_scope"] == "authored_static_extrapolation"
    assert task["source_scope"]["source_contact_verified"] is False


@pytest.mark.parametrize("kind", ["wrist_region_contact", "upward_support"])
def test_complete_static_proxy_and_designated_region_force_can_pass(kind):
    _, _, acc = setup(contact=kind)
    result = finish(acc)
    assert result["success"] and result["completed_interval"]
    assert result["contact_requirements"][0]["qualifying_fraction"] == pytest.approx(1)
    assert result["source_scope"]["grasp_supported"] is False


@pytest.mark.parametrize("change", [
    {"human_segment": "torso"}, {"human_segment": "right_forearm"}, {"object_id": "idle_bell"},
    {"position": (.061, 0., 0.)}, {"distance": .002}, {"normal_force": 0.},
])
def test_other_body_contact_proximity_or_wrong_equipment_does_not_pass(change):
    _, _, acc = setup(contact="wrist_region_contact")
    result = finish(acc, lambda row, _: replace(row, contacts=(wrist_contact(**change),)))
    assert not result["success"] and not result["contact_requirements"][0]["satisfied"]


def test_contact_only_at_control_boundaries_cannot_hide_substep_gaps():
    _, _, acc = setup(contact="wrist_region_contact")
    result = finish(acc, lambda row, i: replace(row, contacts=row.contacts if (i + 1) % acc.substeps == 0 else ()))
    assert result["contact_requirements"][0]["qualifying_fraction"] == pytest.approx(1 / acc.substeps)
    assert not result["success"]


def test_long_contact_gap_fails_even_when_total_coverage_passes():
    _, _, acc = setup(horizon=90, contact="upward_support")
    result = finish(acc, lambda row, i: replace(row, contacts=() if 100 <= i < 155 else row.contacts))
    contact = result["contact_requirements"][0]
    assert contact["qualifying_fraction"] > .95
    assert contact["max_gap_s"] > .1 and not result["success"]


def test_upward_support_uses_declared_gravity_not_unsigned_contact_force():
    scene, task, initial = setup(contact="upward_support")
    scene["gravity"] = [-9.81, 0., 0.]
    acc = _ExerciseMetrics(scene, task, initial.initial.values(), 0., 1.)
    assert not finish(acc)["success"]
    acc = _ExerciseMetrics(scene, task, initial.initial.values(), 0., 1.)
    assert finish(acc, lambda row, _: replace(row, contacts=(wrist_contact(force_on_human_world=(1., 0., 0.)),)))["success"]
    scene["gravity"] = [0., 0., 0.]
    with pytest.raises(ValueError, match="nonzero"):
        _ExerciseMetrics(scene, task, initial.initial.values(), 0., 1.)


@pytest.mark.parametrize("scale", [.5, 1.5])
def test_region_and_displacement_thresholds_use_model_not_world_units(scale):
    _, _, acc = setup(contact="wrist_region_contact", scale=scale)
    assert finish(acc, lambda row, _: replace(row, contacts=(wrist_contact(position=(.059 * scale, 0., 0.)),)))["success"]
    _, _, acc = setup(scale=scale)
    def transient(row, index):
        if index:
            return row
        item, *others = row.equipment
        position = (item.position[0] + .031 * scale, *item.position[1:])
        return replace(row, equipment=(replace(item, position=position), *others))
    result = finish(acc, transient)
    assert not result["success"] and "equipment_stability:working_bell" in result["failed_gates"]
    assert acc.latest["working_bell"].position == acc.initial["working_bell"].position


@pytest.mark.parametrize("field,value", [("velocity", (.151, 0., 0.)), ("angular_velocity", (0., 0., .751)),
                                        ("quaternion", (.99, 0., 0., np.sqrt(1 - .99 ** 2)))])
def test_transient_equipment_motion_cannot_be_repaired_by_terminal_parking(field, value):
    _, _, acc = setup()
    def transient(row, index):
        item, *others = row.equipment
        return replace(row, equipment=(replace(item, **{field: value}), *others)) if index == 1 else row
    assert not finish(acc, transient)["success"]
    assert acc.latest["working_bell"] == acc.initial["working_bell"]


def test_virtual_object_forcing_and_disabled_root_assistance_fail_independently():
    _, _, acc = setup()
    def forced(row, _):
        item, *others = row.equipment
        return replace(row, equipment=(replace(item, virtual_force=(1., 0., 0.)), *others))
    assert finish(acc, forced)["failure"] == "forbidden_object_actuation"
    _, _, acc = setup()
    assert finish(acc, lambda row, _: replace(row, human_action=(0.,) * 19 + (1.,) * 3))["failure"] == "forbidden_human_actuation"
    _, _, acc = setup(root=True)
    result = finish(acc, lambda row, _: replace(row, human_action=(0.,) * 19 + (1.,) * 3))
    assert result["success"] and result["root_assistance_mean_squared"] == pytest.approx(1.)
    assert result["root_assistance"] == "explicit_3_axis"


@pytest.mark.parametrize("gate", ["peak_angular_rmse_rad", "peak_root_rmse_model_units", "max_floor_penetration_model_units"])
def test_transient_human_tracking_or_floor_failure_cannot_disappear_at_end(gate):
    _, _, acc = setup()
    def transient(row, index):
        if index != 1:
            return row
        if gate == "max_floor_penetration_model_units":
            return replace(row, max_human_floor_penetration_model_units=.026)
        q = [.71] * 19 + [0.] * 3 if gate == "peak_angular_rmse_rad" else [0.] * 19 + [.31] * 3
        return replace(row, human_qpos=tuple(q))
    result = finish(acc, transient)
    assert gate in result["failed_gates"] and not result["success"]


def test_only_explicit_contact_interval_contributes_to_contact_coverage():
    scene, task, initial = setup(contact="upward_support")
    task["contact_requirements"][0]["start_step"] = 1
    task = validate_shared_exercise_task(task, scene)
    acc = _ExerciseMetrics(scene, task, initial.initial.values(), 0., 1.)
    result = finish(acc, lambda row, _: replace(row, contacts=() if row.step_index == 1 else row.contacts))
    assert result["success"]
    assert result["contact_requirements"][0]["observed_duration_s"] == pytest.approx(scene["dt"])


def test_invalid_partial_substep_keeps_actual_observed_clock_and_cannot_complete():
    _, _, acc = setup()
    acc.consume(sample(acc, 0))
    acc.consume(sample(acc, 1, valid=False, failure="nonfinite_dynamics", interval_end_s=None,
                       human_qpos=(), human_qvel=(), target_qpos=(), equipment=(), contacts=(),
                       wrist_positions=MappingProxyType({}), max_human_floor_penetration_model_units=None))
    metrics = acc.metrics()
    assert metrics["observed_substeps"] == 2 and metrics["valid_substeps"] == 1
    assert metrics["simulated_duration_s"] == pytest.approx(acc.dt / acc.substeps)
    assert metrics["failure"] == "nonfinite_dynamics" and not metrics["completed_interval"]


def test_missing_or_duplicate_substep_observation_fails_closed():
    _, _, acc = setup()
    assert finish(acc, lambda row, index: replace(row, substep_index=2) if index == 0 else row)["failure"]


def test_real_engine_human_only_end_to_end_seeded_replay_and_detached_snapshots():
    scene, task, _ = setup(root=True)
    before = canonical_hash(scene)
    first, replay = SharedExerciseTaskEnv(scene, task), SharedExerciseTaskEnv(scene, task)
    try:
        a, info = first.reset(seed=17)
        b, _ = replay.reset(seed=17)
        np.testing.assert_array_equal(a, b)
        assert first.action_space.shape == (22,) and first.action_contract()["root_assistance_channels"] == [19, 20, 21]
        assert first.action_contract()["object_force_channels"] == []
        validate_shared_exercise_snapshot(first.snapshot(), scene, task)
        for _ in range(scene["task"]["horizon_steps"]):
            action = first.authored_action()
            a = first.step(action)
            b = replay.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:] == b[1:]
            assert first.snapshot() == replay.snapshot()
            checked = validate_shared_exercise_snapshot(first.snapshot(), scene, task)
            assert all(value == 0 for value in checked["engine_state"]["normalized_action"][22:])
        metrics = first.metrics()
        assert metrics["observed_substeps"] == scene["task"]["horizon_steps"] * scene["substeps"]
        assert a[2] == metrics["success"] and a[3] == (not metrics["success"])
        changed = first.snapshot()
        changed["task_metrics"]["equipment"][0]["within_limits"] = "changed"
        assert first.snapshot()["task_metrics"]["equipment"][0]["within_limits"] is not changed["task_metrics"]["equipment"][0]["within_limits"]
        assert canonical_hash(scene) == before
    finally:
        first.close(); replay.close()


@pytest.mark.parametrize("mutation", ["success", "duration", "gate", "actuation", "stale_status"])
def test_snapshot_rejects_fabricated_success_clock_gates_and_actuation(mutation):
    scene, task, _ = setup()
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=2)
        state = env.snapshot()
        if mutation == "success":
            state["task_metrics"]["success"] = True
        elif mutation == "duration":
            state["task_metrics"]["simulated_duration_s"] = 1
        elif mutation == "gate":
            state["task_metrics"]["gates_sha256"] = "0" * 64
        elif mutation == "actuation":
            state["last_action"] += [1.]
        else:
            state["engine_state_status"] = "last_valid_control_boundary"
        with pytest.raises(ValueError):
            validate_shared_exercise_snapshot(state, scene, task)
    finally:
        env.close()


def test_failed_engine_step_is_explicitly_stale_and_reset_options_are_replayable(monkeypatch):
    scene, task, _ = setup()
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=7, options={"jitter": True, "jitter_scale": .2})
        initial = env.snapshot()
        assert initial["resolved_reset_options"] == {"jitter": True, "jitter_scale": .2}
        assert initial["task_metrics"]["reset_protocol"]["object_velocity_jitter_world_s"] == .12
        def fail(_):
            raise RuntimeError("injected finite observation failure")
        monkeypatch.setattr(env._engine, "step", fail)
        _, reward, terminated, truncated, info = env.step(np.zeros(19))
        assert reward == -1 and terminated and not truncated
        state = env.snapshot()
        assert state["engine_state"] == initial["engine_state"]
        assert state["engine_state_status"] == "last_valid_control_boundary"
        assert info["simulated_duration_s"] == 0
        validate_shared_exercise_snapshot(state, scene, task)
        with pytest.raises(ValueError, match="only explicit"):
            env.reset(options={"position_jitter": 1})
    finally:
        env.close()


def test_exception_after_three_real_substeps_retains_actual_clock_and_stale_geometry():
    scene, task, _ = setup()
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=5)
        initial = env.snapshot()["engine_state"]
        original = env._engine._substep_observer
        def fail_after_observation(row):
            original(row)
            if row.substep_index == 3:
                raise RuntimeError("injected observation failure after completed physics")
        env._engine._substep_observer = fail_after_observation
        _, _, terminated, truncated, info = env.step(np.zeros(19))
        assert terminated and not truncated and info["observed_substeps"] == 3
        assert info["simulated_duration_s"] == pytest.approx(3 * scene["dt"] / scene["substeps"])
        state = env.snapshot()
        assert state["engine_state"] == initial and state["engine_state_status"] == "last_valid_control_boundary"
        validate_shared_exercise_snapshot(state, scene, task)
        with pytest.raises(RuntimeError, match="Reset"):
            env.step(np.zeros(19))
    finally:
        env.close()


def test_finite_engine_failure_preserves_current_partial_state_and_clock():
    scene, _, _ = setup(horizon=12)
    scene["objects"][0]["position"][0] = scene["bounds"][0] - .061
    scene["objects"][0]["velocity"] = [3., 0., 0.]
    scene["target"] = list(scene["objects"][0]["position"])
    task = build_shared_exercise_task(scene)
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=5)
        for _ in range(12):
            _, _, terminated, truncated, _ = env.step(np.zeros(19))
            if terminated or truncated:
                break
        state = env.snapshot()
        assert state["task_metrics"]["failure"] == "object_left_bounds"
        assert state["engine_state_status"] == "current" and state["terminated"] and not state["truncated"]
        assert state["engine_state"]["time_s"] == pytest.approx(state["task_metrics"]["simulated_duration_s"])
        assert state["engine_state"]["partial_action"]["completed_substeps"] < scene["substeps"]
        validate_shared_exercise_snapshot(state, scene, task)
    finally:
        env.close()


def test_prepared_validator_pins_private_inputs_without_rehashing_each_frame(monkeypatch):
    from weft import shared_exercise_task as module

    scene, task, _ = setup(contact="upward_support")
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=9)
        snapshot = env.snapshot()
        validate = prepare_shared_exercise_snapshot_validator(scene, task)
        scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"] = 2.
        task["source_scope"]["grasp_supported"] = True
        task["contact_requirements"][0]["kind"] = "grasp"
        with monkeypatch.context() as context:
            def unexpected(*_):
                raise AssertionError("Immutable scene/task preparation repeated during snapshot validation")
            for name in ("validate_shared_world", "validate_shared_exercise_task", "canonical_hash"):
                context.setattr(module, name, unexpected)
            checked = validate(snapshot)
            assert checked == snapshot and checked is not snapshot
            checked["task_metrics"]["source_scope"]["grasp_supported"] = True
            assert validate(snapshot) == snapshot
        with pytest.raises(ValueError):
            validate_shared_exercise_snapshot(snapshot, scene, task)
    finally:
        env.close()


@pytest.mark.parametrize("mutation", ["source_scope", "gate", "contact_kind", "clock", "geometry", "root_action", "nonfinite"])
def test_prepared_validator_retains_all_dynamic_tamper_checks(mutation):
    scene, task, _ = setup(contact="upward_support")
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=9)
        snapshot = env.snapshot()
        validate = prepare_shared_exercise_snapshot_validator(scene, task)
        assert validate(snapshot) == validate_shared_exercise_snapshot(snapshot, scene, task)
        changed = deepcopy(snapshot)
        if mutation == "source_scope":
            changed["task_metrics"]["source_scope"]["grasp_supported"] = True
        elif mutation == "gate":
            changed["task_metrics"]["gates_sha256"] = "0" * 64
        elif mutation == "contact_kind":
            changed["task_metrics"]["contact_requirements"][0]["kind"] = "grasp"
        elif mutation == "clock":
            changed["task_metrics"]["simulated_duration_s"] = scene["dt"]
        elif mutation == "geometry":
            changed["engine_state"]["objects"][0]["position"][0] = 100.
        elif mutation == "root_action":
            changed["engine_state"]["normalized_action"][19] = .5
        else:
            changed["task_metrics"]["peak_root_rmse_model_units"] = float("nan")
        for check in (validate, lambda value: validate_shared_exercise_snapshot(value, scene, task)):
            with pytest.raises(ValueError):
                check(changed)
    finally:
        env.close()


def test_render_uses_cached_task_renderer_and_detached_failure_geometry(monkeypatch):
    from weft import shared_exercise_recording as recording

    scene, task, _ = setup()
    env = SharedExerciseTaskEnv(scene, task, render_mode="rgb_array")
    seen, prepared = [], []
    original = recording.prepare_renderer
    def prepare(container, width=640):
        prepared.append(container["profile"])
        render = original(container, width=width)
        def frame(snapshot):
            seen.append(snapshot["engine_state_status"])
            return render(snapshot)
        return frame
    monkeypatch.setattr(recording, "prepare_renderer", prepare)
    try:
        env.reset(seed=9)
        def no_live_render():
            raise AssertionError("Exercise render must not use live engine geometry")
        monkeypatch.setattr(env._engine, "render", no_live_render)
        nominal = env.render()
        def fail(_):
            env._engine.data.qpos[:] = np.nan
            raise RuntimeError("injected invalid live geometry")
        monkeypatch.setattr(env._engine, "step", fail)
        env.step(np.zeros(19))
        failed = env.render()
        assert prepared == ["shared-exercise-recording-v1"]
        assert seen == ["current", "last_valid_control_boundary"]
        assert nominal.shape == failed.shape and failed.dtype == np.uint8
        np.testing.assert_array_equal(nominal[:-64], failed[:-64])
        assert not np.array_equal(nominal[-64:], failed[-64:])
    finally:
        env.close()
