"""Analytical contract/accounting fixtures; no native environment or media work."""

import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import pytest

from weft.io import canonical_hash
from weft.shared_support_task import (
    GATES, GATES_SHA256, MAX_CONTACTS, PROFILE, REWARD_PROFILE, SCENE_PROFILE, SupportMetrics,
    build_shared_support_task, validate_shared_support_task,
)
from weft.shared_world_observation import EquipmentState, HumanObjectContact, SubstepSample


def support_fixture(*, horizon=2, scale=1., direction=(0., 0., 1.), root=False):
    scene = {"schema_version": 1, "profile": SCENE_PROFILE, "physical_validation": "unverified",
             "controlled_id": "human", "dt": 1 / 30, "substeps": 8, "gravity": [0., 0., -10.],
             "coordinate_frame": {"scale_bridge": {"body_height_to_world_scale": scale},
                                  "clock_binding": {"control_dt_s": 1 / 30, "physics_substeps": 8}},
             "task": {"type": "human_support_v1", "authored_addition": True, "start_offset_s": 0.,
                      "horizon_steps": horizon, "object_goal": None}, "target": None,
             "objects": [{"id": "support_a", "role": "static", "shape": "box", "size": [1., .4, .2],
                          "position": [1., 1., .5], "quaternion": [1., 0., 0., 0.],
                          "velocity": [0., 0., 0.], "angular_velocity": [0., 0., 0.],
                          "friction": [.8, .01, .001], "provenance": {"mode": "authored_addition"}}]}
    requirement = {"object_id": "support_a", "human_segment": "left_forearm", "direction_world": list(direction),
                   "minimum_load_body_weight_fraction": .05, "start_step": 0, "end_step": horizon}
    task = build_shared_support_task(scene, support_requirements=[requirement], root_assistance=root)
    return scene, task


def initial_supports(scene):
    return tuple(EquipmentState(obj["id"], tuple(obj["position"]), tuple(obj["quaternion"]),
                                (0.,) * 3, (0.,) * 3, (0.,) * 3) for obj in scene["objects"])


def accumulator(scene, task, *, mass=1., penetration=0.):
    return SupportMetrics(scene, task, initial_supports(scene), penetration, mass)


def contact(**changes):
    return replace(HumanObjectContact("left_elbow_geom", "left_forearm", "support_a", (0., 0., 0.),
                                     0., 1., (0., 0., 1.), (0., 0., 1.)), **changes)


def sample(acc, index, **changes):
    tick = acc.dt / acc.substeps
    supports = tuple(EquipmentState(name, position, quaternion, (0.,) * 3, (0.,) * 3, (0.,) * 3)
                     for name, (position, quaternion) in acc._supports.items())
    zero = (0.,) * 22
    value = SubstepSample(index // acc.substeps + 1, index % acc.substeps + 1,
                         index * tick, (index + 1) * tick, index * tick, True, None,
                         zero, zero, zero, zero, supports, (contact(),), MappingProxyType({}), 0.)
    return replace(value, **changes)


def finish(acc, mutate=None):
    for index in range(acc.maximum_samples):
        value = sample(acc, index)
        acc.consume(mutate(value, index) if mutate else value)
    return acc.metrics()


def test_versioned_detached_metadata_pins_complete_static_objects_without_claiming_truth():
    scene, task = support_fixture()
    before = deepcopy(scene)
    assert task["profile"] == PROFILE and task["reward_profile"] == REWARD_PROFILE
    assert task["control"] == {"root_assistance": "disabled", "object_forces": "forbidden"}
    assert task["source_scope"]["supports"] == [{"object_id": "support_a", "object_sha256": canonical_hash(scene["objects"][0])}]
    assert task["source_scope"]["physical_validation"] == "unverified"
    assert task["source_scope"]["source_contact_verified"] is task["source_scope"]["grasp_supported"] is False
    validated = validate_shared_support_task(task, scene)
    validated["support_requirements"][0]["direction_world"][2] = -1.
    assert task["support_requirements"][0]["direction_world"] == [0., 0., 1.]
    assert scene == before


def test_tracking_and_contact_constants_equal_existing_exercise_gates_without_importing_runtime():
    tree = ast.parse((Path(__file__).parents[1] / "src/weft/shared_exercise_task.py").read_text())
    node = next(node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "GATES" for target in node.targets))
    old = ast.literal_eval(node.value.args[0])
    assert all(value == old[key] for key, value in GATES.items())
    assert GATES_SHA256 == canonical_hash(dict(GATES))
    with pytest.raises(TypeError):
        GATES["max_floor_penetration_model_units"] = 1.


@pytest.mark.parametrize("change", [
    {"object_id": "mobile_floor"}, {"human_segment": "left_palm"}, {"human_segment": "left_foot"},
    {"direction_world": [0., 0., 2.]}, {"direction_world": [0., 0., 0.]},
    {"minimum_load_body_weight_fraction": 0.}, {"minimum_load_body_weight_fraction": 2.01},
    {"minimum_load_body_weight_fraction": True}, {"minimum_load_body_weight_fraction": float("nan")},
    {"start_step": True}, {"end_step": 3}, {"kind": "grasp"},
])
def test_unsupported_or_ambiguous_requirements_reject(change):
    scene, task = support_fixture()
    task["support_requirements"][0].update(change)
    with pytest.raises(ValueError):
        validate_shared_support_task(task, scene)


@pytest.mark.parametrize("mutation", [
    lambda scene: scene.update(profile="shared-world-v3"),
    lambda scene: scene.update(physical_validation="verified"),
    lambda scene: scene.update(objects=[]),
    lambda scene: scene["objects"][0].update(role="passive"),
    lambda scene: scene["objects"][0].update(velocity=[0., 0., .1]),
    lambda scene: scene["objects"][0].update(provenance={}),
    lambda scene: scene.update(gravity=[0., 0., 0.]),
    lambda scene: scene["task"].update(horizon_steps=1801),
    lambda scene: scene["task"].update(start_offset_s=1.),
    lambda scene: scene["coordinate_frame"]["clock_binding"].update(physics_substeps=16),
])
def test_metadata_only_context_rejects_incompatible_scenes(mutation):
    scene, task = support_fixture()
    mutation(scene)
    with pytest.raises(ValueError):
        build_shared_support_task(scene, support_requirements=task["support_requirements"])


@pytest.mark.parametrize("field,value", [("scene_sha256", "0" * 64), ("gates_sha256", "0" * 64),
                                          ("authored_addition", False), ("reward_profile", "stability-credit")])
def test_identity_or_criteria_changes_reject(field, value):
    scene, task = support_fixture()
    task[field] = value
    with pytest.raises(ValueError):
        validate_shared_support_task(task, scene)


def test_geometry_material_and_provenance_changes_require_new_scene_and_task_identity():
    scene, task = support_fixture()
    lower = deepcopy(scene)
    lower["objects"][0]["position"][2] -= .25
    with pytest.raises(ValueError):
        validate_shared_support_task(task, lower)
    rebound = build_shared_support_task(lower, support_requirements=task["support_requirements"])
    assert rebound["scene_sha256"] != task["scene_sha256"]
    assert rebound["source_scope"]["supports"] != task["source_scope"]["supports"]
    assert rebound["gates_sha256"] == task["gates_sha256"]
    assert rebound["support_requirements"] == task["support_requirements"]


def test_same_contact_cannot_satisfy_overlapping_requirements_but_disjoint_windows_are_allowed():
    scene, task = support_fixture(horizon=4)
    first = deepcopy(task["support_requirements"][0])
    second = deepcopy(first)
    with pytest.raises(ValueError, match="overlapping"):
        build_shared_support_task(scene, support_requirements=[first, second])
    first["end_step"], second["start_step"] = 2, 2
    assert len(build_shared_support_task(scene, support_requirements=[first, second])["support_requirements"]) == 2


def test_complete_signed_support_passes_and_reports_no_equipment_credit():
    scene, task = support_fixture()
    acc = accumulator(scene, task)
    before = acc.metrics()
    assert before["support_requirements"][0]["load_body_weight_fraction"] is None
    assert before["support_requirements"][0]["qualifying_fraction"] is None
    assert before["angular_rmse_rad"] is None and not before["success"]
    assert before["peak_angular_rmse_rad"] is before["peak_root_rmse_model_units"] is None
    result = finish(acc)
    assert result["completed_interval"] and result["success"]
    assert result["support_requirements"][0]["qualifying_fraction"] == pytest.approx(1.)
    assert result["support_requirements"][0]["load_body_weight_fraction"] == pytest.approx(.1)
    assert result["equipment"] == [] and result["gate_applicability"]["equipment_stability"] == "not_applicable"
    assert result["mean_reward"] == pytest.approx(1.)
    result["support_requirements"][0]["direction_world"][0] = 1.
    assert acc.metrics()["support_requirements"][0]["direction_world"] == [0., 0., 1.]


@pytest.mark.parametrize("contacts", [(), (contact(force_on_human_world=(0., 0., -1.)),),
                                      (contact(force_on_human_world=(1., 0., 0.)),),
                                      (contact(human_segment="right_forearm"),), (contact(normal_force=0.),)])
def test_absence_wrong_segment_wrong_direction_or_forcefree_contact_cannot_pass(contacts):
    scene, task = support_fixture()
    result = finish(accumulator(scene, task), lambda row, _: replace(row, contacts=contacts))
    assert result["completed_interval"] and not result["success"]
    assert result["support_requirements"][0]["qualifying_fraction"] == 0.
    assert result["mean_reward"] == 0.


def test_multiple_contacts_sum_signed_load_before_clamping_and_mass_normalization():
    scene, task = support_fixture()
    pair = (contact(force_on_human_world=(0., 0., .3)), contact(force_on_human_world=(0., 0., .3)))
    assert finish(accumulator(scene, task), lambda row, _: replace(row, contacts=pair))["success"]
    opposing = (contact(force_on_human_world=(0., 0., 1.)), contact(force_on_human_world=(0., 0., -1.)))
    result = finish(accumulator(scene, task), lambda row, _: replace(row, contacts=opposing))
    assert not result["success"] and result["support_requirements"][0]["load_body_weight_fraction"] == 0.
    assert not finish(accumulator(scene, task, mass=3.))["success"]


def test_declared_horizontal_wall_force_is_supported_without_renaming_it_upward_or_grasp():
    scene, task = support_fixture(direction=(1., 0., 0.))
    result = finish(accumulator(scene, task), lambda row, _: replace(row, contacts=(contact(force_on_human_world=(1., 0., 0.)),)))
    assert result["success"]
    assert result["support_requirements"][0]["direction_world"] == [1., 0., 0.]
    assert result["source_scope"]["source_contact_verified"] is False


def test_all_existing_segments_are_generic_not_hardcoded_to_forearm():
    scene, task = support_fixture()
    requirement = task["support_requirements"][0]
    requirement["human_segment"] = "left_thigh"
    task = build_shared_support_task(scene, support_requirements=[requirement])
    result = finish(accumulator(scene, task), lambda row, _: replace(row, contacts=(contact(human_segment="left_thigh"),)))
    assert result["success"]


@pytest.mark.parametrize("scale", [.5, 2.])
def test_contact_distance_gate_remains_in_model_units(scale):
    scene, task = support_fixture(scale=scale)
    near = finish(accumulator(scene, task), lambda row, _: replace(row, contacts=(contact(distance=.001 * scale),)))
    far = finish(accumulator(scene, task), lambda row, _: replace(row, contacts=(contact(distance=.00101 * scale),)))
    assert near["success"] and not far["success"]


def test_boundary_only_contact_cannot_hide_substep_gaps():
    scene, task = support_fixture()
    acc = accumulator(scene, task)
    result = finish(acc, lambda row, index: replace(row, contacts=row.contacts if index % acc.substeps == 7 else ()))
    assert result["support_requirements"][0]["qualifying_fraction"] == pytest.approx(1 / 8)
    assert not result["success"]


def test_long_gap_fails_despite_more_than_95_percent_total_contact():
    scene, task = support_fixture(horizon=90)
    acc = accumulator(scene, task)
    result = finish(acc, lambda row, index: replace(row, contacts=() if 100 <= index < 127 else row.contacts))
    requirement = result["support_requirements"][0]
    assert requirement["qualifying_fraction"] > .95
    assert requirement["max_gap_s"] > .1 and not result["success"]


def test_only_declared_window_scores_contact_and_partial_observations_remain_incomplete():
    scene, task = support_fixture()
    task["support_requirements"][0]["start_step"] = 1
    acc = accumulator(scene, task)
    for index in range(acc.substeps):
        acc.consume(sample(acc, index, contacts=()))
    assert acc.metrics()["support_requirements"][0]["observed_duration_s"] == 0.
    assert not acc.metrics()["completed_interval"]
    for index in range(acc.substeps, acc.maximum_samples):
        acc.consume(sample(acc, index))
    assert acc.metrics()["success"]
    assert acc.metrics()["support_requirements"][0]["observed_duration_s"] == pytest.approx(scene["dt"])


@pytest.mark.parametrize("change", [{"step_index": 2}, {"contact_time_s": .001}, {"interval_end_s": None},
                                    {"valid": False, "failure": "native_failure"}])
def test_missing_misaligned_or_failed_samples_never_complete(change):
    scene, task = support_fixture()
    acc = accumulator(scene, task)
    result = finish(acc, lambda row, index: replace(row, **change) if index == 0 else row)
    assert not result["success"] and not result["completed_interval"]
    assert result["failure"] is not None
    assert result["valid_substeps"] < result["observed_substeps"]


def test_no_silent_root_assistance_and_no_terminal_recovery_of_peak_errors():
    scene, task = support_fixture()
    action = (0.,) * 19 + (.1, 0., 0.)
    result = finish(accumulator(scene, task), lambda row, _: replace(row, human_action=action))
    assert result["failure"] == "forbidden_human_actuation" and not result["success"]
    scene, task = support_fixture(root=True)
    result = finish(accumulator(scene, task), lambda row, _: replace(row, human_action=action))
    assert result["success"] and result["root_assistance"] == "explicit_3_axis"
    assert result["root_assistance_mean_squared"] == pytest.approx(.01 / 3)
    scene, task = support_fixture(horizon=10)
    result = finish(accumulator(scene, task), lambda row, index: replace(row, human_qpos=(.71,) * 19 + (0.,) * 3) if index == 0 else row)
    assert result["angular_rmse_rad"] < .35 and result["peak_angular_rmse_rad"] == pytest.approx(.71)
    assert "peak_angular_rmse_rad" in result["failed_gates"]
    assert not finish(accumulator(scene, task, penetration=.026))["success"]


@pytest.mark.parametrize("field,value", [("position", (1., 1., .6)), ("velocity", (0., 0., .1)),
                                          ("virtual_force", (0., 0., 1.))])
def test_static_pose_motion_and_virtual_force_are_checked_every_sample(field, value):
    scene, task = support_fixture()
    acc = accumulator(scene, task)
    value = sample(acc, 0, equipment=(replace(initial_supports(scene)[0], **{field: value}),))
    with pytest.raises(ValueError, match="Static support moved"):
        acc.consume(value)
    assert not acc.metrics()["completed_interval"]


def test_bounded_contacts_states_and_work_and_idempotent_failure_penalty():
    scene, task = support_fixture()
    acc = accumulator(scene, task)
    with pytest.raises(ValueError, match="bounded engine"):
        acc.consume(sample(acc, 0, contacts=(contact(),) * (MAX_CONTACTS + 1)))
    acc = accumulator(scene, task)
    with pytest.raises(ValueError, match="finite"):
        acc.consume(sample(acc, 0, human_qvel=(float("nan"),) + (0.,) * 21))
    acc = accumulator(scene, task)
    finish(acc)
    with pytest.raises(ValueError, match="budget"):
        acc.consume(sample(acc, acc.maximum_samples))
    assert acc.metrics()["failure"] == "substep_budget_exceeded" and not acc.metrics()["success"]
    before = acc.reward_integral
    acc.penalize_failure()
    acc.penalize_failure()
    assert acc.reward_integral == pytest.approx(before - acc.dt)


@pytest.mark.parametrize("malformed", ["support", "contact"])
def test_malformed_observation_identity_rejects_explicitly(malformed):
    scene, task = support_fixture()
    acc = accumulator(scene, task)
    value = sample(acc, 0)
    if malformed == "support":
        value = replace(value, equipment=(replace(value.equipment[0], object_id=[]),))
    else:
        value = replace(value, contacts=(contact(human_segment=[]),))
    with pytest.raises(ValueError):
        acc.consume(value)


def test_stdlib_only_import_with_native_model_and_media_packages_forbidden():
    root = str(Path(__file__).parents[1] / "src")
    script = '''import builtins, sys
sys.path.insert(0, ROOT)
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'mujoco', 'gymnasium', 'cv2', 'PIL', 'mediapipe', 'torch'}:
        raise AssertionError('Forbidden import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from weft.shared_support_task import GATES, PROFILE
assert PROFILE == 'shared-support-task-v1' and GATES['max_floor_penetration_model_units'] == .025
'''.replace("ROOT", repr(root))
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
