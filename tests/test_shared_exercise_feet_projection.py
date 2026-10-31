"""Analytical projection only; native mass extraction is the sole validation mock."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import socket
import subprocess
from time import perf_counter, process_time
from types import MappingProxyType, SimpleNamespace

import mujoco
import numpy as np
import pytest

from test_shared_support_task import accumulator, contact, sample, support_fixture
from test_shared_world_feet_contract import inputs, raw
from weft import shared_world
from weft.io import canonical_hash
from weft.person_inventory import make_person_inventory
from weft.shared_exercise_feet_projection import (
    HUMAN_OBSERVATION_VALUES, PROJECTION_CONTRACT, PROJECTION_CONTRACT_SHA256,
    derive_base_scoring_scene, project_substep, scatter_task_action, task_motor_indices,
)
from weft.shared_world_observation import EquipmentState, HumanObjectContact, SubstepSample
from weft.whole_body import SUCCESS_LIMITS


@pytest.fixture(autouse=True)
def no_native_or_external_work(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Projection tests cannot compile models, run physics, acquire media or spawn workers")

    monkeypatch.setattr(mujoco, "MjModel", SimpleNamespace(
        from_xml_string=forbidden, from_xml_path=forbidden, from_binary_path=forbidden))
    for name in ("MjData", "Renderer", "mj_forward", "mj_step", "mj_resetData", "mj_contactForce", "mj_kinematics"):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(shared_world, "SharedWorldEnv", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    # Keep the real scene validator; these metadata checks do not qualify native mass.
    monkeypatch.setattr(shared_world, "human_subtree_mass", lambda _spec, scale=1.: 70. * scale**3)


def scene_fixture(profile="shared-world-v3"):
    spec, reference = inputs()
    source = spec["source_sha256"]
    body = {"world_entity_id": "human", "entity_kind": "articulated_human",
            "observation_entity_id": source + ":human:0", "observation_document_sha256": "d" * 64,
            "provenance_mode": "observed_track", "observed_slots": 3, "missing_slots": 0, "pre_init_slots": 0}
    scene = {
        "schema_version": 1, "profile": profile, "physical_validation": "unverified",
        "learned_policy_available": False, "dt": 1 / 30, "substeps": 8,
        "bounds": [3., 3., 3.], "gravity": [0., 0., -9.81],
        "contact": {"time_constant": .012, "damping_ratio": 1.},
        "coordinate_frame": {
            "axes": "x_right_y_depth_z_up", "calibration_status": "unverified",
            "units": {"length": "world_meter_prior", "mass": "kilogram_prior", "time": "second",
                      "orientation": "quaternion_wxyz", "angular_velocity": "world_radians_per_second"},
            "scale_bridge": {"body_height_to_world_scale": 1., "origin_world": [1., 1., 0.],
                             "method": "authored_analytical_fixture", "uncertainty_note": "No physical calibration"},
            "clock_binding": {"control_dt_s": 1 / 30, "physics_substeps": 8, "source_fps": 30.,
                              "source_time_offset_s": 0., "alignment_method": "authored_30hz_fixture"}},
        "source_binding": {"mode": "source_qualified", "source_sha256": source, "total_source_frames": 3,
                           "evidence_documents": {"d" * 64: "evidence/analytical-human.json"},
                           "entity_bindings": [body]},
        "human_spec": spec, "objects": [], "controlled_id": "human", "target": None,
        "task": {"type": "human_motion_v1", "authored_addition": True, "start_offset_s": 0.,
                 "horizon_steps": 2, "human_limits": deepcopy(SUCCESS_LIMITS), "object_goal": None},
        "assumptions": ["Synthetic source identities and authored geometry; no media or physical evidence"],
    }
    if profile == "shared-world-v3":
        scene["inventory"] = make_person_inventory(source)
    else:
        support = profile == "shared-world-v4"
        obj = {"id": "support_a" if support else "ball", "label": "Analytical fixture",
               "role": "static" if support else "passive", "shape": "box", "size": [.2, .2, .2],
               "position": [2., 2., .2], "quaternion": [1., 0., 0., 0.], "velocity": [0., 0., 0.],
               "angular_velocity": [0., 0., 0.], "mass": 1., "friction": [.8, .01, .001],
               "color": "#123456", "provenance": {"mode": "authored_addition"}}
        scene["objects"] = [obj]
        scene["source_binding"]["entity_bindings"].append({
            "world_entity_id": obj["id"], "entity_kind": "rigid_object", "observation_entity_id": None,
            "observation_document_sha256": None, "provenance_mode": "authored_addition",
            "observed_slots": 0, "missing_slots": 3, "pre_init_slots": 0})
        scene["task"]["type"] = "human_support_v1" if support else "coupled_human_object_v1"
        if not support:
            scene["target"] = [2., 2.2, .2]
            scene["task"]["object_goal"] = {"target_object_id": "ball", "position_tolerance": .06,
                "speed_tolerance": .1, "virtual_acceleration": 6., "require_human_contact": True}
    base = shared_world.validate_shared_world(scene)
    reference["body_model_sha256"] = canonical_hash(base["human_spec"])
    return base, shared_world.add_feet_to_shared_world(base, raw(reference))


@pytest.mark.parametrize("profile", ["shared-world-v2", "shared-world-v3", "shared-world-v4"])
def test_base_projection_preserves_all_nonfoot_scene_data_without_mutation(profile):
    base, scene = scene_fixture(profile)
    before = deepcopy(scene)
    projected = derive_base_scoring_scene(scene)
    assert projected == base
    assert projected["action_contract"]["human_motor_count"] == 22
    assert projected["action_contract"]["names"][22:] == scene["action_contract"]["names"][28:]
    assert canonical_hash(projected) != canonical_hash(scene)
    assert scene == before
    projected["human_spec"]["root_reference"][0]["translation"][0] = .1
    assert scene["human_spec"] == before["human_spec"]


@pytest.mark.parametrize("change", ["legacy", "reference_bytes", "body", "clock", "clipped_task", "motor_order"])
def test_base_projection_rejects_changed_source_interval_or_extension(change):
    base, scene = scene_fixture()
    if change == "legacy":
        scene = base
    elif change == "reference_bytes":
        scene["foot_extension"]["reference_json"] += " "
    elif change == "body":
        scene["human_spec"]["mobility"]["base_height"] += .01
    elif change == "clock":
        scene["coordinate_frame"]["clock_binding"]["physics_substeps"] = 16
    elif change == "clipped_task":
        scene["task"]["horizon_steps"] = 1
    else:
        names = scene["action_contract"]["names"]
        names[19], names[22] = names[22], names[19]
    with pytest.raises(ValueError):
        derive_base_scoring_scene(scene)


def test_contract_is_immutable_and_explicitly_excludes_new_scoring_claims():
    assert PROJECTION_CONTRACT_SHA256 == canonical_hash(dict(PROJECTION_CONTRACT))
    assert PROJECTION_CONTRACT["scorer_motor_indices"] == tuple(range(22))
    assert PROJECTION_CONTRACT["angular_scoring_motor_indices"] == tuple(range(19))
    assert dict(PROJECTION_CONTRACT["contact_geometry_segment_pairs"]) == shared_world._HUMAN_SEGMENT_BY_GEOM
    assert HUMAN_OBSERVATION_VALUES == 4 * 28 + 3 * 28 + 1
    assert PROJECTION_CONTRACT["excluded_contacts"] == "separate_telemetry_no_support_credit"
    assert PROJECTION_CONTRACT["ankle_effort"] == "separate_telemetry_not_scored"
    assert PROJECTION_CONTRACT["ground_load_requirements"] == "unsupported"
    with pytest.raises(TypeError):
        PROJECTION_CONTRACT["scorer_motor_indices"] = tuple(range(28))


@pytest.mark.parametrize("root", [False, True])
def test_task_actions_route_ankles_before_optional_roots_with_no_object_forces(root):
    values = np.linspace(-1., 1., 28 if root else 25, dtype=np.float32)
    before = values.copy()
    indices = task_motor_indices(root_assistance=root)
    result = scatter_task_action(values, engine_action_count=64, root_assistance=root)
    assert indices == tuple(range(19)) + tuple(range(22, 28)) + ((19, 20, 21) if root else ())
    assert isinstance(result, tuple) and len(result) == 64
    assert tuple(result[index] for index in indices) == tuple(values)
    assert result[:19] == tuple(values[:19]) and result[22:28] == tuple(values[19:25])
    assert result[19:22] == (tuple(values[25:]) if root else (0., 0., 0.))
    assert result[28:] == (0.,) * 36
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("bad", [True, np.bool_(False), "0.1", float("nan"), float("inf"), 1.001])
def test_actions_reject_coercion_nonfinite_and_out_of_range_values(bad):
    values = [0.] * 25
    values[22] = bad
    with pytest.raises(ValueError):
        scatter_task_action(values, engine_action_count=31)


def test_action_shapes_and_engine_layout_are_bounded_before_allocation():
    for value in ([0.] * 28, [[0.]] * 25, np.zeros((25, 1)), iter([0.] * 25), "0" * 25):
        with pytest.raises(ValueError):
            scatter_task_action(value, engine_action_count=28)
    for count in (True, 28., 27, 29, 67, 10**9):
        with pytest.raises(ValueError):
            scatter_task_action([0.] * 25, engine_action_count=count)
    with pytest.raises(ValueError, match="explicit boolean"):
        task_motor_indices(root_assistance=1)


def foot_contact(side="left", **changes):
    return replace(contact(human_geom=side + "_foot_geom", human_segment=side + "_foot"), **changes)


def engine_sample(**changes):
    equipment = (EquipmentState("support_a", (0., 0., 0.), (1., 0., 0., 0.),
                                (0.,) * 3, (0.,) * 3, (0.,) * 3),)
    value = SubstepSample(2, 3, .04, .04 + 1 / 240, .04, True, None,
        tuple(i / 100 for i in range(28)), tuple(i / 50 for i in range(28)), (0.,) * 28, (0.,) * 28,
        equipment, (foot_contact(), contact(), foot_contact("right")),
        MappingProxyType({"left": (1., 2., 3.), "right": (4., 5., 6.)}), .003)
    return replace(value, **changes)


def test_substep_projection_preserves_clocks_failure_and_detached_nonmotor_state():
    original = engine_sample(failure="finite_dynamics_failure")
    projected, excluded = project_substep(original)
    assert projected is not original
    assert (projected.human_qpos, projected.human_qvel, projected.target_qpos, projected.human_action) == tuple(
        getattr(original, field)[:22] for field in ("human_qpos", "human_qvel", "target_qpos", "human_action"))
    assert projected.contacts == (original.contacts[1],)
    assert excluded == (original.contacts[0], original.contacts[2])
    assert projected.equipment is original.equipment and projected.wrist_positions is original.wrist_positions
    for field in ("step_index", "substep_index", "interval_start_s", "interval_end_s", "contact_time_s", "valid",
                  "failure", "max_human_floor_penetration_model_units"):
        assert getattr(projected, field) == getattr(original, field)
    assert len(original.human_action) == 28 and len(original.contacts) == 3
    with pytest.raises(FrozenInstanceError):
        projected.valid = False


def test_invalid_engine_sample_preserves_missing_state_clock_and_failure():
    original = engine_sample(valid=False, failure="nonfinite_dynamics", interval_end_s=None,
        human_qpos=(), human_qvel=(), target_qpos=(), equipment=(), contacts=(),
        wrist_positions=MappingProxyType({}), max_human_floor_penetration_model_units=None)
    projected, excluded = project_substep(original)
    assert projected == replace(original, human_action=(0.,) * 22)
    assert excluded == ()


@pytest.mark.parametrize("change", [
    {"human_segment": "left_palm"}, {"human_segment": "unrecognized_foot"},
    {"human_geom": "left_elbow_geom"}, {"human_segment": "left_forearm"},
    {"object_id": "unobserved_support"}, {"normal_force": float("nan")},
    {"force_on_human_world": (0., True, 0.)},
])
def test_contact_exclusion_cannot_hide_unknown_aliases_or_invalid_foot_loads(change):
    with pytest.raises(ValueError):
        project_substep(engine_sample(contacts=(foot_contact(**change),)))


@pytest.mark.parametrize("geom,segment", [
    ("unknown_foot_alias_geom", "left_forearm"),
    ("left_knee_geom", "left_forearm"),
    ("left_foot_alias_geom", "left_foot"),
    ("right_foot_geom", "left_foot"),
], ids=["unknown-geom-claims-forearm", "known-shin-claims-forearm", "unknown-foot-alias", "wrong-foot-side"])
def test_every_contact_requires_the_canonical_engine_geometry_segment_pair(geom, segment):
    with pytest.raises(ValueError):
        project_substep(engine_sample(contacts=(contact(human_geom=geom, human_segment=segment),)))


def test_all_canonical_engine_contacts_keep_their_identity_and_scoring_partition():
    contacts = tuple(contact(human_geom=geom, human_segment=segment)
                     for geom, segment in shared_world._HUMAN_SEGMENT_BY_GEOM.items())
    original = engine_sample(contacts=contacts)
    projected, excluded = project_substep(original)
    assert projected.contacts == tuple(item for item in contacts if not item.human_segment.endswith("_foot"))
    assert excluded == tuple(item for item in contacts if item.human_segment.endswith("_foot"))
    assert all(actual is expected for actual, expected in zip(projected.contacts + excluded, contacts))
    assert original.contacts is contacts


def test_projection_does_not_hide_invalid_ankle_values_or_wrong_motor_width():
    for field in ("human_qpos", "human_qvel", "target_qpos", "human_action"):
        values = [0.] * 28
        values[-1] = float("nan")
        with pytest.raises(ValueError):
            project_substep(engine_sample(**{field: tuple(values)}))
        with pytest.raises(ValueError):
            project_substep(engine_sample(**{field: (0.,) * 22}))
    with pytest.raises(ValueError):
        project_substep(engine_sample(human_action=(0.,) * 27 + (1.01,)))


@pytest.mark.parametrize("retained_contact", [False, True])
def test_existing_support_metrics_get_no_foot_credit_or_ankle_effort(retained_contact):
    scene, task = support_fixture()
    expected, actual = accumulator(scene, task), accumulator(scene, task)
    old = sample(expected, 0, contacts=(contact(),) if retained_contact else ())
    extended = replace(old, human_qpos=old.human_qpos + (2.,) * 6, human_qvel=old.human_qvel + (3.,) * 6,
        target_qpos=old.target_qpos + (0.,) * 6, human_action=old.human_action + (1.,) * 6,
        contacts=old.contacts + (foot_contact(normal_force=100., force_on_human_world=(0., 0., 100.)),))
    projected, excluded = project_substep(extended)
    expected.consume(old)
    actual.consume(projected)
    assert actual.metrics() == expected.metrics()
    assert len(excluded) == 1 and excluded[0].normal_force == 100.
    assert actual.requirements[0]["qualifying_now"] is retained_contact


def test_repeated_maximum_contact_projection_is_bounded_and_never_mutates_input():
    contacts = (contact(), foot_contact()) * 128
    original = engine_sample(contacts=contacts)
    start, cpu_start = perf_counter(), process_time()
    for _ in range(128):
        projected, excluded = project_substep(original)
        assert len(projected.contacts) == len(excluded) == 128
        assert projected.equipment is original.equipment
    elapsed, cpu = perf_counter() - start, process_time() - cpu_start
    assert original.contacts is contacts and len(original.human_qpos) == 28
    with pytest.raises(ValueError, match="bounded immutable engine tuple"):
        project_substep(replace(original, contacts=contacts + (contact(),)))
    print(json.dumps({"projection_measurement": {"iterations": 128, "contacts_per_sample": 256,
          "wall_seconds": elapsed, "cpu_seconds": cpu, "native_execution": False}}))
