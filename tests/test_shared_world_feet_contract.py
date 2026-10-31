"""Pure shared-foot custody, XML and initial-target checks; native execution is forbidden."""

from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from weft import shared_world_feet as feet
from weft.foot_body import append_foot_capsules, foot_xml
from weft.foot_reference import FOOT_CONTRACT
from weft.human import human_xml
from weft.io import canonical_hash
from weft.mobile_human import MOBILITY_DEFAULTS
from weft.spatial_pose import SPATIAL_JOINT_NAMES, rotation_zyx


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Shared-foot contract tests cannot compile or run a native model")

    monkeypatch.setattr(mujoco, "MjModel", SimpleNamespace(from_xml_string=forbidden, from_xml_path=forbidden))
    for name in ("MjData", "Renderer", "mj_forward", "mj_step", "mj_resetData"):
        monkeypatch.setattr(mujoco, name, forbidden)


def inputs():
    spec = {
        "schema_version": 3, "profile": "exercise-v3", "source_sha256": "a" * 64, "dt": 1 / 30,
        "lengths": {"torso": .5, "upper_arm": .28, "forearm": .25, "thigh": .42,
                    "shin": .4, "shoulder_width": .36, "hip_width": .24},
        "reference": [{"time_s": i / 30, "joints": {name: 0. for name in SPATIAL_JOINT_NAMES}} for i in range(3)],
        "root_reference": [{"time_s": i / 30, "translation": [0., 0., 0.]} for i in range(3)],
        "mobility": deepcopy(MOBILITY_DEFAULTS),
        "body_frame": {"schema_version": 1, "status": "assumed", "evidence_ids": [], "reference": [],
                       "rotation": np.eye(3).tolist(), "scope": "Authored analytical contract fixture"},
        "assumptions": ["Analytical geometry only; no source study or physical validation"],
    }
    foot = {"status": "kinematic_reference", "reference": {
        "heel_to_index_axis": [1., 0., 0.], "axis_to_ankle_direction": [0., 0., 1.]},
        "endpoint_geometry": {"heel_relative": [-.04, 0., -.08], "foot_index_relative": [.16, 0., -.08]}}
    reference = {
        "schema_version": 1, "profile": "kinematic-foot-reference-v1", "contract": deepcopy(FOOT_CONTRACT),
        "source_sha256": spec["source_sha256"], "body_model_sha256": canonical_hash(spec),
        "fixed_shapes": {side: {"status": "inferred_fixed_triangle", "fixed_parameters": {
            "length": .2, "along": .04, "height": .08}} for side in ("left", "right")},
        "frames": [{"time_s": i / 30, "frame": i, "evidence_id": spec["source_sha256"] + f":frame:{i}",
                    "feet": {side: deepcopy(foot) for side in ("left", "right")}} for i in range(3)],
    }
    return spec, reference


def raw(reference):
    return (json.dumps(reference, indent=2) + "\n").encode()


def missing(foot):
    foot.update(status="unavailable", reference=None, endpoint_geometry=None)


def test_exact_reference_bytes_body_source_and_start_are_bound_without_mutation():
    spec, reference = inputs()
    before = deepcopy((spec, reference))
    original = raw(reference)
    extension = feet.build_foot_extension(spec, original)
    validated = feet.validate_foot_extension(spec, extension)
    assert (spec, reference) == before
    assert extension["reference_json"].encode() == original
    assert extension["reference_sha256"] == sha256(original).hexdigest()
    assert extension["reference_canonical_sha256"] == canonical_hash(reference)
    assert extension["body_model_sha256"] == canonical_hash(spec)
    assert extension["initial_reference"] == {key: reference["frames"][0][key] for key in ("time_s", "frame", "evidence_id")}
    assert extension["target_mode"] == "authored_initial_angle_hold"
    validated["contract"]["ankle_names"][0] = "changed"
    assert extension["contract"] == feet.SHARED_FEET_CONTRACT


@pytest.mark.parametrize("change", ["body", "source", "reference_bytes", "start"])
def test_changed_custody_is_rejected(change):
    spec, reference = inputs()
    extension = feet.build_foot_extension(spec, raw(reference))
    if change == "body":
        spec["mobility"]["base_height"] += .01
    elif change == "source":
        extension["source_sha256"] = "b" * 64
    elif change == "reference_bytes":
        extension["reference_json"] += " "
    else:
        extension["initial_reference"]["frame"] = 1
    with pytest.raises(ValueError, match="match the unchanged|bindings"):
        feet.validate_foot_extension(spec, extension)


@pytest.mark.parametrize("change,match", [("missing_start", "original starting pose"),
                                       ("unavailable_start", "original starting pose"),
                                       ("unavailable_shape", "fixed source-inferred shapes")])
def test_original_start_and_both_fixed_shapes_are_required(change, match):
    spec, reference = inputs()
    if change == "missing_start":
        reference["frames"] = reference["frames"][1:]
    elif change == "unavailable_start":
        missing(reference["frames"][0]["feet"]["left"])
    else:
        reference["fixed_shapes"]["left"].update(status="unavailable", fixed_parameters=None)
        for row in reference["frames"]:
            missing(row["feet"]["left"])
    with pytest.raises(ValueError, match=match):
        feet.build_foot_extension(spec, raw(reference))


def test_later_gaps_stay_explicit_and_never_change_initial_hold():
    spec, reference = inputs()
    original = feet.build_foot_extension(spec, raw(reference))
    missing(reference["frames"][1]["feet"]["left"])
    reference["frames"][2]["feet"]["right"]["reference"].update(
        heel_to_index_axis=[0., 1., 0.], axis_to_ankle_direction=[0., 0., 1.])
    changed = feet.build_foot_extension(spec, raw(reference))
    knees = {side: np.eye(3) for side in ("left", "right")}
    assert feet.initial_ankle_angles(spec, original, knees) == feet.initial_ankle_angles(spec, changed, knees)
    assert json.loads(changed["reference_json"])["frames"] == reference["frames"]
    assert changed["reference_sha256"] != original["reference_sha256"]
    assert changed["contract"]["learning"] == "unsupported"
    assert changed["contract"]["ground_force_qualification"] == "not_qualified"


def test_initial_angles_include_source_body_frame_once_and_use_local_zyx():
    spec, reference = inputs()
    body_frame = rotation_zyx([.7, -.4, .3])
    spec["body_frame"]["rotation"] = body_frame.tolist()
    reference["body_model_sha256"] = canonical_hash(spec)
    targets = {"left": [.2, -.1, .3], "right": [-.15, .2, -.1]}
    knees = {"left": body_frame @ rotation_zyx([.1, .2, -.2]),
             "right": body_frame @ rotation_zyx([-.1, -.2, .2])}
    for side in targets:
        desired = knees[side] @ rotation_zyx(targets[side])
        reference["frames"][0]["feet"][side]["reference"].update(
            heel_to_index_axis=desired[:, 0].tolist(), axis_to_ankle_direction=desired[:, 2].tolist())
    extension = feet.build_foot_extension(spec, raw(reference))
    angles = feet.initial_ankle_angles(spec, extension, knees)
    assert angles == pytest.approx(targets["left"] + targets["right"], abs=1e-12)
    assert spec["body_frame"]["rotation"] == body_frame.tolist()
    assert feet.ANKLE_NAMES == ("left_ankle_z", "left_ankle_y", "left_ankle_x", "right_ankle_z", "right_ankle_y", "right_ankle_x")
    assert extension["contract"]["ankle_engine_indices"] == list(range(22, 28))


def test_invalid_knee_rotations_are_not_repaired():
    spec, reference = inputs()
    extension = feet.build_foot_extension(spec, raw(reference))
    with pytest.raises(ValueError, match="both nominal-start"):
        feet.initial_ankle_angles(spec, extension, {"left": np.eye(3)})
    with pytest.raises(ValueError, match="proper rotation matrices"):
        feet.initial_ankle_angles(spec, extension, {"left": np.eye(3) * 1.001, "right": np.eye(3)})


@pytest.mark.parametrize("scale", [.5, 1., 2.])
def test_append_preserves_original_tree_and_motor_order_with_explicit_scaling(scale):
    spec, reference = inputs()
    before = deepcopy((spec, reference))
    extension = feet.build_foot_extension(spec, raw(reference))
    root = ET.fromstring(human_xml(spec))
    original = ET.tostring(root)
    motors = list(root.find("actuator"))
    original_names = [motor.get("joint") for motor in motors]
    assert len(motors) == 22
    assert feet.append_shared_feet(root, spec, extension, scale=scale) is root
    assert [motor.get("joint") for motor in root.find("actuator")] == original_names + list(feet.ANKLE_NAMES)
    for side in ("left", "right"):
        parent = root.find(f'.//body[@name="{side}_knee_body"]')
        foot = parent.find(f'body[@name="{side}_foot_body"]')
        geom = foot.find("geom")
        assert list(map(float, foot.get("pos").split())) == pytest.approx([0., 0., -.4 * scale])
        assert list(map(float, geom.get("fromto").split())) == pytest.approx(np.array([-.04, 0, -.08, .16, 0, -.08]) * scale)
        assert float(geom.get("size")) == pytest.approx(.018 * scale)
        assert float(geom.get("mass")) == pytest.approx(.5 * scale**3)
        parent.remove(foot)
    for motor in list(root.find("actuator"))[22:]:
        assert motor.get("ctrlrange") == "-12 12"
        root.find("actuator").remove(motor)
    assert ET.tostring(root) == original
    assert feet.added_foot_mass(scale) == pytest.approx(scale**3)
    assert feet.ANKLE_LIMITS == (12.,) * 6
    assert (spec, reference) == before


def test_duplicate_or_incomplete_attachment_is_atomic():
    spec, reference = inputs()
    root = ET.fromstring(human_xml(spec))
    append_foot_capsules(root, spec, reference)
    before = ET.tostring(root)
    with pytest.raises(ValueError, match="already exist"):
        append_foot_capsules(root, spec, reference)
    assert ET.tostring(root) == before
    root.find('.//body[@name="right_knee_body"]').set("name", "unrelated_body")
    before = ET.tostring(root)
    with pytest.raises(ValueError, match="unique existing knee"):
        append_foot_capsules(root, spec, reference)
    assert ET.tostring(root) == before


def test_shared_attachment_rejects_reordered_existing_channels_before_mutation():
    spec, reference = inputs()
    extension = feet.build_foot_extension(spec, raw(reference))
    root = ET.fromstring(human_xml(spec))
    actuator = root.find("actuator")
    actuator.append(actuator[0])
    actuator.remove(actuator[0])
    before = ET.tostring(root)
    with pytest.raises(ValueError, match="unchanged ordered 19 body plus 3 root"):
        feet.append_shared_feet(root, spec, extension, scale=1.)
    assert ET.tostring(root) == before


def test_reference_byte_bound_precedes_json_parse(monkeypatch):
    spec, _ = inputs()
    monkeypatch.setattr(feet.json, "loads", lambda *_a, **_k: pytest.fail("Oversized reference parsed"))
    with pytest.raises(ValueError, match="within 8 MiB"):
        feet.build_foot_extension(spec, b" " * (feet.MAX_REFERENCE_BYTES + 1))


def test_duplicate_json_and_changed_profile_are_not_accepted():
    spec, reference = inputs()
    with pytest.raises(ValueError, match="duplicate keys"):
        feet.build_foot_extension(spec, b'{"profile":"x","profile":"y"}')
    extension = feet.build_foot_extension(spec, raw(reference))
    extension["profile"] = "shared-world-v4"
    with pytest.raises(ValueError, match="Unsupported shared-world foot"):
        feet.validate_foot_extension(spec, extension)


def test_legacy_foot_xml_remains_byte_identical_at_default_scale():
    spec, reference = inputs()
    # Frozen from the committed pre-extraction function with native entry points blocked.
    assert sha256(foot_xml(spec, reference).encode()).hexdigest() == "0475f885ab5dba0adbe7a3e44818925471c514315813660e401ca0251f8bf286"
