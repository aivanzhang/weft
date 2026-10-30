from copy import deepcopy
import json

import pytest

from weft.observations import adapt_exercise
from weft.spatial_pose import SPATIAL_JOINT_NAMES
from test_observations import HASH, exercise


def mobile_scene():
    return {"profile": "exercise-v3", "joint_names": list(SPATIAL_JOINT_NAMES),
            "human": {"profile": "exercise-v3", "source_sha256": HASH,
                      "joint_names": list(SPATIAL_JOINT_NAMES),
                      "mobility": {"translation_axes": 3, "support": "authored virtual motors",
                                   "floor": "assumed first-pose support plane", "root_depth": "unobserved; zero reference"}}}


def test_mobile_export_preserves_pose_slots_clocks_and_inputs_without_fixed_support():
    evidence, scene = exercise(), mobile_scene()
    for row in evidence["frames"]:
        row["decoder_timestamp_s"] = row["time_s"] + .003
    original_evidence, original_scene = deepcopy(evidence), deepcopy(scene)
    baseline = adapt_exercise(evidence)
    result = adapt_exercise(evidence, scene=scene)
    assert evidence == original_evidence and scene == original_scene
    for key in ("source", "time", "observations", "evidence"):
        assert result[key] == baseline[key]
    assert result["time"]["status"] == "assumed"
    assert [row["payload"] for row in result["evidence"]] == evidence["frames"]
    assert [row["partition"] for row in result["observations"]] == ["construction", "unsampled", "construction", "unsampled"]
    relations = result["relationships"]
    assert len(relations) == 23
    assert sum(row["type"] == "internal_hinge_joint" for row in relations) == 19
    assert {row["attributes"]["name"] for row in relations if row["type"] == "root_slide_joint"} == {"root_x", "root_y", "root_z"}
    assert all(row["type"] != "anchored_support" and row["status"] == "assumed" and not row["evidence_ids"] for row in relations)
    actuation = next(row for row in relations if row["type"] == "virtual_actuation")
    assert actuation["attributes"]["mobility"] == scene["human"]["mobility"]
    actuation["attributes"]["mobility"]["support"] = "changed export"
    assert scene == original_scene


@pytest.mark.parametrize("body_model", ["spatial", "mobile"])
def test_exported_joint_types_match_compiled_body_not_legacy_hinge_template(tmp_path, body_model):
    mujoco = pytest.importorskip("mujoco")
    from weft.human import HumanEnv, build_human
    from test_spatial_pose import pose_frame

    frames = json.loads(json.dumps([dict(pose_frame(index), id=f"{HASH}:frame:{index}", frame=index) for index in range(10)]))
    source = {"schema_version": 1, "profile": "exercise-pose-v1", "source_sha256": HASH, "width": 640, "height": 480,
              "fps": 10, "frame_count": 10, "frames": frames,
              "lengths": {"torso": .5, "shoulder_width": .4, "hip_width": .26,
                          "upper_arm": .3, "forearm": .25, "thigh": .3, "shin": .25},
              "reference": [{"time_s": row["time_s"], "evidence_id": row["id"]} for row in frames]}
    env = HumanEnv(build_human(source, tmp_path, body_model=body_model))
    try:
        result = adapt_exercise(source, scene=env.scene)
        types = {"internal_hinge_joint": int(mujoco.mjtJoint.mjJNT_HINGE),
                 "root_slide_joint": int(mujoco.mjtJoint.mjJNT_SLIDE)}
        exported = {row["attributes"]["name"]: types[row["type"]] for row in result["relationships"] if row["type"] in types}
        compiled = {mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, index): int(env.model.jnt_type[index])
                    for index in range(env.model.njnt)}
        assert exported == compiled
        anchors = [row for row in result["relationships"] if row["type"] == "anchored_support"]
        assert len(anchors) == (0 if body_model == "mobile" else 1)
        if anchors:
            assert "articulated torso rotation" in anchors[0]["attributes"]["scope"]
        assert all(row["status"] == "assumed" and not row["evidence_ids"] for row in result["relationships"])
    finally:
        env.close()


def test_legacy_hinge_export_remains_exact():
    result = adapt_exercise(exercise(), scene={"profile": "exercise-v1"})
    expected = [{"id": f"joint-{side}-{joint}", "type": "internal_hinge_joint", "objects": ["person-0"],
                 "status": "assumed", "evidence_ids": [], "interval_s": None,
                 "attributes": {"name": f"{side}_{joint}", "scope": "Authored simulator topology, not recovered joint mechanics"}}
                for side in ("left", "right") for joint in ("shoulder", "elbow", "hip", "knee")]
    expected.append({"id": "anchored-pelvis", "type": "anchored_support", "objects": ["person-0"],
                     "status": "assumed", "evidence_ids": [], "interval_s": None,
                     "attributes": {"scope": "Authored fixed pelvis/torso, not observed support or ground contact"}})
    assert result["relationships"] == expected


@pytest.mark.parametrize("mutation,match", [
    (lambda s: s["human"].update(profile="exercise-v1"), "profiles conflict"),
    (lambda s: s["human"].update(source_sha256="b" * 64), "source identities"),
    (lambda s: s["human"]["joint_names"].pop(), "joint names"),
    (lambda s: s["joint_names"].__setitem__(0, "unknown_axis"), "joint names"),
    (lambda s: s["human"]["joint_names"].__setitem__(0, {}), "joint names"),
    (lambda s: s["human"]["mobility"].update(translation_axes=0), "mobility"),
    (lambda s: s["human"].update(mobility="fixed"), "mobility"),
    (lambda s: s.update(human=[]), "human metadata"),
])
def test_conflicting_profile_source_or_topology_cannot_be_exported(mutation, match):
    scene = mobile_scene()
    mutation(scene)
    with pytest.raises(ValueError, match=match):
        adapt_exercise(exercise(), scene=scene)


def test_controller_wrapper_retains_underlying_mobile_topology():
    scene = mobile_scene()
    scene["profile"] = "exercise-imitation-v1"
    result = adapt_exercise(exercise(), scene=scene)
    assert len(result["relationships"]) == 23
    assert not any(row["type"] == "anchored_support" for row in result["relationships"])


def test_unknown_body_profile_abstains_without_fabricating_fixed_topology():
    result = adapt_exercise(exercise(), scene={"profile": "future-body-v1"})
    assert not result["relationships"]
    assert any("No authored topology/support adapter" in text for text in result["limits"])
