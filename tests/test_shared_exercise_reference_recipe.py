"""Compact execution recipes do not replace sealed source-fit qualification."""

from copy import deepcopy
import json

import numpy as np
import pytest

from test_shared_exercise_reference_controller import (
    FakeEnvironment, controller_fixture, encode, forbid_native_and_media,
    prepare, refresh_hashes, velocity_index,
)
from weft.io import canonical_hash
from weft import shared_exercise_reference_controller as controller


def replay(bundle, document, velocity_index, **kwargs):
    return controller.prepare_reference_recipe(document, scene=bundle["scene"], task=bundle["task"],
        action_contract=bundle["action_contract"], runtime_sha256=kwargs.get("runtime_sha256", dict(controller.RUNTIME_SHA256)),
        reference_velocity_index=velocity_index)


def rehash_candidate(bundle, document):
    candidate = deepcopy(bundle["scene"]["human_spec"])
    for row, target in zip(candidate["reference"], document["targets"], strict=True):
        row["joints"] = deepcopy(target["joints"])
    candidate["whole_body_reference_refinement"] = deepcopy(document["refinement_marker"])
    document["candidate_canonical_sha256"] = canonical_hash(candidate)


@pytest.mark.parametrize("root_assistance", [False, True])
@pytest.mark.parametrize("corrections", [(0., 0., 0., 0.), (0., .04, .01, .03)])
def test_recipe_roundtrip_preserves_exact_actions_not_source_fit_qualification(velocity_index, root_assistance, corrections):
    b = controller_fixture(root_assistance=root_assistance, corrections=corrections)
    original = prepare(b, velocity_index)
    document = original.recipe()
    before = deepcopy((document, b))
    archived = replay(b, document, velocity_index)
    assert archived.recipe() == document
    assert archived.describe()["qualification"] == controller.RECIPE_INTERPRETATION
    assert original.describe()["qualification"] == "saved_passed_proposal_boundary_checked"
    assert archived.describe()["proposal_report_sha256"] == original.describe()["proposal_report_sha256"]
    assert archived.describe()["candidate_canonical_sha256"] == canonical_hash(b["candidate"])
    assert "gates" not in document and "human_spec" not in document
    env = FakeEnvironment(b, original)
    env.features[:] = np.arange(env.features.size).reshape(env.features.shape) * .037 - .7
    for stamp in (0., .05, .1, .15, .2, .3):
        env.state["simulated_duration_s"] = stamp
        assert archived.action(env).tobytes() == original.action(env).tobytes()
    assert (document, b) == before
    document["targets"][1]["joints"]["left_knee"] = 99.
    assert archived.recipe() == before[0]


def test_public_action_contract_matches_existing_shape_and_detaches(velocity_index):
    for assistance in (True, False):
        b = controller_fixture(root_assistance=assistance)
        action = controller.reference_action_contract(b["scene"], b["task"])
        assert action == b["action_contract"]
        action["physical_limits"][0] = 999.
        assert b["scene"]["action_contract"]["physical_limits"][0] == 30.
    b["task"]["control"]["object_forces"] = "allowed"
    with pytest.raises(ValueError):
        controller.reference_action_contract(b["scene"], b["task"])


@pytest.mark.parametrize("kind", ["extra", "missing", "profile", "interpretation", "schema_bool", "scene", "task",
    "gates", "action", "original", "source", "candidate", "target", "target_missing_joint", "target_extra",
    "target_time", "target_evidence", "target_count", "marker_extra", "marker_promoted", "marker_input",
    "runtime", "runtime_extra", "gains", "gain_numeric_type", "clock", "raw_hash_format"])
def test_recipe_tampering_and_unknown_forms_reject(velocity_index, kind):
    b = controller_fixture()
    doc = prepare(b, velocity_index).recipe()
    if kind == "extra": doc["human_spec"] = {}
    elif kind == "missing": del doc["interpretation"]
    elif kind == "profile": doc["profile"] = "shared-exercise-policy-v1"
    elif kind == "interpretation": doc["interpretation"] = "source_fit_qualified"
    elif kind == "schema_bool": doc["schema_version"] = True
    elif kind in {"scene", "task", "gates", "source"}: doc[kind + "_sha256"] = "d" * 64
    elif kind == "action": doc["action_contract_sha256"] = "d" * 64
    elif kind == "original": doc["original_human_canonical_sha256"] = "d" * 64
    elif kind == "candidate": doc["candidate_canonical_sha256"] = "d" * 64
    elif kind == "target": doc["targets"][1]["joints"]["left_knee"] += .001
    elif kind == "target_missing_joint": del doc["targets"][1]["joints"]["left_knee"]
    elif kind == "target_extra": doc["targets"][1]["confidence"] = 1.
    elif kind == "target_time": doc["targets"][1]["time_s"] += .001
    elif kind == "target_evidence": doc["targets"][1]["evidence_id"] = doc["targets"][0]["evidence_id"]
    elif kind == "target_count": doc["targets"].pop()
    elif kind == "marker_extra": doc["refinement_marker"]["gates_passed"] = True
    elif kind == "marker_promoted": doc["refinement_marker"]["promoted"] = True
    elif kind == "marker_input": doc["refinement_marker"]["input_human_sha256"] = "d" * 64
    elif kind == "runtime": doc["runtime_sha256"]["human.py"] = "d" * 64
    elif kind == "runtime_extra": doc["runtime_sha256"]["unknown.py"] = "d" * 64
    elif kind == "gains": doc["feedback_gains"]["joint_position"] = 41.
    elif kind == "gain_numeric_type": doc["feedback_gains"]["joint_position"] = 40
    elif kind == "clock": doc["clock_binding"]["source_time_offset_s"] = 0.
    elif kind == "raw_hash_format": doc["proposal_report_sha256"] = "unknown"
    with pytest.raises(ValueError):
        replay(b, doc, velocity_index)


@pytest.mark.parametrize("kind", ["first_pose", "angle_bound", "rate_bound", "nan", "unknown_clock", "runtime_argument"])
def test_rehashed_recipes_still_obey_executable_constraints(velocity_index, kind):
    b = controller_fixture()
    doc = prepare(b, velocity_index).recipe()
    runtime = dict(controller.RUNTIME_SHA256)
    if kind == "first_pose": doc["targets"][0]["joints"]["left_knee"] += .01
    elif kind == "angle_bound": doc["targets"][1]["joints"]["left_knee"] += .2
    elif kind == "rate_bound":
        doc["targets"][1]["joints"]["left_knee"] = b["original"]["reference"][1]["joints"]["left_knee"] + .15
        doc["targets"][2]["joints"]["left_knee"] = b["original"]["reference"][2]["joints"]["left_knee"] - .15
    elif kind == "nan": doc["targets"][1]["joints"]["left_knee"] = float("nan")
    elif kind == "unknown_clock":
        clock = b["scene"]["coordinate_frame"]["clock_binding"]
        clock["alignment_method"] = "native_decoder_pts"
        b["task"]["scene_sha256"] = canonical_hash(b["scene"])
        doc.update(scene_sha256=canonical_hash(b["scene"]), task_sha256=canonical_hash(b["task"]), clock_binding=deepcopy(clock))
    else: runtime["shared_world.py"] = "d" * 64
    if kind != "nan":
        rehash_candidate(b, doc)
    with pytest.raises(ValueError):
        replay(b, doc, velocity_index, runtime_sha256=runtime)


def test_recipe_raw_hashes_are_identifiers_not_a_new_artifact_authentication(velocity_index):
    b = controller_fixture()
    doc = prepare(b, velocity_index).recipe()
    # Only the containing recording seal can authenticate this otherwise opaque raw-file ID.
    doc["proposal_report_sha256"] = "d" * 64
    result = replay(b, doc, velocity_index)
    assert result.describe()["proposal_report_sha256"] == "d" * 64
    assert result.describe()["qualification"] == controller.RECIPE_INTERPRETATION


def large_recipe_fixture():
    b = controller_fixture(corrections=(0.,) * 4)
    original = b["original"]
    source_sha = original["source_sha256"]
    frames = [6960 + 8 * i for i in range(900)]
    original["reference"] = [{"time_s": frame / 240., "joints": {
        name: 2.5555555555555554 if name.endswith(("_knee", "_elbow")) else -5.555555555555555
        for name in controller.SPATIAL_JOINT_NAMES}} for frame in frames]
    original["reference_evidence"] = [{"evidence_id": f"{source_sha}:frame:{frame}", "time_s": frame / 240.}
                                      for frame in frames]
    original["root_reference"] = [{"time_s": frame / 240., "translation": [0., 0., 0.]} for frame in frames]
    marker = b["candidate"]["whole_body_reference_refinement"]
    b["candidate"] = deepcopy(original) | {"whole_body_reference_refinement": marker}
    b["scene"]["human_spec"] = deepcopy(original)
    b["scene"]["coordinate_frame"]["clock_binding"].update(source_fps=240., source_time_offset_s=29.)
    b["scene"]["task"]["horizon_steps"] = b["task"]["interval"]["horizon_steps"] = 899
    b["report"]["references"] = [{"frame": frame, "evidence_id": f"{source_sha}:frame:{frame}", "time_s": frame / 240.}
                                  for frame in frames]
    refresh_hashes(b)
    return b


def test_nine_hundred_precision_heavy_targets_fit_deliberate_one_mib_bound(velocity_index, record_property):
    b = large_recipe_fixture()
    provider = prepare(b, velocity_index)
    document = provider.recipe()
    compact_size = len(json.dumps(document, separators=(",", ":"), sort_keys=True).encode())
    default_size = len(encode(document))
    assert len(document["targets"]) == 900
    assert 256 * 1024 < compact_size <= default_size < controller.MAX_RECIPE_BYTES == 1 << 20
    assert len(replay(b, document, velocity_index).recipe()["targets"]) == 900
    record_property("900_reference_compact_bytes", compact_size)
    record_property("900_reference_default_json_bytes", default_size)
    record_property("recipe_byte_limit", controller.MAX_RECIPE_BYTES)


def test_recipe_oversize_and_excess_rows_reject_before_execution(velocity_index):
    b = controller_fixture()
    document = prepare(b, velocity_index).recipe()
    with pytest.raises(ValueError, match="bounded"):
        replay(b, document | {"padding": "x" * controller.MAX_RECIPE_BYTES}, velocity_index)
    document["targets"] *= 226
    with pytest.raises(ValueError, match="reference count"):
        replay(b, document, velocity_index)


def test_archived_recipe_preparation_never_calls_proposal_qualification(velocity_index, monkeypatch):
    b = controller_fixture()
    document = prepare(b, velocity_index).recipe()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Replay must not fabricate or qualify a source-fit report")
    monkeypatch.setattr(controller, "_prepare", forbidden)
    assert replay(b, document, velocity_index).recipe() == document
