"""Analytical source contract; original detector admission is explicitly fixture-stubbed."""

from copy import deepcopy
import math
import time

import pytest

from test_moving_equipment_proposal import packet, settings as legacy_settings
from weft.io import canonical_hash
from weft.moving_depth_family import (prepare_moving_depth_family, admit_moving_depth_family,
    materialize_moving_depth_selection, validate_moving_depth_settings)
from weft.moving_equipment_world import construct_moving_equipment_world


def settings():
    old = legacy_settings(); old.pop("depth_offsets_model_units")
    return {**old, "profile": "moving-depth-family-priors-v1", "depth_domain_model_units": [.1, 4.]}


def choice(prepared, ids):
    family = prepared.document
    factors = {row["object_id"]: row for row in family["factors"]}
    equipment = []
    for name in ids:
        low, high = factors[name]["depth_intervals"]["certified_intervals"][0]
        equipment.append({"object_id": name, "interval_index": 0, "depth_model_units": low + (high - low) / 2})
    return {"schema_version": 1, "profile": "moving-depth-selection-v1", "family_sha256": canonical_hash(family),
            "choices": [{"equipment": equipment}]}


@pytest.fixture(scope="module")
def admitted():
    patch = pytest.MonkeyPatch()
    try:
        inputs, calls = packet(patch, raw_tracks=3, image_shift=(300, 190))
        prepared = prepare_moving_depth_family(inputs, settings())
        yield inputs, prepared, calls
    finally:
        patch.undo()


def test_factorized_inventory_does_not_require_flat_seven_worlds(admitted):
    inputs, prepared, calls = admitted
    family = prepared.document
    assert len(calls) == 1 and family["identity_family"]["assignment_count"] == 7
    assert family["identity_family"]["materialized_assignment_count"] == 0
    assert family["identity_family"]["raw_object_ids"] == ["raw-0", "raw-1", "raw-2"]
    assert len(family["factors"]) == 3 and all(len(f["reference_track"]["rows"]) == 10 for f in family["factors"])
    for factor in family["factors"]:
        assert factor["status"] == "feasible_intervals"
        initial = factor["reference_track"]["initial_state_hypothesis"]
        assert initial["unknown_prefix_slot_count"] == 4 and initial["source_slots_filled"] == 0
        assert initial["body_start_s"] == 0 and initial["equipment_source_start_s"] == .017
    selection = choice(prepared, ["raw-0"])
    bundle = materialize_moving_depth_selection(prepared, selection)
    before = deepcopy(bundle)
    scene = construct_moving_equipment_world(inputs, bundle, "candidate-1", depth_family=prepared)
    assert before == bundle and len(calls) == 1
    assert len(bundle["candidates"]) == 1 and bundle["depth_family"] == family
    assert bundle["candidates"][0]["identity_assignment"]["unknown_complement_ids"] == ["raw-1", "raw-2"]
    for key in ("human_spec", "coordinate_frame", "dt", "substeps", "bounds", "gravity", "contact"):
        assert scene[key] == inputs.base_scene[key]
    provenance = scene["objects"][0]["provenance"]
    assert provenance["depth_assumption"]["depth_model_units"] == selection["choices"][0]["equipment"][0]["depth_model_units"]
    assert provenance["depth_assumption"]["status"] == "explicit_authored_depth_not_measured"
    assert provenance["initial_state_hypothesis"]["initial_state_status"] == "authored_prefix_hypothesis"
    assert provenance["initial_geometry_checks"][0]["physics_steps"] == 0
    assert not provenance["trajectory_driving"] and not provenance["task_success_available"]


def test_three_explicit_materializations_retain_full_family_without_geometry_repeats(admitted, monkeypatch):
    inputs, prepared, _ = admitted
    import weft.moving_depth_geometry as geometry
    monkeypatch.setattr(geometry, "extract_depth_geometry", lambda *_a, **_k: pytest.fail("Reuse prepared geometry"))
    selection = choice(prepared, ["raw-0"])
    selection["choices"] += [choice(prepared, [name])["choices"][0] for name in ("raw-1", "raw-2")]
    bundle = materialize_moving_depth_selection(prepared, selection)
    assert len(bundle["candidates"]) == 3 and len(bundle["track_proposals"]) == 3
    assert bundle["depth_family"] == prepared.document
    assert bundle["identity_family"]["assignment_count"] == 7
    assert [row["id"] for row in bundle["candidates"]] == ["candidate-1", "candidate-2", "candidate-3"]


def test_saved_family_requires_one_fresh_geometry_extraction_and_exact_rederivation(admitted, monkeypatch):
    inputs, prepared, calls = admitted
    import weft.moving_depth_geometry as geometry
    original, extracted = geometry.extract_depth_geometry, []
    def capture(*args, **kwargs):
        extracted.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(geometry, "extract_depth_geometry", capture)
    restored = admit_moving_depth_family(inputs, prepared.document_bytes)
    assert extracted == [1] and len(calls) == 1
    assert restored.document_bytes == prepared.document_bytes
    selection = choice(prepared, ["raw-0"])
    assert materialize_moving_depth_selection(restored, selection) == materialize_moving_depth_selection(prepared, selection)


def test_joint_assignment_still_checks_pairwise_native_collisions(admitted):
    inputs, prepared, _ = admitted
    bundle = materialize_moving_depth_selection(prepared, choice(prepared, ["raw-0", "raw-1"]))
    before = deepcopy(bundle)
    with pytest.raises(ValueError, match="intersect|overlap"):
        construct_moving_equipment_world(inputs, bundle, "candidate-1", depth_family=prepared)
    assert bundle == before and len(bundle["depth_family"]["factors"]) == 3


def test_selected_depth_is_not_subtract_add_roundtrip(admitted):
    inputs, prepared, _ = admitted
    from weft.moving_equipment_proposal import _track_proposal
    from weft.moving_depth_family import _source_parts, _legacy_settings
    priors = _legacy_settings(prepared.document["settings"])
    _, camera, tracks = _source_parts(inputs, priors)
    explicit = .1
    assert camera["distance_model_units"] + (explicit - camera["distance_model_units"]) != explicit
    first = inputs.body_references[0]
    result = _track_proposal(tracks["raw-0"], "raw-0", 0, explicit - camera["distance_model_units"], camera,
        priors, first["time_s"], first["source_frame"], time.monotonic() + 5, depth_model_units=explicit)
    slot = tracks["raw-0"]["slots"][4]
    x, y, width, height = slot["measurements"]["box"]["value"]
    assert result["rows"][4]["box_radius_model_units"] == math.hypot(width, height) * explicit / (2 * camera["focal_px"])


@pytest.mark.parametrize("change", ["domain_low", "domain_offset", "free_state", "missing", "extra", "bool"])
def test_explicit_settings_reject_unsupported_assumptions_before_native(change, monkeypatch):
    import weft.moving_depth_geometry as geometry
    monkeypatch.setattr(geometry, "extract_depth_geometry", lambda *_a, **_k: pytest.fail("No native work"))
    value = settings()
    if change == "domain_low": value["depth_domain_model_units"][0] = .05
    elif change == "domain_offset": value["depth_domain_model_units"][1] = 4.1
    elif change == "free_state": value["initialization"] = legacy_settings(free=True)["initialization"]
    elif change == "missing": value.pop("camera")
    elif change == "extra": value["chosen_winner"] = "raw-0"
    else: value["depth_domain_model_units"][0] = True
    with pytest.raises(ValueError): validate_moving_depth_settings(value)
