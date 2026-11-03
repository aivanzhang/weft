"""Independent family-boundary negatives; no real-video processing or candidate stepping."""

from copy import deepcopy
from hashlib import sha256
import json
import math
import pytest

from weft.io import canonical_hash


@pytest.fixture(scope="module")
def admitted_family():
    from test_moving_equipment_proposal import packet
    from weft.moving_depth_family import prepare_moving_depth_family

    patches = pytest.MonkeyPatch()
    try:
        # Original detector/body admission is explicitly stubbed by this analytic fixture.
        inputs, _ = packet(patches, image_shift=(300, 190))
        settings = {"schema_version": 1, "profile": "moving-depth-family-priors-v1",
            "camera": {"distance_model_units": 2.}, "depth_domain_model_units": [.5, 3.5],
            "shape": {"method": "sphere_from_median_image_box_diagonal"}, "mass_kg": 1.,
            "friction": [.8, .01, .001], "initialization": {"method": "first_initialized_box_zero_velocity"}}
        original = dict(inputs.source_files)
        family = prepare_moving_depth_family(inputs, settings)
        assert inputs.source_files == original
        yield inputs, family
    finally:
        patches.undo()


def selection(family):
    document = family.document
    factor = next(row for row in document["factors"] if row["depth_intervals"]["certified_intervals"])
    low, high = factor["depth_intervals"]["certified_intervals"][0]
    return {"schema_version": 1, "profile": "moving-depth-selection-v1",
        "family_sha256": canonical_hash(document), "choices": [{"equipment": [{
            "object_id": factor["object_id"], "interval_index": 0, "depth_model_units": low + (high-low)/2}]}]}


def test_every_raw_identity_slot_and_independent_clock_remains(admitted_family):
    inputs, family = admitted_family
    document = family.document
    assert document["input_pins"]["admission_receipt_sha256"] == sha256(inputs.admission_receipt_bytes).hexdigest()
    assert document["input_pins"]["motion_evidence_sha256"] == sha256(inputs.motion_evidence_bytes).hexdigest()
    assert document["identity_family"]["raw_object_ids"] == ["raw-0", "raw-1"]
    assert len(document["factors"]) == 2
    for factor in document["factors"]:
        rows = factor["reference_track"]["rows"]
        assert [row["source_frame"] for row in rows] == list(range(10))
        assert rows[2]["source_time_s"] == .117 and rows[2]["body_time_s"] == .1
        assert rows[2]["phase_records"][0]["phase"] == "pre_initialization"
        assert rows[3]["position_model_units"] is None
        initial = factor["reference_track"]["initial_state_hypothesis"]
        assert initial["body_start_s"] == 0 and initial["equipment_source_start_s"] == .017
        assert initial["basis_frame"] == 4 and initial["source_slots_filled"] == 0
        assert not initial["physical_state_measured"]
        assert factor["depth_intervals"]["selected_depth"] is None


def test_prepared_authority_is_not_a_plain_document_or_another_packet(admitted_family):
    from weft.moving_depth_family import validate_prepared_family, materialize_moving_depth_selection
    inputs, family = admitted_family
    assert validate_prepared_family(family, inputs=inputs) is family
    with pytest.raises(ValueError): validate_prepared_family(family.document, inputs=inputs)
    with pytest.raises(ValueError): validate_prepared_family(family, inputs=object())
    with pytest.raises(ValueError): materialize_moving_depth_selection(family.document, selection(family))
    detached = family.document
    detached["factors"].clear()
    assert len(family.document["factors"]) == 2


@pytest.mark.parametrize("damage", ["source_pin", "primitive", "native_reference", "ray", "interval", "prefix", "drop_slot", "drop_identity", "promote_complement"])
def test_saved_family_rederivation_rejects_coordinated_metadata_changes(admitted_family, monkeypatch, damage):
    import weft.moving_depth_geometry as geometry
    import weft.moving_depth_family as family_module
    from weft import shared_world
    from weft.moving_depth_family import admit_moving_depth_family

    inputs, family = admitted_family
    original = family.document
    changed = deepcopy(original)
    factor = changed["factors"][0]
    if damage == "source_pin": changed["input_pins"]["motion_evidence_sha256"] = "0" * 64
    elif damage == "primitive": changed["geometry"]["primitives"][0]["radius"] *= .5
    elif damage == "native_reference": changed["geometry"]["source_reference"]["first"]["time_s"] += .01
    elif damage == "ray": factor["ray"]["center_intercept"][0] += .1
    elif damage == "interval": factor["depth_intervals"]["certified_intervals"] = [[.5, 3.5]]
    elif damage == "prefix": factor["reference_track"]["rows"][0]["phase_records"][0]["phase"] = "post_initialization"
    elif damage == "drop_slot": factor["reference_track"]["rows"].pop()
    elif damage == "drop_identity":
        changed["factors"].pop()
        changed["identity_family"]["raw_object_ids"].pop()
    else: changed["identity_family"]["complement_status"] = "verified_person"
    if damage in {"primitive", "native_reference"}:
        # Regenerate declared hashes too: source re-extraction, not stale-hash detection, must refuse.
        digest = sha256(json.dumps(changed["geometry"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for row in changed["factors"]:
            if row["depth_intervals"] is not None: row["depth_intervals"]["geometry_sha256"] = digest
    if damage == "ray":
        factor["depth_intervals"]["ray_sha256"] = sha256(json.dumps(factor["ray"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # Reuse the once-compiled original analytic geometry; admission still rederives all source factors.
    calls = []
    def extract(packet, **kwargs):
        assert packet is inputs
        calls.append(1)
        return deepcopy(original["geometry"])
    monkeypatch.setattr(geometry, "extract_depth_geometry", extract)
    if hasattr(family_module, "extract_depth_geometry"):
        monkeypatch.setattr(family_module, "extract_depth_geometry", extract)
    monkeypatch.setattr(shared_world, "SharedWorldEnv", lambda *a, **k: pytest.fail("Reuse original analytic geometry; no repeated native compilation"))
    with pytest.raises(ValueError): admit_moving_depth_family(inputs, changed)
    assert len(calls) <= 1
    assert family.document == original


@pytest.mark.parametrize("damage", ["family_hash", "unknown_id", "raw_id_alias", "wrong_interval", "outside", "just_outside", "duplicate_object", "bool_depth", "bool_interval", "empty_choice", "duplicate_choice", "too_many_choices", "extra_field"])
def test_selection_cannot_promote_unbound_or_uncertified_choices(admitted_family, damage):
    from weft.moving_depth_family import materialize_moving_depth_selection
    _, family = admitted_family
    chosen = selection(family)
    equipment = chosen["choices"][0]["equipment"]
    if damage == "family_hash": chosen["family_sha256"] = "f" * 64
    elif damage == "unknown_id": equipment[0]["object_id"] = "discarded-track"
    elif damage == "raw_id_alias": equipment[0]["object_id"] = "sam-1"
    elif damage == "wrong_interval": equipment[0]["interval_index"] = 512
    elif damage == "outside": equipment[0]["depth_model_units"] = 100.
    elif damage == "just_outside":
        factor = next(row for row in family.document["factors"] if row["object_id"] == equipment[0]["object_id"])
        low = factor["depth_intervals"]["certified_intervals"][0][0]
        equipment[0]["depth_model_units"] = math.nextafter(low, -math.inf)
    elif damage == "duplicate_object": equipment.append(deepcopy(equipment[0]))
    elif damage == "bool_depth": equipment[0]["depth_model_units"] = True
    elif damage == "bool_interval": equipment[0]["interval_index"] = False
    elif damage == "empty_choice": equipment.clear()
    elif damage == "duplicate_choice": chosen["choices"] *= 2
    elif damage == "too_many_choices": chosen["choices"] *= 4
    else: chosen["automatic_selection"] = True
    with pytest.raises(ValueError): materialize_moving_depth_selection(family, chosen)


def test_selected_bundle_keeps_family_and_constructor_requires_original_capability(admitted_family, monkeypatch):
    from weft.moving_depth_family import materialize_moving_depth_selection
    from weft.moving_equipment_world import construct_moving_equipment_world
    from weft import shared_world_reconstruction
    inputs, family = admitted_family
    original = family.document
    chosen = selection(family)
    bundle = materialize_moving_depth_selection(family, chosen)
    assert family.document == original
    assert bundle["depth_family"] == original and bundle["depth_selection"] == chosen
    assert len(bundle["candidates"]) == 1
    assert len(bundle["track_proposals"]) >= 1
    assignment = bundle["candidates"][0]["identity_assignment"]
    assert assignment["physical_identity_verified"] is False
    assert set(assignment["equipment_ids"]) | set(assignment["unknown_complement_ids"]) == {"raw-0", "raw-1"}
    monkeypatch.setattr(shared_world_reconstruction, "_check_initial_geometry", lambda *a, **k: pytest.fail("No native gate before exact family selection admission"))
    candidate = bundle["candidates"][0]["id"]
    with pytest.raises(ValueError): construct_moving_equipment_world(inputs, bundle, candidate)
    changed = deepcopy(bundle)
    changed["track_proposals"][0]["initial_state_hypothesis"]["position_model_units"][0] += .01
    with pytest.raises(ValueError): construct_moving_equipment_world(inputs, changed, candidate, depth_family=family)


@pytest.mark.parametrize("prefix", [b'{"profile":"discarded-profile",', b'{"discarded":NaN,'])
def test_ambiguous_or_nonfinite_saved_bytes_reject_before_geometry(admitted_family, monkeypatch, prefix):
    from weft.moving_depth_family import admit_moving_depth_family
    import weft.moving_depth_geometry as geometry
    inputs, family = admitted_family
    raw = prefix + family.document_bytes[1:]
    monkeypatch.setattr(geometry, "extract_depth_geometry", lambda *a, **k: pytest.fail("Malformed saved bytes must reject before native extraction"))
    with pytest.raises(ValueError): admit_moving_depth_family(inputs, raw)
