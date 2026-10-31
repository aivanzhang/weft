import json

import pytest

from weft.person_inventory import make_person_inventory, validate_person_inventory


SOURCE = "a" * 64


def test_explicit_scope_is_source_bound_and_never_claims_verified_absence():
    value = make_person_inventory(SOURCE)
    checked = validate_person_inventory(value, source_sha256=SOURCE)
    assert checked == value and checked is not value
    assert checked["basis"] == "user_declared"
    assert checked["selected_entity_id"] == "human"
    assert checked["support"] == "ground_only"
    assert checked["absence_verified"] is False
    assert json.loads(json.dumps(checked)) == checked
    with pytest.raises(ValueError):
        validate_person_inventory(value, source_sha256="b" * 64)


def test_analytic_inventory_cannot_masquerade_as_recorded_evidence():
    value = make_person_inventory(None, analytic=True)
    assert value["basis"] == "analytic_fixture" and value["source_sha256"] is None
    assert validate_person_inventory(value, source_sha256=None, analytic=True) == value
    with pytest.raises(ValueError):
        validate_person_inventory(value, source_sha256=SOURCE)
    with pytest.raises(ValueError):
        make_person_inventory(SOURCE, analytic=True)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("absence_verified", 0), ("absence_verified", True),
    ("profile", "human-only-inventory-v2"), ("basis", "detector_verified"),
    ("selected_entity_id", "person-1"), ("support", "chair"), ("support", "unknown"),
    ("task_relevant_equipment", "unknown"), ("task_relevant_equipment", ["weight"]),
    ("task_relevant_obstacles", "none_verified"), ("source_sha256", None),
    ("source_sha256", "b" * 64), ("support", "x" * 4097), ("support", float("nan")),
])
def test_missing_unknown_or_contradictory_inventory_is_rejected(field, value):
    inventory = make_person_inventory(SOURCE)
    inventory[field] = value
    with pytest.raises(ValueError):
        validate_person_inventory(inventory, source_sha256=SOURCE)


@pytest.mark.parametrize("value", [None, [], {}, {"ignored_equipment": []}])
def test_incomplete_or_extra_fields_cannot_change_declared_scope(value):
    if value == {"ignored_equipment": []}:
        value = make_person_inventory(SOURCE) | value
    with pytest.raises(ValueError):
        validate_person_inventory(value, source_sha256=SOURCE)


@pytest.mark.parametrize("source", [None, "", "A" * 64, "a" * 63, 3, True])
def test_nonanalytic_source_requires_exact_hash(source):
    with pytest.raises(ValueError):
        make_person_inventory(source)


@pytest.mark.parametrize("analytic", [1, 0, "false", None])
def test_analytic_flag_is_not_truthy_coercion(analytic):
    with pytest.raises(ValueError):
        make_person_inventory(SOURCE, analytic=analytic)
