"""Saved native-capture selection preserves the generator's exact source projection."""

from copy import deepcopy
from hashlib import sha256

import pytest

from test_native_capture_custody import captured, package
from weft import shared_exercise_selection as selection
from weft import shared_exercise_residual_learning as generator
from weft.shared_exercise_lab import NATIVE_CAPTURE_VIEWS, NATIVE_SOURCE_VIEWS


def test_selection_matches_generator_projection_without_dropping_capture_proofs(tmp_path, captured):
    files, scene = captured
    original = deepcopy(files)
    root = tmp_path / "source"
    expected_views = generator._source_views(root, package(root, files), scene)
    expected = {name: sha256(raw).hexdigest() for name, raw in expected_views.items()}
    actual = selection._source_view_hashes(files, scene)
    assert actual == expected
    assert set(actual) == NATIVE_SOURCE_VIEWS | NATIVE_CAPTURE_VIEWS | {"body-source.json"}
    assert not set(scene["source_binding"]["evidence_documents"].values()) & actual.keys()
    assert files == original


@pytest.mark.parametrize("damage", ["proof", "receipt", "marker", "raw", "derived"])
def test_invalid_native_custody_is_not_hidden_by_projection(captured, damage):
    files, scene = captured
    name = {"receipt": "native-clock-capture.json", "marker": "body-source.json",
            "raw": "evidence.json", "derived": "reconstruction-evidence.json"}.get(damage)
    if damage == "proof":
        name = next(iter(scene["source_binding"]["evidence_documents"].values()))
    files.pop(name)
    with pytest.raises(ValueError):
        selection._source_view_hashes(files, scene)


def test_legacy_source_projection_remains_unchanged():
    scene = {"profile": "shared-world-v3", "human_spec": {}}
    files = {"source.webm": b"source", "source-clock.json": b"clock", "inventory.json": b"inventory",
             "other.json": b"not a source view"}
    assert selection._source_view_hashes(files, scene) == {
        name: sha256(raw).hexdigest() for name, raw in files.items() if name != "other.json"}
