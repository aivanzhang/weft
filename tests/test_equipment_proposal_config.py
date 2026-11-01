"""Pure compatibility checks; no model admission, geometry, solver or source work."""

import ast
import builtins
from copy import deepcopy
import importlib
import math
import os
from pathlib import Path
import re
import socket
import subprocess

import numpy as np
import pytest


PACKAGE = Path(__file__).resolve().parents[1] / "src/weft"
# Frozen pre-extraction behavior, not a second production validator.
_BASELINE = '''class EquipmentProposalError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def validate_proposal_config(config):
    required = {'schema_version', 'profile', 'source_sha256', 'camera', 'geometry_prior', 'equipment', 'target_object_id'}
    if (not isinstance(config, dict) or not required <= set(config) or set(config) - required - {'frame_translation_xy'} or
            type(config['schema_version']) is not int or config['schema_version'] != 1 or config['profile'] != CONFIG_PROFILE):
        raise EquipmentProposalError('unsupported_config', 'Use exactly the versioned source-equipment proposal configuration')
    if not isinstance(config['source_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', config['source_sha256']):
        raise EquipmentProposalError('invalid_source_identity', 'Configuration must pin the original video SHA256')
    camera = config['camera']
    if (not isinstance(camera, dict) or set(camera) != {'mode', 'distance_model_units'} or
            camera['mode'] != 'fixed_pelvis_perspective_prior'):
        raise EquipmentProposalError('unsupported_camera', 'Only an explicitly authored fixed-pelvis perspective prior is supported')
    distance = camera['distance_model_units']
    if type(distance) not in (int, float) or not math.isfinite(distance) or not .1 <= distance <= 50:
        raise EquipmentProposalError('invalid_camera_distance', 'Camera distance must be between 0.1 and 50 original body-model units')
    if config['geometry_prior'] != 'floor_box_depth_equals_width':
        raise EquipmentProposalError('unsupported_geometry_prior', 'Only unchanged-floor solid boxes with depth equal to width are supported')
    objects = config['equipment']
    if not isinstance(objects, list) or not 1 <= len(objects) <= CONTRACT['maximum_equipment']:
        raise EquipmentProposalError('invalid_equipment_selection', 'Select one to twelve equipment observation entities')
    ids = set()
    for obj in objects:
        if not isinstance(obj, dict) or set(obj) != {'object_id', 'source_frame'}:
            raise EquipmentProposalError('invalid_equipment_selection', 'Each equipment selection requires object_id and source_frame only')
        name, frame = obj['object_id'], obj['source_frame']
        if (not isinstance(name, str) or not 1 <= len(name) <= 80 or name in ids or name == 'human' or
                type(frame) is not int or not 0 <= frame < 1800):
            raise EquipmentProposalError('invalid_equipment_selection', 'Equipment identities must be unique and source frames explicit bounded integers')
        ids.add(name)
    if config['target_object_id'] not in ids:
        raise EquipmentProposalError('invalid_target', 'The authored task target must be one of the selected equipment IDs')
    translation = config.get('frame_translation_xy')
    if translation is not None:
        values = _vector(translation, 2, 'Explicit positive XY frame translation', 0., 6.)
        if not np.any(values > 0):
            raise EquipmentProposalError('invalid_frame_translation', 'Omit frame translation or supply a nonzero positive XY vector')
    return deepcopy(config)'''


def _deny(*args, **kwargs):
    raise AssertionError("Models, media, network and child processes are forbidden")


@pytest.fixture(autouse=True)
def local_only(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if (name.split(".")[0] in {"mujoco", "cv2", "PIL", "mediapipe", "torch", "requests", "httpx"}
                or name in {"weft.human", "weft.shared_world", "weft.equipment_proposal", "weft.shared_world_reconstruction"}):
            _deny()
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    for owner, names in ((socket, ("socket", "create_connection")),
                         (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
                         (os, ("system", "popen", "fork", "posix_spawn", "posix_spawnp"))):
        for name in names:
            if hasattr(owner, name): monkeypatch.setattr(owner, name, _deny)


@pytest.fixture
def api():
    return importlib.import_module("weft.equipment_proposal_config")


@pytest.fixture(scope="module")
def original_validator():
    source = (PACKAGE / "shared_world_reconstruction.py").read_text()
    vector = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_vector")
    namespace = {"deepcopy": deepcopy, "math": math, "re": re, "np": np,
                 "CONFIG_PROFILE": "source-equipment-proposal-config-v1", "CONTRACT": {"maximum_equipment": 12}}
    exec(compile(ast.Module(body=[vector], type_ignores=[]), "original_vector", "exec"), namespace)
    exec(compile(_BASELINE, "frozen_original_validator", "exec"), namespace)
    return namespace["validate_proposal_config"]


def config():
    return {"schema_version": 1, "profile": "source-equipment-proposal-config-v1", "source_sha256": "a" * 64,
            "camera": {"mode": "fixed_pelvis_perspective_prior", "distance_model_units": 2.},
            "geometry_prior": "floor_box_depth_equals_width",
            "equipment": [{"object_id": "weight", "source_frame": 12}], "target_object_id": "weight"}


def outcome(validate, value):
    try:
        return "accepted", validate(value)
    except (ValueError, TypeError, OverflowError) as error:
        return "rejected", type(error).__name__, str(error), getattr(error, "code", None)


@pytest.mark.parametrize("case", [
    "valid", "twelve_objects", "missing", "extra", "boolean_schema", "uppercase_source", "camera_shape",
    "camera_mode", "distance_min", "distance_max", "distance_boolean", "distance_nan", "distance_overflow",
    "geometry", "empty_objects", "too_many_objects", "duplicate_object", "reserved_human", "object_shape",
    "frame_boolean", "frame_upper", "target_missing", "target_unhashable",
])
def test_frozen_validator_equivalence(api, original_validator, case):
    value = config()
    if case == "twelve_objects":
        value["equipment"] = [{"object_id": str(i).zfill(80), "source_frame": 1799} for i in range(12)]
        value["target_object_id"] = value["equipment"][-1]["object_id"]
    elif case == "missing": del value["source_sha256"]
    elif case == "extra": value["assume_contact"] = True
    elif case == "boolean_schema": value["schema_version"] = True
    elif case == "uppercase_source": value["source_sha256"] = "A" * 64
    elif case == "camera_shape": value["camera"]["extra"] = None
    elif case == "camera_mode": value["camera"]["mode"] = "calibrated_intrinsics"
    elif case.startswith("distance_"):
        value["camera"]["distance_model_units"] = {"distance_min": .1, "distance_max": 50,
            "distance_boolean": True, "distance_nan": math.nan, "distance_overflow": 10 ** 400}[case]
    elif case == "geometry": value["geometry_prior"] = "sphere"
    elif case == "empty_objects": value["equipment"] = []
    elif case == "too_many_objects": value["equipment"] = [{"object_id": str(i), "source_frame": 0} for i in range(13)]
    elif case == "duplicate_object": value["equipment"] *= 2
    elif case == "reserved_human": value["equipment"][0]["object_id"] = "human"
    elif case == "object_shape": value["equipment"][0]["contact"] = True
    elif case == "frame_boolean": value["equipment"][0]["source_frame"] = True
    elif case == "frame_upper": value["equipment"][0]["source_frame"] = 1800
    elif case == "target_missing": value["target_object_id"] = "other"
    elif case == "target_unhashable": value["target_object_id"] = []
    before = repr(value)
    assert outcome(api.validate_proposal_config, value) == outcome(original_validator, value)
    assert repr(value) == before


@pytest.mark.parametrize("translation", [None, [0, .1], [6., 6], [5e-324, -0.], [0., -0.], (1., 0.),
    [1.], [True, 0], [math.nan, 0], [math.inf, 0], [-.1, 0], [6.000000000000001, 0], [10 ** 400, 0], [np.float64(1), 0]])
def test_translation_keeps_original_types_bounds_and_exception_semantics(api, original_validator, translation):
    value = config(); value["frame_translation_xy"] = translation
    assert outcome(api.validate_proposal_config, value) == outcome(original_validator, value)


def test_invalid_coordinates_remain_plain_value_error_and_zero_keeps_named_error(api):
    value = config(); value["frame_translation_xy"] = [-.1, 0]
    with pytest.raises(ValueError) as caught: api.validate_proposal_config(value)
    assert type(caught.value) is ValueError
    assert str(caught.value) == "Explicit positive XY frame translation requires 2 finite bounded coordinates"
    value["frame_translation_xy"] = [0, 0]
    with pytest.raises(api.EquipmentProposalError) as caught: api.validate_proposal_config(value)
    assert caught.value.code == "invalid_frame_translation"


def test_valid_result_is_detached_without_numeric_normalization(api):
    value = config(); value["camera"]["distance_model_units"] = 50
    value["frame_translation_xy"] = [0, 6]
    result = api.validate_proposal_config(value)
    assert result == value and type(result["camera"]["distance_model_units"]) is int
    result["equipment"][0]["source_frame"] = 99; result["frame_translation_xy"][0] = 4
    assert value["equipment"][0]["source_frame"] == 12 and value["frame_translation_xy"] == [0, 6]


def test_fresh_module_needs_only_stdlib(monkeypatch):
    source = (PACKAGE / "equipment_proposal_config.py").read_text()
    imports = {n.module for n in ast.parse(source).body if isinstance(n, ast.ImportFrom)}
    imports |= {a.name for n in ast.parse(source).body if isinstance(n, ast.Import) for a in n.names}
    assert imports == {"copy", "math", "re"}
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"weft", "numpy", "mujoco", "cv2", "PIL"}: _deny()
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    namespace = {"__name__": "isolated_pure_config"}
    exec(compile(source, "equipment_proposal_config.py", "exec"), namespace)
    assert namespace["validate_proposal_config"](config()) == config()


def test_legacy_aliases_and_both_snapshot_dependencies_are_preserved(api):
    tree = ast.parse((PACKAGE / "equipment_proposal.py").read_text())
    binding = next(n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "weft.equipment_proposal_config")
    namespace = {}
    exec(compile(ast.Module(body=[binding], type_ignores=[]), "public_aliases", "exec"), namespace)
    for name in ("CONFIG_PROFILE", "MAX_CONFIG_BYTES", "MAX_EQUIPMENT", "EquipmentProposalError", "validate_proposal_config"):
        assert namespace[name] is getattr(api, name)
    contract = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONTRACT" for t in n.targets))
    maximum = next(value for key, value in zip(contract.keys, contract.values) if key.value == "maximum_equipment")
    assert isinstance(maximum, ast.Name) and namespace[maximum.id] == 12
    snapshot = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "IMPLEMENTATION_FILES" for t in n.targets))
    assert {"equipment_proposal.py", "equipment_proposal_config.py"} <= set(ast.literal_eval(snapshot))
    inputs = ast.parse((PACKAGE / "exercise_scene_inputs.py").read_text())
    assert any(isinstance(n, ast.ImportFrom) and n.module == "weft.equipment_proposal_config"
               and any(a.name == "validate_proposal_config" for a in n.names) for n in inputs.body)
    snapshot = next(n.value for n in inputs.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "IMPLEMENTATION_FILES" for t in n.targets))
    assert {"equipment_proposal.py", "equipment_proposal_config.py"} <= {n.value for n in ast.walk(snapshot) if isinstance(n, ast.Constant)}
