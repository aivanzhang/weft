"""Analytical admission/actions only; no source fitting or native scene admission."""

import ast
from bisect import bisect_right
import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from weft.io import canonical_hash
from weft import shared_exercise_reference_controller as shared


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def denied(*args, **kwargs):
    raise AssertionError("Native, fitting, media, network, process or unexpected I/O is forbidden")


@pytest.fixture(autouse=True)
def pure_only(monkeypatch):
    importing = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"mujoco", "cv2", "mediapipe", "torch", "tensorflow"} or name in {
            "weft.human", "weft.shared_world", "weft.camera_shared_task", "weft.camera_body_fit",
            "weft.whole_body_reference_refinement"}:
            denied()
        return importing(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    for owner, names in ((socket, ("socket", "create_connection")),
                         (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
                         (os, ("system", "popen", "fork", "posix_spawn", "posix_spawnp"))):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, denied)


@pytest.fixture
def api(monkeypatch):
    module = importlib.import_module("weft.camera_reference_controller")
    for path, names in (("weft.multiview_pose", ("triangulate_pair", "_triangulate_rays")),
                         ("weft.camera_body", ("make_camera_body", "build_camera_reference")),
                         ("weft.multiview_root", ("make_two_view_root", "triangulate_pair")),
                         ("weft.camera_ground", ("apply_camera_ground", "audit_ground_clearance"))):
        for name in names:
            monkeypatch.setattr(importlib.import_module(path), name, denied)
    return module


@pytest.fixture
def runtime(api):
    directory = Path(__file__).parents[1] / "src/weft"
    hashes = {name: sha256((directory / name).read_bytes()).hexdigest() for name in api.REQUIRED_RUNTIME_FILES}
    assert all(hashes[name] == pin for name, pin in api.RUNTIME_SHA256.items())
    tree = ast.parse((directory / "human.py").read_bytes())
    selector = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "reference_velocity_index")
    namespace = {"math": math, "bisect_right": bisect_right}
    exec(compile(ast.Module(body=[selector], type_ignores=[]), "<exact-pinned-velocity-selector>", "exec"), namespace)
    # Execute only the real pure envelope function; native scene admission remains absent.
    task_tree = ast.parse((directory / "camera_shared_task.py").read_bytes())
    envelope = next(node for node in task_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_envelope")
    scope = next(ast.literal_eval(node.value.args[0]) for node in task_tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "SCOPE" for target in node.targets))
    protected = ast.parse((directory / "shared_exercise_task.py").read_bytes())
    gates = next(ast.literal_eval(node.value.args[0]) for node in protected.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "GATES" for target in node.targets))
    identity = {name: hashes[name] for name in ("camera_shared_task.py", "shared_exercise_task.py")}
    task_namespace = {"canonical_hash": canonical_hash, "deepcopy": deepcopy, "SCOPE": scope,
        "PROFILE": "camera-shared-task-v1", "CONTEXT_PROFILE": "camera-shared-metric-context-v1",
        "implementation_identity": lambda: {"files": identity, "sha256": canonical_hash(identity)},
        "kernel": SimpleNamespace(HUMAN_ONLY_TASK_PROFILE="shared-exercise-task-v2",
                                  GATE_PROFILE="shared-exercise-gates-v1", GATES_SHA256=canonical_hash(gates))}
    exec(compile(ast.Module(body=[envelope], type_ignores=[]), "<exact-pinned-camera-task-envelope>", "exec"), task_namespace)
    return hashes, namespace["reference_velocity_index"], task_namespace["_envelope"]


def fixture(api, runtime, *, corrections=(0., .02, .01, .03), root=True, fps=30000 / 1001, body_dt=1 / 30):
    tools = importlib.import_module("test_camera_shared_contract")
    coupling = importlib.import_module("weft.camera_shared_contract")
    value = tools._fixture(sampled=(1, 2, 3, 4), fps=fps)
    body = value["body"]
    body["dt"] = body_dt
    for index, row in enumerate(body["reference"]):
        row["joints"] = {name: .5 + index * .01 for name in api.SPATIAL_JOINT_NAMES}
    tools._pin(value)
    arguments = tools._scene_arguments(coupling, body)
    steps = max(8, math.ceil(240 / fps))
    arguments["coordinate_frame"]["clock_binding"]["physics_substeps"] = steps
    scene = {"schema_version": 1, "profile": "shared-world-v6", "controlled_id": "human",
        "physical_validation": "unverified", "human_spec": body, "camera_coupling": coupling.build_camera_coupling(body),
        "dt": arguments["dt"], "substeps": steps, "coordinate_frame": arguments["coordinate_frame"],
        "source_binding": arguments["source_binding"], "objects": [], "target": None, "learned_policy_available": False,
        "task": {"start_offset_s": 0., "horizon_steps": 3, "object_goal": None},
        "action_contract": {"names": [*api.SPATIAL_JOINT_NAMES, "root_x", "root_y", "root_z"],
            "human_motor_count": 22, "object_force_channels": 0, "dynamic_object_ids": [],
            "normalized_bounds": [-1., 1.], "substeps_per_action": steps, "hidden_assistance": False,
            "units": ["authored_test_torque"] * 19 + ["authored_test_force"] * 3,
            "physical_limits": [30.] * 19 + [180.] * 3}}
    task = runtime[2](scene, root)
    body_pin = sha256(encode(body)).hexdigest()
    pins = {"body_raw_sha256": body_pin, "evidence_raw_sha256": sha256(encode(value["evidence"][0])).hexdigest(),
        "camera_raw_sha256": sha256(encode(value["cameras"][0])).hexdigest(), "body_canonical_sha256": canonical_hash(body),
        "camera_canonical_sha256": body["camera_reference"]["calibration_sha256"],
        "pose_evidence": {"canonical_sha256": body["camera_reference"]["pose_evidence_sha256"], "scope": "complete_supplied_pose_document"},
        "camera_binding_canonical_sha256": canonical_hash(body["camera_reference"]),
        "coupling_canonical_sha256": canonical_hash(scene["camera_coupling"]),
        "root_reference_canonical_sha256": canonical_hash(body["root_reference"]),
        "ground_canonical_sha256": canonical_hash(body["camera_reference"]["ground"]),
        "original_angular_reference_canonical_sha256": canonical_hash(body["reference"])}
    surface_inputs = {"profile": "source-surface-interval-objective-v1", "source_sha256": body["source_sha256"],
        "body_sha256": body_pin, "binding_sha256": "b" * 64, "envelope_cache_sha256": "c" * 64, "envelope_bindings": {}}
    surface = {"profile": "whole-body-surface-interval-fit-v1", "inputs": surface_inputs, "weight": 1., "weight_status": "authored",
        "work_limits": {"maximum_evaluations": 100, "maximum_journal_bytes": 10000}, "summary": {},
        "weight_units": "authored_camera_loss_units_squared_per_supplied_world_metre_squared", "contact": "unknown", "physical_validation": "unverified"}
    marker = {"schema_version": 1, "method": api.METHOD, "status": "unadmitted_construction_proposal", "promoted": False,
        "input_human_sha256": body_pin, "input_evidence_sha256": pins["evidence_raw_sha256"], "surface_intervals": {
            "profile": surface["profile"], "input_bindings": {"objective_profile": surface_inputs["profile"],
                **{k: v for k, v in surface_inputs.items() if k != "profile"}},
            **{k: surface[k] for k in ("weight", "weight_status", "work_limits", "summary")}}}
    targets = deepcopy(body["reference"])
    for row, delta in zip(targets, corrections, strict=True):
        row["joints"]["left_knee"] += delta
    overlay = {"schema_version": 1, "profile": "camera-body-angle-proposal-v1", "input_bindings": pins,
        "reference_candidate": targets, "whole_body_reference_refinement": marker,
        "installation": "unsupported_angle_overlay_not_a_human_spec", "body_installed": False,
        "raw_camera_changed": False, "root_or_floor_changed": False, "promoted": False,
        "contact": "unknown", "physical_validation": "unverified"}
    report_rows, slots = [], []
    for index, (old, target, meta) in enumerate(zip(body["reference"], targets, body["reference_evidence"], strict=True)):
        report_rows.append({"frame": index + 1, "time_s": old["time_s"], "evidence_id": meta["evidence_id"],
            "correction_rad": {name: target["joints"][name] - old["joints"][name] for name in api.SPATIAL_JOINT_NAMES}})
    retained = [{k: pose.get(k) for k in ("id", "frame", "time_s", "decoder_timestamp_s", "status")}
                for pose in value["evidence"][0]["frames"]]
    for frame in range(12):
        index = frame - 1 if 1 <= frame <= 4 else None
        row = retained[index] if index is not None else None
        slots.append({"source_frame": frame, "camera_time_s": frame / fps, "reference_index": index,
            "evidence_id": row["id"] if row else None, "source_time_s": row["time_s"] if row else None,
            "decoder_timestamp_s": row["decoder_timestamp_s"] if row else None, "status": row["status"] if row else "unsampled",
            "camera_status": "supplied", "landmarks_used": index is not None})
    report = {"schema_version": 1, "profile": api.METHOD, "status": "unadmitted_construction_proposal",
        "candidate_kind": overlay["profile"], "geometry_admission": overlay["installation"], "promoted": False,
        "physical_validation": "unverified", "failures": [], "budget_reason": None,
        "input_human_sha256": body_pin, "input_evidence_sha256": pins["evidence_raw_sha256"],
        "source_sha256": body["source_sha256"], "candidate_sha256": canonical_hash(overlay),
        "gates": {key: True for key in api.PROPOSAL_GATES}, "contract": {"method": api.METHOD, **api._LIMITS},
        "surface_intervals": surface, "objective": {"profile": "camera_surface_interval_v1"}, "references": report_rows,
        "retained_pose_slots": retained, "source_counts": {"declared_frame_count": 12, "declared_decoded_frames": 12,
            "body_reference_knots": 4, "retained_pose_slots": 4}, "camera_fit": {
            "profile": "fixed-camera-body-fit-inputs-v1", "input_bindings": pins, "coupling": scene["camera_coupling"],
            "image_loss_scale_per_pixel": .01, "image_loss_scale_status": "authored", "image_loss_units": "authored_loss_units_not_metres",
            "source_slots": slots, "complete_source_frame_slots": 12}}
    report["root_audit_sha256"] = canonical_hash(report["camera_fit"])
    return {"body": body, "overlay": overlay, "report": report, "scene": scene, "task": task,
            "action": api.camera_reference_action_contract(scene, task)}


def prepare(api, runtime, data, **changes):
    return api.prepare_camera_reference_controller(encode(data["body"]), encode(data["overlay"]), encode(data["report"]),
        **({"scene": data["scene"], "task": data["task"], "action_contract": data["action"],
            "runtime_sha256": runtime[0], "reference_velocity_index": runtime[1]} | changes))


class Environment:
    def __init__(self, data, provider, elapsed=0.):
        self.contract = deepcopy(data["action"])
        self.state = {k: provider.describe()[k] for k in ("scene_sha256", "task_sha256", "gates_sha256")}
        self.state.update(profile="camera-shared-metrics-v1", simulated_duration_s=elapsed, clock_status="observed", failure=None)
        self.features = np.zeros((len(self.contract["names"]), 3))
    def action_contract(self): return self.contract
    def metrics(self): return self.state
    def feature_matrix(self): return self.features


@pytest.mark.parametrize("fps,body_dt", [(30000 / 1001, 1 / 60), (60., 1 / 30)])
def test_source_clock_and_original_body_period_remain_distinct(api, runtime, fps, body_dt):
    data = fixture(api, runtime, fps=fps, body_dt=body_dt)
    before = deepcopy(data)
    provider = prepare(api, runtime, data)
    assert type(provider) is shared.ReferenceController
    assert data == before and data["body"]["camera_reference"] == before["body"]["camera_reference"]
    clock = provider.describe()["clock_binding"]
    assert clock["control_dt_s"] == 1 / fps and clock["source_body_dt_s"] == body_dt
    assert clock["source_time_offset_s"] == 1 / fps and provider._timestep == 1 / fps / data["scene"]["substeps"]
    joint = api.SPATIAL_JOINT_NAMES.index("left_knee")
    p, v = provider.target_delta(.5 / fps)
    assert p[joint] == pytest.approx(.01) and v[joint] == pytest.approx(.02 * fps)
    elapsed = 0.
    for _ in range(data["scene"]["substeps"]):
        elapsed += provider._timestep
    assert provider.target_delta(elapsed)[1][joint] == pytest.approx(-.01 * fps)
    assert np.array_equal(provider.target_delta(3 / fps)[1], np.zeros(19))


@pytest.mark.parametrize("root", [False, True])
def test_unchanged_targets_preserve_authored_bytes_and_roots(api, runtime, root):
    data = fixture(api, runtime, root=root, corrections=(0.,) * 4)
    provider = prepare(api, runtime, data)
    env = Environment(data, provider, .02)
    env.features[:] = np.array([[-0., -0., -0.], [1., -1., .3], [1e5, -1e5, -.7], [2., .2, -.5]])[
        np.arange(len(env.features)) % 4]
    prior = env.features.copy()
    assert provider.action(env).tobytes() == np.clip(env.features.sum(axis=1), -1, 1).tobytes()
    assert prior.tobytes() == env.features.tobytes()
    assert provider.describe()["learned"] is False and provider.describe()["body_installed"] is False


def test_corrections_use_existing_preclip_kernel_with_exact_root_actions(api, runtime):
    data = fixture(api, runtime)
    provider = prepare(api, runtime, data)
    env = Environment(data, provider, 1 / (30000 / 1001))
    env.features[:] = [.8, .4, 0.]
    env.features[19:] = [[-.0, -.0, -.0], [3., -2., -.25], [-1.8, -.2, .1]]
    old = env.features.copy()
    delta, rate = provider.target_delta(env.state["simulated_duration_s"])
    expected = np.clip(env.features.sum(axis=1), -1, 1)
    joint = api.SPATIAL_JOINT_NAMES.index("left_knee")
    expected[joint] = np.clip(1.2 + (40 * delta[joint] + 5 * rate[joint]) / 30., -1, 1)
    actual = provider.action(env)
    assert np.array_equal(actual, expected) and actual[19:].tobytes() == expected[19:].tobytes()
    assert actual[joint] == 1. and 1. + (40 * delta[joint] + 5 * rate[joint]) / 30. < 1.
    assert old.tobytes() == env.features.tobytes()


def test_prepared_actions_are_detached_no_parse_io_or_coupling_work(api, runtime, monkeypatch, record_property):
    data = fixture(api, runtime)
    provider = prepare(api, runtime, data)
    env = Environment(data, provider, .01)
    expected = provider.action(env)
    data["overlay"]["reference_candidate"][1]["joints"]["left_knee"] = 2.
    description, recipe = provider.describe(), provider.recipe()
    description["runtime_sha256"].clear(); recipe["angle_proposal"]["reference_candidate"].clear()
    with pytest.raises(FrozenInstanceError): provider._duration = 99.
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", denied)
        patch.setattr(Path, "read_bytes", denied)
        patch.setattr(api, "validate_camera_coupling", denied)
        patch.setattr(json, "loads", denied); patch.setattr(json, "dumps", denied)
        started = time.perf_counter()
        for _ in range(1000):
            assert provider.action(env).tobytes() == expected.tobytes()
        seconds = time.perf_counter() - started
    record_property("analytical_repeated_actions", 1000)
    record_property("analytical_repeated_action_seconds", seconds)


@pytest.mark.parametrize("change", ["gate_false", "gate_missing", "gate_extra", "gate_boolean_alias", "limits", "first_pose",
    "angle", "rate", "joint_limits", "camera_pin", "installed", "slot_drop", "slot_clock", "correction", "missing_reference"])
def test_rejects_changed_overlay_gates_and_source_inventory(api, runtime, change):
    data = fixture(api, runtime)
    report, overlay = data["report"], data["overlay"]
    if change == "gate_false": report["gates"]["camera_projection_domain"] = False
    elif change == "gate_missing": del report["gates"]["floor_feasibility"]
    elif change == "gate_extra": report["gates"]["physical_contact"] = True
    elif change == "gate_boolean_alias": report["gates"]["surface_interval_nonregression"] = 1
    elif change == "limits": report["contract"]["maximum_angle_correction_rad"] = .16
    elif change == "first_pose": overlay["reference_candidate"][0]["joints"]["left_knee"] += .001
    elif change == "angle": overlay["reference_candidate"][-1]["joints"]["left_knee"] += .2
    elif change == "rate": overlay["reference_candidate"][1]["joints"]["left_knee"] += .08
    elif change == "joint_limits": overlay["reference_candidate"][1]["joints"]["left_knee"] = -1.
    elif change == "camera_pin": overlay["input_bindings"]["camera_canonical_sha256"] = "f" * 64
    elif change == "installed": overlay["body_installed"] = True
    elif change == "slot_drop": report["camera_fit"]["source_slots"].pop()
    elif change == "slot_clock": report["camera_fit"]["source_slots"][0]["camera_time_s"] += .001
    elif change == "correction": report["references"][2]["correction_rad"]["left_knee"] += .001
    elif change == "missing_reference": report["references"].pop()
    report["candidate_sha256"] = canonical_hash(overlay)
    with pytest.raises(ValueError): prepare(api, runtime, data)


@pytest.mark.parametrize("change", ["legacy", "object_force", "body_dt", "root", "gate", "metric_context", "action_order", "runtime"])
def test_scene_task_and_runtime_cannot_be_relabelled(api, runtime, change):
    data = fixture(api, runtime)
    if change == "legacy": data["scene"]["profile"] = "shared-world-v3"
    elif change == "object_force": data["task"]["control"]["object_forces"] = "allowed"
    elif change == "body_dt": data["scene"]["dt"] = data["body"]["dt"]
    elif change == "root": data["body"]["root_reference"][1]["translation"][0] = .1
    elif change == "gate": data["task"]["gates_sha256"] = "f" * 64
    elif change == "metric_context": data["task"]["metric_context"]["valid_legacy_task"] = True
    elif change == "action_order": data["action"]["names"][:2] = reversed(data["action"]["names"][:2])
    elif change == "runtime":
        with pytest.raises(ValueError, match="runtime"):
            prepare(api, runtime, data, runtime_sha256={**runtime[0], "shared_world.py": "f" * 64})
        return
    with pytest.raises(ValueError): prepare(api, runtime, data)


def test_archived_recipe_is_deterministic_and_not_source_readmission(api, runtime, monkeypatch):
    data = fixture(api, runtime)
    fresh = prepare(api, runtime, data)
    options = {"scene": data["scene"], "task": data["task"], "action_contract": data["action"],
               "runtime_sha256": runtime[0], "reference_velocity_index": runtime[1]}
    monkeypatch.setattr(builtins, "open", denied); monkeypatch.setattr(Path, "read_bytes", denied)
    archived = api.prepare_camera_reference_recipe(fresh.recipe(), **options)
    assert archived.recipe() == fresh.recipe()
    assert archived.describe()["qualification"] == api.RECIPE_INTERPRETATION
    for elapsed in (0., .01, data["scene"]["dt"], 3 * data["scene"]["dt"]):
        for old, new in zip(fresh.target_delta(elapsed), archived.target_delta(elapsed), strict=True):
            assert old.tobytes() == new.tobytes()
    for key in ("source_clock_sha256", "task_sha256", "angle_proposal_canonical_sha256", "proposal_report_sha256"):
        recipe = fresh.recipe(); recipe[key] = "f" * 64
        if key == "proposal_report_sha256":
            # Raw artifact pins need an external seal; they are not source authentication.
            assert api.prepare_camera_reference_recipe(recipe, **options).recipe() == recipe
        else:
            with pytest.raises(ValueError): api.prepare_camera_reference_recipe(recipe, **options)
    recipe = fresh.recipe(); recipe["proposal_gates"]["camera_projection_domain"] = False
    with pytest.raises(ValueError): api.prepare_camera_reference_recipe(recipe, **options)


def test_input_byte_and_action_state_mutations_fail_closed(api, runtime):
    data = fixture(api, runtime)
    with pytest.raises(ValueError):
        api.prepare_camera_reference_controller(encode(data["body"]) + b" ", encode(data["overlay"]), encode(data["report"]),
            scene=data["scene"], task=data["task"], action_contract=data["action"], runtime_sha256=runtime[0], reference_velocity_index=runtime[1])
    provider = prepare(api, runtime, data)
    for field, value in (("task_sha256", "f" * 64), ("simulated_duration_s", -.1), ("simulated_duration_s", True), ("failure", "failed")):
        env = Environment(data, provider); env.state[field] = value
        with pytest.raises(ValueError): provider.action(env)
    env = Environment(data, provider); env.features[0, 0] = math.nan
    with pytest.raises(ValueError): provider.action(env)
    env = Environment(data, provider); env.features = env.features.astype(np.float32)
    with pytest.raises(ValueError): provider.action(env)


def test_camera_metadata_audit_pin_and_missing_slot_reject_independently(api, runtime):
    data = fixture(api, runtime)
    data["report"]["root_audit_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="camera objective"):
        prepare(api, runtime, data)
    data = fixture(api, runtime)
    data["report"]["camera_fit"]["source_slots"].pop()
    data["report"]["root_audit_sha256"] = canonical_hash(data["report"]["camera_fit"])
    with pytest.raises(ValueError, match="denominator"):
        prepare(api, runtime, data)


def test_fresh_preparation_has_no_io_and_retains_unknown_decoded_tail(api, runtime, monkeypatch):
    data = fixture(api, runtime)
    report = data["report"]
    report["source_counts"]["declared_decoded_frames"] = 13
    report["camera_fit"]["complete_source_frame_slots"] = 13
    report["camera_fit"]["source_slots"].append({"source_frame": 12, "camera_time_s": None,
        "reference_index": None, "evidence_id": None, "source_time_s": None, "decoder_timestamp_s": None,
        "status": "unsampled", "camera_status": "unavailable_original_decoded_slot", "landmarks_used": False})
    report["root_audit_sha256"] = canonical_hash(report["camera_fit"])
    monkeypatch.setattr(builtins, "open", denied); monkeypatch.setattr(Path, "read_bytes", denied)
    provider = prepare(api, runtime, data)
    assert provider.describe()["proposal_report_sha256"] == sha256(encode(report)).hexdigest()
    assert provider.describe()["reference_count"] == 4 and len(report["camera_fit"]["source_slots"]) == 13
    assert report["camera_fit"]["source_slots"][-1]["source_time_s"] is None
