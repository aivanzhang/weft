"""Analytical public-action checks; no simulator, source selector or native model."""

import ast
from bisect import bisect_right
import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import pytest

from weft.io import canonical_hash
from weft import shared_exercise_reference_controller as controller


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def refresh_hashes(bundle):
    original, candidate, report = (bundle[k] for k in ("original", "candidate", "report"))
    report["input_human_sha256"] = sha256(encode(original)).hexdigest()
    candidate["whole_body_reference_refinement"]["input_human_sha256"] = report["input_human_sha256"]
    report["candidate_sha256"] = canonical_hash(candidate)
    bundle["task"]["scene_sha256"] = canonical_hash(bundle["scene"])


def controller_fixture(*, corrections=(0., .04, .01, .03), root_assistance=True, motor_limit=30.):
    """Minimal analytical boundary objects; not a physics/source admission fixture."""
    names = list(controller.SPATIAL_JOINT_NAMES)
    source_sha, frames = "a" * 64, (2, 5, 8, 11)
    references = [{"time_s": frame / 30., "joints": {
        name: (.5 if name.endswith(("_knee", "_elbow")) else .2) + .01 * i
        for name in reversed(names)}} for i, frame in enumerate(frames)]
    origins = [{"evidence_id": f"{source_sha}:frame:{frame}", "time_s": frame / 30.,
                "pose_status": "tracked_partial"} for frame in frames]
    original = {"schema_version": 3, "profile": "exercise-v3", "source_sha256": source_sha, "dt": 1 / 30,
        "reference": references, "reference_evidence": origins, "lengths": {"torso": .4},
        "motor_limits": {name: motor_limit if name == "left_knee" else 30. for name in names},
        "root_reference": [{"time_s": row["time_s"], "translation": [0., 0., 0.]} for row in references],
        "mobility": {"floor_height": -.5}, "assumptions": ["Analytical fixture; no physical accuracy claim"]}
    candidate = deepcopy(original)
    for row, correction in zip(candidate["reference"], corrections):
        row["joints"]["left_knee"] += correction
    method, status = "blocked-whole-body-floor-reference-v1", "unadmitted_construction_proposal"
    candidate["whole_body_reference_refinement"] = {"schema_version": 1, "method": method, "status": status,
        "promoted": False, "input_human_sha256": "", "input_evidence_sha256": "b" * 64}
    report = {"schema_version": 1, "profile": method, "status": status, "promoted": False,
        "physical_validation": "unverified", "geometry_admission": "unadmitted_pending_independent_current_scene_checks",
        "failures": [], "budget_reason": None, "source_sha256": source_sha, "input_evidence_sha256": "b" * 64,
        "gates": dict.fromkeys(controller.PROPOSAL_GATES, True),
        "contract": {"method": method, "maximum_references": 900, "maximum_angle_correction_rad": .15,
                     "maximum_correction_rate_rad_s": 1.5, "maximum_temporal_gap_s": .25,
                     "floor_penetration_limit_model_units": .025, "numerical_tolerance": 1e-8},
        "references": [{"time_s": frame / 30., "frame": frame, "evidence_id": f"{source_sha}:frame:{frame}"}
                       for frame in frames]}
    world_action = {"names": names + ["root_x", "root_y", "root_z"],
        "physical_limits": [original["motor_limits"][name] for name in names] + [200.] * 3,
        "units": ["kilogram_prior * body_scaled_world_units^2 / second^2"] * 19
                 + ["kilogram_prior * body_scaled_world_units / second^2"] * 3}
    interval = {"start_offset_s": 0., "horizon_steps": 9}
    scene = {"profile": "shared-world-v2", "physical_validation": "unverified", "controlled_id": "human",
        "human_spec": deepcopy(original), "source_binding": {"source_sha256": source_sha}, "dt": 1 / 30,
        "substeps": 16, "task": dict(interval), "action_contract": world_action,
        "coordinate_frame": {"axes": "x_right_y_depth_z_up", "units": dict(controller._UNITS),
            "clock_binding": {"alignment_method": "pose_frame_index_over_fps_prior_v1", "control_dt_s": 1 / 30,
                              "physics_substeps": 16, "source_fps": 30., "source_time_offset_s": 2 / 30},
            "scale_bridge": {"method": "monocular_body_height_world_prior_v1", "body_height_to_world_scale": 1.}}}
    control = "explicit_3_axis" if root_assistance else "disabled"
    task = {"schema_version": 1, "profile": "shared-exercise-task-v1", "interval": interval,
            "gate_profile": "shared-exercise-gates-v1", "gates_sha256": "c" * 64,
            "control": {"root_assistance": control, "object_forces": "forbidden"}}
    count = 22 if root_assistance else 19
    action = {"names": world_action["names"][:count], "physical_limits": world_action["physical_limits"][:count],
              "units": world_action["units"][:count], "normalized_bounds": [-1., 1.], "joint_channels": list(range(19)),
              "root_assistance_channels": list(range(19, count)), "root_assistance": control, "object_force_channels": []}
    bundle = dict(original=original, candidate=candidate, report=report, scene=scene, task=task, action_contract=action)
    refresh_hashes(bundle)
    return bundle


@pytest.fixture(autouse=True)
def forbid_native_and_media(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"mujoco", "gymnasium", "cv2", "mediapipe", "torch", "PIL"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture(scope="module")
def velocity_index():
    package = Path(controller.__file__).parent
    for name, expected in controller.RUNTIME_SHA256.items():
        assert sha256((package / name).read_bytes()).hexdigest() == expected
    source = (package / "human.py").read_bytes()
    definition = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.FunctionDef) and node.name == "reference_velocity_index")
    namespace = {"math": math, "bisect_right": bisect_right}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "<pinned-pure-velocity-selector>", "exec"), namespace)
    return namespace["reference_velocity_index"]


def prepare(bundle, velocity_index, **kwargs):
    return controller.prepare_reference_controller(encode(bundle["original"]), encode(bundle["candidate"]),
        encode(bundle["report"]), scene=bundle["scene"], task=bundle["task"], action_contract=bundle["action_contract"],
        runtime_sha256=kwargs.get("runtime_sha256", dict(controller.RUNTIME_SHA256)),
        reference_velocity_index=velocity_index)


class FakeEnvironment:
    def __init__(self, bundle, provider, elapsed=0.):
        self.contract = deepcopy(bundle["action_contract"])
        description = provider.describe()
        self.state = {key: description[key] for key in ("scene_sha256", "task_sha256", "gates_sha256")}
        self.state.update(simulated_duration_s=elapsed, clock_status="observed", failure=None)
        self.features = np.zeros((len(self.contract["names"]), 3), dtype=float)
        self.calls = []

    def action_contract(self):
        self.calls.append("action_contract")
        return self.contract

    def metrics(self):
        self.calls.append("metrics")
        return self.state

    def feature_matrix(self):
        self.calls.append("feature_matrix")
        return self.features


@pytest.mark.parametrize("root_assistance", [False, True])
def test_noop_exact_authored_equivalence_and_no_mutation(velocity_index, root_assistance):
    bundle = controller_fixture(corrections=(0.,) * 4, root_assistance=root_assistance)
    before = deepcopy(bundle)
    provider = prepare(bundle, velocity_index)
    env = FakeEnvironment(bundle, provider, .1)
    env.features[:] = np.array([[-0., -0., -0.], [1., -1., .3], [1e5, -1e5, -.7], [2., .2, -.5]])[
        np.arange(len(env.features)) % 4]
    frozen = env.features.copy()
    expected = np.clip(env.features.sum(axis=1), -1., 1.)
    result = provider.action(env)
    assert result.tobytes() == expected.tobytes()
    assert env.features.tobytes() == frozen.tobytes() and bundle == before
    assert env.calls == ["action_contract", "metrics", "feature_matrix"]
    result[:] = 0.
    assert provider.action(env).tobytes() == expected.tobytes()
    with pytest.raises(FrozenInstanceError):
        provider._duration = 5.


@pytest.mark.parametrize("elapsed,position,velocity", [(0., 0., .4), (.05, .02, .4),
    (.1, .04, -.3), (.15, .025, -.3), (.2, .01, .2), (.3, .03, 0.)])
def test_analytical_position_velocity_knots_and_endpoint(velocity_index, elapsed, position, velocity):
    bundle = controller_fixture()
    provider = prepare(bundle, velocity_index)
    q, v = provider.target_delta(elapsed)
    index = bundle["action_contract"]["names"].index("left_knee")
    assert q[index] == pytest.approx(position, abs=1e-14)
    assert v[index] == pytest.approx(velocity, abs=1e-14)
    assert np.count_nonzero(q) <= 1 and np.count_nonzero(v) <= 1


def test_native_accumulated_knot_uses_existing_right_derivative(velocity_index):
    provider = prepare(controller_fixture(), velocity_index)
    elapsed = 0.
    for _ in range(48):
        elapsed += (1 / 30) / 16
    assert elapsed != .1
    q, v = provider.target_delta(elapsed)
    index = controller.SPATIAL_JOINT_NAMES.index("left_knee")
    assert q[index] == pytest.approx(.04, abs=1e-14)
    assert v[index] == pytest.approx(-.3, abs=1e-14)


def test_correction_is_added_before_clipping(velocity_index):
    bundle = controller_fixture(corrections=(0., -.03, -.03, -.03), motor_limit=1.)
    provider = prepare(bundle, velocity_index)
    env = FakeEnvironment(bundle, provider)
    index = bundle["action_contract"]["names"].index("left_knee")
    env.features[index, 0] = 2.
    result = provider.action(env)
    assert result[index] == pytest.approx(.5)
    assert result[index] != pytest.approx(np.clip(np.clip(2., -1., 1.) - 1.5, -1., 1.))


def test_contract_channel_order_root_bytes_and_no_step_parsing(velocity_index, monkeypatch):
    bundle = controller_fixture()
    provider = prepare(bundle, velocity_index)
    env = FakeEnvironment(bundle, provider, .05)
    env.features[19:] = [[-0., -0., -0.], [.25, -.5, .1], [2., 1., -1.]]
    roots = np.clip(env.features.sum(axis=1), -1., 1.)[19:]
    def forbidden(*_args, **_kwargs):
        raise AssertionError("No JSON parsing, hashing or filesystem work belongs in action()")
    monkeypatch.setattr(controller.json, "loads", forbidden)
    monkeypatch.setattr(controller, "canonical_hash", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    action = provider.action(env)
    index = bundle["action_contract"]["names"].index("left_knee")
    assert list(bundle["original"]["reference"][0]["joints"]).index("left_knee") != index
    assert action[index] == pytest.approx((40 * .02 + 5 * .4) / 30)
    assert np.count_nonzero(action[:19]) == 1
    assert action[19:].tobytes() == roots.tobytes()


@pytest.mark.parametrize("kind", ["input_hash", "candidate_hash", "unknown_gate", "missing_gate", "false_gate",
    "null_gate", "numeric_gate", "promoted", "rejected", "geometry_admitted", "relaxed_floor", "relaxed_angle",
    "changed_first", "changed_root", "changed_time", "changed_evidence", "changed_identity", "wrong_scene_body",
    "changed_source", "changed_task_scene", "native_clock", "offset_clock", "wrong_units", "wrong_action_order"])
def test_rejects_identity_clock_gate_and_fixed_field_changes(velocity_index, kind):
    b = controller_fixture()
    if kind == "input_hash": b["report"]["input_human_sha256"] = "d" * 64
    elif kind == "candidate_hash": b["report"]["candidate_sha256"] = "d" * 64
    elif kind == "unknown_gate": b["report"]["gates"]["new_gate"] = True
    elif kind == "missing_gate": del b["report"]["gates"]["floor_feasibility"]
    elif kind.endswith("_gate"): b["report"]["gates"]["floor_feasibility"] = {"false_gate": False, "null_gate": None, "numeric_gate": 1}[kind]
    elif kind == "promoted": b["report"]["promoted"] = True
    elif kind == "rejected": b["report"]["status"] = "rejected_construction_proposal"
    elif kind == "geometry_admitted": b["report"]["geometry_admission"] = "passed"
    elif kind == "relaxed_floor": b["report"]["contract"]["floor_penetration_limit_model_units"] = .03
    elif kind == "relaxed_angle": b["report"]["contract"]["maximum_angle_correction_rad"] = .2
    elif kind == "changed_first": b["candidate"]["reference"][0]["joints"]["left_knee"] += .01
    elif kind == "changed_root": b["candidate"]["root_reference"][1]["translation"][0] = .01
    elif kind == "changed_time": b["candidate"]["reference"][1]["time_s"] += .001
    elif kind == "changed_evidence": b["candidate"]["reference_evidence"][1]["pose_status"] = "tracked"
    elif kind == "changed_identity": b["report"]["references"][1]["evidence_id"] = "d" * 64 + ":frame:5"
    elif kind == "wrong_scene_body": b["scene"]["human_spec"]["reference"][1]["joints"]["left_knee"] += .001
    elif kind == "changed_source": b["report"]["source_sha256"] = "d" * 64
    elif kind == "changed_task_scene": b["task"]["scene_sha256"] = "d" * 64
    elif kind == "native_clock": b["scene"]["coordinate_frame"]["clock_binding"]["alignment_method"] = "native_decoder_time"
    elif kind == "offset_clock": b["scene"]["coordinate_frame"]["clock_binding"]["source_time_offset_s"] = 0.
    elif kind == "wrong_units": b["scene"]["coordinate_frame"]["units"]["length"] = "meter"
    elif kind == "wrong_action_order": b["action_contract"]["names"][:2] = reversed(b["action_contract"]["names"][:2])
    if kind.startswith("changed_") and kind not in {"changed_task_scene", "changed_identity", "changed_source"}:
        refresh_hashes(b)
    if kind in {"wrong_scene_body", "native_clock", "offset_clock", "wrong_units"}:
        b["task"]["scene_sha256"] = canonical_hash(b["scene"])
    with pytest.raises(ValueError):
        prepare(b, velocity_index)


@pytest.mark.parametrize("corrections", [(0., .1501, .1501, .1501), (0., .15, -.15, -.15)])
def test_actual_corrections_must_respect_claimed_passing_gates(velocity_index, corrections):
    with pytest.raises(ValueError, match="actual correction/rate"):
        prepare(controller_fixture(corrections=corrections), velocity_index)


def test_runtime_pin_and_bounded_reference_count(velocity_index):
    b = controller_fixture()
    with pytest.raises(ValueError, match="runtime"):
        prepare(b, velocity_index, runtime_sha256={**controller.RUNTIME_SHA256, "shared_world.py": "d" * 64})
    for key in ("original", "candidate"):
        b[key]["reference"] *= 226
    refresh_hashes(b)
    b["scene"]["human_spec"] = deepcopy(b["original"])
    refresh_hashes(b)
    with pytest.raises(ValueError, match="reference counts"):
        prepare(b, velocity_index)


@pytest.mark.parametrize("elapsed", [-.001, .301, float("nan"), float("inf"), True, None])
def test_bad_action_clocks_reject(velocity_index, elapsed):
    b = controller_fixture()
    provider = prepare(b, velocity_index)
    with pytest.raises(ValueError, match="elapsed clock"):
        provider.action(FakeEnvironment(b, provider, elapsed))


@pytest.mark.parametrize("failure", ["scene", "task", "clock", "failed", "shape", "nan", "bool", "oversize", "float32", "action"])
def test_bad_public_feedback_rejects_without_mutating_environment(velocity_index, failure):
    b = controller_fixture()
    provider = prepare(b, velocity_index)
    env = FakeEnvironment(b, provider)
    if failure in {"scene", "task"}: env.state[failure + "_sha256"] = "d" * 64
    elif failure == "clock": env.state["clock_status"] = "unavailable_after_last_observed_time"
    elif failure == "failed": env.state["failure"] = "stopped"
    elif failure == "shape": env.features = np.zeros((22, 2))
    elif failure == "nan": env.features[0, 0] = float("nan")
    elif failure == "bool": env.features = env.features.astype(bool)
    elif failure == "oversize": env.features[0, 0] = 1e6 + 1
    elif failure == "float32": env.features = env.features.astype(np.float32)
    elif failure == "action": env.contract["physical_limits"][0] *= 2
    before = deepcopy((env.state, env.contract)), env.features.tobytes()
    with pytest.raises(ValueError):
        provider.action(env)
    assert (env.state, env.contract) == before[0] and env.features.tobytes() == before[1]


def test_inputs_detached_and_description_is_not_internal_state(velocity_index):
    b = controller_fixture()
    provider = prepare(b, velocity_index)
    expected = provider.target_delta(.1)
    b["candidate"]["reference"][1]["joints"]["left_knee"] = 2.
    b["scene"]["human_spec"]["root_reference"][0]["translation"][0] = 100.
    description = provider.describe()
    description["runtime_sha256"]["human.py"] = "d" * 64
    assert provider.describe()["runtime_sha256"]["human.py"] == controller.RUNTIME_SHA256["human.py"]
    for before, after in zip(expected, provider.target_delta(.1)):
        assert before.tobytes() == after.tobytes()


@pytest.mark.parametrize("target", ["fixed_field", "marker_schema", "report_schema", "task_schema"])
def test_json_boolean_number_aliases_cannot_rewrite_identity(velocity_index, target):
    b = controller_fixture()
    if target == "fixed_field":
        b["original"]["fixed_authored_flag"] = 1
        b["candidate"]["fixed_authored_flag"] = True
        b["scene"]["human_spec"] = deepcopy(b["original"])
    elif target == "marker_schema": b["candidate"]["whole_body_reference_refinement"]["schema_version"] = True
    elif target == "report_schema": b["report"]["schema_version"] = True
    else: b["task"]["schema_version"] = True
    refresh_hashes(b)
    with pytest.raises(ValueError):
        prepare(b, velocity_index)


@pytest.mark.parametrize("raw", [b"{}", b'{"x":1,"x":2}', b'{"x":NaN}', b'[]', b'\xff'])
def test_malformed_bounded_json_rejects(velocity_index, raw):
    b = controller_fixture()
    with pytest.raises(ValueError):
        controller.prepare_reference_controller(raw, encode(b["candidate"]), encode(b["report"]),
            scene=b["scene"], task=b["task"], action_contract=b["action_contract"],
            runtime_sha256=dict(controller.RUNTIME_SHA256), reference_velocity_index=velocity_index)
