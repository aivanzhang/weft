"""Analytical native inputs and real kinematic Jacobians; no video inference or rollouts."""

from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from test_native_reference_inputs import views
from weft import root_leg_source_fit as fit
from weft.body_camera_root_refinement import project_with_jacobians


def test_native_preparation_keeps_full_clock_gauge_and_original_bytes(tmp_path):
    args = views(tmp_path); saved = deepcopy(args)
    human, times, angles, roots, target, confidence, scale, rotation, center, K = fit._prepare(*args)
    assert args == saved
    assert times.tolist() == [row["time_s"] for row in human["reference"]]
    assert len(times) == len(target) == len(confidence) == len(human["reference"])
    assert K[0, 0] == K[1, 1] == json.loads(args[3])["width"]
    assert np.array_equal(angles, [[row["joints"][name] for name in fit.NAMES] for row in human["reference"]])
    pelvis = roots[0] + [0., 0., human["mobility"]["base_height"]]
    p = project_with_jacobians(pelvis[None], rotation, center, K)
    frame = json.loads(args[2])["frames"][0]
    expected = np.mean([[frame["landmarks"][i]["x"] * K[0, 0],
                        frame["landmarks"][i]["y"] * json.loads(args[3])["height"]] for i in (23, 24)], axis=0)
    np.testing.assert_allclose(p["pixels"][0], expected, atol=1e-10)


def test_absolute_root_and_leg_jacobians_match_native_finite_difference(tmp_path):
    human, times, angles, roots, _, _, _, rotation, center, K = fit._prepare(*views(tmp_path))
    engine = fit._NativeGeometry(human)
    try:
        q, root = angles[1].copy(), roots[1].copy()
        q[fit.NAMES.index("left_hip")] = .5; q[fit.NAMES.index("left_knee")] = .7
        state = fit._state(engine, q, root, True)
        projected = project_with_jacobians(state["absolute_points"], rotation, center, K)
        eps = 1e-6
        for axis in range(3):
            shift = np.eye(3)[axis] * eps
            after = fit._state(engine, q, root + shift)
            before = fit._state(engine, q, root - shift)
            numeric = (after["absolute_points"] - before["absolute_points"]) / (2 * eps)
            np.testing.assert_allclose(numeric, np.tile(state["root_jacobian"][:, axis], (12, 1)), atol=1e-8)
            np.testing.assert_allclose((after["gaps"] - before["gaps"]) / (2 * eps),
                                       state["root_gap_jacobian"][axis], atol=1e-8)
        for axis in fit.LEGS[0]:
            shift = np.eye(len(q))[axis] * eps
            after, before = (fit._state(engine, v, root) for v in (q + shift, q - shift))
            np.testing.assert_allclose((after["absolute_points"] - before["absolute_points"]) / (2 * eps),
                                       state["absolute_jacobian"][:, :, axis], atol=1e-8)
            np.testing.assert_allclose((after["gaps"] - before["gaps"]) / (2 * eps), state["gap_jac"][:, axis], atol=1e-8)
            pp, pm = (project_with_jacobians(v["absolute_points"], rotation, center, K)["pixels"] for v in (after, before))
            np.testing.assert_allclose((pp - pm) / (2 * eps),
                (projected["root_jacobian"] @ state["absolute_jacobian"][:, :, axis, None])[:, :, 0], atol=1e-5)
    finally:
        engine.close()


@pytest.mark.parametrize("kind", ["first_root", "root_norm", "root_speed", "angle", "angle_speed", "camera"])
def test_all_protected_correction_bounds(kind):
    q, r, times, rotation = np.zeros((3, 19)), np.zeros((3, 3)), np.array([0., .04, .1]), np.eye(3)
    assert fit._bounded(q, r, times, rotation, rotation)
    if kind == "first_root": r[0, 0] = 1e-12
    elif kind == "root_norm": r[1:, :] = .06
    elif kind == "root_speed": r[1, 0] = .013
    elif kind == "angle": q[:, 0] = .151
    elif kind == "angle_speed": q[1, 0] = .061
    else: rotation = fit._rotation([.151, 0., 0.])
    assert not fit._bounded(q, r, times, rotation, np.eye(3))


def test_fit_retains_native_source_and_protected_nonleg_angles(tmp_path):
    args = views(tmp_path); before = deepcopy(args)
    result = fit.propose_root_leg_source_fit(*args, maximum_seconds=.5, maximum_sweeps=1)
    assert args == before
    assert result["bindings"] == dict(zip(("human", "original_evidence", "derived_evidence", "dimensions"),
                                          (sha256(raw).hexdigest() for raw in args)))
    human = json.loads(args[0]); candidate = result["candidate"]
    assert [r["time_s"] for r in candidate["reference"]] == [r["time_s"] for r in human["reference"]]
    assert [r["time_s"] for r in candidate["root_reference"]] == [r["time_s"] for r in human["root_reference"]]
    assert candidate["root_reference"][0] == human["root_reference"][0]
    protected = set(fit.NAMES) - {fit.NAMES[i] for leg in fit.LEGS for i in leg}
    for original, proposed in zip(human["reference"], candidate["reference"]):
        assert all(original["joints"][name] == proposed["joints"][name] for name in protected)
    assert result["audit"]["nonregression"]
    assert result["physical_validation"] == "unverified"
    assert result["geometry_calls"] <= fit.CONTRACT["maximum_geometry_calls"]
    assert result["journal"] and all(row["qp"]["report"]["variables"] <= 7 for row in result["journal"])
    assert len(json.dumps(result).encode()) < fit.CONTRACT["maximum_output_bytes"]


def test_rejects_changed_native_pose_even_when_sidecar_hash_is_updated(tmp_path):
    h, o, d, dimension = views(tmp_path)
    value, sidecar = json.loads(d), json.loads(dimension)
    value["frames"][1]["landmarks"][27]["y"] += .01
    d = json.dumps(value).encode(); sidecar["derived_evidence_sha256"] = sha256(d).hexdigest()
    with pytest.raises(ValueError, match="Native source clock"):
        fit.propose_root_leg_source_fit(h, o, d, json.dumps(sidecar).encode())


def test_fresh_geometry_audit_rejects_any_new_floor_failure(tmp_path, monkeypatch):
    original = fit._NativeGeometry
    engines = []
    class ChangedAudit(original):
        def __init__(self, spec):
            super().__init__(spec); engines.append(self); self.audit = len(engines) == 2

        def evaluate(self, *args, **kwargs):
            value = super().evaluate(*args, **kwargs)
            if self.audit: value["gaps"][0] -= 10.
            return value
    monkeypatch.setattr(fit, "_NativeGeometry", ChangedAudit)
    result = fit.propose_root_leg_source_fit(*views(tmp_path), maximum_seconds=.5, maximum_sweeps=1)
    assert len(engines) == 2
    assert result["status"] == "rejected_unpromoted"
    assert not result["audit"]["nonregression"]
    assert result["audit"]["floor_after"]["source_pose"]["failed"] == len(result["candidate"]["reference"])
