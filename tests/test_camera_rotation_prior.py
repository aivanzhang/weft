"""Analytical numerical-prior checks; no source fitting or physical calibration."""

from copy import deepcopy
import importlib
import json

import numpy as np
import pytest

from test_multiview_landmark_evidence import denied, local_only


@pytest.fixture
def api():
    return importlib.import_module("weft.camera_rotation_prior")


def proper_rotation():
    x, y, z = .37, -.22, .91
    cx, cy, cz, sx, sy, sz = np.cos(x), np.cos(y), np.cos(z), np.sin(x), np.sin(y), np.sin(z)
    rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
    ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
    rz = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
    return rz @ ry @ rx


def test_rounded_rotation_prior_is_explicit_bounded_and_nonmutating(api):
    supplied = proper_rotation().astype(np.float32).astype(float).tolist()
    original = deepcopy(supplied)
    result = api.prepare_camera_rotation_prior(supplied)
    detail = result.describe()
    assert supplied == original and detail["raw_rotation"] == original
    assert detail["profile"] == "bounded-polar-rounding-prior-v1"
    assert detail["assumption_status"] == "assumed" and detail["physical_validation"] == "unverified"
    assert 1e-8 < detail["raw_orthogonality_residual"] < 1e-5
    assert detail["corrected_orthogonality_residual"] <= 1e-8
    assert abs(detail["corrected_determinant"] - 1.) <= 1e-8
    delta = np.asarray(detail["corrected_rotation"]) - np.asarray(original)
    assert detail["correction_frobenius_norm"] == float(np.linalg.norm(delta, ord="fro"))
    assert detail["correction_max_abs_entry"] == float(np.max(np.abs(delta)))
    assert 0 < detail["correction_frobenius_norm"] <= 2e-5
    assert detail["raw_rotation_sha256"] == api.canonical_hash(original)
    assert detail["corrected_rotation_sha256"] == api.canonical_hash(detail["corrected_rotation"])
    assert detail["raw_rotation_sha256"] != detail["corrected_rotation_sha256"]
    supplied[0][0] = 999
    detail["corrected_rotation"][0][0] = -999
    assert result.describe()["raw_rotation"] == original
    assert abs(result.describe()["corrected_rotation"][0][0]) < 1
    json.dumps(result.describe(), allow_nan=False)


def test_known_isotropic_rounding_has_the_analytic_closest_proper_factor(api):
    true = proper_rotation()
    raw = (true * (1 + 2e-6)).tolist()
    result = api.prepare_camera_rotation_prior(raw)
    actual = np.asarray(result.corrected_rotation)
    assert np.allclose(actual, true, atol=1e-14, rtol=0)
    assert result.correction_frobenius_norm == pytest.approx(np.sqrt(3) * 2e-6, abs=2e-15)
    assert abs(result.corrected_determinant - 1.) < 1e-14


@pytest.mark.parametrize("bad", [
    np.diag([-1., 1., 1.]).tolist(),
    np.diag([1., 1., 0.]).tolist(),
    np.diag([1.000006, 1., 1.]).tolist(),
    [[1., 2e-5, 0.], [0., 1., 0.], [0., 0., 1.]],
])
def test_reflection_singular_and_outside_upstream_rotations_reject_before_svd(api, monkeypatch, bad):
    monkeypatch.setattr(api.np.linalg, "svd", denied)
    with pytest.raises(ValueError, match="unchanged upstream"):
        api.prepare_camera_rotation_prior(bad)


@pytest.mark.parametrize("bad", [[], [[1, 0, 0]] * 2, [[True, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  [[float("nan"), 0, 0], [0, 1, 0], [0, 0, 1]],
                                  [[float("inf"), 0, 0], [0, 1, 0], [0, 0, 1]],
                                  [[10 ** 1000, 0, 0], [0, 1, 0], [0, 0, 1]]])
def test_rotation_input_is_small_finite_and_typed_before_numerical_work(api, monkeypatch, bad):
    monkeypatch.setattr(api.np.linalg, "svd", denied)
    with pytest.raises(ValueError, match="finite bounded"):
        api.prepare_camera_rotation_prior(bad)


def test_exact_identity_has_zero_correction_and_retains_original_number_types(api):
    supplied = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    result = api.prepare_camera_rotation_prior(supplied)
    assert result.correction_frobenius_norm == result.correction_max_abs_entry == 0
    assert result.raw_rotation_sha256 == api.canonical_hash(supplied)
    assert type(result.raw_rotation[0][0]) is int
    assert type(result.corrected_rotation[0][0]) is float
    assert np.array_equal(result.raw_rotation, result.corrected_rotation)


def test_prior_bound_follows_upstream_envelope_not_a_source_residual(api):
    upstream = api.CONTRACT["upstream_maximum_orthogonality_residual"]
    implied_bound = 3 * upstream / (np.sqrt(1 - 3 * upstream) + 1)
    assert implied_bound < api.CONTRACT["maximum_correction_frobenius_norm"] == 2e-5
    assert api.CONTRACT["downstream_maximum_orthogonality_residual"] == 1e-8
    assert api.CONTRACT["downstream_maximum_determinant_error"] == 1e-8


def test_failed_polar_decomposition_never_invents_a_rotation(api, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise np.linalg.LinAlgError("analytical failure")

    monkeypatch.setattr(api.np.linalg, "svd", unavailable)
    with pytest.raises(ValueError, match="polar decomposition unavailable"):
        api.prepare_camera_rotation_prior(np.eye(3).tolist())
