"""Analytical correlated regions and certified projection; no source or physics run."""

from copy import deepcopy
from fractions import Fraction
import time

import numpy as np
import pytest

from weft import constrained_qp as qp
from weft import source_surface_envelopes as envelopes
from weft import source_surface_interval_construction as objective
from test_source_surface_envelopes import fixture, encoded


def rational(value):
    q = Fraction(value)
    return [str(q.numerator), str(q.denominator)]


def box_region(low=(0, 0, 0), high=(1, 1, 1)):
    rows = []
    for axis in range(3):
        for sign, bound in ((1, low[axis]), (-1, high[axis])):
            rows.append([rational(sign if i == axis else 0) for i in range(3)] + [rational(sign*bound)])
    return {"profile": envelopes.REGION_PROFILE, "unit": "m", "convention": "A*x >= b", "halfspaces_q": rows,
            "affine_hull_q": [[rational(int(i == axis)) for i in range(3)] + [rational(low[axis])]
                              for axis in range(3) if low[axis] == high[axis]],
            "feasible_seed_q": [rational((Fraction(a)+Fraction(b))/2) for a, b in zip(low, high, strict=True)]}


def project(record, point):
    prepared = objective._prepare_region(record, time.monotonic()+2)
    result, diagnostic = objective._project_region(prepared, np.asarray(point, dtype=float), time.monotonic()+.25)
    assert diagnostic["status"] == "optimal", diagnostic
    assert result is not None
    return result, diagnostic


def witness(**changes):
    surface = {"height_interval_m": [0, 1], "bounds_xy_m": None} | changes
    return envelopes._plane_box(tuple(map(Fraction, (0, 0, 2))),
        ((Fraction(1), 0, 0), (0, Fraction(1), 0), (0, 0, Fraction(-1))),
        tuple(map(Fraction, (1, 2))), tuple(map(Fraction, (0, 1))), surface,
        constraint_geometry=envelopes.CORRELATED_GEOMETRY)


def test_false_zero_box_loss_has_positive_correlated_distance():
    outer, reason = witness()
    assert reason is None
    point = np.array([1., 0., 0.])
    assert np.array_equal(point-np.clip(point, outer["minimum"], outer["maximum"]), [0, 0, 0])
    (closest, _), diagnostic = project(outer["constraint_region_m"], point)
    np.testing.assert_allclose(closest, [1.5, 0, .5], atol=1e-12)
    assert .5*np.sum((point-closest)**2) == pytest.approx(.25)
    assert diagnostic["primal_violation"] <= qp.CONTRACT["primal_tolerance"]


@pytest.mark.parametrize("point,expected", [
    ([.3, .4, .5], [.3, .4, .5]), ([-1, .5, .5], [0, .5, .5]),
    ([2, .5, .5], [1, .5, .5]), ([2, -1, .5], [1, 0, .5]),
    ([2, 2, 2], [1, 1, 1]), ([-1, -1, -1], [0, 0, 0]),
    ([1, 0, .5], [1, 0, .5]),
])
def test_faces_edges_vertices_and_closed_interior(point, expected):
    (closest, _), _ = project(box_region(), point)
    np.testing.assert_allclose(closest, expected, atol=1e-12)


@pytest.mark.parametrize("high,point,expected", [
    ((1, 1, 0), [.2, .3, 2], [.2, .3, 0]),
    ((1, 0, 0), [.2, 2, 3], [.2, 0, 0]),
    ((0, 0, 0), [1, 2, 3], [0, 0, 0]),
])
def test_planar_line_and_point_regions_are_not_empty(high, point, expected):
    (closest, _), _ = project(box_region(high=high), point)
    np.testing.assert_allclose(closest, expected, atol=1e-12)


@pytest.mark.parametrize("high,point,diagonal", [
    ((1, 1, 0), [.5, .5, 0], [0, 0, 1]),
    ((1, 1, 0), [.2, .3, 0], [0, 0, 1]),
    ((1, 0, 0), [.5, 0, 0], [0, 1, 1]),
    ((1, 0, 0), [.2, 0, 0], [0, 1, 1]),
    ((0, 0, 0), [0, 0, 0], [1, 1, 1]),
])
def test_on_region_affine_hull_derivative_survives_empty_active_set(high, point, diagonal):
    record = box_region(high=high)
    (closest, projector), diagnostic = project(record, point)
    np.testing.assert_allclose(closest, point, atol=1e-12)
    np.testing.assert_allclose(projector, np.diag(diagonal), atol=1e-12)
    assert diagnostic["affine_hull_planes"] == sum(diagonal)
    eps = 1e-6; p = np.array(point)
    finite = []
    for direction in np.eye(3):
        plus, minus = p+eps*direction, p-eps*direction
        finite.append(((plus-project(record, plus)[0][0])-(minus-project(record, minus)[0][0]))/(2*eps))
    np.testing.assert_allclose(projector, np.column_stack(finite), atol=1e-9)


@pytest.mark.parametrize("height,pixel_radius,rank", [([.2, .2], 1., 1), ([.2, .6], 0., 2), ([.2, .2], 0., 3)])
def test_actual_producer_retains_exact_lower_dimensional_hull(height, pixel_radius, rank):
    data = fixture(); data[2]["surfaces"][0]["height_interval_m"] = height
    for row in data[3]["annotations"]:
        row["uncertainty_px"] = pixel_radius
    prepared = envelopes.prepare_source_surface_envelopes(**encoded(data), constraint_geometry=envelopes.CORRELATED_GEOMETRY)
    record = prepared.frame(0)["anchors"][0]["world_envelope_m"]["constraint_region_m"]
    seed = [float(Fraction(int(a), int(b))) for a, b in record["feasible_seed_q"]]
    (closest, projector), diagnostic = project(record, seed)
    np.testing.assert_allclose(closest, seed, atol=1e-12)
    assert len(record["affine_hull_q"]) == diagnostic["affine_hull_planes"] == rank
    assert np.trace(projector) == pytest.approx(rank)


def test_finite_support_clips_correlations_before_projection():
    outer, reason = witness(bounds_xy_m=[1., 0., 1.2, .2], domain_mode=envelopes.CONDITIONAL_DOMAIN_MODE)
    assert reason is None and len(outer["constraint_region_m"]["halfspaces_q"]) == 10
    assert outer["minimum"][2] == pytest.approx(.8)
    (point, _), _ = project(outer["constraint_region_m"], [1., 0., 0.])
    np.testing.assert_allclose(point, [1.2, 0, .8], atol=1e-12)
    unavailable, reason = witness(bounds_xy_m=[0., 0., .5, .5], domain_mode=envelopes.CONDITIONAL_DOMAIN_MODE)
    assert unavailable is None and reason == "ray_height_region_disjoint_from_declared_support"


@pytest.mark.parametrize("radial", [-.01, .01])
def test_nonzero_radial_is_explicitly_unavailable(radial):
    data = fixture(); data[1]["radial_k1"] = radial
    result = envelopes.prepare_source_surface_envelopes(**encoded(data), constraint_geometry=envelopes.CORRELATED_GEOMETRY)
    assert result.describe()["status"] == "unavailable"
    assert all(result.frame(i)["anchors"][0]["reason"] == "correlated_region_requires_undistorted_pinhole"
               for i in range(result.frame_count))


@pytest.mark.parametrize("damage", ["extra_plane", "bad_seed", "denominator", "noncanonical", "shape", "hull_count", "hull_seed", "oversized"])
def test_malformed_or_unbounded_regions_reject(damage):
    record = box_region()
    if damage == "extra_plane":
        record["halfspaces_q"] += [deepcopy(record["halfspaces_q"][0])] * 5
    elif damage == "bad_seed":
        record["feasible_seed_q"][0] = rational(-1)
    elif damage == "denominator":
        record["halfspaces_q"][0][0][1] = "0"
    elif damage == "noncanonical":
        record["halfspaces_q"][0][0] = ["2", "2"]
    elif damage == "shape":
        record["halfspaces_q"][0].append(rational(1))
    elif damage == "hull_count":
        record["affine_hull_q"] = [deepcopy(record["halfspaces_q"][0])] * 4
    elif damage == "hull_seed":
        record["affine_hull_q"] = [deepcopy(record["halfspaces_q"][0])]
    else:
        record["unknown"] = "x" * envelopes.MAX_REGION_BYTES
    with pytest.raises(ValueError):
        objective._prepare_region(record, time.monotonic()+2)


def test_projection_outside_cached_affine_hull_is_unavailable():
    record = box_region()
    record["affine_hull_q"] = [[rational(v) for v in (0, 1, 0, Fraction(1, 2))]]
    prepared = objective._prepare_region(record, time.monotonic()+2)
    result, report = objective._project_region(prepared, np.array([2., .2, .5]), time.monotonic()+.25)
    assert result is None and report["status"] == "projection_affine_hull_mismatch"


def test_solver_has_fixed_dimension_planes_iterations_and_certification(monkeypatch):
    original = qp.solve_feasible_qp
    calls = []
    def solve(h, g, a, b, *, initial, settings):
        calls.append((h.shape, a.shape, settings.copy()))
        return original(h, g, a, b, initial=initial, settings=settings)
    monkeypatch.setattr(qp, "solve_feasible_qp", solve)
    project(box_region(), [2., .5, .5])
    assert calls[0][0] == (3, 3) and calls[0][1] == (6, 3)
    assert calls[0][2]["maximum_iterations"] == 32
    assert 0 < calls[0][2]["maximum_seconds"] <= .05


@pytest.mark.parametrize("failure", ["iteration_limit", "certificate", "dual"])
def test_uncertified_solver_output_never_becomes_a_projection(monkeypatch, failure):
    original = qp.solve_feasible_qp
    def solve(*args, **kwargs):
        result = original(*args, **kwargs)
        if failure == "iteration_limit":
            result.update(status="iteration_limit", step=None)
        elif failure == "certificate":
            result["step"] = [5., 5., 5.]
        else:
            result["report"]["multipliers"] = [-1.] * 6
        return result
    monkeypatch.setattr(qp, "solve_feasible_qp", solve)
    prepared = objective._prepare_region(box_region(), time.monotonic()+2)
    result, report = objective._project_region(prepared, np.array([2., .5, .5]), time.monotonic()+.25)
    assert result is None and report["status"] != "optimal"


def test_elapsed_query_cannot_start_a_solver(monkeypatch):
    prepared = objective._prepare_region(box_region(), time.monotonic()+2)
    monkeypatch.setattr(qp, "solve_feasible_qp", lambda *a, **k: pytest.fail("No solver after deadline"))
    result, report = objective._project_region(prepared, np.array([2., .5, .5]), time.monotonic()-1)
    assert result is None and report["status"] == "projection_deadline"
