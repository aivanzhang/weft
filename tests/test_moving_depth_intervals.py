from copy import deepcopy
from fractions import Fraction as F
import math
import time

import pytest

from weft.moving_depth_geometry import FILTER, GATES, SCOPE
from weft.moving_depth_intervals import _positive, _sqrt_bounds, solve_depth_intervals


def geometry(*primitives):
    planes = []
    for axis in range(3):
        normal = [0., 0., 0.]; normal[axis] = 1.
        planes.extend([{"name": f"axis-{axis}-low", "normal": normal, "offset": 0.},
                       {"name": f"axis-{axis}-high", "normal": [-v for v in normal], "offset": -4.}])
    return {"schema_version": 1, "profile": "moving-depth-geometry-v1",
        "input_pins": {name: "a" * 64 for name in ("admission_receipt_sha256", "base_scene_sha256", "full_human_sha256",
            "source_sha256", "motion_evidence_sha256", "body_binding_sha256")},
        "runtime_sha256": {"moving_depth_geometry.py": "b" * 64},
        "runtime_versions": {name: "analytic" for name in ("python", "system", "machine", "mujoco", "numpy", "gymnasium")},
        "source_reference": {"first": {"index": 0, "source_frame": 0, "evidence_id": "analytic:0", "time_s": 0.},
            "count": 3, **{name: "c" * 64 for name in ("reference_sha256", "root_reference_sha256", "human_qpos_sha256", "native_clock_sha256")}},
        "coordinate_frame": {"scope": "Synthetic geometry only; no source admission"},
        "world_bounds": {"low": [0., 0., 0.], "high": [4., 4., 4.], "planes": planes},
        "floor": {"name": "mobile_floor", "type": "plane", "normal": [0., 0., 1.], "offset": 0.,
                  "contype": 1, "conaffinity": 6, "margin": 0., "gap": 0.},
        "primitives": list(primitives), "candidate_collision_filter": dict(FILTER), "geometric_gates": dict(GATES),
        "ignored_authored_object_ids": [], "physics_steps": 0, "model_compilations": 1, "scope": SCOPE}


def sphere(name="body", center=None, radius=.5):
    return {"name": name, "type": "sphere", "center": center or [2., 2., 2.], "radius": radius,
            "contype": 2, "conaffinity": 5, "margin": 0., "gap": 0.}


def capsule(start, end):
    result = sphere(); result.pop("center")
    return {**result, "type": "capsule", "endpoints": [start, end]}


def ray():
    return {"domain": [.25, 3.75], "center_intercept": [0., 2., 2.], "center_slope": [1., 0., 0.],
            "radius_intercept": .25, "radius_slope": 0., "radius_bounds": [.25, .5]}


@pytest.mark.parametrize("coefficients,expected", [
    ((0, 0, 0), [[0., 4.]]), ((0, 0, -1), []), ((0, 1, -2), [[2., 4.]]),
    ((0, -1, 2), [[0., 2.]]), ((1, -4, 3), [[0., 1.], [3., 4.]]),
    ((-1, 4, -3), [[1., 3.]]), ((-1, 4, -4), [[2., 2.]]),
    ((1, -4, 4), [[0., 4.]]), ((1, 0, 1), [[0., 4.]]), ((-1, 0, -1), []),
])
def test_exact_polynomial_degeneracies(coefficients, expected):
    outer, inner = _positive(*map(F, coefficients), (F(0), F(4)))
    assert outer == inner == [tuple(map(F, row)) for row in expected]


def test_two_disconnected_front_back_intervals_and_no_selection():
    r, g = ray(), geometry(sphere())
    before = deepcopy((r, g)); result = solve_depth_intervals(r, g)
    assert result["possible_intervals"] == result["certified_intervals"] == [[.25, 1.25], [2.75, 3.75]]
    assert result["status"] == "feasible_intervals" and not result["boundary_ambiguity"]
    assert result["selected_depth"] is None and (r, g) == before


@pytest.mark.parametrize("start,end,expected", [
    ([1.5, 2., 2.], [2.5, 2., 2.], [[.25, .75], [3.25, 3.75]]),
    ([2., 1., 2.], [2., 3., 2.], [[.25, 1.25], [2.75, 3.75]]),
    ([2., 2., 2.], [2., 2., 2.], [[.25, 1.25], [2.75, 3.75]]),
])
def test_capsule_projection_branches_and_zero_length(start, end, expected):
    result = solve_depth_intervals(ray(), geometry(capsule(start, end)))
    assert result["possible_intervals"] == result["certified_intervals"] == expected


def test_affine_radius_exact_boundaries_and_certified_points():
    r = ray(); r.update(radius_intercept=0., radius_slope=.125, radius_bounds=[.03125, .5])
    result = solve_depth_intervals(r, geometry(sphere()))
    left, right = F(3, 2) / F(9, 8), F(5, 2) / F(7, 8)
    outer, inner = result["possible_intervals"], result["certified_intervals"]
    assert F(outer[0][1]) >= left >= F(inner[0][1])
    assert F(outer[1][0]) <= right <= F(inner[1][0])
    for a, b in inner:
        for depth in [a, b, (a + b) / 2]:
            d = F(depth)
            assert (d - 2) ** 2 >= (d / 8 + F(1, 2)) ** 2
    assert result["boundary_ambiguity"]


def test_irrational_root_enclosures_are_proven_in_exact_arithmetic():
    low, high = _sqrt_bounds(F(2))
    assert low * low < 2 < high * high and high - low == F(1, 2 ** 192)
    outer, inner = _positive(F(1), F(0), F(-2), (F(0), F(2)))
    assert outer[0][0] ** 2 < 2 < inner[0][0] ** 2


def test_very_small_leading_coefficient_does_not_cancel_small_root():
    a = F(1, 2 ** 500)
    outer, inner = _positive(a, F(-1), F(1), (F(0), F(2)))
    assert outer and inner
    assert a * outer[0][1] ** 2 - outer[0][1] + 1 <= 0
    assert a * inner[0][1] ** 2 - inner[0][1] + 1 >= 0


def test_floor_and_world_bounds_clip_all_components_without_relaxation():
    r = ray(); r["center_intercept"] = [0., 2., 0.]; r["center_slope"] = [1., 0., 1.]
    g = geometry(sphere(center=[2., 3.5, 2.], radius=.1))
    result = solve_depth_intervals(r, g)
    assert result["certified_intervals"] == [[.25, 3.75]]
    r["domain"] = [.1, 4.]
    result = solve_depth_intervals(r, g)
    assert result["certified_intervals"] == [[.25, 3.75]]


def test_non_axis_plane_uses_exact_normal_norm_enclosure():
    r = ray(); g = geometry(sphere(center=[2., 3.5, 2.], radius=.1))
    g["floor"].update(normal=[.6, .8, 0.], offset=2.)
    result = solve_depth_intervals(r, g)
    assert result["status"] == "feasible_intervals"
    n0, n1 = F(.6), F(.8)
    for a, b in result["certified_intervals"]:
        for d in [F(a), F(b)]:
            signed = n0 * d + n1 * 2 - 2
            assert signed >= 0 and signed * signed >= F(.25) ** 2 * (n0 * n0 + n1 * n1)


def test_exact_isolated_tangent_depth_is_retained():
    r = ray(); r.update(domain=[1., 3.], radius_intercept=.5)
    result = solve_depth_intervals(r, geometry(sphere("left", [1., 2., 2.]), sphere("right", [3., 2., 2.])))
    assert result["possible_intervals"] == result["certified_intervals"] == [[2., 2.]]


def test_empty_intersection_is_no_solution_not_unsupported():
    result = solve_depth_intervals(ray(), geometry(sphere(radius=3.)))
    assert result["status"] == "no_solution" and result["possible_intervals"] == []


def test_sub_ulp_feasible_interval_is_retained_as_boundary_only():
    r = ray(); r.update(domain=[1., 2.], radius_intercept=.25,
                        center_intercept=[0., 2., 2.], center_slope=[1., 0., 0.])
    g = geometry(sphere("far", [2., 3.5, 2.], .1))
    # 1/3 is not a binary float; the exact linear constraints meet at 4/3.
    r.update(radius_intercept=0., radius_slope=.375, radius_bounds=[.5, .5])
    result = solve_depth_intervals(r, g)
    assert result["status"] == "boundary_only" and result["certified_intervals"] == []
    assert F(result["possible_intervals"][0][0]) < F(4, 3) < F(result["possible_intervals"][0][1])


@pytest.mark.parametrize("change", ["negative_radius", "nonfinite", "bool", "missing_floor", "changed_gate", "missing_primitive", "planes", "unknown_field"])
def test_malformed_or_changed_contract_refuses(change):
    r, g = ray(), geometry(sphere())
    if change == "negative_radius": g["primitives"][0]["radius"] = -1.
    elif change == "nonfinite": r["center_slope"][0] = math.nan
    elif change == "bool": r["radius_slope"] = True
    elif change == "missing_floor": g.pop("floor")
    elif change == "changed_gate": g["geometric_gates"]["domain_clearance_world"] = -1e-8
    elif change == "missing_primitive": g["primitives"] = []
    elif change == "planes": g["world_bounds"]["planes"].pop()
    else: r["discarded_source_frames"] = [1]
    with pytest.raises(ValueError): solve_depth_intervals(r, g)


def test_deadline_and_numeric_limit_are_explicit():
    with pytest.raises(TimeoutError): solve_depth_intervals(ray(), geometry(sphere()), deadline=time.monotonic() - 1)
    r = ray(); r["center_slope"][0] = math.ldexp(1., -1074); r["center_intercept"][1] = math.ldexp(1., -1074)
    result = solve_depth_intervals(r, geometry(sphere(center=[2., 0., 2.])))
    assert result["status"] == "unsupported" and result["reasons"]


def test_capsule_inner_intervals_satisfy_independent_exact_segment_distance():
    r = ray(); r.update(center_intercept=[0., 1., 1.], center_slope=[1., .5, .25],
                        radius_intercept=.125, radius_slope=.03125, radius_bounds=[.125, .5])
    start, end = [1., 1.5, 1.5], [2.5, 2., 2.]
    result = solve_depth_intervals(r, geometry(capsule(start, end)))
    assert result["certified_intervals"]
    a, edge = list(map(F, start)), [F(b) - F(a) for a, b in zip(start, end)]
    length2 = sum(x * x for x in edge)
    for low, high in result["certified_intervals"]:
        for value in [low, high, (low + high) / 2]:
            d = F(value); p = [F(x) + d * F(y) for x, y in zip(r["center_intercept"], r["center_slope"])]
            t = max(F(0), min(F(1), sum((x - y) * z for x, y, z in zip(p, a, edge)) / length2))
            distance2 = sum((x - y - t * z) ** 2 for x, y, z in zip(p, a, edge))
            assert distance2 >= (F(r["radius_intercept"]) + d * F(r["radius_slope"]) + F(.5)) ** 2


def test_complete_maximum_primitive_inventory_is_not_truncated():
    g = geometry(*(sphere(f"body-{i}", [2., 3.5, 2.], .1) for i in range(64)))
    result = solve_depth_intervals(ray(), g)
    assert result["primitive_count"] == 64 and result["certified_intervals"] == [[.25, 3.75]]
    g["primitives"][-1] = sphere("last-is-blocking", radius=3.)
    assert solve_depth_intervals(ray(), g)["status"] == "no_solution"


def test_module_and_solver_are_native_free():
    import json, subprocess, sys
    script = """import importlib.abc,sys,json
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'numpy','mujoco','gymnasium','cv2'}:
   raise AssertionError(name)
sys.meta_path.insert(0,Guard())
from weft.moving_depth_intervals import solve_depth_intervals
assert solve_depth_intervals(json.loads(sys.argv[1]),json.loads(sys.argv[2]))['certified_intervals'] == [[.25,1.25],[2.75,3.75]]
"""
    run = subprocess.run([sys.executable, "-c", script, json.dumps(ray()), json.dumps(geometry(sphere()))],
                         capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize("change", ["long_vector", "nested_vector", "string_scalar"])
def test_ray_shape_rejected_before_any_serialization(monkeypatch, change):
    import weft.moving_depth_intervals as module
    r = ray()
    if change == "long_vector": r["center_intercept"] = [0.] * 1000
    elif change == "nested_vector": r["center_intercept"] = [[0.] * 1000, 0., 0.]
    else: r["radius_intercept"] = "x" * 1000
    monkeypatch.setattr(module, "_digest", lambda *_: pytest.fail("Malformed ray must reject before serialization"))
    with pytest.raises(ValueError): module.solve_depth_intervals(r, geometry(sphere()))
