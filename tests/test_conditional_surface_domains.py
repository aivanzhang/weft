"""Analytical finite-support conditioning; no source, native or model work."""

from copy import deepcopy
from fractions import Fraction
import importlib
import json
import time

import pytest

from test_multiview_landmark_evidence import denied, local_only
from test_source_surface_envelopes import encoded, fixture


@pytest.fixture
def api():
    return importlib.import_module("weft.source_surface_envelopes")


def conditioned(data, bounds=(.15, -.1, .18, -.07)):
    data[2]["surfaces"][0].update(kind="static_support", bounds_xy_m=list(bounds),
        domain_mode="condition_on_declared_support")
    return data


def prepare(api, data):
    return api.prepare_source_surface_envelopes(**encoded(data))


def test_partial_domain_keeps_original_ranges_and_exact_source_slots(api):
    data = conditioned(fixture())
    before = deepcopy(data)
    result = prepare(api, data)
    assert data == before
    anchor = result.frame(0)["anchors"][0]
    assert anchor["status"] == "conditional_surface_envelope"
    box = anchor["world_envelope_m"]
    condition = box["support_conditioning"]
    assert condition["region_restricted"] is True
    original = deepcopy(data); original[2]["surfaces"][0].update(kind="ground", bounds_xy_m=None)
    original[2]["surfaces"][0].pop("domain_mode")
    assert condition["unconditioned_world_envelope_m"] == prepare(api, original).frame(0)["anchors"][0]["world_envelope_m"]
    assert box["minimum"][:2] == [.15, -.1] and box["maximum"][:2] == [.18, -.07]
    assert anchor["height_interval_m"] == [.2, .6]
    assert anchor["pixel_interval_px"] == {"minimum": [59., 44.], "maximum": [61., 46.]}
    meta = result.describe()
    assert meta["counts"]["retained_source_frames"] == meta["counts"]["conditional_envelopes"] == 4
    assert meta["contact"] == meta["load"] == meta["grasp"] == "unknown"
    assert meta["support_conditioning"]["available_enclosures_retain_original"] is True
    assert "correlated" in condition["scope"] and "infeasible" in condition["scope"]
    assert not meta["body_or_floor_installed"] and not meta["geometry_modified"]


def test_legacy_strict_domain_still_rejects_partial_overlap(api):
    data = conditioned(fixture()); data[2]["surfaces"][0].pop("domain_mode")
    result = prepare(api, data)
    anchor = result.frame(0)["anchors"][0]
    assert anchor["reason"] == "envelope_exceeds_declared_surface_domain"
    assert anchor["world_envelope_m"] is None
    assert "support_conditioning" not in result.describe()


def test_complete_domain_does_not_clip_or_choose_a_point(api):
    data = conditioned(fixture(), (-1., -1., 1., 1.))
    box = prepare(api, data).frame(0)["anchors"][0]["world_envelope_m"]
    condition = box["support_conditioning"]
    assert condition["region_restricted"] is False
    for name in ("minimum", "maximum"):
        assert box[name] == condition["unconditioned_world_envelope_m"][name]
    assert "midpoint" not in box and "position_world_m" not in box


def test_fixed_height_rectangle_intersection_is_exact_and_closed(api):
    data = conditioned(fixture(), (.19, -.11, .21, -.09))
    data[2]["surfaces"][0]["height_interval_m"] = [0., 0.]
    box = prepare(api, data).frame(0)["anchors"][0]["world_envelope_m"]
    assert box["minimum"] == [.19, -.11, 0.]
    assert box["maximum"] == [.21, -.09, 0.]


def test_support_constraint_preserves_height_pixel_correlations(api):
    data = conditioned(fixture(), (.15, -.1, .16, -.05))
    data[2]["surfaces"][0]["height_interval_m"] = [0., .8]
    for row in data[3]["annotations"]:
        row["uncertainty_px"] = 0.
    box = prepare(api, data).frame(0)["anchors"][0]["world_envelope_m"]
    assert box["minimum"][:2] == [.15, -.08]
    assert box["maximum"][:2] == [.16, -.075]
    low, high = 2-10*Fraction(.16), 2-10*Fraction(.15)
    assert Fraction(box["minimum"][2]) <= low <= Fraction(box["maximum"][2])
    assert Fraction(box["minimum"][2]) <= high <= Fraction(box["maximum"][2])
    assert box["maximum"][2]-box["minimum"][2] < .101


@pytest.mark.parametrize("bounds", [(.15, -.065, .16, -.06), (1., 1., 2., 2.)])
def test_empty_correlated_region_rejects_even_when_axis_boxes_overlap(api, bounds):
    data = conditioned(fixture(), bounds)
    data[2]["surfaces"][0]["height_interval_m"] = [0., .8]
    for row in data[3]["annotations"]:
        row["uncertainty_px"] = 0.
    anchor = prepare(api, data).frame(0)["anchors"][0]
    assert anchor["status"] == "unavailable" and anchor["world_envelope_m"] is None
    assert anchor["reason"] == "ray_height_region_disjoint_from_declared_support"


def test_zero_volume_boundary_intersection_remains_a_conditional_point(api):
    data = conditioned(fixture(), (.25, -.2, .3, -.125))
    data[2]["surfaces"][0]["height_interval_m"] = [0., 0.]
    for row in data[3]["annotations"]:
        row.update(pixel=[62.5, 46.25], uncertainty_px=0.)
    box = prepare(api, data).frame(0)["anchors"][0]["world_envelope_m"]
    assert box["minimum"] == box["maximum"] == [.25, -.125, 0.]


def test_oblique_rational_frustum_keeps_every_feasible_interior_sample(api):
    data = conditioned(fixture(), (1., .9, 1.3, 1.3))
    for frame in data[1]["frames"]:
        frame.update(R=[[1., 0., 0.], [0., -.8, -.6], [0., .6, -.8]], t=[-1., 3., 4.])
    baseline = deepcopy(data); baseline[2]["surfaces"][0].update(bounds_xy_m=[-10., -10., 10., 10.])
    original = prepare(api, baseline).frame(0)["anchors"][0]["world_envelope_m"]
    # Clip through the interior instead of choosing a camera-specific hard-coded crop.
    bounds = [original["minimum"][0], original["minimum"][1],
              (original["minimum"][0]+original["maximum"][0])/2, original["maximum"][1]]
    data[2]["surfaces"][0]["bounds_xy_m"] = bounds
    anchor = prepare(api, data).frame(0)["anchors"][0]
    box = anchor["world_envelope_m"]
    decimal_point = importlib.import_module("test_radial_surface_envelopes").decimal_point
    from decimal import Decimal
    lower, upper = ([Decimal.from_float(value) for value in box[key]] for key in ("minimum", "maximum"))
    selected = 0
    for x in (59., 59.5, 60., 60.5, 61.):
        for y in (44., 45., 46.):
            for height in (.2, .4, .6):
                point = decimal_point(data[1], [x, y], height)
                if all(Decimal.from_float(bounds[i]) <= point[i] <= Decimal.from_float(bounds[i+2]) for i in (0, 1)):
                    selected += 1
                    assert all(lo <= value <= hi for lo, value, hi in zip(lower, point, upper))
    assert selected > 10


def test_positive_radial_inverse_retains_its_outer_not_physical_feasibility_claim(api):
    data = conditioned(fixture())
    data[1]["radial_k1"] = .1
    anchor = prepare(api, data).frame(0)["anchors"][0]
    box = anchor["world_envelope_m"]
    assert "radial_outer_box_support_clipping" in box["certificate"]
    assert "radial_outer_box_plane_extrema" in box["support_conditioning"]["unconditioned_world_envelope_m"]["certificate"]
    assert anchor["radial_inverse"]["radial_k1"] == .1
    assert "infeasible" in box["support_conditioning"]["scope"]


@pytest.mark.parametrize("mode", ["occluded", "missing", "excluded", "unknown_height"])
def test_missing_and_excluded_inputs_are_not_filled_by_support_bounds(api, mode, monkeypatch):
    data = conditioned(fixture())
    if mode == "unknown_height":
        data[2]["surfaces"][0].update(height_interval_m=None, quality="unknown")
    elif mode == "missing":
        data[3]["annotations"] = []
    elif mode == "excluded":
        for row in data[0]["observations"]:
            row["partition"] = "audit"
    else:
        for row in data[3]["annotations"]:
            row.update(status="occluded", pixel=None, uncertainty_px=None)
    monkeypatch.setattr(api, "_clip_support", denied)
    result = prepare(api, data)
    assert result.frame_count == 4
    assert result.describe()["counts"]["conditional_envelopes"] == 0
    assert all(result.frame(index)["anchors"][0]["world_envelope_m"] is None for index in range(4))


@pytest.mark.parametrize("mode", [None, True, "clip", "require_complete_enclosure"])
def test_mode_is_explicit_not_truthy_or_implicitly_selected(api, mode):
    data = conditioned(fixture()); data[2]["surfaces"][0]["domain_mode"] = mode
    with pytest.raises(ValueError, match="conditional domain mode"):
        prepare(api, data)


def test_conditioning_does_not_apply_to_ground_or_unknown_support_extents(api):
    data = conditioned(fixture()); data[2]["surfaces"][0]["kind"] = "ground"
    with pytest.raises(ValueError, match="conditional domain mode"):
        prepare(api, data)
    data[2]["surfaces"][0].update(kind="static_support", bounds_xy_m=None)
    with pytest.raises(ValueError, match="finite|bounds|support"):
        prepare(api, data)


def test_representative_1200_conditioned_slots_are_bounded_and_cached(api, monkeypatch, record_property):
    data = conditioned(fixture(count=100, channels=12))
    started = time.monotonic()
    result = prepare(api, data)
    elapsed = time.monotonic()-started
    assert result.describe()["counts"]["conditional_envelopes"] == 1200
    assert result.describe()["cached_row_bytes"] < api.MAX_CACHE_BYTES
    monkeypatch.setattr(api, "_clip_support", denied)
    for index in range(1000):
        assert result.frame(index % 100)["anchors"][index % 12]["status"] == "conditional_surface_envelope"
    record_property("conditioning_seconds_1200_slots", elapsed)
    record_property("cached_row_bytes", result.describe()["cached_row_bytes"])
