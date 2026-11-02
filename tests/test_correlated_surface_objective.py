"""Public optional config, numerical objective and unchanged independent fitter gates."""

from copy import deepcopy
from hashlib import sha256
import json
import resource
import time

import numpy as np
import pytest

from weft import source_surface_envelopes as envelopes
from weft import source_surface_interval_construction as api
from weft import source_surface_fit_inputs as inputs
from weft import whole_body_reference_refinement as fit
from test_source_surface_interval_construction import fixture, raw, evaluate
from test_correlated_surface_regions import box_region, project


def correlated(data=None):
    old, body, binding = fixture(api) if data is None else data
    options = dict(zip(("camera_document", "surface_interval_document", "annotation_document"), old.additional_inputs, strict=True))
    source = envelopes.prepare_source_surface_envelopes(dict(old.observation_inputs), source_sha256=body["source_sha256"],
        constraint_geometry=envelopes.CORRELATED_GEOMETRY, **options)
    return api.prepare_surface_interval_objective(source, body_bytes=raw(body), binding_bytes=raw(binding))


def test_objective_corrects_false_zero_without_relabeling_old_metric():
    result = correlated()
    term = evaluate(result, [.126, -.06, .2])
    row = term["channels"][0]
    assert row["distance_to_outer_envelope_m"] == 0
    assert row["distance_to_constraint_region_m"] > .02 and term["objective_m2"] > 0
    assert result.frame_count == 4 and result.describe()["contact"] == "unknown"
    assert row["projection"]["status"] == "optimal" and row["active_axes"] is None


def test_gradient_chains_model_rotation_scale_and_matches_differences():
    data = fixture(api)
    data[2]["body_to_world"]["spatial"] = {"scale": 2., "rotation_wxyz": [2**-.5, 0., 0., 2**-.5], "translation": [.01, -.02, .03]}
    objective = correlated(data)
    point, j = np.array([.2, .3, .7]), np.array([[1., .2], [.4, -.3], [.1, 2.]])
    term = evaluate(objective, point, jacobians={"left_ankle": j}, parameter_names=api.NAMES[:2])
    eps = 1e-6
    finite = [(evaluate(objective, point+eps*j[:, i])["objective_m2"] - evaluate(objective, point-eps*j[:, i])["objective_m2"])/(2*eps)
              for i in range(2)]
    np.testing.assert_allclose(term["gradient_m2_per_rad"], finite, atol=2e-8)
    hessian = np.array(term["gauss_newton_m2_per_rad2"])
    assert np.linalg.eigvalsh(hessian).min() >= -1e-10


def test_squared_distance_gradient_exists_at_face_change():
    point = np.array([2., 0., .5]); record = box_region()
    closest = project(record, point)[0][0]
    analytic = point-closest
    def loss(p):
        q = project(record, p)[0][0]
        return .5 * np.sum((p-q)**2)
    eps = 1e-6
    numerical = [(loss(point+eps*np.eye(3)[i])-loss(point-eps*np.eye(3)[i]))/(2*eps) for i in range(3)]
    np.testing.assert_allclose(analytic, numerical, atol=3e-7)


def test_query_cache_cannot_reparse_source_or_region(monkeypatch):
    objective = correlated()
    def denied(*args, **kwargs):
        pytest.fail("No preparation during cached query")
    monkeypatch.setattr(api, "_prepare_region", denied)
    monkeypatch.setattr(api, "_sources", denied)
    monkeypatch.setattr(envelopes, "_clip_support", denied)
    first = evaluate(objective, [.126, -.06, .2])
    first["channels"][0]["closest_point_m"][0] = 100
    assert evaluate(objective, [.126, -.06, .2])["channels"][0]["closest_point_m"][0] < 1


def test_unavailable_projection_has_no_zero_or_partial_fitter_cost(monkeypatch):
    objective = correlated()
    monkeypatch.setattr(api, "_project_region", lambda *a: (None, {"status": "iteration_limit", "iterations": 32}))
    term = evaluate(objective, [.126, -.06, .2])
    assert term["objective_m2"] is None and term["projection_failures"] == 1
    assert term["channels"][0]["status"] == "unavailable"
    assert term["channels"][0]["distance_to_outer_envelope_m"] == 0
    assert term["channels"][0]["distance_to_constraint_region_m"] is None
    with pytest.raises(ValueError, match="cannot become zero"):
        fit._SurfaceIntervalFit.cost(term)
    assert not fit._SurfaceIntervalFit.nonregression(term, term)


def test_nonregression_uses_region_not_outer_box_and_binds_mode():
    objective = correlated()
    before, after = evaluate(objective, [.126, -.06, .2]), evaluate(objective, [.13, -.06, .2])
    assert fit._SurfaceIntervalFit.nonregression(before, after)
    assert not fit._SurfaceIntervalFit.nonregression(after, before)
    other = deepcopy(after); other.pop("constraint_geometry")
    assert not fit._SurfaceIntervalFit.nonregression(before, other)


@pytest.mark.parametrize("value", [None, False, "outer_aabb_v1", "unknown", {}])
def test_config_rejects_explicit_unknown_mode(tmp_path, value):
    from test_surface_fit_workflow import bundle
    _, config, _ = bundle(tmp_path/"inputs")
    config["constraint_geometry"] = value
    with pytest.raises(ValueError, match="constraint geometry"):
        inputs.parse_surface_fit_config(raw(config))


def test_public_config_snapshot_reaches_actual_objective(tmp_path):
    from test_surface_fit_workflow import bundle, snapshot
    path, config, body = bundle(tmp_path/"inputs")
    config["constraint_geometry"] = envelopes.CORRELATED_GEOMETRY
    path.write_bytes(raw(config))
    files, descriptor = snapshot(path, config["source_sha256"])
    objective = inputs.prepare_surface_fit_objective(files | {"inputs/human.json": body}, descriptor)["surface_interval_objective"]
    assert files[inputs.CONFIG_MEMBER] == path.read_bytes()
    assert descriptor["constraint_geometry"] == objective.describe()["constraint_geometry"] == envelopes.CORRELATED_GEOMETRY
    term = evaluate(objective, [.126, -.06, .2])
    assert term["objective_m2"] > 0
    broken = deepcopy(descriptor); broken.pop("constraint_geometry")
    with pytest.raises(ValueError, match="frozen descriptor"):
        inputs.validate_surface_fit_snapshot(files, broken)
    assert all(sha256(files[name]).hexdigest() == entry["sha256"] for name, entry in inputs._members(config).items())


def test_original_frame_clock_binding_and_excluded_slots_remain_exact():
    def changes(data):
        data[0]["observations"][1]["partition"] = "audit"
    objective = correlated(fixture(api, source_changes=changes))
    assert objective.frame_count == 4 and objective.frame(1)["channels"][0]["status"] == "unavailable"
    class Denied:
        def __iter__(self):
            pytest.fail("Excluded candidate payload consumed")
    term = objective.evaluate(1, body_sha256=objective.describe()["body_sha256"], body_time_s=1/30,
        coordinate_frame=objective.coordinate_frame, positions=Denied())
    assert term["objective_m2"] is None and term["counts"]["unavailable_channels"] == 1
    with pytest.raises(ValueError, match="clock"):
        evaluate(objective, [.1, -.1, .2], body_time_s=.01)


@pytest.mark.parametrize("mode", ["ordinary", "image_guard", "existing_floor_failure"])
def test_existing_fitter_and_independent_gates_still_apply(monkeypatch, mode):
    import test_whole_body_surface_interval_refinement as old
    data = old.fixture(fit)
    source = envelopes.prepare_source_surface_envelopes(**data[3], constraint_geometry=envelopes.CORRELATED_GEOMETRY)
    data[2]["surface_interval_objective"] = api.prepare_surface_interval_objective(source,
        body_bytes=old.raw(data[0]), binding_bytes=old.raw(data[4]))
    result = old.run(fit, monkeypatch, data, mode=mode)
    report = result["report"]
    assert report["surface_intervals"]["constraint_geometry"] == envelopes.CORRELATED_GEOMETRY
    assert report["gates"]["source_endpoint_image_nonregression"] and report["gates"]["endpoint_depth_change"]
    assert result["human_candidate"]["reference"][0] == data[0]["reference"][0]
    if mode == "image_guard":
        trials = report["surface_intervals"]["trial_journal"]
        assert any(row.get("gate_results", {}).get("source_image_nonregression") is False for row in trials)
        assert report["accepted_updates"] == 0
    elif mode == "existing_floor_failure":
        assert result["status"] == "rejected_construction_proposal" and not report["gates"]["floor_feasibility"]
    else:
        summary = report["surface_intervals"]["summary"]
        assert summary["sum_objective_after_m2"] < summary["sum_objective_before_m2"]


def test_cached_1200_slot_two_channel_workload(monkeypatch, record_property):
    started = time.monotonic()
    original = envelopes.prepare_source_surface_envelopes
    monkeypatch.setattr(envelopes, "prepare_source_surface_envelopes",
        lambda *a, **kw: original(*a, **kw, constraint_geometry=envelopes.CORRELATED_GEOMETRY))
    def coverage(data):
        data[3]["annotations"] = [row for row in data[3]["annotations"] if row["frame"] < 900]
    data = fixture(api, count=1200, channels=2, body_indices=range(900), source_changes=coverage)
    objective = api.prepare_surface_interval_objective(data[0], body_bytes=raw(data[1]), binding_bytes=raw(data[2]))
    prepare_seconds = time.monotonic()-started
    monkeypatch.setattr(api, "_prepare_region", lambda *a: pytest.fail("Cached region rebuilt"))
    body_hash = objective.describe()["body_sha256"]
    begin = time.monotonic(); calls, iterations = 0, 0
    for frame in range(1200):
        term = objective.evaluate(frame, body_sha256=body_hash, body_time_s=frame/30 if frame < 900 else None,
            coordinate_frame=objective.coordinate_frame, positions={"left_ankle": [.126, -.06, .2], "right_ankle": [.25, -.2, .8]})
        assert not term["projection_failures"]
        for channel in term["channels"]:
            if channel["projection"] is not None:
                calls += 1; iterations += channel["projection"]["iterations"]
    assert calls == 1800 and objective.frame_count == 1200
    metadata = objective.describe()
    record_property("source_slots", 1200); record_property("channels", 2)
    record_property("projection_calls", calls); record_property("projection_iterations", iterations)
    record_property("preparation_seconds", prepare_seconds); record_property("query_seconds", time.monotonic()-begin)
    record_property("cached_region_bytes", metadata["cached_region_bytes"])
    record_property("cached_join_bytes", metadata["cached_join_bytes"])
    record_property("child_highwater_bytes", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 512 << 20
