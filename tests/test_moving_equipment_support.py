"""Zero-step analytic support screens; only source admission is fixture-stubbed."""

from copy import deepcopy
from types import SimpleNamespace
import time

import numpy as np
import pytest

from test_moving_equipment_task import example
from weft.io import canonical_hash
from weft.moving_equipment_task_recording import runtime_identity
import weft.moving_equipment_support as support


@pytest.fixture
def admission(example, monkeypatch):
    scene, bundle, task = example
    value = SimpleNamespace(scene=deepcopy(scene), bundle=deepcopy(bundle), task=deepcopy(task),
        manifest_sha256="a"*64, parent_manifest_sha256="b"*64, runtime_sha256=runtime_identity(scene)["runtime_sha256"])
    def validate(candidate):
        if candidate is not value: raise ValueError("Original immutable task admission required")
        return candidate
    monkeypatch.setattr(support, "_admitted", validate)
    return value


def test_complete_native_terminal_boundaries_no_thinning(example):
    scene, _, task = example
    knots, times = support._schedule(scene, task)
    assert times[-1] == .45
    assert times[0] == .45-task["goal"]["terminal_consecutive_hold_s"]
    assert set(knots) <= set(times)
    assert times[-1]-times[-2] == pytest.approx((.45-13/30)/16)
    assert len(times) == len(set(times)) and len(times) <= 256
    longer = deepcopy(task)
    longer["goal"]["terminal_consecutive_hold_s"] = 1.
    with pytest.raises(ValueError, match="interval"): support._schedule(scene, longer)


def test_large_window_refuses_instead_of_claiming_all_time_coverage(example):
    scene, _, task = deepcopy(example)
    scene["task"].pop("native_interval")
    task["interval"]["horizon_steps"] = 90
    scene["task"]["horizon_steps"] = 90
    task["goal"]["terminal_consecutive_hold_s"] = 1.
    with pytest.raises(ValueError, match="no subsampling"): support._schedule(scene, task)


def test_force_deficit_is_necessary_only_and_uses_original_channels(example):
    scene, _, task = deepcopy(example)
    task["goal"]["terminal_consecutive_hold_s"] = .5
    task["goal"]["speed_tolerance_world_s"] = .1
    value = support.force_deficit(scene, task)
    assert value["steady_upward_force_model_units"] == pytest.approx(3.81*scene["objects"][0]["mass"])
    assert value["continuous_window_mean_upward_force_lower_bound_model_units"] == pytest.approx(3.41*scene["objects"][0]["mass"])
    assert "not an exact discrete scorer certificate" in value["scope"]
    scene["gravity"][0] = 1
    with pytest.raises(ValueError, match="vertical"): support.force_deficit(scene, task)


@pytest.mark.parametrize("fraction", [0., .5, 1.])
def test_analytic_forearm_jacobian_matches_finite_difference(example, fraction):
    scene, _, task = example
    geom = support._Geometry(scene, task, support._Budget(time.monotonic()+5, 100))
    try:
        _, indices, gid = geom.arm("left")
        delta = np.array([.03, -.02, .01, .04])
        geom.pose(.42, indices, delta)
        _, jac = geom.residual(gid, indices, fraction, True)
        numerical = np.empty_like(jac)
        for k in range(4):
            high, low = delta.copy(), delta.copy(); high[k] += 1e-6; low[k] -= 1e-6
            geom.pose(.42, indices, high); upper = geom.residual(gid, indices, fraction)
            geom.pose(.42, indices, low); lower = geom.residual(gid, indices, fraction)
            numerical[:, k] = (upper-lower)/2e-6
        assert np.allclose(jac, numerical, atol=1e-8, rtol=1e-6)
    finally: geom.close()


def test_one_actual_model_zero_steps_and_independent_all_boundary_audit(admission, monkeypatch):
    from weft import shared_world
    original = shared_world.SharedWorldEnv
    created = []
    class Watched(original):
        def __init__(self, *a, **kw): super().__init__(*a, **kw); created.append(self)
        def step(self, *a, **kw): pytest.fail("Diagnostic must never step physics")
        def reset(self, *a, **kw): pytest.fail("Diagnostic must never reset/move source props")
        def close(self): self.was_closed = True; super().close()
    monkeypatch.setattr(shared_world, "SharedWorldEnv", Watched)
    before = canonical_hash([admission.scene, admission.task, admission.bundle])
    result = support.diagnose_moving_support(admission, max_seconds=10)
    assert len(created) == result["model_compilations"] == 1 and created[0].was_closed
    assert created[0].steps == 0 and created[0].data.time == 0
    assert result["physics_steps"] == 0 and result["status"] == "complete"
    assert len(result["hypotheses"]) == 6 and result["unattempted_hypotheses"] == 0
    assert result["native_forward_evaluations"] <= 6000 and result["schedule"]["subsampling"] is False
    for row in result["hypotheses"]:
        assert [x["time_s"] for x in row["audit"]] == result["schedule"]["native_boundary_times_s"]
        assert max(abs(x) for point in row["knots"] for x in point["delta_rad"]) <= .5 + 1e-10
        assert all(not any("root" in x for x in row["variables"]) for _ in [0])
    assert result["qualification"]["dynamic_contact_verified"] is False
    assert result["qualification"]["task_success"] is False
    assert result["qualification"]["full_interval_body_score_checked"] is False
    assert canonical_hash([admission.scene, admission.task, admission.bundle]) == before
    assert len(support._encoded(result)) <= 4 << 20


def test_small_budget_preserves_partial_and_unattempted_hypotheses(admission):
    result = support.diagnose_moving_support(admission, max_seconds=5, max_evaluations=1)
    assert result["status"] == "incomplete_resource_bound" and result["native_forward_evaluations"] == 1
    assert result["hypotheses"][0]["status"] == "incomplete_resource_bound"
    assert result["unattempted_hypotheses"] == 5 and result["physics_steps"] == 0
    assert not result["qualification"]["task_success"]


def test_static_motor_capacity_is_not_global_actuator_infeasibility(admission):
    result = support.diagnose_moving_support(admission, max_seconds=5, max_evaluations=1)
    scope = result["qualification"]["static_load_assumptions"]
    assert "Floor load sharing is not solved" in scope
    assert "does not prove actuator infeasibility" in scope
    assert "not an equilibrium certificate" in scope


@pytest.mark.parametrize("options", [{"max_seconds": 61}, {"max_seconds": True}, {"max_evaluations": 6001}, {"max_evaluations": True}])
def test_work_bounds_refuse_before_compilation(admission, monkeypatch, options):
    monkeypatch.setattr(support, "_Geometry", lambda *a: pytest.fail("No model for invalid work bounds"))
    with pytest.raises(ValueError): support.diagnose_moving_support(admission, **options)


def test_source_capability_runtime_and_task_gate_changes_refuse(admission, monkeypatch):
    monkeypatch.setattr(support, "_Geometry", lambda *a: pytest.fail("No model for invalid custody"))
    with pytest.raises(ValueError, match="admission"): support.diagnose_moving_support(object())
    admission.runtime_sha256["recording.py"] = "0"*64
    with pytest.raises(ValueError, match="runtime changed"): support.diagnose_moving_support(admission)
    admission.task["body_gates"]["max_floor_penetration_model_units"] = .9
    with pytest.raises(ValueError): support.diagnose_moving_support(admission)


def test_proximity_does_not_hide_wrong_normal_overlap_or_static_capacity(example):
    scene, _, task = example
    geom = support._Geometry(scene, task, support._Budget(time.monotonic()+5, 100))
    try:
        _, indices, gid = geom.arm("left")
        geom.pose(.42, indices, np.zeros(4))
        # Put the target below a capsule: even a touching pair is not upward support.
        center = geom.data.geom_xpos[gid].copy()
        geom.target = center - [0., 0., geom.object["radius"]+geom.model.geom_size[gid,0]]
        geom.baseline[geom.oq:geom.oq+3] = geom.target
        row = geom.audit(.42, indices, gid, np.zeros(4))
        assert row["checks"]["upward_normal"] is False and row["geometric_static_witness"] is False
        assert row["object_static_capacity_ratio"] > 1
    finally: geom.close()


def test_new_limits_are_immutable():
    with pytest.raises(TypeError): support.LIMITS["angle_correction_rad"] = 9


def test_center_inside_world_does_not_hide_sphere_floor_or_boundary_intersection(example):
    scene, _, task = example
    geom = support._Geometry(scene, task, support._Budget(time.monotonic()+5, 10))
    try:
        _, indices, gid = geom.arm("left")
        radius = geom.object["radius"]
        geom.target = np.array([radius/2, scene["bounds"][1]/2, radius/2])
        geom.baseline[geom.oq:geom.oq+3] = geom.target
        assert np.all(geom.target > 0) and np.all(geom.target < scene["bounds"])
        row = geom.audit(.42, indices, gid, np.zeros(4))
        assert row["object_floor_clearance_world"] < 0
        assert not row["checks"]["object_floor_clearance"]
        assert not row["checks"]["object_world_containment"]
        assert not row["geometric_static_witness"]
    finally: geom.close()
