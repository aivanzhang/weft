from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest

from weft import contact_force_balance as c


def hypothesis():
    return {"schema_version": 1, "profile": c.PROFILE,
        "provenance": {k: "a"*64 for k in ("source_sha256", "scene_sha256", "task_sha256")},
        "contact_model": "point_sliding", "center_of_mass_world": [0, 0, 0], "mass": 1,
        "gravity_world": [0, 0, -9.81], "external_force_world": [0, 0, 0],
        "external_moment_world": [0, 0, 0], "direct_force_max_world": [0, 0, 0],
        "contacts": [{"id": name, "position_world": [sign*.1, 0, 0],
            "normal_on_object_world": [-sign, 0, 0], "tangent_world": [0, 0, 1],
            "sliding_friction": [.5, .5], "normal_force_max": 20} for name, sign in (("left", -1), ("right", 1))]}


def test_opposing_contacts_have_independently_checked_force_and_moment():
    h = hypothesis(); original = deepcopy(h)
    r = c.solve_contact_force_balance(h)
    assert r["status"] == "feasible" and h == original
    w = r["witness"]
    forces = np.array([row["force_world"] for row in w["contacts"]])
    assert np.allclose(forces.sum(axis=0), [0, 0, 9.81], atol=1e-7)
    assert np.max(abs(sum(np.cross(row["position_world"], force) for row, force in zip(h["contacts"], forces)))) < 1e-7
    assert w["motor_capacity"]["status"] == "capacity_unassessed"
    assert not r["source_authenticated"] and not r["global_infeasibility_claim"]
    assert "diamond" in r["approximation"] and "not independently measured" in r["normal_caps"]
    assert len(json.dumps(r)) < c.LIMITS["output_bytes"]


@pytest.mark.parametrize("change", ["zero_friction", "low_friction", "low_cap", "torque", "wrong_normal", "zero_cap"])
def test_refusals_do_not_claim_global_infeasibility(change):
    h = hypothesis()
    if change == "zero_friction":
        for row in h["contacts"]: row["sliding_friction"] = [0, 0]
    if change == "low_friction":
        for row in h["contacts"]: row["sliding_friction"] = [.01, .01]
    if change == "low_cap":
        for row in h["contacts"]: row["normal_force_max"] = 1
    if change == "zero_cap":
        for row in h["contacts"]: row["normal_force_max"] = 0
    if change == "torque": h["external_moment_world"] = [1, 0, 0]
    if change == "wrong_normal": h["contacts"][1]["normal_on_object_world"] = [1, 0, 0]
    r = c.solve_contact_force_balance(h)
    assert r["status"] == "refused" and r["witness"] is None
    assert not r["global_infeasibility_claim"]


def test_nonzero_torque_and_com_offset_are_balanced_not_discarded():
    h = hypothesis(); h["external_moment_world"] = [0, .2, 0]
    r = c.solve_contact_force_balance(h)
    assert r["status"] == "feasible"
    force = r["witness"]["contacts"]
    assert abs(force[0]["force_world"][2]-force[1]["force_world"][2]+2) < 1e-6
    shifted = deepcopy(h)
    shifted["center_of_mass_world"] = [10, -3, 4]
    for contact in shifted["contacts"]:
        contact["position_world"] = (np.asarray(contact["position_world"])+shifted["center_of_mass_world"]).tolist()
    r2 = c.solve_contact_force_balance(shifted)
    assert r2["status"] == "feasible"
    assert r2["hypothesis_sha256"] != r["hypothesis_sha256"]
    assert r2["witness"]["residuals"]["moment_max_abs"] < 1e-7


def test_direct_assistance_keeps_original_per_axis_cap():
    h = hypothesis()
    for row in h["contacts"]: row["normal_force_max"] = 4
    h["direct_force_max_world"] = [6, 6, 6]
    r = c.solve_contact_force_balance(h)
    assert r["status"] == "feasible"
    assert np.max(abs(np.array(r["witness"]["direct_force_world"]))) <= 6+1e-8
    h["direct_force_max_world"][2] = 5
    assert c.solve_contact_force_balance(h)["status"] == "refused"


def test_effective_anisotropic_friction_with_zero_unused_axis():
    h = hypothesis()
    for row in h["contacts"]: row["sliding_friction"] = [.5, 0]
    result = c.solve_contact_force_balance(h)
    assert result["status"] == "feasible"
    for row in result["witness"]["contacts"]:
        assert abs(row["tangent_forces"][1]) <= 1e-8


def test_single_vertical_contact_and_general_gravity_direction():
    h = hypothesis(); h["contacts"] = [h["contacts"][0]]
    h["gravity_world"] = [-9.81, 0, 0]
    r = c.solve_contact_force_balance(h)
    assert r["status"] == "feasible"
    assert np.allclose(r["witness"]["contacts"][0]["force_world"], [9.81, 0, 0], atol=1e-7)


def test_final_audit_encoding_deadline_cannot_publish_success(monkeypatch):
    clock = [0.]; encode = c._bytes
    monkeypatch.setattr(c.time, "monotonic", lambda: clock[0])
    def delayed(value, maximum):
        raw = encode(value, maximum)
        if maximum == c.LIMITS["output_bytes"]: clock[0] = 3.
        return raw
    monkeypatch.setattr(c, "_bytes", delayed)
    result = c.solve_contact_force_balance(hypothesis())
    assert result["status"] == "refused" and result["witness"] is None
    assert result["reason"] == "deadline_exceeded"


def test_tiny_positive_friction_is_explicitly_outside_conditioning(monkeypatch):
    h = hypothesis(); h["contacts"][0]["sliding_friction"] = [1e-300, .5]
    monkeypatch.setattr(c, "solve_feasible_qp", lambda *a, **k: pytest.fail("invalid input reached solver"))
    with pytest.raises(ValueError, match="conditioning"):
        c.solve_contact_force_balance(h)


def test_motor_world_force_rows_and_bias_remain_authoritative():
    h = hypothesis()
    h["motor_limits"] = {"matrix": [[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]],
        "bias": [2, 2], "lower": [-10, -10], "upper": [8, 8]}
    r = c.solve_contact_force_balance(h)
    assert r["status"] == "feasible"
    assert np.allclose(r["witness"]["motor_capacity"]["loads"], [6.905, 6.905], atol=1e-5)
    h["motor_limits"]["upper"] = [6, 6]
    refused = c.solve_contact_force_balance(h)
    assert refused["status"] == "refused" and refused["witness"] is None


@pytest.mark.parametrize("edit", [
    lambda h: h.update(mass=True),
    lambda h: h.update(mass=10**1000),
    lambda h: h.update(gravity_world=[0, 0, float("nan")]),
    lambda h: h.update(gravity_world=[0]*10000),
    lambda h: h["contacts"][0].update(normal_on_object_world=[2, 0, 0]),
    lambda h: h["contacts"][0].update(tangent_world=[1, 0, 0]),
    lambda h: h["contacts"][0].update(sliding_friction=[-.1, .5]),
    lambda h: h["contacts"][0].update(normal_force_max=-1),
    lambda h: h["contacts"][1].update(id="left"),
    lambda h: h.update(extra=True),
    lambda h: h.update(motor_limits={"matrix": [[0]*10000], "bias": [0], "lower": [0], "upper": [1]}),
    lambda h: h.update(motor_limits={"matrix": [[0]*6], "bias": [0], "lower": [2], "upper": [1]}),
])
def test_invalid_inputs_refused_before_solver(edit, monkeypatch):
    h = hypothesis(); edit(h)
    monkeypatch.setattr(c, "solve_feasible_qp", lambda *a, **k: pytest.fail("invalid input reached solver"))
    with pytest.raises(ValueError): c.solve_contact_force_balance(h)


@pytest.mark.parametrize("case", ["many_contacts", "torsional"])
def test_unsupported_is_explicit(case, monkeypatch):
    h = hypothesis()
    if case == "torsional": h["contact_model"] = "torsional"
    else:
        h["contacts"].append(deepcopy(h["contacts"][0])); h["contacts"][-1]["id"] = "third"
    monkeypatch.setattr(c, "solve_feasible_qp", lambda *a, **k: pytest.fail("unsupported reached solver"))
    r = c.solve_contact_force_balance(h)
    assert r["status"] == "unsupported" and r["witness"] is None


def test_false_solver_success_does_not_become_equilibrium(monkeypatch):
    monkeypatch.setattr(c, "solve_feasible_qp", lambda *a, **k: {"status": "optimal", "step": [0.]*7, "report": {"iterations": 1}})
    r = c.solve_contact_force_balance(hypothesis())
    assert r["status"] == "refused" and r["candidate_audit"]["residuals"]["direct_force_cap_violation"] == 9.81


def test_near_unit_frame_cannot_hide_world_normal_cap_violation(monkeypatch):
    h = hypothesis(); h["gravity_world"] = [0, 0, 0]
    for row in h["contacts"]:
        row["normal_force_max"] = 10000
        q = row["normal_on_object_world"][0]*(1+4e-11)/2**.5
        row["normal_on_object_world"] = [q, q, 0]
    monkeypatch.setattr(c, "solve_feasible_qp", lambda *a, **k: {"status": "optimal", "step": [10000.,0.,0.,10000.,0.,0.,0.], "report": {"iterations": 1}})
    result = c.solve_contact_force_balance(h)
    assert result["status"] == "refused"
    assert result["candidate_audit"]["contacts"][0]["normal_cap_violation"] > 1e-8


def test_late_solver_does_not_publish_witness(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(c.time, "monotonic", lambda: clock[0])
    def solver(*a, **k):
        clock[0] = 3.
        return {"status": "optimal", "step": [0.]*7, "report": {"iterations": 1}}
    monkeypatch.setattr(c, "solve_feasible_qp", solver)
    r = c.solve_contact_force_balance(hypothesis())
    assert r["status"] == "refused" and r["reason"] == "deadline_exceeded" and r["witness"] is None


def test_pure_solver_imports_no_native_engine():
    code = """
import sys
class Guard:
 def find_spec(self, fullname, *a):
  if fullname.split('.')[0] in ('mujoco','cv2','mediapipe','scipy'): raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from weft.contact_force_balance import solve_contact_force_balance
import json
print(solve_contact_force_balance(json.loads(sys.stdin.read()))['status'])
"""
    result = subprocess.run([sys.executable, "-c", code], input=json.dumps(hypothesis()), text=True,
        capture_output=True, timeout=10, check=True)
    assert result.stdout.strip() == "feasible"
