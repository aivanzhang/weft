"""Analytic force budgets, input boundaries and state isolation."""

from copy import deepcopy
import json

import mujoco
import numpy as np
import pytest

from weft.reference_dynamics import audit_reference_dynamics, reference_samples


def slider(*, damping=2., gravity="0 0 0", floor=False):
    ground = '<geom type="plane" size="1 1 .1"/>' if floor else ''
    return mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="{gravity}"/>
      <worldbody>{ground}<body pos="0 0 .3"><joint name="slide" type="slide" axis="0 0 1"
      damping="{damping}" armature=".25"/><geom type="sphere" size=".1" mass="2"/></body></worldbody>
      <actuator><motor name="slide_motor" joint="slide" ctrllimited="true" ctrlrange="-10 10"/></actuator>
      </mujoco>''')


def audit(model, positions=((0.,), (.2,), (0.,)), **kwargs):
    return audit_reference_dynamics(model, [0., .1, .2], positions,
        names=["slide"], limits=[10.], transition_s=.02, **kwargs)


def test_linear_motion_and_reversal_match_analytic_mass_damping_and_impulse(monkeypatch):
    model = slider()
    monkeypatch.setattr(mujoco, 'mj_step', lambda *a: pytest.fail('Diagnostic must never step'))
    result = audit(model)
    rows = result['samples']
    assert [row['required_motor_force'][0] for row in rows[:2]] == pytest.approx([4., -4.])
    knot = rows[-1]
    assert knot['velocity_jump'] == [-4.]
    assert knot['acceleration'] == [-200.]
    assert knot['required_motor_force'] == pytest.approx([-450.])
    assert knot['frozen_configuration_impulse'] == pytest.approx([-9.])
    assert knot['motor_only_impulse_time_s'] == pytest.approx([.9])
    assert knot['force_limit_ratio'] == pytest.approx([45.])
    assert result['summary']['interval_midpoint']['over_limit_samples'] == 0
    assert result['summary']['knot_transition_probe']['constraint_free_over_limit_samples'] == 1
    assert result['physics_steps'] == 0 and result['inverse_calls'] == 4
    assert max(row['force_balance_residual_max'] for row in rows) < 1e-12
    json.dumps(result, allow_nan=False)


def test_gravity_is_in_required_force_and_not_clipped():
    result = audit(slider(damping=0., gravity="0 0 -9.81"), positions=((0.,), (0.,), (0.,)))
    assert result['samples'][0]['required_motor_force'] == pytest.approx([19.62])
    assert result['summary']['interval_midpoint']['over_limit_samples'] == 2
    assert result['status'] == 'diagnostic_only'


def test_contacts_stay_present_and_are_not_called_constraint_free():
    result = audit(slider(floor=True), positions=((-0.25,), (-0.25,), (-0.25,)))
    row = result['samples'][0]
    assert row['contacts'] == 1 and row['constraint_rows'] > 0
    assert abs(row['constraint_force'][0]) > 0
    assert row['force_balance_residual_max'] < 1e-9
    assert result['summary']['interval_midpoint']['constraint_free_samples'] == 0


def test_force_predictions_recover_requested_acceleration_without_step():
    model = slider()
    result = audit(model)
    data = mujoco.MjData(model)
    for row in result['samples']:
        data.qpos[:], data.qvel[:] = row['position'], row['velocity']
        # Applied generalized force is unclipped here to test inverse arithmetic, not legal control.
        data.qfrc_applied[:] = row['required_motor_force']
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.qacc, row['acceleration'], atol=1e-10)


def test_live_state_model_inputs_are_untouched():
    model = slider()
    data = mujoco.MjData(model)
    data.qpos[:] = 7.; data.qvel[:] = -3.; data.ctrl[:] = .2; data.time = .031
    arrays = {key: getattr(model, key).copy() for key in ('body_mass', 'dof_damping', 'dof_armature', 'actuator_ctrlrange')}
    state = {key: getattr(data, key).copy() for key in ('qpos', 'qvel', 'ctrl')}
    times, positions = [0., .1, .2], [[0.], [.2], [0.]]
    before = deepcopy((times, positions))
    audit_reference_dynamics(model, times, positions, names=['slide'], limits=[10.], transition_s=.02)
    assert (times, positions) == before
    for key, values in arrays.items(): np.testing.assert_array_equal(getattr(model, key), values)
    for key, values in state.items(): np.testing.assert_array_equal(getattr(data, key), values)
    assert data.time == .031


def test_motor_order_is_not_assumed_to_equal_dof_order():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/><worldbody>
      <body><joint name="x" type="slide" axis="1 0 0" damping="2"/>
      <joint name="y" type="slide" axis="0 1 0" damping="3"/><geom size=".1" mass="1"/></body></worldbody>
      <actuator><motor name="x_motor" joint="x" ctrllimited="true" ctrlrange="-10 10"/>
      <motor name="y_motor" joint="y" ctrllimited="true" ctrlrange="-20 20"/></actuator></mujoco>''')
    result = audit_reference_dynamics(model, [1., 1.1], [[0., 0.], [.1, .2]],
        names=['y', 'x'], limits=[20., 10.], transition_s=.02)
    assert result['samples'][0]['required_motor_force'] == pytest.approx([3., 4.])
    assert result['summary']['knot_transition_probe']['samples'] == 0
    assert result['summary']['knot_transition_probe']['maximum_force_limit_ratio'] is None


def test_stricter_runtime_force_cap_is_preserved():
    model = slider(damping=0., gravity="0 0 -9.81")
    model.actuator_ctrlrange[0] = [-100., 100.]
    model.actuator_forcelimited[0] = True
    model.actuator_forcerange[0] = [-100., 100.]
    result = audit(model, positions=((0.,), (0.,), (0.,)))
    assert result['force_limits'] == [10.]
    assert result['samples'][0]['force_limit_ratio'] == pytest.approx([1.962])
    assert result['engine_limits'][0]['force_range'] == [-100., 100.]


@pytest.mark.parametrize('joint', [False, True])
def test_engine_force_clipping_below_runtime_cap_is_rejected(joint):
    model = slider()
    if joint:
        model.jnt_actfrclimited[0] = True; model.jnt_actfrcrange[0] = [-9., 9.]
    else:
        model.actuator_forcelimited[0] = True; model.actuator_forcerange[0] = [-9., 9.]
    with pytest.raises(ValueError): audit(model)


@pytest.mark.parametrize('times,positions,window', [
    ([0.], [[0.]], .02), ([0., 0.], [[0.], [0.]], .02), ([.1, 0.], [[0.], [0.]], .02),
    ([0., float('nan')], [[0.], [0.]], .02), ([0., 61.], [[0.], [0.]], .02),
    ([0., .1], [[0.], [True]], .02), ([0., .1], [[0.], [float('inf')]], .02),
    ([0., .1], [[0.]], .02), ([0., .1], [[0.], [0.]], True),
    ([0., .1], [[0.], [0.]], 0.), ([0., .1], [[0.], [0.]], .2),
    ([0., .00001], [[0.], [1.]], .00001), ([0., .1], [[0.] * 65] * 2, .02),
    ([0., .1], [[0.], [-(2**63)]], .02),
])
def test_invalid_reference_rejected_before_native_allocation(monkeypatch, times, positions, window):
    model = slider()
    monkeypatch.setattr(mujoco, 'MjData', lambda *a: pytest.fail('Rejected inputs allocate no native state'))
    with pytest.raises(ValueError):
        audit_reference_dynamics(model, times, positions, names=['slide'], limits=[10.], transition_s=window)


@pytest.mark.parametrize('alter', ['gear', 'cap', 'activation', 'bias', 'inverse_discrete', 'gain'])
def test_unsupported_actuation_is_rejected(alter):
    model = slider()
    if alter == 'gear': model.actuator_gear[0, 0] = 2.
    elif alter == 'cap': model.actuator_ctrlrange[0, 1] = 9.
    elif alter == 'activation': model.actuator_dyntype[0] = mujoco.mjtDyn.mjDYN_FILTER
    elif alter == 'bias': model.actuator_biastype[0] = mujoco.mjtBias.mjBIAS_AFFINE
    elif alter == 'gain': model.actuator_gainprm[0, 0] = 2.
    else: model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
    with pytest.raises(ValueError): audit(model)


def test_samples_preserve_absolute_clock_and_explicit_probe_semantics():
    samples = reference_samples([2., 2.1, 2.3], [[0.], [.2], [.2]], .02)
    assert [row['time_s'] for row in samples] == pytest.approx([2.05, 2.2, 2., 2.1])
    assert samples[-1]['source_indices'] == [0, 1, 2]
    assert not any(row['time_s'] == 2.3 for row in samples)
    assert samples[-1]['acceleration'] == pytest.approx([-100.])
