from types import SimpleNamespace
import numpy as np
import pytest
from weft.source_contact_control import authored_target_action,correction_at,prepare_correction,VARIABLES


def engine():
    names=tuple(['unselected'+str(i) for i in range(15)]+list(VARIABLES))
    features=np.arange(66,dtype=float).reshape(22,3)/200-0.1
    return SimpleNamespace(human_action_names=names,scale=2.,_kp=np.arange(1.,23.),_kd=np.arange(2.,24.),
        human_limits=np.full(22,20.),feature_matrix=lambda:features.copy(),data=SimpleNamespace(time=.1),
        source_start=.2,_times=np.array([.2,.4,.6]),model=SimpleNamespace(opt=SimpleNamespace(timestep=1/480)))


def test_zero_matches_original_authored_feedback_bitexact():
    e=engine();expected=np.clip(e.feature_matrix().sum(axis=1),-1.,1.)
    np.testing.assert_array_equal(authored_target_action(e,np.zeros(7),np.zeros(7)),expected)


def test_position_velocity_and_root_scaling_use_original_gains_then_clip():
    e=engine();q=np.arange(7)*.001;v=np.arange(7)*.01;ids=np.arange(15,22)
    expected=e.feature_matrix().sum(axis=1);expected[ids]+=(e._kp[ids]*q+e._kd[ids]*v)*[1,1,1,1,2,2,2]/e.human_limits[ids]
    actual=authored_target_action(e,q,v)
    np.testing.assert_array_equal(actual,np.clip(expected,-1.,1.))
    np.testing.assert_array_equal(actual[:15],e.feature_matrix().sum(axis=1)[:15])


def test_separate_sampling_preserves_original_time_inputs_and_slopes():
    e=engine();times=np.array([.2,.4,.6]);delta=np.zeros((3,7));delta[1]=.02
    before=delta.copy();q,v=correction_at(e,prepare_correction(times,delta))
    np.testing.assert_allclose(q,.01);np.testing.assert_allclose(v,.1)
    np.testing.assert_array_equal(delta,before);assert e.data.time==.1


@pytest.mark.parametrize('change',['nan_time','duplicate_time','reverse_time','too_long','bad_shape','nan_correction','endpoint','angle','root','rate'])
def test_invalid_reference_or_correction_fails_before_control(change):
    times=np.array([.2,.4,.6]);delta=np.zeros((3,7))
    if change=='nan_time':times[1]=np.nan
    elif change=='duplicate_time':times[1]=times[0]
    elif change=='reverse_time':times=times[::-1]
    elif change=='too_long':times=np.arange(129)*.1;delta=np.zeros((129,7))
    elif change=='bad_shape':delta=np.zeros((3,6))
    elif change=='nan_correction':delta[1,0]=np.nan
    elif change=='endpoint':delta[0,0]=.01
    elif change=='angle':delta[1,0]=.16
    elif change=='root':delta[1,4]=.09
    else:times=np.array([.2,.21,.22]);delta[1,0]=.1
    with pytest.raises(ValueError):prepare_correction(times,delta)


def test_prepared_slopes_are_immutable_and_clock_must_match():
    e=engine();delta=np.zeros((3,7));delta[1,0]=.01
    prepared=prepare_correction(e._times,delta);delta[1,0]=.1
    assert prepared.positions[1,0]==.01
    for array in (prepared.times,prepared.positions,prepared.slopes):
        with pytest.raises(ValueError):array.setflags(write=True)
    e._times=e._times+.01
    with pytest.raises(ValueError,match='source clock'):correction_at(e,prepared)
