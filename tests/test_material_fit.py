from copy import deepcopy
import math

import pytest

from weft.material_fit import coast_candidate, surface_hypothesis, validate_inputs
from test_spatial_rigid import collision_scene, scene


def fixture():
    value=scene();value['dt']=1/120
    value['objects'][0].update(position=[.2,.4,.06501],velocity=[.4,0,0])
    value['objects'].insert(1,deepcopy(value['objects'][0])|{'id':'target','label':'Target','role':'passive',
        'position':[.7,.4,.06501],'velocity':[0,0,0]})
    projection={'scope':'construction','source_sha256':'a'*64,'image_size':[1000,800],
        'matrix_px':[[1000,0,0,0],[0,0,-1000,800]],'calibration':'unverified_affine_hypothesis'}
    rows=[]
    for frame in range(17):
        t=frame/120
        for obj,x in (('block',.2+.4*t-.5*.12*9.81*t*t),('target',.7)):
            rows.append({'source_id':f'{"a"*64}:frame:{frame}:{obj}','frame':frame,'time_s':t,'object_id':obj,
                'center_px':[1000*x,734.99], 'identity':{'status':'synthetic_analytic'},'scope':'construction'})
    return value,rows,projection


def test_distinct_contact_materials_recover_known_sliding_friction_from_analytic_positions():
    value,rows,projection=fixture();original=deepcopy(value);scores=[]
    for mu in (.06,.12,.24):
        proposed=surface_hypothesis(value,actor_id='block',target_id='target',support_id='floor',
            actor_friction=mu,target_friction=.25,mass_ratio=1)
        result=coast_candidate(proposed,rows,projection)
        scores.append((result['rmse_px'],mu))
        assert result['slots']==result['compared']==34 and len(result['events'])==17
    assert min(scores)[1]==.12 and min(scores)[0]<.3
    assert value==original


def test_mass_ratio_recovery_uses_both_post_collision_trajectories_not_only_target_motion():
    value=collision_scene();value['objects'].append(scene()['objects'][1]);value['dt']=1/120
    _,_,projection=fixture();rows=[]
    impact=(.5-.25-.08)/.6
    for frame in range(61):
        t=frame/120;after=max(0,t-impact)
        positions={'block':.25+.6*min(t,impact)+.3*after,'target':.5+.3*after}
        for obj,x in positions.items():
            rows.append({'source_id':f'{"a"*64}:frame:{frame}:{obj}','frame':frame,'time_s':t,'object_id':obj,
                'center_px':[1000*x,500], 'identity':{'status':'synthetic_analytic'},'scope':'construction'})
    scores=[]
    for ratio in (.25,1,4):
        proposed=surface_hypothesis(value,actor_id='block',target_id='target',support_id='floor',
            actor_friction=0,target_friction=0,mass_ratio=ratio)
        result=coast_candidate(proposed,rows,projection)
        scores.append((result['rmse_px'],ratio))
    # Soft contacts need not reproduce the ideal instantaneous collision exactly.
    ordered=sorted(scores)
    assert ordered[0][1]==1 and ordered[0][0]<.2*ordered[1][0], scores


def test_missing_observations_remain_slots_and_are_not_filled_or_scored():
    value,rows,projection=fixture();rows[8]['center_px']=None
    result=coast_candidate(value,rows,projection)
    assert result['slots']==34 and result['compared']==33
    assert result['rows'][8]['availability']=='observation_missing'
    assert result['rows'][8]['error_px'] is None and rows[8]['center_px'] is None


def test_episode_end_is_allowed_only_at_the_complete_source_boundary():
    value,rows,projection=fixture();value['task']['max_steps']=10;rows=rows[:22]
    result=coast_candidate(value,rows,projection)
    assert result['events'][-1]['truncated'] and result['compared']==22
    value['gravity']=[0,0,0];value['objects'][0]['velocity']=[0,0,0]
    value['target']=value['objects'][0]['position'][:]
    with pytest.raises(ValueError,match='complete source interval'):coast_candidate(value,rows,projection)


@pytest.mark.parametrize('change',[
    lambda s,r,p:r.pop(),lambda s,r,p:r.append(deepcopy(r[0])),
    lambda s,r,p:r[0].update(scope='holdout'),lambda s,r,p:p.update(scope='validation'),
    lambda s,r,p:r[4].update(time_s=.001),lambda s,r,p:r[0].update(source_id='b'*64+':frame:0:block'),
    lambda s,r,p:r[0].update(center_px=None),lambda s,r,p:p.update(matrix_px=[[0,0,0,0],[0,0,0,0]]),
    lambda s,r,p:r[4].update(center_px=[math.nan,0]),
])
def test_invalid_or_leaking_construction_contract_is_rejected(change):
    value,rows,projection=fixture();change(value,rows,projection)
    with pytest.raises(ValueError):validate_inputs(value,rows,projection,0,1)


@pytest.mark.parametrize('arguments',[
    {'actor_id':'floor'}, {'support_id':'target'}, {'actor_friction':-.1},
    {'mass_ratio':False}, {'target_friction':math.inf},
])
def test_unsafe_material_roles_and_values_are_rejected(arguments):
    value,_,_=fixture()
    parameters={'actor_id':'block','target_id':'target','support_id':'floor',
        'actor_friction':.12,'target_friction':.25,'mass_ratio':1}|arguments
    with pytest.raises(ValueError):surface_hypothesis(value,**parameters)
