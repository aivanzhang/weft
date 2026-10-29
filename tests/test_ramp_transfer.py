from copy import deepcopy
import math

import numpy as np
import pytest

from test_spatial_rigid import scene
from weft.ramp_transfer import initialize_prefix


def fixture():
    value=scene();angle=math.radians(20)
    tangent=np.array([math.cos(angle),0,-math.sin(angle)]);normal=np.array([math.sin(angle),0,math.cos(angle)])
    quaternion=[math.cos(angle/2),0,math.sin(angle/2),0]
    value['objects'][0].update(quaternion=quaternion,friction=[.12,.001,.0001])
    target=deepcopy(value['objects'][0])|{'id':'target','label':'Target','role':'passive',
        'position':[.6,.4,.06501],'quaternion':[1,0,0,0]}
    ramp=deepcopy(value['objects'][1])|{'id':'ramp','label':'Ramp','position':[.3,.4,.15],
        'size':[.4,.2,.02],'quaternion':quaternion}
    value['objects'] += [target,ramp]
    projection={'scope':'construction','source_sha256':'a'*64,'image_size':[1000,800],
        'matrix_px':[[1000,0,0,0],[0,0,-1000,800]],'calibration':'unverified_affine_hypothesis'}
    origin=np.asarray(ramp['position'])+normal*(.01+.025+1e-5)
    start=origin-.12*tangent;acceleration=9.81*(math.sin(angle)-.25*math.cos(angle))
    rows=[]
    for frame in range(19):
        elapsed=(frame-8)/60
        position=start+(.3*elapsed+.5*acceleration*elapsed**2)*tangent
        for obj,point in (('block',position),('target',np.array([.7,.4,.06501]))):
            pixel=(np.asarray(projection['matrix_px'])@np.r_[point,1]).tolist()
            rows.append({'source_id':f'{"a"*64}:frame:{frame}:{obj}','frame':frame,'time_s':frame/60,
                'object_id':obj,'center_px':pixel if frame>=8 else None,'identity':{'status':'analytic'},'scope':'construction'})
    return value,rows,projection,start


def initialize(value,rows,projection):
    return initialize_prefix(value,rows,projection,actor_id='block',target_id='target',ramp_id='ramp',support_id='floor')


def test_prefix_recovers_initial_state_without_changing_fixed_materials_or_gravity():
    value,rows,projection,start=fixture();before=deepcopy(value)
    candidate,report=initialize(value,rows,projection)
    actor=next(obj for obj in candidate['objects'] if obj['id']=='block')
    np.testing.assert_allclose(actor['position'],start,atol=1e-12)
    assert report['initial_downhill_speed']==pytest.approx(.3,abs=1e-12)
    assert report['prefix_max_error_px']<1e-9 and report['later_frames_read'] is False
    assert candidate['gravity']==value['gravity'] and candidate['contact']==value['contact']
    for old,new in zip(value['objects'],candidate['objects']):
        assert {k:old[k] for k in ('size','mass','friction','quaternion')}=={k:new[k] for k in ('size','mass','friction','quaternion')}
    assert value==before


@pytest.mark.parametrize('change',[
    lambda s,r,p:r.__setitem__(20,r[20]|{'center_px':None}),
    lambda s,r,p:r[-1].update(scope='evaluation'),
    lambda s,r,p:r[-1].update(frame=19,time_s=19/60),
    lambda s,r,p:r[24].update(center_px=[900,400]),
    lambda s,r,p:s.update(gravity=[0,0,0]),
    lambda s,r,p:s['objects'][0].update(quaternion=[1,0,0,0]),
])
def test_missing_future_inconsistent_or_nonphysical_prefix_is_rejected(change):
    value,rows,projection,_=fixture();change(value,rows,projection)
    with pytest.raises(ValueError):initialize(value,rows,projection)
