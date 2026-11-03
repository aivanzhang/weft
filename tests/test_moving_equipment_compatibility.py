"""Pure advisory arithmetic; fixtures are structural, not authenticated source bodies."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from weft.io import canonical_hash
from weft import moving_equipment_compatibility as compatibility


def fixture():
    action={'dynamic_object_ids':['raw-0'],'human_motor_count':1,'names':['joint','raw-0_fx','raw-0_fy','raw-0_fz'],
            'physical_limits':[30,6.,6.,6.],'normalized_bounds':[-1,1],'hidden_assistance':False}
    scene={'profile':'shared-world-v2','gravity':[0,0,-9.81],
        'coordinate_frame':{'axes':'x_right_y_depth_z_up','units':{'length':'body_scaled_world_units','mass':'kilogram_prior','time':'second'},
                            'scale_bridge':{'origin_world':[1,2,.2]}},'human_spec':{'mobility':{}},
        'task':{'object_goal':{'virtual_acceleration':6.}},'action_contract':action,
        'objects':[{'id':'raw-0','shape':'sphere','role':'passive','radius':.1,'mass':1.,
                    'provenance':{'profile':'moving-equipment-free-body-v1','hidden_assistance':False,'trajectory_driving':False}}]}
    task={'profile':'moving-equipment-placement-task-v1','binding':{'scene_sha256':canonical_hash(scene)},
          'control':'explicit_human_root_and_virtual_object_forces','action_contract':deepcopy(action),
          'goal':{'object_id':'raw-0','position_world':[1,2,.6],'position_tolerance_world':.03,
                  'speed_tolerance_world_s':.1,'terminal_consecutive_hold_s':.5}}
    return scene,task


def encoded(scene,task):
    task=deepcopy(task);task['binding']['scene_sha256']=canonical_hash(scene)
    return json.dumps(scene).encode(),json.dumps(task).encode()


def test_elevated_support_and_controller_warning_exact_bytes_no_task_change():
    scene,task=fixture();raw=encoded(scene,task)
    result=compatibility.assess_moving_equipment_compatibility(*raw,controller_profile=compatibility.FEEDBACK_PROFILE)
    assert result['status']=='assessed' and result['blocks_manual_exploration'] is False
    assert result['input_sha256']['scene_bytes']==sha256(raw[0]).hexdigest()
    assert result['input_sha256']['task_bytes']==sha256(raw[1]).hexdigest()
    facts=result['facts']
    assert facts['steady_support_force_prior_units']==pytest.approx(3.81)
    assert facts['continuous_unsupported_speed_window_s']==pytest.approx(.2/3.81)
    assert facts['minimum_goal_sphere_bottom_above_floor_world']==pytest.approx(.27)
    assert facts['hold_exceeds_continuous_unsupported_speed_window'] is True
    assert [x['code'] for x in result['warnings']]==['elevated_goal_needs_support_for_sustained_balance','nominal_feedback_has_no_explicit_support_coordination']
    assert encoded(scene,task)==raw and len(json.dumps(result))<4096


@pytest.mark.parametrize('case',['short_hold','floor_reachable','enough_force','unknown_controller','manual'])
def test_no_false_global_infeasibility_or_controller_rejection(case):
    s,t=fixture();controller='manual'
    if case=='short_hold':t['goal']['terminal_consecutive_hold_s']=.01
    elif case=='floor_reachable':t['goal']['position_world'][2]=.3
    elif case=='enough_force':
        s['task']['object_goal']['virtual_acceleration']=12.
        s['action_contract']['physical_limits'][1:]=[12.]*3;t['action_contract']=deepcopy(s['action_contract'])
    elif case=='unknown_controller':controller='future-contact-aware'
    r=compatibility.assess_moving_equipment_compatibility(*encoded(s,t),controller_profile=controller)
    assert r['status']=='assessed' and not r['blocks_manual_exploration']
    if case=='short_hold':assert not r['facts']['hold_exceeds_continuous_unsupported_speed_window']
    if case in ('floor_reachable','enough_force'):assert not r['warnings']
    if case=='unknown_controller':assert r['controller_assessment']=='unassessed_unknown_controller'
    assert 'task impossibility' in r['scope']


@pytest.mark.parametrize('damage',['shape','gravity','floor','units','missing','force_limit','force_order','action_mismatch',
    'hidden','trajectory','static','mass','radius','bool_bound','goal_below','profile'])
def test_unsupported_inputs_explicitly_unassessed(damage):
    s,t=fixture()
    if damage=='shape':s['objects'][0]['shape']='box'
    elif damage=='gravity':s['gravity']=[.1,0,-9.81]
    elif damage=='floor':s['human_spec']['mobility']['ground_plane']={}
    elif damage=='units':s['coordinate_frame']['units']['mass']='measured_kg'
    elif damage=='missing':del s['coordinate_frame']['scale_bridge']
    elif damage=='force_limit':s['action_contract']['physical_limits'][-1]=60;t['action_contract']=deepcopy(s['action_contract'])
    elif damage=='force_order':s['action_contract']['names'][-1]='raw-0_torque';t['action_contract']=deepcopy(s['action_contract'])
    elif damage=='action_mismatch':t['action_contract']['physical_limits'][-1]=2
    elif damage=='hidden':s['objects'][0]['provenance']['hidden_assistance']=True
    elif damage=='trajectory':s['objects'][0]['provenance']['trajectory_driving']=True
    elif damage=='static':s['objects'][0]['role']='static'
    elif damage=='mass':s['objects'][0]['mass']=True
    elif damage=='radius':s['objects'][0]['radius']=0
    elif damage=='bool_bound':s['action_contract']['normalized_bounds']=[-1,True];t['action_contract']=deepcopy(s['action_contract'])
    elif damage=='goal_below':t['goal']['position_world'][2]=0
    else:s['profile']='arbitrary-world'
    r=compatibility.assess_moving_equipment_compatibility(*encoded(s,t))
    assert r['status']=='unassessed' and not r['warnings'] and not r['blocks_manual_exploration']


def test_task_scene_join_and_malformed_json():
    s,t=fixture();a,b=encoded(s,t);s['gravity'][2]=-10
    r=compatibility.assess_moving_equipment_compatibility(json.dumps(s).encode(),b)
    assert r['reason']=='task_scene_binding_mismatch'
    for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}',b'[]',b'no',b'{"x":'+b'9'*5000+b'}'):
        r=compatibility.assess_moving_equipment_compatibility(raw,b)
        assert r['status']=='unassessed' and not r['blocks_manual_exploration']


def test_byte_caps_and_current_pretty_scene_envelope():
    s,t=fixture();s['retained_audit']='x'*650000
    r=compatibility.assess_moving_equipment_compatibility(*encoded(s,t))
    assert r['status']=='assessed'
    r=compatibility.assess_moving_equipment_compatibility(b'x'*(compatibility.MAX_SCENE_BYTES+1),b'{}')
    assert r['reason']=='input_byte_bound_or_type' and r['input_sha256']=={}


def test_time_bound_never_publishes_assessed(monkeypatch):
    times=iter([0,2]);monkeypatch.setattr(compatibility.time,'monotonic',lambda:next(times))
    r=compatibility.assess_moving_equipment_compatibility(*encoded(*fixture()))
    assert r['status']=='unassessed' and r['reason']=='preflight_time_bound' and not r['warnings']


def test_fresh_import_has_no_native_or_provider_modules():
    script='import sys; import weft.moving_equipment_compatibility; assert not any(x in sys.modules for x in ("mujoco","numpy","gymnasium","openai","weft.shared_world","weft.moving_equipment_task"))'
    env={'PATH':os.environ.get('PATH',''),'PYTHONPATH':str(Path('src').absolute())}
    subprocess.run([sys.executable,'-c',script],check=True,env=env,timeout=5)


def test_huge_integer_required_field_is_unassessed_not_overflow():
    scene,task=fixture();scene['objects'][0]['mass']=10**400
    result=compatibility.assess_moving_equipment_compatibility(*encoded(scene,task))
    assert result['status']=='unassessed' and result['reason']=='invalid_mass_prior'
