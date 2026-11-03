from copy import deepcopy
from hashlib import sha256
import json
import subprocess
import sys

import pytest

from weft import portal_task_contract as c
from weft.io import canonical_hash


def create():
    return {"request_id":"1"*32,"parent_manifest_sha256":"2"*64,"task":{
        "target_object_id":"raw-0","target_position_world":[.5,.5,.5],
        "position_tolerance_world":.03,"speed_tolerance_world_s":.1,"hold_duration_s":.5}}


def learning():
    return {"request_id":"1"*32,"task_sha256":"3"*64,
        "training":{"seed":1729,"generations":1,"population":4,"max_seconds":30},
        "evaluation":{"evaluation_seeds":[4101,4102],"position_jitter":.025,"velocity_jitter":.1,"max_seconds":30}}


def bindings(task=False):
    value={name:"a"*64 for name in c._BINDINGS}
    value.update(parent_project_id="a"*32,parent_manifest_sha256="2"*64,
                 task_sha256="3"*64 if task else None,
                 scorer_sha256={name:"b"*64 for name in c._SCORERS} if task else None)
    return value


def scene():
    return {"bounds":[2.,2.,2.],"objects":[{"id":"raw-0","position":[.4,.4,.4]}],
        "action_contract":{"dynamic_object_ids":["raw-0"],"names":["human-knee","raw-0_fx","raw-0_fy","raw-0_fz"]},
        "dt":1/30,"task":{"start_offset_s":0,"horizon_steps":458,"native_interval":{"duration_s":15.233}}}


def metrics(success=False):
    return {"task_sha256":"3"*64,"success":success,"completed_interval":True,
        "failed_criteria":[] if success else ["peak_angular_rmse_rad","terminal_consecutive_placement_hold"],
        "failure":None,"elapsed_s":15.233,"duration_s":15.233,"object_distance_world":.316,
        "object_speed_world_s":.2,"terminal_consecutive_hold_s":0,"angular_rmse_rad":.5,
        "peak_angular_rmse_rad":1.063,"mean_reward":.4}


def outputs():
    source={"task_sha256":"3"*64,"scene_sha256":"4"*64,"task_manifest_sha256":"7"*64,"parent_manifest_sha256":"8"*64}
    result={"status":"complete","policy_sha256":"5"*64,"control_steps":1502,"maximum_control_steps":3656}
    nominal={"profile":"moving-equipment-feedback-policy-v1","source":source,"policy_sha256":"5"*64,
        "independent_changed_start_evaluation":"unavailable","selected_kind":"authored_unchanged",
        "selected_policy_success":False,"comparisons":[{"controller":name,"reused_training_trial":i<2,
         "trial":{"status":"failed" if name=="random" else "complete","failure":"object_left_bounds" if name=="random" else None,
         "control_steps":100 if name=="random" else 458,"metrics":metrics()}} for i,name in enumerate(c.CONTROLLERS)]}
    evaluation={"status":"complete","profile":"moving-equipment-object-reset-evaluation-v1","source":source,
        "policy_sha256":"5"*64,"protocol_sha256":"6"*64,"requested_starts":3,"admitted_changed_starts":2,
        "excluded_or_refused_starts":1,"control_steps":2772,"maximum_control_steps":5496,
        "final_holdout_independence":"not_established","physical_validation":"unverified",
        "comparisons":[{"controller":name,"successes":0,"requested_starts":3,"admitted_starts":2} for name in c.CONTROLLERS]}
    return result,nominal,evaluation


def raw(value): return json.dumps(value).encode()


def test_create_is_explicit_detached_and_no_training_fields():
    value=create();checked=c.validate_task_create_request(value)
    checked["task"]["target_position_world"][0]=1
    assert value["task"]["target_position_world"][0]==.5
    receipt=c.build_task_action_receipt("create",value,bindings())
    assert receipt["api_calls_authorized"]==0 and not receipt["policy_installation_authorized"]
    assert receipt["request_sha256"]==canonical_hash(value)
    assert c.validate_task_action_receipt(receipt)==receipt


@pytest.mark.parametrize("change",["target","tolerance","hold","model","training","request_id","huge"])
def test_closed_create_refusals(change):
    value=create()
    if change=="target":value['task']['target_position_world']=[0,0]
    elif change=="tolerance":value['task']['position_tolerance_world']=0
    elif change=="hold":value['task']['hold_duration_s']=61
    elif change=="model":value['task']['model_path']='/tmp/model'
    elif change=="training":value['training']=True
    elif change=="request_id":value['request_id']='A'*32
    else:value['task']['hold_duration_s']=10**10000
    with pytest.raises(ValueError):c.validate_task_create_request(value)


def test_learning_freezes_prospective_seeds_and_aggregate_work():
    value=learning();assert c.validate_task_learning_request(value,horizon_steps=458)==value
    receipt=c.build_task_action_receipt('train_evaluate',value,bindings(True))
    assert receipt['request']['evaluation']['evaluation_seeds']==[4101,4102]
    assert c.validate_task_action_receipt(receipt)==receipt
    with pytest.raises(ValueError,match='combined'):c.validate_task_learning_request(value,horizon_steps=1251)
    value['evaluation']=None
    assert c.validate_task_learning_request(value,horizon_steps=2000)==value


@pytest.mark.parametrize('change',['same_seed','duplicate','zero_jitter','large_jitter','population','wall','bool','policy'])
def test_learning_settings_cannot_reuse_training_seed_or_expand_authority(change):
    value=learning()
    if change=='same_seed':value['evaluation']['evaluation_seeds'][0]=1729
    elif change=='duplicate':value['evaluation']['evaluation_seeds']=[4101,4101]
    elif change=='zero_jitter':value['evaluation'].update(position_jitter=0,velocity_jitter=0)
    elif change=='large_jitter':value['evaluation']['position_jitter']=.051
    elif change=='population':value['training']['population']=13
    elif change=='wall':value['training']['max_seconds']=61
    elif change=='bool':value['training']['seed']=True
    else:value['policy_path']='/tmp/other-policy'
    with pytest.raises(ValueError):c.validate_task_learning_request(value)


@pytest.mark.parametrize('change',['task','parent','scorer','install','bound'])
def test_receipt_hashes_and_authority_closed(change):
    receipt=c.build_task_action_receipt('train_evaluate',learning(),bindings(True))
    if change=='task':receipt['bindings']['task_sha256']='f'*64
    elif change=='parent':receipt['bindings']['parent_project_id']='wrong'
    elif change=='scorer':receipt['bindings']['scorer_sha256']['extra.py']='f'*64
    elif change=='install':receipt['policy_installation_authorized']=True
    else:receipt['request']['training']['max_seconds']=120
    with pytest.raises(ValueError):c.validate_task_action_receipt(receipt)


def test_options_use_exact_world_and_native_duration_without_native_import():
    world=scene();task={'binding':{'scene_sha256':canonical_hash(world)}}
    result=c.task_options(world,parent_manifest_sha256='2'*64,task=task)
    assert result['duration_s']==15.233 and result['horizon_steps']==458
    assert result['scene_sha256']==canonical_hash(world) and result['task_sha256']==canonical_hash(task)
    assert result['objects']==[{'id':'raw-0','initial_position_world':[.4,.4,.4]}]
    result['objects'][0]['initial_position_world'][0]=1
    assert world['objects'][0]['position'][0]==.4
    p=subprocess.run([sys.executable,'-c',"import sys;import weft.portal_task_contract;assert not any(n in sys.modules for n in ('numpy','mujoco','cv2','weft.human','weft.shared_world'))"],capture_output=True,timeout=10)
    assert p.returncode==0,p.stderr


@pytest.mark.parametrize('change',['inventory','bounds','start','task','malformed'])
def test_options_refuse_unknown_or_mismatched_context(change):
    world=scene();task=None
    if change=='inventory':world['objects'][0]['id']='other'
    elif change=='bounds':world['objects'][0]['position'][0]=3
    elif change=='start':world['task']['start_offset_s']=1
    elif change=='task':task={'binding':{'scene_sha256':'e'*64}}
    else:world['action_contract']['dynamic_object_ids']=[[]]
    with pytest.raises(ValueError):c.task_options(world,parent_manifest_sha256='2'*64,task=task)


def test_report_preserves_failed_baselines_and_actual_refused_denominators():
    result,nominal,evaluation=outputs()
    report=c.summarize_task_learning(raw(result),raw(nominal),task_sha256='3'*64,scene_sha256='4'*64,evaluation_result_raw=raw(evaluation))
    assert report['training']['selected_kind']=='authored_unchanged' and not report['training']['selected_policy_success']
    assert report['training']['comparisons'][2]['trial']['failure']=='object_left_bounds'
    assert report['training']['comparisons'][-1]['trial']['metrics']['failed_criteria']==metrics()['failed_criteria']
    assert report['evaluation']['requested_starts']==3 and report['evaluation']['admitted_changed_starts']==2
    assert report['evaluation']['excluded_or_refused_starts']==1
    assert all(r['successes']==0 for r in report['evaluation']['comparisons'])
    assert not report['policy_installed'] and report['final_holdout_independence']=='not_established'
    assert report['input_sha256']['changed_start_result']==sha256(raw(evaluation)).hexdigest()


@pytest.mark.parametrize('change',['policy','task','missing_baseline','denominator','success','scope','controls','nominal_success'])
def test_report_cannot_promote_wrong_policy_or_missing_failed_trials(change):
    result,nominal,evaluation=outputs()
    if change=='policy':evaluation['policy_sha256']='f'*64
    elif change=='task':nominal['source']['task_sha256']='f'*64
    elif change=='missing_baseline':nominal['comparisons'].pop(2)
    elif change=='denominator':evaluation['excluded_or_refused_starts']=0
    elif change=='success':evaluation['comparisons'][0]['successes']=3
    elif change=='scope':evaluation['final_holdout_independence']='verified'
    elif change=='controls':evaluation['maximum_control_steps']=19000
    else:nominal['comparisons'][-1]['trial']['metrics']['success']=True
    with pytest.raises(ValueError):c.summarize_task_learning(raw(result),raw(nominal),task_sha256='3'*64,scene_sha256='4'*64,evaluation_result_raw=raw(evaluation))


def test_no_admitted_starts_or_optional_evaluation_never_becomes_success():
    result,nominal,evaluation=outputs()
    assert c.summarize_task_learning(raw(result),raw(nominal),task_sha256='3'*64,scene_sha256='4'*64)['evaluation'] is None
    evaluation.update(admitted_changed_starts=0,excluded_or_refused_starts=3)
    for row in evaluation['comparisons']:row['admitted_starts']=0
    checked=c.summarize_task_learning(raw(result),raw(nominal),task_sha256='3'*64,scene_sha256='4'*64,evaluation_result_raw=raw(evaluation))
    assert checked['evaluation']['evidence']=='unavailable_no_admitted_changed_starts'


def test_same_world_task_but_different_admitted_parent_report_rejected():
    result,nominal,evaluation=outputs();evaluation=deepcopy(evaluation)
    evaluation['source']['parent_manifest_sha256']='e'*64
    with pytest.raises(ValueError,match='admitted source package'):
        c.summarize_task_learning(raw(result),raw(nominal),task_sha256='3'*64,scene_sha256='4'*64,evaluation_result_raw=raw(evaluation))


def test_failure_keeps_exact_artifact_pin_and_never_installs_policy():
    failure=raw({'status':'failed','reason':'bounded evaluation timeout','starts':[]})
    result=c.summarize_task_failure(failure,stage='evaluation')
    assert result['failure_sha256']==sha256(failure).hexdigest() and not result['task_success'] and not result['policy_installed']
    with pytest.raises(ValueError):c.summarize_task_failure(b'{"reason":"a","reason":"b"}',stage='training')
    with pytest.raises(ValueError):c.summarize_task_failure(b' '*(2 << 20)+b'{}',stage='training')


def test_encoding_deadline_and_nonfinite_reject(monkeypatch):
    ticks=iter([0.,3.]);monkeypatch.setattr(c.time,'monotonic',lambda:next(ticks))
    with pytest.raises(ValueError,match='work/byte'):c._encode({'x':1})
