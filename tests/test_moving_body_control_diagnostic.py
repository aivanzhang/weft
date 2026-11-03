"""Analytic telemetry parity; only the original sealed admission capability is fixture-stubbed."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from test_moving_equipment_learning import admitted,policy_fixture
from test_moving_equipment_task import example
from weft import moving_equipment_learning as learning
from weft import moving_body_control_diagnostic as module
from weft.moving_equipment_task import MovingEquipmentTaskEnv


def exact_inputs(admission):
    policy=policy_fixture(admission)
    env=MovingEquipmentTaskEnv(admission.scene,admission.task,admission.bundle)
    actions=[]
    try:
        env.reset(seed=policy['training_seed']);feedback=learning._Feedback(env,admission.scene,admission.task,policy['weights'])
        while True:
            action=feedback();actions.append(action.tolist())
            _,_,terminated,truncated,_=env.step(action)
            if terminated or truncated:break
        metrics=env.metrics();qpos=env._engine.data.qpos[env._engine._q].tolist()
    finally:env.close()
    return learning._bytes(policy),learning._bytes(metrics),actions,qpos


def run(admission,tmp_path,inputs,**kw):
    policy,metrics,*_=inputs
    return module.diagnose_moving_body_control(admission,policy,tmp_path/'diagnostic',
        policy_sha256=sha256(policy).hexdigest(),expected_metrics_raw=metrics,
        expected_metrics_sha256=sha256(metrics).hexdigest(),**kw)


def test_full_native_timing_exact_scorer_actions_and_no_extra_native_calls(admitted,tmp_path,monkeypatch):
    import mujoco
    counts={name:0 for name in ('mj_step','mj_forward','mj_inverse')}
    for name in counts:
        original=getattr(mujoco,name)
        def counted(*args,_name=name,_original=original,**kwargs):
            counts[_name]+=1;return _original(*args,**kwargs)
        monkeypatch.setattr(mujoco,name,counted)
    inputs=exact_inputs(admitted);baseline=dict(counts)
    counts.update({n:0 for n in counts})
    result=run(admitted,tmp_path,inputs)
    assert result['status']=='complete' and result['exact_saved_metrics_match'] and result['closed']
    assert counts==baseline and counts['mj_inverse']==0
    assert result['model_compilations']==1
    trace=[json.loads(line) for line in (tmp_path/'diagnostic/substeps.jsonl').read_bytes().splitlines()]
    controls=[json.loads(line) for line in (tmp_path/'diagnostic/controls.jsonl').read_bytes().splitlines()]
    assert [r['action'] for r in controls]==inputs[2]
    assert trace[-1]['qpos']==inputs[3]
    assert trace[-1]['end_s']==pytest.approx(admitted.task['interval']['native_interval']['duration_s'])
    assert len(trace)==admitted.task['interval']['horizon_steps']*admitted.scene['substeps']
    assert max(r['angular_rmse_rad'] for r in trace)==result['metrics']['peak_angular_rmse_rad']
    assert all(r['solve_cache_time_s']==r['start_s'] for r in trace)
    assert all(len(r['bias_force'])==22 and len(r['constraint_force'])==22 for r in trace)
    manifest=json.loads((tmp_path/'diagnostic/manifest.json').read_bytes())
    for name,pin in manifest['files'].items():assert sha256((tmp_path/'diagnostic'/name).read_bytes()).hexdigest()==pin


def test_expected_metrics_mismatch_is_preserved_not_repaired(admitted,tmp_path):
    policy,expected,*tail=exact_inputs(admitted);value=json.loads(expected);value['peak_angular_rmse_rad']+=.01
    result=run(admitted,tmp_path,(policy,learning._bytes(value),*tail))
    assert result['status']=='baseline_mismatch' and result['failure'] is None
    assert not result['exact_saved_metrics_match'] and result['runtime_readback_complete']
    assert result['metrics']['peak_angular_rmse_rad']!=value['peak_angular_rmse_rad']


@pytest.mark.parametrize('damage',['policy_pin','metrics_pin','task_pin','nonfinite'])
def test_input_custody_refuses_before_native_model(admitted,tmp_path,monkeypatch,damage):
    policy=learning._bytes(policy_fixture(admitted));metrics=learning._bytes({'task_sha256':learning.canonical_hash(admitted.task)})
    options={'policy_sha256':sha256(policy).hexdigest(),'expected_metrics_raw':metrics,'expected_metrics_sha256':sha256(metrics).hexdigest()}
    if damage=='policy_pin':options['policy_sha256']='0'*64
    elif damage=='metrics_pin':options['expected_metrics_sha256']='0'*64
    elif damage=='task_pin':
        metrics=learning._bytes({'task_sha256':'0'*64});options.update(expected_metrics_raw=metrics,expected_metrics_sha256=sha256(metrics).hexdigest())
    else:
        policy=b'{"nonfinite":NaN}';options['policy_sha256']=sha256(policy).hexdigest()
    monkeypatch.setattr(module,'MovingEquipmentTaskEnv',lambda *a,**k:pytest.fail('Invalid input cannot compile'))
    with pytest.raises(ValueError):module.diagnose_moving_body_control(admitted,policy,tmp_path/'diagnostic',**options)
    assert not (tmp_path/'diagnostic').exists()


def test_trace_bound_preserves_incomplete_output_and_closes(admitted,tmp_path,monkeypatch):
    inputs=exact_inputs(admitted)
    monkeypatch.setattr(module,'MAX_TRACE_BYTES',1)
    result=run(admitted,tmp_path,inputs)
    assert result['status']=='incomplete' and 'trace exceeds' in result['failure'] and result['closed']
    assert result['native_substeps']==0 and not result['exact_saved_metrics_match']
    assert (tmp_path/'diagnostic/manifest.json').exists()


def test_changed_runtime_is_not_a_qualified_trace(admitted,tmp_path,monkeypatch):
    inputs=exact_inputs(admitted);original=learning._runtime;calls=[]
    def changed(*a,**k):
        hashes,versions,files=original(*a,**k);calls.append(1)
        if len(calls)==3:hashes={**hashes,'unexpected.py':'0'*64}
        return hashes,versions,files
    monkeypatch.setattr(learning,'_runtime',changed)
    result=run(admitted,tmp_path,inputs)
    assert result['status']=='incomplete' and 'runtime changed' in result['failure']
    assert result['exact_saved_metrics_match'] and not result['runtime_readback_complete']


def test_deadline_after_step_preserves_partial_counts_no_retry(admitted,tmp_path,monkeypatch):
    inputs=exact_inputs(admitted);original=module._check;calls=[]
    def limited(deadline):
        calls.append(1)
        if len(calls)>12:raise TimeoutError('analytical deadline')
        original(deadline)
    monkeypatch.setattr(module,'_check',limited)
    result=run(admitted,tmp_path,inputs)
    assert result['status']=='incomplete' and result['closed'] and result['model_compilations']==1
    assert 0<result['native_substeps']<admitted.task['interval']['horizon_steps']*admitted.scene['substeps']
    assert result['failure'] and not result['exact_saved_metrics_match']
    assert any('analytical deadline' in cause for cause in result['failure_chain'])


def test_refuses_output_reuse_and_oversized_interval_before_model(admitted,tmp_path,monkeypatch):
    policy=learning._bytes(policy_fixture(admitted));metrics=learning._bytes({'task_sha256':learning.canonical_hash(admitted.task)})
    monkeypatch.setattr(module,'MAX_CONTROLS',1)
    monkeypatch.setattr(module,'MovingEquipmentTaskEnv',lambda *a,**k:pytest.fail('Cannot truncate full interval'))
    with pytest.raises(ValueError,match='full interval'):
        run(admitted,tmp_path,(policy,metrics))


def test_partial_trace_open_releases_first_handle(tmp_path,monkeypatch):
    original=Path.open;opened=[]
    def controlled(path,*args,**kwargs):
        if path.name=='substeps.jsonl':raise OSError('analytical disk refusal')
        handle=original(path,*args,**kwargs);opened.append(handle);return handle
    monkeypatch.setattr(Path,'open',controlled)
    with pytest.raises(OSError,match='disk refusal'):module._Trace(tmp_path,float('inf'))
    assert len(opened)==1 and opened[0].closed
