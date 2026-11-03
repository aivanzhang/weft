"""Protocol validation and failure accounting, not source reconstruction fixtures."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest

PATH=Path(__file__).parents[1]/'scripts/studies/evaluate_unfamiliar_videos.py'
spec=importlib.util.spec_from_file_location('unseen_runner',PATH)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def plan():
    return {'profile':runner.PROFILE,'settings':deepcopy(runner.SETTINGS),'bounds':deepcopy(runner.BOUNDS),
            'model':{'path':'/tmp/pose.task','sha256':'e'*64},'scope':'development only',
            'cases':[{'id':'case-'+str(i),'original':'/tmp/'+str(i)+'.webm','original_sha256':str(i)*64,
                      'upload':'/tmp/'+str(i)+'.mp4','upload_sha256':str(i+3)*64,'provenance':'/tmp/'+str(i)+'.json'} for i in range(3)]}


def test_plan_detached_and_exact_scope():
    original=plan();got=runner.validate_plan(original);got['settings']['training']=True
    assert original['settings']['training'] is False
    assert runner.INTENT['inventory']=='unknown' and runner.INTENT['absence_verified'] is False
    assert runner.SETTINGS['modeling_scope']=='body_with_authored_ground_equipment_unmodeled'


@pytest.mark.parametrize('change',[
    lambda p:p['settings'].update(model='Full'),lambda p:p['settings'].update(retries=1),
    lambda p:p['settings'].update(root_assistance=True),lambda p:p['bounds'].update(api_calls=1),
    lambda p:p['bounds'].update(case_seconds=91),lambda p:p['settings'].update(retries=False),
    lambda p:p['cases'].pop(),lambda p:p['cases'][0].update(id='../outside'),
    lambda p:p['cases'][1].update(upload_sha256='3'*64),lambda p:p.update(override=True),
])
def test_no_post_outcome_setting_or_source_population_change(change):
    value=plan();change(value)
    with pytest.raises(ValueError):runner.validate_plan(value)


def test_failures_and_unattempted_stages_stay_denominator():
    rows=[{'status':'failed','stage':'preflight','body_reconstructed':True},
          {'status':'failed','stage':'worker'},
          {'status':'complete','preflight_ready':True,'body_reconstructed':True,
           'world_ready':True,'lab_ready':True,'playable':True,'full_authored_rollout':True,'task_success':False}]
    got=runner.summary(rows)
    assert got['denominator']==3 and got['failed_or_refused']==2
    assert got['body_reconstructed']==2 and got['playable']==1 and got['task_success']==0
    assert got['independent_holdout'] is False and got['physical_accuracy']=='not_evaluated'


def test_truthy_strings_do_not_promote_failures():
    got=runner.summary([{'status':'failed','task_success':'true','playable':1}])
    assert got['task_success']==got['playable']==0


def test_source_pin_change_refuses_before_any_pipeline(tmp_path):
    original=tmp_path/'source.webm';original.write_bytes(b'original')
    upload=tmp_path/'upload.mp4';upload.write_bytes(b'derivative')
    model=tmp_path/'pose.task';model.write_bytes(b'model')
    row={'original':str(original),'original_sha256':runner.digest(original),'upload':str(upload),'upload_sha256':runner.digest(upload)}
    modelrow={'path':str(model),'sha256':runner.digest(model)}
    runner._check_inputs(row,modelrow);upload.write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):runner._check_inputs(row,modelrow)
