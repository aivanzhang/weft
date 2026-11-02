"""Explicit local validator recovery; original answers, source pins and paid calls stay immutable."""

from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3

from avatar_assets.safety import artifacts,local_file,new_directory,read_json,require,sha256,write_json

POLICY='cached-planner-runtime-revision-v1'
ALLOWED={'action_intent.py','pipeline.py','pipeline_recovery.py'}
AUTHOR_INPUT_ALLOWED={'pipeline_prompts.py','pipeline_recovery.py','pipeline.py'}


def _change_policy(profile,changes):
    if profile=='author_input_bound':
        return set(changes)<=AUTHOR_INPUT_ALLOWED and {'pipeline_prompts.py','pipeline_recovery.py'}<=set(changes)
    return profile=='planning' and set(changes)<=ALLOWED and {'action_intent.py','pipeline.py','pipeline_recovery.py'}<=set(changes)


def _rows(folder):
    rows=artifacts(folder)
    require(len(rows)<=160 and sum(row['bytes'] for row in rows)<=16*1024**2,'Runtime source inventory exceeds bound')
    return rows


def _job(job):
    job=Path(job)
    root=job.parent.parent
    require(job.resolve()==job and job.parent==root/'outputs' and job.name.startswith('studio-'),
            'Recovery requires an existing Studio output')
    return job,root


def _planner_ledger(root,job,proof):
    path=local_file(root/'.local/openai/ledger.sqlite3',64*1024**2)
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=.2)) as db:
        db.execute('PRAGMA query_only=ON')
        grant=db.execute('SELECT digest,calls,ceiling FROM jobs WHERE id=?',(job.name,)).fetchone()
        calls=db.execute('SELECT role,state,request_digest,response_id,reserved,actual,usage FROM calls WHERE job=?',(job.name,)).fetchall()
        retired=db.execute('SELECT 1 FROM retirements WHERE parent=?',(job.name,)).fetchone()
    ceiling=7 if proof.get('policy')=='author-input-bound-custody-v1' else 3
    roles={'planner'}|{f'{role}-{attempt}' for role in ('author','reviewer') for attempt in range(3 if ceiling==7 else 1)}
    require(proof['call_ceiling_preserved']==ceiling and not retired
            and grant==(proof['request_sha256'],ceiling,proof['dollar_ceiling_microdollars_preserved']),
            'Original API grant changed or was retired')
    require(1<=len(calls)<=ceiling and {row[0] for row in calls}<=roles,
            'Recovery cannot change the original call ceiling or roles')
    planner=[row for row in calls if row[0]=='planner']
    saved=proof['calls'][0]
    from avatar_assets.openai_runner import digest
    request=read_json(job/'agents/planner/request.json')
    require(len(planner)==1 and planner[0][1:6]==('settled',digest(request),saved['response_id'],
            saved['reserved_microdollars'],saved['actual_microdollars'])
            and json.loads(planner[0][6])==saved['usage'],'Settled planner accounting changed')


def prepare_revision(job,revision,*,expected_request_sha256,expected_changes,reason,profile="planning"):
    """Prepare only; an explicit Studio adoption is still required before execution."""
    from avatar_assets.pipeline import _pipeline_sources
    from avatar_assets.pipeline_prompts import validate_plan
    from avatar_assets.studio_inspection import collect_api_planning
    job,root=_job(job);revision=Path(revision)
    require(revision.resolve().parent==root/'outputs' and revision!=job,'Revision must use a new sibling output folder')
    require(isinstance(reason,str) and 1<=len(reason.strip())<=800,'A bounded operator reason is required')
    request=read_json(job/'request.json');receipt=read_json(job/'input-receipt.json')
    require(sha256(job/'request.json')==expected_request_sha256==receipt['request_sha256'],
            'Recovery request differs from the explicitly reviewed request')
    require(profile in ('planning','author_input_bound'),'Unknown runtime recovery profile')
    require((request['max_agent_calls'],request['max_repairs'])==((7,2) if profile=='author_input_bound' else (3,0)),
            'Recovery must preserve its exact call and repair limits')
    state=Path(read_json(root/'.local/studio-registry.json')['state_dir'])
    private=state/'jobs'/job.name.removeprefix('studio-')
    studio_request=read_json(private/'request.json')
    proof=collect_api_planning(root,job,studio_request,private/'pipeline-request.json',profile=profile)
    before={row['path']:row for row in receipt['sources']};sources=_pipeline_sources()
    require(_rows(job/'sources')==receipt['sources'],'Original runtime snapshots changed')
    after={name:{'path':name,'bytes':path.stat().st_size,'sha256':sha256(local_file(path,1024**2))}
           for name,path in sources.items()}
    require(set(before)<=set(after),'Recovery cannot remove a pinned source')
    changes={name:{'before':before.get(name,{}).get('sha256'),'after':row['sha256']}
             for name,row in after.items() if name not in before or before[name]['sha256']!=row['sha256']}
    require(changes==expected_changes and _change_policy(profile,changes),
            'Runtime revision must exactly match the reviewed validator/recovery changes')
    answer=read_json(job/'agents/planner/answer.json');plan=validate_plan(answer,request)
    require(plan['status']=='ready','The unchanged cached answer still does not produce a ready plan')
    if profile=='author_input_bound':
        require(read_json(job/'plan.json')==plan,'Existing paid plan differs from the unchanged cached answer')
    revision=new_directory(revision);(revision/'sources').mkdir()
    for name,path in sources.items():
        target=revision/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    require(_rows(revision/'sources')==sorted(after.values(),key=lambda row:row['path']),
            'Runtime source changed while preparing the revision')
    shutil.copyfile(job/'status.json',revision/'original-failure-status.json')
    write_json(revision/'api-custody.json',proof)
    write_json(revision/'plan.json',plan)
    record={'schema_version':1,'policy':POLICY,'profile':profile,'job':str(job.relative_to(root)),
            'request_sha256':expected_request_sha256,'input_receipt_sha256':sha256(job/'input-receipt.json'),
            'reason':reason,'changes':changes,'sources_before':receipt['sources'],
            'sources':_rows(revision/'sources'),'planner_artifacts':artifacts(job/'agents/planner'),
            'runner_sha256':sha256(job/'agents/runner.json'),'original_events':artifacts(job/'events'),
            'plan_sha256':sha256(revision/'plan.json'),'new_planner_calls':0,'remaining_role_calls':request['max_agent_calls']-1,
            'original_grant_unchanged':True,
            **({'inspection_artifacts':artifacts(job/'inspection'),
                'inspection_manifest_sha256':sha256(job/'inspection/manifest.json'),
                'existing_plan_sha256':sha256(job/'plan.json')} if profile=='author_input_bound' else {})}
    write_json(revision/'revision.json',record)
    write_json(revision/'manifest.json',{'schema_version':1,'policy':POLICY,'status':'prepared_not_adopted',
                                       'artifacts':_rows(revision)})
    return {'revision_path':str(revision/'manifest.json'),'sha256':sha256(revision/'manifest.json'),
            'changes':changes,'new_planner_calls':0,'status':'prepared_not_adopted'}


def verify_runtime_revision(job,live_sources,*,revision=None):
    """Verify saved and live code plus the original planner; this function cannot submit a request."""
    job,root=_job(job);pointer_path=job/'runtime-revision.json'
    pointer=None
    if revision is None:
        if not pointer_path.exists():return None
        pointer=read_json(pointer_path)
        require(set(pointer)=={'schema_version','path','sha256'} and pointer['schema_version']==1,
                'Invalid runtime revision pointer')
        require(isinstance(pointer['path'],str) and not Path(pointer['path']).is_absolute()
                and '..' not in Path(pointer['path']).parts,'Unsafe runtime revision path')
        revision=root/pointer['path']
    revision=Path(revision)
    require(revision.resolve()==revision and revision.parent==root/'outputs' and revision!=job,
            'Unsafe runtime revision folder')
    manifest=read_json(revision/'manifest.json')
    require(manifest.get('policy')==POLICY and manifest.get('artifacts')==_rows(revision),
            'Runtime revision artifacts changed')
    if pointer:
        require(pointer['sha256']==sha256(revision/'manifest.json'),'Runtime revision receipt changed')
        state=Path(read_json(root/'.local/studio-registry.json')['state_dir'])
        with closing(sqlite3.connect((state/'jobs.sqlite3').as_uri()+'?mode=ro',uri=True,timeout=.2)) as db:
            row=db.execute('SELECT status,data FROM jobs WHERE id=?',(job.name.removeprefix('studio-'),)).fetchone()
        require(row is not None and row[0] in ('planned','queued','launching','running','planning','needs_inspection','unsupported','complete')
                and json.loads(row[1]).get('runtime_revision')==pointer,'Runtime revision was not adopted through Studio')
    record=read_json(revision/'revision.json')
    require(record.get('policy')==POLICY and record['job']==str(job.relative_to(root))
            and sha256(job/'request.json')==record['request_sha256']
            and sha256(job/'input-receipt.json')==record['input_receipt_sha256'],
            'Runtime revision request/input binding changed')
    require(_rows(job/'sources')==record['sources_before']==read_json(job/'input-receipt.json')['sources'],
            'Original source snapshots changed')
    require(_rows(revision/'sources')==record['sources'] and set(live_sources)=={r['path'] for r in record['sources']},
            'Effective runtime inventory changed')
    for row in record['sources']:
        require(sha256(local_file(live_sources[row['path']],1024**2))==row['sha256'],'Current runtime differs from reviewed revision')
    before={row['path']:row['sha256'] for row in record['sources_before']}
    changes={row['path']:{'before':before.get(row['path']),'after':row['sha256']}
             for row in record['sources'] if before.get(row['path'])!=row['sha256']}
    profile=record.get('profile','planning')
    require(changes==record['changes'] and _change_policy(profile,changes),
            'Unreviewed runtime source change')
    require(artifacts(job/'agents/planner')==record['planner_artifacts']
            and sha256(job/'agents/runner.json')==record['runner_sha256'],'Original planner cache changed or is missing')
    for row in record['original_events']:
        require(sha256(local_file(job/'events'/row['path'],1024**2))==row['sha256'],'Original failure event changed')
    proof=read_json(revision/'api-custody.json')
    require(proof.get('policy')==('author-input-bound-custody-v1' if profile=='author_input_bound'
                                 else 'completed-api-planning-custody-v1'),'Runtime custody profile changed')
    _planner_ledger(root,job,proof)
    if profile=='author_input_bound':
        require(sha256(job/'plan.json')==record['existing_plan_sha256']
                and artifacts(job/'inspection')==record['inspection_artifacts']
                and sha256(job/'inspection/manifest.json')==record['inspection_manifest_sha256'],
                'Existing paid plan or completed inspection changed')
    from avatar_assets.pipeline_prompts import validate_plan
    plan=validate_plan(read_json(job/'agents/planner/answer.json'),read_json(job/'request.json'))
    require(plan==read_json(revision/'plan.json') and sha256(revision/'plan.json')==record['plan_sha256'],
            'Locally revalidated plan changed')
    return {'revision_path':str(revision/'manifest.json'),'sources':record['sources'],'plan':plan,
            'planner_artifacts':record['planner_artifacts'],'request_sha256':record['request_sha256'],'profile':profile}


def cached_planner_answer(job,verified):
    require(artifacts(Path(job)/'agents/planner')==verified['planner_artifacts'],
            'Verified planner cache changed; no new planner request is allowed')
    return read_json(Path(job)/'agents/planner/answer.json')
