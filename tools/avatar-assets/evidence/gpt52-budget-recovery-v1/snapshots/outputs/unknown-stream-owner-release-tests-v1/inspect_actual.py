from contextlib import closing
from pathlib import Path
import json, sqlite3, time
from avatar_assets.safety import sha256,write_json
from avatar_assets.studio_inspection import collect,custody_locks,include_previous_plans,recheck
from avatar_assets.studio_unknown_release import collect_unknown,recheck_unknown
root=Path.cwd();job='15105a1353404792a185d61d';state=root/'.local/studio';output=root/'outputs'/('studio-'+job)
ledger=root/'.local/openai/ledger.sqlite3';registry=state/'jobs.sqlite3'
before={'ledger':sha256(ledger),'registry':sha256(registry)}
with closing(sqlite3.connect(registry.as_uri()+'?mode=ro',uri=True)) as db:
    db.row_factory=sqlite3.Row
    row=dict(db.execute('SELECT * FROM jobs WHERE id=?',(job,)).fetchone())
    active=db.execute('SELECT job FROM active').fetchone()
assert row['status']=='needs_inspection' and active[0]==job
request=json.loads(row['request']);data=json.loads(row['data']);invocation=state/data['invocation_dir']
locks=[state/'worker.lock',state/'jobs'/job/'worker.lock',output/'pipeline.lock',output/'agents/runner.lock']
started=time.monotonic()
with custody_locks(locks):
    local=include_previous_plans(root,invocation,collect(root,output,invocation,92900,'darwin-start:1793667054:369342'),row['request_digest'])
    local['locks']=[str(path.relative_to(root)) for path in locks]
    unknown=collect_unknown(root,output,request,state/'jobs'/job/'pipeline-request.json')
    local['final_process_check']=recheck(root,local);recheck_unknown(unknown)
after={'ledger':sha256(ledger),'registry':sha256(registry)}
assert before==after
folder=root/'outputs/unknown-stream-owner-release-tests-v1'
write_json(folder/'actual-custody.json',{'schema_version':1,'read_only':True,'no_release':True,'proof':{'local_process_custody':local,'unknown_provider':unknown}})
write_json(folder/'actual-summary.json',{'read_only':True,'no_release':True,'no_api_calls':True,'no_native_calls':True,'seconds':time.monotonic()-started,'before':before,'after':after,'unknown_reserved_microdollars':unknown['uncertain_call']['reserved_microdollars'],'lease_still_held':True,'old_job_restart_allowed':False,'proof_sha256':sha256(folder/'actual-custody.json')})
print(json.dumps({'verified':True,'seconds':time.monotonic()-started,'lease_still_held':True,'unknown_hold':unknown['uncertain_call']['reserved_microdollars']}))
