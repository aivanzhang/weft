from contextlib import closing
import hashlib,importlib.util,json,sqlite3,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
from avatar_assets.safety import sha256,write_json
from avatar_assets.openai_runner import accounting_snapshot,budget_snapshot,call_exposure
from avatar_assets.agent_runner import _call_artifacts
JOBID='b082dc54dbb54f9d9030f75b';JOB=ROOT/'outputs'/('studio-'+JOBID)
LEDGER=ROOT/'.local/openai/ledger.sqlite3';REGISTRY=ROOT/'.local/studio/jobs.sqlite3'
ledger_before=sha256(LEDGER);registry_before=sha256(REGISTRY)
def inventory():
 started=time.monotonic();rows=[]
 for p in sorted(JOB.rglob('*')):
  assert not p.is_symlink()
  if p.is_file():
   assert p.stat().st_size<=128*1024**2 and len(rows)<20000 and time.monotonic()-started<10
   rows.append({'path':str(p.relative_to(JOB)),'bytes':p.stat().st_size,'sha256':sha256(p)})
 return rows
before=inventory();write_json(OUT/'output-hashes.json',before)
with closing(sqlite3.connect(REGISTRY.as_uri()+'?mode=ro',uri=True,timeout=.2)) as db:
 db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
 row=db.execute('SELECT * FROM jobs WHERE id=?',(JOBID,)).fetchone();data=json.loads(row['data'])
 active=db.execute('SELECT job,token FROM active WHERE id=1').fetchone()
 assert row['status']=='needs_inspection' and data['operation']=='execute'
 assert data['owner_pid']==48443 and data['owner_identity']=='darwin-start:1793656432:745879'
 assert active['job']==JOBID and active['token']==data['token']
 invocation=ROOT/'.local/studio'/data['invocation_dir']
 request_digest=row['request_digest']
 registry_summary={'job_id':JOBID,'status':row['status'],'operation':data['operation'],'owner_pid':data['owner_pid'],
  'owner_identity':data['owner_identity'],'invocation':str(invocation.relative_to(ROOT)),
  'studio_request_digest':request_digest,'active_lease_matches':True,'mutated':False}
with closing(sqlite3.connect(LEDGER.as_uri()+'?mode=ro',uri=True,timeout=.2)) as db:
 db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
 budget=budget_snapshot(db);account=accounting_snapshot(db)
 grant=db.execute('SELECT digest,calls,ceiling FROM jobs WHERE id=?',(JOB.name,)).fetchone()
 calls=[dict(r) for r in db.execute('SELECT * FROM calls WHERE job=?',(JOB.name,))]
 assert grant['digest']==sha256(JOB/'request.json') and len(calls)==1 and calls[0]['role']=='planner'
 assert all(r['state']=='settled' for r in calls)
 global_unknown=sum(v['uncertain_reserved_microdollars'] for v in account['jobs'].values())
 assert global_unknown==0
 receipt=json.loads((JOB/'agents/planner/receipt.json').read_text())
 assert receipt['status']=='succeeded' and receipt['artifacts']==_call_artifacts(JOB/'agents/planner')
 assert receipt['response_id']==calls[0]['response_id'] and receipt['actual_microdollars_upper_estimate']==call_exposure(calls[0])
 safe_calls=[{'role':r['role'],'state':r['state'],'reserved_microdollars':r['reserved'],
  'actual_microdollars_upper_estimate':call_exposure(r),'request_digest':r['request_digest'],
  'response_id_sha256':hashlib.sha256(r['response_id'].encode()).hexdigest(),
  'usage':json.loads(r['usage'])} for r in calls]
 ledger_summary={'budget':budget,'project_exposure_microdollars':account['project_exposure_microdollars'],
  'global_uncertain_reserved_microdollars':global_unknown,'job_id':JOB.name,'job_digest':grant['digest'],
  'max_calls':grant['calls'],'grant_microdollars':grant['ceiling'],'calls':safe_calls,
  'planner_receipt_sha256':sha256(JOB/'agents/planner/receipt.json'),'mutated':False}
write_json(OUT/'ledger-summary.json',ledger_summary);write_json(OUT/'registry-summary.json',registry_summary)
roles=sorted(p.name for p in (JOB/'agents').iterdir() if p.is_dir())
assert roles==['planner'] and not any(p.name.startswith(('model-','finish-','author-')) for p in JOB.iterdir())
status=json.loads((JOB/'status.json').read_text());assert status['error']=='Agent input exceeds 160KiB'
write_json(OUT/'predispatch.json',{'saved_error':status['error'],'only_agent_folders':roles,'author_calls':0,
 'author_model_folders_absent':True,'native_inspection_completed':json.loads((JOB/'inspection/manifest.json').read_text())['status'],
 'no_generation_directory':not (JOB/'generation').exists(),'status_sha256':sha256(JOB/'status.json')})
spec=importlib.util.spec_from_file_location('saved_custody',OUT/'sources/studio_inspection.py')
custody=importlib.util.module_from_spec(spec);spec.loader.exec_module(custody)
proof=custody.collect(ROOT,JOB,invocation,48443,'darwin-start:1793656432:745879',operation='execute')
proof=custody.include_previous_plans(ROOT,invocation,proof,request_digest)
proof['process_check']=custody.recheck(ROOT,proof)
write_json(OUT/'process-custody.json',proof)
assert before==inventory()
assert ledger_before==sha256(LEDGER) and registry_before==sha256(REGISTRY)
summary={'status':'verified_read_only','api_calls':[r['role'] for r in calls],'global_unknown_charges':False,
 'known_owned_pids':proof['known_pids'],'all_recorded_owners_children_groups_absent':True,
 'ledger_sha256':ledger_before,'registry_sha256':registry_before,'output_files':len(before),
 'original_outputs_and_databases_unchanged':True,'release_performed':False,
 'next_action':'Parent may review and invoke the narrow public inspection release; this proof does not release or authorize dispatch.'}
write_json(OUT/'summary.json',summary)
print(json.dumps(summary))
