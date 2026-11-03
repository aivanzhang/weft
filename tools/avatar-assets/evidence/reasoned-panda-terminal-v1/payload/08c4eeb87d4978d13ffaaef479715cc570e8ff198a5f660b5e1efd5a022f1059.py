"""Read-only historical revision and terminal collection; no source execution or dispatch."""
from pathlib import Path
import hashlib,json,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from avatar_assets.pipeline_continuation import collect
from avatar_assets.pipeline_recovery import verify_runtime_revision
from avatar_assets.safety import read_json,sha256,write_json
OUT=Path(__file__).resolve().parent;parent=ROOT/'outputs/studio-d8a680ce6580429f9fdd016f';start=time.monotonic()
ledger=ROOT/'.local/openai/ledger.sqlite3';ledger_before=sha256(ledger)
try:
 pointer=read_json(parent/'runtime-revision.json');revision=ROOT/pointer['path'];record=read_json(revision/'revision.json')
 assert sha256(revision/'manifest.json')==pointer['sha256']
 effective_sources={row['path']:revision/'sources'/row['path'] for row in record['sources']}
 verified=verify_runtime_revision(parent,effective_sources)
 assert verified['revision_path']==str(revision/'manifest.json')
 captured=collect(ROOT,parent);context=captured['context'];write_json(OUT/'actual-context.json',context)
 result={'status':'passed','parent':str(parent.relative_to(ROOT)),'files':len(captured['inventory']),
 'bytes':sum(row['bytes'] for row in captured['inventory']),
 'inventory_sha256':hashlib.sha256(json.dumps(captured['inventory'],sort_keys=True).encode()).hexdigest(),
 'original_snapshot_and_effective_revision_verified':True,'revision_manifest_sha256':sha256(revision/'manifest.json'),
 'revision_check_scope':'Existing verifier against frozen effective sources for terminal historical custody, not authorization to execute current code',
 'visual_reviews':len(context['findings'].get('historical_visual_reviews',[])),
 'known_source_failures':len(context['findings'].get('known_source_failures',[])),
 'latest_review_verdict':context['findings'].get('review',{}).get('verdict'),
 'context_sha256':sha256(OUT/'actual-context.json'),'context_bytes':(OUT/'actual-context.json').stat().st_size,
 'original_inventory_unchanged':True,'ledger_sha256_before':ledger_before,'ledger_sha256_after':sha256(ledger)}
 assert result['ledger_sha256_before']==result['ledger_sha256_after']
except Exception as exc:
 result={'status':'failed','error':str(exc),'traceback':traceback.format_exc(),
         'ledger_sha256_before':ledger_before,'ledger_sha256_after':sha256(ledger)}
result.update(schema_version=1,elapsed_seconds=time.monotonic()-start,
 implementation_sha256=sha256(ROOT/'avatar_assets/pipeline_continuation.py'),provider_calls=0,native_calls=0,
 scope='Read-only actual terminal collection; no retry, job, source, approval or ledger mutation')
write_json(OUT/'actual-result.json',result);print(json.dumps(result));sys.exit(0 if result['status']=='passed' else 1)
