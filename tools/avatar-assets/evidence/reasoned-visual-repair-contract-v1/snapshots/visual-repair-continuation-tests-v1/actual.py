import hashlib,json,sys,time,traceback
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from avatar_assets.pipeline_continuation import collect
from avatar_assets.pipeline_prompts import author_prompt
from avatar_assets.safety import read_json,write_json,sha256
OUT=ROOT/'outputs/visual-repair-continuation-tests-v1'
parent=ROOT/'outputs/studio-1f98497e18e74820bda487ce'
started=time.monotonic()
try:
 captured=collect(ROOT,parent);context=captured['context']
 request={**captured['parent_request'],'continuation':{'proof_only':True}}
 inspection=read_json(parent/'inspection/manifest.json')
 prompt=author_prompt(request,context['plan'],{'preflight':inspection['preflight'],'native':inspection['native']},
                      context['previous_answer'],context['findings'])
 size=len(prompt.encode());assert size+1024<=160*1024
 write_json(OUT/'actual-context.json',context)
 result={'status':'passed','scope':'Read-only collected terminal visual repair and reconstructed next prompt; no API/native calls or job creation',
  'parent':str(parent.relative_to(ROOT)),'inventory_files':len(captured['inventory']),
  'inventory_bytes':sum(x['bytes'] for x in captured['inventory']),
  'inventory_sha256':hashlib.sha256(json.dumps(captured['inventory'],sort_keys=True).encode()).hexdigest(),
  'parent_inventory_unchanged':True,'context_bytes':len(json.dumps(context).encode()),'context_sha256':sha256(OUT/'actual-context.json'),
  'author_prompt_bytes':size,'directive_allowance_bytes':1024,'total_bound_bytes':size+1024,'input_limit_bytes':160*1024,
  'author_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'visual_reviews':len(context['findings']['historical_visual_reviews']),
  'known_source_failures':len(context['findings'].get('known_source_failures',[])),
  'paid_seed_role':'author-2','latest_review_verdict':context['findings']['review']['verdict']}
except Exception as exc:
 result={'status':'failed','error':str(exc),'traceback':traceback.format_exc()}
result.update(elapsed_seconds=time.monotonic()-started,implementation_sha256=sha256(ROOT/'avatar_assets/pipeline_continuation.py'))
write_json(OUT/'actual-result.json',result);print(json.dumps(result));sys.exit(0 if result['status']=='passed' else 1)
