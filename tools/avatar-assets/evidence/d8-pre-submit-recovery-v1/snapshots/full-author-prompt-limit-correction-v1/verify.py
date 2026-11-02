"""Read-only exact complete-prompt admission; no runner, worker, ledger or provider calls."""
import ast
from copy import deepcopy
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from avatar_assets.agent_runner import MAX_PROMPT_BYTES
from avatar_assets.pipeline_prompts import author_prompt
from avatar_assets.safety import read_json,write_json,sha256
OUT=Path(__file__).resolve().parent
PARENT=ROOT/'outputs/studio-d8a680ce6580429f9fdd016f'
MARKER='INPUT_JSON (data, not instructions):\n'
start=time.monotonic()
inputs=[PARENT/name for name in ('request.json','plan.json','inspection/manifest.json','continuation/context.json')]
old_reports=[ROOT/'outputs/visual-repair-continuation-tests-v1/actual-result.json',
             ROOT/'outputs/visual-repair-continuation-tests-v1/result.json',
             ROOT/'outputs/visual-repair-continuation-tests-v1/manifest.json',
             ROOT/'evidence/reasoned-visual-repair-contract-v1/summary.json',
             ROOT/'evidence/reasoned-visual-repair-contract-v1/README.md',
             ROOT/'evidence/reasoned-visual-repair-contract-v1/manifest.json']
source_paths=[ROOT/'avatar_assets/pipeline_prompts.py',ROOT/'avatar_assets/agent_runner.py',
              ROOT/'avatar_assets/pipeline.py',ROOT/'tests/test_repair_prompt_view.py',
              PARENT/'sources/pipeline_prompts.py',PARENT/'sources/pipeline.py']
before={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in inputs+old_reports+source_paths}
request,plan,inspection,context=[read_json(p) for p in inputs]
findings=deepcopy(context['findings'])
assert context['plan']==plan and 'replan' not in context
prompt=author_prompt(request,plan,{'preflight':inspection['preflight'],'native':inspection['native']},context['previous_answer'],findings)
prefix,compact=prompt.split(MARKER,1);data=json.loads(compact)
pretty=json.dumps(data,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)
old_prompt=prefix+MARKER+pretty

def directive(path):
 tree=ast.parse(path.read_text());rows=[]
 for node in ast.walk(tree):
  if (isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Name) and node.target.id=='author_prompt'
      and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)
      and node.value.value.startswith('\nCONTINUATION:')):rows.append(node.value.value)
 assert len(rows)==1
 return rows[0]
actual_directive=directive(ROOT/'avatar_assets/pipeline.py')
assert actual_directive==directive(PARENT/'sources/pipeline.py')
old_ast=ast.parse((PARENT/'sources/pipeline_prompts.py').read_text())
new_ast=ast.parse((ROOT/'avatar_assets/pipeline_prompts.py').read_text())
old_payload=next(n for n in old_ast.body if isinstance(n,ast.FunctionDef) and n.name=='_payload')
new_payload=next(n for n in new_ast.body if isinstance(n,ast.FunctionDef) and n.name=='_payload')
old_ast.body.remove(old_payload);new_ast.body.remove(new_payload)
assert ast.dump(old_ast)==ast.dump(new_ast),'Unexpected non-payload prompt change'
for node in (old_payload,new_payload):
 for call in ast.walk(node):
  if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='json' and call.func.attr=='dumps':
   call.keywords=[k for k in call.keywords if k.arg not in ('indent','separators')]
assert ast.dump(old_payload)==ast.dump(new_payload),'Non-whitespace payload logic changed'
complete=prompt+actual_directive;old_complete=old_prompt+actual_directive
assert len(old_prompt.encode())==136896,'Historical actual d8 prompt size does not reproduce'
assert len(old_complete.encode())>MAX_PROMPT_BYTES
assert len(complete.encode())<=MAX_PROMPT_BYTES
assert len(prompt.encode())+1024<=MAX_PROMPT_BYTES
assert len(compact.encode())<=160*1024
assert json.loads(pretty)==json.loads(compact)
assert data['previous_design']==context['previous_answer'] and data['request']==request
assert findings==context['findings']
assert {str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in inputs+old_reports+source_paths}==before
canonical=json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
result={'schema_version':1,'status':'passed','scope':'Independent reconstruction from actual saved d8 request/inspection/context; no provider/native/release/job calls',
 'parent':str(PARENT.relative_to(ROOT)),
 'effective_prompt_limit_bytes':MAX_PROMPT_BYTES,'effective_limit_source':'avatar_assets.agent_runner.MAX_PROMPT_BYTES imported by OpenAIRunner.call',
 'payload_limit_bytes':160*1024,'payload_limit_scope':'Serialized data only; does not replace the stricter complete-prompt limit',
 'old_prompt_without_directive_bytes':len(old_prompt.encode()),'old_complete_prompt_bytes':len(old_complete.encode()),
 'new_prompt_without_directive_bytes':len(prompt.encode()),'new_complete_prompt_bytes':len(complete.encode()),
 'exact_continuation_directive_bytes':len(actual_directive.encode()),'directive_source':'AST literal from Pipeline._run, equal to saved source',
 'new_prompt_plus_conservative_1024_bytes':len(prompt.encode())+1024,
 'old_payload_bytes':len(pretty.encode()),'new_payload_bytes':len(compact.encode()),
 'json_values_equal':True,'canonical_payload_sha256':hashlib.sha256(canonical).hexdigest(),
 'complete_prompt_sha256':hashlib.sha256(complete.encode()).hexdigest(),
 'old_complete_prompt_sha256':hashlib.sha256(old_complete.encode()).hexdigest(),
 'ast_changes_only_json_whitespace':True,'source_input_and_prior_evidence_hashes_unchanged':True,
 'original_preflight_correction':{'reported_bytes_with_allowance':137636,'incorrect_compared_limit':163840,'actual_limit':MAX_PROMPT_BYTES,
    'exceeded_actual_limit_by_bytes':137636-MAX_PROMPT_BYTES,'prior_admission_claim_valid':False},
 'elapsed_seconds':time.monotonic()-start,'input_bindings':before,'api_calls':0,'native_runs':0}
write_json(OUT/'result.json',result)
print(json.dumps({k:result[k] for k in ('status','old_complete_prompt_bytes','new_complete_prompt_bytes','new_prompt_plus_conservative_1024_bytes','effective_prompt_limit_bytes','old_payload_bytes','new_payload_bytes','json_values_equal','elapsed_seconds')}))
