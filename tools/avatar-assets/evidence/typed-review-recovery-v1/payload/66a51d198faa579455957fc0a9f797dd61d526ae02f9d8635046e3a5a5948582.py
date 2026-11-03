import ast,hashlib,json,sys,unittest
from pathlib import Path
from avatar_assets.pipeline_prompts import decode_typed_review,typed_critic_prompt,typed_critic_schema,validate_review
from avatar_assets.agent_runner import MAX_PROMPT_BYTES,MAX_SCHEMA_BYTES
root=Path.cwd();out=Path(sys.argv[1]);job=root/"outputs/studio-d8795ec6c273489da37954ab"
suite=unittest.defaultTestLoader.loadTestsFromNames(["test_typed_critic_contract","test_pipeline_prompts.ReviewContractTests"])
result=unittest.TextTestRunner(verbosity=2).run(suite)
answer=job/"agents/reviewer-2/answer.json";raw=answer.read_bytes();assert hashlib.sha256(raw).hexdigest()=="ca1eed63fefd596cc7c8149d28cecc2aa114fbe77d0ab661d6d5c0bd8ba4dc42"
value=json.loads(raw);prompt=(job/"agents/reviewer-2/prompt.txt").read_bytes();payload,_=json.JSONDecoder().raw_decode(prompt.decode().split("INPUT_JSON (data, not instructions):\n",1)[1]);plan=json.loads((job/"plan.json").read_text());ids=[row["id"] for row in payload["available_citations"]]
errors={}
for name,call in [("legacy_validation",lambda:validate_review(value,plan,ids)),("typed_decoder",lambda:decode_typed_review(value)),("typed_decoder_no_array_fallback",lambda:decode_typed_review({**value,"schema_version":2}))]:
 try:call()
 except ValueError as error:errors[name]=str(error)
 else:raise AssertionError("Original invalid review was accepted: "+name)
assert errors["legacy_validation"]=="Duplicate review check"
native_source=job/"sources/pipeline_prompts.py";old_tree=ast.parse(native_source.read_text());new_tree=ast.parse((root/"avatar_assets/pipeline_prompts.py").read_text());names=("critic_prompt","critic_schema","review_citations","validate_review")
def nodes(tree):return {node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names}
assert nodes(old_tree)==nodes(new_tree)
new=typed_critic_prompt(payload["request"],payload["plan"],payload["authored_config"],payload["independent_proofs"],payload["evidence"])
new_data=json.loads(new.split("INPUT_JSON (data, not instructions):\n",1)[1]);assert new_data==payload
schema=typed_critic_schema(payload["evidence"],payload["independent_proofs"].get("action_audit"));schema_size=len(json.dumps(schema,separators=(",",":")).encode());assert schema_size<=MAX_SCHEMA_BYTES and len(new.encode())+1024<=MAX_PROMPT_BYTES
assert answer.read_bytes()==raw and (job/"agents/reviewer-2/prompt.txt").read_bytes()==prompt
observed={"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"passed":result.wasSuccessful(),"actual_invalid_review":{"path":str(answer.relative_to(root)),"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),"check_names":[row["name"] for row in value["checks"]],"rejections":errors,"source_unchanged":True},"legacy_function_asts_unchanged":list(names),"legacy_source_sha256":hashlib.sha256(native_source.read_bytes()).hexdigest(),"actual_prompt":{"old_bytes":len(prompt),"new_bytes":len(new.encode()),"directive_allowance":1024,"full_prompt_limit":MAX_PROMPT_BYTES,"payload_unchanged":True,"schema_bytes":schema_size,"schema_limit":MAX_SCHEMA_BYTES}}
(out/"observations.json").write_text(json.dumps(observed,indent=2,sort_keys=True)+"\n")
sys.exit(0 if result.wasSuccessful() else 1)
