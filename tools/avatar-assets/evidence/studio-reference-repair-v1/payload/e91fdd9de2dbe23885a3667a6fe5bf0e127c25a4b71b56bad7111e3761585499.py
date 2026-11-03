import hashlib,json,sys,unittest
from pathlib import Path
from avatar_assets.pipeline_prompts import author_prompt
from avatar_assets.agent_runner import MAX_PROMPT_BYTES
out=Path(sys.argv[1]);root=Path.cwd()
suite=unittest.defaultTestLoader.loadTestsFromNames(json.loads(sys.argv[2]))
result=unittest.TextTestRunner(verbosity=2).run(suite)
summary={"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"passed":result.wasSuccessful(),"actual_prompt_checks":[]}
for attempt in range(3):
 p=root/f"outputs/studio-447a849b435b4d2e962b3763/agents/author-{attempt}/prompt.txt"
 raw=p.read_bytes();old=json.loads(raw.decode().split("INPUT_JSON (data, not instructions):\n",1)[1])
 prompt=author_prompt(old["request"],old["plan"],old["inspection"],old["previous_design"],old["repair_findings"])
 new=json.loads(prompt.split("INPUT_JSON (data, not instructions):\n",1)[1])
 preserved=all(new[k]==old[k] for k in ("request","plan","inspection","previous_design"))
 assert preserved and len(prompt.encode())+1024<=MAX_PROMPT_BYTES
 summary["actual_prompt_checks"].append({"attempt":attempt,"source_prompt":str(p.relative_to(root)),"source_prompt_sha256":hashlib.sha256(raw).hexdigest(),"old_bytes":len(raw),"new_bytes":len(prompt.encode()),"directive_allowance_bytes":1024,"full_limit_bytes":MAX_PROMPT_BYTES,"source_inputs_and_previous_design_unchanged":preserved,"envelope":new["resource_envelope"]})
(out/"observations.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
sys.exit(0 if result.wasSuccessful() else 1)
