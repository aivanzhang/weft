import json,sys,platform
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(OUT/'sources'))
from avatar_assets.safety import sha256,write_json
from avatar_assets.action_audit import write_audit
import numpy as np
p=json.loads((OUT/'prepared.json').read_text())
assert all(sha256(OUT/r['snapshot'])==r['sha256'] for r in p['sources'])
assert all(sha256(ROOT/r['path'])==r['sha256'] for r in p['inputs'].values())
assert sha256(OUT/'intent.json')==p['intent_sha256']
r=write_audit(ROOT/p['inputs']['model']['path'],OUT/'intent.json',OUT/'report.json')
assert all(sha256(ROOT/v['path'])==v['sha256'] for v in p['inputs'].values())
state=r['states'][0];endpoint=[v for v in state['requirement_results']['requirements'] if v['kind']=='return_pose']
summary={'status':'measured','analytic_projection_only':True,'numeric_passed':r['passed'],'source_unchanged':True,'endpoint_requirements':endpoint,'body_evidence':state['evidence']['roles']['body'],'endpoint_scope':state['role_details']['body']['endpoint_measurement'],'sampled_frames':len(state['sample_seconds']),'full_skin_sample_seconds':state['full_skin_sample_seconds'],'audit_elapsed_seconds':r['elapsed_seconds'],'python':platform.python_version(),'numpy':np.__version__,'limitations':['No new authored or generated assets.','No visual, continuous-motion, loop or physical-contact acceptance.']}
write_json(OUT/'summary.json',summary)
print(json.dumps(summary,sort_keys=True))
