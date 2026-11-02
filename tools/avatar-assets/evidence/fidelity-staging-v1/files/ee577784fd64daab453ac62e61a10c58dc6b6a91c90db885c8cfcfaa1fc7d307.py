"""Apply established tangent policies to the exact experimental skin-restored model."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from avatar_assets.fringe_export import omit_unused_fringe_tangents
from avatar_assets.generated_tangents import repair
from avatar_assets.finish_generated import _undercoat_frame_policy
from avatar_assets.assemble_generated import rig_contract
from avatar_assets.three_d import _GLB,validate_glb,validate_render_attributes
from avatar_assets.safety import require,sha256,write_json,new_directory
cfg=json.loads((HERE/'prepared.json').read_text())
for row in cfg['inputs']+cfg['sources']: require(sha256(ROOT/row['path'])==row['sha256'],'Frozen dependency changed')
out=new_directory(ROOT/cfg['output']);source=ROOT/cfg['source'];native=ROOT/cfg['native']
fringe=json.loads((native/'fringe-report.json').read_text());report=json.loads((native/'prepared-finish-report.json').read_text())
require(report['portable_fringe']==fringe and report['portable_fur']==[fringe],'Native fringe report differs')
omitted=omit_unused_fringe_tangents(source,out/'model-unused-tangents-omitted.glb',[fringe['material']['material']])
write_json(out/'unused-tangent-omission.json',omitted)
repaired=repair(out/'model-unused-tangents-omitted.glb',out/'model.glb',correct_normal_uv1=True,
                normal_uv1_frame_policy=_undercoat_frame_policy(report,True))
write_json(out/'tangent-repair.json',repaired)
contract=rig_contract(_GLB(out/'model.glb').doc,authored=True);write_json(out/'rig-contract.json',contract)
write_json(out/'rig-verification.json',validate_glb(out/'model.glb',contract))
write_json(out/'render-attributes.json',validate_render_attributes(out/'model.glb'))
require(all(sha256(ROOT/r['path'])==r['sha256']for r in cfg['inputs']+cfg['sources']),'Inputs changed')
write_json(out/'result.json',{'status':'attributes_valid_not_quality_accepted','model_sha256':sha256(out/'model.glb'),
 'source_sha256':sha256(source),'source_unchanged':True,'quality_approved':False,'new_generation_calls':0,
 'limitation':'Experimental source-skin restore plus existing tangent policies; native reload, appearance and independent audit pending.'})
print(json.dumps({'output':str(out.relative_to(ROOT)),'model_sha256':sha256(out/'model.glb')}))
