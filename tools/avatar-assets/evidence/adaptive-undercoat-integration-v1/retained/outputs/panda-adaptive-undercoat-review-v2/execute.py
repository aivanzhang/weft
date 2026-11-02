from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from avatar_assets.safety import sha256, require, write_json
from avatar_assets.provider_model import preview
p=json.loads((HERE/'prepared.json').read_text())
require(not (HERE/'execution-started.json').exists(),'Preview already started')
for row in p['source_pins']+p['original_pack']:
    require(sha256(ROOT/row['path'])==row['sha256'],'Prepared input changed: '+row['path'])
pack=ROOT/'outputs/panda-adaptive-undercoat-finish-v2'
report=json.loads((pack/'finish-report.json').read_text())
states=[{'id':s['id'],'seconds':s['seconds']} for s in report['states']]
write_json(HERE/'execution-started.json',{'prepared_sha256':sha256(HERE/'prepared.json'),'native_attempts':1,'new_paid_calls':0})
try:
    result=preview(pack/'model.glb', ROOT/'outputs/panda-adaptive-undercoat-export-preview-v2',resolution=720,samples=16,motion_states=states)
    for row in p['original_pack']:
        require(sha256(ROOT/row['path'])==row['sha256'],'Original output changed: '+row['path'])
    write_json(HERE/'execution-result.json',{'result':result,'all_original_files_unchanged':True,'original_files':len(p['original_pack']),'quality_approved':False,'full_media_rendered':False,'new_paid_calls':0})
    print(json.dumps(result))
except Exception as error:
    write_json(HERE/'execution-result.json',{'status':'failed_preserved','error':type(error).__name__+': '+str(error),'automatic_retry':False,'quality_approved':False})
    raise
