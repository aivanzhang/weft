"""Read-only analytic attribution of a rejected UV bake; not a relaxed asset gate."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root))
from scripts import blender_fur_undercoat as helper
from avatar_assets.three_d import _GLB
from avatar_assets.safety import write_json,sha256

out=Path(__file__).resolve().parents[1]
failed=root/'outputs/panda-portable-undercoat-v2'
inputs=np.load(failed/'source-body.npz',allow_pickle=False)
selected=np.load(failed/'selected-triangles.npy',allow_pickle=False)
model=_GLB(root/'outputs/muse-panda-dance-flight-v2/model.glb')
assert model.hash=='693f8af27d9d18b5beeb344e6168f9c79471f82b9660780867b8d71956ff2c67'
materials=[];names=[]
for primitive in model.doc['meshes'][0]['primitives']:
    count=model.doc['accessors'][primitive['indices']]['count']//3
    names.append(model.doc['materials'][primitive['material']]['name'])
    materials.extend([len(names)-1]*count)
materials=np.asarray(materials,dtype=np.int16)
pair_counts={};face_hits=np.zeros(len(materials),dtype=np.int32)
conflict_pixels=np.zeros((2048,2048),dtype=bool)
source=inspect.getsource(helper._raster)
original='                overlap += int(np.count_nonzero(np.max(np.abs(old_p-new_p), axis=1) > 1e-5))'
replacement='''                bad = np.max(np.abs(old_p-new_p), axis=1) > 1e-5
                overlap += int(bad.sum())
                np.add.at(face_hits, old_ids[bad], 1)
                face_hits[index] += int(bad.sum())
                for other, count in zip(*np.unique(materials[old_ids[bad]], return_counts=True)):
                    pair = tuple(sorted((int(other), int(materials[index]))))
                    pair_counts[pair] = pair_counts.get(pair, 0) + int(count)
                local_y, local_x = np.where(collision)
                conflict_pixels[y0+local_y[bad], low[0]+local_x[bad]] = True'''
assert original in source
source=source.replace(original,replacement)
condition='    if covered == 0 or overlap > max(8, covered*.01):'
assert condition in source
source=source.replace(condition,'    if False:  # Extraction only; no texture/model is produced or approved.')
scope={'np':np,'time':time,'MAX_SECONDS':20.,'MAX_RASTER_WORK':32000000,
       'materials':materials,'pair_counts':pair_counts,'face_hits':face_hits,'conflict_pixels':conflict_pixels}
exec(compile(source,'analytic_instrumented_raster','exec'),scope)
owner,bary,report=scope['_raster'](inputs['points'],inputs['faces'],inputs['uvs'],selected,2048,time.monotonic())
affected=np.flatnonzero(face_hits)
rows=[]
for index,name in enumerate(names):
    ids=np.flatnonzero((materials==index)&(face_hits>0))
    points=inputs['points'][inputs['faces'][ids]].reshape(-1,3)
    rows.append({'material':name,'affected_faces':len(ids),'selected_faces':int((selected&(materials==index)).sum()),
        'overlap_endpoint_samples':int(face_hits[ids].sum()),
        'bounds':{'min':points.min(axis=0).tolist(),'max':points.max(axis=0).tolist()} if len(points) else None})
result={'schema_version':1,'kind':'analytic_uv_overlap_attribution','status':'diagnostic_only_rejected_bake_remains_rejected',
    'source_glb_sha256':model.hash,'failed_manifest_sha256':sha256(failed/'manifest.json'),
    'selected_mask_sha256':sha256(failed/'selected-triangles.npy'),'raster_helper_sha256':sha256(Path(helper.__file__)),
    'original_gate_percent':1,'raster':report,'unique_conflicting_texels':int(conflict_pixels.sum()),
    'pair_counts':[{'materials':[names[a],names[b]],'overlap_samples':count} for (a,b),count in sorted(pair_counts.items())],
    'materials':rows,'native_calls':0,'provider_calls':0,'original_files_changed':False,
    'limitation':'Uses original GLB coordinates rather than native bind-rounding values; no full appearance or generated texture. Gate is bypassed only to extract counts in this isolated diagnostic, not in production.'}
write_json(out/'report.json',result)
print(json.dumps(result,sort_keys=True))
