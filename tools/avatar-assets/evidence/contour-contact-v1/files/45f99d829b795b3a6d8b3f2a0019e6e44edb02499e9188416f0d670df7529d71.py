"""Prepare source-normal-only controls, without modifying either model."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
DIAG=ROOT/'outputs/panda-contact-defects-diagnosis-v1'
CURRENT=ROOT/'outputs/panda-staged-portable-model-v1/model.glb'
SOURCE=ROOT/'outputs/companion-separation-v7/diagnostic-model.glb'


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def glb(path):
    assert path.stat().st_size<64*1024**2
    raw=path.read_bytes();size=struct.unpack_from('<I',raw,12)[0]
    doc=json.loads(raw[20:20+size]);binary=memoryview(raw)[28+size:]
    def array(index):
        a=doc['accessors'][index];v=doc['bufferViews'][a['bufferView']]
        assert 'sparse'not in a and a['count']<=300000
        dtype={5121:'<u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
        width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']];item=np.dtype(dtype).itemsize
        return np.ndarray((a['count'],width),dtype=dtype,buffer=binary,
            offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',width*item),item))
    return doc,array


def blender(values):
    out=np.asarray(values,dtype=np.float64)[:,[0,2,1]].copy();out[:,1]*=-1
    return out


mapping=json.loads((DIAG/'normal-comparison.json').read_text())
selectors=json.loads((DIAG/'contact-selectors.json').read_text())
assert digest(CURRENT)==mapping['model_sha256']==selectors['source_model_sha256']
assert digest(SOURCE)==mapping['source_sha256']
selected={(r['primitive'],r['triangle']):r for r in selectors['regions']['cheek'] if r['kind']!='original'}
assert len(selected)==198 and {key[0]for key in selected}=={6}
current,ca=glb(CURRENT);source,sa=glb(SOURCE)
rows=[];position_errors=[];uv_errors=[];normalization_errors=[]
for key,row in sorted(selected.items()):
    joins=[r for r in mapping['rows']if r['region']=='cheek'and(r['primitive'],r['triangle'])==key]
    assert len(joins)==1
    join=joins[0]
    cp=current['meshes'][0]['primitives'][key[0]];sp=source['meshes'][0]['primitives'][join['source_primitive']]
    ci=ca(cp['indices']).reshape(-1,3)[key[1]];si=sa(sp['indices']).reshape(-1,3)[join['source_triangle']]
    assert ci.tolist()==row['vertex_ids']
    cp3=ca(cp['attributes']['POSITION'])[ci];cu=ca(cp['attributes']['TEXCOORD_0'])[ci]
    sp3=sa(sp['attributes']['POSITION'])[si];su=sa(sp['attributes']['TEXCOORD_0'])[si]
    old=sa(sp['attributes']['NORMAL'])[si]
    matches=[]
    for shift in range(3):
        pe=float(np.max(np.abs(cp3-np.roll(sp3,shift,axis=0))));ue=float(np.max(np.abs(cu-np.roll(su,shift,axis=0))))
        if pe<=2e-7 and ue<=2e-7:matches.append((shift,pe,ue))
    assert len(matches)==1
    shift,pe,ue=matches[0];original=blender(np.roll(old,shift,axis=0))
    lengths=np.linalg.norm(original,axis=1)
    assert np.isfinite(original).all()and np.max(np.abs(lengths-1))<.001
    normalized=original/lengths[:,None]
    delta=float(np.max(np.abs(normalized-original)))
    position_errors.append(pe);uv_errors.append(ue);normalization_errors.append(delta)
    uv_blender=cu.astype(float).copy();uv_blender[:,1]=1-uv_blender[:,1]
    rows.append({'target_mesh':0,'target_primitive':key[0],'target_triangle':key[1],
        'target_vertex_ids':ci.tolist(),'source_mesh':0,'source_primitive':join['source_primitive'],
        'source_triangle':join['source_triangle'],'source_vertex_ids_in_target_corner_order':np.roll(si,shift).tolist(),
        'source_to_target_cyclic_shift':shift,'kind':row['kind'],
        'closure_loop':row.get('closure_loop'),'source_face_id':row.get('source_face_id'),
        'rest_positions_blender_xyz_m':blender(cp3).tolist(),'uv0_blender_v_up':uv_blender.tolist(),
        'original_v7_normal_blender_xyz':normalized.tolist(),
        'normalization_max_component_delta':delta,'match_max_position_error_m':pe,'match_max_uv_error':ue})
inputs=[CURRENT,SOURCE,DIAG/'normal-comparison.json',DIAG/'contact-selectors.json']
payload={'schema_version':1,'status':'prepared_normal_only_ablation_not_repair',
    'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p),'bytes':p.stat().st_size}for p in inputs],
    'current_model_sha256':digest(CURRENT),'source_model_sha256':digest(SOURCE),
    'coordinates':'Blender XYZ Z-up meters; UV0 Blender V-up; normalized original v7 corner normals',
    'tolerances':{'rest_position_component_m':2e-7,'uv_component':2e-7,'source_normal_length_error':.001},
    'faces':rows,'face_count':198,'normalized_corner_count':594,
    'geometry_uv_topology_skin_animation_modified':False,'native_execution':False,
    'claim':'Strict bounded correspondence payload only. Source v7 normals are a diagnostic control, not accepted repaired shading.',
    'application_contract':'Match exact triangle rest positions and UV0 cyclic order; replace only matched corner normals via FLOAT_VECTOR CORNER custom_normal. Preserve all other corner normals, geometry, UVs, materials, skin and actions. Fail on missing/ambiguous faces; no fallback.'}
with(OUT/'payload.json').open('x')as f:json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
report={'status':'prepared','faces':len(rows),'corners':594,'max_position_component_error_m':max(position_errors),
    'max_uv_component_error':max(uv_errors),'max_normalization_component_delta':max(normalization_errors),
    'payload_sha256':digest(OUT/'payload.json'),'sources_unchanged':all(digest(p)==r['sha256']for p,r in zip(inputs,payload['inputs'])),
    'native_jobs':0,'provider_calls':0,'quality_approved':False}
with(OUT/'report.json').open('x')as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(report,sort_keys=True))
