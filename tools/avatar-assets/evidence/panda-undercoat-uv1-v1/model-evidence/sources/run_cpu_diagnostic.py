"""Bounded procedural normal bake and append-only diagnostic patch; no generation service."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

here=Path(__file__).resolve().parent
root=here.parents[2]
sys.path.insert(0,str(here));sys.path.insert(1,str(root))
from blender_fur_undercoat import bake_short_pile
from groom_flow import prepare_flow
from patch_uv1 import patch,raw_accessor
from avatar_assets.safety import sha256,write_json
from avatar_assets.three_d import _GLB


def face_signatures(glb,mesh_index):
    result=Counter()
    for primitive in glb.doc['meshes'][mesh_index]['primitives']:
        arrays=[raw_accessor(glb,primitive['attributes'][name]) for name in sorted(primitive['attributes'])
                if name not in ('TANGENT','TEXCOORD_1')]
        name=glb.doc['materials'][primitive['material']]['name'].removesuffix('__short_pile_v1')
        for triangle in raw_accessor(glb,primitive['indices']).reshape(-1,3):
            digest=hashlib.sha256(name.encode())
            for array in arrays:digest.update(array[triangle].tobytes())
            result[digest.hexdigest()]+=1
    return result


def run(out):
    out=Path(out);cfg=json.loads((out/'diagnostic-config.json').read_text())
    for name,key in [('source-body.npz','source_body_sha256'),('authored-normal-atlases.npz','authored_atlases_sha256'),('uv-plan.json','uv_plan_sha256')]:
        if (out/name).stat().st_size>16*1024*1024 or sha256(out/name)!=cfg[key]:raise ValueError('Prepared array/plan changed')
    if sha256(Path(cfg['design']))!=cfg['design_sha256'] or sha256(Path(cfg['source_model']))!=cfg['source_model_sha256']:
        raise ValueError('Original model/design changed')
    previous=cfg['prior_mask_evidence'];prior=Path(previous['path'])
    if sha256(prior/'manifest.json')!=previous['manifest_sha256'] or sha256(prior/'selected-triangles.npy')!=previous['selected_mask_sha256']:
        raise ValueError('Measured prior mask evidence changed')
    data=np.load(out/'source-body.npz',allow_pickle=False);atlas=np.load(out/'authored-normal-atlases.npz',allow_pickle=False)
    plan=json.loads((out/'uv-plan.json').read_text());design=json.loads(Path(cfg['design']).read_text())
    points,faces,uv0=data['points'],data['faces'],data['uvs']
    selected,regions=data['selected'],data['regions']
    np.save(out/'selected-triangles.npy',selected,allow_pickle=False)
    prepared=prepare_flow(points.min(axis=0),np.ptp(points,axis=0),design['groom']['direction'],design['groom']['comb'],design['groom_flow'])
    normals=data['normals'][faces];source_tangents=data['tangents'][faces]
    (out/'textures').mkdir();rows=[]
    for row in plan['regions']:
        region=row['region_id'];chosen=selected&(regions==region);ids=np.flatnonzero(chosen)
        if len(ids)!=row['selected_faces'] or hashlib.sha256(ids.astype('<i4').tobytes()).hexdigest()!=row['selection_sha256']:
            raise ValueError('Planned measured region changed')
        uvs=uv0.copy();tangents=source_tangents.copy();settings=dict(cfg['settings'])
        if row['method']=='authored_uv1':
            if not np.array_equal(ids,atlas[f'faces_{region}']):raise ValueError('Authored atlas face mapping changed')
            uvs[ids]=atlas[f'uv1_{region}'];tangents[ids]=atlas[f'tangents_{region}']
            settings['resolution']=row['atlas']['resolution']
        pixels,report=bake_short_pile(points,faces,uvs,normals,tangents[:,:,:3],tangents[:,:,3],chosen,prepared,settings)
        size=settings['resolution'];encoded=np.empty((size,size,3),dtype=np.uint8)
        for channel in range(3):encoded[:,:,channel]=np.clip(np.rint(pixels[::-1,:,channel]*255),0,255).astype(np.uint8)
        path=out/'textures'/f'normal-region-{region:02d}.png'
        Image.fromarray(encoded).save(path)
        del pixels,encoded,uvs,tangents
        rows.append({'region_id':region,'method':row['method'],'texture_path':str(path.relative_to(out)),
            'texture_sha256':sha256(path),'bake':report,'encoding':'RGB8 PNG, linear normal components; bottom-up math rows flipped to PNG top-down',
            'normal_texcoord':1 if row['method']=='authored_uv1' else 0})
    write_json(out/'normal-bake-report.json',{'schema_version':1,'kind':'actual_procedural_short_pile_normals',
        'status':'baked_not_visually_accepted','regions':rows,'selected_triangle_mask_sha256':sha256(out/'selected-triangles.npy'),
        'source_glb_sha256':cfg['source_model_sha256'],'source_color_texture_bytes_changed':False,
        'new_model_calls':0,'new_provider_calls':0,'not_recovered_hair':True})
    result=patch(out)
    original=_GLB(cfg['source_model']);actual=_GLB(out/'model.glb')
    if face_signatures(original,cfg['body_mesh_index'])!=face_signatures(actual,cfg['body_mesh_index']):
        raise ValueError('Original position/normal/UV0/color/skin triangle rows changed')
    uv1_count=0
    for primitive in actual.doc['meshes'][cfg['body_mesh_index']]['primitives']:
        material=actual.doc['materials'][primitive['material']]
        if material.get('normalTexture',{}).get('texCoord')==1:
            uv=raw_accessor(actual,primitive['attributes']['TEXCOORD_1'])
            tangent=raw_accessor(actual,primitive['attributes']['TANGENT']);normal=raw_accessor(actual,primitive['attributes']['NORMAL'])
            if (np.any(uv<0) or np.any(uv>1) or np.max(np.abs(np.linalg.norm(tangent[:,:3],axis=1)-1))>1e-6 or
                    np.max(np.abs(np.sum(normal*tangent[:,:3],axis=1)))>2e-6 or not np.all(np.abs(tangent[:,3])==1)):
                raise ValueError('Authored UV1 tangent frames failed the actual GLB audit')
            uv1_count+=len(tangent)
    proof={'schema_version':1,'kind':'actual_append_only_uv1_model_audit','status':'passed_format_and_preservation_not_quality',
        'source_model_sha256':original.hash,'model_sha256':actual.hash,'original_binary_prefix_unchanged':actual.binary[:len(original.binary)]==original.binary,
        'position_normal_uv0_color_skin_triangle_multiset_exact':True,'body_triangles_verified':len(selected),
        'authored_uv1_tangent_vertices':uv1_count,'uv1_tangent_unit_orthogonal_handedness_checks':True,
        'skin_animation_and_portable_primitives_unchanged':True,'hd_only':True,'weft_appearance_compatible':False,
        'appearance_accepted':False,'new_provider_calls':0}
    write_json(out/'actual-model-audit.json',proof)
    if sha256(Path(cfg['source_model']))!=cfg['source_model_sha256']:raise ValueError('Original source model changed')
    print(json.dumps({'status':'baked_patched_audited','preflight':result['preflight'],'model_sha256':actual.hash,'uv1_vertices':uv1_count},sort_keys=True))


if __name__=='__main__':run(sys.argv[1])
