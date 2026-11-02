"""Append-only diagnostic GLB patch: original skin, motions and portable mesh bytes stay exact."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

from avatar_assets.provider_model import preflight
from avatar_assets.safety import sha256,write_json
from avatar_assets.three_d import _GLB


DTYPES={5120:'i1',5121:'u1',5122:'<i2',5123:'<u2',5125:'<u4',5126:'<f4'}
WIDTHS={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}


def raw_accessor(glb,index):
    row=glb.doc['accessors'][index];view=glb.doc['bufferViews'][row['bufferView']]
    dtype=np.dtype(DTYPES[row['componentType']]);width=WIDTHS[row['type']]
    return np.ndarray((row['count'],width),dtype=dtype,buffer=glb.binary,
        offset=view.get('byteOffset',0)+row.get('byteOffset',0),
        strides=(view.get('byteStride',width*dtype.itemsize),dtype.itemsize))


def prepare(out):
    out=Path(out);cfg=json.loads((out/'diagnostic-config.json').read_text());glb=_GLB(cfg['source_model'])
    if glb.hash!=cfg['source_model_sha256']: raise ValueError('Source model changed')
    mesh=next(row for row in glb.doc['meshes'] if row.get('name')==cfg['body_mesh_name'])
    points,faces,uvs,offset=[],[],[],0
    for primitive in mesh['primitives']:
        xyz=raw_accessor(glb,primitive['attributes']['POSITION']).copy()
        xyz=xyz[:,[0,2,1]];xyz[:,1]*=-1
        indices=raw_accessor(glb,primitive['indices']).reshape(-1,3).astype(np.int32)
        uv=raw_accessor(glb,primitive['attributes']['TEXCOORD_0']).copy();uv[:,1]=1-uv[:,1]
        points.append(xyz);faces.append(indices+offset);uvs.append(uv[indices]);offset+=len(xyz)
    np.savez(out/'source-body.npz',points=np.concatenate(points),faces=np.concatenate(faces),uvs=np.concatenate(uvs))
    return {'triangles':sum(len(row) for row in faces),'vertices':offset,'sha256':sha256(out/'source-body.npz')}


def patch(out):
    out=Path(out);cfg=json.loads((out/'diagnostic-config.json').read_text());glb=_GLB(cfg['source_model'])
    if glb.hash!=cfg['source_model_sha256']: raise ValueError('Source model changed')
    baked=json.loads((out/'native-bake-report.json').read_text())
    mask_path=out/'selected-triangles.npy'
    if sha256(mask_path)!=baked['selected_triangle_mask_sha256']: raise ValueError('Measured selection changed')
    selected=np.load(mask_path,allow_pickle=False)
    if selected.dtype!=np.bool_ or selected.ndim!=1 or not 1<=len(selected)<=100000:
        raise ValueError('Invalid bounded face selection')
    texture=out/baked['texture_path']
    if sha256(texture)!=baked['texture_sha256'] or texture.stat().st_size>32*1024*1024:
        raise ValueError('Baked normal changed or exceeded bounds')
    doc=deepcopy(glb.doc);binary=bytearray(glb.binary)
    def view(data,target=None):
        binary.extend(b'\0'*(-len(binary)%4));offset=len(binary);binary.extend(data)
        row={'buffer':0,'byteOffset':offset,'byteLength':len(data)}
        if target is not None: row['target']=target
        doc['bufferViews'].append(row);return len(doc['bufferViews'])-1
    def accessor(original,values,target):
        row=deepcopy(original);row['bufferView']=view(values.tobytes(),target);row['byteOffset']=0;row['count']=len(values)
        for key,fn in [('min',np.min),('max',np.max)]:
            if key in row: row[key]=fn(values,axis=0).tolist()
        doc['accessors'].append(row);return len(doc['accessors'])-1
    doc['images'].append({'name':'AuthoredShortPileNormal','bufferView':view(texture.read_bytes()),'mimeType':'image/png'})
    doc.setdefault('samplers',[]).append({'magFilter':9729,'minFilter':9987,'wrapS':10497,'wrapT':10497})
    doc['textures'].append({'source':len(doc['images'])-1,'sampler':len(doc['samplers'])-1})
    normal_index=len(doc['textures'])-1
    mesh=next(row for row in doc['meshes'] if row.get('name')==cfg['body_mesh_name'])
    originals=deepcopy(mesh['primitives']);primitives=[];offset=0;new_materials={}
    for original in originals:
        indices=raw_accessor(glb,original['indices']).reshape(-1,3)
        mask=selected[offset:offset+len(indices)];offset+=len(indices)
        for coated in (False,True):
            subset=indices[mask==coated]
            if not len(subset): continue
            primitive=deepcopy(original)
            if len(subset)!=len(indices):
                used,inverse=np.unique(subset,return_inverse=True)
                primitive['attributes']={name:accessor(glb.doc['accessors'][index],
                    raw_accessor(glb,index)[used],34962) for name,index in original['attributes'].items()}
                dtype='<u2' if len(used)<=65535 else '<u4'
                component=5123 if len(used)<=65535 else 5125
                primitive['indices']=accessor({'componentType':component,'type':'SCALAR'},
                                             inverse.astype(dtype).reshape(-1,1),34963)
            if coated:
                source=original['material']
                if source not in new_materials:
                    material=deepcopy(doc['materials'][source]);material['name']+='__short_pile_v1'
                    pbr=material.setdefault('pbrMetallicRoughness',{})
                    pbr['metallicFactor']=0;pbr['roughnessFactor']=cfg['settings']['roughness']
                    pbr.pop('metallicRoughnessTexture',None)
                    material['normalTexture']={'index':normal_index,'texCoord':0,'scale':1}
                    new_materials[source]=len(doc['materials']);doc['materials'].append(material)
                primitive['material']=new_materials[source]
            primitives.append(primitive)
    if offset!=len(selected): raise ValueError('Selection length differs from original body triangles')
    mesh['primitives']=primitives
    binary.extend(b'\0'*(-len(binary)%4));doc['buffers'][0]['byteLength']=len(binary)
    encoded=json.dumps(doc,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    encoded+=b' '*(-len(encoded)%4)
    output=out/'model.glb'
    with output.open('xb') as stream:
        stream.write(struct.pack('<4sII',b'glTF',2,12+8+len(encoded)+8+len(binary)))
        stream.write(struct.pack('<II',len(encoded),0x4e4f534a));stream.write(encoded)
        stream.write(struct.pack('<II',len(binary),0x004e4942));stream.write(binary)
    facts=preflight(output);modified=_GLB(output)
    if modified.binary[:len(glb.binary)]!=glb.binary: raise ValueError('Original binary prefix changed')
    for key in ('nodes','skins','animations'):
        if modified.doc.get(key)!=glb.doc.get(key): raise ValueError('Original '+key+' changed')
    for old,new in zip(glb.doc['meshes'],modified.doc['meshes']):
        if old.get('name')!=cfg['body_mesh_name'] and old!=new: raise ValueError('Original portable mesh changed')
    checks={}
    for label,keys in [('skin_animation',['nodes','skins','animations']),('portable_meshes',[]),('basecolor_images',[])]:
        value=({key:glb.doc.get(key) for key in keys} if keys else
               [row for row in glb.doc['meshes'] if row.get('name')!=cfg['body_mesh_name']] if label=='portable_meshes' else
               glb.doc['images'])
        checks[label+'_json_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    report={'schema_version':1,'status':'patched_awaiting_visual_review','source_sha256':glb.hash,'model_sha256':modified.hash,
        'original_binary_prefix_sha256':hashlib.sha256(glb.binary).hexdigest(),'original_binary_prefix_bytes':len(glb.binary),
        'original_binary_prefix_unchanged':True,'source_basecolor_images_unchanged':True,'source_uv_accessor_rows_unchanged':True,
        'source_skin_and_animation_arrays_unchanged':True,'source_portable_primitives_unchanged':True,
        'source_geometry_rows_unchanged':True,'selected_triangles':int(selected.sum()),'excluded_triangles':int((~selected).sum()),
        'body_material_boundary_may_duplicate_vertices':True,'preflight':facts,'native_bake':baked,**checks,
        'scope':'Only copied base-body face ownership, cloned normal/roughness materials and an added authored normal PNG differ. No geometry, color repaint, UV, rig, portable-fiber or motion edit.',
        'new_provider_calls':0,'new_provider_cost_usd':0,'visual_approval':False}
    write_json(out/'diagnostic-report.json',report)
    return report


if __name__=='__main__':
    import sys
    print(json.dumps(prepare(sys.argv[2]) if sys.argv[1]=='prepare' else patch(sys.argv[2]),sort_keys=True))
