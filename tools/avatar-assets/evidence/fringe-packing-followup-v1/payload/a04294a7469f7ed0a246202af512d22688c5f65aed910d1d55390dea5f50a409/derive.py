"""Repack unchanged fin RGBA cells; preserve geometry, skin and animation."""
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import struct
import sys
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from avatar_assets.normalize_generated import _array
from avatar_assets.provider_model import preflight
from avatar_assets.safety import require,sha256,write_json
from avatar_assets.three_d import _GLB,validate_render_attributes


def sample_rgba(pixels, uv):
    size=np.array([pixels.shape[1],pixels.shape[0]])
    xy=uv*size-.5;low=np.floor(xy).astype(int);fraction=xy-low
    high=np.clip(low+1,0,size-1);low=np.clip(low,0,size-1)
    u,v=fraction.T;result=np.zeros((len(uv),4))
    for x,y,factor in ((low[:,0],low[:,1],(1-u)*(1-v)),(high[:,0],low[:,1],u*(1-v)),
                       (low[:,0],high[:,1],(1-u)*v),(high[:,0],high[:,1],u*v)):
        rgba=pixels[y,x].astype(float)/255;rgb=rgba[:,:3]
        rgba[:,:3]=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
        result+=rgba*factor[:,None]
    return result


def main():
    cfg=json.loads((HERE/'prepared.json').read_text())
    for row in cfg['inputs']:require(sha256(ROOT/row['path'])==row['sha256'],'Input changed')
    out=ROOT/cfg['output'];require(out.is_dir() and not any(out.iterdir()),'Require fresh monitored output')
    source=_GLB(ROOT/cfg['source']);original=source.doc
    fin=original['meshes'][0]['primitives'][7]
    uv=_array(source,fin['attributes']['TEXCOORD_0'],2,5126,100000).reshape(4800,4,2).astype(float)
    material=original['materials'][fin['material']]
    texture=original['textures'][material['pbrMetallicRoughness']['baseColorTexture']['index']]
    require(material['doubleSided'] is False and material['alphaMode']=='MASK','Require combined source policy')
    image=original['images'][texture['source']];view=original['bufferViews'][image['bufferView']]
    raw=source.binary[view.get('byteOffset',0):view.get('byteOffset',0)+view['byteLength']]
    require(hashlib.sha256(raw).hexdigest()==sha256(ROOT/cfg['atlas']),'Source atlas differs from embedded image')
    del raw
    with Image.open(ROOT/cfg['atlas']) as image:
        require(image.size==(4096,2400) and image.mode=='RGBA','Unexpected bounded atlas')
        atlas=np.asarray(image).copy()
    with np.load(ROOT/cfg['permutation'],allow_pickle=False) as data:
        order=data['source_fin_at_destination'];destinations=data['destination_for_source_fin']
        expected_uv=data['simulated_uv']
    require(order.shape==destinations.shape==(4800,) and order.dtype.kind in 'iu'
            and destinations.dtype.kind in 'iu' and np.array_equal(np.sort(order),np.arange(4800))
            and np.array_equal(destinations[order],np.arange(4800)),'Invalid permutation')
    packed=np.empty_like(atlas)
    for target,old in enumerate(order):
        sy,sx=old//64*32,old%64*64;ty,tx=target//64*32,target%64*64
        packed[ty:ty+32,tx:tx+64]=atlas[sy:sy+32,sx:sx+64]
    for old,target in enumerate(destinations):
        require(np.array_equal(atlas[old//64*32:(old//64+1)*32,old%64*64:(old%64+1)*64],
            packed[target//64*32:(target//64+1)*32,target%64*64:(target%64+1)*64]),'RGBA cell changed')
    expected=json.loads((ROOT/cfg['screen_result']).read_text())['packing_analytic_control']['atlas_array_sha256']
    require(hashlib.sha256(packed.tobytes()).hexdigest()==expected,'Packed pixel hash differs from analytic screen')
    slots=np.arange(4800);old_origins=np.column_stack((slots%64*64,slots//64*32))
    new_origins=np.column_stack((destinations%64*64,destinations//64*32))
    mapped=(uv*np.array([4096,2400])-old_origins[:,None]+new_origins[:,None])/np.array([4096,2400])
    require(np.array_equal(mapped,expected_uv),'Recomputed UV differs from frozen permutation')
    new_uv=mapped.reshape(-1,2).astype('<f4')
    with np.load(ROOT/cfg['tip_reference'],allow_pickle=False) as data:
        truth=data['source_rgb'].reshape(9600,3);fractions=data['tip_fraction']
        old_rgb=data['new_rgb'].reshape(9600,3)
    exported_uv=new_uv.reshape(4800,4,2).astype(float)
    query=exported_uv[:,None,0]+fractions[:,:,None]*(exported_uv[:,None,1]-exported_uv[:,None,0])
    query[:,:,1]=((exported_uv[:,0,1]+exported_uv[:,2,1])*.5)[:,None]
    sampled=sample_rgba(packed,query.reshape(-1,2));errors=np.max(np.abs(sampled[:,:3]-truth),axis=1)
    gate={'tip_count':len(errors),'p95':float(np.quantile(errors,.95)),'p99':float(np.quantile(errors,.99)),
        'maximum_linear_component_error':float(errors.max()),'tips_over_010':int((errors>.1).sum()),
        'maximum_change_from_original_atlas':float(np.max(np.abs(sampled[:,:3]-old_rgb))),
        'minimum_alpha':float(sampled[:,3].min()),'tips_below_mask_cutoff':int((sampled[:,3]<.18).sum())}
    require(gate['p95']<=.02 and gate['p99']<=.05 and gate['tips_over_010']<=48
            and gate['tips_below_mask_cutoff']==0,'Packed atlas failed unchanged nominal-tip gate')
    write_json(out/'lod0-color-audit.json',{'status':'passed','float32_export_uvs':True,'measurements':gate,
        'limits':['Finite nominal-tip samples only; no GPU/mip or whole-surface acceptance.']})
    Image.fromarray(packed).save(out/'packed-color-atlas.png')
    doc=deepcopy(original);binary=bytearray(source.binary)
    def append(raw,target=None):
        binary.extend(b'\0'*(-len(binary)%4));record={'buffer':0,'byteOffset':len(binary),'byteLength':len(raw)}
        if target is not None:record['target']=target
        binary.extend(raw);doc['bufferViews'].append(record);return len(doc['bufferViews'])-1
    doc['accessors'].append({'bufferView':append(new_uv.tobytes(),34962),'componentType':5126,'count':len(new_uv),'type':'VEC2'})
    doc['images'].append({'bufferView':append((out/'packed-color-atlas.png').read_bytes()),'mimeType':'image/png','name':'ColorCoherentFinCells'})
    new_texture=deepcopy(texture);new_texture['source']=len(doc['images'])-1;doc['textures'].append(new_texture)
    new_material=deepcopy(material);new_material['name']='ExperimentalPackedSourceSurfaceFins'
    new_material['pbrMetallicRoughness']['baseColorTexture']['index']=len(doc['textures'])-1;doc['materials'].append(new_material)
    target=doc['meshes'][0]['primitives'][7];target['attributes']['TEXCOORD_0']=len(doc['accessors'])-1;target['material']=len(doc['materials'])-1
    doc['buffers'][0]['byteLength']=len(binary)
    doc['asset']['generator']='Experimental color-coherent atlas packing; RGBA cells, geometry and motion unchanged'
    restored=deepcopy(doc);restored['meshes'][0]['primitives'][7]=deepcopy(fin)
    for key in ('accessors','bufferViews','images','textures','materials'):restored[key]=restored[key][:len(original[key])]
    restored['buffers']=deepcopy(original['buffers']);restored['asset']=deepcopy(original['asset'])
    require(restored==original and binary[:len(source.binary)]==source.binary,'Unexpected mutation beyond atlas/UV allowlist')
    encoded=json.dumps(doc,separators=(',',':'),allow_nan=False).encode();encoded+=b' '*(-len(encoded)%4);binary.extend(b'\0'*(-len(binary)%4))
    total=28+len(encoded)+len(binary);require(total<=64*1024*1024,'File exceeds fixed bound')
    with (out/'model.glb').open('xb') as stream:
        stream.write(struct.pack('<4sII',b'glTF',2,total));stream.write(struct.pack('<II',len(encoded),0x4e4f534a));stream.write(encoded)
        stream.write(struct.pack('<II',len(binary),0x004e4942));stream.write(binary)
    report={'schema_version':1,'status':'packing_verified_pending_actual_review','quality_approved':False,
        'source_sha256':source.hash,'candidate_sha256':sha256(out/'model.glb'),'file_bytes':total,'all_4800_rgba_cells_exact':True,
        'geometry_normals_indices_skin_actions_exact':True,'json_allowlist_passed':True,'source_binary_prefix_exact':True,
        'packed_rgba_array_sha256':expected,'uv_float32_max_rounding':float(np.max(np.abs(new_uv.astype(float)-mapped.reshape(-1,2)))),
        'limits':['Neighbor filtering and alpha at small scales change; this is not pixel-identical rendering.','Analytic modeled improvements do not prove actual GPU causality or final quality.','Original body/contact defects and deep-mip limitations remain.'],
        'new_provider_calls':0}
    del binary,atlas,packed,source
    gc.collect();write_json(out/'preflight.json',preflight(out/'model.glb'))
    write_json(out/'render-attributes.json',validate_render_attributes(out/'model.glb'));write_json(out/'build-report.json',report)
    for row in cfg['inputs']:require(sha256(ROOT/row['path'])==row['sha256'],'Input changed during build')
    print(json.dumps({k:report[k]for k in ['candidate_sha256','file_bytes','uv_float32_max_rounding']}))


if __name__=='__main__':main()
