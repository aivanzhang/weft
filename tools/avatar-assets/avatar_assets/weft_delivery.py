"""Explicit low-detail delivery derivative; consumer admission remains independently authoritative."""
from copy import deepcopy
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import struct
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image

from avatar_assets.runner import run_bounded
from avatar_assets.safety import require,read_json,write_json,new_directory,preserve_failure
from avatar_assets.three_d import _GLB,MAX_TRIANGLE_INDICES

ROOT=Path(__file__).resolve().parents[1]
PROFILE='weft-delivery-derivative-v1'
ROLES={'mobile_root','torso',*(side+'_'+joint+'_body' for side in ('left','right') for joint in ('shoulder','elbow','hip','knee'))}
ROOT_NAME='WeftAvatarRoot'


def _check(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Delivery packing deadline exceeded')


def _pin(path):
    with path.open('rb') as f:return sha256(f.read()).hexdigest()


def _file(path,maximum):
    path=Path(path).absolute()
    require('..' not in path.parts and path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)), 'Safe regular input required')
    require(0<path.stat().st_size<=maximum,'Input byte bound exceeded');return path


def validate_settings(value):
    require(type(value) is dict and set(value)=={'asset_id','display_name','rights','provenance','body_roles','scale','orientation_xyzw'},'Closed delivery settings required')
    roles=value['body_roles'];require(type(roles) is dict and set(roles)==ROLES,'Exactly ten explicit semantic body roles required')
    require(all(type(n) is str and 0<len(n)<=128 for n in roles.values()) and len(set(roles.values()))==10 and roles['mobile_root']==ROOT_NAME,'Unique explicit source bone names and authored container required')
    scale=value['scale'];require(type(scale) in (int,float) and .01<=scale<=100 and math.isfinite(scale),'Explicit bounded authored visual scale required')
    q=value['orientation_xyzw'];require(type(q) is list and len(q)==4 and all(type(x) in (int,float) and -1<=x<=1 for x in q) and abs(sum(x*x for x in q)-1)<1e-6,'Explicit unit orientation required')
    require(len(json.dumps(value,allow_nan=False).encode())<=32<<10,'Delivery metadata exceeds32KiB')
    return deepcopy(value)


def _png(image):
    f=io.BytesIO();image.save(f,format='PNG');return f.getvalue()


def _texture(glb,material,cache,deadline):
    pbr=material.get('pbrMetallicRoughness',{});ref=pbr.get('baseColorTexture')
    if ref is None:return np.full((1,1,4),255,dtype=np.uint8),{'wrapS':10497,'wrapT':10497}
    require(ref.get('texCoord',0)==0 and not ref.get('extensions'),'Only original UV0 basecolor is supported')
    texture=glb.doc['textures'][ref['index']];i=texture['source']
    if i not in cache:
        _check(deadline);image=glb.doc['images'][i];view=glb.doc['bufferViews'][image['bufferView']];offset=view.get('byteOffset',0)
        with Image.open(io.BytesIO(glb.binary[offset:offset+view['byteLength']])) as im:
            require(im.width*im.height<=16_777_216 and getattr(im,'n_frames',1)==1,'Basecolor image exceeds16MP still limit')
            rgba=im.convert('RGBA');rgba.thumbnail((2048,2048),Image.Resampling.LANCZOS);cache[i]=np.asarray(rgba).copy()
    sampler=glb.doc.get('samplers',[])[texture['sampler']] if 'sampler' in texture else {}
    require(all(sampler.get(k,10497) in (10497,33071,33648) for k in ('wrapS','wrapT')),'Unsupported texture wrapping')
    return cache[i],sampler


def _sample(image,uv,sampler):
    values=[]
    for axis,key in enumerate(('wrapS','wrapT')):
        x=uv[...,axis];mode=sampler.get(key,10497)
        x=np.clip(x,0,1) if mode==33071 else 1-np.abs((x%2)-1) if mode==33648 else x%1
        values.append(x)
    h,w=image.shape[:2];x=values[0]*w-.5;y=values[1]*h-.5;ix=np.floor(x).astype(int);iy=np.floor(y).astype(int);fx=x-ix;fy=y-iy
    def idx(v,size,key):return np.clip(v,0,size-1) if sampler.get(key,10497) in (33071,33648) else v%size
    a=image[idx(iy,h,'wrapT'),idx(ix,w,'wrapS')]/255.;b=image[idx(iy,h,'wrapT'),idx(ix+1,w,'wrapS')]/255.
    c=image[idx(iy+1,h,'wrapT'),idx(ix,w,'wrapS')]/255.;d=image[idx(iy+1,h,'wrapT'),idx(ix+1,w,'wrapS')]/255.
    # glTF colors multiply in linear light; alpha remains linear.
    def linear(v):return np.concatenate([np.where(v[...,:3]<=.04045,v[...,:3]/12.92,((v[...,:3]+.055)/1.055)**2.4),v[...,3:]],axis=-1)
    return (linear(a)*(1-fx)[...,None]+linear(b)*fx[...,None])*(1-fy)[...,None]+(linear(c)*(1-fx)[...,None]+linear(d)*fx[...,None])*fy[...,None]


def bake_vertex_colors(uv,colors,image,sampler,*,deadline):
    """Retain per-corner tint/alpha through a bounded explicitly resampled triangle atlas."""
    require(isinstance(uv,np.ndarray) and uv.ndim==3 and uv.shape[1:]==(3,2) and 1<=len(uv)<=6500,'Bounded triangle UV array required')
    require(isinstance(colors,np.ndarray) and colors.shape==(len(uv),3,4) and np.isfinite(colors).all() and np.all((colors>=0)&(colors<=1)),'Finite linear vertex colors required')
    cols=math.ceil(math.sqrt(len(uv)));rows=math.ceil(len(uv)/cols);tile=16;atlas=np.zeros((rows*tile,cols*tile,4),dtype=np.uint8);mapped=np.empty_like(uv)
    y,x=np.mgrid[:tile,:tile];b=(x+.5-2)/12;c=(y+.5-2)/12;a=1-b-c;weights=np.stack([a,b,c],axis=-1);weights=np.maximum(weights,0);weights/=weights.sum(axis=-1,keepdims=True)
    for i in range(len(uv)):
        if i%32==0:_check(deadline)
        value=_sample(image,weights@uv[i],sampler)*(weights@colors[i]);rgb=value[...,:3]
        value[...,:3]=np.where(rgb<=.0031308,12.92*rgb,1.055*np.power(np.maximum(rgb,0),1/2.4)-.055)
        cx,cy=(i%cols)*tile,(i//cols)*tile;atlas[cy:cy+tile,cx:cx+tile]=np.rint(np.clip(value,0,1)*255).astype(np.uint8)
        mapped[i]=np.array([[cx+2,cy+2],[cx+14,cy+2],[cx+2,cy+14]])/[cols*tile,rows*tile]
    return mapped.reshape(-1,2),_png(Image.fromarray(atlas)),{'width':cols*tile,'height':rows*tile,'tile_pixels':16,'method':'linear_vertex_color_times_bilinear_basecolor_to_srgb_triangle_atlas'}


def pack_delivery(native_model,settings,output,*,deadline):
    settings=validate_settings(settings);glb=_GLB(native_model);source=glb.doc
    require(len(source.get('skins',[]))==1 and len(source.get('nodes',[]))<=127,'One bounded skin and room for containing root required')
    require(sum(len(m.get('primitives',[])) for m in source['meshes'])<=16,'Primitive limit exceeded')
    doc={k:deepcopy(source[k]) for k in ('asset','nodes','skins','scenes')};doc['scene']=source.get('scene',0)
    names=[n.get('name') for n in doc['nodes']];require(ROOT_NAME not in names,'Containing root name already exists')
    roots=doc['scenes'][doc['scene']]['nodes'];root=len(doc['nodes']);doc['nodes'].append({'name':ROOT_NAME,'children':roots});doc['scenes']=[{'nodes':[root]}];doc['scene']=0
    for role,name in settings['body_roles'].items():
        if role!='mobile_root':require(names.count(name)==1,'Explicit semantic bone is absent or ambiguous: '+name)
    doc.update(bufferViews=[],accessors=[],meshes=[],materials=[],images=[],textures=[],samplers=[]);binary=bytearray();image_cache={};report=[];total=0
    def append(data):
        binary.extend(b'\0'*(-len(binary)%4));offset=len(binary);binary.extend(data);doc['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':len(data)});return len(doc['bufferViews'])-1
    def accessor(value,kind,component):
        value=np.asarray(value,dtype={5126:'<f4',5123:'<u2',5125:'<u4'}[component]);row={'bufferView':append(value.tobytes()),'componentType':component,'count':len(value),'type':kind}
        if kind=='VEC3':row.update(min=value.min(axis=0).tolist(),max=value.max(axis=0).tolist())
        doc['accessors'].append(row);return len(doc['accessors'])-1
    for skin in doc['skins']:
        require(len(skin['joints'])<=64 and 'inverseBindMatrices' in skin,'Original complete bind matrices required')
        skin['inverseBindMatrices']=accessor(glb.accessor(skin['inverseBindMatrices'],'MAT4',{5126}),'MAT4',5126)
    triangles_total=0
    for mesh in source['meshes']:
        packed={'name':mesh.get('name','mesh'),'primitives':[]}
        for primitive in mesh['primitives']:
            _check(deadline);require(primitive.get('mode',4)==4 and not primitive.get('targets'),'Only un-morphed triangles supported')
            attrs=primitive['attributes'];required={'POSITION','NORMAL','TEXCOORD_0','JOINTS_0','WEIGHTS_0'};require(required<=attrs.keys(),'Complete skinned textured vertices required')
            positions=np.asarray(glb.accessor(attrs['POSITION'],'VEC3',{5126}));indices=np.asarray(glb.accessor(primitive['indices'],'SCALAR',{5121,5123,5125},count_limit=MAX_TRIANGLE_INDICES)).reshape(-1)
            require(len(indices)%3==0 and indices.min()>=0 and indices.max()<len(positions),'Complete triangle indices required')
            triangles_total+=len(indices)//3;require(triangles_total<=39000,'Delivery triangle bound exceeded')
            uv=np.asarray(glb.accessor(attrs['TEXCOORD_0'],'VEC2',{5126}))
            material=deepcopy(source['materials'][primitive['material']]);image,sampler=_texture(glb,material,image_cache,deadline)
            normals=np.asarray(glb.accessor(attrs['NORMAL'],'VEC3',{5126}));length=np.linalg.norm(normals,axis=1);require(np.all(length>1e-8),'Nonzero normals required');normals=normals/length[:,None]
            joints=np.asarray(glb.accessor(attrs['JOINTS_0'],'VEC4',{5121,5123}));weights=np.asarray(glb.accessor(attrs['WEIGHTS_0'],'VEC4',{5126,5121,5123}))
            require(np.all(weights>=0) and np.allclose(weights.sum(axis=1),1,atol=1e-4),'Original normalized four-influence skin required')
            colors=None
            if 'COLOR_0' in attrs:
                # The triangle atlas needs unique per-corner UV islands, so this path
                # alone expands corners; every corner counts against the vertex bound.
                colors=np.asarray(glb.accessor(attrs['COLOR_0'],components={5121,5123,5126}))[indices]
                if colors.shape[1]==3:colors=np.column_stack([colors,np.ones(len(colors))])
                uv,image_raw,bake=bake_vertex_colors(uv[indices].reshape(-1,3,2),colors.reshape(-1,3,4),image,sampler,deadline=deadline)
                sampler={'wrapS':33071,'wrapT':33071,'magFilter':9729,'minFilter':9729}
                positions=positions[indices];normals=normals[indices];joints=joints[indices];weights=weights[indices]
                indices=np.arange(len(indices))
            else:image_raw=_png(Image.fromarray(image));bake=None
            total+=len(positions);require(total<=19500,'Delivery vertex bound exceeded')
            image_index=len(doc['images']);doc['images'].append({'bufferView':append(image_raw),'mimeType':'image/png'});doc['samplers'].append({k:sampler[k] for k in ('wrapS','wrapT','magFilter','minFilter') if k in sampler});doc['textures'].append({'source':image_index,'sampler':image_index})
            pbr=material.get('pbrMetallicRoughness',{});simple={'name':material.get('name','material'),'pbrMetallicRoughness':{'baseColorFactor':pbr.get('baseColorFactor',[1,1,1,1]),'baseColorTexture':{'index':image_index},'metallicFactor':0,'roughnessFactor':1},'alphaMode':material.get('alphaMode','OPAQUE'),'alphaCutoff':material.get('alphaCutoff',.5),'doubleSided':material.get('doubleSided',False)}
            require(simple['alphaMode'] in ('OPAQUE','MASK'),'Unsupported blended material')
            doc['materials'].append(simple)
            packed['primitives'].append({'attributes':{'POSITION':accessor(positions,'VEC3',5126),'NORMAL':accessor(normals,'VEC3',5126),'TEXCOORD_0':accessor(uv,'VEC2',5126),'JOINTS_0':accessor(joints,'VEC4',5123),'WEIGHTS_0':accessor(weights,'VEC4',5126)},'indices':accessor(indices,'SCALAR',5123),'material':len(doc['materials'])-1})
            report.append({'mesh':mesh.get('name'),'vertices':len(positions),'triangles':len(indices)//3,'color_bake':bake,'material_prior':'diffuse metallic0 roughness1; normal/occlusion/emissive/metallic-roughness maps omitted','source_material':material})
        doc['meshes'].append(packed)
    doc['buffers']=[{'byteLength':len(binary)}];binary.extend(b'\0'*(-len(binary)%4));encoded=json.dumps(doc,separators=(',',':'),allow_nan=False).encode();encoded+=b' '*(-len(encoded)%4)
    result=struct.pack('<4sII',b'glTF',2,28+len(encoded)+len(binary))+struct.pack('<II',len(encoded),0x4e4f534a)+encoded+struct.pack('<II',len(binary),0x004e4942)+binary
    require(len(result)<=32<<20,'Delivery exceeds original32MiB consumer cap');_check(deadline)
    with Path(output).open('xb') as f:f.write(result)
    roles={k:(root if k=='mobile_root' else names.index(v)) for k,v in settings['body_roles'].items()}
    mapping={k:settings[k] for k in ('asset_id','display_name','rights','provenance')}
    mapping['retargeting']={'profile':'weft-avatar-retarget-v2','root_node':root,'scale':settings['scale'],'orientation_xyzw':settings['orientation_xyzw'],'bindings':[{'node':roles[role],'body':role} for role in sorted(roles)]}
    mapping['provenance']={**mapping['provenance'],'delivery_profile':PROFILE,'native_model_sha256':_pin(Path(native_model)),'simplifications':['Explicit connected-mesh decimation and whole-card fur thinning; surviving/interpolated skin weights retained.','Vertex color/basecolor atlas is bounded resampling, not exact original pixels.','Unsupported PBR channels removed; diffuse material prior.','Original clips are retained in the source, not copied or played in the RL delivery.'],'authored_visual_scale':settings['scale'],'retarget_scope':'Appearance only; no physics/anatomical changes.'}
    return mapping,{'profile':PROFILE,'vertices':total,'triangles':triangles_total,'joints':len(doc['skins'][0]['joints']),'nodes':len(doc['nodes']),'primitives':report,'output_sha256':sha256(result).hexdigest(),'animations_copied':False,'consumer_validation':'pending_independent_Weft_import'}


def prepare_weft_delivery(source,settings_path,out,*,expected_source_sha256,triangles=6000,timeout=120):
    """One local conversion with a new output directory; never changes the source or consumer caps."""
    source=_file(source,32<<20);settings_path=_file(settings_path,32<<10);settings=validate_settings(read_json(settings_path))
    require(type(triangles) is int and 100<=triangles<=39000,'Explicit triangle budget must be100..39000')
    require(type(timeout) in (int,float) and 1<=timeout<=120,'Conversion wall bound must be1..120seconds')
    require(_pin(source)==expected_source_sha256,'Original Studio model hash differs')
    glb=_GLB(source);require(len(glb.doc.get('skins',[]))==1,'One source semantic skin required')
    original_joints=sorted(glb.doc['nodes'][i]['name'] for i in glb.doc['skins'][0]['joints']);original_meshes=sorted(n['name'] for n in glb.doc['nodes'] if 'mesh' in n);del glb
    out=Path(out).absolute();require(not out.is_relative_to(source.parent),'Derivative output must be outside immutable original folder')
    out=new_directory(out);script=ROOT/'scripts/blender_weft_delivery.py';script_raw=script.read_bytes();mesh_helper=ROOT/'scripts/weft_delivery_mesh.py';mesh_raw=mesh_helper.read_bytes();settings_raw=settings_path.read_bytes();started=time.monotonic()
    try:
        (out/'blender_weft_delivery.py').write_bytes(script_raw);(out/'weft_delivery_mesh.py').write_bytes(mesh_raw);(out/'settings.json').write_bytes(settings_raw)
        command=['/usr/bin/env','-i','PATH=/usr/bin:/bin','TMPDIR='+str(out),'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
                 'BLENDER_USER_CONFIG='+str(out/'blender-config'),'BLENDER_USER_SCRIPTS='+str(out/'blender-scripts'),'BLENDER_USER_EXTENSIONS='+str(out/'blender-extensions'),
                 str(ROOT/'.tools/Blender.app/Contents/MacOS/Blender'),'--background','--factory-startup','--disable-autoexec','--python-exit-code','1','--threads','2','--python',str(out/'blender_weft_delivery.py'),'--',str(source),str(out),str(triangles)]
        job=run_bounded(command,out/'convert.log',out,timeout=timeout,memory_mb=1536,output_mb=128);write_json(out/'job.json',job)
        require(job['exit_code']==0 and not job['failure'],'Native conversion failed; preserved original and attempt')
        native_report=read_json(out/'native-report.json')
        require(sorted(r['name'] for r in native_report['meshes'])==original_meshes,'Original source mesh inventory changed')
        native=_GLB(out/'decimated.glb');names=sorted(native.doc['nodes'][i]['name'] for i in native.doc['skins'][0]['joints']);require(names==original_joints,'Semantic skin joint inventory changed');del native
        mapping,report=pack_delivery(out/'decimated.glb',settings,out/'model.glb',deadline=time.monotonic()+20)
        require(_pin(source)==expected_source_sha256 and settings_path.read_bytes()==settings_raw and script.read_bytes()==script_raw and mesh_helper.read_bytes()==mesh_raw,'Source/settings/helper changed')
        report.update(source_sha256=expected_source_sha256,settings_sha256=sha256(settings_raw).hexdigest(),native_report=native_report,seconds=time.monotonic()-started,api_calls=0,source_modified=False)
        mapping['provenance']['original_studio_glb_sha256']=expected_source_sha256
        write_json(out/'mapping.json',mapping)
        with ZipFile(out/'weft-avatar.zip','x',compression=ZIP_DEFLATED) as archive:
            for name in ('model.glb','mapping.json'):archive.write(out/name,arcname=name)
        report['zip_sha256']=_pin(out/'weft-avatar.zip')
        write_json(out/'delivery.json',report)
        files={p.name:_pin(p) for p in out.iterdir() if p.is_file()}
        write_json(out/'manifest.json',{'profile':PROFILE,'status':'converted_pending_consumer_import','files':files,'scope':'Derived render asset only; original source and consumer limits unchanged.'})
        return report
    except Exception as error:preserve_failure(out,error);raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('settings');parser.add_argument('out');parser.add_argument('--source-sha256',required=True);parser.add_argument('--triangles',type=int,default=17000)
    args=parser.parse_args();print(json.dumps(prepare_weft_delivery(args.source,args.settings,args.out,expected_source_sha256=args.source_sha256,triangles=args.triangles)))
