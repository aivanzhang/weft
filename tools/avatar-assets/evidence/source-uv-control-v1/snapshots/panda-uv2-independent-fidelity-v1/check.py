"""Independent source-ID, normal, skin and root-motion audit of a retained native export."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from math_oracle import GLB, canonical, skin, tracks, world

START=time.monotonic()
NORMAL_LIMIT=.0002
WEIGHT_LIMIT=1e-6
POSITION_LIMIT=1e-6
ROOT_LIMIT=1e-6
POSE_LIMIT=1e-6


def require(value,message):
    if not value:raise ValueError(message)
    if time.monotonic()-START>25:raise ValueError('Independent audit internal25second limit')


def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def save(path,value):
    encoded=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    with path.open('x') as handle:handle.write(encoded)


def pinned(row):
    path=ROOT/row['path']
    require(path.resolve()==path.absolute() and path.is_file() and not path.is_symlink()
            and path.stat().st_size<=64*1024**2 and digest(path)==row['sha256'],'Pinned input changed')
    return path


def load_npz(path,keys):
    with zipfile.ZipFile(path) as archive:
        rows=archive.infolist()
        require(len(rows)<=20 and len({r.filename for r in rows})==len(rows)
                and sum(r.file_size for r in rows)<=16*1024**2,'NPZ expansion bound')
    with np.load(path,allow_pickle=False,max_header_size=4096) as saved:
        require(set(keys)<=set(saved.files),'Missing correspondence arrays')
        data={key:saved[key].copy() for key in keys}
    require(all(a.dtype.kind in 'iuf' and np.isfinite(a).all() for a in data.values()),'Invalid NPZ array')
    return data


def gltf_vectors(a):
    result=a[:,[0,2,1]].copy();result[:,2]*=-1
    return result


def dense(ids,weights,order,names):
    result=np.zeros((len(ids),len(names)),dtype=np.float64)
    require(ids.min()>=0 and ids.max()<len(order),'Joint index bounds')
    for slot,name in enumerate(order):
        result[:,names.index(name)]=np.where(ids==slot,weights,0).sum(axis=1)
    return result


def derivative_custody(source,target,out,paths):
    expected=deepcopy(source.doc);allowed=set();uses={};removed=[];tangent_ids=set()
    native=json.loads(paths['finish_report'].read_text())
    fringe=json.loads(paths['fringe_report'].read_text())['material']['material']
    targets={row['material']:row for row in native['fur_undercoat']['materials'] if row['normal_texcoord']==2}
    require(len(targets)==3 and native['fur_undercoat']['authored_uv']['index']==2,'Unexpected authored UV2 scope')
    require(native['fur_undercoat']['authored_uv1_sha256'] is None,'Source UV1 mislabeled authored')
    uv2_proofs=[];fringe_removals=0;normal_bytes=0
    for mi,mesh in enumerate(source.doc['meshes']):
        for pi,primitive in enumerate(mesh['primitives']):
            attrs=primitive['attributes']
            require('_AUTHOR_VERTEX_ID' in attrs,'Raw primitive lacks proof IDs')
            del expected['meshes'][mi]['primitives'][pi]['attributes']['_AUTHOR_VERTEX_ID']
            removed.append(f'meshes/{mi}/primitives/{pi}/attributes/_AUTHOR_VERTEX_ID')
            for semantic,accessor in attrs.items():uses.setdefault(accessor,[]).append(semantic)
            uses.setdefault(primitive['indices'],[]).append('indices')
            allowed.update((attrs['JOINTS_0'],attrs['WEIGHTS_0']))
            material=source.doc['materials'][primitive['material']]
            name=material['name'];normal=material.get('normalTexture')
            require(normal is None or normal.get('texCoord',0)!=1,'Live source UV1 normal material survived')
            if name==fringe:
                require(normal is None and 'TANGENT' in attrs and material.get('alphaMode')=='MASK'
                    and abs(material.get('alphaCutoff',.5)-.18)<1e-7 and material.get('doubleSided',False) is False,
                    'Fringe omission is not an unused MASK material tangent')
                del expected['meshes'][mi]['primitives'][pi]['attributes']['TANGENT']
                removed.append(f'meshes/{mi}/primitives/{pi}/attributes/TANGENT');fringe_removals+=1
            if normal is not None and normal.get('texCoord',0)==2:
                require(name in targets,'UV2 primitive lacks exact native bake binding')
                row=targets[name]
                require(row['frame_policy']=='mikk-unshared-triangle-v-up-v2','Unexpected native frame policy')
                texture=source.doc['textures'][normal['index']];image=source.doc['images'][texture['source']]
                view=source.doc['bufferViews'][image['bufferView']];start=view.get('byteOffset',0)
                embedded=hashlib.sha256(source.binary[start:start+view['byteLength']]).hexdigest()
                require(embedded==row['texture_sha256']==digest(paths['texture_'+str(row['source_material_index'])]),'UV2 embedded PNG differs from actual bake')
                faces=target.array(primitive['indices']).ravel().astype(int).reshape(-1,3)
                p=target.array(attrs['POSITION']).astype(float);n=target.array(attrs['NORMAL']).astype(float)
                uv=target.array(attrs['TEXCOORD_2']).astype(float);actual=target.array(attrs['TANGENT'])
                require(len(faces)==row['selected_faces'] and len(p)==3*len(faces)
                    and np.all(np.bincount(faces.ravel(),minlength=len(p))==1),'UV2 topology/count mismatch')
                require(np.isfinite(actual).all() and np.max(abs(np.linalg.norm(n,axis=1)-1))<.001,'Invalid UV2 normal/tangent data')
                n/=np.linalg.norm(n,axis=1)[:,None];uv[:,1]=1-uv[:,1]
                a,b=uv[faces[:,1]]-uv[faces[:,0]],uv[faces[:,2]]-uv[faces[:,0]]
                determinant=a[:,0]*b[:,1]-a[:,1]*b[:,0]
                require(np.all(abs(determinant)>1e-12),'Degenerate UV2 chart')
                du=((p[faces[:,1]]-p[faces[:,0]])*b[:,1,None]-(p[faces[:,2]]-p[faces[:,0]])*a[:,1,None])/determinant[:,None]
                projected=du[:,None,:]-n[faces]*np.sum(n[faces]*du[:,None,:],axis=2)[:,:,None]
                require(np.all(np.linalg.norm(projected,axis=2)>1e-8),'Undefined UV2 frame')
                projected/=np.linalg.norm(projected,axis=2)[:,:,None]
                wanted=np.empty_like(actual);wanted[faces,:3]=projected;wanted[faces,3]=np.sign(determinant)[:,None]
                require(np.array_equal(actual,wanted),'Selected UV2 tangent bytes differ from independent face projection')
                tangent_ids.add(attrs['TANGENT']);allowed.add(attrs['TANGENT'])
                uv2_proofs.append({'mesh':mi,'primitive':pi,'material':name,'faces':len(faces),
                    'tangent_accessor':attrs['TANGENT'],'embedded_png_sha256':embedded,
                    'normal_texcoord':2,'exact_independent_projected_frames':True,
                    'frame_policy':row['frame_policy'],'scope':'Explicit per-face V-up bake convention; native Mikk parity was separately tested by the fixture.'})
            require(np.array_equal(source.array(attrs['NORMAL']),target.array(attrs['NORMAL'])),'A body/fringe normal changed')
            normal_bytes+=source.array(attrs['NORMAL']).nbytes
    for spec in source.doc['skins']:uses.setdefault(spec['inverseBindMatrices'],[]).append('inverse_bind')
    for animation in source.doc.get('animations',[]):
        for sampler in animation['samplers']:
            for field in ('input','output'):uses.setdefault(sampler[field],[]).append('animation')
    require(fringe_removals==1 and len(uv2_proofs)==3 and {r['material'] for r in uv2_proofs}==set(targets),
        'Incomplete fringe omission or UV2 target coverage')
    require(expected==target.doc,'Derivative JSON changed beyond temporary ID and unused fringe T bindings')
    require(len(source.binary)==len(target.binary),'Padded BIN length changed')
    mask=np.zeros(len(source.binary),dtype=np.uint8);allowed_views=set();spans=[]
    for accessor in sorted(allowed):
        expected_semantics={'TANGENT'} if accessor in tangent_ids else {'JOINTS_0','WEIGHTS_0'}
        require(set(uses[accessor])<=expected_semantics,'Allowed accessor aliases other semantics')
        begin,end,stride,dtype,width=source.span(accessor);spec=source.doc['accessors'][accessor]
        require(width==4 and 'min' not in spec and 'max' not in spec,'Skin accessor shape/bounds changed')
        np.ndarray((spec['count'],width*dtype.itemsize),dtype=np.uint8,buffer=mask,offset=begin,strides=(stride,1))[:]=1
        allowed_views.add(spec['bufferView'])
        spans.append({'accessor':accessor,'start':begin,'stride':stride,'element_bytes':width*dtype.itemsize,'count':spec['count']})
    for accessor in range(len(source.doc['accessors'])):
        if accessor in allowed:continue
        begin,end,stride,dtype,width=source.span(accessor);count=source.doc['accessors'][accessor]['count']
        require(not np.ndarray((count,width*dtype.itemsize),dtype=np.uint8,buffer=mask,offset=begin,strides=(stride,1)).any(),
                'Skin rewrite aliases a non-skin or orphan accessor')
    images={image['bufferView'] for image in source.doc.get('images',[])}
    for vi,view in enumerate(source.doc['bufferViews']):
        if vi not in allowed_views or vi in images:
            begin=view.get('byteOffset',0)
            require(not mask[begin:begin+view['byteLength']].any(),'Skin rewrite aliases image/unproven bufferView')
    raw=np.frombuffer(source.binary,dtype=np.uint8);result=np.frombuffer(target.binary,dtype=np.uint8);changed=0
    for start in range(0,len(raw),1024*1024):
        end=min(start+1024*1024,len(raw));difference=raw[start:end]!=result[start:end]
        require(not np.any(difference & (mask[start:end]==0)),'Unselected BIN bytes changed')
        changed+=int(np.count_nonzero(difference))
    changes_by_semantic={'skin_joints_weights':0,'authored_uv2_tangents':0}
    for row in spans:
        a=np.ndarray((row['count'],row['element_bytes']),dtype=np.uint8,buffer=source.binary,
            offset=row['start'],strides=(row['stride'],1))
        b=np.ndarray(a.shape,dtype=np.uint8,buffer=target.binary,offset=row['start'],strides=(row['stride'],1))
        key='authored_uv2_tangents' if row['accessor'] in tangent_ids else 'skin_joints_weights'
        changes_by_semantic[key]+=int(np.count_nonzero(a!=b))
    require(sum(changes_by_semantic.values())==changed,'Overlapping admitted write spans')
    save(out/'derivative-custody.json',{'status':'passed','changed_binary_bytes':changed,'allowed_spans':spans,
        'changed_bytes_by_semantic':changes_by_semantic,'removed_bindings':removed,
        'every_other_json_value_exact':True,'every_other_padded_binary_byte_exact':True,
        'all_normal_bytes_exact':normal_bytes,'uv2_targets':uv2_proofs,
        'unused_original_id_and_fringe_tangent_accessors_bytes_retained':True,'quality_approved':False})


def main(request,out):
    paths={key:pinned(row) for key,row in request['inputs'].items()}
    require(digest(paths['independent_math'])=='a0230b3a5104ae27ea3868b8985b7614e8d2eb932745c7276a22f7a38e2bd372','Independent math changed')
    prepare=json.loads(paths['prepare_receipt'].read_text())
    require(prepare['job']['exit_code']==0 and not prepare['job']['failure']
            and prepare['job']['process_exit_verified'] and prepare['originals_unchanged'],'Preparation incomplete')
    plan=json.loads(paths['scene_plan'].read_text())
    require(plan['status']=='prepared_not_rendered' and plan['config_sha256']==digest(paths['config']),'Prepared config binding changed')
    spec=json.loads(paths['skin_report'].read_text());fr=json.loads(paths['fringe_report'].read_text())
    require(spec['policy']=='source-skin-id-v1' and spec['attribute']=='_AUTHOR_VERTEX_ID'
            and digest(paths['skin_npz'])==spec['npz_sha256'],'Source ID snapshot mismatch')
    source=load_npz(paths['skin_npz'],['positions','joints','weights','triangles','triangle_materials'])
    helper=load_npz(paths['helper'],['positions','normals','joints','weights','triangles','uv_gltf',
                    'root_vertex_ids','native_vertex_to_helper_vertex'])
    for key,value in source.items():
        require(list(value.shape)==spec['arrays'][key]['shape'] and value.dtype.str==spec['arrays'][key]['dtype']
                and hashlib.sha256(value.tobytes()).hexdigest()==spec['arrays'][key]['sha256'],'Source snapshot array changed')
    count=spec['vertex_count'];fins=fr['cards'];names=spec['bone_names'];owners=spec['owners']
    require(0<count<=100000 and 0<fins<=4800 and len(owners)==2 and len(names)<=64,'Actual source size bound')
    fringe_name=fr['material']['material']
    fin_owner=[o for o in owners if o['materials']==[fringe_name]]
    require(len(fin_owner)==1,'Fringe owner identity ambiguous')
    fin_owner=fin_owner[0];body_owner=next(o for o in owners if o!=fin_owner)
    require(fin_owner['vertex_count']==12*fins and len(helper['positions'])==4*fins
            and len(helper['native_vertex_to_helper_vertex'])==12*fins,'Fringe corner map bounds')
    require(np.array_equal(helper['triangles'].ravel(),helper['native_vertex_to_helper_vertex']),'Fringe source corner map changed')
    source_p=gltf_vectors(source['positions'])
    source_dense=dense(source['joints'],source['weights'],names,names)
    helper_dense=dense(helper['joints'],helper['weights'],fr['bone_order'],names)
    actual_p=np.zeros((count,3),dtype=np.float32);actual_dense=np.zeros((count,len(names)))
    actual_j=np.zeros((count,4),dtype=np.int32);actual_w=np.zeros((count,4),dtype=np.float32)
    actual_owner=np.full(count,-1,dtype=np.int32);covered=np.zeros(count,dtype=bool)
    raw=GLB(paths['model']);glb=GLB(paths['derivative']) if 'derivative' in paths else raw
    if glb is not raw:derivative_custody(raw,glb,out,paths)
    records=[];topology=Counter();fin_record=None
    for node_index,node in enumerate(glb.doc['nodes']):
        if 'mesh' not in node:continue
        require('skin' in node,'Unskinned selected owner')
        skin_names=[glb.doc['nodes'][j]['name'] for j in glb.doc['skins'][node['skin']]['joints']]
        require(len(skin_names)==len(set(skin_names)) and set(skin_names)==set(names),'Skin names differ')
        for primitive_index,primitive in enumerate(glb.doc['meshes'][node['mesh']]['primitives']):
            attrs=primitive['attributes'];material=glb.doc['materials'][primitive['material']]['name']
            id_attrs=raw.doc['meshes'][node['mesh']]['primitives'][primitive_index]['attributes']
            ai=id_attrs.get('_AUTHOR_VERTEX_ID');require(type(ai)is int,'Missing exported source IDs')
            accessor=raw.doc['accessors'][ai]
            require(accessor['componentType']==5126 and accessor['type']=='SCALAR'
                    and not accessor.get('normalized',False),'Source IDs lost exact float32 scalar encoding')
            ids=raw.array(ai).ravel();require(np.all(ids==np.floor(ids)) and ids.min()>=0 and ids.max()<count,'Invalid exported IDs')
            ids=ids.astype(np.int32)
            matches=[o for o in owners if ids.min()>=o['vertex_start'] and ids.max()<o['vertex_start']+o['vertex_count']]
            require(len(matches)==1 and material in matches[0]['materials'],'Primitive crosses source owner/material boundaries')
            owner=matches[0];p=glb.array(attrs['POSITION']);j=glb.array(attrs['JOINTS_0']).astype(np.int32);w=glb.array(attrs['WEIGHTS_0'])
            dw=dense(j,w,skin_names,names)
            require(len(ids)==len(p) and np.max(abs(p-source_p[ids]))<=POSITION_LIMIT,'ID-bound rest position changed')
            record={'node':node,'node_index':node_index,'ids':ids,'positions':p,'joints':j,'weights':w,
                    'skin_names':skin_names,'primitive':primitive,'owner':owner['index']}
            records.append(record)
            record_index=len(records)-1
            for local,global_id in enumerate(ids):
                if covered[global_id]:
                    require(np.array_equal(actual_p[global_id],p[local]) and np.array_equal(actual_dense[global_id],dw[local])
                            and records[actual_owner[global_id]]['node_index']==node_index,'One source ID has inconsistent exported position/skin')
                else:
                    actual_p[global_id]=p[local];actual_dense[global_id]=dw[local]
                    actual_j[global_id]=j[local];actual_w[global_id]=w[local]
                    actual_owner[global_id]=record_index;covered[global_id]=True
            indices=glb.array(primitive['indices']).ravel().astype(int)
            require(indices.min()>=0 and indices.max()<len(ids) and len(indices)%3==0,'Invalid actual triangles')
            topology.update((owner['index'],material,canonical(t)) for t in ids[indices].reshape(-1,3))
            if material==fringe_name:
                require(fin_record is None,'Multiple fringe primitives');fin_record=record
    require(covered.all() and fin_record is not None,'Unreferenced/missing authored source IDs')
    wanted=Counter()
    for owner in owners:
        a=owner['triangle_start'];b=a+owner['triangle_count']
        wanted.update((owner['index'],owner['materials'][int(m)],canonical(t))
                      for t,m in zip(source['triangles'][a:b],source['triangle_materials'][a:b]))
    require(topology==wanted,'Source-ID oriented material triangle multiplicity changed')
    fin_ids=fin_record['ids'];local_ids=fin_ids-fin_owner['vertex_start']
    require(np.array_equal(np.sort(local_ids),np.arange(12*fins)),'Fringe ID/corner coverage changed')
    mapped=helper['native_vertex_to_helper_vertex'][local_ids]
    expected_p=gltf_vectors(helper['positions']);expected_n=gltf_vectors(helper['normals'])
    uv=helper['uv_gltf'].copy();uv[:,1]=np.float32(1)-np.float32(np.float32(1)-uv[:,1])
    attrs=fin_record['primitive']['attributes'];actual_n=glb.array(attrs['NORMAL'])
    require(np.array_equal(fin_record['positions'],expected_p[mapped])
            and np.array_equal(glb.array(attrs['TEXCOORD_0']),uv[mapped]),'Fringe helper position/UV differs under source IDs')
    normal_errors=np.max(abs(actual_n-expected_n[mapped]),axis=1)
    weight_errors=np.max(abs(actual_dense-source_dense),axis=1)
    body_range=np.arange(body_owner['vertex_start'],body_owner['vertex_start']+body_owner['vertex_count'])
    fin_range=np.arange(fin_owner['vertex_start'],fin_owner['vertex_start']+fin_owner['vertex_count'])
    helper_root_indices=(np.arange(fins)[:,None]*4+np.array([0,1])).ravel()
    body_root_ids=helper['root_vertex_ids'].ravel()+body_owner['vertex_start']
    require(body_root_ids.min()>=body_range[0] and body_root_ids.max()<=body_range[-1],'Helper root ID outside body owner')
    helper_to_corner=np.full(4*fins,-1,dtype=np.int32)
    for corner,h in enumerate(helper['native_vertex_to_helper_vertex']):
        if helper_to_corner[h]<0:helper_to_corner[h]=corner
    fin_root_ids=helper_to_corner[helper_root_indices]+fin_owner['vertex_start']
    require(np.array_equal(expected_p[helper_root_indices],source_p[body_root_ids]),'Exact helper root source identity changed position')
    helper_source_weight=float(np.max(abs(helper_dense[helper_root_indices]-source_dense[body_root_ids])))
    fin_helper_weight=float(np.max(abs(actual_dense[fin_ids]-helper_dense[mapped])))
    rows=[];worst=[]
    for owner,ids in [('body',body_range),('fringe',fin_range)]:
        bad=ids[weight_errors[ids]>WEIGHT_LIMIT]
        rows.append({'owner':owner,'vertices':len(ids),'over_original_weight_limit':len(bad),
                     'maximum_named_weight_error':float(weight_errors[ids].max())})
        for gid in ids[np.argsort(weight_errors[ids])[-4:]][::-1]:
            if weight_errors[gid]>0:
                worst.append({'source_id':int(gid),'owner':owner,'error':float(weight_errors[gid]),
                  'native_weights':{n:float(v) for n,v in zip(names,source_dense[gid]) if v>0},
                  'exported_weights':{n:float(v) for n,v in zip(names,actual_dense[gid]) if v>0}})
    save(out/'id-and-fidelity.json',{'source_id_coverage':count,'source_topology_exact':True,
        'fringe_exported_corners':len(fin_ids),'normal_limit':NORMAL_LIMIT,'normal_max_component_error':float(normal_errors.max()),
        'normal_over_limit':int(np.count_nonzero(normal_errors>NORMAL_LIMIT)),'weight_limit':WEIGHT_LIMIT,'weights':rows,
        'helper_vs_native_body_root_weight_error':helper_source_weight,'raw_fin_vs_helper_weight_error':fin_helper_weight,
        'worst_weight_rows':worst,'normal_quality_approved':False})
    witnesses=[]
    for index in request.get('witness_helper_roots',[]):
        require(type(index)is int and 0<=index<len(body_root_ids),'Witness root outside measured roots')
        witnesses.append({'helper_root':index,'body_source_id':int(body_root_ids[index]),
            'fin_source_id':int(fin_root_ids[index]),'lookup':'Exact preserved IDs, no position/UV proximity selection',
            'source_named_weights':{n:float(v) for n,v in zip(names,source_dense[body_root_ids[index]]) if v>0}})
    save(out/'formerly-ambiguous-root.json',{'witnesses':witnesses})
    animation=tracks(glb);require(set(animation)==set(request['expected_clips']),'Unexpected actual clips')
    samples=[]
    for name,clip in animation.items():
        times=np.linspace(min(t[0] for _,t,_,_ in clip),max(t[-1] for _,t,_,_ in clip),9)
        raw_gap=helper_raw_gap=source_gap=pose_error=0.;per_time=[]
        for second in times:
            matrices=world(glb,clip,float(second));actual_pose=np.zeros((count,3));source_pose=np.zeros((count,3))
            for index,record in enumerate(records):
                ids=np.flatnonzero(actual_owner==index)
                if len(ids)==0:continue
                node=record['node'];remap=np.array([record['skin_names'].index(n) for n in names])
                actual_pose[ids]=skin(glb,node,actual_p[ids],actual_j[ids],actual_w[ids],matrices)
                source_pose[ids]=skin(glb,node,source_p[ids],remap[source['joints'][ids]],source['weights'][ids],matrices)
            remap=np.array([fin_record['skin_names'].index(n) for n in fr['bone_order']])
            helper_pose=skin(glb,fin_record['node'],expected_p,remap[helper['joints']],helper['weights'],matrices)
            raw=float(np.linalg.norm(actual_pose[fin_root_ids]-actual_pose[body_root_ids],axis=1).max())
            hybrid=float(np.linalg.norm(helper_pose[helper_root_indices]-actual_pose[body_root_ids],axis=1).max())
            native=float(np.linalg.norm(source_pose[fin_root_ids]-source_pose[body_root_ids],axis=1).max())
            error=float(np.linalg.norm(actual_pose-source_pose,axis=1).max())
            raw_gap=max(raw_gap,raw);helper_raw_gap=max(helper_raw_gap,hybrid);source_gap=max(source_gap,native);pose_error=max(pose_error,error)
            per_time.append({'second':float(second),'raw_fin_to_raw_body_m':raw,'helper_fin_to_raw_body_m':hybrid,
                             'native_snapshot_fin_to_body_m':native,'all_vertices_raw_vs_source_skin_m':error})
        samples.append({'clip':name,'samples':per_time,'maximum_raw_root_gap_m':raw_gap,
                        'maximum_helper_to_pruned_body_gap_m':helper_raw_gap,'maximum_native_snapshot_root_gap_m':source_gap,
                        'maximum_raw_vs_source_skin_pose_error_m':pose_error})
    save(out/'motion.json',{'clips':samples,'root_limit_m':ROOT_LIMIT,'source_skin_pose_limit_m':POSE_LIMIT,
        'scope':'Actual exported skeleton CPU skin sampling; no native evaluated-pose array or continuous-time proof.',
        'quality_approved':False})
    passed=bool(np.max(normal_errors)<=NORMAL_LIMIT and np.max(weight_errors)<=WEIGHT_LIMIT
                and helper_source_weight<=WEIGHT_LIMIT and fin_helper_weight<=WEIGHT_LIMIT
                and all(s['maximum_raw_root_gap_m']<=ROOT_LIMIT and s['maximum_helper_to_pruned_body_gap_m']<=ROOT_LIMIT
                        and s['maximum_native_snapshot_root_gap_m']<=ROOT_LIMIT
                        and s['maximum_raw_vs_source_skin_pose_error_m']<=POSE_LIMIT for s in samples))
    for row in request['inputs'].values():pinned(row)
    return {'status':'passed_component_gates' if passed else 'failed_component_gates',
        'passed':passed,'quality_approved':False,'whole_model_normal_fidelity_approved':False,
        'asset_kind':'final_uv2_portable_derivative' if 'derivative' in paths else 'raw_native_export',
        'elapsed_seconds':time.monotonic()-START,'limits':{'normal_component':NORMAL_LIMIT,'weight':WEIGHT_LIMIT,'root_m':ROOT_LIMIT,
        'source_skin_pose_m':POSE_LIMIT},'pose_gate_origin':'New conservative exported-skin versus source-snapshot CPU invariant; not a historical native evaluated-pose comparator.'}


if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_text());out=ROOT/request['output'];out.mkdir()
    save(out/'request.json',request)
    try:
        result=main(request,out);save(out/'result.json',result)
    except Exception as error:
        save(out/'result.json',{'status':'audit_failed','error':str(error),'quality_approved':False,
            'elapsed_seconds':time.monotonic()-START})
        (out/'failure.txt').write_text(traceback.format_exc());raise
    if not result['passed']:raise SystemExit(1)
