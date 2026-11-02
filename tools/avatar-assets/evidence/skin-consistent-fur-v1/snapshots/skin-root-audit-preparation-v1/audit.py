"""Prepared all-root proof. Requires an explicit released, hash-bound request."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback
import zipfile

import numpy as np

PREPARATION = Path(__file__).resolve().parent
ROOT = PREPARATION.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.three_d import _GLB, _animation_tracks, _pose, _world_matrices

START = time.monotonic()
RESULT = {'schema_version':1,'status':'running','quality_approved':False,'native_provider_AI_calls':0}
OUT = None


def require(value, message):
    if not value:
        raise ValueError(message)


def check(value, message):
    if not value:
        RESULT.setdefault('failures',[]).append(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    text=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    with (OUT/name).open('x') as stream:
        stream.write(text)


def array(glb, index, kind, types, indices=False):
    return np.asarray(glb.accessor(index,kind,types,count_limit=300000 if indices else 100000))


def context(glb, node, names):
    skin=glb.doc['skins'][glb.doc['nodes'][node]['skin']]
    actual=[glb.doc['nodes'][i]['name'] for i in skin['joints']]
    require(len(actual)==len(set(actual))==len(names) and set(actual)==set(names),'Named skin changed')
    inverse=array(glb,skin['inverseBindMatrices'],'MAT4',(5126,)).reshape(-1,4,4).transpose(0,2,1)
    order=[actual.index(n) for n in names]
    return {'joints':[skin['joints'][i] for i in order],'inverse':inverse[order],'source_names':actual}


def data(glb, primitive, ctx, names):
    a=primitive['attributes'];j=array(glb,a['JOINTS_0'],'VEC4',(5121,5123)).astype(np.int32)
    w=array(glb,a['WEIGHTS_0'],'VEC4',(5121,5123,5126));dense=np.zeros((len(w),len(names)))
    for i,name in enumerate(ctx['source_names']):
        dense[:,names.index(name)]=np.where(j==i,w,0).sum(axis=1)
    return {'points':array(glb,a['POSITION'],'VEC3',(5126,)),'weights':dense,
            'faces':array(glb,primitive['indices'],'SCALAR',(5121,5123,5125),True).reshape(-1,3).astype(np.int32),
            'uv':array(glb,a['TEXCOORD_0'],'VEC2',(5126,))}


def matrices(world, ctx):
    return np.array([np.asarray(world[i]).reshape(4,4).T for i in ctx['joints']])@ctx['inverse']


def weighted(points, weights, transforms):
    p=np.column_stack((points,np.ones(len(points))))
    return np.einsum('nb,bij,nj->ni',weights,transforms[:,:3,:],p)


def decode_archive(pack, descriptor, roots, pins):
    require(descriptor.get('schema_version')==1 and descriptor.get('roots')==roots
            and descriptor.get('space')=='blender_local_z_up_meters'
            and descriptor.get('root_inset_convention')=='surface_root_minus_bottom_edge_midpoint',
            'Binding archive schema differs')
    rel=Path(descriptor['path'])
    require(not rel.is_absolute() and len(rel.parts)==2 and rel.parts[0]=='portable-root-bindings'
            and rel.suffix=='.npz' and '..' not in rel.parts,'Binding archive path escapes its directory')
    path=pack/rel
    require(not path.is_symlink() and path.is_file() and path.stat().st_size==descriptor['bytes']<=512*1024,
            'Binding archive size/type differs')
    digest=sha(path)
    require(digest==descriptor['sha256']==pins.get(str(path.relative_to(ROOT))),'Binding archive missing exact request/report pin')
    with zipfile.ZipFile(path) as z:
        infos=z.infolist()
        require(len(infos)==3 and {x.filename for x in infos}=={'triangle_index.npy','barycentric.npy','root_inset.npy'}
                and sum(x.file_size for x in infos)<=roots*52+4096,'Binding archive decoded layout exceeds its bound')
    with np.load(path,allow_pickle=False) as z:
        triangle=z['triangle_index'].copy();bary=z['barycentric'].copy();inset=z['root_inset'].copy()
    require(triangle.dtype==np.dtype('<i4') and triangle.shape==(roots,)
            and bary.dtype==np.dtype('<f8') and bary.shape==(roots,3)
            and inset.dtype==np.dtype('<f8') and inset.shape==(roots,3), 'Binding archive dtype/shape differs')
    require(np.isfinite(bary).all() and np.isfinite(inset).all() and (bary>=0).all()
            and np.max(abs(bary.sum(axis=1)-1))<=1e-12,'Invalid source barycentrics/insets')
    return triangle,bary,inset,path.stat().st_size


def run(request_path):
    global OUT
    request=json.loads(request_path.read_text());plan=json.loads((PREPARATION/'plan.json').read_text())
    require(request.get('status')=='released' and request.get('schema_version')==1,'Candidate execution is not released')
    require(request['gates']==plan['gates'] and request['limits']==plan['limits'],'Prepared gates/bounds changed')
    require(request['sample_seconds']==[i/2 for i in range(9)],'Motion samples changed')
    for row in plan['existing_evidence']:
        require(sha(ROOT/row['path'])==row['sha256'],'Prior source correspondence evidence changed')
    pins={r['path']:r['sha256'] for r in request['inputs']}
    require(len(pins)==len(request['inputs']),'Duplicate input pins')
    for path,digest in pins.items():
        require(sha(ROOT/path)==digest,'Pinned input changed '+path)
    required=[request['source'],request['model'],request['body_fidelity_receipt'],
              request['pack']+'/finish-report.json',request['pack']+'/finish-config.json',
              request['pack']+'/material-groom/report.json','avatar_assets/three_d.py']
    require(set(required)<=set(pins),'Required inputs lack custody pins')
    target=ROOT/request['output']
    require(target.resolve().parent==ROOT/'outputs' and target.is_dir() and not target.is_symlink(),
            'Audit output must be a fresh direct outputs child')
    require(not any((target/n).exists() for n in ['result.json','roots.json','motion.json','sampled-motion.npz']),
            'Audit result already exists')
    OUT=target
    body=json.loads((ROOT/request['body_fidelity_receipt']).read_text())
    require(body.get('status')=='passed' and body.get('source_sha256')==pins[request['source']]
            and body.get('final_sha256')==pins[request['model']], 'Exact new-model body fidelity proof is absent')
    source,final=_GLB(ROOT/request['source']),_GLB(ROOT/request['model']);pack=ROOT/request['pack']
    finish=json.loads((pack/'finish-report.json').read_text());cfg=json.loads((pack/'finish-config.json').read_text())
    groom=json.loads((pack/'material-groom/report.json').read_text())
    old_identity=json.loads((ROOT/'outputs/fox-material-fur-independent-audit-v2/native-source-identity.json').read_text())
    require(pins[request['source']]==plan['source_sha256'] and old_identity['equal'], 'Known fox source identity changed')
    require(all(groom['body_before'][k]==old_identity['native_report'][k]
                for k in ['positions','faces','loops','material_slots'])
            and groom['body_before']['uvs'][0]['sha256']==old_identity['native_report']['uvs'],
            'Native source topology/UV IDs changed')
    require(groom['body_before']==groom['body_after'],'Native material groom changed the source body')
    parts=finish['portable_fur'];require(1<=len(parts)<=128,'Portable part count bound')
    total=sum(p['cards'] for p in parts)
    require(total==finish['material_coat']['created_cards']==sum(b['count'] for b in cfg['portable_fur']['bands'])
            and 1<=total<=5000,'Global root count changed/exceeded bound')
    sn,fn=source.doc['nodes'],final.doc['nodes'];owners=[i for i,n in enumerate(sn) if 'mesh' in n]
    require(len(owners)==1,'Source body is ambiguous')
    names=[sn[i]['name'] for i in source.doc['skins'][sn[owners[0]]['skin']]['joints']]
    sctx=context(source,owners[0],names);source_rows={};vo=fo=0
    for slot,p in enumerate(source.doc['meshes'][sn[owners[0]]['mesh']]['primitives']):
        name=source.doc['materials'][p['material']]['name'];nv=source.doc['accessors'][p['attributes']['POSITION']]['count']
        nf=source.doc['accessors'][p['indices']]['count']//3
        if name in {x['source_material'] for x in parts}:
            source_rows[name]={**data(source,p,sctx,names),'vertex_offset':vo,'face_offset':fo,'slot':slot}
        vo+=nv;fo+=nf
    records=[];metadata=[];domain_reports=[];archives=set();archive_bytes=0
    max_weight=max_rest=max_detail=0.;contexts={}
    for part_id,part in enumerate(parts):
        summary=part['root_summary'];require(summary.get('root_placement')=='skin-consistent-v1','Wrong placement policy')
        original=source_rows[part['source_material']];band=part['source_band']
        require(cfg['portable_fur']['bands'][band]['brightness']==[0,1.01],
                'This source-ID checker admits the frozen full-material brightness band only')
        allocations=[x for x in finish['material_coat']['allocations'] if x['source_band']==band]
        require(len(allocations)==1,'Eligible face list is ambiguous')
        a=[x for x in allocations[0]['parts'] if x['material_name']==part['source_material']]
        require(len(a)==1,'Material allocation missing/ambiguous')
        global_faces=np.arange(original['face_offset'],original['face_offset']+len(original['faces']),dtype='<i4')
        require(a[0]['cards']==part['cards'] and a[0]['eligible_triangles']==len(global_faces)
                and a[0]['source_triangle_ids_sha256']==hashlib.sha256(global_faces.tobytes()).hexdigest(),
                'Eligible source-face list is not the proved ordered full-material list')
        ti,bary,insets,byte_count=decode_archive(pack,summary['binding_archive'],part['cards'],pins)
        path=summary['binding_archive']['path'];require(path not in archives,'Duplicate root archive');archives.add(path);archive_bytes+=byte_count
        require(archive_bytes<=1024*1024,'Total root archive byte limit')
        require((ti>=0).all() and (ti<len(global_faces)).all()
                and np.max(np.linalg.norm(insets,axis=1))<=request['gates']['max_authored_inset_m']+1e-12,
                'Binding triangle/inset escapes its declared bound')
        matches=[(i,p) for i,n in enumerate(fn) if 'mesh' in n and n.get('name','').startswith('PortableCoat_')
                 for p in final.doc['meshes'][n['mesh']]['primitives']
                 if final.doc['materials'][p['material']]['name']==part['material']['material']]
        require(len(matches)==1,'Final coat material primitive is ambiguous')
        node,p=matches[0];ctx=context(final,node,names);actual=data(final,p,ctx,names);contexts[part_id]=ctx
        require(len(actual['faces'])==part['cards']*4,'Final card triangle count changed')
        selected_kind={'uniform_interior':0,'source_vertex':0,'varying_triangle_vertex':0}
        details={x['root_index']:x for x in part['root_samples']}
        require(len(details)==len(part['root_samples']) and len(details)==min(128,part['cards'])
                and all(type(i) is int and 0<=i<part['cards'] for i in details),'Detailed root sample count/indices differ')
        sw=original['weights'];sums=sw.sum(axis=1);require((sums>0).all(),'Source skin has zero support')
        normalized=sw/sums[:,None];f=original['faces']
        uniform=np.all(normalized[f[:,0]]==normalized[f[:,1]],axis=1)&np.all(normalized[f[:,0]]==normalized[f[:,2]],axis=1)
        for root in range(part['cards']):
            ids=f[ti[root]];b=bary[root];source_p=original['points'][ids];source_w=sw[ids]
            hot=np.flatnonzero(b==1.)
            one_hot=len(hot)==1 and np.count_nonzero(b)==1
            require(one_hot or uniform[ti[root]],'Interior root lies on a varying-weight source triangle')
            inherited=b@source_w;require(int(np.count_nonzero(inherited>0))<=4,'Root requires truncating source influences')
            expected=inherited/inherited.sum()
            kind='source_vertex' if one_hot else 'uniform_interior';selected_kind[kind]+=1
            selected_kind['varying_triangle_vertex']+=int(one_hot and not uniform[ti[root]])
            card_ids=np.unique(actual['faces'][root*4:root*4+4]);weight_error=float(abs(actual['weights'][card_ids]-expected).max())
            max_weight=max(max_weight,weight_error)
            root_ids=card_ids[actual['uv'][card_ids,1]==actual['uv'][card_ids,1].max()]
            points,groups=np.unique(actual['points'][root_ids],axis=0,return_inverse=True)
            require(len(points)==2,'Bottom UV row does not prove two root endpoints')
            endpoints=[root_ids[groups==i] for i in range(2)]
            inset=insets[root][[0,2,1]]*np.array([1,1,-1]);center=b@source_p-inset
            rest_error=float(abs(points.mean(axis=0)-center).max());max_rest=max(max_rest,rest_error)
            if root in details:
                d=details[root];require(d['triangle_index']==int(ti[root]) and np.array_equal(d['barycentric'],b)
                    and d['source_polygon']==int(global_faces[ti[root]])
                    and d['source_vertices']==(ids+original['vertex_offset']).tolist(), 'Detailed sample/archive source identity differs')
                if 'source_vertex_index' in d:
                    require(one_hot and d['source_vertex_index']==int(ids[hot[0]]+original['vertex_offset']),
                            'Detailed one-hot source vertex ID differs')
                max_detail=max(max_detail,float(abs(np.asarray(d['normal'])*d['root_inset_m']-insets[root]).max()))
            records.append({'part':part_id,'actual':actual,'endpoints':endpoints,'source_p':source_p,'source_w':source_w,
                            'bary':b,'expected':expected,'center':center,'inset':inset})
            metadata.append({'part':part_id,'material':part['source_material'],'root':root,'source_polygon':int(global_faces[ti[root]]),
                'source_vertices':(ids+original['vertex_offset']).tolist(),'barycentric':b.tolist(),'binding_kind':kind,
                'varying_source_triangle':bool(not uniform[ti[root]]),'weight_error':weight_error,'rest_root_error_m':rest_error})
        require(not (~uniform).any() or selected_kind['varying_triangle_vertex']>0,
                'Varying-weight source domain has no selected vertex root')
        domain_reports.append({'material':part['source_material'],'source_band':band,'eligible_triangles':len(f),
            'uniform_triangles':int(uniform.sum()),'varying_triangles':int((~uniform).sum()),'selected':selected_kind})
    require(len(records)==total,'Not every root was audited')
    check(max_weight<=request['gates']['named_weights'],'Exported inherited weight mismatch')
    check(max_rest<=request['gates']['rest_detailed_root_m'],'Exported all-root midpoint/inset mismatch')
    check(max_detail<=1e-12,'Detailed samples differ from archived insets')
    save('roots.json',{'samples':metadata,'domain_coverage':domain_reports,'archive_bytes':archive_bytes,
        'max_weight_error':max_weight,'max_all_root_rest_error_m':max_rest,'max_detailed_inset_error_m':max_detail})
    sp={c:i for i,n in enumerate(sn) for c in n.get('children',[])};fp={c:i for i,n in enumerate(fn) for c in n.get('children',[])}
    st={name:t for name,t,_ in _animation_tracks(source,sn,set(sctx['joints']))}
    all_final_joints={j for ctx in contexts.values() for j in ctx['joints']}
    ft={name:t for name,t,_ in _animation_tracks(final,fn,all_final_joints)}
    require(set(st)==set(ft)=={'idle','making_something','milestone_level_up','working'},'Source action names changed')
    clips=[];all_actual=[];all_expected=[];all_surface=[];motion_max=attachment_max=duplicate_max=0.
    for clip in st:
        rows=[];clip_actual=[];clip_expected=[];clip_surface=[]
        for second in request['sample_seconds']:
            sm=matrices(_world_matrices(sn,sp,_pose(st[clip],second)),sctx)
            fw=_world_matrices(fn,fp,_pose(ft[clip],second));fm={i:matrices(fw,ctx) for i,ctx in contexts.items()}
            actual=[];expected=[];surface=[]
            for record in records:
                endpoints=[]
                for ids in record['endpoints']:
                    posed=weighted(record['actual']['points'][ids],record['actual']['weights'][ids],fm[record['part']])
                    duplicate_max=max(duplicate_max,float(abs(posed-posed[0]).max()));endpoints.append(posed.mean(axis=0))
                actual.append(np.mean(endpoints,axis=0))
                expected.append(weighted(record['center'][None,:],record['expected'][None,:],sm)[0])
                body_surface=record['bary']@weighted(record['source_p'],record['source_w'],sm)
                inset=np.einsum('b,bij,j->i',record['expected'],sm[:,:3,:3],record['inset'])
                surface.append(body_surface-inset)
            actual,expected,surface=map(np.asarray,(actual,expected,surface))
            me=abs(actual-expected).max(axis=1);ae=abs(actual-surface).max(axis=1)
            motion_max=max(motion_max,float(me.max()));attachment_max=max(attachment_max,float(ae.max()))
            worst=np.argsort(ae)[-8:][::-1]
            rows.append({'second':second,'declared_skin_max_error_m':float(me.max()),'surface_attachment_max_error_m':float(ae.max()),
                'roots_over_attachment_gate':int((ae>request['gates']['source_surface_attachment_m']).sum()),
                'witnesses':[{'record':int(i),'material':metadata[i]['material'],'root':metadata[i]['root'],
                              'source_polygon':metadata[i]['source_polygon'],'error_m':float(ae[i])} for i in worst]})
            clip_actual.append(actual);clip_expected.append(expected);clip_surface.append(surface)
        clips.append({'clip':clip,'samples':rows});all_actual.append(clip_actual);all_expected.append(clip_expected);all_surface.append(clip_surface)
    check(motion_max<=request['gates']['sampled_motion_m'],'Actual root motion differs from declared inherited skin')
    check(attachment_max<=request['gates']['source_surface_attachment_m'],'Actual roots drift from their deformed source surface')
    check(duplicate_max<=request['gates']['rest_detailed_root_m'],'Duplicated root vertices separate')
    np.savez_compressed(OUT/'sampled-motion.npz',actual_roots=np.asarray(all_actual),declared_skin_roots=np.asarray(all_expected),
                        source_surface_minus_inset=np.asarray(all_surface),seconds=np.asarray(request['sample_seconds']))
    save('motion.json',{'clips':clips,'max_declared_skin_error_m':motion_max,'max_surface_attachment_error_m':attachment_max,
                       'max_duplicate_root_error_m':duplicate_max,'all_roots_checked':total,'domain_coverage':domain_reports})
    RESULT.update(status='failed' if RESULT.get('failures') else 'passed',source_sha256=source.hash,final_sha256=final.hash,
        roots=total,clips=4,times_per_clip=9,max_named_weight_error=max_weight,max_rest_root_error_m=max_rest,
        max_declared_skin_error_m=motion_max,max_source_surface_attachment_error_m=attachment_max,
        scope='All archived roots only; no hair tips, native hair, GPU, appearance, loop or physics acceptance')
    require(not RESULT.get('failures'),'Full-root skin/attachment proof failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);args=parser.parse_args()
    try:
        run(Path(args.request))
    except BaseException as error:
        RESULT.update(status='failed',exception={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
    finally:
        if OUT is not None:
            RESULT.update(elapsed_seconds=time.monotonic()-START,
                          process_high_water_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            save('result.json',RESULT)
