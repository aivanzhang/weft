"""One bounded authored curved-patch candidate; fail before export on numeric defects."""
from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import time

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
START=time.monotonic()
SOURCE=ROOT/'outputs/companion-separation-v7/diagnostic-model.glb'
DOMAIN=ROOT/'outputs/panda-cheek-targeted-selection-v2/candidate.json'


class Unsupported(Exception):pass


def require(condition,message):
    if not condition:raise Unsupported(message)


def digest(path):
    with path.open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(name,value):
    with(OUT/name).open('x')as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def to_blender(values):
    result=np.asarray(values,dtype=float)[:,[0,2,1]].copy();result[:,1]*=-1;return result


def to_gltf(values):
    result=np.asarray(values,dtype=float)[:,[0,2,1]].copy();result[:,2]*=-1;return result


def stats(values):
    a=np.asarray(values,dtype=float)
    return {'count':int(a.size),'min':float(a.min()),'median':float(np.median(a)),
        'p95':float(np.percentile(a,95)),'max':float(a.max()),'rms':float(np.sqrt(np.mean(a*a)))}


def main():
    config=json.loads((OUT/'config.json').read_text());domain=json.loads(DOMAIN.read_text())
    require(digest(SOURCE)==domain['source_model_sha256']==config['source_sha256'],'Source hash changed')
    require(digest(DOMAIN)==config['domain_sha256'],'Domain hash changed')
    raw=SOURCE.read_bytes();require(len(raw)<=64*1024**2,'Source file exceeds64MiB')
    size=struct.unpack_from('<I',raw,12)[0];doc=json.loads(raw[20:20+size]);binary=memoryview(raw)[28+size:]
    require(not doc.get('skins')and not doc.get('animations')and doc['nodes']==[{'mesh':0,'name':'PartitionedBodyAndPart'}],
            'This candidate requires the pinned unrigged identity source')
    def array(index):
        a=doc['accessors'][index];v=doc['bufferViews'][a['bufferView']]
        require('sparse'not in a and a['count']<=300000,'Unsupported accessor')
        dtype={5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
        width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']];item=np.dtype(dtype).itemsize
        return np.ndarray((a['count'],width),dtype=dtype,buffer=binary,
            offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',width*item),item))
    primitives=doc['meshes'][0]['primitives']
    require([doc['materials'][p['material']]['name']for p in primitives]==['body_surface','companion_surface','body_cap','companion_cap'],
            'Source part/material inventory changed')
    geometry=[]
    for pi,p in enumerate(primitives):
        a=p['attributes'];f=array(p['indices']).reshape(-1,3)
        geometry.append({'p':to_blender(array(a['POSITION'])),'n':to_blender(array(a['NORMAL'])),
                         'uv':array(a['TEXCOORD_0']),'f':f})
    selected=[tuple(x)for x in domain['replacement_source_faces']]
    retired=[tuple(x)for x in domain['retired_without_replacement_source_faces']]
    require(len(selected)==124 and {p for p,_ in selected}=={2}and retired==[(0,4961)],'Unexpected local replacement ownership')
    # Domain welded IDs are used only to reconstruct its exact, already-tested body topology.
    body_positions=np.concatenate([geometry[pi]['p']for pi in(0,2)])
    # Match the prior GLB-axis lexicographic unique order before converting coordinates.
    unique_gltf,inverse=np.unique(to_gltf(body_positions).astype('<f4'),axis=0,return_inverse=True)
    unique=to_blender(unique_gltf)
    offset=len(geometry[0]['p']);source_global_faces={}
    for pi in(0,2):
        off=0 if pi==0 else offset
        for fid,tri in enumerate(geometry[pi]['f']):source_global_faces[(pi,fid)]=inverse[tri+off]
    global_ids=sorted({int(v)for key in selected for v in source_global_faces[key]})
    local={v:i for i,v in enumerate(global_ids)}
    points=unique[global_ids].copy()
    triangles=np.asarray([[local[int(v)]for v in source_global_faces[key]]for key in selected],dtype=np.int32)
    original_triangle_origins=[list(key)for key in selected]
    boundary_global=domain['oriented_boundary_body_welded_ids'];boundary=np.asarray([local[v]for v in boundary_global],dtype=int)
    require(np.array_equal(points[boundary],np.asarray(domain['oriented_boundary_blender_m'])),'Boundary position custody failed')
    forbidden=domain['forbidden_existing_body_diagonals'];require(len(forbidden)==1,'Expected one forbidden source-body diagonal')
    a,b=[local[v]for v in forbidden[0]['body_welded_vertices']]
    new_index=len(points);points=np.vstack([points,(points[a]+points[b])*.5]);global_ids.append(-1)
    split=[];triangle_origins=[]
    for tri,origin in zip(triangles,original_triangle_origins):
        rotated=None
        for shift in range(3):
            t=np.roll(tri,-shift)
            if {int(t[0]),int(t[1])}=={a,b}:rotated=t;break
        if rotated is None:split.append(tri.tolist());triangle_origins.append(origin)
        else:
            x,y,z=map(int,rotated);split.extend([[x,new_index,z],[new_index,y,z]])
            triangle_origins.extend([origin,origin])
    triangles=np.asarray(split,dtype=np.int32)
    require(len(triangles)==126 and len(triangles)<=config['max_triangles'],'Unexpected/oversized split topology')
    require(all(not({a,b}<=set(map(int,t)))for t in triangles),'Forbidden diagonal remains')
    original=points.copy();bset=set(map(int,boundary));interior=np.asarray([i for i in range(len(points))if i not in bset],dtype=int)
    spur=geometry[0]['f'][4961]
    np.savez_compressed(OUT/'retired-source-spur.npz',positions=geometry[0]['p'][spur],normals=geometry[0]['n'][spur],uv0=geometry[0]['uv'][spur])

    refs=domain['healthy_remaining_original_body_reference_triangles']
    refp=np.asarray([r['rest_positions_blender_m']for r in refs],dtype=float).reshape(-1,3)
    refn=np.asarray([r['corner_normals_blender']for r in refs],dtype=float).reshape(-1,3)
    require(np.isfinite(refp).all()and np.isfinite(refn).all(),'Nonfinite reference data')
    refn/=np.linalg.norm(refn,axis=1)[:,None]
    axis=np.median(refn,axis=0);axis/=np.linalg.norm(axis)
    require(np.isfinite(axis).all(),'Reference normal axis is undefined')
    helper=np.eye(3)[np.argmin(np.abs(axis))];u=np.cross(helper,axis);u/=np.linalg.norm(u);v=np.cross(axis,u)
    basis=np.column_stack([u,v,axis]);origin=points[boundary].mean(0)
    scale=float(np.ptp(points[boundary],axis=0).max());require(.001<scale<1,'Invalid local fit scale')
    local_ref=(refp-origin)@basis/scale;normal_ref=refn@basis
    support=normal_ref[:,2]>=config['minimum_reference_normal_cosine']
    require(support.sum()>=config['minimum_reference_samples'],'Insufficient consistently oriented healthy reference support')
    x,y,z=local_ref[support].T;nr=normal_ref[support]
    A=np.column_stack([np.ones(len(x)),x,y,x*x,x*y,y*y])
    Au=np.column_stack([np.zeros(len(x)),np.ones(len(x)),np.zeros(len(x)),2*x,y,np.zeros(len(x))])
    Av=np.column_stack([np.zeros(len(x)),np.zeros(len(x)),np.ones(len(x)),np.zeros(len(x)),x,2*y])
    w=config['hermite_normal_weight'];design=np.vstack([A,Au*w,Av*w]);target=np.concatenate([z,-nr[:,0]/nr[:,2]*w,-nr[:,1]/nr[:,2]*w])
    weights=np.ones(len(target));coefficients=None
    for _ in range(3):
        matrix=design*np.sqrt(weights)[:,None]
        coefficients=np.linalg.lstsq(matrix,target*np.sqrt(weights),rcond=None)[0]
        residual=design@coefficients-target
        scales=np.concatenate([np.full(len(x),config['point_huber_m']/scale),np.full(2*len(x),config['slope_huber']*w)])
        weights=1/np.maximum(1,np.abs(residual)/scales)
    def polynomial(coords):
        xx,yy=coords[:,0],coords[:,1]
        return np.column_stack([np.ones(len(xx)),xx,yy,xx*xx,xx*yy,yy*yy])@coefficients
    def fitted_normals(coords):
        xx,yy=coords[:,0],coords[:,1]
        du=coefficients[1]+2*coefficients[3]*xx+coefficients[4]*yy
        dv=coefficients[2]+coefficients[4]*xx+2*coefficients[5]*yy
        normals=np.column_stack([-du,-dv,np.ones(len(xx))])@basis.T
        return normals/np.linalg.norm(normals,axis=1)[:,None]
    reference_residual=(polynomial(local_ref)-local_ref[:,2])*scale
    reference_angle=np.degrees(np.arccos(np.clip(np.sum(fitted_normals(local_ref)*refn,axis=1),-1,1)))
    fit={'reference_faces':len(refs),'reference_samples':len(refp),'supported_samples':int(support.sum()),
        'supported_fraction':float(support.mean()),'excluded_reference_corner_ids':np.flatnonzero(~support).tolist(),
        'basis_blender_columns':basis.tolist(),'origin_blender_m':origin.tolist(),'normalization_scale_m':scale,
        'quadratic_coefficients':coefficients.tolist(),'condition_number':float(np.linalg.cond(matrix)),
        'supported_point_residual_m':stats(np.abs(reference_residual[support])),
        'all_reference_point_residual_m':stats(np.abs(reference_residual)),
        'supported_normal_error_degrees':stats(reference_angle[support]),
        'assumption':'Authored local Hermite quadratic inferred from retained body references, not recovered hidden anatomy'}
    write('fit-report.json',fit)
    require(fit['condition_number']<=config['max_condition_number'],'Ill-conditioned local curved fit')
    require(fit['supported_fraction']>=config['minimum_support_fraction'],'Too many reference normals lack support')
    require(fit['supported_point_residual_m']['rms']<=config['max_fit_rms_m']and fit['supported_point_residual_m']['max']<=config['max_fit_max_m'],
            'Reference point residual exceeds curved-fit gate')
    require(fit['supported_normal_error_degrees']['p95']<=config['max_fit_normal_p95_degrees'],'Reference normal residual exceeds curved-fit gate')

    coordinates=(points-origin)@basis/scale;predicted=polynomial(coordinates)
    correction=np.zeros(len(points));correction[boundary]=coordinates[boundary,2]-predicted[boundary]
    adjacency=[set()for _ in points]
    for tri in triangles:
        for x,y in zip(tri,np.roll(tri,-1)):adjacency[int(x)].add(int(y));adjacency[int(y)].add(int(x))
    L=np.zeros((len(interior),len(interior)));rhs=np.zeros(len(interior));ii={int(q):i for i,q in enumerate(interior)}
    for row,vertex in enumerate(interior):
        for other in adjacency[int(vertex)]:
            weight=1/max(np.linalg.norm(original[vertex]-original[other]),1e-6)
            L[row,row]+=weight
            if other in ii:L[row,ii[other]]-=weight
            else:rhs[row]+=weight*correction[other]
    require(np.linalg.cond(L)<1e8,'Ill-conditioned bounded boundary correction')
    correction[interior]=np.linalg.solve(L,rhs)
    delta=(predicted+correction-coordinates[:,2])*scale
    points[interior]+=delta[interior,None]*axis
    points[boundary]=original[boundary]
    displacement=np.linalg.norm(points-original,axis=1)
    cross=np.cross(points[triangles[:,1]]-points[triangles[:,0]],points[triangles[:,2]]-points[triangles[:,0]])
    cross_length=np.linalg.norm(cross,axis=1)
    geometry_report={'vertices':len(points),'triangles':len(triangles),'boundary_vertices':len(boundary),
        'boundary_position_max_component_error_m':float(np.max(np.abs(points[boundary]-original[boundary]))),
        'interior_displacement_m':stats(displacement[interior]),'minimum_triangle_area_m2':float(cross_length.min()*.5),
        'forbidden_diagonal_removed':True,'new_steiner_vertex':new_index}
    np.savez_compressed(OUT/'patch-candidate.npz',positions=points.astype('<f4'),original_positions=original.astype('<f4'),
        triangles=triangles.astype('<u4'),boundary=boundary.astype('<u4'),global_source_position_ids=np.asarray(global_ids,dtype='<i4'))
    write('geometry-report.json',geometry_report)
    require(displacement[interior].max()<=config['max_displacement_m'],'Interior displacement exceeds15mm gate')
    require(cross_length.min()>=config['min_double_area_m2'],'Degenerate/sliver triangle below area gate')
    geometric_normals=cross/cross_length[:,None]
    field=fitted_normals((points[triangles].mean(1)-origin)@basis/scale)
    facing=np.sum(geometric_normals*field,axis=1)
    write('orientation-report.json',{'geometric_vs_fit_normal_dot':stats(facing),'nonpositive_face_ids':np.flatnonzero(facing<=0).tolist()})
    require(np.all(facing>0),'Replacement has reversed oriented triangles against supported fitted surface')
    normals=np.zeros_like(points)
    np.add.at(normals,triangles.ravel(),np.repeat(cross,3,axis=0));normals/=np.linalg.norm(normals,axis=1)[:,None]
    # Boundary normals come from retained body samples, never from old cap normals.
    boundary_support=[]
    for vertex in boundary:
        exact=np.max(np.abs(refp-points[vertex]),axis=1)<=2e-7
        require(exact.any(),'Missing exact retained normal support at boundary')
        mean=refn[exact].sum(0);require(np.linalg.norm(mean)>1e-8,'Conflicting boundary normal support')
        normals[vertex]=mean/np.linalg.norm(mean);boundary_support.append(int(exact.sum()))
    dots=np.sum(normals[triangles]*geometric_normals[:,None,:],axis=2)
    write('normal-report.json',{'minimum_corner_geometric_dot':float(dots.min()),
        'corner_geometric_angle_degrees':stats(np.degrees(np.arccos(np.clip(dots,-1,1)))),
        'boundary_exact_reference_sample_counts':boundary_support})
    require(np.all(dots>0),'New custom normals oppose replacement geometry')

    spec=importlib.util.spec_from_file_location('trusted_partition_geometry',OUT/'partition_geometry.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    retained=[];retained_ids=[]
    owned=set(selected)|set(retired)
    for pi in(0,2):
        g=geometry[pi]
        for fid,tri in enumerate(g['f']):
            if(pi,fid)not in owned:retained.append(g['p'][tri]);retained_ids.append([pi,fid])
    retained=np.asarray(retained);lower=retained.min(1);upper=retained.max(1);new_tri=points[triangles]
    hits=[];tested=0
    for fid,tri in enumerate(new_tri):
        require(time.monotonic()-START<40,'Intersection screen exceeded aggregate time envelope')
        nearby=np.flatnonzero(np.all(lower<=tri.max(0)+1e-8,axis=1)&np.all(upper>=tri.min(0)-1e-8,axis=1))
        for other in nearby:
            tested+=1;require(tested<=config['max_intersection_pairs'],'Intersection candidate bound exceeded')
            if module.triangles_intersect_3d(tri,retained[other]):hits.append([fid,*retained_ids[int(other)]])
    self_hits=[]
    for fid,tri in enumerate(new_tri):
        for other in range(fid):
            if module.triangles_intersect_3d(tri,new_tri[other]):self_hits.append([fid,other])
    intersection={'scope':'Proper 3D/coplanar triangle screen; shared-edge/corner pairs are not blindly skipped',
        'retained_body_tested_pairs':tested,'patch_body_hits':hits,'patch_self_hits':self_hits,
        'companion_contact_scope':'Protected companion geometry is unchanged; at-rest contact is not accepted or visually validated by this body-only gate'}
    write('intersection-report.json',intersection)
    require(not hits and not self_hits,'New patch intersects retained body or itself')

    # Append-only GLB derivative keeps every original binary byte and non-owned JSON field.
    target=copy.deepcopy(doc);blob=bytearray(binary)
    def append(values,kind,component=5126):
        dtype={5123:'<u2',5125:'<u4',5126:'<f4'}[component];values=np.ascontiguousarray(values,dtype=dtype)
        while len(blob)%4:blob.append(0)
        offset=len(blob);blob.extend(values.tobytes());vi=len(target['bufferViews']);target['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':values.nbytes})
        ai=len(target['accessors']);entry={'bufferView':vi,'componentType':component,'count':len(values),'type':kind}
        if kind=='VEC3':entry.update(min=values.min(0).astype(float).tolist(),max=values.max(0).astype(float).tolist())
        target['accessors'].append(entry);return ai
    p0=target['meshes'][0]['primitives'][0]
    p0['indices']=append(np.delete(geometry[0]['f'],4961,axis=0).reshape(-1,1),'SCALAR',5125)
    p2=target['meshes'][0]['primitives'][2];attrs=primitives[2]['attributes']
    require(set(attrs)=={'POSITION','NORMAL','TEXCOORD_0','TANGENT'},'Unsupported cap attribute inventory')
    old_count=len(array(attrs['POSITION']));new_positions=to_gltf(points)[triangles].reshape(-1,3)
    new_normals=to_gltf(normals)[triangles].reshape(-1,3)
    palette=json.loads((ROOT/'outputs/companion-separation-v7/partition-plan.json').read_text())['loop_palette_patches'][27]['uv_rect_blender']
    uv=np.asarray([[.1,.1],[.9,.1],[.1,.9]])*(np.asarray(palette[2:])-palette[:2])+palette[:2];uv[:,1]=1-uv[:,1]
    new_uv=np.tile(uv,(len(triangles),1))
    tangents=[]
    for pts,ns in zip(new_positions.reshape(-1,3,3),new_normals.reshape(-1,3,3)):
        e1,e2=pts[1]-pts[0],pts[2]-pts[0];d1,d2=uv[1]-uv[0],uv[2]-uv[0];det=d1[0]*d2[1]-d1[1]*d2[0]
        t=(e1*d2[1]-e2*d1[1])/det;bvec=(e2*d1[0]-e1*d2[0])/det
        for n in ns:
            tangent=t-n*np.dot(n,t);require(np.linalg.norm(tangent)>1e-10,'Degenerate authored tangent')
            tangent/=np.linalg.norm(tangent);sign=1. if np.dot(np.cross(n,tangent),bvec)>=0 else -1.
            tangents.append([*tangent,sign])
    for name,values,kind in [('POSITION',new_positions,'VEC3'),('NORMAL',new_normals,'VEC3'),('TEXCOORD_0',new_uv,'VEC2'),('TANGENT',tangents,'VEC4')]:
        p2['attributes'][name]=append(np.concatenate([array(attrs[name]),np.asarray(values)],axis=0),kind)
    retained_cap=np.asarray([tri for fid,tri in enumerate(geometry[2]['f'])if(2,fid)not in set(selected)],dtype=np.uint32)
    p2['indices']=append(np.concatenate([retained_cap.ravel(),np.arange(old_count,old_count+len(new_positions),dtype=np.uint32)]).reshape(-1,1),'SCALAR',5125)
    while len(blob)%4:blob.append(0)
    target['buffers'][0]['byteLength']=len(blob)
    encoded=json.dumps(target,separators=(',',':'),allow_nan=False).encode();encoded+=b' '*((-len(encoded))%4)
    output=struct.pack('<4sII',b'glTF',2,28+len(encoded)+len(blob))+struct.pack('<II',len(encoded),0x4E4F534A)+encoded+struct.pack('<II',len(blob),0x004E4942)+blob
    require(len(output)<=64*1024**2,'Derivative exceeds64MiB')
    require(bytes(blob[:len(binary)])==binary,'Original BIN prefix changed')
    with(OUT/'model.glb').open('xb')as stream:stream.write(output)
    np.savez_compressed(OUT/'editable-patch.npz',positions_blender=points.astype('<f4'),triangles=triangles.astype('<u4'),
        normals_blender=normals.astype('<f4'),uv0_gltf=new_uv.reshape(-1,3,2).astype('<f4'),boundary_local_ids=boundary.astype('<u4'))
    mapping={'source_sha256':digest(SOURCE),'model_sha256':digest(OUT/'model.glb'),
        'removed_original_faces':[list(x)for x in retired],'replaced_source_faces':[list(x)for x in selected],
        'new_patch_triangle_sources':triangle_origins,'new_patch_primitive':2,'new_patch_face_start':len(retained_cap),
        'retained_original_cap_triangle_ids':[fid for fid in range(len(geometry[2]['f']))if(2,fid)not in set(selected)],
        'source_body_face_mapping':'primitive0: old IDs below4961 unchanged; above4961 become old-1;4961retired',
        'original_binary_prefix_unchanged':True,'original_binary_bytes':len(binary),'original_binary_sha256':hashlib.sha256(binary).hexdigest(),
        'source_images_materials_and_companion_primitives_exact':target['images']==doc['images']and target['materials']==doc['materials']and target['meshes'][0]['primitives'][1::2]==primitives[1::2],
        'source_unrigged':True,'quality_approved':False,'authored_assumption':'Local unseen cheek patch and its flat ivory source-palette UVs are authored; no recovered anatomy or texture claim'}
    write('mapping.json',mapping)
    return {'status':'cpu_candidate_numeric_gates_passed_not_visual_acceptance','model_sha256':mapping['model_sha256'],
        'replacement_triangles':len(triangles),'retired_original_faces':1,'elapsed_seconds':time.monotonic()-START,
        'source_unchanged':digest(SOURCE)==config['source_sha256'],'native_jobs':0,'provider_calls':0,'quality_approved':False}


try:result=main()
except Unsupported as error:
    result={'status':'unsupported','gate':str(error),'elapsed_seconds':time.monotonic()-START,
        'model_written':(OUT/'model.glb').exists(),'native_jobs':0,'provider_calls':0,'quality_approved':False}
    write('failure.json',result)
write('result.json',result)
print(json.dumps(result,sort_keys=True))
