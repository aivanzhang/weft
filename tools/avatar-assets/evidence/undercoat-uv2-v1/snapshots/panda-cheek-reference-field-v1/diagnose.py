"""Reference-only diagnosis against a frozen failed fit; no new fit or mesh mutation."""
from collections import Counter, defaultdict, deque
import hashlib
import itertools
import json
from pathlib import Path
import struct
import time

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
START=time.monotonic()
V7=ROOT/'outputs/companion-separation-v7'
DOMAIN=ROOT/'outputs/panda-cheek-targeted-selection-v2/candidate.json'
FAILED=ROOT/'outputs/panda-cheek-curved-candidate-v1'


def digest(path):
    with path.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(name,value):
    with(OUT/name).open('x')as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def require(condition,message):
    if not condition:raise ValueError(message)


def stats(values):
    a=np.asarray(values,dtype=float)
    return {'count':int(a.size),'median':float(np.median(a))if a.size else None,
        'p95':float(np.percentile(a,95))if a.size else None,'max':float(a.max())if a.size else None,
        'rms':float(np.sqrt(np.mean(a*a)))if a.size else None}


def angle(a,b):
    return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1),-1,1)))


def blender(values):
    out=np.asarray(values,dtype=float)[:,[0,2,1]].copy();out[:,1]*=-1;return out


config=json.loads((OUT/'config.json').read_text())
domain=json.loads(DOMAIN.read_text());fit=json.loads((FAILED/'fit-report.json').read_text())
inputs=[V7/'diagnostic-model.glb',V7/'partition-arrays.npz',V7/'partition-plan.json',DOMAIN,
        FAILED/'manifest.json',FAILED/'fit-report.json',FAILED/'config.json']
write('inputs.json',[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)}for p in inputs])
require(digest(V7/'diagnostic-model.glb')==domain['source_model_sha256'],'Source custody mismatch')
with np.load(V7/'partition-arrays.npz',allow_pickle=False)as data:
    points=data['points'].astype(float);faces=data['faces'];parts=data['face_labels'];materials=data['face_material_indices']
require(len(faces)<=100000 and len(points)<=100000,'Source exceeds geometry limit')
triangles=points[faces];centers=triangles.mean(1)
cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);double_area=np.linalg.norm(cross,axis=1)
geo=cross/np.maximum(double_area[:,None],1e-30)
raw=(V7/'diagnostic-model.glb').read_bytes();require(len(raw)<=64*1024**2,'Source file too large')
size=struct.unpack_from('<I',raw,12)[0];doc=json.loads(raw[20:20+size]);binary=memoryview(raw)[28+size:]


def array(index):
    a=doc['accessors'][index];v=doc['bufferViews'][a['bufferView']]
    require('sparse'not in a and a['count']<=300000,'Unsupported accessor')
    dtype={5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
    width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']];item=np.dtype(dtype).itemsize
    return np.ndarray((a['count'],width),dtype=dtype,buffer=binary,
        offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',width*item),item))


primitive=doc['meshes'][0]['primitives'][0]
require(doc['materials'][primitive['material']]['name']=='body_surface','Reference material changed')
attrs=primitive['attributes'];gf=array(primitive['indices']).reshape(-1,3)
gp=blender(array(attrs['POSITION']));gn=blender(array(attrs['NORMAL']));gn/=np.linalg.norm(gn,axis=1)[:,None]
gtri=gp[gf];gcenters=gtri.mean(1)
bins=defaultdict(list);gbins=defaultdict(list)
for fid in np.flatnonzero((parts==0)&(materials==0)):
    bins[tuple(np.floor(centers[fid]/1e-4).astype(int))].append(int(fid))
for fid,center in enumerate(gcenters):gbins[tuple(np.floor(center/1e-4).astype(int))].append(fid)
offsets=list(itertools.product((-1,0,1),repeat=3))
source_map={};reverse_map={};match_errors=[]


def join_to_raw(gid):
    if gid in source_map:return source_map[gid]
    q=np.floor(gcenters[gid]/1e-4).astype(int)
    candidates=[f for d in offsets for f in bins.get(tuple(q+d),[])]
    matches=[]
    for fid in candidates:
        for shift in range(3):
            error=float(np.max(np.abs(np.roll(triangles[fid],shift,axis=0)-gtri[gid])))
            if error<=2e-7:matches.append((fid,shift,error))
    require(len(matches)==1,'Reference source provenance is missing or ambiguous')
    fid,shift,error=matches[0];source_map[gid]=fid;reverse_map[fid]=gid;match_errors.append(error)
    return fid


def join_to_glb(fid):
    if fid in reverse_map:return reverse_map[fid]
    q=np.floor(centers[fid]/1e-4).astype(int)
    candidates=[f for d in offsets for f in gbins.get(tuple(q+d),[])]
    matches=[]
    for gid in candidates:
        for shift in range(3):
            error=float(np.max(np.abs(np.roll(triangles[fid],shift,axis=0)-gtri[gid])))
            if error<=2e-7:matches.append((gid,error))
    require(len(matches)==1,'Candidate-band source provenance is missing or ambiguous')
    gid,error=matches[0];source_map[gid]=fid;reverse_map[fid]=gid;match_errors.append(error)
    return gid


# Reproduce v7's recorded contact-interface policy, not a new spatial guess.
_,weld=np.unique(np.round(points,6),axis=0,return_inverse=True)
wf=weld[faces];edge_faces=defaultdict(list)
for fid,tri in enumerate(wf):
    for a,b in zip(tri,np.roll(tri,-1)):edge_faces[tuple(sorted((int(a),int(b))))].append(fid)
body_adj=defaultdict(set);contact=set();nonmanifold=set()
for incident in edge_faces.values():
    body=[fid for fid in incident if parts[fid]==0]
    if len({int(parts[fid])for fid in incident})>1:contact.update(body)
    if len(body)>2:nonmanifold.update(body)
    for fid in body:body_adj[fid].update(other for other in body if other!=fid)
distance={fid:0 for fid in contact};queue=deque(sorted(contact))
while queue:
    fid=queue.popleft()
    if distance[fid]>=8:continue
    for other in body_adj[fid]:
        if other not in distance:distance[other]=distance[fid]+1;queue.append(other)
reference_glb=[int(r['source_face'][1])for r in domain['healthy_remaining_original_body_reference_triangles']]
require(len(reference_glb)==132 and all(r['source_face'][0]==0 for r in domain['healthy_remaining_original_body_reference_triangles']),
        'Unexpected source reference list')
reference_raw={join_to_raw(gid)for gid in reference_glb}
band_distance={fid:0 for fid in reference_raw};queue=deque(sorted(reference_raw))
while queue:
    fid=queue.popleft()
    if band_distance[fid]>=config['reference_only_edge_rings']:continue
    for other in body_adj[fid]:
        if other not in band_distance:band_distance[other]=band_distance[fid]+1;queue.append(other)
boundary=np.asarray(domain['oriented_boundary_blender_m']);low=boundary.min(0)-config['reference_margin_m'];high=boundary.max(0)+config['reference_margin_m']
pool=sorted(fid for fid in band_distance if materials[fid]==0 and np.all(centers[fid]>=low)and np.all(centers[fid]<=high))
require(len(pool)<=config['max_candidate_faces'],'Reference-only band exceeds face cap')
origin=np.asarray(fit['origin_blender_m']);basis=np.asarray(fit['basis_blender_columns']);scale=fit['normalization_scale_m'];coeff=np.asarray(fit['quadratic_coefficients'])


def fixed_fit(position):
    local=(np.asarray(position)-origin)@basis/scale;x,y=local[...,0],local[...,1]
    height=coeff[0]+coeff[1]*x+coeff[2]*y+coeff[3]*x*x+coeff[4]*x*y+coeff[5]*y*y
    du=coeff[1]+2*coeff[3]*x+coeff[4]*y;dv=coeff[2]+coeff[4]*x+2*coeff[5]*y
    n=np.stack([-du,-dv,np.ones_like(du)],axis=-1)@basis.T;n/=np.linalg.norm(n,axis=-1)[...,None]
    return n,(height-local[...,2])*scale


records=[];kept=[]
for fid in pool:
    require(time.monotonic()-START<25,'Diagnostic deadline reached')
    gid=join_to_glb(fid);p=gtri[gid];custom=gn[gf[gid]];geometric=geo[fid]
    predicted,residual=fixed_fit(p)
    custom_geo=angle(custom,np.broadcast_to(geometric,custom.shape))
    custom_fit=angle(custom,predicted);geometry_fit=angle(np.broadcast_to(geometric,custom.shape),predicted)
    neighbors=[other for other in body_adj[fid]if double_area[other]>1e-12]
    dihedral=angle(np.broadcast_to(geometric,(len(neighbors),3)),geo[neighbors])if neighbors else np.asarray([])
    maximum_dihedral=float(dihedral.max())if dihedral.size else 180.
    support_distance=distance.get(fid,9)
    rejection=[]
    if support_distance<config['minimum_contact_edge_distance']:rejection.append('contact_interface_or_first_body_ring')
    if maximum_dihedral>config['max_reference_dihedral_degrees']:rejection.append('original_body_geometric_crease')
    if custom_geo.max()>config['max_custom_geometric_degrees']:rejection.append('custom_geometric_normal_disagreement')
    if fid in nonmanifold:rejection.append('original_body_nonmanifold_edge')
    if double_area[fid]<=1e-10:rejection.append('near_degenerate_original_triangle')
    record={'source_primitive':0,'source_triangle':gid,'partition_original_face_id':fid,
        'was_original_reference':fid in reference_raw,'reference_only_ring':band_distance[fid],
        'contact_interface_edge_distance':support_distance,'contact_distance_censored_at':9,
        'centroid_blender_m':centers[fid].tolist(),'area_m2':float(double_area[fid]*.5),
        'custom_geometric_corner_degrees':custom_geo.tolist(),'custom_vs_failed_fit_corner_degrees':custom_fit.tolist(),
        'geometric_vs_failed_fit_corner_degrees':geometry_fit.tolist(),'point_residual_to_failed_fit_m':residual.tolist(),
        'maximum_original_body_edge_dihedral_degrees':maximum_dihedral,
        'body_neighbor_source_face_ids':neighbors,'rejected_reasons':rejection}
    records.append(record)
    if not rejection:
        kept.append({'source_primitive':0,'source_triangle':gid,'partition_original_face_id':fid,
            'rest_positions_blender_m':p.tolist(),'source_custom_corner_normals_blender':custom.tolist(),
            'proposed_geometric_corner_normals_blender':[geometric.tolist()]*3,
            'area_m2':float(double_area[fid]*.5),'reference_only_ring':band_distance[fid],
            'contact_interface_edge_distance':support_distance})
write('face-diagnostics.json',records)
require(len(kept)<=config['max_selected_faces'],'Cleaner field exceeds selected face cap')
write('proposed-reference-field.json',{'source_model_sha256':domain['source_model_sha256'],
    'replacement_domain_sha256':digest(DOMAIN),'selection_policy':config,
    'field':kept,'field_count':len(kept),'no_fit_performed':True,
    'normal_scope':'Geometric source-face normals are proposed support, not custom caps, recovered anatomy or finished patch normals',
    'selection_does_not_use_failed_fit_error':True,'model_modified':False,'quality_approved':False})


def collect(rows,key):return [v for r in rows for v in r[key]]


old=[r for r in records if r['was_original_reference']]
require(len(old)==132,'Original reference coverage changed')
outliers=[r for r in old if max(r['custom_vs_failed_fit_corner_degrees'])>45]
selected_records=[r for r in records if not r['rejected_reasons']]
patch_boundary_set={tuple(x)for x in domain['replacement_source_faces']}|{tuple(x)for x in domain['retired_without_replacement_source_faces']}
require(not patch_boundary_set&{(0,r['source_triangle'])for r in kept},'Reference field overlaps repair ownership')
angular=(np.asarray([r['rest_positions_blender_m']for r in kept]).mean(1)-origin)@basis if kept else np.empty((0,3))
sectors=np.floor((np.arctan2(angular[:,1],angular[:,0])+np.pi)/(np.pi/4)).astype(int)%8
summary={'schema_version':1,'status':'reference_diagnosis_not_new_fit','source_model_sha256':domain['source_model_sha256'],
    'failed_fit_sha256':digest(FAILED/'fit-report.json'),'source_match_max_component_error_m':max(match_errors),
    'original_reference_faces':len(old),'original_reference_custom_geometric_degrees':stats(collect(old,'custom_geometric_corner_degrees')),
    'original_reference_custom_vs_failed_fit_degrees':stats(collect(old,'custom_vs_failed_fit_corner_degrees')),
    'original_reference_geometric_vs_failed_fit_degrees':stats(collect(old,'geometric_vs_failed_fit_corner_degrees')),
    'outlier_reference_faces':len(outliers),'outlier_faces_touching_contact_band':sum(r['contact_interface_edge_distance']<2 for r in outliers),
    'outlier_faces_with_geometric_crease':sum(r['maximum_original_body_edge_dihedral_degrees']>config['max_reference_dihedral_degrees']for r in outliers),
    'outlier_faces_with_custom_disagreement':sum(max(r['custom_geometric_corner_degrees'])>config['max_custom_geometric_degrees']for r in outliers),
    'reference_only_band_faces':len(records),'selected_cleaner_faces':len(kept),
    'rejected_reason_counts':dict(Counter(reason for r in records for reason in r['rejected_reasons'])),
    'cleaner_field_custom_geometric_degrees':stats(collect(selected_records,'custom_geometric_corner_degrees')),
    'cleaner_field_geometric_vs_fixed_failed_fit_degrees':stats(collect(selected_records,'geometric_vs_failed_fit_corner_degrees')),
    'cleaner_field_point_residual_to_fixed_failed_fit_m':stats(collect(selected_records,'point_residual_to_failed_fit_m')),
    'cleaner_field_eight_sector_counts':dict(Counter(map(int,sectors))),
    'original_contact_body_faces':len(contact),'max_reference_only_edge_rings':config['reference_only_edge_rings'],
    'elapsed_seconds':time.monotonic()-START,'source_unchanged':digest(V7/'diagnostic-model.glb')==domain['source_model_sha256'],
    'model_modified':False,'new_fit_performed':False,'native_jobs':0,'provider_calls':0,'textures_decoded':0,'quality_approved':False,
    'limitations':['Low dihedral and source contact distance are evidence-informed support rules, not proof that an authored hidden surface matches ground truth.',
        'Comparison against the frozen failed coefficients is diagnostic only. No gate was relaxed and no replacement geometry was produced.',
        'Sector counts are coarse support coverage, not a convex-hull or full extrapolation certificate.']}
write('summary.json',summary)
print(json.dumps(summary,sort_keys=True))
