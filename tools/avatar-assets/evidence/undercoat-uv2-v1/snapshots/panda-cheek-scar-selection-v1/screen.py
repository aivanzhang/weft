"""Targeted body fan selection screen; no model derivative or texture decode."""
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
SOURCE=ROOT/'outputs/companion-separation-v7/diagnostic-model.glb'
PRIOR=ROOT/'outputs/panda-cheek-patch-topology-v1'


def digest(path):
    with path.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(name,value):
    with(OUT/name).open('x')as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def groups(nodes,adj):
    remaining=set(nodes);result=[]
    for start in sorted(remaining):
        if start not in remaining:continue
        remaining.remove(start);queue=deque([start]);group=[]
        while queue:
            item=queue.popleft();group.append(item)
            for other in adj[item]:
                if other in remaining:remaining.remove(other);queue.append(other)
        result.append(sorted(group))
    return sorted(result,key=lambda x:(-len(x),x[0]))


prior=json.loads((PRIOR/'boundary-attachment.json').read_text())
assert digest(SOURCE)==prior['source_model_sha256']
pins=[SOURCE,PRIOR/'manifest.json',PRIOR/'boundary-attachment.json',PRIOR/'ring-0.json',
      ROOT/'outputs/panda-cheek-source-normal-payload-v1/payload.json']
write('inputs.json',[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)}for p in pins])
raw=SOURCE.read_bytes();assert len(raw)<64*1024**2
size=struct.unpack_from('<I',raw,12)[0];doc=json.loads(raw[20:20+size]);binary=memoryview(raw)[28+size:]


def array(index):
    a=doc['accessors'][index];v=doc['bufferViews'][a['bufferView']]
    assert 'sparse'not in a and a['count']<=300000
    dtype={5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
    width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']];item=np.dtype(dtype).itemsize
    return np.ndarray((a['count'],width),dtype=dtype,buffer=binary,
        offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',item*width),item))


points=[];normals=[];faces=[];uvs=[];source_ids=[];offset=0
for pi in(0,2):
    prim=doc['meshes'][0]['primitives'][pi]
    assert doc['materials'][prim['material']]['name']in('body_surface','body_cap')
    attrs=prim['attributes'];p=array(attrs['POSITION']);f=array(prim['indices']).reshape(-1,3)
    points.append(p.copy());normals.append(array(attrs['NORMAL']).copy());uvs.append(array(attrs['TEXCOORD_0']).copy())
    faces.append(f.astype(np.int32)+offset);offset+=len(p);source_ids.extend((pi,int(i))for i in range(len(f)))
points=np.concatenate(points);normals=np.concatenate(normals);uvs=np.concatenate(uvs);faces=np.concatenate(faces)
assert len(points)<=100000 and len(faces)<=100000
unique,inverse=np.unique(points,axis=0,return_inverse=True);wf=inverse[faces]
vertex_faces=defaultdict(list);edge_faces=defaultdict(list);edge_directed=defaultdict(list)
for fid,tri in enumerate(wf):
    for v in tri:vertex_faces[int(v)].append(fid)
    for a,b in zip(tri,np.roll(tri,-1)):
        a,b=int(a),int(b);edge=tuple(sorted((a,b)));edge_faces[edge].append(fid);edge_directed[edge].append((fid,a,b))
adj=[set()for _ in faces]
for incident in edge_faces.values():
    for fid in incident:adj[fid].update(x for x in incident if x!=fid)
lookup={key:i for i,key in enumerate(source_ids)}
seed={lookup[tuple(x)]for x in prior['components'][0]['source_face_ids']}
assert len(seed)==124
problem_vertices={int(x['body_welded_vertex'])for x in prior['components'][0]['boundary_problem_vertices']}
assert problem_vertices=={2573,3009,3091}


def link(v,incident):
    neighbors=defaultdict(set);edges=[]
    for fid in incident:
        a,b=[int(x)for x in wf[fid]if x!=v]
        neighbors[a].add(b);neighbors[b].add(a);edges.append(tuple(sorted((a,b))))
    components=groups(neighbors,neighbors);degrees=[len(x)for x in neighbors.values()]
    interval=degrees.count(1)==2 and all(x in(1,2)for x in degrees)
    circle=bool(degrees)and all(x==2 for x in degrees)
    return {'healthy':len(components)==1 and len(edges)==len(set(edges))and(interval or circle),
            'link_components':components,'link_edges':[list(x)for x in edges],
            'degree_histogram':dict(Counter(degrees))}


health_cache={}


def healthy(v):
    if v not in health_cache:health_cache[v]=link(v,vertex_faces[v])['healthy']
    return health_cache[v]


def blender(a):
    b=np.asarray(a,dtype=float)[:,[0,2,1]].copy();b[:,1]*=-1;return b


junctions=[]
for v in sorted(problem_vertices):
    incident=vertex_faces[v]
    junctions.append({'body_welded_vertex':v,'position_blender_m':blender(unique[[v]])[0].tolist(),
        'whole_body_link':link(v,incident),
        'incident_faces':[{'source_face':list(source_ids[f]),'selected':f in seed,
             'body_welded_vertices':wf[f].tolist(),'rest_positions_blender_m':blender(points[faces[f]]).tolist()}
             for f in incident]})
write('junction-fans.json',{'source_sha256':digest(SOURCE),'junctions':junctions,
    'scope':'Exact body-only position aliases; no companion vertices or faces participate'})


def screen(selected, deleted=frozenset()):
    assert time.monotonic()-START<25
    local_edges=defaultdict(list);local_vertices=defaultdict(list)
    for fid in selected:
        for v in wf[fid]:local_vertices[int(v)].append(fid)
        for a,b in zip(wf[fid],np.roll(wf[fid],-1)):local_edges[tuple(sorted((int(a),int(b))))].append(fid)
    boundary=[e for e,fs in local_edges.items()if len(fs)==1]
    boundary_graph=defaultdict(set)
    for a,b in boundary:boundary_graph[a].add(b);boundary_graph[b].add(a)
    boundary_groups=groups(boundary_graph,boundary_graph)
    face_groups=groups(selected,adj)
    invalid_selected=[v for v,fs in local_vertices.items()if not link(v,fs)['healthy']]
    invalid_outer=[v for v in boundary_graph if not link(v,[f for f in vertex_faces[v]if f not in selected and f not in deleted])['healthy']]
    interior_external=[v for v in local_vertices if v not in boundary_graph and any(f not in selected and f not in deleted for f in vertex_faces[v])]
    orientation=[]
    for e,fs in local_edges.items():
        if len(fs)==2:
            directed=[(a,b)for fid,a,b in edge_directed[e]if fid in selected]
            if directed[0]==directed[1]:orientation.append(e)
    attachment_edges=[e for e in boundary if len([f for f in edge_faces[e]if f not in selected and f not in deleted])!=1]
    chi=len(local_vertices)-len(local_edges)+len(selected)
    ok=(0<len(selected)<=1000 and len(face_groups)==1 and len(boundary_groups)==1 and chi==1
        and all(len(fs)<=2 for fs in local_edges.values())and all(len(ns)==2 for ns in boundary_graph.values())
        and not invalid_selected and not invalid_outer and not interior_external and not orientation and not attachment_edges)
    return {'passes':ok,'faces':len(selected),'face_components':list(map(len,face_groups)),
        'boundary_components':list(map(len,boundary_groups)),'boundary_edges':len(boundary),
        'boundary_degree_histogram':dict(Counter(map(len,boundary_graph.values()))),
        'edge_valence_histogram':dict(Counter(map(len,local_edges.values()))),'euler_characteristic':chi,
        'invalid_selected_vertex_links':invalid_selected,'invalid_outer_attachment_vertices':invalid_outer,
        'interior_vertices_with_unselected_body_faces':interior_external,
        'orientation_conflict_edges':[list(e)for e in orientation],
        'boundary_edges_without_exactly_one_retained_body_face':[list(e)for e in attachment_edges]},boundary


REF=ROOT/'outputs/panda-cheek-reference-field-v1'
ref_report=json.loads((REF/'support-reconciliation.json').read_text())
diagnostics=json.loads((REF/'face-diagnostics.json').read_text())
field=json.loads((REF/'proposed-reference-field.json').read_text())['field']
old_fit=json.loads((ROOT/'outputs/panda-cheek-curved-candidate-v1/fit-report.json').read_text())
parameters=json.loads((OUT/'parameters.json').read_text())
extra_inputs=[REF/'manifest.json',REF/'support-reconciliation.json',REF/'face-diagnostics.json',
              REF/'proposed-reference-field.json',ROOT/'outputs/panda-cheek-curved-candidate-v1/fit-report.json']
write('extra-inputs.json',[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)}for p in extra_inputs])
by_id={(int(r['source_primitive']),int(r['source_triangle'])):r for r in diagnostics}
witness_ids={(0,int(r['source_triangle']))for r in ref_report['supported_outlier_faces']}
require_witnesses=len(witness_ids)==24
assert require_witnesses
witnesses={lookup[x]for x in witness_ids}
retired={lookup[(0,4961)]}
protected_ids={(0,int(r['source_triangle']))for r in field}
for component in prior['components'][1:]:protected_ids.update(map(tuple,component['source_face_ids']))
protected={lookup[x]for x in protected_ids}
protected_vertices={int(v)for f in protected for v in wf[f]}
assert not(seed|witnesses|retired)&protected
allowed={lookup[key]for key,r in by_id.items()
    if r['maximum_original_body_edge_dihedral_degrees']>parameters['scar_dihedral_degrees']
    and r['contact_interface_edge_distance']<=parameters['maximum_contact_edge_distance']
    and lookup[key]not in protected|retired}
selected=set(seed)|witnesses;parent={f:None for f in selected};queue=deque(sorted(selected))
while queue:
    f=queue.popleft()
    for other in sorted(adj[f]):
        if other in allowed and other not in selected:
            selected.add(other);parent[other]=f;queue.append(other)
assert len(selected)<=parameters['max_replacement_faces'],'Targeted scar ownership exceeds512faces'
selection=[]
for f in sorted(selected):
    key=source_ids[f];r=by_id.get(key)
    reason='existing_loop27_patch'if f in seed else 'exact_supported_reference_outlier'if f in witnesses else 'connected_high_dihedral_contact_face'
    selection.append({'source_face':list(key),'reason':reason,
        'connected_from_source_face':list(source_ids[parent[f]])if parent.get(f)is not None else None,
        'partition_original_face_id':r['partition_original_face_id']if r else None,
        'contact_interface_edge_distance':r['contact_interface_edge_distance']if r else None,
        'original_body_dihedral_degrees':r['maximum_original_body_edge_dihedral_degrees']if r else None})
write('selection.json',{'source_model_sha256':digest(SOURCE),'replacement':selection,
    'retired_without_replacement':[[0,4961]],'protected_source_faces':[list(x)for x in sorted(protected_ids)],
    'held_boundaries':'Both other ring0 cheek disks and44 cleaner reference faces are protected; any shared vertex must remain on the proposed fixed boundary',
    'parameters':parameters,'source_model_modified':False})


def normal_angle(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))


cross=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
areas=np.linalg.norm(cross,axis=1);geometric=cross/np.maximum(areas[:,None],1e-30)
normal_basis=np.asarray(old_fit['basis_blender_columns']);normal_origin=np.asarray(old_fit['origin_blender_m']);coeff=np.asarray(old_fit['quadratic_coefficients'])


def boundary_support(chosen,boundary):
    removed=set(chosen)|retired
    boundary_vertices=sorted({int(v)for edge in boundary for v in edge});outside_faces=set();edge_issues=[]
    for edge in boundary:
        remaining=[f for f in edge_faces[edge]if f not in removed]
        outside_faces.update(remaining)
        if len(remaining)!=1:edge_issues.append({'body_edge':list(edge),'retained_source_faces':[list(source_ids[f])for f in remaining]})
    face_rows=[]
    for f in sorted(outside_faces):
        key=source_ids[f];row=by_id.get(key);reasons=[]
        if key[0]!=0:reasons.append('retained_authored_or_repainted_cap_not_original_reference')
        if row is None:reasons.append('outside_measured_reference_only_band')
        elif row['rejected_reasons']:reasons.extend(row['rejected_reasons'])
        face_rows.append({'source_face':list(key),'reference_support_rejections':reasons,
            'known_cleaner_reference_face':key in protected_ids and key[0]==0})
    vertex_rows=[]
    for v in boundary_vertices:
        contributions=[]
        for f in vertex_faces[v]:
            if f in removed:continue
            for corner,vertex in enumerate(wf[f]):
                if int(vertex)==v:contributions.append(normals[faces[f,corner]])
        if not contributions:
            vertex_rows.append({'body_welded_vertex':v,'normal_supported':False});continue
        mean=np.sum(contributions,axis=0);mean/=np.linalg.norm(mean)
        position=blender(unique[[v]])[0];p=(position-normal_origin)@normal_basis/old_fit['normalization_scale_m'];x,y=p[:2]
        n=np.asarray([-(coeff[1]+2*coeff[3]*x+coeff[4]*y),-(coeff[2]+coeff[4]*x+2*coeff[5]*y),1.])@normal_basis.T
        vertex_rows.append({'body_welded_vertex':v,'position_blender_m':position.tolist(),'normal_supported':True,
            'retained_custom_mean_vs_frozen_fit_degrees':normal_angle(blender(mean[None,:])[0],n)})
    valid=[r['retained_custom_mean_vs_frozen_fit_degrees']for r in vertex_rows if r['normal_supported']]
    intrusions=sorted((set(int(v)for f in chosen for v in wf[f])&protected_vertices)-set(boundary_vertices))
    return {'boundary_vertices':len(boundary_vertices),'outside_faces':face_rows,'edge_attachment_issues':edge_issues,
        'unsupported_retained_boundary_faces':sum(bool(r['reference_support_rejections'])for r in face_rows),
        'protected_vertices_that_would_become_interior':intrusions,
        'retained_custom_mean_vs_frozen_fit_degrees':{'count':len(valid),'median':float(np.median(valid))if valid else None,
            'p95':float(np.percentile(valid,95))if valid else None,'max':max(valid)if valid else None},
        'vertex_rows':vertex_rows,
        'smooth_join_evidence_passes':not edge_issues and not intrusions and all(not r['reference_support_rejections']for r in face_rows),
        'scope':'Support uses provenance/crease rules only. Frozen failed-fit comparison is diagnostic, not a new fit or altered45-degree fit gate.'}


variants=[]
for name,chosen in [('exact24_outlier_addition',seed|witnesses),('connected_provenance_scar_selection',selected)]:
    outcome,boundary=screen(chosen,retired)
    support=boundary_support(chosen,boundary)
    variants.append({'name':name,'topology':outcome,'boundary_support':support,
        'candidate_supported':outcome['passes']and support['smooth_join_evidence_passes']})
write('variant-results.json',variants)
accepted=variants[-1]['candidate_supported']
result={'schema_version':1,'source_model_sha256':digest(SOURCE),
    'status':'supported_reference_domain_only'if accepted else 'unsupported_stop_branch',
    'initial_patch_faces':len(seed),'exact_outlier_faces':len(witnesses),'allowed_high_dihedral_contact_faces':len(allowed),
    'connected_added_scar_faces':len(selected-seed-witnesses),'replacement_faces':len(selected),'retired_original_faces':1,
    'total_owned_face_count':len(selected|retired),'protected_face_count':len(protected),
    'protected_source_faces_unchanged':not selected&protected,
    'last_topology':variants[-1]['topology'],
    'unsupported_retained_boundary_faces':variants[-1]['boundary_support']['unsupported_retained_boundary_faces'],
    'boundary_normal_error_to_frozen_fit':variants[-1]['boundary_support']['retained_custom_mean_vs_frozen_fit_degrees'],
    'model_modified':False,'new_fit_performed':False,'native_jobs':0,'provider_calls':0,'quality_approved':False,
    'elapsed_seconds':time.monotonic()-START,
    'next_action':'Report domain and boundary evidence before any new fit.'if accepted else
        'Stop this local repair branch under the fixed512-face/protected-boundary envelope; do not run another fit or native preview. Exact failed topology/retained-normal support is preserved.',
    'sources_unchanged':all(digest(ROOT/r['path'])==r['sha256']for r in json.loads((OUT/'inputs.json').read_text())+json.loads((OUT/'extra-inputs.json').read_text()))}
write('report.json',result);print(json.dumps(result,sort_keys=True))
