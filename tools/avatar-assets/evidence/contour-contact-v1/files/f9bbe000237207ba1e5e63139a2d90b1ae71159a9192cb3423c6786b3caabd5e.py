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


spur=lookup[(0,4961)]
spur_tri=wf[spur]
spur_edges=[tuple(sorted((int(a),int(b))))for a,b in zip(spur_tri,np.roll(spur_tri,-1))]
spur_counts=[len(edge_faces[e])for e in spur_edges]
spur_points=points[faces[spur]].astype(float)
spur_area=float(np.linalg.norm(np.cross(spur_points[1]-spur_points[0],spur_points[2]-spur_points[0]))*.5)
edge_lengths=np.linalg.norm(spur_points-np.roll(spur_points,-1,axis=0),axis=1)
spur_info={'source_face':[0,4961],'body_welded_vertices':spur_tri.tolist(),
    'positions_blender_m':blender(spur_points).tolist(),'area_m2':spur_area,
    'edge_lengths_m':edge_lengths.tolist(),'edge_total_body_incidence':spur_counts,
    'incident_source_faces_by_edge':[[list(source_ids[f])for f in edge_faces[e]]for e in spur_edges],
    'vertex_total_body_incidence':[len(vertex_faces[int(v)])for v in spur_tri],
    'one_triangle_attached_along_only_one_edge':sorted(spur_counts)==[1,1,3],
    'claim':'Actual source triangle, not a zero-area or duplicate-face assumption'}
write('spur.json',spur_info)

fan_neighbors=set(vertex_faces[2573])-seed-{spur}
assert len(fan_neighbors)<=8
config={'source_model_sha256':digest(SOURCE),'seed_faces':124,'seed_source_faces':[list(source_ids[f])for f in sorted(seed)],
    'explicit_original_spur_face':[0,4961],
    'optional_incident_original_faces':[list(source_ids[f])for f in sorted(fan_neighbors)],
    'protected_primitives':[1,3],'max_added_original_faces':8,'max_target_faces':1000,
    'post_removal_gate':'Selected disk with simple oriented boundary; every boundary edge has exactly one retained body face; remaining-body vertex links healthy. Original cap diagonals are removed and need not remain manifold before removal.',
    'two_scopes':'First test a single connected replacement selection including the spur. If unsupported, separately screen explicit leaf-face retirement plus the unchanged disk. These are not silently equivalent.'}
write('config.json',config)
trials=[];integrated=None
for count in range(len(fan_neighbors)+1):
    for added in itertools.combinations(sorted(fan_neighbors),count):
        candidate=seed|{spur}|set(added)
        outcome,boundary=screen(candidate)
        trials.append({'variant':'single_replacement_selection_including_spur',
            'added_source_faces':[[0,4961]]+[list(source_ids[f])for f in added], 'screen':outcome})
        if outcome['passes']:integrated=(candidate,boundary);break
    if integrated:break

isolated_result,isolated_boundary=screen(seed,{spur})
trials.append({'variant':'separate_leaf_retirement_plus_original_124_face_disk','screen':isolated_result,
    'retired_source_faces':[[0,4961]],'leaf_gate':spur_info['one_triangle_attached_along_only_one_edge']})
write('trials.json',trials)
winner=integrated
mode='single_replacement_selection'if integrated else 'separate_leaf_retirement_plus_disk'
if winner is None and isolated_result['passes']and spur_info['one_triangle_attached_along_only_one_edge']:
    winner=(seed,isolated_boundary)
result={'schema_version':1,'source_model_sha256':digest(SOURCE),'config_sha256':digest(OUT/'config.json'),
    'status':'unsupported'if winner is None else 'bounded_composite_candidate_not_implemented',
    'tests':len(trials),'single_selection_with_spur_passed':integrated is not None,
    'separate_leaf_retirement_disk_screen_passed':isolated_result['passes'],
    'spur_leaf_gate':spur_info['one_triangle_attached_along_only_one_edge'],
    'native_jobs':0,'provider_calls':0,'textures_decoded':0,'model_modified':False,'quality_approved':False}
if winner is not None:
    selected,boundary=winner;deleted=set()if integrated else {spur};edges=set(boundary);directed={}
    for fid in selected:
        for a,b in zip(wf[fid],np.roll(wf[fid],-1)):
            a,b=int(a),int(b)
            if tuple(sorted((a,b)))in edges:
                assert a not in directed;directed[a]=b
    first=min(directed);cycle=[first];current=directed[first]
    while current!=first:
        assert current not in cycle;cycle.append(current);current=directed[current]
    assert len(cycle)==len(boundary)
    references={f for v in cycle for f in vertex_faces[v]if f not in selected and f not in deleted and source_ids[f][0]==0}
    forbidden_diagonals=[]
    for edge,incident in edge_faces.items():
        if set(edge)<=set(cycle)and edge not in edges and any(f in selected for f in incident)and any(f not in selected and f not in deleted for f in incident):
            forbidden_diagonals.append({'body_welded_vertices':list(edge),
                'retained_source_faces':[list(source_ids[f])for f in incident if f not in selected and f not in deleted],
                'removed_source_faces':[list(source_ids[f])for f in incident if f in selected]})
    refs=[]
    for fid in sorted(references):
        tri=faces[fid]
        refs.append({'source_face':list(source_ids[fid]),'rest_positions_blender_m':blender(points[tri]).tolist(),
            'corner_normals_blender':blender(normals[tri]).tolist(),'uv0_gltf_v_down':uvs[tri].tolist()})
    candidate={'source_model_sha256':digest(SOURCE),'mode':mode,'source_mesh':0,
        'replacement_source_faces':[list(source_ids[f])for f in sorted(selected)],
        'retired_without_replacement_source_faces':[list(source_ids[f])for f in sorted(deleted)],
        'source_files_mutated':False,'oriented_boundary_body_welded_ids':cycle,
        'oriented_boundary_blender_m':blender(unique[cycle]).tolist(),
        'boundary_winding':'Same directed order as selected source triangles; preserve orientation in any new patch',
        'healthy_remaining_original_body_reference_triangles':refs,
        'forbidden_existing_body_diagonals':forbidden_diagonals,
        'scope':'Hypothetical local composite repair data only. Retirement deletes an explicit tiny source flap in a future derivative; it is not a source-preserving no-op.',
        'screen':screen(selected,deleted)[0]}
    write('candidate.json',candidate)
    result.update(candidate_sha256=digest(OUT/'candidate.json'),mode=mode,replacement_faces=len(selected),
        retired_without_replacement_faces=len(deleted),oriented_boundary_vertices=len(cycle),
        reference_faces=len(refs),forbidden_shared_body_diagonals=len(forbidden_diagonals))
else:
    result['next_dependency']='Explicit local face-fan reconstruction is still required; this envelope did not establish a healthy replacement disk.'
result.update(elapsed_seconds=time.monotonic()-START,
    sources_unchanged=all(digest(ROOT/p['path'])==p['sha256']for p in json.loads((OUT/'inputs.json').read_text())))
write('report.json',result)
print(json.dumps(result,sort_keys=True))
