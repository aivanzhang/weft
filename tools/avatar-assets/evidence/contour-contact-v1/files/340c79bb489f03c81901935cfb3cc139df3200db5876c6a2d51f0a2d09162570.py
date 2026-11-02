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


def screen(selected):
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
    invalid_outer=[v for v in boundary_graph if not healthy(v)or
                   not link(v,[f for f in vertex_faces[v]if f not in selected])['healthy']]
    interior_external=[v for v in local_vertices if v not in boundary_graph and any(f not in selected for f in vertex_faces[v])]
    orientation=[]
    for e,fs in local_edges.items():
        if len(fs)==2:
            directed=[(a,b)for fid,a,b in edge_directed[e]if fid in selected]
            if directed[0]==directed[1]:orientation.append(e)
    chi=len(local_vertices)-len(local_edges)+len(selected)
    ok=(0<len(selected)<=1000 and len(face_groups)==1 and len(boundary_groups)==1 and chi==1
        and all(len(fs)<=2 for fs in local_edges.values())and all(len(ns)==2 for ns in boundary_graph.values())
        and not invalid_selected and not invalid_outer and not interior_external and not orientation)
    return {'passes':ok,'faces':len(selected),'face_components':list(map(len,face_groups)),
        'boundary_components':list(map(len,boundary_groups)),'boundary_edges':len(boundary),
        'boundary_degree_histogram':dict(Counter(map(len,boundary_graph.values()))),
        'edge_valence_histogram':dict(Counter(map(len,local_edges.values()))),'euler_characteristic':chi,
        'invalid_selected_vertex_links':invalid_selected,'invalid_outer_attachment_vertices':invalid_outer,
        'interior_vertices_with_unselected_body_faces':interior_external,
        'orientation_conflict_edges':[list(e)for e in orientation]},boundary


fan_faces={f for v in problem_vertices for f in vertex_faces[v]}
mandatory_remove=seed&fan_faces
assert len(mandatory_remove)<=16
core=seed-mandatory_remove
# Excluding a bad boundary junction is allowed only as an explicit selection change, not a global repair.
pool={f for fid in fan_faces for f in adj[fid]if f not in seed and not(problem_vertices&set(wf[f]))
      and all(healthy(int(v))for v in wf[f])}
assert len(pool)<=40
configuration={'seed_component':0,'seed_faces':124,'protected_primitives':[1,3],
    'mandatory_remove_source_faces':[list(source_ids[f])for f in sorted(mandatory_remove)],
    'addition_pool_source_faces':[list(source_ids[f])for f in sorted(pool)],
    'max_added_faces':2,'max_candidate_faces':1000,'max_candidate_tests':1024,
    'selection_scope':'Remove only seed faces incident on the three known bad junctions, then consider at most two healthy faces adjacent to those exact fans',
    'invariant':'No tested face includes any of the three original bad vertices; all remaining seed faces stay required',
    'minimality_scope':'Mandatory removals are necessary under the stated healthy-global-vertex and no-interior-external-attachment constraints; optional additions tested in cardinality order',
    'limitation':'Moving the patch boundary off inherited bad junctions does not repair those junctions in the untouched original body'}
write('config.json',configuration)
trials=[];winner=None;winning_boundary=None
initial,_=screen(seed);trials.append({'variant':'original_seed','screen':initial})
for count in range(3):
    for added in itertools.combinations(sorted(pool),count):
        assert len(trials)<=1024
        selected=core|set(added);outcome,boundary=screen(selected)
        trials.append({'variant':'targeted_fan_avoidance','added_source_faces':[list(source_ids[f])for f in added],
                       'screen':outcome})
        if outcome['passes']:
            winner=selected;winning_boundary=boundary;break
    if winner is not None:break
write('trials.json',trials)
result={'schema_version':1,'status':'unsupported'if winner is None else 'boundary_candidate_only',
    'source_model_sha256':digest(SOURCE),'config_sha256':digest(OUT/'config.json'),
    'tests':len(trials),'mandatory_removed_faces':len(mandatory_remove),'addition_pool_faces':len(pool),
    'native_jobs':0,'provider_calls':0,'textures_decoded':0,'model_modified':False,'quality_approved':False,
    'elapsed_seconds':time.monotonic()-START,'original_junction_geometry_repaired':False,
    'scope':'Local selection screen only; global original nonmanifold geometry remains unchanged'}
if winner is not None:
    edges=set(winning_boundary);directed={}
    for fid in winner:
        for a,b in zip(wf[fid],np.roll(wf[fid],-1)):
            a,b=int(a),int(b)
            if tuple(sorted((a,b)))in edges:
                assert a not in directed;directed[a]=b
    first=min(directed);cycle=[first];current=directed[first]
    while current!=first:
        assert current not in cycle;cycle.append(current);current=directed[current]
    assert len(cycle)==len(winning_boundary)
    references={f for v in cycle for f in vertex_faces[v]if f not in winner and source_ids[f][0]==0
                and all(healthy(int(x))for x in wf[f])}
    refs=[]
    for fid in sorted(references):
        tri=faces[fid]
        refs.append({'source_face':list(source_ids[fid]),'rest_positions_blender_m':blender(points[tri]).tolist(),
            'corner_normals_blender':blender(normals[tri]).tolist(),'uv0_gltf_v_down':uvs[tri].tolist()})
    candidate={'source_model_sha256':digest(SOURCE),'source_mesh':0,
        'source_face_ids':[list(source_ids[f])for f in sorted(winner)],
        'removed_from_selection':[list(source_ids[f])for f in sorted(seed-winner)],
        'added_to_selection':[list(source_ids[f])for f in sorted(winner-seed)],
        'oriented_boundary_body_welded_ids':cycle,'oriented_boundary_blender_m':blender(unique[cycle]).tolist(),
        'boundary_winding':'Same directed edge order as selected oriented source triangles; a future replacement must preserve this orientation',
        'healthy_original_body_reference_triangles':refs,
        'protected_bad_junction_positions_blender_m':[r['position_blender_m']for r in junctions],
        'screen':trials[-1]['screen'],
        'caveat':'The three inherited bad junctions and removed seed triangles remain outside this candidate. This is not a topology repair of those original faces.'}
    write('candidate.json',candidate)
    result.update(candidate_faces=len(winner),added_faces=len(winner-seed),boundary_vertices=len(cycle),
        healthy_reference_faces=len(refs),candidate_sha256=digest(OUT/'candidate.json'))
else:
    result['next_dependency']='Resolve the explicit incident face-fan ownership at the three saved body junctions; no valid selection under this local envelope. No projected-cap or whole-mesh remesh fallback.'
result['sources_unchanged']=all(digest(ROOT/p['path'])==p['sha256']for p in json.loads((OUT/'inputs.json').read_text()))
write('report.json',result)
print(json.dumps(result,sort_keys=True))
