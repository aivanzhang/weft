"""Check remaining body-side links at the three ring-zero candidate boundaries."""
from collections import defaultdict, deque, Counter
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'outputs/companion-separation-v7/diagnostic-model.glb'
report=json.loads((OUT/'report.json').read_text())
raw=SOURCE.read_bytes();assert hashlib.sha256(raw).hexdigest()==report['source_model_sha256']
size=struct.unpack_from('<I',raw,12)[0];doc=json.loads(raw[20:20+size]);binary=memoryview(raw)[28+size:]


def array(index):
    a=doc['accessors'][index];v=doc['bufferViews'][a['bufferView']]
    dtype={5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
    width={'SCALAR':1,'VEC3':3}[a['type']];item=np.dtype(dtype).itemsize
    return np.ndarray((a['count'],width),dtype=dtype,buffer=binary,
        offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',width*item),item))


points=[];faces=[];sources=[];offset=0
for pi in(0,2):
    primitive=doc['meshes'][0]['primitives'][pi];p=array(primitive['attributes']['POSITION']);f=array(primitive['indices']).reshape(-1,3)
    points.append(p.copy());faces.append(f.astype(np.int32)+offset);offset+=len(p)
    sources.extend((pi,int(i))for i in range(len(f)))
points=np.concatenate(points);faces=np.concatenate(faces)
unique,inverse=np.unique(points,axis=0,return_inverse=True);welded=inverse[faces]
vertex_faces=defaultdict(list)
for fid,tri in enumerate(welded):
    for vertex in tri:vertex_faces[int(vertex)].append(fid)
source_to_face={v:i for i,v in enumerate(sources)}
seed_payload=json.loads((ROOT/'outputs/panda-cheek-source-normal-payload-v1/payload.json').read_text())
provenance={(r['source_primitive'],r['source_triangle']):r for r in seed_payload['faces']}


def link_health(vertex,incident):
    neighbors=defaultdict(set);edges=[]
    for fid in incident:
        a,b=[int(x)for x in welded[fid]if x!=vertex]
        neighbors[a].add(b);neighbors[b].add(a);edges.append(tuple(sorted((a,b))))
    remaining=set(neighbors);count=0
    for start in sorted(remaining):
        if start not in remaining:continue
        count+=1;remaining.remove(start);queue=deque([start])
        while queue:
            item=queue.popleft()
            for other in neighbors[item]:
                if other in remaining:remaining.remove(other);queue.append(other)
    degrees=[len(x)for x in neighbors.values()]
    circle=bool(degrees)and all(x==2 for x in degrees)
    interval=degrees.count(1)==2 and all(x in(1,2)for x in degrees)
    duplicates=len(edges)-len(set(edges))
    return {'healthy_one_fan':count==1 and not duplicates and(circle or interval),
        'components':count,'degree_histogram':dict(Counter(degrees)),'duplicate_link_edges':duplicates,
        'incident_source_faces':[list(sources[fid])for fid in incident]}


rows=[]
ring=json.loads((OUT/'ring-0.json').read_text())
for component_index,component in enumerate(ring['components']):
    selected={source_to_face[tuple(x)]for x in component['source_face_ids']}
    boundary_edges=component['boundary_welded_body_edges'];boundary_vertices=sorted({v for edge in boundary_edges for v in edge})
    neighbors=defaultdict(list)
    for a,b in boundary_edges:neighbors[a].append(b);neighbors[b].append(a)
    assert all(len(x)==2 for x in neighbors.values())
    first=min(boundary_vertices);cycle=[first];previous=None;current=first
    while True:
        choices=sorted(x for x in neighbors[current]if x!=previous)
        nxt=choices[0]
        if nxt==first:break
        assert nxt not in cycle
        cycle.append(nxt);previous,current=current,nxt
    assert len(cycle)==len(boundary_vertices)
    checks=[];references=set()
    for vertex in boundary_vertices:
        all_faces=vertex_faces[vertex];remaining=[fid for fid in all_faces if fid not in selected]
        references.update(remaining)
        whole=link_health(vertex,all_faces);outside=link_health(vertex,remaining)
        if not whole['healthy_one_fan']or not outside['healthy_one_fan']:
            checks.append({'body_welded_vertex':vertex,'whole_body':whole,'remaining_body':outside})
    bp=unique[cycle][:,[0,2,1]].copy();bp[:,1]*=-1
    kinds=Counter(provenance[tuple(x)]['kind']for x in component['source_face_ids'])
    loops=sorted({provenance[tuple(x)]['closure_loop']for x in component['source_face_ids']if provenance[tuple(x)]['closure_loop']is not None})
    rows.append({'component':component_index,'faces':component['faces'],'origin_counts':dict(kinds),'closure_loops':loops,
        'selected_disk_passes':component['single_manifold_disk'],
        'healthy_attachment_boundary':not checks,'boundary_problem_vertices':checks,
        'ordered_boundary_body_welded_vertex_ids':cycle,'ordered_boundary_blender_positions_m':bp.tolist(),
        'boundary_order':'Deterministic undirected cycle, not an authored winding or new triangulation',
        'reference_body_source_faces':[list(sources[fid])for fid in sorted(references)],
        'source_face_ids':component['source_face_ids']})
value={'schema_version':1,'source_model_sha256':report['source_model_sha256'],'ring':0,
    'status':'screen_only_not_replacement','components':rows,
    'scope':'Original v7 body only; checks selected disk and remaining body vertex links; companion never merged',
    'limitation':'Boundary health is a necessary geometric/topological screen, not fit/intersection/motion/appearance proof',
    'quality_approved':False,'model_mutated':False,'native_jobs':0,'provider_calls':0}
with(OUT/'boundary-attachment.json').open('x')as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
print(json.dumps([{'component':r['component'],'healthy_attachment_boundary':r['healthy_attachment_boundary'],
                  'problem_vertices':len(r['boundary_problem_vertices'])}for r in rows]))
