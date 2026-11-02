"""Body-only exact-position topology screen; no repair, rendering or texture decoding."""
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
START=time.monotonic()
SOURCE=ROOT/'outputs/companion-separation-v7/diagnostic-model.glb'
PAYLOAD=ROOT/'outputs/panda-cheek-source-normal-payload-v1/payload.json'
MAX_FACES=100000


def digest(path):
    with path.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(name,value):
    with(OUT/name).open('x')as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def components(ids,neighbors):
    remaining=set(map(int,ids));groups=[]
    for start in sorted(remaining):
        if start not in remaining:continue
        remaining.remove(start);queue=deque([start]);group=[]
        while queue:
            item=queue.popleft();group.append(item)
            for other in neighbors[item]:
                if other in remaining:remaining.remove(other);queue.append(other)
        groups.append(sorted(group))
    return sorted(groups,key=lambda x:(-len(x),x[0]))


assert SOURCE.stat().st_size<=64*1024**2
payload=json.loads(PAYLOAD.read_text())
assert digest(SOURCE)==payload['source_model_sha256']
pins=[SOURCE,PAYLOAD,ROOT/'outputs/panda-contact-defects-diagnosis-v1/manifest.json',
      ROOT/'outputs/panda-cheek-normal-ablation-v1/normal-ablation.json']
write('inputs.json',[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)}for p in pins])
raw=SOURCE.read_bytes();size=struct.unpack_from('<I',raw,12)[0]
assert size<=2*1024**2
doc=json.loads(raw[20:20+size]);binary=memoryview(raw)[28+size:]


def array(index):
    a=doc['accessors'][index];v=doc['bufferViews'][a['bufferView']]
    assert 'sparse'not in a and a['count']<=300000
    dtype={5121:'<u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
    width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']];item=np.dtype(dtype).itemsize
    return np.ndarray((a['count'],width),dtype=dtype,buffer=binary,
        offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',width*item),item))


points=[];faces=[];face_sources=[];offset=0;protected=[]
for pi,primitive in enumerate(doc['meshes'][0]['primitives']):
    name=doc['materials'][primitive['material']]['name']
    if name not in {'body_surface','body_cap'}:
        protected.append({'primitive':pi,'material':name,'triangles':doc['accessors'][primitive['indices']]['count']//3})
        continue
    p=array(primitive['attributes']['POSITION']);f=array(primitive['indices']).reshape(-1,3)
    points.append(p.copy());faces.append(f.astype(np.int32)+offset);offset+=len(p)
    face_sources.extend((pi,int(i))for i in range(len(f)))
points=np.concatenate(points);faces=np.concatenate(faces)
assert len(points)<=100000 and len(faces)<=MAX_FACES
# Exact floats only; UV seams are analytical aliases, not a mesh mutation.
unique,inverse=np.unique(points,axis=0,return_inverse=True)
welded=inverse[faces]
assert np.all(welded[:,0]!=welded[:,1])and np.all(welded[:,1]!=welded[:,2])and np.all(welded[:,2]!=welded[:,0])
by_source={tuple(source):i for i,source in enumerate(face_sources)}
seeds={by_source[(r['source_primitive'],r['source_triangle'])]for r in payload['faces']}
assert len(seeds)==198 and len(payload['faces'])==198
edge_faces=defaultdict(list);edge_directions=defaultdict(list)
for fid,tri in enumerate(welded):
    a,b,c=map(int,tri)
    for x,y in ((a,b),(b,c),(c,a)):
        edge=tuple(sorted((x,y)));edge_faces[edge].append(fid);edge_directions[edge].append((fid,x,y))
adj=[set()for _ in faces]
for incident in edge_faces.values():
    for fid in incident:adj[fid].update(other for other in incident if other!=fid)


def analyze_patch(ids):
    selected=set(ids);local_edges=defaultdict(list);vertex_faces=defaultdict(list)
    for fid in ids:
        tri=welded[fid]
        for vertex in tri:vertex_faces[int(vertex)].append(fid)
        for x,y in zip(tri,np.roll(tri,-1)):local_edges[tuple(sorted((int(x),int(y))))].append(fid)
    boundary=[edge for edge,incident in local_edges.items()if len(incident)==1]
    boundary_graph=defaultdict(set)
    for a,b in boundary:boundary_graph[a].add(b);boundary_graph[b].add(a)
    boundary_components=components(boundary_graph,boundary_graph)
    boundary_bad={str(v):len(n)for v,n in boundary_graph.items()if len(n)!=2}
    pinches=[]
    for vertex,incident in vertex_faces.items():
        link=defaultdict(set);link_edges=[]
        for fid in incident:
            other=[int(x)for x in welded[fid]if int(x)!=vertex]
            assert len(other)==2
            a,b=other;link[a].add(b);link[b].add(a);link_edges.append(tuple(sorted(other)))
        groups=components(link,link)
        degrees=[len(n)for n in link.values()]
        interval=degrees.count(1)==2 and all(n in(1,2)for n in degrees)
        circle=all(n==2 for n in degrees)
        duplicate_links=len(link_edges)-len(set(link_edges))
        if len(groups)!=1 or not(interval or circle) or duplicate_links:
            pinches.append({'welded_body_vertex':vertex,'fan_components':len(groups),'link_degree_histogram':dict(Counter(degrees)),
                'duplicate_link_edges':duplicate_links,'incident_source_faces':[list(face_sources[fid])for fid in incident]})
    orientation=[]
    for edge,incident in local_edges.items():
        if len(incident)!=2:continue
        dirs=[(a,b)for fid,a,b in edge_directions[edge]if fid in selected]
        if dirs[0]==dirs[1]:orientation.append(list(edge))
    chi=len(vertex_faces)-len(local_edges)+len(ids)
    disk=len(boundary_components)==1 and not boundary_bad and not pinches and not orientation and chi==1 and all(len(x)<=2 for x in local_edges.values())
    coords=unique[list(vertex_faces)][:,[0,2,1]].copy();coords[:,1]*=-1
    return {'faces':len(ids),'vertices':len(vertex_faces),'edges':len(local_edges),'euler_characteristic':chi,
        'boundary_edges':len(boundary),'boundary_components':len(boundary_components),
        'boundary_component_vertex_counts':[len(x)for x in boundary_components],
        'boundary_degree_histogram':dict(Counter(len(x)for x in boundary_graph.values())),
        'non_degree_two_boundary_vertices':boundary_bad,
        'selected_edge_valence_histogram':dict(Counter(len(x)for x in local_edges.values())),
        'boundary_edges_already_open_in_body':sum(len(edge_faces[e])==1 for e in boundary),
        'boundary_edges_with_nonmanifold_body_incidence':sum(len(edge_faces[e])>2 for e in boundary),
        'pinched_or_invalid_vertex_links':pinches,'orientation_conflict_edges':orientation,
        'single_manifold_disk':disk,'bounds_blender_m':{'min':coords.min(0).tolist(),'max':coords.max(0).tolist()},
        'candidate_removed_faces_under_1000':len(ids)<=1000,
        'minimal_disk_triangles_without_interior_vertices':len(boundary)-2 if disk else None,
        'source_face_ids':[list(face_sources[fid])for fid in sorted(ids)],
        'boundary_welded_body_edges':[list(edge)for edge in boundary]}


selected=set(seeds);rings=[]
for ring in range(4):
    assert time.monotonic()-START<25
    groups=components(selected,adj)
    details=[analyze_patch(group)for group in groups]
    row={'ring':ring,'faces':len(selected),'new_faces':len(selected)-(rings[-1]['faces']if rings else 0),
        'seed_faces':len(seeds),'face_components':len(groups),'components':details,
        'single_connected_manifold_disk':len(details)==1 and details[0]['single_manifold_disk'],
        'source_material_counts':dict(Counter(doc['materials'][doc['meshes'][0]['primitives'][face_sources[f][0]]['material']]['name']for f in selected))}
    write(f'ring-{ring}.json',row);rings.append(row)
    selected|={other for fid in selected for other in adj[fid]}
report={'schema_version':1,'status':'screened_not_repaired','source_model_sha256':digest(SOURCE),
    'seed_payload_sha256':digest(PAYLOAD),'seed_faces':198,
    'coordinates':'Source GLB exact float32 positions; reported bounds converted to Blender XYZ Z-up meters',
    'alias_policy':'Only exact position tuples within body materials0/2; no tolerance weld, no companion vertices, no output model mutation',
    'protected_primitives':protected,'body_faces':len(faces),'body_vertices_before_aliasing':len(points),'body_unique_positions':len(unique),
    'body_edge_valence_histogram':dict(Counter(len(x)for x in edge_faces.values())),
    'rings':[{'ring':r['ring'],'faces':r['faces'],'components':r['face_components'],
              'single_connected_manifold_disk':r['single_connected_manifold_disk'],
              'component_faces':[d['faces']for d in r['components']],
              'boundary_components':[d['boundary_components']for d in r['components']],
              'pinched_vertex_counts':[len(d['pinched_or_invalid_vertex_links'])for d in r['components']]}for r in rings],
    'elapsed_seconds':time.monotonic()-START,'sources_unchanged':all(digest(ROOT/p['path'])==p['sha256']for p in json.loads((OUT/'inputs.json').read_text())),
    'native_jobs':0,'provider_calls':0,'textures_decoded':0,'quality_approved':False,
    'limits':{'neighbor_rings':3,'whole_mesh_faces':100000,'candidate_removal_screen_faces':1000,'wall_seconds':30,'rss_mib':256},
    'limitation':'A topological disk is only a necessary screen. This does not prove a smooth curved fit, valid triangulation, nonintersection, deformation quality or recovered hidden geometry.'}
write('report.json',report)
print(json.dumps(report,sort_keys=True))
