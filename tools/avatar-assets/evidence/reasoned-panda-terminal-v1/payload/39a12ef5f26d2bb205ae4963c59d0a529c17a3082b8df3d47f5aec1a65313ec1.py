"""Read-only body weight adjacency inspection; no evaluation or asset changes."""
import hashlib,json,resource,struct,time
from pathlib import Path
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(55,55))
start=time.monotonic(); root=Path(__file__).resolve().parents[2]; job=root/'outputs/studio-d8a680ce6580429f9fdd016f'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 assert p.stat().st_size<64*1024**2
 b=p.read_bytes(); magic,version,length=struct.unpack_from('<III',b);assert magic==0x46546c67 and version==2 and length==len(b)
 n,typ=struct.unpack_from('<II',b,12);assert typ==0x4e4f534a;d=json.loads(b[20:20+n]);off=20+n;n,typ=struct.unpack_from('<II',b,off);assert typ==0x004e4942;blob=memoryview(b)[off+8:off+8+n]
 def a(i):
  v=d['accessors'][i];assert 'sparse' not in v and not v.get('normalized');view=d['bufferViews'][v['bufferView']];dt=np.dtype({5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}[v['componentType']]);cols={'SCALAR':1,'VEC3':3,'VEC4':4}[v['type']]; stride=view.get('byteStride',dt.itemsize*cols);assert v['count']<200000
  return np.ndarray((v['count'],cols),dtype=dt,buffer=blob,offset=view.get('byteOffset',0)+v.get('byteOffset',0),strides=(stride,dt.itemsize)).copy()
 return d,a
rows=[]; saved=[]
for rel in ['inputs/source-model.glb','model-0/model.glb','model-1/model.glb']:
 p=job/rel;d,a=read(p);q=d['meshes'][0]['primitives'][0];attr=q['attributes'];pos=a(attr['POSITION']).astype(float);tris=a(q['indices']).ravel().reshape(-1,3); unique,first,inv=np.unique(pos,axis=0,return_index=True,return_inverse=True); edges=np.concatenate([inv[tris[:,[0,1]]],inv[tris[:,[1,2]]],inv[tris[:,[2,0]]]]);edges=np.unique(np.sort(edges,axis=1),axis=0);edges=edges[edges[:,0]!=edges[:,1]]
 row={'path':str(p.relative_to(root)),'sha256':h(p),'body_vertices':len(pos),'triangles':len(tris),'exact_position_groups':len(unique),'split_duplicate_vertices':len(pos)-len(unique),'true_geometric_edges':len(edges),'has_skin_weights':'WEIGHTS_0' in attr}
 if 'WEIGHTS_0' in attr:
  joints=a(attr['JOINTS_0']);weights=a(attr['WEIGHTS_0']);names=[d['nodes'][j]['name'] for j in d['skins'][0]['joints']];w=np.zeros((len(pos),len(names)));np.add.at(w,(np.arange(len(pos))[:,None],joints),weights)
  mx=np.full((len(unique),len(names)),-np.inf);mn=np.full_like(mx,np.inf);np.maximum.at(mx,inv,w);np.minimum.at(mn,inv,w);seam=(mx-mn).sum(axis=1); ww=w[first]; delta=np.abs(ww[edges[:,0]]-ww[edges[:,1]]).sum(axis=1); length=np.linalg.norm(unique[edges[:,0]]-unique[edges[:,1]],axis=1)
  row.update(seam_weight_disagreement_groups=int(np.count_nonzero(seam>1e-5)),seam_max_l1=float(seam.max()),edge_l1_percentiles=dict(zip(['p50','p90','p95','p99','max'],map(float,np.percentile(delta,[50,90,95,99,100])))),edge_l1_above_0_5=int((delta>.5).sum()),edge_l1_above_1=int((delta>1).sum()),edge_l1_above_1_length_range=[float(length[delta>1].min()),float(length[delta>1].max())] if (delta>1).any() else None)
  idx=np.argsort(delta)[-10:][::-1];row['largest_edges']=[{'l1':float(delta[k]),'length_local_units':float(length[k]),'endpoints':unique[edges[k]].tolist(),'weights':[{names[j]:round(float(ww[v,j]),6) for j in np.flatnonzero(ww[v]>1e-5)} for v in edges[k]]} for k in idx]
  saved.append((unique,edges,ww))
 rows.append(row)
compare={'same_body_positions':bool(np.array_equal(saved[0][0],saved[1][0])),'same_welded_edges':bool(np.array_equal(saved[0][1],saved[1][1]))}
if compare['same_body_positions']:
 change=np.abs(saved[0][2]-saved[1][2]).sum(axis=1);compare.update(changed_position_groups=int((change>1e-5).sum()),weight_l1_change_percentiles=np.percentile(change,[50,90,95,99,100]).tolist())
out={'classification':'Measured exported rest-pose body skin-weight adjacency, not a causal deformation/appearance test','method':'Exact-position welding; only triangle-connected nonzero edges, excluding portable coat meshes. Check split duplicate weights before selecting representative weights. L1 maximum possible is 2. No spatial nearest-neighbor joins.','limits':['Input source has no skin weights; source-versus-finished weight comparison unavailable.','No before-constraint weights saved here; cannot isolate weight_constraints from automatic rigging, authored rig layers or top-four export truncation.','Exact welding treats coincident disconnected surface positions as one; only original triangle-connected edge pairs are measured.','Mesh-local edge lengths; no world unit assumption.'], 'models':rows,'comparison':compare,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'max_runtime_seconds':55,'memory_acceptance_limit_bytes':512*1024**2,'source_sha256':{str(p.relative_to(root)):h(p) for p in [root/'scripts/blender_finish_generated.py',root/'avatar_assets/author_contract.py',Path(__file__)]}}
assert out['peak_rss_bytes']<512*1024**2
Path(__file__).with_name('result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'elapsed':out['elapsed_seconds'],'rss':out['peak_rss_bytes'],'comparison':compare,'models':[{k:v for k,v in r.items() if k!='largest_edges'} for r in rows]},indent=2))
