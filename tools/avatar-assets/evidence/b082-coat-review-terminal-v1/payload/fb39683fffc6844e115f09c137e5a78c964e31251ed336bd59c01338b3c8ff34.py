from pathlib import Path
import sys,json,io,hashlib
import numpy as np
from PIL import Image
O=Path(__file__).resolve().parent;R=O.parents[1];sys.path.insert(0,str(R))
from avatar_assets.three_d import _GLB
from avatar_assets.safety import sha256,write_json
p=R/'outputs/studio-b082dc54dbb54f9d9030f75b/model-0';g=_GLB(p/'model.glb')
def access(pr,key):
 a=pr['attributes'][key];s=g.doc['accessors'][a];return np.asarray(g.accessor(a,s['type'],(5126,5125,5123,5121)))
def img(n):
 v=g.doc['bufferViews'][g.doc['images'][n]['bufferView']];b=g.binary[v.get('byteOffset',0):v.get('byteOffset',0)+v['byteLength']]
 with Image.open(io.BytesIO(b)) as im:
  assert im.width*im.height<=16777216
  return np.asarray(im.convert('RGB')).copy()
body=g.doc['meshes'][0]['primitives'][0];ids=access(body,'_AVATAR_FUR_VERTEX_ID').reshape(-1).astype(int);uv=access(body,'TEXCOORD_0')
s=g.doc['accessors'][body['indices']];ix=np.asarray(g.accessor(body['indices'],'SCALAR',(5125,5123))).reshape(-1,3).astype(int)
lookup={}
for face in ix:
 key=tuple(sorted(ids[face]));row={int(ids[v]):uv[v] for v in face};lookup.setdefault(key,[]).append(row)
pixels=img(1);H,W=pixels.shape[:2];out=[]
for band in range(2):
 coat=g.doc['meshes'][band+1]['primitives'][0];ci=access(coat,'_AVATAR_FUR_VERTEX_ID').reshape(-1).astype(int);cv=access(coat,'COLOR_0')
 with np.load(p/f'portable-root-bindings/PortableCoat_{band}.npz',allow_pickle=False) as a:
  tris=a['source_vertex_ids'];bary=a['barycentric']
  selected=np.unique(np.linspace(0,len(tris)-1,128,dtype=int));errors=[];ambiguous=0
  for root in selected:
   rows=lookup.get(tuple(sorted(tris[root])),[])
   candidates=[np.array([r[int(v)] for v in tris[root]]) for r in rows]
   if not candidates or any(not np.array_equal(candidates[0],v) for v in candidates[1:]):ambiguous+=1;continue
   q=bary[root]@candidates[0];xy=np.mod(q,1)*[W,H]-.5;lo=np.floor(xy).astype(int);f=xy-lo
   rgb=pixels[np.ix_([lo[1]%H,(lo[1]+1)%H],[lo[0]%W,(lo[0]+1)%W])].astype(float)/255
   linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
   expected=(linear*np.outer([1-f[1],f[1]],[1-f[0],f[0]])[:,:,None]).sum((0,1))
   observed=cv[ci//6==root];assert len(observed)>0
   errors.append(float(np.max(np.abs(observed[:,:3]-expected))))
  out.append({'band':band,'requested_sample_roots':len(selected),'matched_roots':len(errors),'ambiguous_roots':ambiguous,'max_color_component_error':max(errors,default=None),'median_color_component_error':float(np.median(errors)) if errors else None})
del pixels
mr=img(2);mrstats={'size':list(mr.shape[:2]),'roughness_green_quantiles':np.quantile(mr[:,:,1],[0,.1,.5,.9,1]).tolist(),'metallic_blue_quantiles':np.quantile(mr[:,:,2],[0,.1,.5,.9,1]).tolist()};del mr
report=json.loads((p/'finish-report.json').read_text());areas=[]
for row in report['portable_fur']:
 s=row['root_summary'];area=s['eligible_area_m2'];areas.append({'cards':row['cards'],'area_m2':area,'roots_per_m2':row['cards']/area,'area_per_root_sqrt_m':(area/row['cards'])**.5,'length_range_m':s['length_range_m'],'width_range_m':s['width_range_m']})
write_json(O/'result.json',{'source_model_sha256':g.hash,'model_unchanged':sha256(p/'model.glb')==g.hash,'kind':'read_only_shader_and_color_diagnosis','materials':g.doc['materials'],'sampled_color_check':out,'color_check_convention':'Actual root source IDs and oriented UV corners; repeat bilinear LOD0; per-texel sRGB decode; compare every exported duplicate of selected root COLOR_0. No mip/lighting equivalence claim.','body_metal_roughness_8bit':mrstats,'allocation':areas,'limitations':['No rendering or source changes','Shader differences and sparse upright geometry are not causally isolated by this CPU check','Square root area per root is a nominal allocation scale, not measured nearest-neighbor spacing']})
print(json.dumps({'samples':out,'mr':mrstats,'allocation':areas}))
