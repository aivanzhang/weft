"""Join parent's native pixel witnesses to the completed CPU provenance checkpoint."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
path=ROOT/'outputs/panda-contact-pixel-attribution-v1/attribution.json'
witness=json.loads(path.read_text())
checkpoint=json.loads((OUT/'contact-selectors.json').read_text())
source=ROOT/'outputs/panda-staged-portable-model-v1/model.glb'
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()==witness['source_model_sha256']==checkpoint['source_model_sha256']
size=struct.unpack_from('<I',raw,12)[0]
doc=json.loads(raw[20:20+size]); binary=memoryview(raw)[28+size:]


def array(index):
    a=doc['accessors'][index];v=doc['bufferViews'][a['bufferView']]
    dtype={5121:'<u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
    width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
    item=np.dtype(dtype).itemsize
    return np.ndarray((a['count'],width),dtype=dtype,buffer=binary,
        offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',width*item),item))


provenance={(r['primitive'],r['triangle']):r for rows in checkpoint['regions'].values() for r in rows}
rows=[]
for ray in witness['pixels']:
    assert ray['hit'] and ray['object']=='PartitionedBodyAndPart'
    p=np.asarray(ray['rest_positions_gltf_xyz']); uv=np.asarray(ray['uv0_corners']);uv[:,1]=1-uv[:,1]
    matches=[]
    for pi,primitive in enumerate(doc['meshes'][0]['primitives']):
        if doc['materials'][primitive['material']]['name']!=ray['material']:
            continue
        attrs=primitive['attributes'];f=array(primitive['indices']).reshape(-1,3)
        positions=array(attrs['POSITION'])[f];texcoords=array(attrs['TEXCOORD_0'])[f]
        for shift in range(3):
            hit=np.flatnonzero((np.max(np.abs(np.roll(positions,shift,axis=1)-p),axis=(1,2))<=2e-7)&
                (np.max(np.abs(np.roll(texcoords,shift,axis=1)-uv),axis=(1,2))<=2e-7))
            matches.extend((pi,int(fid),shift)for fid in hit)
    assert len(matches)==1, 'Pixel rest geometry/UV join is missing or ambiguous'
    pi,fid,shift=matches[0]; item=provenance.get((pi,fid))
    rows.append({'pixel':ray['pixel'],'observed_kind':ray['kind'],'observed_rgb8':ray['rgb'],
        'material':ray['material'],'primitive':pi,'triangle':fid,'cyclic_shift':shift,
        'provenance':item,'inside_saved_region':item is not None,
        'native_polygon_index':ray['polygon_index']})
result={'schema_version':1,'source_model_sha256':checkpoint['source_model_sha256'],
    'ray_attribution_path':str(path.relative_to(ROOT)),'ray_attribution_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'checkpoint_sha256':hashlib.sha256((OUT/'contact-selectors.json').read_bytes()).hexdigest(),
    'matches':rows,'ambiguous':0,'ray_hits_body':len(rows),
    'scope':'Exact unique material+rest-position+UV join; pixels come from parent actual native raycast, not this CPU calculation',
    'sampling_limit':'Sparse LOD0 UV colors do not predict native mipmapped/antialiased pixel footprints',
    'quality_approved':False}
with (OUT/'pixel-provenance.json').open('x')as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
print(json.dumps({'body_hits':len(rows),'outside_region':sum(not r['inside_saved_region']for r in rows),'ambiguous':0}))
