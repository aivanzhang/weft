"""Bounded checkpoint finalization; no second texture decode or full correspondence census."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
START = time.monotonic()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, data):
    with (OUT/name).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def glb(path):
    assert path.stat().st_size < 64*1024**2
    data = path.read_bytes()
    size = struct.unpack_from('<I', data, 12)[0]
    doc = json.loads(data[20:20+size]); binary = memoryview(data)[28+size:]
    def array(index):
        a = doc['accessors'][index]; v = doc['bufferViews'][a['bufferView']]
        assert 'sparse' not in a and a['count'] <= 300000
        dtype = {5121:'<u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
        width = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
        item = np.dtype(dtype).itemsize
        return np.ndarray((a['count'],width),dtype=dtype,buffer=binary,
            offset=v.get('byteOffset',0)+a.get('byteOffset',0), strides=(v.get('byteStride',item*width),item))
    return doc,array


def stats(values):
    a=np.asarray(values,dtype=float)
    return {'count':len(a),'min':float(a.min()) if len(a) else None,
            'median':float(np.median(a)) if len(a) else None,'p90':float(np.percentile(a,90)) if len(a) else None,
            'max':float(a.max()) if len(a) else None}


pins=json.loads((OUT/'inputs.json').read_text())
assert all(digest(ROOT/p['path'])==p['sha256'] for p in pins)
selection=json.loads((OUT/'contact-selectors.json').read_text())
provenance=json.loads((OUT/'provenance.json').read_text())
source_doc,source_array=glb(ROOT/'outputs/companion-separation-v7/diagnostic-model.glb')
current_doc,current_array=glb(ROOT/'outputs/panda-staged-portable-model-v1/model.glb')
source_triangles=[]
for pi,prim in enumerate(source_doc['meshes'][0]['primitives']):
    if pi != 2:
        continue
    attrs=prim['attributes']; faces=source_array(prim['indices']).reshape(-1,3)
    p=source_array(attrs['POSITION']);uv=source_array(attrs['TEXCOORD_0']);n=source_array(attrs['NORMAL'])
    for fid,tri in enumerate(faces):
        source_triangles.append((fid,p[tri],uv[tri],n[tri]))
sp=np.asarray([r[1] for r in source_triangles]);su=np.asarray([r[2] for r in source_triangles]);sn=np.asarray([r[3] for r in source_triangles])
normal_comparison=[]
for region,rows in selection['regions'].items():
    for row in rows:
        if row['kind']=='original':
            continue
        prim=current_doc['meshes'][0]['primitives'][row['primitive']];attrs=prim['attributes'];tri=row['vertex_ids']
        p=current_array(attrs['POSITION'])[tri];uv=current_array(attrs['TEXCOORD_0'])[tri];n=current_array(attrs['NORMAL'])[tri]
        matches=[]
        for shift in range(3):
            good=np.flatnonzero((np.max(np.abs(np.roll(sp,shift,axis=1)-p),axis=(1,2))<=2e-7)&
                (np.max(np.abs(np.roll(su,shift,axis=1)-uv),axis=(1,2))<=2e-7))
            matches.extend((int(i),shift)for i in good)
        assert len(matches)==1, 'Cap provenance ambiguous during normal-only comparison'
        source_id,shift=matches[0]
        old=np.roll(sn[source_id],shift,axis=0)
        old=old/np.linalg.norm(old,axis=1)[:,None];new=n/np.linalg.norm(n,axis=1)[:,None]
        angle=np.degrees(np.arccos(np.clip(np.sum(old*new,axis=1),-1,1)))
        geo=np.cross(p[1]-p[0],p[2]-p[0]);geo/=np.linalg.norm(geo)
        oldmean=old.mean(0);oldmean/=np.linalg.norm(oldmean)
        normal_comparison.append({'region':region,'primitive':row['primitive'],'triangle':row['triangle'],
            'kind':row['kind'],'source_primitive':2,'source_triangle':source_id,
            'source_face_id':row.get('source_face_id'),'closure_loop':row.get('closure_loop'),
            'old_geometric_dot':float(oldmean@geo),'current_geometric_dot':row['normal_geo_dot'],
            'corner_delta_degrees':angle.tolist()})
report={}
for region,rows in selection['regions'].items():
    report[region]={}
    for kind in ['original','repainted_original','authored_closure']:
        subset=[r for r in rows if r['kind']==kind]
        compare=[r for r in normal_comparison if r['region']==region and r['kind']==kind]
        report[region][kind]={'faces':len(subset),
            'green_sample_faces':sum(r['green_samples']>0 for r in subset),
            'light_sample_faces':sum(r['light_samples']>0 for r in subset),
            'current_normal_vs_geometric_degrees':stats(np.degrees(np.arccos(np.clip([r['normal_geo_dot']for r in subset],-1,1)))),
            'current_opposite_hemisphere_faces':sum(r['normal_geo_dot']<0 for r in subset),
            'v7_opposite_hemisphere_faces':sum(r['old_geometric_dot']<0 for r in compare) if compare else None,
            'v7_to_current_normal_corner_delta_degrees':stats([x for r in compare for x in r['corner_delta_degrees']])}

# The original failure is retained; only complete written checkpoints are reused.
write('normal-comparison.json',{'model_sha256':selection['source_model_sha256'],
    'source_sha256':digest(ROOT/'outputs/companion-separation-v7/diagnostic-model.glb'),
    'matching':'Position and UV only, unique cyclic corner match; no normal-based selection',
    'geometry_tolerance_m':2e-7,'uv_tolerance':2e-7,'rows':normal_comparison})
write('summary.json',{'status':'read_only_diagnosis_not_acceptance', 'quality_approved':False,
    'current_model_sha256':selection['source_model_sha256'], 'regions':report,
    'current_base_faces_matched':provenance['current']['matched_faces'],
    'ambiguous_correspondences':len(provenance['current']['ambiguous_faces']),
    'position_error_m':provenance['position_errors_m'],
    'original_cpu_failure':json.loads((OUT/'cpu-job.json').read_text()),
    'recovery_scope':'Complete JSON checkpoints reused; no second texture decode or full census; added bounded cap-normal-only comparison',
    'closure_material_warning':'cap materials mix authored closures and UV-repainted original faces',
    'color_attribution_limit':'Sparse body-face UV samples, not visible pixel hits; companion overlap requires separate ray attribution',
    'native_jobs':0,'provider_calls':0,'sources_unchanged':True,'elapsed_seconds':time.monotonic()-START})
print(json.dumps({'status':'completed','regions':report},sort_keys=True))
