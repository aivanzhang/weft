"""Join eight actual subpixel surfaces to frozen source/color decisions; no texture decode."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
import time

import numpy as np

OUT = Path(__file__).resolve().parent
PREP = json.loads((OUT/'prepared.json').read_text())
START = time.monotonic()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


class GLB:
    def __init__(self, path):
        self.raw = Path(path).read_bytes()
        assert self.raw[:4] == b'glTF' and len(self.raw) <= 64*1024**2
        size, kind = struct.unpack_from('<II', self.raw, 12)
        assert kind == 0x4E4F534A and size <= 2*1024**2
        self.doc = json.loads(self.raw[20:20+size]); offset = 20+size
        length, kind = struct.unpack_from('<II', self.raw, offset)
        assert kind == 0x004E4942 and offset+8+length == len(self.raw)
        self.binary = memoryview(self.raw)[offset+8:]

    def array(self, index):
        a = self.doc['accessors'][index]; v = self.doc['bufferViews'][a['bufferView']]
        assert 'sparse' not in a and a['count'] <= 300000
        dtype = {5121:'<u1', 5123:'<u2', 5125:'<u4', 5126:'<f4'}[a['componentType']]
        width = {'SCALAR':1, 'VEC2':2, 'VEC3':3, 'VEC4':4}[a['type']]; size = np.dtype(dtype).itemsize
        return np.ndarray((a['count'], width), dtype=dtype, buffer=self.binary,
            offset=v.get('byteOffset',0)+a.get('byteOffset',0), strides=(v.get('byteStride',width*size),size))

    def triangles(self, primitive):
        p = self.doc['meshes'][0]['primitives'][primitive]
        f = self.array(p['indices']).reshape(-1,3)
        return f, self.array(p['attributes']['POSITION']), self.array(p['attributes']['TEXCOORD_0'])


def main():
    pins = PREP['inputs']
    for row in pins.values():
        assert sha(row['path']) == row['sha256'], 'Input changed'
    get = lambda key: json.loads(Path(pins[key]['path']).read_text())
    trace, census, old, selection = get('trace'), get('census'), get('old_selectors'), get('selection')
    current, source = GLB(pins['current']['path']), GLB(pins['source']['path'])
    assert trace['source_model_sha256'] == pins['current']['sha256']
    target = set(PREP['native_triangles'])
    faces = {r['triangle_index']: r for r in trace['faces'] if r['triangle_index'] in target}
    assert set(faces) == target and len(target) == 8
    saved = {(r['current_primitive'],r['current_triangle']):r for r in census['rows']}
    sparse = {(r['primitive'],r['triangle']):r for rows in old['regions'].values() for r in rows}
    selected = {(r['source_primitive'],r['source_triangle']) for r in selection['rows']}
    ff, pp, uu = source.triangles(0)
    point_members = defaultdict(set)
    for pi, fid in selected:
        if pi == 0:
            for point in pp[ff[fid]]:
                point_members[point.tobytes()].add(fid)
    first_counts = defaultdict(Counter); front_normals = defaultdict(list)
    for ray in trace['rays']:
        index = ray['first_visible_triangle']
        if index in target:
            first_counts[index][tuple(ray['pixel'])] += 1
            hit = next(h for h in ray['ordered_hits'] if h['triangle'] == index)
            assert not hit['backface_culled']
            front_normals[index].append(hit['normal_dot_forward'])
    result = []
    for index in sorted(target):
        ray = faces[index]
        assert ray['selected_source_face'] is None
        p = np.asarray(ray['rest_positions_blender_xyz_m'])[:,[0,2,1]]*[1,1,-1]
        uv = np.asarray(ray['uv0_blender_v_up'])*[1,-1]+[0,1]
        mapped = []
        for model, primitives in ((current,[5]), (source,[0])):
            matches = []
            for pi in primitives:
                f, points, tex = model.triangles(pi)
                positions, texcoords = points[f], tex[f]
                for shift in range(3):
                    yes = np.flatnonzero((np.max(np.abs(np.roll(positions,shift,axis=1)-p),axis=(1,2))<=2e-7) &
                        (np.max(np.abs(np.roll(texcoords,shift,axis=1)-uv),axis=(1,2))<=2e-7))
                    matches.extend((pi,int(fid),shift) for fid in yes)
            assert len(matches) == 1, 'Missing or ambiguous oriented position/UV correspondence'
            mapped.append(matches[0])
        cp, cf, cshift = mapped[0]; sp, sf, sshift = mapped[1]
        decision = saved.get((cp,cf)); historical = sparse.get((cp,cf))
        assert decision is not None and not decision['selected']
        assert decision['protected_white'] == (ray['protected_white_selector'] is not None)
        neighbors = Counter()
        for point in pp[ff[sf]]:
            neighbors.update(point_members.get(point.tobytes(),()))
        protected = decision['protected_white']
        eligible = not protected and decision['dense_white_fraction'] == 0 and decision['dense_green_samples'] == 0 and bool(first_counts[index]) and bool(neighbors)
        outcome = 'keep_protected_white' if protected else 'eligible_exact_observed_dark_contact_exception' if eligible else 'insufficient_evidence_keep_unchanged'
        result.append({'native_triangle':index,'native_polygon':ray['native_polygon'],
            'current_face':[cp,cf], 'v7_face':[sp,sf], 'source_vertex_rows':ff[sf].tolist(),
            'source_corner_to_native_rotation':sshift, 'material':ray['material'],
            'frontmost_grid_hits_by_pixel':[{'pixel':list(k),'count':v} for k,v in sorted(first_counts[index].items())],
            'frontmost_geometric_normal_dot_range':[min(front_normals[index]),max(front_normals[index])],
            'frozen_dense_samples':91,'frozen_green_samples':decision['dense_green_samples'],
            'frozen_white_fraction':decision['dense_white_fraction'], 'protected_white':protected,
            'original_selection_reasons':decision['reasons'], 'inside_unchanged_centroid_region':True,
            'existing_selected_neighbors':[{'v7_face':[0,fid],'shared_exact_positions':n}for fid,n in sorted(neighbors.items())],
            'saved_sparse_color':{k:historical[k] for k in ('sample_mean_rgb','sample_max_rgb','sample_min_rgb','green_samples','light_samples') if k in historical} if historical else None,
            'provenance': decision['provenance'], 'recommendation':outcome,
            'reason': 'Existing protected-white rule applies; subpixel contribution does not authorize recoloring white surface.' if protected else
                'Actual frontmost residual-pixel contribution on original body material, no saved white samples, and exact adjacency to an already selected contact face. Eligible only as an explicit face-bound authored exception, not a broader green threshold or proof of dark anatomy.' if eligible else
                'Frozen evidence does not justify an additional color edit.',
            'old_uv0_gltf_v_down':uu[ff[sf]].tolist(),
            'rest_positions_blender_xyz_m':(pp[ff[sf]][:,[0,2,1]]*[1,-1,1]).tolist()})
    counts = Counter(r['recommendation'] for r in result)
    coverage = []
    for pixel in [(271,361),(268,365),(287,403),(301,340),(311,385),(283,388)]:
        rr = [r for r in trace['rays'] if tuple(r['pixel']) == pixel]
        coverage.append({'pixel':list(pixel),'rays':len(rr),'selected_frontmost':sum(r['first_visible_selected_for_uv_repair']for r in rr),
            'misses':sum(r['first_visible_triangle'] is None for r in rr),
            'unselected_frontmost':dict(Counter(str(r['first_visible_triangle']) for r in rr if r['first_visible_triangle'] is not None and not r['first_visible_selected_for_uv_repair']))})
    for row in pins.values():
        assert sha(row['path']) == row['sha256'], 'Input changed during review'
    report={'schema_version':1,'status':'residual_face_review_complete','source_v7_sha256':pins['source']['sha256'],
        'current_model_sha256':pins['current']['sha256'],'trace_sha256':pins['trace']['sha256'],
        'rows':result,'counts':dict(counts),'grid_summary':coverage,
        'model_mutated':False,'texture_decodes':0,'native_jobs':0,'thresholds_changed':False,'region_expanded':False,
        'source_files_unchanged':True,'elapsed_seconds':time.monotonic()-START,
        'self_peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
        'next_safe_action':'Parent review of eligible exact face exceptions listed above; keep every protected white face unchanged. Any new UV candidate must be append-only with independent color/white-protection review. No automatic mutation follows this report.',
        'limitations':['These regular subpixel grids are not the exact EEVEE16-sample TAA/filter footprint.',
            'No new texture pixels were decoded or inspected; the color decisions are the immutable earlier census.',
            'Eligibility is an authored local contact-color decision, not recovered anatomy or automatic material classification.',
            'Missing rays at the silhouette and remaining bright surface contributions are not solved by face selection.'],
        'quality_approved':False}
    write('review.json',report)
    print(json.dumps({'counts':dict(counts),'v7_faces':[(r['native_triangle'],r['v7_face'],r['recommendation'])for r in result],
        'elapsed_seconds':report['elapsed_seconds'],'self_peak_rss_bytes':report['self_peak_rss_bytes']}))


try:
    main()
except Exception as error:
    write('failure.json',{'status':'failed','error':str(error),'type':type(error).__name__,'model_mutated':False})
    raise
