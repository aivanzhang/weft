"""Read-only contact provenance and sparse texture census; no scene execution."""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
START = time.monotonic()


def write(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class GLB:
    def __init__(self, path):
        assert path.stat().st_size <= 64 * 1024**2
        self.path, self.raw = path, path.read_bytes()
        assert self.raw[:4] == b'glTF'
        size, kind = struct.unpack_from('<II', self.raw, 12)
        assert kind == 0x4E4F534A and size <= 2 * 1024**2
        self.doc = json.loads(self.raw[20:20+size])
        offset = 20+size
        length, kind = struct.unpack_from('<II', self.raw, offset)
        assert kind == 0x004E4942 and offset+8+length == len(self.raw)
        self.binary = memoryview(self.raw)[offset+8:]

    def array(self, index):
        a = self.doc['accessors'][index]
        assert 'sparse' not in a and a['count'] <= 300000
        v = self.doc['bufferViews'][a['bufferView']]
        dtype = {5121: '<u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}[a['componentType']]
        width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
        item = np.dtype(dtype).itemsize
        stride = v.get('byteStride', width*item)
        return np.ndarray((a['count'], width), dtype=dtype, buffer=self.binary,
                          offset=v.get('byteOffset', 0)+a.get('byteOffset', 0),
                          strides=(stride, item))


def canonical(ids):
    a, b, c = map(int, ids)
    return min((a, b, c), (b, c, a), (c, a, b))


def summary(values):
    a = np.asarray(values)
    return {'count': int(a.size), 'min': float(a.min()) if a.size else None,
            'median': float(np.median(a)) if a.size else None,
            'max': float(a.max()) if a.size else None}


def linear(value):
    return np.where(value <= .04045, value/12.92, ((value+.055)/1.055)**2.4)


def encoded(value):
    return np.where(value <= .0031308, value*12.92, 1.055*np.maximum(value, 0)**(1/2.4)-.055)


def sample(pixels, uv):
    height, width = pixels.shape[:2]
    p = (np.asarray(uv) % 1)*[width, height]-.5
    low = np.floor(p).astype(np.int64)
    t = p-low
    result = np.zeros((*p.shape[:-1], 3), dtype=np.float64)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        weight = (t[..., 0] if dx else 1-t[..., 0])*(t[..., 1] if dy else 1-t[..., 1])
        result += linear(pixels[(low[..., 1]+dy) % height, (low[..., 0]+dx) % width]/255)*weight[..., None]
    return result


v7 = ROOT/'outputs/companion-separation-v7'
current_path = ROOT/'outputs/panda-staged-portable-model-v1/model.glb'
pins = [current_path, v7/'diagnostic-model.glb', v7/'original.glb', v7/'partition-plan.json',
        v7/'partition-arrays.npz', v7/'native-report.json', v7/'config.json',
        v7/'sources/partition_generated.py', v7/'sources/blender_partition_generated.py',
        ROOT/'outputs/panda-staged-fringe-preparation-v1/finish-config.json',
        ROOT/'outputs/panda-staged-fringe-v1/prepared-finish-report.json',
        ROOT/'outputs/panda-staged-fringe-v1/source-skin.json',
        ROOT/'outputs/panda-contact-ablation-v1/diagnostics.json']
write('inputs.json', [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in pins])
plan = json.loads((v7/'partition-plan.json').read_text())
native = json.loads((v7/'native-report.json').read_text())
with np.load(v7/'partition-arrays.npz', allow_pickle=False) as arrays:
    points, faces = arrays['points'], arrays['faces']
    uv, normal = arrays['uv_per_corner'], arrays['normal_per_corner']
    face_mats, labels = arrays['face_material_indices'], arrays['vertex_labels']
assert len(faces) <= 100000 and len(points) <= 100000
spec = importlib.util.spec_from_file_location('frozen_partition', v7/'sources/partition_generated.py')
partition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(partition)
vertex_normals = np.zeros_like(points)
np.add.at(vertex_normals, faces.ravel(), normal.reshape(-1, 3))
vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1)[:, None], 1e-12)
all_faces, all_uv, all_mats = list(faces), list(uv), list(face_mats)
origins = [{'source_face_id': i, 'kind': 'repainted_original' if face_mats[i] >= 2 else 'original'} for i in range(len(faces))]
cap_rows = []
for number, loop in enumerate(plan['loops']):
    tris = partition.triangulate_boundary_3d(points[loop], vertex_normals[loop], deadline=START+30)
    rect = plan['loop_palette_patches'][number]['uv_rect_blender']
    local_uv = np.asarray([[.1, .1], [.9, .1], [.1, .9]])*np.subtract(rect[2:], rect[:2])+rect[:2]
    first = len(all_faces)
    for tri in tris:
        all_faces.append(np.asarray(loop)[tri]); all_uv.append(local_uv); all_mats.append(2+int(labels[loop[0]]))
        origins.append({'kind': 'authored_closure', 'closure_loop': number, 'closure_triangle': len(all_faces)-1-len(faces)})
    cap_rows.append({'loop': number, 'part': int(labels[loop[0]]), 'expected_face_ids': [first, len(all_faces)],
                     'bounds_m': {'min': points[loop].min(0).tolist(), 'max': points[loop].max(0).tolist()}})
all_faces, all_uv, all_mats = np.asarray(all_faces), np.asarray(all_uv), np.asarray(all_mats)
assert len(all_faces)-len(faces) == 497

# A bounded spatial hash admits import rounding but never chooses between distinct close points.
unique, point_inverse = np.unique(points, axis=0, return_inverse=True)
bins = defaultdict(list)
for i, q in enumerate(np.floor(unique/.00001).astype(np.int64)):
    bins[tuple(q)].append(i)
neighbors = list(itertools.product((-1, 0, 1), repeat=3))
lookup = {}
position_errors, position_ambiguous = [], []


def position_ids(values):
    result = []
    for value in values:
        key = tuple(map(float, value))
        if key not in lookup:
            q = np.floor(value/.00001).astype(np.int64)
            candidates = [i for offset in neighbors for i in bins.get(tuple(q+offset), [])]
            errors = np.max(np.abs(unique[candidates]-value), axis=1) if candidates else np.asarray([])
            good = [candidates[i] for i in np.flatnonzero(errors <= 2e-7)]
            if len(good) != 1:
                position_ambiguous.append({'position': list(key), 'candidates': good})
                raise ValueError('Position correspondence is missing or ambiguous')
            lookup[key] = good[0]
            position_errors.append(float(np.max(np.abs(unique[good[0]]-value))))
        result.append(lookup[key])
    return np.asarray(result)


expected = defaultdict(list)
for i, tri in enumerate(all_faces):
    expected[(int(all_mats[i]), canonical(point_inverse[tri]))].append(i)
all_uv_down = all_uv.copy(); all_uv_down[..., 1] = 1-all_uv_down[..., 1]
actual = GLB(current_path)
source = GLB(v7/'diagnostic-model.glb')
original_material = {'body_surface': 0, 'companion_surface': 1, 'body_cap': 2, 'companion_cap': 3}
for name in ['body_surface__color_rear-tail-base', 'body_surface__short_pile_adaptive_v1']:
    original_material[name] = 0
original_material['body_cap__short_pile_adaptive_v1'] = 2


def inspect(model):
    rows, ambiguous, used = [], [], Counter()
    for pi, primitive in enumerate(model.doc['meshes'][0]['primitives']):
        attrs = primitive['attributes']; material = model.doc['materials'][primitive['material']]['name']
        p = model.array(attrs['POSITION'])[:, [0, 2, 1]].copy(); p[:, 1] *= -1
        n = model.array(attrs['NORMAL'])[:, [0, 2, 1]].copy(); n[:, 1] *= -1
        f = model.array(primitive['indices']).reshape(-1, 3)
        tex = model.array(attrs['TEXCOORD_0'])
        ids = position_ids(p)
        for fid, tri in enumerate(f):
            candidates = expected.get((original_material[material], canonical(ids[tri])), [])
            # Rotation-preserving UV comparison disambiguates geometrically coincident faces.
            good = []
            for candidate in candidates:
                source_ids = point_inverse[all_faces[candidate]]
                for shift in range(3):
                    if np.array_equal(np.roll(source_ids, shift), ids[tri]) and np.max(np.abs(np.roll(all_uv_down[candidate], shift, axis=0)-tex[tri])) <= 2e-7:
                        good.append(candidate); break
            if len(good) != 1:
                ambiguous.append({'primitive': pi, 'triangle': fid, 'candidate_source_faces': good, 'geometry_candidates': candidates})
                continue
            eid = good[0]; used[eid] += 1
            geo = np.cross(p[tri[1]]-p[tri[0]], p[tri[2]]-p[tri[0]])
            area = np.linalg.norm(geo)*.5
            geo /= max(area*2, 1e-30)
            mean_n = n[tri].mean(0); mean_n /= max(np.linalg.norm(mean_n), 1e-30)
            rows.append({'primitive': pi, 'triangle': fid, 'material': material,
                'vertex_ids': tri.tolist(), 'position_accessor': attrs['POSITION'],
                'expected_face_id': eid, **origins[eid], 'area_m2': float(area),
                'centroid_blender_m': p[tri].mean(0).tolist(),
                'normal_geo_dot': float(np.dot(geo, mean_n)), 'uv_centroid': tex[tri].mean(0).tolist()})
    return rows, ambiguous, used


source_rows, source_ambiguous, source_used = inspect(source)
rows, ambiguous, used = inspect(actual)
write('provenance.json', {'position_tolerance_m': 2e-7, 'position_errors_m': summary(position_errors),
    'source_export': {'matched_faces': len(source_rows), 'ambiguous_faces': source_ambiguous,
        'unexported_expected_face_ids': sorted(set(range(len(all_faces)))-set(source_used)),
        'duplicate_expected_matches': {str(k):v for k,v in source_used.items() if v>1}},
    'current': {'matched_faces': len(rows), 'ambiguous_faces': ambiguous,
        'unexported_expected_face_ids': sorted(set(range(len(all_faces)))-set(used)),
        'duplicate_expected_matches': {str(k):v for k,v in used.items() if v>1}},
    'material_origin_counts': {name: dict(Counter(r['kind'] for r in rows if r['material']==name)) for name in original_material},
    'expected_origin_counts': dict(Counter(r['kind'] for r in origins)),
    'claim': 'Exact oriented triangle and UV-corner provenance within stated import float tolerances; no normals or weights used to choose correspondence.'})

# Sparse actual-UV samples retain palette bytes; no edited texture is written.
img = actual.doc['images'][1]; view = actual.doc['bufferViews'][img['bufferView']]
blob = actual.binary[view.get('byteOffset', 0):view.get('byteOffset', 0)+view['byteLength']]
with Image.open(io.BytesIO(blob)) as handle:
    assert handle.size == (4096, 4096)
    pixels = np.asarray(handle.convert('RGB'))
texture_sha = hashlib.sha256(blob).hexdigest()
bary = np.asarray([[1/3]*3, [.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.5,.5,0],[0,.5,.5],[.5,0,.5]])
regions = {'cheek': ([-.295,-.27,.745],[-.19,-.10,.89]),
           'shoulder': ([-.39,-.30,.48],[-.10,-.04,.725])}
selected = {key: [] for key in regions}
for row in rows:
    if not row['material'].startswith('body_'):
        continue
    center = np.asarray(row['centroid_blender_m'])
    for name, (low, high) in regions.items():
        if not (np.all(center >= low) and np.all(center <= high)):
            continue
        primitive = actual.doc['meshes'][0]['primitives'][row['primitive']]
        tex = actual.array(primitive['attributes']['TEXCOORD_0'])[row['vertex_ids']]
        rgb = encoded(sample(pixels, bary@tex))
        green = rgb[:,1] > 1.2*np.maximum(rgb[:,0],rgb[:,2])
        light = rgb.mean(1) > .45
        item = dict(row, sample_max_rgb=rgb.max(0).tolist(), sample_mean_rgb=rgb.mean(0).tolist(),
                    green_samples=int(green.sum()), light_samples=int(light.sum()))
        selected[name].append(item)
write('contact-selectors.json', {'source_model_sha256':digest(current_path), 'mesh':0,
    'index_space':'Exact current GLB primitive-local triangle and vertex IDs; do not reuse on a re-export',
    'regions_blender_m':regions, 'samples_barycentric':bary.tolist(),
    'texture_sha256':texture_sha, 'sampling':'Seven sparse Linear REPEAT texel-center samples per face, decoded before interpolation; top-down image/glTF V-down',
    'warning':'Color counts are sampled candidates, not pixel-perfect visible attribution. Shoulder light is not automatically contamination outside the authored contact region.',
    'regions':selected})
stats = {}
for name, subset in selected.items():
    stats[name] = {'faces':len(subset), 'origins':dict(Counter(r['kind'] for r in subset)),
        'green_sample_faces':sum(r['green_samples']>0 for r in subset),
        'light_sample_faces':sum(r['light_samples']>0 for r in subset),
        'green_by_origin':dict(Counter(r['kind'] for r in subset if r['green_samples']>0)),
        'light_by_origin':dict(Counter(r['kind'] for r in subset if r['light_samples']>0)),
        'normal_vs_geometric_angle_degrees':summary(np.degrees(np.arccos(np.clip([r['normal_geo_dot'] for r in subset],-1,1)))),
        'closure_loops':sorted({r['closure_loop'] for r in subset if 'closure_loop' in r})}
write('summary.json', {'status':'read_only_diagnosis_not_repair_or_acceptance', 'elapsed_seconds':time.monotonic()-START,
    'current_model_sha256':digest(current_path), 'v7_source_sha256':digest(v7/'diagnostic-model.glb'),
    'regions':stats, 'closure_loops':cap_rows, 'source_intersection_report_retained':native['intersection_screen'],
    'new_provider_calls':0, 'native_jobs':0, 'source_mutated':False, 'quality_approved':False})
assert all(digest(p)==pin['sha256'] for p,pin in zip(pins,json.loads((OUT/'inputs.json').read_text())))
print(json.dumps({'status':'completed_read_only','mapped_current_faces':len(rows),'ambiguous':len(ambiguous),'regions':stats},sort_keys=True))
