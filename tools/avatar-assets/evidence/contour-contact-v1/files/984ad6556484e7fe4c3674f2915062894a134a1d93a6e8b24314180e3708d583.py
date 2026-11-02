"""Independent CPU custody audit; no producer import or texture decode."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
import time
import zipfile

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT/'sources'))
from three_d import _GLB, MAX_TRIANGLE_INDICES

START = time.monotonic()


def require(value, message):
    if not value:
        raise ValueError(message)
    if time.monotonic()-START > 29:
        raise TimeoutError('Independent CPU audit deadline')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def array(glb, index, kind):
    return np.asarray(glb.accessor(index, kind, (5126,)), dtype='<f4')


def primitive(glb, spec):
    attrs = {k: array(glb, index, 'VEC4' if k == 'TANGENT' else 'VEC3' if k in ('POSITION', 'NORMAL') else 'VEC2')
             for k, index in spec['attributes'].items()}
    idx = np.asarray(glb.accessor(spec['indices'], 'SCALAR', (5121, 5123, 5125),
                                 count_limit=MAX_TRIANGLE_INDICES), dtype=np.int64).reshape(-1, 3)
    n = len(attrs['POSITION'])
    require(all(len(a) == n for a in attrs.values()), 'Primitive attribute count mismatch')
    require(idx.min() >= 0 and idx.max() < n and spec.get('mode', 4) == 4, 'Triangle bounds')
    require(np.array_equal(np.unique(idx), np.arange(n)), 'Loose output/source vertex')
    return attrs, idx


def main():
    pins = json.loads((OUT/'inputs.json').read_text())
    paths = {r['role']: ROOT/r['path'] for r in pins['inputs']}
    for row in pins['inputs']:
        require(sha(ROOT/row['path']) == row['sha256'], 'Input hash changed: '+row['role'])
    require(sha(OUT/'sources/three_d.py') == pins['safe_loader_sha256'], 'Safe loader snapshot changed')
    old, new = _GLB(paths['source']), _GLB(paths['candidate'])
    require(len(old.doc['meshes']) == len(new.doc['meshes']) == 1, 'Mesh ownership')
    oldp, newp = old.doc['meshes'][0]['primitives'], new.doc['meshes'][0]['primitives']
    require(len(oldp) == 4 and len(newp) == 6, 'Expected four source and six derivative primitives')
    require(not old.doc.get('skins') and not old.doc.get('animations'), 'Expected unrigged source')
    require(new.binary[:len(old.binary)] == old.binary, 'Original BIN prefix changed')
    for key in ('accessors', 'bufferViews', 'materials'):
        require(new.doc[key][:len(old.doc[key])] == old.doc[key], 'Original '+key+' changed')
    require(len(new.doc['materials']) == 6, 'Unexpected materials')
    for i in (1, 3):
        require(oldp[i] == newp[i], 'Companion primitive changed')
    for i in (0, 2):
        require({k:v for k,v in oldp[i].items() if k not in ('attributes', 'indices')} ==
                {k:v for k,v in newp[i].items() if k not in ('attributes', 'indices')}, 'Retained primitive metadata changed')
    normalized = deepcopy(new.doc)
    normalized['buffers'][0]['byteLength'] = old.doc['buffers'][0]['byteLength']
    for key in ('accessors', 'bufferViews', 'materials'):
        normalized[key] = normalized[key][:len(old.doc[key])]
    normalized['meshes'][0]['primitives'] = deepcopy(oldp)
    require(normalized == old.doc, 'JSON changed outside declared append/remap allowlist')
    oldarrays = [primitive(old, p) for p in oldp]
    newarrays = [primitive(new, p) for p in newp]
    selector = json.loads(paths['selector'].read_text())
    plan = json.loads(paths['plan'].read_text())
    census = json.loads(paths['census'].read_text())
    selected = {(r['source_primitive'], r['source_triangle']) for r in selector['rows']}
    require(len(selected) == len(selector['rows']) == 242 and all(r['selected'] for r in selector['rows']), 'Unique selected face set')
    require(all(pi in (0, 2) for pi, _ in selected), 'Companion selected')
    protected = {(r['current_primitive'], r['current_triangle']) for r in census['rows'] if r['protected_white']}
    selected_current = {(r['current_primitive'], r['current_triangle']) for r in selector['rows']}
    require(len(protected) == 244 and not protected.intersection(selected_current), 'Protected white selection')
    require(selector['source_sha256'] == old.hash and plan['source_model_sha256'] == old.hash, 'Plan bound to another source')
    with zipfile.ZipFile(paths['correspondence']) as z:
        require(sum(r.file_size for r in z.infolist()) <= 4*1024**2 and len(z.infolist()) == 2, 'Mapping archive bounds')
    with np.load(paths['correspondence'], allow_pickle=False) as values:
        require(set(values.files) == {'face_map', 'vertex_map'}, 'Mapping schema')
        fmap, vmap = values['face_map'], values['vertex_map']
    require(fmap.shape == (62581, 4) and vmap.shape == (51358, 4) and
            fmap.dtype == vmap.dtype == np.uint32, 'Mapping shape/type')
    require(len({tuple(map(int, r[:2])) for r in fmap}) == len(fmap) and
            len({tuple(map(int, r[2:])) for r in fmap}) == len(fmap), 'Duplicate face mapping')
    expected_source = {(i, j) for i, (_, faces) in enumerate(oldarrays) for j in range(len(faces))}
    expected_output = {(i, j) for i, (_, faces) in enumerate(newarrays) for j in range(len(faces))}
    require({tuple(map(int, r[:2])) for r in fmap} == expected_source and
            {tuple(map(int, r[2:])) for r in fmap} == expected_output, 'Incomplete triangle coverage')
    routes, mappings = [], {}
    for op, (attrs, idx) in enumerate(newarrays):
        vm = vmap[vmap[:, 0] == op]
        require(len(vm) == len(attrs['POSITION']) and np.array_equal(np.sort(vm[:, 1]), np.arange(len(vm))), 'Incomplete output vertex map')
        vm = vm[np.argsort(vm[:, 1])]
        sp = op if op < 4 else 0 if op == 4 else 2
        require(np.all(vm[:, 2] == sp), 'Source material/vertex ownership changed')
        source_rows = vm[:, 3].astype(np.int64)
        original = oldarrays[sp][0]
        require(source_rows.max() < len(original['POSITION']), 'Source vertex map bounds')
        expected_attrs = set(original) | ({'TEXCOORD_1'} if op >= 4 else set())
        require(set(attrs) == expected_attrs, 'Unexpected attribute added/removed')
        for semantic in ('POSITION', 'NORMAL', 'TANGENT'):
            require(exact(attrs[semantic], original[semantic][source_rows]), semantic+' payload changed')
        if op < 4:
            require(exact(attrs['TEXCOORD_0'], original['TEXCOORD_0'][source_rows]), 'Unselected UV0 changed')
            require(newp[op]['material'] == oldp[sp]['material'], 'Unselected material changed')
        else:
            require(exact(attrs['TEXCOORD_1'], original['TEXCOORD_0'][source_rows]), 'Selected UV1 differs from original UV0')
            material = deepcopy(old.doc['materials'][oldp[sp]['material']])
            material['name'] = ('body_surface' if sp == 0 else 'body_cap')+'__contact_uv_dark_v1'
            route_rows = []
            for container, keys, prefix in [(material, ('normalTexture', 'occlusionTexture', 'emissiveTexture'), ''),
                (material.get('pbrMetallicRoughness', {}), ('baseColorTexture', 'metallicRoughnessTexture'), 'pbrMetallicRoughness.')]:
                for key in keys:
                    if key not in container:
                        continue
                    texture = container[key]
                    require(texture.get('texCoord', 0) == 0, 'Unexpected original texture route')
                    if key != 'baseColorTexture':
                        texture['texCoord'] = 1
                    route_rows.append({'slot': prefix+key, 'texture': texture['index'], 'texCoord': texture.get('texCoord', 0)})
            require(newp[op]['material'] == op and new.doc['materials'][op] == material, 'Clone changed material/image route beyond UV selection')
            palette = np.asarray(plan['palette']['uv_triangle_gltf_v_down'], dtype='<f4')
            require(exact(attrs['TEXCOORD_0'][idx], np.broadcast_to(palette, (len(idx), 3, 2)).copy()), 'New basecolor UV outside exact planned triangle')
            routes.append({'primitive': op, 'source_primitive': sp, 'routes': route_rows,
                           'tangent_basis': 'Exact original N/T including handedness; UV1 exactly original UV0'})
        mappings[op] = (sp, source_rows)
    selected_observed = set()
    rotated = 0
    for sp in range(4):
        for op in range(6):
            rows = fmap[(fmap[:, 0] == sp) & (fmap[:, 2] == op)]
            if not len(rows):
                continue
            require(mappings[op][0] == sp, 'Face/vertex source mismatch')
            a = oldarrays[sp][1][rows[:, 1]]
            b = mappings[op][1][newarrays[op][1][rows[:, 3]]]
            rotation = np.stack([np.all(a == np.roll(b, shift, axis=1), axis=1) for shift in range(3)])
            require(np.all(rotation.any(axis=0)), 'Oriented source triangle changed')
            rotated += int((~rotation[0]).sum())
            pairs = {(sp, int(r[1])) for r in rows}
            if op >= 4:
                selected_observed.update(pairs)
            else:
                require(not selected.intersection(pairs), 'Selected face retained duplicate original material')
    require(selected_observed == selected, 'UV mutation selector mismatch')
    for row in pins['inputs']:
        require(sha(ROOT/row['path']) == row['sha256'], 'Input modified during check')
    return {'schema_version': 1, 'status': 'passed_exact_uv_component_custody',
            'source_sha256': old.hash, 'candidate_sha256': new.hash,
            'triangles': len(fmap), 'referenced_vertices': len(vmap), 'selected_faces': len(selected),
            'protected_white_candidates_current_id_space': len(protected), 'protected_companion_triangles': len(oldarrays[1][1])+len(oldarrays[3][1]),
            'unselected_faces_uv0_exact': len(fmap)-len(selected), 'oriented_triangle_coverage_exact': True,
            'cyclically_reordered_triangles': rotated, 'positions_normals_tangents_exact': True,
            'all_original_material_image_texture_sampler_node_skin_animation_bindings_exact': True,
            'original_binary_prefix_bytes': len(old.binary), 'original_binary_prefix_exact': True,
            'original_accessor_buffer_view_prefix_exact': True, 'no_loose_vertices': True,
            'clone_routes': routes, 'all_original_files_unchanged': True,
            'texture_decodes': 0, 'native_calls': 0, 'quality_approved': False,
            'limitations': ['Unrigged v7 component: no completed skin or animation claim.',
                'Exact source tangent preservation is not a new derivation/normal-map visual validation.',
                'Palette/filter claims are reused from the frozen plan; no image redecoding or GPU sampling test.',
                'No acceptance of residual source geometry, finished coat or complete appearance.'],
            'elapsed_seconds': time.monotonic()-START,
            'process_high_water_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


try:
    result = main()
except Exception as error:
    result = {'status': 'failed', 'type': type(error).__name__, 'error': str(error),
              'elapsed_seconds': time.monotonic()-START}
    data = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with (OUT/'failure.json').open('x') as f:
        f.write(data)
    raise
else:
    data = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with (OUT/'result.json').open('x') as f:
        f.write(data)
    print(json.dumps(result, sort_keys=True))
