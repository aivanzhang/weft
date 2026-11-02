"""Hash-bound source UV repair planning; no image or model mutation."""
from collections import defaultdict
import hashlib
import io
import itertools
import json
from pathlib import Path
import resource
import struct
import sys
import time

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
START = time.monotonic()
PREP = json.loads((OUT / 'prepared.json').read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


class GLB:
    def __init__(self, path):
        self.raw = Path(path).read_bytes()
        require(len(self.raw) <= 64 * 1024**2 and self.raw[:4] == b'glTF', 'Invalid bounded GLB')
        length, kind = struct.unpack_from('<II', self.raw, 12)
        require(kind == 0x4E4F534A and length <= 2 * 1024**2, 'Invalid JSON chunk')
        self.doc = json.loads(self.raw[20:20+length])
        offset = 20 + length
        size, kind = struct.unpack_from('<II', self.raw, offset)
        require(kind == 0x004E4942 and offset + 8 + size == len(self.raw), 'Invalid BIN chunk')
        self.binary = memoryview(self.raw)[offset+8:]

    def array(self, index):
        a = self.doc['accessors'][index]
        require('sparse' not in a and a['count'] <= 300000, 'Unsupported accessor')
        v = self.doc['bufferViews'][a['bufferView']]
        dtype = {5121: '<u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}[a['componentType']]
        width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[a['type']]
        size = np.dtype(dtype).itemsize
        return np.ndarray((a['count'], width), dtype=dtype, buffer=self.binary,
            offset=v.get('byteOffset', 0)+a.get('byteOffset', 0),
            strides=(v.get('byteStride', width*size), size))

    def mesh(self, pi):
        p = self.doc['meshes'][0]['primitives'][pi]
        a = p['attributes']
        return p, self.array(a['POSITION']), self.array(a['TEXCOORD_0']), self.array(p['indices']).reshape(-1, 3)


def linear(rgb):
    value = np.asarray(rgb, dtype=np.float32) / 255
    return np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055)**2.4)


def encoded(rgb):
    return np.where(rgb <= .0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0)**(1/2.4) - .055)


def sample(pixels, uv):
    height, width = pixels.shape[:2]
    points = (np.asarray(uv) % 1) * [width, height] - .5
    lo = np.floor(points).astype(np.int64)
    frac = points - lo
    result = np.zeros((*points.shape[:-1], 3), dtype=np.float32)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        weight = (frac[..., 0] if dx else 1-frac[..., 0]) * (frac[..., 1] if dy else 1-frac[..., 1])
        result += linear(pixels[(lo[..., 1]+dy) % height, (lo[..., 0]+dx) % width]) * weight[..., None]
    return encoded(result)


def box_samples(pixels, uv):
    height, width = pixels.shape[:2]
    x, y = np.asarray(uv) % 1 * [width, height]
    rows = []
    for lod in range(6):
        radius = max(1, 2**lod // 2)
        ix = np.arange(int(np.floor(x))-radius, int(np.floor(x))+radius+1) % width
        iy = np.arange(int(np.floor(y))-radius, int(np.floor(y))+radius+1) % height
        rgb = encoded(linear(pixels[np.ix_(iy, ix)]).mean(axis=(0, 1)))
        rows.append({'approximate_lod': lod, 'square_radius_source_texels': radius, 'rgb_srgb': rgb.tolist()})
    return rows


def main():
    for row in PREP['inputs'].values():
        require(sha(row['path']) == row['sha256'], 'Input changed: ' + row['path'])
    current = GLB(PREP['inputs']['current']['path'])
    source = GLB(PREP['inputs']['v7']['path'])
    saved = json.loads(Path(PREP['inputs']['selectors']['path']).read_text())
    rays = json.loads(Path(PREP['inputs']['rays']['path']).read_text())
    joined = json.loads(Path(PREP['inputs']['pixel_join']['path']).read_text())
    old_plan = json.loads(Path(PREP['inputs']['v7_plan']['path']).read_text())
    geom = json.loads(Path(PREP['inputs']['geometry_selector']['path']).read_text())
    provenance = {(r['primitive'], r['triangle']): r for rows in saved['regions'].values() for r in rows}
    witnesses = {(r['primitive'], r['triangle']): r for r in joined['matches']}
    require(len(witnesses) == 17, 'Expected 17 distinct bound witnesses')
    base = source.doc['materials'][0]['pbrMetallicRoughness']['baseColorTexture']
    image = source.doc['images'][source.doc['textures'][base['index']]['source']]
    view = source.doc['bufferViews'][image['bufferView']]
    blob = source.binary[view.get('byteOffset', 0):view.get('byteOffset', 0)+view['byteLength']]
    texture_sha = hashlib.sha256(blob).hexdigest()
    require(texture_sha == saved['texture_sha256'] == old_plan['basecolor_original_encoded_sha256'], 'Texture mismatch')
    # Decode exactly one image; retain uint8 pixels, not a full float texture.
    with Image.open(io.BytesIO(blob)) as im:
        require(im.size == (4096, 4096) and im.mode in ('RGB', 'RGBA') and getattr(im, 'n_frames', 1) == 1, 'Unsupported source image')
        pixels = np.array(im.convert('RGB'), dtype=np.uint8)
    require(pixels.nbytes == 4096 * 4096 * 3, 'Unexpected decoded budget')

    primitive_map = {0: 0, 2: 2, 4: 0, 5: 0, 6: 2}
    low = np.asarray(PREP['settings']['region_blender_m']['min'])
    high = np.asarray(PREP['settings']['region_blender_m']['max'])
    bary = np.asarray([(i/12, j/12, 1-(i+j)/12) for i in range(13) for j in range(13-i)])
    candidates = []
    for pi in primitive_map:
        primitive, p, uv, faces = current.mesh(pi)
        blender = p[:, [0, 2, 1]].copy(); blender[:, 1] *= -1
        center = blender[faces].mean(axis=1)
        ids = np.flatnonzero(np.all((center >= low) & (center <= high), axis=1))
        require(len(ids) <= 5000, 'Candidate face cap exceeded')
        for start in range(0, len(ids), 64):
            batch = ids[start:start+64]
            tex = uv[faces[batch]]
            colors = sample(pixels, np.einsum('bc,fcd->fbd', bary, tex))
            mean = colors.mean(axis=2)
            green = (colors[..., 1] > np.maximum(colors[..., 0], colors[..., 2]) * 1.06 + .006)
            white = (mean >= .5) & (np.ptp(colors, axis=2) <= .18)
            for j, fid in enumerate(batch):
                fid = int(fid); key = (pi, fid); prov = provenance.get(key, {})
                confirmed = key in witnesses
                white_fraction = float(white[j].mean())
                protected_white = white_fraction >= .25 and not confirmed
                cap = prov.get('kind') == 'authored_closure' and prov.get('closure_loop') in (1, 4)
                reasons = []
                if confirmed: reasons.append('actual_pixel_witness')
                if int(green[j].sum()) and not protected_white: reasons.append('dense_green_contact_candidate')
                if cap and not protected_white: reasons.append('same_dark_closure_loop_as_actual_witness')
                candidates.append({'current_primitive': pi, 'current_triangle': fid,
                    'material': current.doc['materials'][primitive['material']]['name'],
                    'center_blender_m': center[fid].tolist(),
                    'area_m2': float(np.linalg.norm(np.cross(p[faces[fid,1]]-p[faces[fid,0]], p[faces[fid,2]]-p[faces[fid,0]]))*.5),
                    'dense_green_samples': int(green[j].sum()), 'dense_white_fraction': white_fraction,
                    'protected_white': protected_white, 'selected': bool(reasons), 'reasons': reasons,
                    'provenance': {k: prov[k] for k in ('kind', 'source_face_id', 'closure_loop', 'closure_triangle', 'expected_face_id') if k in prov}})
    require(set(witnesses) <= {(r['current_primitive'], r['current_triangle']) for r in candidates}, 'Region misses actual witness')
    chosen = [r for r in candidates if r['selected']]
    require(17 <= len(chosen) <= 1500, 'Selected face cap exceeded')

    # Quantized centroid bins only index candidates; full oriented geometry + UV decide each match.
    bins = {}
    for pi in (0, 2):
        primitive, p, uv, f = source.mesh(pi)
        index = defaultdict(list)
        for fid, q in enumerate(np.floor(p[f].mean(axis=1) / .001).astype(np.int32)):
            index[tuple(q)].append(fid)
        bins[pi] = (p, uv, f, index)
    neighbor = list(itertools.product((-1, 0, 1), repeat=3))
    for row in chosen:
        pi, fid = row['current_primitive'], row['current_triangle']
        _, cp, cuv, cf = current.mesh(pi)
        tri, tex = cp[cf[fid]], cuv[cf[fid]]
        spi = primitive_map[pi]; p, uv, faces, index = bins[spi]
        q = np.floor(tri.mean(0) / .001).astype(np.int32)
        possibles = set(itertools.chain.from_iterable(index.get(tuple(q+offset), []) for offset in neighbor))
        matches = []
        for sf in sorted(possibles):
            for shift in range(3):
                if np.max(np.abs(np.roll(p[faces[sf]], shift, axis=0)-tri)) <= 2e-7 and np.max(np.abs(np.roll(uv[faces[sf]], shift, axis=0)-tex)) <= 2e-7:
                    matches.append((sf, shift))
        require(len(matches) == 1, 'Source correspondence missing or ambiguous')
        sf, shift = matches[0]
        row.update(source_primitive=spi, source_triangle=sf, source_vertex_rows=faces[sf].tolist(),
            source_corner_to_current_rotation=shift, source_uv0=uv[faces[sf]].tolist(),
            source_triangle_blender_m=(p[faces[sf]][:, [0, 2, 1]] * [1, -1, 1]).tolist())

    palette_levels = []
    for size in (128, 96, 64):
        best = None; passing = 0
        for y in range(0, 4096-size+1, 64):
            for x in range(0, 4096-size+1, 64):
                tile = pixels[y:y+size, x:x+size]
                lo, hi = tile.min(axis=(0, 1)), tile.max(axis=(0, 1))
                # Cheap extrema screen prevents large float allocations on unsuitable blocks.
                if int(hi.max()) > 38 or int(lo.min()) < 3:
                    continue
                rgb = tile.astype(np.float32) / 255
                variance = rgb.var(axis=(0, 1)); mean = rgb.mean(axis=(0, 1))
                chroma = float(np.ptp(rgb, axis=2).max())
                green_excess = float((rgb[..., 1]-np.maximum(rgb[..., 0], rgb[..., 2])).max())
                if chroma > .04 or green_excess > .012 or float(variance.sum()) > .00012 or float(mean.mean()) > .1:
                    continue
                passing += 1
                score = float(np.sum((mean-np.asarray([.055, .055, .06]))**2) + 32*variance.sum())
                if best is None or score < best['score']:
                    core = [x+size//2-4, y+size//2-4, x+size//2+4, y+size//2+4]
                    best = {'score': score, 'source_pixel_rect': [x, y, x+size, y+size], 'core_pixel_rect': core,
                        'mean_srgb': mean.tolist(), 'variance_srgb': variance.tolist(), 'min_rgb8': lo.tolist(), 'max_rgb8': hi.tolist(),
                        'max_per_pixel_srgb_chroma': chroma, 'max_per_pixel_green_excess': green_excess,
                        'all_pixels_checked': size*size, 'padding_texels': (size-8)//2,
                        'maximum_certified_trilinear_lod': int(np.floor(np.log2((size-8)//2)))-1}
        palette_levels.append({'tile_size': size, 'passing_tiles': passing, 'best': best})
    palette = next((r['best'] for r in palette_levels if r['best'] is not None), None)
    require(palette is not None, 'No padded dark neutral palette satisfies the frozen gates')
    rect = palette['core_pixel_rect']; x0, y0, x1, y1 = rect
    new_uv = np.asarray([[x0+.5, y0+.5], [x1-.5, y0+.5], [x0+.5, y1-.5]]) / 4096
    palette['uv_triangle_gltf_v_down'] = new_uv.tolist()
    palette['uv_triangle_blender_v_up'] = (new_uv * [1, -1] + [0, 1]).tolist()
    palette['filter_scope'] = 'Every source texel in padded rectangle meets gates; positive bilinear/BOX-mip combinations inside it stay within extrema. Trilinear support bound is conservative. GPU anisotropy, cross-face antialiasing and larger LOD are not certified.'

    by_key = {(r['current_primitive'], r['current_triangle']): r for r in chosen}
    witness_checks = []
    for ray, join in zip(rays['pixels'], joined['matches']):
        require(ray['pixel'] == join['pixel'], 'Witness ordering changed')
        row = by_key[(join['primitive'], join['triangle'])]
        old = np.asarray(ray['uv0_corners']); old[:, 1] = 1-old[:, 1]
        uv = np.asarray(ray['barycentric']) @ old
        witness_checks.append({'pixel': ray['pixel'], 'observed_kind': ray['kind'],
            'source_selector': [row['source_primitive'], row['source_triangle']], 'selected': True,
            'old_lod0_srgb': sample(pixels, uv).tolist(), 'old_box_footprints': box_samples(pixels, uv),
            'new_lod0_srgb': sample(pixels, np.asarray(ray['barycentric']) @ new_uv).tolist(),
            'limitation': 'Old boxes are CPU footprint probes, not a reproduction of native mip/lighting/AA.'})

    geometry_faces = {tuple(x) for x in geom['replacement_source_faces']} | {tuple(x) for x in geom['retired_without_replacement_source_faces']}
    reference_faces = {tuple(x['source_face']) for x in geom['healthy_remaining_original_body_reference_triangles']}
    selected_faces = {(r['source_primitive'], r['source_triangle']) for r in chosen}
    split_report = []
    for spi in (0, 2):
        _, _, _, f = source.mesh(spi)
        selected_ids = sorted(fid for pi, fid in selected_faces if pi == spi)
        mask = np.zeros(len(f), dtype=bool); mask[selected_ids] = True
        selected_vertices = np.unique(f[mask]); protected_vertices = np.unique(f[~mask])
        shared = np.intersect1d(selected_vertices, protected_vertices)
        split_report.append({'source_primitive': spi, 'selected_faces': selected_ids,
            'selected_source_vertex_rows': selected_vertices.tolist(), 'boundary_uv_seam_vertices': shared.tolist(),
            'new_independent_corner_rows': len(selected_ids)*3,
            'target_material': ('body_surface' if spi == 0 else 'body_cap')+'__contact_uv_dark_v1',
            'non_basecolor_routes_to_copy_uv1': ['normalTexture', 'pbrMetallicRoughness.metallicRoughnessTexture'] if spi == 0 else []})
    uv_routes = []
    for pi in (5, 6):
        p = current.doc['meshes'][0]['primitives'][pi]
        uv_routes.append({'primitive': pi, 'uv0_uv1_equal': bool(np.array_equal(current.array(p['attributes']['TEXCOORD_0']), current.array(p['attributes']['TEXCOORD_1']))),
            'normal_texture': current.doc['materials'][p['material']].get('normalTexture')})
    selected_area = sum(r['area_m2'] for r in chosen)
    report = {'schema_version': 1, 'status': 'prepared_source_uv_repair', 'model_edited': False, 'image_edited': False,
        'source_model_sha256': PREP['inputs']['v7']['sha256'], 'current_model_sha256': PREP['inputs']['current']['sha256'],
        'texture_sha256': texture_sha, 'texture_decode_calls': 1, 'candidate_faces': len(candidates), 'selected_faces': len(chosen),
        'selected_area_m2': selected_area, 'actual_witness_coverage': {'covered': len(witness_checks), 'total': 17, 'includes_actual_closure_hits': 2},
        'protected_white_candidate_faces': sum(r['protected_white'] for r in candidates),
        'protected_white_intersection': sum(r['protected_white'] and r['selected'] for r in candidates),
        'explicit_light_witness_overrides': [r['source_selector'] for r in witness_checks if r['observed_kind'] == 'light'],
        'selected_companion_faces': 0, 'outside_region_selected': 0,
        'source_correspondence': 'Unique oriented face positions + UV corners <=2e-7; no normal/color/desired-weight tie breaking',
        'geometry_repair_intersection': [list(x) for x in sorted(selected_faces & geometry_faces)],
        'geometry_repair_target_faces': len(geometry_faces),
        'geometry_healthy_reference_intersection': [list(x) for x in sorted(selected_faces & reference_faces)],
        'geometry_healthy_reference_faces': len(reference_faces), 'palette': palette, 'current_normal_routes': uv_routes,
        'future_mutation_contract': {'new_textured_materials': [x['target_material'] for x in split_report if x['selected_faces']],
            'preserve': ['all geometric triangle positions', 'NORMAL', 'TANGENT', 'skin', 'animations', 'original materials', 'original image bytes', 'all unselected UV0', 'existing UV1 if present'],
            'allowed': ['selected face UV0', 'duplicate identical source corner rows for explicit seams', 'selected textured material clones', 'copied original UV0 as UV1 on clones', 'selected-clone non-basecolor texCoord1'],
            'requires': ['Rebuild native/portable fur colors from repaired source UV0', 'Keep selected material undercoat handling explicit; overlapping UV0 must use authored UV1 or fail', 'Independent source normal/geometry fixes composed using exact face mapping', 'Native basecolor/appearance review before acceptance']},
        'limits': PREP['limits'], 'elapsed_seconds': time.monotonic()-START,
        'self_peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024),
        'quality_approved': False,
        'limitations': ['Dense 91-point color sampling is a discovery aid, not complete texture integration; exact actual witnesses override it.',
            'The observed light defect is an explicit authored dark-surface exception, not inferred anatomy.',
            'This selected set covers all17 confirmed pixels and recorded dark closure loops, not all future views or unseen contact paint.',
            'No geometry/topology/normal defect is repaired by a UV edit.']}
    for row in PREP['inputs'].values():
        require(sha(row['path']) == row['sha256'], 'Input changed during planning')
    write('candidate-census.json', {'sampling_barycentric': bary.tolist(), 'rows': candidates})
    write('face-selector.json', {'source_sha256': report['source_model_sha256'], 'mesh': 0, 'rows': chosen, 'primitive_splits': split_report})
    write('palette-search.json', {'levels': palette_levels, 'selected': palette})
    write('witness-coverage.json', {'rows': witness_checks})
    write('summary.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'selected_faces', 'actual_witness_coverage', 'protected_white_candidate_faces', 'protected_white_intersection', 'geometry_repair_intersection', 'elapsed_seconds', 'self_peak_rss_bytes')}))


try:
    main()
except Exception as error:
    write('failure.json', {'status': 'failed', 'error': str(error), 'type': type(error).__name__, 'elapsed_seconds': time.monotonic()-START, 'model_edited': False})
    raise
