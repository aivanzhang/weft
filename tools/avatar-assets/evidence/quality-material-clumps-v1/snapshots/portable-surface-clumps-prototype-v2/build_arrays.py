"""One actual-source CPU prototype; no model patch, renderer or provider calls."""
import hashlib
import collections
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE/'sources/scripts'))
from surface_clumps import build_surface_clumps, SurfaceClumpError
from groom_flow import prepare_flow
from fringe_color_atlas import sample_source_rgb
from avatar_assets.fringe_export import _array
from avatar_assets.three_d import _GLB

PREP = json.loads((HERE/'prepared.json').read_text())
PATHS = {key: ROOT/value['path'] for key, value in PREP['inputs'].items()}
START = time.monotonic()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, data):
    with (HERE/name).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True); stream.write('\n')


def npz(path):
    with zipfile.ZipFile(path) as z:
        require(len(z.infolist()) <= 20 and sum(i.file_size for i in z.infolist()) <= 16*1024**2,
                'Input NPZ exceeds bounds')
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def image_bytes(glb, image_index):
    image = glb.doc['images'][image_index]
    view = glb.doc['bufferViews'][image['bufferView']]
    a, n = view.get('byteOffset', 0), view['byteLength']
    require(n <= 32*1024**2, 'Packed image exceeds32MiB')
    return glb.binary[a:a+n]


report = {'schema_version': 1, 'status': 'incomplete', 'quality_approved': False,
          'native_provider_ai_role_calls': 0, 'model_written': False}
try:
    require(all(sha(PATHS[k]) == v['sha256'] for k, v in PREP['inputs'].items()), 'Input pin changed')
    data = npz(PATHS['source_arrays']); fringe = npz(PATHS['fringe_npz'])
    selected = data.pop('surface_fin_ids'); data.pop('surface_face_pairs'); data.pop('raw_primitive_face')
    skin = json.loads(PATHS['skin_report'].read_text()); owner = skin['owners'][1]
    raw = _GLB(PATHS['raw_model']); final = _GLB(PATHS['final_model'])
    require(raw.doc['meshes'][1]['name'] == final.doc['meshes'][1]['name'] == owner['mesh_name'], 'Coat mesh identity differs')
    primitive_raw, primitive_final = raw.doc['meshes'][1]['primitives'][0], final.doc['meshes'][1]['primitives'][0]
    ids = _array(raw, primitive_raw['attributes']['_AUTHOR_VERTEX_ID'], 'SCALAR', {5126}).ravel()
    require(np.all(ids == np.floor(ids)) and np.array_equal(np.sort(ids),
            np.arange(owner['vertex_start'], owner['vertex_start']+owner['vertex_count'])), 'Raw coat source ID coverage differs')
    helper_ids = fringe['native_vertex_to_helper_vertex'][ids.astype(np.int32)-owner['vertex_start']]
    first = np.full(len(fringe['positions']), -1, dtype=np.int32)
    for vertex, helper_id in enumerate(helper_ids):
        if first[helper_id] == -1:
            first[helper_id] = vertex
    require(np.all(first >= 0), 'Actual export does not cover each frozen helper vertex')
    mapped = {}
    for name, attribute, kind, components in (('positions', 'POSITION', 'VEC3', {5126}),
            ('normals', 'NORMAL', 'VEC3', {5126}), ('uv_gltf', 'TEXCOORD_0', 'VEC2', {5126}),
            ('joints', 'JOINTS_0', 'VEC4', {5121, 5123}), ('weights', 'WEIGHTS_0', 'VEC4', {5126})):
        values = _array(final, primitive_final['attributes'][attribute], kind, components)
        require(len(values) == len(helper_ids), 'Final coat vertex order/count differs')
        if name in ('positions', 'normals'):
            values = values[:, [0, 2, 1]].copy(); values[:, 1] *= -1
        require(np.array_equal(values, values[first[helper_ids]]), 'Actual duplicate corner attributes disagree: '+name)
        mapped[name] = values[first].copy()
    actual_bones = [final.doc['nodes'][j]['name'] for j in final.doc['skins'][0]['joints']]
    require(actual_bones == skin['bone_names'], 'Actual final/source skin bone order differs')
    expected_uv = fringe['uv_gltf'].copy()
    expected_uv[:, 1] = np.float32(1)-(np.float32(1)-expected_uv[:, 1])
    require(np.array_equal(mapped['positions'], fringe['positions'])
            and np.array_equal(mapped['uv_gltf'], expected_uv), 'Frozen fin positions or exact V-flip UVs differ from final model')
    actual_triangles = _array(final, primitive_final['indices'], 'SCALAR', {5121, 5123, 5125}).reshape(-1, 3)
    helper_triangles = helper_ids[actual_triangles]
    owners = helper_triangles//4
    require(np.all(owners == owners[:, :1]), 'Actual triangle crosses fin ownership')
    selected_set = set(map(int, selected))
    retained = np.array([int(fin) not in selected_set for fin in owners[:, 0]])
    def canonical(triangle):
        value = tuple(map(int, triangle))
        return min(value, value[1:]+value[:1], value[2:]+value[:2])
    actual_oriented = collections.Counter(canonical(t) for t in helper_triangles[retained])
    expected_oriented = collections.Counter()
    selected_set = set(map(int, selected))
    for fin in range(len(first)//4):
        if fin not in selected_set:
            for triangle in ((0, 1, 3), (0, 3, 2), (3, 1, 0), (2, 3, 0)):
                expected_oriented[canonical([fin*4+x for x in triangle])] += 1
    require(actual_oriented == expected_oriented, 'Retained oriented contour triangles differ')
    fringe.update(mapped)
    material = final.doc['materials'][primitive_final['material']]
    require(material['name'] == 'SourceColoredFringe' and 'normalTexture' not in material
            and material.get('alphaMode') == 'MASK' and not material.get('doubleSided', False), 'Coat material route differs')
    tex_index = material['pbrMetallicRoughness']['baseColorTexture']['index']
    atlas_image_index = final.doc['textures'][tex_index]['source']
    require(hashlib.sha256(image_bytes(final, atlas_image_index)).hexdigest() == sha(PATHS['atlas']), 'Actual final atlas differs')
    with Image.open(PATHS['atlas']) as im:
        require(im.mode == 'RGBA' and im.size == (4096, 1824), 'Actual atlas format/dimensions differ')
        atlas = np.asarray(im).copy()
    source_materials = skin['owners'][0]['materials']
    final_names = [m['name'] for m in final.doc['materials']]
    textures, reads, source_pixels = {}, [], [0]

    def sampler(slot, uv):
        material_name = source_materials[slot]
        match = [index for index, name in enumerate(final_names) if name == material_name]
        require(len(match) == 1, 'Source material lookup is ambiguous')
        m = final.doc['materials'][match[0]]
        require(m.get('alphaMode', 'OPAQUE') == 'OPAQUE', 'Source color must be opaque')
        pbr = m['pbrMetallicRoughness']; texture_info = pbr['baseColorTexture']
        require(texture_info.get('texCoord', 0) == 0 and not texture_info.get('extensions'), 'Source color must use direct UV0')
        texture = final.doc['textures'][texture_info['index']]
        sampling = final.doc['samplers'][texture['sampler']]
        require(sampling.get('wrapS', 10497) == sampling.get('wrapT', 10497) == 10497
                and sampling.get('magFilter', 9729) == 9729, 'Source sampling route differs')
        image_id = texture['source']
        if image_id not in textures:
            packed = image_bytes(final, image_id)
            with Image.open(io.BytesIO(packed)) as image:
                require(image.format in ('PNG', 'JPEG') and max(image.size) <= 4096, 'Source texture format/size exceeds bounds')
                source_pixels[0] += image.width*image.height
                require(source_pixels[0] <= 16_777_216, 'Source texture aggregate exceeds16MP')
                textures[image_id] = np.asarray(image.convert('RGB')).copy()
            reads.append({'material': material_name, 'image': image_id, 'packed_sha256': hashlib.sha256(packed).hexdigest(),
                          'dimensions': [textures[image_id].shape[1], textures[image_id].shape[0]]})
        factor = np.asarray(pbr.get('baseColorFactor', [1, 1, 1, 1])[:3], dtype=float)
        return sample_source_rgb(textures[image_id], np.asarray(uv, dtype=float), factor)

    config = json.loads(PATHS['finish_config'].read_text())
    flow = prepare_flow(data['positions'].min(axis=0), np.ptp(data['positions'], axis=0),
                        config['groom']['direction'], config['groom']['comb'], config['groom_flow'])
    settings = {key: PREP['settings'][key] for key in ('profile', 'seed', 'lift_m')}
    settings['surface_fin_ids'] = selected
    report['actual_contour_binding'] = {'source_model_sha256': final.hash, 'helper_vertices': len(first),
        'actual_corner_rows': len(helper_ids), 'duplicates_identical': True, 'bone_names': actual_bones,
        'retained_oriented_triangles_exact': True, 'uv_admission': 'Exact float32 V flip roundtrip, no epsilon',
        'policy': 'Retained contour attributes come from final GLB, not pre-export approximations'}
    write('actual-input-checkpoint.json', report)
    del raw
    result = build_surface_clumps(data, fringe, atlas, flow, sampler, settings,
                                  source_export_vertices=51709, source_export_triangles=62581)
    pixels, component = result.pop('pixels'), result.pop('report')
    Image.fromarray(pixels).save(HERE/'portable-surface-clumps-atlas.png')
    np.savez_compressed(HERE/'component-arrays.npz', **result)
    report.update(status='cpu_arrays_built_pending_root_review', component=component,
        source_images_decoded=reads, array_sha256=sha(HERE/'component-arrays.npz'),
        atlas_sha256=sha(HERE/'portable-surface-clumps-atlas.png'),
        coordinate_contract=PREP['coordinate_contract'])
except SurfaceClumpError as exc:
    report.update(status='unsupported_cpu_candidate', component=exc.report,
                  source_images_decoded=reads if 'reads' in locals() else [])
except (ValueError, KeyError, AssertionError) as exc:
    report.update(status='unsupported_input_or_execution', failure=str(exc))
report.update(elapsed_seconds=time.monotonic()-START,
              inputs_unchanged=all(sha(PATHS[k]) == v['sha256'] for k, v in PREP['inputs'].items()))
write('result.json', report)
print(json.dumps(report, indent=2))
