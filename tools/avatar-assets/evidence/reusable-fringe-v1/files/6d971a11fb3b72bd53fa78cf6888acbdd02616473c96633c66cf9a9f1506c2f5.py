"""Independent CPU checks of actual exported adapter geometry, textures and skin."""

import hashlib
import io
import json
from pathlib import Path
import struct
import time

import numpy as np
from PIL import Image


def check(value, message):
    if not value:
        raise ValueError(message)


def convert(points):
    return np.asarray(points)[..., [0, 2, 1]] * [1, 1, -1]


def canonical(face):
    face = tuple(map(int, face))
    return min(face, face[1:]+face[:1], face[2:]+face[:2])


def rotation(q):
    x, y, z, w = np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def sample_rgb(pixels, uv, repeat):
    pixels = np.asarray(pixels)
    xy = (np.mod(uv, 1) if repeat else uv)*[pixels.shape[1], pixels.shape[0]]-.5
    low = np.floor(xy).astype(int)
    fraction = xy-low
    output = np.zeros((len(uv), 4))
    for dx in (0, 1):
        for dy in (0, 1):
            xx, yy = low[:, 0]+dx, low[:, 1]+dy
            if repeat:
                xx, yy = xx % pixels.shape[1], yy % pixels.shape[0]
            else:
                xx, yy = xx.clip(0, pixels.shape[1]-1), yy.clip(0, pixels.shape[0]-1)
            values = pixels[yy, xx].astype(float)/255
            rgb = values[:, :3]
            rgb = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
            alpha = values[:, 3] if values.shape[1] == 4 else np.ones(len(values))
            weight = (fraction[:, 0] if dx else 1-fraction[:, 0])*(fraction[:, 1] if dy else 1-fraction[:, 1])
            output += np.column_stack([rgb, alpha])*weight[:, None]
    return output


def verify(out):
    started = time.monotonic()
    out = Path(out)
    model = out/'model.glb'
    check(28 <= model.stat().st_size < 64*1024**2, 'Model byte bounds')
    blob = model.read_bytes()
    check(blob[:4] == b'glTF' and struct.unpack_from('<I', blob, 8)[0] == len(blob), 'GLB header')
    length, kind = struct.unpack_from('<I4s', blob, 12)
    check(kind == b'JSON' and length < 2*1024**2, 'GLB JSON chunk')
    doc = json.loads(blob[20:20+length])
    size, kind = struct.unpack_from('<I4s', blob, 20+length)
    check(kind == b'BIN\0', 'Embedded BIN required')
    binary = memoryview(blob)[28+length:28+length+size]
    check(len(doc['animations']) == 1, 'One real action required')
    check(not any('uri' in row for row in doc.get('images', [])+doc.get('buffers', [])), 'No external data')

    def accessor(index):
        a = doc['accessors'][index]
        check(not a.get('sparse') and 0 < a['count'] <= 100000, 'Accessor bounds')
        v = doc['bufferViews'][a['bufferView']]
        columns = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
        dtype = np.dtype({5121: '<u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}[a['componentType']])
        offset, stride = v.get('byteOffset', 0)+a.get('byteOffset', 0), v.get('byteStride', columns*dtype.itemsize)
        check(offset+(a['count']-1)*stride+columns*dtype.itemsize <= len(binary), 'Accessor storage')
        result = np.ndarray((a['count'], columns), dtype=dtype, buffer=binary, offset=offset,
                            strides=(stride, dtype.itemsize)).copy()
        if a.get('normalized'):
            result = result.astype(float)/np.iinfo(dtype).max
        return result

    def texture_image(material):
        binding = material['pbrMetallicRoughness']['baseColorTexture']
        check(binding.get('texCoord', 0) == 0, 'Expected UV0')
        texture = doc['textures'][binding['index']]
        image = doc['images'][texture['source']]
        view = doc['bufferViews'][image['bufferView']]
        raw = bytes(binary[view.get('byteOffset', 0):view.get('byteOffset', 0)+view['byteLength']])
        with Image.open(io.BytesIO(raw)) as im:
            check(im.width*im.height <= 16777216, 'Image limit')
            pixels = np.asarray(im.convert('RGBA'))
        return pixels

    expected = np.load(out/'expected-source.npz')
    fins = np.load(out/'fringe-correspondence.npz')
    native = np.load(out/'native-corner-attributes.npz')
    native_map = fins['native_vertex_to_helper_vertex']
    check(np.array_equal(native_map, fins['triangles'].ravel()), 'Native corner correspondence changed')
    check(np.array_equal(native['native_vertex_to_helper_vertex'], native_map), 'Native attribute correspondence changed')
    check(np.array_equal(native['positions'], fins['positions'][native_map]), 'Native corner position copy changed')
    check(np.max(np.abs(native['uv']-fins['uv_gltf'][native_map])) < 1e-7, 'Native corner UV copy changed')
    check(np.max(np.abs(native['normals']-fins['normals'][native_map])) < .0002, 'Native corner normal copy changed')
    helper_first = np.asarray([np.flatnonzero(native_map == i)[0] for i in range(len(fins['positions']))])
    for pose in expected['posed_coat']:
        check(np.array_equal(pose, pose[helper_first][native_map]), 'Duplicated native corners diverge during motion')
    source_points = convert(expected['points'])
    fin_points = convert(fins['positions'])
    check(len(fins['positions']) == 4*len(fins['root_vertex_ids']), 'Four source vertices per fin')
    check(expected['positive_mask'][fins['root_vertex_ids']].all(), 'Excluded root region leaked')
    raw_uv = expected['raw_uv_topdown'][fins['root_vertex_ids']]
    check(np.max(np.abs(raw_uv-fins['root_source_uv'])) < 1e-7, 'Source UV orientation/interpolation changed')
    check(np.any(raw_uv < 0) and set(fins['material_ids']) == {0, 1}, 'Fixture lacks negative UV or either material')
    mesh_nodes = [(i, n) for i, n in enumerate(doc['nodes']) if 'mesh' in n]
    check(len(mesh_nodes) == 2, 'Exactly source and fringe mesh nodes required')
    by_name = {n['name']: (i, n) for i, n in mesh_nodes}
    check(set(by_name) == {'AnalyticSurface', 'PortableEdgeFringe'}, 'Unexpected mesh identities')
    retained_triangles = []
    measurements = {'normal_error': 0., 'uv_error': 0., 'skin_error': 0., 'pose_error_m': 0., 'root_error_m': 0.}
    decoded = {}
    atlas = None
    for name, (node_id, node) in by_name.items():
        check('skin' in node, 'Both meshes need a real skin')
        skin = doc['skins'][node['skin']]
        joint_names = [doc['nodes'][i]['name'] for i in skin['joints']]
        check(set(joint_names) == {'Root', 'Tip'}, 'Two-bone skin mapping changed')
        decoded[name] = []
        for primitive in doc['meshes'][node['mesh']]['primitives']:
            attributes = primitive['attributes']
            check({'POSITION', 'NORMAL', 'TANGENT', 'TEXCOORD_0', 'JOINTS_0', 'WEIGHTS_0'} <= set(attributes), 'Missing exported attribute')
            p, n, tangent, uv, j, w = [accessor(attributes[k]) for k in
                                     ('POSITION', 'NORMAL', 'TANGENT', 'TEXCOORD_0', 'JOINTS_0', 'WEIGHTS_0')]
            check(np.isfinite(np.c_[p, n, tangent, uv, w]).all(), 'Nonfinite exported data')
            check(np.max(np.abs(np.linalg.norm(tangent[:, :3], axis=1)-1)) < .001 and
                  np.isin(tangent[:, 3], [-1, 1]).all(), 'Invalid tangent frame')
            tri = accessor(primitive['indices']).reshape(-1, 3)
            material = doc['materials'][primitive['material']]
            skin_weight = np.zeros((len(p), 2))
            for column in range(4):
                for bone, label in enumerate(('Root', 'Tip')):
                    skin_weight[:, bone] += np.where(j[:, column] == joint_names.index(label), w[:, column], 0)
            if name == 'AnalyticSurface':
                distances = np.max(np.abs(p[:, None]-source_points[None]), axis=2)
                mapping = distances.argmin(axis=1)
                check(distances[np.arange(len(p)), mapping].max() < 1e-8, 'Source positions changed')
                slot = int(material['name'][-1])
                pixels = texture_image(material)
                original = np.asarray(Image.open(out/f'source-{slot}.png').convert('RGBA'))
                check(np.array_equal(pixels, original), 'Source texture pixels changed')
                expected_uv = expected['raw_uv_topdown'][mapping]
                measurements['uv_error'] = max(measurements['uv_error'], float(np.max(np.abs(uv-expected_uv))))
                expected_w = expected['weights'][mapping]
                retained_triangles.extend((slot, canonical(row)) for row in mapping[tri])
            else:
                check(material.get('alphaMode') == 'MASK' and abs(material.get('alphaCutoff', .5)-.18) < 1e-6
                      and material.get('doubleSided', False) is False, 'Fringe MASK/culling contract')
                check('COLOR_0' not in attributes and 'normalTexture' not in material, 'Unexpected fin tint/normal texture')
                pbr = material['pbrMetallicRoughness']
                check(pbr.get('metallicFactor', 1) == 0 and abs(pbr.get('roughnessFactor', 1)-.92) < 1e-6, 'Fringe PBR contract')
                check('baseColorFactor' not in pbr or np.allclose(pbr['baseColorFactor'], 1), 'Unexpected double tint')
                distances = np.max(np.abs(p[:, None]-fin_points[None]), axis=2)+np.max(np.abs(uv[:, None]-fins['uv_gltf'][None]), axis=2)
                mapping = distances.argmin(axis=1)
                check(distances[np.arange(len(p)), mapping].max() < 1e-7, 'Exported fin position/UV mismatch')
                check(sorted(canonical(row) for row in mapping[tri]) == sorted(canonical(row) for row in fins['triangles']), 'Reversed/original fin triangles changed')
                error = np.max(np.abs(n-convert(fins['normals'])[mapping]))
                measurements['normal_error'] = max(measurements['normal_error'], float(error))
                expected_w = np.zeros((len(mapping), 2))
                for column in range(4):
                    for bone in range(2):
                        expected_w[:, bone] += np.where(fins['joints'][mapping, column] == bone,
                                                       fins['weights'][mapping, column], 0)
                atlas = texture_image(material)
                saved_atlas = np.asarray(Image.open(out/'textures/portable-fringe-color-atlas.png').convert('RGBA'))
                check(np.array_equal(atlas, saved_atlas), 'Embedded atlas differs from saved PNG')
            measurements['skin_error'] = max(measurements['skin_error'], float(np.max(np.abs(skin_weight-expected_w))))
            decoded[name].append({'node': node_id, 'skin': skin, 'positions': p, 'joints': j, 'weights': w, 'mapping': mapping})
    expected_triangles = [(int(slot), canonical(face)) for slot, face in zip(expected['material_ids'], expected['faces'])]
    check(sorted(retained_triangles) == sorted(expected_triangles), 'Source topology/material assignment changed')
    check(measurements['uv_error'] < 1e-7 and measurements['normal_error'] < .0002 and measurements['skin_error'] < 2e-6,
          'Exported attribute preservation threshold')
    check(atlas is not None, 'Missing embedded fringe atlas')
    parents = {child: i for i, node in enumerate(doc['nodes']) for child in node.get('children', [])}
    animation = doc['animations'][0]
    check(animation.get('name') == 'analytic_bend', 'Action identity changed')
    channels = []
    for channel in animation['channels']:
        sampler = animation['samplers'][channel['sampler']]
        check(sampler.get('interpolation', 'LINEAR') in ('LINEAR', 'STEP'), 'Unexpected interpolation')
        channels.append((channel['target'], accessor(sampler['input']).ravel(), accessor(sampler['output'])))
    for ti, seconds in enumerate(expected['times']):
        overrides, globals_ = {}, {}
        for target, times, values in channels:
            near = np.flatnonzero(np.abs(times-seconds) < 1e-6)
            check(len(near) == 1, 'Expected actual baked sample time')
            overrides[(target['node'], target['path'])] = values[near[0]]

        def global_matrix(index):
            if index in globals_:
                return globals_[index]
            node = doc['nodes'][index]
            if 'matrix' in node:
                local = np.asarray(node['matrix']).reshape(4, 4).T
                check(not any(k[0] == index for k in overrides), 'Matrix/TRS animation ambiguity')
            else:
                t = overrides.get((index, 'translation'), node.get('translation', [0, 0, 0]))
                q = overrides.get((index, 'rotation'), node.get('rotation', [0, 0, 0, 1]))
                s = overrides.get((index, 'scale'), node.get('scale', [1, 1, 1]))
                local = np.eye(4)
                local[:3, :3], local[:3, 3] = rotation(q)*s, t
            globals_[index] = global_matrix(parents[index])@local if index in parents else local
            return globals_[index]

        reconstructed = {}
        for name, primitives in decoded.items():
            target = convert(expected['posed_source'][ti] if name == 'AnalyticSurface' else expected['posed_coat'][ti][helper_first])
            full = np.full(target.shape, np.nan)
            for primitive in primitives:
                skin = primitive['skin']
                inverse = accessor(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
                matrices = np.asarray([global_matrix(node)@bind for node, bind in zip(skin['joints'], inverse)])
                local = np.c_[primitive['positions'], np.ones(len(primitive['positions']))]
                world = np.zeros((len(local), 4))
                for column in range(4):
                    world += np.einsum('nij,nj->ni', matrices[primitive['joints'][:, column]], local)*primitive['weights'][:, column, None]
                error = np.linalg.norm(world[:, :3]-target[primitive['mapping']], axis=1).max()
                measurements['pose_error_m'] = max(measurements['pose_error_m'], float(error))
                full[primitive['mapping']] = world[:, :3]
            check(np.isfinite(full).all(), 'Unmapped source/fin vertices')
            reconstructed[name] = full
        native_roots = np.flatnonzero(native_map % 4 < 2)
        root_source_ids = fins['root_vertex_ids'][native_map[native_roots]//4, native_map[native_roots] % 4]
        roots = reconstructed['PortableEdgeFringe'][native_map[native_roots]]
        error = np.linalg.norm(roots-reconstructed['AnalyticSurface'][root_source_ids], axis=1).max()
        measurements['root_error_m'] = max(measurements['root_error_m'], float(error))
    check(measurements['pose_error_m'] < 2e-6 and measurements['root_error_m'] < 1e-6, 'Exported animated skin attachment failed')
    uv4 = fins['uv_gltf'].reshape(-1, 4, 2)
    span = np.linalg.norm(fins['positions'].reshape(-1, 4, 3)[:, 1]-fins['positions'].reshape(-1, 4, 3)[:, 0], axis=1)
    errors, alpha = [], []
    source_textures = [np.asarray(Image.open(out/f'source-{i}.png').convert('RGBA')) for i in range(2)]
    for physical_offset in (-.0025, .0025):
        fraction = .5+physical_offset/span
        source_uv = raw_uv[:, 0]*(1-fraction[:, None])+raw_uv[:, 1]*fraction[:, None]
        atlas_uv = ((uv4[:, 0]+uv4[:, 2])*(1-fraction[:, None])+(uv4[:, 1]+uv4[:, 3])*fraction[:, None])*.5
        actual = sample_rgb(atlas, atlas_uv, False)
        wanted = np.empty_like(actual)
        for slot in (0, 1):
            selected = fins['material_ids'] == slot
            wanted[selected] = sample_rgb(source_textures[slot], source_uv[selected], True)
        errors.extend(np.max(np.abs(actual[:, :3]-wanted[:, :3]), axis=1).tolist())
        alpha.extend(actual[:, 3].tolist())
    color = {'samples': len(errors), 'p95': float(np.quantile(errors, .95)), 'p99': float(np.quantile(errors, .99)),
             'maximum': max(errors), 'minimum_tip_alpha': min(alpha), 'count_error_over_0_1': int(np.sum(np.asarray(errors) > .1))}
    check(color['p95'] <= .02 and color['p99'] <= .05 and min(alpha) > .18, 'Independent exported tip RGB/orientation gate')
    result = {'status': 'passed', 'provenance': 'analytic_fixture', 'model_sha256': hashlib.sha256(blob).hexdigest(),
              'measurements': measurements, 'nominal_tip_color': color, 'sample_times_seconds': expected['times'].tolist(),
              'source_material_pixels_preserved': True, 'source_topology_preserved': True,
              'source_geometry_and_weights_preserved': True, 'atlas_embedded_pixel_identical': True,
              'fringe_masks_verified': True, 'selected_fins': len(fins['root_vertex_ids']),
              'native_corners_verified': len(native_map), 'duplicate_roots_checked_per_pose': len(native_roots),
              'elapsed_seconds': time.monotonic()-started,
              'limits': ['Analytic component fixture, not a generated or accepted avatar.',
                         'CPU LOD0 sampling does not measure GPU mip/filtering, visibility or appearance.',
                         'Source normals may use the exporter\'s bounded float rounding; measurements are retained.',
                         'Five actual animation samples do not prove continuous-time collision freedom.']}
    with (out/'verification.json').open('x') as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    return result
