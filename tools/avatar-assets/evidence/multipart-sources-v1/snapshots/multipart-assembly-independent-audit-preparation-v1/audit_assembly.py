"""Independent CPU custody check; invoke only with a released candidate hash."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
from PIL import Image

PREP = Path(__file__).resolve().parent
ROOT = PREP.parents[1]
sys.path.insert(0, str(PREP / 'sources'))
from avatar_assets.three_d import _GLB


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)


def rows(glb, index):
    spec = glb.doc['accessors'][index]
    require(not spec.get('sparse') and not spec.get('normalized'), 'Unexpected sparse/normalized accessor')
    width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[spec['type']]
    dtype = np.dtype({5121:'u1', 5123:'<u2', 5125:'<u4', 5126:'<f4'}[spec['componentType']])
    view = glb.doc['bufferViews'][spec['bufferView']]
    stride = view.get('byteStride', width*dtype.itemsize)
    offset = spec.get('byteOffset', 0)
    require(0 < spec['count'] <= 300000 and stride >= width*dtype.itemsize, 'Accessor count/stride bound')
    require(offset + (spec['count']-1)*stride + width*dtype.itemsize <= view['byteLength'], 'Accessor overrun')
    result = np.ndarray((spec['count'], width), dtype=dtype, buffer=glb.binary,
                        offset=view.get('byteOffset', 0)+offset,
                        strides=(stride, dtype.itemsize))
    require(np.isfinite(result).all(), 'Nonfinite accessor')
    return result


def row_bytes_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def rotation(quaternion):
    x, y, z, w = map(float, quaternion)
    require(all(math.isfinite(v) for v in (x, y, z, w)), 'Invalid quaternion')
    length = math.sqrt(x*x+y*y+z*z+w*w)
    require(abs(length-1) <= 1e-6, 'Quaternion normalization bound')
    x, y, z, w = (v/length for v in (x, y, z, w))
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float64)


def identity_node(node):
    if 'matrix' in node:
        require(node['matrix'] == np.eye(4).reshape(-1).tolist(), 'Output node matrix is not identity')
    require(node.get('translation', [0, 0, 0]) == [0, 0, 0], 'Hidden node translation')
    require(node.get('rotation', [0, 0, 0, 1]) == [0, 0, 0, 1], 'Hidden node rotation')
    require(node.get('scale', [1, 1, 1]) == [1, 1, 1], 'Hidden node scale')
    require(not node.get('children') and 'skin' not in node, 'Unexpected hierarchy or skin')


def image_records(glb):
    records = []
    for source in glb.doc.get('images', []):
        view = glb.doc['bufferViews'][source['bufferView']]
        start = view.get('byteOffset', 0)
        data = glb.binary[start:start+view['byteLength']]
        with Image.open(io.BytesIO(data)) as image:
            require(image.width*image.height <= 16777216, 'Image pixel bound')
            require(getattr(image, 'n_frames', 1) == 1, 'Animated texture')
            records.append({'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                'mimeType': source['mimeType'], 'width': image.width, 'height': image.height})
    return records


def texture_semantics(glb, texture_index, images):
    texture = glb.doc['textures'][texture_index]
    sampler = glb.doc.get('samplers', [])[texture['sampler']] if 'sampler' in texture else {}
    require(set(texture) <= {'name', 'source', 'sampler'}, 'Unknown texture semantics')
    require(set(sampler) <= {'name', 'wrapS', 'wrapT', 'minFilter', 'magFilter'}, 'Unknown sampler semantics')
    return {'image': images[texture['source']], 'sampler': {
        'wrapS': sampler.get('wrapS', 10497), 'wrapT': sampler.get('wrapT', 10497),
        'minFilter': sampler.get('minFilter'), 'magFilter': sampler.get('magFilter')}}


def material_semantics(glb, index, images):
    material = json.loads(json.dumps(glb.doc['materials'][index]))
    material.pop('name', None)
    material.setdefault('alphaMode', 'OPAQUE')
    material.setdefault('doubleSided', False)
    if material['alphaMode'] == 'MASK':
        material.setdefault('alphaCutoff', .5)
    pbr = material.setdefault('pbrMetallicRoughness', {})
    pbr.setdefault('baseColorFactor', [1, 1, 1, 1])
    pbr.setdefault('metallicFactor', 1.)
    pbr.setdefault('roughnessFactor', 1.)
    for owner, keys in ((pbr, ('baseColorTexture', 'metallicRoughnessTexture')),
                        (material, ('normalTexture', 'occlusionTexture', 'emissiveTexture'))):
        for key in keys:
            if key in owner:
                route = owner[key]
                route.setdefault('texCoord', 0)
                if key == 'normalTexture':
                    route.setdefault('scale', 1.)
                if key == 'occlusionTexture':
                    route.setdefault('strength', 1.)
                route['index'] = texture_semantics(glb, route['index'], images)
    return material


def close_array(actual, expected, tolerance, label):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(actual.shape == expected.shape and np.isfinite(actual).all(), label+' shape/finite mismatch')
    error = float(np.abs(actual-expected).max())
    require(error <= tolerance, label+' mismatch')
    return error


def check_receipt(args, config, points, evidence):
    receipt_path = args.receipt.resolve()
    require(receipt_path == args.model.resolve().parent/'worker-result.json', 'Unexpected receipt path')
    require(not args.receipt.is_symlink() and receipt_path.stat().st_size <= 512*1024, 'Receipt type/size bound')
    require(sha(receipt_path) == args.receipt_sha256, 'Receipt hash mismatch')
    receipt = json.loads(receipt_path.read_text())
    expected_ids = {component['id'] for component in config['components']}
    require(len(receipt['components']) == len(expected_ids) and {c['id'] for c in receipt['components']} == expected_ids, 'Receipt component coverage')
    for component in config['components']:
        record = next(c for c in receipt['components'] if c['id'] == component['id'])
        measured = next(c for c in evidence['components'] if c['id'] == component['id'])
        require(record['source_sha256'] == component['source_sha256'], 'Receipt source hash mismatch')
        transform = record['source_transform']
        require(transform['placement'] == component['placement'], 'Receipt placement changed')
        require(transform['source_meters_per_unit'] == component['source_meters_per_unit'], 'Receipt source scale changed')
        require(transform['no_implicit_centering'] is True, 'Receipt declares hidden centering')
        close_array(transform['scale_to_meters'], measured['scale'], 1e-12, 'Receipt scale')
        close_array(transform['rotation_matrix_rows'], measured['rotation_matrix'], 1e-12, 'Receipt rotation')
        q = np.asarray(component['placement']['rotation_xyzw'], dtype=np.float64)
        close_array(transform['effective_unit_quaternion_xyzw'], q/np.linalg.norm(q), 1e-12, 'Receipt effective quaternion')
        require(len(record['primitives']) == len(measured['primitives']), 'Receipt primitive count')
        require(sum(p['vertices'] for p in record['primitives']) == sum(p['vertices'] for p in measured['primitives']), 'Receipt vertex count')
        require(sum(p['triangles'] for p in record['primitives']) == sum(p['triangles'] for p in measured['primitives']), 'Receipt triangle count')
    totals = receipt['measured_totals']
    for field in ('vertices', 'triangles'):
        require(totals[field] == evidence[field], 'Receipt measured '+field)
    require(totals['image_pixels'] == evidence['texture_pixels'], 'Receipt image pixel total')
    low, high = points.min(axis=0), points.max(axis=0)
    close_array(receipt['assembly_bounds_m']['min'], low, 0, 'Receipt glTF minimum')
    close_array(receipt['assembly_bounds_m']['max'], high, 0, 'Receipt glTF maximum')
    blender_points = points[:, [0, 2, 1]] * np.array([1., -1., 1.])
    blender_low, blender_high = blender_points.min(axis=0), blender_points.max(axis=0)
    offset = (blender_low+blender_high)*.5
    offset[2] = blender_low[2]
    factor = config['requested_rig_height_m']/(blender_high[2]-blender_low[2])
    canonical = receipt['requested_native_canonical_transform']
    require(canonical['applied'] is False, 'Canonical rig transform was already applied')
    require(canonical['gltf_to_blender'] == '[x,-z,y]', 'Canonical axis mapping changed')
    require(canonical['requested_height_m'] == config['requested_rig_height_m'], 'Requested rig height mismatch')
    errors = {
        'offset':close_array(canonical['blender_offset_before_scale'], offset, 1e-12, 'Canonical offset'),
        'translation':close_array(canonical['blender_translation_before_scale'], -offset, 1e-12, 'Canonical translation'),
        'scale':close_array(canonical['uniform_scale'], factor, 1e-12, 'Canonical scale')}
    evidence['canonical_rig_prediction'] = {'status':'verified_prediction_not_applied',
        'blender_offset_before_scale':offset.tolist(), 'translation_before_scale':(-offset).tolist(),
        'uniform_scale':float(factor), 'requested_height_m':config['requested_rig_height_m'],
        'receipt_errors':errors, 'actual_rig_or_motion_verified':False}
    evidence['receipt_sha256'] = args.receipt_sha256
    evidence['producer_status_not_used_as_acceptance'] = receipt.get('status')


def audit(args, evidence):
    prepared = json.loads((PREP/'prepared.json').read_text())
    for pin in prepared['source_pins']:
        require(sha(ROOT/pin['path']) == pin['sha256'] == sha(PREP/pin['snapshot']), 'Prepared source changed')
    config_path = ROOT/'outputs/multipart-source-layout-v1/compose.json'
    config = json.loads((PREP/'inputs/compose.json').read_text())
    layout = json.loads((PREP/'inputs/layout.json').read_text())
    target_path = args.model.resolve()
    require(target_path == ROOT/prepared['candidate_path'], 'Unexpected candidate path')
    require(len(args.expected_sha256) == 64 and sha(target_path) == args.expected_sha256, 'Candidate hash mismatch')
    output = _GLB(target_path)
    doc = output.doc
    require(not doc.get('skins') and not doc.get('animations'), 'Composition must remain unrigged and unanimated')
    require(len(doc['meshes']) == len(doc['nodes']) == 1, 'Require one identity node/mesh')
    identity_node(doc['nodes'][0])
    require(doc['nodes'][0]['mesh'] == 0, 'Wrong mesh owner')
    require(len(doc['scenes']) == 1 and doc['scenes'][0]['nodes'] == [0] and doc.get('scene', 0) == 0, 'Scene coverage changed')
    output_images = image_records(output)
    output_primitives = doc['meshes'][0]['primitives']
    materials = doc.get('materials', [])
    names = [m.get('name') for m in materials]
    require(len(names) == len(set(names)), 'Ambiguous output material name')
    covered_primitives, covered_materials, source_images = set(), set(), []
    total_vertices, total_triangles, source_bytes = 0, 0, 0
    aggregate_positions = []
    for component in config['components']:
        component_id = component['id']
        path = (config_path.parent/component['source']).resolve()
        require(sha(path) == component['source_sha256'], 'Source pin changed')
        source = _GLB(path)
        source_bytes += path.stat().st_size
        require(not source.doc.get('skins') and not source.doc.get('animations'), 'Unexpected source skin/clip')
        require(len(source.doc['meshes']) == len(source.doc['nodes']) == 1, 'Source structure changed')
        identity_node(source.doc['nodes'][0])
        images = image_records(source)
        source_images.extend(images)
        placement = component['placement']
        matrix = rotation(placement['rotation_xyzw'])
        scale = component['source_meters_per_unit']*placement['uniform_scale']
        translation = np.asarray(placement['translation_m'], dtype=np.float64)
        require(math.isfinite(scale) and scale > 0 and translation.shape == (3,) and np.isfinite(translation).all(), 'Invalid placement')
        part_evidence = {'id': component_id, 'source_sha256': source.hash,
            'scale': scale, 'rotation_matrix': matrix.tolist(), 'translation_m': translation.tolist(),
            'primitives': []}
        evidence['components'].append(part_evidence)
        for mi in range(len(source.doc.get('materials', []))):
            name = component_id+'__m'+str(mi)
            require(names.count(name) == 1, 'Missing namespaced material '+name)
            mapped = names.index(name)
            require(material_semantics(source, mi, images) == material_semantics(output, mapped, output_images), 'Changed material/image/sampler semantics '+name)
            covered_materials.add(mapped)
        originals = source.doc['meshes'][0]['primitives']
        for primitive in originals:
            material_index = names.index(component_id+'__m'+str(primitive['material']))
            candidates = [(i, p) for i, p in enumerate(output_primitives) if p.get('material') == material_index]
            require(len(candidates) == 1, 'This audit requires unique primitive/material correspondence')
            oi, actual_primitive = candidates[0]
            require(oi not in covered_primitives, 'Output primitive covered twice')
            covered_primitives.add(oi)
            require(primitive.get('mode', 4) == actual_primitive.get('mode', 4) == 4 and not actual_primitive.get('targets'), 'Changed triangle topology mode')
            attrs, actual_attrs = primitive['attributes'], actual_primitive['attributes']
            require(set(attrs) == set(actual_attrs), 'Changed attribute set')
            indices = rows(source, primitive['indices'])
            actual_indices = rows(output, actual_primitive['indices'])
            require(row_bytes_equal(indices, actual_indices), 'Changed index bytes/order')
            original_positions = rows(source, attrs['POSITION'])
            positions = rows(output, actual_attrs['POSITION'])
            require(positions.dtype == original_positions.dtype and positions.shape == original_positions.shape, 'Changed position representation')
            expected_positions = original_positions.astype(np.float64) @ matrix.T * scale + translation
            position_error = float(np.abs(positions-expected_positions).max())
            current = {'output_primitive': oi, 'vertices': len(positions), 'triangles': len(indices)//3,
                'position_error_m': position_error, 'normal_error': None, 'exact_other_attributes': []}
            part_evidence['primitives'].append(current)
            require(position_error <= prepared['gates']['position_absolute_tolerance_m'], 'Position transform/hidden recenter mismatch')
            for attribute in sorted(attrs):
                if attribute == 'POSITION':
                    continue
                a, b = rows(source, attrs[attribute]), rows(output, actual_attrs[attribute])
                require(a.dtype == b.dtype and a.shape == b.shape, 'Changed attribute representation '+attribute)
                if attribute in ('NORMAL', 'TANGENT'):
                    expected = a[:, :3].astype(np.float64) @ matrix.T
                    delta = float(np.abs(b[:, :3]-expected).max())
                    current[attribute.lower()+'_error'] = delta
                    require(delta <= prepared['gates']['rotated_normal_tangent_absolute_tolerance'], 'Frame transform mismatch '+attribute)
                    if np.array_equal(matrix, np.eye(3)):
                        require(row_bytes_equal(a, b), 'Identity rotation changed frame bytes')
                    if attribute == 'TANGENT':
                        require(row_bytes_equal(a[:, 3:], b[:, 3:]), 'Tangent handedness changed')
                else:
                    require(row_bytes_equal(a, b), 'Changed '+attribute+' bytes')
                    current['exact_other_attributes'].append(attribute)
            require(len(indices)%3 == 0 and int(indices.min()) >= 0 and int(indices.max()) < len(positions), 'Invalid final triangle indices')
            aggregate_positions.append(positions.astype(np.float64))
            total_vertices += len(positions)
            total_triangles += len(indices)//3
        actual_points = np.concatenate([rows(output, output_primitives[i]['attributes']['POSITION']) for i in sorted(covered_primitives)
                                       if names[output_primitives[i]['material']].startswith(component_id+'__m')])
        part_evidence['output_bounds'] = {'min': actual_points.min(axis=0).tolist(), 'max': actual_points.max(axis=0).tolist()}
        declared = next(r for r in layout['components'] if r['id'] == component_id)
        part_evidence['declared_height_m'] = declared['authored_height_m']
        require(abs(float(np.ptp(actual_points[:, 1]))-declared['authored_height_m']) <= 1e-6, 'Authored height changed')
        require(abs(float(actual_points[:, 1].min())) <= 1e-6, 'Authored base not on y=0')
    require(covered_primitives == set(range(len(output_primitives))), 'Uncovered/extra output primitives')
    require(covered_materials == set(range(len(materials))), 'Uncovered/extra output materials')
    used_textures = set()
    for material in materials:
        for owner, keys in ((material.get('pbrMetallicRoughness', {}), ('baseColorTexture', 'metallicRoughnessTexture')),
                            (material, ('normalTexture', 'occlusionTexture', 'emissiveTexture'))):
            used_textures.update(owner[key]['index'] for key in keys if key in owner)
    require(used_textures == set(range(len(doc.get('textures', [])))), 'Extra/unreferenced output texture')
    require({doc['textures'][i]['source'] for i in used_textures} == set(range(len(output_images))), 'Unreferenced image')
    used_samplers = {doc['textures'][i]['sampler'] for i in used_textures if 'sampler' in doc['textures'][i]}
    require(used_samplers == set(range(len(doc.get('samplers', [])))), 'Extra/unreferenced sampler')
    canonical_images = lambda records: sorted(json.dumps(r, sort_keys=True) for r in records)
    require(canonical_images(source_images) == canonical_images(output_images), 'Missing/extra/changed embedded images')
    require(total_vertices <= 100000 and total_triangles <= 100000, 'Geometry budget exceeded')
    require(source_bytes <= 67108864 and target_path.stat().st_size <= 67108864, '64MiB aggregate source/output bound')
    pixels = sum(r['width']*r['height'] for r in output_images)
    require(pixels <= 100663296, '96Mi-pixel aggregate texture bound')
    points = np.concatenate(aggregate_positions)
    evidence.update({'status':'assembly_custody_checked', 'candidate_sha256': output.hash,
        'vertices':total_vertices, 'triangles':total_triangles, 'source_file_bytes':source_bytes,
        'output_bytes':target_path.stat().st_size, 'texture_pixels':pixels, 'images':output_images,
        'output_bounds_gltf':{'min':points.min(axis=0).tolist(), 'max':points.max(axis=0).tolist()},
        'new_rig':False, 'new_animation':False, 'appearance_accepted':False})
    check_receipt(args, config, points, evidence)
    require(sha(target_path) == args.expected_sha256, 'Candidate changed during audit')
    evidence['status'] = 'verified_assembly_custody_and_canonical_prediction'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(args.out.is_dir() and not any(args.out.glob('result.json')) and not any(args.out.glob('failure.json')), 'Fresh result folder required')
    started = time.monotonic()
    evidence = {'schema_version':1, 'status':'running', 'components':[],
        'scope':'Independent attribute/transform/material custody only; no visual or animation acceptance',
        'native_calls':0, 'provider_calls':0, 'ai_role_calls':0}
    try:
        audit(args, evidence)
    except Exception as error:
        evidence.update({'status':'failed', 'error':str(error), 'traceback':traceback.format_exc()})
        write(args.out/'failure.json', evidence)
        raise
    evidence.update({'elapsed_seconds':time.monotonic()-started,
        'process_high_water_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    write(args.out/'result.json', evidence)


if __name__ == '__main__':
    main()
