"""Read only four pinned originals; no image decoding, native work or geometry edits."""

import gc
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT / 'sources'))
from avatar_assets.three_d import _GLB


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(name, value):
    encoded = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    with (OUT / name).open('xb') as stream:
        stream.write(encoded)


def array(glb, index):
    a = glb.doc['accessors'][index]
    assert not a.get('sparse') and not a.get('normalized')
    width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[a['type']]
    dtype = np.dtype({5121: 'u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}[a['componentType']])
    view = glb.doc['bufferViews'][a['bufferView']]
    stride = view.get('byteStride', width * dtype.itemsize)
    start = view.get('byteOffset', 0) + a.get('byteOffset', 0)
    assert a['count'] <= 300000 and stride >= width * dtype.itemsize
    assert a.get('byteOffset', 0) + (a['count'] - 1) * stride + width * dtype.itemsize <= view['byteLength']
    result = np.ndarray((a['count'], width), dtype=dtype, buffer=glb.binary,
                        offset=start, strides=(stride, dtype.itemsize))
    assert np.isfinite(result).all()
    return result


def inspect(case):
    path = ROOT / case['path']
    glb = _GLB(path)
    assert glb.hash == case['sha256']
    manifest_path = ROOT / case['download_manifest']
    assert sha(manifest_path) == case['download_manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    matches = [r for r in manifest['artifacts'] if r['path'] == 'model.glb']
    assert len(matches) == 1 and matches[0]['sha256'] == glb.hash and matches[0]['bytes'] == path.stat().st_size
    cfg = json.loads((OUT / 'inputs' / (case['subject_id'] + '-config.json')).read_text())
    plan_path = ROOT / 'outputs' / case['subject_id'] / 'plan.json'
    assert manifest['plan_sha256'] == sha(plan_path)
    plan = json.loads(plan_path.read_text())
    doc = glb.doc
    assert not doc.get('skins') and not doc.get('animations')
    assert len(doc['nodes']) == len(doc['meshes']) == 1
    node = doc['nodes'][0]
    assert node['mesh'] == 0 and node.get('matrix', [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]) == [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]
    assert node.get('translation', [0., 0., 0.]) == [0., 0., 0.] and node.get('rotation', [0., 0., 0., 1.]) == [0., 0., 0., 1.] and node.get('scale', [1., 1., 1.]) == [1., 1., 1.]
    primitives = []
    for primitive in doc['meshes'][0]['primitives']:
        assert primitive.get('mode', 4) == 4 and not primitive.get('targets')
        attrs = primitive['attributes']
        points = array(glb, attrs['POSITION'])
        indices = array(glb, primitive['indices'])
        assert len(indices) % 3 == 0 and int(indices.min()) >= 0 and int(indices.max()) < len(points)
        hashes = {'indices': hashlib.sha256(indices.tobytes()).hexdigest()}
        for name, accessor in attrs.items():
            values = array(glb, accessor)
            assert len(values) == len(points)
            hashes[name] = hashlib.sha256(values.tobytes()).hexdigest()
        normals = array(glb, attrs['NORMAL']).astype(np.float64)
        low, high = points.min(axis=0), points.max(axis=0)
        primitives.append({'vertices': len(points), 'triangles': len(indices) // 3,
            'attributes': sorted(attrs), 'material': primitive.get('material'),
            'local_gltf_bounds': {'min': low.tolist(), 'max': high.tolist(), 'extent': (high-low).tolist()},
            'normal_unit_error_max': float(np.abs(np.linalg.norm(normals, axis=1)-1).max()),
            'decoded_attribute_hashes': hashes})
    images = []
    for index, image in enumerate(doc.get('images', [])):
        view = doc['bufferViews'][image['bufferView']]
        start = view.get('byteOffset', 0)
        raw = glb.binary[start:start + view['byteLength']]
        with Image.open(io.BytesIO(raw)) as bitmap:
            assert bitmap.width * bitmap.height <= 16777216 and getattr(bitmap, 'n_frames', 1) == 1
            images.append({'index': index, 'format': bitmap.format, 'mode': bitmap.mode,
                'width': bitmap.width, 'height': bitmap.height, 'pixels': bitmap.width * bitmap.height,
                'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
            bitmap.verify()
    return {**case, 'bytes': path.stat().st_size, 'vertices': sum(p['vertices'] for p in primitives),
        'triangles': sum(p['triangles'] for p in primitives), 'primitives': primitives,
        'materials': doc.get('materials', []), 'textures': doc.get('textures', []),
        'samplers': doc.get('samplers', []), 'images': images, 'nodes': doc['nodes'],
        'skins': 0, 'animations': 0, 'source_unchanged': sha(path) == glb.hash,
        'provenance': {'provider': 'Meshy', 'recorded_model': plan['model'], 'generation': cfg['generation'],
            'subject': cfg['subject'], 'rights': cfg['rights'], 'task_id': manifest['task_id'],
            'download_status': manifest['status'], 'plan_sha256': manifest['plan_sha256'],
            'reference_image': cfg.get('reference'), 'new_calls_this_inventory': 0},
        'units_axes': {'storage_convention': 'glTF right-handed +Y up; distances conventionally metres',
            'source_unit_calibration_verified': False, 'requested_later_authored_height_meters': cfg['height_meters'],
            'raw_height_is_not_that_authoring_height': True,
            'forward_facing_verified_from_geometry': False},
        'limitations': ['No watertightness, hidden-surface completeness, visual acceptance or motion claim.',
            'Image headers/verify only; no decoded-pixel or new rendering review.',
            'No tangent attribute: future native export must derive valid UV0 frames without rerouting textures.']}


def main():
    started = time.monotonic()
    prepared = json.loads((OUT / 'prepared.json').read_text())
    for item in prepared['source_pins']:
        assert sha(ROOT / item['path']) == item['sha256'] == sha(OUT / item['snapshot'])
    records = []
    for case in prepared['cases']:
        records.append(inspect(case))
        gc.collect()
    comparisons = []
    for name in sorted({r['subject_id'] for r in records}):
        a, b = [r for r in records if r['subject_id'] == name]
        equal = [p['decoded_attribute_hashes'] for p in a['primitives']] == [p['decoded_attribute_hashes'] for p in b['primitives']]
        assert equal
        comparisons.append({'subject_id': name, 'mesh_texture_positions_normals_uv0_indices_exact': equal})
    pairs = []
    for stage in ['mesh', 'texture']:
        pair = [r for r in records if r['stage'] == stage]
        unique = {image['sha256']: image for row in pair for image in row['images']}
        pixels = sum(i['pixels'] for i in unique.values())
        total = {'stage': stage, 'vertices': sum(r['vertices'] for r in pair),
            'triangles': sum(r['triangles'] for r in pair), 'source_file_bytes': sum(r['bytes'] for r in pair),
            'unique_images': len(unique), 'unique_texture_pixels': pixels,
            'compressed_image_bytes': sum(i['bytes'] for i in unique.values()),
            'rgb8_decode_bytes': pixels * 3, 'rgba8_decode_bytes': pixels * 4,
            'within_16mp_per_image': all(i['pixels'] <= 16777216 for i in unique.values()),
            'within_16mp_aggregate_if_required': pixels <= 16777216,
            'within_64mib_source_bytes': sum(r['bytes'] for r in pair) <= 67108864,
            'assembly_not_built': True}
        assert total['vertices'] <= 100000 and total['triangles'] <= 100000 and total['within_64mib_source_bytes']
        pairs.append(total)
    write('report.json', {'schema_version': 1, 'status': 'verified_named_source_inventory',
        'sources': records, 'raw_vs_textured': comparisons, 'pair_totals': pairs,
        'elapsed_seconds': time.monotonic()-started,
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'memory_scope': 'Process high-water RSS on macOS; images not fully decoded',
        'claims': {'new_generation': False, 'new_assembly': False, 'animation': False,
            'shoulder_dragon_identity': False, 'quality_accepted': False, 'independent_rights_verification': False}})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write('failure.json', {'status': 'failed', 'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc()})
        raise
