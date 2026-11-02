"""Prepare known bytes and sparse source-UV samples; never start Blender."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
from PIL import Image, PngImagePlugin, __version__ as pillow_version

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT/'outputs/panda-adaptive-undercoat-finish-v2/model.glb'
MODEL_HASH = 'e0488074965a89a9f7eda7788ebb26e7a8f14f98bbb44e1d9856fa091915e5d5'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def decode(rgb):
    return [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in rgb]


def main():
    assert not (HERE/'samples.json').exists()
    assert MODEL.stat().st_size <= 64*1024*1024 and sha(MODEL) == MODEL_HASH
    raw = MODEL.read_bytes()
    assert raw[:4] == b'glTF' and struct.unpack_from('<I', raw, 4)[0] == 2
    length = struct.unpack_from('<I', raw, 12)[0]
    assert length <= 2*1024*1024 and raw[16:20] == b'JSON'
    doc = json.loads(raw[20:20+length]); offset = 28+length
    assert raw[24+length:28+length] == b'BIN\x00'
    image_id = doc['textures'][doc['materials'][0]['pbrMetallicRoughness']['baseColorTexture']['index']]['source']
    image = doc['images'][image_id]
    assert image['mimeType'] == 'image/jpeg' and 'uri' not in image
    view = doc['bufferViews'][image['bufferView']]
    assert view.get('buffer', 0) == 0 and 0 < view['byteLength'] <= 16*1024*1024
    target = HERE/'inputs/actual-basecolor.jpg'; target.parent.mkdir(exist_ok=True)
    with target.open('xb') as handle:
        start = offset+view.get('byteOffset', 0)
        handle.write(raw[start:start+view['byteLength']])

    def accessor(index, kind):
        entry = doc['accessors'][index]; assert not entry.get('sparse') and entry['type'] == kind
        width = {'VEC2': 2, 'SCALAR': 1}[kind]
        dtype = {5123: '<u2', 5125: '<u4', 5126: '<f4'}[entry['componentType']]
        size = np.dtype(dtype).itemsize; count = entry['count']; assert 0 < count <= 300000
        view = doc['bufferViews'][entry['bufferView']]; stride = view.get('byteStride', size*width)
        begin = offset+view.get('byteOffset', 0)+entry.get('byteOffset', 0)
        assert size*width <= stride <= 252
        assert entry.get('byteOffset', 0)+(count-1)*stride+size*width <= view['byteLength']
        return np.ndarray((count, width), dtype=dtype, buffer=raw, offset=begin, strides=(stride, size))

    with Image.open(target) as source:
        assert source.mode == 'RGB' and source.width*source.height <= 16777216
        width, height = source.size; candidates = []
        for mi, mesh in enumerate(doc['meshes']):
            for pi, primitive in enumerate(mesh['primitives']):
                mat_id = primitive['material']; material = doc['materials'][mat_id]
                texture = material.get('pbrMetallicRoughness', {}).get('baseColorTexture')
                if not texture or doc['textures'][texture['index']]['source'] != image_id:
                    continue
                uv = accessor(primitive['attributes']['TEXCOORD_0'], 'VEC2')
                faces = accessor(primitive['indices'], 'SCALAR').reshape(-1, 3)
                for triangle in np.unique(np.linspace(0, len(faces)-1, min(24, len(faces)), dtype=int)):
                    ids = faces[triangle]; corners = uv[ids].astype(float)
                    bary = np.array([.2, .3, .5]); point = bary@corners
                    xy = [int(np.floor((point[0] % 1)*width)), int(np.floor((point[1] % 1)*height))]
                    rgb = list(source.getpixel(tuple(xy)))
                    candidates.append({'mesh': mi, 'primitive': pi, 'material': material['name'],
                        'triangle': int(triangle), 'vertex_indices': ids.tolist(), 'uv0_gltf_v_down': corners.tolist(),
                        'barycentric': bary.tolist(), 'sample_uv_gltf': point.tolist(), 'pixel_top_left_xy': xy,
                        'pixel_bottom_left_xy': [xy[0], height-1-xy[1]], 'encoded_rgb8': rgb,
                        'basecolor_factor': material['pbrMetallicRoughness'].get('baseColorFactor', [1, 1, 1, 1])})
        assert 12 <= len(candidates) <= 768
        colors = np.asarray([r['encoded_rgb8'] for r in candidates], dtype=float)
        chosen = [int(np.argmin(colors.sum(axis=1)))]; nearest = np.full(len(candidates), np.inf)
        while len(chosen) < 12:
            nearest = np.minimum(nearest, np.sum((colors-colors[chosen[-1]])**2, axis=1))
            nearest[chosen] = -1
            chosen.append(int(np.argmax(nearest)))
        actual = [candidates[index] for index in chosen]
        for number, row in enumerate(actual):
            row['id'] = 'actual-'+str(number).zfill(2)
            x, y = row['pixel_bottom_left_xy']
            wrong = list(source.getpixel((x, y)))
            row.update(encoded_rgb=[v/255 for v in row['encoded_rgb8']],
                linear_rgb=decode([v/255 for v in row['encoded_rgb8']]),
                opposite_row_encoded_rgb=[v/255 for v in wrong],
                opposite_row_linear_rgb=decode([v/255 for v in wrong]))

    levels = [0, 32, 128, 224]
    known = Image.new('RGB', (4, 4))
    for y in range(4):
        for x in range(4):
            known.putpixel((x, y), (levels[x], levels[(x+2*y) % 4], levels[y]))
    png = PngImagePlugin.PngInfo(); png.add(b'sRGB', b'\x00')
    known_path = HERE/'inputs/known-srgb.png'; assert not known_path.exists()
    known.save(known_path, pnginfo=png)
    samples = []
    for y in range(4):
        for x in range(4):
            rgb = list(known.getpixel((x, y))); opposite = list(known.getpixel((x, 3-y)))
            samples.append({'id': f'known-{x}-{y}', 'pixel_top_left_xy': [x, y],
                'pixel_bottom_left_xy': [x, 3-y], 'encoded_rgb8': rgb,
                'encoded_rgb': [v/255 for v in rgb], 'linear_rgb': decode([v/255 for v in rgb]),
                'opposite_row_encoded_rgb': [v/255 for v in opposite],
                'opposite_row_linear_rgb': decode([v/255 for v in opposite])})
    write(HERE/'samples.json', {'schema_version': 1, 'model_path': str(MODEL), 'model_sha256': MODEL_HASH,
        'image_binding': {'image_index': image_id, 'image_metadata': image, 'buffer_view': view},
        'selection': 'Deterministic source-triangle samples, then twelve farthest encoded-RGB candidates; nearest texel, not a rendered or aesthetic selection.',
        'pillow_version': pillow_version, 'python_version': sys.version,
        'source_pixel_note': 'JPEG RGB8 are Pillow-decoded bytes; small decoder rounding differences are possible. Known PNG bytes are lossless.',
        'images': [{'id': 'known-srgb', 'path': 'inputs/known-srgb.png', 'sha256': sha(known_path),
                    'dimensions': [4, 4], 'samples': samples, 'full_foreach_get': True, 'classification': 'analytic_known_bytes'},
                   {'id': 'actual-basecolor', 'path': 'inputs/actual-basecolor.jpg', 'sha256': sha(target),
                    'dimensions': [width, height], 'samples': actual, 'full_foreach_get': False,
                    'classification': 'exact_embedded_generated_source_texture'}]})
    copies = {'scripts/fur_root_sampling.py': 'sources/native/fur_root_sampling.py',
              'scripts/blender_finish_generated.py': 'sources/context/blender_finish_generated.py',
              'avatar_assets/__init__.py': 'sources/host/avatar_assets/__init__.py',
              'avatar_assets/runner.py': 'sources/host/avatar_assets/runner.py',
              'avatar_assets/safety.py': 'sources/host/avatar_assets/safety.py'}
    inventory = []
    for original, copied in copies.items():
        destination = HERE/copied; destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists(); shutil.copyfile(ROOT/original, destination)
        inventory.append({'origin': original, 'path': copied, 'sha256': sha(destination)})
    write(HERE/'source-copies.json', {'files': inventory, 'no_source_files_changed': True})
    print(json.dumps({'status': 'inputs_prepared_no_native_call', 'actual_dimensions': [width, height],
        'actual_samples': len(actual), 'known_samples': len(samples), 'actual_texture_bytes': target.stat().st_size,
        'actual_texture_sha256': sha(target)}))


if __name__ == '__main__':
    main()
