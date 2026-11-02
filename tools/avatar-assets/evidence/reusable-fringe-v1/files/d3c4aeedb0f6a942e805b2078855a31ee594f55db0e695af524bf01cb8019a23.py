"""Read-only comparison of two rejected native attempts; no rendering or repair."""
import hashlib
import io
import json
from pathlib import Path
import struct
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
START = time.monotonic()
Image.MAX_IMAGE_PIXELS = 16_777_216


def check(value, message):
    if not value:
        raise ValueError(message)
    if time.monotonic()-START > 28:
        raise TimeoutError('Comparison exceeded its28-second internal deadline')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(name, value):
    with (HERE/name).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def load_glb(path):
    check(path.is_file() and not path.is_symlink() and path.stat().st_size <= 64*1024**2, 'Invalid GLB path/size')
    raw = path.read_bytes()
    check(len(raw) >= 28 and struct.unpack_from('<4sII', raw) == (b'glTF', 2, len(raw)), 'Invalid GLB header')
    size, kind = struct.unpack_from('<II', raw, 12)
    check(kind == 0x4e4f534a and size <= 1024**2 and size % 4 == 0, 'Invalid GLB JSON chunk')
    document = json.loads(raw[20:20+size])
    binary_size, kind = struct.unpack_from('<II', raw, 20+size)
    check(kind == 0x004e4942 and 28+size+binary_size == len(raw), 'Invalid binary chunk')
    check(len(document['buffers']) == 1 and 'uri' not in document['buffers'][0], 'Only embedded single buffers admitted')
    check(0 <= binary_size-document['buffers'][0]['byteLength'] <= 3, 'Invalid binary padding')
    return {'raw': raw, 'doc': document, 'binary': memoryview(raw)[28+size:], 'json': memoryview(raw)[20:20+size]}


def view(blob, index):
    rows = blob['doc']['bufferViews']
    check(type(index) is int and 0 <= index < len(rows) <= 2048, 'Invalid buffer view')
    row = rows[index]
    start, size = row.get('byteOffset', 0), row['byteLength']
    check(row.get('buffer', 0) == 0 and type(start) is int and type(size) is int
          and 0 <= start < start+size <= len(blob['binary']), 'Invalid buffer view range')
    return blob['binary'][start:start+size]


def accessor_hash(blob, index):
    row = blob['doc']['accessors'][index]
    check(not row.get('sparse'), 'Sparse accessor comparison is unsupported')
    width = {'SCALAR':1, 'VEC2':2, 'VEC3':3, 'VEC4':4, 'MAT4':16}[row['type']]
    item_bytes = width*{5120:1, 5121:1, 5122:2, 5123:2, 5125:4, 5126:4}[row['componentType']]
    check(row['type'] != 'MAT4' or row['componentType'] == 5126, 'Unsupported matrix storage')
    count, start = row['count'], row.get('byteOffset', 0)
    stride = blob['doc']['bufferViews'][row['bufferView']].get('byteStride', item_bytes)
    data = view(blob, row['bufferView'])
    check(type(count) is int and 1 <= count <= 300000 and type(start) is int and start >= 0
          and type(stride) is int and item_bytes <= stride <= 252
          and start+(count-1)*stride+item_bytes <= len(data), 'Accessor exceeds bounded storage')
    if stride == item_bytes:
        value = digest(data[start:start+count*item_bytes])
    else:
        array = np.ndarray((count, item_bytes), dtype=np.uint8, buffer=data, offset=start, strides=(stride, 1))
        result = hashlib.sha256()
        for first in range(0, count, 1024):
            result.update(array[first:first+1024].tobytes())
        value = result.hexdigest()
    return {'sha256': value, 'payload_bytes': count*item_bytes, 'count': count}


def png_pixels(encoded):
    check(bytes(encoded[:8]) == b'\x89PNG\r\n\x1a\n' and encoded[24] == 8, 'Require an eight-bit PNG')
    with Image.open(io.BytesIO(encoded)) as image:
        check(image.format == 'PNG' and image.width*image.height <= 16_777_216
              and image.width > 0 and image.height > 0 and image.mode in ('RGB', 'RGBA')
              and getattr(image, 'n_frames', 1) == 1, 'PNG format/dimensions exceed comparison bounds')
        image.load()
        rgba = image.convert('RGBA')
        try:
            return {'width': image.width, 'height': image.height, 'stored_mode': image.mode,
                    'canonical_mode': 'RGBA8', 'pixel_sha256': digest(rgba.tobytes())}
        finally:
            rgba.close()


def compare():
    prepared = json.loads((HERE/'prepared.json').read_text())
    roots = [Path(prepared['roots'][key]) for key in ('v1', 'v2')]
    blobs = [load_glb(root/'model.glb') for root in roots]
    docs = [b['doc'] for b in blobs]
    report = {'schema_version':1, 'scope':'Actual rejected native artifacts; byte/pixel preservation, not accepted avatars',
              'raw_model_sha256':[digest(b['raw']) for b in blobs], 'raw_model_bytes':[len(b['raw']) for b in blobs],
              'raw_model_equal':blobs[0]['raw'] == blobs[1]['raw'], 'json_semantically_equal':docs[0] == docs[1],
              'json_chunk_equal':blobs[0]['json'] == blobs[1]['json'],
              'json_sections':{key:docs[0].get(key) == docs[1].get(key) for key in sorted(set(docs[0])|set(docs[1]))}}
    for field in ('bufferViews','accessors','images'):
        check(len(docs[0][field]) == len(docs[1][field]), 'Mismatched '+field+' counts; preserve as failed comparison')
    report['buffer_views'] = []
    for i in range(len(docs[0]['bufferViews'])):
        hashes = [digest(view(b,i)) for b in blobs]
        report['buffer_views'].append({'index':i,'metadata_equal':docs[0]['bufferViews'][i] == docs[1]['bufferViews'][i],
                                      'payload_equal':hashes[0] == hashes[1], 'sha256':hashes})
    check(len(docs[0]['accessors']) <= 2048, 'Accessor inventory exceeds limit')
    report['accessors'] = []
    for i in range(len(docs[0]['accessors'])):
        rows = [accessor_hash(b,i) for b in blobs]
        report['accessors'].append({'index':i,'metadata_equal':docs[0]['accessors'][i] == docs[1]['accessors'][i],
                                   'payload_equal':rows[0] == rows[1], 'measurements':rows})
    report['embedded_images'], differing_image_ranges = [], []
    for i in range(len(docs[0]['images'])):
        images = [d['images'][i] for d in docs]
        check(all('bufferView' in row and 'uri' not in row for row in images), 'External GLB image unsupported')
        payloads = [view(b,row['bufferView']) for b,row in zip(blobs,images)]
        hashes = [digest(data) for data in payloads]
        entry = {'index':i, 'metadata_equal':images[0] == images[1], 'encoded_sha256':hashes,
                 'encoded_equal':hashes[0] == hashes[1], 'decoded':False}
        if not entry['encoded_equal']:
            check(all(row.get('mimeType') == 'image/png' for row in images), 'Differing non-PNG image requires separate review')
            entry['pixels'] = [png_pixels(data) for data in payloads]
            entry['decoded'] = True
            entry['pixels_equal'] = entry['pixels'][0] == entry['pixels'][1]
            check(images[0]['bufferView'] == images[1]['bufferView'], 'Changed image binding prevents offset attribution')
            row = docs[0]['bufferViews'][images[0]['bufferView']]
            differing_image_ranges.append((row.get('byteOffset',0),row.get('byteOffset',0)+row['byteLength']))
        else:
            entry['pixels_equal'] = True
            entry['pixel_basis'] = 'Identical encoded PNG/image bytes; decode not repeated'
        report['embedded_images'].append(entry)
    binary = [b['binary'] for b in blobs]
    check(len(binary[0]) == len(binary[1]), 'Binary lengths differ')
    changes, first, last = 0, None, None
    for start in range(0,len(binary[0]),1024*1024):
        ids = np.flatnonzero(np.frombuffer(binary[0][start:start+1024*1024],dtype=np.uint8)
                            != np.frombuffer(binary[1][start:start+1024*1024],dtype=np.uint8))
        if len(ids):
            changes += len(ids)
            if first is None: first = start+int(ids[0])
            last = start+int(ids[-1])
    cursor, outside_equal = 0, True
    for start,end in sorted(differing_image_ranges):
        check(start >= cursor, 'Differing image byte ranges overlap')
        outside_equal &= binary[0][cursor:start] == binary[1][cursor:start]
        cursor = end
    outside_equal &= binary[0][cursor:] == binary[1][cursor:]
    report['binary_difference'] = {'changed_byte_count':changes,'first_offset':first,'last_offset':last,
        'offset_basis':'Relative to complete padded BIN chunk','differing_image_ranges':differing_image_ranges,
        'all_bytes_outside_differing_embedded_images_equal':outside_equal}
    report['external_exact'] = []
    for name in ('textures/portable-fringe-color-atlas.png','fringe-correspondence.npz'):
        hashes = [file_hash(root/name) for root in roots]
        report['external_exact'].append({'path':name,'sha256':hashes,'equal':hashes[0] == hashes[1]})
    report['render_images'] = []
    for name in ('portrait.png','views/front.png','views/side.png','views/back.png','views/three_quarter.png'):
        paths = [root/name for root in roots]
        present = [path.is_file() for path in paths]
        row = {'path':name,'present':present}
        if all(present):
            check(all(path.stat().st_size <= 16*1024**2 for path in paths), 'Render PNG exceeds byte limit')
            row['encoded_sha256'] = [file_hash(path) for path in paths]
            row['pixels'] = [png_pixels(path.read_bytes()) for path in paths]
            row['pixels_equal'] = row['pixels'][0] == row['pixels'][1]
        report['render_images'].append(row)
    report['structural_and_pixel_preservation_passed'] = bool(report['json_semantically_equal'] and outside_equal
        and all(row['metadata_equal'] and row['payload_equal'] for row in report['accessors'])
        and all(row['pixels_equal'] for row in report['embedded_images'])
        and all(row['equal'] for row in report['external_exact'])
        and all(row.get('pixels_equal') is True for row in report['render_images']))
    report['elapsed_seconds'] = round(time.monotonic()-START,6)
    report['quality_approved'] = False
    write('preservation-comparison.json',report)
    print(json.dumps({'comparison_completed':True,'preservation_passed':report['structural_and_pixel_preservation_passed'],
        'changed_binary_bytes':changes,'changed_image_indices':[row['index'] for row in report['embedded_images'] if row['decoded']],
        'render_pixel_equality':{row['path']:row.get('pixels_equal') for row in report['render_images']}}))


if __name__ == '__main__':
    try:
        compare()
    except Exception as error:
        write('comparison-failure.json',{'error':str(error),'type':type(error).__name__,
            'elapsed_seconds':round(time.monotonic()-START,6),'native_calls':0})
        raise
