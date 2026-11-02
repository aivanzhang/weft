"""Remove only the three declared legacy card primitives; keep the entire source BIN."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = json.loads((HERE/'config.json').read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def parse(blob):
    assert 28 <= len(blob) <= 40*1024**2
    magic, version, size, json_size, kind = struct.unpack_from('<4sIIII', blob)
    assert magic == b'glTF' and version == 2 and size == len(blob) and kind == 0x4E4F534A
    assert 0 < json_size <= 2*1024**2
    binary_size, kind = struct.unpack_from('<II', blob, 20+json_size)
    assert kind == 0x004E4942 and 28+json_size+binary_size == len(blob)
    return json.loads(blob[20:20+json_size]), blob[28+json_size:]


def main():
    source = ROOT/CONFIG['source']
    blob = source.read_bytes()
    assert sha(blob) == CONFIG['source_sha256']
    doc, binary = parse(blob)
    assert len(doc['meshes']) == 1 and len(doc['skins']) == 1
    assert len(doc['skins'][0]['joints']) == CONFIG['expected_bones']
    assert [a['name'] for a in doc['animations']] == CONFIG['expected_actions']
    primitives = doc['meshes'][0]['primitives']
    remove = CONFIG['remove_original_mesh0_primitives']
    assert [doc['materials'][primitives[i]['material']]['name'] for i in remove] == CONFIG['removed_materials']
    derivative = deepcopy(doc)
    derivative['meshes'][0]['primitives'] = [p for i, p in enumerate(primitives) if i not in remove]
    encoded = json.dumps(derivative, separators=(',', ':'), allow_nan=False).encode()
    encoded += b' '*((-len(encoded)) % 4)
    result = (struct.pack('<4sII', b'glTF', 2, 28+len(encoded)+len(binary))
              +struct.pack('<I4s', len(encoded), b'JSON')+encoded
              +struct.pack('<I4s', len(binary), b'BIN\0')+binary)
    actual, actual_bin = parse(result)
    assert actual == derivative and actual_bin == binary
    restored = deepcopy(actual); restored['meshes'][0]['primitives'] = primitives
    assert restored == doc
    destination = HERE/CONFIG['baseline']; destination.parent.mkdir(exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(result)
    assert sha(source.read_bytes()) == CONFIG['source_sha256']
    factors = []
    for material in doc['materials']:
        if material.get('name') in CONFIG['selected_materials']:
            pbr = material['pbrMetallicRoughness']
            texture = pbr['baseColorTexture']; definition = doc['textures'][texture['index']]
            sampler = doc['samplers'][definition.get('sampler', 0)]
            factor = pbr.get('baseColorFactor', [1, 1, 1, 1])
            assert factor == [1, 1, 1, 1] and material.get('alphaMode', 'OPAQUE') == 'OPAQUE'
            assert texture.get('texCoord', 0) == 0 and not texture.get('extensions')
            assert sampler.get('magFilter', 9729) == 9729 and sampler.get('minFilter', 9987) == 9987
            assert sampler.get('wrapS', 10497) == sampler.get('wrapT', 10497) == 10497
            factors.append({'material': material['name'], 'linear_factor': factor, 'texture_coord': 0,
                            'filter': 'Linear/trilinear', 'addressing': 'REPEAT', 'source_alpha': 'OPAQUE'})
    save(HERE/'baseline-proof.json', {
        'status': 'baseline_only_exact_preservation_passed', 'source': CONFIG['source'],
        'source_sha256': sha(blob), 'baseline': CONFIG['baseline'], 'baseline_sha256': sha(result),
        'baseline_bytes': len(result), 'original_bin_bytes': len(binary), 'original_bin_sha256': sha(binary),
        'entire_original_bin_exact': True, 'retained_primitive_json_exact': True,
        'other_document_fields_exact': True, 'removed_primitives': remove,
        'remaining_original_primitive_indices': [i for i in range(len(primitives)) if i not in remove],
        'source_material_contract': factors, 'source_geometry_or_texture_bytes_rewritten': False,
        'source_file_unchanged': True, 'native_executed': False, 'quality_approved': False,
        'limits': ['Header/JSON encoding changes in this derivative; the complete original file is not a byte prefix.',
                   'Legacy card buffers/materials/images stay unreferenced in this exact-custody baseline.',
                   'This baseline proof does not claim that a later Blender export is byte-identical.']})
    print(json.dumps({'status': 'baseline_prepared', 'sha256': sha(result), 'bytes': len(result)}))


if __name__ == '__main__':
    main()
