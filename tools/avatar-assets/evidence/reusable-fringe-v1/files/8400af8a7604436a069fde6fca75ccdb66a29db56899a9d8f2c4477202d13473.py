"""Call the released unused-fringe-tangent helper; preserve the actual raw native export."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE/'sources/host'))
from avatar_assets.fringe_export import omit_unused_fringe_tangents

CONFIG = json.loads((HERE/'config.json').read_text())
OUT = ROOT/CONFIG['output']


def parse(path):
    blob = path.read_bytes()
    assert len(blob) <= 64*1024**2 and struct.unpack_from('<I', blob, 8)[0] == len(blob)
    length, kind = struct.unpack_from('<I4s', blob, 12)
    assert kind == b'JSON' and length <= 2*1024**2
    binary_length, kind = struct.unpack_from('<I4s', blob, 20+length)
    assert kind == b'BIN\0' and 28+length+binary_length == len(blob)
    return json.loads(blob[20:20+length]), blob[28+length:], hashlib.sha256(blob).hexdigest()


def main():
    raw, target = OUT/'raw-export.glb', OUT/'model.glb'
    entries = list(OUT.rglob('*'))
    assert len(entries) <= 256 and all(not path.is_symlink() for path in entries)
    existing = sum(path.stat().st_size for path in entries if path.is_file())
    # Keep both GLBs plus eight MiB of numeric-audit space inside the same aggregate cap.
    assert existing+raw.stat().st_size+10*1024**2 <= 128*1024**2, 'Insufficient aggregate output headroom'
    before, binary, raw_hash = parse(raw)
    report = omit_unused_fringe_tangents(raw, target, ['SourceColoredFringe'])
    after, new_binary, target_hash = parse(target)
    expected = deepcopy(before)
    omissions = []
    for mesh_index, mesh in enumerate(expected['meshes']):
        for primitive_index, primitive in enumerate(mesh['primitives']):
            material = expected['materials'][primitive['material']]
            if material['name'] == 'SourceColoredFringe':
                assert 'normalTexture' not in material
                if 'TANGENT' in primitive['attributes']:
                    omissions.append({'mesh': mesh_index, 'primitive': primitive_index,
                                      'unbound_accessor': primitive['attributes'].pop('TANGENT')})
    assert after == expected and new_binary == binary
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == raw_hash
    with (OUT/'unused-fringe-tangent-omission.json').open('x') as stream:
        json.dump({'status': 'strict_unused_binding_omission_verified', 'helper_report': report,
                   'raw_sha256': raw_hash, 'model_sha256': target_hash, 'omitted_bindings': omissions,
                   'entire_original_binary_unchanged': True, 'other_json_fields_unchanged': True,
                   'raw_file_unchanged': True, 'quality_approved': False,
                   'limits': ['This does not repair or validate geometric normals; the independent export audit still runs.',
                              'Only unused non-normalmapped fringe tangent bindings are removed; raw export stays retained.']},
                  stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
