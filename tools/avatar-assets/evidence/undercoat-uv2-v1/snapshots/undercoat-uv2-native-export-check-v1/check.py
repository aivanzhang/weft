"""Check actual analytic native export; never promote fixture to avatar acceptance."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.finish_generated import _undercoat_tangent_targets
from avatar_assets.generated_tangents import repair
from avatar_assets.normalize_generated import _array
from avatar_assets.safety import require, sha256, write_json
from avatar_assets.three_d import _GLB, validate_render_attributes

source = ROOT/'outputs/undercoat-uv2-native-v2'
inputs = json.loads((OUT/'inputs.json').read_text())
for row in inputs['files']:
    require(sha256(ROOT/row['path']) == row['sha256'], 'Pinned source changed')
native = _GLB(source/'model-native.glb')
report = json.loads((source/'finish-report.json').read_text())
targets = _undercoat_tangent_targets(report, True)
require(len(targets) == 1 and targets[0]['normal_texcoord'] == 2, 'Expected exactly one authored UV2 material')
correction = repair(source/'model-native.glb', OUT/'model.glb', authored_normal_targets=targets)
fixed = _GLB(OUT/'model.glb')
require(fixed.doc == native.doc, 'Correction changed native JSON')
allowed = np.zeros(len(native.binary), dtype=bool)
for row in correction['authored_normal_correction']['primitives']:
    accessor = native.doc['accessors'][row['tangent_accessor']]
    view = native.doc['bufferViews'][accessor['bufferView']]
    start = view.get('byteOffset', 0)
    allowed[start:start+view['byteLength']] = True
old, new = np.frombuffer(native.binary, dtype='u1'), np.frombuffer(fixed.binary, dtype='u1')
require(np.array_equal(old[~allowed], new[~allowed]), 'Non-authored bytes changed')
require(not correction['repaired_vertices'], 'Unrelated tangent repair occurred')


def image_hash(texture):
    image = native.doc['images'][native.doc['textures'][texture['index']]['source']]
    view = native.doc['bufferViews'][image['bufferView']]
    start = view.get('byteOffset', 0)
    return hashlib.sha256(native.binary[start:start+view['byteLength']]).hexdigest()


rows = []
for primitive in native.doc['meshes'][0]['primitives']:
    material = native.doc['materials'][primitive['material']]
    base = material['pbrMetallicRoughness']['baseColorTexture']
    response = material['pbrMetallicRoughness']['metallicRoughnessTexture']
    require(base.get('texCoord', 0) == 0 and image_hash(base) == sha256(source/'analytic-color.png'),
            'Original basecolor bytes or UV0 route changed')
    require(response.get('texCoord', 0) == 1 and image_hash(response) == sha256(source/'analytic-source-response.png'),
            'Original response bytes or UV1 route changed')
    attrs = primitive['attributes']
    uv0 = _array(native, attrs['TEXCOORD_0'], 2, 5126, 100000)
    uv1 = _array(native, attrs['TEXCOORD_1'], 2, 5126, 100000)
    require(np.max(np.abs(uv1-(uv0*.6+.2))) < 1e-7, 'Source UV1 relationship changed during native export')
    if material['name'] == 'HealthyUV0':
        require(material['normalTexture']['texCoord'] == 1
                and image_hash(material['normalTexture']) == sha256(source/'analytic-source-normal.png'),
                'Unselected original normal texture or route changed')
    rows.append({'material': material['name'], 'vertices': len(uv0),
                 'normal_texcoord': material['normalTexture'].get('texCoord', 0),
                 'basecolor_uv': base.get('texCoord', 0), 'response_uv': response['texCoord']})
attributes = validate_render_attributes(OUT/'model.glb')
write_json(OUT/'correction.json', correction)
write_json(OUT/'render-attributes.json', attributes)
write_json(OUT/'result.json', {'schema_version': 1, 'status': 'analytic_native_component_verified',
    'native_sha256': native.hash, 'corrected_sha256': fixed.hash, 'materials': rows,
    'changed_binary_bytes': int((old != new).sum()), 'only_authored_tangent_bytes_changed': True,
    'original_texture_bytes_routes_and_uv1_preserved': True,
    'generated_avatar': False, 'appearance_accepted': False, 'motion_accepted': False})
