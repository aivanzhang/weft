"""Source-normal fin derivative; unchanged shape, colors, skin and action."""
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.normalize_generated import _array
from avatar_assets.provider_model import preflight
from avatar_assets.safety import require, sha256, write_json
from avatar_assets.three_d import _GLB, validate_glb, validate_render_attributes


def main():
    config = json.loads((HERE/'prepared.json').read_text())
    for row in config['inputs']:
        require(sha256(ROOT/row['path']) == row['sha256'], 'Pinned input changed')
    out = ROOT/config['output']
    require(out.is_dir() and not any(out.iterdir()), 'Require empty monitored output')
    source = _GLB(ROOT/config['source'])
    original = source.doc
    mesh_id, primitive_id = config['fin_primitive']
    root_mesh, root_primitive = config['root_primitive']
    fin = original['meshes'][mesh_id]['primitives'][primitive_id]
    root = original['meshes'][root_mesh]['primitives'][root_primitive]
    require(fin['mode'] == 4 and fin['attributes'].keys() >= {'POSITION','NORMAL','JOINTS_0','WEIGHTS_0'},
            'Require explicit triangle skinned fins')
    require('TANGENT' not in fin['attributes'], 'Normal-only derivative cannot retain stale tangents')
    material = original['materials'][fin['material']]
    require(material.get('doubleSided') is True and material.get('alphaMode') == 'MASK'
            and 'normalTexture' not in material, 'Require the frozen masked geometric-normal source')
    with np.load(ROOT/config['correspondence'], allow_pickle=False) as data:
        ids = data['root_vertex_ids'].copy()
    require(ids.ndim == 2 and ids.shape[1] == 2 and ids.dtype.kind in 'iu' and 1 <= len(ids) <= 4800,
            'Invalid bounded root correspondence')
    count = len(ids)*4
    normals = _array(source, root['attributes']['NORMAL'], 3, 5126, 100000)
    require(ids.min() >= 0 and ids.max() < len(normals), 'Invalid source normal IDs')
    lengths = np.linalg.norm(normals, axis=1)
    require(np.max(np.abs(lengths-1)) < .001, 'Require source unit normals')
    normals = normals/lengths[:,None]
    authored = np.tile(normals[ids], (1,2,1)).reshape(-1,3).astype('<f4')
    old_normal = _array(source, fin['attributes']['NORMAL'], 3, 5126, 100000)
    require(old_normal.shape == authored.shape and len(authored) == count, 'Fin/root count mismatch')
    old_indices = _array(source, fin['indices'], 1, 5125, 300000).reshape(-1,3)
    expected = (np.arange(len(ids))[:,None,None]*4+np.array([[0,1,3],[0,3,2]])).reshape(-1,3)
    require(np.array_equal(old_indices, expected), 'Unsupported fin topology; do not infer anchors')
    indices = np.concatenate((old_indices, old_indices[:,::-1])).astype('<u4')
    doc = deepcopy(original)
    binary = bytearray(source.binary)

    def append(values, kind, target):
        binary.extend(b'\0'*(-len(binary)%4))
        raw = values.tobytes()
        doc['bufferViews'].append({'buffer':0,'byteOffset':len(binary),'byteLength':len(raw),'target':target})
        binary.extend(raw)
        record = {'bufferView':len(doc['bufferViews'])-1,'componentType':5126 if kind=='VEC3' else 5125,
                  'count':len(values),'type':kind}
        if kind=='VEC3':record.update(min=values.min(axis=0).tolist(),max=values.max(axis=0).tolist())
        doc['accessors'].append(record)
        return len(doc['accessors'])-1

    new_fin = doc['meshes'][mesh_id]['primitives'][primitive_id]
    new_fin['attributes']['NORMAL'] = append(authored,'VEC3',34962)
    new_fin['indices'] = append(indices.reshape(-1,1),'SCALAR',34963)
    new_material = deepcopy(material)
    new_material.update(name='ExperimentalSourceSurfaceFinNormals',doubleSided=False)
    doc['materials'].append(new_material)
    new_fin['material'] = len(doc['materials'])-1
    doc['buffers'][0]['byteLength'] = len(binary)
    doc['asset']['generator'] = 'Experimental source-surface fin shading; original geometry and motion unchanged'
    vertex_count = sum(doc['accessors'][p['attributes']['POSITION']]['count'] for m in doc['meshes'] for p in m['primitives'])
    triangles = sum(doc['accessors'][p['indices']]['count']//3 for m in doc['meshes'] for p in m['primitives'])
    require(vertex_count <= 100000 and triangles <= 100000,'Unchanged HD bounds exceeded')
    encoded = json.dumps(doc,separators=(',',':'),allow_nan=False).encode()
    encoded += b' '*(-len(encoded)%4)
    binary.extend(b'\0'*(-len(binary)%4))
    total = 28+len(encoded)+len(binary)
    require(total <= 64*1024*1024,'File exceeds HD bound')
    with (out/'model.glb').open('xb') as stream:
        stream.write(struct.pack('<4sII',b'glTF',2,total))
        stream.write(struct.pack('<II',len(encoded),0x4e4f534a));stream.write(encoded)
        stream.write(struct.pack('<II',len(binary),0x004e4942));stream.write(binary)
    require(binary[:len(source.binary)] == source.binary,'Original buffer changed')
    restored = deepcopy(doc)
    restored['meshes'][mesh_id]['primitives'][primitive_id] = deepcopy(fin)
    for key in ('accessors','bufferViews','materials'):restored[key] = restored[key][:len(original[key])]
    restored['buffers'] = deepcopy(original['buffers'])
    restored['asset'] = deepcopy(original['asset'])
    require(restored == original,'Unexpected JSON mutation beyond explicit allowlist')
    report = {'schema_version':1,'status':'pending_native_culling_and_appearance_review','quality_approved':False,
              'source_sha256':source.hash,'candidate_sha256':sha256(out/'model.glb'),'file_bytes':total,
              'vertices':vertex_count,'triangles':triangles,'fins':len(ids),
              'policy':'source-surface-normals-opposed-culled-faces-v1',
              'source_normals_unit_error':float(np.max(np.abs(np.linalg.norm(authored,axis=1)-1))),
              'median_absolute_source_geometric_dot':float(np.median(np.abs(np.sum(authored*old_normal,axis=1)))),
              'original_binary_prefix_exact':True,'preserved_binary_bytes':len(source.binary),
              'position_color_uv_skin_animation_exact':True,'json_allowlist_verified':True,
              'new_provider_calls':0,
              'limitations':['Authored shading normals intentionally do not match plane geometry.',
                  'Culling compatibility must be independently verified; Cycles is not automatically glTF-conformant.',
                  'Color/alpha resampling limits are inherited from the source color-atlas report; not arbitrary texel or mip equivalence.',
                  'Same sampled rig/action evidence applies to unchanged positions/skin, not new physical motion.',
                  'Not dense fur, fresh prompt acceptance, Muse parity or Weft consumer compatibility.']}
    del binary, source, authored, normals, old_normal
    gc.collect()
    contract = json.loads((ROOT/config['rig_contract']).read_text())
    write_json(out/'rig-contract.json',contract)
    write_json(out/'preflight.json',preflight(out/'model.glb'))
    write_json(out/'render-attributes.json',validate_render_attributes(out/'model.glb'))
    write_json(out/'rig-verification.json',validate_glb(out/'model.glb',contract))
    write_json(out/'build-report.json',report)
    for row in config['inputs']:require(sha256(ROOT/row['path'])==row['sha256'],'Input changed during build')
    print(json.dumps({k:report[k] for k in ('candidate_sha256','file_bytes','vertices','triangles')}))


if __name__ == '__main__':
    main()
