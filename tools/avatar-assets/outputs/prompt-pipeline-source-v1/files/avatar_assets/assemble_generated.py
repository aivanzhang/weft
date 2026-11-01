"""Combine generated motions only when their named hierarchy and rest pose match."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import struct

from avatar_assets.provider_model import preflight
from avatar_assets.safety import artifacts, local_file, new_directory, read_json, require, sha256, slug, write_json
from avatar_assets.three_d import _GLB, validate_glb


def rig_contract(doc, assumptions=None, *, authored=False):
    names = [doc['nodes'][i]['name'] for i in sorted({i for s in doc['skins'] for i in s['joints']})]
    return {'schema_version': 1, 'units': 'meters',
            'axes': {'up': '+Y', 'forward': '+Z', 'handedness': 'right'},
            'rest_pose': 'Default exported node transforms and inverse bind matrices; authored, not recovered anatomy' if authored else
                         'Default exported node transforms and inverse bind matrices; generated, not recovered anatomy',
            'bone_to_simulated_joint': {name: name for name in names},
            'target': {'name': 'Locally authored creature controls, NOT Weft' if authored else
                              'Local provider bone controls, NOT Weft', 'joint_names': names},
            'authored_assumptions': assumptions or ([
                'Hidden surfaces are generated; bone placement, weights, scale and motion are locally authored assumptions'
            ] if authored else ['Hidden surfaces and humanoid skeleton are generated assumptions']),
            'attachments': [], 'weft_runtime_verified': False}


def _hierarchy(doc):
    nodes = doc['nodes']
    names = [n.get('name') for n in nodes]
    require(all(isinstance(n, str) and n for n in names) and len(set(names)) == len(names),
            'Assembly requires unique named nodes')
    parents = {i: p for p, n in enumerate(nodes) for i in n.get('children', [])}
    return {n['name']: {'parent': names[parents[i]] if i in parents else None,
                       **{k: n[k] for k in ('translation', 'rotation', 'scale', 'matrix') if k in n}}
            for i, n in enumerate(nodes)}


def assemble(config_path, output):
    path = local_file(config_path)
    cfg = read_json(path)
    require(set(cfg) == {'schema_version', 'id', 'source', 'source_sha256', 'states'}
            and cfg['schema_version'] == 1, 'Invalid assembly config')
    slug(cfg['id'])
    require(isinstance(cfg['states'], list) and 1 <= len(cfg['states']) <= 8, 'Require 1–8 motions')
    source = local_file(path.parent/cfg['source'], 64*1024*1024)
    require(sha256(source) == cfg['source_sha256'], 'Assembly source changed')
    preflight(source)
    base = _GLB(source)
    doc, binary = deepcopy(base.doc), bytearray(base.binary)
    expected = _hierarchy(doc)
    node_ids = {node['name']: i for i, node in enumerate(doc['nodes'])}
    doc['animations'] = []
    evidence, seen = [], set()
    for state in cfg['states']:
        require(set(state) == {'id', 'source', 'sha256', 'meaning'}, 'Invalid motion config')
        name = slug(state['id'])
        require(name not in seen and isinstance(state['meaning'], str) and 0 < len(state['meaning']) <= 4000,
                'Duplicate state or invalid meaning')
        seen.add(name)
        motion_path = local_file(path.parent/state['source'], 64*1024*1024)
        require(sha256(motion_path) == state['sha256'], 'Motion source changed')
        motion = _GLB(motion_path)
        require(_hierarchy(motion.doc) == expected, 'Motion rest hierarchy differs; explicit retargeting required')
        require(len(motion.doc.get('animations', [])) == 1, 'Require exactly one source clip per state')
        proof = validate_glb(motion_path, rig_contract(motion.doc))
        require(proof['motion']['animation_verified'], 'Source clip contains no verified skin motion')
        animation = deepcopy(motion.doc['animations'][0])
        copied = {}
        for sampler in animation['samplers']:
            for field in ('input', 'output'):
                old = sampler[field]
                if old not in copied:
                    accessor = deepcopy(motion.doc['accessors'][old])
                    view = deepcopy(motion.doc['bufferViews'][accessor['bufferView']])
                    offset = view.get('byteOffset', 0)
                    binary.extend(b'\0'*(-len(binary)%4))
                    raw = motion.binary[offset:offset+view['byteLength']]
                    view['byteOffset'] = len(binary)
                    binary.extend(raw)
                    doc['bufferViews'].append(view)
                    accessor['bufferView'] = len(doc['bufferViews'])-1
                    doc['accessors'].append(accessor)
                    copied[old] = len(doc['accessors'])-1
                sampler[field] = copied[old]
        for channel in animation['channels']:
            channel['target']['node'] = node_ids[motion.doc['nodes'][channel['target']['node']]['name']]
        animation['name'] = name
        doc['animations'].append(animation)
        times = proof['motion']['animations'][0]['sample_times_seconds']
        duration = max(times)-min(times)
        require(0 < duration <= 10, 'Motion exceeds the bounded ten-second assembly profile')
        evidence.append({**state, 'duration_seconds': duration,
                         'loop_verified': False, 'source_motion_verified': True})
    doc['buffers'] = [{'byteLength': len(binary)}]
    encoded = json.dumps(doc, separators=(',', ':'), allow_nan=False).encode()
    encoded += b' '*(-len(encoded)%4)
    binary.extend(b'\0'*(-len(binary)%4))
    raw = (struct.pack('<4sII', b'glTF', 2, 28+len(encoded)+len(binary))
           +struct.pack('<II', len(encoded), 0x4E4F534A)+encoded
           +struct.pack('<II', len(binary), 0x004E4942)+binary)
    require(len(raw) <= 64*1024*1024, 'Assembled GLB exceeds 64MiB')
    out = new_directory(output)
    (out/'model.glb').write_bytes(raw)
    shutil.copyfile(path, out/'assembly-config.json')
    shutil.copyfile(__file__, out/'assemble_generated.py')
    contract = rig_contract(doc)
    write_json(out/'rig-contract.json', contract)
    proof = validate_glb(out/'model.glb', contract)
    write_json(out/'rig-verification.json', proof)
    result = {'schema_version': 1, 'kind': 'generated_motion_assembly', 'id': cfg['id'],
              'status': 'motion_verified_visual_review_pending', 'source_sha256': base.hash,
              'geometry_and_skin_unchanged': True, 'states': evidence, 'rig_verification': proof,
              'new_provider_calls': 0, 'artifacts': artifacts(out)}
    write_json(out/'manifest.json', result)
    return {'out': str(out), 'status': result['status'], 'states': len(evidence)}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    print(json.dumps(assemble(args.config, args.out), indent=2))
