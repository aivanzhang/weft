"""Disabled experimental restoration of exact authored fringe attributes.

Use a fresh target after unused-fringe-tangent omission. This is a diagnostic
derivative, not completion or reuse of a failed native run. Caller job limits
remain30seconds/512MiB; local checks also bound elapsed time and sampled RSS.
"""

from collections import Counter
import hashlib
import math
import os
from pathlib import Path
import struct
import time
import zipfile

import numpy as np

from avatar_assets.fringe_export import _array, _index, _material, _strict_json
from avatar_assets.runner import resident_bytes
from avatar_assets.safety import local_file, read_json, require, sha256
from avatar_assets.three_d import _GLB


POLICY = 'experimental-authored-fringe-attributes-v1'
MAX_BYTES = 64*1024**2
MAX_NPZ_BYTES = 8*1024**2
MAX_FINS = 4800
ARRAY_NAMES = {'positions', 'normals', 'triangles', 'joints', 'weights', 'uv_gltf',
               'root_vertex_ids', 'root_face_ids', 'root_corner_ids', 'root_source_uv',
               'material_ids', 'native_vertex_to_helper_vertex',
               'destination_for_source_fin', 'source_fin_at_destination'}
REWRITES = ('NORMAL', 'JOINTS_0', 'WEIGHTS_0')


def _limits(started, samples):
    require(time.monotonic()-started < 30, 'Fringe fidelity exceeded30seconds')
    memory = resident_bytes(os.getpid())
    require(0 < memory <= 512*1024**2, 'Fringe fidelity exceeded512MiB sampled process RSS')
    samples.append(memory)


def _hash_array(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def _load_helper(path, count):
    expected = {name+'.npy' for name in ARRAY_NAMES}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) == len(expected) and {i.filename for i in entries} == expected,
                'Unknown or duplicate helper NPZ entries')
        require(sum(i.file_size for i in entries) <= MAX_NPZ_BYTES,
                'Helper NPZ decoded bytes exceed8MiB')
        for entry in entries:
            require(not entry.is_dir() and not entry.flag_bits & 1
                    and entry.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    and 0 < entry.file_size <= MAX_NPZ_BYTES, 'Invalid helper NPZ entry')
            with archive.open(entry) as handle:
                version = np.lib.format.read_magic(handle)
                require(version in ((1, 0), (2, 0)), 'Unsupported helper NPY header')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(handle, max_header_size=4096)
                require(not dtype.hasobject and dtype.kind in 'fiu' and not fortran
                        and 1 <= len(shape) <= 3 and all(type(n) is int and 0 < n <= 100000 for n in shape),
                        'Invalid helper NPY type or dimensions')
                require(int(np.prod(shape))*dtype.itemsize+handle.tell() == entry.file_size,
                        'Helper NPY declared bytes do not match its shape')
    with np.load(path, allow_pickle=False, max_header_size=4096) as archive:
        data = {name: archive[name] for name in ARRAY_NAMES}
    nv, nt = count*4, count*4
    schema = {'positions':((nv,3),'<f4'), 'normals':((nv,3),'<f4'),
        'triangles':((nt,3),'<i4'), 'joints':((nv,4),'<u2'), 'weights':((nv,4),'<f4'),
        'uv_gltf':((nv,2),'<f4'), 'root_vertex_ids':((count,2),'<i4'),
        'root_face_ids':((count,),'<i8'), 'root_corner_ids':((count,2),'<i8'),
        'root_source_uv':((count,2,2),'<f8'), 'material_ids':((count,),'<i4'),
        'native_vertex_to_helper_vertex':((count*12,),'<i4'),
        'destination_for_source_fin':((count,),'<i4'), 'source_fin_at_destination':((count,),'<i4')}
    for name,(shape,dtype) in schema.items():
        value = data[name]
        require(value.shape == shape and value.dtype == np.dtype(dtype) and np.isfinite(value).all(),
                'Helper array violates exact schema: '+name)
    require(np.abs(data['positions']).max() <= 10000
            and np.abs(np.linalg.norm(data['normals'].astype(np.float64),axis=1)-1).max() < 1e-6,
            'Invalid helper positions or unit normals')
    require(np.all((data['uv_gltf'] >= 0)&(data['uv_gltf'] <= 1))
            and np.abs(data['root_source_uv']).max() <= 1e6, 'Invalid helper UVs')
    require(data['joints'].max() < 64 and np.all((data['weights'] >= 0)&(data['weights'] <= 1))
            and np.abs(data['weights'].astype(np.float64).sum(axis=1)-1).max() < 1e-6,
            'Invalid helper four-weight skin')
    for joints, weights in zip(data['joints'],data['weights']):
        active = joints[weights > 0]
        require(len(active) == len(set(map(int,active))), 'Duplicate positive helper joint influences')
    front = (np.arange(count)[:,None,None]*4+np.array([[0,1,3],[0,3,2]])).reshape(-1,3)
    expected_triangles = np.concatenate((front,front[:,::-1]))
    require(np.array_equal(data['triangles'],expected_triangles)
            and np.array_equal(data['native_vertex_to_helper_vertex'],expected_triangles.ravel()),
            'Helper does not use exact opposed twelve-corner fin topology')
    for name in ('root_vertex_ids','root_face_ids'):
        require(data[name].min() >= 0 and data[name].max() < 100000, 'Invalid source root identifiers')
    require(data['root_corner_ids'].min() >= 0 and data['root_corner_ids'].max() < 3
            and data['material_ids'].min() >= 0 and data['material_ids'].max() < 32, 'Invalid source corner/material identifiers')
    order = np.arange(count)
    a,b = data['destination_for_source_fin'],data['source_fin_at_destination']
    require(np.array_equal(np.sort(a),order) and np.array_equal(np.sort(b),order)
            and np.array_equal(b[a],order), 'Invalid fin-cell permutation')
    for name in ('normals','joints','weights'):
        rows = data[name].reshape(count,4,-1)
        require(np.array_equal(rows[:,:2],rows[:,2:]), 'Helper tips do not inherit their own roots: '+name)
    return data


def _identity_node(node):
    require('matrix' not in node or not {'translation','rotation','scale'} & set(node),
            'Mixed fringe matrix/TRS transform is unsupported')
    for name,expected in (('matrix',[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]),
                          ('translation',[0,0,0]),('rotation',[0,0,0,1]),('scale',[1,1,1])):
        value = node.get(name,expected)
        require(isinstance(value,list) and len(value) == len(expected)
                and all(type(v) in (int,float) and math.isfinite(v) for v in value)
                and value == expected, 'Fringe mesh/ancestor transform is unsupported')
    require('weights' not in node, 'Morphed fringe node is unsupported')


def _skin_map(doc, mesh_id, names):
    nodes = doc['nodes']
    owners = [i for i,node in enumerate(nodes) if node.get('mesh') == mesh_id]
    require(len(owners) == 1, 'Fringe requires one unambiguous skinned mesh node')
    owner = owners[0]
    parents = {}
    for i,node in enumerate(nodes):
        for child in node.get('children',[]):
            _index(child,len(nodes),'node child')
            require(child not in parents, 'Node has multiple parents')
            parents[child] = i
    chain, current = [], owner
    while current is not None:
        require(current not in chain, 'Cyclic node ancestry')
        chain.append(current)
        _identity_node(nodes[current])
        current = parents.get(current)
    scene = doc['scenes'][_index(doc.get('scene',0),len(doc.get('scenes',[])),'active scene')]
    require(chain[-1] in scene.get('nodes',[]), 'Fringe mesh is absent from the active scene')
    for animation in doc.get('animations',[]):
        require(all(channel.get('target',{}).get('node') not in chain for channel in animation.get('channels',[])),
                'Animated fringe mesh/ancestor transform is unsupported')
    skin_id = _index(nodes[owner].get('skin'),len(doc.get('skins',[])),'fringe skin')
    skin = doc['skins'][skin_id]
    joints = skin.get('joints')
    require(isinstance(joints,list) and 1 <= len(joints) <= 64 and len(set(joints)) == len(joints), 'Invalid fringe skin joints')
    actual = [nodes[_index(i,len(nodes),'skin joint')].get('name') for i in joints]
    require(all(isinstance(name,str) and name for name in actual) and len(set(actual)) == len(actual)
            and set(names) == set(actual), 'Fringe bone names do not match the bound native order')
    return np.array([actual.index(name) for name in names],dtype=np.uint16), owner, skin_id, actual


def _exclusive_range(glb, accessor_id, mesh_id, primitive_id, semantic):
    doc = glb.doc
    accessor = doc['accessors'][accessor_id]
    view_id = accessor['bufferView']
    view = doc['bufferViews'][view_id]
    width = 3 if semantic == 'NORMAL' else 4
    size = {5121:1,5123:2,5126:4}[accessor['componentType']]*width
    require(accessor.get('byteOffset',0) == 0 and view.get('byteStride',size) == size
            and view['byteLength'] == accessor['count']*size
            and 'min' not in accessor and 'max' not in accessor,
            'Restoration requires exclusive contiguous storage without stale bounds')
    require(all(i == accessor_id or other.get('bufferView') != view_id for i,other in enumerate(doc['accessors'])),
            'Fringe attribute view is shared by another accessor')
    begin, end = view.get('byteOffset',0),view.get('byteOffset',0)+view['byteLength']
    require(all(i == view_id or max(begin,other.get('byteOffset',0)) >=
                min(end,other.get('byteOffset',0)+other['byteLength']) for i,other in enumerate(doc['bufferViews'])),
            'Fringe attribute storage overlaps another view')
    uses = []
    for mi,mesh in enumerate(doc['meshes']):
        for pi,primitive in enumerate(mesh['primitives']):
            uses.extend((mi,pi,key) for key,value in primitive.get('attributes',{}).items() if value == accessor_id)
            require(primitive.get('indices') != accessor_id
                    and all(accessor_id not in target.values() for target in primitive.get('targets',[])),
                    'Fringe accessor is shared by indices or morphs')
    require(uses == [(mesh_id,primitive_id,semantic)], 'Fringe accessor ownership is ambiguous')
    require(all(image.get('bufferView') != view_id for image in doc.get('images',[]))
            and all(skin.get('inverseBindMatrices') != accessor_id for skin in doc.get('skins',[]))
            and all(sampler.get(key) != accessor_id for animation in doc.get('animations',[])
                    for sampler in animation.get('samplers',[]) for key in ('input','output')),
            'Fringe storage aliases image, inverse-bind or animation data')
    return begin,end


def _oriented_triangles(faces):
    # Cyclic corner rotations preserve winding; reversed triples remain distinct.
    return Counter(min(tuple(row),tuple(np.roll(row,-1)),tuple(np.roll(row,-2))) for row in faces)


def restore_authored_fringe(source, target, correspondence, native_report, *, experimental=False, expected_hashes=None):
    """Restore only three exclusive fringe attributes after exact correspondence."""
    require(experimental is True, 'Experimental fringe fidelity is disabled by default')
    require(isinstance(expected_hashes,dict) and set(expected_hashes) == {'source','correspondence','native_report'}
            and all(isinstance(value,str) and len(value) == 64 and set(value) <= set('0123456789abcdef')
                    for value in expected_hashes.values()), 'Require exact frozen input hashes')
    started, memory_samples = time.monotonic(), []
    _limits(started,memory_samples)
    paths = {'source':local_file(source,MAX_BYTES), 'correspondence':local_file(correspondence,MAX_NPZ_BYTES),
             'native_report':local_file(native_report,1024**2)}
    require(all(sha256(path) == expected_hashes[name] for name,path in paths.items()), 'Frozen fidelity input hash changed')
    target = Path(target)
    require(not target.exists() and not target.is_symlink() and target.parent.is_dir()
            and not target.parent.is_symlink() and target.resolve() not in paths.values(), 'Require a fresh local derivative target')
    report = read_json(paths['native_report'])
    require(isinstance(report,dict) and isinstance(report.get('geometry'),dict)
            and isinstance(report.get('material'),dict), 'Invalid native fringe report')
    count = report.get('cards')
    require(type(count) is int and 1 <= count <= MAX_FINS and report.get('vertices') == count*12
            and report.get('geometry_profile') == 'edge-fringe-v1'
            and report.get('geometry',{}).get('profile') == 'edge-fringe-source-surface-v1'
            and report.get('geometry_arrays_npz') == paths['correspondence'].name,
            'Invalid native fringe report/profile/counts')
    names = report.get('bone_order')
    require(isinstance(names,list) and 1 <= len(names) <= 64
            and all(isinstance(name,str) and 1 <= len(name) <= 128 for name in names)
            and len(set(names)) == len(names), 'Invalid native bone-name order')
    helper = _load_helper(paths['correspondence'],count)
    require(helper['joints'].max() < len(names), 'Helper names an absent native bone')
    glb = _GLB(paths['source'])
    raw = bytearray(paths['source'].read_bytes())
    require(hashlib.sha256(raw).hexdigest() == glb.hash, 'Source changed while reading')
    json_size = struct.unpack_from('<I',raw,12)[0]
    require(_strict_json(raw[20:20+json_size]) == glb.doc, 'Ambiguous source JSON')
    doc = glb.doc
    material_name = report.get('material',{}).get('material')
    require(isinstance(material_name,str) and material_name, 'Native material name is missing')
    matches = [i for i,material in enumerate(doc['materials']) if material.get('name') == material_name]
    require(len(matches) == 1, 'Reported fringe material is missing or ambiguous')
    material_id = matches[0]
    _material(doc['materials'][material_id])
    selected, total_vertices, total_indices = [], 0, 0
    for mi,mesh in enumerate(doc['meshes']):
        for pi,primitive in enumerate(mesh['primitives']):
            attrs = primitive['attributes']
            position = doc['accessors'][_index(attrs.get('POSITION'),len(doc['accessors']),'position accessor')]
            indices = doc['accessors'][_index(primitive.get('indices'),len(doc['accessors']),'indices')]
            require(type(position.get('count')) is int and type(indices.get('count')) is int
                    and position['count'] > 0 and indices['count'] > 0, 'Invalid source counts')
            total_vertices += position['count']; total_indices += indices['count']
            require(total_vertices <= 100000 and total_indices <= 300000, 'Source exceeds100k vertex/triangle bounds')
            if primitive.get('material') == material_id:
                selected.append((mi,pi,primitive))
                require('weights' not in mesh, 'Morphed fringe mesh is unsupported')
    require(len(selected) == 1, 'Require one exact reported fringe primitive')
    mi,pi,primitive = selected[0]
    attrs = primitive['attributes']
    require(set(attrs) == {'POSITION','NORMAL','TEXCOORD_0','JOINTS_0','WEIGHTS_0'}
            and primitive.get('mode',4) == 4 and type(primitive.get('mode',4)) is int
            and 'targets' not in primitive and 'extensions' not in primitive,
            'Require omitted unused tangents and exactly one skinned UV0 attribute set')
    arrays = {key:_array(glb,attrs[key],kind,components) for key,kind,components in
        (('POSITION','VEC3',{5126}),('NORMAL','VEC3',{5126}),('TEXCOORD_0','VEC2',{5126}),
         ('JOINTS_0','VEC4',{5121,5123}),('WEIGHTS_0','VEC4',{5126}))}
    require(all(len(value) == count*12 for value in arrays.values()), 'Export does not retain twelve corners per fin')
    indices = _array(glb,primitive['indices'],'SCALAR',{5121,5123,5125},300000).ravel()
    require(len(indices) == count*12 and indices.max() < count*12
            and np.all(np.bincount(indices,minlength=count*12) == 1), 'Every exported fringe corner must be indexed exactly once')
    bone_map, owner, skin_id, exported_names = _skin_map(doc,mi,names)
    require(arrays['JOINTS_0'].max() < len(exported_names) and np.all(arrays['WEIGHTS_0'] >= 0)
            and np.abs(arrays['WEIGHTS_0'].astype(np.float64).sum(axis=1)-1).max() < 1e-5,
            'Invalid original exported fringe skin')
    texture = doc['materials'][material_id]['pbrMetallicRoughness']['baseColorTexture']['index']
    image = doc['images'][doc['textures'][texture]['source']]
    image_view = doc['bufferViews'][image['bufferView']]
    image_bytes = glb.binary[image_view.get('byteOffset',0):image_view.get('byteOffset',0)+image_view['byteLength']]
    require(hashlib.sha256(image_bytes).hexdigest() == report['material'].get('sha256'),
            'Native fringe atlas hash does not bind the selected exported material')
    _limits(started,memory_samples)
    points = helper['positions'][:,[0,2,1]].copy(); points[:,2] *= -1
    uv = helper['uv_gltf'].copy()
    uv[:,1] = np.float32(1)-np.float32(np.float32(1)-uv[:,1])
    source_keys = np.column_stack((points,uv)).astype('<f4')
    target_keys = np.column_stack((arrays['POSITION'],arrays['TEXCOORD_0'])).astype('<f4')
    source_keys[source_keys == 0] = 0; target_keys[target_keys == 0] = 0
    lookup = {row.tobytes():i for i,row in enumerate(source_keys)}
    require(len(lookup) == count*4, 'Ambiguous authored position/UV keys')
    mapping = np.empty(count*12,dtype='<i4')
    for i,row in enumerate(target_keys):
        if i % 2048 == 0: _limits(started,memory_samples)
        key = row.tobytes()
        require(key in lookup, 'Exported fringe position/UV lacks an exact authored key')
        mapping[i] = lookup[key]
    require(_oriented_triangles(mapping[indices].reshape(-1,3)) == _oriented_triangles(helper['triangles']),
            'Fringe winding or triangle multiplicity differs from authored geometry')
    require(np.array_equal(np.bincount(mapping,minlength=count*4),
                           np.bincount(helper['triangles'].ravel(),minlength=count*4)),
            'Fringe corner multiplicity differs from authored geometry')
    normals = helper['normals'][:,[0,2,1]].copy(); normals[:,2] *= -1
    replacements = {'NORMAL':normals[mapping], 'JOINTS_0':bone_map[helper['joints'][mapping]],
                    'WEIGHTS_0':helper['weights'][mapping]}
    ranges, changed = [], []
    binary_start = 28+json_size
    for semantic in REWRITES:
        begin,end = _exclusive_range(glb,attrs[semantic],mi,pi,semantic)
        value = replacements[semantic].astype(arrays[semantic].dtype)
        encoded = value.tobytes(order='C')
        require(len(encoded) == end-begin, 'Restored attribute byte count changed')
        old = bytes(raw[binary_start+begin:binary_start+end])
        raw[binary_start+begin:binary_start+end] = encoded
        ranges.append((begin,end))
        changed.append({'semantic':semantic,'accessor':attrs[semantic],'buffer_view':doc['accessors'][attrs[semantic]]['bufferView'],
                        'binary_offset':begin,'bytes':end-begin,'old_sha256':hashlib.sha256(old).hexdigest(),
                        'new_sha256':hashlib.sha256(encoded).hexdigest(),
                        'changed_vertices':int(np.count_nonzero(np.any(arrays[semantic] != value,axis=1)))})
    original = paths['source'].read_bytes()
    cursor = 0
    for begin,end in sorted(ranges):
        require(raw[cursor:binary_start+begin] == original[cursor:binary_start+begin], 'Non-attribute bytes changed')
        cursor = binary_start+end
    require(raw[cursor:] == original[cursor:] and len(raw) <= MAX_BYTES,
            'Trailing/non-attribute bytes or output bounds changed')
    require(all(sha256(path) == expected_hashes[name] for name,path in paths.items()), 'Frozen fidelity input changed')
    _limits(started,memory_samples)
    with target.open('xb') as handle:
        handle.write(raw)
    require(sha256(target) == hashlib.sha256(raw).hexdigest(), 'Written fidelity derivative differs')
    return {'schema_version':1,'policy':POLICY,'experimental':True,'public_guard_enabled':False,
        'input_sha256':dict(expected_hashes),'result_sha256':sha256(target),'source_unchanged':True,
        'mesh':mi,'primitive':pi,'material':material_id,'material_name':material_name,'mesh_node':owner,'skin':skin_id,
        'native_bone_order':names,'exported_bone_order':exported_names,'native_to_exported_joint_slots':bone_map.tolist(),
        'fins':count,'exported_corners':count*12,'unique_authored_position_uv_keys':count*4,
        'correspondence_sha256':_hash_array(mapping),'normalization':'No new normal/weight normalization or threshold change',
        'frame':'Applied Blender XYZ meters to glTF XYZ meters: [x,z,-y]',
        'uv_policy':'Exact float32 adapter/exporter double V flip; signed zero numerically canonicalized for keys only',
        'oriented_triangle_multiplicity_exact':True,'changed_attributes':changed,
        'json_and_all_other_bytes_unchanged':True,'all_original_accessors_and_views_retained':True,
        'limits':{'file_bytes':MAX_BYTES,'npz_decoded_bytes':MAX_NPZ_BYTES,'vertices':100000,'triangles':100000,
                  'bones':64,'seconds':30,'sampled_process_rss_bytes':512*1024**2},
        'sampled_peak_rss_bytes':max(memory_samples),'elapsed_seconds':time.monotonic()-started,
        'quality_approved':False,'motion_verified':False,'body_fidelity_verified':False,
        'limitation':'Exact helper parity is not proof of body-root attachment, motion, inherited body quality, or completion of a failed native run.'}
