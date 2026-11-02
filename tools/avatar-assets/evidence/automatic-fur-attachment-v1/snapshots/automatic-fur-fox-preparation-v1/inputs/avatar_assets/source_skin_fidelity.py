"""Experimental hash-bound restoration of complete authored skin weights.

Temporary global vertex IDs prove correspondence across exported seams. Only
JOINTS_0/WEIGHTS_0 bytes and those ID bindings may change; normals remain intact.
"""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import zipfile

import numpy as np

from avatar_assets.fringe_export import _array, _index, _strict_json
from avatar_assets.runner import resident_bytes
from avatar_assets.safety import local_file, read_json, require, sha256
from avatar_assets.three_d import _GLB

POLICY = 'experimental-source-skin-fidelity-v1'
SNAPSHOT_POLICY = 'source-skin-id-v1'
ATTRIBUTE = '_AUTHOR_VERTEX_ID'
MAX_BYTES = 64*1024**2
MAX_DECODED = 8*1024**2
NAMES = {'positions','joints','weights','triangles','triangle_materials'}


def _budget(started, samples):
    require(time.monotonic()-started < 30, 'Source skin restoration exceeded30seconds')
    memory = resident_bytes(os.getpid())
    require(0 < memory <= 512*1024**2, 'Source skin restoration exceeded512MiB sampled RSS')
    samples.append(memory)


def _array_hash(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def _name(value):
    return (isinstance(value,str) and 0 < len(value.encode('utf-8')) <= 255
            and not any(ord(c) < 32 for c in value))


def _snapshot(path, report):
    require(isinstance(report,dict) and report.get('schema_version') == 1
            and report.get('policy') == SNAPSHOT_POLICY and report.get('attribute') == ATTRIBUTE
            and report.get('space') == 'applied_rest_Blender_XYZ_meters', 'Unsupported source-skin snapshot')
    n,t = report.get('vertex_count'),report.get('triangle_count')
    require(type(n) is int and 3 <= n <= 100000 and type(t) is int and 1 <= t <= 100000,
            'Snapshot geometry exceeds100k bounds')
    names = report.get('bone_names')
    require(isinstance(names,list) and 1 <= len(names) <= 64
            and all(_name(x) for x in names)
            and len(set(names)) == len(names), 'Invalid snapshot bone names')
    require(isinstance(report.get('rig'),dict) and report['rig'].get('pose_position') == 'REST',
            'Snapshot requires actual REST data')
    require(report.get('npz_file') == path.name and report.get('npz_sha256') == sha256(path)
            and report.get('npz_bytes') == path.stat().st_size, 'Snapshot NPZ binding changed')
    schema = {'positions':((n,3),'<f4'),'joints':((n,4),'<u2'),'weights':((n,4),'<f4'),
              'triangles':((t,3),'<u4'),'triangle_materials':((t,),'<u2')}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) == 5 and {i.filename for i in entries} == {name+'.npy' for name in NAMES},
                'Unknown or duplicate source-skin NPZ schema')
        require(sum(i.file_size for i in entries) <= MAX_DECODED, 'Source-skin NPZ decode exceeds8MiB')
        for entry in entries:
            require(not entry.flag_bits & 1 and entry.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED),
                    'Unsupported NPZ entry encoding')
            with archive.open(entry) as handle:
                version = np.lib.format.read_magic(handle)
                require(version in ((1,0),(2,0)), 'Unsupported NPY version')
                read = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
                shape,fortran,dtype = read(handle,max_header_size=4096)
                expected_shape,expected_dtype = schema[entry.filename[:-4]]
                require(shape == expected_shape and dtype == np.dtype(expected_dtype) and not fortran
                        and not dtype.hasobject and int(np.prod(shape))*dtype.itemsize+handle.tell() == entry.file_size,
                        'Source-skin NPY shape/type/length mismatch')
    with np.load(path,allow_pickle=False,max_header_size=4096) as archive:
        arrays = {name:archive[name] for name in NAMES}
    require(isinstance(report.get('arrays'),dict) and set(report['arrays']) == NAMES,
            'Missing exact source-skin array metadata')
    for name,value in arrays.items():
        meta = report['arrays'][name]
        require(np.isfinite(value).all() and isinstance(meta,dict)
                and meta.get('dtype') == value.dtype.str and meta.get('shape') == list(value.shape)
                and meta.get('sha256') == _array_hash(value), 'Snapshot array content hash mismatch: '+name)
    require(np.abs(arrays['positions']).max() <= 10000 and arrays['joints'].max() < len(names)
            and np.all((arrays['weights'] >= 0)&(arrays['weights'] <= 1))
            and np.abs(arrays['weights'].astype(np.float64).sum(axis=1)-1).max() <= 1e-6,
            'Invalid snapshot coordinates/skin')
    for joints,weights in zip(arrays['joints'],arrays['weights']):
        active = joints[weights > 0]
        require(len(active) == len(set(map(int,active))), 'Duplicate positive snapshot bone influences')
    triangles = arrays['triangles']
    require(triangles.max() < n and np.all(triangles[:,0] != triangles[:,1])
            and np.all(triangles[:,1] != triangles[:,2]) and np.all(triangles[:,2] != triangles[:,0])
            and np.array_equal(np.unique(arrays['triangles']),np.arange(n)), 'Snapshot contains invalid or loose source IDs')
    owners = report.get('owners')
    require(isinstance(owners,list) and 1 <= len(owners) <= 32, 'Invalid snapshot owner inventory')
    vertex_start = triangle_start = 0
    seen_names,seen_meshes = set(),set()
    for index,owner in enumerate(owners):
        require(isinstance(owner,dict) and set(owner) == {'index','name','mesh_name','vertex_start','vertex_count',
                'triangle_start','triangle_count','materials','ignored_non_skin_groups'}, 'Invalid snapshot owner schema')
        for key,seen in (('name',seen_names),('mesh_name',seen_meshes)):
            require(_name(owner[key]) and owner[key] not in seen, 'Ambiguous snapshot owner identity')
            seen.add(owner[key])
        require(owner['index'] == index and type(owner['index']) is int
                and type(owner['vertex_start']) is int and type(owner['triangle_start']) is int
                and owner['vertex_start'] == vertex_start and owner['triangle_start'] == triangle_start
                and all(type(owner[k]) is int and owner[k] > 0 for k in ('vertex_count','triangle_count')),
                'Source-skin owner ranges must partition arrays')
        materials = owner['materials']
        require(isinstance(materials,list) and 1 <= len(materials) <= 128
                and all(_name(name) for name in materials), 'Invalid owner material names')
        ignored = owner['ignored_non_skin_groups']
        require(isinstance(ignored,list) and len(ignored) <= 128 and all(_name(name) for name in ignored)
                and len(set(ignored)) == len(ignored) and not set(ignored)&set(names), 'Invalid ignored non-skin groups')
        end_v,end_t = vertex_start+owner['vertex_count'],triangle_start+owner['triangle_count']
        require(end_v <= n and end_t <= t and arrays['triangles'][triangle_start:end_t].min() >= vertex_start
                and arrays['triangles'][triangle_start:end_t].max() < end_v
                and arrays['triangle_materials'][triangle_start:end_t].max() < len(materials),
                'Source triangle crosses owner or material ranges')
        vertex_start,triangle_start = end_v,end_t
    require(vertex_start == n and triangle_start == t, 'Snapshot owner coverage is incomplete')
    return arrays,owners,names


def _node_owner(doc, mesh_index, owner, names):
    nodes = doc['nodes']
    matches = [i for i,node in enumerate(nodes) if node.get('mesh') == mesh_index]
    require(len(matches) == 1 and nodes[matches[0]].get('name') == owner['name']
            and doc['meshes'][mesh_index].get('name') == owner['mesh_name'], 'Exported owner node/mesh identity differs')
    node_id = matches[0]
    parents = {}
    for i,node in enumerate(nodes):
        for child in node.get('children',[]):
            _index(child,len(nodes),'node child'); require(child not in parents,'Node has multiple parents')
            parents[child] = i
    chain,current = [],node_id
    while current is not None:
        require(current not in chain,'Cyclic owner ancestry')
        chain.append(current); node = nodes[current]
        require('matrix' not in node or not {'translation','rotation','scale'} & set(node),'Mixed node matrix/TRS')
        for key,expected in (('matrix',[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]),
                             ('translation',[0,0,0]),('rotation',[0,0,0,1]),('scale',[1,1,1])):
            value = node.get(key,expected)
            require(isinstance(value,list) and len(value) == len(expected)
                    and all(type(x) in (int,float) and math.isfinite(x) for x in value)
                    and value == expected,'Nonidentity source-skin owner frame')
        require('weights' not in node,'Morphed source-skin owner is unsupported')
        current = parents.get(current)
    scene = doc['scenes'][_index(doc.get('scene',0),len(doc.get('scenes',[])),'active scene')]
    require(chain[-1] in scene.get('nodes',[]),'Source owner is absent from the active scene')
    require(all(channel.get('target',{}).get('node') not in chain for animation in doc.get('animations',[])
                for channel in animation.get('channels',[])), 'Animated owner frame is unsupported')
    skin_id = _index(nodes[node_id].get('skin'),len(doc.get('skins',[])),'source skin')
    skin = doc['skins'][skin_id]
    ids = skin.get('joints')
    require(isinstance(ids,list) and 1 <= len(ids) <= 64 and len(set(ids)) == len(ids),'Invalid skin joint set')
    actual = [nodes[_index(i,len(nodes),'bone')].get('name') for i in ids]
    require(all(isinstance(name,str) and name for name in actual)
            and len(set(actual)) == len(actual) and set(actual) == set(names),'Snapshot/exported bone names differ')
    return np.array([actual.index(name) for name in names],dtype=np.uint16),node_id,skin_id


def _layout(doc,index,binary_size):
    accessor = doc['accessors'][_index(index,len(doc['accessors']),'accessor')]
    require('sparse' not in accessor,'Sparse accessor storage is unsupported')
    size = {5120:1,5121:1,5122:2,5123:2,5125:4,5126:4}.get(accessor.get('componentType'))
    kind = accessor.get('type'); width = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT2':4,'MAT3':9,'MAT4':16}.get(kind)
    count = accessor.get('count')
    require(size is not None and width is not None and type(count) is int and 0 < count <= 300000,
            'Invalid bounded accessor layout')
    item_size = size*width
    if kind.startswith('MAT'):
        dimension = int(kind[-1]); item_size = dimension*((dimension*size+3)//4*4)
    view_id = _index(accessor.get('bufferView'),len(doc['bufferViews']),'accessor view')
    view = doc['bufferViews'][view_id]
    start,stride = accessor.get('byteOffset',0),view.get('byteStride',item_size)
    require(type(start) is int and start >= 0 and type(stride) is int and item_size <= stride <= 252
            and start+(count-1)*stride+item_size <= view['byteLength'],'Accessor range exceeds its view')
    absolute = view.get('byteOffset',0)+start
    require(absolute+(count-1)*stride+item_size <= binary_size,'Accessor exceeds binary bytes')
    return view_id,absolute,stride,count,item_size


def _canonical(row):
    a,b,c = map(int,row)
    return min((a,b,c),(b,c,a),(c,a,b))


def restore_source_skin(source,target,snapshot,snapshot_report,*,experimental=False,expected_hashes=None):
    require(experimental is True,'Experimental source-skin fidelity is disabled by default')
    require(isinstance(expected_hashes,dict) and set(expected_hashes) == {'source','snapshot','snapshot_report'}
            and all(isinstance(x,str) and len(x) == 64 and set(x) <= set('0123456789abcdef')
                    for x in expected_hashes.values()),'Require exact frozen source-skin input hashes')
    started,samples = time.monotonic(),[]
    _budget(started,samples)
    paths = {'source':local_file(source,MAX_BYTES),'snapshot':local_file(snapshot,MAX_DECODED),
             'snapshot_report':local_file(snapshot_report,1024**2)}
    require(all(sha256(path) == expected_hashes[key] for key,path in paths.items()),'Frozen source-skin input hash changed')
    target = Path(target)
    require(not target.exists() and not target.is_symlink() and target.parent.is_dir()
            and not target.parent.is_symlink() and target.resolve() not in paths.values(),'Require a fresh local target')
    report = read_json(paths['snapshot_report'])
    arrays,owners,names = _snapshot(paths['snapshot'],report)
    glb = _GLB(paths['source']); doc = glb.doc
    with paths['source'].open('rb') as handle:
        prefix = handle.read(20); json_size = struct.unpack_from('<I',prefix,12)[0]
        original_json = handle.read(json_size)
        require(_strict_json(original_json) == doc,'Ambiguous source JSON')
        handle.seek(28+json_size+len(glb.binary)); padding = handle.read(3)
    require(sha256(paths['source']) == glb.hash,'Source changed during parse')
    binary = bytearray(glb.binary); binary.extend(padding)
    mask = np.zeros(len(binary),dtype=np.uint8)
    updated = deepcopy(doc)
    owner_by_name = {owner['mesh_name']:owner for owner in owners}
    require(len(doc['meshes']) == len(owners),'Exported mesh inventory differs from selected owners')
    material_names = [material.get('name') for material in doc.get('materials',[])]
    require(all(isinstance(name,str) and name for name in material_names)
            and len(set(material_names)) == len(material_names),'Ambiguous exported material names')
    seen_ids = np.zeros(report['vertex_count'],dtype=bool)
    plans,uses,primitive_rows,changed_paths = {},{},[],[]
    joint_nodes = None
    observed_triangles,expected_triangles = Counter(),Counter()
    total_vertices = total_triangles = 0
    for owner in owners:
        begin,end = owner['triangle_start'],owner['triangle_start']+owner['triangle_count']
        for triangle,slot in zip(arrays['triangles'][begin:end],arrays['triangle_materials'][begin:end]):
            expected_triangles[(owner['index'],owner['materials'][int(slot)],_canonical(triangle))] += 1
    for mi,mesh in enumerate(doc['meshes']):
        _budget(started,samples)
        require(mesh.get('name') in owner_by_name and 'weights' not in mesh,'Unknown/morphed exported owner mesh')
        owner = owner_by_name[mesh['name']]
        bone_map,node_id,skin_id = _node_owner(doc,mi,owner,names)
        current_nodes = [doc['skins'][skin_id]['joints'][int(slot)] for slot in bone_map]
        require(joint_nodes is None or current_nodes == joint_nodes, 'Owners do not share the same named rig nodes')
        joint_nodes = current_nodes
        for pi,primitive in enumerate(mesh.get('primitives',[])):
            attrs = primitive.get('attributes',{})
            require({ATTRIBUTE,'POSITION','JOINTS_0','WEIGHTS_0'} <= set(attrs)
                    and not any(key.startswith(('JOINTS_','WEIGHTS_')) and key not in ('JOINTS_0','WEIGHTS_0') for key in attrs)
                    and type(primitive.get('mode',4)) is int and primitive.get('mode',4) == 4
                    and 'targets' not in primitive and 'extensions' not in primitive,'Every source primitive requires one ID-bound four-weight skin')
            ids_float = _array(glb,attrs[ATTRIBUTE],'SCALAR',{5126}).ravel()
            require(np.array_equal(ids_float,np.floor(ids_float)) and ids_float.min() >= owner['vertex_start']
                    and ids_float.max() < owner['vertex_start']+owner['vertex_count'],'Exported global IDs are invalid or cross owners')
            ids = ids_float.astype(np.int64); seen_ids[ids] = True
            positions = _array(glb,attrs['POSITION'],'VEC3',{5126})
            expected = arrays['positions'][ids][:,[0,2,1]].copy(); expected[:,2] *= -1
            require(np.array_equal(positions,expected),'Exported positions do not exactly match source IDs and axes')
            original_joints = _array(glb,attrs['JOINTS_0'],'VEC4',{5121,5123})
            original_weights = _array(glb,attrs['WEIGHTS_0'],'VEC4',{5126})
            require(len(original_joints) == len(original_weights) == len(ids)
                    and original_joints.max() < len(names) and np.all(original_weights >= 0)
                    and np.abs(original_weights.astype(np.float64).sum(axis=1)-1).max() <= 1e-5,'Invalid exported source skin')
            triangles = _array(glb,primitive.get('indices'),'SCALAR',{5121,5123,5125},300000).ravel()
            require(len(triangles)%3 == 0 and triangles.max() < len(ids),'Invalid source primitive triangle indices')
            total_vertices += len(ids); total_triangles += len(triangles)//3
            require(total_vertices <= 100000 and total_triangles <= 100000,'Export exceeds100k vertices/triangles')
            material_id = _index(primitive.get('material'),len(material_names),'primitive material')
            material = material_names[material_id]
            require(material in owner['materials'],'Primitive material is absent from its owner snapshot')
            for triangle in ids[triangles].reshape(-1,3):
                observed_triangles[(owner['index'],material,_canonical(triangle))] += 1
            values = {'JOINTS_0':bone_map[arrays['joints'][ids]].astype(original_joints.dtype),
                      'WEIGHTS_0':arrays['weights'][ids]}
            for key,value in values.items():
                aid = attrs[key]
                if aid in plans:
                    require(np.array_equal(plans[aid],value),'Shared skin accessor demands conflicting source values')
                else:
                    plans[aid] = value
                uses.setdefault(aid,[]).append((mi,pi,key))
            changed_paths.append(f'/meshes/{mi}/primitives/{pi}/attributes/{ATTRIBUTE}')
            del updated['meshes'][mi]['primitives'][pi]['attributes'][ATTRIBUTE]
            primitive_rows.append({'mesh':mi,'primitive':pi,'owner':owner['index'],'node':node_id,'skin':skin_id,
                                   'vertices':len(ids),'source_ids_sha256':_array_hash(ids.astype('<u4')),
                                   'native_to_exported_joint_slots':bone_map.tolist()})
    require(seen_ids.all() and observed_triangles == expected_triangles,
            'Source ID coverage or oriented owner/material triangle multiplicity differs')
    # Every live alias must be a validated skin reference, not an orphan or another attribute.
    references = {}
    for mi,mesh in enumerate(doc['meshes']):
        for pi,primitive in enumerate(mesh['primitives']):
            for key,aid in primitive['attributes'].items():references.setdefault(aid,[]).append((mi,pi,key))
            references.setdefault(primitive['indices'],[]).append((mi,pi,'indices'))
    for skin in doc.get('skins',[]):
        if 'inverseBindMatrices' in skin:references.setdefault(skin['inverseBindMatrices'],[]).append(('skin',))
    for animation in doc.get('animations',[]):
        for sampler in animation.get('samplers',[]):
            for key in ('input','output'):references.setdefault(sampler[key],[]).append(('animation',key))
    for aid,rows in uses.items():require(references.get(aid) == rows,'Skin accessor has an unproven non-skin alias')
    writes,write_views = [],set()
    for aid,value in plans.items():
        view_id,start,stride,count,width = _layout(doc,aid,len(binary))
        require('min' not in doc['accessors'][aid] and 'max' not in doc['accessors'][aid], 'Skin accessor bounds would become stale')
        require(width == value.dtype.itemsize*4 and count == len(value),'Skin storage shape changed')
        destination = np.ndarray((count,width),dtype=np.uint8,buffer=binary,offset=start,strides=(stride,1))
        touched = np.ndarray((count,width),dtype=np.uint8,buffer=mask,offset=start,strides=(stride,1))
        desired = value.view(np.uint8).reshape(count,width)
        overlap = touched != 0
        require(np.array_equal(destination[overlap],desired[overlap]),'Overlapping skin writes demand conflicting encoded bytes')
        original_value = np.ndarray((count,width),dtype=np.uint8,buffer=glb.binary,offset=start,strides=(stride,1))
        old_hash = _array_hash(original_value)
        changed = int(np.count_nonzero(np.any(original_value != desired,axis=1)))
        destination[:] = desired; touched[:] = 1; write_views.add(view_id)
        writes.append({'accessor':aid,'buffer_view':view_id,'byte_offset':start,'byte_stride':stride,
                       'element_bytes':width,'count':count,'old_sha256':old_hash,'new_sha256':_array_hash(desired),
                       'changed_rows':changed,'uses':[list(row) for row in uses[aid]]})
    for aid in range(len(doc['accessors'])):
        if aid in plans:continue
        _,start,stride,count,width = _layout(doc,aid,len(binary))
        require(not np.ndarray((count,width),dtype=np.uint8,buffer=mask,offset=start,strides=(stride,1)).any(),
                'Skin writes overlap non-skin or orphan accessor bytes')
    image_views = {image['bufferView'] for image in doc.get('images',[])}
    for vi,view in enumerate(doc['bufferViews']):
        if vi not in write_views or vi in image_views:
            start=view.get('byteOffset',0)
            require(not mask[start:start+view['byteLength']].any(),'Skin writes overlap an unproven view/image')
    original = np.frombuffer(glb.binary,dtype=np.uint8)
    current = np.frombuffer(binary,dtype=np.uint8,count=len(original))
    for start in range(0,len(original),1024*1024):
        end=min(start+1024*1024,len(original));keep=mask[start:end] == 0
        require(np.array_equal(original[start:end][keep],current[start:end][keep]),'Non-skin bytes changed')
    encoded = json.dumps(updated,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    encoded += b' '*(-len(encoded)%4)
    length = 28+len(encoded)+len(binary)
    require(length <= MAX_BYTES and len(encoded) <= 1024**2,'Restored GLB exceeds file/JSON bounds')
    require(all(sha256(path) == expected_hashes[key] for key,path in paths.items()),'Frozen input changed before writing')
    _budget(started,samples)
    with target.open('xb') as handle:
        handle.write(struct.pack('<4sII',b'glTF',2,length));handle.write(struct.pack('<II',len(encoded),0x4e4f534a))
        handle.write(encoded);handle.write(struct.pack('<II',len(binary),0x004e4942));handle.write(binary)
    return {'schema_version':1,'policy':POLICY,'experimental':True,'public_guard_enabled':False,
        'input_sha256':dict(expected_hashes),'result_sha256':sha256(target),'source_unchanged':True,
        'source_vertices':report['vertex_count'],'exported_vertex_entries':total_vertices,'triangles':total_triangles,
        'owners':len(owners),'primitives':primitive_rows,'skin_writes':writes,'removed_id_bindings':changed_paths,
        'all_source_ids_represented':True,'oriented_owner_material_triangles_exact':True,
        'all_non_skin_binary_bytes_unchanged':True,'json_changes_only_removed_id_bindings':True,
        'normals_positions_uv_images_rig_actions_unchanged':True,'original_accessors_views_and_id_payloads_retained':True,
        'shared_storage_policy':'All live write aliases demand identical proven source bytes; non-skin/orphan overlap rejected',
        'frame':'Exact applied Blender XYZ meters -> glTF [x,z,-y]; no nearest matching or renormalization',
        'sampled_peak_rss_bytes':max(samples),'elapsed_seconds':time.monotonic()-started,
        'limits':{'file_bytes':MAX_BYTES,'npz_decoded_bytes':MAX_DECODED,'vertices':100000,'triangles':100000,
                  'bones':64,'seconds':30,'sampled_rss_bytes':512*1024**2},
        'motion_verified':False,'body_normal_fidelity_verified':False,'quality_approved':False,
        'limitation':'Exact complete source-skin restoration is not action, appearance, physical validity, or failed-native completion proof.'}
