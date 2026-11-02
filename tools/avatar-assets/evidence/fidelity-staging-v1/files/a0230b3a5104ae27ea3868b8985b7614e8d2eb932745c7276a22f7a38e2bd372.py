"""Independent read-only fidelity audit; never imports or invokes a repair module."""

from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import time
import zipfile

import numpy as np


START = time.monotonic()
MAX_BYTES = 64*1024**2
ROOT = Path(__file__).resolve().parents[2]
WIDTH = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
DTYPE = {5121: 'u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
THRESHOLDS_SHA256 = '4b151d9b77835ee6f7d4231cc0c8b1bcadf9fd5be45fdd45ae628412b6acbedf'


def require(value, message):
    if not value:
        raise ValueError(message)
    if time.monotonic()-START > 50:
        raise ValueError('Independent audit exceeded its50second CPU envelope')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with Path(path).open('x') as handle:
        handle.write(encoded)


def pinned(row):
    path = ROOT/row['path']
    require(path.is_relative_to(ROOT) and path.is_file() and path.absolute() == path.resolve()
            and not path.is_symlink() and path.stat().st_size <= MAX_BYTES, 'Invalid bounded local evidence path')
    require(digest(path) == row['sha256'], 'Evidence hash changed: '+row['path'])
    return path


class GLB:
    def __init__(self, path):
        self.raw = Path(path).read_bytes()
        require(28 <= len(self.raw) <= MAX_BYTES, 'GLB size bound')
        magic, version, total, size, kind = struct.unpack_from('<4sIIII', self.raw)
        require(magic == b'glTF' and version == 2 and total == len(self.raw) and kind == 0x4E4F534A
                and size <= 2*1024**2, 'GLB header')
        self.doc = json.loads(self.raw[20:20+size])
        length, kind = struct.unpack_from('<II', self.raw, 20+size)
        self.start = 28+size
        require(kind == 0x004E4942 and self.start+length == len(self.raw), 'Embedded BIN bounds')
        require(not any('uri' in x for x in self.doc.get('images', [])+self.doc.get('buffers', [])), 'External URI')
        self.binary = memoryview(self.raw)[self.start:]
        self.cache = {}
        self.parents = {}
        for i, node in enumerate(self.doc['nodes']):
            for child in node.get('children', []):
                require(child not in self.parents, 'Shared child node')
                self.parents[child] = i
        require(len(self.doc['nodes']) <= 256, 'Node bound')

    def span(self, index):
        require(type(index) is int and 0 <= index < len(self.doc['accessors']), 'Accessor index')
        a = self.doc['accessors'][index]
        require('sparse' not in a and type(a['count']) is int and 0 < a['count'] <= 300000, 'Sparse/oversized accessor')
        width, dtype = WIDTH[a['type']], np.dtype(DTYPE[a['componentType']])
        v = self.doc['bufferViews'][a['bufferView']]
        require(v.get('buffer', 0) == 0, 'Multiple buffers')
        offset, stride = v.get('byteOffset', 0)+a.get('byteOffset', 0), v.get('byteStride', width*dtype.itemsize)
        stop = offset+(a['count']-1)*stride+width*dtype.itemsize
        require(0 <= offset < stop <= len(self.binary) and stop <= v.get('byteOffset', 0)+v['byteLength'], 'Accessor storage bounds')
        return offset, stop, stride, dtype, width

    def array(self, index):
        if index in self.cache:
            return self.cache[index]
        a = self.doc['accessors'][index]
        begin, end, stride, dtype, width = self.span(index)
        values = np.ndarray((a['count'], width), dtype=dtype, buffer=self.binary, offset=begin,
                            strides=(stride, dtype.itemsize)).copy()
        require(np.isfinite(values).all(), 'Nonfinite accessor')
        if a.get('normalized'):
            require(dtype.kind == 'u', 'Unexpected normalized float')
            values = values.astype(float)/np.iinfo(dtype).max
        self.cache[index] = values
        return values


def rotation(q):
    q = np.asarray(q, dtype=float)
    require(abs(np.linalg.norm(q)-1) < .001, 'Invalid node quaternion')
    x, y, z, w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def slerp(a, b, fraction):
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    dot = float(a@b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        q = a+(b-a)*fraction
        return q/np.linalg.norm(q)
    angle = math.acos(min(1, dot))
    return (a*math.sin((1-fraction)*angle)+b*math.sin(fraction*angle))/math.sin(angle)


def tracks(glb):
    result = {}
    for animation in glb.doc.get('animations', []):
        name = animation.get('name')
        require(isinstance(name, str) and name not in result, 'Unique clip names required')
        rows, targets = [], set()
        for channel in animation['channels']:
            target = channel['target']; key = (target['node'], target['path'])
            require(key not in targets and key[1] in ('translation', 'rotation', 'scale'), 'Unsupported/duplicate animation target')
            targets.add(key)
            sampler = animation['samplers'][channel['sampler']]
            mode = sampler.get('interpolation', 'LINEAR')
            require(mode in ('LINEAR', 'STEP'), 'Unsupported interpolation; no fallback')
            times, values = glb.array(sampler['input']).ravel(), glb.array(sampler['output'])
            require(len(times) == len(values) and len(times) <= 1000 and np.all(np.diff(times) > 0)
                    and 0 <= times[0] < times[-1] <= 30, 'Animation time bounds')
            rows.append((key, times, values, mode))
        require(rows and len(rows) <= 256, 'Empty/oversized animation')
        result[name] = rows
    require(1 <= len(result) <= 8, 'Clip count bound')
    return result


def world(glb, rows, seconds):
    overrides = {}
    for key, times, values, mode in rows:
        index = int(np.searchsorted(times, seconds, side='right')-1)
        if index < 0:
            value = values[0]
        elif index >= len(times)-1 or mode == 'STEP':
            value = values[min(index, len(times)-1)]
        else:
            fraction = float((seconds-times[index])/(times[index+1]-times[index]))
            value = slerp(values[index], values[index+1], fraction) if key[1] == 'rotation' else values[index]*(1-fraction)+values[index+1]*fraction
        overrides[key] = value
    cache, pending = {}, set()

    def node_matrix(index):
        if index in cache:
            return cache[index]
        require(index not in pending, 'Cyclic nodes')
        pending.add(index)
        node = glb.doc['nodes'][index]
        if 'matrix' in node:
            require(not any(key[0] == index for key in overrides), 'Matrix/TRS ambiguity')
            matrix = np.asarray(node['matrix'], dtype=float).reshape(4, 4).T
        else:
            t = overrides.get((index, 'translation'), node.get('translation', [0, 0, 0]))
            q = overrides.get((index, 'rotation'), node.get('rotation', [0, 0, 0, 1]))
            s = overrides.get((index, 'scale'), node.get('scale', [1, 1, 1]))
            matrix = np.eye(4); matrix[:3, :3] = rotation(q)*s; matrix[:3, 3] = t
        cache[index] = node_matrix(glb.parents[index])@matrix if index in glb.parents else matrix
        pending.remove(index)
        return cache[index]
    return np.asarray([node_matrix(i) for i in range(len(glb.doc['nodes']))])


def skin(glb, node, positions, ids, weights, matrices):
    spec = glb.doc['skins'][node['skin']]
    inverse = glb.array(spec['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
    transforms = matrices[spec['joints']]@inverse
    require(ids.min() >= 0 and ids.max() < len(transforms), 'Skin slot outside bound')
    moved = np.zeros((len(positions), 3))
    for column in range(4):
        m = transforms[ids[:, column]]
        moved += (np.einsum('nij,nj->ni', m[:, :3, :3], positions)+m[:, :3, 3])*weights[:, column, None]
    return moved


def canonical(row):
    row = tuple(map(int, row))
    return min(row, row[1:]+row[:1], row[2:]+row[:2])


def original_motion(request, result, body, names, animation, limit, out):
    if 'original_baseline' not in request:
        write(out/'original-body-motion.json', {'status':'not_available',
              'scope':'No original whole-native pose comparator is asserted for this case.'})
        return True
    baseline = GLB(pinned(request['original_baseline']))
    previous = {}
    for node in baseline.doc['nodes']:
        if 'mesh' not in node:continue
        source_names = [baseline.doc['nodes'][j]['name'] for j in baseline.doc['skins'][node['skin']]['joints']]
        require(set(source_names) == set(names), 'Original body skin names changed')
        for primitive in baseline.doc['meshes'][node['mesh']]['primitives']:
            material = baseline.doc['materials'][primitive['material']]['name']; attrs = primitive['attributes']
            require(material not in previous, 'Ambiguous original material binding')
            previous[material] = {'node':node, 'names':source_names,
                'positions':baseline.array(attrs['POSITION']), 'uv':baseline.array(attrs['TEXCOORD_0']),
                'joints':baseline.array(attrs['JOINTS_0']).astype(int), 'weights':baseline.array(attrs['WEIGHTS_0'])}
    require(set(previous) == {b['material'] for b in body}, 'Original body material set changed')

    def keys(record):
        dense = np.zeros((len(record['positions']),len(names)),dtype=np.float32)
        for slot,name in enumerate(record['names']):
            dense[:,names.index(name)] = np.where(record['joints']==slot,record['weights'],0).sum(axis=1)
        return [tuple(np.r_[p,u,w]) for p,u,w in zip(record['positions'],record['uv'],dense)]

    pairs=[]
    for actual in body:
        before=previous[actual['material']]; old_keys=keys(before); lookup={key:i for i,key in enumerate(old_keys)}
        new_keys=keys(actual)
        require(set(new_keys)==set(old_keys), 'Original body position/UV/named-weight groups changed')
        mapping=np.asarray([lookup[key] for key in new_keys],dtype=np.int32)
        pairs.append((before,actual,mapping))
    source_tracks=tracks(baseline)
    require(set(source_tracks)==set(animation), 'Original clip names changed')
    rows=[]
    for name,current in animation.items():
        original=source_tracks[name]
        start=min(t[0] for _,t,_,_ in current);end=max(t[-1] for _,t,_,_ in current)
        require(abs(start-min(t[0] for _,t,_,_ in original))<=1e-6
                and abs(end-max(t[-1] for _,t,_,_ in original))<=1e-6, 'Original clip range changed')
        times=np.linspace(start,end,9);maximum=0.
        for seconds in times:
            old_world=world(baseline,original,float(seconds));new_world=world(result,current,float(seconds))
            for before,actual,mapping in pairs:
                old=skin(baseline,before['node'],before['positions'],before['joints'],before['weights'],old_world)[mapping]
                new=skin(result,actual['node'],actual['positions'],actual['joints'],actual['weights'],new_world)
                maximum=max(maximum,float(np.linalg.norm(new-old,axis=1).max()))
        rows.append({'clip':name,'sample_times_seconds':times.tolist(),'maximum_weighted_pose_error_m':maximum,
                     'passed':maximum<=limit['weighted_pose_m']})
    write(out/'original-body-motion.json',{'status':'passed' if all(r['passed'] for r in rows) else 'failed',
          'clips':rows,'original_weighted_pose_limit_m':limit['weighted_pose_m'],
          'matched_vertices':sum(len(b['positions']) for b in body),
          'correspondence':'Exact position/UV/named weights; aliases with different normals are not normal-fidelity proof.',
          'normal_fidelity_approved':False,'quality_approved':False,
          'native_pose_comparison':'Original GLB versus actual native-export GLB CPU skin; not native evaluated-pose arrays.'})
    pinned(request['original_baseline'])
    return all(r['passed'] for r in rows)


def inspect(request, out):
    thresholds_path = pinned(request['thresholds'])
    require(digest(thresholds_path) == THRESHOLDS_SHA256, 'Independent thresholds cannot be changed by the repair request')
    all_limits = json.loads(thresholds_path.read_text())
    limit = all_limits['cases'][request['case']]
    source_path, result_path = pinned(request['source']), pinned(request['derivative'])
    helper_path, report_path = pinned(request['helper']), pinned(request['native_report'])
    source, result = GLB(source_path), GLB(result_path)
    require(source.doc == result.doc and source.start == result.start and len(source.raw) == len(result.raw), 'Only selected payload bytes may change; JSON/header layout differs')
    with zipfile.ZipFile(helper_path) as z:
        require(len(z.infolist()) <= 24 and sum(i.file_size for i in z.infolist()) <= 16*1024**2, 'Helper archive expansion bound')
    with np.load(helper_path, allow_pickle=False) as saved:
        helper = {key: saved[key].copy() for key in ('positions','normals','triangles','joints','weights','uv_gltf',
                  'root_vertex_ids','root_source_uv','native_vertex_to_helper_vertex')}
    report = json.loads(report_path.read_text())
    name, bone_order = report['material']['material'], report['bone_order']
    require(report['geometry_profile'] == 'edge-fringe-v1' and name == request['material_name'], 'Material/report profile mismatch')
    require(1 <= len(bone_order) <= 64 and len(set(bone_order)) == len(bone_order), 'Ambiguous bone order')
    fins = len(helper['root_vertex_ids']); count = len(helper['positions'])
    require(1 <= fins <= 4800 and count == 4*fins and helper['normals'].shape == (count,3)
            and helper['joints'].shape == helper['weights'].shape == (count,4), 'Helper array dimensions')
    for array in helper.values():
        require(array.dtype.kind in 'uif' and np.isfinite(array).all(), 'Invalid helper array')
    require(all(helper[name].dtype == np.dtype('<f4') for name in ('positions','normals','weights','uv_gltf'))
            and helper['joints'].dtype == np.dtype('<u2') and helper['triangles'].dtype == np.dtype('<i4'),
            'Helper dtype differs from the frozen authoring representation')
    require(np.array_equal(helper['triangles'].ravel(), helper['native_vertex_to_helper_vertex']), 'Helper native-corner map changed')
    require(helper['joints'].min() >= 0 and helper['joints'].max() < len(bone_order)
            and np.all(helper['weights'] >= 0) and np.max(abs(helper['weights'].sum(axis=1)-1)) <= 1e-6, 'Helper skin invalid')
    selected = [(mi, pi, p) for mi,m in enumerate(source.doc['meshes']) for pi,p in enumerate(m['primitives'])
                if source.doc['materials'][p['material']].get('name') == name]
    require(len(selected) == 1, 'Selected material/primitive ambiguous')
    mi, pi, primitive = selected[0]; attrs = primitive['attributes']
    material = source.doc['materials'][primitive['material']]
    require('normalTexture' not in material and 'extensions' not in material and 'TANGENT' not in attrs
            and 'extensions' not in primitive and 'targets' not in primitive, 'Unsupported selected shading/morph route')
    require(not any(k.startswith(('JOINTS_','WEIGHTS_')) and k not in ('JOINTS_0','WEIGHTS_0') for k in attrs), 'Additional skin influence sets')
    instances = [(i,n) for i,n in enumerate(source.doc['nodes']) if n.get('mesh') == mi]
    require(len(instances) == 1, 'Instanced selected mesh is ambiguous')
    node_id, node = instances[0]
    rest_world = world(source, [], 0)
    parent = node_id
    while True:
        require(np.array_equal(rest_world[parent], np.eye(4)), 'Selected mesh/ancestor transform is not identity')
        if parent not in source.parents:break
        parent = source.parents[parent]
    sk = source.doc['skins'][node['skin']]
    names = [source.doc['nodes'][j]['name'] for j in sk['joints']]
    require(len(names) == len(set(names)) and set(names) == set(bone_order), 'Actual skin names differ from helper')
    position, uv = source.array(attrs['POSITION']), source.array(attrs['TEXCOORD_0'])
    expected_p = helper['positions'][:, [0,2,1]].copy(); expected_p[:,2] *= -1
    expected_n = helper['normals'][:, [0,2,1]].copy(); expected_n[:,2] *= -1
    expected_uv = helper['uv_gltf'].copy(); expected_uv[:,1] = np.float32(1)-np.float32(np.float32(1)-expected_uv[:,1])
    lookup = {tuple(np.r_[p,u]):i for i,(p,u) in enumerate(zip(expected_p,expected_uv))}
    require(len(lookup) == count, 'Duplicate helper POSITION+UV keys')
    mapped = np.asarray([lookup.get(tuple(np.r_[p,u]),-1) for p,u in zip(position,uv)], dtype=np.int32)
    require(len(mapped) == 12*fins and mapped.min() >= 0 and set(mapped) == set(range(count)), 'Exact exported/helper correspondence failed')
    indices = source.array(primitive['indices']).ravel()
    require(np.array_equal(np.sort(indices), np.arange(len(mapped))), 'Orphan/reused native corner index')
    require(Counter(canonical(t) for t in mapped[indices].reshape(-1,3)) == Counter(canonical(t) for t in helper['triangles']), 'Oriented triangle multiplicity changed')
    refs = defaultdict(list)
    for mesh_id,mesh in enumerate(source.doc['meshes']):
        for prim_id,p in enumerate(mesh['primitives']):
            for semantic,accessor in p['attributes'].items():refs[accessor].append(('attribute',mesh_id,prim_id,semantic))
            if 'indices' in p:refs[p['indices']].append(('index',mesh_id,prim_id))
    for skin_id,s in enumerate(source.doc['skins']):refs[s['inverseBindMatrices']].append(('inverse_bind',skin_id))
    for ai,a in enumerate(source.doc.get('animations', [])):
        for si,s in enumerate(a['samplers']):
            refs[s['input']].append(('animation_input',ai,si));refs[s['output']].append(('animation_output',ai,si))
    spans=[]
    for semantic in ('NORMAL','JOINTS_0','WEIGHTS_0'):
        ai=attrs[semantic];a=source.doc['accessors'][ai];v=source.doc['bufferViews'][a['bufferView']]
        begin,end,stride,dtype,width=source.span(ai)
        require(refs[ai] == [('attribute',mi,pi,semantic)], 'Orphan or multiply referenced selected accessor')
        require(stride == width*dtype.itemsize and not a.get('normalized') and a.get('byteOffset',0)==0
                and end-begin == v['byteLength'] and 'min' not in a and 'max' not in a, 'Selected accessor is interleaved, padded or carries stale bounds')
        for oi in range(len(source.doc['accessors'])):
            if oi == ai:continue
            ob,oe,*_=source.span(oi)
            require(end <= ob or oe <= begin, 'Selected bytes alias another accessor')
        for vi,other in enumerate(source.doc['bufferViews']):
            if vi == a['bufferView']:continue
            ob,oe=other.get('byteOffset',0),other.get('byteOffset',0)+other['byteLength']
            require(end <= ob or oe <= begin, 'Selected bytes alias another bufferView/image payload')
        spans.append({'semantic':semantic,'accessor':ai,'start':source.start+begin,'end':source.start+end})
    cursor=0; changed=0
    for span in sorted(spans,key=lambda r:r['start']):
        require(source.raw[cursor:span['start']] == result.raw[cursor:span['start']], 'Unselected payload/header/JSON bytes changed')
        left=np.frombuffer(source.raw[span['start']:span['end']],dtype=np.uint8)
        right=np.frombuffer(result.raw[span['start']:span['end']],dtype=np.uint8)
        changed+=int(np.count_nonzero(left!=right));cursor=span['end']
    require(source.raw[cursor:] == result.raw[cursor:], 'Trailing unselected bytes changed')
    require(np.array_equal(helper['weights'].reshape(fins,4,4)[:,:2], helper['weights'].reshape(fins,4,4)[:,2:])
            and np.array_equal(helper['joints'].reshape(fins,4,4)[:,:2],helper['joints'].reshape(fins,4,4)[:,2:]), 'Helper tip/root skin differs')
    remap=np.asarray([names.index(name) for name in bone_order])
    normal=result.array(attrs['NORMAL']);ids=result.array(attrs['JOINTS_0']).astype(int);weights=result.array(attrs['WEIGHTS_0'])
    require(np.array_equal(normal,expected_n[mapped]) and np.array_equal(ids,remap[helper['joints'][mapped]])
            and np.array_equal(weights,helper['weights'][mapped]), 'Restored values are not exact helper values')
    fidelity={'normal_component_error':float(abs(normal-expected_n[mapped]).max()),
              'weight_error':float(abs(weights-helper['weights'][mapped]).max()),'joint_names_exact':True}
    require(fidelity['normal_component_error'] < limit['normal_component'] and fidelity['weight_error'] <= limit['weight'], 'Original fidelity gate failed')
    write(out/'custody-and-fidelity.json', {'status':'passed','allowed_spans':spans,'changed_bytes':changed,
          'all_other_file_bytes_exact':True,'original_body_payload_exact':True,'helper_fidelity':fidelity,
          'mapped_exported_corners':len(mapped),'helper_vertices':count,'normals_requantized':False})
    np.savez_compressed(out/'exported-to-helper.npz', mapping=mapped)
    # Root anchoring uses actual unchanged body vertices, never the repaired fin as its own oracle.
    body=[];buckets=defaultdict(list)
    for bi,bn in enumerate(source.doc['nodes']):
        if 'mesh' not in bn or bi==node_id:continue
        require('skin' in bn, 'Unskinned body correspondence unsupported')
        bnames=[source.doc['nodes'][j]['name'] for j in source.doc['skins'][bn['skin']]['joints']]
        require(set(bnames)==set(names), 'Body skin joint set differs')
        for bp in source.doc['meshes'][bn['mesh']]['primitives']:
            ba=bp['attributes'];p=source.array(ba['POSITION']);u=source.array(ba['TEXCOORD_0']);j=source.array(ba['JOINTS_0']).astype(int);w=source.array(ba['WEIGHTS_0'])
            row={'node':bn,'positions':p,'uv':u,'joints':j,'weights':w,'names':bnames,
                 'material':source.doc['materials'][bp['material']]['name']};rid=len(body);body.append(row)
            for vi,point in enumerate(p):buckets[tuple(np.floor(point/limit['position_m']).astype(np.int64))].append((rid,vi))
    root_map=[]; root_weight_error=0.; offsets=list(itertools.product((-1,0,1),repeat=3)); root_positions=expected_p.reshape(fins,4,3)[:,:2].reshape(-1,3)
    root_uv=helper['root_source_uv'].astype(np.float32).reshape(-1,2)
    for ri,(point,uv0) in enumerate(zip(root_positions,root_uv)):
        if ri%128==0:require(True,'deadline')
        key=np.floor(point/limit['position_m']).astype(np.int64); candidates=[]
        for offset in offsets:
            for record,vertex in buckets.get(tuple(key+offset),()):
                b=body[record]
                if np.max(abs(b['positions'][vertex]-point))<=limit['position_m'] and np.max(abs(b['uv'][vertex]-uv0))<=limit['uv']:
                    candidates.append((record,vertex))
        if not 0<len(candidates)<=256:
            write(out/'root-correspondence-failure.json',{'helper_root':ri,'position':point.tolist(),'uv':uv0.tolist(),
                  'candidate_count':len(candidates),'reason':'Missing or oversized body-root correspondence'})
        require(0<len(candidates)<=256,'Missing/oversized body-root correspondence')
        signatures=[]
        for record,vertex in candidates:
            b=body[record]; weighted=tuple(sorted((b['names'][int(j)],float(w)) for j,w in zip(b['joints'][vertex],b['weights'][vertex]) if w>0))
            signatures.append((tuple(b['positions'][vertex]),weighted))
        if len(set(signatures))!=1:
            write(out/'root-correspondence-failure.json',{'helper_root':ri,'position':point.tolist(),'uv':uv0.tolist(),
                  'candidate_count':len(candidates),'signatures':signatures[:8],
                  'reason':'Different source-root positions/skin; no nearest-match selection'})
        require(len(set(signatures))==1,'Body-root candidates have different positions or skin; no nearest-match guess')
        root_map.append(candidates[0])
        helper_index=(ri//2)*4+ri%2
        expected_weight={name:0. for name in names}
        for slot,value in zip(helper['joints'][helper_index],helper['weights'][helper_index]):
            expected_weight[bone_order[int(slot)]]+=float(value)
        actual_weight={name:0. for name in names}
        record,vertex=candidates[0]; b=body[record]
        for slot,value in zip(b['joints'][vertex],b['weights'][vertex]):actual_weight[b['names'][int(slot)]]+=float(value)
        root_weight_error=max(root_weight_error,max(abs(expected_weight[name]-actual_weight[name]) for name in names))
    write(out/'body-root-correspondence.json',{'matched_roots':len(root_map),'maximum_helper_vs_body_weight_error':root_weight_error,
          'original_weight_gate':limit['weight'],'weight_gate_passed':root_weight_error<=limit['weight'],
          'ambiguity_policy':'Every surviving position/UV candidate must have identical position and positive named weights.'})
    np.savez_compressed(out/'body-root-correspondence.npz',body_primitive_and_vertex=np.asarray(root_map,dtype=np.int32))
    animation=tracks(result);require(set(animation)==set(request['expected_clips']),'Actual clip set changed')
    original_motion_passed=original_motion(request,result,body,names,animation,limit,out)
    inverse_helper_ids=remap[helper['joints']];motion=[]
    for name,rows in animation.items():
        times=np.linspace(min(t[0] for _,t,_,_ in rows),max(t[-1] for _,t,_,_ in rows),9)
        helper_error=0.;root_error=0.;raw_fringe_delta=0.
        for seconds in times:
            matrices=world(result,rows,float(seconds))
            moved=skin(result,node,position,ids,weights,matrices)
            wanted=skin(result,node,expected_p,inverse_helper_ids,helper['weights'],matrices)[mapped]
            helper_error=max(helper_error,float(np.linalg.norm(moved-wanted,axis=1).max()))
            raw=skin(source,node,position,source.array(attrs['JOINTS_0']).astype(int),source.array(attrs['WEIGHTS_0']),matrices)
            raw_fringe_delta=max(raw_fringe_delta,float(np.linalg.norm(moved-raw,axis=1).max()))
            helper_roots=skin(result,node,expected_p,inverse_helper_ids,helper['weights'],matrices).reshape(fins,4,3)[:,:2].reshape(-1,3)
            body_poses=[skin(source,b['node'],b['positions'],b['joints'],b['weights'],matrices) for b in body]
            actual_roots=np.asarray([body_poses[record][vertex] for record,vertex in root_map])
            root_error=max(root_error,float(np.linalg.norm(helper_roots-actual_roots,axis=1).max()))
        motion.append({'clip':name,'sample_times_seconds':times.tolist(),'helper_pose_error_m':helper_error,
                       'actual_body_root_gap_m':root_error,'changed_fringe_vs_raw_native_export_max_m':raw_fringe_delta,
                       'helper_pose_passed':helper_error<=limit['weighted_pose_m'],'root_passed':root_error<=limit['root_m']})
    write(out/'motion.json',{'clips':motion,'status':'passed' if all(m['helper_pose_passed'] and m['root_passed'] for m in motion) else 'failed',
          'native_pose_comparison':'unavailable: no actual native pose array supplied; raw exported skin is a different evidence level',
          'unchanged_body_byte_custody':True,'source_body_native_roundtrip':'not repaired or silently accepted',
          'continuous_time_or_physics_verified':False})
    require(original_motion_passed and root_weight_error<=limit['weight'] and all(m['helper_pose_passed'] and m['root_passed'] for m in motion),
            'Original body-root weight/weighted-pose/attachment gate failed')
    for row in ('source','derivative','helper','native_report','thresholds'):pinned(request[row])
    return {'status':'passed_selected_fringe_fidelity_and_sampled_attachment','quality_approved':False,
            'whole_model_fidelity_approved':False,'native_pose_comparison_verified':False,'elapsed_seconds':time.monotonic()-START,
            'body_limitations':request['inherited_body_limitations']}


if __name__=='__main__':
    request_path=Path(sys.argv[1]);out=Path(sys.argv[2])
    out=out if out.is_absolute() else ROOT/out
    require(out.absolute()==out.resolve() and out.is_relative_to(ROOT/'outputs'),'New output must be under this workspace outputs')
    out.mkdir()
    request=json.loads(request_path.read_text())
    write(out/'request.json',request)
    try:
        result=inspect(request,out)
    except Exception as error:
        import traceback
        write(out/'result.json',{'status':'failed','error':str(error),'quality_approved':False,'elapsed_seconds':time.monotonic()-START})
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    else:
        write(out/'result.json',result)
