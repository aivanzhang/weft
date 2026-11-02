"""Sampled new-coat skin proof; source-body proof is consumed, not rerun."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.three_d import _GLB, _animation_tracks, _pose, _world_matrices

START = time.monotonic()
RESULT = {'schema_version': 1, 'status': 'running', 'failures': [],
          'quality_approved': False, 'native_provider_AI_calls': 0}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with (HERE/name).open('x') as stream:
        stream.write(text)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check(ok, message):
    if not ok:
        RESULT['failures'].append(message)


def array(glb, index, kind, types, indices=False):
    return np.asarray(glb.accessor(index, kind, types, count_limit=300000 if indices else 100000))


def skin_context(glb, node_id, names):
    skin = glb.doc['skins'][glb.doc['nodes'][node_id]['skin']]
    joint_names = [glb.doc['nodes'][i]['name'] for i in skin['joints']]
    require(len(joint_names) == len(set(joint_names)) == len(names) and set(joint_names) == set(names),
            'Named bone set changed')
    inverse = array(glb, skin['inverseBindMatrices'], 'MAT4', (5126,)).reshape(-1, 4, 4).transpose(0, 2, 1)
    order = [joint_names.index(name) for name in names]
    return {'joints': [skin['joints'][i] for i in order], 'inverse': inverse[order],
            'joint_names': joint_names}


def vertex_data(glb, primitive, ctx, names):
    a = primitive['attributes']
    points = array(glb, a['POSITION'], 'VEC3', (5126,))
    indices = array(glb, primitive['indices'], 'SCALAR', (5121, 5123, 5125), True).reshape(-1, 3).astype(np.int32)
    joints = array(glb, a['JOINTS_0'], 'VEC4', (5121, 5123)).astype(np.int32)
    weights = array(glb, a['WEIGHTS_0'], 'VEC4', (5121, 5123, 5126))
    dense = np.zeros((len(points), len(names)))
    for i, name in enumerate(ctx['joint_names']):
        dense[:, names.index(name)] = np.where(joints == i, weights, 0).sum(axis=1)
    return {'points': points, 'faces': indices, 'weights': dense,
            'uv': array(glb, a['TEXCOORD_0'], 'VEC2', (5126,))}


def matrices(world, ctx):
    return np.array([np.asarray(world[j]).reshape(4, 4).T for j in ctx['joints']]) @ ctx['inverse']


def weighted(points, weights, transforms):
    p = np.column_stack((points, np.ones(len(points))))
    return np.einsum('nb,bij,nj->ni', weights, transforms[:, :3, :], p)


def main():
    plan = json.loads((HERE/'prepared.json').read_text())
    for row in plan['inputs']:
        require(sha(ROOT/row['path']) == row['sha256'], 'Pinned input changed '+row['path'])
    prior = ROOT/plan['prior_audit']
    require(json.loads((prior/'result.json').read_text())['status'] == 'passed'
            and json.loads((prior/'native-source-identity.json').read_text())['equal'] is True,
            'Prior exact source/native ID proof missing')
    source, final = _GLB(ROOT/plan['source']), _GLB(ROOT/plan['final'])
    report = json.loads((ROOT/plan['pack']/'finish-report.json').read_text())
    sn, fn = source.doc['nodes'], final.doc['nodes']
    source_nodes = [i for i, n in enumerate(sn) if 'mesh' in n]
    require(len(source_nodes) == 1, 'Source must have one body mesh')
    source_node = source_nodes[0]
    names = [sn[j]['name'] for j in source.doc['skins'][sn[source_node]['skin']]['joints']]
    source_ctx = skin_context(source, source_node, names)
    source_rows, vertex_offset, face_offset = {}, 0, 0
    selected = {x['source_material'] for x in report['portable_fur']}
    for p in source.doc['meshes'][sn[source_node]['mesh']]['primitives']:
        name = source.doc['materials'][p['material']]['name']
        nv = source.doc['accessors'][p['attributes']['POSITION']]['count']
        nf = source.doc['accessors'][p['indices']]['count']//3
        if name in selected:
            source_rows[name] = {**vertex_data(source, p, source_ctx, names),
                                 'vertex_offset': vertex_offset, 'face_offset': face_offset}
        vertex_offset += nv
        face_offset += nf
    records, metadata = [], []
    max_weight_error = max_root_error = 0.
    for part in report['portable_fur']:
        name = part['source_material']; original = source_rows[name]
        material_name = part['material']['material']
        matches = [(i, p) for i, n in enumerate(fn) if 'mesh' in n and n.get('name', '').startswith('PortableCoat_')
                   for p in final.doc['meshes'][n['mesh']]['primitives']
                   if final.doc['materials'][p['material']]['name'] == material_name]
        require(len(matches) == 1, 'Portable material primitive identity ambiguous')
        node_id, primitive = matches[0]
        ctx = skin_context(final, node_id, names)
        data = vertex_data(final, primitive, ctx, names)
        require(len(data['faces']) == part['cards']*4, 'Portable card topology count changed')
        for sample in part['root_samples']:
            local_face = sample['source_polygon']-original['face_offset']
            require(0 <= local_face < len(original['faces']), 'Source face outside named material')
            ids = original['faces'][local_face]
            require((ids+original['vertex_offset']).tolist() == sample['source_vertices'], 'Exact source vertex IDs changed')
            bary = np.asarray(sample['barycentric'])
            source_p, source_w = original['points'][ids], original['weights'][ids]
            combined = bary @ source_w
            order = sorted(range(len(names)), key=lambda i: (-combined[i], names[i]))[:4]
            expected_w = np.zeros(len(names)); denominator = float(combined[order].sum())
            require(denominator > 0, 'Source root has no positive skin support')
            expected_w[order] = combined[order]/denominator
            card_id = sample['root_index']
            card_ids = np.unique(data['faces'][card_id*4:card_id*4+4])
            weight_error = float(abs(data['weights'][card_ids]-expected_w).max())
            max_weight_error = max(max_weight_error, weight_error)
            root_ids = card_ids[data['uv'][card_ids, 1] == data['uv'][card_ids, 1].max()]
            root_points, groups = np.unique(data['points'][root_ids], axis=0, return_inverse=True)
            require(len(root_points) == 2, 'Bottom UV row does not prove two root endpoints')
            endpoint_ids = [root_ids[groups == i] for i in range(2)]
            normal = np.asarray(sample['normal'])[[0, 2, 1]]*np.array([1, 1, -1])
            surface_p = bary @ source_p
            center = surface_p-normal*sample['root_inset_m']
            root_error = float(abs(root_points.mean(axis=0)-center).max())
            max_root_error = max(max_root_error, root_error)
            records.append({'ctx':ctx, 'data':data, 'endpoints':endpoint_ids, 'source_p':source_p,
                            'source_w':source_w, 'bary':bary, 'expected_w':expected_w,
                            'center':center, 'normal_inset':normal*sample['root_inset_m']})
            metadata.append({'material':name, 'root_index':card_id, 'source_polygon':sample['source_polygon'],
                'source_vertices':sample['source_vertices'], 'barycentric':bary.tolist(),
                'expected_named_weights':{names[i]:float(v) for i,v in enumerate(expected_w) if v>0},
                'discarded_pre_normalization_weight':float(combined.sum()-denominator),
                'max_exported_named_weight_error':weight_error, 'rest_root_center_error_m':root_error,
                'exported_card_vertex_ids':card_ids.tolist(), 'exported_root_endpoint_ids':[x.tolist() for x in endpoint_ids]})
    require(len(records) == 384, 'Expected384 sealed sampled roots')
    check(max_weight_error <= plan['gates']['named_weights'], 'Sampled coat named weights differ from normalized top4 source weights')
    check(max_root_error <= plan['gates']['rest_root_m'], 'Sampled coat rest roots differ from their exact source anchors')
    save('roots.json', {'samples':metadata,'max_weight_error':max_weight_error,'max_rest_root_error_m':max_root_error})
    sp = {c:i for i,n in enumerate(sn) for c in n.get('children', [])}
    fp = {c:i for i,n in enumerate(fn) for c in n.get('children', [])}
    st = {name:t for name,t,_ in _animation_tracks(source,sn,set(source_ctx['joints']))}
    ft = {name:t for name,t,_ in _animation_tracks(final,fn,set(final.doc['skins'][0]['joints']))}
    require(set(st) == set(ft) == {'idle','making_something','milestone_level_up','working'}, 'Source clips changed')
    clip_results, all_actual, all_expected, all_surface = [], [], [], []
    total_max_motion = total_max_attachment = total_max_duplicates = 0.
    for clip in st:
        clip_rows, clip_actual, clip_expected, clip_surface = [], [], [], []
        for second in plan['sample_seconds']:
            sw = _world_matrices(sn,sp,_pose(st[clip],second))
            fw = _world_matrices(fn,fp,_pose(ft[clip],second))
            source_matrix = matrices(sw,source_ctx)
            final_matrices = {}
            actual, expected, surface, duplicate_errors = [], [], [], []
            for record in records:
                key = tuple(record['ctx']['joints'])
                if key not in final_matrices:
                    final_matrices[key] = matrices(fw,record['ctx'])
                endpoints = []
                for ids in record['endpoints']:
                    posed = weighted(record['data']['points'][ids],record['data']['weights'][ids],final_matrices[key])
                    duplicate_errors.append(float(abs(posed-posed[0]).max()))
                    endpoints.append(posed.mean(axis=0))
                actual.append(np.mean(endpoints,axis=0))
                expected.append(weighted(record['center'][None,:],record['expected_w'][None,:],source_matrix)[0])
                body_surface = record['bary'] @ weighted(record['source_p'],record['source_w'],source_matrix)
                inset = np.einsum('b,bij,j->i',record['expected_w'],source_matrix[:,:3,:3],record['normal_inset'])
                surface.append(body_surface-inset)
            actual,expected,surface = map(np.asarray,(actual,expected,surface))
            motion_errors = abs(actual-expected).max(axis=1)
            attachment_errors = abs(actual-surface).max(axis=1)
            duplicate_max = max(duplicate_errors)
            total_max_motion = max(total_max_motion,float(motion_errors.max()))
            total_max_attachment = max(total_max_attachment,float(attachment_errors.max()))
            total_max_duplicates = max(total_max_duplicates,duplicate_max)
            worst = np.argsort(attachment_errors)[-5:][::-1]
            clip_rows.append({'second':second,'declared_skin_max_component_error_m':float(motion_errors.max()),
                'source_surface_attachment_max_component_error_m':float(attachment_errors.max()),
                'source_surface_attachment_over_gate':int((attachment_errors>plan['gates']['source_surface_attachment_m']).sum()),
                'root_duplicate_pose_max_error_m':duplicate_max,
                'attachment_witnesses':[{'sample':int(i),'material':metadata[i]['material'],'root_index':metadata[i]['root_index'],
                    'source_polygon':metadata[i]['source_polygon'],'error_m':float(attachment_errors[i])} for i in worst]})
            clip_actual.append(actual);clip_expected.append(expected);clip_surface.append(surface)
        clip_results.append({'clip':clip,'samples':clip_rows})
        all_actual.append(clip_actual);all_expected.append(clip_expected);all_surface.append(clip_surface)
    check(total_max_motion <= plan['gates']['sampled_motion_m'], 'Sampled exported root motion differs from declared inherited top4 skin')
    check(total_max_duplicates <= plan['gates']['rest_root_m'], 'Duplicated root vertices separate during animation')
    check(total_max_attachment <= plan['gates']['source_surface_attachment_m'], 'Sampled coat roots drift from the deformed source triangle beyond the declared inset')
    np.savez_compressed(HERE/'sampled-motion.npz', actual_roots=np.asarray(all_actual), declared_skin_roots=np.asarray(all_expected),
                        source_surface_minus_inset=np.asarray(all_surface), seconds=np.asarray(plan['sample_seconds']))
    save('motion.json',{'clips':clip_results,'max_declared_skin_error_m':total_max_motion,
        'max_source_surface_attachment_error_m':total_max_attachment,'max_duplicate_pose_error_m':total_max_duplicates,
        'attachment_definition':'Actual root center versus barycentric deformed source triangle minus the declared inset transformed by the independently derived top4 skin; no other offset is compensated.'})
    RESULT.update(status='passed' if not RESULT['failures'] else 'failed',samples=len(records),clips=len(st),
        times_per_clip=len(plan['sample_seconds']),max_named_weight_error=max_weight_error,max_rest_root_error_m=max_root_error,
        max_declared_skin_motion_error_m=total_max_motion,max_source_surface_attachment_error_m=total_max_attachment,
        source_sha256=source.hash,final_sha256=final.hash)
    if RESULT['failures']:
        raise ValueError('Coat skin addon recorded failures')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        RESULT.update(status='failed',exception={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
    finally:
        RESULT.update(elapsed_seconds=time.monotonic()-START,
            process_high_water_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save('result.json',RESULT)
