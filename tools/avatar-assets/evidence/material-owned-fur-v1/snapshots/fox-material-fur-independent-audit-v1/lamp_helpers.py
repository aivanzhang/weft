"""Independent source/final lamp semantics and all-vertex sampled-skin audit."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.three_d import _GLB, _animation_tracks, _pose, _world_matrices

START = time.monotonic()
RESULT = {'schema_version': 1, 'status': 'running', 'failures': [],
          'recorded_local': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
          'new_native_provider_or_ai_calls': 0, 'quality_approved': False,
          'source_kind': 'prior_agent_authored_procedural_asset_not_neural_or_fresh_Studio_acceptance'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with (HERE / name).open('x') as stream:
        stream.write(text)


def check(ok, message):
    if not ok:
        RESULT['failures'].append(message)


def parents(doc):
    return {child: i for i, node in enumerate(doc['nodes']) for child in node.get('children', [])}


def arrays(glb, index, kind, types, *, indices=False):
    return np.asarray(glb.accessor(index, kind, types, count_limit=300000 if indices else 100000))


def image_records(glb):
    result = []
    for image in glb.doc.get('images', []):
        view = glb.doc['bufferViews'][image['bufferView']]
        start = view.get('byteOffset', 0)
        payload = glb.binary[start:start + view['byteLength']]
        result.append({'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload), 'mimeType': image['mimeType']})
    return result


def material_semantics(glb, images):
    doc, output = glb.doc, {}
    def texture(info):
        if info is None:
            return None
        item = doc['textures'][info['index']]
        sampler = doc.get('samplers', [])[item['sampler']] if 'sampler' in item else {}
        return {'image': images[item['source']], 'texCoord': info.get('texCoord', 0),
                'scale': info.get('scale', 1), 'strength': info.get('strength', 1),
                'sampler': {'wrapS': sampler.get('wrapS', 10497), 'wrapT': sampler.get('wrapT', 10497),
                            'minFilter': sampler.get('minFilter'), 'magFilter': sampler.get('magFilter')}}
    for item in doc['materials']:
        name = item['name']
        if name in output:
            raise ValueError('Duplicate named material')
        pbr = item.get('pbrMetallicRoughness', {})
        output[name] = {'baseColorFactor': pbr.get('baseColorFactor', [1, 1, 1, 1]),
                        'metallicFactor': pbr.get('metallicFactor', 1), 'roughnessFactor': pbr.get('roughnessFactor', 1),
                        'baseColorTexture': texture(pbr.get('baseColorTexture')),
                        'metallicRoughnessTexture': texture(pbr.get('metallicRoughnessTexture')),
                        'normalTexture': texture(item.get('normalTexture')),
                        'occlusionTexture': texture(item.get('occlusionTexture')),
                        'emissiveTexture': texture(item.get('emissiveTexture')),
                        'emissiveFactor': item.get('emissiveFactor', [0, 0, 0]),
                        'alphaMode': item.get('alphaMode', 'OPAQUE'), 'alphaCutoff': item.get('alphaCutoff', .5),
                        'doubleSided': item.get('doubleSided', False)}
    return output


def records(glb, names):
    result = {}
    for node in glb.doc['nodes']:
        if 'mesh' not in node:
            continue
        skin = glb.doc['skins'][node['skin']]
        skin_names = [glb.doc['nodes'][j]['name'] for j in skin['joints']]
        if set(skin_names) != set(names):
            raise ValueError('Changed named skeleton')
        for p in glb.doc['meshes'][node['mesh']]['primitives']:
            material = glb.doc['materials'][p['material']]['name']
            if material in result:
                raise ValueError('Ambiguous repeated material primitive')
            a = p['attributes']
            points = arrays(glb, a['POSITION'], 'VEC3', (5126,))
            joints = arrays(glb, a['JOINTS_0'], 'VEC4', (5121, 5123)).astype(np.int32)
            weights = arrays(glb, a['WEIGHTS_0'], 'VEC4', (5121, 5123, 5126))
            dense = np.zeros((len(points), len(names)))
            for j, name in enumerate(skin_names):
                dense[:, names.index(name)] = np.where(joints == j, weights, 0).sum(axis=1)
            result[material] = {'positions': points, 'uv': arrays(glb, a['TEXCOORD_0'], 'VEC2', (5126,)),
                'normals': arrays(glb, a['NORMAL'], 'VEC3', (5126,)),
                'tangents': arrays(glb, a['TANGENT'], 'VEC4', (5126,)) if 'TANGENT' in a else None,
                'weights': weights, 'joints': joints, 'dense_weights': dense, 'skin': skin['joints'],
                'inverse': arrays(glb, skin['inverseBindMatrices'], 'MAT4', (5126,)).reshape(-1, 4, 4).transpose(0, 2, 1),
                'faces': arrays(glb, p['indices'], 'SCALAR', (5121, 5123, 5125), indices=True).reshape(-1, 3).astype(np.int32)}
    return result


OFFSETS = list(itertools.product((-1, 0, 1), repeat=3))


def correspond(src, dst):
    buckets = defaultdict(list)
    for i, cell in enumerate(np.floor(src['positions'] / 1e-5).astype(np.int64)):
        buckets[tuple(cell)].append(i)
    canonical = {}
    source_ids = []
    for i, values in enumerate(zip(src['positions'], src['uv'], src['dense_weights'])):
        key = tuple(float(x) for row in values for x in row)
        source_ids.append(canonical.setdefault(key, i))
    source_ids = np.asarray(source_ids, np.int32)
    mapped = np.empty(len(dst['positions']), np.int32)
    metrics = {'position_max_m': 0., 'uv_max': 0., 'weight_max': 0., 'normal_min_candidate_max': 0.,
               'normal_identity_ambiguous_vertices': 0, 'tangent_min_candidate_max': 0.,
               'source_vertices': len(src['positions']), 'final_vertices': len(dst['positions'])}
    limits = {'positions': 1e-6, 'uv': 1e-6, 'dense_weights': 1e-6}
    for i, point in enumerate(dst['positions']):
        if i % 512 == 0 and time.monotonic() - START > 27:
            raise TimeoutError('Audit internal deadline')
        base = np.floor(point / 1e-5).astype(np.int64)
        candidate = np.asarray([j for off in OFFSETS for j in buckets.get(tuple(base + off), ())], np.int32)
        if not 0 < len(candidate) <= 256:
            raise ValueError('Missing or oversized correspondence bucket')
        for field, limit in limits.items():
            candidate = candidate[np.max(abs(src[field][candidate] - dst[field][i]), axis=1) <= limit]
        if not len(candidate):
            raise ValueError('No position/UV/named-skin correspondence')
        # Prefer exact stored source identity before admitting tiny import/export rounding.
        exact = candidate.copy()
        for field in limits:
            exact = exact[np.max(abs(src[field][exact] - dst[field][i]), axis=1) == 0]
        if len(exact):
            candidate = exact
        ids = np.unique(source_ids[candidate])
        if len(ids) != 1:
            raise ValueError('Ambiguous non-equivalent source point/UV/weight identity')
        mapped[i] = ids[0]
        for field, key in [('positions', 'position_max_m'), ('uv', 'uv_max'), ('dense_weights', 'weight_max')]:
            metrics[key] = max(metrics[key], float(np.max(abs(src[field][mapped[i]] - dst[field][i]))))
        nerr = np.max(abs(src['normals'][candidate] - dst['normals'][i]), axis=1)
        metrics['normal_min_candidate_max'] = max(metrics['normal_min_candidate_max'], float(nerr.min()))
        metrics['normal_identity_ambiguous_vertices'] += int(len(np.unique(src['normals'][candidate], axis=0)) > 1)
        if src['tangents'] is not None and dst['tangents'] is not None:
            terr = np.max(abs(src['tangents'][candidate] - dst['tangents'][i]), axis=1)
            metrics['tangent_min_candidate_max'] = max(metrics['tangent_min_candidate_max'], float(terr.min()))
    def oriented(record, ids):
        return Counter(min(tuple(row), tuple(np.roll(row, 1)), tuple(np.roll(row, 2))) for row in ids[record['faces']])
    if set(source_ids) != set(mapped) or oriented(src, source_ids) != oriented(dst, mapped):
        raise ValueError('Source point groups or oriented material triangle coverage changed')
    metrics['oriented_position_uv_weight_triangle_multiset_equal'] = True
    metrics['normals_tangents_scope'] = 'Measured lower bounds where coincident source normals differ; not full shading-frame byte identity'
    return mapped, metrics


def skin(record, world):
    matrices = np.array([np.asarray(world[j]).reshape(4, 4).T for j in record['skin']]) @ record['inverse']
    points = np.column_stack((record['positions'], np.ones(len(record['positions']))))
    result = np.zeros((len(points), 3))
    for column in range(4):
        result += np.einsum('nij,nj->ni', matrices[record['joints'][:, column], :3], points) * record['weights'][:, column, None]
    return result


def main():
    plan = json.loads((HERE / 'prepared.json').read_text())
    for row in plan['inputs']:
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError('Pinned input changed: ' + row['path'])
    source, final = _GLB(ROOT / plan['source']), _GLB(ROOT / plan['final'])
    images = {'source': image_records(source), 'final': image_records(final)}
    materials = {'source': material_semantics(source, images['source']), 'final': material_semantics(final, images['final'])}
    check(Counter(x['sha256'] for x in images['source']) == Counter(x['sha256'] for x in images['final']), 'Embedded image bytes differ')
    check(materials['source'] == materials['final'], 'Named PBR/basecolor/UV/alpha/culling material semantics differ')
    save('materials.json', {'images': images, 'materials': materials, 'semantic_equal': materials['source'] == materials['final']})
    ns, nf = source.doc['nodes'], final.doc['nodes']
    by_name_s, by_name_f = {n['name']: i for i, n in enumerate(ns)}, {n['name']: i for i, n in enumerate(nf)}
    check(len(by_name_s) == len(ns) and len(by_name_f) == len(nf) and set(by_name_s) == set(by_name_f), 'Named nodes changed or duplicated')
    names = [ns[j]['name'] for j in source.doc['skins'][0]['joints']]
    check(len(source.doc['skins']) == len(final.doc['skins']) == 1, 'Skin count changed')
    ps, pf = parents(source.doc), parents(final.doc)
    ws, wf = _world_matrices(ns, ps), _world_matrices(nf, pf)
    rest_errors = {name: float(np.max(abs(np.asarray(ws[by_name_s[name]]) - wf[by_name_f[name]]))) for name in by_name_s}
    bone_parents_s = {name: ns[ps[by_name_s[name]]]['name'] if by_name_s[name] in ps else None for name in names}
    bone_parents_f = {name: nf[pf[by_name_f[name]]]['name'] if by_name_f[name] in pf else None for name in names}
    check(bone_parents_s == bone_parents_f, 'Bone parent names changed')
    check(max(rest_errors.values()) <= 1e-6, 'Named rest world transform changed beyond 1e-6')
    save('rig.json', {'node_ids': {name: [by_name_s[name], by_name_f[name]] for name in by_name_s},
                     'bone_parents_source': bone_parents_s, 'bone_parents_final': bone_parents_f,
                     'rest_world_matrix_max_errors': rest_errors, 'units_axes': 'Declared meters, glTF +Y up/+Z forward; transform comparison, not recovered physical measurement',
                     'mesh_parent_change': 'Original mesh parent AvatarRig becomes scene-root; compare world transforms rather than node IDs.'})
    sr, fr = records(source, names), records(final, names)
    check(set(sr) == set(fr), 'Material primitive ownership changed')
    mappings, geometry = {}, {}
    for name in sr:
        try:
            mappings[name], geometry[name] = correspond(sr[name], fr[name])
        except Exception as error:
            save('geometry-partial.json', {'completed': geometry, 'failed_material': name, 'reason': str(error)})
            raise
    save('geometry.json', geometry)
    np.savez_compressed(HERE / 'correspondence.npz', **mappings)
    st = {name: tracks for name, tracks, _ in _animation_tracks(source, ns, set(source.doc['skins'][0]['joints']))}
    ft = {name: tracks for name, tracks, _ in _animation_tracks(final, nf, set(final.doc['skins'][0]['joints']))}
    check(set(st) == set(ft) == {'idle', 'making_something', 'milestone_level_up', 'working'}, 'Animation names changed')
    motion = []
    for name in st:
        bounds_s = [min(t[3][0] for t in st[name]), max(t[3][-1] for t in st[name])]
        bounds_f = [min(t[3][0] for t in ft[name]), max(t[3][-1] for t in ft[name])]
        check(bounds_s == bounds_f == [0, 4], 'Clip duration changed: ' + name)
        track_ids_s = {(ns[t[0]]['name'], t[1]) for t in st[name]}
        track_ids_f = {(nf[t[0]]['name'], t[1]) for t in ft[name]}
        check(track_ids_s == track_ids_f, 'Animation targets changed: ' + name)
        row = {'id': name, 'source_seconds': bounds_s, 'final_seconds': bounds_f, 'samples': []}
        initial, maximum_motion = {}, 0.
        for second in np.linspace(0, 4, 9):
            sw = _world_matrices(ns, ps, _pose(st[name], float(second)))
            fw = _world_matrices(nf, pf, _pose(ft[name], float(second)))
            error = 0.
            for material in sr:
                before = skin(sr[material], sw)[mappings[material]]
                after = skin(fr[material], fw)
                error = max(error, float(np.max(np.abs(before - after))))
                if second == 0:
                    initial[material] = after
                maximum_motion = max(maximum_motion, float(np.linalg.norm(after - initial[material], axis=1).max()))
            row['samples'].append({'seconds': float(second), 'max_weighted_position_component_error_m': error})
            check(error <= 1e-5, 'Weighted source pose changed beyond 1e-5 m: ' + name + '/' + str(second))
        row['maximum_observed_weighted_displacement_from_first_sample_m'] = maximum_motion
        check(maximum_motion > 1e-5, 'No actual weighted motion: ' + name)
        motion.append(row)
    save('motion.json', {'states': motion, 'all_exported_vertices_per_sample': sum(len(r['positions']) for r in fr.values()),
                         'pose_component_tolerance_m': 1e-5, 'sampling': 'Nine fixed times per clip; not all-frame or loop/physical/semantic quality acceptance'})
    pack = ROOT / 'outputs/multi-material-lamp-finish-v1'
    manifest = json.loads((pack / 'manifest.json').read_text())
    finish = json.loads((pack / 'finish-report.json').read_text())
    guard = json.loads((pack / 'normal-uv-export.json').read_text())
    sampling = finish['source_color_sampling']
    check(sampling['status'] == 'skipped' and sampling['selection_sampling_calls'] == 0 and sampling['sampled_vertices'] == 0 and sampling['required_by'] == [], 'Color selection was not skipped')
    check(guard['status'] == 'supported' and guard['errors'] == [] and guard['texture_pixels_read'] is False and finish['normal_uv_export'] == guard, 'Dedicated normal-UV guard did not pass unchanged')
    check(finish['groom'] is None and not finish.get('portable_fur') and not finish.get('portable_fringe') and not finish.get('fur_undercoat'), 'Unrequested fur added')
    pngs = []
    for item in manifest['artifacts']:
        if item['path'] in ('normal-uv-export.json', 'finish-report.json') or item['path'].endswith('.png'):
            path = pack / item['path']
            check(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Artifact custody mismatch: ' + item['path'])
        if item['path'].endswith('.png'):
            with Image.open(path) as image:
                row = {'path': item['path'], 'sha256': item['sha256'], 'format': image.format, 'dimensions': list(image.size), 'mode': image.mode, 'frames': getattr(image, 'n_frames', 1)}
                check(image.format == 'PNG' and image.size == (256, 256) and image.mode == 'RGB' and row['frames'] == 1, 'Unexpected draft image encoding')
                image.verify()
            pngs.append(row)
    check(len(pngs) == 21, 'Expected one portrait plus twenty actual pose PNGs')
    save('artifacts.json', {'source_color_sampling': sampling, 'normal_guard': guard, 'draft_pngs': pngs,
                            'video_created': False, 'skeletal_clips': 4, 'final_media_acceptance': False})
    RESULT.update(status='passed_requested_integration_checks' if not RESULT['failures'] else 'failed',
                  geometry_normal_tangent_byte_equality_claimed=False,
                  scope='Material/image identity and sampled all-vertex pose equivalence; normal/tangent diagnostic limits recorded separately',
                  source_sha256=source.hash, final_sha256=final.hash,
                  source_vertices=sum(len(r['positions']) for r in sr.values()), final_vertices=sum(len(r['positions']) for r in fr.values()),
                  source_triangles=sum(len(r['faces']) for r in sr.values()), final_triangles=sum(len(r['faces']) for r in fr.values()),
                  embedded_image_records_identical_ordered=images['source'] == images['final'],
                  embedded_image_hash_multiset_equal=Counter(x['sha256'] for x in images['source']) == Counter(x['sha256'] for x in images['final']),
                  materials_semantic_equal=materials['source'] == materials['final'])
    for row in plan['inputs']:
        check(sha(ROOT / row['path']) == row['sha256'], 'Input changed during audit: ' + row['path'])
    if RESULT['failures']:
        RESULT['status'] = 'failed'
        raise ValueError('Recorded independent audit failures')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        RESULT['status'] = 'failed'
        RESULT['exception'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        RESULT['elapsed_seconds'] = time.monotonic() - START
        RESULT['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save('result.json', RESULT)
