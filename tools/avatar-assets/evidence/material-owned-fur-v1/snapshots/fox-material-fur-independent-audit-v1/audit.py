"""One bounded independent fox body, action and sampled tuft-root audit."""

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import lamp_helpers as h
from avatar_assets.three_d import _GLB, _animation_tracks, _pose, _world_matrices

START = time.monotonic()
RESULT = {'schema_version': 1, 'status': 'running', 'failures': [],
          'native_provider_role_calls': 0, 'appearance_accepted': False}


def save(name, value):
    with (HERE/name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def check(value, message):
    if not value:
        RESULT['failures'].append(message)


def require(value, message):
    if not value:
        raise ValueError(message)


def ahash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def body_view(glb):
    nodes = [dict(node) for node in glb.doc['nodes']]
    excluded = []
    for index, node in enumerate(nodes):
        if 'mesh' in node and node.get('name', '').startswith('PortableCoat_'):
            excluded.append({'node': index, 'name': node['name'], 'mesh': node.pop('mesh')})
    view = SimpleNamespace(doc={**glb.doc, 'nodes': nodes}, accessor=glb.accessor)
    return view, excluded


def native_identity(source, sr, groom):
    points, faces, uvs, slots = [], [], [], []
    offset = 0
    for slot, primitive in enumerate(source.doc['meshes'][0]['primitives']):
        name = source.doc['materials'][primitive['material']]['name']
        row = sr[name]
        positions = row['positions'].astype('<f4')[:, [0, 2, 1]].copy()
        positions[:, 1] *= -1
        uv = row['uv'].astype('<f4').copy()
        uv[:, 1] = np.float32(1)-uv[:, 1]
        points.append(positions)
        faces.append(row['faces']+offset)
        uvs.append(uv[row['faces']].reshape(-1, 2))
        slots.append(np.full(len(row['faces']), slot, dtype='<i4'))
        offset += len(positions)
    points = np.concatenate(points).astype('<f4')
    faces = np.concatenate(faces).astype('<i4')
    uvs = np.concatenate(uvs).astype('<f4')
    slots = np.concatenate(slots).astype('<i4')
    loops = np.arange(len(faces)*3, dtype='<i4').reshape(-1, 3)
    actual = {'positions': ahash(points), 'faces': ahash(faces), 'loops': ahash(loops),
              'material_slots': ahash(slots), 'uvs': ahash(uvs)}
    before = groom['body_before']
    expected = {key: before[key] for key in ('positions', 'faces', 'loops', 'material_slots')}
    require(len(before['uvs']) == 1, 'Source identity requires its single native UV layer')
    expected['uvs'] = before['uvs'][0]['sha256']
    save('native-source-identity.json', {'reconstructed': actual, 'native_report': expected,
        'equal': actual == expected,
        'method': 'Source primitive-order concatenation, glTF-to-Blender [x,-z,y], float32 V flip; exact hashes required before interpreting native IDs.'})
    require(actual == expected, 'Native vertex/triangle/UV identity cannot be proven; do not guess root IDs')
    return points, faces, uvs, slots


def root_audit(source, final, sr, finish, groom, pack, gates):
    points, faces, uvs, slots = native_identity(source, sr, groom)
    path = pack/'material-groom/correspondence.npz'
    with zipfile.ZipFile(path) as archive:
        require(len(archive.infolist()) <= 16 and sum(x.file_size for x in archive.infolist()) <= 8*1024**2,
                'Native correspondence declared size bound')
    with np.load(path, allow_pickle=False) as z:
        require(np.array_equal(z['source_vertex_ids'], np.arange(len(points), dtype=np.int32)), 'Native vertex IDs changed')
        membership = {}
        for part in groom['parts']:
            ids = z[f"part_{part['index']:02d}_face_ids"].copy()
            loop_ids = z[f"part_{part['index']:02d}_loop_ids"].copy()
            require(ids.dtype == np.dtype('<i4') and len(ids) == part['faces']
                    and ahash(ids) == part['source_face_ids_sha256']
                    and ahash(loop_ids) == part['source_loop_ids_sha256'], 'Part correspondence hash/count mismatch')
            require(np.array_equal(ids, np.flatnonzero(slots == part['source_material_slot']))
                    and np.array_equal(loop_ids, (ids[:, None]*3+np.arange(3)).reshape(-1)),
                    'This fox full-material mask differs from native face membership')
            membership[part['source_material']] = set(map(int, ids))
    require(groom['body_before'] == groom['body_after'] and groom['original_body_unchanged'] is True,
            'Native adapter body receipt differs')
    require(groom['actual_parents'] == sum(p['parents'] for p in groom['parts']) == 5000, 'Global native parent count changed')
    require(sum(p['cards'] for p in finish['portable_fur']) == finish['material_coat']['created_cards'] == 3000,
            'Global card count changed')
    material_names = [m['name'] for m in final.doc['materials']]
    results = []
    for report in finish['portable_fur']:
        name = report['source_material']
        source_material = next(m for m in source.doc['materials'] if m['name'] == name)
        route = source_material['pbrMetallicRoughness']['baseColorTexture']
        require(route.get('texCoord', 0) == 0, 'Root audit requires source UV0')
        image = source.doc['images'][source.doc['textures'][route['index']]['source']]
        view = source.doc['bufferViews'][image['bufferView']]
        a = view.get('byteOffset', 0)
        encoded = source.binary[a:a+view['byteLength']]
        with Image.open(io.BytesIO(encoded)) as bitmap:
            require(bitmap.width*bitmap.height <= 16777216 and bitmap.format == 'PNG', 'Root decode bound/format')
            rgb = np.asarray(bitmap.convert('RGB'))
        height, width = rgb.shape[:2]
        portable_material = report['material']['material']
        matches = [p for mesh in final.doc['meshes'] if mesh.get('name', '').startswith('PortableCoat_')
                   for p in mesh['primitives'] if material_names[p['material']] == portable_material]
        require(len(matches) == 1, 'Ambiguous portable material primitive')
        primitive = matches[0]; attrs = primitive['attributes']
        card_faces = h.arrays(final, primitive['indices'], 'SCALAR', (5121, 5123, 5125), indices=True).reshape(-1, 3).astype(int)
        card_points = h.arrays(final, attrs['POSITION'], 'VEC3', (5126,))[:, [0, 2, 1]].copy()
        card_points[:, 1] *= -1
        card_colors = h.arrays(final, attrs['COLOR_0'], 'VEC4', (5121, 5123, 5126))
        color_type = final.doc['accessors'][attrs['COLOR_0']]['componentType']
        color_tolerance = gates['root_linear_rgb'] + {5121: .5/255, 5123: .5/65535, 5126: 0}[color_type]
        require(len(card_faces) == report['cards']*4, 'Portable triangle/card count mismatch')
        metrics = {'material': name, 'cards': report['cards'], 'sampled_roots': len(report['root_samples']),
                   'root_position_error_m': 0., 'root_linear_rgb_error': 0., 'exported_root_rgb_error': 0.,
                   'exported_card_root_center_error_m': 0., 'color_component_type': color_type,
                   'export_color_tolerance': color_tolerance, 'all_samples_owned': True}
        seen = set()
        for sample in report['root_samples']:
            face, root = sample['source_polygon'], sample['root_index']
            require(type(face) is int and face in membership[name] and root not in seen
                    and type(root) is int and 0 <= root < report['cards'], 'Root ownership/index mismatch')
            seen.add(root)
            require(sample['source_vertices'] == faces[face].tolist()
                    and sample['source_loops'] == list(range(face*3, face*3+3))
                    and sample['source_material_slot'] == int(slots[face]), 'Root face/loop/vertex correspondence changed')
            bary = np.asarray(sample['barycentric'], dtype=float)
            require(bary.shape == (3,) and np.isfinite(bary).all() and bary.min() >= 0
                    and abs(float(bary.sum())-1) <= 1e-8, 'Invalid source root barycentrics')
            position = bary @ points[faces[face]].astype(float)
            metrics['root_position_error_m'] = max(metrics['root_position_error_m'], float(abs(position-sample['position']).max()))
            uv = bary @ uvs[face*3:face*3+3].astype(float)
            px, py = uv[0]*width-.5, (1-uv[1])*height-.5
            ix, iy = int(np.floor(px)), int(np.floor(py)); fx, fy = px-ix, py-iy
            samples = rgb[np.array([iy, iy, iy+1, iy+1])%height,
                          np.array([ix, ix+1, ix, ix+1])%width].astype(np.float32)/np.float32(255)
            samples = samples.astype(float)
            linear = np.where(samples <= .04045, samples/12.92, ((samples+.055)/1.055)**2.4)
            expected = np.array([(1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy]) @ linear
            expected *= source_material['pbrMetallicRoughness'].get('baseColorFactor', [1, 1, 1, 1])[:3]
            metrics['root_linear_rgb_error'] = max(metrics['root_linear_rgb_error'], float(abs(expected-sample['root_linear_rgb']).max()))
            ids = np.unique(card_faces[root*4:root*4+4])
            metrics['exported_root_rgb_error'] = max(metrics['exported_root_rgb_error'], float(abs(card_colors[ids, :3]-expected).max()))
            unique = np.unique(card_points[ids], axis=0)
            target = np.asarray(sample['position'])-np.asarray(sample['normal'])*sample['root_inset_m']
            errors = [float(abs((unique[i]+unique[j])*.5-target).max())
                      for i in range(len(unique)) for j in range(i+1, len(unique))]
            require(errors, 'No exported card root pair')
            metrics['exported_card_root_center_error_m'] = max(metrics['exported_card_root_center_error_m'], min(errors))
        check(metrics['root_position_error_m'] <= gates['root_position_m'], 'Sample source root position mismatch '+name)
        check(metrics['root_linear_rgb_error'] <= gates['root_linear_rgb'], 'Independent bilinear source root color mismatch '+name)
        check(metrics['exported_root_rgb_error'] <= color_tolerance, 'Exported sampled tuft color mismatch '+name)
        check(metrics['exported_card_root_center_error_m'] <= gates['root_position_m'], 'Exported card-group root identity mismatch '+name)
        results.append(metrics)
        del rgb, encoded
    check(sum(r['sampled_roots'] for r in results) == 384, 'Expected384 independent root samples')
    save('roots.json', {'materials':results, 'sampling':'Independent sRGB-per-texel decode, REPEAT texel centers, bilinear LOD0; one RGB8 image at a time',
        'native_groom_texture_sampling_verified':False, 'GPU_filtering_verified':False,
        'all_3000_root_colors_verified':False, 'native_parent_total':5000, 'portable_card_total':3000})


def main():
    plan = json.loads((HERE/'prepared.json').read_text())
    for pin in plan['inputs']:
        require(h.sha(ROOT/pin['path']) == pin['sha256'], 'Input hash changed '+pin['path'])
    source, final = _GLB(ROOT/plan['source']), _GLB(ROOT/plan['final'])
    pack = ROOT/plan['pack']
    finish = json.loads((pack/'finish-report.json').read_text())
    groom = json.loads((pack/'material-groom/report.json').read_text())
    names = [source.doc['nodes'][j]['name'] for j in source.doc['skins'][0]['joints']]
    require(len(names) == 20 and len(set(names)) == 20 and len(final.doc['skins']) == 1, 'Skeleton count mismatch')
    view, excluded = body_view(final)
    require(len(excluded) == 3, 'Expected three explicitly excluded PortableCoat meshes')
    sr, fr = h.records(source, names), h.records(view, names)
    require(set(sr) == set(fr), 'Retained body material primitive set changed')
    images = {'source':h.image_records(source), 'final':h.image_records(final)}
    materials = {'source':h.material_semantics(source, images['source']), 'final':h.material_semantics(final, images['final'])}
    retained_source = {name:materials['source'][name] for name in sr}
    retained_final = {name:materials['final'][name] for name in fr}
    check(retained_source == retained_final, 'Retained body material/active image semantics changed')
    save('materials.json', {'source_body':retained_source, 'final_body':retained_final,
        'equal':retained_source == retained_final, 'excluded_new_coat_meshes':excluded,
        'unused_original_card_materials_not_required_in_export':sorted(set(materials['source'])-set(sr))})
    try:
        root_audit(source, final, sr, finish, groom, pack, plan['gates'])
    except Exception as error:
        save('roots-failure.json', {'error':str(error), 'traceback':traceback.format_exc()})
        check(False, 'Root audit failed: '+str(error))
    ns, nf = source.doc['nodes'], final.doc['nodes']
    bs, bf = {n['name']:i for i,n in enumerate(ns)}, {n['name']:i for i,n in enumerate(nf)}
    require(set(bs) <= set(bf) and len(bf) == len(nf), 'Original named nodes missing/duplicated')
    ps, pf = h.parents(source.doc), h.parents(final.doc)
    ws, wf = _world_matrices(ns, ps), _world_matrices(nf, pf)
    rest = {name:float(np.max(abs(np.asarray(ws[bs[name]])-wf[bf[name]]))) for name in bs}
    old_parents = {name:ns[ps[bs[name]]]['name'] if bs[name] in ps else None for name in names}
    new_parents = {name:nf[pf[bf[name]]]['name'] if bf[name] in pf else None for name in names}
    check(old_parents == new_parents and max(rest.values()) <= 1e-6, 'Bone topology/rest world transforms changed')
    save('rig.json', {'bone_names':names, 'source_parents':old_parents, 'final_parents':new_parents,
        'rest_world_matrix_errors':rest, 'units_axes':'Authored metres, glTF +Y up/+Z forward; no measured-anatomy or Weft claim'})
    mapping, geometry = {}, {}
    for name in sr:
        try:
            mapping[name], geometry[name] = h.correspond(sr[name], fr[name])
        except Exception as error:
            geometry[name] = {'status':'failed', 'error':str(error)}
            check(False, 'Body correspondence failed '+name+': '+str(error))
    save('geometry.json', geometry)
    if set(mapping) == set(sr):
        st = {name:t for name,t,_ in _animation_tracks(source, ns, set(source.doc['skins'][0]['joints']))}
        ft = {name:t for name,t,_ in _animation_tracks(final, nf, set(final.doc['skins'][0]['joints']))}
        require(set(st) == set(ft) == {'idle','making_something','milestone_level_up','working'}, 'Source action names changed')
        motion = []
        for name in st:
            bounds = lambda t:[min(x[3][0] for x in t),max(x[3][-1] for x in t)]
            check(bounds(st[name]) == bounds(ft[name]) == [0,4], 'Source clip duration changed '+name)
            check({(ns[t[0]]['name'],t[1]) for t in st[name]} == {(nf[t[0]]['name'],t[1]) for t in ft[name]}, 'Action targets changed '+name)
            samples = []
            for second in np.linspace(0,4,9):
                sw = _world_matrices(ns, ps, _pose(st[name],float(second)))
                fw = _world_matrices(nf, pf, _pose(ft[name],float(second)))
                error = max(float(np.max(abs(h.skin(sr[m],sw)[mapping[m]]-h.skin(fr[m],fw)))) for m in sr)
                check(error <= plan['gates']['sampled_pose_m'], 'Body source pose changed '+name+'/'+str(second))
                samples.append({'second':float(second),'max_component_error_m':error})
            motion.append({'id':name,'duration_seconds':4,'samples':samples})
        save('motion.json', {'states':motion,'body_vertices_per_sample':sum(len(r['positions']) for r in fr.values()),
            'scope':'All retained body vertices at nine times per source clip; excludes new coat deformation and native hair/GPU/loop/physics claims'})
    else:
        save('motion-skipped.json', {'reason':'Ambiguous/failed body correspondence; no source-motion equivalence inferred'})
    RESULT.update({'status':'passed' if not RESULT['failures'] else 'failed',
        'source_sha256':source.hash,'final_sha256':final.hash,
        'source_body_vertices':sum(len(r['positions']) for r in sr.values()),
        'final_body_vertices':sum(len(r['positions']) for r in fr.values()),
        'source_body_triangles':sum(len(r['faces']) for r in sr.values()),
        'final_body_triangles':sum(len(r['faces']) for r in fr.values()),
        'excluded_new_coat_meshes':excluded,
        'native_hair_texture_sampling_verified':False,'portable_motion_verified':False,
        'normal_tangent_byte_equivalence_claimed':False})
    if RESULT['failures']:
        raise ValueError('Independent fox audit recorded failures')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        RESULT.update(status='failed', exception={'type':type(error).__name__, 'message':str(error)})
        raise
    finally:
        RESULT.update(elapsed_seconds=time.monotonic()-START,
            process_high_water_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save('result.json', RESULT)
