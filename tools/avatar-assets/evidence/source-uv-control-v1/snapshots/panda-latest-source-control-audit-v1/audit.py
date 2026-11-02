"""Read-only source/control audit; no texture decode, Blender, or material mutation."""
import copy
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
import time

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class GLB:
    def __init__(self, path):
        require(Path(path).is_file() and not Path(path).is_symlink(), 'Invalid model path')
        require(Path(path).stat().st_size <= 64*1024**2, 'Model exceeds64MiB')
        self.raw = Path(path).read_bytes()
        require(struct.unpack_from('<4sII', self.raw) == (b'glTF', 2, len(self.raw)), 'Invalid GLB header')
        n, kind = struct.unpack_from('<II', self.raw, 12)
        require(kind == 0x4e4f534a and n <= 2*1024**2, 'Invalid GLB JSON')
        self.doc = json.loads(self.raw[20:20+n])
        count, kind = struct.unpack_from('<II', self.raw, 20+n)
        require(kind == 0x004e4942 and 28+n+count == len(self.raw), 'Invalid GLB BIN')
        self.binary = memoryview(self.raw)[28+n:]

    def array(self, index):
        a = self.doc['accessors'][index]
        require('sparse' not in a and 0 < a['count'] <= 300000 and not a.get('normalized'), 'Unsupported accessor')
        view = self.doc['bufferViews'][a['bufferView']]
        require(view.get('buffer') == 0, 'External buffer')
        dtype = np.dtype({5121:'<u1', 5123:'<u2', 5125:'<u4', 5126:'<f4'}[a['componentType']])
        width = {'SCALAR':1, 'VEC2':2, 'VEC3':3, 'VEC4':4}[a['type']]
        step = view.get('byteStride', width*dtype.itemsize)
        start = view.get('byteOffset', 0)+a.get('byteOffset', 0)
        require(start+(a['count']-1)*step+width*dtype.itemsize <= len(self.binary), 'Accessor exceeds BIN')
        result = np.ndarray((a['count'], width), dtype=dtype, buffer=self.binary,
                            offset=start, strides=(step, dtype.itemsize))
        require(np.isfinite(result).all(), 'Nonfinite accessor')
        return result


def tangent_summary(points, normals, stored, uv, faces):
    xyz = points[faces].astype(np.float64)
    tex = uv[faces].astype(np.float64)
    e1, e2 = xyz[:,1]-xyz[:,0], xyz[:,2]-xyz[:,0]
    a, b = tex[:,1]-tex[:,0], tex[:,2]-tex[:,0]
    determinant = a[:,0]*b[:,1]-a[:,1]*b[:,0]
    valid = np.abs(determinant) > 1e-12
    du = (e1[valid]*b[valid,1,None]-e2[valid]*a[valid,1,None])/determinant[valid,None]
    n = normals[faces[valid]].astype(np.float64)
    n /= np.linalg.norm(n, axis=2)[...,None]
    t = np.repeat(du[:,None,:], 3, axis=1)
    t -= n*np.sum(n*t, axis=2)[...,None]
    sizes = np.linalg.norm(t, axis=2)
    good = sizes > 1e-10
    t /= np.maximum(sizes, 1e-10)[...,None]
    source = stored[faces[valid],:3].astype(np.float64)
    source /= np.linalg.norm(source, axis=2)[...,None]
    angles = np.degrees(np.arccos(np.clip(np.sum(t*source, axis=2)[good], -1, 1)))
    return {'nondegenerate_uv_faces': int(valid.sum()), 'valid_corner_frames': int(good.sum()),
            'angle_degrees_min_median_max': [float(v) for v in (angles.min(), np.median(angles), angles.max())],
            'corners_over_15_degrees': int(np.sum(angles > 15)),
            'corners_over_45_degrees': int(np.sum(angles > 45)),
            'uv_determinant_positive_negative_zero': [int(np.sum(determinant > 1e-12)),
                int(np.sum(determinant < -1e-12)), int(np.sum(~valid))],
            'scope': 'Face-local projected dPdu versus stored tangent XYZ; not a Mikk shared-vertex oracle or shader result.'}


def main():
    start = time.monotonic()
    prep = json.loads((OUT/'prepared.json').read_text())
    for item in prep['inputs']:
        require(sha(ROOT/item['path']) == item['sha256'], 'Changed pinned input: '+item['path'])
    model = GLB(ROOT/prep['source'])
    previous = GLB(ROOT/prep['original_source'])
    cfg = json.loads((ROOT/prep['staged_config']).read_text())
    old_report = json.loads((ROOT/prep['staged_report']).read_text())
    doc = model.doc
    require(len(doc['meshes']) == 1, 'Unexpected mesh count')
    require(not doc.get('skins') and not doc.get('animations'), 'Expected unrigged upstream source')
    primitives = doc['meshes'][0]['primitives']
    material_names = [m.get('name') for m in doc['materials']]
    require(len(material_names) == len(set(material_names)), 'Ambiguous material names')
    totals = {'vertices':0, 'triangles':0}
    rows, points_all = [], []
    for pi, primitive in enumerate(primitives):
        require(primitive.get('mode',4) == 4 and not primitive.get('targets'), 'Unsupported primitive')
        attrs = primitive['attributes']
        points = model.array(attrs['POSITION'])
        faces = model.array(primitive['indices']).ravel().reshape(-1,3)
        require(faces.max() < len(points), 'Invalid triangle index')
        counts = np.bincount(faces.ravel(), minlength=len(points))
        mat = doc['materials'][primitive['material']]
        uv_rows = {}
        for name in ('TEXCOORD_0','TEXCOORD_1'):
            if name in attrs:
                uv = model.array(attrs[name]); require(len(uv) == len(points) and np.abs(uv).max() <= 1e6, 'Invalid UV')
                uv_rows[name] = {'accessor':attrs[name], 'min':uv.min(axis=0).tolist(), 'max':uv.max(axis=0).tolist(),
                                 'raw_sha256':hashlib.sha256(uv.tobytes()).hexdigest()}
        material_routes = {}
        for name in ('normalTexture','occlusionTexture','emissiveTexture'):
            if name in mat: material_routes[name] = mat[name]
        for name in ('baseColorTexture','metallicRoughnessTexture'):
            if name in mat.get('pbrMetallicRoughness',{}): material_routes['pbrMetallicRoughness.'+name] = mat['pbrMetallicRoughness'][name]
        rows.append({'primitive':pi,'material_index':primitive['material'],'material':mat.get('name'),
            'vertices':len(points),'triangles':len(faces),'attributes':attrs,'texture_routes':material_routes,
            'uv':uv_rows,'indices_unique':int(np.count_nonzero(counts)), 'index_use_max':int(counts.max()),
            'vertices_shared_by_multiple_faces':int(np.sum(counts>1)), 'fully_unshared_triangle_corners':bool(np.all(counts == 1)),
            'worst_all_faces_atlas_additional_vertices':int(3*len(faces)-np.count_nonzero(counts))})
        points_all.append(points)
        totals['vertices'] += len(points); totals['triangles'] += len(faces)
    require(totals == {'vertices':51376,'triangles':62581}, 'Unexpected source counts')
    source_low = np.minimum.reduce([p.min(axis=0) for p in points_all])
    source_high = np.maximum.reduce([p.max(axis=0) for p in points_all])
    with np.load(ROOT/prep['correspondence'], allow_pickle=False) as archive:
        face_map = archive['face_map']
    clone_reports = []
    for pi in (4,5):
        primitive = primitives[pi]; attrs = primitive['attributes']
        faces = model.array(primitive['indices']).ravel().reshape(-1,3)
        mapped = face_map[face_map[:,2] == pi]
        mapped = mapped[np.argsort(mapped[:,3])]
        require(np.array_equal(mapped[:,3], np.arange(len(faces))), 'Incomplete clone correspondence')
        parent_ids = np.unique(mapped[:,0]); require(len(parent_ids) == 1, 'Ambiguous clone source')
        old = previous.doc['meshes'][0]['primitives'][int(parent_ids[0])]
        old_faces = previous.array(old['indices']).ravel().reshape(-1,3)[mapped[:,1]]
        exact = {}
        for name in ('POSITION','NORMAL','TANGENT'):
            exact[name] = bool(np.array_equal(model.array(attrs[name])[faces], previous.array(old['attributes'][name])[old_faces]))
        exact['TEXCOORD_1_equals_original_UV0'] = bool(np.array_equal(model.array(attrs['TEXCOORD_1'])[faces], previous.array(old['attributes']['TEXCOORD_0'])[old_faces]))
        require(all(exact.values()), 'Clone source frame payload changed')
        points, normals, tangent = (model.array(attrs[name]) for name in ('POSITION','NORMAL','TANGENT'))
        norms = np.linalg.norm(normals.astype(np.float64),axis=1)
        tangents = np.linalg.norm(tangent[:,:3].astype(np.float64),axis=1)
        local = {'primitive':pi,'material':rows[pi]['material'], 'source_primitive':int(parent_ids[0]),
            'exact_preservation':exact,'topology':{key:rows[pi][key] for key in ('triangles','vertices','vertices_shared_by_multiple_faces','index_use_max','fully_unshared_triangle_corners')},
            'source_normal_map_present': 'normalTexture' in doc['materials'][primitive['material']],
            'stored_normal_length_error_max':float(np.abs(norms-1).max()),
            'stored_tangent_length_error_max':float(np.abs(tangents-1).max()),
            'stored_tangent_w_counts':{str(v):int(np.sum(tangent[:,3] == v)) for v in np.unique(tangent[:,3])},
            'stored_T_vs_current_UV0':tangent_summary(points,normals,tangent,model.array(attrs['TEXCOORD_0']),faces),
            'stored_T_vs_preserved_UV1':tangent_summary(points,normals,tangent,model.array(attrs['TEXCOORD_1']),faces)}
        clone_reports.append(local)
    bones = [b['name'] for b in cfg['authored_rig']['bones']]
    named_part_materials = {name for part in cfg['part_rigs'] for name in part['materials']}
    require(named_part_materials <= set(material_names), 'A part material is missing')
    part_rows = []
    for part in cfg['part_rigs']:
        chosen = [row for row in rows if row['material'] in part['materials']]
        part_rows.append({'name':part['name'],'materials':part['materials'],
            'source_export_vertices':sum(r['vertices'] for r in chosen),
            'triangles':sum(r['triangles'] for r in chosen),
            'native_count_scope':'Export seam copies are not native welded vertex counts; native min/max coverage must be rechecked.',
            'root_bone':part['root_bone'],'root_present_in_unchanged_authored_rig':part['root_bone'] in bones})
    control = copy.deepcopy(cfg)
    control['id'] = 'panda-latest-source-normal-control-v1'
    control['source'] = str((ROOT/prep['source']).resolve())
    control['source_sha256'] = prep['source_sha256']
    del control['normal_continuity']
    control['portable_fringe']['excluded_materials'] = ['body_cap','body_cap__contact_uv_dark_v1']
    from scripts.edge_fringe import export_budget
    control['fringe_export_budget'] = export_budget(totals['vertices'], totals['triangles'], control['portable_fringe']['settings']['max_fins'])
    control['undercoat_export_budget']['source_export_vertices'] = totals['vertices']
    from avatar_assets.finish_generated import validate
    internal = {'export_model','sample_states','fringe_export_budget','undercoat_export_budget'}
    host_control = {k:v for k,v in control.items() if k not in internal}
    validate(host_control)
    changed = [key for key in sorted(set(cfg)|set(control)) if cfg.get(key) != control.get(key)]
    protected = ['authored_rig','authored_actions','states','camera_fit','camera_view','weight_constraints','part_rigs',
                 'groom','groom_flow','groom_exclusions','groom_excluded_parts','groom_shader','fur_undercoat',
                 'color_corrections','normal_strength','resolution','pose_resolution','samples','review_views','portrait_denoise']
    invariant = {key:{'unchanged':cfg[key] == control[key], 'canonical_sha256':digest(cfg[key])} for key in protected}
    require(all(row['unchanged'] for row in invariant.values()), 'Control changed protected authoring intent')
    write('control-native-config.json',control)
    write('control-host-design.json',host_control)
    write('control-delta.json',{'status':'prepared_not_released_for_native','base_config':prep['staged_config'],
        'base_sha256':sha(ROOT/prep['staged_config']), 'changed_top_level_fields':changed,
        'remove':{'normal_continuity':{'old':cfg['normal_continuity'],
            'reason':'The old transferred normal field is proved invalid. Omission selects unmodified source normals for a CONTROL; no guard threshold or normal repair implementation is changed.'}},
        'set':{key:control[key] for key in changed if key in control}, 'unchanged_bindings':invariant,
        'normal_map_strength_retained':control['normal_strength'],
        'public_fringe_guard_unchanged_and_disabled':True,'native_execution_allowed':False,
        'quality_claim':False})
    budget = control['fringe_export_budget']
    previous_plan = old_report['fur_undercoat']['plan']
    report = {'schema_version':1,'status':'held_for_source_uv1_tangent_fidelity', 'git_head':prep['git_head'],
        'source_sha256':prep['source_sha256'],'source_bytes':len(model.raw),**totals,
        'scene':{'scene':doc.get('scene'),'scenes':doc.get('scenes'),'nodes':doc.get('nodes'),
                 'meshes':[{'name':m.get('name'),'primitives':len(m['primitives'])} for m in doc['meshes']],
                 'skins':len(doc.get('skins',[])), 'animations':len(doc.get('animations',[])),
                 'local_gltf_y_up_bounds':{'min':source_low.tolist(),'max':source_high.tolist()}},
        'material_primitives':rows,'clone_source_frame_evidence':clone_reports,'part_selectors':part_rows,
        'material_policy':{
            'part_exclusions_preserved':['companion_surface','companion_cap'],
            'new_body_surface_clone':'body_surface__contact_uv_dark_v1',
            'new_body_cap_clone':'body_cap__contact_uv_dark_v1',
            'portable_fringe_closure_exclusions':control['portable_fringe']['excluded_materials'],
            'potential_undercoat_body_regions':['body_surface','body_cap','body_surface__contact_uv_dark_v1','body_cap__contact_uv_dark_v1'],
            'region_limit':4,'no_implicit_material_merge':True,
            'tail_color_clone':'All corrected vertices remain excluded by the unchanged color-correction return mask. If a fifth region remains selected, the4-region guard must reject it.',
            'adaptive_socket_policy':'Existing source Roughness/Metallic links remain preserved; authored constants apply only to unlinked sockets.',
            'source_uv_policy':'Existing UV0 and UV1 must remain exact; only a newly authored atlas can use UV2. Original source UV1 is never an authored correction target.'},
        'export_budget':{**budget,'headroom_for_additional_adaptive_seams':100000-budget['projected_vertices'],
            'adaptive_seams_measured_for_latest_source':False,
            'historical_old_source_additional_seams':previous_plan['additional_export_vertices'],
            'historical_value_is_not_current_measurement':True,
            'potential_all_faces_chart_split_extras':sum(rows[i]['worst_all_faces_atlas_additional_vertices'] for i in (2,4,5)),
            'latest_actual_mask_and_material_seams_require_bounded_native_preflight':True,
            'overall_budget_pass':False},
        'hard_blockers':[
            'Installed Blender exporter calls calc_tangents() without uvmap; it recomputes source contact normal-map frames from active UV0 although the material samples preserved UV1. Native/full-control export must not proceed until source frame preservation is independently verified.',
            'The latest regenerated chroma/spatial/part/color mask and material/chart splits have not been measured. Only1328 vertices remain within the conservative100k budget.',
            'Public portable_fringe remains intentionally disabled; this audit does not bypass it or establish full-avatar acceptance.'],
        'scope_limits':['No texture pixels decoded; final chroma/color-correction masks not inferred from material names.',
            'No Blender import, UV active/render state observation, rig build, bake, render, action playback or appearance assessment.',
            'Face-local tangent comparisons are diagnostic, not shared-Mikk reconstruction.',
            'Existing contact/nonmanifold geometry and authored source-normal defects remain; this CONTROL is not a repair or quality promotion.'],
        'inputs_unchanged':all(sha(ROOT/item['path']) == item['sha256'] for item in prep['inputs']),
        'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'elapsed_seconds':round(time.monotonic()-start,6),'texture_decodes':0,'native_calls':0,
        'paid_calls':0,'quality_approved':False}
    require(report['inputs_unchanged'],'Input mutation')
    write('report.json',report)
    print(json.dumps({'status':report['status'], 'vertices':totals['vertices'],
        'budget':report['export_budget'], 'clone_topology':[r['topology'] for r in clone_reports],
        'self_peak_rss_bytes':report['self_peak_rss_bytes'],'elapsed_seconds':report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
