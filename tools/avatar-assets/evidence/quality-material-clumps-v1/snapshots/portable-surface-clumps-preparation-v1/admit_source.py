"""Exact raw source-ID/corner admission before any experimental patch construction."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.fringe_export import _array
from avatar_assets.source_skin_fidelity import _snapshot, _canonical, _node_owner
from avatar_assets.three_d import _GLB

START = time.monotonic()
PREP = json.loads((HERE/'prepared.json').read_text())
PATHS = {key: ROOT/row['path'] for key, row in PREP['inputs'].items()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, data):
    with (HERE/name).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True); stream.write('\n')


report = {'schema_version': 1, 'status': 'incomplete', 'geometry_constructed': False,
          'native_provider_ai_role_calls': 0, 'quality_approved': False}
try:
    require(all(sha(PATHS[k]) == v['sha256'] for k, v in PREP['inputs'].items()), 'Input pin changed')
    snap = json.loads(PATHS['skin_report'].read_text())
    skin, owners, bones = _snapshot(PATHS['skin_npz'], snap)
    owner = owners[0]
    require(owner['vertex_start'] == owner['triangle_start'] == 0, 'Unexpected body range')
    nv, nt = owner['vertex_count'], owner['triangle_count']
    points, faces = skin['positions'][:nv], skin['triangles'][:nt]
    materials = skin['triangle_materials'][:nt]
    current = _GLB(PATHS['raw_model'])
    mesh_ids = [i for i, m in enumerate(current.doc['meshes']) if m['name'] == owner['mesh_name']]
    require(len(mesh_ids) == 1, 'Ambiguous body mesh identity')
    mi = mesh_ids[0]
    _node_owner(current.doc, mi, owner, bones)
    lookup = {}
    for index, (triangle, material) in enumerate(zip(faces, materials)):
        key = (int(material), *_canonical(triangle))
        require(key not in lookup, 'Ambiguous oriented source triangle')
        lookup[key] = index
    normals = np.empty((nt, 3, 3), dtype=np.float32)
    uv0 = np.empty((nt, 3, 2), dtype=np.float32)
    provenance = np.full((nt, 2), -1, dtype=np.int32)
    position_error = 0.
    for pi, primitive in enumerate(current.doc['meshes'][mi]['primitives']):
        attrs = primitive['attributes']
        ids = _array(current, attrs['_AUTHOR_VERTEX_ID'], 'SCALAR', {5126}).ravel()
        require(np.all(ids == np.floor(ids)) and ids.min() >= 0 and ids.max() < nv, 'Invalid body source IDs')
        ids = ids.astype(np.int32)
        pos = _array(current, attrs['POSITION'], 'VEC3', {5126})[:, [0, 2, 1]]
        pos[:, 1] *= -1
        position_error = max(position_error, float(np.max(abs(pos-points[ids]))))
        require(np.array_equal(pos, points[ids]), 'Source-ID position mismatch')
        n = _array(current, attrs['NORMAL'], 'VEC3', {5126})[:, [0, 2, 1]]
        n[:, 1] *= -1
        tex = _array(current, attrs['TEXCOORD_0'], 'VEC2', {5126})
        indices = _array(current, primitive['indices'], 'SCALAR', {5121, 5123, 5125}, 300000).ravel().reshape(-1, 3)
        name = current.doc['materials'][primitive['material']]['name']
        slot = owner['materials'].index(name)
        for ti, corners in enumerate(indices):
            vertex_ids = ids[corners]
            target = lookup.get((slot, *_canonical(vertex_ids)))
            require(target is not None and provenance[target, 0] == -1, 'Missing or repeated triangle correspondence')
            rotation = next((k for k in range(3) if np.array_equal(np.roll(vertex_ids, -k), faces[target])), None)
            require(rotation is not None, 'Triangle winding mismatch')
            corners = np.roll(corners, -rotation)
            normals[target], uv0[target], provenance[target] = n[corners], tex[corners], (pi, ti)
    require(np.all(provenance >= 0), 'Incomplete source-face correspondence')
    with np.load(PATHS['fringe_npz'], allow_pickle=False) as archive:
        fringe = {k: archive[k] for k in archive.files}
    root_faces, root_corners = fringe['root_face_ids'], fringe['root_corner_ids']
    require(np.array_equal(faces[root_faces[:, None], root_corners], fringe['root_vertex_ids']),
            'Root face/corner IDs mismatch')
    root_uv_error = float(np.max(abs(uv0[root_faces[:, None], root_corners]-fringe['root_source_uv'])))
    source_root_normals = normals[root_faces[:, None], root_corners].astype(float)
    source_root_normals /= np.linalg.norm(source_root_normals, axis=2, keepdims=True)
    expected_normals = fringe['normals'].reshape(-1, 4, 3)[:, :2]
    normal_errors = np.max(abs(source_root_normals-expected_normals), axis=2)
    max_normal_error = float(normal_errors.max())
    config = json.loads(PATHS['finish_config'].read_text())
    native_report = json.loads(PATHS['finish_report'].read_text())
    excluded = set(config['portable_fringe']['excluded_materials'])
    parts = native_report.get('part_rigs', {}).get('parts', [])
    excluded_part_materials = {name for p in parts if p['name'] in config['groom_excluded_parts'] for name in p['materials']}
    allowed_slots = set()
    for row in native_report['fur_undercoat']['materials']:
        if row['source_material'] not in excluded|excluded_part_materials:
            slot = owner['materials'].index(row['material'])
            require(np.count_nonzero(materials == slot) == row['selected_faces'], 'Undercoat fur-mask face receipt differs')
            allowed_slots.add(slot)
    allowed = np.isin(materials, sorted(allowed_slots))
    vertex_parts = np.zeros(nv, dtype=np.int16)
    for label, part in enumerate(parts, 1):
        member = np.isin(skin['joints'][:nv], [bones.index(name) for name in part['bone_names']]) & (skin['weights'][:nv] > 0)
        selected = member.any(axis=1)
        require(not np.any(selected & (vertex_parts != 0)), 'Part skin ownership conflict')
        vertex_parts[selected] = label
    face_parts = vertex_parts[faces]
    allowed &= (face_parts == 0).all(axis=1)
    selectors = json.loads(PATHS['selectors'].read_text())
    pair_faces = np.asarray(selectors['surface_source_face_pairs'], dtype=np.int32)
    surface_ids = np.asarray(selectors['surface_fin_ids'], dtype=np.int32)
    require(len(surface_ids) == 2700 and pair_faces.shape == (2700, 2), 'Fixed selection changed')
    require(allowed[pair_faces].all(), 'Selected source patch crosses excluded fur/material/part ownership')
    xyz = points[faces].astype(float)
    geometric = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    geometric /= np.linalg.norm(geometric, axis=1, keepdims=True)
    corner_n = normals.astype(float)
    corner_n /= np.linalg.norm(corner_n, axis=2, keepdims=True)
    dots = np.sum(corner_n*geometric[:, None], axis=2)
    selected_dots = dots[pair_faces]
    bad = np.argwhere(selected_dots < PREP['allow_source_normal_dot_min'])
    witnesses = [{'fin': int(surface_ids[r]), 'side': int(s), 'source_face': int(pair_faces[r, s]),
                  'corner': int(c), 'outward_dot': float(selected_dots[r, s, c]),
                  'raw_primitive_face': provenance[pair_faces[r, s]].tolist()} for r, s, c in bad[:32]]
    report.update(status='correspondence_measured', source_faces=nt, source_vertices=nv,
        raw_source_model_sha256=current.hash, body_position_error=position_error,
        root_uv_max_component_error=root_uv_error, root_normal_max_component_error=max_normal_error,
        root_normal_components_over_0002=int(np.count_nonzero(normal_errors > PREP['normal_component_limit'])),
        allowed_fur_materials=[owner['materials'][slot] for slot in sorted(allowed_slots)],
        fur_mask_provenance='Exact current adaptive-undercoat material face receipts; clone assignment requires all original final-mask triangle vertices. Part ownership independently derived from positive snapshot skin.',
        selected_patch_faces_allowed=int(allowed[pair_faces].sum()),
        selected_source_corner_normal_min_outward_dot=float(selected_dots.min()),
        unsafe_source_normal_corners=int(len(bad)), unsafe_source_normal_witnesses=witnesses,
        source_normal_scope='Actual raw-export source-body corner normals, not a new claim of full-body native-normal equivalence')
    write('correspondence-checkpoint.json', report)
    require(root_uv_error <= 1e-6, 'Source UV correspondence exceeds float32 rounding bound')
    require(max_normal_error <= PREP['normal_component_limit'], 'Body/fringe source endpoint normals disagree')
    require(len(bad) == 0, 'Selected source normals cannot support outward surface patches')
    require(time.monotonic()-START < 27, 'Source admission exceeded27seconds')
    data = {'positions': points, 'triangles': faces, 'corner_normals': normals, 'uv0': uv0,
            'material_ids': materials, 'part_ids': face_parts[:, 0], 'fur_faces': allowed,
            'protected_faces': ~allowed, 'joints': skin['joints'][:nv], 'weights': skin['weights'][:nv],
            'raw_primitive_face': provenance, 'surface_fin_ids': surface_ids, 'surface_face_pairs': pair_faces}
    np.savez_compressed(HERE/'admitted-source.npz', **data)
    report.update(status='source_correspondence_admitted_for_cpu_prototype', arrays_sha256=sha(HERE/'admitted-source.npz'))
except (ValueError, KeyError, AssertionError) as exc:
    report.update(status='unsupported_source_admission', failure=str(exc))
report.update(elapsed_seconds=time.monotonic()-START,
              inputs_unchanged=all(sha(PATHS[k]) == v['sha256'] for k, v in PREP['inputs'].items()))
write('admission.json', report)
print(json.dumps(report, indent=2))
