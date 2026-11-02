"""Actual-source CPU rejection proof; a read-only facade never invokes Blender."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT / 'sources'))
from three_d import _GLB, MAX_TRIANGLE_INDICES
from legacy_normal_continuity import plan_normal_continuity as legacy_plan
from normal_continuity import NormalContinuityError
from blender_normal_continuity import apply_normal_continuity

START = time.monotonic()


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)
    if time.monotonic() - START > 29:
        raise TimeoutError('Replay CPU deadline')


def write(name, value):
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with (OUT / name).open('x') as handle:
        handle.write(data)


class LazyRows:
    def __init__(self, rows, make):
        self.rows, self.make = rows, make

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        for index, row in enumerate(self.rows):
            yield self.make(index, row)


class MutationTrap(dict):
    def __init__(self, mutations):
        self.mutations = mutations

    def new(self, *args, **kwargs):
        self.mutations.append('attributes.new')
        raise AssertionError('Adapter attempted mutation before rejecting actual source')


def main():
    pins = json.loads((OUT / 'input-pins.json').read_text())
    require(sha(ROOT / pins['source']['path']) == pins['source']['sha256'], 'Source pin')
    require(sha(OUT / 'config.json') == pins['config']['sha256'], 'Config pin')
    for row in pins['sources']:
        require(sha(OUT / row['snapshot']) == row['sha256'], 'Helper snapshot changed')
    cfg = json.loads((OUT / 'config.json').read_text())
    require(cfg['source_sha256'] == pins['source']['sha256'], 'Config source mismatch')
    glb = _GLB(ROOT / pins['source']['path'])
    require(len(glb.doc['meshes']) == 1 and len(glb.doc['nodes']) == 1 and glb.doc['nodes'][0] ==
            {'mesh': 0, 'name': 'PartitionedBodyAndPart'}, 'Unexpected source transform/ownership')
    points, faces, normals, slots, correspondence = [], [], [], [], []
    count, face_start = 0, 0
    for pi, primitive in enumerate(glb.doc['meshes'][0]['primitives']):
        require(primitive.get('mode', 4) == 4, 'Nontriangle primitive')
        attrs = primitive['attributes']
        p = np.asarray(glb.accessor(attrs['POSITION'], 'VEC3', (5126,)), dtype=np.float64)
        n = np.asarray(glb.accessor(attrs['NORMAL'], 'VEC3', (5126,)), dtype=np.float64)
        idx = np.asarray(glb.accessor(primitive['indices'], 'SCALAR', (5121, 5123, 5125),
                                    count_limit=MAX_TRIANGLE_INDICES), dtype=np.int64).reshape(-1, 3)
        require(p.shape == n.shape and idx.min() >= 0 and idx.max() < len(p), 'Primitive array bounds')
        # Same fixed axis conversion and float32 storage boundary as create_rig.
        p = p[:, [0, 2, 1]] * [1, -1, 1]
        n = n[:, [0, 2, 1]] * [1, -1, 1]
        points.append(p); faces.append(idx + count); normals.append(n[idx].reshape(-1, 3))
        slots.append(np.full(len(idx), primitive['material'], dtype=np.int64))
        correspondence.append({'primitive': pi, 'material': primitive['material'], 'vertex_start': count,
                               'vertices': len(p), 'face_start': face_start, 'faces': len(idx)})
        count += len(p); face_start += len(idx)
    points, faces = np.concatenate(points), np.concatenate(faces)
    normals, slots = np.concatenate(normals), np.concatenate(slots)
    names = [m['name'] for m in glb.doc['materials']]
    low, high = points.min(axis=0), points.max(axis=0)
    height = cfg['authored_rig']['height_meters']
    factor = height / (high[2] - low[2])
    offset = np.array([(low[0]+high[0])/2, (low[1]+high[1])/2, low[2]])
    points = ((points-offset)*factor).astype(np.float32).astype(np.float64)
    arrays = {'points': points, 'faces': faces, 'corner_normals': normals, 'face_materials': slots}
    before = {key: array_hash(value) for key, value in arrays.items()}
    for value in arrays.values():
        value.flags.writeable = False
    config = cfg['normal_continuity']
    require(len(config) == 1, 'Expected one saved transfer')
    legacy = legacy_plan(points, faces, slots, names, normals, config)
    flat = faces.reshape(-1)
    selected = legacy['repaired_vertices'][flat] & np.isin(
        np.repeat(slots, 3), [names.index(n) for n in config[0]['materials']])
    transferred = legacy['corner_normals']
    triangle_area = np.cross(points[faces[:, 1]]-points[faces[:, 0]],
                             points[faces[:, 2]]-points[faces[:, 0]])
    length = np.linalg.norm(triangle_area, axis=1)
    relative_scale = np.maximum(np.sum((points[faces[:, 1]]-points[faces[:, 0]])**2, axis=1),
                                np.sum((points[faces[:, 2]]-points[faces[:, 0]])**2, axis=1))
    invalid = (relative_scale == 0) | (length <= 1e-12*relative_scale)
    direction = np.divide(triangle_area, length[:, None], out=np.zeros_like(triangle_area), where=length[:, None]>0)
    corner_ids = np.flatnonzero(selected)
    face_ids = corner_ids // 3
    dots = np.einsum('ij,ij->i', transferred[corner_ids], direction[face_ids])
    outside = (dots <= 0) & ~invalid[face_ids]
    require(outside.any(), 'Legacy input no longer demonstrates the unsafe transfer')
    independent = {'legacy_plan_status': legacy['report']['status'],
                   'checked_faces': int(len(np.unique(face_ids))), 'checked_corners': int(len(corner_ids)),
                   'degenerate_faces': int(len(np.unique(face_ids[invalid[face_ids]]))),
                   'outside_hemisphere_corners': int(outside.sum()),
                   'minimum_face_dot': float(dots[~invalid[face_ids]].min()),
                   'source_primitive_ranges': correspondence,
                   'method': 'Independent triangle cross-product; transferred values from immutable pre-guard planner'}
    write('independent-legacy-diagnosis.json', independent)
    mutations = []
    mesh = SimpleNamespace(
        vertices=LazyRows(points, lambda i, row: SimpleNamespace(co=tuple(map(float, row)), groups=[])),
        polygons=LazyRows(faces, lambda i, row: SimpleNamespace(vertices=tuple(map(int, row)),
                                                            material_index=int(slots[i]), use_smooth=False)),
        loops=LazyRows(flat, lambda i, value: SimpleNamespace(vertex_index=int(value))),
        corner_normals=LazyRows(normals, lambda i, row: SimpleNamespace(vector=tuple(map(float, row)))),
        materials=[SimpleNamespace(name=name) for name in names],
        uv_layers=[], attributes=MutationTrap(mutations))
    def forbidden(*args, **kwargs):
        mutations.append('normal_write_or_update')
        raise AssertionError('Unexpected mutation')
    mesh.normals_split_custom_set = forbidden
    mesh.update = forbidden
    obj = SimpleNamespace(type='MESH', data=mesh, matrix_world=np.eye(4), vertex_groups=[])
    try:
        apply_normal_continuity(obj, config)
    except NormalContinuityError as error:
        guard = error.report
        write('guard-rejection.json', guard)
    else:
        raise AssertionError('Actual source was not rejected')
    require(not mutations, 'Mutation occurred')
    for name in ['checked_faces', 'checked_corners', 'degenerate_faces', 'outside_hemisphere_corners']:
        require(guard[name] == independent[name], 'Independent count mismatch: '+name)
    require(abs(guard['minimum_face_dot']-independent['minimum_face_dot']) <= 2e-14,
            'Independent minimum dot mismatch')
    require(guard['position_sha256'] == before['points'] and
            guard['face_corner_indices_sha256'] == array_hash(flat), 'Rejection belongs to another geometry')
    require(guard['tested_corner_ids_sha256'] == array_hash(corner_ids) and
            guard['tested_normals_sha256'] == array_hash(transferred[corner_ids]), 'Rejection tested another transfer')
    witness_rows = []
    for witness in guard['witnesses']:
        corner, face, vertex = witness['corner'], witness['face'], witness['vertex']
        require(selected[corner] and face == corner//3 and vertex == flat[corner], 'Witness index mismatch')
        dot = float(transferred[corner] @ direction[face])
        require(witness['reason'] == 'outside_face_hemisphere' and dot <= 0 and
                abs(witness['dot']-dot) <= 2e-14, 'Witness not independently reproduced')
        owner = next(row for row in correspondence if row['face_start'] <= face < row['face_start']+row['faces'])
        witness_rows.append({**witness, 'source_primitive': owner['primitive'],
                             'primitive_triangle': face-owner['face_start'], 'independent_dot': dot})
    require(0 < len(witness_rows) <= 32, 'Bounded witnesses')
    after = {key: array_hash(value) for key, value in arrays.items()}
    require(before == after and sha(ROOT/pins['source']['path']) == pins['source']['sha256'], 'Original data changed')
    require(sha(ROOT/pins['config']['path']) == pins['config']['sha256'], 'Original config changed')
    return {'schema_version': 1, 'status': 'actual_source_rejected_before_mutation',
            'asset_quality_accepted': False, 'vertices': len(points), 'triangles': len(faces),
            'source_sha256': pins['source']['sha256'], 'source_config_sha256': pins['config']['sha256'],
            'config': config, 'transform': {'blender_axes': '[gltf.x,-gltf.z,gltf.y]',
                'height_meters': height, 'translation_before_scale': (-offset).tolist(), 'uniform_scale': float(factor),
                'position_storage': 'float32, then float64 CPU view'},
            'original_data_before': before, 'original_data_after': after, 'original_file_bytes_unchanged': True,
            'adapter_mutation_attempts': mutations, 'guard_report_sha256': sha(OUT/'guard-rejection.json'),
            'independent_counts': independent, 'independently_checked_witnesses': witness_rows,
            'proof_scope': 'Actual original GLB arrays and saved config through the production Python adapter with read-only mesh facade; not an analytic character fixture.',
            'limits': ['No Blender/native call; import custom-normal encoding was not re-executed.',
                       'No new rigging, texture decoding, image generation, or repaired avatar.',
                       'This proves preventive rejection, not smooth geometry or accepted appearance.'],
            'elapsed_seconds': time.monotonic()-START,
            'process_high_water_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


try:
    result = main()
except Exception as error:
    write('failure.json', {'status': 'failed', 'error': str(error), 'type': type(error).__name__,
                          'elapsed_seconds': time.monotonic()-START})
    raise
else:
    write('result.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'vertices', 'triangles', 'adapter_mutation_attempts',
        'elapsed_seconds', 'process_high_water_rss_bytes')}, sort_keys=True))
