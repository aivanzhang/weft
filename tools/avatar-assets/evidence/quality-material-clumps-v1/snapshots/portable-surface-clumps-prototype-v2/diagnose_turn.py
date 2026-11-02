"""Read-only localization of the unchanged failed lift, not a new candidate."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
started = time.monotonic()
prepared = json.loads((HERE/'prepared.json').read_text())
source = ROOT/prepared['inputs']['source_arrays']['path']
fringe_path = ROOT/prepared['inputs']['fringe_npz']['path']
assert hashlib.sha256(source.read_bytes()).hexdigest() == prepared['inputs']['source_arrays']['sha256']
with np.load(source, allow_pickle=False) as archive:
    data = {key: archive[key] for key in archive.files}
with np.load(fringe_path, allow_pickle=False) as archive:
    roots = archive['root_vertex_ids']
order = np.argsort(data['surface_fin_ids'])
selected = data['surface_fin_ids'][order]
pairs = np.sort(data['surface_face_pairs'][order], axis=1)
face_ids = pairs.ravel()
source_ids = data['triangles'][face_ids]
original = data['positions'][source_ids].astype(float)
normals = data['corner_normals'][face_ids].astype(float)
normals /= np.linalg.norm(normals, axis=2)[..., None]
own_roots = np.repeat(roots[selected], 2, axis=0)
is_root = (source_ids[:, :, None] == own_roots[:, None]).any(axis=2)
patched = (original+normals*.0012*(~is_root)[..., None]).astype(np.float32).astype(float)
cross = np.cross(original[:, 1]-original[:, 0], original[:, 2]-original[:, 0])
cross_after = np.cross(patched[:, 1]-patched[:, 0], patched[:, 2]-patched[:, 0])
unit = cross/np.linalg.norm(cross, axis=1)[:, None]
unit_after = cross_after/np.linalg.norm(cross_after, axis=1)[:, None]
degrees = np.degrees(np.arccos(np.clip((unit*unit_after).sum(axis=1), -1, 1)))
root_positions = data['positions'][own_roots].astype(float)
root_length = np.linalg.norm(root_positions[:, 1]-root_positions[:, 0], axis=1)
altitude = np.linalg.norm(cross, axis=1)/root_length
edge = (root_positions[:, 1]-root_positions[:, 0])/root_length[:, None]
opposite = (~is_root).argmax(axis=1)
tip = original[np.arange(len(original)), opposite]
tip_normal = normals[np.arange(len(normals)), opposite]
transverse = tip-root_positions[:, 0]
transverse -= edge*(transverse*edge).sum(axis=1)[:, None]
transverse /= np.linalg.norm(transverse, axis=1)[:, None]
a = (tip_normal*transverse).sum(axis=1)
b = (tip_normal*unit).sum(axis=1)
tangent = np.tan(np.deg2rad(15))
assert np.all(b-a*tangent > 0)
ceiling = altitude*tangent/(b-a*tangent)
card_ceiling = np.minimum(.0012, ceiling.reshape(-1, 2).min(axis=1))
continuous_angle = np.degrees(np.arctan2(.0012*b, altitude+.0012*a))
bad = np.flatnonzero(degrees > 15)
witnesses = []
for index in bad[np.argsort(-degrees[bad])]:
    opposite = int(np.flatnonzero(~is_root[index])[0])
    witnesses.append({'fin': int(selected[index//2]), 'native_source_face': int(face_ids[index]),
        'raw_glb_primitive_face': data['raw_primitive_face'][face_ids[index]].tolist(),
        'source_vertex_ids': source_ids[index].tolist(), 'degrees': float(degrees[index]),
        'source_altitude_m': float(altitude[index]), 'root_length_m': float(root_length[index]),
        'continuous_15degree_lift_ceiling_m': float(ceiling[index]),
        'source_points_m': original[index].tolist(), 'offset_points_m': patched[index].tolist(),
        'offset_corner_source_normal': normals[index, opposite].tolist(),
        'source_normal_geometric_dot': float(np.dot(normals[index, opposite], unit[index]))})
report = {'schema_version': 1, 'status': 'read_only_failed_lift_localization',
    'source_arrays_sha256': prepared['inputs']['source_arrays']['sha256'],
    'selection_changed': False, 'new_candidate_constructed': False,
    'model_written': False, 'native_provider_ai_role_calls': 0,
    'selected_source_faces': len(face_ids), 'failed_source_faces': len(bad),
    'affected_fins': len(np.unique(selected[bad//2])), 'maximum_turn_degrees': float(degrees.max()),
    'all_source_altitude_quantiles_mm': np.quantile(altitude*1000, [0, .01, .5, .95, 1]).tolist(),
    'failed_source_altitude_quantiles_mm': np.quantile(altitude[bad]*1000, [0, .01, .5, .95, 1]).tolist(),
    'continuous_ceiling_derivation': {'fixed_roots': True, 'gate_degrees': 15,
        'formula': 'L_max = h*tan(15deg)/(b-a*tan(15deg)); h=source altitude, a=dot(source corner normal, transverse altitude direction), b=dot(source corner normal, original geometric normal)',
        'meaning': 'Continuous float64 ceiling only; actual float32 offset and intersection gates must still pass in any separately approved candidate.',
        'observed_float32_angle_vs_continuous_max_delta_degrees': float(np.max(abs(degrees-continuous_angle)))},
    'per_card_capped_ceiling_quantiles_mm': np.quantile(card_ceiling*1000, [0, .01, .05, .5, .95, 1]).tolist(),
    'cards_requiring_less_than_1p2mm': int(np.count_nonzero(card_ceiling < .0012)),
    'cards_with_ceiling_below_mm': {str(mm): int(np.count_nonzero(card_ceiling < mm/1000)) for mm in (.1, .2, .4, .6, 1)},
    'witnesses': witnesses, 'elapsed_seconds': time.monotonic()-started,
    'limitation': 'The fixed lift failed before intersection or atlas work; no safe alternative is established.'}
with (HERE/'normal-turn-diagnosis.json').open('x') as stream:
    json.dump(report, stream, indent=2); stream.write('\n')
print(json.dumps({key: value for key, value in report.items() if key != 'witnesses'}, indent=2))
