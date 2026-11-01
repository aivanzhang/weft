"""Read-only CPU skin/edge inspection of the rejected robot wave."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from avatar_assets.three_d import _GLB, _world_matrices, _animation_tracks, _pose, MAX_TRIANGLE_INDICES

MODEL = ROOT / 'outputs/agent-robot-v2/model-2/model.glb'
CONFIG = ROOT / 'outputs/agent-robot-v2/model-2/finish-config.json'
OUT = Path(__file__).parent
before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (MODEL, CONFIG)}
glb = _GLB(MODEL)
doc = glb.doc
nodes = doc['nodes']
parents = {child: i for i, node in enumerate(nodes) for child in node.get('children', [])}
mesh_nodes = [(i, node) for i, node in enumerate(nodes) if 'mesh' in node]
assert len(mesh_nodes) == 1
node_id, mesh_node = mesh_nodes[0]
primitives = doc['meshes'][mesh_node['mesh']]['primitives']
assert len(primitives) == 1
primitive = primitives[0]
skin = doc['skins'][mesh_node['skin']]
joint_nodes = skin['joints']
joint_names = [nodes[i].get('name', str(i)) for i in joint_nodes]
attrs = primitive['attributes']
positions = np.asarray(glb.accessor(attrs['POSITION'], 'VEC3', (5126,)))
weights = np.asarray(glb.accessor(attrs['WEIGHTS_0'], 'VEC4'))
joints = np.asarray(glb.accessor(attrs['JOINTS_0'], 'VEC4'), dtype=np.int32)
inverse = np.asarray(glb.accessor(skin['inverseBindMatrices'], 'MAT4', (5126,))).reshape(-1, 4, 4).transpose(0, 2, 1)
triangles = np.asarray(glb.accessor(primitive['indices'], 'SCALAR', count_limit=MAX_TRIANGLE_INDICES), dtype=np.int32).reshape(-1, 3)
edges = np.unique(np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1), axis=0)
homogeneous = np.column_stack([positions, np.ones(len(positions))])

def deform(overrides=None):
    world = np.asarray(_world_matrices(nodes, parents, overrides)).reshape(-1, 4, 4).transpose(0, 2, 1)
    matrices = world[joint_nodes] @ inverse
    points = np.zeros_like(positions)
    for k in range(4):
        if np.any(weights[:, k]):
            points += np.einsum('nij,nj->ni', matrices[joints[:, k], :3, :], homogeneous) * weights[:, k, None]
    return points[:, [0, 2, 1]] * [1, -1, 1]

rest = deform()
low, high = rest.min(axis=0), rest.max(axis=0)
normalized = (rest-low)/(high-low)
rest_lengths = np.linalg.norm(rest[edges[:, 0]]-rest[edges[:, 1]], axis=1)
name, tracks, _ = next(row for row in _animation_tracks(glb, nodes, set(joint_nodes)) if row[0] == 'wave')
times = sorted(set(t for track in tracks for t in track[3]))
assert len(times) <= 4096
maximum = rest_lengths.copy()
maximum_seconds = np.zeros(len(edges))
sample_summary = []
for second in times:
    posed = deform(_pose(tracks, second))
    lengths = np.linalg.norm(posed[edges[:, 0]]-posed[edges[:, 1]], axis=1)
    update = lengths > maximum
    maximum[update] = lengths[update]
    maximum_seconds[update] = second
    sample_summary.append({'second': second, 'max_edge_growth_m': float((lengths-rest_lengths).max())})

config = json.loads(CONFIG.read_text())
rig = config['authored_rig']
def influences(index):
    return {joint_names[j]: float(w) for j, w in zip(joints[index], weights[index]) if w > 1e-6}

def layer_tests(index):
    point = normalized[index]
    result = []
    for layer in rig['layers']:
        failures = []
        for axis, axis_name in enumerate('XYZ'):
            if point[axis] < layer['min'][axis]-1e-7:
                failures.append({'axis': axis_name, 'boundary': 'min', 'value': layer['min'][axis], 'miss_by': float(layer['min'][axis]-point[axis])})
            if point[axis] > layer['max'][axis]+1e-7:
                failures.append({'axis': axis_name, 'boundary': 'max', 'value': layer['max'][axis], 'miss_by': float(point[axis]-layer['max'][axis])})
        result.append({'bones': layer['bones'], 'inside': not failures, 'outside': failures})
    return result

def row(e):
    a, b = map(int, edges[e])
    return {'vertex_ids': [a, b], 'rest_length_m': float(rest_lengths[e]), 'max_posed_length_m': float(maximum[e]),
            'growth_m': float(maximum[e]-rest_lengths[e]), 'stretch_ratio': float(maximum[e]/max(rest_lengths[e], 1e-12)),
            'second': float(maximum_seconds[e]), 'rest_normalized_blender_z_up': normalized[[a,b]].tolist(),
            'rest_blender_z_up_m': rest[[a,b]].tolist(), 'influences': [influences(a), influences(b)],
            'layer_tests': [layer_tests(a), layer_tests(b)]}

growth = maximum-rest_lengths
ranked = np.argsort(growth)[::-1]
top = [row(e) for e in ranked[:20]]
root_joint = joint_names.index('Root')
root_mask = np.any((joints == root_joint)&(weights > .999), axis=1)
hand_left_joint = joint_names.index('HandLeft')
hand_mask = np.any((joints == hand_left_joint)&(weights > .999), axis=1)
root_hand_edges = (root_mask[edges[:,0]]&hand_mask[edges[:,1]]) | (hand_mask[edges[:,0]]&root_mask[edges[:,1]])
bad_root_vertices = np.unique(edges[root_hand_edges].ravel())
bad_root_vertices = bad_root_vertices[root_mask[bad_root_vertices]]
root_boundary = {'root_hand_edge_count': int(root_hand_edges.sum()), 'root_vertex_count_on_hand_edges': int(len(bad_root_vertices)),
                 'root_boundary_normalized_min': normalized[bad_root_vertices].min(axis=0).tolist() if len(bad_root_vertices) else None,
                 'root_boundary_normalized_max': normalized[bad_root_vertices].max(axis=0).tolist() if len(bad_root_vertices) else None,
                 'worst_root_hand_edges': [row(e) for e in ranked if root_hand_edges[e]][:6]}
after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (MODEL, CONFIG)}
assert after == before
report = {'schema_version': 1, 'scope': 'Read-only all-exported-vertex CPU linear-blend skinning and triangle-edge growth at every exported wave key; not a render or physical validation',
          'inputs_sha256': before, 'inputs_unchanged': True, 'vertices': len(positions), 'triangles': len(triangles), 'unique_triangle_edges': len(edges),
          'wave_samples': len(times), 'sample_range_seconds': [times[0], times[-1]], 'joint_names': joint_names,
          'normalized_space': 'normalized_bounds_blender_z_up', 'rest_bounds_m': {'min': low.tolist(), 'max': high.tolist()},
          'top_edges_by_absolute_growth': top, 'root_hand_boundary': root_boundary,
          'edge_growth_threshold_counts': {str(t): int((growth>t).sum()) for t in (.01,.025,.05,.1)},
          'sample_summary': sample_summary}
with (OUT/'report.json').open('x') as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps({key: report[key] for key in ('vertices','triangles','unique_triangle_edges','wave_samples','rest_bounds_m','edge_growth_threshold_counts')}))
print(json.dumps({'worst_edge': {k:v for k,v in top[0].items() if k != 'layer_tests'},
                  'root_hand_boundary': {k:v for k,v in root_boundary.items() if k != 'worst_root_hand_edges'}}))
