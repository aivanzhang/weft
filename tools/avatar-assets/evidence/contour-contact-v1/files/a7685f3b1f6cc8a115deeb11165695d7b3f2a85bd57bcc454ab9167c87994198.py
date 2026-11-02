"""Read-only subpixel intersections; never render, edit a model, or emulate TAA."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'sources'))
from blender_verify import bind_action, hash_file, imported_bindings, require, set_frame, write_json


def prepare_rays(positions, triangles, direction):
    p = np.asarray(positions, dtype=np.float64)
    f = np.asarray(triangles, dtype=np.int32)
    d = np.asarray(direction, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) > 100000 or f.ndim != 2 or f.shape[1] != 3 or len(f) > 100000:
        raise ValueError('Bounded triangle geometry required')
    if not np.isfinite(p).all() or not np.isfinite(d).all() or d.shape != (3,) or abs(np.linalg.norm(d)-1) > 1e-5:
        raise ValueError('Finite geometry and unit ray required')
    if f.size == 0 or f.min() < 0 or f.max() >= len(p):
        raise ValueError('Invalid triangle indices')
    a = p[f[:, 0]]; e1 = p[f[:, 1]]-a; e2 = p[f[:, 2]]-a
    cross = np.cross(e1, e2); lengths = np.linalg.norm(cross, axis=1)
    normal = cross / np.maximum(lengths, 1e-30)[:, None]
    pvec = np.cross(d, e2); det = np.einsum('ij,ij->i', e1, pvec)
    valid = np.abs(det) > 1e-12
    inverse = np.divide(1., det, out=np.zeros_like(det), where=valid)
    return {'a': a, 'e1': e1, 'e2': e2, 'direction': d, 'pvec': pvec,
        'valid': valid, 'inverse': inverse, 'normals': normal, 'dot': normal@d}


def ordered_intersections(origin, cache, culling, limit=8):
    o = np.asarray(origin, dtype=np.float64)
    c = np.asarray(culling)
    if o.shape != (3,) or not np.isfinite(o).all() or c.dtype != np.dtype(bool) or c.shape != cache['valid'].shape or limit != 8:
        raise ValueError('Invalid ray/culling/limit')
    tvec = o-cache['a']; qvec = np.cross(tvec, cache['e1'])
    u = np.einsum('ij,ij->i', tvec, cache['pvec'])*cache['inverse']
    v = (qvec@cache['direction'])*cache['inverse']
    distance = np.einsum('ij,ij->i', cache['e2'], qvec)*cache['inverse']
    hit = np.flatnonzero(cache['valid'] & (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (distance >= 0))
    hit = hit[np.lexsort((hit, distance[hit]))]
    visible = hit[~(c[hit] & (cache['dot'][hit] >= 0))]
    selected = hit[:limit].tolist()
    first_visible = int(visible[0]) if len(visible) else None
    if first_visible is not None and first_visible not in selected:
        selected.append(first_visible)
    rows = [{'triangle': int(i), 'distance_m': float(distance[i]),
        'barycentric': [float(1-u[i]-v[i]), float(u[i]), float(v[i])],
        'normal_dot_forward': float(cache['dot'][i]), 'backface_culled': bool(c[i] and cache['dot'][i] >= 0)} for i in selected]
    near = [int(i) for i in hit if first_visible is not None and abs(distance[i]-distance[first_visible]) <= 1e-5]
    return {'ordered_hits': rows, 'total_intersections': len(hit), 'retained_intersections': len(rows),
        'first_visible_triangle': first_visible, 'near_depth_10um_triangles': near[:16], 'near_depth_10um_total': len(near),
        'truncated': len(hit) > limit}


def main():
    start = time.monotonic()
    args = sys.argv[sys.argv.index('--')+1:]
    require(len(args) == 2, 'Require frozen config and fresh output')
    cfg = json.loads(Path(args[0]).read_text()); out = Path(args[1])
    source = Path(cfg['scene'])
    require(hash_file(source) == cfg['scene_sha256'] and source.stat().st_size < 128*1024**2, 'Source scene changed')
    for row in cfg['pins']:
        require(hash_file(Path(row['path'])) == row['sha256'], 'Input changed')
    require(len(cfg['rays']) <= 320 and cfg['resolution'] == 720, 'Ray budget changed')
    bpy.ops.wm.open_mainfile(filepath=str(source), use_scripts=False)
    scene = bpy.context.scene
    require(scene.camera.data.type == 'ORTHO', 'Require the saved orthographic camera')
    rigs = [o for o in scene.objects if o.type == 'ARMATURE']
    actions = list(bpy.data.actions)
    require(len(rigs) == len(actions) == 1 and actions[0].name == cfg['action'], 'Action ownership changed')
    bindings, owners = imported_bindings(list(scene.objects), actions)
    bind_action(actions[0], bindings, owners, rigs); rigs[0].data.pose_position = 'POSE'
    set_frame(cfg['second']*24)
    identity = {'camera': [[float(x) for x in row] for row in scene.camera.matrix_world],
        'ortho_scale': scene.camera.data.ortho_scale, 'frame': scene.frame_current+scene.frame_subframe,
        'bones': {b.name: [float(x) for row in b.matrix for x in row] for b in rigs[0].pose.bones}}
    require(identity == cfg['pose_camera'], 'Saved pose/camera changed')
    require(scene.render.engine == 'BLENDER_EEVEE_NEXT' and scene.render.resolution_x == scene.render.resolution_y == 720,
        'Saved renderer/resolution changed')
    obj = bpy.data.objects[cfg['body_object']]; original = obj.data
    original.calc_loop_triangles(); tris = list(original.loop_triangles)
    require(len(tris) <= 100000 and len(original.vertices) <= 100000 and len(original.uv_layers) >= 1, 'Body limits changed')
    materials = []
    for i, material in enumerate(original.materials):
        image_routes = []
        for node in material.node_tree.nodes if material.use_nodes else []:
            if node.type == 'TEX_IMAGE':
                vector = node.inputs.get('Vector')
                image_routes.append({'node': node.name, 'image': node.image.name if node.image else None,
                    'interpolation': node.interpolation, 'extension': node.extension,
                    'vector_links': [{'node_type': link.from_node.type, 'uv_map': getattr(link.from_node, 'uv_map', None)}
                        for link in vector.links] if vector else []})
        materials.append({'slot': i, 'name': material.name, 'use_backface_culling': bool(material.use_backface_culling),
            'surface_render_method': getattr(material, 'surface_render_method', None), 'image_routes': image_routes})
    culling = np.asarray([materials[t.material_index]['use_backface_culling'] for t in tris], dtype=bool)
    selected_report = json.loads(Path(cfg['ablation_report']).read_text())
    selected = {row['native_polygon']: row for row in selected_report['mapping']}
    protected = json.loads(Path(cfg['protected_faces']).read_text())['faces']
    protected_bins = {}
    for candidate in protected:
        point = np.asarray(candidate['rest_positions_blender_xyz_m'])[0]
        key = (candidate['material'], *np.floor(point*100000).astype(int))
        protected_bins.setdefault(key, []).append(candidate)
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        require(len(mesh.vertices) == len(original.vertices) and len(mesh.loop_triangles) == len(tris), 'Evaluated topology changed')
        require(all(tuple(a.vertices) == tuple(b.vertices) for a, b in zip(tris, mesh.loop_triangles)), 'Evaluated index ordering changed')
        points = np.asarray([tuple(evaluated.matrix_world@v.co) for v in mesh.vertices])
        faces = np.asarray([tuple(t.vertices) for t in tris], dtype=np.int32)
        camera = scene.camera; basis = camera.matrix_world.to_3x3()
        forward = np.asarray(basis@Vector((0, 0, -1))); right = np.asarray(basis@Vector((1, 0, 0))); up = np.asarray(basis@Vector((0, 1, 0)))
        cache = prepare_rays(points, faces, forward)
        rows = []; unique_faces = {}
        for spec in cfg['rays']:
            require(time.monotonic()-start < 25, 'Internal ray deadline exceeded')
            x, y = spec['pixel']; dx, dy = spec['offset_from_center']
            origin = np.asarray(camera.matrix_world.translation)+right*((x+.5+dx)/720-.5)*camera.data.ortho_scale+up*(.5-(y+.5+dy)/720)*camera.data.ortho_scale
            result = ordered_intersections(origin, cache, culling)
            result.update(pixel=spec['pixel'], offset_from_center=spec['offset_from_center'], role=spec['role'])
            first = result['first_visible_triangle']
            result['first_visible_selected_for_uv_repair'] = first is not None and tris[first].polygon_index in selected
            for hit in result['ordered_hits']:
                index = hit['triangle']; tri = tris[index]
                hit['native_polygon'] = tri.polygon_index
                hit['selected_for_uv_repair'] = tri.polygon_index in selected
                if index in unique_faces:
                    continue
                rest = [obj.matrix_world@original.vertices[i].co for i in tri.vertices]
                uv = np.asarray([tuple(original.uv_layers[0].data[i].uv) for i in tri.loops])
                pos = np.asarray(rest)
                protected_matches = []
                for shift in range(3):
                    key = np.floor(np.roll(pos, shift, axis=0)[0]*100000).astype(int)
                    candidates = []
                    for delta in itertools.product((-1, 0, 1), repeat=3):
                        candidates.extend(protected_bins.get((original.materials[tri.material_index].name, *(key+delta)), []))
                    for candidate in candidates:
                        if np.max(np.abs(np.roll(pos, shift, axis=0)-candidate['rest_positions_blender_xyz_m'])) <= 1e-6 and np.max(np.abs(np.roll(uv, shift, axis=0)-candidate['uv0_blender_v_up'])) <= 2e-6:
                            protected_matches.append([candidate['current_primitive'], candidate['current_triangle']])
                require(len(protected_matches) <= 1, 'Protected-face correspondence ambiguous')
                unique_faces[index] = {'triangle_index': index, 'native_polygon': tri.polygon_index,
                    'material_slot': tri.material_index, 'material': original.materials[tri.material_index].name,
                    'native_vertex_ids': list(tri.vertices), 'native_loops': list(tri.loops),
                    'rest_positions_blender_xyz_m': pos.tolist(), 'uv0_blender_v_up': uv.tolist(),
                    'posed_positions_world_blender_xyz_m': points[list(tri.vertices)].tolist(),
                    'selected_source_face': [selected[tri.polygon_index]['source_primitive'], selected[tri.polygon_index]['source_triangle']] if tri.polygon_index in selected else None,
                    'protected_white_selector': protected_matches[0] if protected_matches else None}
            rows.append(result)
        require(hash_file(source) == cfg['scene_sha256'], 'Source scene changed during trace')
        write_json(out/'subpixel-attribution.json', {'schema_version': 1, 'source_scene_sha256': cfg['scene_sha256'],
            'source_model_sha256': cfg['model_sha256'], 'derivative_model_sha256': cfg['derivative_model_sha256'],
            'pose_camera': identity, 'rays': rows, 'faces': list(unique_faces.values()), 'materials': materials,
            'active_uv_index': original.uv_layers.active_index, 'uv_names': [x.name for x in original.uv_layers],
            'actual_render_filter_size': scene.render.filter_size, 'actual_taa_samples': scene.eevee.taa_render_samples,
            'ray_count': len(rows), 'elapsed_seconds': time.monotonic()-start, 'geometry_edited': False,
            'render_called': False, 'quality_approved': False,
            'scope': 'Exact posed triangle intersections for bounded subpixel rays, honoring actual material backface culling. Not raster TAA/filter reconstruction.',
            'limits': {'rays': 320, 'vertices': 100000, 'triangles': 100000, 'ordered_hits_per_ray': 8,
                'extra_first_visible_if_outside_first8': 1, 'near_depth_record_count': 16, 'near_depth_m': 1e-5},
            'limitations': ['Geometrically coincident hits retain separate triangle indices; sorting ties by index does not predict GPU depth resolution.',
                'The regular7x7 grids are not the engine16-sample TAA jitter pattern.', 'Shader opacity/displacement and other scene objects are not represented by this body-only trace.']})
    finally:
        evaluated.to_mesh_clear()


main()
