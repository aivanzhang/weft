"""Measure bounded saved animation without rendering, hair evaluation or physics claims."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np


VERSION = 'compound-motion-audit-v1'
MAX_VERTICES, MAX_EDGES, MAX_FRAMES = 100000, 300000, 241
MAX_SECONDS, MAX_REPORT_BYTES = 60, 8*1024*1024
WEIGHT_EPS, MOTION_EPS_M = 1e-6, .001


def require(value, message):
    if not value:
        raise ValueError(message)


def frame_schedule(ranges):
    """Sample actual frame ranges at one-frame intervals, retaining fractional endpoints."""
    require(isinstance(ranges, dict) and 1 <= len(ranges) <= 8, 'Require 1–8 actual actions')
    result, total = {}, 0
    for name, bounds in ranges.items():
        require(isinstance(name, str) and 0 < len(name) <= 128 and len(bounds) == 2,
                'Invalid action frame range')
        require(all(type(v) in (int, float) and math.isfinite(v) for v in bounds), 'Nonfinite action range')
        start, end = bounds
        require(0 <= start < end <= 240 and end-start <= 240, 'Action exceeds bounded 10-second timeline')
        frames = [start+i for i in range(math.floor(end-start)+1)]
        if end-frames[-1] > 1e-7:
            frames.append(end)
        total += len(frames)
        require(total <= MAX_FRAMES, 'Aggregate action samples exceed 241 frames')
        result[name] = frames
    return result


def owned_parts(weights, bone_names, parts, subsets, expected_counts):
    """Final weights survive material cloning; mixed ownership cannot identify a clean part."""
    require(isinstance(weights, np.ndarray) and weights.ndim == 2 and
            1 <= weights.shape[0] <= MAX_VERTICES and weights.shape[1] == len(bone_names) <= 64,
            'Invalid bounded weight matrix')
    require(np.isfinite(weights).all() and np.all(weights >= 0), 'Invalid actual skin weights')
    require(np.max(np.abs(weights.sum(axis=1)-1)) <= .001 and np.max((weights > 0).sum(axis=1)) <= 4,
            'Actual skin requires normalized weights and at most four influences')
    labels = np.zeros(len(weights), dtype=np.int8)
    result = []
    for index, part in enumerate(parts):
        names = subsets[part['name']]
        columns = [bone_names.index(name) for name in names]
        inside = (weights[:, columns] > WEIGHT_EPS).any(axis=1)
        outside = [i for i, name in enumerate(bone_names) if name not in names]
        mixed = inside & (weights[:, outside] > WEIGHT_EPS).any(axis=1) if outside else np.zeros(len(weights), bool)
        require(not mixed.any(), 'Part vertices have weights outside their root/descendant subset: '+part['name'])
        require(not np.any(labels[inside]), 'Actual part skin memberships overlap')
        ids = np.flatnonzero(inside)
        require(part['min_vertices'] <= len(ids) <= part['max_vertices'], 'Actual part vertex count is outside bounds')
        require(expected_counts.get(part['name']) == len(ids), 'Actual part count differs from finishing proof: '+part['name'])
        labels[ids] = index+1
        result.append(ids)
    return labels, result


def camera_projection(points, inverse, view_frame, orthographic, margin, clip):
    """Project real vertices plus a conservative fur-expanded world AABB; no rendered-hair claim."""
    require(isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 3 and
            1 <= len(points) <= MAX_VERTICES and np.isfinite(points).all(), 'Invalid camera points')
    inverse, view_frame = np.asarray(inverse), np.asarray(view_frame)
    require(inverse.shape == (4, 4) and view_frame.shape == (4, 3) and
            np.isfinite(inverse).all() and np.isfinite(view_frame).all(), 'Invalid camera matrices')
    require(type(orthographic) is bool and (orthographic or np.all(view_frame[:, 2] < -1e-10)),
            'Invalid perspective camera plane')
    require(type(margin) in (int, float) and math.isfinite(margin) and 0 <= margin <= .2,
            'Invalid configured fur margin')
    require(len(clip) == 2 and all(math.isfinite(v) for v in clip) and 0 < clip[0] < clip[1],
            'Invalid camera clipping range')
    plane = view_frame[:, :2] if orthographic else view_frame[:, :2]/(-view_frame[:, 2, None])
    minimum, extent = plane.min(axis=0), np.ptp(plane, axis=0)
    require(np.all(extent > 1e-10), 'Degenerate camera frame')
    def project(value):
        local = value@inverse[:3, :3].T+inverse[:3, 3]
        depth = -local[:, 2]
        denominator = np.where(np.abs(depth) > 1e-10, depth, 1e-10)
        xy = local[:, :2] if orthographic else local[:, :2]/denominator[:, None]
        uv = (xy-minimum)/extent
        low, high = uv.min(axis=0), uv.max(axis=0)
        return {'uv_min': low.tolist(), 'uv_max': high.tolist(),
                'minimum_image_margin': float(min(low.min(), (1-high).min())),
                'depth_min_m': float(depth.min()), 'depth_max_m': float(depth.max()),
                'points_at_or_behind_camera': int(np.count_nonzero(depth <= 1e-10)),
                'inside_image': bool(np.all(low >= 0) and np.all(high <= 1)),
                'inside_depth': bool(depth.min() >= clip[0] and depth.max() <= clip[1])}
    low, high = points.min(axis=0)-margin, points.max(axis=0)+margin
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    return {'base_vertices': project(points), 'fur_margin_world_aabb': project(corners)}


def root_local_points(points, matrix):
    """Remove root rigid motion while retaining distances in world meters."""
    matrix = np.asarray(matrix, dtype=float)
    require(matrix.shape == (4, 4) and np.isfinite(matrix).all(), 'Invalid actual root matrix')
    scale = np.linalg.norm(matrix[:3, :3], axis=0)
    require(np.all(scale > 1e-10), 'Singular actual root transform')
    basis = matrix[:3, :3]/scale
    require(np.max(np.abs(basis.T@basis-np.eye(3))) < 1e-5 and np.linalg.det(basis) > 0,
            'Sheared/reflected root transform is unsupported')
    return (points-matrix[:3, 3])@basis


def edge_peak(edges, edge_ids, rest_lengths, posed_lengths, frame):
    usable = edge_ids[rest_lengths[edge_ids] > 1e-9]
    if not len(usable):
        return {'extension_m': 0., 'frame': None, 'edge_index': None, 'ignored_tiny_edges': len(edge_ids)}
    edge = int(usable[np.argmax(posed_lengths[usable]-rest_lengths[usable])])
    return {'extension_m': max(0., float(posed_lengths[edge]-rest_lengths[edge])),
            'frame': float(frame), 'second': float(frame/24), 'edge_index': edge,
            'vertex_indices': edges[edge].tolist(), 'rest_length_m': float(rest_lengths[edge]),
            'posed_length_m': float(posed_lengths[edge]), 'ratio': float(posed_lengths[edge]/rest_lengths[edge]),
            'ignored_tiny_edges': int(len(edge_ids)-len(usable))}


def _file(path, maximum):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= maximum,
            'Require a bounded regular local file: '+str(path))
    return path.resolve()


def _digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _deadline(start):
    if time.monotonic()-start > MAX_SECONDS:
        raise TimeoutError('Compound motion audit exceeded 60 seconds')


def audit(blend_path, config_path, report_path, finish_report_path=None):
    import bpy
    sys.path.insert(0, str(Path(__file__).parent))
    from blender_verify import imported_bindings, bind_action, clear_actions, set_frame
    try:
        from part_rigs import validate_part_rigs, part_rig_config
        part_module = sys.modules['part_rigs']
    except ModuleNotFoundError as error:
        if error.name != 'part_rigs':
            raise
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from avatar_assets.part_rigs import validate_part_rigs, part_rig_config
        part_module = sys.modules['avatar_assets.part_rigs']

    started = time.monotonic()
    blend = _file(blend_path, 512*1024*1024)
    config_path = _file(config_path, 1024*1024)
    finish_path = _file(finish_report_path or blend.parent/'finish-report.json', 8*1024*1024)
    out = Path(report_path)
    require(out.parent.is_dir() and not out.exists(), 'Report requires an existing directory and a new output path')
    inputs = {name: {'path': str(path), 'sha256': _digest(path)} for name, path in
              (('blend', blend), ('finish_config', config_path), ('finish_report', finish_path))}
    config, finish = json.loads(config_path.read_text()), json.loads(finish_path.read_text())
    parts = validate_part_rigs(config.get('part_rigs'), config.get('authored_rig'))
    require(isinstance(finish.get('part_rigs'), dict) and finish.get('source_sha256') == config.get('source_sha256'),
            'Finishing proof must identify the same source and part-rig pass')
    expected = {row['name']: row['selected_vertices'] for row in finish['part_rigs'].get('parts', [])}
    require(set(expected) == {part['name'] for part in parts}, 'Finishing proof does not cover every configured part')
    designs = config.get('authored_actions', [])
    require(isinstance(designs, list) and 1 <= len(designs) <= 8, 'Audit requires explicit authored actions')
    names = [design['id'] for design in designs]
    require(len(set(names)) == len(names), 'Duplicate requested action names')
    bpy.context.preferences.filepaths.use_scripts_auto_execute = False
    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False, use_scripts=False)
    require(not bpy.data.libraries and len(bpy.data.objects) <= 256 and len(bpy.data.actions) <= 16,
            'Linked or oversized scenes are unsupported')
    scene, objects = bpy.context.scene, list(bpy.context.scene.objects)
    rigs = [obj for obj in objects if obj.type == 'ARMATURE']
    require(len(rigs) == 1 and 1 <= len(rigs[0].data.bones) <= 64, 'Audit requires one bounded authored armature')
    rig = rigs[0]
    helpers = {bone.custom_shape for bone in rig.pose.bones if bone.custom_shape}
    excluded, bodies = [], []
    for obj in objects:
        role = ('rig_display' if obj in helpers else 'native_groom' if obj.get('native_groom') or
                obj.name.startswith('NativeGroom_') or obj.type == 'CURVES' or
                any(system.settings.type == 'HAIR' for system in obj.particle_systems) else
                'portable_coat' if obj.name.startswith('PortableCoat_') else None)
        if role:
            obj.hide_viewport, obj.hide_render = True, True
            for modifier in obj.modifiers:
                modifier.show_viewport, modifier.show_render = False, False
            excluded.append({'name': obj.name, 'role': role})
        elif obj.type == 'MESH':
            bodies.append(obj)
        for owner in (obj, obj.data):
            data = getattr(owner, 'animation_data', None)
            require(not data or not data.drivers, 'Driver evaluation is unsupported')
    require(len(bodies) == 1, 'Audit requires exactly one base body mesh after helper exclusions')
    obj = bodies[0]
    require(1 <= len(obj.data.vertices) <= MAX_VERTICES and len(obj.data.edges) <= MAX_EDGES and
            len(obj.data.polygons) <= 100000 and not obj.data.shape_keys, 'Base geometry exceeds audit bounds')
    modifiers = [modifier for modifier in obj.modifiers if modifier.show_viewport]
    require(len(modifiers) == 1 and modifiers[0].type == 'ARMATURE' and modifiers[0].object == rig,
            'Base geometry requires only its actual armature deformation')
    require(abs(scene.render.fps/scene.render.fps_base-24) < 1e-7, 'Saved animation must use 24 fps')
    camera = scene.camera
    require(camera and camera.data.type in ('ORTHO', 'PERSP'), 'Saved orthographic or perspective camera required')
    require(all(name in bpy.data.actions for name in names), 'Saved blend is missing a requested action')
    actions = [bpy.data.actions[name] for name in names]
    schedule = frame_schedule({action.name: list(map(float, action.frame_range)) for action in actions})
    for design, action in zip(designs, actions):
        require(abs(action.frame_range[0]-design['keys'][0]['second']*24) < 1e-4 and
                abs(action.frame_range[1]-design['keys'][-1]['second']*24) < 1e-4,
                'Actual action duration differs from the finishing config')
    bindings, owners = imported_bindings(objects, actions)
    binding_modes = {}
    for action in actions:
        if bindings[action.name]:
            require(all(owner == rig for owner, _ in bindings[action.name]), 'Action has unexpected saved owners')
            binding_modes[action.name] = 'saved_owner_slot'
        else:
            require(len(action.slots) <= 1, 'Unbound multi-slot authored action is unsupported')
            bindings[action.name] = [(rig, next(iter(action.slots), None))]
            binding_modes[action.name] = 'single_authored_armature_slot'
    clear_actions(owners, [rig])
    rig.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    bone_names = list(rig.data.bones.keys())
    require(set(bone_names) == {bone['name'] for bone in config['authored_rig']['bones']},
            'Saved bone set differs from the authored rig')
    subsets = {part['name']: [bone['name'] for bone in part_rig_config(part, config['authored_rig'])['bones']] for part in parts}
    require(all(set(subset) <= set(bone_names) for subset in subsets.values()), 'Actual part bones are missing')
    group_columns = {group.index: bone_names.index(group.name) for group in obj.vertex_groups if group.name in bone_names}
    weights = np.zeros((len(obj.data.vertices), len(bone_names)))
    memberships = 0
    for vertex in obj.data.vertices:
        memberships += len(vertex.groups)
        require(memberships <= 800000, 'Too many stored vertex-group memberships')
        for group in vertex.groups:
            if group.group in group_columns:
                weights[vertex.index, group_columns[group.group]] = group.weight
    labels, selected = owned_parts(weights, bone_names, parts, subsets, expected)
    edges = np.empty((len(obj.data.edges), 2), dtype=np.int32)
    obj.data.edges.foreach_get('vertices', edges.ravel())
    require(not len(edges) or (edges.min() >= 0 and edges.max() < len(weights)), 'Invalid stored mesh edges')
    require(not np.any(labels[edges[:, 0]] != labels[edges[:, 1]]), 'Actual topology still joins separate skin parts')
    def points():
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            require(len(mesh.vertices) == len(weights), 'Evaluated topology changed')
            values = np.empty((len(weights), 3), dtype=np.float64)
            mesh.vertices.foreach_get('co', values.ravel())
            matrix = np.asarray(evaluated.matrix_world)
            values = values@matrix[:3, :3].T+matrix[:3, 3]
            require(np.isfinite(values).all(), 'Nonfinite native deformation')
            return values
        finally:
            evaluated.to_mesh_clear()
    rest = points()
    rest_lengths = np.linalg.norm(rest[edges[:, 0]]-rest[edges[:, 1]], axis=1)
    regions = [{'name': '__body__', 'kind': 'outside_parts', 'ids': np.flatnonzero(labels == 0)}]
    regions += [{'name': part['name'], 'kind': 'part', 'ids': ids, 'design': part} for part, ids in zip(parts, selected)]
    require(len(regions[0]['ids']) > 0, 'Compound audit needs a remaining body outside articulated parts')
    for index, region in enumerate(regions):
        ids = region['ids']
        region['edge_ids'] = np.flatnonzero(labels[edges[:, 0]] == index)
        region['rest_centroid'] = rest[ids].mean(axis=0)
        if index:
            part = region['design']
            root = rig.matrix_world@rig.data.bones[part['root_bone']].matrix_local
            region['rest_root_local'] = root_local_points(rest[ids], root)
            region['rest_root_matrix'] = np.asarray(root)
            region['control_ids'] = {name: np.flatnonzero(weights[ids, bone_names.index(name)] > WEIGHT_EPS)
                                     for name in subsets[part['name']] if name != part['root_bone']}
            region['rest_controls'] = {name: np.linalg.solve(region['rest_root_matrix'],
                np.asarray(rig.matrix_world@rig.data.bones[name].matrix_local)) for name in region['control_ids']}
    material_coverage = {region['name']: {} for region in regions}
    for polygon in obj.data.polygons:
        owners_on_face = {int(labels[i]) for i in polygon.vertices}
        require(len(owners_on_face) == 1, 'A stored face mixes actual skin parts')
        owner = next(iter(owners_on_face))
        require(0 <= polygon.material_index < len(obj.data.materials), 'Invalid stored material slot')
        material = obj.data.materials[polygon.material_index]
        name = material.name if material else '<missing>'
        counts = material_coverage[regions[owner]['name']]
        counts[name] = counts.get(name, 0)+1
    margin = max(config.get('groom', {}).get('length', 0)*3,
                 config.get('portable_fur', {}).get('settings', {}).get('length', 0))
    camera_inverse = np.asarray(camera.matrix_world.inverted())
    camera_frame = np.asarray([tuple(point) for point in camera.data.view_frame(scene=scene)])
    camera_matrix = np.asarray(camera.matrix_world).copy()
    report = {'schema_version': 1, 'audit_version': VERSION, 'status': 'measured', 'inputs': inputs,
              'blender_version': bpy.app.version_string, 'body': obj.name, 'armature': rig.name,
              'numpy_version': np.__version__, 'actual_bone_order': bone_names,
              'base_vertices': len(weights), 'base_edges': len(edges), 'excluded_objects': excluded,
              'fps': 24, 'sampled_frames': sum(map(len, schedule.values())),
              'camera': {'name': camera.name, 'type': camera.data.type, 'matrix_world': camera_matrix.tolist(),
                         'view_frame': camera_frame.tolist(), 'fur_margin_m': margin,
                         'fur_margin_rule': 'max(3*native strand length, portable card length); conservative world AABB, not evaluated fur'},
              'part_ownership': [{'name': part['name'], 'vertices': len(ids), 'expected_vertices': expected[part['name']],
                  'bones': subsets[part['name']], 'design_material_names': part['materials'],
                  'actual_material_face_counts': material_coverage[part['name']], 'mixed_weight_vertices': 0,
                  'max_influences': int((weights[ids] > 0).sum(axis=1).max()),
                  'max_weight_sum_error': float(np.max(np.abs(weights[ids].sum(axis=1)-1))),
                  'positive_vertices_per_bone': {name: int(np.count_nonzero(weights[ids, bone_names.index(name)] > WEIGHT_EPS))
                                                 for name in subsets[part['name']]}}
                  for part, ids in zip(parts, selected)], 'actions': [],
              'limits': {'vertices': MAX_VERTICES, 'edges': MAX_EDGES, 'frames': MAX_FRAMES,
                         'seconds': MAX_SECONDS, 'report_bytes': MAX_REPORT_BYTES},
              'scope': 'Actual saved skin measured at 24 fps. Not renders, continuous-time bounds, collision, physics or action-meaning acceptance.',
              'limitations': ['Native groom and portable coat evaluation are disabled; configured margin is not strand-motion proof.',
                             'Centroids are vertex averages, not physical mass centers.',
                             'Control coverage reports overlapping weighted vertices, not isolated causal attribution.',
                             'Final skin membership is compared to finishing counts; material clones do not redefine membership.',
                             'Caller must enforce a process-level 60-second bound in addition to this script deadline.']}
    for action in actions:
        _deadline(started)
        bind_action(action, bindings, owners, [rig])
        rig.data.pose_position = 'POSE'
        item = {'id': action.name, 'binding_mode': binding_modes[action.name],
                'bindings': [{'owner': owner.name, 'slot': slot.identifier if slot else None} for owner, slot in bindings[action.name]],
                'frame_range': list(map(float, action.frame_range)), 'frames': [], 'parts': {}}
        prior, control_stats = {}, {}
        for region in regions:
            item['parts'][region['name']] = {'centroid_travel_m': 0., 'max_centroid_distance_from_rest_m': 0.,
                'world_min_m': [math.inf]*3, 'world_max_m': [-math.inf]*3, 'worst_edge': {'extension_m': -1.}}
            if region['kind'] == 'part':
                control_stats[region['name']] = {name: {'weighted_vertices': len(ids), 'max_relative_matrix_change': 0.,
                    'max_own_pose_matrix_change': 0., 'max_covered_vertex_articulation_m': 0.,
                    '_moving': np.zeros(len(ids), bool)} for name, ids in region['control_ids'].items()}
        for frame in schedule[action.name]:
            _deadline(started)
            set_frame(frame)
            require(np.max(np.abs(np.asarray(camera.matrix_world)-camera_matrix)) < 1e-7,
                    'Animated camera is outside this fixed-camera audit')
            posed = points()
            lengths = np.linalg.norm(posed[edges[:, 0]]-posed[edges[:, 1]], axis=1)
            row = {'frame': frame, 'second': frame/24,
                   'camera': camera_projection(posed, camera_inverse, camera_frame, camera.data.type == 'ORTHO', margin,
                                               (camera.data.clip_start, camera.data.clip_end)), 'parts': {}}
            for region in regions:
                name, ids = region['name'], region['ids']
                cloud = posed[ids]
                center, low, high = cloud.mean(axis=0), cloud.min(axis=0), cloud.max(axis=0)
                peak = edge_peak(edges, region['edge_ids'], rest_lengths, lengths, frame)
                summary = item['parts'][name]
                summary['centroid_travel_m'] += float(np.linalg.norm(center-prior.get(name, center)))
                prior[name] = center
                displacement = float(np.linalg.norm(center-region['rest_centroid']))
                summary['max_centroid_distance_from_rest_m'] = max(summary['max_centroid_distance_from_rest_m'], displacement)
                summary['world_min_m'] = np.minimum(summary['world_min_m'], low).tolist()
                summary['world_max_m'] = np.maximum(summary['world_max_m'], high).tolist()
                if peak['extension_m'] > summary['worst_edge']['extension_m']:
                    summary['worst_edge'] = peak
                row['parts'][name] = {'centroid_m': center.tolist(), 'world_min_m': low.tolist(), 'world_max_m': high.tolist(),
                                      'centroid_distance_from_rest_m': displacement, 'worst_edge': peak}
                if region['kind'] != 'part':
                    continue
                root = np.asarray(rig.matrix_world@rig.pose.bones[region['design']['root_bone']].matrix)
                shifts = np.linalg.norm(root_local_points(cloud, root)-region['rest_root_local'], axis=1)
                for control, covered in region['control_ids'].items():
                    stat = control_stats[name][control]
                    bone = rig.pose.bones[control]
                    relative = np.linalg.solve(root, np.asarray(rig.matrix_world@bone.matrix))
                    rest_relative = region['rest_controls'][control]
                    stat['max_relative_matrix_change'] = max(stat['max_relative_matrix_change'], float(np.max(np.abs(relative-rest_relative))))
                    stat['max_own_pose_matrix_change'] = max(stat['max_own_pose_matrix_change'], float(np.max(np.abs(np.asarray(bone.matrix_basis)-np.eye(4)))))
                    if len(covered):
                        stat['max_covered_vertex_articulation_m'] = max(stat['max_covered_vertex_articulation_m'], float(shifts[covered].max()))
                        stat['_moving'] |= shifts[covered] > MOTION_EPS_M
            item['frames'].append(row)
        for name, controls in control_stats.items():
            for stat in controls.values():
                stat['covered_vertices_moving_over_1mm'] = int(stat.pop('_moving').sum())
                stat['relative_control_motion_observed'] = stat['max_relative_matrix_change'] > 1e-6
                stat['skin_motion_observed'] = stat['relative_control_motion_observed'] and stat['covered_vertices_moving_over_1mm'] > 0
            item['parts'][name]['actual_descendant_controls'] = controls
        item['all_frames_inside_base_camera'] = all(row['camera']['base_vertices']['inside_image'] and
            row['camera']['base_vertices']['inside_depth'] for row in item['frames'])
        item['all_frames_inside_fur_margin_camera'] = all(row['camera']['fur_margin_world_aabb']['inside_image'] and
            row['camera']['fur_margin_world_aabb']['inside_depth'] for row in item['frames'])
        report['actions'].append(item)
    _deadline(started)
    report['source_files_unchanged'] = all(_digest(Path(value['path'])) == value['sha256'] for value in inputs.values())
    require(report['source_files_unchanged'], 'An input changed during the audit')
    script_paths = [Path(__file__), Path(sys.modules['blender_verify'].__file__), Path(part_module.__file__),
                    Path(sys.modules['creature_rig_math'].__file__)]
    report['implementation_sha256'] = {path.name: _digest(path) for path in script_paths}
    report['elapsed_seconds'] = time.monotonic()-started
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n'
    require(len(payload.encode()) <= MAX_REPORT_BYTES, 'Audit output exceeds 8 MiB')
    _deadline(started)
    with out.open('x') as stream:
        stream.write(payload)
    return report


def main():
    args = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('blend')
    parser.add_argument('config')
    parser.add_argument('report')
    parser.add_argument('finish_report', nargs='?')
    options = parser.parse_args(args)
    def timeout(_signum, _frame):
        raise TimeoutError('Compound motion audit exceeded 60 seconds')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        audit(options.blend, options.config, options.report, options.finish_report)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    main()
