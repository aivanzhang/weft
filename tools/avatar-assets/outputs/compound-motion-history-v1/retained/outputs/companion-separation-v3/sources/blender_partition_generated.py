"""Triangulate explicit authored closure loops and render separation diagnostics."""

import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.geometry import tessellate_polygon

sys.path.insert(0, str(Path(__file__).parent))
from partition_generated import MAX_CAPS, MAX_FACES, MAX_VERTICES, require, validate_config, _edges, _components


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main(output):
    out = Path(output)
    config = validate_config(json.loads((out/'config.json').read_text()))
    plan = json.loads((out/'partition-plan.json').read_text())
    require(digest(out/'original.glb') == config['source_sha256'] == plan['source_sha256'], 'Source hash mismatch')
    with np.load(out/'partition-arrays.npz', allow_pickle=False) as data:
        require(set(data.files) == {'points', 'faces', 'vertex_labels', 'face_labels', 'uv_per_corner', 'normal_per_corner'},
                'Unexpected partition arrays')
        points, faces = data['points'], data['faces']
        labels, face_labels = data['vertex_labels'], data['face_labels']
        uvs, normals = data['uv_per_corner'], data['normal_per_corner']
    require(len(points) <= MAX_VERTICES and len(faces) <= MAX_FACES and np.isfinite(points).all(), 'Oversized prepared mesh')
    require(faces.shape == (len(faces), 3) and uvs.shape == (len(faces), 3, 2)
            and normals.shape == (len(faces), 3, 3), 'Invalid prepared corner attributes')
    source_faces = len(faces)
    all_faces, all_uvs, all_normals = faces.tolist(), uvs.tolist(), normals.tolist()
    materials = [int(label) for label in face_labels]
    closure_rows = []
    for number, loop in enumerate(plan['loops']):
        require(3 <= len(loop) <= MAX_CAPS+2 and all(type(i) is int and 0 <= i < len(points) for i in loop),
                'Invalid closure loop')
        coordinates = points[loop]
        center = coordinates.mean(axis=0)
        _, _, basis = np.linalg.svd(coordinates-center, full_matrices=False)
        projected = (coordinates-center)@basis[:2].T
        span = np.ptp(projected, axis=0)
        require(min(span) > 1e-7, 'Degenerate closure projection')
        polygon = [Vector((float(p[0]), float(p[1]), 0)) for p in projected]
        lookup = {tuple(p): i for i, p in enumerate(polygon)}
        require(len(lookup) == len(loop), 'Projected closure has coincident vertices')
        triangles = tessellate_polygon([polygon])
        require(len(triangles) == len(loop)-2, 'Closure triangulation is incomplete')
        part = int(labels[loop[0]])
        rect = plan['palette_patches'][part]['uv_rect_blender']
        local_uv = (projected-projected.min(axis=0))/span
        local_uv = local_uv*np.array([rect[2]-rect[0], rect[3]-rect[1]])+np.array(rect[:2])
        signed_area = sum(float(a[0]*b[1]-b[0]*a[1]) for a, b in zip(projected, np.roll(projected, -1, axis=0)))
        require(abs(signed_area) > 1e-12, 'Closure projected area is zero')
        triangle_area = 0.
        for triangle in triangles:
            # Blender 4.5 returns indices; older builds return the supplied vectors.
            ids = list(triangle) if isinstance(triangle[0], int) else [lookup[tuple(v)] for v in triangle]
            require(all(type(i) is int and 0 <= i < len(loop) for i in ids), 'Invalid tessellation indices')
            signed = float(np.cross(projected[ids[1]]-projected[ids[0]], projected[ids[2]]-projected[ids[0]]))
            if signed*signed_area < 0:
                ids[1], ids[2] = ids[2], ids[1]
            indices = [loop[i] for i in ids]
            normal = np.cross(points[indices[1]]-points[indices[0]], points[indices[2]]-points[indices[0]])
            length = float(np.linalg.norm(normal))
            require(length > 1e-12, 'Closure triangle has zero area')
            normal /= length
            triangle_area += abs(signed)*.5
            all_faces.append(indices)
            all_uvs.append(local_uv[ids].tolist())
            all_normals.append([normal.tolist()]*3)
            materials.append(2+part)
        require(abs(triangle_area-abs(signed_area)*.5) <= max(1e-8, abs(signed_area)*.0001),
                'Closure triangulation overlaps or misses projected surface')
        closure_rows.append({'loop': number, 'part': part, 'vertices': len(loop), 'triangles': len(triangles),
            'bounds_m': {'min': coordinates.min(axis=0).tolist(), 'max': coordinates.max(axis=0).tolist()},
            'max_distance_from_best_fit_plane_m': float(np.abs((coordinates-center)@basis[2]).max()),
            'authored_not_recovered': True, 'uv_palette_patch': plan['palette_patches'][part]})
    require(len(all_faces)-source_faces <= MAX_CAPS and len(all_faces) <= MAX_FACES, 'Closure geometry exceeds caps')
    edges = _edges(np.asarray(all_faces))
    require(all(len(incident) == 2 for incident in edges.values()), 'Closed derivative still has open/nonmanifold edges')
    adjacency = [set() for _ in all_faces]
    for incident in edges.values():
        a, b = incident
        adjacency[a].add(b)
        adjacency[b].add(a)
    components = _components(np.ones(len(all_faces), dtype=bool), adjacency)
    component_parts = [{int(labels[vertex]) for fid in group for vertex in all_faces[fid]} for group in components]
    require(all(len(parts) == 1 for parts in component_parts), 'A geometric component bridges part labels')
    bpy.context.preferences.filepaths.use_scripts_auto_execute = False
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(out/'original.glb'))
    originals = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    require(len(originals) == 1 and not bpy.data.armatures and len(originals[0].data.materials) == 1,
            'Native import differs from prepared single unrigged mesh')
    source_material = originals[0].data.materials[0]
    mesh = bpy.data.meshes.new('PartitionedMesh')
    mesh.from_pydata(points.tolist(), [], all_faces)
    mesh.update()
    obj = bpy.data.objects.new('PartitionedBodyAndPart', mesh)
    bpy.context.scene.collection.objects.link(obj)
    for original in originals:
        bpy.data.objects.remove(original, do_unlink=True)
    for number, name in enumerate(('body_surface', 'companion_surface', 'body_cap', 'companion_cap')):
        material = source_material.copy()
        material.name = name
        if number >= 2:
            bsdf = next(n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
            for key, value in [('Normal', None), ('Metallic', 0.), ('Roughness', 1.)]:
                socket = bsdf.inputs[key]
                for link in list(socket.links):
                    material.node_tree.links.remove(link)
                if value is not None:
                    socket.default_value = value
        mesh.materials.append(material)
    uv_layer = mesh.uv_layers.new(name='UVMap')
    uv_layer.data.foreach_set('uv', np.asarray(all_uvs).ravel())
    for polygon, material_index in zip(mesh.polygons, materials):
        polygon.material_index = material_index
        polygon.use_smooth = material_index < 2
    mesh.normals_split_custom_set(np.asarray(all_normals).reshape((-1, 3)).tolist())
    for part, name in enumerate(('body', config['part_name'])):
        obj.vertex_groups.new(name=name).add(np.flatnonzero(labels == part).tolist(), 1, 'REPLACE')
    attr = mesh.attributes.new(name='partition_id', type='INT', domain='POINT')
    attr.data.foreach_set('value', labels.astype(np.int32))
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(out/'model.glb'), export_format='GLB', use_selection=True,
        export_animations=False, export_skins=False, export_morph=False, export_normals=True,
        export_tangents=True, export_yup=True, export_image_format='AUTO')
    write(out/'component-vertices.json', {'schema_version': 1, 'native_mesh_name': mesh.name,
        'index_space': 'source.blend mesh vertex IDs; exported GLB splits corners by UV/normal/material',
        'material_contract': {'body': ['body_surface', 'body_cap'], config['part_name']: ['companion_surface', 'companion_cap']},
        'components': {name: np.flatnonzero(labels == part).tolist() for part, name in enumerate(('body', config['part_name']))}})
    for image in bpy.data.images:
        if image.size[0]:
            image.pack()
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'source.blend'), check_existing=False)
    # Diagnostic matte overrides reduce texture memory; the saved model retains source material channels.
    for material in mesh.materials:
        bsdf = next(n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        for key, value in [('Normal', None), ('Metallic', 0.), ('Roughness', .8)]:
            for link in list(bsdf.inputs[key].links):
                material.node_tree.links.remove(link)
            if value is not None:
                bsdf.inputs[key].default_value = value
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device, scene.cycles.samples, scene.cycles.use_denoising = 'CPU', 4, False
    scene.render.threads_mode, scene.render.threads = 'FIXED', 2
    scene.render.resolution_x = scene.render.resolution_y = 384
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format, scene.render.image_settings.color_mode = 'PNG', 'RGB'
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new('DiagnosticWorld')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value = (.8, .8, .8, 1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value = .5
    low, high = points.min(axis=0), points.max(axis=0)
    center, extent = Vector((low+high)*.5), float(max(high-low))
    camera_data = bpy.data.cameras.new('DiagnosticCamera')
    camera_data.type, camera_data.ortho_scale = 'ORTHO', extent*1.5
    camera = bpy.data.objects.new('DiagnosticCamera', camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    for name, offset, energy in [('Key', (-2, -3, 4), 600), ('Fill', (3, 1, 2), 400)]:
        light = bpy.data.lights.new(name, 'AREA')
        light.energy, light.size = energy, 3
        light_obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(light_obj)
        light_obj.location = center+Vector(offset)*extent
        light_obj.rotation_euler = (center-light_obj.location).to_track_quat('-Z', 'Y').to_euler()
    renders = []
    for name, offset, separated in [('front', (0, -3, .05), False), ('back', (0, 3, .05), False),
                                    ('open-gap-front', (-.7, -3, .2), True), ('open-gap-back', (.7, 3, .2), True)]:
        displaced = points.copy()
        if separated:
            displaced[labels == 1] += config['preview_offset_m']
        mesh.vertices.foreach_set('co', displaced.ravel())
        mesh.update()
        camera.location = center+Vector(offset)*extent
        camera.rotation_euler = (center-camera.location).to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = str(out/(name+'.png'))
        bpy.ops.render.render(write_still=True)
        renders.append({'path': name+'.png', 'sha256': digest(out/(name+'.png')), 'exploded_diagnostic': separated})
    write(out/'native-report.json', {'schema_version': 1, 'source_sha256': config['source_sha256'],
        'model_sha256': digest(out/'model.glb'), 'source_unchanged': digest(out/'original.glb') == config['source_sha256'],
        'topology': plan['topology'], 'base_faces_retained': source_faces, 'closure_triangles': len(all_faces)-source_faces,
        'vertices': len(points), 'triangles': len(all_faces), 'closed_manifold_edge_counts': True,
        'geometric_component_face_counts': list(map(len, components)),
        'geometric_component_part_labels': [next(iter(parts)) for parts in component_parts],
        'cross_part_edges': 0, 'closure_surfaces': closure_rows, 'renders': renders,
        'preview_shading': 'Diagnostic matte override, same source basecolor; not final lookdev quality',
        'preserved': ['Every original triangle', 'Original surface UVs and corner normals', 'Shared source basecolor image'],
        'authored_assumptions': ['Missing body/companion contact surfaces triangulated from cut boundaries',
            'Cap colors use planar UVs into measured source palette patches, not recovered hidden textures',
            'Pre-existing nonmanifold/open edges are split and closed; no full remesh',
            'Exploded part displacement is diagnostic only, not animation'],
        'animation_verified': False, 'provider_calls': 0})


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--')+1])
