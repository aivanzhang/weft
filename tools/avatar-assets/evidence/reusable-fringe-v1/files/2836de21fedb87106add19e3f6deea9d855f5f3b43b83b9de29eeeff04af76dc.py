"""One bounded native adapter fixture; this is not an avatar or an appearance test."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def native(out):
    import math
    import bpy
    import numpy as np
    from mathutils import Quaternion
    sys.path.insert(0, str(Path(__file__).parent))
    from blender_edge_fringe import attach_edge_fringe
    from blender_finish_generated import exact_root_texture_contract

    out = Path(out)
    prepared = json.loads((out/'prepared.json').read_text())
    for row in prepared['sources']:
        assert digest(out/row['snapshot']) == row['sha256']
    for row in prepared['inputs']:
        assert digest(out/row['path']) == row['sha256']
    config = prepared['fringe']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps, scene.frame_start, scene.frame_end = 24, 1, 25
    n = 13
    points, normals, uv = [], [], []
    for zi in range(n):
        for xi in range(n):
            theta = -.65 + 1.3*xi/(n-1)
            points.append([.1*math.sin(theta), -.1*math.cos(theta), .012*zi])
            normals.append([math.sin(theta), -math.cos(theta), 0])
            uv.append([-.35 + 1.7*xi/(n-1), -.15 + 1.4*zi/(n-1)])
    points, normals, uv = np.asarray(points), np.asarray(normals), np.asarray(uv)
    faces, material_ids = [], []
    for zi in range(n-1):
        for xi in range(n-1):
            a = zi*n+xi
            faces.extend([[a, a+1, a+n+1], [a, a+n+1, a+n]])
            material_ids.extend([int(xi >= 6)]*2)
    faces = np.asarray(faces, dtype=np.int32)
    mesh = bpy.data.meshes.new('AnalyticCurvedSurface')
    mesh.from_pydata(points.tolist(), [], faces.tolist())
    mesh.update()
    layer = mesh.uv_layers.new(name='SourceUV')
    layer.active_render = True
    for loop in mesh.loops:
        u, v = uv[loop.vertex_index]
        layer.data[loop.index].uv = [u, 1-v]
    for polygon in mesh.polygons:
        polygon.use_smooth = True
        polygon.material_index = material_ids[polygon.index]
    mesh.normals_split_custom_set_from_vertices(normals.tolist())
    obj = bpy.data.objects.new('AnalyticSurface', mesh)
    scene.collection.objects.link(obj)
    images = {}
    for slot in range(2):
        image = bpy.data.images.load(str(out/f'source-{slot}.png'), check_existing=False)
        image.colorspace_settings.name = 'sRGB'
        images[slot] = image
        mat = bpy.data.materials.new(f'AnalyticSource{slot}')
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Roughness'].default_value, bsdf.inputs['Metallic'].default_value = .92, 0
        node = mat.node_tree.nodes.new('ShaderNodeTexImage')
        node.image, node.interpolation, node.extension = image, 'Linear', 'REPEAT'
        mat.node_tree.links.new(node.outputs['Color'], bsdf.inputs['Base Color'])
        mesh.materials.append(mat)
    rig = bpy.data.objects.new('AnalyticRig', bpy.data.armatures.new('AnalyticBones'))
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    base = rig.data.edit_bones.new('Root')
    base.head, base.tail = (0, 0, 0), (0, 0, .07)
    tip = rig.data.edit_bones.new('Tip')
    tip.head, tip.tail, tip.parent = (0, 0, .07), (0, 0, .15), base
    bpy.ops.object.mode_set(mode='OBJECT')
    groups = [obj.vertex_groups.new(name=name) for name in ('Root', 'Tip')]
    values = np.clip((points[:, 2]-.035)/.07, 0, 1)
    for index, value in enumerate(values):
        for group, weight in zip(groups, [1-value, value]):
            if weight > 0:
                group.add([index], float(weight), 'REPLACE')
    obj.modifiers.new('ActualSkin', 'ARMATURE').object = rig
    rig.data.pose_position = 'REST'
    mask = points[:, 2] >= .024
    before = np.asarray([tuple(v.co) for v in mesh.vertices], dtype=np.float32)
    source_normals = np.asarray([tuple(c.vector) for c in mesh.corner_normals], dtype=np.float32)
    cached_uv_before = np.asarray([tuple(c.uv) for c in layer.data], dtype=np.float32)
    source_uv = np.asarray([tuple(c.uv) for c in mesh.uv_layers['SourceUV'].data], dtype=np.float32)
    formula_uv = np.column_stack([uv[:, 0], 1-uv[:, 1]]).astype(np.float32)[faces.ravel()]

    def uv_difference(actual, reference):
        same_shape = actual.shape == reference.shape
        bad = np.argwhere(actual != reference)[:8] if same_shape else np.empty((0, 2), dtype=int)
        difference = np.abs(actual-reference) if same_shape else np.empty(0)
        return {'shape': list(actual.shape), 'reference_shape': list(reference.shape),
                'exact_equal': bool(np.array_equal(actual, reference)),
                'nonfinite_values': int((~np.isfinite(actual)).sum()),
                'max_abs_difference': float(difference.max()) if difference.size and np.isfinite(difference).all() else None,
                'first_bad_indices': bad.tolist()}

    np.savez_compressed(out/'source-uv-before.npz', formula=formula_uv, source_before=source_uv,
                        cached_handle_before=cached_uv_before)
    write(out/'source-uv-before.json', {'fresh_vs_formula': uv_difference(source_uv, formula_uv),
                                      'cached_vs_formula': uv_difference(cached_uv_before, formula_uv)})
    assert np.array_equal(source_uv, formula_uv)
    coats, reports = attach_edge_fringe(obj, rig, mask, images, config,
        {'direction': [0, 0, 1], 'comb': .2}, prepared['groom_flow'], out,
        {'source_export_vertices': 3*len(faces), 'source_export_triangles': len(faces)},
        root_contract=exact_root_texture_contract)
    assert len(coats) == 1 and len(reports) == 1
    assert np.array_equal(before, np.asarray([tuple(v.co) for v in mesh.vertices], dtype=np.float32))
    assert np.array_equal(source_normals, np.asarray([tuple(c.vector) for c in mesh.corner_normals], dtype=np.float32))
    source_uv_after = np.asarray([tuple(c.uv) for c in mesh.uv_layers['SourceUV'].data], dtype=np.float32)
    cached_uv_after = np.asarray([tuple(c.uv) for c in layer.data], dtype=np.float32)
    np.savez_compressed(out/'source-uv-after.npz', formula=formula_uv, source_before=source_uv,
                        source_after=source_uv_after, cached_handle_before=cached_uv_before,
                        cached_handle_after=cached_uv_after)
    write(out/'source-uv-after.json', {'fresh_after_vs_before': uv_difference(source_uv_after, source_uv),
                                     'fresh_after_vs_formula': uv_difference(source_uv_after, formula_uv),
                                     'cached_after_vs_before': uv_difference(cached_uv_after, cached_uv_before),
                                     'cached_after_vs_fresh': uv_difference(cached_uv_after, source_uv_after)})
    assert np.array_equal(source_uv, source_uv_after)
    coat = coats[0]
    correspondence = np.load(out/'fringe-correspondence.npz')
    root_ids = correspondence['root_vertex_ids']
    assert mask[root_ids].all() and len(np.unique(correspondence['material_ids'])) == 2
    assert np.any(correspondence['root_source_uv'] < 0)
    rig.data.pose_position = 'POSE'
    bone = rig.pose.bones['Tip']
    bone.rotation_mode = 'QUATERNION'
    for frame, angle in ((1, 0), (13, math.radians(30)), (25, 0)):
        bone.rotation_quaternion = Quaternion((1, 0, 0), angle)
        bone.keyframe_insert(data_path='rotation_quaternion', frame=frame, group='Tip')
    rig.animation_data.action.name = 'analytic_bend'
    # Exact sampled Blender poses are a separate oracle for the exported animation.
    posed_source, posed_coat, root_errors = [], [], []
    for frame in (1, 7, 13, 19, 25):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        dependency = bpy.context.evaluated_depsgraph_get()
        sampled = []
        for target in (obj, coat):
            evaluated = target.evaluated_get(dependency)
            sample_mesh = evaluated.to_mesh()
            sampled.append(np.asarray([tuple(evaluated.matrix_world @ v.co) for v in sample_mesh.vertices]))
            evaluated.to_mesh_clear()
        posed_source.append(sampled[0])
        posed_coat.append(sampled[1])
        roots = sampled[1].reshape(-1, 4, 3)[:, :2]
        root_errors.append(float(np.linalg.norm(roots-sampled[0][root_ids], axis=2).max()))
    assert max(root_errors) < 1e-6
    assert np.linalg.norm(posed_source[2]-posed_source[0], axis=1).max() > .02
    np.savez_compressed(out/'expected-source.npz', points=before, faces=faces,
        corner_normals=source_normals.reshape(-1, 3, 3), uv_blender=source_uv.reshape(-1, 3, 2),
        raw_uv_topdown=uv, material_ids=np.asarray(material_ids), weights=np.column_stack([1-values, values]),
        positive_mask=mask, times=np.array([0, .25, .5, .75, 1]),
        posed_source=np.asarray(posed_source), posed_coat=np.asarray(posed_coat))
    scene.frame_set(1)
    for target in bpy.context.selected_objects:
        target.select_set(False)
    for target in (obj, coat, rig):
        target.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(filepath=str(out/'model.glb'), export_format='GLB', use_selection=True,
        export_animations=True, export_animation_mode='ACTIONS', export_force_sampling=True,
        export_frame_range=True, export_skins=True, export_normals=True, export_tangents=True,
        export_yup=True, export_morph=False, export_extras=False)
    for image in bpy.data.images:
        if image.has_data:
            image.pack()
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'source.blend'), check_existing=False)
    write(out/'native-result.json', {'status': 'passed_native_checks', 'provenance': 'analytic_fixture',
        'blender_version': bpy.app.version_string, 'source_vertices': len(points), 'source_faces': len(faces),
        'selected_fins': len(root_ids), 'materials_selected': sorted(map(int, np.unique(correspondence['material_ids']))),
        'native_root_attachment_max_error_m': max(root_errors), 'source_arrays_unchanged': True,
        'sampled_frames': [1, 7, 13, 19, 25], 'native_groom_created': False,
        'limits': ['Component geometry/texture/skin fixture only; no avatar or rendered appearance acceptance.']})


def prepare(out):
    import numpy as np
    from PIL import Image
    root = out.parents[1]
    (out/'sources').mkdir()
    filenames = ['scripts/blender_edge_fringe.py', 'scripts/edge_fringe.py', 'scripts/fringe_color_atlas.py',
                 'scripts/groom_flow.py', 'scripts/blender_finish_generated.py', 'scripts/blender_verify.py',
                 'scripts/blender_native_groom.py', 'scripts/motion_samples.py', 'scripts/fur_root_sampling.py',
                 'avatar_assets/runner.py', 'avatar_assets/safety.py']
    records = []
    for name in filenames:
        source = root/name
        target = out/'sources'/source.name
        shutil.copyfile(source, target)
        assert digest(source) == digest(target)
        records.append({'path': name, 'snapshot': str(target.relative_to(out)), 'sha256': digest(target)})
    for name in ('fixture.py', 'verify.py'):
        shutil.copyfile(out/name, out/'sources'/name)
        records.append({'path': str((out/name).relative_to(root)), 'snapshot': 'sources/'+name,
                        'sha256': digest(out/'sources'/name)})
    assert digest(out/'sources/fringe_color_atlas.py') == '127c3aa3d3cbfbc52686527b61ede0a2ba7182cb0314ee37feb8812e4b9ec8d4'
    assert digest(out/'sources/edge_fringe.py') == '24eb6347abccae2a885758aa3f8e49e92dcca26695c2bc55aaf220963c7870b9'
    y, x = np.mgrid[:8, :8]
    images = [np.stack([32+24*x, 24+22*y, 32+8*x+4*y], axis=-1),
              np.stack([200-20*x, 20+10*x+14*y, 210-22*y], axis=-1)]
    for slot, pixels in enumerate(images):
        Image.fromarray(pixels.astype(np.uint8)).save(out/f'source-{slot}.png')
    config = {'profile': 'edge-fringe-v1', 'excluded_materials': [],
              'settings': {'seed': 7, 'min_fins': 24, 'max_fins': 80, 'length_m': .008,
                           'min_spacing_m': .003, 'min_edge_m': .0085, 'max_edge_m': .020, 'comb': .2},
              'design_views': [[0, -1, 0], [.6, -.8, 0], [-.6, -.8, 0]]}
    flow = {'regions': [{'name': 'analytic-flow', 'center': [.5, .5, .5], 'radius': [2, 2, 2],
                         'feather': .5, 'direction': [0, 0, 1], 'comb': .2, 'length_scale': 1}]}
    command = [str(root/'.tools/Blender.app/Contents/MacOS/Blender'), '--background', '--factory-startup',
               '--disable-autoexec', '--python-exit-code', '1', '--threads', '4', '--python',
               str(out/'sources/fixture.py'), '--', '--native', str(out)]
    inputs = [{'path': f'source-{i}.png', 'sha256': digest(out/f'source-{i}.png')} for i in range(2)]
    previous = root/'outputs/fringe-native-adapter-fixture-v1/manifest.json'
    check_previous = '333b5a8363b1a9cd281ea29df28039905497b12a7c36fc5e7a7fcbab899fb39c'
    assert digest(previous) == check_previous
    write(out/'prepared.json', {'schema_version': 1, 'provenance': 'analytic_fixture', 'sources': records, 'inputs': inputs,
        'prior_failure': {'path': str(previous.relative_to(root)), 'sha256': check_previous,
                          'preserved': True, 'changed_hypothesis': 'Reacquire UV layer and prove exact before/after/formula values; cached handle is diagnostic only.'},
        'command': command, 'fringe': config, 'groom_flow': flow, 'provider_calls': 0,
        'limits': {'wall_seconds': 60, 'rss_mb': 1024, 'threads': 4, 'output_mb': 64},
        'acceptance': {'native_root_max_error_m': 1e-6, 'export_pose_max_error_m': 2e-6,
                       'color_tip_p95_max': .02, 'color_tip_p99_max': .05,
                       'normalized_export_normal_max_component_error': .0002,
                       'no_quality_claim': True}})


def launch(out):
    root = out.parents[1]
    sys.path.insert(0, str(root))
    from avatar_assets.runner import run_bounded
    prepared = json.loads((out/'prepared.json').read_text())
    for row in prepared['sources']:
        assert digest(out/row['snapshot']) == row['sha256']
        assert digest(root/row['path']) == row['sha256']
    for row in prepared['inputs']:
        assert digest(out/row['path']) == row['sha256']
    job = run_bounded(prepared['command'], out/'native.log', out, timeout=60, memory_mb=1024, output_mb=64)
    write(out/'native-job.json', job)
    passed = not job.get('failure') and job.get('exit_code') == 0 and job.get('process_exit_verified') is True
    if passed:
        sys.path.insert(0, str(out/'sources'))
        import verify
        try:
            verify.verify(out)
        except Exception:
            passed = False
            (out/'verification-error.txt').write_text(traceback.format_exc())
    write(out/'result.json', {'status': 'passed' if passed else 'failed', 'provenance': 'analytic_fixture',
        'native_exit_verified': job.get('process_exit_verified'), 'elapsed_seconds': job['elapsed_seconds'],
        'sampled_peak_rss_bytes': job['sampled_peak_rss_bytes'], 'provider_calls': 0,
        'no_native_retry': True, 'avatar_or_appearance_accepted': False})
    artifacts = [{'path': str(p.relative_to(out)), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                 for p in sorted(out.rglob('*')) if p.is_file()]
    write(out/'manifest.json', {'schema_version': 1, 'status': 'passed' if passed else 'failed',
        'provenance': 'analytic_fixture', 'artifacts': artifacts})
    print((out/'result.json').read_text())
    return 0 if passed else 1


if __name__ == '__main__':
    args = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    mode = args[0]
    out = Path(args[1]).resolve() if len(args) == 2 else Path(__file__).resolve().parent
    if mode == '--native':
        native(out)
    elif mode == '--prepare':
        prepare(out)
    elif mode == '--run':
        raise SystemExit(launch(out))
    else:
        raise ValueError('Expected explicit prepare, run or native mode')
