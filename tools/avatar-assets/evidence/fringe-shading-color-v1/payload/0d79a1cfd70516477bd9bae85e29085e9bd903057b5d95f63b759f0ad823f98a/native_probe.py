"""One imported-glTF control; never change imported material or normal settings."""
from array import array
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

HERE = Path(__file__).resolve().parent


def write(name, value):
    with (HERE/name).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixels(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.colorspace_settings.name = 'Non-Color'
        assert tuple(image.size) == (128, 128)
        count = len(image.pixels)
        assert count in (128*128*3, 128*128*4)
        values = array('f', [0.])*count
        image.pixels.foreach_get(values)
        return values, count//(128*128)
    finally:
        image.buffers_free()
        bpy.data.images.remove(image)


def main():
    started = time.monotonic()
    prepared = json.loads((HERE/'prepared.json').read_text())
    assert sha(HERE/'fixture-v2.glb') == prepared['fixture_sha256']
    spec = json.loads((HERE/'fixture-spec-v2.json').read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(HERE/'fixture-v2.glb'))
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    imported = []
    for obj in sorted((o for o in scene.objects if o.type == 'MESH'), key=lambda o: o.name):
        matrix = obj.matrix_world.to_3x3().inverted().transposed()
        imported.append({'name': obj.name,
            'materials': [{'name': m.name, 'use_backface_culling': m.use_backface_culling} for m in obj.data.materials],
            'polygon_world_normals': [list((matrix@p.normal).normalized()) for p in obj.data.polygons],
            'corner_world_normals': [list((matrix@n.vector).normalized()) for n in obj.data.corner_normals]})
    assert len(imported) == 4
    write('imported.json', {'blender_version': bpy.app.version_string,
        'build_hash': bpy.app.build_hash.decode(), 'objects': imported,
        'materials_modified': False, 'source_normals_modified': False})
    scene.render.resolution_x = scene.render.resolution_y = 128
    scene.render.resolution_percentage = 100
    scene.render.threads_mode, scene.render.threads = 'FIXED', 4
    scene.render.film_transparent = True
    scene.render.use_persistent_data = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.view_settings.view_transform = 'Standard'
    scene.world = bpy.data.worlds.new('BlackWorld')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0
    camera = bpy.data.objects.new('ProbeCamera', bpy.data.cameras.new('ProbeCamera'))
    scene.collection.objects.link(camera); scene.camera = camera
    camera.data.type, camera.data.ortho_scale = 'ORTHO', 2.6
    light = bpy.data.objects.new('ProbeSun', bpy.data.lights.new('ProbeSun', 'SUN'))
    scene.collection.objects.link(light); light.data.energy, light.data.angle = 3., 0.
    scene.render.engine = 'CYCLES'
    scene.cycles.device, scene.cycles.samples = 'CPU', 4
    scene.cycles.use_denoising = scene.cycles.use_adaptive_sampling = False
    scene.cycles.max_bounces = 1
    bpy.context.view_layer.use_pass_normal = True
    scene.use_nodes = True
    tree = scene.node_tree; tree.nodes.clear()
    layers = tree.nodes.new('CompositorNodeRLayers')
    output = tree.nodes.new('CompositorNodeComposite')
    tree.links.new(layers.outputs['Image'], output.inputs['Image'])
    normals = tree.nodes.new('CompositorNodeOutputFile')
    normals.base_path = str(HERE)
    normals.format.file_format, normals.format.color_mode = 'OPEN_EXR', 'RGB'
    normals.format.color_depth, normals.format.exr_codec = '32', 'ZIP'
    tree.links.new(layers.outputs['Normal'], normals.inputs[0])
    scene.frame_set(1)
    results = []
    for engine in ('CYCLES', 'BLENDER_EEVEE_NEXT'):
        key = 'cycles' if engine == 'CYCLES' else 'eevee'
        try:
            assert time.monotonic()-started < 50, 'Inner deadline before engine'
            scene.render.engine = engine
            eevee = getattr(scene, 'eevee', None)
            if engine != 'CYCLES' and eevee is not None and hasattr(eevee, 'taa_render_samples'):
                eevee.taa_render_samples = 4
            for view, sign in (('front', -1), ('back', 1)):
                assert time.monotonic()-started < 55, 'Inner deadline before render'
                label = key+'-'+view
                camera.location = (0, 3*sign, 0)
                camera.rotation_euler = (-camera.location).to_track_quat('-Z', 'Y').to_euler()
                light.rotation_euler = camera.rotation_euler
                bpy.context.view_layer.update()
                beauty = HERE/(label+'.png')
                assert not beauty.exists()
                scene.render.filepath = str(beauty)
                normals.file_slots[0].path = label+'-normal-'
                bpy.ops.render.render(write_still=True)
                normal_path = HERE/(label+'-normal-0001.exr')
                color, cs = pixels(beauty)
                normal, ns = pixels(normal_path)
                regions = []
                for case in spec['cases']:
                    point = world_to_camera_view(scene, camera, Vector(case['center_blender']))
                    x, y = int(point.x*128), int(point.y*128)
                    assert 5 <= x < 123 and 5 <= y < 123
                    offsets = [yy*128+xx for yy in range(y-4, y+5) for xx in range(x-4, x+5)]
                    visible = [i for i in offsets if color[i*cs+3] > .5]
                    rgb = [sum(color[i*cs+c] for i in visible)/len(visible) for c in range(3)] if visible else None
                    mean_normal = [sum(normal[i*ns+c] for i in visible)/len(visible) for c in range(3)] if visible else None
                    regions.append({'case': case['id'], 'center_pixel_bottom_up': [x,y],
                        'expected_glTF_visible': case['expected_'+view+'_visible'],
                        'visible_pixels': len(visible), 'roi_pixels': len(offsets),
                        'mean_png_noncolor_rgb': rgb, 'mean_engine_Normal_pass': mean_normal})
                row = {'engine': engine, 'view': view, 'regions': regions,
                    'beauty': beauty.name, 'beauty_sha256': sha(beauty),
                    'normal_pass': normal_path.name, 'normal_sha256': sha(normal_path),
                    'cycles_samples': 4 if engine == 'CYCLES' else None,
                    'eevee_samples': getattr(eevee, 'taa_render_samples', None) if engine != 'CYCLES' else None,
                    'elapsed_seconds': time.monotonic()-started}
                write(label+'.json', row); results.append(row)
                print(json.dumps({'checkpoint': label, 'visible_pixels': [r['visible_pixels'] for r in regions]}), flush=True)
                del color, normal
                result = bpy.data.images.get('Render Result')
                if result: result.buffers_free()
                gc.collect()
        except Exception as error:
            write(key+'-failure.json', {'engine': engine, 'error': type(error).__name__+': '+str(error),
                'completed_render_checkpoints': len(results), 'automatic_retry': False})
            break
    write('native-result.json', {'status': 'renderer_control_complete' if len(results)==4 else 'partial_control_preserved',
        'render_checkpoints': len(results), 'elapsed_seconds': time.monotonic()-started,
        'materials_modified': False, 'source_normals_modified': False, 'fixture_unchanged': sha(HERE/'fixture-v2.glb')==prepared['fixture_sha256'],
        'normal_scope': 'Engine Normal render pass sampled at visible region interiors; imported explicit corner normals are separately recorded. Not a physical or avatar-quality claim.',
        'native_invocations': 1, 'provider_ai_role_calls': 0, 'quality_approved': False})


if __name__ == '__main__':
    main()
