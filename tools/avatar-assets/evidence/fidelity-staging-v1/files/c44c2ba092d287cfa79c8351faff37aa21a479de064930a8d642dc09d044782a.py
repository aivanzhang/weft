"""Tiny native custom-normal storage comparison; no render or source mutation."""
import hashlib
import json
from pathlib import Path
import sys

import bpy

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE/'config.json').read_text())


def save(name, data):
    with (HERE/name).open('x') as handle:
        json.dump(data,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def observation(obj, case, method):
    mesh = obj.data
    attr = mesh.attributes.get('custom_normal')
    return {'name':obj.name,'case':case['id'],'method':method,
        'positions':[list(v.co) for v in mesh.vertices],
        'corner_normals':[list(n.vector) for n in mesh.corner_normals],
        'loop_vertices':[loop.vertex_index for loop in mesh.loops],
        'polygons_smooth':[p.use_smooth for p in mesh.polygons],
        'attribute':{'name':attr.name,'domain':attr.domain,'data_type':attr.data_type} if attr else None,
        'expected_normals':case['normals_blender']}


def main():
    assert not (HERE/'model.glb').exists() and not (HERE/'source.blend').exists()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.threads_mode='FIXED'
    bpy.context.scene.render.threads=4
    material=bpy.data.materials.new('DiffuseControl')
    material.use_nodes=True
    shader=material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value=(.5,.5,.5,1)
    shader.inputs['Metallic'].default_value=0
    shader.inputs['Roughness'].default_value=1
    objects=[]; before=[]
    for case in CONFIG['cases']:
        for method in CONFIG['methods']:
            name=case['id']+'__'+method
            mesh=bpy.data.meshes.new(name)
            mesh.from_pydata(case['positions_blender_m'],[],[(0,1,2)])
            mesh.update()
            # Read first so the float-attribute path also tests cache invalidation.
            geometric=[list(n.vector) for n in mesh.corner_normals]
            if method=='legacy-short2':
                mesh.normals_split_custom_set_from_vertices(case['normals_blender'])
            else:
                domain='POINT' if method=='point-float-vector' else 'CORNER'
                attr=mesh.attributes.new(name='custom_normal',type='FLOAT_VECTOR',domain=domain)
                values=[value for normal in case['normals_blender'] for value in normal]
                attr.data.foreach_set('vector',values)
            mesh.update()
            mesh.materials.append(material)
            obj=bpy.data.objects.new(name,mesh)
            bpy.context.scene.collection.objects.link(obj)
            objects.append((obj,case,method))
            row=observation(obj,case,method);row['geometric_normals_before_setting']=geometric
            before.append(row)
    save('native-before-export.json',{'observations':before,'blender':bpy.app.version_string,
        'build_hash':bpy.app.build_hash.decode(),'config_sha256':sha(HERE/'config.json'),
        'objects':len(objects),'triangles':len(objects),'rendering_performed':False})
    bpy.ops.wm.save_as_mainfile(filepath=str(HERE/'source.blend'),check_existing=False)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.gltf(filepath=str(HERE/'model.glb'),export_format='GLB',use_selection=True,
        export_yup=True,export_normals=True,export_tangents=False,export_materials='EXPORT',
        export_animations=False,export_skins=False,export_morph=False,export_cameras=False,
        export_lights=False,export_draco_mesh_compression_enable=False,export_attributes=False)
    import io_scene_gltf2
    save('native-after-export.json',{'observations':[observation(obj,case,method) for obj,case,method in objects],
        'blender':bpy.app.version_string,'build_hash':bpy.app.build_hash.decode(),
        'exporter_version':io_scene_gltf2.get_version_string(),'model_sha256':sha(HERE/'model.glb'),
        'blend_sha256':sha(HERE/'source.blend'),'rendering_performed':False})
    print(json.dumps({'status':'native_fixture_exported','objects':len(objects),'renders':0}))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        save('native-failure.json',{'type':type(error).__name__,'message':str(error),'accepted':False})
        raise
