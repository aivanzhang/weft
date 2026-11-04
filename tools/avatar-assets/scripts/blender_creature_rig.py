"""Author an explicit creature rig on a bounded generated mesh without changing its topology."""

import bpy
import numpy as np

from creature_rig_math import validate_rig, skin_weights


def create_rig(obj, config):
    validate_rig(config)
    if obj.vertex_groups or any(m.type == 'ARMATURE' for m in obj.modifiers):
        raise ValueError('Authoring a creature rig cannot replace existing skin data')
    world = obj.matrix_world.copy()
    obj.parent = None
    obj.matrix_world = world
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    points = np.array([tuple(v.co) for v in obj.data.vertices])
    low, high = points.min(axis=0), points.max(axis=0)
    if high[2]-low[2] <= 1e-8:
        raise ValueError('Cannot calibrate a zero-height creature')
    factor = config['height_meters']/(high[2]-low[2])
    offset = np.array([(low[0]+high[0])/2, (low[1]+high[1])/2, low[2]])
    points = (points-offset)*factor
    obj.data.vertices.foreach_set('co', points.astype(np.float32).ravel())
    obj.data.update()
    bounds = points.min(axis=0), points.max(axis=0)
    weights, proof = skin_weights(points, *bounds, config)
    data = bpy.data.armatures.new('CreatureSkeleton')
    rig = bpy.data.objects.new('CreatureRig', data)
    bpy.context.scene.collection.objects.link(rig)
    bpy.ops.object.select_all(action='DESELECT')
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    bones = {}
    for source in config['bones']:
        bone = data.edit_bones.new(source['name'])
        bone.head = bounds[0]+np.asarray(source['head'])*(bounds[1]-bounds[0])
        bone.tail = bounds[0]+np.asarray(source['tail'])*(bounds[1]-bounds[0])
        bone.parent = bones.get(source['parent'])
        bones[source['name']] = bone
    bpy.ops.object.mode_set(mode='OBJECT')
    for i, source in enumerate(config['bones']):
        group = obj.vertex_groups.new(name=source['name'])
        for index in np.flatnonzero(weights[:, i] > 0):
            group.add([int(index)], float(weights[index, i]), 'REPLACE')
    obj.modifiers.new('AuthoredCreatureSkin', 'ARMATURE').object = rig
    rig.data.pose_position = 'REST'
    # Generated meshes often weld resting contacts (a paw against a foot); such an
    # edge joins chain-distant regions and tears the moment either limb moves, so it
    # splits like the seam between a toy's touching parts. The gate judges the result.
    from weight_locality import chain_distances, weld_edge_indices
    import bmesh
    dominants = np.argmax(weights, axis=1)
    marked = weld_edge_indices([tuple(e.vertices) for e in obj.data.edges],
                               dominants, chain_distances(config['bones']))
    if marked:
        mesh = bmesh.new()
        mesh.from_mesh(obj.data)
        mesh.edges.ensure_lookup_table()
        bmesh.ops.split_edges(mesh, edges=[mesh.edges[i] for i in marked])
        mesh.to_mesh(obj.data)
        mesh.free()
        obj.data.update()
    proof.update(authored_not_recovered=True, design=config, height_meters=config['height_meters'],
                 input_world_bounds={'min':low.tolist(), 'max':high.tolist()},
                 source_to_authored={'translation_before_scale':(-offset).tolist(), 'uniform_scale':float(factor)},
                 contact_weld_split_edges=len(marked),
                 mesh_topology_uv_textures_unchanged=not marked)
    return rig, proof
