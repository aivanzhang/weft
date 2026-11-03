"""Bone-heat re-skin of the delivered Studio body via a watertight proxy; weights-only export.

The delivered body is too degenerate for bone heat directly, so a voxel-remeshed proxy is
heat-skinned against the existing armature and its weights transfer back to the original
vertices by nearest-face interpolation. Geometry, rig, materials and the GLB itself are
never written here; the caller splices the weights into a byte-identical derivative and
validates with pose probes.
"""

import json
import sys
from pathlib import Path

import bpy

ARGS = sys.argv[sys.argv.index("--") + 1:]
SOURCE, OUT = Path(ARGS[0]), Path(ARGS[1])
BODY_MESH_NAME = ARGS[2] if len(ARGS) > 2 else "Mesh_0"
VOXEL_DIVISIONS = float(ARGS[3]) if len(ARGS) > 3 else 150.0

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(SOURCE))

armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
assert len(armatures) == 1, [o.name for o in bpy.data.objects]
armature = armatures[0]
meshes = {o.name: o for o in bpy.data.objects if o.type == "MESH"}
body = next(o for name, o in meshes.items() if name.startswith(BODY_MESH_NAME))

report = {"blender": bpy.app.version_string, "source": str(SOURCE),
          "objects": sorted(meshes), "body_object": body.name,
          "armature": armature.name, "bones": [b.name for b in armature.data.bones],
          "body_vertices": len(body.data.vertices)}

def select_only(*objects):
    for o in bpy.data.objects:
        o.select_set(False)
    for o in objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objects[-1]

# Watertight proxy: voxel remesh guarantees manifold shells so bone heat can solve.
select_only(body)
bpy.ops.object.duplicate()
proxy = bpy.context.view_layer.objects.active
proxy.name = "HeatProxy"
proxy.vertex_groups.clear()
for modifier in list(proxy.modifiers):
    proxy.modifiers.remove(modifier)
bounds = [proxy.dimensions[i] for i in range(3)]
voxel = max(bounds) / VOXEL_DIVISIONS
remesh = proxy.modifiers.new("Voxel", "REMESH")
remesh.mode = "VOXEL"
remesh.voxel_size = voxel
select_only(proxy)
bpy.ops.object.modifier_apply(modifier=remesh.name)
report["proxy_vertices"] = len(proxy.data.vertices)
report["voxel_size"] = voxel

select_only(proxy)
bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
select_only(proxy, armature)
bpy.ops.object.parent_set(type="ARMATURE_AUTO")
proxy_unweighted = sum(1 for v in proxy.data.vertices if not any(g.weight > 1e-6 for g in v.groups))
report["proxy_unweighted"] = proxy_unweighted
assert proxy_unweighted < len(proxy.data.vertices) // 10, "Bone heat failed on the proxy too"

# Transfer proxy weights onto the original body vertices (operator: active -> selected,
# and destination groups must already exist by name).
body.vertex_groups.clear()
for group in proxy.vertex_groups:
    body.vertex_groups.new(name=group.name)
select_only(body, proxy)
bpy.ops.object.data_transfer(use_reverse_transfer=False, data_type="VGROUP_WEIGHTS",
                             vert_mapping="POLYINTERP_NEAREST", layers_select_src="ALL",
                             layers_select_dst="NAME")
groups = {g.index: g.name for g in body.vertex_groups}
report["vertex_groups"] = sorted(groups.values())
matrix = body.matrix_world
rows = []
unweighted = 0
for v in body.data.vertices:
    world = matrix @ v.co
    weights = {groups[g.group]: g.weight for g in v.groups if g.weight > 1e-6 and g.group in groups}
    if not weights:
        unweighted += 1
    rows.append({"p": [world.x, world.y, world.z], "w": weights})
report["unweighted_vertices"] = unweighted

OUT.write_text(json.dumps({"report": report, "vertices": rows}) + "\n")
print("RESKIN_OK", len(rows), "unweighted", unweighted, "proxy_unweighted", proxy_unweighted)
