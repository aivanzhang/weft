"""Prune anatomically distant skin influences from the original authored weights.

The authored panda weights pose coherently under the ten-role retarget except for a
few small vertex clusters blended to far-away bones (a face patch following a leg),
which sweep long triangle blades across the body in motion. Heat re-skins replace the
whole solution and collapse under partial driving, so instead keep every authored
weight whose bone is plausibly local and drop only influences whose bone segment is
far from the vertex relative to its nearest influence; empty results rebind rigidly
to the nearest bone. Geometry, rig, materials and textures are unchanged.
"""
import json
import sys
from pathlib import Path

import bpy

ARGS = sys.argv[sys.argv.index("--") + 1:]
SOURCE, OUT_GLB, OUT_REPORT = Path(ARGS[0]), Path(ARGS[1]), Path(ARGS[2])
RELATIVE_FACTOR = 2.5   # influence dropped when farther than this times the nearest influence
ABSOLUTE_FLOOR = 0.12   # of the mesh bounding-sphere diameter; closer bones are never dropped
# Distal bones travel farthest under retargeting, so a stray influence from one drags
# geometry across the whole pose; they only deserve influence in a tight local tube.
DISTAL_MARKERS = ("hand", "foot", "_lower")
DISTAL_FACTOR = 1.6
DISTAL_FLOOR = 0.05
# A vertex split between a limb chain and the torso stretches into a wide membrane
# when the limb travels; outside a narrow blend margin, the dominant side owns it.
LIMB_PREFIXES = ("arm_", "hand_", "leg_", "foot_")
CONTRAST_LOW = 0.35
CONTRAST_HIGH = 0.65

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(SOURCE), import_pack_images=True)

armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
assert len(armatures) == 1
armature = armatures[0]
bones = {b.name: (armature.matrix_world @ b.head_local, armature.matrix_world @ b.tail_local)
         for b in armature.data.bones}


def segment_distance(point, head, tail):
    direction = tail - head
    length2 = direction.length_squared
    if length2 < 1e-12:
        return (point - head).length
    t = max(0.0, min(1.0, (point - head).dot(direction) / length2))
    return (point - head - direction * t).length


report = {"blender": bpy.app.version_string, "objects": {}}
targets = [o for o in bpy.data.objects
           if o.type == "MESH" and (o.name.startswith("Mesh_0") or o.name.startswith("PortableCoat"))]
for obj in targets:
    groups = {g.index: g.name for g in obj.vertex_groups}
    by_name = {g.name: g for g in obj.vertex_groups}
    matrix = obj.matrix_world
    diameter = max(obj.dimensions)
    pruned = rebound = 0
    for v in obj.data.vertices:
        world = matrix @ v.co
        influences = [(groups[g.group], g.weight) for g in v.groups
                      if g.weight > 1e-6 and groups.get(g.group) in bones]
        if not influences:
            continue
        distances = {name: segment_distance(world, *bones[name]) for name, _ in influences}
        nearest = min(distances.values())
        keep = []
        for name, weight in influences:
            distal = any(marker in name for marker in DISTAL_MARKERS)
            factor = DISTAL_FACTOR if distal else RELATIVE_FACTOR
            limit = max(factor * nearest, diameter * (DISTAL_FLOOR if distal else ABSOLUTE_FLOOR))
            if distances[name] <= limit:
                keep.append((name, weight))
        limb = sum(w for name, w in keep if name.startswith(LIMB_PREFIXES))
        total = sum(w for _, w in keep)
        if keep and 0 < limb < total:
            share = limb / total
            if share < CONTRAST_LOW:
                keep = [(n, w) for n, w in keep if not n.startswith(LIMB_PREFIXES)]
            elif share > CONTRAST_HIGH:
                keep = [(n, w) for n, w in keep if n.startswith(LIMB_PREFIXES)]
        if len(keep) == len(influences):
            continue
        for name, _ in influences:
            by_name[name].remove([v.index])
        if keep:
            total = sum(w for _, w in keep)
            for name, weight in keep:
                by_name[name].add([v.index], weight / total, "REPLACE")
            pruned += 1
        else:
            name = min(bones, key=lambda n: segment_distance(world, *bones[n]))
            if name not in by_name:
                by_name[name] = obj.vertex_groups.new(name=name)
            by_name[name].add([v.index], 1.0, "REPLACE")
            rebound += 1
    report["objects"][obj.name] = {"vertices": len(obj.data.vertices),
                                   "pruned": pruned, "rebound": rebound}

meshes = [o for o in bpy.data.objects if o.type == "MESH" and o in targets]
for o in bpy.data.objects:
    o.select_set(False)
for o in [armature, *meshes]:
    o.select_set(True)
bpy.context.view_layer.objects.active = armature
bpy.ops.export_scene.gltf(filepath=str(OUT_GLB), export_format="GLB", export_animations=False,
    use_selection=True, export_skins=True, export_all_influences=False, export_def_bones=False,
    export_tangents=False, export_attributes=False, export_yup=True, export_image_format="AUTO",
    export_extras=False)
OUT_REPORT.write_text(json.dumps(report, indent=2) + "\n")
print("HYGIENE_OK", json.dumps(report))
