"""Independently reload a GLB, sample native deformation, and render three views."""

import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


MAX_BYTES = 64 * 1024 * 1024
MAX_VERTICES = 100_000
MAX_PROBES = 128
MAX_ACTIONS = 16


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def source_header(path):
    require(path.is_file() and not path.is_symlink(), "Require a regular local GLB")
    size = path.stat().st_size
    require(28 <= size <= MAX_BYTES, "GLB size outside 28 bytes–64 MiB")
    with path.open("rb") as handle:
        header = handle.read(20)
        magic, version, total, length, kind = struct.unpack("<4sIIII", header)
        require(magic == b"glTF" and version == 2 and total == size and kind == 0x4E4F534A,
                "Invalid GLB header")
        require(0 < length <= 2*1024*1024 and 20+length < size, "GLB JSON size outside bounds")
        doc = json.loads(handle.read(length))
    pending = [doc]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            require("uri" not in value, "External/data URIs are unsupported in independent reload")
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    require(doc.get("asset", {}).get("version") == "2.0", "Require glTF 2.0")
    require(not doc.get("extensionsRequired") and not doc.get("extensionsUsed"),
            "Independent verifier expects standard uncompressed glTF")
    for key, limit in (("nodes", 256), ("meshes", 64), ("skins", 16), ("animations", MAX_ACTIONS)):
        require(isinstance(doc.get(key, []), list) and len(doc.get(key, [])) <= limit,
                f"Too many {key}")
    return doc


def hash_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def imported_bindings(objects, actions):
    bindings = {action.name: [] for action in actions}
    owners = []
    for obj in objects:
        owners.append(obj)
        if obj.data:
            owners.append(obj.data)
            if hasattr(obj.data, "shape_keys") and obj.data.shape_keys:
                owners.append(obj.data.shape_keys)
    seen = set()
    for owner in owners:
        data = owner.animation_data
        if not data:
            continue
        candidates = []
        if data.action:
            candidates.append((data.action, data.action_slot))
        for track in data.nla_tracks:
            for strip in track.strips:
                if strip.action:
                    candidates.append((strip.action, strip.action_slot))
        for action, slot in candidates:
            key = (owner.as_pointer(), action.as_pointer(), slot.handle if slot else None)
            if key not in seen and action.name in bindings:
                seen.add(key)
                bindings[action.name].append((owner, slot))
    return bindings, owners


def clear_actions(owners, armatures):
    for owner in owners:
        if owner.animation_data:
            owner.animation_data.action = None
            owner.animation_data.use_nla = False
            for track in owner.animation_data.nla_tracks:
                track.mute = True
    for rig in armatures:
        for bone in rig.pose.bones:
            bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def bind_action(action, bindings, owners, armatures):
    clear_actions(owners, armatures)
    require(bindings[action.name], f"No imported owner/slot found for action {action.name}")
    for owner, slot in bindings[action.name]:
        data = owner.animation_data_create()
        data.action = action
        if slot:
            data.action_slot = slot
        data.action_blend_type = "REPLACE"
        data.action_influence = 1
    bpy.context.view_layer.update()


def inspect_geometry(meshes, armatures):
    require(meshes and armatures, "Imported GLB must contain mesh and armature objects")
    vertices = sum(len(obj.data.vertices) for obj in meshes)
    require(0 < vertices <= MAX_VERTICES, "Imported vertex count outside bounds")
    require(sum(len(obj.data.bones) for obj in armatures) <= 256, "Too many imported bones")
    groups, records, world_points = [], [], []
    for obj in meshes:
        require(obj.data.materials and all(obj.data.materials), f"Missing mesh materials: {obj.name}")
        require(obj.data.polygons, f"Empty mesh: {obj.name}")
        skin_modifiers = [m for m in obj.modifiers if m.type == "ARMATURE" and m.object in armatures]
        require(len(skin_modifiers) <= 1, "Multiple armature modifiers are unsupported")
        weighted, unweighted = 0, 0
        if skin_modifiers:
            bone_names = set(skin_modifiers[0].object.data.bones.keys())
            group_names = {group.index: group.name for group in obj.vertex_groups}
            for vertex in obj.data.vertices:
                weights = [g.weight for g in vertex.groups if group_names.get(g.group) in bone_names]
                require(all(math.isfinite(w) and w >= 0 for w in weights), "Invalid imported skin weight")
                if sum(weights) > 0:
                    require(abs(sum(weights)-1) < .001, "Imported skin weights do not sum to one")
                    weighted += 1
                else:
                    unweighted += 1
            require(unweighted == 0, f"Unweighted vertices in skinned mesh: {obj.name}")
        material_vertices = {}
        for polygon in obj.data.polygons:
            require(0 <= polygon.material_index < len(obj.data.materials), "Invalid imported material index")
            material_vertices.setdefault(polygon.material_index, set()).update(polygon.vertices)
        for material_index, indices in sorted(material_vertices.items()):
            groups.append((obj, material_index, sorted(indices)))
        world_points.extend(obj.matrix_world @ v.co for v in obj.data.vertices)
        records.append({"name": obj.name, "vertices": len(obj.data.vertices),
                        "polygons": len(obj.data.polygons),
                        "materials": [m.name for m in obj.data.materials],
                        "armature": skin_modifiers[0].object.name if skin_modifiers else None,
                        "weighted_vertices": weighted, "unweighted_vertices": unweighted})
    require(any(r["weighted_vertices"] for r in records), "No skin-weighted imported geometry")
    require(len(groups) <= MAX_PROBES, "More material subsets than the 128-probe verification budget")
    # Allocate probes across every material subset before adding denser samples.
    counts = [1] * len(groups)
    while sum(counts) < MAX_PROBES:
        expanded = False
        for i, (_, _, indices) in enumerate(groups):
            if counts[i] < len(indices) and sum(counts) < MAX_PROBES:
                counts[i] += 1
                expanded = True
        if not expanded:
            break
    probes, coverage = {}, []
    for (obj, material_index, indices), count in zip(groups, counts):
        chosen = ([indices[len(indices)//2]] if count == 1 else
                  [indices[round(i*(len(indices)-1)/(count-1))] for i in range(count)])
        probes.setdefault(obj.name, set()).update(chosen)
        coverage.append({"object": obj.name, "material": obj.data.materials[material_index].name,
                         "material_index": material_index, "probe_vertices": chosen})
    probes = {name: sorted(indices) for name, indices in probes.items()}
    low = Vector(tuple(min(point[axis] for point in world_points) for axis in range(3)))
    high = Vector(tuple(max(point[axis] for point in world_points) for axis in range(3)))
    dimensions = high-low
    require(all(math.isfinite(x) and x > 1e-5 for x in dimensions), "Geometry lacks finite 3D extent")
    return records, probes, coverage, low, high


def evaluated_points(meshes, probes):
    graph = bpy.context.evaluated_depsgraph_get()
    result = {}
    for obj in meshes:
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            require(len(mesh.vertices) == len(obj.data.vertices), "Evaluated topology unexpectedly changed")
            for index in probes[obj.name]:
                point = evaluated.matrix_world @ mesh.vertices[index].co
                require(all(math.isfinite(x) for x in point), "Non-finite native deformation")
                result[(obj.name, index)] = point.copy()
        finally:
            evaluated.to_mesh_clear()
    return result


def set_frame(frame):
    integer = math.floor(frame)
    bpy.context.scene.frame_set(integer, subframe=frame-integer)
    bpy.context.view_layer.update()


def action_evidence(actions, bindings, owners, armatures, meshes, probes):
    result = []
    for action in actions:
        bind_action(action, bindings, owners, armatures)
        start, end = map(float, action.frame_range)
        require(math.isfinite(start) and math.isfinite(end) and end > start, f"Static action: {action.name}")
        fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
        require((end-start)/fps <= 120, "Imported action exceeds 120 seconds")
        # Off-quarter samples avoid aliasing repetitive motions whose quarters are all rest poses.
        frames = [start+(end-start)*fraction for fraction in (0, .137, .25, .367, .5, .613, .75, .887, 1)]
        first, first_bones, maximum = None, None, 0.
        by_object, moved_bones = {obj.name: 0. for obj in meshes}, set()
        for frame in frames:
            set_frame(frame)
            points = evaluated_points(meshes, probes)
            bone_matrices = {(rig.name, bone.name): tuple(value for row in bone.matrix for value in row)
                             for rig in armatures for bone in rig.pose.bones}
            if first is None:
                first = points
                first_bones = bone_matrices
            for key, values in bone_matrices.items():
                require(all(math.isfinite(value) for value in values), "Non-finite native bone transform")
                if max(abs(a-b) for a, b in zip(values, first_bones[key])) > 1e-7:
                    moved_bones.add("/".join(key))
            for key, point in points.items():
                distance = (point-first[key]).length
                maximum = max(maximum, distance)
                by_object[key[0]] = max(by_object[key[0]], distance)
        result.append({"name": action.name, "frame_range": [start, end],
                       "sample_frames": frames, "duration_seconds": (end-start)/fps,
                       "max_sampled_displacement_m": maximum,
                       "displacement_by_mesh_m": by_object,
                       "moving_pose_bones": sorted(moved_bones),
                       "owners": [{"name": owner.name, "slot": slot.identifier if slot else None}
                                  for owner, slot in bindings[action.name]],
                       "motion_observed": maximum > 1e-6 and bool(moved_bones), "loop": "not_claimed"})
    return result


def render_views(out, low, high):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 8
    scene.cycles.use_denoising = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.resolution_x = scene.render.resolution_y = 256
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.world = bpy.data.worlds.new("IndependentVerificationWorld")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.93, .95, .98, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = .65
    center = (low+high)*.5
    extent = max(high-low)
    camera_data = bpy.data.cameras.new("IndependentVerificationCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = (high-low).length*1.18
    camera_data.clip_start = max(.0001, extent/1000)
    camera_data.clip_end = max(100, extent*20)
    camera = bpy.data.objects.new("IndependentVerificationCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    for name, offset, power in (("Key", (2, -3, 4), 450), ("Fill", (-3, 2, 3), 300)):
        data = bpy.data.lights.new("Independent"+name, "AREA")
        data.energy, data.shape, data.size = power*extent*extent, "DISK", extent*2
        light = bpy.data.objects.new("Independent"+name, data)
        scene.collection.objects.link(light)
        light.location = center + Vector(offset)*extent
        light.rotation_euler = (center-light.location).to_track_quat("-Z", "Y").to_euler()
    results = []
    for name, offset in (("front", (0, -3, .3)), ("back", (0, 3, .3)), ("side", (3, 0, .3))):
        camera.location = center + Vector(offset)*extent
        camera.rotation_euler = (center-camera.location).to_track_quat("-Z", "Y").to_euler()
        path = out/f"independent-{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        require(path.is_file() and path.stat().st_size > 100, f"Missing {name} proof render")
        results.append({"view": name, "path": path.name, "sha256": hash_file(path),
                        "camera_position_blender_z_up": list(camera.location),
                        "camera_target_blender_z_up": list(center)})
    return results


def verify(model, output):
    started = time.monotonic()
    source, out = Path(model).resolve(), Path(output).resolve()
    require(out.is_dir(), "Caller must create a fresh independent verification output directory")
    for name in ("independent_reload.json", "independent-front.png", "independent-back.png", "independent-side.png"):
        require(not (out/name).exists(), "Refusing to overwrite independent verification evidence")
    report = {"schema_version": 1, "status": "failed", "source": str(source),
              "tool": "Blender built-in glTF importer and evaluated dependency graph",
              "verifier_sha256": hash_file(Path(__file__).resolve()),
              "blender_version": bpy.app.version_string, "weft_runtime_verified": False,
              "limits": {"source_bytes": MAX_BYTES, "vertices": MAX_VERTICES,
                         "probe_vertices": MAX_PROBES, "actions": MAX_ACTIONS,
                         "action_time_samples": 9, "render_resolution": 256,
                         "render_samples": 8, "render_threads": 4},
              "scope": "Native GLB reload, material/rig inspection, sampled deformation and three-view renders; not physical simulation or exhaustive motion review"}
    try:
        doc = source_header(source)
        report["source_sha256"] = hash_file(source)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(source))
        objects = list(bpy.context.scene.objects)
        armatures = sorted((o for o in objects if o.type == "ARMATURE"), key=lambda o: o.name)
        custom_shapes = {bone.custom_shape for rig in armatures for bone in rig.pose.bones if bone.custom_shape}
        meshes = sorted((o for o in objects if o.type == "MESH" and o not in custom_shapes), key=lambda o: o.name)
        report["excluded_importer_display_helpers"] = sorted(obj.name for obj in custom_shapes)
        actions = sorted(bpy.data.actions, key=lambda a: a.name)
        require(0 < len(actions) <= MAX_ACTIONS, "Require 1–16 imported skeletal actions")
        require(len(actions) == len(doc.get("animations", [])), "Imported action count differs from GLB animation count")
        bindings, owners = imported_bindings(objects, actions)
        clear_actions(owners, armatures)
        records, probes, coverage, low, high = inspect_geometry(meshes, armatures)
        report["meshes"] = records
        report["probe_coverage"] = coverage
        report["probe_vertices"] = sum(len(indices) for indices in probes.values())
        report["armatures"] = [{"name": rig.name, "bones": [{"name": b.name,
                                 "parent": b.parent.name if b.parent else None,
                                 "head_local": list(b.head_local), "tail_local": list(b.tail_local)}
                                 for b in rig.data.bones]} for rig in armatures]
        report["materials"] = [{"name": material.name, "uses_nodes": material.use_nodes,
                                "diffuse_rgba": list(material.diffuse_color)} for material in bpy.data.materials]
        report["bounds_blender_z_up_m"] = {"min": list(low), "max": list(high), "dimensions": list(high-low)}
        report["actions"] = action_evidence(actions, bindings, owners, armatures, meshes, probes)
        require(all(action["motion_observed"] for action in report["actions"]),
                "No native skeletal/geometry movement observed for: " +
                ", ".join(a["name"] for a in report["actions"] if not a["motion_observed"]))
        clear_actions(owners, armatures)
        set_frame(0)
        report["renders"] = render_views(out, low, high)
        require(hash_file(source) == report["source_sha256"], "Source GLB changed during verification")
        report["source_unchanged"] = True
        report["status"] = "passed"
        report["native_loading_verified"] = True
        report["sampled_animation_verified"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.monotonic()-started
        write_json(out/"independent_reload.json", report)
    return report


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--")+1:]
    require(len(arguments) == 2, "Usage: blender_verify.py -- MODEL.glb OUTPUT_DIR")
    verify(arguments[0], arguments[1])
