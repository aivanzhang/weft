"""Rest-space native-hair combing using Blender's supported original-key setter.

Native hair remains parented to the same skinned emitter; this is not a GLB groom
conversion or a hair-dynamics simulation. Run the bounded analytic probe first.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))

import bpy
from mathutils import Vector

SOURCE_API = "https://github.com/blender/blender/blob/v4.5.3/source/blender/makesrna/intern/rna_particle.cc"
MAX_PARENTS, MAX_KEYS, MAX_SECONDS = 10000, 150000, 20


def require(value, message):
    if not value:
        raise ValueError(message)


def _evaluated_system(groom):
    evaluated = groom.evaluated_get(bpy.context.evaluated_depsgraph_get())
    modifiers = [modifier for modifier in evaluated.modifiers if modifier.type == "PARTICLE_SYSTEM"]
    require(len(modifiers) == 1 and len(evaluated.particle_systems) == 1, "Use one native particle-hair system per emitter")
    return evaluated, modifiers[0], evaluated.particle_systems[0]


def _initialize_original_keys(groom):
    """Particle edit creates persistent original keys from the evaluated groom."""
    require(bpy.context.mode == "OBJECT", "Initialize native keys only from Object mode")
    original = groom.particle_systems[0]
    count = len(_evaluated_system(groom)[2].particles)
    require(0 < count <= MAX_PARENTS, "Evaluated native parent count is outside the comb budget")
    if len(original.particles) == count:
        return {"method":"existing_original_keys","parents":count}
    require(not original.particles, "Original and evaluated parent counts disagree")
    active, selection = bpy.context.view_layer.objects.active, list(bpy.context.selected_objects)
    try:
        bpy.ops.object.select_all(action="DESELECT")
        groom.select_set(True)
        bpy.context.view_layer.objects.active = groom
        require(bpy.ops.object.mode_set.poll(), "Particle-edit initialization has no valid object context")
        require(bpy.ops.object.mode_set(mode="PARTICLE_EDIT") == {"FINISHED"}, "Could not enter Particle Edit mode")
        require(groom.mode == "PARTICLE_EDIT", "Particle Edit mode did not become active")
        require(bpy.ops.object.mode_set(mode="OBJECT") == {"FINISHED"}, "Could not leave Particle Edit mode")
        bpy.context.view_layer.update()
        require(len(original.particles) == count, "Particle Edit did not create persistent original hair keys")
    finally:
        if groom.mode == "PARTICLE_EDIT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        for obj in selection:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = active
    return {"method":"Particle Edit enter/exit on owned emitter","parents":count}


def _original_coordinates(groom):
    evaluated, modifier, system = _evaluated_system(groom)
    original = groom.particle_systems[0]
    require(0 < len(original.particles) == len(system.particles) <= MAX_PARENTS,
            "Original and evaluated hair parents must be initialized and matched")
    rows, key_count = [], 0
    for index, particle in enumerate(original.particles):
        require(2 <= len(particle.hair_keys) <= 33, "Use two to thirty-three stored keys per strand")
        key_count += len(particle.hair_keys)
        require(key_count <= MAX_KEYS, "Native comb key budget exceeded")
        evaluated_particle = system.particles[index]
        rows.append([Vector(key.co_object(evaluated,modifier,evaluated_particle)) for key in particle.hair_keys])
    return evaluated, modifier, system, rows


def _settings_signature(system):
    names = ("type","count","hair_step","hair_length","child_type","rendered_child_count",
             "child_percent","use_modifier_stack","use_regrow_hair","clump_factor","roughness_1",
             "roughness_2","radius_scale","root_radius","tip_radius","material")
    return {"settings":{name:getattr(system.settings,name) for name in names},
            "seed":system.seed,"child_seed":system.child_seed,"hair_dynamics":system.use_hair_dynamics}


def comb_native_groom(groom, direction, amount=.65, flow=None):
    """Comb once in rest axes; direction components must be in [-1,1], amount in [0,.95]."""
    from blender_native_groom import _skin_signature

    started = time.monotonic()
    require(groom.type == "MESH" and len(groom.particle_systems) == 1, "Expected a native hair emitter mesh")
    require(isinstance(direction,(list,tuple,Vector)) and len(direction) == 3
            and all(type(value) in (int,float) and math.isfinite(value) and -1 <= value <= 1 for value in direction),
            "Comb direction must be three finite numbers in [-1,1]")
    direction = Vector(direction)
    require(direction.length > 1e-8, "Comb direction cannot be zero")
    direction.normalize()
    require(type(amount) in (int,float) and math.isfinite(amount) and 0 <= amount <= .95, "Comb amount must be in [0,.95]")
    prepared_flow = None
    if flow is not None:
        from groom_flow import prepare_flow, evaluate_prepared_flow
        vertices = [v.co for v in groom.data.vertices]
        low = [min(v[i] for v in vertices) for i in range(3)]
        span = [max(v[i] for v in vertices)-low[i] for i in range(3)]
        prepared_flow = prepare_flow(low, span, list(direction), amount, flow)
    require(not groom.get("native_comb_settings"), "This groom is already combed; use a preserved uncombed source")
    require(all(abs(value-1) < 1e-6 for value in groom.matrix_world.to_scale()), "Apply emitter scale before combing")
    original = groom.particle_systems[0]
    require(original.settings.type == "HAIR" and not original.use_hair_dynamics, "Only native hair without dynamics is supported")
    armatures = [modifier.object for modifier in groom.modifiers if modifier.type == "ARMATURE"]
    require(len(armatures) == 1 and armatures[0] is not None, "Preserve one explicit skin armature")
    rig = armatures[0]
    require(rig.data.pose_position == "REST" or all(
        abs(bone.matrix_basis[row][column]-(row == column)) < 1e-6
        for bone in rig.pose.bones for row in range(4) for column in range(4)), "Comb only in the skin's rest pose")
    require("co_object_set" in bpy.types.ParticleHairKey.bl_rna.functions,
            "Installed Blender lacks the supported persistent hair-key setter")
    skin_before, settings_before = _skin_signature(groom,rig), _settings_signature(original)
    bpy.context.view_layer.update()
    initialization = _initialize_original_keys(groom)
    evaluated, modifier, system, before = _original_coordinates(groom)
    planned, parallel, arc_lengths, expected_lengths, scales = [], 0, [], [], []
    for points in before:
        lengths = [(b-a).length for a,b in zip(points,points[1:])]
        total = sum(lengths)
        require(1e-6 < total <= .08 and all(length > 1e-10 for length in lengths), "Stored strand length is zero or outside .08m")
        arc_lengths.append(total)
        strand_direction, strand_amount, scale = direction, amount, 1.
        if prepared_flow is not None:
            d, strand_amount, scale = evaluate_prepared_flow(tuple(points[0]), prepared_flow)
            strand_direction = Vector(d)
        require(1e-6 < total*scale <= .08, 'Regional hair length exceeds .08m')
        expected_lengths.append(total*scale)
        scales.append(scale)
        normal = (points[1]-points[0]).normalized()
        tangent = strand_direction-normal*strand_direction.dot(normal)
        if tangent.length < 1e-6:
            planned.append([points[0]+(point-points[0])*scale for point in points])
            parallel += 1
            continue
        tangent.normalize()
        changed, distance = [points[0].copy()], 0.
        for index,length in enumerate(lengths):
            distance += length
            lean = strand_amount*(.30+.70*distance/total)
            target = (normal*(1-lean)+tangent*lean).normalized()
            rotation = normal.rotation_difference(target)
            changed.append(changed[-1]+(rotation@(points[index+1]-points[index]))*scale)
        require(all(all(math.isfinite(value) for value in point) for point in changed), "Comb produced nonfinite keys")
        planned.append(changed)
    require(time.monotonic()-started < MAX_SECONDS, "Native comb preparation exceeded time budget")
    # This API marks original keys edited; assigning .co alone can be regenerated away.
    for index,(particle,points) in enumerate(zip(original.particles,planned,strict=True)):
        for key,point in zip(list(particle.hair_keys)[1:],points[1:],strict=True):
            key.co_object_set(evaluated,modifier,system.particles[index],point)
        require(time.monotonic()-started < MAX_SECONDS, "Native comb exceeded time budget; discard this new attempt")
    bpy.context.view_layer.update()
    _,_,_,after = _original_coordinates(groom)
    root_error = max((a[0]-b[0]).length for a,b in zip(before,after,strict=True))
    key_error = max((a-b).length for actual,expected in zip(after,planned,strict=True) for a,b in zip(actual,expected,strict=True))
    arc_after = [sum((b-a).length for a,b in zip(row,row[1:])) for row in after]
    length_error = max(abs(a-b) for a,b in zip(expected_lengths,arc_after,strict=True))
    require(root_error < 1e-6 and key_error < 1e-5 and length_error < 1e-5,
            "Evaluated persistent keys differ from planned rooted strand lengths")
    require(_skin_signature(groom,rig) == skin_before and _settings_signature(original) == settings_before,
            "Combing changed skin weights, children or hair settings")
    result = {"schema_version":1,"status":"combed_pending_render_review","method":"original ParticleHairKey.co_object_set with evaluated emitter/modifier/particle",
              "initialization":initialization,"direction_rest_object_axes":list(direction),"amount":amount,"parents":len(before),
              "stored_keys":sum(len(row) for row in before),"parallel_direction_strands_unchanged":parallel,
              "max_root_shift_m":root_error,"max_key_error_m":key_error,
              "max_arc_length_change_m":max(abs(a-b) for a,b in zip(arc_lengths,arc_after,strict=True)),
              "max_planned_arc_length_error_m":length_error,"flow_config":flow,
              "regional_length_scale_range":[min(scales),max(scales)],
              "max_stored_arc_length_m":max(arc_after),"mean_tip_displacement_m":sum((a[-1]-b[-1]).length for a,b in zip(before,after))/len(before),
              "skin_signature":skin_before,"settings_preserved":settings_before,"seconds":time.monotonic()-started,
              "native_only":True,"hair_dynamics":False,"api_source":SOURCE_API,
              "limits":{"parents":MAX_PARENTS,"stored_keys":MAX_KEYS,"wall_seconds":MAX_SECONDS,
                        "direction_component_range":[-1,1],"amount_range":[0,.95],"stored_parent_arc_length_m":.08},
              "limitations":["Apply once to a fresh groom in its rest pose; counts/topology changes can regenerate hair.",
                             "The vector is in emitter object axes; no panda-specific direction or recovered grooming.",
                             "Stored parent arc length follows explicit regional scale (otherwise preserved); children add configured roughness/clump variation."]}
    groom["native_comb_settings"] = json.dumps(result,sort_keys=True)
    return result


def _write(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")


def _rest_snapshot(groom):
    from blender_native_groom import _hair_snapshot
    result = _hair_snapshot(groom)
    _,_,_,keys = _original_coordinates(groom)
    local = [[list(key.co_local) for key in particle.hair_keys] for particle in groom.particle_systems[0].particles]
    result["stored_keys_sha256"] = hashlib.sha256(json.dumps(local,separators=(",",":")).encode()).hexdigest()
    result["max_stored_arc_length_m"] = max(sum((b-a).length for a,b in zip(row,row[1:])) for row in keys)
    return result


def probe(output):
    from blender_native_groom import add_native_groom,_material,_aim,_skin_signature

    started, out = time.monotonic(),Path(output)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine,scene.cycles.device = "CYCLES","CPU"
    scene.cycles.samples,scene.cycles.use_denoising = 24,True
    scene.cycles.denoising_use_gpu = False
    scene.render.threads_mode,scene.render.threads = "FIXED",4
    scene.render.resolution_x = scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format,scene.render.image_settings.color_mode = "PNG","RGBA"
    scene.view_settings.view_transform = "AgX"
    scene.world = bpy.data.worlds.new("CombProbeWorld")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = .07
    bpy.ops.object.armature_add(enter_editmode=True)
    rig = bpy.context.object
    rig.name = "AnalyticCombTwoBoneRig"
    root = rig.data.edit_bones[0]
    root.name,root.head,root.tail = "root",(.5,0,.12),(.5,0,.75)
    tip = rig.data.edit_bones.new("tip")
    tip.head,tip.tail,tip.parent,tip.use_connect = (.5,0,.75),(.5,0,1.36),root,True
    bpy.ops.object.mode_set(mode="OBJECT")
    grooms,bases = [],[]
    for name,center,scale,tint,count in (("sphere",(-.55,0,.72),(.42,.42,.42),(.55,.46,.32),1800),
                                         ("limb",(.5,0,.75),(.20,.20,.62),(.018,.025,.035),1600)):
        material = _material("Analytic_"+name,tint)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=40,ring_count=28,location=center)
        base = bpy.context.object
        base.name,base.scale = "Analytic_"+name,scale
        bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
        base.data.materials.append(material)
        for polygon in base.data.polygons:polygon.use_smooth = True
        groups = {key:base.vertex_groups.new(name=key) for key in ("root","tip")}
        for vertex in base.data.vertices:
            weight = min(1.,max(0.,(vertex.co.z-.52)/.38)) if name == "limb" else 0
            if weight < 1:groups["root"].add([vertex.index],1-weight,"REPLACE")
            if weight > 0:groups["tip"].add([vertex.index],weight,"REPLACE")
        modifier = base.modifiers.new("AnalyticSkin","ARMATURE")
        modifier.object,base.parent = rig,rig
        groom,_ = add_native_groom(base,rig,{"parents":count,"children":15,"length":.035,
                                            "radius":.00025,"roughness":.001,"seed":227},material)
        bases.append(base);grooms.append(groom)
    bpy.ops.mesh.primitive_plane_add(size=200)
    bpy.context.object.data.materials.append(_material("Floor",(.12,.14,.17)))
    for name,position,power,size in (("Key",(-3,-4,5),900,3),("Fill",(3,-2,2),140,3),("Rim",(0,3,4),600,2)):
        lamp = bpy.data.lights.new(name,"AREA")
        lamp.energy,lamp.shape,lamp.size = power,"DISK",size
        obj = bpy.data.objects.new(name,lamp)
        bpy.context.collection.objects.link(obj)
        obj.location = position
        _aim(obj,(0,0,.75))
    camera = bpy.data.objects.new("FrontOnCamera",bpy.data.cameras.new("FrontOnCamera"))
    bpy.context.collection.objects.link(camera)
    camera.location = (0,-6,.75)
    camera.data.type,camera.data.ortho_scale = "ORTHO",2.65
    _aim(camera,(0,0,.75));scene.camera = camera
    rig.pose.bones["tip"].rotation_mode = "XYZ"
    scene.frame_set(1);bpy.context.view_layer.update()
    properties = {name:{"readonly":prop.is_readonly,"description":prop.description}
                  for name,prop in bpy.types.ParticleHairKey.bl_rna.properties.items() if name in ("co","co_local","time")}
    _write(out/"api-inspection.json",{"version":bpy.app.version_string,"hair_key_properties":properties,
           "functions":list(bpy.types.ParticleHairKey.bl_rna.functions.keys()),
           "parents_original":[len(g.particle_systems[0].particles) for g in grooms],
           "parents_evaluated":[len(_evaluated_system(g)[2].particles) for g in grooms]})
    _write(out/"initialization.json",{g.name:_initialize_original_keys(g) for g in grooms})
    baseline = {g.name:_rest_snapshot(g) for g in grooms}
    _write(out/"baseline-hair-samples.json",baseline)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out/"uncombed.blend"),check_existing=False)
    scene.render.filepath = str(out/"front-radial.png")
    bpy.ops.render.render(write_still=True)
    reports = [comb_native_groom(g,[0,0,-1],.65) for g in grooms]
    combed = {g.name:_rest_snapshot(g) for g in grooms}
    projection = []
    for groom in grooms:
        a,b = baseline[groom.name],combed[groom.name]
        radial,leaning = [],[]
        for index,(root,tip) in enumerate(zip(a["roots"],a["tips"],strict=True)):
            v = Vector(tip)-Vector(root)
            if v.length > 1e-6 and v.y/v.length < -.9:
                w = Vector(b["tips"][index])-Vector(b["roots"][index])
                radial.append(math.hypot(v.x,v.z));leaning.append(math.hypot(w.x,w.z))
        require(radial,"No front-facing parent strand samples available")
        projection.append({"object":groom.name,"front_facing_samples":len(radial),
                           "radial_mean_projected_length_m":sum(radial)/len(radial),
                           "combed_mean_projected_length_m":sum(leaning)/len(leaning)})
    _write(out/"combed-hair-samples.json",combed)
    _write(out/"comb-reports.json",reports)
    scene.render.filepath = str(out/"front-combed.png")
    bpy.ops.render.render(write_still=True)
    rig.pose.bones["tip"].rotation_euler.z = math.radians(-50)
    bpy.context.view_layer.update()
    bent = {g.name:_rest_snapshot(g) for g in grooms}
    _write(out/"bent-hair-samples.json",bent)
    scene.render.filepath = str(out/"combed-bent.png")
    bpy.ops.render.render(write_still=True)
    checks = []
    for groom,base in zip(grooms,bases,strict=True):
        before,after = combed[groom.name],bent[groom.name]
        shift = max(math.dist(a,b) for a,b in zip(before["roots"],after["roots"],strict=True))
        root_error = max(before["max_root_surface_distance"],after["max_root_surface_distance"])
        require(root_error < .003,"Combed hair roots detached from skinned surface")
        require(shift > .1 if "limb" in groom.name else shift < 1e-6,"Hair roots do not follow the assigned analytic bones")
        require(_skin_signature(groom,rig) == _skin_signature(base,rig),"Source skin differs after combing")
        require(after["max_stored_arc_length_m"] < .08,"Bent hair length exceeded bound")
        checks.append({"object":groom.name,"root_surface_distance_m":root_error,"max_root_motion_m":shift,
                       "max_bent_stored_arc_length_m":after["max_stored_arc_length_m"],"weights_preserved":True})
    rig.pose.bones["tip"].rotation_euler.z = 0
    bpy.context.view_layer.update()
    bpy.ops.wm.save_as_mainfile(filepath=str(out/"combed.blend"),check_existing=False)
    names = [g.name for g in grooms]
    bpy.ops.wm.open_mainfile(filepath=str(out/"combed.blend"),load_ui=False,use_scripts=False)
    reload_results = []
    for name in names:
        groom = bpy.data.objects[name]
        reloaded = _rest_snapshot(groom)
        require(reloaded["stored_keys_sha256"] == combed[name]["stored_keys_sha256"],"Persistent comb changed after save/reload")
        reload_results.append({"object":name,"stored_keys_identical":True,"comb_metadata_preserved":bool(groom.get("native_comb_settings"))})
    report = {"schema_version":1,"status":"mechanical_probe_passed_needs_visual_review","scope":"Analytic sphere and skinned two-bone limb, not a generated character",
              "blender_version":bpy.app.version_string,"direction":[0,0,-1],"amount":.65,
              "configured_native_fibers":3400*16,"comb_reports":reports,"bone_checks":checks,
              "front_on_projection":projection,
              "reload_checks":reload_results,"wall_seconds":time.monotonic()-started,
              "provider_calls":0,"provider_cost_usd":0,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "api_source":SOURCE_API,"artifacts":[{"path":p.name,"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
                                                     for p in sorted(out.iterdir()) if p.is_file() and p.suffix in (".png",".blend",".json")]}
    _write(out/"probe-result.json",report)
    print(json.dumps({"status":report["status"],"wall_seconds":report["wall_seconds"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    try:
        probe(args.output)
    except Exception as error:
        _write(Path(args.output)/"probe-failure.json",{"status":"unsupported_or_failed","error":str(error),
                                                       "blender_version":bpy.app.version_string,"api_source":SOURCE_API})
        raise


if __name__ == "__main__":
    main()
