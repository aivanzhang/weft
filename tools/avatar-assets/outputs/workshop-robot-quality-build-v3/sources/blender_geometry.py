"""Deterministic v2 geometry and baked surfaces; no user code or external inputs."""

from bisect import bisect_left
import hashlib
import math
from pathlib import Path
import random


def normalized_weights(weights, maximum=4):
    ordered = sorted(((name, value) for name, value in weights.items() if value > 1e-10),
                     key=lambda item: (-item[1], item[0]))[:maximum]
    total = sum(value for _, value in ordered)
    if total <= 0:
        raise ValueError("Geometry has an unweighted vertex")
    return {name: value/total for name, value in ordered}


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _unit(value):
    length = math.sqrt(sum(component*component for component in value))
    return tuple(component/length for component in value) if length > 1e-14 else (0., 0., 1.)


def fur_geometry(vertices, triangles, weights, uv, settings, normals=None):
    """Generate surface-area sampled triangle strands with inherited skinning."""
    rng = random.Random(settings["seed"])
    areas, usable, total = [], [], 0.
    for triangle in triangles:
        a, b, c = (vertices[index] for index in triangle)
        cross = _cross(tuple(b[i]-a[i] for i in range(3)), tuple(c[i]-a[i] for i in range(3)))
        area = math.sqrt(sum(value*value for value in cross))/2
        if area > 1e-14:
            total += area
            areas.append(total)
            usable.append((triangle, _unit(cross)))
    if not usable and settings["count"]:
        raise ValueError("Fur requires a nondegenerate surface")
    output = {"vertices": [], "faces": [], "weights": [], "uv": []}
    for _ in range(settings["count"]):
        triangle, face_normal = usable[min(bisect_left(areas, rng.random()*total), len(usable)-1)]
        root = math.sqrt(rng.random())
        fraction = rng.random()
        bary = (1-root, root*(1-fraction), root*fraction)
        position = tuple(sum(vertices[index][axis]*factor for index, factor in zip(triangle, bary)) for axis in range(3))
        normal = _unit(tuple(sum(normals[index][axis]*factor for index, factor in zip(triangle, bary)) for axis in range(3))) if normals else face_normal
        basis = (0., 0., 1.) if abs(normal[2]) < .9 else (0., 1., 0.)
        tangent = _unit(_cross(normal, basis))
        bitangent = _cross(normal, tangent)
        angle = rng.random()*math.tau
        side = tuple(tangent[i]*math.cos(angle)+bitangent[i]*math.sin(angle) for i in range(3))
        length = settings["length"]*(.75+.5*rng.random())
        width = settings["width"]*(.8+.4*rng.random())
        lean = (rng.random()-.5)*length*.25
        points = [tuple(position[i]-side[i]*width/2 for i in range(3)),
                  tuple(position[i]+side[i]*width/2 for i in range(3)),
                  tuple(position[i]+normal[i]*length+bitangent[i]*lean for i in range(3))]
        inherited = {}
        for index, factor in zip(triangle, bary):
            for bone, weight in weights[index].items():
                inherited[bone] = inherited.get(bone, 0)+factor*weight
        inherited = normalized_weights(inherited)
        base_uv = tuple(sum(uv[index][axis]*factor for index, factor in zip(triangle, bary)) for axis in range(2))
        first = len(output["vertices"])
        output["vertices"].extend(points)
        output["faces"].append((first, first+1, first+2))
        output["weights"].extend(dict(inherited) for _ in range(3))
        output["uv"].extend(((base_uv[0]-.001, base_uv[1]), (base_uv[0]+.001, base_uv[1]),
                             (base_uv[0], base_uv[1]+.004)))
    return output


def surface_pixels(color, settings):
    """Return sRGB color and linear tangent-normal pixels from a seeded height field."""
    import numpy as np

    size, strength = settings["resolution"], settings["strength"]
    rng = np.random.default_rng(settings["seed"])
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)/size
    noise = rng.uniform(-1, 1, (size, size)).astype(np.float32)
    if settings["pattern"] == "fur":
        fx = np.fft.rfftfreq(size)[None, :]
        fy = np.fft.fftfreq(size)[:, None]
        pixel_scale = size/256

        def strands(field, width, length):
            kernel = np.exp(-2*math.pi**2*((fx*width*pixel_scale)**2+(fy*length*pixel_scale)**2))
            filtered = np.fft.irfft2(np.fft.rfft2(field)*kernel, s=(size, size)).astype(np.float32)
            return np.tanh(filtered/max(float(filtered.std()), 1e-6))

        # Short anisotropic noise strokes break up the surface without repeated corrugated bands.
        fine = rng.uniform(-1, 1, (size, size)).astype(np.float32)
        height = .5+.095*strands(noise, .55, 3.5)+.025*strands(fine, .3, 1.1)
    elif settings["pattern"] == "scales":
        row = np.floor(y*16)
        u = (x*16+(row % 2)*.5) % 1-.5
        v = (y*16) % 1-.5
        dome = np.maximum(0, 1-(u/.51)**2-(v/.56)**2)
        height = .15+.7*dome+noise*.025
    else:
        height = .5+.24*np.sin(x*math.tau*83)+.13*np.sin(x*math.tau*117+y*math.tau)+noise*.06
    height = np.asarray(height, dtype=np.float32)
    linear = np.asarray([value/12.92 if value < .04045 else ((value+.055)/1.055)**2.4 for value in color[:3]], dtype=np.float32)
    rgba = np.ones((size, size, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(linear[None, None, :]*(1+(height[:, :, None]-.5)*strength*.28), 0, 1)
    # Byte-backed Blender images store encoded sRGB; labeling linear bytes causes double darkening.
    rgba[:, :, :3] = np.where(rgba[:, :, :3] <= .0031308, rgba[:, :, :3]*12.92,
                              1.055*rgba[:, :, :3]**(1/2.4)-.055)
    dx = (np.roll(height, -1, axis=1)-np.roll(height, 1, axis=1))*strength*1.7
    dy = (np.roll(height, -1, axis=0)-np.roll(height, 1, axis=0))*strength*1.7
    normal = np.stack((-dx, -dy, np.ones_like(dx)), axis=2)
    normal /= np.linalg.norm(normal, axis=2)[:, :, None]
    normal_rgba = np.ones_like(rgba)
    normal_rgba[:, :, :3] = normal*.5+.5
    return rgba, normal_rgba


def attach_surface(material, name, source, out):
    import bpy

    folder = Path(out)/"textures"
    folder.mkdir(exist_ok=True)
    settings = source["surface"]
    color, normal = surface_pixels(source["color"], settings)
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    report = []
    for kind, pixels, color_space in (("basecolor", color, "sRGB"), ("normal", normal, "Non-Color")):
        image = bpy.data.images.new(name+"_"+kind, width=settings["resolution"], height=settings["resolution"], alpha=True)
        image.colorspace_settings.name = color_space
        image.pixels.foreach_set(pixels.ravel())
        image.filepath_raw = str(folder/(name+"-"+kind+".png"))
        image.file_format = "PNG"
        image.save()
        image.pack()
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = image
        texture.interpolation = "Linear"
        texture.extension = "REPEAT"
        if kind == "basecolor":
            links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            normal_map = nodes.new("ShaderNodeNormalMap")
            normal_map.space = "TANGENT"
            normal_map.inputs["Strength"].default_value = 1
            links.new(texture.outputs["Color"], normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
        raw = Path(image.filepath_raw).read_bytes()
        report.append({"material": name, "kind": kind, "path": "textures/"+Path(image.filepath_raw).name,
                       "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                       "resolution": settings["resolution"], "pattern": settings["pattern"],
                       "algorithm": "anisotropic_irregular_fur_v2" if settings["pattern"] == "fur" else settings["pattern"]+"_v1",
                       "strength": settings["strength"], "seed": settings["seed"]})
    return report


def _uv_mesh(mesh, coordinates, explicit=False):
    layer = mesh.uv_layers.new(name="UVMap")
    for polygon in mesh.polygons:
        values = [coordinates[mesh.loops[index].vertex_index] for index in polygon.loop_indices]
        seam = not explicit and max(uv[0] for uv in values)-min(uv[0] for uv in values) > .5
        for index, uv in zip(polygon.loop_indices, values):
            layer.data[index].uv = (uv[0]+1 if seam and uv[0] < .5 else uv[0], uv[1])


def _set_weights(obj, weights):
    groups = {name: obj.vertex_groups.new(name=name) for name in sorted({name for weight in weights for name in weight})}
    for index, weight in enumerate(weights):
        for name, value in normalized_weights(weight).items():
            groups[name].add([index], value, "REPLACE")


def _get_weights(obj):
    names = {group.index: group.name for group in obj.vertex_groups}
    return [normalized_weights({names[group.group]: group.weight for group in vertex.groups}) for vertex in obj.data.vertices]


def planar_triangle_uv(points, previous):
    """Replace a collapsed UV triangle using its own geometric plane, keeping its texture neighborhood."""
    edge = tuple(points[1][i]-points[0][i] for i in range(3))
    other = tuple(points[2][i]-points[0][i] for i in range(3))
    normal = _unit(_cross(edge, other))
    tangent = _unit(edge)
    bitangent = _cross(normal, tangent)
    length = max(math.dist(points[i], points[(i+1) % 3]) for i in range(3))
    if length < 1e-12:
        raise ValueError("Cannot project a zero-size triangle")
    span = max(.05, min(1., max(max(uv[i] for uv in previous)-min(uv[i] for uv in previous) for i in range(2))))
    center = tuple(sum(uv[i] for uv in previous)/3 for i in range(2))
    coordinates = [(sum((point[i]-points[0][i])*axis[i] for i in range(3))*span/length)
                   for point in points for axis in (tangent, bitangent)]
    means = (sum(coordinates[::2])/3, sum(coordinates[1::2])/3)
    return [(center[0]+coordinates[2*i]-means[0], center[1]+coordinates[2*i+1]-means[1]) for i in range(3)]


def repair_surface_uvs(obj):
    import bmesh

    mesh = obj.data
    original_normals = [tuple(normal.vector) for normal in mesh.corner_normals] if mesh.has_custom_normals else None
    bm = bmesh.new()
    bm.from_mesh(mesh)
    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        bm.free()
        raise ValueError("Quality mesh requires UV0 before tangent repair")
    corner_ids = bm.loops.layers.int.new("_original_corner")
    for face, polygon in zip(bm.faces, mesh.polygons):
        for loop, index in zip(face.loops, polygon.loop_indices):
            loop[corner_ids] = index
    # Sub-micrometer bevel remnants collapse at float32 precision and cannot support stable frames.
    def collapsed(face):
        lengths = [edge.calc_length() for edge in face.edges]
        area = face.calc_area()
        return min(lengths) <= 1e-6 or area <= 5e-11 or 2*area/max(lengths)**2 <= 1e-6

    degenerate = [face for face in bm.faces if collapsed(face)]
    if degenerate:
        bmesh.ops.delete(bm, geom=degenerate, context="FACES_ONLY")
    repaired = 0
    minimum_determinant = float("inf")
    for face in bm.faces:
        if len(face.loops) != 3:
            bm.free()
            raise ValueError("UV repair requires stable triangulation first")
        uv = [tuple(loop[uv_layer].uv) for loop in face.loops]
        determinant = (uv[1][0]-uv[0][0])*(uv[2][1]-uv[0][1])-(uv[2][0]-uv[0][0])*(uv[1][1]-uv[0][1])
        if abs(determinant) <= 1e-10:
            uv = planar_triangle_uv([tuple(loop.vert.co) for loop in face.loops], uv)
            for loop, value in zip(face.loops, uv):
                loop[uv_layer].uv = value
            repaired += 1
            uv = [tuple(loop[uv_layer].uv) for loop in face.loops]
            determinant = (uv[1][0]-uv[0][0])*(uv[2][1]-uv[0][1])-(uv[2][0]-uv[0][0])*(uv[1][1]-uv[0][1])
        minimum_determinant = min(minimum_determinant, abs(determinant))
    normals = [original_normals[loop[corner_ids]] for face in bm.faces for loop in face.loops] if original_normals else None
    bm.loops.layers.int.remove(corner_ids)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    if normals:
        mesh.normals_split_custom_set(normals)
    ambiguous, passes = set(), 0
    for attempt in range(3):
        mesh.calc_tangents(uvmap=mesh.uv_layers.active.name)
        bad = set()
        for polygon in mesh.polygons:
            for index in polygon.loop_indices:
                tangent = mesh.loops[index].tangent
                normal = mesh.corner_normals[index].vector
                if abs(tangent.length_squared-1) >= .0005 or abs(normal.dot(tangent)) >= .001:
                    bad.add(polygon.index)
                    break
        mesh.free_tangents()
        if not bad:
            break
        if attempt == 2:
            raise ValueError(f"Cannot derive valid MikkTSpace frames for {len(bad)} triangles")
        # Overlapping UV fans can cancel a shared tangent despite nonzero individual determinants.
        for index in bad:
            polygon = mesh.polygons[index]
            previous = [tuple(mesh.uv_layers.active.data[loop].uv) for loop in polygon.loop_indices]
            points = [tuple(mesh.vertices[vertex].co) for vertex in polygon.vertices]
            for loop, value in zip(polygon.loop_indices, planar_triangle_uv(points, previous)):
                mesh.uv_layers.active.data[loop].uv = value
        ambiguous.update(bad)
        passes += 1
        mesh.update()
    minimum_determinant = float("inf")
    for polygon in mesh.polygons:
        uv = [mesh.uv_layers.active.data[index].uv for index in polygon.loop_indices]
        determinant = (uv[1].x-uv[0].x)*(uv[2].y-uv[0].y)-(uv[2].x-uv[0].x)*(uv[1].y-uv[0].y)
        minimum_determinant = min(minimum_determinant, abs(determinant))
    if minimum_determinant <= 1e-12:
        raise ValueError("UV repair left an ill-conditioned triangle")
    return {"degenerate_faces_removed": len(degenerate), "uv_degenerate_faces_reprojected": repaired,
            "ambiguous_tangent_faces_reprojected": len(ambiguous), "tangent_repair_passes": passes,
            "minimum_absolute_uv_determinant": minimum_determinant,
            "minimum_geometric_edge_m": 1e-6, "minimum_relative_triangle_area": 1e-6,
            "method": "Per-loop geometric-plane UV projection for collapsed UVs or cancelling UV fans; other UV seams unchanged. Tangents are computed by Blender MikkTSpace, not replaced."}


def build_part(part, materials, quality):
    import bpy

    common = {"location": part["location"], "rotation": part.get("rotation", [0, 0, 0])}
    shape = part["shape"]
    if shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=part.get("segments", 20), ring_count=part.get("rings", 12), radius=1, **common)
    elif shape == "cube":
        bpy.ops.mesh.primitive_cube_add(size=2, **common)
    elif shape == "cone":
        bpy.ops.mesh.primitive_cone_add(vertices=part.get("segments", 24), radius1=1, radius2=0, depth=2, **common)
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=part.get("segments", 24), radius=1, depth=2, **common)
    elif shape == "torus":
        bpy.ops.mesh.primitive_torus_add(major_segments=part.get("segments", 24), minor_segments=part.get("rings", 10),
                                       major_radius=1, minor_radius=.25, **common)
    elif shape == "mesh":
        data = bpy.data.meshes.new(part["name"]+"_surface")
        data.from_pydata(part["vertices"], [], part["faces"])
        data.update()
        obj = bpy.data.objects.new(part["name"], data)
        bpy.context.collection.objects.link(obj)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        obj.location, obj.rotation_euler = common["location"], common["rotation"]
        uv = part.get("uv")
        if uv is None:
            uv = [(math.atan2(vertex[1], vertex[0])/math.tau+.5,
                   math.acos(max(-1, min(1, vertex[2]/max(1e-12, math.sqrt(sum(x*x for x in vertex))))))/math.pi)
                  for vertex in part["vertices"]]
        _uv_mesh(data, uv, explicit="uv" in part)
    else:
        raise ValueError("Unsupported geometry")
    obj = bpy.context.object
    obj.name, obj.scale = part["name"], part["scale"]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if shape != "cube":
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    obj.data.materials.append(materials[part["material"]])
    if "weights" in part:
        _set_weights(obj, part["weights"])
    else:
        group = obj.vertex_groups.new(name=part["bone"])
        group.add(list(range(len(obj.data.vertices))), 1, "REPLACE")
    if quality and part.get("bevel", 0):
        modifier = obj.modifiers.new("AuthoredEdgeBevel", "BEVEL")
        modifier.width = part["bevel"]
        modifier.segments = part.get("bevel_segments", 2)
        modifier.limit_method = "ANGLE"
        modifier.harden_normals = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        weights = _get_weights(obj)
        obj.vertex_groups.clear()
        _set_weights(obj, weights)
    objects, fur_count = [obj], part.get("fur", {}).get("count", 0)
    if fur_count:
        obj.data.calc_loop_triangles()
        uv = [(0., 0.) for _ in obj.data.vertices]
        if obj.data.uv_layers.active:
            for loop in obj.data.loops:
                uv[loop.vertex_index] = tuple(obj.data.uv_layers.active.data[loop.index].uv)
        generated = fur_geometry([tuple(vertex.co) for vertex in obj.data.vertices],
                                 [tuple(triangle.vertices) for triangle in obj.data.loop_triangles],
                                 _get_weights(obj), uv, part["fur"],
                                 [tuple(vertex.normal) for vertex in obj.data.vertices])
        data = bpy.data.meshes.new(part["name"]+"_fur")
        data.from_pydata(generated["vertices"], [], generated["faces"])
        data.update()
        fur = bpy.data.objects.new(part["name"]+"_fur", data)
        bpy.context.collection.objects.link(fur)
        data.materials.append(materials[part["material"]])
        _uv_mesh(data, generated["uv"], explicit=True)
        _set_weights(fur, generated["weights"])
        objects.append(fur)
    return objects, {"name": part["name"], "shape": shape, "control_vertices": len(part.get("vertices", [])),
                     "surface_vertices": len(obj.data.vertices), "surface_faces": len(obj.data.polygons),
                     "fur_strands": fur_count, "fur_vertices": 3*fur_count,
                     "mixed_weight_vertices": sum(len(weight) > 1 for item in objects for weight in _get_weights(item))}
