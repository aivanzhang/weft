"""Seeded curved hair cards with inherited skinning and embedded cutout atlases."""

from bisect import bisect_left
import hashlib
import math
from numbers import Real
from pathlib import Path
import random

ATLAS_SIZE = 1024
ATLAS_TILES = 4
FIBERS_PER_TILE = 14
ATLAS_PROFILES = {"fine-v1": {"fibers": 14, "cutoff": .45},
                  "distance-v1": {"fibers": 10, "cutoff": .18},
                  "tuft-v1": {"fibers": 3, "cutoff": .22}}
TUFT_VERTEX_LIMIT = 100_000
TUFT_CARD_LIMIT = 5000
TUFT_EXPORTED_CORNERS = 12


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _unit(v):
    length = math.sqrt(sum(x*x for x in v))
    return tuple(x/length for x in v) if length > 1e-14 else (0., 0., 1.)


def card_geometry(vertices, triangles, weights, settings, normals=None, *, profile="legacy", colors=None,
                  prepared_flow=None, root_color_sampler=None, record_bindings=False):
    """Each six-vertex ribbon carries an atlas tile, not individual mesh hairs."""
    if not isinstance(profile, str) or profile not in ("legacy", "combed-v1", "tuft-v1"):
        raise ValueError("Unknown fur card geometry profile")
    if type(record_bindings) is not bool:
        raise ValueError("Root recording must be explicitly boolean")
    if profile == "tuft-v1":
        if colors is not None:
            raise ValueError("Tuft colors must come from the exact root UV sampler, not vertex interpolation")
        return _tuft_geometry(vertices, triangles, weights, settings, normals, prepared_flow, root_color_sampler,
                              record_bindings=record_bindings)
    if prepared_flow is not None or root_color_sampler is not None:
        raise ValueError("Regional flow and exact root UV sampling require tuft-v1")
    if 'root_placement' in settings:
        raise ValueError("Explicit root placement requires tuft-v1")
    if colors is not None:
        try:
            if isinstance(colors, (str, bytes, dict)) or len(colors) != len(vertices):
                raise ValueError("Fur root colors must match the source vertex count")
            for row in colors:
                if isinstance(row, (str, bytes, dict)) or len(row) != 3 or any(
                        isinstance(value, bool) or not isinstance(value, Real)
                        or not math.isfinite(value) or not 0 <= value <= 1 for value in row):
                    raise ValueError("Fur root colors require finite RGB triples in 0..1")
        except (TypeError, AttributeError):
            raise ValueError("Fur root colors require one finite RGB triple per source vertex") from None
    combed = profile == "combed-v1"
    rng = random.Random(settings["seed"])
    cumulative, usable, total = [], [], 0.
    for ordinal, triangle in enumerate(triangles):
        a, b, c = (vertices[i] for i in triangle)
        cross = _cross(tuple(b[i]-a[i] for i in range(3)), tuple(c[i]-a[i] for i in range(3)))
        area = math.sqrt(sum(x*x for x in cross))/2
        if area > 1e-14:
            total += area
            cumulative.append(total)
            usable.append((ordinal, triangle, _unit(cross)))
    if not usable and settings["count"]:
        raise ValueError("Hair cards require nondegenerate emitter geometry")
    output = {"vertices": [], "faces": [], "weights": [], "uv": [], "normals": []}
    if colors is not None:
        output["colors"] = []
    if record_bindings:
        output['root_bindings'] = {'triangle_index': [], 'barycentric': [], 'root_inset': []}
        output['root_summary'] = {'root_placement': 'area-random-v1', 'recorded_roots': settings['count']}
    direction = settings.get("direction", [0., 0., -1.])
    for _ in range(settings["count"]):
        ordinal, triangle, face_normal = usable[min(bisect_left(cumulative, rng.random()*total), len(usable)-1)]
        r, fraction = math.sqrt(rng.random()), rng.random()
        bary = (1-r, r*(1-fraction), r*fraction)
        root = tuple(sum(vertices[index][axis]*f for index, f in zip(triangle, bary)) for axis in range(3))
        normal = (_unit(tuple(sum(normals[index][axis]*f for index, f in zip(triangle, bary))
                             for axis in range(3))) if normals else face_normal)
        if record_bindings:
            output['root_bindings']['triangle_index'].append(ordinal)
            output['root_bindings']['barycentric'].append(list(bary))
            output['root_bindings']['root_inset'].append([value*.0005 for value in normal])
        basis = (0., 0., 1.) if abs(normal[2]) < .9 else (0., 1., 0.)
        tangent = _unit(_cross(normal, basis))
        bitangent = _cross(normal, tangent)
        angle = rng.random()*math.tau
        side = tuple(tangent[i]*math.cos(angle)+bitangent[i]*math.sin(angle) for i in range(3))
        projected = tuple(direction[i]-normal[i]*sum(direction[k]*normal[k] for k in range(3)) for i in range(3))
        comb = _unit(projected) if sum(x*x for x in projected) > 1e-8 else bitangent
        if combed:
            across = _unit(_cross(normal, comb))
            # Reuse the legacy angle draw so root, size and atlas samples stay aligned.
            roll = (angle/math.tau-.5)*.3
            side = tuple(across[i]*math.cos(roll)+normal[i]*math.sin(roll) for i in range(3))
        length = settings["length"]*(.72+.45*rng.random())
        width = settings["width"]*(.8+.4*rng.random())
        inherited = {}
        for index, factor in zip(triangle, bary):
            for bone, weight in weights[index].items():
                inherited[bone] = inherited.get(bone, 0)+factor*weight
        inherited = dict(sorted(inherited.items(), key=lambda pair: (-pair[1], pair[0]))[:4])
        weight_sum = sum(inherited.values())
        inherited = {name: weight/weight_sum for name, weight in inherited.items() if weight > 0}
        tile = rng.randrange(ATLAS_TILES**2)
        tile_x, tile_y = tile % ATLAS_TILES, tile // ATLAS_TILES
        first = len(output["vertices"])
        root_color = (tuple(sum(float(colors[index][axis])*f for index, f in zip(triangle, bary))
                            for axis in range(3)) if colors is not None else None)
        rows = ((0., .92), (.5, .8), (1., .15)) if combed else ((0., .92), (.5, 1.), (1., .82))
        lift, lean = (.35, .8) if combed else (.82, .45)
        for t, span in rows:
            # The root intersects the emitter slightly; fibers lean with the authored comb.
            center = tuple(root[i]+normal[i]*(length*t*lift-.0005)+comb[i]*length*lean*t*t for i in range(3))
            for sign in (-1, 1):
                output["vertices"].append(tuple(center[i]+side[i]*width*span*sign/2 for i in range(3)))
                output["normals"].append(normal)
                output["weights"].append(dict(inherited))
                output["uv"].append(((tile_x+(.015 if sign < 0 else .985))/ATLAS_TILES,
                                     (tile_y+.015+.97*t)/ATLAS_TILES))
                if root_color is not None:
                    output["colors"].append(root_color)
        for a, b, c in ((0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)):
            if combed:
                pa, pb, pc = (output["vertices"][first+i] for i in (a, b, c))
                cross = _cross(tuple(pb[i]-pa[i] for i in range(3)), tuple(pc[i]-pa[i] for i in range(3)))
                if sum(cross[i]*normal[i] for i in range(3)) < 0:
                    b, c = c, b
            output["faces"].append((first+a, first+b, first+c))
    return output


def _finite(value, label, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Invalid finite tuft " + label)
    try:
        value = float(value)
    except (ValueError, OverflowError):
        raise ValueError("Invalid finite tuft " + label) from None
    if not math.isfinite(value) or low is not None and value < low or high is not None and value > high:
        raise ValueError("Invalid finite tuft " + label)
    return value


def _triple(value, label, low=None, high=None):
    try:
        if isinstance(value, (str, bytes, dict)) or len(value) != 3:
            raise ValueError("Tuft " + label + " requires three numbers")
        return tuple(_finite(v, label, low, high) for v in value)
    except TypeError:
        raise ValueError("Tuft " + label + " requires three numbers") from None


def tuft_export_budget(base_export_vertices, total_cards, reserve_vertices=4096):
    """Call once across all bands; twelve corners per card bounds glTF attribute splits."""
    for value, maximum in ((base_export_vertices, TUFT_VERTEX_LIMIT), (total_cards, TUFT_CARD_LIMIT),
                           (reserve_vertices, TUFT_VERTEX_LIMIT)):
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError("Invalid tuft exported vertex budget")
    remaining = TUFT_VERTEX_LIMIT-base_export_vertices-reserve_vertices
    projected = base_export_vertices+total_cards*TUFT_EXPORTED_CORNERS+reserve_vertices
    if remaining < 0 or projected > TUFT_VERTEX_LIMIT:
        raise ValueError("Tuft coat exceeds the aggregate 100000 exported vertex budget")
    return {"base_export_vertices": base_export_vertices, "cards": total_cards,
            "exported_corners_per_card": TUFT_EXPORTED_CORNERS, "reserve_vertices": reserve_vertices,
            "projected_max_vertices": projected, "limit": TUFT_VERTEX_LIMIT,
            "maximum_cards": min(TUFT_CARD_LIMIT, remaining//TUFT_EXPORTED_CORNERS),
            "scope": "Conservative corner allowance, not measured export size or visible fur coverage"}


def _tuft_geometry(vertices, triangles, weights, settings, normals, prepared_flow, root_color_sampler, *,
                   record_bindings=False):
    try:
        if not 1 <= len(vertices) <= TUFT_VERTEX_LIMIT or not 0 <= len(triangles) <= 100_000:
            raise ValueError("Tuft emitter vertex/triangle limit exceeded")
        if len(weights) != len(vertices) or normals is not None and len(normals) != len(vertices):
            raise ValueError("Tuft emitter weights/normals must match vertices")
    except TypeError:
        raise ValueError("Tuft emitter requires bounded sequences") from None
    required = {"seed", "count", "length", "width", "direction"}
    if (not isinstance(settings, dict) or not required <= set(settings)
            or not set(settings) <= required | {"comb", "mode", "profile", "sample_root_color", "root_placement"}):
        raise ValueError("Invalid tuft settings")
    if 'root_placement' in settings and settings['root_placement'] != 'skin-consistent-v1':
        raise ValueError("Unsupported tuft root placement")
    if type(settings["count"]) is not int or not 0 <= settings["count"] <= TUFT_CARD_LIMIT:
        raise ValueError("Tuft card count exceeds its bound")
    if type(settings["seed"]) is not int or not 0 <= settings["seed"] <= 2**31-1:
        raise ValueError("Invalid tuft seed")
    base_length = _finite(settings["length"], "length", .001, .04)
    base_width = _finite(settings["width"], "width", .0003, .02)
    base_direction = _triple(settings["direction"], "direction", -1, 1)
    if math.hypot(*base_direction) <= 1e-8:
        raise ValueError("Tuft direction must be nonzero")
    base_direction = _unit(base_direction)
    base_comb = _finite(settings.get("comb", .65), "comb", 0, .95)
    if not callable(root_color_sampler):
        raise ValueError("Tuft coat requires a trusted exact root UV color sampler")
    evaluate_flow = None
    if prepared_flow is not None:
        if __package__:
            from .groom_flow import PreparedFlow, evaluate_prepared_flow
        else:
            from groom_flow import PreparedFlow, evaluate_prepared_flow
        if not isinstance(prepared_flow, PreparedFlow):
            raise ValueError("Tuft regional field must come from prepare_flow")
        evaluate_flow = evaluate_prepared_flow
    vertices = [_triple(v, "vertex") for v in vertices]
    if normals is not None:
        normals = [_triple(v, "normal", -1, 1) for v in normals]
        if any(math.hypot(*v) <= 1e-8 for v in normals):
            raise ValueError("Tuft source normals must be nonzero")
    for row in weights:
        if (not isinstance(row, dict) or not 1 <= len(row) <= 64
                or any(not isinstance(name, str) or not 1 <= len(name) <= 128 or '\0' in name for name in row)):
            raise ValueError("Invalid tuft source skin influences")
        if sum(_finite(value, "skin weight", 0, 1) for value in row.values()) <= 0:
            raise ValueError("Tuft source skin weights must have positive support")
    cumulative, usable, total = [], [], 0.
    for ordinal, triangle in enumerate(triangles):
        if (not isinstance(triangle, (list, tuple)) or len(triangle) != 3
                or any(type(i) is not int or not 0 <= i < len(vertices) for i in triangle)):
            raise ValueError("Invalid tuft emitter triangle")
        a, b, c = (vertices[i] for i in triangle)
        cross = _cross(tuple(b[i]-a[i] for i in range(3)), tuple(c[i]-a[i] for i in range(3)))
        area = math.hypot(*cross)/2
        if not math.isfinite(area):
            raise ValueError("Tuft emitter area is nonfinite")
        if area > 1e-14:
            total += area
            cumulative.append(total)
            usable.append((ordinal, triangle, _unit(cross)))
    if not math.isfinite(total) or settings["count"] and not usable:
        raise ValueError("Tuft coat requires finite nondegenerate emitter area")
    anchors = None
    if settings.get('root_placement') == 'skin-consistent-v1':
        if __package__:
            from .fur_skin_roots import select_skin_roots
        else:
            from fur_skin_roots import select_skin_roots
        anchors = select_skin_roots(vertices, triangles, weights, settings['count'], settings['seed'])
        face_normals = {ordinal: normal for ordinal, _, normal in usable}
    rng = random.Random(settings["seed"])
    layers = [index % 3 != 2 for index in range(settings["count"])]
    rng.shuffle(layers)
    output = {"vertices": [], "faces": [], "weights": [], "uv": [], "normals": [],
              "colors": [], "root_samples": []}
    sample_count = min(128, settings["count"])
    sample_indices = {round(i*(settings["count"]-1)/max(1, sample_count-1)) for i in range(sample_count)}
    summary = {"roots": settings["count"], "pile_cards": sum(layers), "tuft_cards": len(layers)-sum(layers),
               "eligible_area_m2": total, "regional_flow": prepared_flow is not None,
               "root_color_source": "Exact emitter triangle UV callback; linear RGB constant per card",
               "length_clamped": 0, "width_clamped": 0, "length_range_m": None, "width_range_m": None,
               "length_scale_range": None, "sample_records": sample_count, "maximum_sample_records": 128}
    output["root_summary"] = summary
    if anchors is not None:
        summary.update(root_placement='skin-consistent-v1', binding_selection=anchors['summary'])
    if anchors is not None or record_bindings:
        summary.setdefault('root_placement', 'area-stratified-v1')
        output['root_bindings'] = {'triangle_index': [], 'barycentric': [], 'root_inset': []}
    for index in range(settings["count"]):
        # Equal-area strata avoid spending the finite card budget on random face clusters.
        ordinal, triangle, face_normal = usable[min(bisect_left(cumulative,
            (index+rng.random())/settings["count"]*total), len(usable)-1)]
        r, fraction = math.sqrt(rng.random()), rng.random()
        bary = (1-r, r*(1-fraction), r*fraction)
        if anchors is not None:
            # Retain the existing style draws while changing only the attachment domain.
            anchor = anchors['roots'][index]
            ordinal = anchor['triangle_index']
            triangle, face_normal, bary = triangles[ordinal], face_normals[ordinal], anchor['barycentric']
        root = tuple(sum(vertices[v][axis]*f for v, f in zip(triangle, bary)) for axis in range(3))
        normal = tuple(sum(normals[v][axis]*f for v, f in zip(triangle, bary)) for axis in range(3)) if normals else face_normal
        if math.hypot(*normal) <= 1e-8:
            raise ValueError("Tuft root normals cancel at the sampled point")
        normal = _unit(normal)
        direction, amount, scale = evaluate_flow(root, prepared_flow) if evaluate_flow else (base_direction, base_comb, 1.)
        direction = _triple(direction, "regional direction", -1, 1)
        if math.hypot(*direction) <= 1e-8:
            raise ValueError("Tuft regional direction must be nonzero")
        amount, scale = _finite(amount, "regional comb", 0, .95), _finite(scale, "regional length scale", .25, 2)
        projected = tuple(direction[i]-normal[i]*sum(direction[k]*normal[k] for k in range(3)) for i in range(3))
        basis = (0., 0., 1.) if abs(normal[2]) < .9 else (0., 1., 0.)
        comb = _unit(projected) if math.hypot(*projected) > 1e-8 else _unit(_cross(normal, basis))
        across, roll = _unit(_cross(normal, comb)), rng.uniform(-.10, .10)
        side = tuple(across[i]*math.cos(roll)+normal[i]*math.sin(roll) for i in range(3))
        pile = layers[index]
        proposed_length = base_length*scale*(.58 if pile else 1.)*rng.uniform(.9, 1.1)
        proposed_width = base_width*(1.7 if pile else .9)*rng.uniform(.95, 1.05)
        length, width = min(.04, max(.001, proposed_length)), min(.02, max(.0003, proposed_width))
        inherited = {}
        for vertex, factor in zip(triangle, bary):
            for bone, weight in weights[vertex].items():
                inherited[bone] = inherited.get(bone, 0)+factor*weight
        inherited = dict(sorted(inherited.items(), key=lambda pair: (-pair[1], pair[0]))[:4])
        weight_sum = sum(inherited.values())
        inherited = {name: weight/weight_sum for name, weight in inherited.items() if weight > 0}
        color = _triple(root_color_sampler(ordinal, bary), "root UV color", 0, 1)
        tile = rng.randrange(ATLAS_TILES**2)
        tile_x, tile_y = tile % ATLAS_TILES, tile // ATLAS_TILES
        first = len(output["vertices"])
        inset = min(.0005, length*.05)
        if anchors is not None or record_bindings:
            output['root_bindings']['triangle_index'].append(ordinal)
            output['root_bindings']['barycentric'].append(list(bary))
            output['root_bindings']['root_inset'].append([normal[i]*inset for i in range(3)])
        center = tuple(root[i]-normal[i]*inset for i in range(3))
        for t, span in ((0., .96), (.5, 1.), (1., .38)):
            if t:
                # Match the native comb's per-segment direction, using two portable segments.
                lean = amount*(.30+.70*t)
                segment = _unit(tuple(normal[i]*(1-lean)+comb[i]*lean for i in range(3)))
                center = tuple(center[i]+segment[i]*length/2 for i in range(3))
            for sign in (-1, 1):
                output["vertices"].append(tuple(center[i]+side[i]*width*span*sign/2 for i in range(3)))
                output["normals"].append(normal)
                output["weights"].append(dict(inherited))
                output["colors"].append(color)
                output["uv"].append(((tile_x+(.015 if sign < 0 else .985))/ATLAS_TILES,
                                     (tile_y+.015+.97*t)/ATLAS_TILES))
        for a, b, c in ((0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)):
            pa, pb, pc = (output["vertices"][first+i] for i in (a, b, c))
            cross = _cross(tuple(pb[i]-pa[i] for i in range(3)), tuple(pc[i]-pa[i] for i in range(3)))
            if sum(cross[i]*normal[i] for i in range(3)) < 0:
                b, c = c, b
            output["faces"].append((first+a, first+b, first+c))
        summary["length_clamped"] += length != proposed_length
        summary["width_clamped"] += width != proposed_width
        for name, value in (("length_range_m", length), ("width_range_m", width), ("length_scale_range", scale)):
            prior = summary[name]
            summary[name] = [min(prior[0], value), max(prior[1], value)] if prior else [value, value]
        if index in sample_indices:
            output["root_samples"].append({"root_index": index, "triangle_index": ordinal,
                "barycentric": bary, "position": root, "normal": normal, "direction": direction,
                "comb": amount, "length_scale": scale, "layer": "pile" if pile else "tuft",
                "length_m": length, "width_m": width, "root_inset_m": inset})
            if anchors is not None:
                output['root_samples'][-1]['root_binding_kind'] = anchor['binding_kind']
                if 'vertex_index' in anchor:
                    output['root_samples'][-1]['source_vertex_index'] = anchor['vertex_index']
    return output


def _tuft_atlas(color, seed):
    import numpy as np

    if not isinstance(color, (list, tuple)) or len(color) not in (3, 4):
        raise ValueError("Tuft neutral atlas color requires RGB or RGBA")
    if _triple(color[:3], "neutral atlas color", 0, 1) != (1., 1., 1.):
        raise ValueError("Tuft atlas must be neutral white; source color belongs to exact root vertex colors")
    if type(seed) is not int or not 0 <= seed <= 2**31-1:
        raise ValueError("Invalid tuft atlas seed")
    rng = random.Random(seed)
    tile_size = ATLAS_SIZE//ATLAS_TILES
    yy, xx = np.mgrid[0:tile_size, 0:tile_size].astype(np.float32)/(tile_size-1)
    pixels = np.ones((ATLAS_SIZE, ATLAS_SIZE, 4), dtype=np.float32)
    pixels[:, :, 3] = 0
    for tile in range(ATLAS_TILES**2):
        mask = np.zeros_like(xx)
        for index in range(3):
            start, tip = .14+index*.36+rng.uniform(-.004, .004), rng.uniform(.86, .97)
            center = start+rng.uniform(-.008, .008)*yy+rng.uniform(-.008, .008)*np.sin(yy*math.pi)
            radius = rng.uniform(.100, .117)*np.maximum(0., 1-yy/tip)**.38
            coverage = np.clip((radius-np.abs(xx-center))*tile_size+.5, 0, 1)
            coverage *= ((yy > .025) & (yy < tip) & (xx > .025) & (xx < .975)).astype(np.float32)
            np.maximum(mask, coverage, out=mask)
        row, col = tile//ATLAS_TILES, tile % ATLAS_TILES
        pixels[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size, 3] = mask
    return pixels


def atlas_pixels(color, seed=193, profile="fine-v1"):
    """Native-resolution authored fibers with a transparent border around each tile."""
    import numpy as np

    if profile not in ATLAS_PROFILES:
        raise ValueError("Unknown fur atlas profile")
    if profile == "tuft-v1":
        return _tuft_atlas(color, seed)
    distance = profile == "distance-v1"
    fibers = ATLAS_PROFILES[profile]["fibers"]
    rng = random.Random(seed)
    tile_size = ATLAS_SIZE//ATLAS_TILES
    yy, xx = np.mgrid[0:tile_size, 0:tile_size].astype(np.float32)/(tile_size-1)
    pixels = np.ones((ATLAS_SIZE, ATLAS_SIZE, 4), dtype=np.float32)
    pixels[:, :, :3] = color[:3]
    pixels[:, :, 3] = 0
    for tile in range(ATLAS_TILES**2):
        mask = np.zeros_like(xx)
        tint = np.ones_like(xx)
        for index in range(fibers):
            start = .055+index*.89/(fibers-1)+rng.uniform(-.012, .012)
            tip = rng.uniform(.69, .96)
            lean, curve = rng.uniform(-.045, .045), rng.uniform(-.04, .04)
            center = start+lean*yy+curve*np.sin(yy*math.pi)
            # Wider tapered locks survive minification; fine-v1 retains its original bytes.
            radius = rng.uniform(.022, .034) if distance else rng.uniform(.0055, .009)
            radius *= np.maximum(.08, 1-yy/tip)**.7
            coverage = np.clip((radius-np.abs(xx-center))*tile_size+.5, 0, 1)
            coverage *= ((yy > .025) & (yy < tip)).astype(np.float32)
            active = coverage > mask
            tint[active] = rng.uniform(.88, 1.08)
            np.maximum(mask, coverage, out=mask)
        row, col = tile//ATLAS_TILES, tile % ATLAS_TILES
        region = pixels[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size]
        region[:, :, :3] = np.clip(np.asarray(color[:3], dtype=np.float32)[None, None, :]
                                   *tint[:, :, None]*(.85+.18*yy[:, :, None]), 0, 1)
        region[:, :, 3] = mask
    return pixels


def attach_fur_material(name, source, output):
    import bpy

    material = bpy.data.materials.new(name+"_cutout_fur")
    material.use_nodes = True
    profile = source.get("fur_atlas", "fine-v1")
    # glTF flips back-face normals; that inverts these inherited outward shell normals.
    material.use_backface_culling = profile in ("distance-v1", "tuft-v1")
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = .90 if profile == "tuft-v1" else .78
    bsdf.inputs["Metallic"].default_value = 0
    settings = ATLAS_PROFILES[profile]
    pixels = atlas_pixels(source["color"], profile=profile)
    image = bpy.data.images.new(name+"_fur_atlas", width=ATLAS_SIZE, height=ATLAS_SIZE, alpha=True)
    image.colorspace_settings.name = "sRGB"
    image.pixels.foreach_set(pixels.ravel())
    folder = Path(output)/"textures"
    folder.mkdir(exist_ok=True)
    path = folder/(name+"-fur-atlas.png")
    image.filepath_raw, image.file_format = str(path), "PNG"
    image.save()
    image.pack()
    texture = nodes.new("ShaderNodeTexImage")
    texture.image, texture.extension = image, "CLIP"
    links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    clip = nodes.new("ShaderNodeMath")
    clip.operation = "GREATER_THAN"
    clip.inputs[1].default_value = settings["cutoff"]
    links.new(texture.outputs["Alpha"], clip.inputs[0])
    links.new(clip.outputs[0], bsdf.inputs["Alpha"])
    report = {"material": material.name, "kind": "fur_atlas", "path": str(path.relative_to(output)),
              "resolution": ATLAS_SIZE, "pattern": f"{settings['fibers']} curved fibers per tile; sixteen seeded tiles",
              "profile": profile, "seed": 193, "alpha_mode": "MASK", "alpha_cutoff": settings["cutoff"],
              "double_sided": not material.use_backface_culling,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return material, report
