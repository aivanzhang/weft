"""Seeded curved hair cards with inherited skinning and embedded cutout atlases."""

from bisect import bisect_left
import hashlib
import math
from pathlib import Path
import random

ATLAS_SIZE = 1024
ATLAS_TILES = 4
FIBERS_PER_TILE = 14


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _unit(v):
    length = math.sqrt(sum(x*x for x in v))
    return tuple(x/length for x in v) if length > 1e-14 else (0., 0., 1.)


def card_geometry(vertices, triangles, weights, settings, normals=None):
    """Each six-vertex ribbon carries fourteen masked fibers, not fourteen mesh hairs."""
    rng = random.Random(settings["seed"])
    cumulative, usable, total = [], [], 0.
    for triangle in triangles:
        a, b, c = (vertices[i] for i in triangle)
        cross = _cross(tuple(b[i]-a[i] for i in range(3)), tuple(c[i]-a[i] for i in range(3)))
        area = math.sqrt(sum(x*x for x in cross))/2
        if area > 1e-14:
            total += area
            cumulative.append(total)
            usable.append((triangle, _unit(cross)))
    if not usable and settings["count"]:
        raise ValueError("Hair cards require nondegenerate emitter geometry")
    output = {"vertices": [], "faces": [], "weights": [], "uv": [], "normals": []}
    direction = settings.get("direction", [0., 0., -1.])
    for _ in range(settings["count"]):
        triangle, face_normal = usable[min(bisect_left(cumulative, rng.random()*total), len(usable)-1)]
        r, fraction = math.sqrt(rng.random()), rng.random()
        bary = (1-r, r*(1-fraction), r*fraction)
        root = tuple(sum(vertices[index][axis]*f for index, f in zip(triangle, bary)) for axis in range(3))
        normal = (_unit(tuple(sum(normals[index][axis]*f for index, f in zip(triangle, bary))
                             for axis in range(3))) if normals else face_normal)
        basis = (0., 0., 1.) if abs(normal[2]) < .9 else (0., 1., 0.)
        tangent = _unit(_cross(normal, basis))
        bitangent = _cross(normal, tangent)
        angle = rng.random()*math.tau
        side = tuple(tangent[i]*math.cos(angle)+bitangent[i]*math.sin(angle) for i in range(3))
        projected = tuple(direction[i]-normal[i]*sum(direction[k]*normal[k] for k in range(3)) for i in range(3))
        comb = _unit(projected) if sum(x*x for x in projected) > 1e-8 else bitangent
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
        for t, span in ((0., .92), (.5, 1.), (1., .82)):
            # The root intersects the emitter slightly; fibers lean with the authored comb.
            center = tuple(root[i]+normal[i]*(length*t*.82-.0005)+comb[i]*length*.45*t*t for i in range(3))
            for sign in (-1, 1):
                output["vertices"].append(tuple(center[i]+side[i]*width*span*sign/2 for i in range(3)))
                output["normals"].append(normal)
                output["weights"].append(dict(inherited))
                output["uv"].append(((tile_x+(.015 if sign < 0 else .985))/ATLAS_TILES,
                                     (tile_y+.015+.97*t)/ATLAS_TILES))
        output["faces"].extend((first+a, first+b, first+c) for a, b, c in
                               ((0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)))
    return output


def atlas_pixels(color, seed=193):
    """Native-resolution authored fibers with a transparent border around each tile."""
    import numpy as np

    rng = random.Random(seed)
    tile_size = ATLAS_SIZE//ATLAS_TILES
    yy, xx = np.mgrid[0:tile_size, 0:tile_size].astype(np.float32)/(tile_size-1)
    pixels = np.ones((ATLAS_SIZE, ATLAS_SIZE, 4), dtype=np.float32)
    pixels[:, :, :3] = color[:3]
    pixels[:, :, 3] = 0
    for tile in range(ATLAS_TILES**2):
        mask = np.zeros_like(xx)
        tint = np.ones_like(xx)
        for index in range(FIBERS_PER_TILE):
            start = .055+index*.89/(FIBERS_PER_TILE-1)+rng.uniform(-.012, .012)
            tip = rng.uniform(.69, .96)
            lean, curve = rng.uniform(-.045, .045), rng.uniform(-.04, .04)
            center = start+lean*yy+curve*np.sin(yy*math.pi)
            radius = rng.uniform(.0055, .009)*np.maximum(.08, 1-yy/tip)**.7
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
    material.use_backface_culling = False
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = .78
    bsdf.inputs["Metallic"].default_value = 0
    pixels = atlas_pixels(source["color"])
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
    clip.inputs[1].default_value = .45
    links.new(texture.outputs["Alpha"], clip.inputs[0])
    links.new(clip.outputs[0], bsdf.inputs["Alpha"])
    report = {"material": material.name, "kind": "fur_atlas", "path": str(path.relative_to(output)),
              "resolution": ATLAS_SIZE, "pattern": "fourteen curved fibers per tile; sixteen seeded tiles",
              "seed": 193, "alpha_mode": "MASK", "alpha_cutoff": .45,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return material, report
