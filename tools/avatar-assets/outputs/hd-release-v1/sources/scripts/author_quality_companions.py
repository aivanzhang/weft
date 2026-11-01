"""Author two detailed, editable scene recipes without a modeling service."""

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def mul(a, k):
    return [x * k for x in a]


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def norm(a):
    return mul(a, 1 / math.sqrt(sum(x*x for x in a)))


def rounded(a):
    return [round(x, 6) for x in a]


def color(hex_value, roughness=.5, metallic=0, surface=None):
    rgb = [int(hex_value[i:i+2], 16)/255 for i in (0, 2, 4)]
    value = {"color": rgb + [1], "roughness": roughness, "metallic": metallic}
    if surface:
        value["surface"] = surface
    return value


def surface(pattern, strength, seed):
    return {"pattern": pattern, "resolution": 512, "strength": strength, "seed": seed}


def primitive(name, shape, at, scale, mat, bone, rotation=None, **kwargs):
    p = {"name": name, "shape": shape, "location": rounded(at), "scale": rounded(scale),
         "rotation": rounded(rotation or [0, 0, 0]), "material": mat, "bone": bone}
    if shape != "cube":
        p.update(segments=32, rings=20)
    p.update(kwargs)
    return p


def mesh(name, vertices, faces, mat, bone, uv=None, weights=None):
    p = primitive(name, "mesh", [0, 0, 0], [1, 1, 1], mat, bone)
    p.pop("segments", None)
    p.pop("rings", None)
    p.update(vertices=[rounded(v) for v in vertices], faces=faces)
    if uv is not None:
        p["uv"] = [rounded(x) for x in uv]
    if weights is not None:
        p["weights"] = weights
    return p


def fur(p, count, length=.012, width=.001, seed=1):
    p["fur"] = {"count": count, "length": length, "width": width, "seed": seed}
    return p


def loft(name, rings, mat, bone, segments=32, ring_weights=None):
    """Z-oriented oval sections preserve deliberate silhouette and UV seams."""
    verts, faces, uv, weights = [], [], [], []
    for j, (z, x, y, rx, ry) in enumerate(rings):
        for k in range(segments + 1):
            angle = 2 * math.pi * k / segments
            verts.append([x + rx*math.cos(angle), y + ry*math.sin(angle), z])
            uv.append([k/segments, j/(len(rings)-1)])
            weights.append(deepcopy(ring_weights[j] if ring_weights else {bone: 1}))
    for j in range(len(rings)-1):
        for k in range(segments):
            a = j*(segments+1) + k
            faces.append([a, a+1, a+segments+2, a+segments+1])
    for j, flip in [(0, True), (len(rings)-1, False)]:
        z, x, y, _, _ = rings[j]
        index = len(verts)
        verts.append([x, y, z])
        uv.append([.5, .5])
        weights.append(deepcopy(ring_weights[j] if ring_weights else {bone: 1}))
        for k in range(segments):
            a = j*(segments+1) + k
            faces.append([index, a+1, a] if flip else [index, a, a+1])
    return mesh(name, verts, faces, mat, bone, uv, weights)


def tube(name, points, radii, mat, bone, segments=14, weights=None, closed=False):
    verts, faces, uv, vweights = [], [], [], []
    for j, point in enumerate(points):
        tangent = norm(sub(points[min(j+1, len(points)-1)], points[max(j-1, 0)]))
        ref = [0, 1, 0] if abs(tangent[1]) < .9 else [1, 0, 0]
        x_axis = norm(cross(ref, tangent))
        y_axis = cross(tangent, x_axis)
        radius = radii[j] if isinstance(radii, list) else radii
        rx, ry = radius if isinstance(radius, tuple) else (radius, radius)
        for k in range(segments+1):
            theta = math.tau*k/segments
            verts.append(add(point, add(mul(x_axis, rx*math.cos(theta)), mul(y_axis, ry*math.sin(theta)))))
            uv.append([k/segments, j/(len(points)-1)])
            vweights.append(deepcopy(weights[j] if weights else {bone: 1}))
    for j in range(len(points)-1):
        for k in range(segments):
            a = j*(segments+1)+k
            faces.append([a, a+1, a+segments+2, a+segments+1])
    for j, reverse in [(0, True), (len(points)-1, False)]:
        v = len(verts)
        verts.append(points[j])
        uv.append([.5, .5])
        vweights.append(deepcopy(weights[j] if weights else {bone: 1}))
        for k in range(segments):
            a = j*(segments+1)+k
            faces.append([v, a+1, a] if reverse else [v, a, a+1])
    return mesh(name, verts, faces, mat, bone, uv, vweights)


def soft_limb(name, start, joint, end, start_radius, joint_radius, end_radius, bone_a, bone_b, mat, hand_bone=None):
    points, radii, weights = [], [], []
    controls = [start, joint, end]
    for half in range(2):
        a, b = controls[half:half+2]
        for n in range(6):
            if half and not n:
                continue
            t = n/5
            point = add(mul(a, 1-t), mul(b, t))
            point[1] -= .016*math.sin(math.pi*t)
            points.append(point)
            r0, r1 = (start_radius, joint_radius) if not half else (joint_radius, end_radius)
            radii.append(r0*(1-t)+r1*t+.016*math.sin(math.pi*t))
            blend = max(0, min(1, (half+t-.65)/.7))
            w = {bone_a: 1-blend, bone_b: blend}
            if hand_bone and half == 1 and t > .65:
                h = (t-.65)/.35
                w = {bone_b: 1-h, hand_bone: h}
            weights.append({key: round(value, 6) for key, value in w.items() if value > 0})
    return tube(name, points, radii, mat, bone_a, segments=24, weights=weights)


def arc(name, center, rx, rz, start, end, y, radius, mat, bone, steps=20):
    points = [[center[0]+rx*math.cos(t), y, center[2]+rz*math.sin(t)]
              for t in [start+(end-start)*i/steps for i in range(steps+1)]]
    return tube(name, points, radius, mat, bone, segments=8)


def smooth_sections(values, steps=4):
    result = []
    for j in range(len(values)-1):
        a, b, c, d = [values[max(0, min(len(values)-1, n))] for n in (j-1, j, j+1, j+2)]
        for k in range(steps):
            t = k/steps
            result.append([.5*((2*y)+(-x+z)*t+(2*x-5*y+4*z-w)*t*t+(-x+3*y-3*z+w)*t*t*t)
                           for x, y, z, w in zip(a, b, c, d)])
    result.append(values[-1])
    return result


def rounded_arm(side, sign, bones):
    upper, forearm, hand = [f"{name}_{side}" for name in ("upper_arm", "forearm", "hand")]
    shoulder, elbow, wrist = [bones[name]["head"] for name in (upper, forearm, hand)]
    controls = [
        [sign*.21, .014, 1.50, .018],
        [sign*.29, .005, 1.515, .10],
        [*shoulder, .145],
        [sign*.485, -.038, 1.393, .147],
        [*elbow, .122],
        [sign*.43, -.27, elbow[2]+(wrist[2]-elbow[2])*.55, .113],
        [*wrist, .096],
        [wrist[0]-sign*.025, wrist[1]-.028, wrist[2]-.004, .018],
    ]
    control_weights = [
        {"torso": 1}, {"torso": .55, upper: .45}, {"torso": .1, upper: .9},
        {upper: .94, forearm: .06}, {upper: .5, forearm: .5},
        {forearm: .92, hand: .08}, {hand: 1}, {hand: 1},
    ]
    rows = smooth_sections(controls, 4)
    weights = []
    for j in range(len(rows)):
        n, t = min(j//4, len(control_weights)-2), (j % 4)/4
        if j == len(rows)-1:
            weights.append(control_weights[-1])
            continue
        keys = set(control_weights[n]) | set(control_weights[n+1])
        weights.append({key: round(control_weights[n].get(key, 0)*(1-t)+control_weights[n+1].get(key, 0)*t, 6)
                        for key in sorted(keys) if control_weights[n].get(key, 0)*(1-t)+control_weights[n+1].get(key, 0)*t > 0})
    return tube(f"continuous_arm_{side}", [row[:3] for row in rows], [row[3] for row in rows],
                "coat", upper, segments=28, weights=weights)


def fox():
    old = json.loads((ROOT / "examples/scenes/fox-craftsperson-v2.json").read_text())
    d = deepcopy(old)
    d.update(schema_version=2, id="fox-quality-v4", output_profile="general")
    d["source_prompt"] = old["prompt"]
    d["prompt"] = (
        "Refine the existing orange-and-cream fox craftsperson into a premium, expressive, fully modeled "
        "storybook character. Preserve the teal scarf, navy sketchbook, gold pencil, twenty named joints and "
        "the four authored actions. Replace ball-separated arms and knees with continuous weighted skin; "
        "shape a broad-cheeked tapered face, cupped pointed ears, curved fluffy tail, glossy amber eyes with "
        "sclera, eyelids and small catchlights, a soft triangular nose and a subtle smile. Add short geometric "
        "fur and embedded procedural fur textures, cloth folds and scarf fringes, a bound book with page "
        "edges, elastic strap and cover emblem, and a real wood/graphite pencil tip. Use physically based "
        "opaque surfaces; preserve every visible detail in the animated GLB, not just the portrait. Author "
        "the back, underside, topology and skin weights explicitly; do not claim recovered anatomy or fur simulation. "
        "Refinement after v3 review: bury rounded shoulder roots inside the torso, blend shoulder weights to the torso, "
        "smooth the complete elbow centerline and head cross sections, and replace twin rounded cheek lobes with a "
        "single tapered fox muzzle. Retain the denser 12,700-strand fur pass and all four action meanings."
    )
    d["style"] = "Detailed warm storybook fox: soft short fur, expressive eyes, tailored teal scarf and crafted props"
    d["authored_assumptions"] = [
        "This is a new authored 3D interpretation of the existing original fox design, not recovered geometry or a neural text-to-3D result.",
        "Complete hidden surfaces and twenty-bone rest layout are authored; the original fox left/right bone names are preserved as identifiers.",
        "The continuous arm and leg lofts use explicit multi-bone skin blends around elbows and knees; they are not muscle or cloth simulations.",
        "The shoulder roots are rounded and buried within the torso, with torso/arm weight blending; skin is not a single watertight anatomical retopology.",
        "Short fur is bounded opaque tapered triangle geometry attached with interpolated skin weights, plus embedded procedural albedo/normal textures. It is not a strand groom or dynamically simulated fur.",
        "The scarf, eyelids, fingers and face are sculpted fixed geometry without independent expressions or cloth dynamics.",
        "The book drawings and pencil strokes remain authored gestures: no new ink or page simulation is claimed.",
        "This PBR browser-quality variant is separate from Weft's restricted diffuse-only importer profile.",
        "The preserved four skeletal actions are authored animation, not learned behavior or Weft physics."
    ]
    d["materials"] = {
        "coat": color("D66C2E", .83, surface=surface("fur", .28, 51)),
        "cream": color("F5E6CA", .88, surface=surface("fur", .2, 52)),
        "dark": color("382C2E", .75, surface=surface("fur", .18, 53)),
        "scarf": color("287F7A", .79, surface=surface("brushed", .13, 54)),
        "book": color("253B59", .51),
        "gold": color("DDB75C", .27, .58),
        "eye_white": color("FFFAEB", .19),
        "iris": color("9E5C28", .2),
        "pupil": color("14131A", .12),
        "nose": color("312125", .31),
        "ear_inner": color("D7A189", .9),
        "paper": color("F1E4CA", .92),
        "page_edge": color("BDAA87", .91),
        "scarf_seam": color("61A7A0", .8),
        "pencil_wood": color("D8B580", .8),
        "graphite": color("30343D", .38, .1),
    }
    p = []
    p.append(fur(loft("pear_shaped_body", [
        [.68, 0, .015, .12, .12], [.72, 0, .02, .25, .2], [.85, 0, .015, .35, .275],
        [1.02, 0, 0, .385, .29], [1.22, 0, -.005, .37, .285], [1.4, 0, .005, .31, .24],
        [1.53, 0, .005, .25, .2], [1.65, 0, 0, .17, .15]], "coat", "torso", 40), 900, .012, .0009, 91))
    p.append(fur(primitive("cream_chest", "sphere", [0, -.245, 1.19], [.262, .078, .36], "cream", "torso", segments=40, rings=28), 350, .009, .0008, 92))
    p.append(fur(loft("sculpted_head", smooth_sections([
        [1.69, 0, -.02, .16, .16], [1.75, 0, -.025, .32, .26], [1.86, 0, -.025, .445, .32],
        [2.0, 0, -.025, .47, .35], [2.16, 0, -.015, .43, .335],
        [2.3, 0, 0, .33, .27], [2.39, 0, .015, .17, .15], [2.415, 0, .015, .04, .04],
        [2.42, 0, .015, .003, .003]], 4), "coat", "head", 64), 1400, .012, .0008, 93))
    muzzle = smooth_sections([
        [0, -.24, 1.865, .145, .27], [0, -.31, 1.865, .175, .29],
        [0, -.4, 1.878, .145, .25], [0, -.49, 1.905, .10, .177],
        [0, -.555, 1.939, .052, .092], [0, -.58, 1.948, .019, .034]], 4)
    p.append(fur(tube("tapered_fox_muzzle", [row[:3] for row in muzzle], [(row[3], row[4]) for row in muzzle],
                      "cream", "head", 40), 440, .01, .0007, 94))
    for side, sign in (("left", -1), ("right", 1)):
        p.append(fur(primitive(f"outer_cheek_{side}", "sphere", [sign*.34, -.12, 1.87], [.13, .21, .13],
                              "cream", "head", rotation=[0, sign*.25, 0]), 170, .014, .0009, 97+(sign+1)))
        p.append(primitive(f"eye_socket_{side}", "sphere", [sign*.202, -.319, 2.108], [.112, .047, .131], "nose", "head"))
        p.append(primitive(f"eye_white_{side}", "sphere", [sign*.202, -.348, 2.109], [.092, .041, .111], "eye_white", "head", segments=40, rings=28))
        p.append(primitive(f"iris_{side}", "sphere", [sign*.188, -.383, 2.106], [.057, .019, .073], "iris", "head"))
        p.append(primitive(f"pupil_{side}", "sphere", [sign*.183, -.398, 2.107], [.033, .012, .051], "pupil", "head"))
        p.append(primitive(f"catchlight_{side}", "sphere", [sign*.181-.014, -.409, 2.137], [.012, .005, .016], "eye_white", "head", segments=16, rings=12))
        p.append(arc(f"upper_lid_{side}", [sign*.202, 0, 2.108], .097, .119, .12, math.pi-.12, -.364, .009, "coat", "head"))
        p.append(arc(f"brow_{side}", [sign*.202, 0, 2.19], .108, .08, .4, math.pi-.4, -.297, .025, "coat", "head", 12))
        ear_x = sign*.30
        ear = loft(f"cupped_ear_{side}", [
            [2.19, ear_x, .012, .15, .105], [2.31, ear_x+sign*.017, .018, .15, .075],
            [2.45, ear_x+sign*.035, .025, .108, .048], [2.58, ear_x+sign*.052, .034, .045, .022],
            [2.65, ear_x+sign*.062, .042, .005, .005]], "coat", f"ear_{side}", 28)
        p.append(fur(ear, 180, .01, .0008, 102+(sign+1)))
        inner = loft(f"inset_ear_{side}", [
            [2.27, ear_x, -.069, .082, .012], [2.34, ear_x+sign*.017, -.054, .09, .015],
            [2.45, ear_x+sign*.035, -.024, .063, .012], [2.56, ear_x+sign*.049, .011, .009, .004]],
            "ear_inner", f"ear_{side}", 20)
        p.append(inner)
        bones = {b["name"]: b for b in d["bones"]}
        start, joint, end = [bones[f"{prefix}_{side}"]["head"] for prefix in ("upper_arm", "forearm", "hand")]
        p.append(fur(rounded_arm(side, sign, bones), 310, .01, .0008, 110+(sign+1)))
        p.append(fur(primitive(f"paw_{side}", "sphere", end, [.12, .105, .116], "dark", f"hand_{side}"), 90, .007, .0007, 114+(sign+1)))
        for finger in range(3):
            p.append(primitive(f"finger_{side}_{finger}", "sphere", [end[0]+(finger-1)*.047, end[1]-.072, end[2]-.03],
                               [.029, .047, .058], "dark", f"hand_{side}", segments=16, rings=12))
        start, joint, end = [bones[f"{prefix}_{side}"]["head"] for prefix in ("thigh", "shin", "foot")]
        p.append(fur(soft_limb(f"continuous_leg_{side}", start, joint, end, .165, .13, .11,
                               f"thigh_{side}", f"shin_{side}", "coat", f"foot_{side}"), 220, .01, .0008, 116+(sign+1)))
        p.append(fur(primitive(f"dark_sock_{side}", "sphere", [sign*.23, -.023, .33], [.135, .135, .18], "dark", f"shin_{side}"), 100, .008, .0007, 119+(sign+1)))
        p.append(fur(primitive(f"soft_foot_{side}", "sphere", [sign*.23, -.105, .15], [.17, .24, .13], "dark", f"foot_{side}"), 110, .008, .0007, 122+(sign+1)))
        for toe in range(3):
            p.append(primitive(f"toe_{side}_{toe}", "sphere", [sign*.23+(toe-1)*.07, -.278, .118], [.041, .064, .065],
                               "dark", f"foot_{side}", segments=16, rings=12))
    p.append(primitive("soft_triangular_nose", "sphere", [0, -.585, 1.952], [.074, .048, .05], "nose", "head", segments=32, rings=20))
    def muzzle_front(x, z):
        candidates = [row[1] for row in smooth_sections(muzzle, 4)
                      if (x/row[4])**2 + ((z-row[2])/row[3])**2 <= 1]
        return min(candidates)-.006 if candidates else -.4
    smile_points = [[x, muzzle_front(x, z), z] for x, z in [(-.18, 1.84), (-.12, 1.821), (-.06, 1.815),
                                                          (0, 1.827), (.06, 1.815), (.12, 1.821), (.18, 1.84)]]
    p.append(tube("smile", smile_points, .0045, "nose", "head", 8))
    p.append(tube("philtrum", [[0, muzzle_front(0, z), z] for z in (1.926, 1.9, 1.865, 1.827)], .0035, "nose", "head", 8))
    tail_points = [[0, .18, .98], [.18, .26, .97], [.39, .31, .99], [.62, .34, 1.04], [.82, .36, 1.15], [.98, .36, 1.33]]
    p.append(fur(tube("curved_tail", tail_points, [.1, .155, .195, .21, .18, .105], "coat", "tail", 32), 600, .018, .001, 127))
    p.append(fur(tube("tapered_tail_tip", [[.79, .36, 1.13], [.89, .36, 1.23], [.98, .36, 1.34], [1.03, .35, 1.45]],
                       [.183, .155, .105, .005], "cream", "tail", 28), 300, .016, .0009, 128))
    collar_points = [[.295*math.cos(a), .22*math.sin(a), 1.64+.017*math.sin(2*a)]
                     for a in [math.tau*i/40 for i in range(41)]]
    p.append(tube("folded_scarf_collar", collar_points, .068, "scarf", "torso", 12))
    scarf_ring = [[.30*math.cos(a), .222*math.sin(a), 1.652+.017*math.sin(2*a)]
                  for a in [math.tau*i/40 for i in range(41)]]
    p.append(tube("collar_topstitch", scarf_ring, .0055, "scarf_seam", "torso", 6))
    p.append(primitive("scarf_knot", "sphere", [-.13, -.24, 1.59], [.094, .065, .083], "scarf", "torso", rotation=[0, .3, -.2]))
    p.append(tube("draped_scarf", [[-.15, -.265, 1.57], [-.16, -.315, 1.48], [-.14, -.335, 1.35], [-.12, -.35, 1.2], [-.145, -.355, 1.11]],
                  [(.075, .022), (.077, .025), (.071, .022), (.07, .021), (.072, .017)], "scarf", "torso", 12))
    for k in range(5):
        p.append(tube(f"scarf_fringe_{k}", [[-.2+k*.026, -.355, 1.12], [-.202+k*.026, -.36, 1.063]], .006, "scarf_seam", "torso", 6))
    old_parts = {x["name"]: x for x in old["parts"]}
    for name in ("sketchbook_cover", "sketchbook_pages"):
        q = deepcopy(old_parts[name])
        q.update(bevel=.009 if "cover" in name else .004, bevel_segments=3)
        if "pages" in name:
            q["material"] = "paper"
        p.append(q)
    for z in range(7):
        p.append(primitive(f"bound_page_line_{z}", "cube", [-.16, -.555, .99+z*.037], [.205, .002, .002],
                           "page_edge", "sketchbook", rotation=[.08, 0, -.12]))
    p.append(primitive("book_spine", "cube", [-.409, -.491, 1.12], [.018, .048, .201], "gold", "sketchbook", bevel=.008, bevel_segments=3))
    p.append(primitive("book_clasp", "cube", [.017, -.559, 1.12], [.019, .006, .153], "book", "sketchbook", bevel=.004, bevel_segments=2))
    p.append(primitive("cover_emblem", "sphere", [-.16, -.562, 1.14], [.049, .005, .055], "gold", "sketchbook", segments=24, rings=16))
    shaft = deepcopy(old_parts["pencil_shaft"])
    shaft["segments"] = 12
    shaft["rings"] = 12
    p.append(shaft)
    tip = deepcopy(old_parts["pencil_tip"])
    tip["material"] = "pencil_wood"
    tip["segments"] = 24
    tip["rings"] = 12
    p.append(tip)
    p.append(primitive("graphite_point", "cone", [-.148, -.586, 1.116], [.01, .01, .016], "graphite", "pencil", rotation=tip["rotation"], segments=16, rings=12))
    for part in p:
        if "fur" in part:
            part["fur"]["count"] *= 2
            part["fur"]["width"] *= 2
            part["fur"]["length"] *= 1.2
    d["parts"] = p
    # Softer elbow bends prevent the new blended skin from folding like a toy hinge.
    for state in d["states"]:
        if state["id"] != "milestone_level_up":
            continue
        for key in state["keyframes"]:
            for side in ("left", "right"):
                key["bones"][f"upper_arm_{side}"]["rotation"][1] *= 2.5/1.7
                key["bones"][f"forearm_{side}"]["rotation"][1] *= 1.05/2.3
                key["bones"][f"hand_{side}"]["rotation"] = [0, 0, 0]
    d["camera"] = {"target": [.08, -.01, 1.35], "position": [2.7, -7.5, 2.85], "orthographic_scale": 3.12}
    d["background"] = [.91, .925, .93, 1]
    return d


def shade_shell(name, center, rotation, material, bone, inner=False):
    profile = [(-.21, .375), (-.18, .372), (-.11, .34), (.01, .27), (.11, .19), (.17, .12), (.195, .08)]
    if inner:
        profile = [(z+.012, r-.015) for z, r in profile]
    p = loft(name, [[z, 0, 0, r, r] for z, r in profile], material, bone, 48)
    # A thin open bell, not a cone with a face covering the light aperture.
    p["faces"] = p["faces"][:(len(profile)-1)*48]
    p["vertices"] = p["vertices"][:-2]
    p["uv"] = p["uv"][:-2]
    p["weights"] = p["weights"][:-2]
    if inner:
        p["faces"] = [list(reversed(face)) for face in p["faces"]]
    p["location"] = center
    p["rotation"] = rotation
    return p


def lamp():
    old = json.loads((ROOT / "examples/scenes/articulated-lamp-v2.json").read_text())
    d = deepcopy(old)
    d.update(schema_version=2, id="lamp-quality-v3", output_profile="general")
    d["source_prompt"] = old["prompt"]
    d["prompt"] = (
        "Refine the original burgundy-and-brass animated workshop lamp as a complete high-quality designer "
        "object. Preserve its articulated two-link arm, expressive two-eye base display, eight named joints, "
        "four actions, sketch pad and independently animated circular tracing stylus. Replace the solid cone "
        "with an open, curved bell shade with an interior reflector, rim and real bulb. Model beveled enamel "
        "base tiers, layered metal hinge disks, axle caps with screw slots, paired arm rails, small springs "
        "and power-cord sections that stay attached during existing motion. Distinguish brushed brass, "
        "glossy burgundy enamel, dark rubber, glassy warm diffuser and paper through opaque PBR materials. "
        "Preserve every visible feature in the GLB with actual skin weights; no render-only props or light "
        "simulation. Complete the back and underside rather than relying on one camera angle."
    )
    d["style"] = "Premium miniature task lamp: burgundy enamel, brushed brass, layered working hinges and a hollow bell shade"
    d["authored_assumptions"] = [
        "This is an original detailed interpretation of the existing lamp, not a scanned or recovered industrial product.",
        "The eight named joints and their rest coordinates are preserved; all hidden hardware, cord routing and shade interior are authored.",
        "The open shade includes separate outer and interior wall surfaces, aperture trim, rear cap and bulb; the warm diffuser is geometry, not an emitted simulated light.",
        "Arms, hinge caps, springs and cord sections use explicit rigid or blended skin weights; no mechanical spring, cable or electrical simulation is claimed.",
        "The pad retains a fixed circular guide; the stylus traces gestures without dynamically creating ink.",
        "Opaque PBR metal/roughness and embedded procedural brushed textures target the browser, not Weft's restricted diffuse-only profile.",
        "The four skeletal actions are authored local animations and do not duplicate a simulator or change Weft actions or rewards."
    ]
    d["materials"] = {
        "body": color("753C4B", .24, .24),
        "brass": color("CFA253", .28, .79, surface("brushed", .16, 71)),
        "joint": color("303442", .34, .58),
        "bulb": color("FFE7B3", .18),
        "paper": color("F0E6D2", .95),
        "ink": color("406878", .85),
        "rubber": color("1F2630", .82),
        "reflector": color("E4D5B7", .32, .66),
        "steel": color("A9ADB3", .22, .85),
        "display": color("121F2B", .16),
        "page_line": color("B5B7AB", .85),
    }
    old_parts = {p["name"]: p for p in old["parts"]}
    p = []
    retain = ["base_body", "base_upper_ring", "turntable", "upright_stem", "base_display", "display_eye_left", "display_eye_right",
              "work_pad_cover", "work_pad_paper", "pad_blueprint_ring", "trace_mount", "trace_support", "trace_hub", "trace_radial_arm",
              "trace_pen_holder", "trace_pen_shaft", "trace_pen_tip", "trace_pen_cap"]
    for name in retain:
        q = deepcopy(old_parts[name])
        if q["shape"] in ("cube", "cylinder"):
            q["bevel"] = .018 if name == "base_body" else .005
            q["bevel_segments"] = 3
        if q["shape"] != "cube":
            q.update(segments=48 if "base" in name else 32, rings=20)
        if name == "base_display":
            q["material"] = "display"
        p.append(q)
    p.append(primitive("lower_base_plinth", "cylinder", [0, 0, .052], [.464, .464, .027], "rubber", "root", bevel=.014, bevel_segments=3, segments=64))
    p.append(primitive("base_lower_trim", "cylinder", [0, 0, .079], [.478, .478, .012], "brass", "root", bevel=.006, bevel_segments=3, segments=64))
    p.append(primitive("base_upper_step", "cylinder", [0, 0, .278], [.405, .405, .017], "body", "root", bevel=.011, bevel_segments=3, segments=64))
    p.append(primitive("turntable_bearing", "cylinder", [0, 0, .35], [.19, .19, .018], "brass", "turntable", bevel=.006, bevel_segments=3, segments=48))
    for index, (x, y) in enumerate([(-.28, -.26), (-.28, .26), (.28, -.26), (.28, .26)]):
        p.append(primitive(f"rubber_foot_{index}", "cylinder", [x, y, .022], [.075, .075, .021], "rubber", "root", bevel=.009, bevel_segments=3))
        p.append(primitive(f"underside_screw_{index}", "cylinder", [x, y, .0025], [.022, .022, .003], "steel", "root", segments=24))
    for stage, a, b, bone in [
        ("lower", [0, 0, .58], [.3, 0, 1.25], "lower_arm"),
        ("upper", [.3, 0, 1.25], [-.1, 0, 1.9], "upper_arm")]:
        delta = sub(b, a)
        length = math.sqrt(sum(x*x for x in delta))
        rotation = [0, math.atan2(delta[0], delta[2]), 0]
        for side, y in [("front", -.095), ("back", .095)]:
            at = mul(add(a, b), .5)
            at[1] = y
            p.append(primitive(f"{stage}_enamel_rail_{side}", "cube", at, [.041, .029, length*.5], "body", bone,
                               rotation=rotation, bevel=.02, bevel_segments=3))
            inset = list(at)
            inset[1] += -.031 if y < 0 else .031
            p.append(primitive(f"{stage}_rail_inlay_{side}", "cube", inset, [.012, .004, length*.34], "brass", bone,
                               rotation=rotation, bevel=.004, bevel_segments=2))
        wire_points = []
        for i in range(12):
            t = i/11
            q = add(mul(a, 1-t), mul(b, t))
            q[1] = .15
            q[0] += .035*math.sin(math.pi*t)
            wire_points.append(q)
        p.append(tube(f"{stage}_cord", wire_points, .015, "rubber", bone, 8))
        spring_points = []
        tangent = norm(delta)
        cross_axis = norm(cross([0, 1, 0], tangent))
        for i in range(97):
            t = i/96
            base = add(mul(a, .72-.44*t), mul(b, .28+.44*t))
            phase = t*math.tau*8
            point = add(base, mul(cross_axis, .032*math.cos(phase)))
            point[1] += .033*math.sin(phase)
            spring_points.append(point)
        p.append(tube(f"{stage}_coil_spring", spring_points, .006, "steel", bone, 6))
    for name, xyz, radius, bone in [("base", [0, 0, .58], .13, "lower_arm"),
                                    ("elbow", [.3, 0, 1.25], .133, "upper_arm"),
                                    ("head", [-.1, 0, 1.9], .105, "shade")]:
        p.append(primitive(f"{name}_hinge_barrel", "cylinder", xyz, [radius, radius, .121], "joint", bone,
                           rotation=[math.pi/2, 0, 0], bevel=.009, bevel_segments=3, segments=40))
        for side, sign in [("front", -1), ("back", 1)]:
            cap = add(xyz, [0, sign*.128, 0])
            p.append(primitive(f"{name}_hinge_washer_{side}", "cylinder", cap, [radius*.9, radius*.9, .016], "body", bone,
                               rotation=[math.pi/2, 0, 0], bevel=.007, bevel_segments=3, segments=40))
            cap[1] += sign*.019
            p.append(primitive(f"{name}_axle_cap_{side}", "cylinder", cap, [radius*.52, radius*.52, .018], "brass", bone,
                               rotation=[math.pi/2, 0, 0], bevel=.007, bevel_segments=3, segments=32))
            p.append(primitive(f"{name}_screw_slot_{side}", "cube", add(cap, [0, sign*.019, 0]), [radius*.29, .002, .004], "joint", bone,
                               rotation=[0, .45, 0], bevel=.002, bevel_segments=2))
    center, rotation = [-.1, -.045, 2.035], [-.38, 0, 0]
    p.append(shade_shell("enamel_bell_shade", center, rotation, "body", "shade"))
    p.append(shade_shell("interior_reflector", center, rotation, "reflector", "shade", inner=True))
    p.append(primitive("shade_lip", "torus", [-.1, -.123, 1.84], [.302, .302, .019], "brass", "shade", rotation=rotation, segments=48, rings=12))
    p.append(primitive("shade_top_cap", "sphere", [-.1, .025, 2.216], [.083, .079, .024], "brass", "shade", rotation=rotation, segments=32, rings=16))
    p.append(primitive("bulb_socket", "cylinder", [-.1, -.021, 2.093], [.075, .075, .07], "joint", "shade", rotation=rotation, bevel=.009, bevel_segments=3))
    p.append(primitive("warm_bulb", "sphere", [-.1, -.076, 1.965], [.12, .12, .15], "bulb", "shade", rotation=rotation, segments=40, rings=28))
    for i in range(8):
        angle = i*math.tau/8
        p.append(primitive(f"shade_vent_{i}", "sphere", [-.1+.107*math.cos(angle), .01+.1*math.sin(angle), 2.185],
                           [.014, .008, .026], "joint", "shade", segments=16, rings=12))
    for side, x in [("left", -.077), ("right", .077)]:
        p.append(arc(f"display_brow_{side}", [x, 0, .173], .028, .018, .25, math.pi-.25, -.475, .004, "bulb", "root", 10))
    p.append(arc("display_smile", [0, 0, .151], .032, .015, math.pi+.1, math.tau-.1, -.477, .0035, "bulb", "root", 12))
    p.append(primitive("power_button", "cylinder", [.29, -.17, .281], [.041, .041, .012], "brass", "root", bevel=.005, bevel_segments=3))
    p.append(primitive("power_button_center", "cylinder", [.29, -.17, .295], [.022, .022, .003], "joint", "root", segments=24))
    rear_points = [[0, .421, .14], [0, .49, .13], [.1, .54, .08], [.22, .56, .05], [.34, .63, .045]]
    p.append(tube("rear_power_lead", rear_points, .022, "rubber", "root", 10))
    p.append(primitive("cord_grommet", "sphere", [0, .423, .14], [.051, .025, .044], "rubber", "root", segments=24, rings=16))
    for i in range(4):
        p.append(primitive(f"pad_rule_{i}", "cube", [.53, -.73+i*.105, .11], [.232, .002, .002], "page_line", "work_pad"))
    p.append(primitive("pad_clip", "cube", [.53, -.352, .12], [.11, .024, .012], "brass", "work_pad", bevel=.009, bevel_segments=3))
    d["parts"] = p
    d["camera"] = {"target": [.1, -.04, 1.1], "position": [3.5, -7, 3.05], "orthographic_scale": 2.83}
    d["background"] = [.91, .925, .93, 1]
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("fox", "lamp"), default="fox")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for scene in (fox() if args.only == "fox" else lamp(),):
        path = ROOT / "examples/scenes" / (scene["id"] + ".json")
        if args.check:
            assert json.loads(path.read_text()) == scene, "Recipe differs from saved scene"
            print(json.dumps({"id": scene["id"], "exact_scene_match": True}))
            continue
        with path.open("x") as handle:
            handle.write(json.dumps(scene, separators=(",", ":"), ensure_ascii=False) + "\n")
        print(json.dumps({"path": str(path), "parts": len(scene["parts"]), "materials": len(scene["materials"]),
                          "control_vertices": sum(len(p.get("vertices", [])) for p in scene["parts"]),
                          "fur_strands": sum(p.get("fur", {}).get("count", 0) for p in scene["parts"]),
                          "bytes": path.stat().st_size}))


if __name__ == "__main__":
    main()
