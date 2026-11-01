"""Deterministically author a complete robot scene; never overwrite an existing asset."""

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/scenes/humanoid-robot-weft-v1.json"
DESTINATION = ROOT / "examples/scenes/workshop-robot-quality-v2.json"
EDIT_PROMPT = """Refine the existing blue-and-cream workshop robot into a coherent,
polished character model, preserving its dark friendly visor, warm eyes, compact
antenna, anatomical left-hand circuit board and right-hand assembly probe. Replace
the inflated primitive appearance with deliberately shaped, closed shell geometry:
a tapered rounded-rectangle blue body, inset ivory chest armor, layered panel gaps,
recessed controls, a two-piece head enclosure and a dark visor in a substantial
gasket. Keep the face warm and legible, with softly rounded eyes and a curved smile.
Use chamfered enamel shells, satin ivory panels, dark joint seals and selectively
brushed aluminum/brass hardware, all as native opaque glTF metallic-roughness PBR.
Construct shoulder, elbow, hip, knee and wrist collars around the real pivots.
Give the hands a palm, three visible static grip fingers and a thumb; these are
authored rigid details, not newly actuated finger joints. Model a purposeful rear
service hatch, recessed vent bars, fasteners, heel guards and rubber soles so the
back and sides are complete, not implied by the portrait. Detail the circuit board
with a raised chip, fixed copper traces, pads, pins and edge hardware; make the
probe a layered grip, collar, shaft and tapered tip. Preserve all 17 named bones,
four meaningful states and rigid skin weights. Idle gently settles, working lifts
the board for inspection, making_something makes short probe gestures near its
fixed guide, and milestone_level_up raises the hands into an open V with a small
hop. Limit the celebration forearm rotations to 1.10 radians so the tools stay
outside the head silhouette. Keep hidden geometry, mechanics, material finishes
and gestures explicitly authored; do not claim reconstructed reference geometry,
physical manufacture, finger actuation, seamless loops or Weft compatibility for
this richer general-profile model. Use no external models, textures or fonts.
""".strip()


def clean(value):
    if isinstance(value, float):
        return round(value, 7) + 0.0
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    return value


def material(color, roughness, metallic=0, brushed=False, seed=0):
    result = {"color": [*color, 1], "roughness": roughness, "metallic": metallic}
    if brushed:
        result["surface"] = {"pattern": "brushed", "resolution": 256, "strength": .12, "seed": seed}
    return result


def towards(a, b, t):
    return [x + (y - x) * t for x, y in zip(a, b)]


def direction_rotation(a, b):
    d = [y - x for x, y in zip(a, b)]
    length = math.sqrt(sum(x * x for x in d))
    # Rz(azimuth) Ry(inclination) sends the primitive's +Z axis along the segment.
    return [0, math.acos(d[2] / length), math.atan2(d[1], d[0])], length


class Geometry:
    def __init__(self):
        self.parts = []

    def primitive(self, name, shape, location, scale, material_name, bone,
                  rotation=None, bevel=0, segments=32, rings=16):
        part = {"name": name, "shape": shape, "location": location, "scale": scale,
                "rotation": rotation or [0, 0, 0], "material": material_name, "bone": bone}
        if shape != "cube":
            part.update(segments=segments, rings=rings)
        if bevel:
            part.update(bevel=bevel, bevel_segments=3)
        self.parts.append(part)

    def mesh(self, name, vertices, faces, material_name, bone, location=None, rotation=None, uv=None):
        part = {"name": name, "shape": "mesh", "location": location or [0, 0, 0],
                "rotation": rotation or [0, 0, 0], "scale": [1, 1, 1],
                "material": material_name, "bone": bone, "vertices": vertices, "faces": faces,
                "weights": [{bone: 1} for _ in vertices]}
        if uv is not None:
            part["uv"] = uv
        self.parts.append(part)

    def shell(self, name, location, radii, material_name, bone, rotation=None, power=3.8):
        # Closed superellipse lofts replace swollen ellipsoids with purposeful shell profiles.
        profile = [(-1, .64), (-.98, .78), (-.92, .91), (-.79, .98), (-.52, 1),
                   (.42, 1), (.73, .96), (.9, .87), (.98, .72), (1, .58)]
        count = 32
        vertices, faces, uv = [], [], []
        rx, ry, rz = radii
        for z, width in profile:
            for i in range(count):
                angle = i * math.tau / count
                x, y = math.cos(angle), math.sin(angle)
                vertices.append([rx * width * math.copysign(abs(x) ** (2 / power), x),
                                 ry * width * math.copysign(abs(y) ** (2 / power), y), rz * z])
                uv.append([i / count, (z + 1) / 2])
        for ring in range(len(profile) - 1):
            for i in range(count):
                j = (i + 1) % count
                faces.append([ring * count + i, ring * count + j,
                              (ring + 1) * count + j, (ring + 1) * count + i])
        bottom, top = len(vertices), len(vertices) + 1
        vertices.extend([[0, 0, -rz], [0, 0, rz]])
        uv.extend([[.5, 0], [.5, 1]])
        offset = (len(profile) - 1) * count
        for i in range(count):
            j = (i + 1) % count
            faces.extend([[bottom, j, i], [top, offset + i, offset + j]])
        self.mesh(name, vertices, faces, material_name, bone, location, rotation, uv)

    def link(self, name, a, b, radius, material_name, bone, shape="cylinder", bevel=0):
        rotation, length = direction_rotation(a, b)
        self.primitive(name, shape, towards(a, b, .5), [radius, radius, length / 2],
                       material_name, bone, rotation, bevel)

    def sleeve(self, name, a, b, radii, material_name, bone):
        rotation, length = direction_rotation(a, b)
        self.shell(name, towards(a, b, .5), [*radii, length / 2], material_name, bone, rotation)

    def boxes(self, name, boxes, material_name, bone):
        vertices, faces = [], []
        corners = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                   (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        sides = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                 [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        for location, scale in boxes:
            base = len(vertices)
            vertices.extend([[location[i] + c[i] * scale[i] for i in range(3)] for c in corners])
            faces.extend([[base + index for index in face] for face in sides])
        self.mesh(name, vertices, faces, material_name, bone)

    def smile(self):
        vertices, faces = [], []
        count, sides = 20, 8
        for i in range(count + 1):
            t = i / count
            x = -.067 + .134 * t
            z = 1.995 + .027 * ((2 * t - 1) ** 2)
            for j in range(sides):
                angle = j * math.tau / sides
                vertices.append([x, -.310 + .008 * math.cos(angle), z + .008 * math.sin(angle)])
        for i in range(count):
            for j in range(sides):
                k = (j + 1) % sides
                faces.append([i * sides + j, i * sides + k, (i + 1) * sides + k, (i + 1) * sides + j])
        vertices.extend([[-.067, -.310, 2.022], [.067, -.310, 2.022]])
        for j in range(sides):
            k = (j + 1) % sides
            faces.extend([[len(vertices) - 2, k, j],
                          [len(vertices) - 1, count * sides + j, count * sides + k]])
        self.mesh("curved_smile", vertices, faces, "eye", "head")


def add_body(g):
    g.shell("tapered_body_shell", [0, 0, 1.365], [.385, .239, .38], "blue", "torso")
    g.primitive("chest_panel_gasket", "cube", [0, -.237, 1.388], [.302, .025, .265], "graphite", "torso", bevel=.058)
    g.primitive("ivory_chest_armor", "cube", [0, -.258, 1.397], [.285, .024, .247], "cream", "torso", bevel=.054)
    g.primitive("recessed_control_bezel", "cube", [0, -.285, 1.426], [.155, .014, .108], "metal", "torso", bevel=.024)
    g.primitive("control_glass", "cube", [0, -.303, 1.43], [.137, .009, .091], "visor", "torso", bevel=.020)
    g.primitive("power_ring", "torus", [-.075, -.317, 1.446], [.030, .030, .009], "brass", "torso", [math.pi / 2, 0, 0], segments=24, rings=8)
    g.primitive("power_indicator", "cylinder", [-.075, -.319, 1.446], [.018, .018, .004], "eye", "torso", [math.pi / 2, 0, 0])
    g.boxes("display_meter", [([.025 + i * .027, -.316, 1.455], [.008, .004, .010 + i * .006]) for i in range(3)], "eye", "torso")
    g.boxes("chest_recessed_vents", [([0, -.285, 1.218 + i * .035], [.112, .005, .007]) for i in range(3)], "graphite", "torso")
    g.boxes("chest_fasteners", [([x, -.286, z], [.012, .005, .012]) for x in (-.23, .23) for z in (1.205, 1.58)], "metal", "torso")
    g.primitive("rear_hatch_gasket", "cube", [0, .234, 1.365], [.256, .021, .246], "graphite", "torso", bevel=.046)
    g.primitive("rear_service_hatch", "cube", [0, .255, 1.375], [.237, .018, .226], "blue_edge", "torso", bevel=.038)
    g.boxes("rear_hatch_vents", [([0, .277, 1.275 + i * .032], [.135, .005, .006]) for i in range(4)], "metal", "torso")
    g.primitive("rear_hatch_latch", "cube", [0, .284, 1.51], [.053, .011, .026], "brass", "torso", bevel=.011)
    g.boxes("rear_hatch_fasteners", [([x, .277, z], [.011, .005, .011]) for x in (-.19, .19) for z in (1.20, 1.55)], "metal", "torso")
    g.primitive("waist_flexible_seal", "cylinder", [0, 0, .986], [.235, .208, .072], "graphite", "torso", bevel=.014)
    g.primitive("waist_upper_collar", "cylinder", [0, 0, 1.041], [.246, .219, .018], "metal", "torso", bevel=.007)
    g.shell("pelvis_enclosure", [0, .005, .902], [.31, .205, .145], "blue", "root")
    g.primitive("pelvis_front_guard", "cube", [0, -.203, .92], [.163, .017, .058], "cream", "root", bevel=.025)
    g.primitive("neck_gimbal", "cylinder", [0, 0, 1.778], [.114, .114, .072], "graphite", "head", bevel=.01)
    g.primitive("neck_upper_collar", "cylinder", [0, 0, 1.822], [.13, .13, .018], "metal", "head", bevel=.008)


def add_head(g):
    g.primitive("head_main_enclosure", "cube", [0, 0, 2.061], [.355, .234, .255], "cream", "head", bevel=.112)
    g.primitive("crown_parting_seam", "cube", [0, .016, 2.234], [.336, .222, .068], "graphite", "head", bevel=.060)
    g.primitive("blue_crown_shell", "cube", [0, .016, 2.247], [.34, .225, .067], "blue", "head", bevel=.061)
    g.primitive("visor_gasket", "cube", [0, -.236, 2.071], [.301, .037, .161], "graphite", "head", bevel=.065)
    g.primitive("recessed_visor", "cube", [0, -.273, 2.075], [.278, .025, .137], "visor", "head", bevel=.063)
    for sign, side in ((-1, "right"), (1, "left")):
        g.primitive(f"warm_eye_{side}", "cube", [sign * .103, -.307, 2.105], [.027, .008, .043], "eye", "head", bevel=.012)
        g.primitive(f"eye_glint_{side}", "sphere", [sign * .103 - .007, -.316, 2.126], [.006, .003, .008], "cream", "head", segments=16, rings=8)
        g.primitive(f"temple_gasket_{side}", "cylinder", [sign * .352, .012, 2.077], [.095, .095, .025], "graphite", "head", [0, math.pi / 2, 0], bevel=.008)
        g.primitive(f"temple_disc_{side}", "cylinder", [sign * .382, .012, 2.077], [.076, .076, .018], "blue", "head", [0, math.pi / 2, 0], bevel=.012)
    g.smile()
    g.boxes("rear_head_vent_array", [([x, .238, 2.05], [.006, .006, .050]) for x in (-.10, -.05, 0, .05, .10)], "graphite", "head")
    g.primitive("antenna_socket", "cylinder", [0, .035, 2.322], [.055, .055, .028], "graphite", "head", bevel=.01)
    g.primitive("antenna_telescoping_stem", "cylinder", [0, .035, 2.412], [.020, .020, .077], "metal", "head", bevel=.005)
    g.primitive("antenna_brass_tip", "sphere", [0, .035, 2.501], [.049, .049, .049], "brass", "head")


def add_limb(g, sign, side, heads):
    upper, forearm, hand = f"upper_arm_{side}", f"forearm_{side}", f"hand_{side}"
    shoulder, elbow, wrist = heads[upper], heads[forearm], heads[hand]
    thigh, shin, foot = f"thigh_{side}", f"shin_{side}", f"foot_{side}"
    hip, knee, ankle = heads[thigh], heads[shin], heads[foot]
    sideways = [0, math.pi / 2, 0]
    g.primitive(f"shoulder_gimbal_{side}", "cylinder", shoulder, [.131, .131, .095], "graphite", upper, sideways, bevel=.018)
    g.primitive(f"shoulder_outer_collar_{side}", "cylinder", [sign * .517, 0, 1.58], [.145, .145, .020], "metal", upper, sideways, bevel=.009)
    g.primitive(f"shoulder_shell_cap_{side}", "cylinder", [sign * .542, 0, 1.58], [.121, .121, .017], "cream", upper, sideways, bevel=.014)
    g.sleeve(f"tapered_upper_arm_{side}", towards(shoulder, elbow, .15), towards(shoulder, elbow, .88), [.102, .098], "blue", upper)
    g.link(f"upper_arm_cuff_{side}", towards(shoulder, elbow, .71), towards(shoulder, elbow, .80), .103, "metal", upper, bevel=.005)
    g.primitive(f"elbow_bearing_{side}", "cylinder", elbow, [.115, .115, .125], "graphite", forearm, sideways, bevel=.012)
    g.primitive(f"elbow_outer_hub_{side}", "cylinder", [elbow[0] + sign * .125, elbow[1], elbow[2]], [.086, .086, .014], "metal", forearm, sideways, bevel=.010)
    g.primitive(f"elbow_hub_center_{side}", "cylinder", [elbow[0] + sign * .142, elbow[1], elbow[2]], [.033, .033, .008], "brass", forearm, sideways, bevel=.005)
    g.sleeve(f"tapered_forearm_shell_{side}", towards(elbow, wrist, .18), towards(elbow, wrist, .86), [.112, .108], "cream", forearm)
    g.link(f"wrist_seal_{side}", towards(elbow, wrist, .82), towards(elbow, wrist, 1.02), .083, "graphite", forearm, bevel=.008)
    g.link(f"wrist_blue_cuff_{side}", towards(elbow, wrist, .85), towards(elbow, wrist, .94), .096, "blue", forearm, bevel=.007)
    g.primitive(f"hand_palm_{side}", "cube", wrist, [.087, .08, .075], "blue", hand, bevel=.035)
    for digit, z in enumerate((1.005, 1.050, 1.095)):
        x = .421 if sign == 1 else -.405
        g.primitive(f"static_grip_finger_{side}_{digit}", "cube", [x, -.360, z], [.051, .027, .018], "cream", hand, bevel=.013)
    g.primitive(f"static_thumb_{side}", "cube", [sign * .374, -.314, 1.102], [.034, .029, .027], "graphite", hand, bevel=.02)
    g.primitive(f"hip_bearing_{side}", "cylinder", hip, [.137, .137, .085], "graphite", thigh, sideways, bevel=.014)
    g.sleeve(f"thigh_shell_{side}", towards(hip, knee, .18), towards(hip, knee, .81), [.123, .125], "blue", thigh)
    g.link(f"thigh_lower_trim_{side}", towards(hip, knee, .72), towards(hip, knee, .82), .123, "metal", thigh, bevel=.007)
    g.primitive(f"knee_bearing_{side}", "cylinder", knee, [.123, .123, .134], "graphite", shin, sideways, bevel=.012)
    g.primitive(f"knee_front_guard_{side}", "cube", [knee[0], -.112, knee[2]], [.103, .036, .084], "blue", shin, bevel=.040)
    g.primitive(f"knee_guard_fastener_{side}", "cylinder", [knee[0], -.151, knee[2]], [.031, .031, .006], "brass", shin, [math.pi / 2, 0, 0], bevel=.004)
    g.sleeve(f"shin_armor_{side}", towards(knee, ankle, .16), towards(knee, ankle, .86), [.123, .115], "cream", shin)
    g.primitive(f"shin_inset_{side}", "cube", [sign * .255, -.117, .341], [.065, .014, .095], "blue", shin, bevel=.019)
    g.primitive(f"ankle_collar_{side}", "cylinder", [sign * .26, -.02, .187], [.095, .10, .035], "graphite", foot, bevel=.010)
    g.primitive(f"rubber_sole_{side}", "cube", [sign * .26, -.104, .049], [.178, .24, .037], "graphite", foot, bevel=.028)
    g.primitive(f"boot_shell_{side}", "cube", [sign * .26, -.104, .139], [.169, .224, .079], "blue", foot, bevel=.052)
    g.primitive(f"ivory_toe_cap_{side}", "cube", [sign * .26, -.279, .147], [.147, .053, .053], "cream", foot, bevel=.032)
    g.primitive(f"heel_guard_{side}", "cube", [sign * .26, .112, .127], [.127, .023, .052], "metal", foot, bevel=.016)
    g.boxes(f"sole_tread_{side}", [([sign * .26, y, .014], [.135, .008, .005]) for y in (-.25, -.15, -.05, .05)], "blue_edge", foot)


def add_tools(g):
    g.primitive("board_frame", "cube", [.19, -.390, 1.05], [.235, .033, .157], "metal", "workpiece", bevel=.014)
    g.primitive("board_substrate", "cube", [.19, -.430, 1.05], [.214, .012, .137], "circuit", "workpiece", bevel=.008)
    g.primitive("board_microchip", "cube", [.194, -.457, 1.061], [.059, .012, .040], "graphite", "workpiece", bevel=.006)
    tracks = [([.050, -.448, 1.127], [.044, .003, .004]), ([.095, -.448, 1.100], [.004, .003, .031]),
              ([.114, -.448, 1.072], [.025, .003, .004]), ([.312, -.448, 1.144], [.058, .003, .004]),
              ([.255, -.448, 1.121], [.004, .003, .027]), ([.300, -.448, 1.035], [.044, .003, .004]),
              ([.343, -.448, .992], [.004, .003, .047]), ([.282, -.448, .949], [.065, .003, .004]),
              ([.111, -.448, .978], [.060, .003, .004]), ([.054, -.448, 1.014], [.004, .003, .040])]
    g.boxes("fixed_copper_trace_network", tracks, "copper", "workpiece")
    pins = [([.145 + i * .019, -.453, z], [.004, .004, .014]) for i in range(6) for z in (1.007, 1.115)]
    pins += [([x, -.453, 1.040 + i * .021], [.012, .004, .004]) for x in (.123, .265) for i in range(3)]
    g.boxes("chip_pin_array", pins, "metal", "workpiece")
    g.boxes("board_edge_contacts", [([.09 + i * .022, -.448, .924], [.006, .003, .011]) for i in range(10)], "brass", "workpiece")
    g.boxes("board_corner_fasteners", [([x, -.447, z], [.009, .005, .009]) for x in (-.010, .39) for z in (.935, 1.165)], "brass", "workpiece")
    g.primitive("probe_contact_pad", "cylinder", [.056, -.450, 1.126], [.016, .016, .004], "brass", "workpiece", [math.pi / 2, 0, 0])
    g.link("probe_soft_grip", [-.478, -.340, 1.085], [-.263, -.392, 1.104], .034, "blue_edge", "probe", bevel=.008)
    g.link("probe_painted_barrel", [-.370, -.367, 1.094], [-.241, -.397, 1.106], .038, "blue", "probe", bevel=.008)
    g.link("probe_brass_collar", [-.261, -.392, 1.104], [-.217, -.402, 1.108], .041, "brass", "probe", bevel=.005)
    g.link("probe_metal_shaft", [-.217, -.402, 1.108], [.024, -.454, 1.123], .013, "metal", "probe", bevel=.003)
    g.link("probe_tapered_contact_tip", [.022, -.454, 1.123], [.058, -.462, 1.126], .014, "brass", "probe", shape="cone")


def animated_states():
    states = []
    idle = []
    for frame, rock in ((0, 0), (24, 1), (48, 0), (72, -1), (96, 0)):
        idle.append({"frame": frame, "bones": {
            "torso": {"rotation": [0, 0, rock * .025]}, "head": {"rotation": [0, rock * .035, 0]},
            "upper_arm_right": {"rotation": [0, 0, rock * .012]},
            "upper_arm_left": {"rotation": [0, 0, -rock * .012]}}})
    states.append({"id": "idle", "meaning": "An attentive settling sway with a small head tilt, tools held steadily at waist level.",
                   "loop": True, "duration": 4, "keyframes": idle})
    work = []
    for frame, look in ((0, -1), (24, 0), (48, 1), (72, 0), (96, -1)):
        work.append({"frame": frame, "bones": {
            "torso": {"rotation": [.09, 0, look * .028]}, "head": {"rotation": [.10, 0, -look * .025]},
            "upper_arm_left": {"rotation": [-.97, 0, 0]},
            "forearm_left": {"rotation": [-.32, 0, look * .018]},
            "upper_arm_right": {"rotation": [0, 0, 0]}, "forearm_right": {"rotation": [0, 0, 0]}}})
    states.append({"id": "working", "meaning": "Lifts the circuit board below the visor and studies it with a gentle head-and-body scan; the probe rests low.",
                   "loop": True, "duration": 4, "keyframes": work})
    craft = []
    for frame in range(0, 97, 12):
        sign = 1 if frame % 24 == 0 else -1
        craft.append({"frame": frame, "bones": {
            "torso": {"rotation": [.045, 0, 0]}, "head": {"rotation": [.12, 0, -.025]},
            "forearm_right": {"rotation": [-sign * .065, 0, sign * .045]},
            "forearm_left": {"rotation": [0, 0, .01]}}})
    states.append({"id": "making_something", "meaning": "Makes short precise probe gestures near the board's contact guide while the left hand steadies it. Copper tracks are fixed geometry, not newly manufactured marks.",
                   "loop": True, "duration": 4, "keyframes": craft})
    celebrate = []
    for frame, lift, hop, crouch in ((0, 0, 0, 0), (16, .15, -.004, .1),
                                    (32, 1, .085, 0), (48, 1, .12, 0),
                                    (64, .85, 0, .035), (80, .2, 0, 0), (96, 0, 0, 0)):
        celebrate.append({"frame": frame, "bones": {
            "root": {"location": [0, 0, hop]}, "torso": {"rotation": [-.025 * lift, 0, 0]},
            "head": {"rotation": [-.035 * lift, 0, 0]},
            "upper_arm_left": {"rotation": [0, -1.9 * lift, 0]},
            "upper_arm_right": {"rotation": [0, 1.9 * lift, 0]},
            "forearm_left": {"rotation": [0, -1.10 * lift, 0]},
            "forearm_right": {"rotation": [0, 1.10 * lift, 0]},
            "thigh_left": {"rotation": [-crouch, 0, 0]}, "thigh_right": {"rotation": [-crouch, 0, 0]},
            "shin_left": {"rotation": [crouch * 2, 0, 0]}, "shin_right": {"rotation": [crouch * 2, 0, 0]},
            "foot_left": {"rotation": [-crouch, 0, 0]}, "foot_right": {"rotation": [-crouch, 0, 0]}}})
    states.append({"id": "milestone_level_up", "meaning": "Raises both tools into a wide open V with a small hop and preparatory knee bend; hands stay outside the face silhouette.",
                   "loop": True, "duration": 4, "keyframes": celebrate})
    return states


def scene():
    original = json.loads(SOURCE.read_text())
    g = Geometry()
    add_body(g)
    add_head(g)
    heads = {bone["name"]: bone["head"] for bone in original["bones"]}
    add_limb(g, -1, "right", heads)
    add_limb(g, 1, "left", heads)
    add_tools(g)
    result = {
        "schema_version": 2, "id": "workshop-robot-quality-v2", "output_profile": "general",
        "subject": "A polished blue-and-ivory workshop robot with a friendly inset visor, metal joint hardware, a detailed circuit board and a precision probe.",
        "style": "Designer retro-future character: shaped enamel shells, satin ivory armor, dark rubber seals, brushed metal collars and restrained warm brass accents.",
        "source_prompt": original["prompt"], "prompt": EDIT_PROMPT,
        "references": [], "rights": deepcopy(original["rights"]),
        "authored_assumptions": [
            "All geometry, hidden surfaces, mechanical details and material finishes are authored locally; no image-to-3D recovery or external assets are used.",
            "The original blue/cream robot identity, all 17 bone names, parent relationships, major pivots, left-hand board and right-hand probe are retained.",
            "Anatomical left is world +X, right is world -X, the rest-facing direction is -Y and units are meters with Z up before glTF export.",
            "Closed body lofts and beveled panels are overlapping manufactured components, not one welded watertight outer surface.",
            "The visible grip fingers and thumb are rigidly weighted to each hand bone; they are not independent finger joints or finger controls.",
            "Board traces, pads and chip pins are static decorative geometry; probe motion and inspection are authored gestures, not contact-constrained manufacturing or a circuit simulation.",
            "The four clips have matching authored endpoints as loop intent only. Rendered loop boundaries and state quality require downstream verification.",
            "This general-profile revision uses real metallic-roughness PBR and richer geometry. It is not the matte diffuse weft-v1 deliverable and has not been validated against Weft.",
            "Original bone rests are retained, while new shell surfaces, backside panels and small surface patterns are deliberate design assumptions.",
            "Keyframe rotations and translations use model-rest axes; the backend converts them through bone rest bases.",
        ],
        "materials": {
            "blue": material([.19, .405, .565], .30, .14),
            "blue_edge": material([.085, .205, .285], .43, .12),
            "cream": material([.925, .885, .775], .34, .02),
            "graphite": material([.115, .155, .177], .65, .06),
            "metal": material([.48, .54, .56], .31, .78, True, 173),
            "brass": material([.755, .555, .255], .31, .72, True, 271),
            "visor": material([.035, .078, .105], .19, .10),
            "eye": material([1, .845, .49], .31),
            "circuit": material([.08, .27, .29], .55, .05),
            "copper": material([.73, .43, .20], .34, .70),
        },
        "bones": deepcopy(original["bones"]), "parts": g.parts, "states": animated_states(),
        "attachments": deepcopy(original["attachments"]),
        "camera": {"target": [0, -.02, 1.32], "position": [2.8, -6, 2.85], "orthographic_scale": 3.5},
        "background": [.925, .945, .95, 1],
        "authoring": {
            "method": "Deterministic local Python geometry and keyframe authoring; no generation service calls",
            "script": "scripts/author_quality_robot.py",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_scene": "examples/scenes/humanoid-robot-weft-v1.json",
            "source_scene_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "cost_usd": 0, "seed": 173,
            "preserved_mapping_design": deepcopy(original["binding_design"]),
            "status": "Authored scene; rendering, exported mesh validation and visual review are downstream work",
        },
    }
    return clean(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--check", action="store_true", help="Check deterministic equivalence without writing")
    args = parser.parse_args()
    data = scene()
    if len(data["parts"]) > 128:
        raise ValueError("Authored scene exceeds the 128-part budget")
    encoded = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.check:
        if args.output.read_text() != encoded:
            raise SystemExit("Scene differs from deterministic authoring output")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(encoded)
    print(json.dumps({"path": str(args.output), "check": args.check, "parts": len(data["parts"]),
                      "bones": len(data["bones"]), "materials": len(data["materials"]),
                      "explicit_vertices": sum(len(p.get("vertices", [])) for p in data["parts"]),
                      "explicit_faces": sum(len(p.get("faces", [])) for p in data["parts"]),
                      "bytes": len(encoded.encode()), "sha256": hashlib.sha256(encoded.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
