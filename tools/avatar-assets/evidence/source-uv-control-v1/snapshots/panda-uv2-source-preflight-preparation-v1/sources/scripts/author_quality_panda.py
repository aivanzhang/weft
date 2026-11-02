"""Reproducible local modeling recipe; the reference is data, never executable input."""

import json
import math
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from author_quality_companions import color, fur, mesh, primitive, surface, tube

ROOT = Path(__file__).resolve().parents[1]


def ellipsoid(name, center, radii, material, bone, cheeks=0, segments=40, rings=24):
    vertices, uv, faces = [], [], []
    for j in range(rings + 1):
        theta = math.pi * j / rings
        nz = math.cos(theta)
        widen = 1 + cheeks * math.exp(-((nz + .28) / .55) ** 2)
        for i in range(segments + 1):
            phi = math.tau * i / segments
            vertices.append([center[0] + radii[0] * math.sin(theta) * math.cos(phi) * widen,
                             center[1] + radii[1] * math.sin(theta) * math.sin(phi),
                             center[2] + radii[2] * nz])
            uv.append([i / segments, 1 - j / rings])
    for j in range(rings):
        for i in range(segments):
            a = j * (segments + 1) + i
            if j == 0:
                faces.append([a, a + segments + 1, a + segments + 2])
            elif j == rings - 1:
                faces.append([a, a + segments + 1, a + 1])
            else:
                faces.append([a, a + segments + 1, a + segments + 2, a + 1])
    return mesh(name, vertices, faces, material, bone, uv)


def front_y(x, z, center, radii, cheeks=0):
    nz = (z - center[2]) / radii[2]
    width = radii[0] * (1 + cheeks * math.exp(-((nz + .28) / .55) ** 2))
    return center[1] - radii[1] * math.sqrt(max(.001, 1 - (x / width) ** 2 - nz ** 2))


def patch(name, x, z, rx, rz, angle, parent_center, parent_radii, mat, bone, cheeks=0, offset=.007):
    vertices, uv, faces = [], [], []
    segments, rings = 40, 8
    for j in range(rings + 1):
        radius = j / rings
        for i in range(segments + 1):
            theta = math.tau * i / segments
            a, b = rx * radius * math.cos(theta), rz * radius * math.sin(theta)
            xx = x + a * math.cos(angle) - b * math.sin(angle)
            zz = z + a * math.sin(angle) + b * math.cos(angle)
            vertices.append([xx, front_y(xx, zz, parent_center, parent_radii, cheeks) - offset, zz])
            uv.append([.5 + radius * math.cos(theta) / 2, .5 + radius * math.sin(theta) / 2])
    for j in range(rings):
        for i in range(segments):
            a = j * (segments + 1) + i
            if j == 0:
                faces.append([a, a + segments + 1, a + segments + 2])
            else:
                faces.append([a, a + segments + 1, a + segments + 2, a + 1])
    return mesh(name, vertices, faces, mat, bone, uv)


def rotation(v):
    x, y, z = v
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def euler_xyz(matrix):
    y = math.asin(float(np.clip(-matrix[2, 0], -1, 1)))
    if abs(math.cos(y)) < 1e-6:
        x, z = math.atan2(-matrix[1, 2], matrix[1, 1]), 0
    else:
        x, z = math.atan2(matrix[2, 1], matrix[2, 2]), math.atan2(matrix[1, 0], matrix[0, 0])
    return [round(float(v), 6) for v in (x, y, z)]


def arm_pose(shoulder, elbow, hand, target, side):
    # Solve a bounded author-time reach; there is no runtime IK or physics substitute.
    a, b, c, goal = [np.array(p, dtype=float) for p in (shoulder, elbow, hand, target)]
    preferred = np.array([-.45, side * .4, 0, -.35, side * .55, 0])
    angles = preferred.copy()

    def end(v):
        r = rotation(v[:3])
        return a + r @ (b - a) + r @ rotation(v[3:]) @ (c - b)

    for _ in range(100):
        error = goal - end(angles)
        jac = np.column_stack([(end(angles + np.eye(6)[i] * .0001) - end(angles)) / .0001 for i in range(6)])
        step = np.linalg.solve(jac.T @ jac + np.eye(6) * .008,
                               jac.T @ error + .00005 * (preferred - angles))
        angles += np.clip(step, -.12, .12)
        angles = np.clip(angles, -1.8, 1.8)
    return [round(float(x), 6) for x in angles[:3]], [round(float(x), 6) for x in angles[3:]], float(np.linalg.norm(goal-end(angles)))


def build_scene():
    old = json.loads((ROOT / "examples/scenes/muse-panda-weft-v1.json").read_text())
    d = deepcopy(old)
    d.update(schema_version=2, id="muse-panda-quality-v5", output_profile="general")
    d.pop("binding_design", None)
    d["source_prompt"] = old["prompt"]
    d["user_quality_request"] = "ok nice but why are these assets not as high quality as the Muse one? Can you make them as nice as them"
    d["user_detail_request"] = "make sur eto have fur and all other necessary components"
    d["prompt"] = (
        "Re-author the Muse-inspired panda as a compact, expressive, complete animated 3D character. "
        "Use the supplied Muse portrait for proportions, black-and-ivory palette, warm brown eyes, broad cheeks, "
        "soft muzzle, smile and the small green dragon on its anatomical right/viewer-left shoulder. "
        "Replace stacked joint balls with continuous weighted arms and legs. Model visible short fur, "
        "eyelids, layered glossy eyes, nostrils, mouth crease, ears, paws, toes and a rear tail. The shoulder "
        "dragon must have shaped head/body, eyes, nostrils, smile, horns, frills, scaled green skin, segmented "
        "belly, claws, curved tail and complete scalloped wings with raised spars. Include all these features "
        "in the skinned GLB, not only in the render. Use embedded procedural fur/scale surface textures and "
        "opaque physically based materials. Preserve a small golden crafting orb as an explicitly authored "
        "action prop. Idle breathes and the dragon nods; making_something rubs and rolls the orb between "
        "the paws; working lifts it for inspection; milestone_level_up presents it with a happy arm lift. "
        "Author all unseen surfaces, rig, fur placement and motion; do not claim recovered geometry, "
        "hair simulation, facial capture, Weft compatibility or exact Muse-quality equivalence."
    )
    d["style"] = "Soft, compact storybook panda with actual short geometric fur, expressive inset features and a detailed green dragon; browser PBR edition"
    d["authored_assumptions"] = [
        "The supplied opaque 2D portrait supplies appearance only; back, underside, rig, topology, UVs, weights and fur placement are newly authored.",
        "Fur is a bounded short triangular-blade groom plus embedded surface textures, not a physically simulated or film-density hair system.",
        "Arms and legs have continuous meshes with blended two-bone skinning. Torso, head, paws and facial components remain separate authored surfaces.",
        "Eyes, eyelids, nose, mouth and paw digits are modeled geometry, not independently actuated facial muscles or fingers.",
        "The green dragon stays attached to the torso-side anatomical-right shoulder helper; head, tail and wings have local authored controls only.",
        "The golden orb is an authored action prop, not part of the reference portrait. Crafting gestures do not change its geometry or simulate manufacture.",
        "General PBR edition: embedded normal maps, short fur and glossy surfaces have not been reduced to or approved by the Weft diffuse-only profile.",
        "All native glTF transforms are authored meters/Y-up/+Z-front after Blender export; scale is not calibrated to the live simulator.",
        "The neutral pose is arms resting beside the rounded body, feet flat, face forward; it is not an observed simulator pose."
    ]
    d["materials"] = {
        "ivory": color("F5F1E6", .82, surface=surface("fur", .24, 71)),
        "charcoal": color("171613", .8, surface=surface("fur", .27, 72)),
        "ear_inner": color("443D38", .9, surface=surface("fur", .2, 73)),
        "nose": color("34241E", .3),
        "sclera": color("F9F6E8", .22),
        "iris": color("836132", .17),
        "pupil": color("090C0D", .07),
        "dragon_green": color("3A9659", .66, surface=surface("scales", .42, 74)),
        "dragon_belly": color("ACBC67", .72, surface=surface("scales", .17, 75)),
        "wing": color("6A9B51", .78, surface=surface("scales", .15, 76)),
        "horn": color("B6BA8B", .6),
        "dragon_horn": color("7D9D66", .64),
        "gold": color("D99D38", .36, .16),
    }
    positions = {"root": [0, 0, .08], "torso": [0, 0, .58], "head": [0, 0, 1.40],
                 "right_shoulder": [-.43, 0, 1.32], "dragon_root": [-.57, -.045, 1.40],
                 "dragon_head": [-.60, -.10, 1.61], "dragon_tail": [-.60, .04, 1.30],
                 "dragon_wing_outer": [-.66, .035, 1.47], "dragon_wing_inner": [-.48, .035, 1.47],
                 "craft_orb": [-.49, -.34, .69]}
    for side, sign in (("left", 1), ("right", -1)):
        positions.update({f"upper_arm_{side}": [sign*.43, 0, 1.32],
                          f"forearm_{side}": [sign*.60, -.035, .95],
                          f"hand_{side}": [sign*.49, -.25, .66],
                          f"thigh_{side}": [sign*.24, 0, .52],
                          f"shin_{side}": [sign*.26, -.005, .30],
                          f"foot_{side}": [sign*.27, -.04, .13]})
    for b in d["bones"]:
        b["head"] = positions[b["name"]]
        b["tail"] = [b["head"][0], b["head"][1], round(b["head"][2] + .18, 6)]
    p = []
    head_center, head_radii = [0, -.015, 1.785], [.59, .43, .505]
    body_center, body_radii = [0, 0, .89], [.46, .345, .535]
    p.append(fur(ellipsoid("rounded_body", body_center, body_radii, "charcoal", "torso", .08), 2500, .016, .0028, 80))
    p.append(fur(patch("white_belly", 0, .825, .365, .435, 0, body_center, body_radii,
                       "ivory", "torso", .08, .009), 1800, .013, .0024, 81))
    p.append(fur(ellipsoid("cheeked_head", head_center, head_radii, "ivory", "head", .13), 7800, .017, .0026, 82))
    for side, sign in (("left", 1), ("right", -1)):
        p.append(fur(primitive(f"ear_{side}", "sphere", [sign*.445, .015, 2.205], [.165, .115, .18],
                               "charcoal", "head", segments=32, rings=20), 350, .015, .0022, 83+(sign>0)))
        p.append(primitive(f"inner_ear_{side}", "sphere", [sign*.445, -.095, 2.205], [.101, .015, .115],
                           "ear_inner", "head", segments=24, rings=16))
        p.append(fur(patch(f"eye_patch_{side}", sign*.242, 1.855, .15, .207, sign*.30, head_center,
                           head_radii, "charcoal", "head", .13, .011), 600, .009, .0015, 85+(sign>0)))
        ex, ez = sign*.222, 1.885
        ey = front_y(ex, ez, head_center, head_radii, .13)
        for name, delta, size, mat in (("eye", -.032, [.071, .038, .081], "sclera"),
                                       ("iris", -.066, [.044, .019, .052], "iris"),
                                       ("pupil", -.081, [.028, .010, .036], "pupil")):
            p.append(primitive(f"{name}_{side}", "sphere", [ex-sign*.005, ey+delta, ez-.003],
                               size, mat, "head", segments=32, rings=20))
        lid_points = [[ex+.075*math.cos(t), ey-.044, ez+.085*math.sin(t)]
                      for t in [i*math.pi/12 for i in range(13)]]
        p.append(tube(f"upper_lid_{side}", lid_points, .009, "charcoal", "head", 8))
        p.append(fur(primitive(f"muzzle_{side}", "sphere", [sign*.083, -.427, 1.675], [.143, .081, .105],
                               "ivory", "head", segments=32, rings=20), 150, .004, .0008, 90+(sign>0)))
        shoulder, elbow, hand = [positions[f"{name}_{side}"] for name in ("upper_arm", "forearm", "hand")]
        points = [[shoulder[0]*.92, shoulder[1], shoulder[2]+.13],
                  [shoulder[0]*.97, shoulder[1], shoulder[2]+.09],
                  [shoulder[0], shoulder[1], shoulder[2]+.045]]
        radii = [.004,.085,.13]
        weights = [{f"upper_arm_{side}":1} for _ in points]
        for half, (start, end) in enumerate(((shoulder, elbow), (elbow, hand))):
            for i in range(7):
                if half and i == 0:
                    continue
                t = i/6
                before, after = (shoulder, hand) if half == 0 else (shoulder, hand)
                m0 = [(end[k]-before[k])*.5 if half else (end[k]-start[k]) for k in range(3)]
                m1 = [(after[k]-start[k])*.5 if not half else (end[k]-start[k]) for k in range(3)]
                points.append([(2*t**3-3*t*t+1)*start[k]+(t**3-2*t*t+t)*m0[k]
                               +(-2*t**3+3*t*t)*end[k]+(t**3-t*t)*m1[k] for k in range(3)])
                radii.append(.164-.035*(half+t)/2)
                blend = max(0, min(1, (half+t-.65)/.7))
                values = {f"upper_arm_{side}": 1-blend, f"forearm_{side}": blend}
                weights.append({k: round(v, 6) for k, v in values.items() if v>0})
        p.append(fur(tube(f"continuous_arm_{side}", points, radii, "charcoal", f"upper_arm_{side}", 24, weights),
                     950, .016, .0024, 92+(sign>0)))
        p.append(fur(primitive(f"paw_{side}", "sphere", hand, [.155, .13, .155], "charcoal", f"hand_{side}",
                               segments=28, rings=18), 260, .011, .0018, 94+(sign>0)))
        for finger in range(3):
            p.append(primitive(f"paw_digit_{side}_{finger}", "sphere",
                               [hand[0]+(finger-1)*.068, hand[1]-.073, hand[2]-.07],
                               [.046, .065, .079], "charcoal", f"hand_{side}", segments=20, rings=12))
        hip, knee, foot = [positions[f"{name}_{side}"] for name in ("thigh", "shin", "foot")]
        lpoints = [[hip[0], 0, .58], hip, [sign*.25, 0, .40], knee, foot]
        lweights = [{f"thigh_{side}": 1}, {f"thigh_{side}": 1},
                    {f"thigh_{side}": .6, f"shin_{side}": .4},
                    {f"thigh_{side}": .25, f"shin_{side}": .75}, {f"shin_{side}": 1}]
        p.append(fur(tube(f"continuous_leg_{side}", lpoints, [.13,.175,.17,.155,.12], "charcoal", f"thigh_{side}", 24, lweights),
                     500, .014, .0023, 96+(sign>0)))
        p.append(primitive(f"foot_{side}", "sphere", [sign*.27,-.11,.125], [.177,.23,.12], "charcoal", f"foot_{side}", segments=32, rings=18))
        for toe in range(3):
            p.append(primitive(f"toe_{side}_{toe}", "sphere", [sign*.27+(toe-1)*.08,-.29,.105],
                               [.055,.065,.065], "charcoal", f"foot_{side}", segments=20, rings=12))
    p.append(primitive("soft_nose", "sphere", [0,-.505,1.738], [.084,.046,.049], "nose", "head", segments=32, rings=20))
    p.append(tube("philtrum", [[0,-.533,1.711],[0,-.523,1.659],[0,-.51,1.625]], .005, "nose", "head", 8))
    smile = [[x, -.503+.085*(abs(x)/.19)**1.7, 1.625+.061*(abs(x)/.19)**2] for x in np.linspace(-.19,.19,19)]
    p.append(tube("smile", smile, .006, "nose", "head", 8))
    p.append(fur(primitive("rear_tail", "sphere", [-.16,.318,.635], [.15,.16,.14], "ivory", "torso",
                           segments=28, rings=18), 400, .017, .0021, 99))
    for i in range(3):
        p.append(tube(f"forehead_tuft_{i}", [[-.028+i*.027,.01,2.27],[-.055+i*.032,.018,2.305],[-.07+i*.035,.03,2.32]],
                      [.025,.014,.002], "ivory", "head", 10))
    # The little dragon is complete from side/back, including a folded inner wing.
    p.append(primitive("dragon_body", "sphere", [-.58,-.015,1.43], [.135,.105,.185], "dragon_green", "dragon_root", segments=32, rings=20))
    p.append(primitive("dragon_belly", "sphere", [-.58,-.112,1.43], [.088,.025,.135], "dragon_belly", "dragon_root", segments=28, rings=18))
    for j in range(4):
        z = 1.345+j*.048
        points = [[-.58+x,-.137+.022*(abs(x)/.073)**2,z-.012*(abs(x)/.073)**2] for x in np.linspace(-.073,.073,9)]
        p.append(tube(f"belly_ridge_{j}", points, .004, "dragon_green", "dragon_root", 6))
    p.append(ellipsoid("dragon_head_shape", [-.60,-.10,1.72], [.175,.142,.166], "dragon_green", "dragon_head", .09, 32,20))
    p.append(primitive("dragon_snout", "sphere", [-.60,-.226,1.665], [.133,.073,.077], "dragon_green", "dragon_head", segments=32,rings=20))
    for side, sign in (("left",1),("right",-1)):
        x = -.60+sign*.074
        for name, y, size, mat in (("dragon_eye",-.232,[.043,.024,.051],"sclera"),
                                   ("dragon_iris",-.254,[.027,.013,.033],"dragon_belly"),
                                   ("dragon_pupil",-.266,[.018,.007,.024],"pupil")):
            p.append(primitive(f"{name}_{side}","sphere",[x-sign*.003,y,1.745],size,mat,"dragon_head",segments=24,rings=16))
        p.append(primitive(f"dragon_nostril_{side}","sphere",[-.60+sign*.054,-.291,1.69], [.012,.005,.009],"nose","dragon_head",segments=16,rings=10))
        hornpoints = [[-.60+sign*.108,-.03,1.84],[-.60+sign*.137,.005,1.906],[-.60+sign*.16,.013,1.952],[-.60+sign*.18,.002,1.968]]
        p.append(tube(f"horn_{side}",hornpoints,[.035,.027,.017,.002],"dragon_horn","dragon_head",12))
        for j in range(2):
            p.append(tube(f"frill_{side}_{j}", [[-.60+sign*.153,.005,1.69+j*.056],[-.60+sign*.217,.012,1.71+j*.065],[-.60+sign*.24,.01,1.73+j*.063]],
                          [.026,.018,.002],"dragon_green","dragon_head",10))
        hand = [-.58+sign*.112,-.14,1.35]
        p.append(tube(f"dragon_arm_{side}", [[-.58+sign*.105,-.01,1.51],[-.58+sign*.15,-.06,1.4],hand], [.045,.035,.032],"dragon_green","dragon_root",14))
        p.append(primitive(f"dragon_hand_{side}","sphere",hand,[.051,.043,.025],"dragon_green","dragon_root",segments=20,rings=12))
        for j in range(3):
            claw = [hand[0]+(j-1)*.028,hand[1]-.033,hand[2]-.005]
            p.append(tube(f"dragon_claw_{side}_{j}",[claw,[claw[0],claw[1]-.025,claw[2]-.014]],[.009,.002],"horn","dragon_root",8))
        p.append(primitive(f"dragon_haunch_{side}","sphere",[-.58+sign*.10,.01,1.29],[.07,.065,.075],"dragon_green","dragon_root",segments=24,rings=16))
        p.append(primitive(f"dragon_foot_{side}","sphere",[-.58+sign*.103,-.055,1.25],[.068,.085,.027],"dragon_green","dragon_root",segments=24,rings=16))
    p.append(tube("dragon_smile", [[-.695,-.274,1.654],[-.65,-.298,1.629],[-.60,-.304,1.624],[-.55,-.298,1.629],[-.505,-.274,1.654]], .004,"dragon_green","dragon_head",8))
    tailpoints = [[-.60,.035,1.30],[-.69,.04,1.22],[-.79,-.015,1.20],[-.81,-.105,1.22],[-.76,-.17,1.235],[-.69,-.19,1.25],[-.65,-.20,1.275]]
    p.append(tube("curved_dragon_tail",tailpoints,[.058,.049,.039,.029,.023,.014,.002],"dragon_green","dragon_tail",16))
    for j in range(4):
        point = tailpoints[j]
        p.append(tube(f"tail_spine_{j}",[[point[0],point[1]+.03,point[2]+.03],[point[0]-.01,point[1]+.035,point[2]+.078]],[.018,.002],"dragon_belly","dragon_tail",8))
    for label, sign, root in (("outer",-1,[-.67,.035,1.49]),("inner",1,[-.49,.035,1.49])):
        outline = [[0,0,0],[sign*.10,.025,.14],[sign*.25,.045,.19],[sign*.24,.02,.02],
                   [sign*.23,.005,-.10],[sign*.15,-.01,-.055],[sign*.12,-.01,-.18],[sign*.06,-.01,-.095]]
        vertices = [[root[k]+v[k] for k in range(3)] for v in outline]
        vertices += [[v[0],v[1]+.006,v[2]] for v in vertices]
        faces = []
        for i in range(1,7):
            face = [0,i,i+1]
            if sign < 0:
                face.reverse()
            faces += [face,[x+8 for x in reversed(face)]]
        for i in range(8):
            faces.append([i,(i+1)%8,(i+1)%8+8,i+8])
        bone = f"dragon_wing_{label}"
        p.append(mesh(f"scalloped_wing_{label}",vertices,faces,"wing",bone,
                      [[abs(v[0])/ .25,(v[2]+.18)/.37] for v in outline]*2))
        for j, end in enumerate((2,4,6)):
            endpoint = vertices[end]
            midpoint = [(root[k]+endpoint[k])/2 for k in range(3)]
            midpoint[2] += .023
            p.append(tube(f"wing_spar_{label}_{j}",[root,midpoint,endpoint],[.012,.008,.003],"dragon_green",bone,8))
    p.append(primitive("craft_orb", "sphere",positions["craft_orb"],[.104,.104,.104],"gold","craft_orb",segments=40,rings=24))
    d["parts"] = p
    d["camera"] = {"position":[.7,-6,2.45],"target":[-.055,0,1.15],"orthographic_scale":2.70}
    d["background"] = [.95,.955,.95,1]
    d["attachments"][0]["placement"] = "Anatomical right/viewer-left shoulder; dragon root [-0.57,-0.045,1.40] meters in author Z-up space."
    d["attachments"][0]["authored_assumption"] = "Fixed shoulder parent; local dragon head/tail/wing keyframes are authored gestures, not recovered or simulated controls."
    d["attachments"][1]["placement"] = "Small golden orb at the right paw; follows the right hand through actual skeletal transforms."
    d["states"] = []
    target_poses, residuals = {}, {}
    for name, right_target, left_target in (
        ("craft",[-.10,-.41,.96],[.08,-.43,.96]),
        ("inspect",[-.11,-.43,1.25],[.13,-.43,1.20]),
        ("celebrate",[-.90,-.40,1.75],[.90,-.40,1.75]),
    ):
        pose = {}
        for side, sign, target in (("right",-1,right_target),("left",1,left_target)):
            a,b,error = arm_pose(positions[f"upper_arm_{side}"],positions[f"forearm_{side}"],positions[f"hand_{side}"],target,sign)
            pose[f"upper_arm_{side}"] = {"rotation":a}
            pose[f"forearm_{side}"] = {"rotation":b}
            residuals[f"{name}_{side}"] = round(error,6)
        target_poses[name] = pose
    meanings = {
        "idle":"Panda breathes and gives a small friendly head tilt while its shoulder dragon nods and curls its tail.",
        "making_something":"Panda brings the small golden orb between both paws and works it with alternating short rubbing/rolling gestures; the orb is not replaced or reshaped.",
        "working":"Panda raises the golden orb toward its eyes and angles its head for close inspection, then lowers it.",
        "milestone_level_up":"Panda lifts both paws and presents the orb in a celebratory pose while the shoulder dragon opens its wings; no physical jump is claimed."
    }
    for state, meaning in meanings.items():
        keys = []
        for frame in range(0,97,8):
            t = frame/96
            envelope = math.sin(math.pi*t)**2
            wave = math.sin(math.tau*t)
            values = {"head":{"rotation":[-.022*envelope,.025*wave,.026*wave]},
                      "dragon_head":{"rotation":[.045*wave,0,.05*wave]},
                      "dragon_tail":{"rotation":[0,0,.07*wave]}}
            if state == "idle":
                values["torso"] = {"rotation":[.009*wave,0,.008*wave]}
            else:
                name = {"making_something":"craft","working":"inspect","milestone_level_up":"celebrate"}[state]
                for bone, transform in target_poses[name].items():
                    v = [x*envelope for x in transform["rotation"]]
                    if state == "making_something" and bone.startswith("forearm"):
                        v[2] += .06*math.sin(math.tau*3*t)*envelope*(1 if bone.endswith("left") else -1)
                    values[bone] = {"rotation":[round(x,6) for x in v]}
                for side in ("left", "right"):
                    upper = rotation(values[f"upper_arm_{side}"]["rotation"])
                    fore = rotation(values[f"forearm_{side}"]["rotation"])
                    wrist = euler_xyz((upper @ fore).T)
                    assert np.linalg.norm(upper @ fore @ rotation(wrist) - np.eye(3)) < .00001
                    values[f"hand_{side}"] = {"rotation": wrist}
                if state == "working":
                    values["head"] = {"rotation":[-.11*envelope,0,.045*wave]}
                if state == "milestone_level_up":
                    values["dragon_wing_outer"] = {"rotation":[0,0,-.35*envelope]}
                    values["dragon_wing_inner"] = {"rotation":[0,0,.25*envelope]}
            keys.append({"frame":frame,"bones":values})
        d["states"].append({"id":state,"meaning":meaning,"duration":4,"loop":True,"keyframes":keys})
    d["authoring_checks"] = {"arm_reach_residual_m":residuals,"provider_calls":0,"provider_cost_usd":0}
    return d


def main():
    parser = argparse.ArgumentParser(description="Author the detailed panda or check its exact reproducibility")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    scene = build_scene()
    path = ROOT / "examples/scenes/muse-panda-quality-v5.json"
    if args.check:
        assert json.loads(path.read_text()) == scene, "Panda recipe differs from authored scene"
        print("Panda recipe matches the saved scene exactly")
        return
    with path.open("x") as handle:
        json.dump(scene,handle,separators=(",",":"),allow_nan=False)
        handle.write("\n")
    print(json.dumps({"path":str(path),"parts":len(scene["parts"]),"fur_strands":sum(p.get("fur",{}).get("count",0) for p in scene["parts"]),
                      "control_vertices":sum(len(p.get("vertices",[])) for p in scene["parts"]),"bytes":path.stat().st_size,
                      "arm_reaches":scene["authoring_checks"]}))


if __name__ == "__main__":
    main()
