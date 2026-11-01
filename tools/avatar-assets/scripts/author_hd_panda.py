"""Deterministic Muse-inspired face/dragon refinement; no recovered geometry or paid calls."""

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from author_quality_companions import color, fur, mesh, primitive, smooth_sections, tube
from author_quality_panda import build_scene as build_base_scene
from author_quality_panda import ellipsoid

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "examples/scenes/muse-panda-hd-v1.json"
HEAD_CENTER, HEAD_RADII = [0, -.015, 1.785], [.59, .43, .505]
DRAGON_CENTER, DRAGON_RADII = [-.64, -.16, 1.725], [.174, .136, .167]


def muzzle_depth(x, z):
    return (.075*math.exp(-(x/.235)**4-((z-1.690)/.145)**2)
            + .024*math.exp(-(x/.19)**2-((z-1.602)/.083)**2))


def dragon_snout_depth(x, z):
    return .070*math.exp(-(x/.128)**4-((z-1.666)/.080)**2)


def front_surface(x, z, center, radii, cheeks, sculpt):
    nz = (z-center[2])/radii[2]
    width = radii[0]*(1+cheeks*math.exp(-((nz+.28)/.55)**2))
    nx = (x-center[0])/width
    sine = math.sqrt(max(0, 1-nz*nz))
    front = math.sqrt(max(.0001, 1-nx*nx-nz*nz))
    facing = min(1, front/max(sine, .0001))
    return center[1]-radii[1]*front-sculpt(x-center[0], z)*facing**4


def head_front(x, z):
    return front_surface(x, z, HEAD_CENTER, HEAD_RADII, .13, muzzle_depth)


def dragon_front(x, z):
    return front_surface(x, z, DRAGON_CENTER, DRAGON_RADII, .09, dragon_snout_depth)


def sculpted_head(name, center, radii, cheeks, sculpt, material, bone, segments, rings):
    part = ellipsoid(name, center, radii, material, bone, cheeks, segments, rings)
    for vertex in part["vertices"]:
        nz = (vertex[2]-center[2])/radii[2]
        span = radii[1]*math.sqrt(max(0, 1-nz*nz))
        facing = max(0, min(1, (center[1]-vertex[1])/max(span, .00001)))
        vertex[1] = round(vertex[1]-sculpt(vertex[0]-center[0], vertex[2])*facing**4, 6)
    return part


def eye_patch(name, side):
    vertices, uv, faces = [[side*.234, head_front(side*.234, 1.879)-.023, 1.879]], [[.5, .5]], []
    segments, rings, angle = 48, 7, side*.25
    for j in range(1, rings+1):
        r = j/rings
        for i in range(segments):
            t = math.tau*i/segments
            a, b = .160*r*math.cos(t)*(1-.14*math.sin(t)), .213*r*math.sin(t)
            x = side*.234+a*math.cos(angle)-b*math.sin(angle)
            z = 1.879+a*math.sin(angle)+b*math.cos(angle)
            vertices.append([x, head_front(x, z)-.023, z])
            uv.append([.5+r*math.cos(t)/2, .5+r*math.sin(t)/2])
            if j == 1:
                faces.append([0, 1+i, 1+(i+1)%segments])
            else:
                a0, b0 = 1+(j-2)*segments+i, 1+(j-1)*segments+i
                faces.append([a0, b0, 1+(j-1)*segments+(i+1)%segments,
                              1+(j-2)*segments+(i+1)%segments])
    return fur(mesh(name, vertices, faces, "charcoal", "head", uv), 600, .009, .0015, 85+(side>0))


def lens(name, center, radius_x, radius_z, depth, material, bone, segments=48, rings=5):
    """Convex opaque eye shell with radial UV0; highlights come from lighting, not paint."""
    x, y, z = center
    vertices, uv, faces = [[x, y-depth, z]], [[.5, .5]], []
    for j in range(1, rings+1):
        r = j/rings
        for i in range(segments):
            t = math.tau*i/segments
            vertices.append([x+radius_x*r*math.cos(t), y-depth*(1-r*r), z+radius_z*r*math.sin(t)])
            uv.append([.5+r*math.cos(t)/2, .5+r*math.sin(t)/2])
            if j == 1:
                faces.append([0, 1+i, 1+(i+1)%segments])
            else:
                a, b = 1+(j-2)*segments+i, 1+(j-1)*segments+i
                faces.append([a, b, 1+(j-1)*segments+(i+1)%segments,
                              1+(j-2)*segments+(i+1)%segments])
    back = len(vertices)
    vertices.append([x, y+.003, z])
    uv.append([.5, .5])
    for i in range(segments):
        faces.append([back, 1+(rings-1)*segments+(i+1)%segments, 1+(rings-1)*segments+i])
    return mesh(name, vertices, faces, material, bone, uv)


def shaped_nose():
    y = head_front(0, 1.741)-.012
    part = ellipsoid("soft_nose", [0, y, 1.741], [.087, .034, .045], "nose", "head", 0, 32, 20)
    for vertex in part["vertices"]:
        height = (vertex[2]-1.741)/.045
        vertex[0] = round(vertex[0]*(1+.27*height), 6)
        vertex[2] = round(vertex[2]-.004*(1-height*height), 6)
    return part


def facial_parts():
    parts = [fur(sculpted_head("cheeked_head", HEAD_CENTER, HEAD_RADII, .13, muzzle_depth,
                              "ivory", "head", 48, 32), 7800, .017, .0026, 82), shaped_nose()]
    for label, sign in (("left", 1), ("right", -1)):
        parts.append(eye_patch(f"eye_patch_{label}", sign))
        x, z = sign*.216, 1.851
        y = head_front(x, z)
        parts.append(primitive(f"eye_{label}", "sphere", [x, y-.036, z], [.082, .039, .090],
                               "sclera", "head", segments=32, rings=20))
        iris_center, iris_depth = [x-sign*.005, y-.071, z-.002], .013
        parts.append(lens(f"iris_{label}", iris_center, .059, .065, iris_depth, "iris", "head"))
        ratio = .65
        pupil_center = [iris_center[0], iris_center[1]-iris_depth*(1-ratio*ratio)-.0007, iris_center[2]]
        parts.append(lens(f"pupil_{label}", pupil_center, .059*ratio, .065*ratio,
                          iris_depth*ratio*ratio+.0012, "pupil", "head", 32, 4))
        for upper in (True, False):
            start, end = (.03, math.pi-.03) if upper else (math.pi+.10, math.tau-.10)
            samples = 24 if upper else 20
            points = []
            for i in range(samples+1):
                t = start+(end-start)*i/samples
                dx, dz = .079*math.cos(t), .0875*math.sin(t)
                eye_depth = .039*math.sqrt(1-(dx/.082)**2-(dz/.090)**2)
                points.append([x+dx, y-.036-eye_depth-.0015, z+dz])
            radii = [(.008 if upper else .0045)*(.42+.58*math.sin(math.pi*i/samples)) for i in range(samples+1)]
            parts.append(tube(f"{'upper' if upper else 'lower'}_lid_{label}", points, radii, "charcoal", "head", 8))
        parts.append(primitive(f"nostril_{label}", "sphere", [sign*.047, head_front(0, 1.741)-.035, 1.719],
                               [.012, .005, .007], "pupil", "head", segments=16, rings=10))
    philtrum = [[0, head_front(0, z)-.005, z] for z in [1.706-i*.010 for i in range(8)]]
    parts.append(tube("philtrum", philtrum, [.0035]*7+[.0025], "nose", "head", 8))
    smile, radii = [], []
    for i in range(33):
        x = -.205+.410*i/32
        z = 1.624+.072*(abs(x)/.205)**1.8
        smile.append([x, head_front(x, z)-.0055, z])
        radii.append(.0022+.0024*math.sin(math.pi*i/32))
    parts.append(tube("smile", smile, radii, "nose", "head", 8))
    return parts


def wing_parts(label, sign, origin):
    """Two scalloped membrane bays share a root and retain closed thickness."""
    tips = [[sign*.262, .032, .199], [sign*.244, .010, -.106], [sign*.115, -.010, -.184]]
    controls = [[sign*.128, -.014, .005], [sign*.084, -.018, -.012]]
    edge = []
    for bay in range(2):
        for i in range(13):
            if bay and i == 0:
                continue
            t = i/12
            edge.append([(1-t)**2*tips[bay][k]+2*(1-t)*t*controls[bay][k]+t*t*tips[bay+1][k] for k in range(3)])
    width, rows = len(edge), 6
    front = [origin]
    uv = [[0, .5]]
    for j in range(1, rows+1):
        r = j/rows
        for i, endpoint in enumerate(edge):
            p = [origin[k]+r*endpoint[k] for k in range(3)]
            p[1] -= .016*math.sin(math.pi*r)*(.35+.65*math.sin(math.pi*i/(width-1))**2)
            front.append(p)
            uv.append([r, i/(width-1)])
    faces = []
    for j in range(rows):
        for i in range(width-1):
            if j == 0:
                faces.append([0, 1+i, 1+i+1])
            else:
                a, b = 1+(j-1)*width+i, 1+j*width+i
                faces.append([a, b, b+1, a+1])
    if sign > 0:
        faces = [list(reversed(face)) for face in faces]
    count = len(front)
    vertices = front+[[v[0], v[1]+.005, v[2]] for v in front]
    surface_faces = deepcopy(faces)
    faces += [[i+count for i in reversed(face)] for face in surface_faces]
    edges = {}
    for face in surface_faces:
        for a, b in zip(face, face[1:]+face[:1]):
            edges.setdefault(tuple(sorted((a,b))), []).append((a,b))
    for occurrences in edges.values():
        if len(occurrences) == 1:
            a, b = occurrences[0]
            faces.append([b, a, a+count, b+count])
    bone = f"dragon_wing_{label}"
    result = [mesh(f"scalloped_wing_{label}", vertices, faces, "wing", bone, uv*2)]
    for index, tip in enumerate(tips):
        points = []
        for j in range(9):
            r = j/8
            points.append([origin[0]+r*tip[0], origin[1]+r*tip[1]-.006*math.sin(math.pi*r),
                           origin[2]+r*tip[2]+.012*math.sin(math.pi*r)])
        result.append(tube(f"wing_spar_{label}_{index}", points, [.011*(1-j/8)+.002 for j in range(9)],
                           "dragon_green", bone, 8))
    rim = [[origin[k]+point[k] for k in range(3)] for point in edge]
    result.append(tube(f"wing_rim_{label}", rim, .003, "dragon_green", bone, 6))
    return result


def crown_scales():
    vertices, uv, faces = [], [], []
    centers = []
    for row in range(3):
        for i in range(5-row):
            centers.append((-.64+(i-(4-row)/2)*.026, 1.819+row*.023, .014-row*.0015, .012))
    for sign in (-1, 1):
        for row in range(3):
            centers.append((-.64+sign*(.134-row*.004), 1.718+row*.028, .009, .015))
    for x, z, rx, rz in centers:
        start = len(vertices)
        vertices.append([x, dragon_front(x, z)-.004, z])
        uv.append([.5, .5])
        for i in range(12):
            t = math.tau*i/12
            xx, zz = x+rx*math.cos(t), z+rz*math.sin(t)
            vertices.append([xx, dragon_front(xx, zz)-.0008, zz])
            uv.append([.5+math.cos(t)/2, .5+math.sin(t)/2])
            faces.append([start, start+1+i, start+1+(i+1)%12])
        back = len(vertices)
        vertices.append([x, dragon_front(x, z)+.001, z])
        uv.append([.5, .5])
        for i in range(12):
            faces.append([back, start+1+(i+1)%12, start+1+i])
    return mesh("dragon_crown_scales", vertices, faces, "dragon_green", "dragon_head", uv)


def dragon_parts():
    parts = [sculpted_head("dragon_head_shape", DRAGON_CENTER, DRAGON_RADII, .09, dragon_snout_depth,
                           "dragon_green", "dragon_head", 40, 24), crown_scales()]
    neck = smooth_sections([[-.58,-.035,1.51,.064],[-.605,-.085,1.575,.074],[-.64,-.14,1.636,.090]], 4)
    weights = [{"dragon_root": round(1-i/(len(neck)-1), 6), "dragon_head": round(i/(len(neck)-1), 6)} for i in range(len(neck))]
    weights = [{key: value for key, value in row.items() if value > 0} for row in weights]
    parts.append(tube("dragon_neck", [row[:3] for row in neck], [row[3] for row in neck],
                      "dragon_green", "dragon_root", 16, weights))
    for label, sign in (("left", 1), ("right", -1)):
        x, z = -.64+sign*.075, 1.753
        y = dragon_front(x, z)
        parts.append(primitive(f"dragon_eye_{label}", "sphere", [x, y-.019, z], [.042,.024,.049],
                               "sclera", "dragon_head", segments=24, rings=16))
        iris = [x-sign*.003, y-.041, z-.001]
        parts.append(lens(f"dragon_iris_{label}", iris, .029, .034, .007, "iris", "dragon_head", 32, 4))
        ratio = .68
        parts.append(lens(f"dragon_pupil_{label}", [iris[0],iris[1]-.007*(1-ratio*ratio)-.0007,iris[2]],
                          .029*ratio,.034*ratio,.007*ratio*ratio+.0008,"pupil","dragon_head",24,3))
        lid = []
        for i in range(17):
            t = math.pi*i/16
            dx, dz = .040*math.cos(t), .047*math.sin(t)
            depth = .024*math.sqrt(1-(dx/.042)**2-(dz/.049)**2)
            lid.append([x+dx,y-.019-depth-.001,z+dz])
        parts.append(tube(f"dragon_eyelid_{label}", lid, .004, "dragon_green", "dragon_head", 6))
        nostril_x, nostril_z = -.64+sign*.053, 1.685
        parts.append(primitive(f"dragon_nostril_{label}", "sphere",
                               [nostril_x,dragon_front(nostril_x,nostril_z)-.003,nostril_z],
                               [.010,.004,.007],"nose","dragon_head",segments=16,rings=10))
        controls = [[-.64+sign*.111,-.105,1.841,.034],[-.64+sign*.139,-.061,1.902,.027],
                    [-.64+sign*.177,-.023,1.946,.015],[-.64+sign*.194,-.031,1.962,.0018]]
        rows = smooth_sections(controls, 5)
        radii = [max(.0018,row[3])*(1+.055*math.sin(i*math.tau/4)) for i,row in enumerate(rows)]
        parts.append(tube(f"horn_{label}",[row[:3] for row in rows],radii,"dragon_horn","dragon_head",12))
        for index in range(2):
            controls = [[-.64+sign*.154,-.058,1.696+index*.056,.023],
                        [-.64+sign*.213,-.043,1.708+index*.065,.016],
                        [-.64+sign*.239,-.058,1.734+index*.062,.0018]]
            rows = smooth_sections(controls, 4)
            parts.append(tube(f"frill_{label}_{index}",[row[:3] for row in rows],
                              [max(.0018,row[3]) for row in rows],"dragon_green","dragon_head",10))
    smile = []
    for i in range(25):
        x = -.64-.104+.208*i/24
        z = 1.637+.028*((x+.64)/.104)**2
        smile.append([x, dragon_front(x,z)-.003, z])
    parts.append(tube("dragon_smile", smile, .0025, "dragon_horn", "dragon_head", 6))
    for label, sign, origin in (("outer",-1,[-.67,.035,1.49]),("inner",1,[-.49,.035,1.49])):
        parts.extend(wing_parts(label, sign, origin))
    return parts


def refine_grooms(parts):
    """The GLB ribbons and native-only hairs share emitters and skin weights, not topology."""
    original_count = sum(part.get("fur", {}).get("count", 0) for part in parts)
    for part in parts:
        if "fur" not in part:
            continue
        old = deepcopy(part["fur"])
        face = part["name"] == "cheeked_head" or part["name"].startswith("eye_patch_")
        length = round(old["length"]*.85, 6)
        part["fur"].update(mode="cards", count=round(old["count"]*.4), length=length,
                           width=.018 if face else .025, direction=[0, 0, -1])
        part["native_groom"] = {"parents":max(16, round(old["count"]*6500/original_count)), "children":10,
                                "length":length, "radius":.00020 if face else .00025,
                                "roughness":.0007, "seed":old["seed"]}
    return original_count


def build_scene():
    scene = build_base_scene()
    original_bones, original_states, original_attachments = [deepcopy(scene[key]) for key in ("bones", "states", "attachments")]
    scene["id"] = "muse-panda-hd-v1"
    scene["source_edit_prompt"] = scene["prompt"]
    scene["user_hd_request"] = "you're working on making these avatars better right? I want the same textures and hd quality as the muse ones for example fura and hari"
    scene["prompt"] = (
        "Refine the existing Muse-inspired, complete animated panda and shoulder dragon as real 3D geometry. "
        "Use the supplied Muse portrait for its softly integrated muzzle, curved smile, warm large glossy eyes, "
        "slightly pear-shaped eye patches, fine fur and organic small green dragon. Sculpt the panda muzzle into "
        "the continuous head surface, not two attached cheek balls. Add shaped nostrils, curved fitted lids and "
        "convex warm iris geometry with explicit radial UV0 for native 1024 procedural iris color/normal maps. "
        "Preserve physical light reflections instead of painting glints. Give the dragon an integrated snout, "
        "a smooth weighted neck, curved tapered horns and frills, raised forehead scales, fitted lids and "
        "closed curved scalloped wing membranes with smooth finger spars. Use native 2048 fur surfaces on "
        "ivory/charcoal and 1024 scale surfaces on green skin, with bounded 512 secondary surfaces. Preserve "
        "all twenty-two bones, existing four motion clips, attachment hierarchy, proportions, framing and orb. "
        "Use a bounded fine-fiber cutout-card groom in the GLB and separately editable native Blender hair "
        "on matching emitters for native previews; native hairs are not secretly exported as browser hair. "
        "This is an authored visual interpretation, not recovered unseen anatomy, measured rigging, Weft "
        "runtime acceptance, hair simulation or demonstrated Muse-quality parity."
    )
    scene["style"] = "Soft storybook panda with sculpted expressive face, layered glossy warm eyes and an organically modeled emerald shoulder dragon"
    scene["authored_assumptions"][1] = "Browser fur is a bounded skinned cutout-card groom; native Blender hair is separate editable geometry and is not exported as GLB hair. Neither representation simulates hair dynamics."
    scene["authored_assumptions"].extend([
        "The new continuous head deformation authors the muzzle and chin from visual interpretation; it is not a scan, image-to-mesh recovery or a new face control rig.",
        "Iris and pupil geometry are glossy opaque curved shells. There is no refractive transparent cornea, painted catchlight or eye-tracking behavior.",
        "Every iris uses disk UV0 centered at [0.5,0.5], rim radius 0.5; the central pupil shell covers part of the native radial color texture.",
        "Dragon horn ridges, visible crown scales and thickened curved wing membranes are authored geometry; the remaining scales are procedural surface detail.",
        "The retained emitters previously used 17,820 triangular strands. This recipe requests 7,128 six-vertex skinned ribbons with fiber cutouts and approximately 71,500 native parent/child hairs; counts are configuration, not independently counted rendered fibers.",
        "The two-channel surface plan allocates 22,544,384 texture pixels; HD surface support must be validated before rendering this recipe.",
        "No new clip boundary, visual quality or hair-density claim follows from preserving authored keys. Native render and exported-model inspection remain required."
    ])
    materials = scene["materials"]
    materials["ear_inner"] = color("251F1B", .86, surface={"pattern":"fur","resolution":512,"strength":.18,"seed":73})
    materials["iris"] = color("967039", .12, surface={"pattern":"iris","resolution":1024,"strength":.50,"seed":177})
    materials["pupil"]["roughness"] = .045
    materials["sclera"]["roughness"] = .18
    materials["nose"]["roughness"] = .27
    materials["ivory"]["surface"].update(resolution=2048, strength=.29)
    materials["charcoal"]["surface"].update(resolution=2048, strength=.32)
    materials["dragon_green"]["surface"].update(resolution=1024, strength=.48)
    materials["dragon_green"]["roughness"] = .53
    materials["wing"]["surface"]["strength"] = .10
    replacements = facial_parts()+dragon_parts()
    by_name = {part["name"]:part for part in replacements}
    removed = {"muzzle_left", "muzzle_right", "dragon_snout"}
    parts, used = [], set()
    for part in scene["parts"]:
        name = part["name"]
        if name in removed:
            continue
        if name in by_name:
            parts.append(by_name[name])
            used.add(name)
        else:
            parts.append(part)
    parts.extend(part for part in replacements if part["name"] not in used)
    for part in parts:
        if part["name"].startswith(("paw_digit_", "toe_", "dragon_hand_")):
            part.update(segments=16, rings=10)
        if part["name"].startswith("inner_ear_"):
            part["scale"] = [.086, .012, .101]
    original_groom_count = refine_grooms(parts)
    scene["parts"] = parts
    scene["studio"] = {"ground_z":0, "world_strength":.3, "exposure":.25}
    texture_pixels = sum(2*m["surface"]["resolution"]**2 for m in materials.values() if "surface" in m)
    card_materials = {part["material"] for part in parts if part.get("fur",{}).get("mode") == "cards"}
    control_vertices = sum(len(p.get("vertices",[])) for p in parts)
    primitive_vertices = sum(p["segments"]*(p["rings"]-1)+2 for p in parts if p["shape"] == "sphere")
    scene["authoring_checks"].update({
        "parts":len(parts),"explicit_mesh_control_vertices":control_vertices,
        "primitive_vertex_estimate_before_uv_splits":primitive_vertices,
        "surface_vertex_estimate_before_uv_splits":control_vertices+primitive_vertices,
        "two_channel_texture_pixels":texture_pixels,
        "groom_strands_before_card_conversion":original_groom_count,
        "glb_hair_cards":sum(p.get("fur",{}).get("count",0) for p in parts),
        "glb_card_vertices":sum(p.get("fur",{}).get("count",0)*6 for p in parts),
        "texture_pixels_including_card_atlases":texture_pixels+len(card_materials)*1024**2,
        "native_hair_configured_upper_bound":sum(p.get("native_groom",{}).get("parents",0)*(p.get("native_groom",{}).get("children",0)+1) for p in parts),
        "bones_unchanged":scene["bones"]==original_bones,
        "clips_unchanged":scene["states"]==original_states,
        "attachments_unchanged":scene["attachments"]==original_attachments,
        "native_render_performed":False,"muse_quality_parity_verified":False,
        "reference_sha256":"c072bc988b0116ba6ac02641467cde3cddf65ac76a98be5a19c0c54553c9f3b2",
        "recipe_dependencies":{name:hashlib.sha256((ROOT/"scripts"/name).read_bytes()).hexdigest()
                               for name in ("author_quality_panda.py", "author_quality_companions.py")},
    })
    assert len(parts) <= 128 and control_vertices <= 30000
    assert texture_pixels <= 24_000_000
    assert scene["authoring_checks"]["native_hair_configured_upper_bound"] <= 80000
    assert len(scene["bones"]) == 22 and len(scene["states"]) == 4
    assert scene["bones"] == original_bones and scene["states"] == original_states and scene["attachments"] == original_attachments
    return scene


def main():
    parser = argparse.ArgumentParser(description="Author the HD panda geometry recipe or verify exact saved bytes")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    scene = build_scene()
    raw = (json.dumps(scene,separators=(",",":"),allow_nan=False)+"\n").encode()
    if args.check:
        if DESTINATION.read_bytes() != raw:
            raise SystemExit("HD panda recipe differs from the saved scene bytes")
    else:
        with DESTINATION.open("xb") as handle:
            handle.write(raw)
    print(json.dumps({"path":str(DESTINATION),"mode":"check" if args.check else "author",
                      "sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),
                      "checks":scene["authoring_checks"]},sort_keys=True))


if __name__ == "__main__":
    main()
