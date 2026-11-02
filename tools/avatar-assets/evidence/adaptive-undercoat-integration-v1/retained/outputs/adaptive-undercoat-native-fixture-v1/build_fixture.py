"""Analytic mixed-UV adapter/export fixture; no hair, renderer or avatar-quality claim."""

import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"sources/native"))
from blender_fur_undercoat import apply_undercoat, NORMAL_UV1_NAME
from groom_flow import prepare_flow


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    with (HERE/name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes()).hexdigest()


def pixels(image):
    values = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(values)
    return array_hash(values)


def basecolor(material):
    shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    links = list(shader.inputs["Base Color"].links)
    require(len(links) == 1 and links[0].from_node.type == "TEX_IMAGE", "Fixture requires direct source base color")
    texture = links[0].from_node
    return {"default": list(shader.inputs["Base Color"].default_value), "source_node": texture.name,
            "image": texture.image.name, "image_pixels_sha256": pixels(texture.image),
            "image_color_space": texture.image.colorspace_settings.name,
            "vector_links": [(link.from_node.name, link.from_socket.name) for link in texture.inputs["Vector"].links],
            "extension": texture.extension, "interpolation": texture.interpolation}


def material_graph(material):
    nodes = []
    for node in material.node_tree.nodes:
        inputs = {}
        for socket in node.inputs:
            value = getattr(socket, "default_value", None)
            if isinstance(value, (int, float, bool, str)):
                inputs[socket.identifier] = value
            elif value is not None:
                try:
                    inputs[socket.identifier] = list(value)
                except TypeError:
                    inputs[socket.identifier] = getattr(value, "name", str(type(value)))
        nodes.append({"name": node.name, "type": node.type, "inputs": inputs,
                      "image": node.image.name if node.type == "TEX_IMAGE" and node.image else None})
    return {"nodes": sorted(nodes, key=lambda row: row["name"]),
            "links": sorted((link.from_node.name, link.from_socket.identifier, link.to_node.name,
                             link.to_socket.identifier) for link in material.node_tree.links)}


def geometry(mesh):
    return {"vertices": [list(vertex.co) for vertex in mesh.vertices],
            "faces": [list(face.vertices) for face in mesh.polygons],
            "uv0": [list(value.uv) for value in mesh.uv_layers[0].data],
            "active_uv_index": mesh.uv_layers.active_index,
            "uv0_active_render": mesh.uv_layers[0].active_render}


def main():
    started = time.monotonic()
    prepared = json.loads((HERE/"prepared.json").read_text())
    for row in prepared["frozen_files"]:
        require(sha(HERE/row["path"]) == row["sha256"], "Frozen fixture input changed")
    config = json.loads((HERE/"fixture.json").read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.use_scripts_auto_execute = False
    mesh = bpy.data.meshes.new("AnalyticMixedUVSurface")
    mesh.from_pydata(config["vertices"], [], config["faces"])
    mesh.update()
    obj = bpy.data.objects.new("AnalyticMixedUVFixture", mesh)
    bpy.context.scene.collection.objects.link(obj)
    uv0 = mesh.uv_layers.new(name="OriginalColorUV0")
    for face, values in zip(mesh.polygons, config["triangle_uv0"]):
        for loop, uv in zip(face.loop_indices, values):
            uv0.data[loop].uv = uv
    mesh.uv_layers.active_index, uv0.active_render = 0, True
    (HERE/"textures").mkdir()
    image = bpy.data.images.new("OriginalFixtureBaseColor", width=4, height=4, alpha=True)
    image.colorspace_settings.name = "sRGB"
    rgba = np.asarray(config["basecolor_pixels"], dtype=np.float32)
    image.pixels.foreach_set(rgba.ravel())
    image.filepath_raw, image.file_format = str(HERE/"textures/original-basecolor.png"), "PNG"
    image.save()
    image.pack()
    originals = []
    for index, name in enumerate(config["material_names"]):
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
        shader.inputs["Roughness"].default_value = .63+index*.08
        shader.inputs["Metallic"].default_value = .17
        texture = material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.name, texture.image = "UnchangedSourceColor", image
        material.node_tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
        mesh.materials.append(material)
        originals.append(material)
    for face, index in zip(mesh.polygons, config["material_indices"]):
        face.material_index = index
    mask = np.asarray(config["selected_vertices"], dtype=bool)
    points = np.asarray(config["vertices"], dtype=float)
    flow = prepare_flow(points.min(axis=0).tolist(), np.ptp(points, axis=0).tolist(), [.3, .8, .1], .6,
                        config["groom_flow"])
    before = geometry(mesh)
    graphs = [material_graph(material) for material in originals]
    colors = [basecolor(material) for material in originals]
    original_pixels = pixels(image)
    negatives = []
    for kind in ("existing_uv1", "conflicting_attribute"):
        copy = obj.copy()
        copy.data = mesh.copy()
        bpy.context.scene.collection.objects.link(copy)
        if kind == "existing_uv1":
            copy.data.uv_layers.new(name="PreexistingUV1")
        else:
            copy.data.attributes.new(name=NORMAL_UV1_NAME, type="FLOAT", domain="POINT")
        folder = HERE/kind
        folder.mkdir()
        state = geometry(copy.data)
        counts = (len(bpy.data.materials), len(bpy.data.images))
        try:
            apply_undercoat(copy, mask, flow, config["settings"], folder,
                            export_vertex_budget=config["export_vertex_budget"])
        except ValueError as error:
            expected = "preexisting UV1" if kind == "existing_uv1" else "conflicts with existing"
            require(expected in str(error), "Unexpected negative-case refusal: "+str(error))
            require(geometry(copy.data) == state and counts == (len(bpy.data.materials), len(bpy.data.images))
                    and not list(folder.iterdir()), "Refused input was mutated")
            negatives.append({"case": kind, "rejected": True, "reason": str(error), "no_bake_or_material_mutation": True})
        else:
            raise ValueError("Invalid existing UV configuration was accepted")
        data = copy.data
        bpy.data.objects.remove(copy, do_unlink=True)
        bpy.data.meshes.remove(data)
    report = apply_undercoat(obj, mask, flow, config["settings"], HERE,
                            export_vertex_budget=config["export_vertex_budget"])
    require(geometry(mesh) == before, "Original positions/topology/UV0/active settings changed")
    require([material_graph(material) for material in originals] == graphs and pixels(image) == original_pixels,
            "Original material graph or source color pixels changed")
    require(len(mesh.uv_layers) == 2 and mesh.uv_layers[1].name == NORMAL_UV1_NAME
            and not mesh.uv_layers[1].active_render, "Authored UV1 or UV flags missing")
    methods = {row["region_id"]: row["method"] for row in report["plan"]["regions"]}
    require(methods == {0: "authored_uv1", 1: "keep_uv0"}, "Fixture did not exercise both adaptive branches")
    for row in report["materials"]:
        clone = bpy.data.materials[row["material"]]
        require(basecolor(clone) == colors[row["source_material_index"]], "Cloned basecolor graph changed")
    require(mesh.materials[mesh.polygons[3].material_index] is originals[0], "Excluded triangle material was replaced")
    faces = []
    for index, face in enumerate(mesh.polygons):
        faces.append({"face_index": index, "positions_blender_m": [list(mesh.vertices[v].co) for v in face.vertices],
                      "uv0_blender": [list(mesh.uv_layers[0].data[v].uv) for v in face.loop_indices],
                      "uv1_blender": [list(mesh.uv_layers[1].data[v].uv) for v in face.loop_indices],
                      "material": mesh.materials[face.material_index].name,
                      "expected_normal_uv": 1 if index < 2 else 0 if index == 2 else None})
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(filepath=str(HERE/"model.glb"), export_format="GLB", use_selection=True,
        export_animations=False, export_yup=True, export_skins=False, export_normals=True, export_tangents=True,
        export_image_format="AUTO", export_morph=False, export_extras=False, export_materials="EXPORT")
    require(geometry(mesh) == before and [material_graph(material) for material in originals] == graphs
            and [basecolor(material) for material in originals] == colors and not mesh.uv_layers[1].active_render
            and pixels(image) == original_pixels, "Export changed source geometry, UV flags or color data")
    bpy.ops.wm.save_as_mainfile(filepath=str(HERE/"source.blend"), check_existing=False)
    write("native-fixture.json", {"schema_version": 1, "kind": "analytic_adapter_export_fixture", "quality_approved": False,
        "blender_version": bpy.app.version_string, "source_graphs": graphs, "source_basecolor": colors,
        "source_basecolor_pixel_sha256": original_pixels, "original_geometry": before, "faces": faces,
        "original_data_unchanged": True, "uv_layers": [uv.name for uv in mesh.uv_layers],
        "active_uv_index": mesh.uv_layers.active_index, "active_render_flags": [uv.active_render for uv in mesh.uv_layers],
        "negative_cases": negatives, "undercoat": report, "model_sha256": sha(HERE/"model.glb"),
        "renders": 0, "provider_calls": 0, "elapsed_seconds": time.monotonic()-started})
    print(json.dumps({"status": "analytic_fixture_exported_needs_host_verification", "model_sha256": sha(HERE/"model.glb"),
                      "renders": 0, "elapsed_seconds": time.monotonic()-started}, sort_keys=True))


if __name__ == "__main__":
    main()
