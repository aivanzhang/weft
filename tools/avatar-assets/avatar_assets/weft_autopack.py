"""Pack a generated avatar for Weft without hand-written per-asset configuration.

Delivery previously needed an author to write the ten role-to-bone names, a visual scale
and an orientation for every asset. This derives all three: roles come from the automatic
rig resolver, scale from the model's own height against Weft's body height, and the
orientation is the fixed Y-up-to-Z-up turn the viewer expects. An asset whose rig cannot
be resolved is refused here rather than delivered with a guessed mapping.

Scale and orientation are appearance only; they never change collision geometry, physics
or task scoring. A packed asset is importable, which is not the same as visually
qualified under posed deformation.
"""

import json
import math
from pathlib import Path

import numpy as np

from avatar_assets.rig_mapping import RigMappingError, resolve_from_glb
from avatar_assets.safety import require
from avatar_assets.three_d import _GLB
from avatar_assets.weft_delivery import ROOT_NAME, prepare_weft_delivery

# Weft's reconstructed bodies stand about this tall in model units; matching it keeps a
# delivered avatar proportionate to the physics body it dresses.
TARGET_BODY_HEIGHT = 1.4975
# glTF is Y-up; Weft's world is Z-up, so every delivery turns +90 degrees about X.
Y_UP_TO_Z_UP = [math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]


def model_extent(glb):
    """Axis-aligned extent of every skinned and unskinned mesh position."""
    points = []
    for mesh in glb.doc.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            accessor = primitive.get("attributes", {}).get("POSITION")
            if accessor is not None:
                points.append(np.array(glb.accessor(accessor, "VEC3", {5126}), dtype=float))
    require(points, "Model has no mesh positions to measure")
    stacked = np.concatenate(points)
    size = stacked.max(axis=0) - stacked.min(axis=0)
    require(np.isfinite(size).all() and size.max() > 1e-6, "Model extent is degenerate")
    return size


def _world_positions(doc):
    """Rest world translation of every node, following parent transforms."""
    parents = {}
    for index, node in enumerate(doc.get("nodes", [])):
        for child in node.get("children", []):
            parents[child] = index

    def local(node):
        if "matrix" in node:
            return np.array(node["matrix"], dtype=float).reshape(4, 4).T
        matrix = np.eye(4)
        x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
        matrix[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=float) * np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=float)
        matrix[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
        return matrix

    world = {}

    def resolve(index):
        if index in world:
            return world[index]
        matrix = local(doc["nodes"][index])
        parent = parents.get(index)
        world[index] = matrix if parent is None else resolve(parent) @ matrix
        return world[index]

    return {index: resolve(index)[:3, 3] for index in range(len(doc.get("nodes", [])))}


def derive_orientation(glb, mapping):
    """Rotation taking the rig's own torso-up direction to Weft's +Z, as a unit quaternion.

    Generated rigs do not share an up axis, so a fixed turn lays some avatars on their
    side. Measuring hips-to-torso in the model's rest pose orients each asset by its own
    skeleton instead.
    """
    names = {node.get("name"): index for index, node in enumerate(glb.doc.get("nodes", []))}
    positions = _world_positions(glb.doc)
    try:
        torso = positions[names[mapping["torso"]]]
        hips = (positions[names[mapping["left_hip_body"]]]
                + positions[names[mapping["right_hip_body"]]]) / 2
    except KeyError as error:
        raise RigMappingError("Cannot measure rig orientation: missing mapped bone") from error
    up = torso - hips
    norm = float(np.linalg.norm(up))
    require(norm > 1e-6, "Rig torso and hips coincide; orientation cannot be measured")
    up = up / norm
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(up, target))
    if sine < 1e-9:
        # Already aligned, or exactly inverted: turn about X so the result stays finite.
        return ([0.0, 0.0, 0.0, 1.0] if cosine > 0 else [1.0, 0.0, 0.0, 0.0]), up.tolist()
    axis = axis / sine
    angle = math.atan2(sine, cosine)
    half = angle / 2
    quaternion = [float(axis[0] * math.sin(half)), float(axis[1] * math.sin(half)),
                  float(axis[2] * math.sin(half)), float(math.cos(half))]
    length = math.sqrt(sum(value * value for value in quaternion))
    return [value / length for value in quaternion], up.tolist()


def derive_settings(model, *, asset_id, display_name, rights, provenance,
                    target_height=TARGET_BODY_HEIGHT):
    """Return complete delivery settings for a model, or raise RigMappingError."""
    require(isinstance(asset_id, str) and 1 <= len(asset_id) <= 100, "Explicit asset id required")
    require(isinstance(display_name, str) and 1 <= len(display_name) <= 128, "Explicit display name required")
    require(type(target_height) in (int, float) and math.isfinite(target_height)
            and 0.1 <= target_height <= 10, "Target body height must be bounded")
    glb = _GLB(Path(model))
    mapping, report = resolve_from_glb(glb, root_name=ROOT_NAME)
    size = model_extent(glb)
    orientation, measured_up = derive_orientation(glb, mapping)
    # Height is measured along the rig's own up direction, not a fixed axis.
    scale = float(target_height / max(float(abs(np.asarray(measured_up) @ size)), 1e-6))
    require(0.01 <= scale <= 100, "Derived visual scale falls outside the supported delivery range")
    settings = {"asset_id": asset_id, "display_name": display_name, "rights": rights,
                "provenance": provenance, "body_roles": mapping, "scale": scale,
                "orientation_xyzw": orientation}
    return settings, {**report, "derived_scale": scale, "model_extent": size.tolist(),
                      "measured_up": measured_up, "orientation_xyzw": orientation,
                      "target_body_height": target_height, "source_sha256": glb.hash}


def autopack(model, out_dir, *, asset_id, display_name, rights, provenance,
             target_height=TARGET_BODY_HEIGHT, triangles=6000, timeout=120):
    """Convert a native generated avatar into a Weft-importable ZIP with no authored config.

    Runs the ordinary bounded delivery (native decimation then packing) against settings
    derived from the model itself, so a generated asset needs no hand-written mapping.
    """
    model = Path(model)
    out_dir = Path(out_dir)
    require(not out_dir.exists(), "Require a fresh delivery directory")
    settings, report = derive_settings(model, asset_id=asset_id, display_name=display_name,
                                       rights=rights, provenance=provenance, target_height=target_height)
    out_dir.mkdir(parents=True)
    settings_path = out_dir / "derived-settings.json"
    settings_path.write_text(json.dumps(settings, indent=1) + "\n")
    delivery = prepare_weft_delivery(model, settings_path, out_dir / "delivery",
                                     expected_source_sha256=report["source_sha256"],
                                     triangles=triangles, timeout=timeout)
    archive = out_dir / "delivery/weft-avatar.zip"
    require(archive.is_file(), "Delivery did not produce its ZIP")
    receipt = {"profile": "weft-autopack-v1", "zip": str(archive), "zip_bytes": archive.stat().st_size,
               "rig": report, "delivery": delivery, "settings": settings,
               "scope": "Automatic role, scale and orientation derivation; appearance only. "
                        "Importable is not visually qualified under posed deformation."}
    (out_dir / "autopack-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    return receipt


def can_pack(model):
    """Report whether a model's rig resolves, without writing anything."""
    try:
        glb = _GLB(Path(model))
        mapping, report = resolve_from_glb(glb, root_name=ROOT_NAME)
        return {"supported": True, "reason": None, "mapping": mapping, "bones": report["bones"]}
    except (RigMappingError, ValueError) as error:
        return {"supported": False, "reason": str(error)[:400], "mapping": None, "bones": None}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    receipt = autopack(Path(args.model).absolute(), Path(args.out).absolute(),
        asset_id=args.asset_id, display_name=args.name,
        rights={"holder": "Studio generation", "license": "internal development",
                "source": "avatar-assets studio",
                "declaration": "Generated asset; rights not independently verified"},
        provenance={"scope": "Automatic rig-role delivery of a studio generation"})
    print(json.dumps({"zip": receipt["zip"], "zip_bytes": receipt["zip_bytes"]}))


if __name__ == "__main__":
    main()
