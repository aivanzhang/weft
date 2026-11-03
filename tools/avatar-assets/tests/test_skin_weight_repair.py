"""Chain-local repair removes cross-body blends without touching geometry or the document."""

import json
import math
import struct

import numpy as np
import pytest

from avatar_assets import skin_weight_repair as m
from avatar_assets.three_d import _GLB


def biped_glb(path, *, contaminate=True):
    """Two-leg rig: root -> {leg_l_upper -> leg_l_lower, leg_r_upper -> leg_r_lower}."""
    binary = bytearray()
    doc = {"asset": {"version": "2.0"}, "bufferViews": [], "accessors": []}

    def view(raw):
        binary.extend(b"\0" * (-len(binary) % 4))
        offset = len(binary)
        binary.extend(raw)
        doc["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw)})
        return len(doc["bufferViews"]) - 1

    def accessor(values, kind, component=5126):
        dtype = {5126: "<f4", 5123: "<u2", 5121: "<u1"}[component]
        array = np.asarray(values, dtype=dtype)
        row = {"bufferView": view(array.tobytes()), "componentType": component,
               "count": len(array), "type": kind}
        if kind == "VEC3":
            row.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        doc["accessors"].append(row)
        return len(doc["accessors"]) - 1

    # Two vertical strips of quads, one per leg; foot rows use millimeter edges like the
    # observed defect, with the opposite-leg blend alternating across adjacent vertices.
    positions, joints, weights, triangles = [], [], [], []
    for leg, x in ((0, -0.25), (1, 0.25)):
        upper, lower = 1 + 2 * leg, 2 + 2 * leg
        base = len(positions)
        for level in range(5):
            y = -level * 0.25
            near_foot = level >= 3
            for column, dx in enumerate((-0.002, 0.002) if near_foot else (-0.05, 0.05)):
                positions.append([x + dx, y, 0.0])
                primary = lower if near_foot else upper
                if contaminate and near_foot and leg == 0 and column == 0:
                    # The generated-source defect: a left-foot vertex blending the opposite leg.
                    joints.append([primary, 4, 0, 0])
                    weights.append([0.6, 0.4, 0.0, 0.0])
                else:
                    joints.append([primary, 0, 0, 0])
                    weights.append([1.0, 0.0, 0.0, 0.0])
        for level in range(4):
            a = base + 2 * level
            triangles += [[a, a + 1, a + 2], [a + 1, a + 3, a + 2]]
    attributes = {"POSITION": accessor(positions, "VEC3"),
                  "JOINTS_0": accessor(joints, "VEC4", 5123),
                  "WEIGHTS_0": accessor(weights, "VEC4")}
    indices = accessor(np.asarray(triangles).reshape(-1, 1), "SCALAR", 5123)
    names = ["root", "leg_l_upper", "leg_l_lower", "leg_r_upper", "leg_r_lower"]
    inverse = []
    offsets = {0: [0, 0, 0], 1: [-0.25, 0, 0], 2: [-0.25, -0.5, 0], 3: [0.25, 0, 0], 4: [0.25, -0.5, 0]}
    for slot in range(5):
        matrix = np.eye(4)
        matrix[:3, 3] = -np.asarray(offsets[slot], dtype=float)
        inverse.append(matrix.T.reshape(16).tolist())
    ibm = accessor(inverse, "MAT4")
    doc["nodes"] = [
        {"name": "root", "children": [1, 3], "translation": [0.0, 0.0, 0.0]},
        {"name": "leg_l_upper", "children": [2], "translation": [-0.25, 0.0, 0.0]},
        {"name": "leg_l_lower", "translation": [0.0, -0.5, 0.0]},
        {"name": "leg_r_upper", "children": [4], "translation": [0.25, 0.0, 0.0]},
        {"name": "leg_r_lower", "translation": [0.0, -0.5, 0.0]},
        {"name": "mesh", "mesh": 0, "skin": 0},
        {"name": "Root", "children": [0, 5]},
    ]
    doc.update(skins=[{"joints": [0, 1, 2, 3, 4], "inverseBindMatrices": ibm}],
               meshes=[{"name": "body", "primitives": [{"attributes": attributes, "indices": indices}]}],
               scenes=[{"nodes": [6]}], scene=0, buffers=[{"byteLength": len(binary)}])
    binary.extend(b"\0" * (-len(binary) % 4))
    encoded = json.dumps(doc, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary))
                     + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
                     + struct.pack("<II", len(binary), 0x004E4942) + binary)
    return path


def test_distances_follow_the_bone_tree():
    doc = {"nodes": [
        {"name": "root", "children": [1, 3]}, {"name": "l_up", "children": [2]}, {"name": "l_low"},
        {"name": "r_up", "children": [4]}, {"name": "r_low"},
    ]}
    distances = m.joint_tree_distances(doc, [0, 1, 2, 3, 4])
    assert distances[2][4] == 4 and distances[1][2] == 1 and distances[0][2] == 2
    assert distances[2][3] == 3


def test_repair_removes_cross_leg_influence_and_leaves_document_unchanged(tmp_path):
    source = biped_glb(tmp_path / "biped.glb")
    before = source.read_bytes()
    audit = m.repair_skinned_glb(source, tmp_path / "repaired.glb")
    assert source.read_bytes() == before
    original, repaired = _GLB(source), _GLB(tmp_path / "repaired.glb")
    assert repaired.doc == original.doc
    attrs = repaired.doc["meshes"][0]["primitives"][0]["attributes"]
    joints = np.array(repaired.accessor(attrs["JOINTS_0"]))
    weights = np.array(repaired.accessor(attrs["WEIGHTS_0"]))
    # Left-foot vertices must no longer blend the right leg; weights stay normalized.
    for row_joints, row_weights, position in zip(joints, weights, repaired.accessor(attrs["POSITION"])):
        active = {int(j) for j, w in zip(row_joints, row_weights) if w > 0}
        if position[0] < 0:
            assert active <= {0, 1, 2}
        assert abs(sum(row_weights) - 1) < 1e-5
    assert audit["primitives"][0]["repaired_groups"] >= 1
    # Surgical repair: only the defective region changes, the rest keeps authored weights.
    assert audit["primitives"][0]["changed_vertices"] <= 6
    assert audit["converged"] is True
    assert audit["source_sha256"] != audit["target_sha256"]


def test_probes_expose_contamination_and_accept_the_repair(tmp_path):
    source = biped_glb(tmp_path / "biped.glb")
    clean = biped_glb(tmp_path / "clean.glb", contaminate=False)
    contaminated = m.limb_probe_metrics(source)
    assert contaminated["total_severe_edges"] > 0
    m.repair_skinned_glb(source, tmp_path / "repaired.glb")
    repaired = m.limb_probe_metrics(tmp_path / "repaired.glb")
    baseline = m.limb_probe_metrics(clean)
    assert repaired["total_severe_edges"] == 0 == baseline["total_severe_edges"]
    # Geometry-driven assignment may differ slightly from the authored fixture, but it must
    # stay near-rigid and far below the contaminated clip's stretching.
    assert repaired["worst_edge_ratio"] < 1.5 < contaminated["worst_edge_ratio"]


def test_configuration_bounds_are_enforced(tmp_path):
    source = biped_glb(tmp_path / "biped.glb")
    with pytest.raises(ValueError, match="Unknown skin repair configuration"):
        m.repair_skinned_glb(source, tmp_path / "out.glb", config={"surprise": 1})
    with pytest.raises(ValueError, match="Chain distance"):
        m.repair_skinned_glb(source, tmp_path / "out.glb", config={"chain_distance": 9})
    m.repair_skinned_glb(source, tmp_path / "out.glb")
    with pytest.raises(ValueError, match="fresh"):
        m.repair_skinned_glb(source, tmp_path / "out.glb")
