"""Source-normal relationships and a cosine diagnostic, not a renderer or quality gate."""
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from glb_read import GLB, require

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(name, value):
    with (HERE/name).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def unit(values):
    lengths = np.linalg.norm(values, axis=-1)
    require(np.isfinite(values).all() and (lengths > 1e-8).all(), "Invalid normal/vector")
    return values/lengths[..., None]


def stats(values):
    values = np.asarray(values).reshape(-1)
    require(len(values) and np.isfinite(values).all(), "Invalid metric")
    return dict(zip(("min", "p05", "median", "p95", "max"), np.quantile(values, [0, .05, .5, .95, 1]).tolist()))


def angle(a, b, ignore_sign=False):
    dot = np.sum(a*b, axis=-1)
    if ignore_sign:
        dot = np.abs(dot)
    return np.degrees(np.arccos(np.clip(dot, -1, 1)))


def main():
    start = time.monotonic()
    prepared = json.loads((HERE/"prepared.json").read_text())
    for row in prepared["inputs"]+prepared["sources"]:
        require(sha(row["path"]) == row["sha256"], "Pinned evidence changed")
    source, candidate = GLB(prepared["source"]), GLB(prepared["candidate"])
    with zipfile.ZipFile(prepared["correspondence"]) as archive:
        require(len(archive.infolist()) <= 8 and sum(i.file_size for i in archive.infolist()) <= 4*1024**2, "NPZ bound")
    with np.load(prepared["correspondence"], allow_pickle=False) as arrays:
        ids = arrays["root_vertex_ids"]
        source_face_ids = arrays["root_face_ids"]
    require(ids.shape == (4800, 2) and ids.dtype.kind in "iu" and source_face_ids.shape == (4800,), "Unexpected root count")
    sp = source.doc["meshes"][0]["primitives"][5]
    fp = candidate.doc["meshes"][0]["primitives"][-1]
    source_points = source.array(sp["attributes"]["POSITION"], "VEC3", (5126,)).astype(np.float64)
    source_normals = unit(source.array(sp["attributes"]["NORMAL"], "VEC3", (5126,)).astype(np.float64))
    source_faces = source.array(sp["indices"], "SCALAR", (5121, 5123, 5125), 300000).reshape(-1, 3)
    positions = candidate.array(fp["attributes"]["POSITION"], "VEC3", (5126,)).astype(np.float64)
    normals_stored = candidate.array(fp["attributes"]["NORMAL"], "VEC3", (5126,)).astype(np.float64)
    colors = candidate.array(fp["attributes"]["COLOR_0"], "VEC3", (5126,)).astype(np.float64).reshape(4800, 4, 3)
    triangles = candidate.array(fp["indices"], "SCALAR", (5125,), 300000).reshape(-1, 3)
    require(positions.shape == (19200, 3) and normals_stored.shape == positions.shape and triangles.shape == (9600, 3)
            and (ids >= 0).all() and (ids < len(source_points)).all() and (source_face_ids >= 0).all()
            and (source_face_ids < len(source_faces)).all(), "Array/index bounds")
    geometric = unit(np.cross(positions[triangles[:, 1]]-positions[triangles[:, 0]], positions[triangles[:, 2]]-positions[triangles[:, 0]]))
    normals4 = unit(normals_stored).reshape(4800, 4, 3)
    require(np.max(np.abs(normals4-normals4[:, :1])) < 1e-12, "Existing fin normals are not flat")
    fin = normals4[:, 0]
    endpoint = source_normals[ids]
    surface = unit(endpoint.sum(axis=1))
    root = source_points[ids].mean(axis=1)
    base_xyz = source_points[source_faces[source_face_ids]]
    base_geometric = unit(np.cross(base_xyz[:, 1]-base_xyz[:, 0], base_xyz[:, 2]-base_xyz[:, 0]))
    fin_geo_dot = np.einsum("fvc,fc->fv", unit(normals_stored)[triangles], geometric)
    require((fin_geo_dot > .99999).all(), "Prior geometric-normal relationship no longer holds")
    oriented = angle(fin[:, None], endpoint)
    sign_independent = angle(fin[:, None], endpoint, True)
    endpoint_difference = angle(endpoint[:, 0], endpoint[:, 1])
    luminance = colors[:, :2].mean(axis=1)@np.array([.2126, .7152, .0722])
    bands = np.searchsorted([.1, .5], luminance, side="right")
    band_rows = [{"band": name, "fins": int((bands == i).sum()),
                  "fin_vs_source_endpoint_angle_degrees": stats(oriented[bands == i]),
                  "angle_after_best_global_sign_choice_degrees": stats(sign_independent[bands == i]),
                  "source_endpoint_normal_difference_degrees": stats(endpoint_difference[bands == i])}
                 for i, name in enumerate(("low_luminance", "middle_luminance", "high_luminance"))]
    hypothetical = {}
    for amount in prepared["settings"]["source_blend_weights"]:
        mixed = unit((1-amount)*fin+amount*surface)
        front = np.einsum("nc,nc->n", mixed, surface)
        hypothetical[str(amount)] = {"scope": "Hypothetical vectors only; no model or normals written",
            "angle_to_average_source_degrees": stats(angle(mixed, surface)),
            "front_side_dot_source": stats(front), "back_side_dot_source_after_gltf_reversal": stats(-front),
            "front_side_inward_count": int((front < 0).sum()), "back_side_inward_count": int((-front < 0).sum())}
    inspection = json.loads(Path(prepared["inspection"]).read_text())
    bounds = inspection["bounds_blender_z_up"]
    lo, hi = np.asarray(bounds["min"]), np.asarray(bounds["max"])
    center, extent = (lo+hi)/2, float((hi-lo).max())
    native_root = root[:, [0, 2, 1]]*np.array([1, -1, 1])
    native_surface = surface[:, [0, 2, 1]]*np.array([1, -1, 1])
    native_fin = fin[:, [0, 2, 1]]*np.array([1, -1, 1])
    native_geo = geometric[::2, [0, 2, 1]]*np.array([1, -1, 1])
    views = {name: unit(np.asarray(offset, dtype=float)[None])[0] for name, offset in prepared["settings"]["preview_view_offsets"].items()}
    light_directions = {name: unit(center+np.asarray(offset)*extent-native_root)
                        for name, offset in prepared["settings"]["preview_light_offsets"].items()}
    coefficients, backface_rows = [], []
    for view_name, view in views.items():
        subset = (native_surface@view) > prepared["settings"]["source_view_facing_dot"]
        front_face = (native_geo@view) > 0
        require(subset.any(), "Empty source-facing subset")
        signs = np.where(front_face, 1., -1.)
        shading = native_fin*signs[:, None]
        blend = unit(.25*native_fin+.75*native_surface)*signs[:, None]
        backface_rows.append({"view": view_name, "source_facing_fins": int(subset.sum()),
            "geometric_backfacing_fins": int((subset&~front_face).sum()),
            "75pct_source_blend_inward_after_double_sided_flip": int((subset&(np.einsum("nc,nc->n", blend, native_surface) < 0)).sum()),
            "scope": "Source-normal-facing subset, not visibility; body occlusion and pixel footprint are not evaluated."})
        for light_name, light in light_directions.items():
            source_cos = np.maximum(0, np.einsum("nc,nc->n", native_surface, light))
            current_cos = np.maximum(0, np.einsum("nc,nc->n", shading, light))
            mixed_cos = np.maximum(0, np.einsum("nc,nc->n", blend, light))
            difference = np.abs(current_cos-source_cos)
            coefficients.append({"view": view_name, "light_center": light_name,
                "source_facing_fins": int(subset.sum()), "source_cosine": stats(source_cos[subset]),
                "current_fin_cosine": stats(current_cos[subset]), "absolute_cosine_difference": stats(difference[subset]),
                "fraction_abs_difference_over_025": float((difference[subset] > .25).mean()),
                "75pct_source_blend_abs_difference_median": float(np.median(np.abs(mixed_cos[subset]-source_cos[subset])))})
    witnesses = []
    witness_order = np.argsort(-sign_independent.min(axis=1), kind="stable")
    for band in (0, 2):
        for ordinal in witness_order[bands[witness_order] == band][:4]:
            witnesses.append({"fin_ordinal": int(ordinal), "exported_fin_vertex_indices": [int(4*ordinal+i) for i in range(4)],
                "source_primitive": 5, "source_exported_vertex_indices": ids[ordinal].tolist(),
                "root_glTF_Y_up_m": root[ordinal].tolist(), "source_endpoint_normals_glTF": endpoint[ordinal].tolist(),
                "serialized_fin_normal_glTF": fin[ordinal].tolist(), "linear_endpoint_mean_rgb": colors[ordinal, :2].mean(axis=0).tolist(),
                "fin_vs_source_endpoint_angle_degrees": oriented[ordinal].tolist()})
    sm = source.doc["materials"][sp["material"]]
    fm = candidate.doc["materials"][fp["material"]]
    require(fm["doubleSided"] is True and fm["pbrMetallicRoughness"]["metallicFactor"] == 0
            and fm["pbrMetallicRoughness"]["roughnessFactor"] == .92, "Material evidence changed")
    require(time.monotonic()-start < 14, "Internal deadline")
    for row in prepared["inputs"]+prepared["sources"]:
        require(sha(row["path"]) == row["sha256"], "Pinned evidence changed during measurement")
    build = json.loads(Path(prepared["build_report"]).read_text())
    result = {"schema_version": 1, "status": "measured_shading_relationships", "quality_approved": False,
        "source_sha256": prepared["source_sha256"], "candidate_sha256": prepared["candidate_sha256"],
        "elapsed_seconds": round(time.monotonic()-start, 6), "inputs_unchanged": True, "fins": len(ids),
        "relationships": {"fin_vs_source_endpoint_angle_degrees": stats(oriented),
            "angle_after_best_global_sign_choice_degrees": stats(sign_independent),
            "source_endpoint_normal_difference_degrees": stats(endpoint_difference),
            "average_source_vs_source_face_geometric_angle_degrees": stats(angle(surface, base_geometric)),
            "fin_vs_average_source_abs_dot": stats(np.abs(np.einsum("nc,nc->n", fin, surface))),
            "fin_vs_actual_triangle_dot": stats(fin_geo_dot),
            "fraction_endpoint_angles_between_60_and_120_degrees": float(((oriented >= 60)&(oriented <= 120)).mean())},
        "luminance_bands": band_rows, "source_blend_vector_diagnostics": hypothetical,
        "double_sided_view_diagnostics": backface_rows,
        "lambert_cosine_diagnostic": {"method": "max(0,N dot direction_to_saved_area_light_center), before any albedo/intensity/distance/environment/occlusion/PBR effects",
            "camera_backface_rule": "Use geometric winding sign against preserved orthographic view direction; glTF double-sided rule reverses back-side shading normals.",
            "scope": "An explanatory cosine calculation, not a rendered brightness prediction or specular model.", "rows": coefficients},
        "materials": {"source": sm, "fin": fm, "source_has_normal_texture": "normalTexture" in sm,
            "fin_has_normal_texture": "normalTexture" in fm,
            "limitation": "The source normal texture is not decoded; these comparisons use serialized vertex normals only."},
        "known_color_defect_not_resolved": build["long_edge_color_interpolation"], "witnesses": witnesses,
        "duplicated_winding_proposal": {"implemented": False, "candidate_vertices_unchanged_if_sharing_accessors": 70567,
            "candidate_triangles_before": 72181, "candidate_triangles_after": 81781,
            "mechanism": "Add reversed index triangles with same positions/UVs/weights/source-outward shading normals; doubleSided=false intends exactly one front winding per view.",
            "risks": ["Coincident opposite surfaces have no physical thickness; culling and shadow behavior must be verified in the target renderer.",
                "Source-outward shading normals intentionally differ from one or both geometric face normals; this requires an explicit authored policy.",
                "The Cycles preview/importer culling behavior is not established by the glTF flag alone.",
                "This does not correct known long-edge color interpolation errors."]},
        "normative_reference": {"url": "https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#double-sided",
            "section": "3.9.5", "verified_by": "Parent session primary-source lookup",
            "paraphrase": "Single-sided materials enable backface culling. Double-sided materials disable it and reverse back-facing shading normals before lighting."},
        "limitations": ["No model, geometry, normal, image, material, action, or production file was changed.",
            "Normal mismatch supports a shading-discontinuity hypothesis; it does not isolate the cause of each visible speckle or prove a blend improves appearance.",
            "No texture decode, native render, browser, provider, AI-role call, collision, or physical fur claim.",
            "Bands use actual stored endpoint colors and are not anatomical labels or exact visible-tip colors."]}
    write("measurement.json", result)
    print(json.dumps({"status": result["status"], "fins": len(ids), "median_angle_degrees": result["relationships"]["fin_vs_source_endpoint_angle_degrees"]["median"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write("failure.json", {"status": "rejected", "type": type(error).__name__, "message": str(error)})
        raise
