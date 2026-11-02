"""Read-only area/root allocation census; counts are not visible fur coverage."""
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from glb_read import GLB, require
from color_sampling import repeat_root_colors

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(name, value):
    with (HERE/name).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def load_npz(path, names):
    with zipfile.ZipFile(path) as archive:
        require(len(archive.infolist()) <= 8 and sum(i.file_size for i in archive.infolist()) <= 4*1024**2,
                "NPZ expansion bound")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in names}
    require(all(a.dtype.kind in "iuf" and np.isfinite(a).all() for a in arrays.values()), "Invalid correspondence array")
    return arrays


def weighted_quantiles(values, weights):
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    total = weights.sum()
    require(total > 0, "No positive area")
    cumulative = np.cumsum(weights)/total
    return dict(zip(("p10", "median", "p90"), np.interp([.1, .5, .9], cumulative, values).tolist()))


def main():
    start = time.monotonic()
    prepared = json.loads((HERE/"prepared.json").read_text())
    settings = prepared["settings"]
    for row in prepared["inputs"]+prepared["sources"]:
        require(sha(row["path"]) == row["sha256"], "Pinned input changed")
    source = GLB(prepared["source"])
    candidate = GLB(prepared["candidate"])
    census = json.loads(Path(prepared["source_census"]).read_text())
    build = json.loads(Path(prepared["build_report"]).read_text())
    require(census["model_sha256"] == prepared["source_sha256"] and census["body_node"] == 19 and census["mesh"] == 0,
            "Source census binding mismatch")
    require(np.array_equal(census["nodes"][19]["world_matrix_column_major"], np.eye(4).reshape(-1)), "Nonidentity emitter")
    primitive = source.doc["meshes"][0]["primitives"][5]
    material = source.doc["materials"][primitive["material"]]
    require(material["name"] == "body_surface__short_pile_adaptive_v1"
            and material["name"] in census["selection"]["eligible_materials"], "Not an eligible recorded fur material")
    attrs = primitive["attributes"]
    positions = source.array(attrs["POSITION"], "VEC3", (5126,))
    uv = source.array(attrs["TEXCOORD_0"], "VEC2", (5126,))
    faces = source.array(primitive["indices"], "SCALAR", (5121, 5123, 5125), 300000).reshape(-1, 3)
    require(len(positions) <= 100000 and len(faces) <= 100000 and (faces < len(positions)).all(), "Source geometry bound")
    safe = load_npz(prepared["safe_edges"], ["primitive_5_edges", "primitive_5_faces"])
    edges, adjacent = safe["primitive_5_edges"], safe["primitive_5_faces"]
    require(edges.shape == adjacent.shape and edges.ndim == 2 and edges.shape[1] == 2 and len(edges) <= 150000,
            "Safe edge shape bound")
    require((edges >= 0).all() and (edges < len(positions)).all() and (adjacent >= 0).all() and (adjacent < len(faces)).all(),
            "Safe edge index bound")
    roots = load_npz(prepared["correspondence"], ["root_vertex_ids", "root_face_ids", "root_source_uv", "root_centers"])
    root_ids = roots["root_vertex_ids"]
    require(root_ids.ndim == 2 and root_ids.shape[1] == 2 and 1 <= len(root_ids) <= 4800
            and (root_ids >= 0).all() and (root_ids < len(positions)).all(), "Old root bound")
    require(roots["root_source_uv"].tobytes() == uv[root_ids].tobytes()
            and roots["root_centers"].tobytes() == positions[root_ids].mean(axis=1).tobytes(), "Old source correspondence mismatch")
    nv = len(positions)
    codes = edges[:, 0].astype(np.int64)*nv+edges[:, 1]
    order = np.argsort(codes)
    codes = codes[order]
    root_codes = root_ids[:, 0].astype(np.int64)*nv+root_ids[:, 1]
    root_lookup = np.searchsorted(codes, root_codes)
    require((root_lookup < len(codes)).all() and np.array_equal(codes[root_lookup], root_codes), "Old root outside safe whitelist")
    root_safe_ids = order[root_lookup]
    require(len(np.unique(root_safe_ids)) == len(root_safe_ids), "Duplicate old roots")
    # Reconstruct the existing face buffer so edge-length and later selection effects stay separate.
    all_edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    all_codes = all_edges[:, 0].astype(np.int64)*nv+all_edges[:, 1]
    unique, inverse, occurrences = np.unique(all_codes, return_inverse=True, return_counts=True)
    unsafe_occurrence = (occurrences[inverse] == 2)&~np.isin(all_codes, codes)
    buffered_faces = unsafe_occurrence.reshape(-1, 3).any(axis=1)
    after_buffer = ~buffered_faces[adjacent].any(axis=1)
    require(after_buffer[root_safe_ids].all(), "Old fin uses a buffered face")
    require(int(buffered_faces.sum()) == build["source_selection"]["conservatively_excluded_faces_near_unsafe_shared_edges"],
            "Existing face-buffer receipt mismatch")
    xyz = positions.astype(np.float64)[faces]
    areas = np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=1)/2
    require((areas > 0).all(), "Degenerate source area")
    edge_vectors = positions[edges[:, 1]]-positions[edges[:, 0]]
    lengths = np.linalg.norm(edge_vectors, axis=1)
    edge_midpoints = positions[edges].astype(np.float64).mean(axis=1)
    low_gltf, high_gltf = (np.asarray(census["world_bounds"][key], dtype=float) for key in ("min", "max"))
    low = np.array([low_gltf[0], -high_gltf[2], low_gltf[1]])
    high = np.array([high_gltf[0], -low_gltf[2], high_gltf[1]])
    span = high-low
    require((span > 0).all(), "Degenerate bounds")
    grid_size = settings["grid_axis_cells"]

    def cells(points):
        native = points[:, [0, 2, 1]]*np.array([1, -1, 1])
        normalized = (native-low)/span
        require((normalized >= -1e-6).all() and (normalized <= 1+1e-6).all(), "Point outside original body bounds")
        ijk = np.clip(np.floor(normalized*grid_size).astype(int), 0, grid_size-1)
        return (ijk[:, 0]*grid_size+ijk[:, 1])*grid_size+ijk[:, 2]

    pbr = material["pbrMetallicRoughness"]
    texture_info = pbr["baseColorTexture"]
    require(texture_info.get("texCoord", 0) == 0 and not texture_info.get("extensions"), "Unsupported source UV routing")
    texture = source.doc["textures"][texture_info["index"]]
    sampler = source.doc["samplers"][texture["sampler"]]
    require(sampler.get("wrapS", 10497) == 10497 and sampler.get("wrapT", 10497) == 10497
            and sampler.get("magFilter", 9729) == 9729, "Source filtering differs from candidate convention")
    image_spec = source.doc["images"][texture["source"]]
    image_bytes = source.view_bytes(image_spec["bufferView"])
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    require(image_hash == build["source_selection"]["exact_root_texture_sha256"], "Wrong source root-color image")
    Image.MAX_IMAGE_PIXELS = 16777216
    with Image.open(io.BytesIO(image_bytes)) as image:
        require(image.width <= 4096 and image.height <= 4096 and image.width*image.height <= 16777216
                and image.format in ("JPEG", "PNG"), "Texture decode bound")
        pixels = np.array(image.convert("RGB"), dtype=np.uint8)
    factor = np.asarray(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3], dtype=np.float64)
    require(factor.shape == (3,) and np.isfinite(factor).all() and (factor >= 0).all() and (factor <= 1).all(), "Invalid color factor")

    def sample(uvs):
        require(uvs.ndim == 2 and uvs.shape[1] == 2 and len(uvs) <= 300000 and np.isfinite(uvs).all(), "Sample bound")
        result = np.empty((len(uvs), 3), dtype=np.float64)
        for begin in range(0, len(uvs), settings["sampling_batch"]):
            end = begin+settings["sampling_batch"]
            result[begin:end] = repeat_root_colors(pixels, uvs[begin:end], factor)
        return result

    def bands(rgb):
        luminance = rgb@np.array([.2126, .7152, .0722])
        require(np.isfinite(luminance).all() and (luminance >= 0).all() and (luminance <= 1+1e-12).all(), "Luminance outside bounds")
        return np.searchsorted(settings["linear_luminance_breaks"], luminance, side="right")

    edge_rgb = sample(uv[edges].reshape(-1, 2)).reshape(-1, 2, 3).mean(axis=1)
    edge_bands = bands(edge_rgb)
    edge_cells = cells(edge_midpoints)
    # Compare the actual serialized old vertex colors, not only the correspondence claims.
    old_primitive = candidate.doc["meshes"][0]["primitives"][-1]
    old_colors = candidate.array(old_primitive["attributes"]["COLOR_0"], "VEC3", (5126,)).reshape(-1, 4, 3)
    root_rgb = sample(uv[root_ids].reshape(-1, 2)).reshape(-1, 2, 3)
    require(old_colors.shape == (len(root_ids), 4, 3)
            and old_colors[:, :2].tobytes() == root_rgb.astype("<f4").tobytes(), "Actual old root colors differ from exact candidate filtering")
    root_bands = bands(root_rgb.mean(axis=1))
    root_cells = cells(positions[root_ids].astype(np.float64).mean(axis=1))
    quadrature = np.asarray(settings["area_barycentric_samples"], dtype=np.float64)
    require(quadrature.shape == (3, 3) and np.allclose(quadrature.sum(axis=1), 1), "Invalid area quadrature")
    active_faces = np.flatnonzero(~buffered_faces)
    surface_points = np.einsum("qc,fck->fqk", quadrature, xyz[active_faces]).reshape(-1, 3)
    surface_uvs = np.einsum("qc,fck->fqk", quadrature, uv[faces[active_faces]].astype(float)).reshape(-1, 2)
    sample_area = np.repeat(areas[active_faces]/3, 3)
    surface_cells, surface_bands = cells(surface_points), bands(sample(surface_uvs))
    grid_area = np.bincount(surface_cells, weights=sample_area, minlength=grid_size**3)
    band_area = np.bincount(surface_bands, weights=sample_area, minlength=3)
    root_grid = np.bincount(root_cells, minlength=grid_size**3)
    root_band = np.bincount(root_bands, minlength=3)
    total_area = float(sample_area.sum())
    require(abs(grid_area.sum()-total_area) < 1e-10 and abs(band_area.sum()-total_area) < 1e-10, "Area allocation lost samples")
    range_reports, grid_ranges, band_ranges = {}, {}, {}
    for label, minimum, maximum in settings["ranges_m"]:
        length_ok = (lengths >= minimum)&(lengths <= maximum)
        available = length_ok&after_buffer
        count = int(available.sum())
        require(count > 0, "No available edges in configured range")
        grid_counts = np.bincount(edge_cells[available], minlength=grid_size**3)
        band_counts = np.bincount(edge_bands[available], minlength=3)
        incident_faces = np.unique(adjacent[available])
        grid_ranges[label], band_ranges[label] = grid_counts, band_counts
        range_reports[label] = {"minimum_m_inclusive": minimum, "maximum_m_inclusive": maximum,
            "safe_whitelist_edges": int(length_ok.sum()), "after_existing_face_buffer_edges": count,
            "after_buffer_edge_length_m": dict(zip(("min", "median", "max"), np.quantile(lengths[available], [0, .5, 1]).tolist())),
            "incident_eligible_faces": len(incident_faces), "incident_eligible_face_area_m2": float(areas[incident_faces].sum()),
            "incident_face_area_fraction": float(areas[incident_faces].sum()/total_area),
            "grid_distribution_total_variation_vs_area": float(np.abs(grid_counts/count-grid_area/total_area).sum()/2),
            "luminance_distribution_total_variation_vs_area": float(np.abs(band_counts/count-band_area/total_area).sum()/2)}
    old_range = range_reports["8.5_to_10_mm"]
    require(old_range["after_existing_face_buffer_edges"] == build["fin_report"]["eligible_edges"], "Old eligible-edge count does not reproduce")
    old_grid_share = root_grid/len(root_ids)
    valid_grid = grid_area > 0
    relative_density = root_grid[valid_grid]/grid_area[valid_grid]/(len(root_ids)/total_area)
    substantial = grid_area >= total_area*settings["substantial_cell_area_fraction"]
    grid_rows = []
    for cell in np.flatnonzero((grid_area > 0)|(root_grid > 0)):
        ijk = [int(cell//grid_size**2), int((cell//grid_size)%grid_size), int(cell%grid_size)]
        area = float(grid_area[cell])
        grid_rows.append({"cell": ijk, "eligible_area_m2": area, "area_fraction": area/total_area,
            "old_fin_roots": int(root_grid[cell]), "old_roots_per_m2": float(root_grid[cell]/area) if area else None,
            "old_relative_density": float(root_grid[cell]/area/(len(root_ids)/total_area)) if area else None,
            "available_edges_after_face_buffer": {label: int(counts[cell]) for label, counts in grid_ranges.items()}})
    band_rows = []
    for band, name in enumerate(settings["band_names"]):
        area = float(band_area[band])
        band_rows.append({"band": name, "eligible_area_m2": area, "area_fraction": area/total_area,
            "old_fin_roots": int(root_band[band]), "old_root_fraction": float(root_band[band]/len(root_ids)),
            "old_roots_per_m2": float(root_band[band]/area) if area else None,
            "old_relative_density": float(root_band[band]/area/(len(root_ids)/total_area)) if area else None,
            "available_edges_after_face_buffer": {label: int(counts[band]) for label, counts in band_ranges.items()},
            "available_edges_per_m2": {label: float(counts[band]/area) if area else None for label, counts in band_ranges.items()}})
    require(time.monotonic()-start < 28, "Internal allocation deadline")
    for row in prepared["inputs"]+prepared["sources"]:
        require(sha(row["path"]) == row["sha256"], "Pinned input changed during census")
    result = {"schema_version": 1, "status": "measured_source_allocation", "quality_approved": False,
        "source_sha256": prepared["source_sha256"], "candidate_sha256": prepared["candidate_sha256"], "inputs_unchanged": True,
        "elapsed_seconds": round(time.monotonic()-start, 6), "settings": settings,
        "selection": {"source_primitive": 5, "source_material": material["name"], "safe_whitelist_edges": len(edges),
            "source_faces": len(faces), "source_surface_area_m2": float(areas.sum()),
            "existing_unsafe_edge_buffer_faces": int(buffered_faces.sum()), "buffered_area_m2": float(areas[buffered_faces].sum()),
            "eligible_faces_after_buffer": len(active_faces), "eligible_surface_area_m2": total_area,
            "recorded_cap_target_materials_excluded": census["selection"]["normal_continuity_target_source_names_excluded"]},
        "sampling": {"texture_sha256": image_hash, "texture_size": [pixels.shape[1], pixels.shape[0]], "source_sampler": sampler,
            "color_convention": "glTF normalized texel centers, REPEAT, linear LOD0, per-texel sRGB decode before bilinear filtering, then material RGB factor",
            "source_color_function": "AST-extracted unchanged repeat_root_colors from pinned candidate v2 source",
            "surface_area_quadrature": "Three equal-area barycentric samples per eligible triangle; cell/color bin areas are quadrature estimates, total area is exact triangle-area sum.",
            "edge_color": "Mean of two exact linear endpoint colors, then Rec.709 linear luminance; same convention for old and available roots.",
            "actual_serialized_old_root_colors_exact": True, "image_decodes": 1, "texture_sample_points": 2*len(edges)+2*len(root_ids)+len(surface_uvs),
            "edge_length_filter": "Float32 point subtraction/norm, matching the frozen candidate builder; inclusive endpoints.",
            "grid_coordinate_space": "Original whole-body normalized bounds in Blender right-handed Z-up; glTF point [x,y,z] maps to [x,-z,y]",
            "original_bounds_blender": {"min": low.tolist(), "max": high.tolist()}},
        "ranges": range_reports, "luminance_bands": band_rows, "grid_cells": grid_rows,
        "old_allocation_summary": {"fin_roots": len(root_ids), "global_roots_per_m2": len(root_ids)/total_area,
            "grid_distribution_total_variation_vs_area": float(np.abs(old_grid_share-grid_area/total_area).sum()/2),
            "luminance_distribution_total_variation_vs_area": float(np.abs(root_band/len(root_ids)-band_area/total_area).sum()/2),
            "grid_total_variation_vs_old_range_availability": float(np.abs(old_grid_share-grid_ranges["8.5_to_10_mm"]/old_range["after_existing_face_buffer_edges"]).sum()/2),
            "area_weighted_relative_density_quantiles": weighted_quantiles(relative_density, grid_area[valid_grid]),
            "substantial_cells": int(substantial.sum()), "substantial_cells_with_zero_roots": int((substantial&(root_grid == 0)).sum()),
            "area_fraction_in_zero_root_cells": float(grid_area[root_grid == 0].sum()/total_area)},
        "limitations": ["Root counts and triangle-incidence opportunity are not visible fur area, screen-space coverage, softness, or appearance approval.",
            "No new fin selection, spacing, geometry, render, or motion was generated; wider ranges report available source edges only.",
            "Cell/color quadrature can misassign triangles crossing grid/texture-band boundaries; no anatomy labels or subject-specific coordinates are used.",
            "Texture bands describe stored basecolor, not lighting or perceived rendered brightness; LOD/mip/GPU behavior is not measured.",
            "No distance/nearest-neighbor scan is needed for this allocation comparison; existing root spacing and direction scoring remain separate selection effects."],
        "limits": {"wall_seconds": 30, "rss_mb": 512, "output_mb": 16, "attempts": 1}, "native_calls": 0, "provider_calls": 0}
    write("allocation.json", result)
    print(json.dumps({"status": result["status"], "old_fin_roots": len(root_ids), "ranges": {k:v["after_existing_face_buffer_edges"] for k,v in range_reports.items()}}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write("failure.json", {"status": "rejected", "type": type(error).__name__, "message": str(error)})
        raise
