"""Area-weighted roots on exact source vertex IDs, without replacement."""

import heapq
import math
from numbers import Integral, Real
import random

import numpy as np


POLICY = "area-weighted-source-vertices-v1"
MAX_VERTICES = 100_000
MAX_TRIANGLES = 100_000
MAX_ROOTS = 5000
MIN_AREA = 1e-14


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _shape(value, label, maximum, *, empty=False):
    _require(isinstance(value, (list, tuple, np.ndarray)), label + " requires bounded triples")
    if isinstance(value, np.ndarray):
        _require(value.ndim == 2 and value.shape[1:] == (3,)
                 and (0 if empty else 1) <= len(value) <= maximum,
                 label + " shape/count exceeds bounds")
    else:
        _require((0 if empty else 1) <= len(value) <= maximum, label + " count exceeds bounds")


def _triples(value, label, *, integer=False):
    if isinstance(value, np.ndarray):
        _require(value.dtype.kind in ("iu" if integer else "fiu"), label + " has invalid numeric types")
        return value
    kind = Integral if integer else Real
    for row in value:
        _require(isinstance(row, (list, tuple, np.ndarray))
                 and (not isinstance(row, np.ndarray) or row.ndim == 1) and len(row) == 3,
                 label + " requires triples")
        _require(all(isinstance(item, kind) and not isinstance(item, (bool, np.bool_)) for item in row),
                 label + " has invalid numeric types")
    try:
        result = np.asarray(value).reshape((-1, 3))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(label + " cannot be represented as bounded numeric triples") from None
    _require(not len(result) or result.dtype.kind in ("iu" if integer else "fiu"),
             label + " has invalid numeric types")
    return result


def select_vertex_roots(vertices, triangles, count, seed):
    """Return one-hot roots in input triangle/corner order; no source attributes change.

    Every eligible face contributes area/3 to each source vertex. Log-domain
    exponential keys preserve the weighted ordering without overflow in division.
    """
    _require(type(count) is int and 0 <= count <= MAX_ROOTS, "Root count must be an integer in0..5000")
    _require(type(seed) is int and 0 <= seed <= 2**31-1, "Seed must be an integer in0..2147483647")
    _shape(vertices, "Vertices", MAX_VERTICES)
    _shape(triangles, "Triangles", MAX_TRIANGLES, empty=True)
    points = _triples(vertices, "Vertices")
    faces = _triples(triangles, "Triangles", integer=True)
    _require(np.isfinite(points).all(), "Vertices must be finite")
    _require(not len(faces) or faces.min() >= 0 and faces.max() < len(points), "Triangle indices exceed source vertices")
    # Validate both input shapes before allocating geometry-sized working arrays.
    points = points.astype(np.float64, copy=False)
    faces = faces.astype(np.int64, copy=False)
    _require(np.isfinite(points).all(), "Vertices exceed finite float64 coordinates")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        cross = np.cross(points[faces[:, 1]]-points[faces[:, 0]], points[faces[:, 2]]-points[faces[:, 0]])
        areas = np.hypot(np.hypot(cross[:, 0], cross[:, 1]), cross[:, 2])*.5
    _require(np.isfinite(areas).all(), "Triangle areas exceed finite numeric range")
    usable = np.flatnonzero(areas > MIN_AREA)
    weights = np.bincount(faces[usable].reshape(-1), weights=np.repeat(areas[usable]/3, 3), minlength=len(points))
    _require(np.isfinite(weights).all(), "Incident vertex areas exceed finite numeric range")
    eligible = np.flatnonzero(weights > 0)
    _require(count <= len(eligible), "Requested roots exceed unique eligible source vertices: " + str(len(eligible)))
    first = np.full(len(points), 3*len(faces), dtype=np.int64)
    for corner in range(3):
        np.minimum.at(first, faces[usable, corner], usable*3+corner)
    rng = random.Random(seed)

    def keys():
        for index in eligible:
            draw = rng.random()
            key = math.log(-math.log1p(-draw))-math.log(float(weights[index])) if draw else -math.inf
            yield key, int(index)

    selected = sorted((index for _, index in heapq.nsmallest(count, keys())), key=lambda index: int(first[index]))
    roots = []
    for index in selected:
        triangle, corner = divmod(int(first[index]), 3)
        bary = [0., 0., 0.]; bary[corner] = 1.
        roots.append({"triangle_index": triangle, "corner": corner, "vertex_index": index, "barycentric": bary})
    _, duplicates = np.unique(points[eligible], axis=0, return_counts=True)
    try:
        total_area = math.fsum(float(areas[index]) for index in usable)
    except OverflowError:
        raise ValueError("Total eligible area exceeds finite numeric range") from None
    return {"roots": roots, "summary": {
        "policy": POLICY, "seed": seed, "requested_roots": count, "selected_roots": len(roots),
        "source_vertices": len(points), "source_triangles": len(faces),
        "eligible_triangles": len(usable), "excluded_degenerate_triangles": len(faces)-len(usable),
        "eligible_source_vertices": len(eligible), "total_eligible_area": total_area,
        "minimum_triangle_area_exclusive": MIN_AREA,
        "weight_policy": "sum of incident eligible triangle area/3 per source vertex ID",
        "sampling_policy": "seeded exponential keys without replacement; exact key ties use source vertex ID",
        "representative_policy": "first eligible input triangle/corner; returned roots sorted by that pair",
        "coincident_position_groups": int(np.count_nonzero(duplicates > 1)),
        "coincident_eligible_vertices": int(duplicates[duplicates > 1].sum()),
        "coincident_extra_source_ids": int(len(eligible)-len(duplicates)),
        "coincidence_policy": "exact numeric position equality only; distinct source IDs are never merged",
        "source_attributes_modified": False,
    }}
