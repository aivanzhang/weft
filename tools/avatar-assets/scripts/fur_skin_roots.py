"""Interior roots only where exact named skin weights commute with interpolation."""

import math
from numbers import Real
import random

import numpy as np

if __package__:
    from . import fur_vertex_roots as _vertex_roots
else:
    import fur_vertex_roots as _vertex_roots


POLICY = "exact-skin-domain-roots-v1"
UNIFORM = "uniform-skin-interior"
VARYING = "source-vertex"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _weight_classes(weights, referenced, vertex_count):
    classes = np.full(vertex_count, -1, dtype=np.int32)
    canonical = {}
    zero_entries, maximum_positive = 0, 0
    for index in referenced:
        row = weights[int(index)]
        _require(isinstance(row, dict) and 1 <= len(row) <= 64, "Referenced vertex needs1..64 named skin entries")
        positive = []
        for name, value in row.items():
            _require(isinstance(name, str) and 1 <= len(name) <= 128 and "\0" not in name, "Invalid source bone name")
            _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)), "Skin weights must be finite numbers")
            try:
                value = float(value)
            except (ValueError, OverflowError):
                raise ValueError("Skin weights exceed finite numeric range") from None
            _require(math.isfinite(value) and value >= 0, "Skin weights must be finite and nonnegative")
            if value > 0:
                positive.append((name, value))
            else:
                zero_entries += 1
        _require(1 <= len(positive) <= 4, "Referenced vertex needs1..4 positive skin influences; none may be dropped")
        maximum_positive = max(maximum_positive, len(positive))
        positive.sort()
        try:
            total = math.fsum(value for _, value in positive)
        except OverflowError:
            raise ValueError("Skin weight normalization exceeds finite numeric range") from None
        normalized = tuple((name, value/total) for name, value in positive)
        _require(all(math.isfinite(value) and value > 0 for _, value in normalized),
                 "Positive skin influence underflowed during normalization; none may be dropped")
        classes[int(index)] = canonical.setdefault(normalized, len(canonical))
    return classes, {"canonical_weight_classes": len(canonical), "referenced_vertices": len(referenced),
                     "ignored_zero_weight_entries": zero_entries, "maximum_positive_influences": maximum_positive}


def _allocation(areas, count):
    total = math.fsum(areas)
    expected = [count*(area/total) if total else 0. for area in areas]
    resolved = [math.floor(value) for value in expected]
    remaining = count-sum(resolved)
    present = [index for index, area in enumerate(areas) if area > 0]
    _require(present or count == 0, "Skin roots require eligible nondegenerate geometry")
    _require(0 <= remaining <= len(present), "Invalid skin-domain count allocation")
    order = sorted(present, key=lambda index: (-(expected[index]-resolved[index]), index))
    for index in order[:remaining]:
        resolved[index] += 1
    hamilton = resolved.copy()
    adjustment = None
    if count >= len(present) and len(present) == 2 and 0 in resolved:
        receiver = resolved.index(0); donor = 1-receiver
        resolved[receiver] += 1; resolved[donor] -= 1
        adjustment = {"from": (UNIFORM, VARYING)[donor], "to": (UNIFORM, VARYING)[receiver], "count": 1}
    _require(sum(resolved) == count and min(resolved) >= 0, "Skin-domain counts differ from the requested total")
    return expected, hamilton, resolved, adjustment


def select_skin_roots(vertices, triangles, weights, count, seed):
    """Keep exact count/ownership; insufficient varying-domain capacity is unsupported."""
    _require(type(count) is int and 0 <= count <= _vertex_roots.MAX_ROOTS, "Root count must be an integer in0..5000")
    _require(type(seed) is int and 0 <= seed <= 2**31-1, "Seed must be an integer in0..2147483647")
    _vertex_roots._shape(vertices, "Vertices", _vertex_roots.MAX_VERTICES)
    _vertex_roots._shape(triangles, "Triangles", _vertex_roots.MAX_TRIANGLES, empty=True)
    _require(isinstance(weights, (list, tuple)) and len(weights) == len(vertices), "Weights must match source vertices")
    points = _vertex_roots._triples(vertices, "Vertices")
    faces = _vertex_roots._triples(triangles, "Triangles", integer=True)
    _require(np.isfinite(points).all(), "Vertices must be finite")
    _require(not len(faces) or faces.min() >= 0 and faces.max() < len(points), "Triangle indices exceed source vertices")
    points = points.astype(np.float64, copy=False)
    faces = faces.astype(np.int64, copy=False)
    _require(np.isfinite(points).all(), "Vertices exceed finite float64 coordinates")
    classes, skin_summary = _weight_classes(weights, np.unique(faces), len(points))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        cross = np.cross(points[faces[:, 1]]-points[faces[:, 0]], points[faces[:, 2]]-points[faces[:, 0]])
        face_areas = np.hypot(np.hypot(cross[:, 0], cross[:, 1]), cross[:, 2])*.5
    _require(np.isfinite(face_areas).all(), "Triangle areas exceed finite numeric range")
    eligible = face_areas > _vertex_roots.MIN_AREA
    same = (classes[faces[:, 0]] == classes[faces[:, 1]]) & (classes[faces[:, 0]] == classes[faces[:, 2]])
    uniform = np.flatnonzero(eligible & same)
    varying = np.flatnonzero(eligible & ~same)
    try:
        areas = [math.fsum(float(face_areas[index]) for index in domain) for domain in (uniform, varying)]
        expected, hamilton, resolved, adjustment = _allocation(areas, count)
    except OverflowError:
        raise ValueError("Eligible skin-domain area exceeds finite numeric range") from None
    capacity = len(np.unique(faces[varying]))
    _require(resolved[1] <= capacity, "Varying-skin domain requires " + str(resolved[1])
             + " roots but has " + str(capacity) + " unique source vertices; refusing count reduction or reallocation")
    output = []
    if resolved[1]:
        selected = _vertex_roots.select_vertex_roots(points, faces[varying], resolved[1], seed)
        for row in selected["roots"]:
            output.append({**row, "triangle_index": int(varying[row["triangle_index"]]), "binding_kind": VARYING})
    if resolved[0]:
        cumulative = np.cumsum(face_areas[uniform]/areas[0]); cumulative[-1] = 1.
        rng = random.Random(seed)
        for ordinal in range(resolved[0]):
            position = (ordinal+rng.random())/resolved[0]
            index = min(int(np.searchsorted(cumulative, position, side="right")), len(uniform)-1)
            # Open draws keep the declared interior binding off edges and vertices.
            radius = math.sqrt(max(2**-53, rng.random()))
            fraction = max(2**-53, rng.random())
            bary = [1-radius, radius*(1-fraction), radius*fraction]
            _require(all(math.isfinite(value) and value > 0 for value in bary), "Interior root is numerically degenerate")
            output.append({"triangle_index": int(uniform[index]), "barycentric": bary, "binding_kind": UNIFORM})
    output.sort(key=lambda row: row["triangle_index"])
    domains = [{"binding_kind": name, "eligible_triangles": len(domain), "eligible_area": area,
                "expected_count": quota, "hamilton_count": original, "resolved_count": selected}
               for name, domain, area, quota, original, selected in zip(
                   (UNIFORM, VARYING), (uniform, varying), areas, expected, hamilton, resolved)]
    domains[1]["unique_vertex_capacity"] = capacity
    _require(len(output) == count, "Skin roots differ from the requested count")
    return {"roots": output, "summary": {
        "policy": POLICY, "seed": seed, "requested_roots": count, "selected_roots": len(output),
        "source_vertices": len(points), "source_triangles": len(faces),
        "eligible_triangles": int(eligible.sum()), "excluded_degenerate_triangles": int((~eligible).sum()),
        "minimum_triangle_area_exclusive": _vertex_roots.MIN_AREA, "domains": domains,
        "minimum_domain_adjustment": adjustment, **skin_summary,
        "normalization_policy": "sort positive named influences; divide by math.fsum; exact tuple equality, no cutoff/tolerance",
        "allocation_policy": "Hamilton largest remainder; uniform wins ties; when possible give each nonempty domain one root",
        "uniform_sampling_policy": "one seeded draw per cumulative-area stratum; open barycentric draws",
        "varying_sampling_policy": _vertex_roots.POLICY,
        "ordering_policy": "original triangle index; stable draw/corner order within each face",
        "source_attributes_modified": False, "count_reduction": False, "capacity_reallocation": False,
    }}
