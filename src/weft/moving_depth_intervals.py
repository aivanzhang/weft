"""Analytic depth enclosures for an authored sphere ray, never a recovered initial state."""

from fractions import Fraction
from hashlib import sha256
import json
import math
import time

PROFILE = "moving-depth-intervals-v1"
MAX_PRIMITIVES = 64
MAX_INTERVALS = 512
ROOT_BITS = 192
MAX_BITS = 4096


class _Unsupported(Exception):
    pass


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Analytic depth intervals exceeded their deadline")


def _number(value):
    if type(value) not in (int, float) or abs(value) > 1e6 or not math.isfinite(value):
        raise ValueError("Depth geometry requires finite bounded numbers")
    result = Fraction(value)
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > MAX_BITS:
        raise _Unsupported("input_numeric_precision")
    return result


def _ray_shape(ray):
    for name, length in (("domain", 2), ("radius_bounds", 2), ("center_intercept", 3), ("center_slope", 3)):
        if not isinstance(ray[name], list) or len(ray[name]) != length:
            raise ValueError("Depth ray requires bounded fixed-length vectors")
        for value in ray[name]:
            _number(value)
    _number(ray["radius_intercept"])
    _number(ray["radius_slope"])


def _vector(value):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("Depth geometry requires three-dimensional vectors")
    return tuple(map(_number, value))


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


def _bounded(value):
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_BITS:
        raise _Unsupported("coefficient_numeric_precision")
    return value


def _sqrt_bounds(value, bits=ROOT_BITS):
    if value < 0 or bits > MAX_BITS:
        raise _Unsupported("root_numeric_precision")
    n, d = math.isqrt(value.numerator), math.isqrt(value.denominator)
    if n * n == value.numerator and d * d == value.denominator:
        return Fraction(n, d), Fraction(n, d)
    scale = 1 << bits
    low = math.isqrt((value.numerator << (2 * bits)) // value.denominator)
    return Fraction(low, scale), Fraction(low + 1, scale)


def _union(intervals):
    output = []
    for low, high in sorted((a, b) for a, b in intervals if a <= b):
        if output and low <= output[-1][1]:
            output[-1] = (output[-1][0], max(high, output[-1][1]))
        else:
            output.append((low, high))
    if len(output) > MAX_INTERVALS:
        raise _Unsupported("interval_count")
    return output


def _intersect(left, right):
    output, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a, b = left[i]; c, d = right[j]
        if max(a, c) <= min(b, d):
            output.append((max(a, c), min(b, d)))
        if b < d:
            i += 1
        else:
            j += 1
    return _union(output)


def _positive(a, b, c, domain):
    """Outer and inner closed sets for a*d²+b*d+c >= 0, with exact coefficients."""
    a, b, c = map(_bounded, (a, b, c))
    low, high = domain
    full = [(low, high)]
    if not a:
        if not b:
            return (full, full) if c >= 0 else ([], [])
        root = -c / b
        span = [(max(low, root), high)] if b > 0 else [(low, min(high, root))]
        result = _union(span)
        return result, result
    discriminant = _bounded(b * b - 4 * a * c)
    if discriminant < 0:
        return (full, full) if a > 0 else ([], [])
    if not discriminant:
        if a > 0:
            return full, full
        root = -b / (2 * a)
        result = [(root, root)] if low <= root <= high else []
        return result, result
    # Scale root precision when a small leading coefficient amplifies sqrt error.
    bits = ROOT_BITS + max(0, a.denominator.bit_length() - abs(a.numerator).bit_length())
    sl, sh = _sqrt_bounds(discriminant, bits)
    denominator = 2 * a
    roots = sorted((tuple(sorted(((-b - sl) / denominator, (-b - sh) / denominator))),
                    tuple(sorted(((-b + sl) / denominator, (-b + sh) / denominator)))))
    first, second = roots
    if a > 0:
        outer = [(low, min(high, first[1])), (max(low, second[0]), high)]
        inner = [(low, min(high, first[0])), (max(low, second[1]), high)]
    else:
        outer = [(max(low, first[0]), min(high, second[1]))]
        inner = [(max(low, first[1]), min(high, second[0]))]
    return _union(outer), _union(inner)


def _distance_polynomial(p0, p1, r0, r1):
    return _dot(p1, p1) - r1 * r1, 2 * (_dot(p0, p1) - r0 * r1), _dot(p0, p0) - r0 * r0


def _sphere(p0, p1, r0, r1, primitive, domain):
    center, radius = _vector(primitive["center"]), _number(primitive["radius"])
    if radius < 0:
        raise ValueError("Fixed geometry radius cannot be negative")
    return (*_positive(*_distance_polynomial(tuple(x - y for x, y in zip(p0, center)), p1, r0 + radius, r1), domain), 1)


def _capsule(p0, p1, r0, r1, primitive, domain):
    endpoints = primitive["endpoints"]
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise ValueError("Capsule requires two endpoints")
    start, end = map(_vector, endpoints)
    edge = tuple(b - a for a, b in zip(start, end))
    length2 = _dot(edge, edge)
    if not length2:
        return _sphere(p0, p1, r0, r1, {"center": endpoints[0], "radius": primitive["radius"]}, domain)
    radius = _number(primitive["radius"])
    if radius < 0:
        raise ValueError("Fixed geometry radius cannot be negative")
    delta = tuple(a - b for a, b in zip(p0, start))
    t0, t1 = _dot(delta, edge) / length2, _dot(p1, edge) / length2
    cuts = {domain[0], domain[1]}
    if t1:
        cuts.update(value for value in (-t0 / t1, (1 - t0) / t1) if domain[0] < value < domain[1])
    cuts = sorted(cuts)
    outer, inner = [], []
    pieces = list(zip(cuts, cuts[1:])) or [domain]
    for low, high in pieces:
        projection = t0 + t1 * ((low + high) / 2)
        # The midpoint chooses an exact affine branch, not a sampled feasibility test.
        if projection <= 0:
            q0, q1 = delta, p1
        elif projection >= 1:
            q0, q1 = tuple(a - b for a, b in zip(p0, end)), p1
        else:
            q0 = tuple(a - t0 * b for a, b in zip(delta, edge))
            q1 = tuple(a - t1 * b for a, b in zip(p1, edge))
        possible, certified = _positive(*_distance_polynomial(q0, q1, r0 + radius, r1), (low, high))
        outer.extend(possible); inner.extend(certified)
    return _union(outer), _union(inner), len(pieces)


def _plane(p0, p1, r0, r1, plane, domain):
    normal, offset = _vector(plane["normal"]), _number(plane["offset"])
    norm2 = _dot(normal, normal)
    if not norm2:
        raise ValueError("Plane normal cannot be zero")
    nl, nh = _sqrt_bounds(norm2)
    base, slope = _dot(normal, p0) - offset, _dot(normal, p1)
    # Positive radii make the larger normal norm the conservative inner constraint.
    possible, _ = _positive(Fraction(0), slope - r1 * nl, base - r0 * nl, domain)
    _, certified = _positive(Fraction(0), slope - r1 * nh, base - r0 * nh, domain)
    return possible, certified, 2


def _rounded(value, upward):
    result = float(value)
    if not math.isfinite(result):
        raise _Unsupported("output_numeric_range")
    if (upward and Fraction(result) < value) or (not upward and Fraction(result) > value):
        result = math.nextafter(result, math.inf if upward else -math.inf)
    return result


def _output(intervals, inner):
    spans = [(_rounded(a, inner), _rounded(b, not inner)) for a, b in intervals]
    return [list(x) for x in _union(spans)]


def _digest(value):
    raw = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    if len(raw) > 1 << 20:
        raise ValueError("Depth geometry exceeds the metadata bound")
    return sha256(raw).hexdigest()


def solve_depth_intervals(ray, geometry, *, deadline=None):
    """Return all possible depth components and certified interiors; never choose a depth."""
    from weft.moving_depth_geometry import validate_depth_geometry

    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("Depth interval deadline must be finite")
    stop = min(time.monotonic() + 5, deadline) if deadline is not None else time.monotonic() + 5
    _check(stop)
    if not isinstance(ray, dict) or set(ray) != {"domain", "center_intercept", "center_slope", "radius_intercept", "radius_slope", "radius_bounds"}:
        raise ValueError("Unsupported affine depth ray")
    _ray_shape(ray)
    if not isinstance(geometry, dict) or geometry.get("profile") != "moving-depth-geometry-v1":
        raise ValueError("Unsupported fixed depth geometry profile")
    geometry = validate_depth_geometry(geometry)
    _check(stop)
    result = {"schema_version": 1, "profile": PROFILE, "ray_sha256": _digest(ray), "geometry_sha256": _digest(geometry),
        "status": "unsupported", "possible_intervals": [], "certified_intervals": [], "boundary_ambiguity": False,
        "primitive_count": 0, "polynomial_pieces": 0, "reasons": [], "selected_depth": None,
        "scope": "Conditional no-penetration depth prior; all source identity/prefix uncertainty remains. Not source reconstruction, multi-object admission or native geometry qualification.",
        "limits": {"seconds": 5, "primitives": MAX_PRIMITIVES, "intervals": MAX_INTERVALS,
                   "root_fractional_bits": ROOT_BITS, "coefficient_bits": MAX_BITS, "max_root_fractional_bits": MAX_BITS},
        "numeric_contract": "Exact rational input-float coefficients; complete outer enclosures and conservative inner intervals. Uncertain boundaries and nonrepresentable tangent points are never silently certified."}
    try:
        for name in ("domain", "radius_bounds"):
            if not isinstance(ray[name], list) or len(ray[name]) != 2:
                raise ValueError("Depth and radius bounds require two endpoints")
        domain = tuple(map(_number, ray["domain"]))
        minimum, maximum = map(_number, ray["radius_bounds"])
        if not 0 < domain[0] <= domain[1] <= 100 or not 0 < minimum <= maximum <= 100:
            raise ValueError("Depth and radius bounds must be positive and ordered")
        p0, p1 = _vector(ray["center_intercept"]), _vector(ray["center_slope"])
        r0, r1 = _number(ray["radius_intercept"]), _number(ray["radius_slope"])
        gates = geometry.get("geometric_gates", {})
        if (gates.get("domain_clearance_world") != 0 or type(gates.get("domain_clearance_world")) not in (int, float)
                or gates.get("mask_filtering_applied") is not False):
            raise ValueError("Depth proposal requires explicit zero-clearance, unfiltered geometry")
        primitives = geometry.get("primitives")
        if not isinstance(primitives, list) or not 1 <= len(primitives) <= MAX_PRIMITIVES:
            raise ValueError("Depth geometry needs a bounded complete primitive inventory")
        result["primitive_count"] = len(primitives)
        floor, world = geometry.get("floor"), geometry.get("world_bounds")
        if not isinstance(floor, dict) or floor.get("type") != "plane" or not isinstance(world, dict):
            raise ValueError("Depth geometry needs its original floor and world bounds")
        low, high = _vector(world["low"]), _vector(world["high"])
        if any(a >= b for a, b in zip(low, high)):
            raise ValueError("World bounds must be strictly ordered")
        planes = [{"normal": [int(i == j) for i in range(3)], "offset": float(low[j])} for j in range(3)]
        planes += [{"normal": [-int(i == j) for i in range(3)], "offset": -float(high[j])} for j in range(3)]
        possible, certified = [domain], [domain]

        def constrain(outer, inner, count):
            nonlocal possible, certified
            possible, certified = _intersect(possible, outer), _intersect(certified, inner)
            result["polynomial_pieces"] += count
            _check(stop)

        for slope, intercept in ((r1, r0 - minimum), (-r1, maximum - r0)):
            outer, inner = _positive(Fraction(0), slope, intercept, domain)
            constrain(outer, inner, 1)
        # Restrict first: the plane norm enclosure assumes a nonnegative radius.
        radius_domains = possible
        if not radius_domains:
            result.update(status="no_solution")
            return result
        names = set()
        for primitive in primitives:
            _check(stop)
            if not isinstance(primitive, dict) or not isinstance(primitive.get("name"), str) or primitive["name"] in names:
                raise ValueError("Depth primitive identities must be unique")
            names.add(primitive["name"])
            kind = primitive.get("type")
            if kind not in {"sphere", "capsule"}:
                raise _Unsupported("primitive_type:" + str(kind))
            method = _sphere if kind == "sphere" else _capsule
            outer, inner, count = method(p0, p1, r0, r1, primitive, domain)
            constrain(outer, inner, count)
        for plane in [floor, *planes]:
            outer, inner, count = [], [], 0
            for span in radius_domains:
                a, b, n = _plane(p0, p1, r0, r1, plane, span)
                outer.extend(a); inner.extend(b); count += n
            constrain(_union(outer), _union(inner), count)
        result["possible_intervals"] = _output(possible, False)
        result["certified_intervals"] = _output(certified, True)
        result["boundary_ambiguity"] = result["possible_intervals"] != result["certified_intervals"]
        result["status"] = "no_solution" if not possible else "feasible_intervals" if result["certified_intervals"] else "boundary_only"
        _check(stop)
        return result
    except _Unsupported as exc:
        result.update(status="unsupported", possible_intervals=[], certified_intervals=[], reasons=[str(exc)])
        return result
