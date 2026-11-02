"""Conditional static-surface anchor ranges; no chosen height, contact or body installation."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
import time
from typing import NotRequired, TypedDict

from weft.camera_pose import validate_camera
from weft.source_surface_anchors import (
    _annotations, _document, _fields, _receipt_bound, _sources, _surfaces,
)
from weft.world_alignment import load_world_alignment


PROFILE = "source-surface-envelopes-v1"
INTERVAL_PROFILE = "source-horizontal-surface-intervals-v1"
MAX_SECONDS, MAX_FRAMES = 10., 1800
MAX_EXTRA_BYTES, MAX_CACHE_BYTES, MAX_ROW_BYTES, MAX_METADATA_BYTES = 4 << 20, 8 << 20, 128 << 10, 128 << 10
RADIAL_PROFILE = "certified-nonnegative-radial-inverse-v1"
RADIAL_BISECTIONS, RADIAL_CACHE_ENTRIES, RADIAL_RADIUS_SQUARED_MAX = 48, 256, 16
CONDITIONAL_DOMAIN_MODE = "condition_on_declared_support"
CORRELATED_GEOMETRY = "correlated_pinhole_region_v1"
REGION_PROFILE = "source-surface-correlated-region-v1"
MAX_REGION_BYTES, MAX_REGION_PLANES = 8 << 10, 10


class AxisAlignedEnvelope(TypedDict):
    minimum: list[float]
    maximum: list[float]
    unit: str
    kind: str
    certificate: str
    support_conditioning: NotRequired[dict]
    constraint_region_m: NotRequired[dict]


def _require(value, message):
    if not value:
        raise ValueError("Source surface envelopes: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Source surface envelopes exceeded the cooperative ten-second bound")


def _number(value, low, high):
    return type(value) in (int, float) and low <= value <= high and math.isfinite(value)


def _encode(value, maximum, deadline):
    pieces, size = [], 0
    for part in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
        _check(deadline)
        raw = part.encode()
        size += len(raw)
        _require(size <= maximum, "complete output exceeds its bound; no slots were dropped")
        pieces.append(raw)
    return b"".join(pieces)


def _intervals(document, source, camera_hash, evidence, receipts):
    if document is None:
        return {}, [], None
    _fields(document, ("schema_version", "profile", "source_sha256", "calibration_sha256", "world_frame",
                       "calibration_evidence_refs", "surfaces"), "surface interval document")
    _require(type(document["schema_version"]) is int and document["schema_version"] == 1
             and document["profile"] == INTERVAL_PROFILE, "versioned surface interval document required")
    _require(type(document["surfaces"]) is list and len(document["surfaces"]) <= 12, "at most twelve static surfaces")
    intervals, checked = {}, []
    for item in document["surfaces"]:
        _require(type(item) is dict, "surface interval must be an object")
        fields = ("id", "kind", "height_interval_m", "bounds_xy_m", "quality", "provenance", "evidence_refs")
        _fields(item, fields + (("domain_mode",) if "domain_mode" in item else ()), "surface interval")
        _require("domain_mode" not in item or item["domain_mode"] == CONDITIONAL_DOMAIN_MODE
                 and item["kind"] == "static_support", "conditional domain mode requires an explicit static support")
        _require(type(item["provenance"]) is str and 1 <= len(item["provenance"].strip()) <= 2000,
                 "surface interval needs bounded explicit provenance")
        bounds = item["height_interval_m"]
        _require(bounds is None and item["quality"] == "unknown" or
                 type(bounds) is list and len(bounds) == 2 and all(_number(v, -1000, 1000) for v in bounds)
                 and bounds[0] <= bounds[1] and item["quality"] == "authored",
                 "height ranges must be explicit authored bounds or unknown; measured ranges are not inferred")
        # The existing validator checks custody/domain only; no nominal or midpoint height is invented.
        checked.append({key: value for key, value in item.items() if key not in ("height_interval_m", "quality", "domain_mode")}
                       | {"height_m": None, "quality": "unknown"})
    shadow = document | {"profile": "source-horizontal-surfaces-v1", "surfaces": checked}
    validated, calibration, world = _surfaces(shadow, source, camera_hash, evidence, receipts)
    for item in document["surfaces"]:
        _require(item["id"] in validated, "surface identity did not survive schema validation")
        intervals[item["id"]] = item
    return intervals, calibration, world


def _inverse(matrix):
    a, b, c, d, e, f, g, h, i = (Fraction(value) for row in matrix for value in row)
    adj = ((e*i-f*h, c*h-b*i, b*f-c*e), (f*g-d*i, a*i-c*g, c*d-a*f), (d*h-e*g, b*g-a*h, a*e-b*d))
    determinant = a*adj[0][0] + b*adj[1][0] + c*adj[2][0]
    _require(determinant != 0, "singular supplied camera matrix")
    return tuple(tuple(value / determinant for value in row) for row in adj)


def _multiply(matrix, vector):
    return tuple(sum(a*b for a, b in zip(row, vector, strict=True)) for row in matrix)


def _camera_frame(camera, frame, inverse_k):
    transform = camera["frames"][frame]
    inverse_r = _inverse(transform["R"])
    origin = tuple(-value for value in _multiply(inverse_r, tuple(map(Fraction, transform["t"]))))
    if camera["radial_k1"] > 0:
        return origin, inverse_r
    directions = tuple(tuple(sum(inverse_r[i][k] * inverse_k[k][j] for k in range(3))
                             for j in range(3)) for i in range(3))
    return origin, directions


def _outward(value, lower):
    result = float(value)
    represented = Fraction(result)
    if lower and represented > value or not lower and represented < value:
        result = math.nextafter(result, -math.inf if lower else math.inf)
    return result


def _cap_hull(points, axis):
    axes = [index for index in range(3) if index != axis]
    ordered = sorted(set(points), key=lambda point: tuple(point[index] for index in axes))
    if len(ordered) < 3:
        return ordered
    def cross(a, b, c):
        i, j = axes
        return (b[i]-a[i])*(c[j]-a[j]) - (b[j]-a[j])*(c[i]-a[i])
    halves = []
    for sequence in (ordered, reversed(ordered)):
        half = []
        for point in sequence:
            while len(half) > 1 and cross(half[-2], half[-1], point) <= 0:
                half.pop()
            half.append(point)
        halves.append(half)
    return halves[0][:-1] + halves[1][:-1]


def _clip_support(points, domain):
    # Positive ray depth makes the eight-vertex height slab convex; clipping keeps correlations.
    faces = [[points[index] for index in face] for face in (
        (0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4),
        (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5))]
    for axis, bound, sign in ((0, domain[0], 1), (1, domain[1], 1),
                              (0, domain[2], -1), (1, domain[3], -1)):
        bound = Fraction(bound)
        clipped, cap = [], []
        for face in faces:
            output = []
            previous, distance = face[-1], sign*(face[-1][axis]-bound)
            for current in face:
                next_distance = sign*(current[axis]-bound)
                if (distance >= 0) != (next_distance >= 0):
                    weight = distance/(distance-next_distance)
                    point = tuple(a+weight*(b-a) for a, b in zip(previous, current, strict=True))
                    output.append(point); cap.append(point)
                if next_distance >= 0:
                    output.append(current)
                    if next_distance == 0:
                        cap.append(current)
                previous, distance = current, next_distance
            deduplicated = list(dict.fromkeys(output))
            if deduplicated:
                clipped.append(deduplicated)
        if not clipped:
            return []
        if cap:
            clipped.append(_cap_hull(cap, axis))
        faces = clipped
    return list(dict.fromkeys(point for face in faces for point in face))


def _bounds_envelope(minimum, maximum):
    return {"minimum": [_outward(value, True) for value in minimum],
        "maximum": [_outward(value, False) for value in maximum], "unit": "m",
        "kind": "outer_axis_aligned_enclosure", "certificate": "exact_rational_eight_vertex_extrema_outward_binary64"}


def _affine_hull(points):
    points = list(dict.fromkeys(points))
    _require(1 <= len(points) <= 16, "correlated region vertex bound exceeded")
    origin = points[0]
    rows = [[a-b for a, b in zip(point, origin, strict=True)] for point in points[1:]]
    pivots = []
    for column in range(3):
        pivot = next((i for i in range(len(pivots), len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        index = len(pivots)
        rows[index], rows[pivot] = rows[pivot], rows[index]
        divisor = rows[index][column]
        rows[index] = [value/divisor for value in rows[index]]
        for i, row in enumerate(rows):
            if i != index:
                factor = row[column]
                rows[i] = [value-factor*other for value, other in zip(row, rows[index], strict=True)]
        pivots.append(column)
    planes = []
    for free in (i for i in range(3) if i not in pivots):
        normal = [Fraction(int(i == free)) for i in range(3)]
        for row, pivot in zip(rows, pivots):
            normal[pivot] = -row[free]
        planes.append((*normal, sum(a*b for a, b in zip(normal, origin, strict=True))))
    return planes


def _correlated_region(origin, transform, xs, ys, surface, points):
    inverse = _inverse(transform)
    planes = []
    for axis, interval in ((0, xs), (1, ys)):
        for bound, sign in ((interval[0], 1), (interval[1], -1)):
            normal = tuple(sign * (a - bound * b) for a, b in zip(inverse[axis], inverse[2], strict=True))
            planes.append((*normal, sum(a*b for a, b in zip(normal, origin, strict=True))))
    lo, hi = map(Fraction, surface["height_interval_m"])
    planes.extend(((0, 0, 1, lo), (0, 0, -1, -hi)))
    if surface["bounds_xy_m"] is not None:
        left, bottom, right, top = map(Fraction, surface["bounds_xy_m"])
        planes.extend(((1, 0, 0, left), (0, 1, 0, bottom), (-1, 0, 0, -right), (0, -1, 0, -top)))
    seed = tuple(sum(point[i] for point in points) / len(points) for i in range(3))
    _require(all(sum(Fraction(a)*b for a, b in zip(plane[:3], seed, strict=True)) >= plane[3] for plane in planes),
             "correlated region seed is not feasible")
    def rational(value):
        value = Fraction(value)
        return [str(value.numerator), str(value.denominator)]
    result = {"profile": REGION_PROFILE, "unit": "m", "convention": "A*x >= b",
              "halfspaces_q": [[rational(value) for value in plane] for plane in planes],
              "affine_hull_q": [[rational(value) for value in plane] for plane in _affine_hull(points)],
              "feasible_seed_q": [rational(value) for value in seed]}
    _require(len(json.dumps(result, separators=(",", ":")).encode()) <= MAX_REGION_BYTES,
             "complete correlated region exceeds eight KiB")
    return result


def _plane_box(origin, transform, xs, ys, surface, *, constraint_geometry=None):
    directions = [_multiply(transform, (u, v, Fraction(1))) for u in xs for v in ys]
    denominators = [direction[2] for direction in directions]
    if min(denominators) <= 0 <= max(denominators):
        return None, "ray_interval_crosses_surface_horizon"
    points = []
    for height in map(Fraction, surface["height_interval_m"]):
        for direction in directions:
            depth = (height-origin[2]) / direction[2]
            if not 0 < depth <= 1000:
                return None, "height_interval_behind_camera_or_beyond_depth_bound"
            points.append(tuple(a+depth*b for a, b in zip(origin, direction, strict=True)))
    minimum = tuple(min(point[axis] for point in points) for axis in range(3))
    maximum = tuple(max(point[axis] for point in points) for axis in range(3))
    if any(abs(value) > 1_000_000 for value in (*minimum, *maximum)):
        return None, "envelope_exceeds_coordinate_bound"
    domain = surface["bounds_xy_m"]
    envelope: AxisAlignedEnvelope = _bounds_envelope(minimum, maximum)
    exceeds = domain is not None and any(minimum[axis] < Fraction(domain[axis]) or maximum[axis] > Fraction(domain[axis+2]) for axis in (0, 1))
    if surface.get("domain_mode") != CONDITIONAL_DOMAIN_MODE:
        if exceeds:
            return None, "envelope_exceeds_declared_surface_domain"
        if constraint_geometry is not None:
            envelope["constraint_region_m"] = _correlated_region(origin, transform, xs, ys, surface, points)
        return envelope, None
    original = dict(envelope)
    if exceeds:
        points = _clip_support(points, domain)
        if not points:
            return None, "ray_height_region_disjoint_from_declared_support"
        minimum = tuple(min(point[axis] for point in points) for axis in range(3))
        maximum = tuple(max(point[axis] for point in points) for axis in range(3))
        envelope = _bounds_envelope(minimum, maximum)
    envelope["certificate"] = "exact_rational_convex_ray_height_region_support_clipping_outward_binary64"
    envelope["support_conditioning"] = {"mode": CONDITIONAL_DOMAIN_MODE, "bounds_xy_m": domain,
        "unconditioned_world_envelope_m": original, "region_restricted": exceeds,
        "scope": "Conditional on the supplied static-support hypothesis; no contact, load or grasp evidence. "
                 "An enclosed point need not satisfy the correlated pixel/height constraints; radial outer rays may be infeasible."}
    if constraint_geometry is not None:
        envelope["constraint_region_m"] = _correlated_region(origin, transform, xs, ys, surface, points)
    return envelope, None


def _pixel_box(annotation, camera):
    pixel, uncertainty = annotation["pixel"], annotation["uncertainty_px"]
    _require(type(pixel) is list and len(pixel) == 2 and all(_number(v, -1e6, 1e6) for v in pixel)
             and _number(uncertainty, 0, 64), "finite authored pixel and bounded uncertainty required")
    x, y, radius = Fraction(pixel[0]), Fraction(pixel[1]), Fraction(uncertainty)
    xs, ys = (x-radius, x+radius), (y-radius, y+radius)
    if xs[0] < 0 or xs[1] > camera["width"] or ys[0] < 0 or ys[1] > camera["height"]:
        return None
    return xs, ys


def _enclose(annotation, surface, camera, prepared, *, constraint_geometry=None):
    bounds = _pixel_box(annotation, camera)
    if bounds is None:
        return None, "pixel_interval_outside_source_image"
    xs, ys = bounds
    envelope, reason = _plane_box(*prepared, xs, ys, surface, constraint_geometry=constraint_geometry)
    if envelope is None:
        return None, reason
    return {"world_envelope_m": envelope, "height_interval_m": surface["height_interval_m"],
            "pixel_interval_px": {"minimum": [_outward(xs[0], True), _outward(ys[0], True)],
                                  "maximum": [_outward(xs[1], False), _outward(ys[1], False)]},
            "uncertainty_scope": "Complete supplied pixel/height ranges under fixed pinhole coefficients; excludes calibration error",
            "surface_quality_claim": "authored", "annotation_quality": "authored"}, None


class _RadialInverse:
    """Exact outer inverse boxes for the monotone k>=0 branch; no selected inverse point."""

    def __init__(self, inverse_k, radial, deadline):
        self.inverse_k, self.radial, self.deadline = inverse_k, Fraction(radial), deadline
        _require(0 < self.radial <= 1, "certified radial inverse requires 0<k<=1")
        self.cache = {}
        self.computed_boxes = self.cache_hits = self.bisections = 0

    def scale(self, squared_radius):
        _check(self.deadline)
        _require(0 <= squared_radius <= RADIAL_RADIUS_SQUARED_MAX, "radial squared radius exceeds the fixed inverse domain")
        if squared_radius == 0:
            return Fraction(1), Fraction(1)
        low, high = Fraction(0), Fraction(1)
        coefficient = self.radial * squared_radius
        for _ in range(RADIAL_BISECTIONS):
            _check(self.deadline)
            probe = (low+high) / 2
            forward = probe + coefficient * probe**3
            self.bisections += 1
            if forward == 1:
                return probe, probe
            if forward < 1:
                low = probe
            else:
                high = probe
        return low, high

    def bounds(self, xs, ys):
        _check(self.deadline)
        key = (*xs, *ys)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.computed_boxes += 1
        corners = []
        for x in xs:
            for y in ys:
                _check(self.deadline)
                point = _multiply(self.inverse_k, (x, y, Fraction(1)))
                _require(point[2] == 1, "radial inverse requires the validated affine pinhole intrinsics")
                corners.append(point[:2])
        intervals = tuple((min(point[i] for point in corners), max(point[i] for point in corners)) for i in range(2))
        minimum = sum(0 if lo <= 0 <= hi else min(lo*lo, hi*hi) for lo, hi in intervals)
        # Squared norm is convex: its maximum over the parallelogram is attained at a vertex.
        maximum = max(x*x+y*y for x, y in corners)
        if maximum > RADIAL_RADIUS_SQUARED_MAX:
            result = None, "radial_pixel_interval_exceeds_inverse_domain"
        else:
            at_minimum = self.scale(minimum)
            at_maximum = at_minimum if minimum == maximum else self.scale(maximum)
            scales = (at_maximum[0], at_minimum[1])
            undistorted = []
            for coordinate in intervals:
                _check(self.deadline)
                products = tuple(value*scale for value in coordinate for scale in scales)
                undistorted.append((min(products), max(products)))
            result = {"intervals": tuple(undistorted), "distorted": intervals, "squared_radius": (minimum, maximum),
                      "scales": scales, "brackets_at_radius_bounds": (at_minimum, at_maximum)}, None
        if len(self.cache) == RADIAL_CACHE_ENTRIES:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = result
        _check(self.deadline)
        return result


def _outward_pair(bounds):
    return [_outward(bounds[0], True), _outward(bounds[1], False)]


def _enclose_radial(annotation, surface, camera, prepared, inverse):
    pixels = _pixel_box(annotation, camera)
    if pixels is None:
        return None, "pixel_interval_outside_source_image"
    box, reason = inverse.bounds(*pixels)
    if box is None:
        return None, reason
    envelope, reason = _plane_box(*prepared, *box["intervals"], surface)
    if envelope is None:
        return None, reason
    envelope["certificate"] = ("exact_rational_monotone_radial_outer_box_support_clipping_outward_binary64"
        if "support_conditioning" in envelope else "exact_rational_monotone_radial_outer_box_plane_extrema_outward_binary64")
    if "support_conditioning" in envelope:
        envelope["support_conditioning"]["unconditioned_world_envelope_m"]["certificate"] = "exact_rational_monotone_radial_outer_box_plane_extrema_outward_binary64"
    xs, ys = pixels
    diagnostic = {"profile": RADIAL_PROFILE, "radial_k1": camera["radial_k1"],
        "distorted_normalized_intervals": [_outward_pair(pair) for pair in box["distorted"]],
        "distorted_radius_squared_interval": _outward_pair(box["squared_radius"]),
        "inverse_scale_interval": _outward_pair(box["scales"]),
        "scale_brackets_at_radius_bounds": [_outward_pair(pair) for pair in box["brackets_at_radius_bounds"]],
        "undistorted_normalized_intervals": [_outward_pair(pair) for pair in box["intervals"]],
        "scope": "Outer enclosure of the exact supplied radial model; correlations discarded, no point or calibration selected"}
    _check(inverse.deadline)
    return {"world_envelope_m": envelope, "height_interval_m": surface["height_interval_m"],
        "pixel_interval_px": {"minimum": [_outward(xs[0], True), _outward(ys[0], True)],
                              "maximum": [_outward(xs[1], False), _outward(ys[1], False)]},
        "radial_inverse": diagnostic,
        "uncertainty_scope": "Complete supplied pixel/height ranges under fixed nonnegative-radial coefficients; excludes calibration error",
        "surface_quality_claim": "authored", "annotation_quality": "authored"}, None


@dataclass(frozen=True)
class SourceSurfaceEnvelopes:
    """Retain exact inputs once and inspect bounded immutable cached rows."""

    observation_inputs: tuple
    additional_inputs: tuple
    _metadata: bytes
    _rows: tuple[bytes, ...]

    @property
    def frame_count(self):
        return len(self._rows)

    def describe(self):
        return json.loads(self._metadata)

    def frame(self, index: int):
        _require(type(index) is int and 0 <= index < len(self._rows), "original source frame index outside range")
        return json.loads(self._rows[index])

    def iter_frame_bytes(self):
        return iter(self._rows)


def prepare_source_surface_envelopes(observation_documents: Mapping[str, bytes], *, source_sha256: str,
        camera_document: bytes | None = None, surface_interval_document: bytes | None = None,
        annotation_document: bytes | None = None, constraint_geometry: str | None = None,
        deadline: float | None = None) -> SourceSurfaceEnvelopes:
    """Build source-complete conditional anchor ranges; perform no I/O or geometry installation."""
    _require(constraint_geometry is None or constraint_geometry == CORRELATED_GEOMETRY,
             "unsupported explicit constraint geometry")
    _require(deadline is None or type(deadline) in (int, float) and math.isfinite(deadline), "invalid caller deadline")
    own_deadline = time.monotonic() + (min(MAX_SECONDS, 5.) if constraint_geometry else MAX_SECONDS)
    deadline = own_deadline if deadline is None else min(own_deadline, deadline)
    supplied = (camera_document, surface_interval_document, annotation_document)
    _require(all(raw is None or type(raw) is bytes for raw in supplied)
             and sum(len(raw) for raw in supplied if raw is not None) <= MAX_EXTRA_BYTES, "additional exact inputs exceed 4 MiB")
    source, documents, observations, evidence, allowed_ids, receipts = _sources(observation_documents, source_sha256, deadline)
    frame_count = max(source["decoded_frames"], source["metadata_frames"])
    _require(frame_count <= MAX_FRAMES, "complete original source exceeds 1800 slots")
    camera, surface_input, annotation_input = [_document(raw, deadline) for raw in supplied]
    camera_hash = sha256(camera_document).hexdigest() if camera_document is not None else None
    surfaces, calibration, world = _intervals(surface_input, source, camera_hash, evidence, receipts)
    channels, annotations = _annotations(annotation_input, source, documents, observations, deadline)
    camera_reason, inverse_k, alignment, radial_inverse = "missing_camera_calibration", None, None, None
    if camera is not None:
        validate_camera(camera, {"source_sha256": source_sha256, "width": source["width_px"], "height": source["height_px"],
                                "frame_count": source["metadata_frames"], "fps": source.get("fps", camera.get("fps"))})
        camera_reason = None
        if camera["quality"] == "measured" and not any(row["kind"] == "camera_calibration_measurement" and row["status"] == "observed"
                and _receipt_bound(row, receipts, source_sha256, camera_hash) for row in calibration):
            camera_reason = "missing_supplied_camera_measurement_receipt"
        elif constraint_geometry is not None and camera["radial_k1"] != 0:
            camera_reason = "correlated_region_requires_undistorted_pinhole"
        elif camera["radial_k1"] < 0:
            camera_reason = "radial_distortion_has_no_certified_interval_bound"
        if camera_reason is None:
            inverse_k = _inverse(camera["K"])
            if camera["radial_k1"] > 0:
                radial_inverse = _RadialInverse(inverse_k, camera["radial_k1"], deadline)
    if camera is not None and world is not None:
        claim = {"status": "assumed", "method": "Explicit interval surface in the supplied calibration world; no fitted mapping",
                 "uncertainty": "Source/input custody is checked, not physical calibration or contact", "evidence_refs": surface_input["calibration_evidence_refs"]}
        mapping = {"schema_version": 1, "profile": "world-alignment-v1",
            "from_frame": {"id": "calibration-world:"+camera_hash, "length_unit": "m", "axes": "x_right_y_away_z_up"},
            "to_frame": world, "spatial": {"scale": 1., "rotation_wxyz": [1., 0., 0., 0.], "translation": [0., 0., 0.]},
            "clock": {"rate": 1., "source_origin_s": 0., "target_origin_s": 0.}, "claims": {"spatial": claim,
                "clock": {"status": "assumed", "method": "Exact original source-clock identity; no retiming",
                          "uncertainty": "No physical synchronization independently measured", "evidence_refs": []}}}
        alignment = load_world_alignment(mapping, evidence_ids_by_document=allowed_ids).document()
    rows, counts, prepared_frames, cache_bytes = [], Counter(), 0, 0
    for frame in range(frame_count):
        _check(deadline)
        row, prepared = {"source_frame": frame, "source_status": "decoded" if frame < source["decoded_frames"] else "metadata_only_not_decoded", "anchors": []}, None
        for channel in channels:
            ref = channel["entity_ref"]
            observation = observations.get((ref["document_sha256"], ref["object_id"], frame))
            annotation = annotations.get((frame, channel["id"]))
            anchor = {"channel_id": channel["id"], "entity_ref": ref, "surface_id": channel["surface_id"],
                "annotation_id": None if annotation is None else annotation["id"],
                "observation_id": None if observation is None else observation["id"],
                "source_time_s": None if observation is None else observation["time_s"],
                "partition": None if observation is None else observation["partition"],
                "status": "unavailable", "reason": None, "world_envelope_m": None}
            if frame >= source["decoded_frames"]:
                reason = "metadata_only_not_decoded"
            elif observation is None:
                reason = "missing_source_observation"
            elif observation["partition"] == "unsampled":
                reason = "source_not_sampled"
            elif observation["admission"] != "included":
                reason = "excluded_evidence_partition"
            elif observation["visibility"] not in ("detected", "partial"):
                reason = "source_entity_unavailable"
            elif annotation is None or annotation["status"] != "annotated":
                reason = "missing_annotation" if annotation is None else "annotation_"+annotation["status"]
            elif camera_reason:
                reason = camera_reason
            elif channel["surface_id"] not in surfaces:
                reason = "missing_surface_geometry"
            elif surfaces[channel["surface_id"]]["height_interval_m"] is None:
                reason = "unknown_surface_height_interval"
            elif frame >= len(camera["frames"]) or not math.isclose(observation["time_s"], camera["frames"][frame]["time_s"], abs_tol=1e-9, rel_tol=0):
                reason = "camera_and_source_clocks_differ"
            else:
                if prepared is None:
                    prepared = _camera_frame(camera, frame, inverse_k)
                    prepared_frames += 1
                if radial_inverse is None:
                    geometry, reason = _enclose(annotation, surfaces[channel["surface_id"]], camera, prepared,
                        **({"constraint_geometry": constraint_geometry} if constraint_geometry is not None else {}))
                else:
                    geometry, reason = _enclose_radial(annotation, surfaces[channel["surface_id"]], camera, prepared, radial_inverse)
                if geometry is not None:
                    anchor.update(geometry, status="conditional_surface_envelope", camera_quality_claim=camera["quality"])
            anchor["reason"] = reason
            counts[reason or anchor["status"]] += 1
            row["anchors"].append(anchor)
        raw = _encode(row, MAX_ROW_BYTES, deadline)
        cache_bytes += len(raw)
        _require(cache_bytes <= MAX_CACHE_BYTES, "complete source cache exceeds 8 MiB; no slots were dropped")
        rows.append(raw)
    metadata = {"schema_version": 1, "profile": PROFILE, "source_sha256": source_sha256, "world_frame": world,
        "status": "conditional_envelopes_available" if counts["conditional_surface_envelope"] else "unavailable",
        "bindings": {"observation_document_sha256s": sorted(documents), "camera_sha256": camera_hash,
            "surface_intervals_sha256": None if surface_interval_document is None else sha256(surface_interval_document).hexdigest(),
            "annotations_sha256": None if annotation_document is None else sha256(annotation_document).hexdigest()},
        "source_clocks": {digest: value["time"] for digest, value in documents.items()}, "channels": channels,
        "surface_intervals": list(surfaces.values()), "calibration_world_alignment": alignment,
        "counts": {"decoded_frames": source["decoded_frames"], "metadata_frames": source["metadata_frames"],
            "retained_source_frames": frame_count, "retained_anchor_slots": frame_count*len(channels),
            "supplied_annotation_slots": len(annotations), "prepared_camera_frames": prepared_frames,
            "conditional_envelopes": counts["conditional_surface_envelope"], "slot_status_counts": dict(counts)},
        "missing_inputs": [name for name, raw in zip(("camera", "surface_intervals", "annotations"), supplied, strict=True) if raw is None],
        "consumer_contract": {"kind": "outer_axis_aligned_world_anchor_enclosure", "midpoint": "not_a_measurement_or_selected_target",
            "required_join": "Exact source/entity/frame/clock plus explicit world-to-body alignment and interval-aware construction objective",
            "existing_consumer": "No interval consumer installed; point-only construction must not silently substitute a midpoint"},
        "certificate": {"arithmetic": "Exact rationals of supplied numeric values; coordinate extrema rounded outward to binary64",
            "proof": "Constant-sign affine ray denominator gives linear-fractional pixel extrema at corners; height dependence is affine",
            "conditions": "Undistorted fixed pinhole coefficients, authored closed pixel/height ranges, forward depths within1000m",
            "excludes": "Calibration accuracy, support membership truth, body scale, anatomy and physical contact"},
        "bounds": {"maximum_seconds": MAX_SECONDS, "maximum_source_frames": MAX_FRAMES,
            "maximum_extra_bytes": MAX_EXTRA_BYTES, "maximum_cache_bytes": MAX_CACHE_BYTES}, "cached_row_bytes": cache_bytes,
        "body_or_floor_installed": False, "geometry_modified": False, "contact": "unknown", "load": "unknown", "grasp": "unknown",
        "physical_validation": "unverified", "moving_or_handheld_supports": "unsupported"}
    if radial_inverse is not None:
        metadata["radial_inverse"] = {"profile": RADIAL_PROFILE, "radial_k1": camera["radial_k1"],
            "maximum_distorted_radius": 4., "bisections_per_scalar_bracket": RADIAL_BISECTIONS,
            "maximum_cache_entries": RADIAL_CACHE_ENTRIES, "computed_boxes": radial_inverse.computed_boxes,
            "cache_hits": radial_inverse.cache_hits, "bisections": radial_inverse.bisections,
            "scope": "Unique nonnegative-k inverse under supplied coefficients, not independently validated calibration"}
        metadata["certificate"]["proof"] = ("Monotone exact-rational radial scale brackets enclose every normalized ray; "
            "constant-sign affine plane denominator gives extrema on the outer ray rectangle and height endpoints")
        metadata["certificate"]["conditions"] = ("Fixed 0<radial_k1<=1 pinhole coefficients, complete distorted radius<=4, "
            "authored closed pixel/height ranges, forward depths within1000m")
    conditioned = [surface["id"] for surface in surfaces.values() if surface.get("domain_mode") == CONDITIONAL_DOMAIN_MODE]
    if conditioned:
        metadata["support_conditioning"] = {"mode": CONDITIONAL_DOMAIN_MODE, "surface_ids": conditioned,
            "scope": "Explicit static-support hypothesis, not recovered membership or physical contact; "
                     "original pixel/height ranges remain saved and no midpoint is selected",
            "empty_intersection": "unavailable", "available_enclosures_retain_original": True}
        metadata["certificate"]["proof"] += "; conditional supports clip the convex ray/height region with four exact half-planes before outward enclosure"
    if constraint_geometry is not None:
        metadata["constraint_geometry"] = constraint_geometry
        metadata["correlated_region"] = {"profile": REGION_PROFILE, "maximum_planes": MAX_REGION_PLANES,
            "maximum_region_bytes": MAX_REGION_BYTES, "preparation_seconds_cooperative": min(MAX_SECONDS, 5.),
            "scope": "Exact rational supplied pinhole/height/support region; numerical projection is tolerance-qualified, not physical contact"}
    encoded = _encode(metadata, MAX_METADATA_BYTES, deadline)
    _require(cache_bytes+len(encoded) <= MAX_CACHE_BYTES, "complete metadata/source cache exceeds 8 MiB")
    _check(deadline)
    return SourceSurfaceEnvelopes(tuple(sorted(observation_documents.items())), supplied, encoded, tuple(rows))
