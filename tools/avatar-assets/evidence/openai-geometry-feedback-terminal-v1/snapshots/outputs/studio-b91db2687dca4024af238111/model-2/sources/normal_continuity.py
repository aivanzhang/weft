"""Bounded transfer of authored surface normals; positions and topology never change."""

from copy import deepcopy
import hashlib
import math
import re

import numpy as np


MAX_VERTICES = 100000
MAX_FACES = 100000
MAX_CORNERS = 300000
MAX_TARGET_VERTICES = 10000
MAX_PAIRS = 2000000
MAX_WITNESSES = 32
POINT_ATTRIBUTE = 'AuthoredSurfaceNormal'
FACE_ATTRIBUTE = 'AuthoredNormalRepair'
FIELDS = {'name', 'materials', 'reference_materials', 'max_distance_m', 'min_vertices', 'max_vertices'}


class NormalContinuityError(ValueError):
    """A rejected plan with bounded, JSON-safe geometric witnesses."""

    def __init__(self, report):
        self.report = deepcopy(report)
        super().__init__(f"Normal continuity {report['name']} rejected before mutation: "
            f"{report['degenerate_faces']} degenerate/cancelling target faces, "
            f"{report['outside_hemisphere_corners']} transferred corners outside their face hemisphere")


def _material_names(value):
    if (not isinstance(value, list) or not 1 <= len(value) <= 64 or any(
            not isinstance(name, str) or not 1 <= len(name) <= 128 or name.strip() != name
            or any(ord(char) < 32 or ord(char) == 127 for char in name) for name in value)
            or len(set(value)) != len(value)):
        raise ValueError('Normal continuity requires 1–64 unique exact material names')


def validate_normal_continuity(configs):
    """Empty means disabled; target sets cannot overlap or serve as references."""
    if not isinstance(configs, list) or len(configs) > 4:
        raise ValueError('Normal continuity supports at most four records')
    names, targets, references = set(), set(), set()
    for config in configs:
        if not isinstance(config, dict) or set(config) != FIELDS:
            raise ValueError('Invalid normal continuity fields')
        name = config['name']
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', name) or name in names:
            raise ValueError('Normal continuity names must be unique safe slugs')
        names.add(name)
        for key in ('materials', 'reference_materials'):
            _material_names(config[key])
        if targets.intersection(config['materials']):
            raise ValueError('Normal continuity target material sets overlap')
        targets.update(config['materials'])
        references.update(config['reference_materials'])
        distance = config['max_distance_m']
        if type(distance) not in (int, float) or not math.isfinite(distance) or not 0 <= distance <= .03:
            raise ValueError('Normal continuity distance must be finite and within 0–0.03 meters')
        if (type(config['min_vertices']) is not int or type(config['max_vertices']) is not int
                or not 1 <= config['min_vertices'] <= config['max_vertices'] <= MAX_TARGET_VERTICES):
            raise ValueError('Normal continuity requires ordered vertex bounds within 1–10000')
    if targets.intersection(references):
        raise ValueError('Normal continuity target and reference material sets overlap')
    return deepcopy(configs)


def _vectors(value, count_limit, name):
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape[1] != 3 or value.dtype.kind not in 'fiu':
            raise ValueError('Invalid normal continuity '+name)
    elif (not isinstance(value, (list, tuple)) or not 1 <= len(value) <= count_limit or any(
            not isinstance(row, (list, tuple)) or len(row) != 3
            or any(type(component) not in (int, float) for component in row) for row in value)):
        raise ValueError('Invalid normal continuity '+name)
    if not 1 <= len(value) <= count_limit:
        raise ValueError('Oversized or empty normal continuity '+name)
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(np.abs(result) > 1000000):
        raise ValueError('Normal continuity '+name+' exceeds finite numerical bounds')
    return result


def _faces(value, vertices):
    if isinstance(value, np.ndarray):
        if (value.ndim != 2 or value.dtype.kind not in 'iu' or not 1 <= len(value) <= MAX_FACES
                or not 3 <= value.shape[1] <= 64 or value.size > MAX_CORNERS):
            raise ValueError('Invalid normal continuity faces')
        sizes = np.full(len(value), value.shape[1], dtype=np.int64)
        flat = value.reshape(-1)
    else:
        if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_FACES:
            raise ValueError('Invalid or oversized normal continuity faces')
        sizes, count = [], 0
        for face in value:
            if (not isinstance(face, (list, tuple)) or not 3 <= len(face) <= 64
                    or any(type(i) is not int or not 0 <= i < vertices for i in face)):
                raise ValueError('Normal continuity faces require bounded integer indices')
            count += len(face)
            if count > MAX_CORNERS:
                raise ValueError('Normal continuity exceeds 300000 corners')
            sizes.append(len(face))
        sizes = np.asarray(sizes, dtype=np.int64)
        flat = np.asarray([i for face in value for i in face], dtype=np.int64)
    if (not 1 <= len(sizes) <= MAX_FACES or len(flat) > MAX_CORNERS
            or flat.min() < 0 or flat.max() >= vertices):
        raise ValueError('Invalid or oversized normal continuity face indices')
    return flat, sizes


def array_hash(value, dtype):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def _unit(rows, label):
    lengths = np.linalg.norm(rows, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError('Normal continuity has cancelling or zero '+label)
    return rows/lengths[:, None]


def _nearest(points, references, distance, budget):
    exact = {tuple(point): i for i, point in enumerate(references)}
    result, distances = np.full(len(points), -1, dtype=np.int64), np.zeros(len(points))
    cell = max(distance, 1e-6)
    buckets = {}
    if distance:
        for i, point in enumerate(references):
            key = tuple(math.floor(float(v)/cell) for v in point)
            buckets.setdefault(key, []).append(i)
    for i, point in enumerate(points):
        found = exact.get(tuple(point))
        if found is not None:
            result[i] = found
            continue
        if not distance:
            continue
        key = tuple(math.floor(float(v)/cell) for v in point)
        candidates = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    candidates.extend(buckets.get((key[0]+x, key[1]+y, key[2]+z), ()))
        budget[0] += len(candidates)
        if budget[0] > MAX_PAIRS:
            raise ValueError('Normal continuity nearest search exceeds two million candidate pairs')
        if candidates:
            candidates.sort()  # Lexicographic reference ordering resolves equal-distance ties.
            squared = np.sum((references[candidates]-point)**2, axis=1)
            nearest = int(np.argmin(squared))
            if squared[nearest] <= distance*distance:
                result[i] = candidates[nearest]
                distances[i] = math.sqrt(float(squared[nearest]))
    return result, distances


def _face_directions(points, flat, sizes):
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    following = np.arange(len(flat))+1
    following[starts+sizes-1] = starts
    # Translation-relative Newell area handles polygons with collinear first corners.
    relative = points[flat]-np.repeat(points[flat[starts]], sizes, axis=0)
    area = np.add.reduceat(np.cross(relative, relative[following]), starts, axis=0)
    length = np.linalg.norm(area, axis=1)
    extent_squared = np.maximum.reduceat(np.sum(relative*relative, axis=1), starts)
    invalid = (extent_squared == 0) | (length <= 1e-12*extent_squared)
    direction = np.zeros_like(area)
    direction[~invalid] = area[~invalid]/length[~invalid, None]
    return direction, invalid


def _check_hemispheres(points, flat, loop_faces, output, selected_corners, face_geometry, name):
    direction, invalid = face_geometry
    corners = np.flatnonzero(selected_corners)
    faces = loop_faces[corners]
    invalid_corners = invalid[faces]
    dots = np.sum(output[corners]*direction[faces], axis=1)
    outside = (~invalid_corners) & (dots <= 0)
    bad = invalid_corners | outside
    witnesses = []
    for index in np.flatnonzero(bad)[:MAX_WITNESSES]:
        corner, face = int(corners[index]), int(faces[index])
        witnesses.append({'face': face, 'corner': corner, 'vertex': int(flat[corner]),
            'reason': 'degenerate_or_cancelling_face' if invalid_corners[index] else 'outside_face_hemisphere',
            'dot': None if invalid_corners[index] else float(dots[index])})
    report = {'schema_version': 1, 'status': 'rejected' if bad.any() else 'passed', 'name': name,
        'checked_faces': int(len(np.unique(faces))), 'checked_corners': int(len(corners)),
        'degenerate_faces': int(len(np.unique(faces[invalid_corners]))),
        'outside_hemisphere_corners': int(outside.sum()),
        'minimum_face_dot': float(dots[~invalid_corners].min()) if (~invalid_corners).any() else None,
        'witnesses': witnesses, 'witness_limit': MAX_WITNESSES,
        'position_sha256': array_hash(points, '<f8'),
        'face_corner_indices_sha256': array_hash(flat, '<i8'),
        'tested_corner_ids_sha256': array_hash(corners, '<i8'),
        'tested_normals_sha256': array_hash(output[corners], '<f8'),
        'policy': 'Transferred base-body corners require positive dot with their own Newell face normal; no flips or fallback.',
        'scope': 'Pre-mutation shading rejection, not a topology repair or appearance acceptance.'}
    if bad.any():
        raise NormalContinuityError(report)
    return report


def plan_normal_continuity(points, faces, face_materials, material_names, corner_normals, configs):
    """Return detached arrays and a JSON receipt; average original corners at exact positions."""
    configs = validate_normal_continuity(configs)
    points = _vectors(points, MAX_VERTICES, 'points')
    normals = _vectors(corner_normals, MAX_CORNERS, 'corner normals')
    _material_names(material_names)
    flat, sizes = _faces(faces, len(points))
    if len(normals) != len(flat):
        raise ValueError('Normal continuity corner-normal count mismatch')
    if isinstance(face_materials, np.ndarray):
        if face_materials.ndim != 1 or face_materials.dtype.kind not in 'iu':
            raise ValueError('Invalid normal continuity material indices')
    elif (not isinstance(face_materials, (list, tuple)) or not 1 <= len(face_materials) <= MAX_FACES
          or any(type(v) is not int for v in face_materials)):
        raise ValueError('Invalid normal continuity material indices')
    slots = np.asarray(face_materials, dtype=np.int64)
    if len(slots) != len(sizes) or slots.min() < 0 or slots.max() >= len(material_names):
        raise ValueError('Normal continuity material count or index mismatch')
    unit = _unit(normals, 'corner normals')
    if np.any(np.abs(np.linalg.norm(normals, axis=1)-1) > .001):
        raise ValueError('Original corner normals must be unit length')
    counts = np.bincount(flat, minlength=len(points))
    if not np.all(counts):
        raise ValueError('Normal continuity requires every vertex to have an original corner normal')
    vertex_original = np.zeros_like(points)
    np.add.at(vertex_original, flat, unit)
    cancelled = np.linalg.norm(vertex_original, axis=1) < 1e-12
    # Point attributes cannot represent opposing split corners; keep the first original corner.
    if cancelled.any():
        _, first = np.unique(flat, return_index=True)
        vertex_original[cancelled] = unit[first[cancelled]]
    vertex_original = _unit(vertex_original, 'averaged vertex normals')
    vertex_normals, output = vertex_original.copy(), normals.copy()
    repaired_vertices, repaired_faces = np.zeros(len(points), bool), np.zeros(len(sizes), bool)
    loop_faces = np.repeat(np.arange(len(sizes)), sizes)
    targets_seen, target_total, budget, reports = set(), 0, [0], []
    name_to_slot = {name: i for i, name in enumerate(material_names)}
    face_geometry = _face_directions(points, flat, sizes) if configs else None
    for config in configs:
        names = config['materials']+config['reference_materials']
        if any(name not in name_to_slot for name in names):
            raise ValueError('Normal continuity names an absent exact material')
        target_faces = np.isin(slots, [name_to_slot[name] for name in config['materials']])
        reference_faces = np.isin(slots, [name_to_slot[name] for name in config['reference_materials']])
        target_corners, reference_corners = target_faces[loop_faces], reference_faces[loop_faces]
        target_vertices = np.unique(flat[target_corners])
        if not reference_corners.any() or not len(target_vertices):
            raise ValueError('Normal continuity requires populated target and reference materials')
        target_total += len(target_vertices)
        if target_total > MAX_TARGET_VERTICES or targets_seen.intersection(map(int, target_vertices)):
            raise ValueError('Normal continuity exceeds 10000 target vertices or target vertex sets overlap')
        targets_seen.update(map(int, target_vertices))
        reference_points, inverse = np.unique(points[flat[reference_corners]], axis=0, return_inverse=True)
        reference_normals = np.zeros_like(reference_points)
        np.add.at(reference_normals, inverse, unit[reference_corners])
        reference_normals = _unit(reference_normals, 'reference normals at shared positions')
        nearest, distances = _nearest(points[target_vertices], reference_points, config['max_distance_m'], budget)
        matched = nearest >= 0
        selected = target_vertices[matched]
        if not config['min_vertices'] <= len(selected) <= config['max_vertices']:
            raise ValueError(f"Normal continuity {config['name']} matched {len(selected)} vertices outside its bounds")
        vertex_normals[selected] = reference_normals[nearest[matched]]
        repaired_vertices[selected] = True
        changed_corners = target_corners & np.isin(flat, selected)
        output[changed_corners] = vertex_normals[flat[changed_corners]]
        hemisphere = _check_hemispheres(points, flat, loop_faces, output, changed_corners,
                                       face_geometry, config['name'])
        changed_faces = np.unique(loop_faces[changed_corners])
        repaired_faces[changed_faces] = True
        angles = np.degrees(np.arccos(np.clip(np.sum(unit[changed_corners]*output[changed_corners], axis=1), -1, 1)))
        reports.append({'name': config['name'], 'materials': config['materials'],
            'reference_materials': config['reference_materials'], 'target_vertices': len(target_vertices),
            'matched_vertices': len(selected), 'unmatched_vertices': int((~matched).sum()),
            'reference_unique_positions': len(reference_points), 'reference_corners': int(reference_corners.sum()),
            'repaired_faces': len(changed_faces), 'transferred_corners': int(changed_corners.sum()),
            'maximum_distance_m': float(distances[matched].max()),
            'hemisphere_check': hemisphere,
            'angle_degrees': {'min': float(angles.min()), 'median': float(np.median(angles)), 'max': float(angles.max())},
            'matched_vertex_ids_sha256': array_hash(selected, '<i8'),
            'reference_position_sha256': array_hash(reference_points, '<f8'),
            'reference_normal_sha256': array_hash(reference_normals, '<f8')})
    report = {'schema_version': 1, 'status': 'planned' if configs else 'not_requested', 'corrections': reports,
        'repaired_vertices': int(repaired_vertices.sum()), 'repaired_faces': int(repaired_faces.sum()),
        'candidate_pairs_tested': budget[0], 'point_normal_cancellation_fallbacks': int(cancelled.sum()),
        'point_attribute': POINT_ATTRIBUTE, 'face_attribute': FACE_ATTRIBUTE,
        'position_sha256': array_hash(points, '<f8'), 'face_corner_indices_sha256': array_hash(flat, '<i8'),
        'original_corner_normals_sha256': array_hash(normals, '<f8'),
        'planned_corner_normals_sha256': array_hash(output, '<f8'),
        'authored_point_normals_sha256': array_hash(vertex_normals.astype(np.float32), '<f4'),
        'repair_face_mask_sha256': array_hash(repaired_faces, '?'),
        'method': 'Unweighted original reference-corner normals averaged at exact positions, then nearest-position transfer within each configured distance; target-face corners only.',
        'space': 'Object-local meters with applied rotation and scale',
        'scope': 'Authored shading and fur-direction metadata; no geometry, UV, weight or topology repair.',
        'limits': {'vertices': MAX_VERTICES, 'faces': MAX_FACES, 'corners': MAX_CORNERS,
                   'target_vertices': MAX_TARGET_VERTICES, 'candidate_pairs': MAX_PAIRS}}
    return {'corner_normals': output, 'vertex_normals': vertex_normals, 'repaired_vertices': repaired_vertices,
            'repaired_faces': repaired_faces, 'report': report}
