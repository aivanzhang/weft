"""Bounded spatial/color masks for authored constant-color material corrections."""

from copy import deepcopy
import math
import re

import numpy as np


MAX_VERTICES = 100000
MAX_FACES = 100000
MAX_CORNERS = 300000
FIELDS = {'name', 'regions', 'dominant_channel', 'dominant_ratio', 'color_srgb',
          'roughness', 'min_faces', 'max_faces'}


def _number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'Color correction number outside {low}–{high}')


def validate_corrections(corrections):
    """Return detached data; all face bounds apply after last-match precedence."""
    if not isinstance(corrections, list) or not 1 <= len(corrections) <= 4:
        raise ValueError('Color corrections require 1–4 entries')
    names = set()
    for correction in corrections:
        if not isinstance(correction, dict) or set(correction) != FIELDS:
            raise ValueError('Invalid color correction fields')
        name = correction['name']
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', name) or name in names:
            raise ValueError('Color correction names must be unique safe slugs')
        names.add(name)
        regions = correction['regions']
        if not isinstance(regions, list) or not 1 <= len(regions) <= 4:
            raise ValueError('Color corrections require 1–4 spatial regions')
        for region in regions:
            if not isinstance(region, dict) or set(region) != {'center', 'radius'}:
                raise ValueError('Invalid color correction region fields')
            for key, minimum in (('center', -1), ('radius', .001)):
                values = region[key]
                if not isinstance(values, list) or len(values) != 3:
                    raise ValueError('Color correction regions require three coordinates')
                for value in values:
                    _number(value, minimum, 3)
        if type(correction['dominant_channel']) is not int or correction['dominant_channel'] not in (0, 1, 2):
            raise ValueError('Color correction dominant channel must be 0, 1 or 2')
        _number(correction['dominant_ratio'], 1.05, 5)
        color = correction['color_srgb']
        if not isinstance(color, list) or len(color) != 3:
            raise ValueError('Color correction needs three sRGB channels')
        for value in color:
            _number(value, 0, 1)
        _number(correction['roughness'], .04, 1)
        if (type(correction['min_faces']) is not int or type(correction['max_faces']) is not int or
                not 1 <= correction['min_faces'] <= correction['max_faces'] <= MAX_FACES):
            raise ValueError('Color correction needs ordered face bounds within 1–100000')
    return deepcopy(corrections)


def srgb_to_linear(color):
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        raise ValueError('sRGB conversion requires three channels')
    for value in color:
        _number(value, 0, 1)
    return tuple(value/12.92 if value <= .04045 else ((value+.055)/1.055)**2.4 for value in color)


def _points(value, name):
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape[1] != 3 or not 1 <= len(value) <= MAX_VERTICES or value.dtype.kind not in 'fiu':
            raise ValueError(f'Invalid or oversized color correction {name}')
    else:
        if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_VERTICES:
            raise ValueError(f'Invalid or oversized color correction {name}')
        if any(not isinstance(row, (list, tuple)) or len(row) != 3 or
               any(type(v) not in (int, float) for v in row) for row in value):
            raise ValueError(f'Invalid color correction {name} row')
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f'Color correction {name} must be finite')
    return result


def _vector(value):
    if isinstance(value, np.ndarray):
        if value.shape != (3,) or value.dtype.kind not in 'fiu':
            raise ValueError('Invalid color correction bounds')
    elif not isinstance(value, (list, tuple)) or len(value) != 3 or any(type(v) not in (int, float) for v in value):
        raise ValueError('Invalid color correction bounds')
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Color correction bounds must be finite')
    return result


def _face_indices(faces, vertex_count):
    if isinstance(faces, np.ndarray):
        if (faces.ndim != 2 or not 1 <= len(faces) <= MAX_FACES or not 3 <= faces.shape[1] <= 64
                or faces.size > MAX_CORNERS or faces.dtype.kind not in 'iu'):
            raise ValueError('Invalid or oversized color correction face array')
        sizes, flat = np.full(len(faces), faces.shape[1], dtype=np.int64), faces.reshape(-1)
    else:
        if not isinstance(faces, (list, tuple)) or not 1 <= len(faces) <= MAX_FACES:
            raise ValueError('Invalid or oversized color correction faces')
        sizes, total = [], 0
        for face in faces:
            if not isinstance(face, (list, tuple)) or not 3 <= len(face) <= 64 or any(
                    type(i) is not int or not 0 <= i < vertex_count for i in face):
                raise ValueError('Color correction faces need bounded integer indices')
            total += len(face)
            if total > MAX_CORNERS:
                raise ValueError('Color correction faces exceed 300000 corners')
            sizes.append(len(face))
        flat = np.asarray([index for face in faces for index in face], dtype=np.int64)
        sizes = np.asarray(sizes, dtype=np.int64)
    if flat.min() < 0 or flat.max() >= vertex_count:
        raise ValueError('Color correction face index is out of bounds')
    return flat, sizes, np.concatenate(([0], np.cumsum(sizes[:-1])))


def select_color_faces(faces, colors, points, low, span, corrections):
    """A face must meet both the spatial centroid and mean-RGB dominance tests."""
    corrections = validate_corrections(corrections)
    points, colors = _points(points, 'points'), _points(colors, 'colors')
    if points.shape != colors.shape or np.any(colors < 0) or np.any(colors > 1):
        raise ValueError('Source RGB colors must match vertices and stay within 0–1')
    low, span = _vector(low), _vector(span)
    if np.any(span < 0) or not np.any(span > 0):
        raise ValueError('Invalid color correction normalization bounds')
    flat, sizes, starts = _face_indices(faces, len(points))
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            centroids = np.add.reduceat(points[flat], starts, axis=0)/sizes[:, None]
            rgb = np.add.reduceat(colors[flat], starts, axis=0)/sizes[:, None]
            normalized = (centroids-low)/np.maximum(span, 1e-8)
    except FloatingPointError as error:
        raise ValueError('Color correction geometry exceeds finite numerical range') from error
    owners, reports = np.full(len(sizes), -1, dtype=np.int16), []
    for index, correction in enumerate(corrections):
        inside = np.zeros(len(sizes), dtype=bool)
        for region in correction['regions']:
            try:
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    inside |= np.sum(((normalized-region['center'])/region['radius'])**2, axis=1) <= 1
            except FloatingPointError as error:
                raise ValueError('Color correction region exceeds finite numerical range') from error
        channel = correction['dominant_channel']
        others = np.max(rgb[:, [i for i in range(3) if i != channel]], axis=1)
        selected = inside & (rgb[:, channel] > correction['dominant_ratio']*others)
        owners[selected] = index
        reports.append({'name': correction['name'], 'matched_faces': int(selected.sum())})
    for index, correction in enumerate(corrections):
        count = int(np.count_nonzero(owners == index))
        if not correction['min_faces'] <= count <= correction['max_faces']:
            raise ValueError(f"Color correction {correction['name']} selected {count} final faces outside its bounds")
        reports[index]['selected_faces'] = count
    return owners, reports
