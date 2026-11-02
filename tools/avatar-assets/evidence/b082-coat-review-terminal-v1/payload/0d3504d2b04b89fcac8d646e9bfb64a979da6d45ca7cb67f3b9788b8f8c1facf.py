"""Material-scoped authored skin fields using unchanged full-mesh coordinates."""

from copy import deepcopy
import re

import numpy as np

try:
    from creature_rig_math import validate_rig, skin_weights
except ModuleNotFoundError as error:
    if error.name != 'creature_rig_math':
        raise
    from scripts.creature_rig_math import validate_rig, skin_weights


MAX_VERTICES = 100000
FIELDS = {'name', 'materials', 'root_bone', 'layers', 'min_vertices', 'max_vertices'}


def _derived_rig(part, authored_rig):
    included, bones = set(), []
    for original in authored_rig['bones']:
        if original['name'] == part['root_bone'] or original['parent'] in included:
            bone = deepcopy(original)
            if not bones:
                bone['parent'] = None
            bones.append(bone)
            included.add(bone['name'])
    if not bones or bones[0]['name'] != part['root_bone']:
        raise ValueError('Part root bone is absent from the authored rig')
    derived = {'space': authored_rig['space'], 'height_meters': authored_rig['height_meters'],
               'bones': bones, 'base_bones': [part['root_bone']], 'layers': deepcopy(part['layers'])}
    validate_rig(derived)
    return derived


def validate_part_rigs(parts, authored_rig):
    """Validate exact material ownership and descendant-only controls; return a copy."""
    validate_rig(authored_rig)
    if not isinstance(parts, list) or not 1 <= len(parts) <= 4:
        raise ValueError('Part rigs require 1–4 entries')
    names, ownership = set(), set()
    for part in parts:
        if not isinstance(part, dict) or set(part) != FIELDS:
            raise ValueError('Invalid part rig fields')
        name = part['name']
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', name) or name in names:
            raise ValueError('Part names must be unique safe slugs')
        names.add(name)
        materials = part['materials']
        if not isinstance(materials, list) or not 1 <= len(materials) <= 8:
            raise ValueError('Each part requires 1–8 exact material names')
        for material in materials:
            if (not isinstance(material, str) or not 1 <= len(material) <= 256 or
                    any(ord(character) < 32 for character in material) or material in ownership):
                raise ValueError('Material names must be bounded and uniquely owned by one part')
            ownership.add(material)
        if not isinstance(part['root_bone'], str) or not 1 <= len(part['root_bone']) <= 64:
            raise ValueError('Invalid part root bone name')
        if (type(part['min_vertices']) is not int or type(part['max_vertices']) is not int or
                not 1 <= part['min_vertices'] <= part['max_vertices'] <= MAX_VERTICES):
            raise ValueError('Part vertex bounds must be ordered within 1–100000')
        _derived_rig(part, authored_rig)
    return deepcopy(parts)


def part_rig_config(part, authored_rig):
    """Detach the subtree for weighting only; never alter the real rig hierarchy."""
    part = validate_part_rigs([part], authored_rig)[0]
    return _derived_rig(part, authored_rig)


def validate_part_points(points, low, high):
    """Reject oversized data before conversion; bounds are the ORIGINAL whole mesh."""
    if isinstance(points, np.ndarray):
        if (points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= MAX_VERTICES or
                points.dtype.kind not in 'fiu'):
            raise ValueError('Invalid or oversized part points')
    else:
        if not isinstance(points, (list, tuple)) or not 1 <= len(points) <= MAX_VERTICES:
            raise ValueError('Invalid or oversized part points')
        for row in points:
            if not isinstance(row, (list, tuple)) or len(row) != 3 or any(
                    type(value) not in (int, float) for value in row):
                raise ValueError('Part points require finite nonboolean three-number rows')
    def bounds(value):
        if isinstance(value, np.ndarray):
            if value.shape != (3,) or value.dtype.kind not in 'fiu':
                raise ValueError('Invalid original whole-mesh bounds')
        elif not isinstance(value, (list, tuple)) or len(value) != 3 or any(
                type(number) not in (int, float) for number in value):
            raise ValueError('Invalid original whole-mesh bounds')
        return np.asarray(value, dtype=float)
    points, low, high = np.asarray(points, dtype=float), bounds(low), bounds(high)
    if not (np.isfinite(points).all() and np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError('Part points and whole-mesh bounds must be finite')
    with np.errstate(over='ignore', invalid='ignore'):
        span = high-low
    if not np.isfinite(span).all() or np.any(span <= 1e-8):
        raise ValueError('Part weighting requires nondegenerate original whole-mesh bounds')
    tolerance = float(span.max())*1e-7
    if np.any(points < low-tolerance) or np.any(points > high+tolerance):
        raise ValueError('Selected part points fall outside the original whole-mesh bounds')
    return points, low, high


def part_skin_weights(points, low, high, part, authored_rig):
    """Use the existing spatial field, without renormalizing a part to its own bounds."""
    derived = part_rig_config(part, authored_rig)
    points, low, high = validate_part_points(points, low, high)
    if not part['min_vertices'] <= len(points) <= part['max_vertices']:
        raise ValueError('Selected part vertex count is outside its configured bounds')
    weights, proof = skin_weights(points, low, high, derived)
    names = [bone['name'] for bone in derived['bones']]
    required = {name for layer in part['layers'] for name in layer['bones']}
    absent = sorted(name for name in required if not proof['positive_vertices_per_bone'][name])
    if absent:
        raise ValueError('Configured part layer controls have no final weight coverage: '+', '.join(absent))
    positive = weights[weights > 0]
    proof.update(name=part['name'], root_bone=part['root_bone'], bone_names=names,
                 selected_vertices=len(points), min_positive_weight=float(positive.min()),
                 max_weight=float(positive.max()), weight_sum_max_error=float(np.max(np.abs(weights.sum(axis=1)-1))),
                 original_mesh_bounds={'min': low.tolist(), 'max': high.tolist()},
                 coverage_threshold=1e-6, actual_rig_reparented=False,
                 scope='Descendant-only material-part weights in original normalized bounds; not inferred anatomy or motion acceptance')
    return weights, proof
