"""Subject-independent, ordered ellipsoid fields for native hair and portable cards.

Use evaluate_flow for one checked sample. For strand loops, call prepare_flow
once, then evaluate_prepared_flow per root; only the root is checked per sample.
Roots/bounds share one rest coordinate basis; directions use that same basis.
Fields are authored grooming controls, not recovered fur or hair simulation.

Schema: {regions: [1..12 regions]}. Each region has exactly name (unique slug),
center (three numbers -1..3), radius (three numbers >.001..3), feather (.05..1),
direction (nonzero three-vector -1..1), comb (0...95), length_scale (.25..2).
Example region: {"name": "upper-flow", "center": [.5, .5, .8],
"radius": [.3, .4, .2], "feather": .5, "direction": [0, 0, -1],
"comb": .65, "length_scale": 1.2}. No coordinates imply any subject anatomy.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
import re


MAX_REGIONS = 12
_FIELDS = {'name', 'center', 'radius', 'feather', 'direction', 'comb', 'length_scale'}


def _number(value, name, low=None, high=None):
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f'Invalid finite groom flow {name}')
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f'Invalid finite groom flow {name}') from error
    if (not math.isfinite(result) or (low is not None and result < low) or
            (high is not None and result > high)):
        raise ValueError(f'Invalid finite groom flow {name}')
    return result


def _vector(value, name, low=None, high=None):
    try:
        if len(value) != 3:
            raise ValueError(f'Groom flow {name} requires three coordinates')
        return tuple(_number(value[index], name, low, high) for index in range(3))
    except (TypeError, KeyError, IndexError) as error:
        raise ValueError(f'Groom flow {name} requires three numeric coordinates') from error


def _direction(value):
    direction = _vector(value, 'direction', -1, 1)
    length = math.hypot(*direction)
    if length <= 1e-8:
        raise ValueError('Groom flow direction cannot be zero or near zero')
    return tuple(component/length for component in direction)


def validate_flow(config):
    """Return a detached validated config; later regions blend over earlier ones."""
    if not isinstance(config, dict) or set(config) != {'regions'}:
        raise ValueError('Groom flow requires exactly regions')
    regions = config['regions']
    if not isinstance(regions, list) or not 1 <= len(regions) <= MAX_REGIONS:
        raise ValueError('Groom flow requires 1–12 regions')
    names = set()
    for region in regions:
        if not isinstance(region, dict) or set(region) != _FIELDS:
            raise ValueError('Invalid groom flow region fields')
        name = region['name']
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', name) or name in names:
            raise ValueError('Groom flow names must be unique safe slugs')
        names.add(name)
        for key in ('center', 'radius', 'direction'):
            if not isinstance(region[key], list):
                raise ValueError(f'Groom flow {key} must be a three-number list')
        _vector(region['center'], 'center', -1, 3)
        radii = _vector(region['radius'], 'radius', .001, 3)
        if min(radii) <= .001:
            raise ValueError('Groom flow radii must exceed .001')
        _number(region['feather'], 'feather', .05, 1)
        _direction(region['direction'])
        _number(region['comb'], 'comb', 0, .95)
        _number(region['length_scale'], 'length_scale', .25, 2)
    return deepcopy(config)


@dataclass(frozen=True)
class _Region:
    center: tuple
    inverse_radius: tuple
    feather: float
    direction: tuple
    comb: float
    length_scale: float


@dataclass(frozen=True)
class PreparedFlow:
    """Validated immutable coefficients; construct through prepare_flow only."""

    low: tuple
    inverse_span: tuple
    base_direction: tuple
    base_comb: float
    regions: tuple


def prepare_flow(low, span, base_direction, base_comb, config):
    """Validate once and precompute immutable coefficients for a bounded strand loop."""
    config = validate_flow(config)
    low, span = _vector(low, 'low'), _vector(span, 'span')
    if min(span) <= 0:
        raise ValueError('Groom flow bounds need positive spans')
    inverse_span = tuple(1/value for value in span)
    if not all(math.isfinite(value) for value in inverse_span):
        raise ValueError('Groom flow bounds exceed the finite numerical range')
    direction = _direction(base_direction)
    comb = _number(base_comb, 'base_comb', 0, .95)
    regions = tuple(_Region(tuple(region['center']), tuple(1/value for value in region['radius']),
                            float(region['feather']), _direction(region['direction']),
                            float(region['comb']), float(region['length_scale']))
                    for region in config['regions'])
    return PreparedFlow(low, inverse_span, direction, comb, regions)


def evaluate_prepared_flow(root, prepared):
    """Blend one finite root against prepare_flow output without revalidating regions.

    Influence is one inside each ellipsoid and smoothstep-fades to zero over
    feather units outside it. Length starts at one. Exact opposing cancellation
    keeps the previous direction while comb/length still blend normally.
    """
    if not isinstance(prepared, PreparedFlow):
        raise ValueError('Expected a prepared groom flow')
    root = _vector(root, 'root')
    point = tuple((value-low)*scale for value, low, scale in zip(root, prepared.low, prepared.inverse_span))
    if not all(math.isfinite(value) for value in point):
        raise ValueError('Groom root exceeds the finite normalized numerical range')
    direction, comb, length_scale = prepared.base_direction, prepared.base_comb, 1.
    for region in prepared.regions:
        distance = math.hypot(*((value-center)*scale for value, center, scale in
                                zip(point, region.center, region.inverse_radius)))
        if distance >= 1+region.feather:
            continue
        if distance <= 1:
            direction, comb, length_scale = region.direction, region.comb, region.length_scale
            continue
        fraction = (distance-1)/region.feather
        influence = 1-fraction*fraction*(3-2*fraction)
        blended = tuple((1-influence)*old+influence*new for old, new in zip(direction, region.direction))
        magnitude = math.hypot(*blended)
        if magnitude > 1e-8:
            direction = tuple(value/magnitude for value in blended)
        comb = (1-influence)*comb+influence*region.comb
        length_scale = (1-influence)*length_scale+influence*region.length_scale
    return direction, comb, length_scale


def evaluate_flow(root, low, span, base_direction, base_comb, config):
    """Checked one-shot API; use a prepared flow when sampling many strand roots."""
    return evaluate_prepared_flow(root, prepare_flow(low, span, base_direction, base_comb, config))
