"""Opt-in edge-fringe settings; technical admission is not appearance approval."""

from copy import deepcopy
import math


def validate_fringe(value):
    from scripts.edge_fringe import validate_settings

    if (not isinstance(value, dict)
            or set(value) != {'profile', 'settings', 'design_views', 'excluded_materials'}
            or value['profile'] != 'edge-fringe-v1'):
        raise ValueError('Invalid portable fringe config')
    validate_settings(value['settings'])
    views = value['design_views']
    if not isinstance(views, list) or not 1 <= len(views) <= 16:
        raise ValueError('Fringe needs 1..16 fixed object-space design views')
    for vector in views:
        if (not isinstance(vector, list) or len(vector) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1 for v in vector)
                or math.hypot(*vector) <= 1e-8):
            raise ValueError('Invalid portable fringe design view')
    excluded = value['excluded_materials']
    if (not isinstance(excluded, list) or len(excluded) > 32
            or any(not isinstance(name, str) or not 1 <= len(name) <= 128 for name in excluded)
            or len(set(excluded)) != len(excluded)):
        raise ValueError('Invalid portable fringe excluded materials')
    return deepcopy(value)
