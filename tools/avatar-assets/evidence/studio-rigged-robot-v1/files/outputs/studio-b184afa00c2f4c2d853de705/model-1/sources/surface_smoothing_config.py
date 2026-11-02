"""Strict bounds for authored, color-independent surface refinement."""

from copy import deepcopy
import math


def _number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'Surface smoothing number outside {low}–{high}')


def validate_surface_smoothing(config):
    fields = {'iterations', 'strength', 'max_displacement_ratio', 'feather', 'exclusions'}
    if not isinstance(config, dict) or not fields <= set(config) <= fields|{'method'}:
        raise ValueError('Invalid surface smoothing fields')
    if config.get('method', 'taubin') not in ('taubin', 'laplacian'):
        raise ValueError('Surface smoothing method must be taubin or laplacian')
    if type(config['iterations']) is not int or not 1 <= config['iterations'] <= 20:
        raise ValueError('Surface smoothing requires 1–20 iterations')
    for key, low, high in (('strength', .01, .5), ('max_displacement_ratio', .001, .05),
                           ('feather', .05, 1)):
        _number(config[key], low, high)
    exclusions = config['exclusions']
    if not isinstance(exclusions, list) or len(exclusions) > 16:
        raise ValueError('Surface smoothing supports at most 16 exclusions')
    for region in exclusions:
        if not isinstance(region, dict) or set(region) != {'center', 'radius'}:
            raise ValueError('Invalid smoothing exclusion fields')
        for key in ('center', 'radius'):
            values = region[key]
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError('Smoothing exclusions require three coordinates')
            for value in values:
                _number(value, .001 if key == 'radius' else -1, 3)
    return deepcopy(config)
