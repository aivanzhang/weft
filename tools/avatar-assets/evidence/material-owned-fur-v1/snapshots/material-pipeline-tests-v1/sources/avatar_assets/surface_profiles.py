"""Data-only bounds for texture-preserving material-response refinements."""

from copy import deepcopy
import math
import re


FIELDS = {'name', 'regions', 'brightness', 'max_chroma', 'roughness', 'metallic',
          'normal_strength', 'min_faces', 'max_faces'}


def _number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'Surface profile number outside {low}–{high}')


def validate_profiles(profiles):
    """Return a detached validated list; later matching profiles own overlap faces."""
    if not isinstance(profiles, list) or not 1 <= len(profiles) <= 8:
        raise ValueError('Surface profiles require 1–8 entries')
    names = set()
    for profile in profiles:
        if not isinstance(profile, dict) or set(profile) != FIELDS:
            raise ValueError('Invalid surface profile fields')
        name = profile['name']
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', name) or name in names:
            raise ValueError('Surface profile names must be unique safe slugs')
        names.add(name)
        regions = profile['regions']
        if not isinstance(regions, list) or len(regions) > 8:
            raise ValueError('Surface profiles support at most eight regions')
        for region in regions:
            if not isinstance(region, dict) or set(region) != {'center', 'radius'}:
                raise ValueError('Invalid surface region fields')
            for key in ('center', 'radius'):
                values = region[key]
                if not isinstance(values, list) or len(values) != 3:
                    raise ValueError('Surface regions require three coordinates')
                for value in values:
                    _number(value, .001 if key == 'radius' else -1, 3)
        brightness = profile['brightness']
        if not isinstance(brightness, list) or len(brightness) != 2:
            raise ValueError('Surface brightness requires a lower and upper bound')
        for value in brightness:
            _number(value, 0, 1.01)
        if brightness[0] >= brightness[1]:
            raise ValueError('Surface brightness range must increase')
        for key in ('max_chroma', 'roughness', 'metallic', 'normal_strength'):
            _number(profile[key], .04 if key == 'roughness' else 0, 1)
        if (type(profile['min_faces']) is not int or type(profile['max_faces']) is not int or
                not 1 <= profile['min_faces'] <= profile['max_faces'] <= 100000):
            raise ValueError('Surface profile needs ordered face bounds within 1–100000')
    return deepcopy(profiles)
