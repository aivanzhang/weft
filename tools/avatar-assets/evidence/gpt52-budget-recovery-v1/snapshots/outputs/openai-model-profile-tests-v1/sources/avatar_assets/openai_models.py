"""Pinned standard-tier prices; full-context reservations never estimate input tokens."""
from copy import deepcopy
from fractions import Fraction
import math

from avatar_assets.safety import require

MODEL = 'gpt-5.4-2026-03-05'
GPT52 = 'gpt-5.2-2025-12-11'
_PROFILES = {
    MODEL: {'model': MODEL, 'checked': '2026-11-02',
        'source': 'https://developers.openai.com/api/docs/models/gpt-5.4',
        'context_tokens': 1050000, 'long_context_threshold': 272000,
        'input_microdollars_per_token': 2.5, 'output_microdollars_per_token': 15,
        'long_input_multiplier': 2, 'long_output_multiplier': 1.5},
    GPT52: {'model': GPT52, 'checked': '2026-11-02',
        'source': 'https://developers.openai.com/api/docs/models/gpt-5.2',
        'context_tokens': 400000, 'long_context_threshold': None,
        'input_microdollars_per_token': 1.75, 'output_microdollars_per_token': 14,
        'long_input_multiplier': 1, 'long_output_multiplier': 1},
}


def price_profile(model=MODEL):
    require(isinstance(model, str) and model in _PROFILES, 'Model lacks an approved pinned price profile')
    return deepcopy(_PROFILES[model])


PRICING = price_profile()


def usage_cost_microdollars(model, inp, output):
    profile = price_profile(model)
    require(type(inp) is int and 0 <= inp <= profile['context_tokens'], 'Invalid priced input tokens')
    require(type(output) is int and 0 <= output <= 32768, 'Invalid priced output tokens')
    threshold = profile['long_context_threshold']
    long = threshold is not None and inp > threshold
    input_rate = Fraction(str(profile['input_microdollars_per_token']))
    output_rate = Fraction(str(profile['output_microdollars_per_token']))
    if long:
        input_rate *= Fraction(str(profile['long_input_multiplier']))
        output_rate *= Fraction(str(profile['long_output_multiplier']))
    return math.ceil(inp * input_rate + output * output_rate)


def reserve_microdollars(max_output_tokens=16384, model=MODEL):
    require(type(max_output_tokens) is int and 256 <= max_output_tokens <= 32768, 'Invalid output token ceiling')
    profile = price_profile(model)
    return usage_cost_microdollars(model, profile['context_tokens'], max_output_tokens)
