"""Bounded authored bone fields; identical positions receive identical skin weights."""

import math
import re

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def vector(value):
    require(isinstance(value, list) and len(value) == 3
            and all(type(x) in (int, float) and math.isfinite(x) and -1 <= x <= 2 for x in value),
            'Rig coordinates must be three finite normalized values in -1..2')


def validate_rig(config):
    require(isinstance(config, dict) and set(config) == {'space', 'height_meters', 'bones', 'base_bones', 'layers'},
            'Invalid authored rig fields')
    require(config['space'] == 'normalized_bounds_blender_z_up', 'Unsupported authored rig coordinate space')
    require(type(config['height_meters']) in (int, float) and math.isfinite(config['height_meters'])
            and .05 <= config['height_meters'] <= 10, 'Authored rig height outside .05–10m')
    bones = config['bones']
    require(isinstance(bones, list) and 1 <= len(bones) <= 64, 'Authored rig requires 1–64 bones')
    names = []
    for bone in bones:
        require(isinstance(bone, dict) and set(bone) == {'name', 'parent', 'head', 'tail'}, 'Invalid authored bone')
        name = bone['name']
        require(isinstance(name, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', name)
                and name not in names, 'Invalid or duplicate authored bone name')
        require(bone['parent'] in names if names else bone['parent'] is None,
                'Bones need one root and parents before children')
        vector(bone['head']); vector(bone['tail'])
        require(math.dist(bone['head'], bone['tail']) > .001, 'Zero-length authored bone')
        names.append(name)
    def subset(value):
        require(isinstance(value, list) and 1 <= len(value) <= 64 and len(set(value)) == len(value)
                and all(name in names for name in value), 'Invalid weight bone subset')
    subset(config['base_bones'])
    require(isinstance(config['layers'], list) and len(config['layers']) <= 32, 'Too many weight layers')
    for layer in config['layers']:
        require(isinstance(layer, dict) and set(layer) == {'bones', 'min', 'max', 'feather', 'min_vertices'},
                'Invalid weight layer')
        subset(layer['bones']); vector(layer['min']); vector(layer['max'])
        require(all(a < b for a, b in zip(layer['min'], layer['max'])), 'Empty weight box')
        require(type(layer['feather']) in (int, float) and math.isfinite(layer['feather'])
                and 0 <= layer['feather'] <= .3, 'Invalid weight feather')
        require(type(layer['min_vertices']) is int and 1 <= layer['min_vertices'] <= 100000,
                'Invalid minimum region coverage')
    return config


def skin_weights(points, low, high, config):
    validate_rig(config)
    points, low, high = np.asarray(points, dtype=float), np.asarray(low), np.asarray(high)
    require(points.ndim == 2 and points.shape[1] == 3 and 1 <= len(points) <= 100000
            and np.isfinite(points).all() and np.all(high-low > 1e-8), 'Invalid bounded skin points')
    names = [b['name'] for b in config['bones']]
    span = high-low
    normalized = (points-low)/span
    heads = np.array([b['head'] for b in config['bones']])*span+low
    tails = np.array([b['tail'] for b in config['bones']])*span+low
    distance = np.empty((len(points), len(names)))
    for i, (head, tail) in enumerate(zip(heads, tails)):
        direction = tail-head
        fraction = np.clip(((points-head)@direction)/(direction@direction), 0, 1)
        distance[:, i] = np.linalg.norm(points-head-fraction[:, None]*direction, axis=1)
    def field(subset):
        indices = [names.index(name) for name in subset]
        selected = 1/np.maximum(distance[:, indices], .002*span.max())**4
        selected /= selected.sum(axis=1)[:, None]
        result = np.zeros_like(distance)
        result[:, indices] = selected
        return result
    weights, coverage = field(config['base_bones']), []
    for layer in config['layers']:
        edge = np.minimum(normalized-layer['min'], np.asarray(layer['max'])-normalized).min(axis=1)
        alpha = np.clip(edge/layer['feather'], 0, 1) if layer['feather'] else (edge >= 0).astype(float)
        alpha = alpha*alpha*(3-2*alpha)
        count = int((alpha > 0).sum())
        require(count >= layer['min_vertices'], 'Authored weight layer missed expected geometry: '+','.join(layer['bones']))
        weights = weights*(1-alpha[:, None])+field(layer['bones'])*alpha[:, None]
        coverage.append({'bones':layer['bones'], 'selected_vertices':count})
    # Anatomical chain locality: spatially overlapping envelopes can pair bones that
    # are far apart on the chain (a forearm resting beside a hip), and such blends
    # tear the moment the limb articulates. Each vertex keeps only bones within two
    # chain links of its dominant bone; the deformation gate still judges the result.
    index_of = {name: i for i, name in enumerate(names)}
    adjacent = np.eye(len(names), dtype=bool)
    for bone in config['bones']:
        if bone['parent'] is not None:
            i, j = index_of[bone['name']], index_of[bone['parent']]
            adjacent[i, j] = adjacent[j, i] = True
    within_two = adjacent @ adjacent
    dominant = weights.argmax(axis=1)
    allowed = within_two[dominant]
    chain_pruned = int(((weights > 0) & ~allowed).any(axis=1).sum())
    weights = np.where(allowed, weights, 0.)
    weights = weights/np.maximum(weights.sum(axis=1)[:, None], 1e-12)
    if len(names) > 4:
        top = np.argpartition(weights, -4, axis=1)[:, -4:]
        trimmed = np.zeros_like(weights)
        np.put_along_axis(trimmed, top, np.take_along_axis(weights, top, axis=1), axis=1)
        weights = trimmed
    weights /= weights.sum(axis=1)[:, None]
    require(np.isfinite(weights).all() and np.max(np.abs(weights.sum(axis=1)-1)) < 1e-6,
            'Invalid authored skin weights')
    return weights, {'method':'Authored spatial layers with nearest-segment blending and chain locality; not recovered anatomy',
                     'chain_localized_vertices':chain_pruned,
                     'max_influences':int((weights > 0).sum(axis=1).max()), 'layers':coverage,
                     'positive_vertices_per_bone':{name:int((weights[:,i] > 1e-6).sum()) for i,name in enumerate(names)}}
