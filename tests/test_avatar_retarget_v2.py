"""Explicit visual retargeting contracts; no simulator or authoring execution."""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import pytest

from weft import avatar_gltf as module

ROOT = Path(__file__).resolve().parents[1]


def builtin(name):
    folder = ROOT / 'tools/avatar-assets/outputs' / name
    mapping = json.loads((folder / 'mapping.json').read_bytes())['retargeting']
    mapping['profile'] = module.REST_RETARGET_PROFILE
    mapping['bindings'] = [dict(node=row['node'], body=row['body'].replace('left', 'TEMP').replace('right', 'left').replace('TEMP', 'right')) for row in mapping['bindings']]
    return folder, mapping


def normalize(name, mutate=None):
    folder, mapping = builtin(name)
    if mutate:
        mutate(mapping)
    return module.normalize_avatar_gltf('model.glb', {'model.glb': (folder / 'model.glb').read_bytes()}, mapping, deadline=time.monotonic() + 10)[0]


@pytest.mark.parametrize('name', ['weft-muse-panda-v1', 'weft-humanoid-robot-v1'])
def test_actual_builtin_explicit_rest_mapping(name):
    result = normalize(name)
    assert result['retargeting']['profile'] == module.REST_RETARGET_PROFILE
    assert len(result['retargeting']['bindings']) == 10
    assert result['counts']['stored_animations'] > 0
    assert 'not played' in result['scope']


@pytest.mark.parametrize('mutate', [
    lambda m: m['bindings'].pop(),
    lambda m: m['bindings'][1].update(body=m['bindings'][2]['body']),
    lambda m: m['bindings'][1].update(translation=[0, 0, 0]),
    lambda m: m['bindings'][1].update(body='guessed_wing'),
    lambda m: m.update(scale=0),
    lambda m: m.update(orientation_xyzw=[0, 0, 0, 1]),
    lambda m: [r.update(body=r['body'].replace('left', 'TEMP').replace('right', 'left').replace('TEMP', 'right')) for r in m['bindings']],
])
def test_incompatible_mapping_refused(mutate):
    with pytest.raises(module.AvatarAssetError):
        normalize('weft-humanoid-robot-v1', mutate)


def test_generic_rest_shape_and_bind_refusal():
    result = normalize('weft-humanoid-robot-v1')
    parents = [r['parent'] for r in result['nodes']]
    local = [np.array(r['local_matrix']).reshape(4, 4, order='F') for r in result['nodes']]
    skin = result['skins'][0]
    changed = deepcopy(local)
    changed[0][:3, 0] *= 2
    with pytest.raises(module.AvatarAssetError, match='uniform'):
        module._rest_relative(result['retargeting'], parents, changed, skin['joints'], skin['inverse_bind_matrices'])
    inverse = deepcopy(skin['inverse_bind_matrices'])
    inverse[0][12] += .1
    with pytest.raises(module.AvatarAssetError, match='bind pose'):
        module._rest_relative(result['retargeting'], parents, local, skin['joints'], inverse)


def test_mapping_does_not_change_original_document():
    folder, mapping = builtin('weft-muse-panda-v1')
    raw = (folder / 'model.glb').read_bytes()
    before = deepcopy(mapping)
    module.normalize_avatar_gltf('model.glb', {'model.glb': raw}, mapping, deadline=time.monotonic() + 10)
    assert mapping == before
    assert raw == (folder / 'model.glb').read_bytes()
