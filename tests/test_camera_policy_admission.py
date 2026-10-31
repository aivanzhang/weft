"""Reject unqualified camera policies before native validation or action construction."""

from types import SimpleNamespace

import pytest

from weft import shared_world_learning as learning


@pytest.mark.parametrize('entrypoint', ['validation', 'action'])
def test_camera_policy_rejects_before_native_or_legacy_controller(monkeypatch, entrypoint):
    def forbidden(*_args, **_kwargs):
        pytest.fail('Camera policy rejection must not compile a model or construct a legacy action')

    monkeypatch.setattr(learning, 'validate_shared_world', forbidden)
    monkeypatch.setattr(learning, '_action_from_weights', forbidden)
    scene = {'profile': 'shared-world-v6'}
    with pytest.raises(ValueError, match='separately qualified'):
        if entrypoint == 'validation':
            learning.validate_shared_policy({}, scene)
        else:
            learning.policy_action({}, SimpleNamespace(scene=scene))
