"""Compact recipe replay checks use public-feedback stubs, never native models."""

import ast
from bisect import bisect_right
import builtins
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import socket
import subprocess

import numpy as np
import pytest

from weft import shared_exercise_reference_controller as controller
from weft import shared_exercise_reference_recording as recording
from test_shared_exercise_reference_controller import controller_fixture, encode, prepare, FakeEnvironment


@pytest.fixture(autouse=True)
def no_native_media_or_processes(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'mujoco', 'gymnasium', 'cv2', 'PIL', 'mediapipe', 'torch'}
        return original(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('No native, process or network execution belongs in these pure checks')

    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)


@pytest.fixture(scope='module')
def velocity_index():
    path = Path(controller.__file__).parent / 'human.py'
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == controller.RUNTIME_SHA256['human.py']
    definition = next(row for row in ast.parse(raw).body
                      if isinstance(row, ast.FunctionDef) and row.name == 'reference_velocity_index')
    scope = {'math': math, 'bisect_right': bisect_right}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), scope)
    return scope['reference_velocity_index']


@pytest.fixture
def case(monkeypatch, velocity_index):
    monkeypatch.setattr(recording, '_trusted_velocity_index', lambda: velocity_index)

    def build(root_assistance=True, corrections=(0., .04, .01, .03)):
        bundle = controller_fixture(root_assistance=root_assistance, corrections=corrections)
        provider = prepare(bundle, velocity_index)
        raw = encode(provider.recipe())
        envelope = recording.reference_controller_envelope(raw, bundle['scene'], bundle['task'])
        env = FakeEnvironment(bundle, provider, .05)
        env.scene, env.task = deepcopy(bundle['scene']), deepcopy(bundle['task'])
        return bundle, provider, raw, envelope, env

    return build


def test_exact_recipe_envelope_roundtrip_and_detached_values(case):
    bundle, provider, raw, envelope, _ = case()
    assert set(envelope) == {'document_utf8', 'sha256', 'implementation_sha256'}
    assert envelope['document_utf8'].encode('utf-8') == raw
    assert envelope['sha256'] == sha256(raw).hexdigest()
    assert set(envelope['implementation_sha256']) == set(recording.IMPLEMENTATION_FILES)
    for name, digest in envelope['implementation_sha256'].items():
        assert digest == sha256((Path(recording.__file__).parent / name).read_bytes()).hexdigest()
    validated = recording.validate_reference_controller_envelope(envelope, bundle['scene'], bundle['task'])
    assert validated == envelope and validated is not envelope
    validated['implementation_sha256']['human.py'] = 'f' * 64
    assert validated != envelope
    recipe = json.loads(raw)
    assert recipe['interpretation'] == 'archived_execution_only_not_source_fit_admission'
    assert recipe['proposal_report_sha256'] == provider.describe()['proposal_report_sha256']
    assert len(recipe['targets']) == 4
    assert not {'original_human', 'candidate_human', 'proposal_report'} & set(recipe)


@pytest.mark.parametrize('root_assistance', [False, True])
@pytest.mark.parametrize('corrections', [(0.,) * 4, (0., .04, .01, .03)])
def test_recorded_callback_matches_provider_and_preserves_root_channels(case, root_assistance, corrections):
    bundle, provider, _, envelope, env = case(root_assistance, corrections)
    env.features[:] = np.arange(env.features.size).reshape(env.features.shape) * .01 - .2
    before = deepcopy((bundle, envelope)); features = env.features.copy()
    expected = provider.action(env)
    action = recording.prepare_recorded_reference_controller(envelope, env)
    actual = action()
    assert actual.tobytes() == expected.tobytes()
    assert env.features.tobytes() == features.tobytes() and (bundle, envelope) == before
    assert actual[19:].tobytes() == np.clip(features.sum(axis=1), -1., 1.)[19:].tobytes()


def test_prepared_action_does_not_reparse_rehash_or_read_files(case, monkeypatch):
    _, provider, _, envelope, env = case()
    expected = provider.action(env)
    action = recording.prepare_recorded_reference_controller(envelope, env)

    def forbidden(*args, **kwargs):
        raise AssertionError('No repeated recipe or filesystem work on the action path')

    monkeypatch.setattr(recording, '_read_implementation', forbidden)
    monkeypatch.setattr(controller.json, 'loads', forbidden)
    monkeypatch.setattr(controller, 'canonical_hash', forbidden)
    envelope['document_utf8'] = 'mutated caller object'
    assert action().tobytes() == expected.tobytes()
    assert action().tobytes() == expected.tobytes()


@pytest.mark.parametrize('kind', ['unknown', 'missing', 'not_dict', 'null', 'text_type', 'hash_type', 'hash_mismatch',
                                 'surrogate', 'map_missing', 'map_unknown', 'map_type', 'map_hash_format'])
def test_malformed_envelope_is_rejected(case, kind):
    bundle, _, _, envelope, _ = case()
    if kind == 'unknown': envelope['unknown'] = False
    elif kind == 'missing': del envelope['sha256']
    elif kind == 'not_dict': envelope = []
    elif kind == 'null': envelope = None
    elif kind == 'text_type': envelope['document_utf8'] = b'{}'
    elif kind == 'hash_type': envelope['sha256'] = True
    elif kind == 'hash_mismatch': envelope['document_utf8'] += ' '
    elif kind == 'surrogate': envelope['document_utf8'] = '\ud800'
    elif kind == 'map_missing': del envelope['implementation_sha256']['human.py']
    elif kind == 'map_unknown': envelope['implementation_sha256']['unknown.py'] = 'a' * 64
    elif kind == 'map_type': envelope['implementation_sha256'] = []
    elif kind == 'map_hash_format': envelope['implementation_sha256']['human.py'] = 'not-a-hash'
    with pytest.raises(ValueError):
        recording.validate_reference_controller_envelope(envelope, bundle['scene'], bundle['task'])


@pytest.mark.parametrize('name', recording.IMPLEMENTATION_FILES)
def test_each_recorded_runtime_pin_is_checked_before_callback_loading(case, monkeypatch, name):
    bundle, _, _, envelope, _ = case()
    envelope['implementation_sha256'][name] = 'f' * 64
    monkeypatch.setattr(recording, '_trusted_velocity_index', lambda: pytest.fail('Loaded callback before pins matched'))
    with pytest.raises(ValueError, match='implementation pins differ'):
        recording.validate_reference_controller_envelope(envelope, bundle['scene'], bundle['task'])


@pytest.mark.parametrize('name', ['human.py', 'shared_world.py'])
def test_new_envelopes_cannot_pin_unreviewed_feedback_runtime(case, monkeypatch, name):
    bundle, _, raw, _, _ = case()
    original = recording._read_implementation
    monkeypatch.setattr(recording, '_read_implementation', lambda p: original(p) + (b'\n' if p.name == name else b''))
    monkeypatch.setattr(recording, '_trusted_velocity_index', lambda: pytest.fail('Loaded unsupported callback'))
    with pytest.raises(ValueError, match='unsupported human'):
        recording.reference_controller_envelope(raw, bundle['scene'], bundle['task'])


@pytest.mark.parametrize('kind', ['scene', 'task', 'action'])
def test_environment_binding_mismatch_rejected_at_preparation(case, kind):
    _, _, _, envelope, env = case()
    if kind == 'scene': env.scene['human_spec']['lengths']['torso'] = .41
    elif kind == 'task': env.task['gates_sha256'] = 'f' * 64
    else: env.contract['names'][0] = 'unknown'
    with pytest.raises(ValueError): recording.prepare_recorded_reference_controller(envelope, env)


@pytest.mark.parametrize('raw', [b'', bytearray(b'{}'), b'\xff', b'[]', b'{"x":1,"x":2}', b'{"x":NaN}', b'{'])
def test_recipe_requires_finite_unique_utf8_object(case, raw):
    bundle, _, _, _, _ = case()
    with pytest.raises(ValueError): recording.reference_controller_envelope(raw, bundle['scene'], bundle['task'])


@pytest.mark.parametrize('kind', ['unknown_field', 'wrong_interpretation', 'scene', 'task', 'candidate', 'first_target'])
def test_rehashed_tampered_recipe_cannot_bypass_provider(case, kind):
    bundle, _, raw, envelope, _ = case()
    recipe = json.loads(raw)
    if kind == 'unknown_field': recipe['unknown'] = False
    elif kind == 'wrong_interpretation': recipe['interpretation'] = 'new_source_fit_admission'
    elif kind == 'first_target': recipe['targets'][0]['joints']['left_knee'] += .01
    else: recipe[{'scene': 'scene_sha256', 'task': 'task_sha256', 'candidate': 'candidate_canonical_sha256'}[kind]] = 'f' * 64
    changed = encode(recipe); envelope['document_utf8'] = changed.decode(); envelope['sha256'] = sha256(changed).hexdigest()
    with pytest.raises(ValueError): recording.validate_reference_controller_envelope(envelope, bundle['scene'], bundle['task'])


def test_exact_document_byte_limit_and_utf8_expansion(case):
    bundle, _, raw, _, _ = case()
    maximum = recording.MAX_DOCUMENT_BYTES
    padded = raw + b' ' * (maximum - len(raw))
    envelope = recording.reference_controller_envelope(padded, bundle['scene'], bundle['task'])
    assert len(envelope['document_utf8'].encode()) == maximum
    with pytest.raises(ValueError, match='at most 1 MiB'):
        recording.reference_controller_envelope(padded + b' ', bundle['scene'], bundle['task'])
    expanded = '\u00e9' * (maximum // 2 + 1)
    envelope['document_utf8'] = expanded; envelope['sha256'] = sha256(expanded.encode()).hexdigest()
    with pytest.raises(ValueError, match='byte bound'):
        recording.validate_reference_controller_envelope(envelope, bundle['scene'], bundle['task'])


def test_implementation_reads_reject_symlinks_and_oversized_sources(tmp_path, monkeypatch):
    source = tmp_path / 'source.py'; source.write_bytes(b'12345')
    link = tmp_path / 'link.py'; link.symlink_to(source)
    with pytest.raises(OSError): recording._read_implementation(link)
    monkeypatch.setattr(recording, 'MAX_IMPLEMENTATION_BYTES', 4)
    with pytest.raises(ValueError, match='bounded regular'): recording._read_implementation(source)


def test_legacy_controller_methods_are_not_called_or_replaced(case):
    _, _, _, envelope, env = case(corrections=(0.,) * 4)

    def legacy():
        return 'legacy method remains owned by the existing environment'

    env.authored_action = legacy
    env.step = env.reset = lambda *a, **k: pytest.fail('Helper must not control an environment')
    action = recording.prepare_recorded_reference_controller(envelope, env)
    assert np.array_equal(action(), np.zeros(22))
    assert env.authored_action is legacy and env.authored_action().startswith('legacy method')
