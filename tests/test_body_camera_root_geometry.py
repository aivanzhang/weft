"""Analytical extraction/diagnostic fixtures; no native model is constructed."""

from copy import deepcopy
from hashlib import sha256
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from weft import body_camera_root_geometry as geometry
from weft.io import canonical_hash
from test_whole_body_reference_refinement import fixture, encode


RUNTIME = {'implementation_sha256': {'human.py': 'b' * 64},
           'versions': {'python': 'fixture', 'numpy': 'fixture', 'mujoco': 'fixture', 'gymnasium': 'fixture'}}


class AnalyticGeometry:
    support_names = ('sphere:0.0', 'capsule:-1.0', 'capsule:1.0')

    def __init__(self, *, corruption=0.):
        self.floor, self.closed, self.calls, self.corruption = 0, False, 0, corruption
        self.env = SimpleNamespace(_site_ids={'pelvis': 0})
        rotation = np.array([[.8, 0., .6], [0., 1., 0.], [-.6, 0., .8]])
        self.data = SimpleNamespace(site_xpos=np.zeros((1, 3)), geom_xmat=np.array([rotation.ravel()]),
                                    geom_xpos=np.zeros((1, 3)))
        self.relative = np.arange(36).reshape(12, 3) / 100

    def evaluate(self, q, root, derivatives=False):
        assert derivatives is False
        self.calls += 1
        pelvis = np.array(root) + [.2, -.3, 2.]
        self.data.site_xpos[0] = pelvis
        normal = self.data.geom_xmat[0].reshape(3, 3)[:, 2]
        centers = pelvis + [[0., 0., .1], [0., 0., -.5], [0., 0., .5]]
        gaps = (centers - self.data.geom_xpos[0]) @ normal - [.05, .02, .02] + self.corruption
        return {'points': self.relative.copy(), 'gaps': gaps}

    def close(self):
        self.closed = True


def setup_geometry(monkeypatch, *, corrupt_final=False):
    engines = []
    def make(spec):
        instance = AnalyticGeometry(corruption=.01 if engines and corrupt_final else 0.)
        engines.append(instance)
        return instance
    monkeypatch.setattr(geometry, '_make_geometry', make)
    monkeypatch.setattr(geometry, '_xml', lambda spec: b'<unchanged-original-model/>')
    return engines


def extract(monkeypatch, **kwargs):
    engines = setup_geometry(monkeypatch, **kwargs)
    spec, evidence = fixture()
    spec['mobility']['ground_plane'] = {'normal': [.6, 0., .8]}
    human_raw, evidence_raw = encode(spec), encode(evidence)
    raw = geometry.extract_original_geometry(human_raw, evidence_raw, runtime=RUNTIME, deadline=time.monotonic() + 2)
    return engines, spec, evidence, human_raw, evidence_raw, raw


def proposal(spec, *, change=True):
    roots = deepcopy(spec['root_reference'])
    if change:
        for index, row in enumerate(roots):
            row['translation'] = [index * .004, index * -.002, index * .003]
    return encode({'root_reference_candidate': roots})


def test_extract_absolute_world_points_exact_clocks_and_deduplicated_floor_schedule(monkeypatch):
    engines, spec, evidence, human_raw, evidence_raw, raw = extract(monkeypatch)
    result = json.loads(raw)
    assert len(engines) == 1 and engines[0].calls == 5 and engines[0].closed
    assert len(result['source_rows']) == 3 and len(result['floor_rows']) == 8
    assert [row['kind'] for row in result['floor_rows']] == ['source_pose'] * 3 + ['control_grid'] * 5
    assert result['support_names'] == list(engines[0].support_names)
    first = result['source_rows'][0]
    assert first['points']['left_shoulder'] == [.2, -.29, 2.02]
    assert first['points']['left_shoulder'] != (engines[0].relative[0] + spec['root_reference'][0]['translation']).tolist()
    assert [row['decoder_timestamp_s'] for row in result['source_rows']] == [row['decoder_timestamp_s'] for row in evidence['frames']]
    assert result['floor'] == {'normal': [.6, 0., .8], 'offset_model_units': 0.}
    assert result['bindings']['body_sha256'] == sha256(human_raw).hexdigest()
    assert result['bindings']['evidence_sha256'] == sha256(evidence_raw).hexdigest()
    assert result['bindings']['body_angles_sha256'] == canonical_hash(spec['reference'])
    assert result['bindings']['body_shape_sha256'] == canonical_hash({k: v for k, v in spec.items() if k not in {'reference', 'root_reference'}})
    assert result['physics_steps'] == 0 and result['runtime']['model_xml_sha256'] == sha256(b'<unchanged-original-model/>').hexdigest()
    assert encode(spec) == human_raw and encode(evidence) == evidence_raw


def test_changed_roots_get_one_fresh_original_model_with_inclined_plane_translation_check(monkeypatch):
    engines, spec, _, human_raw, evidence_raw, raw = extract(monkeypatch)
    candidate = proposal(spec)
    checked = json.loads(geometry.check_proposed_geometry(human_raw, evidence_raw, raw, candidate, deadline=time.monotonic() + 2))
    assert len(engines) == 2 and [engine.calls for engine in engines] == [5, 5]
    assert all(engine.closed for engine in engines)
    assert checked['status'] == 'completed' and checked['unique_evaluations'] == 5
    assert checked['source_pose_checks'] == 3 and checked['control_grid_checks'] == 5
    assert all(checked['gates'].values())
    before = json.loads(raw)['floor_rows'][1]['gaps']['sphere:0.0']
    after = checked['rows'][1]['gaps']['sphere:0.0']
    assert after - before == pytest.approx(.6 * .004 + .8 * .003, abs=1e-12)
    assert checked['bindings']['geometry_sha256'] == sha256(raw).hexdigest()
    assert checked['bindings']['proposal_sha256'] == sha256(candidate).hexdigest()
    assert not checked['admitted'] and not checked['promoted']


def test_unchanged_roots_skip_second_model_without_claiming_a_fresh_pass(monkeypatch):
    engines, spec, _, human_raw, evidence_raw, raw = extract(monkeypatch)
    checked = json.loads(geometry.check_proposed_geometry(human_raw, evidence_raw, raw, proposal(spec, change=False),
                                                        deadline=time.monotonic() + 2))
    assert len(engines) == 1
    assert checked['status'] == 'not_required' and checked['reason'] == 'original_roots_unchanged'
    assert checked['rows'] == [] and checked['unique_evaluations'] == 0
    assert all(value is None for value in checked['gates'].values())


def test_fresh_geometry_disagreement_is_visible_not_admitted(monkeypatch):
    engines, spec, _, human_raw, evidence_raw, raw = extract(monkeypatch, corrupt_final=True)
    checked = json.loads(geometry.check_proposed_geometry(human_raw, evidence_raw, raw, proposal(spec), deadline=time.monotonic() + 2))
    assert checked['status'] == 'completed' and not checked['gates']['translation_consistency']
    assert checked['maximum_gap_translation_error_model_units'] == pytest.approx(.01)
    assert all(engine.closed for engine in engines) and not checked['admitted']


@pytest.mark.parametrize('phase', ['extraction', 'final'])
def test_expired_deadline_never_constructs_a_model(monkeypatch, phase):
    if phase == 'extraction':
        engines = setup_geometry(monkeypatch); spec, evidence = fixture()
        call = lambda: geometry.extract_original_geometry(encode(spec), encode(evidence), runtime=RUNTIME, deadline=0.)
        expected = 0
    else:
        engines, spec, _, human_raw, evidence_raw, raw = extract(monkeypatch)
        call = lambda: geometry.check_proposed_geometry(human_raw, evidence_raw, raw, proposal(spec), deadline=0.)
        expected = 1
    with pytest.raises(TimeoutError): call()
    assert len(engines) == expected


@pytest.mark.parametrize('mutate', [
    lambda value: value['root_reference_candidate'][0]['translation'].__setitem__(0, .001),
    lambda value: value['root_reference_candidate'][1].__setitem__('time_s', .03),
    lambda value: value['root_reference_candidate'][1].__setitem__('translation', [True, 0., 0.]),
    lambda value: value['root_reference_candidate'][1].__setitem__('translation', [.081, 0., 0.]),
    lambda value: value['root_reference_candidate'][1].__setitem__('translation', [.04, 0., 0.]),
])
def test_invalid_proposed_identity_clock_and_bounds_reject_before_second_model(monkeypatch, mutate):
    engines, spec, _, human_raw, evidence_raw, raw = extract(monkeypatch)
    value = json.loads(proposal(spec)); mutate(value)
    with pytest.raises(ValueError):
        geometry.check_proposed_geometry(human_raw, evidence_raw, raw, encode(value), deadline=time.monotonic() + 2)
    assert len(engines) == 1


def test_geometry_size_failure_closes_original_model(monkeypatch):
    engines = setup_geometry(monkeypatch); spec, evidence = fixture()
    spec['mobility']['ground_plane'] = {'normal': [.6, 0., .8]}
    monkeypatch.setattr(geometry, 'MAX_BYTES', 32)
    with pytest.raises(ValueError, match='8 MiB'):
        geometry.extract_original_geometry(encode(spec), encode(evidence), runtime=RUNTIME, deadline=time.monotonic() + 2)
    assert len(engines) == 1 and engines[0].closed


def test_unsupported_source_and_duplicate_supports_fail_without_silent_repair(monkeypatch):
    engines = setup_geometry(monkeypatch); spec, evidence = fixture()
    spec['camera_reference'] = {}
    with pytest.raises(ValueError, match='stacked'):
        geometry.extract_original_geometry(encode(spec), encode(evidence), runtime=RUNTIME, deadline=time.monotonic() + 2)
    assert not engines
    del spec['camera_reference']
    monkeypatch.setattr(AnalyticGeometry, 'support_names', ('duplicate', 'duplicate'))
    with pytest.raises(ValueError, match='distinct'):
        geometry.extract_original_geometry(encode(spec), encode(evidence), runtime=RUNTIME, deadline=time.monotonic() + 2)
    assert engines[0].closed


def test_native_floor_mismatch_rejects_instead_of_relabeling_the_plane(monkeypatch):
    engines = setup_geometry(monkeypatch); spec, evidence = fixture()
    with pytest.raises(ValueError, match='native floor differs'):
        geometry.extract_original_geometry(encode(spec), encode(evidence), runtime=RUNTIME, deadline=time.monotonic() + 2)
    assert engines[0].closed and engines[0].calls == 1
