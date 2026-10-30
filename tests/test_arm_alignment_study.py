"""Normal report serialization must retain native flags and every inspected source row."""

import importlib
import json
from pathlib import Path

import pytest
import numpy as np


@pytest.fixture
def study(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts' / 'studies'))
    module = importlib.import_module('check_source_arm_alignment')
    def forbidden(*args, **kwargs):
        raise AssertionError('Report serialization must not fit or evaluate geometry')
    monkeypatch.setattr(module, 'align_arm_reference', forbidden)
    monkeypatch.setattr(module, 'geometry_comparison', forbidden)
    return module


def source_rows():
    return [{'frame': index, 'object_id': 'person', 'admission': 'included', 'visibility': 'detected',
             'measurements': {'landmarks': {'value': [{'index': 13, 'position': [20., 30.],
                                                     'visibility': confidence, 'presence': 1.}]}}}
            for index, confidence in enumerate((.9, .2))]


def image_rows():
    return [{'frame': index, 'joint': 'left_elbow', 'before_projection_px': [24., 33.],
             'after_projection_px': [20., 30.], 'before_model_units': [0., 0., .5],
             'after_model_units': [.01, 0., .5]} for index in range(3)]


def test_normal_image_report_serializes_numpy_derived_eligibility(tmp_path, study):
    observations = source_rows() + [{'frame': 2, 'object_id': 'person', 'admission': 'withheld',
                                     'visibility': None, 'measurements': {}}]
    evidence = json.dumps({'sources': [{'document_sha256': 'a' * 64, 'image': {'width_px': 100, 'height_px': 80},
                                      'observations': observations}]}).encode()
    rows = image_rows()
    audit = {'references': [{'endpoints': [{'joint': 'left_elbow', 'image_px': [20., 30.],
                                          'fit_weight': .9, 'error_before_px': 5., 'error_after_px': 0.}]}]}
    summary = study.summarize_images(audit, rows, evidence, {'document_sha256': 'a' * 64, 'object_id': 'person'})
    assert [row['eligible'] for row in rows] == [True, False, False]
    assert all(type(row['eligible']) is bool for row in rows)
    path = tmp_path / 'normal-body-image-comparison.json'
    study.write(path, {'points': rows, 'summary': summary})
    saved = json.loads(path.read_text())
    assert len(saved['points']) == 3
    assert saved['summary']['whole_body_eligible']['before']['count'] == 1
    assert saved['summary']['whole_body_eligible']['before']['rmse'] == 5.
    assert saved['summary']['whole_body_all_finite']['after']['count'] == 2
    assert saved['points'][2]['source_image_px'] is None
    assert saved['points'][2]['error_after_px'] is None


def test_normal_image_report_retains_all_finite_rows_when_none_are_eligible(tmp_path, study):
    observations = source_rows()
    observations[0]['measurements']['landmarks']['value'][0]['position'] = [101., 30.]
    evidence = json.dumps({'sources': [{'document_sha256': 'a' * 64, 'image': {'width_px': 100, 'height_px': 80},
                                      'observations': observations}]}).encode()
    rows = image_rows()[:2]
    summary = study.summarize_images({'references': []}, rows, evidence,
                                    {'document_sha256': 'a' * 64, 'object_id': 'person'})
    path = tmp_path / 'no-eligible-image-comparison.json'
    study.write(path, {'points': rows, 'summary': summary})
    saved = json.loads(path.read_text())
    assert [row['eligible'] for row in saved['points']] == [False, False]
    assert saved['summary']['whole_body_eligible']['after']['count'] == 0
    assert saved['summary']['whole_body_eligible']['after']['rmse'] is None
    assert saved['summary']['whole_body_all_finite']['after']['count'] == 2


@pytest.fixture
def projection(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts' / 'studies'))
    return importlib.import_module('check_source_contact_projection')


def test_contact_projection_capsule_entry_exit_includes_caps_and_parallel_rays(projection):
    first, second = [-.5, 2., 0.], [.5, 2., 0.]
    lo, hi = projection.ray_capsule_interval([0., 0., 0.], np.array([[0., 1., 0.], [0., 1., 1.]]), first, second, .25)
    assert lo[0] == pytest.approx(1.75) and hi[0] == pytest.approx(2.25)
    assert lo[1] > hi[1]
    lo, hi = projection.ray_capsule_interval([-2., 2., 0.], np.array([[1., 0., 0.]]), first, second, .25)
    assert lo[0] == pytest.approx(1.25) and hi[0] == pytest.approx(2.75)


def test_contact_projection_keeps_image_overlap_distinct_from_volume_intersection(projection):
    rays = np.array([[0., 1., 0.], [1., 0., 0.]])
    a, b = projection.ray_capsule_interval([0., 0., 0.], rays, [-.5, 2., 0.], [.5, 2., 0.], .25)
    c, d = projection.ray_box_interval([0., 0., 0.], rays, [0., 4., 0.], [.5, .5, .5])
    assert a[0] <= b[0] and c[0] <= d[0]
    assert max(a[0], c[0]) > min(b[0], d[0])
    assert c[1] > d[1]


def test_contact_projection_retains_conflicting_masks_and_unknown_pixels(projection):
    equipment = np.array([[True, False, False, True, False]])
    person = np.array([[False, True, False, True, False]])
    other = np.array([[False, False, True, False, False]])
    labels = projection.mask_labels(equipment, person, other)
    assert labels.tolist() == [[0, 1, 2, 3, 4]]
    counts = projection.counts(np.ones_like(labels, dtype=bool), labels)
    assert list(counts.values()) == [1] * 5
    assert counts['background_or_unresolved'] == 1
    json.dumps(counts, allow_nan=False)
