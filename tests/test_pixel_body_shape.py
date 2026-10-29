from copy import deepcopy
import json

import mujoco
import numpy as np
import pytest

from weft.body_shape_diagnostic import INDEX
from weft.io import canonical_hash
from weft.pixel_body_shape import fit_pixel_shape
from test_body_shape_diagnostic import fixture, run


def inputs():
    spec, evidence = fixture()
    return spec, run(spec, evidence)


def test_exact_geometry_unchanged_without_dynamics_or_floor_refit(monkeypatch):
    spec, diagnosis = inputs()
    before = deepcopy((spec, diagnosis))
    def forbidden(*args, **kwargs):
        raise AssertionError('No dynamics or floor refitting')
    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    monkeypatch.setattr('weft.mobile_human.calibrate_floor', forbidden)
    result = fit_pixel_shape(spec, diagnosis)
    assert (spec, diagnosis) == before
    assert result['candidate']['lengths'] == pytest.approx(spec['lengths'])
    for group in result['summary'].values():
        assert group['candidate']['pixels']['rmse'] < 1e-12
    json.dumps(result, allow_nan=False)


def length_change(diagnosis, ratio, partitions=('training', 'heldout')):
    for row in diagnosis['rows']:
        if row['partition'] not in partitions:
            continue
        for side in ('left', 'right'):
            a, b = INDEX[side + '_elbow'], INDEX[side + '_wrist']
            points = np.array(row['source_pixels_hip_centered'])
            points[b] = points[a] + ratio * (points[b] - points[a])
            row['source_pixels_hip_centered'] = points.tolist()


def test_known_shared_length_change_improves_later_pixels_with_prior_shrinkage():
    spec, diagnosis = inputs()
    length_change(diagnosis, 1.1)
    result = fit_pixel_shape(spec, diagnosis)
    assert result['estimates']['forearm']['bounded_ratio'] == pytest.approx(1 + .1 / 1.1)
    later = result['summary']['heldout']
    assert later['candidate']['pixels']['rmse'] < later['baseline']['pixels']['rmse'] / 5
    assert result['candidate']['reference'] == spec['reference']
    assert result['candidate']['root_reference'] == spec['root_reference']
    assert result['candidate']['mobility'] == spec['mobility']


def test_later_changes_cannot_fit_lengths_and_row_partition_is_not_trusted():
    spec, diagnosis = inputs()
    baseline = fit_pixel_shape(spec, diagnosis)
    length_change(diagnosis, 2, ('heldout',))
    for row in diagnosis['rows']:
        row['partition'] = 'training'
    altered = fit_pixel_shape(spec, diagnosis)
    assert altered['estimates'] == baseline['estimates']
    assert altered['summary']['heldout']['candidate']['pixels']['rmse'] > 1


@pytest.mark.parametrize('cause', ['low_confidence', 'foreshortened', 'short_image_segment'])
def test_unidentifiable_segments_retain_original_length(cause):
    spec, diagnosis = inputs()
    for row in diagnosis['rows']:
        for side in ('left', 'right'):
            a, b = INDEX[side + '_elbow'], INDEX[side + '_wrist']
            if cause == 'low_confidence':
                row['effective_confidence'][side + '_wrist'] = 0
            elif cause == 'short_image_segment':
                row['source_pixels_hip_centered'][b] = row['source_pixels_hip_centered'][a].copy()
            else:
                points = np.array(row['baseline_model_hip_centered'])
                points[b] = points[a] + [0, spec['lengths']['forearm'], 0]
                row['baseline_model_hip_centered'] = points.tolist()
    result = fit_pixel_shape(spec, diagnosis)
    estimate = result['estimates']['forearm']
    assert estimate['status'] == 'unidentified_kept_original'
    assert estimate['candidate_length'] == spec['lengths']['forearm']
    assert all(row['reason'] == cause for row in estimate['samples'])


def test_cap_and_stale_clearance_claim_are_explicit():
    spec, diagnosis = inputs()
    spec['mobility']['floor_fit'] = {'historical_claim': 'not reusable after changing geometry'}
    diagnosis['model_sha256'] = canonical_hash(spec)
    length_change(diagnosis, 2)
    result = fit_pixel_shape(spec, diagnosis)
    assert result['estimates']['forearm']['bounded_ratio'] == 1.15
    assert result['estimates']['forearm']['cap_hit']
    assert 'floor_fit' not in result['candidate']['mobility']
    assert result['candidate']['mobility']['base_height'] == spec['mobility']['base_height']
    assert result['original_floor_fit_not_reapplied'] == spec['mobility']['floor_fit']


@pytest.mark.parametrize('field', ['source', 'model', 'time', 'duplicate', 'missing', 'nonfinite', 'confidence', 'geometry'])
def test_conflicting_or_invalid_evidence_rejected(field):
    spec, diagnosis = inputs()
    if field == 'source': diagnosis['source_sha256'] = 'b' * 64
    if field == 'model': diagnosis['model_sha256'] = 'b' * 64
    if field == 'time': diagnosis['rows'][0]['time_s'] = .01
    if field == 'duplicate': diagnosis['rows'][1] = deepcopy(diagnosis['rows'][0])
    if field == 'missing': diagnosis['rows'].pop()
    if field == 'nonfinite': diagnosis['rows'][0]['source_pixels_hip_centered'][0][0] = float('nan')
    if field == 'confidence': diagnosis['rows'][0]['effective_confidence']['left_wrist'] = 2
    if field == 'geometry': diagnosis['rows'][0]['baseline_model_hip_centered'][INDEX['left_wrist']][2] += 1
    with pytest.raises(ValueError):
        fit_pixel_shape(spec, diagnosis)


def test_unavailable_slots_are_retained_not_zero_scored():
    spec, diagnosis = inputs()
    diagnosis['rows'][-1]['status'] = 'missing_landmarks'
    result = fit_pixel_shape(spec, diagnosis)
    assert len(result['rows']) == 6
    assert 'comparisons' not in result['rows'][-1]
    assert result['summary']['heldout']['candidate']['pixels']['count'] == 12
