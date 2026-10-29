from copy import deepcopy

import pytest

from weft.projected_joint_evaluation import grade_projected_joints


def rows(points):
    return [{'frame': i, 'joint': 23, 'pixel': p} for i, p in enumerate(points)]


def test_shared_errors_and_missing_slots_are_separate():
    truth = rows([[0., 0.], [0., 0.], [0., 0.], None, [0., 0.]])
    predictions = {'raw': rows([[3., 4.], None, [0., 0.], [1., 2.], [0., 0.]]),
                   'candidate': rows([[0., 0.], [3., 4.], None, [1., 2.], [0., 0.]])}
    before = deepcopy((truth, predictions))
    result = grade_projected_joints(predictions, truth, baseline='raw', limit_px=4.)
    assert result['methods']['raw']['paired_slots'] == 3
    assert result['methods']['raw']['within_limit'] == 2
    assert result['methods']['raw']['outside_limit'] == 1
    assert result['methods']['raw']['observation_without_prediction'] == 1
    assert result['methods']['raw']['prediction_without_observation'] == 1
    paired = result['paired_comparisons']['candidate']
    assert paired['shared_slots'] == 2
    assert paired['baseline_on_shared']['rmse_px'] == pytest.approx(5/2**.5)
    assert paired['candidate_on_shared']['rmse_px'] == 0
    assert paired['improved_slots'] == 1 and paired['worsened_slots'] == 0
    assert paired['newly_covered_observations'] == paired['lost_observed_predictions'] == 1
    assert result['all_slot_errors'][2]['errors_px']['candidate'] is None
    assert (truth, predictions) == before


def test_unavailable_comparison_is_not_zero_error():
    result = grade_projected_joints({'raw': rows([None])}, rows([[1., 1.]]), baseline='raw', limit_px=1.)
    assert result['methods']['raw']['paired_slots'] == 0
    assert result['methods']['raw']['rmse_px'] is None


def test_far_predictions_are_not_clipped_to_the_image():
    result = grade_projected_joints({'raw': rows([[-100., 0.]])}, rows([[0., 0.]]), baseline='raw', limit_px=10.)
    assert result['methods']['raw']['maximum_px'] == 100.
    assert result['methods']['raw']['outside_limit'] == 1


@pytest.mark.parametrize('problem', ['omitted_slot', 'duplicate_slot', 'nan', 'infinite', 'too_large', 'bad_baseline', 'bad_limit'])
def test_invalid_grading_inputs_fail(problem):
    observations = rows([[0., 0.], [1., 1.]])
    predictions = {'raw': deepcopy(observations)}; baseline = 'raw'; limit = 1.
    if problem == 'omitted_slot': predictions['raw'].pop()
    if problem == 'duplicate_slot': observations[1]['frame'] = 0
    if problem == 'nan': predictions['raw'][0]['pixel'] = [float('nan'), 0.]
    if problem == 'infinite': predictions['raw'][0]['pixel'] = [float('inf'), 0.]
    if problem == 'too_large': predictions['raw'][0]['pixel'] = [1e6, 0.]
    if problem == 'bad_baseline': baseline = 'unknown'
    if problem == 'bad_limit': limit = True
    with pytest.raises(ValueError):
        grade_projected_joints(predictions, observations, baseline=baseline, limit_px=limit)
