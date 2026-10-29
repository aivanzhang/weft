import numpy as np
import pytest

from weft.line_fragments import merge_vertical_fragments


def test_occluded_continuation_is_one_supported_edge():
    source = [[10., 0., 10., 100.], [10., 120., 10., 220.]]
    result = merge_vertical_fragments(source)
    assert result['groups'][0]['input_indices'] == [0, 1]
    np.testing.assert_allclose(result['groups'][0]['pixels'], [[10., 0.], [10., 220.]])
    assert result['links'][0]['gap_px'] == 20.
    assert result['groups'][0]['status'] == 'straight_group'


def test_distinct_parallel_edges_are_not_joined():
    result = merge_vertical_fragments([[10., 0., 10., 200.], [15., 0., 15., 200.]])
    assert len(result['groups']) == 2 and not result['links']


def test_long_unobserved_gap_is_not_bridged():
    result = merge_vertical_fragments([[10., 0., 10., 100.], [10., 200., 10., 300.]])
    assert len(result['groups']) == 2


def test_bent_chain_stays_rejected_instead_of_retaining_one_piece():
    result = merge_vertical_fragments([[10., 0., 10., 400.], [10., 400., 26., 800.]])
    assert len(result['groups']) == 1
    assert result['groups'][0]['status'] == 'nonstraight_group_rejected'
    assert result['groups'][0]['maximum_endpoint_residual_px'] > 2.


def test_endpoint_order_does_not_change_fitted_edge():
    source = np.array([[10., 0., 12., 100.], [12.4, 120., 14.4, 220.]])
    old = source.copy()
    forward = merge_vertical_fragments(source)
    reverse = merge_vertical_fragments(source[:, [2, 3, 0, 1]])
    np.testing.assert_allclose(forward['groups'][0]['pixels'], reverse['groups'][0]['pixels'], atol=1e-12)
    np.testing.assert_array_equal(source, old)


@pytest.mark.parametrize('source', [[], [[1., 2., 1., 2.]], [[0., 0., 100., 1.]],
    np.zeros((129, 4)), [[float('nan'), 0., 1., 100.]], [[1e6, 0., 1e6, 100.]], [[1., 2., 3.]]])
def test_invalid_fragments_are_rejected(source):
    with pytest.raises(ValueError):
        merge_vertical_fragments(source)
