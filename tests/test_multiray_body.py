from copy import deepcopy

import numpy as np
import pytest

from test_camera_pose import fixture
from test_multiview_root import CLOCK, WORLD, inputs
from weft.human import validate_human
from weft.io import canonical_hash
from weft.multiview_body import MULTI_METHOD as BODY_METHOD, make_two_view_body
from weft.multiview_root import MULTI_METHOD as ROOT_METHOD, make_two_view_root, shape_hash


def setup(tmp_path):
    base, first, second, roots, *_ = inputs(tmp_path)
    evidence, camera, _, _ = fixture(roots=roots, translations=[[.8, -.3, 0.]]*len(roots))
    evidence['source_sha256'] = camera['source_sha256'] = 'c'*64
    for row in evidence['frames']:
        row['id'] = f"{'c'*64}:frame:{row['frame']}"
    third = {'name': 'third', 'camera': camera, 'evidence': evidence, 'clock_provenance': CLOCK}
    return base, first, second, third, roots


def test_extra_view_recovers_missing_root_without_changing_shape_ground_or_slots(tmp_path):
    base, first, second, third, expected = setup(tmp_path)
    second['evidence']['frames'].pop(3)
    before = canonical_hash([base, first, second, third])
    root, report, predictions = make_two_view_root(base, first, second, additional_views=[third], world_provenance=WORLD)
    assert root['camera_reference']['method'] == ROOT_METHOD
    assert shape_hash(root) == shape_hash(base)
    assert root['camera_reference']['ground'] == base['camera_reference']['ground']
    assert len(root['reference']) == len(base['reference']) == len(report['source_slots']) == 10
    rows = root['camera_reference']['root_refinement']['roots']
    assert rows[3]['source_evidence_ids'][1] is None
    assert rows[3]['hip_view_indices'] == [[0, 2], [0, 2]]
    np.testing.assert_allclose([r['root_world_m'] for r in rows], expected, atol=1e-10)
    assert predictions['frames'][3]['joints'][23]['observation_statuses'][1] == 'pose_not_sampled'
    assert canonical_hash([base, first, second, third]) == before
    validate_human(root)


@pytest.mark.parametrize('mode', ['directions', 'dimensions', 'both'])
def test_multiview_body_uses_only_joint_contributors_and_keeps_fixed_world(mode, tmp_path):
    base, first, second, third, _ = setup(tmp_path)
    for row in second['evidence']['frames']:
        row['landmarks'][25]['visibility'] = .1
        row['landmarks'][27]['presence'] = .1
    root, _, predictions = make_two_view_root(base, first, second, additional_views=[third], world_provenance=WORLD)
    body, report = make_two_view_body(root, predictions, first['evidence'], second['evidence'],
        additional_evidence=[third['evidence']], mode=mode)
    assert body['camera_reference']['method'] == BODY_METHOD
    assert body['root_reference'] == root['root_reference']
    assert body['mobility'] == root['mobility'] and body['motor_limits'] == root['motor_limits']
    assert body['camera_reference']['ground'] == root['camera_reference']['ground']
    assert report['group_coverage']['left_hip'] == {'multi_view_estimate': 10}
    assert all(row['groups']['left_hip']['joint_view_indices']['25'] == [0, 2] for row in report['frames'])
    if mode != 'dimensions':
        assert body['reference_evidence'][0]['geometry_view_indices']['left_hip']['25'] == [0, 2]
    if mode == 'directions':
        assert body['lengths'] == root['lengths']
    validate_human(body)


def test_missing_every_second_ray_prevents_root_instead_of_dropping_a_pose(tmp_path):
    base, first, second, third, _ = setup(tmp_path)
    for view in (second, third):
        view['evidence']['frames'][3]['landmarks'][23]['visibility'] = .1
    with pytest.raises(ValueError, match='no fallback or dropped poses'):
        make_two_view_root(base, first, second, additional_views=[third], world_provenance=WORLD)


@pytest.mark.parametrize('damage', ['contributors', 'missing_identity', 'remove_field', 'view_count', 'legacy_method'])
def test_multi_root_custody_is_checked(damage, tmp_path):
    base, first, second, third, _ = setup(tmp_path)
    root, _, _ = make_two_view_root(base, first, second, additional_views=[third], world_provenance=WORLD)
    binding = root['camera_reference']; refinement = binding['root_refinement']
    if damage == 'contributors': refinement['roots'][0]['hip_view_indices'][0] = [0, 2]
    elif damage == 'missing_identity': refinement['roots'][0]['source_evidence_ids'][2] = None
    elif damage == 'remove_field': del refinement['roots'][0]['hip_view_indices']
    elif damage == 'view_count': refinement['views'].pop()
    elif damage == 'legacy_method': binding['method'] = 'two-view-root-body-v1'
    with pytest.raises(ValueError):
        validate_human(root)


def test_invalid_body_contributors_do_not_pass_even_if_evidence_hash_is_resealed(tmp_path):
    base, first, second, third, _ = setup(tmp_path)
    root, _, predictions = make_two_view_root(base, first, second, additional_views=[third], world_provenance=WORLD)
    body, _ = make_two_view_body(root, predictions, first['evidence'], second['evidence'], additional_evidence=[third['evidence']])
    body['reference_evidence'][0]['geometry_view_indices']['left_hip']['25'] = [0]
    body['camera_reference']['body_refinement']['output_evidence_sha256'] = canonical_hash(body['reference_evidence'])
    with pytest.raises(ValueError, match='at least two distinct'):
        validate_human(body)


def test_unqualified_extra_sources_reject(tmp_path):
    base, first, second, third, _ = setup(tmp_path)
    root, _, predictions = make_two_view_root(base, first, second, additional_views=[third], world_provenance=WORLD)
    wrong = deepcopy(third['evidence']); wrong['source_sha256'] = 'd'*64
    with pytest.raises(ValueError, match='pinned reconstruction'):
        make_two_view_body(root, predictions, first['evidence'], second['evidence'], additional_evidence=[wrong])
