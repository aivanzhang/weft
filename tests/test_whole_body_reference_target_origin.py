from copy import deepcopy
import builtins

import numpy as np
import pytest

from test_whole_body_reference_refinement import encode, fixture, run_toy
from weft import whole_body_reference_refinement as refinement
from weft.io import canonical_hash


@pytest.fixture(autouse=True)
def no_native_or_media_imports(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'mujoco', 'gymnasium', 'cv2', 'mediapipe', 'torch', 'PIL'}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)


def crop_fixture():
    spec, evidence = fixture()
    selected = evidence['frames'][1]
    original = deepcopy(selected)
    original.update(status='no_pose', pose_count=0, landmarks=[], world_landmarks=[])
    crop = {'frame': selected['frame'], 'box_xyxy': [120, 60, 1080, 540]}
    # The saved crop's x=.45 maps to full-image x=.46; the fitter must not map it again.
    selected['landmarks'][27]['x'] = (120 + .45 * 960) / 1200
    selected.update(crop_proposal=deepcopy(crop), inference_seconds=.001)
    marker = {'method': 'single-view-selective-crop-v1', 'crop': crop,
              'original_observation_sha256': canonical_hash(original),
              'proposal_sha256': canonical_hash(selected),
              'decision_sha256': canonical_hash({'frame': selected['frame'], 'accepted': True}),
              'anatomical_identity_verified': False}
    selected.update(image_refinement=marker, image_refinement_raw=original)
    spec['reference_evidence'][1]['image_refinement'] = deepcopy(marker)
    return spec, evidence


def test_selected_crop_targets_keep_mapped_coordinates_and_fixed_root(monkeypatch):
    spec, evidence = crop_fixture()
    before = deepcopy((spec, evidence))
    result = run_toy(monkeypatch, spec, evidence, 'curved', joint_mask=['left_knee'])
    candidate, report = result['human_candidate'], result['report']
    row = report['references'][1]
    marker = evidence['frames'][1]['image_refinement']
    assert row['target_origin'] == {
        'kind': 'selected_crop_estimate_mapped_to_source', 'method': marker['method'],
        **{key: marker[key] for key in ('original_observation_sha256', 'proposal_sha256', 'decision_sha256')},
        'anatomical_identity_verified': False}
    assert row['points'][10]['target_model_units'] == pytest.approx([-.2, .75])
    assert all(point['landmark_source'] == 'landmarks' for point in row['points'])
    assert all('target_origin' not in other for i, other in enumerate(report['references']) if i != 1)
    assert evidence['frames'][1]['image_refinement_raw']['status'] == 'no_pose'
    assert evidence['frames'][1]['image_refinement_raw']['landmarks'] == []
    assert (spec, evidence) == before
    assert candidate['root_reference'] == spec['root_reference']
    assert candidate['reference'][0] == spec['reference'][0]
    assert all(candidate[key] == value for key, value in spec.items() if key != 'reference')
    assert report['gates']['source_endpoint_image_nonregression']
    assert report['gates']['pointwise_floor_nonregression'] and report['gates']['floor_feasibility']
    assert report['gates']['endpoint_depth_change']
    assert report['contract']['floor_penetration_limit_model_units'] == .025
    assert report['contract']['maximum_endpoint_depth_change_model_units'] == .08
    assert report['image_summary']['grid_image_targets'] == report['physics_steps'] == 0


def test_origin_annotation_does_not_change_any_prepared_numeric_input():
    spec, evidence = crop_fixture()
    ordinary = deepcopy(evidence)
    ordinary['frames'][1].pop('image_refinement')
    ordinary['frames'][1].pop('image_refinement_raw')
    annotated = refinement._prepare(spec, evidence)
    unannotated = refinement._prepare(spec, ordinary)
    for before, after in zip(annotated[:3], unannotated[:3], strict=True):
        assert np.array_equal(before, after)
    assert annotated[4] == unannotated[4]
    for before, after in zip(annotated[3], unannotated[3], strict=True):
        assert np.array_equal(before['target'], after['target'])
        assert np.array_equal(before['weights'], after['weights'])
        assert before['points'] == after['points']
        assert {key: value for key, value in before.items() if key not in {'target_origin', 'target', 'weights'}} == {
            key: value for key, value in after.items() if key not in {'target', 'weights'}}


@pytest.mark.parametrize('identity_raw', [False, True])
def test_ordinary_or_identity_raw_rows_keep_the_original_report_schema(monkeypatch, identity_raw):
    spec, evidence = fixture()
    if identity_raw:
        for frame in evidence['frames']:
            frame['identity_raw'] = deepcopy(frame)
            frame['landmarks'][27]['x'] += .2
    result = run_toy(monkeypatch, spec, evidence, 'curved', joint_mask=['left_knee'])
    expected = {'evidence_id', 'frame', 'time_s', 'root_status', 'status', 'points', 'correction_rad'}
    assert all(set(row) == expected for row in result['report']['references'])
    point = result['report']['references'][1]['points'][10]
    assert point['landmark_source'] == ('identity_raw.landmarks' if identity_raw else 'landmarks')
    assert point['target_model_units'] == pytest.approx([-.25, .75])
    assert 'target_origin' not in point


def test_crop_provenance_does_not_make_a_weak_endpoint_available(monkeypatch):
    spec, evidence = crop_fixture()
    frame = evidence['frames'][1]
    frame['landmarks'][15]['presence'] = .1
    proposal = {key: value for key, value in frame.items() if key not in {'image_refinement', 'image_refinement_raw'}}
    frame['image_refinement']['proposal_sha256'] = canonical_hash(proposal)
    spec['reference_evidence'][1]['image_refinement'] = deepcopy(frame['image_refinement'])
    result = run_toy(monkeypatch, spec, evidence, 'curved', joint_mask=['left_knee'])
    row = result['report']['references'][1]
    point = next(point for point in row['points'] if point['joint'] == 'left_wrist')
    assert row['target_origin']['kind'] == 'selected_crop_estimate_mapped_to_source'
    assert point['status'] == 'low_confidence' and point['weight'] == 0
    assert point['error_before_px'] is point['error_after_px'] is None


@pytest.mark.parametrize('change', [
    lambda f: f.update(image_refinement=None),
    lambda f: f['image_refinement'].update(method='two-view-temporal-crop-v1'),
    lambda f: f['image_refinement'].update(anatomical_identity_verified=True),
    lambda f: f['image_refinement'].update(original_observation_sha256='A' * 64),
    lambda f: f['image_refinement'].update(original_observation_sha256='c' * 64),
    lambda f: f['image_refinement'].update(proposal_sha256='d' * 64),
    lambda f: f['image_refinement'].update(decision_sha256=None),
    lambda f: f['landmarks'][27].update(x=.9),
    lambda f: f['image_refinement_raw'].update(id='different-source'),
    lambda f: f['image_refinement_raw'].update(decoder_timestamp_s=None),
    lambda f: f['image_refinement_raw'].pop('decoder_timestamp_s'),
    lambda f: f.update(identity_raw=deepcopy(f['image_refinement_raw'])),
    lambda f: f['image_refinement_raw'].update(image_refinement={'method': 'prior'}),
    lambda f: f.pop('image_refinement'),
])
def test_invalid_or_mixed_crop_claims_cannot_acquire_a_descriptor(change):
    _, evidence = crop_fixture()
    frame = evidence['frames'][1]
    change(frame)
    with pytest.raises(ValueError, match='crop'):
        refinement._target_origin(frame)


def test_bad_crop_provenance_is_rejected_before_geometry(monkeypatch):
    spec, evidence = crop_fixture()
    evidence['frames'][1]['image_refinement']['proposal_sha256'] = 'd' * 64
    def forbidden(_):
        pytest.fail('Malformed crop provenance must fail before geometry')
    monkeypatch.setattr(refinement, '_make_geometry', forbidden)
    with pytest.raises(ValueError, match='crop target observation identity'):
        refinement.propose_whole_body_reference_refinement(encode(spec), encode(evidence))
