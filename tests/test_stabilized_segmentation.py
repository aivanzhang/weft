"""Coordinate-adapter tests; registration quality is tested separately against pixels."""

from copy import deepcopy

import numpy as np
import pytest

from test_segmentation_scene import assumptions, evidence
from weft.io import canonical_hash
from weft.segmentation_scene import generate_candidate, evaluate_candidate
from weft.stabilized_segmentation import materialize_stabilized, stabilized_evidence


@pytest.fixture
def adapter_boundary(monkeypatch):
    # Isolate coordinate/provenance logic from the independently tested registration estimator.
    monkeypatch.setattr('weft.camera_stabilization.validate_registration', deepcopy)
    monkeypatch.setattr('weft.camera_stabilization.registration_evidence_hash', canonical_hash)


def translated():
    data = evidence()
    for row in data['observations']:
        offset = 2 * row['frame']
        row['geometry']['box_xyxy_halfopen'][0] += offset
        row['geometry']['box_xyxy_halfopen'][2] += offset
        row['geometry']['box_xywh'][0] += offset
        row['geometry']['center_px'][0] += offset
        row['raw_model_output'] = {'opaque_mask': 'unchanged-source-payload'}
    report = {'ready': True, 'source_sha256': data['source']['submitted_video_sha256'],
              'width': 200, 'height': 100, 'timestamps_s': data['time']['timestamps_s'].copy(),
              'reference_frame': 0, 'evidence_sha256': canonical_hash(data),
              'frames': [{'frame': frame, 'time_s': stamp, 'status': 'registered',
                          'current_to_reference': [[1., 0., -2.*frame], [0., 1., 0.], [0., 0., 1.]]}
                         for frame, stamp in enumerate(data['time']['timestamps_s'])]}
    return data, report


def test_adapter_restores_reference_coordinates_and_preserves_all_raw_evidence(adapter_boundary):
    data, report = translated()
    originals = deepcopy(data), deepcopy(report)
    document = stabilized_evidence(data, report)
    normalized = materialize_stabilized(document)
    expected = evidence()
    assert (data, report) == originals
    assert document['source_evidence'] == data
    assert normalized['source']['coordinate_system'] == 'registered_reference_image_pixels'
    assert normalized['source']['reference_frame'] == 0
    assert len(normalized['observations']) == len(data['observations'])
    for original, row, target in zip(data['observations'], normalized['observations'], expected['observations']):
        assert row['id'] == original['id'] and row['identity'] == original['identity']
        assert row['raw_model_output'] == original['raw_model_output']
        assert row['geometry']['center_px'] == pytest.approx(target['geometry']['center_px'])
        assert row['geometry']['original_box_xyxy_halfopen'] == original['geometry']['box_xyxy_halfopen']
    assert normalized['identity_groups'] == data['identity_groups']
    assert normalized['physics_ready'] is False


def test_known_moving_camera_requires_explicit_validated_coordinate_hypothesis(adapter_boundary):
    data, report = translated()
    config = assumptions()
    config['known_camera_motion'] = 'moving'
    with pytest.raises(ValueError, match='moving camera'):
        generate_candidate(data, config)
    document = stabilized_evidence(data, report)
    with pytest.raises(ValueError, match='Unsupported camera'):
        generate_candidate(document, config)
    config['camera_model'] = 'similarity_stabilized_orthographic_assumed'
    candidate = generate_candidate(document, config)
    assert candidate['authored_assumptions']['known_camera_motion'] == 'moving'
    assert candidate['scene']['objects'][0]['velocity'] == pytest.approx([.1, 0.], abs=1e-10)
    replay = evaluate_candidate(candidate, document)
    assert replay['partitions']['development_frames']['fixed_raw_id_hypothesis']['conditional_rmse_px'] < 1e-10
    assert candidate['physics_ready'] is False
    assert any('not reclassified as fixed' in item for item in candidate['scene']['assumptions'])


@pytest.mark.parametrize('mutation', [
    lambda report: report.update(ready=False),
    lambda report: report.update(source_sha256='b'*64),
    lambda report: report.update(width=201),
    lambda report: report.update(height=101),
    lambda report: report['timestamps_s'].__setitem__(1, .02),
    lambda report: report.update(reference_frame=1),
    lambda report: report.update(evidence_sha256='b'*64),
    lambda report: report['frames'][1].update(status='unavailable', current_to_reference=None),
    lambda report: report['frames'][1]['current_to_reference'][0].__setitem__(2, -1000.),
])
def test_adapter_rejects_wrong_or_incomplete_registration_without_filling_or_clipping(adapter_boundary, mutation):
    data, report = translated()
    mutation(report)
    with pytest.raises(ValueError):
        stabilized_evidence(data, report)


def test_rotated_box_uses_transformed_envelope_and_retains_four_corners(adapter_boundary):
    data, report = translated()
    angle = .05
    matrix = [[np.cos(angle), -np.sin(angle), 5.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
    report['frames'][1]['current_to_reference'] = matrix
    row = materialize_stabilized(stabilized_evidence(data, report))['observations'][2]
    points = np.asarray(row['geometry']['transformed_corners_px'])
    assert points.shape == (4, 2)
    np.testing.assert_allclose(row['geometry']['center_px'], points.mean(axis=0))
    assert row['geometry']['box_xywh'][2] > 4.


@pytest.mark.parametrize('mutation', [
    lambda doc: doc['source_evidence']['observations'][1]['identity'].update(physical_identity_verified=True),
    lambda doc: doc['registration']['frames'][1]['current_to_reference'][0].__setitem__(2, 5.),
    lambda doc: doc.update(physics_ready=True),
    lambda doc: doc.update(scope='measured physical truth'),
])
def test_saved_wrapper_rejects_changed_inputs_and_claims(adapter_boundary, mutation):
    data, report = translated()
    wrapped = stabilized_evidence(data, report)
    mutation(wrapped)
    with pytest.raises(ValueError):
        materialize_stabilized(wrapped)


def test_missing_original_slots_stay_missing_and_raw_wrapper_cannot_stack(adapter_boundary):
    data, report = translated()
    data['observations'][-1].update(proposal_status='missing', geometry=None)
    report['evidence_sha256'] = canonical_hash(data)
    wrapped = stabilized_evidence(data, report)
    row = materialize_stabilized(wrapped)['observations'][-1]
    assert row['proposal_status'] == 'missing' and row['geometry'] is None
    with pytest.raises(ValueError, match='stacked'):
        stabilized_evidence(wrapped, report)


def test_private_materialized_coordinates_cannot_masquerade_as_original_video(adapter_boundary):
    data, report = translated()
    materialized = materialize_stabilized(stabilized_evidence(data, report))
    with pytest.raises(ValueError, match='validated original-evidence wrapper'):
        generate_candidate(materialized, assumptions())


def test_real_registration_validator_and_pixel_worker_integrate_with_candidate(tmp_path):
    import cv2
    from test_camera_stabilization import texture, write_video
    from weft.camera_stabilization import register_video
    from weft.io import digest

    data = evidence(times=[i*.02 for i in range(8)])
    data['source'].update(width_px=320, height_px=240)
    reference = texture()
    images = []
    for frame in range(8):
        current = reference.copy()
        for row in data['observations'][frame*2:frame*2+2]:
            x, y = row['geometry']['center_px']
            cv2.circle(current, (round(x), round(y)), 2, 0, -1)
            for key in ('box_xyxy_halfopen', 'box_xywh', 'center_px'):
                row['geometry'][key][0] += frame
            row['geometry']['box_xyxy_halfopen'][2] += frame
        images.append(cv2.warpAffine(current, np.float32([[1, 0, frame], [0, 1, 0]]), (320, 240)))
    video = tmp_path/'known-camera.avi'
    write_video(video, images)
    data['source']['submitted_video_sha256'] = digest(video)
    for row in data['observations']:
        row['id'] = f"{digest(video)}:frame:{row['frame']}:model-object:{row['raw_model_id']}"
    for i, obj in enumerate(data['objects']):
        obj['initialization'] = {'frame': 0, 'roi': data['observations'][i]['geometry']['box_xywh'].copy()}
    registration = register_video(video, data, reference_frame=0)
    assert registration['ready'] is True
    wrapped = stabilized_evidence(data, registration)
    config = assumptions()
    config.update(camera_model='similarity_stabilized_orthographic_assumed', known_camera_motion='moving')
    for obj in config['objects']:
        obj['radius_px'] = 3.
    candidate = generate_candidate(wrapped, config)
    assert candidate['scene']['objects'][0]['velocity'] == pytest.approx([20/320, 0], abs=.03)
    replay = evaluate_candidate(candidate, wrapped)
    assert replay['status'] == 'completed_diagnostic_replay'
    assert replay['partitions']['development_frames']['fixed_raw_id_hypothesis']['conditional_rmse_px'] < 1.5
    assert wrapped['source_evidence'] == data
