"""Cache reuse must preserve evidence identity, clocks and the whole-body failure gate."""

from copy import deepcopy
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from weft import pose_densify as module
from weft.benchmark import seal_suite
from weft.io import digest, write_json


@pytest.fixture
def cached_video(tmp_path, monkeypatch):
    video = tmp_path / 'source.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 64))
    assert writer.isOpened()
    for index in range(12): writer.write(np.full((64, 64, 3), index*10, np.uint8))
    writer.release()
    source = digest(video)
    provenance = {'name': 'Test pinned model', 'sha256': 'a'*64, 'delegate': 'CPU', 'network_inference': False}
    model = tmp_path / 'model.task'; model.write_bytes(b'fixture')
    no_pose = {'pose_count': 0, 'status': 'no_pose', 'landmarks': [], 'world_landmarks': []}
    evidence = {'schema_version': 1, 'profile': 'exercise-pose-v1', 'source_sha256': source,
        'width': 64, 'height': 64, 'fps': 30., 'frame_count': 12, 'decoded_frames': 12,
        'inference_stride': 2, 'sample_rate_fps': 15, 'assumptions': [], 'scope': 'Synthetic cache tests only',
        'model': provenance | {'running_mode': 'IMAGE', 'temporal_tracking': False,
            'maximum_pose_hypotheses': 2, 'mediapipe_version': 'fixture'},
        'pose_identity': {'mode': 'raw', 'repaired_frames': 0},
        'frames': [deepcopy(no_pose) | {'frame': i, 'time_s': i/30, 'decoder_timestamp_s': i/30,
            'id': f'{source}:frame:{i}', 'inference_seconds': .001} for i in range(0, 12, 2)]}
    run = tmp_path / 'cached'; run.mkdir()
    write_json(run / 'evidence.json', evidence)
    write_json(run / 'report.json', {'status': 'synthetic_cached_input'})
    seal_suite(run)
    calls = []
    class Detector:
        closed = False
        def detect(self, image): calls.append(float(image.mean())); return no_pose
        def close(self): self.closed = True
    detector = Detector()
    mp = SimpleNamespace(Image=lambda **kwargs: kwargs['data'], ImageFormat=SimpleNamespace(SRGB='RGB'))
    monkeypatch.setattr(module, 'model_provenance', lambda _: provenance)
    monkeypatch.setattr(module.importlib.metadata, 'version', lambda _: 'fixture')
    monkeypatch.setattr(module, '_detector', lambda *_: (mp, detector))
    monkeypatch.setattr(module, 'summarize_result', deepcopy)
    return run, video, model, evidence, provenance, calls, detector


@pytest.mark.parametrize('stride,expected_new', [(1, [1, 3, 5, 7, 9, 11]), (3, [3, 9])])
def test_only_missing_frames_are_inferred_and_all_originals_remain_exact(tmp_path, cached_video, stride, expected_new):
    run, video, model, original, _, calls, detector = cached_video
    before = {path.name: digest(path) for path in run.iterdir()}
    out = tmp_path / 'dense'
    report = module.densify_image_poses(run, video, model, out, stride=stride)
    evidence = json.loads((out / 'evidence.json').read_text())
    by_frame = {row['frame']: row for row in evidence['frames']}
    assert report['new_inference_calls'] == len(calls) == len(expected_new)
    assert report['reused_observations'] == 6 and report['decoded_frames'] == 12
    assert report['whole_body_quality_passes'] is False
    assert report['whole_body_quality_error']
    assert detector.closed
    assert [json.loads(line)['frame'] for line in (out/'new-observations.jsonl').read_text().splitlines()] == expected_new
    assert set(by_frame) == set(range(0, 12, 2)) | set(expected_new)
    assert all(by_frame[row['frame']] == row for row in original['frames'])
    assert all(row['time_s'] == row['frame']/30 for row in by_frame.values())
    assert evidence['inference_stride'] == (1 if stride == 1 else None)
    assert before == {path.name: digest(path) for path in run.iterdir()}


@pytest.mark.parametrize('damage', ['video_mode', 'tracking', 'version', 'model', 'identity', 'clock',
                                  'id', 'duplicate', 'count', 'source', 'schema', 'crop_refinement'])
def test_cache_mismatches_reject_before_inference(cached_video, damage):
    _, _, _, evidence, provenance, _, _ = cached_video
    evidence = deepcopy(evidence)
    source = evidence['source_sha256']
    if damage == 'video_mode': evidence['model']['running_mode'] = 'VIDEO'
    elif damage == 'tracking': evidence['model']['temporal_tracking'] = True
    elif damage == 'version': evidence['model']['mediapipe_version'] = 'changed'
    elif damage == 'model': evidence['model']['sha256'] = 'b'*64
    elif damage == 'identity': evidence['pose_identity']['mode'] = 'temporal'
    elif damage == 'clock': evidence['frames'][1]['decoder_timestamp_s'] += .001
    elif damage == 'id': evidence['frames'][1]['id'] = 'changed'
    elif damage == 'duplicate': evidence['frames'].append(evidence['frames'][-1])
    elif damage == 'count': evidence['decoded_frames'] = 11
    elif damage == 'source': evidence['source_sha256'] = 'c'*64
    elif damage == 'schema': evidence['schema_version'] = True
    elif damage == 'crop_refinement': evidence['pose_reinference'] = {'method': 'test'}
    with pytest.raises(ValueError): module._plan(evidence, source, provenance, 'fixture', 1)


def test_existing_samples_are_not_recomputed_and_union_is_bounded(cached_video):
    _, _, _, evidence, provenance, _, _ = cached_video
    with pytest.raises(ValueError, match='already cached'):
        module._plan(evidence, evidence['source_sha256'], provenance, 'fixture', 2)
    evidence.update(frame_count=1800, decoded_frames=1800, fps=30)
    with pytest.raises(ValueError, match='exceeds 900'):
        module._plan(evidence, evidence['source_sha256'], provenance, 'fixture', 1)


def test_failed_inference_retains_partial_journal_and_closes_detector(tmp_path, cached_video, monkeypatch):
    run, video, model, _, _, _, detector = cached_video
    original_detect = detector.detect
    calls = 0
    def fail(image):
        nonlocal calls
        calls += 1
        if calls == 3: raise RuntimeError('Controlled inference failure')
        return original_detect(image)
    monkeypatch.setattr(detector, 'detect', fail)
    out = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='Controlled'):
        module.densify_image_poses(run, video, model, out)
    assert detector.closed
    report = json.loads((out/'report.json').read_text())
    assert report['status'] == 'failed_densification' and report['new_observations'] == 2
    assert [json.loads(line)['frame'] for line in (out/'new-observations.jsonl').read_text().splitlines()] == [1, 3]
    assert [row['frame'] for row in json.loads((out/'evidence.json').read_text())['frames']] == [0, 1, 2, 3, 4]
