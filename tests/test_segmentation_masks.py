from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from weft import segmentation_masks as masks
from weft.benchmark import seal_suite
from weft.io import digest, write_json
from weft.pipeline import verify_artifacts
from weft.segmentation_evidence import _geometry


PAYLOAD = '!!!!!(QO(0lu8?'
PIXELS = bytes([1,0,0,1,0, 0,0,1,1,0, 0,0,0,0,0, 1,0,1,1,1, 1,0,0,0,1])


def raw_mask(frame=0, status='returned'):
    raw = {'frame': frame, 'object_id': '1', 'status': status, 'source_dimensions': [96, 64],
           'box_xyxy_halfopen': [10, 12, 20, 32], 'box_xywh': [10, 12, 10, 20],
           'confidence': .9, 'box_confidence': .9, 'mask_confidence': .8,
           'mask_encoded': PAYLOAD, 'mask_encoding': 'one_bit', 'mask_dimensions': [5, 5],
           'mask_status': 'opaque_not_decoded_or_codec_validated'}
    if status == 'missing':
        for key in ('box_xyxy_halfopen', 'box_xywh', 'confidence', 'box_confidence', 'mask_confidence',
                    'mask_encoded', 'mask_encoding', 'mask_dimensions'):
            raw[key] = None
        raw['mask_status'] = 'not_returned'
    return raw


def evidence(source_hash='a'*64, count=5):
    result = {'profile': 'segmentation-evidence-v1', 'physics_ready': False,
              'source': {'submitted_video_sha256': source_hash, 'width_px': 96, 'height_px': 64, 'frame_count': count},
              'time': {'timestamps_s': [i/10 for i in range(count)]},
              'objects': [{'id': 'sam-1', 'raw_model_id': '1'}], 'identity_groups': [], 'observations': []}
    for i in range(count):
        raw = raw_mask(i, 'missing' if i in (0, count-1) else 'returned')
        geometry, confidence, provenance = _geometry(raw, 96, 64)
        result['observations'].append({'id': f'{source_hash}:frame:{i}:model-object:1', 'object_id': 'sam-1',
            'raw_model_id': '1', 'frame': i, 'time_s': i/10, 'original_source': None,
            'proposal_status': raw['status'], 'phase': 'pre_initialization' if i == 0 else 'initialization' if i == 1 else 'post_initialization',
            'raw_model_output': raw, 'geometry': geometry, 'confidence': confidence, 'mask_provenance': provenance,
            'identity': {'physical_identity_verified': False, 'status': 'assumed_model_identity'}})
    return result


@pytest.fixture
def bundle(tmp_path):
    video = tmp_path/'source.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (96, 64))
    assert writer.isOpened()
    for frame in range(5):
        image = np.full((64, 96, 3), frame*30, np.uint8)
        image[12:32, 10:20] = [50, 170, 230]
        writer.write(image)
    writer.release()
    data = evidence(digest(video))
    run = tmp_path/'evidence'; run.mkdir()
    write_json(run/'segmentation-evidence.json', data)
    write_json(run/'report.json', {'status': 'synthetic_fixture'})
    seal_suite(run)
    return run, video, data


def test_published_codec_example_and_immutable_raw_record():
    raw = raw_mask(); original = deepcopy(raw)
    assert masks.decode_mask(raw).tobytes() == PIXELS
    assert raw == original


@pytest.mark.parametrize('payload', [PAYLOAD[:-1], PAYLOAD+'!', PAYLOAD+'!!!!!', '!!!!!!', '~'+PAYLOAD[1:], '!{{{{{'])
def test_invalid_or_wrong_codec_payload_never_becomes_empty_mask(payload):
    from meta_sam_parser import InvalidSegmentationMaskError
    raw = raw_mask(); raw['mask_encoded'] = payload
    with pytest.raises(InvalidSegmentationMaskError):
        masks.decode_mask(raw)


@pytest.mark.parametrize('dimensions', [[0, 5], [-1, 5], [True, 5], [5.0, 5], [4097, 5], [4096, 4096], [5], None])
def test_decode_dimensions_reject_before_codec(monkeypatch, dimensions):
    monkeypatch.setattr('meta_sam_parser.decode_mask_to_raster', lambda *a: pytest.fail('Unbounded codec call'))
    raw = raw_mask(); raw['mask_dimensions'] = dimensions
    with pytest.raises(ValueError, match='bounds'):
        masks.decode_mask(raw)


def test_pixel_cells_map_to_box_not_full_frame_or_raster_dimensions():
    raster = np.array([[0, 1, 1], [0, 0, 1]], np.uint8)
    result = masks.mask_geometry(raster, [10, 20, 16, 28])
    assert result['foreground_pixels'] == 3 and result['mapped_area_px2'] == 24
    assert result['occupancy_fraction'] == .5
    assert result['centroid_px'] == pytest.approx([10+(5/3+.5)*2, 20+(1/3+.5)*4])
    assert result['foreground_bounds_px'] == [12, 20, 16, 28]
    assert result['components_8_connected'] == 1


def test_empty_mask_is_valid_but_has_no_position_or_axis():
    result = masks.mask_geometry(np.zeros((4, 6), np.uint8), [10, 20, 16, 24])
    assert result['foreground_pixels'] == result['mapped_area_px2'] == result['components_8_connected'] == 0
    assert result['centroid_px'] is result['foreground_bounds_px'] is result['principal_axis_radians'] is None


def test_shape_axis_uses_source_scale_and_abstains_for_square():
    raster = np.ones((2, 2), np.uint8)
    assert masks.mask_geometry(raster, [10, 10, 14, 14])['principal_axis_radians'] is None
    assert masks.mask_geometry(raster, [10, 10, 18, 14])['principal_axis_radians'] == pytest.approx(0)
    assert masks.mask_geometry(raster, [10, 10, 14, 18])['principal_axis_radians'] == pytest.approx(math.pi/2)


def test_disconnected_components_and_diagonal_connectivity():
    raster = np.eye(4, dtype=np.uint8)
    assert masks.mask_geometry(raster, [0, 0, 4, 4])['components_8_connected'] == 1
    raster = np.array([[1, 0, 1], [0, 0, 0]], np.uint8)
    assert masks.mask_geometry(raster, [0, 0, 3, 2])['components_8_connected'] == 2


@pytest.mark.parametrize('raster', [np.zeros((0, 2), np.uint8), np.array([[2]], np.uint8), np.ones((2, 2)), np.ones(2, np.uint8)])
def test_geometry_rejects_nonbinary_or_malformed_rasters(raster):
    with pytest.raises(ValueError):
        masks.mask_geometry(raster, [0, 0, 2, 2])


def test_palette_png_preserves_binary_bytes_and_transparency(tmp_path):
    raster = np.frombuffer(PIXELS, dtype=np.uint8).reshape(5, 5)
    path = tmp_path/'mask.png'; masks._write_raster(path, raster, (0, 215, 255))
    with Image.open(path) as image:
        assert image.mode == 'P' and image.tobytes() == PIXELS
        rgba = np.array(image.convert('RGBA'))
    np.testing.assert_array_equal(rgba[:, :, 3], raster*120)


def test_plan_retains_missing_slots_without_decoding():
    data = evidence(); original = deepcopy(data)
    source, ids, times, rows, demand = masks._plan(data)
    assert len(rows) == 5 and len(times) == 5 and ids == ['sam-1']
    assert demand['mask_pixels'] == 75
    assert data == original


@pytest.mark.parametrize('mutation', [
    lambda d: d['observations'].pop(),
    lambda d: d['observations'][1]['raw_model_output'].update(frame=2),
    lambda d: d['observations'][1]['raw_model_output'].update(object_id='2'),
    lambda d: d['observations'][1]['raw_model_output'].update(source_dimensions=[97, 64]),
    lambda d: d['observations'][1]['mask_provenance'].update(encoded_sha256='b'*64),
    lambda d: d['observations'][1]['raw_model_output'].update(mask_encoded=PAYLOAD+'!!!!!'),
    lambda d: d['observations'][1]['geometry'].update(center_px=[15., 21.]),
    lambda d: d['observations'][0]['raw_model_output'].update(mask_encoded=PAYLOAD),
    lambda d: d['observations'][2].update(time_s=.1),
    lambda d: d.update(profile='stabilized-segmentation-v1'),
])
def test_input_tampering_rejected_before_decoding(mutation):
    data = evidence(); mutation(data)
    with pytest.raises(ValueError): masks._plan(data)


@pytest.mark.parametrize('key, bound', [('maximum_frame_mask_pixels', 24), ('maximum_total_mask_pixels', 74), ('maximum_total_payload_characters', 10)])
def test_combined_work_bounds_are_enforced(monkeypatch, key, bound):
    monkeypatch.setitem(masks.CONTRACT, key, bound)
    with pytest.raises(ValueError, match='bound'): masks._plan(evidence())


def test_real_worker_exports_full_frames_masks_and_preserves_source(bundle, tmp_path):
    run, video, data = bundle; before = verify_artifacts(run)
    out = tmp_path/'decoded'
    report = masks.export_masks(run, video, out)
    assert report['paid_requests'] == 0 and report['physics_ready'] is False
    assert report['status'] == 'decoded_proposals' and report['counts'] == {'not_returned': 2, 'decoded': 3}
    assert verify_artifacts(run) == before
    result = json.loads((out/'mask-evidence.json').read_text())
    assert len(result['frames']) == len(result['observations']) == 5
    assert result['identity_groups'] == data['identity_groups']
    for raw, row in zip(data['observations'], result['observations']):
        assert all(row[k] == raw[k] for k in ('id', 'frame', 'phase', 'identity', 'original_source', 'time_s'))
        if row['raster_path']:
            with Image.open(out/row['raster_path']) as image:
                assert image.tobytes() == PIXELS
                assert sha256(image.tobytes()).hexdigest() == row['raster_sha256']
    assert len(list((out/'frames').glob('*.jpg'))) == 5
    assert 'approximate browser video seeking' in (out/'index.html').read_text()
    verify_artifacts(out)


def test_codec_failure_stays_unavailable_while_other_frames_survive(bundle, tmp_path):
    run, video, data = bundle
    data['observations'][2]['raw_model_output']['mask_encoded'] = '!!!!!!'
    raw = data['observations'][2]['raw_model_output']
    data['observations'][2]['mask_provenance'] = _geometry(raw, 96, 64)[2]
    out = tmp_path/'partial';out.mkdir()
    write_json(out/'original-evidence.json', data)
    __import__('shutil').copyfile(video, out/'source.mp4')
    report = masks._decode_run(out)
    assert report['status'] == 'partial_decode'
    assert report['counts'] == {'not_returned': 2, 'decoded': 2, 'decode_failed': 1}
    result = json.loads((out/'mask-evidence.json').read_text())
    row = result['observations'][2]
    assert row['raster_path'] is row['silhouette'] is None and row['mask_status'] == 'decode_failed'
    assert len(result['frames']) == 5


def test_export_rejects_wrong_video_and_output_reuse_before_worker(bundle, tmp_path, monkeypatch):
    run, video, data = bundle
    monkeypatch.setattr(masks, '_run_bounded', lambda *a: pytest.fail('Unexpected worker'))
    with pytest.raises(ValueError, match='new mask directory'): masks.export_masks(run, video, run/'new')
    other = tmp_path/'wrong.mp4';other.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='exact bounded'): masks.export_masks(run, other, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_worker_deadline_is_sealed_failure_not_success(bundle, tmp_path, monkeypatch):
    run, video, _ = bundle
    def timeout(*args): raise TimeoutError('synthetic worker deadline')
    monkeypatch.setattr(masks, '_run_bounded', timeout)
    out = tmp_path/'timeout'
    with pytest.raises(TimeoutError): masks.export_masks(run, video, out)
    report = json.loads((out/'report.json').read_text())
    assert report['status'] == 'failed' and report['error_type'] == 'TimeoutError'
    assert not (out/'index.html').exists()
    verify_artifacts(out)


def test_actual_worker_timeout_cleans_up_child(bundle, tmp_path, monkeypatch):
    import multiprocessing
    run, video, data = bundle
    out = tmp_path/'deadline'; out.mkdir()
    write_json(out/'original-evidence.json', data)
    __import__('shutil').copyfile(video, out/'source.mp4')
    before = {process.pid for process in multiprocessing.active_children()}
    monkeypatch.setitem(masks.CONTRACT, 'deadline_seconds', .00001)
    with pytest.raises(TimeoutError): masks._run_bounded(out)
    assert {process.pid for process in multiprocessing.active_children()} <= before


def test_source_mutation_during_decode_is_not_certified(bundle, tmp_path, monkeypatch):
    _, video, data = bundle
    out = tmp_path/'changed'; out.mkdir()
    write_json(out/'original-evidence.json', data)
    __import__('shutil').copyfile(video, out/'source.mp4')
    native = cv2.VideoCapture
    class ChangingCapture:
        def __init__(self, path): self.capture = native(path)
        def read(self):
            result = self.capture.read()
            if not result[0]:
                with (out/'source.mp4').open('ab') as stream: stream.write(b'changed')
            return result
        def get(self, prop): return self.capture.get(prop)
        def release(self): self.capture.release()
    monkeypatch.setattr(masks.cv2, 'VideoCapture', ChangingCapture)
    with pytest.raises(ValueError, match='changed during'):
        masks._decode_run(out)
    assert not (out/'mask-evidence.json').exists()


def test_direct_source_without_optional_conversion_mapping_is_preserved(bundle, tmp_path):
    _, video, data = bundle
    for row in data['observations']: row.pop('original_source')
    run = tmp_path/'direct'; run.mkdir()
    write_json(run/'segmentation-evidence.json', data)
    write_json(run/'report.json', {'status': 'direct_source_fixture'})
    seal_suite(run)
    out = tmp_path/'direct-decoded'
    result = masks.export_masks(run, video, out)
    assert result['status'] == 'decoded_proposals'
    decoded = json.loads((out/'mask-evidence.json').read_text())
    assert all('original_source' not in row for row in decoded['observations'])
    assert [row['time_s'] for row in decoded['observations']] == [row['time_s'] for row in data['observations']]
    assert decoded['source']['submitted_video_sha256'] == digest(video)
