from io import BytesIO
from contextlib import contextmanager
import json
from pathlib import Path
import time

import pytest

from weft import meta_access
from weft import meta_segmentation as sam
from weft.io import digest


def test_events_chunked_multiline_and_comments():
    body = b': heartbeat\n\nevent: example\ndata: {"type":\ndata: "example"}\n\ndata: [DONE]\n\n'
    raw = BytesIO()
    assert list(sam._events(BytesIO(body), raw, time.monotonic())) == [{'type': 'example'}]
    assert raw.getvalue() == body


@pytest.mark.parametrize('body', [b'data: {}', b'data: {}\n', b'data: broken\n\n'])
def test_rejects_partial_or_invalid_event(body):
    with pytest.raises(ValueError):
        list(sam._events(BytesIO(body), BytesIO(), time.monotonic()))


def test_event_size_and_deadline(monkeypatch):
    monkeypatch.setattr(sam, 'MAX_EVENT_BYTES', 8)
    with pytest.raises(ValueError):
        list(sam._events(BytesIO(b'data: 12345\n\n'), BytesIO(), time.monotonic()))
    with pytest.raises(TimeoutError):
        list(sam._events(BytesIO(b''), BytesIO(), time.monotonic() - 121))


@pytest.mark.parametrize('usage, expected', [(None, None), ({}, None), ({'video_frames_processed': True}, None),
    ({'video_frames_processed': -1}, None), ({'video_frames_processed': 3}, .01),
    ({'video_frames_processed': 354}, .0708)])
def test_usage_cost(usage, expected):
    if expected is None:
        assert sam._cost(usage) is None
    else:
        assert sam._cost(usage) == pytest.approx(expected)


def test_usage_price_has_no_spurious_microusd_tail():
    assert str(sam._cost({'video_frames_processed': 141})) == '0.0282'


def test_payload_only_uses_initialization_and_checked_media(tmp_path):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'fixture')
    args = dict(expected_sha256=digest(video), frame_count=354, width=1920, height=1080, roi=[970,538,108,68])
    payload = json.loads(sam.build_request(video, **args))
    assert payload['store'] is False and payload['stream'] is True
    assert payload['input'][0]['content'][0]['text'] == '<0f>1<|box;x1=970;y1=538;x2=1077;y2=605;w=1920;h=1080|>'
    video.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        sam.build_request(video, **args)


def test_connection_failure_keeps_reservation_and_never_retries(tmp_path, monkeypatch):
    meta_access.configure_access(tmp_path, total_budget_usd=.2, key='test-secret')
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'fixture')
    calls = []
    class Connection:
        def __init__(self, host, timeout):
            assert host == 'api.meta.ai' and timeout == 30
        def request(self, method, path, **kwargs):
            calls.append((method, path))
            raise TimeoutError('fixture')
        def close(self):
            pass
    @contextmanager
    def fake_network(body, key, started):
        connection = Connection('api.meta.ai', timeout=30)
        connection.request('POST', '/v1/responses', body=body,
                           headers={'Authorization': 'Bearer ' + key})
        yield connection.getresponse()
    monkeypatch.setattr(sam, '_network_response', fake_network)
    with pytest.raises(TimeoutError):
        sam.segment_video(video, tmp_path / 'result', expected_sha256=digest(video), frame_count=354,
                          width=1920, height=1080, roi=[970,538,108,68], project_root=tmp_path)
    assert calls == [('POST', '/v1/responses')]
    status = meta_access.access_status(tmp_path)
    assert status['committed_usd'] == .15 and status['unsettled_count'] == 1
    record = json.loads((tmp_path / 'result/call.json').read_text())
    assert record['status'] == 'failed' and record['cost_usd'] is None
    assert 'test-secret' not in ''.join(p.read_text() for p in (tmp_path / 'result').iterdir())


def test_empty_success_settles_reported_usage(tmp_path, monkeypatch):
    meta_access.configure_access(tmp_path, total_budget_usd=.2, key='test-secret')
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'fixture')
    lane = {'item_id': 'item1', 'output_index': 0, 'content_index': 0}
    events = [{'type': 'response.output_text.done', **lane, 'text': ''},
              {'type': 'response.completed', 'response': {'id': 'response1', 'status': 'completed',
                                                        'usage': {'video_frames_processed': 354}}}]
    class Response(BytesIO):
        status = 200
        def getheaders(self):
            return [('Content-Type', 'text/event-stream')]
        def getheader(self, key, default=None):
            return 'text/event-stream'
    class Connection:
        def __init__(self, *args, **kwargs):
            pass
        def request(self, method, path, **kwargs):
            assert kwargs['headers']['Authorization'] == 'Bearer test-secret'
        def getresponse(self):
            return Response(b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events))
        def close(self):
            pass
    @contextmanager
    def fake_network(body, key, started):
        connection = Connection('api.meta.ai', timeout=30)
        connection.request('POST', '/v1/responses', body=body,
                           headers={'Authorization': 'Bearer ' + key})
        yield connection.getresponse()
    monkeypatch.setattr(sam, '_network_response', fake_network)
    record = sam.segment_video(video, tmp_path / 'result', expected_sha256=digest(video), frame_count=354,
                               width=1920, height=1080, roi=[970,538,108,68], project_root=tmp_path)
    assert record['status'] == 'completed' and record['cost_usd'] == pytest.approx(.0708)
    status = meta_access.access_status(tmp_path)
    assert status['unsettled_count'] == 0 and status['committed_usd'] == .0708
    frames = json.loads((tmp_path / 'result/segmentation.json').read_text())['frames']
    assert len(frames) == 354 and all(f['box_xyxy_halfopen'] is None for f in frames)


def test_comparison_preserves_missing_and_unknown_slots():
    import importlib.util
    spec = importlib.util.spec_from_file_location('meta_tracking_study', Path(__file__).parents[1] / 'scripts/studies/check_meta_tracking.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frames = [{'frame': i, 'status': 'missing', 'box_xyxy_halfopen': None} for i in range(354)]
    for i in (0, 70, 140):
        frames[i].update(status='returned', box_xyxy_halfopen=[10,20,12,24])
    manual = {'observations': [{'frame': i, 'center_px': None if i == 140 else [10.5,21.5]}
                               for i in (0,70,140,210,280,353)]}
    result = module.grade(frames, manual)
    assert result['returned_center_slots'] == 1 and result['conditional_rmse_px'] == 0
    assert result['rows'][0]['scored_error_px'] is None
    assert result['rows'][2]['unknown_crossfade_has_prediction']
    assert result['rows'][-1]['scored_error_px'] is None


def test_multi_prompt_serialization_initialization_frames_and_ids(tmp_path):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'fixture')
    objects = [{'object_id': 8, 'frame': 4, 'roi': [0, 0, 1, 1]},
               {'object_id': 2, 'frame': 0, 'roi': [10, 20, 5, 6]},
               {'object_id': 1, 'frame': 0, 'roi': [5, 5, 2, 2]}]
    body = sam.build_request(video, expected_sha256=digest(video), frame_count=5, width=100, height=80, objects=objects)
    prompt = json.loads(body)['input'][0]['content'][0]['text']
    assert prompt == ('<0f>1<|box;x1=5;y1=5;x2=6;y2=6;w=100;h=80|>,'
                      '2<|box;x1=10;y1=20;x2=14;y2=25;w=100;h=80|>'
                      '<4f>8<|box;x1=0;y1=0;x2=0;y2=0;w=100;h=80|>')
    assert objects[0]['object_id'] == 8


@pytest.mark.parametrize('objects', [[], {}, [{'object_id': True, 'frame': 0, 'roi': [0,0,1,1]}],
    [{'object_id': 9, 'frame': 0, 'roi': [0,0,1,1]}],
    [{'object_id': 1, 'frame': 5, 'roi': [0,0,1,1]}],
    [{'object_id': 1, 'frame': -1, 'roi': [0,0,1,1]}],
    [{'object_id': 1, 'frame': 0, 'roi': [0,0,101,1]}],
    [{'object_id': 1, 'frame': 0, 'roi': [0,0,1,1], 'concept': 'ball'}],
    [{'object_id': 1, 'frame': 0, 'roi': [0,0,1,1]}] * 2])
def test_multi_prompt_validation_precedes_provider_access(objects):
    with pytest.raises(ValueError):
        sam._prompts(5, 100, 80, None, objects)


def test_roi_and_objects_are_mutually_exclusive():
    with pytest.raises(ValueError):
        sam._prompts(5, 100, 80, [0,0,1,1], [{'object_id': 1, 'frame': 0, 'roi': [0,0,1,1]}])


def _stalled_network(pipe, body, key):
    pipe.send(('headers', 200, [('Content-Type', 'text/event-stream')]))
    time.sleep(10)


def _trickling_network(pipe, body, key):
    pipe.send(('headers', 200, [('Content-Type', 'text/event-stream')]))
    for _ in range(100):
        pipe.send(('bytes', b': keepalive\n\n'))
        time.sleep(.02)


@pytest.mark.parametrize('worker', [_stalled_network, _trickling_network])
def test_network_process_deadline_terminates_stall_and_trickle_without_orphans(monkeypatch, worker):
    before = {child.pid for child in sam.multiprocessing.active_children()}
    monkeypatch.setattr(sam, '_network_worker', worker)
    monkeypatch.setattr(sam, 'TOTAL_SECONDS', .35)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        with sam._network_response(b'fixture', 'test-key', started) as response:
            list(sam._events(response, BytesIO(), started))
    elapsed = time.monotonic() - started
    assert elapsed < 2.35
    assert {child.pid for child in sam.multiprocessing.active_children()} <= before


def test_multi_timeout_preserves_conservative_reservation_once(tmp_path, monkeypatch):
    meta_access.configure_access(tmp_path, total_budget_usd=1, key='test-secret')
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'fixture')
    calls = []
    @contextmanager
    def failed_network(body, key, started):
        calls.append(body)
        raise TimeoutError('fixture')
        yield
    monkeypatch.setattr(sam, '_network_response', failed_network)
    with pytest.raises(TimeoutError):
        sam.segment_video(video, tmp_path / 'result', expected_sha256=digest(video), frame_count=500,
            width=100, height=80, objects=[{'object_id': i, 'frame': i, 'roi': [0,0,1,1]} for i in range(1,9)],
            project_root=tmp_path)
    assert len(calls) == 1
    result = json.loads((tmp_path / 'result/call.json').read_text())
    assert result['reserved_usd'] == pytest.approx(.8)
    assert result['cost_usd'] is None and result['request_count'] == 1
    assert len(result['initialization']) == 8
    assert meta_access.access_status(tmp_path)['committed_usd'] == pytest.approx(.8)


def test_multi_success_preserves_preinitialization_slots_and_actual_usage(tmp_path, monkeypatch):
    meta_access.configure_access(tmp_path, total_budget_usd=.5, key='test-secret')
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'fixture')
    lane = {'item_id': 'item1', 'output_index': 0, 'content_index': 0}
    text = '<2f>8<|box;x1=0;y1=0;x2=0;y2=0;w=100;h=80|><|mask;x=0;y=0;data=1,1,!abcde|>'
    events = [{'type': 'response.output_text.done', **lane, 'text': text},
              {'type': 'response.completed', 'response': {'id': 'response1', 'usage': {'video_frames_processed': 5}}}]
    class Response(BytesIO):
        status = 200
        def getheaders(self):
            return [('Content-Type', 'text/event-stream')]
        def getheader(self, key, default=None):
            return 'text/event-stream'
    @contextmanager
    def fake_network(body, key, started):
        yield Response(b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events))
    monkeypatch.setattr(sam, '_network_response', fake_network)
    prompts = [{'object_id': 1, 'frame': 0, 'roi': [1,1,1,1]}, {'object_id': 8, 'frame': 2, 'roi': [0,0,1,1]}]
    result = sam.segment_video(video, tmp_path / 'result', expected_sha256=digest(video), frame_count=5,
                               width=100, height=80, objects=prompts, project_root=tmp_path)
    parsed = json.loads((tmp_path / 'result/segmentation.json').read_text())
    assert parsed['initialization'] == prompts
    assert parsed['frames'][0]['objects'][1]['status'] == 'missing'
    assert parsed['frames'][2]['objects'][1]['status'] == 'returned'
    assert result['cost_usd'] == .01 and result['reserved_usd'] == .15
    assert meta_access.access_status(tmp_path)['unsettled_count'] == 0


def test_network_worker_forwards_raw_chunks_and_closes_once(monkeypatch):
    calls, sent = [], []
    class Response(BytesIO):
        status = 200
        def getheaders(self):
            return [('Content-Type', 'text/event-stream')]
    class Connection:
        def __init__(self, host, timeout):
            calls.append(('connect', host, timeout))
        def request(self, method, path, **kwargs):
            calls.append((method, path, kwargs['headers']['Authorization']))
        def getresponse(self):
            return Response(b'data: {}\n\n')
        def close(self):
            calls.append(('close',))
    class Pipe:
        def send(self, value):
            sent.append(value)
        def close(self):
            calls.append(('pipe_close',))
    monkeypatch.setattr(sam.http.client, 'HTTPSConnection', Connection)
    sam._network_worker(Pipe(), b'fixture', 'local-secret')
    assert calls == [('connect', 'api.meta.ai', 30), ('POST', '/v1/responses', 'Bearer local-secret'), ('close',), ('pipe_close',)]
    assert sent == [('headers', 200, [('Content-Type', 'text/event-stream')]), ('bytes', b'data: {}\n'), ('bytes', b'\n'), ('end',)]


@pytest.mark.parametrize('stage',['entry','after_payload','after_key','after_reserve'])
def test_absolute_deadline_prevents_send_after_local_preparation(tmp_path,monkeypatch,stage):
    video=tmp_path/'video.mp4';video.write_bytes(b'fixture')
    now=[0.];events=[]
    monkeypatch.setattr(sam.time,'monotonic',lambda:now[0])
    original=sam.build_request
    def build(*a,**k):
        value=original(*a,**k)
        if stage=='after_payload':now[0]=10.
        return value
    def key(*a):
        events.append('key')
        if stage=='after_key':now[0]=10.
        return 'synthetic-key'
    def reserve(*a,**k):
        events.append('reserve')
        if stage=='after_reserve':now[0]=10.
        return {'reservation_id':'synthetic'}
    monkeypatch.setattr(sam,'build_request',build);monkeypatch.setattr(sam,'resolve_key',key)
    monkeypatch.setattr(sam,'reserve_call',reserve)
    monkeypatch.setattr(sam,'settle_call',lambda *a,**k: events.append('preserved_unknown') or {})
    monkeypatch.setattr(sam,'_network_response',lambda *a,**k:pytest.fail('Expired aggregate deadline must not send'))
    with pytest.raises(TimeoutError,match='nothing sent'):
        sam.segment_video(video,tmp_path/'run',expected_sha256=digest(video),frame_count=50,width=160,height=100,
                          roi=[10,10,20,20],deadline=121 if stage=='entry' else 125)
    assert ('reserve' in events)==(stage=='after_reserve')
    if stage=='after_reserve':
        assert 'preserved_unknown' in events
        receipt=json.loads((tmp_path/'run/call.json').read_text())
        assert receipt['request_count']==0 and receipt['reservation']['reservation_id']=='synthetic'
