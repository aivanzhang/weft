"""Client cancellations stop writes without changing normal error responses."""

import errno
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from weft.lab import LabHandler
from weft.portal import PortalHandler


class FailingWriter(io.BytesIO):
    def __init__(self, error, fail_on):
        super().__init__()
        self.error = error
        self.fail_on = fail_on
        self.writes = 0

    def write(self, value):
        self.writes += 1
        if self.writes == self.fail_on:
            raise self.error
        return super().write(value)


def handler_for(handler_type, path, root, writer):
    video = root / 'run' / 'video.mp4'
    video.parent.mkdir(exist_ok=True)
    video.write_bytes(b'v' * 200_000)
    lab = SimpleNamespace(root=root, token='test-token')
    demo = SimpleNamespace(showcase_video=lambda name: video, gallery_video=lambda name: video)
    portal = SimpleNamespace(token='test-token', demo=demo, lab=lambda identity: lab)
    handler = object.__new__(handler_type)
    handler.server = SimpleNamespace(server_port=8765, lab=lab, portal=portal)
    handler.rfile = io.BytesIO((f'GET {path} HTTP/1.1\r\nHost: 127.0.0.1:8765\r\n'
                               'Authorization: Bearer test-token\r\n\r\n').encode())
    handler.wfile = writer
    return handler


@pytest.mark.parametrize('handler_type,path', [
    (LabHandler, '/files/run/video.mp4'),
    (LabHandler, '/missing'),
    (PortalHandler, '/api/demo/showcase/example/video'),
    (PortalHandler, '/api/demo/gallery/example/video'),
    (PortalHandler, '/labs/' + 'a' * 32 + '/files/run/video.mp4'),
    (PortalHandler, '/demo'),
])
@pytest.mark.parametrize('error', [BrokenPipeError(errno.EPIPE, 'closed'),
                                  ConnectionResetError(errno.ECONNRESET, 'reset')])
@pytest.mark.parametrize('fail_on', [1, 2])
def test_disconnected_headers_or_body_never_trigger_an_error_response(tmp_path, handler_type, path, error, fail_on):
    writer = FailingWriter(error, fail_on)
    handler = handler_for(handler_type, path, tmp_path, writer)
    handler.handle_one_request()
    assert handler.close_connection
    assert writer.writes == fail_on
    assert writer.getvalue().count(b'HTTP/1.0 ') <= 1


def test_stream_stops_on_a_later_chunk_and_closes_the_file(tmp_path, monkeypatch):
    writer = FailingWriter(BrokenPipeError(), 3)
    handler = handler_for(LabHandler, '/files/run/video.mp4', tmp_path, writer)
    opened = []
    original = Path.open

    def track(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        if path.name == 'video.mp4':
            opened.append(stream)
        return stream

    monkeypatch.setattr(Path, 'open', track)
    handler.handle_one_request()
    assert writer.writes == 3
    assert opened and all(stream.closed for stream in opened)


def test_real_io_failures_are_not_treated_as_client_cancellation(tmp_path):
    writer = FailingWriter(OSError(errno.EIO, 'disk or transport failure'), 1)
    handler = handler_for(LabHandler, '/missing', tmp_path, writer)
    with pytest.raises(OSError, match='disk or transport failure'):
        handler.write_body(b'payload')
    assert writer.writes == 1


def test_missing_file_still_returns_an_error_response(tmp_path):
    writer = io.BytesIO()
    handler = handler_for(LabHandler, '/files/run/missing.mp4', tmp_path, writer)
    handler.handle_one_request()
    response = writer.getvalue()
    assert response.startswith(b'HTTP/1.0 400 ')
    assert b'File is outside this lab or does not exist' in response
