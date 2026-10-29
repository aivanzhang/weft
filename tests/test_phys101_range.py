import io
import re
import zipfile

import pytest

from scripts.studies import phys101_range as source


URL = 'https://phys101.csail.mit.edu/data/phys101_v1.0.zip'
ETAG = '"pinned"'


class Response(io.BytesIO):
    status = 206


def server(monkeypatch, data, *, status=206, wrong_etag=False, wrong_range=False, short=False):
    reads = []

    def fetch(request, timeout):
        start, end = map(int, re.fullmatch(r'bytes=(\d+)-(\d+)', request.get_header('Range')).groups())
        assert request.get_header('If-range') == ETAG
        assert timeout == 15
        response = Response(data[start:end+1-(1 if short else 0)])
        response.status = status
        response.headers = {'ETag': '"changed"' if wrong_etag else ETAG,
                            'Content-Range': 'wrong' if wrong_range else f'bytes {start}-{end}/{len(data)}'}
        original = response.read

        def read(size):
            reads.append(size)
            return original(size)

        response.read = read
        return response

    monkeypatch.setattr(source, 'urlopen', fetch)
    return reads


def test_selected_zip_member_without_whole_file(monkeypatch, tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('video.mp4', b'v'*100000)
        archive.writestr('readme.txt', b'Physical sample metadata')
    raw = buffer.getvalue()
    server(monkeypatch, raw)
    remote = source.BoundedRangeFile(URL, len(raw), ETAG, tmp_path, byte_limit=4096)
    with zipfile.ZipFile(remote) as archive:
        assert archive.read('readme.txt') == b'Physical sample metadata'
    assert remote.received < 4096 < len(raw)
    assert remote.requests == len(remote.log)


@pytest.mark.parametrize('options', [{'status': 200}, {'wrong_etag': True}, {'wrong_range': True}])
def test_bad_response_rejected_before_body(monkeypatch, tmp_path, options):
    reads = server(monkeypatch, b'a'*100, **options)
    remote = source.BoundedRangeFile(URL, 100, ETAG, tmp_path)
    with pytest.raises(ValueError, match='exact pinned'):
        remote.read(10)
    assert reads == [] and remote.received == 0


def test_short_response_rejected(monkeypatch, tmp_path):
    server(monkeypatch, b'a'*100, short=True)
    remote = source.BoundedRangeFile(URL, 100, ETAG, tmp_path)
    with pytest.raises(ValueError, match='length differs'):
        remote.read(10)
    assert remote.tell() == 0 and remote.received == 9


@pytest.mark.parametrize('byte_limit,request_limit', [(4, 20), (100, 0)])
def test_transfer_limits_reject_before_request(monkeypatch, tmp_path, byte_limit, request_limit):
    def forbidden(*args, **kwargs): raise AssertionError('Unexpected network request')
    monkeypatch.setattr(source, 'urlopen', forbidden)
    remote = source.BoundedRangeFile(URL, 100, ETAG, tmp_path, byte_limit=byte_limit, request_limit=request_limit)
    with pytest.raises(ValueError, match='transfer bound'):
        remote.read(5)


def test_repeated_reads_count_aggregate(monkeypatch, tmp_path):
    server(monkeypatch, b'a'*100)
    remote = source.BoundedRangeFile(URL, 100, ETAG, tmp_path, byte_limit=8)
    assert remote.read(5) == b'a'*5
    remote.seek(0)
    with pytest.raises(ValueError, match='transfer bound'):
        remote.read(5)


def test_saved_ranges_skip_network_and_preserve_bounds(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs): raise AssertionError('Unexpected network request')
    monkeypatch.setattr(source, 'urlopen', forbidden)
    saved = tmp_path/'saved.bin'; saved.write_bytes(b'23456')
    remote = source.BoundedRangeFile(URL, 10, ETAG, tmp_path, request_limit=0, cached=[(2, 6, saved)])
    remote.seek(3)
    assert remote.read(3) == b'345'
    assert remote.cache_hits == 1 and remote.received == remote.requests == 0
    with pytest.raises(ValueError, match='transfer bound'):
        remote.read(2)


def test_eof_and_seek_bounds(monkeypatch, tmp_path):
    server(monkeypatch, b'1234567890')
    remote = source.BoundedRangeFile(URL, 10, ETAG, tmp_path)
    remote.seek(-3, io.SEEK_END)
    assert remote.read(100) == b'890'
    assert remote.read() == b''
    for offset, origin in [(-11, io.SEEK_END), (11, io.SEEK_SET), (-1, io.SEEK_SET)]:
        with pytest.raises(ValueError, match='outside'):
            remote.seek(offset, origin)


@pytest.mark.parametrize('url,etag', [('http://phys101.csail.mit.edu/data.zip', ETAG),
                                  ('https://example.com/data.zip', ETAG), (URL, '')])
def test_origin_and_version_binding(tmp_path, url, etag):
    with pytest.raises(ValueError, match='same-origin'):
        source.BoundedRangeFile(url, 10, etag, tmp_path)
