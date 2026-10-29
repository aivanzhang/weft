"""Inspect or retrieve explicit members of the pinned public Physics 101 ZIP."""

import argparse
import io
import json
from pathlib import Path
import re
import shutil
import time
from urllib.request import Request, urlopen
import zipfile

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts


class BoundedRangeFile(io.RawIOBase):
    def __init__(self, url, size, etag, out, *, byte_limit=8*1024*1024, request_limit=20, cached=()):
        super().__init__()
        if not url.startswith('https://phys101.csail.mit.edu/') or not etag:
            raise ValueError('A pinned same-origin HTTPS archive is required')
        self.url, self.size, self.etag, self.out = url, size, etag, out
        self.byte_limit, self.request_limit = byte_limit, request_limit
        self.position = self.received = self.requests = 0
        self.cached, self.cache_hits = cached, 0
        self.log = []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ValueError('Unsupported seek origin')
        position = offset+(self.position if whence == io.SEEK_CUR else self.size if whence == io.SEEK_END else 0)
        if not 0 <= position <= self.size:
            raise ValueError('Archive seek outside pinned size')
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        if size == 0: return b''
        for start, end, path in self.cached:
            if start <= self.position and self.position+size <= end+1:
                with path.open('rb') as saved:
                    saved.seek(self.position-start)
                    data = saved.read(size)
                if len(data) != size: raise ValueError('Saved range is incomplete')
                self.cache_hits += 1
                self.position += size
                return data
        if size > self.byte_limit-self.received or self.requests >= self.request_limit:
            raise ValueError('Archive metadata/member request exceeds the declared transfer bound')
        start, end = self.position, self.position+size-1
        request = Request(self.url, headers={'User-Agent': 'Weft-local-research/1.0',
            'Range': f'bytes={start}-{end}', 'If-Range': self.etag, 'Accept-Encoding': 'identity'})
        self.requests += 1
        tick = time.monotonic()
        with urlopen(request, timeout=15) as response:
            # Reject ignored ranges before reading a multi-gigabyte response body.
            if (response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}'
                    or response.headers.get('ETag') != self.etag):
                raise ValueError('Server did not honor the exact pinned archive range')
            data = response.read(size+1)
            self.received += len(data)
            if len(data) != size:
                raise ValueError('Archive range length differs from its declaration')
        name = f'range-{self.requests:02d}.bin'
        (self.out/name).write_bytes(data)
        self.log.append({'start': start, 'end_inclusive': end, 'bytes': len(data),
                         'file': name, 'sha256': digest(self.out/name), 'seconds': time.monotonic()-tick})
        self.position += size
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--member', action='append', default=[])
    parser.add_argument('--metadata-run', type=Path, help='Reuse byte-pinned archive ranges from a sealed metadata inspection')
    args = parser.parse_args()
    source = Path('artifacts/phys101-source-access-02')
    source_verification = verify_artifacts(source)
    metadata = json.loads((source/'report.json').read_text())
    size = int(metadata['archive_headers']['Content-Length'])
    etag = metadata['archive_headers']['ETag']
    cached, cache_verification = [], None
    if args.metadata_run:
        cache_root = args.metadata_run.resolve()
        cache_verification = verify_artifacts(cache_root)
        cache_pre = json.loads((cache_root/'pre-run.json').read_text())
        if any(cache_pre[k] != v for k, v in [('archive_url', metadata['archive_url']), ('archive_bytes', size), ('etag', etag)]):
            raise ValueError('Saved ranges belong to a different archive version')
        cache_report = json.loads((cache_root/'report.json').read_text())
        if len(cache_report['ranges']) > 20: raise ValueError('Saved range inventory exceeds bounds')
        for row in cache_report['ranges']:
            path = (cache_root/row['file']).resolve()
            if (not path.is_relative_to(cache_root) or path.stat().st_size != row['bytes']
                    or row['end_inclusive']-row['start']+1 != row['bytes'] or digest(path) != row['sha256']):
                raise ValueError('Saved range does not match its pinned bytes')
            cached.append((row['start'], row['end_inclusive'], path))
    out = args.out.resolve()
    if out.is_relative_to(source.resolve()): raise ValueError('Output must not alter source metadata')
    if args.metadata_run and out.is_relative_to(args.metadata_run.resolve()):
        raise ValueError('Output must not alter cached metadata')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'source_verification': source_verification,
        'archive_url': metadata['archive_url'], 'archive_bytes': size, 'etag': etag,
        'transfer_limit_bytes': 8*1024*1024, 'request_limit': 20, 'timeout_seconds_each': 15,
        'uncompressed_member_limit_bytes': 6*1024*1024, 'maximum_members': 4,
        'requested_members': args.member, 'paid_requests': 0, 'automatic_retries': 0,
        'cached_metadata': str(args.metadata_run.resolve()) if args.metadata_run else None,
        'cache_verification': cache_verification,
        'scope': 'Public research sample, no complete dataset download or license acceptance'})
    remote = BoundedRangeFile(metadata['archive_url'], size, etag, out, cached=cached)
    report = {'created_at': local_time(), 'paid_requests': 0, 'full_archive_downloaded': False}
    try:
        with zipfile.ZipFile(remote) as archive:
            members = archive.infolist()
            if len(members) > 25000: raise ValueError('Archive inventory exceeds expected dataset bound')
            inventory = [{'name': row.filename, 'bytes': row.file_size, 'compressed_bytes': row.compress_size,
                          'crc32': row.CRC, 'compression': row.compress_type, 'header_offset': row.header_offset}
                         for row in members]
            write_json(out/'inventory.json', inventory)
            selected = args.member or [row.filename for row in members
                if not row.is_dir() and row.filename.lower().endswith(('readme.txt', 'readme.md'))
                and len(Path(row.filename).parts) <= 2]
            if not 1 <= len(selected) <= 4 or len(set(selected)) != len(selected):
                raise ValueError('Choose one to four distinct explicit members')
            extracted = []
            for index, name in enumerate(selected):
                info = archive.getinfo(name)
                if (info.is_dir() or info.flag_bits & 1 or info.file_size > 6*1024*1024
                        or info.compress_size > 6*1024*1024 or info.compress_type not in (0, 8)):
                    raise ValueError('Member exceeds size or compression bounds')
                with archive.open(info) as member:
                    data = member.read(6*1024*1024+1)
                if len(data) != info.file_size: raise ValueError('Member length differs from its inventory')
                safe = re.sub(r'[^A-Za-z0-9_.-]', '_', Path(name).name)
                path = out/f'member-{index:02d}-{safe}'
                path.write_bytes(data)
                extracted.append({'member': name, 'file': path.name, 'bytes': len(data), 'sha256': digest(path)})
            report.update(status='selected_members_saved', inventory_members=len(inventory), extracted=extracted)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, KeyError) as error:
        report.update(status='failure_retained', error=str(error))
    finally:
        remote.close()
    if verify_artifacts(source) != source_verification: raise ValueError('Source metadata changed')
    if args.metadata_run and verify_artifacts(args.metadata_run) != cache_verification:
        raise ValueError('Cached metadata changed')
    report.update(range_requests=remote.requests, received_bytes=remote.received, ranges=remote.log, cache_hits=remote.cache_hits)
    write_json(out/'report.json', report)
    print(json.dumps({'report': {k: v for k, v in report.items() if k != 'ranges'},
                      'verification': seal_suite(out)}, indent=2))
    if report['status'] != 'selected_members_saved': raise SystemExit(1)


if __name__ == '__main__':
    main()
