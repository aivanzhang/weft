"""Fetch bounded public source metadata, never the complete multi-gigabyte dataset."""

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from weft.benchmark import seal_suite
from weft.io import local_time, write_json


URL = 'https://phys101.csail.mit.edu/'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.urls.extend(urljoin(URL, value) for key, value in attrs if key == 'href' and value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'url': URL, 'maximum_html_bytes': 1_000_000,
        'maximum_requests': 2, 'timeout_seconds_each': 12, 'paid_requests': 0,
        'scope': 'Public page plus HEAD of the advertised ZIP; no archive body, account or agreement'})
    report = {'created_at': local_time(), 'url': URL, 'paid_requests': 0, 'archive_downloaded': False}
    try:
        with urlopen(Request(URL, headers={'User-Agent': 'Weft-local-research/1.0'}), timeout=12) as response:
            raw = response.read(1_000_001)
            if len(raw) > 1_000_000: raise ValueError('Source page exceeds byte bound')
            report['page'] = {'status': response.status, 'url': response.url, 'bytes': len(raw)}
            (out/'source.html').write_bytes(raw)
        links = Links(); links.feed(raw.decode('utf-8'))
        archives = [url for url in links.urls if url.endswith('.zip')]
        report['links'] = links.urls
        if len(archives) != 1 or not archives[0].startswith(URL):
            raise ValueError('Expected one same-origin advertised ZIP')
        report['archive_url'] = archives[0]
        with urlopen(Request(archives[0], method='HEAD', headers={'User-Agent': 'Weft-local-research/1.0'}), timeout=12) as response:
            report['archive_headers'] = dict(response.headers.items())
            report['archive_status'] = response.status
        report['status'] = 'public_metadata_saved'
    except (OSError, URLError, ValueError) as error:
        report.update(status='access_failed_retained', error=str(error))
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
