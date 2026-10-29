"""One bounded public-source acquisition; no provider calls or media transformation."""

import html
from pathlib import Path
import re
import time
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from weft.io import digest, local_time, write_json


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/general-cylinder-source-review-01'
PAGE = 'https://commons.wikimedia.org/wiki/File:Aufw%C3%A4rtsrollender_Doppelkegel.webm'
ALLOWED = {'commons.wikimedia.org', 'upload.wikimedia.org'}


def allowed(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in ALLOWED or parsed.username or parsed.password:
        raise ValueError('Only the public Wikimedia source hosts are allowed')


class Redirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        allowed(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url, path, maximum):
    allowed(url)
    start = time.monotonic()
    total = 0
    with build_opener(Redirects()).open(Request(url, headers={'User-Agent': 'WeftLocalResearch/1.0'}), timeout=20) as response:
        allowed(response.url)
        with path.open('xb') as target:
            while chunk := response.read(min(65536, maximum - total + 1)):
                total += len(chunk)
                if total > maximum or time.monotonic() - start > 40:
                    raise ValueError('Public download exceeded its byte/time bound')
                target.write(chunk)
    return {'url': url, 'bytes': total, 'sha256': digest(path), 'retrieved_at': local_time()}


def main():
    page = OUT / 'source-page.html'
    metadata = fetch(PAGE, page, 1_500_000)
    write_json(OUT / 'source-page-fetch.json', metadata)
    content = html.unescape(page.read_text())
    urls = sorted(set(re.findall(r'(?:(?:https:)?//upload\.wikimedia\.org/[^\s"<>]+)', content)))
    urls = [url if url.startswith('https:') else 'https:' + url for url in urls]
    choices = [url for url in urls if url.endswith('.240p.vp9.webm')]
    write_json(OUT / 'available-media.json', {'urls': urls, 'selected_candidates': choices})
    if len(choices) != 1:
        raise ValueError('Expected one official complete 240p VP9 transcode; no guessed URL')
    destination = ROOT / 'data/general/bicone-cylinder-240p.webm'
    source = fetch(choices[0], destination, 30_000_000 - metadata['bytes'])
    source.update(source_page=PAGE, file=destination.name, author='Rhetos',
                  license='CC BY-SA 4.0', license_url='https://creativecommons.org/licenses/by-sa/4.0/',
                  changes='Official complete 240p VP9 transcode retained unchanged; no local crop or retiming',
                  scope='Source candidate only; motion, camera, physical interpretation and suitability not yet verified')
    write_json(destination.with_suffix('.source.json'), source)
    write_json(OUT / 'acquisition.json', source | {'page': metadata, 'total_bytes': source['bytes'] + metadata['bytes'],
                                               'maximum_bytes': 30_000_000, 'new_api_calls': 0, 'cost_usd': 0})
    print(source)


if __name__ == '__main__':
    main()
