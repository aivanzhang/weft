from pathlib import Path
from urllib.request import Request, urlopen

from weft.io import digest, local_time, write_json

SAMPLE = {
    "title": "Rollender Ball",
    "author": "Ecole Science (provider attribution; individual author not listed)",
    "page_url": "https://hubbs.schule/mediathek/rollender-ball",
    "media_url": "https://download.sodix.de/api/content/data/SODIX-0001130497.mp4",
    "license": "CC0-1.0",
    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "license_evidence": "HubbS lists Nutzungsrechte: CC0 and Ecole Science as provider.",
    "sha256": "75621f8624ce677336f9bbd6c110b638063d0f75e1678460d85602026e4d3c86",
    "bytes": 293634,
    "modifications": "Original input unchanged; generated overlays, crops, and simulations are labeled derivatives.",
}


def fetch_sample(destination):
    path = Path(destination)
    if path.exists():
        if digest(path) != SAMPLE["sha256"]:
            raise ValueError("Existing sample hash differs; refusing to overwrite it")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(SAMPLE["media_url"], headers={"User-Agent": "WeftResearch/0.1"})
    with urlopen(request, timeout=30) as response:
        body = response.read(SAMPLE["bytes"] + 1)
    from hashlib import sha256
    if len(body) != SAMPLE["bytes"] or sha256(body).hexdigest() != SAMPLE["sha256"]:
        raise ValueError("Remote sample changed or exceeds its pinned length")
    with path.open("xb") as stream:
        stream.write(body)
    return path


def record_source(path, out):
    actual_hash = digest(path)
    record = dict(SAMPLE) if actual_hash == SAMPLE["sha256"] else {
        "title": Path(path).name,
        "license": "user-supplied; not independently established",
        "page_url": None,
    }
    record.update(sha256=actual_hash, bytes=Path(path).stat().st_size, inspected_at=local_time())
    write_json(Path(out) / "source.json", record)
    return record
