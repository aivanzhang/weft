from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def local_time():
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()
