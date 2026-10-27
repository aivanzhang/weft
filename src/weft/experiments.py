import json
from pathlib import Path
import re

from weft.config import RunConfig
from weft.io import digest
from weft.pipeline import run


def load_experiment(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    required = {"schema_version", "name", "hypothesis", "video", "source_sha256", "run"}
    if not isinstance(spec, dict) or set(spec) != required or type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("Experiment must use schema_version=1 and the documented fields")
    if not isinstance(spec["name"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", spec["name"]):
        raise ValueError("Experiment name must be a short lowercase slug")
    if not isinstance(spec["hypothesis"], str) or not 1 <= len(spec["hypothesis"].strip()) <= 2000:
        raise ValueError("Record a hypothesis of 1–2000 characters")
    if not isinstance(spec["source_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", spec["source_sha256"]):
        raise ValueError("Experiment requires a SHA-256 pinned source")
    if not isinstance(spec["video"], str) or not spec["video"]:
        raise ValueError("Experiment requires a local video path")
    config = RunConfig.from_dict(spec["run"])
    video = (path.parent / spec["video"]).resolve()
    if not video.is_file() or digest(video) != spec["source_sha256"]:
        raise ValueError("Experiment source is missing or its SHA-256 differs")
    return spec, config, video


def run_experiment(path, out):
    spec, config, video = load_experiment(path)
    settings = config.to_dict()
    settings["tracking"] = config.tracking
    return run(video, out, **settings, experiment=spec)
