"""Explicit support priors before a video's body-manifest identity exists."""

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re

from weft.shared_support_task import MAX_REQUIREMENTS, MAX_STEPS, SEGMENTS, _number, _vector
from weft.shared_world_human import _json
from weft.shared_world_packaging import read_package_file
from weft.shared_world_support import CONFIG_PROFILE, MAX_CONFIG_BYTES, _path, validate_authored_supports


PROFILE = "shared-static-support-priors-v1"
SCOPE = ("Support geometry, material values and load windows are authored simulator hypotheses. "
         "They do not establish recovered equipment, ground, contact, grasp or physical accuracy.")


def validate_support_priors(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_CONFIG_BYTES:
        raise ValueError("Static-support priors require at most 128 KiB of exact JSON bytes")
    value = _json(raw)
    if (set(value) != {"schema_version", "profile", "source_sha256", "scale", "objects",
                       "support_requirements", "root_assistance"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["profile"] != PROFILE):
        raise ValueError("Static-support prior fields differ from the versioned authoring contract")
    if not isinstance(value["source_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["source_sha256"]):
        raise ValueError("Static-support priors require the exact original video SHA256")
    if (type(value["scale"]) not in (int, float) or not math.isfinite(value["scale"])
            or not .5 <= value["scale"] <= 2. or type(value["root_assistance"]) is not bool):
        raise ValueError("Static-support scale and root assistance must be explicit bounded priors")
    identifiers = validate_authored_supports(value["objects"])
    rows = value["support_requirements"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_REQUIREMENTS:
        raise ValueError("Static-support priors require 1–24 explicit load windows")
    windows = {}
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"object_id", "human_segment", "direction_world",
                "minimum_load_body_weight_fraction", "start_step", "end_step"}
                or not isinstance(row["object_id"], str) or row["object_id"] not in identifiers
                or not isinstance(row["human_segment"], str) or row["human_segment"] not in SEGMENTS):
            raise ValueError("Support windows require declared objects and body segments, not inferred palms, soles or ground")
        direction = _vector(row["direction_world"], 3, "Support direction")
        if not math.isclose(sum(item * item for item in direction), 1., rel_tol=0, abs_tol=1e-9):
            raise ValueError("Support direction must be an explicit unit vector")
        if _number(row["minimum_load_body_weight_fraction"], "Minimum support load", 0., 2.) <= 0:
            raise ValueError("Minimum support load must be positive")
        # The extracted body's actual horizon is checked again during task construction.
        if (type(row["start_step"]) is not int or type(row["end_step"]) is not int
                or not 0 <= row["start_step"] < row["end_step"] <= MAX_STEPS):
            raise ValueError("Support windows must fit the bounded half-open control-step envelope")
        key = row["object_id"], row["human_segment"]
        start, end = row["start_step"], row["end_step"]
        if any(start < old_end and old_start < end for old_start, old_end in windows.get(key, [])):
            raise ValueError("The same support/segment load cannot count in overlapping requirements")
        windows.setdefault(key, []).append((start, end))
    return value


def read_support_priors(path, *, expected_sha256=None):
    path = _path(path)
    raw = read_package_file(path.parent, path.name, max_bytes=MAX_CONFIG_BYTES, expected_sha256=expected_sha256)
    return validate_support_priors(raw), raw, path


def resolve_support_config(priors_raw, *, source_sha256, body_manifest_sha256):
    value = validate_support_priors(priors_raw)
    if value["source_sha256"] != source_sha256:
        raise ValueError("Static-support priors and extracted body name different original videos")
    if not isinstance(body_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", body_manifest_sha256):
        raise ValueError("Static supports require the actual completed body-source manifest identity")
    return {**deepcopy(value), "profile": CONFIG_PROFILE, "body_source_manifest_sha256": body_manifest_sha256}


def recheck_support_priors(path, raw):
    read_support_priors(Path(path), expected_sha256=hashlib.sha256(raw).hexdigest())
