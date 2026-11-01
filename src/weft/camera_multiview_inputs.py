"""Exact optional second-view inputs for the existing body-fitting worker."""

from hashlib import sha256
import json
import math
from pathlib import Path
import time

from weft.source_surface_fit_inputs import _entry, _json, _SHA


PROFILE = "camera-secondary-view-fit-config-v1"
CONFIG_MEMBER = "inputs/camera-secondary-config.json"
CAMERA_MEMBER = "inputs/secondary-camera.json"
EVIDENCE_MEMBER = "inputs/secondary-evidence.json"
ALIGNMENT_MEMBER = "inputs/camera-secondary-alignment.json"
MAX_CONFIG_BYTES, MAX_FILE_BYTES, MAX_INPUT_BYTES = 64 << 10, 8 << 20, 16 << 20
MAX_ALIGNMENT_BYTES = 16 << 10
RUNTIME_MODULES = ("camera_multiview_inputs", "camera_multiview_fit")


def _require(condition, message):
    if not condition:
        raise ValueError("Secondary camera inputs: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Secondary camera input snapshot exceeded its deadline")


def _canonical(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _hash(value):
    return type(value) is str and _SHA.fullmatch(value) is not None


def _scales(scale, weight):
    _require(type(scale) in (int, float) and math.isfinite(scale) and 1e-6 <= scale <= 1,
             "explicit image scale must be within [1e-6,1]")
    _require(type(weight) in (int, float) and math.isfinite(weight) and 0 < weight <= 1000,
             "explicit authored weight must be within (0,1000]")


def _correspondence(value):
    _require(type(value) is dict and set(value) == {
        "primary_entity_ref", "secondary_entity_ref", "status", "method", "uncertainty"},
        "explicit same-person correspondence required")
    _require(value["status"] == "assumed", "correspondence must remain an authored assumption")
    for key in ("method", "uncertainty"):
        _require(type(value[key]) is str and 0 < len(value[key].strip()) <= 2000,
                 "bounded correspondence method and uncertainty required")
    for key in ("primary_entity_ref", "secondary_entity_ref"):
        ref = value[key]
        _require(type(ref) is dict and set(ref) == {"document_sha256", "object_id"}
                 and _hash(ref["document_sha256"]) and type(ref["object_id"]) is str
                 and 0 < len(ref["object_id"]) <= 80, "exact qualified entity reference required")


def parse_camera_multiview_config(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_CONFIG_BYTES, "config exceeds 64 KiB")
    config = _json(raw)
    _require(set(config) == {"schema_version", "profile", "primary_source_sha256", "secondary_source_sha256",
        "observations", "world_alignment", "correspondence", "image_scale", "weight"}, "unexpected config fields")
    _require(type(config["schema_version"]) is int and config["schema_version"] == 1
             and config["profile"] == PROFILE, "versioned config required")
    _require(all(_hash(config[key]) for key in ("primary_source_sha256", "secondary_source_sha256"))
             and config["primary_source_sha256"] != config["secondary_source_sha256"],
             "two distinct exact source identities required")
    _scales(config["image_scale"], config["weight"])
    _correspondence(config["correspondence"])
    observations = config["observations"]
    _require(type(observations) is list and 1 <= len(observations) <= 8, "one to eight observation documents required")
    for entry in observations:
        _entry(entry)
    _require(len({entry["sha256"] for entry in observations}) == len(observations), "duplicate observation documents")
    if config["world_alignment"] is not None:
        _entry(config["world_alignment"])
    return config


def _members(config):
    return {**{f"inputs/camera-secondary-observation-{index:02d}.json": entry
               for index, entry in enumerate(config["observations"])},
            **({ALIGNMENT_MEMBER: config["world_alignment"]} if config["world_alignment"] is not None else {})}


def _descriptor(config, raw, files):
    return {"profile": PROFILE, "config_sha256": sha256(raw).hexdigest(),
        "primary_source_sha256": config["primary_source_sha256"], "secondary_source_sha256": config["secondary_source_sha256"],
        "camera_sha256": sha256(files[CAMERA_MEMBER]).hexdigest(), "evidence_sha256": sha256(files[EVIDENCE_MEMBER]).hexdigest(),
        "observation_sha256s": [entry["sha256"] for entry in config["observations"]],
        "world_alignment_sha256": config["world_alignment"]["sha256"] if config["world_alignment"] else None,
        "correspondence_sha256": _canonical(config["correspondence"]),
        "image_scale": config["image_scale"], "weight": config["weight"]}


def camera_multiview_member_names(descriptor):
    _require(type(descriptor) is dict and set(descriptor) == {"profile", "config_sha256", "primary_source_sha256",
        "secondary_source_sha256", "camera_sha256", "evidence_sha256", "observation_sha256s",
        "world_alignment_sha256", "correspondence_sha256", "image_scale", "weight"}, "invalid frozen descriptor")
    _require(descriptor["profile"] == PROFILE, "invalid frozen profile")
    _require(all(_hash(descriptor[key]) for key in ("config_sha256", "primary_source_sha256", "secondary_source_sha256",
        "camera_sha256", "evidence_sha256", "correspondence_sha256")), "exact frozen identities required")
    _require(descriptor["primary_source_sha256"] != descriptor["secondary_source_sha256"], "distinct sources required")
    observations = descriptor["observation_sha256s"]
    _require(type(observations) is list and 1 <= len(observations) <= 8 and all(map(_hash, observations))
             and len(set(observations)) == len(observations), "invalid frozen observation identities")
    alignment = descriptor["world_alignment_sha256"]
    _require(alignment is None or _hash(alignment), "invalid frozen world-alignment identity")
    _scales(descriptor["image_scale"], descriptor["weight"])
    return {CONFIG_MEMBER, CAMERA_MEMBER, EVIDENCE_MEMBER,
            *(f"inputs/camera-secondary-observation-{index:02d}.json" for index in range(len(observations))),
            *({ALIGNMENT_MEMBER} if alignment is not None else set())}


def validate_camera_multiview_snapshot(files, descriptor, *, source_sha256=None):
    names = camera_multiview_member_names(descriptor)
    _require(type(files) is dict and names <= set(files), "missing frozen secondary input")
    _require(all(type(files[name]) is bytes and 0 < len(files[name]) <= MAX_FILE_BYTES for name in names)
             and sum(len(files[name]) for name in names) <= MAX_INPUT_BYTES, "secondary inputs exceed their byte bound")
    raw = files[CONFIG_MEMBER]
    config = parse_camera_multiview_config(raw)
    _require(descriptor == _descriptor(config, raw, files), "config or source copies differ from their frozen descriptor")
    if source_sha256 is not None:
        _require(config["primary_source_sha256"] == source_sha256, "primary source differs from the original body")
    for name in (CAMERA_MEMBER, EVIDENCE_MEMBER):
        _require(_json(files[name]).get("source_sha256") == config["secondary_source_sha256"],
                 "secondary source differs from its original camera or evidence")
    for name, entry in _members(config).items():
        _require(sha256(files[name]).hexdigest() == entry["sha256"], "input differs from its exact pin: " + name)
    _require(ALIGNMENT_MEMBER not in names or len(files[ALIGNMENT_MEMBER]) <= MAX_ALIGNMENT_BYTES,
             "world alignment exceeds 16 KiB")
    return config


def prepare_camera_multiview_inputs(config_path, *, source_files, source_sha256, deadline):
    from weft.shared_world_packaging import read_package_file

    _check(deadline)
    path = Path(config_path).absolute()
    _require(len(path.parts) <= 64 and ".." not in path.parts
             and not any(part.is_symlink() for part in (path, *path.parents)), "config must not traverse symlinks")
    raw = read_package_file(path.parent, path.name, max_bytes=MAX_CONFIG_BYTES)
    config = parse_camera_multiview_config(raw)
    _require(config["primary_source_sha256"] == source_sha256, "primary source differs from the original body")
    _require(type(source_files) is dict and all(name in source_files for name in (CAMERA_MEMBER, EVIDENCE_MEMBER)),
             "the original sealed body must supply both secondary source members")
    files = {CONFIG_MEMBER: raw, **{name: source_files[name] for name in (CAMERA_MEMBER, EVIDENCE_MEMBER)}}
    _require(all(type(value) is bytes and 0 < len(value) <= MAX_FILE_BYTES for value in files.values()),
             "source copies exceed their byte bound")
    total = sum(map(len, files.values()))
    for member, entry in _members(config).items():
        _check(deadline)
        bound = min(MAX_ALIGNMENT_BYTES if member == ALIGNMENT_MEMBER else MAX_FILE_BYTES, MAX_INPUT_BYTES - total)
        _require(bound > 0, "secondary inputs exceed their byte bound")
        value = read_package_file(path.parent, entry["path"], max_bytes=bound, expected_sha256=entry["sha256"])
        files[member], total = value, total + len(value)
    descriptor = _descriptor(config, raw, files)
    validate_camera_multiview_snapshot(files, descriptor, source_sha256=source_sha256)
    _check(deadline)
    return files, descriptor


def prepare_camera_multiview_options(files, descriptor):
    config = validate_camera_multiview_snapshot(files, descriptor)
    _require(_json(files["inputs/human.json"]).get("source_sha256") == config["primary_source_sha256"],
             "primary source differs from the original body")
    return {"camera_secondary_view": {
        "camera_document": files[CAMERA_MEMBER], "evidence_document": files[EVIDENCE_MEMBER],
        "observation_documents": {digest: files[f"inputs/camera-secondary-observation-{index:02d}.json"]
                                  for index, digest in enumerate(descriptor["observation_sha256s"])},
        "world_alignment_document": files[ALIGNMENT_MEMBER] if config["world_alignment"] is not None else None,
        "correspondence": config["correspondence"],
        "image_scale": config["image_scale"], "weight": config["weight"]}}
