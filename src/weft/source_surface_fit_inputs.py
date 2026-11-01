"""Exact optional surface inputs for the existing bounded body-proposal worker."""

from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import time


PROFILE = "source-surface-fit-config-v1"
CONFIG_MEMBER = "inputs/surface-config.json"
MAX_CONFIG_BYTES, MAX_FILE_BYTES, MAX_INPUT_BYTES = 64 << 10, 8 << 20, 16 << 20
OPTIONAL_MEMBERS = {
    "camera": "inputs/surface-camera.json",
    "surface_intervals": "inputs/surface-intervals.json",
    "annotations": "inputs/surface-annotations.json",
    "body_binding": "inputs/surface-body-binding.json",
}
RUNTIME_MODULES = (
    "source_surface_fit_inputs", "source_surface_envelopes", "source_surface_interval_construction",
    "source_surface_anchors", "camera_pose", "camera_ground", "world_alignment", "world_evidence", "observations",
)
CAMERA_RUNTIME_MODULES = (
    "camera_body_fit", "camera_shared_contract", "camera_body", "camera_reference",
    "multiview_root", "multiview_body", "multiview_pose", "multiview_segments",
)
FIT_SETTINGS = {"maximum_evaluations": 20000, "maximum_journal_bytes": 4 << 20}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _require(value, message):
    if not value:
        raise ValueError("Surface fit inputs: " + message)


def _json(raw):
    def unique(items):
        value = {}
        for key, item in items:
            _require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    try:
        value = json.loads(raw, object_pairs_hook=unique)
        json.dumps(value, allow_nan=False)
    except (ValueError, RecursionError) as error:
        raise ValueError("Surface fit inputs: finite strict JSON required") from error
    _require(type(value) is dict, "JSON object required")
    return value


def _entry(value):
    _require(type(value) is dict and set(value) == {"path", "sha256"}, "each input needs a path and exact SHA256")
    name, digest = value["path"], value["sha256"]
    _require(type(name) is str and 0 < len(name) <= 256 and "\\" not in name and ":" not in name
             and all(ord(c) >= 32 for c in name), "invalid relative input path")
    parts = name.split("/")
    _require(len(parts) <= 8 and all(p not in ("", ".", "..") for p in parts)
             and not PurePosixPath(name).is_absolute(), "inputs must be beneath the config directory")
    _require(type(digest) is str and _SHA.fullmatch(digest), "exact input SHA256 required")


def parse_surface_fit_config(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_CONFIG_BYTES, "config must be at most 64 KiB")
    config = _json(raw)
    _require(set(config) == {"schema_version", "profile", "source_sha256", "weight", "observations", *OPTIONAL_MEMBERS},
             "unexpected config fields")
    _require(type(config["schema_version"]) is int and config["schema_version"] == 1 and config["profile"] == PROFILE,
             "versioned config required")
    _require(type(config["source_sha256"]) is str and _SHA.fullmatch(config["source_sha256"]), "exact source SHA256 required")
    weight = config["weight"]
    _require(type(weight) in (int, float) and math.isfinite(weight) and 0 < weight <= 1000,
             "explicit surface weight must be greater than zero and at most 1000")
    observations = config["observations"]
    _require(type(observations) is list and 1 <= len(observations) <= 8, "one to eight exact observation documents required")
    for entry in observations:
        _entry(entry)
    _require(len({e["sha256"] for e in observations}) == len(observations), "duplicate observation documents")
    for role in OPTIONAL_MEMBERS:
        if config[role] is not None:
            _entry(config[role])
    return config


def _members(config):
    return {**{f"inputs/surface-observation-{i:02d}.json": entry for i, entry in enumerate(config["observations"])},
            **{name: config[role] for role, name in OPTIONAL_MEMBERS.items() if config[role] is not None}}


def _descriptor(config, raw):
    return {"profile": PROFILE, "config_sha256": sha256(raw).hexdigest(), "source_sha256": config["source_sha256"],
            "weight": config["weight"], "observation_count": len(config["observations"]),
            "optional_inputs": [role for role in OPTIONAL_MEMBERS if config[role] is not None]}


def surface_fit_member_names(descriptor):
    _require(type(descriptor) is dict and set(descriptor) == {
        "profile", "config_sha256", "source_sha256", "weight", "observation_count", "optional_inputs"}, "invalid frozen surface descriptor")
    _require(descriptor["profile"] == PROFILE, "invalid frozen surface profile")
    for key in ("config_sha256", "source_sha256"):
        _require(type(descriptor[key]) is str and _SHA.fullmatch(descriptor[key]), "invalid frozen surface identity")
    count, roles, weight = descriptor["observation_count"], descriptor["optional_inputs"], descriptor["weight"]
    _require(type(count) is int and 1 <= count <= 8, "invalid frozen observation count")
    _require(type(roles) is list and all(type(role) is str and role in OPTIONAL_MEMBERS for role in roles)
             and roles == [role for role in OPTIONAL_MEMBERS if role in roles], "invalid frozen optional inputs")
    _require(type(weight) in (int, float) and math.isfinite(weight) and 0 < weight <= 1000, "invalid frozen surface weight")
    return {CONFIG_MEMBER, *(f"inputs/surface-observation-{i:02d}.json" for i in range(count)),
            *(OPTIONAL_MEMBERS[role] for role in roles)}


def validate_surface_fit_snapshot(files, descriptor, *, source_sha256=None):
    names = surface_fit_member_names(descriptor)
    _require(names <= set(files), "missing frozen surface input")
    _require(all(type(files[name]) is bytes and 0 < len(files[name]) <= MAX_FILE_BYTES for name in names)
             and sum(len(files[name]) for name in names) <= MAX_INPUT_BYTES, "surface inputs exceed their byte bound")
    raw = files[CONFIG_MEMBER]
    config = parse_surface_fit_config(raw)
    _require(descriptor == _descriptor(config, raw), "config differs from its frozen descriptor")
    if source_sha256 is not None:
        _require(config["source_sha256"] == source_sha256, "surface source differs from the original body")
    for name, entry in _members(config).items():
        _require(sha256(files[name]).hexdigest() == entry["sha256"], "surface input differs from its exact pin: " + name)
    _require(sum(len(files[name]) for role, name in OPTIONAL_MEMBERS.items()
                 if role != "body_binding" and name in files) <= 4 << 20, "camera/surface/annotation inputs exceed 4 MiB")
    binding = files.get(OPTIONAL_MEMBERS["body_binding"])
    _require(binding is None or len(binding) <= 64 << 10, "body binding exceeds 64 KiB")
    return config


def prepare_surface_fit_inputs(config_path, *, source_sha256, deadline):
    from weft.shared_world_packaging import read_package_file

    path = Path(config_path).absolute()
    _require(len(path.parts) <= 64 and ".." not in path.parts
             and not any(part.is_symlink() for part in (path, *path.parents)), "config path must not traverse symlinks")
    raw = read_package_file(path.parent, path.name, max_bytes=MAX_CONFIG_BYTES)
    config = parse_surface_fit_config(raw)
    _require(config["source_sha256"] == source_sha256, "surface source differs from the original body")
    files, total = {CONFIG_MEMBER: raw}, len(raw)
    for member, entry in _members(config).items():
        if time.monotonic() >= deadline:
            raise TimeoutError("Surface input snapshot exceeded its deadline")
        value = read_package_file(path.parent, entry["path"], max_bytes=min(MAX_FILE_BYTES, MAX_INPUT_BYTES - total),
                                  expected_sha256=entry["sha256"])
        files[member], total = value, total + len(value)
    descriptor = _descriptor(config, raw)
    validate_surface_fit_snapshot(files, descriptor, source_sha256=source_sha256)
    return files, descriptor


def prepare_surface_fit_objective(files, descriptor, *, camera_image_scale=None):
    from weft.source_surface_envelopes import prepare_source_surface_envelopes
    from weft.source_surface_interval_construction import prepare_surface_interval_objective

    config = validate_surface_fit_snapshot(files, descriptor)
    camera_options = {}
    if camera_image_scale is not None:
        _require(type(camera_image_scale) in (int, float) and math.isfinite(camera_image_scale)
                 and 1e-6 <= camera_image_scale <= 1, "camera image scale must be explicit within [1e-6,1]")
        _require(OPTIONAL_MEMBERS["camera"] in files, "camera fitting requires the exact supplied camera")
        camera_options = {"camera_document": files[OPTIONAL_MEMBERS["camera"]], "camera_image_scale": camera_image_scale}
    body = files["inputs/human.json"]
    _require(_json(body).get("source_sha256") == config["source_sha256"], "surface source differs from the original body")
    documents = {entry["sha256"]: files[name] for name, entry in _members(config).items()
                 if name.startswith("inputs/surface-observation-")}
    envelopes = prepare_source_surface_envelopes(documents, source_sha256=config["source_sha256"],
        camera_document=files.get(OPTIONAL_MEMBERS["camera"]),
        surface_interval_document=files.get(OPTIONAL_MEMBERS["surface_intervals"]),
        annotation_document=files.get(OPTIONAL_MEMBERS["annotations"]))
    objective = prepare_surface_interval_objective(envelopes, body_bytes=body,
        binding_bytes=files.get(OPTIONAL_MEMBERS["body_binding"]),
        **({"include_image_admission": True} if camera_image_scale is not None else {}))
    return {"surface_interval_objective": objective, "surface_interval_weight": config["weight"],
            "surface_interval_settings": dict(FIT_SETTINGS), **camera_options}
