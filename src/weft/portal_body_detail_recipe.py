"""Closed local body-detail recipe; source qualification stays with package readers."""

from hashlib import sha256
import json
import re
import time

from weft.person_region_proposal import propose_person_region

PROFILE = "full-dense-region-v1"
MODEL_FILENAME = "pose_landmarker_full.task"
MODEL_SHA256 = "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"
MODEL_BYTES = 9_398_198
MAX_RECIPE_BYTES = 2 << 20
IMPLEMENTATIONS = ("portal_body_detail_recipe.py", "person_region_proposal.py", "pose_identity_windows.py")
INPUT_FILES = ("body-detail/evidence.json", "body-detail/pose-landmarks.json")
RECIPE_FILE = "body-detail/recipe.json"
ARTIFACTS = (*INPUT_FILES, RECIPE_FILE, *("implementation/" + name for name in IMPLEMENTATIONS))
_PARENT = {"project_id", "job_sha256", "status_sha256", "preflight_failure_sha256"}


def _parent(value):
    if (not isinstance(value, dict) or set(value) != _PARENT
            or any(not isinstance(v, str) or not re.fullmatch("[0-9a-f]{32}" if k == "project_id" else "[0-9a-f]{64}", v)
                   for k, v in value.items())):
        raise ValueError("Body-detail parent requires exact project/job/status/failure pins")
    return dict(value)


def _encode(value):
    result = bytearray()
    deadline = time.monotonic() + 5
    try:
        for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
            raw = chunk.encode()
            if len(result) + len(raw) > MAX_RECIPE_BYTES or time.monotonic() >= deadline:
                raise ValueError("Body-detail recipe exceeds its bounded encoding")
            result.extend(raw)
    except (TypeError, RecursionError, OverflowError) as error:
        raise ValueError("Body-detail recipe must be bounded JSON") from error
    return bytes(result)


def build_body_detail_recipe(evidence_raw, raw_poses, *, parent):
    """All original sampled hypotheses define attention, never identity or absence."""
    parent = _parent(parent)
    region = propose_person_region([(evidence_raw, raw_poses)])
    result = {"schema_version": 1, "profile": PROFILE, "parent": parent,
        "source_sha256": region["source_sha256"], "evidence_sha256": sha256(evidence_raw).hexdigest(),
        "raw_pose_sha256": sha256(raw_poses).hexdigest(), "region_proposal": region,
        "model": {"filename": MODEL_FILENAME, "sha256": MODEL_SHA256, "bytes": MODEL_BYTES},
        "inference": {"pose_mode": "video", "pose_identity": "raw", "max_pose_hypotheses": 2,
                      "pose_sampling": "all_frames", "native_timing": True},
        "scope": "Explicit new local measurement hypothesis; all original frames and quality gates remain. "
                 "The inferred region does not verify identity or absence outside it. Existing model metadata calls regions manual; "
                 "this recipe instead records their all-hypothesis derivation, not a user-drawn region. No provider call or physical qualification.",
        "inference_executed": False, "simulation_ready": False}
    _encode(result)
    return result


def validate_body_detail_recipe(recipe, evidence_raw, raw_poses, *, parent=None):
    """Reconstruct the whole recipe, including every original hypothesis contribution."""
    if not isinstance(recipe, dict):
        raise ValueError("Body-detail recipe must be an object")
    encoded = _encode(recipe)
    expected = build_body_detail_recipe(evidence_raw, raw_poses,
        parent=recipe.get("parent") if parent is None else parent)
    if encoded != _encode(expected):
        raise ValueError("Body-detail recipe differs from its complete original evidence or fixed options")
    return expected


def body_detail_options(recipe):
    """Call only after recipe validation; no caller-supplied path or ROI is accepted."""
    return {**recipe["inference"], "person_region": list(recipe["region_proposal"]["region_xyxy"])}


def validate_body_detail_body(recipe, marker, model):
    """Join the new recipe to an ordinarily validated native body, not merely its seal."""
    expected = body_detail_options(recipe)
    expected.pop("native_timing"); expected.pop("max_pose_hypotheses")
    if (marker.get("source_sha256") != recipe["source_sha256"] or marker.get("pose_options") != expected
            or marker.get("construction", {}).get("clock_profile") != "source-native-clock-v1"
            or model.get("sha256") != MODEL_SHA256 or model.get("bytes") != MODEL_BYTES):
        raise ValueError("Body-detail body differs from its fixed Full/native/dense/region recipe")
