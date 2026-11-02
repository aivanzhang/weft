"""Compare complete changed-model pose branches without selecting frames or weakening gates."""

from collections import Counter
from hashlib import sha256
import math

from weft.exercise_reconstruction_readiness import _inputs
from weft.exercise_vision import MODEL_VARIANTS, USABLE_STATUSES, _quality
from weft.io import canonical_hash


PROFILE = "complete-pose-model-comparison-v1"


def _require(condition, message):
    if not condition:
        raise ValueError("Pose model comparison: " + message)


def _quality_and_model(document):
    model = document.get("model", {})
    _require(isinstance(model, dict), "retained model provenance required")
    variant = next((name for name, entry in MODEL_VARIANTS.items()
                    if all(model.get(key) == value for key, value in entry.items())), None)
    _require(variant is not None, "exact supported model identity required")
    _require(model.get("running_mode") == "IMAGE" and model.get("temporal_tracking") is False
             and type(model.get("maximum_pose_hypotheses")) is int and model["maximum_pose_hypotheses"] == 2
             and model.get("delegate") == "CPU" and model.get("network_inference") is False,
             "compare complete independent IMAGE observations with unchanged two-hypothesis capacity")
    _require(document.get("pose_identity", {}).get("mode") == "raw", "retain original raw pose identity")
    stride, decoded = document.get("inference_stride"), document["decoded_frames"]
    _require(type(stride) is int and 1 <= stride <= 120
             and [row["frame"] for row in document["frames"]] == list(range(0, decoded, stride)),
             "complete original sampling required; no frame selection")
    try:
        quality = _quality(document)
    except (KeyError, IndexError, TypeError, ZeroDivisionError) as error:
        raise ValueError("Pose model comparison requires original pose geometry and confidence") from error
    _require(canonical_hash(document.get("quality")) == canonical_hash(quality), "saved quality differs from unchanged gates")
    return variant, quality


def compare_pose_models(baseline_bytes, candidate_bytes):
    """Return exact branch identities and transitions; neither branch is mutated or merged."""
    baseline, candidate = _inputs(baseline_bytes), _inputs(candidate_bytes)
    fields = ("source_sha256", "width", "height", "fps", "frame_count", "decoded_frames", "inference_stride")
    _require(all(baseline.get(key) == candidate.get(key) for key in fields), "source, geometry or complete sampling changed")
    old_model, old_quality = _quality_and_model(baseline)
    new_model, new_quality = _quality_and_model(candidate)
    _require(old_model != new_model, "unchanged model retries are not a changed reconstruction hypothesis")
    _require(baseline["model"].get("mediapipe_version") == candidate["model"].get("mediapipe_version"),
             "model comparison must retain the detector runtime")
    rows, transitions = [], Counter()
    for old, new in zip(baseline["frames"], candidate["frames"], strict=True):
        _require(all(old.get(key) == new.get(key) for key in ("id", "frame", "time_s", "decoder_timestamp_s")),
                 "candidate changed a source frame, nominal time or decoder time")
        stamp = old.get("decoder_timestamp_s")
        _require(stamp is None or type(stamp) in (int, float) and math.isfinite(stamp), "finite retained decoder history required")
        old_usable, new_usable = old["status"] in USABLE_STATUSES, new["status"] in USABLE_STATUSES
        transition = ("retained_usable" if old_usable else "recovered_usable") if new_usable else (
            "lost_usable" if old_usable else "still_unusable")
        transitions[transition] += 1
        rows.append({"frame": old["frame"], "evidence_id": old["id"], "time_s": old["time_s"],
            "decoder_timestamp_s": stamp, "baseline_status": old["status"], "candidate_status": new["status"],
            "baseline_hypothesis_count": old["pose_count"], "candidate_hypothesis_count": new["pose_count"],
            "baseline_observation_sha256": canonical_hash(old), "candidate_observation_sha256": canonical_hash(new),
            "transition": transition, "selected_for_merged_output": False})
    return {"schema_version": 1, "profile": PROFILE, "source_sha256": baseline["source_sha256"],
        "baseline_evidence_sha256": sha256(baseline_bytes).hexdigest(), "candidate_evidence_sha256": sha256(candidate_bytes).hexdigest(),
        "baseline_model": {"variant": old_model, "sha256": baseline["model"]["sha256"]},
        "candidate_model": {"variant": new_model, "sha256": candidate["model"]["sha256"]},
        "source_frames": baseline["decoded_frames"], "sampled_frames": len(rows),
        "baseline_quality": old_quality, "candidate_quality": new_quality,
        "net_usable_samples": new_quality["tracked_frames"] - old_quality["tracked_frames"],
        "transitions": dict(transitions), "rows": rows,
        "decision": "candidate_requires_independent_body_construction" if new_quality["passes"] else "candidate_remains_source_quality_rejected",
        "candidate_pose_quality_passes": new_quality["passes"], "body_admitted": False,
        "merged_frames": 0, "physical_validation": "unverified", "contact_state": "unknown",
        "scope": "Complete alternative model observation, not framewise selection, measured anatomy, recovered camera motion or verified contact. Original construction and simulation checks remain required."}
