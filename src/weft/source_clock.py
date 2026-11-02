"""Native presentation timing is a derived view, never a rewrite of detector history."""

from copy import deepcopy
from fractions import Fraction
import json
import math
import re

from weft.io import canonical_hash

PROFILE = "source-native-clock-v1"
VIEW_PROFILE = "source-native-reconstruction-view-v1"
SUMMARY_PROFILE = "source-native-clock-summary-v1"
MAX_FRAMES = 1800
MAX_CLOCK_BYTES = 512 * 1024
MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
TIME_TOLERANCE = 1e-7
DETECTOR_CLOCK = "preserved_nominal_frame_index_divided_by_fps"
SUMMARY_FIELDS = {"schema_version", "source_sha256", "evidence_sha256", "evidence_canonical_sha256",
    "metadata_frames", "decoded_frames", "nominal_fps", "time_base", "native_origin_pts",
    "decoder_minus_native_s", "native_to_reconstruction_offset_s", "capture_clock_status", "detector_clock_status"}
BINDING_FIELDS = SUMMARY_FIELDS | {"profile", "rows"}
_SHA = re.compile(r"[a-f0-9]{64}")


def _require(condition, message):
    if not condition:
        raise ValueError("Native source clock: " + message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _bounded_json(value, maximum):
    try:
        size = len(json.dumps(value, allow_nan=False, separators=(",", ":")).encode())
    except (ValueError, TypeError, RecursionError, OverflowError) as error:
        raise ValueError("Native source clock requires bounded finite JSON") from error
    _require(size <= maximum, "JSON exceeds its byte bound")


def _time_base(value):
    _require(isinstance(value, str) and re.fullmatch(r"[1-9][0-9]{0,8}/[1-9][0-9]{0,8}", value),
             "time_base must be a bounded positive rational")
    numerator, denominator = map(int, value.split('/'))
    _require(numerator <= denominator and math.gcd(numerator, denominator) == 1,
             "time_base must be reduced and no coarser than one second")
    return Fraction(numerator, denominator)


def _metadata(value):
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1,
             "unsupported schema version")
    for key in ("source_sha256", "evidence_sha256", "evidence_canonical_sha256"):
        _require(isinstance(value[key], str) and _SHA.fullmatch(value[key]), "invalid " + key)
    for key in ("metadata_frames", "decoded_frames"):
        _require(type(value[key]) is int and 1 <= value[key] <= MAX_FRAMES, "invalid " + key)
    _require(_finite(value["nominal_fps"]) and 1 <= value["nominal_fps"] <= 240, "invalid nominal FPS")
    tick = _time_base(value["time_base"])
    _require(type(value["native_origin_pts"]) is int and abs(value["native_origin_pts"]) <= 2**53,
             "native origin requires bounded integer ticks")
    for key in ("decoder_minus_native_s", "native_to_reconstruction_offset_s"):
        _require(_finite(value[key]) and abs(value[key]) <= 3600, "invalid signed clock offset")
    _require(math.isclose(value["native_to_reconstruction_offset_s"], -float(value["native_origin_pts"] * tick),
                         abs_tol=TIME_TOLERANCE, rel_tol=0), "native-to-reconstruction offset differs from explicit origin")
    _require(value["capture_clock_status"] == "unknown" and value["detector_clock_status"] == DETECTOR_CLOCK,
             "capture/detector clocks must not be upgraded to observed native inference")
    return tick


def validate_clock_binding(binding):
    """Validate clock structure/math, not the provenance of the supplied measurements."""
    _bounded_json(binding, MAX_CLOCK_BYTES)
    _require(isinstance(binding, dict) and set(binding) == BINDING_FIELDS and binding["profile"] == PROFILE,
             "binding fields differ from the versioned contract")
    tick = _metadata(binding)
    rows = binding["rows"]
    _require(isinstance(rows, list) and len(rows) == binding["decoded_frames"], "every decoded slot needs one clock row")
    previous = None
    for index, row in enumerate(rows):
        _require(isinstance(row, dict) and set(row) == {"frame", "pts", "decoder_timestamp_s"},
                 "clock row fields differ")
        _require(type(row["frame"]) is int and row["frame"] == index, "clock slots must be complete and ordered")
        pts = row["pts"]
        _require(type(pts) is int and abs(pts) <= 2**53 and (previous is None or pts > previous),
                 "native ticks must be bounded and strictly increasing")
        _require(index != 0 or pts == binding["native_origin_pts"], "first native tick must equal the declared origin")
        stamp = row["decoder_timestamp_s"]
        _require(_finite(stamp) and -3600 <= stamp <= 3600, "every decoder timestamp must be finite")
        _require(math.isclose(stamp - float(pts * tick), binding["decoder_minus_native_s"],
                             abs_tol=TIME_TOLERANCE, rel_tol=0), "decoder/native origin drifts across frames")
        _require(0 <= float((pts - binding["native_origin_pts"]) * tick) <= 60,
                 "native elapsed time exceeds the 60-second envelope")
        previous = pts
    return deepcopy(binding)


def frame_times(binding):
    """Return exact-tick elapsed times in decoded-frame order."""
    value = validate_clock_binding(binding)
    tick = _time_base(value["time_base"])
    return [float((row["pts"] - value["native_origin_pts"]) * tick) for row in value["rows"]]


def validate_clock(binding, evidence, *, evidence_sha256=None):
    value = validate_clock_binding(binding)
    _bounded_json(evidence, MAX_EVIDENCE_BYTES)
    _require(isinstance(evidence, dict) and evidence.get("profile") == "exercise-pose-v1"
             and type(evidence.get("schema_version")) is int and evidence["schema_version"] == 1,
             "original exercise-pose-v1 evidence required")
    _require("source_clock" not in evidence, "clock derivations cannot be chained")
    _require(value["source_sha256"] == evidence.get("source_sha256")
             and value["evidence_canonical_sha256"] == canonical_hash(evidence), "original evidence identity differs")
    if evidence_sha256 is not None:
        _require(evidence_sha256 == value["evidence_sha256"], "exact evidence byte hash differs")
    _require(value["metadata_frames"] == evidence.get("frame_count")
             and value["decoded_frames"] == evidence.get("decoded_frames")
             and value["nominal_fps"] == evidence.get("fps"), "metadata/decoded counts or nominal FPS differ")
    if value["metadata_frames"] != value["decoded_frames"]:
        audit = evidence.get("container_frame_audit", {})
        encoded = (isinstance(audit, dict) and audit.get("matches") is True
                   and type(audit.get("encoded_video_frames")) is int
                   and audit["encoded_video_frames"] == value["decoded_frames"]
                   and type(audit.get("decoded_frames")) is int and audit["decoded_frames"] == value["decoded_frames"])
        discrepancy = evidence.get("decode_metadata_discrepancy", {})
        difference = value["decoded_frames"] - value["metadata_frames"]
        # Preserve the ingest contract's one-frame rounding, never normalize the counts.
        rounding = (abs(difference) == 1 and isinstance(discrepancy, dict)
                    and set(discrepancy) == {"metadata_frames", "decoded_frames", "difference", "note"}
                    and all(type(discrepancy.get(key)) is int and discrepancy[key] == expected for key, expected in
                            (("metadata_frames", value["metadata_frames"]), ("decoded_frames", value["decoded_frames"]),
                             ("difference", difference)))
                    and isinstance(discrepancy["note"], str) and 1 <= len(discrepancy["note"]) <= 500)
        _require(encoded or rounding, "metadata discrepancy requires an encoded/decoded frame audit or exact one-frame rounding receipt")
    frames, references = evidence.get("frames"), evidence.get("reference")
    _require(isinstance(frames, list) and 1 <= len(frames) <= 900 and isinstance(references, list)
             and len(references) <= len(frames), "bounded original pose rows and references required")
    indexed, previous = {}, -1
    for row in frames:
        _require(isinstance(row, dict), "pose sample must be an object")
        frame = row.get("frame")
        _require(type(frame) is int and previous < frame < value["decoded_frames"], "pose frame order or bounds differ")
        identifier = f"{value['source_sha256']}:frame:{frame}"
        _require(row.get("id") == identifier and _finite(row.get("time_s"))
                 and math.isclose(row["time_s"], frame / value["nominal_fps"], abs_tol=TIME_TOLERANCE, rel_tol=0),
                 "original detector nominal time/frame identity differs")
        decoder = row.get("decoder_timestamp_s")
        _require(decoder is None or _finite(decoder) and math.isclose(decoder, value["rows"][frame]["decoder_timestamp_s"],
                 abs_tol=TIME_TOLERANCE, rel_tol=0), "retained sampled decoder timestamp differs")
        indexed[identifier] = row
        previous = frame
    previous = -1
    for row in references:
        _require(isinstance(row, dict) and row.get("evidence_id") in indexed, "reference lacks exact sampled evidence identity")
        sample = indexed[row["evidence_id"]]
        _require(sample["frame"] > previous and _finite(row.get("time_s")) and row["time_s"] == sample["time_s"],
                 "reference clocks/order differ from original samples")
        previous = sample["frame"]
    _require("origin_time_s" not in evidence or evidence["origin_time_s"] == (references[0]["time_s"] if references else None),
             "reference origin differs from its first sample")
    _require(not {"duration_s", "duration", "end_time_s"} & evidence.keys(),
             "duration-bearing pose contracts need an explicit clock adapter")
    return value


def _derive(evidence, binding):
    view = deepcopy(evidence)
    times = frame_times(binding)
    originals = {"frames": [row["time_s"] for row in evidence["frames"]],
                 "reference": [row["time_s"] for row in evidence["reference"]],
                 "origin_present": "origin_time_s" in evidence, "origin_time_s": evidence.get("origin_time_s")}
    indexed = {}
    for row in view["frames"]:
        row["time_s"] = times[row["frame"]]
        indexed[row["id"]] = row["time_s"]
    for row in view["reference"]:
        row["time_s"] = indexed[row["evidence_id"]]
    if originals["origin_present"]:
        view["origin_time_s"] = view["reference"][0]["time_s"] if view["reference"] else None
    view["source_clock"] = {"profile": VIEW_PROFILE, "binding": deepcopy(binding), "original_times": originals}
    return view


def derive_evidence(evidence, binding):
    """Change only reconstruction times; raw detector payloads and identities stay intact."""
    return _derive(evidence, validate_clock(binding, evidence))


def validate_derived_evidence(view):
    _bounded_json(view, MAX_EVIDENCE_BYTES + MAX_CLOCK_BYTES)
    _require(isinstance(view, dict), "derived evidence must be an object")
    marker = view.get("source_clock")
    _require(isinstance(marker, dict) and set(marker) == {"profile", "binding", "original_times"}
             and marker["profile"] == VIEW_PROFILE, "derived view marker differs")
    original = deepcopy(view)
    original.pop("source_clock")
    times = marker["original_times"]
    _require(isinstance(times, dict) and set(times) == {"frames", "reference", "origin_present", "origin_time_s"}
             and type(times["origin_present"]) is bool, "original timestamp custody differs")
    for key in ("frames", "reference"):
        rows, stamps = original.get(key), times.get(key)
        _require(isinstance(rows, list) and isinstance(stamps, list) and len(rows) == len(stamps),
                 "original timestamp slots differ")
        for row, stamp in zip(rows, stamps, strict=True):
            _require(isinstance(row, dict) and _finite(stamp), "invalid original timestamp row")
            row["time_s"] = stamp
    if times["origin_present"]:
        original["origin_time_s"] = times["origin_time_s"]
    else:
        _require(times["origin_time_s"] is None, "absent original origin must remain absent")
        original.pop("origin_time_s", None)
    binding = validate_clock(marker["binding"], original)
    _require(canonical_hash(_derive(original, binding)) == canonical_hash(view),
             "derived reconstruction changed data beyond its exact native timing")
    return binding


def clock_summary(binding):
    value = validate_clock_binding(binding)
    return {key: deepcopy(value[key]) for key in SUMMARY_FIELDS} | {
        "profile": SUMMARY_PROFILE, "binding_sha256": canonical_hash(value),
        "reconstruction_time_basis": "native_pts_minus_first_pts"}


def validate_summary(summary, *, source_sha256=None):
    _bounded_json(summary, 4096)
    _require(isinstance(summary, dict) and set(summary) == SUMMARY_FIELDS | {
        "profile", "binding_sha256", "reconstruction_time_basis"} and summary["profile"] == SUMMARY_PROFILE,
        "summary fields differ")
    _metadata(summary)
    _require(isinstance(summary["binding_sha256"], str) and _SHA.fullmatch(summary["binding_sha256"])
             and summary["reconstruction_time_basis"] == "native_pts_minus_first_pts", "summary binding/basis differs")
    if source_sha256 is not None:
        _require(summary["source_sha256"] == source_sha256, "summary source differs")
    return deepcopy(summary)
