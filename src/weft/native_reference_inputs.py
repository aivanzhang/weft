"""Compose sealed native timing with dimension provenance without editing either pose view."""

from dataclasses import dataclass
from types import MappingProxyType

from weft.source_clock import clock_summary, validate_clock, validate_derived_evidence
from weft.whole_body_reference_inputs import (
    _DECODERS, _PRODUCER, _Reader, _SOURCE_PRODUCERS, _bytes, _check, _json, _manifest,
    _require, _root_path, _sha, _SHA, _name, MAX_INPUT_BYTES, MAX_RECOVERY_BYTES, MAX_MANIFEST_BYTES, MAX_MEDIA_BYTES,
)

PROFILE = "native-reference-dimension-binding-v1"
VIEW_RUNTIME_FILES = ("native_reference_inputs.py", "whole_body_reference_inputs.py", "source_clock.py", "io.py")
CAPTURE_FILES = frozenset({"decoder-clock.json", "native-clock-capture.json", "native-clock-decode.log"})
DIMENSION_FIELDS = {"schema_version", "profile", "body_manifest_sha256", "source_sha256", "source_artifact",
    "human_sha256", "original_evidence_sha256", "derived_evidence_sha256", "width", "height", "decoded_frames",
    "metadata_frames", "decoder_producer_sha256", "decoder_profile", "producer_manifest_member", "clock",
    "capture_artifacts", "pose_options", "coordinate_scope", "physical_validation",
    "body_constructor_decoder_sha256", "original_decoder_attachment"}
COORDINATE_SCOPE = "Original source image dimensions; body validation preserves any explicit region/crop mapping. Native presentation timing is separate from detector time and unknown camera capture time."


@dataclass(frozen=True)
class NativeReferenceInputs:
    human_bytes: bytes
    original_evidence_bytes: bytes
    derived_evidence_bytes: bytes
    dimensions_bytes: bytes
    body_manifest_bytes: bytes
    files: object

    @property
    def binding(self):
        return validate_derived_evidence(_json(self.derived_evidence_bytes))

    @property
    def dimensions(self):
        return _json(self.dimensions_bytes)


def validate_native_reference_view(human_bytes, original_evidence_bytes, derived_evidence_bytes, dimensions_bytes):
    """Check detached worker inputs; parent seals separately qualify media, producer and crop custody."""
    _require(all(type(raw) is bytes and 0 < len(raw) <= 8 << 20 for raw in
                 (human_bytes, original_evidence_bytes, derived_evidence_bytes, dimensions_bytes)),
             "Native reference views require bounded exact bytes")
    human, original, derived, dimensions = map(_json, (human_bytes, original_evidence_bytes,
                                                     derived_evidence_bytes, dimensions_bytes))
    _require(set(dimensions) == DIMENSION_FIELDS and type(dimensions["schema_version"]) is int
             and dimensions["schema_version"] == 1 and dimensions["profile"] == PROFILE
             and dimensions["physical_validation"] == "unverified"
             and dimensions["coordinate_scope"] == COORDINATE_SCOPE, "Native dimension binding fields differ")
    for key in ("body_manifest_sha256", "source_sha256", "human_sha256", "original_evidence_sha256",
                "derived_evidence_sha256", "decoder_producer_sha256", "body_constructor_decoder_sha256"):
        _require(type(dimensions[key]) is str and _SHA.fullmatch(dimensions[key]),
                 "Native dimension binding requires exact SHA256 identities")
    _name(dimensions["source_artifact"])
    _require(dimensions["source_artifact"] in {"source" + suffix for suffix in
             (".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv")}
             and dimensions["producer_manifest_member"] in (*_SOURCE_PRODUCERS, "recovery/" + _PRODUCER),
             "Native dimension binding source/producer paths differ")
    captures = dimensions["capture_artifacts"]
    _require(type(captures) is dict and (not captures or captures.keys() == CAPTURE_FILES)
             and all(type(value) is str and _SHA.fullmatch(value) for value in captures.values()),
             "Native dimension capture binding must be complete")
    options = dimensions["pose_options"]
    _require(type(options) is dict and {"pose_mode", "pose_identity"} <= options.keys()
             and options.keys() <= {"pose_mode", "pose_identity", "max_pose_hypotheses", "person_region"}
             and options["pose_mode"] in ("video", "image") and options["pose_identity"] in ("raw", "temporal")
             and type(options.get("max_pose_hypotheses", 2)) is int
             and options.get("max_pose_hypotheses", 2) in (1, 2), "Native dimension pose options differ")
    binding = validate_derived_evidence(derived)
    validate_clock(binding, original, evidence_sha256=_sha(original_evidence_bytes))
    _require(dimensions["human_sha256"] == _sha(human_bytes)
             and dimensions["original_evidence_sha256"] == _sha(original_evidence_bytes)
             and dimensions["derived_evidence_sha256"] == _sha(derived_evidence_bytes)
             and dimensions["source_sha256"] == original["source_sha256"] == human.get("source_sha256")
             and human.get("native_clock") == dimensions["clock"] == clock_summary(binding),
             "Native dimension/body/evidence identities differ")
    _require(human.get("profile") == "exercise-v3" and type(human.get("schema_version")) is int
             and human["schema_version"] == 3 and human.get("dt") == 1 / 30
             and "camera_reference" not in human and not any("refinement" in key or "alignment" in key for key in human),
             "Native reference inputs require an original mobile body")
    for key, evidence_key in (("width", "width"), ("height", "height"), ("decoded_frames", "decoded_frames"),
                              ("metadata_frames", "frame_count")):
        _require(type(dimensions[key]) is int and dimensions[key] == original[evidence_key],
                 "Native dimension binding differs from original decoder metadata")
    _require(64 <= dimensions["width"] <= 4096 and 64 <= dimensions["height"] <= 2160,
             "Native reference image dimensions exceed the reviewed decoder envelope")
    producer = dimensions["decoder_producer_sha256"]
    _require(producer in _DECODERS and dimensions["decoder_profile"] == _DECODERS[producer],
             "Native dimension binding requires a reviewed decoder")
    attachment = dimensions["original_decoder_attachment"]
    if attachment is None:
        _require(producer == dimensions["body_constructor_decoder_sha256"],
                 "A different original decoder needs its explicit source attachment")
    else:
        _require(type(attachment) is dict and set(attachment) == {"manifest_sha256", "manifest_artifact", "producer_artifact", "producer_sha256"}
                 and attachment["producer_artifact"] == "reference-inputs/original-decoder.py"
                 and attachment["manifest_artifact"] == "reference-inputs/original-source-manifest.json"
                 and attachment["producer_sha256"] == producer
                 and type(attachment["manifest_sha256"]) is str and _SHA.fullmatch(attachment["manifest_sha256"]),
                 "Native original decoder attachment differs")
    usable = [row for row in derived["frames"] if row.get("status") in {"tracked", "tracked_partial"}]
    refs, provenance = human.get("reference"), human.get("reference_evidence")
    _require(type(refs) is list and type(provenance) is list and 3 <= len(refs) == len(provenance) == len(usable) <= 900
             and [row["time_s"] for row in refs] == [row["time_s"] for row in usable]
             and [row["time_s"] for row in provenance] == [row["time_s"] for row in usable]
             and [row["evidence_id"] for row in provenance] == [row["id"] for row in usable],
             "Native references must retain every original usable identity and timestamp")
    _require("image_dimension_provenance" not in original and "image_dimension_provenance" not in derived,
             "Dimension provenance must remain outside both original and native-derived evidence")
    return human, derived, dimensions


def prepare_native_reference_inputs(source, *, expected_manifest_sha256=None, original_pose_source=None, deadline):
    """Read and validate one sealed body once; no media decode, inference, rendering or physics."""
    from weft.exercise_body_source import validate_body_source_files

    _check(deadline)
    reader = _Reader(_root_path(source), deadline)
    manifest_raw = reader.read("manifest.json", expected_manifest_sha256, MAX_MANIFEST_BYTES)
    manifest, declared = _manifest(manifest_raw)
    _require(manifest.get("profile") == "exercise-body-source-v1" and manifest.get("status") == "construction_only"
             and manifest.get("simulation_ready") is False
             and {"body-source.json", "native-clock.json", "reconstruction-evidence.json", _PRODUCER} <= declared.keys(),
             "Native reference preparation requires an original sealed native body source")
    files = {"manifest.json": manifest_raw}
    marker_raw = reader.read("body-source.json", declared["body-source.json"])
    files["body-source.json"] = marker_raw
    marker = _json(marker_raw)
    source_name = marker.get("source_artifact")
    _require(source_name in declared, "Original source video is missing from body seal")
    for name, digest in sorted(declared.items()):
        if name in files:
            continue
        files[name] = reader.read(name, digest, MAX_MEDIA_BYTES if name == source_name else 8 << 20,
                                  media=name == source_name)
    human, _ = validate_body_source_files(files, deadline=deadline)
    implementation = _json(files["implementation.json"])
    constructor_producer = declared[_PRODUCER]
    _require(implementation.get("files", {}).get("exercise_vision.py") == constructor_producer,
             "Native reference construction implementation differs")
    producer, attachment, attachment_reader = constructor_producer, None, None
    producer_member = _PRODUCER
    if marker.get("input_kind") == "sealed_pose_evidence":
        parent_raw = files.get("inputs/source-manifest.json")
        _require(parent_raw is not None and marker.get("input_manifest_sha256") == _sha(parent_raw),
                 "Native saved evidence requires its exact original producer manifest")
        _, parent_files = _manifest(parent_raw)
        names = ("recovery/" + _PRODUCER,) if "pose-recovery.json" in files else _SOURCE_PRODUCERS
        candidates = [name for name in names if name in parent_files]
        _require(candidates and len({parent_files[name] for name in candidates}) == 1
                 and all(parent_files.get(name) == declared[name] for name in
                 ("evidence.json", "pose-landmarks.json", "identity-audit.json", "model.json", "source.json", "run-config.json", source_name)),
                 "Native source manifest lost its reviewed decoder/evidence binding")
        producer_member = candidates[0]
        producer = parent_files[producer_member]
        if producer != constructor_producer or original_pose_source is not None:
            _require(original_pose_source is not None,
                     "Original decoder differs from the body constructor; provide original_pose_source for explicit sealed attachment")
            attachment_reader = _Reader(_root_path(original_pose_source), deadline)
            raw = attachment_reader.read("manifest.json", _sha(parent_raw), MAX_MANIFEST_BYTES)
            remaining = min(MAX_INPUT_BYTES - (reader.total - reader.media_bytes) - len(raw),
                            MAX_RECOVERY_BYTES - reader.total - len(raw), 8 << 20)
            _require(remaining > 0, "Original decoder attachment exceeds aggregate input bound")
            original_decoder = attachment_reader.read(producer_member, producer, remaining)
            files["reference-inputs/original-decoder.py"] = original_decoder
            files["reference-inputs/original-source-manifest.json"] = raw
            attachment = {"manifest_sha256": _sha(raw), "manifest_artifact": "reference-inputs/original-source-manifest.json",
                          "producer_artifact": "reference-inputs/original-decoder.py", "producer_sha256": producer}
    else:
        _require(marker.get("input_kind") == "local_video" and marker.get("input_manifest_sha256") is None,
                 "Unsupported native body input route")
        _require(original_pose_source is None, "Direct native sources must not attach an unrelated producer")
    _require(producer in _DECODERS, "Native reference source has an unreviewed original decoder producer")
    original = _json(files["evidence.json"])
    binding = validate_derived_evidence(_json(files["reconstruction-evidence.json"]))
    dimensions = {"schema_version": 1, "profile": PROFILE, "body_manifest_sha256": _sha(manifest_raw),
        "source_sha256": marker["source_sha256"], "source_artifact": source_name,
        "human_sha256": _sha(files["human.json"]), "original_evidence_sha256": _sha(files["evidence.json"]),
        "derived_evidence_sha256": _sha(files["reconstruction-evidence.json"]),
        "width": original["width"], "height": original["height"], "decoded_frames": original["decoded_frames"],
        "metadata_frames": original["frame_count"], "decoder_producer_sha256": producer,
        "decoder_profile": _DECODERS[producer], "producer_manifest_member": producer_member,
        "body_constructor_decoder_sha256": constructor_producer, "original_decoder_attachment": attachment,
        "clock": clock_summary(binding), "capture_artifacts": {name: declared[name] for name in sorted(CAPTURE_FILES & files.keys())},
        "pose_options": marker["pose_options"], "coordinate_scope": COORDINATE_SCOPE, "physical_validation": "unverified"}
    dimension_raw = _bytes(dimensions)
    validate_native_reference_view(files["human.json"], files["evidence.json"], files["reconstruction-evidence.json"], dimension_raw)
    # Recheck the small seal and root identity after validation, without rereading the original video.
    _Reader(reader.root, deadline, reader.identity).read("manifest.json", _sha(manifest_raw), MAX_MANIFEST_BYTES)
    if attachment_reader is not None:
        _Reader(attachment_reader.root, deadline, attachment_reader.identity).read(
            "manifest.json", attachment["manifest_sha256"], MAX_MANIFEST_BYTES)
    _check(deadline)
    return NativeReferenceInputs(files["human.json"], files["evidence.json"], files["reconstruction-evidence.json"],
                                 dimension_raw, manifest_raw, MappingProxyType(files))
