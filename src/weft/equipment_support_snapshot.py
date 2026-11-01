"""Restore sealed image diagnostics without repeating pose or raster processing."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time

from weft.io import canonical_hash
from weft.shared_world_packaging import read_package_file


MAX_INPUT = 32 << 20
MEMBERS = {
    "inputs/context.json": 8 << 20,
    "inputs/equipment.json": 16 << 20,
    "inputs/masks.json": 8 << 20,
    "human-observations.json": 16 << 20,
    "source-annotations.json": 4 << 20,
    "annotation-metadata.json": 128 << 10,
    "annotation-frames.jsonl": (8 << 20) + 1800,
    "silhouette/metadata.json": 256 << 10,
    "silhouette/frames.jsonl": (8 << 20) + 1800,
}
SCOPE = ("Restored sealed image diagnostics, not freshly recomputed silhouettes or precursor qualification. "
         "Recorded source-equivalence evidence is verified when needed; no media is decoded. "
         "Image consistency does not verify physical contact, support, grasp or load.")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _require(value, message):
    if not value:
        raise ValueError("Equipment support snapshot: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Equipment support snapshot exceeded its cooperative deadline")


def _path(value):
    path = Path(value).absolute()
    _require(len(path.parts) <= 64 and ".." not in path.parts
             and not any(p.is_symlink() for p in (path, *path.parents)), "paths cannot traverse symlinks")
    return path


def _identity(path):
    stat = _path(path).stat(follow_symlinks=False)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _document(raw, deadline):
    from weft.world_evidence import _check_tree
    def unique(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    _check(deadline)
    try:
        value = json.loads(raw, object_pairs_hook=unique)
        _check_tree(value)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError("Equipment support snapshot requires bounded finite JSON") from error
    _require(type(value) is dict, "JSON object required")
    _check(deadline)
    return value


def _compact_cache(raw, maximum):
    # The study adds exactly one LF; keep archived bytes and record this normalization.
    _require(type(raw) is bytes and 0 < len(raw) <= maximum + 1 and raw.endswith(b"\n")
             and not raw.endswith(b"\n\n"), "saved compact cache needs one terminal LF")
    return raw[:-1]


def _rows(raw, count, deadline):
    _require(type(count) is int and 1 <= count <= 1800 and raw.count(b"\n") == count
             and raw.endswith(b"\n"), "complete original-frame JSONL cache required")
    values = tuple(raw[:-1].split(b"\n"))
    _require(all(0 < len(row) <= 128 << 10 for row in values), "cached frame exceeds its byte bound")
    _check(deadline)
    return values


@dataclass(frozen=True)
class EquipmentSupportSnapshot:
    pose: dict
    silhouette: object
    equipment_ids: tuple[str, ...]
    source_files: dict[str, bytes]
    source_manifest_sha256: str
    read_work: dict
    source_root: Path
    source_identities: dict

    def recheck(self, deadline):
        for name, expected in self.source_identities.items():
            _check(deadline)
            _require(_identity(self.source_root / name) == expected, "selected input changed before publication")


def _restore(files, proof, deadline):
    from weft.equipment_silhouette_evidence import EquipmentSilhouetteEvidence
    from weft.equipment_silhouette_report import _metadata, _row
    from weft.exercise_support_evidence import _equipment_mapping
    from weft.pose_surface_annotations import PoseSurfaceAnnotations
    from weft.world_evidence import load_world_evidence

    pose = _document(files["inputs/context.json"], deadline)
    equipment = _document(files["inputs/equipment.json"], deadline)
    human = _document(files["human-observations.json"], deadline)
    annotations = _document(files["source-annotations.json"], deadline)
    annotation_meta = _document(files["annotation-metadata.json"], deadline)
    meta = _document(files["silhouette/metadata.json"], deadline)
    channels = _metadata(meta)
    provenance = annotation_meta["provenance"]
    pose_pin = sha256(files["inputs/context.json"]).hexdigest()
    human_pin = sha256(files["human-observations.json"]).hexdigest()
    equipment_pin = sha256(files["inputs/equipment.json"]).hexdigest()
    count = meta["counts"]["source_frames"]
    _require(annotation_meta.get("profile") == "source-pose-surface-annotations-v1"
             and meta["pose_provenance"] == provenance
             and provenance["pose_raw_sha256"] == pose_pin
             and provenance["pose_canonical_sha256"] == canonical_hash(pose)
             and meta["source_sha256"] == provenance["source_sha256"] == pose["source_sha256"]
             and meta["equipment_document_sha256"] == equipment_pin
             and meta["mask_document_sha256"] == sha256(files["inputs/masks.json"]).hexdigest(),
             "saved pose/equipment/mask provenance differs")
    _require(annotation_meta["observation_document_sha256s"] == [human_pin]
             and annotation_meta["annotation_sha256"] == sha256(files["source-annotations.json"]).hexdigest()
             and human["provenance"]["input_evidence_sha256"] == provenance["pose_canonical_sha256"]
             and annotation_meta["source_clocks"] == {human_pin: human["time"]}
             and meta["source_clocks"] == {human_pin: human["time"], equipment_pin: equipment["time"]},
             "saved annotation/input/clock pins differ")
    _require(count == max(pose["frame_count"], pose["decoded_frames"])
             and meta["counts"]["decoded_frames"] == pose["decoded_frames"]
             and meta["counts"]["metadata_frames"] == pose["frame_count"]
             and annotation_meta["counts"]["retained_source_frames"] == count
             and annotation_meta["counts"]["retained_pose_slots"] == len(pose["frames"]),
             "saved source coverage differs")
    selected = [row["frame"] for row in pose["frames"][:math.floor(.6 * len(pose["frames"]))]]
    _require(meta["construction_frames"] == selected, "saved image evidence must use the original construction prefix")
    mapping, proof_summary = _equipment_mapping(equipment, pose, proof, deadline)
    _require(meta["source_equivalence"] == proof_summary, "saved source-equivalence summary differs")
    aliases = None if mapping is None else {equipment["source"]["sha256"]: pose["source_sha256"]}
    registry = load_world_evidence({human_pin: files["human-observations.json"], equipment_pin: files["inputs/equipment.json"]})
    exported = _document(registry.construction_bytes(source_aliases=aliases,
        source_frame_selection={pose["source_sha256"]: selected}), deadline)
    _require(meta.get("canonical_frame_selection") == exported["source_frame_selection"],
             "selected-frame policy differs from exact canonical inputs")
    _require(set(channels) == {row["id"] for row in annotations["channels"]}
             and len(annotations["channels"]) == len(channels), "annotation wrist channels differ")
    for channel in annotations["channels"]:
        _require(channel["point_role"] in {"left_wrist", "right_wrist"}
                 and channel["entity_ref"]["document_sha256"] == human_pin,
                 "wrist channels require the exact human document")
    source_poses = {row["frame"]: row for row in pose["frames"]}
    _require(len(source_poses) == len(pose["frames"]) <= 900, "duplicate or excessive pose slots")
    annotation_rows = _rows(files["annotation-frames.jsonl"], count, deadline)
    silhouette_rows = _rows(files["silhouette/frames.jsonl"], count, deadline)
    statuses, annotation_statuses, identities = Counter(), Counter(), None
    selected_set = frozenset(selected)
    annotation_count = 0
    copied = 0
    for index, (annotation_raw, silhouette_raw) in enumerate(zip(annotation_rows, silhouette_rows, strict=True)):
        _check(deadline)
        row = _document(silhouette_raw, deadline)
        keys, _ = _row(row, index, meta, channels, selected_set)
        if identities is None:
            identities = keys
        _require(keys == identities, "saved equipment/wrist identities changed across frames")
        supplied = source_poses.get(index)
        for target, source in (("pose_evidence_id", "id"), ("pose_time_s", "time_s"),
                               ("pose_decoder_timestamp_s", "decoder_timestamp_s")):
            _require(row[target] == (None if supplied is None else supplied.get(source)), "saved original pose clock/identity differs")
        native = None if mapping is None else mapping.get(index)
        _require(row["source_mapping"] == native and row["equipment_source_frame"] ==
                 (index if mapping is None else None if native is None else native["derivative_frame"]),
                 "saved original/derivative frame mapping differs")
        statuses.update(pair["reason"] or pair["status"] for pair in row["pairs"])
        annotation = _document(annotation_raw, deadline)
        _require(annotation["source_frame"] == index and annotation["source_status"] == row["source_status"]
                 and annotation["pose_evidence_id"] == row["pose_evidence_id"]
                 and annotation["pose_time_s"] == row["pose_time_s"]
                 and annotation["decoder_timestamp_s"] == row["pose_decoder_timestamp_s"], "saved annotation frame identity differs")
        points = annotation["channels"]
        _require(type(points) is list and len(points) == len(channels)
                 and {point["channel_id"] for point in points} == set(channels), "incomplete annotation channel slots")
        for point in points:
            _require(point["contact"] == "unknown" and point["status"] in {"copied_source_image_estimate", "unavailable"},
                     "saved annotations cannot declare physical contact")
            if index not in selected_set:
                _require(point["pixel"] is None, "excluded annotation exposes coordinates")
            annotation_statuses[point["reason"] or point["status"]] += 1
            copied += point["status"] == "copied_source_image_estimate"
            annotation_count += 1
    _require(dict(statuses) == meta["counts"]["status_counts"]
             and count * len(identities) == meta["counts"]["pair_slots"]
             and len(source_poses) == meta["counts"]["pose_slots"]
             and dict(annotation_statuses) == annotation_meta["counts"]["status_counts"]
             and annotation_count == annotation_meta["counts"]["channel_slots"]
             and copied == annotation_meta["counts"]["copied_estimates"], "complete saved status/slot counts differ")
    equipment_ids = tuple(dict.fromkeys(key[2] for key in identities))
    wrists = list(dict.fromkeys((key[0], key[1], key[3], key[4]) for key in identities))
    _require(1 <= len(equipment_ids) <= 8 and len(wrists) == len(channels)
             and identities == [(w[0], w[1], obj, w[2], w[3]) for obj in equipment_ids for w in wrists],
             "complete ordered wrist/equipment pairs required")
    annotation_cache = _compact_cache(files["annotation-metadata.json"], 128 << 10)
    silhouette_cache = _compact_cache(files["silhouette/metadata.json"], 256 << 10)
    _require(sum(map(len, annotation_rows)) == annotation_meta["cached_row_bytes"]
             and sum(map(len, silhouette_rows)) == meta["cached_row_bytes"]
             and len(silhouette_cache) + sum(map(len, silhouette_rows)) <= 8 << 20
             and len(annotation_cache) + sum(map(len, annotation_rows)) + len(files["source-annotations.json"]) <= 8 << 20,
             "complete saved cache byte counts differ")
    _require(all(annotation_meta[key] == "unknown" for key in ("contact", "load", "grasp"))
             and annotation_meta["body_or_floor_installed"] is False
             and annotation_meta["physical_validation"] == "unverified", "annotation physical claims differ")
    prepared_annotations = PoseSurfaceAnnotations(((human_pin, files["human-observations.json"]),),
        files["inputs/context.json"], files["source-annotations.json"], annotation_cache, annotation_rows)
    prepared = EquipmentSilhouetteEvidence(prepared_annotations, files["inputs/equipment.json"],
        files["inputs/masks.json"], proof, silhouette_cache, silhouette_rows)
    _check(deadline)
    return pose, prepared, equipment_ids


def load_equipment_support_snapshot(run, *, expected_manifest_sha256=None, deadline):
    _require(type(deadline) in (int, float) and math.isfinite(deadline), "finite deadline required")
    _require(expected_manifest_sha256 is None or type(expected_manifest_sha256) is str
             and _SHA.fullmatch(expected_manifest_sha256), "invalid expected manifest SHA256")
    source = _path(run)
    files, identities, read_bytes = {}, {}, 0
    def read(name, maximum, expected=None):
        nonlocal read_bytes
        _check(deadline)
        before = _identity(source / name)
        raw = read_package_file(source, name, max_bytes=min(maximum, MAX_INPUT - read_bytes), expected_sha256=expected)
        after = _identity(source / name)
        _require(before == after, "selected source changed during read")
        identities[name] = after
        files[name] = raw
        read_bytes += len(raw)
        return raw
    try:
        seal_raw = read("manifest.json", 1 << 20, expected_manifest_sha256)
    except FileNotFoundError as error:
        raise ValueError("Equipment support needs a completed study seal; inspect the original failure/process receipt") from error
    seal = _document(seal_raw, deadline)
    _require(type(seal.get("schema_version")) is int and seal["schema_version"] == 1
             and seal.get("profile") == "pose-equipment-silhouette-composition-v1"
             and seal.get("status") == "completed", "completed pose/equipment composition-v1 required")
    pins = seal.get("files")
    _require(type(pins) is dict and len(MEMBERS) <= len(pins) <= 384, "bounded study file map required")
    for name, maximum in MEMBERS.items():
        _require(type(pins.get(name)) is str and _SHA.fullmatch(pins[name]), "study omits an exact required member")
        read(name, maximum, pins[name])
    meta = _document(files["silhouette/metadata.json"], deadline)
    _require(all(type(meta.get(key)) is str and _SHA.fullmatch(meta[key])
                 for key in ("source_sha256", "equipment_source_sha256")),
             "exact source identities required before proof loading")
    proof = None
    if meta["source_sha256"] != meta["equipment_source_sha256"]:
        from weft.source_equivalence import load_source_equivalence
        proof_names = [name for name in pins if name.startswith("source-equivalence/")]
        _require(2 <= len(proof_names) <= 129, "bounded complete recorded source proof required")
        before_proof = {}
        for name in proof_names:
            _check(deadline)
            before_proof[name] = _identity(source / name)
        proof = load_source_equivalence(source / "source-equivalence", max_input_bytes=MAX_INPUT-read_bytes,
                                        max_files=129, deadline=deadline)
        _require(set(proof_names) == {"source-equivalence/" + name for name in proof.file_names},
                 "source proof inventory differs from the outer study")
        for name in proof.file_names:
            _check(deadline)
            member = "source-equivalence/" + name
            raw = proof.raw_bytes(name)
            _require(pins.get(member) == sha256(raw).hexdigest(), "source-equivalence member differs from the study seal")
            after = _identity(source / member)
            _require(before_proof[member] == after, "source proof changed during verification")
            identities[member] = after
            files[member] = raw
        read_bytes += proof.total_input_bytes
    _check(deadline)
    try:
        pose, silhouette, equipment_ids = _restore(files, proof, deadline)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Equipment support snapshot has incomplete or incompatible saved fields") from error
    return EquipmentSupportSnapshot(pose, silhouette, equipment_ids, files, sha256(seal_raw).hexdigest(),
        {"selected_files": len(files), "selected_bytes": read_bytes,
         "proof_files": 0 if proof is None else len(proof.file_names),
         "cache_normalization": "One terminal LF removed per saved metadata/JSONL record in memory only; archived bytes unchanged",
         "scope": SCOPE}, source, identities)
