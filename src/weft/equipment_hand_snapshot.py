"""Extend sealed wrist evidence with named hand points and cached binary masks."""

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import PurePosixPath
import time

from weft.equipment_support_snapshot import (
    MAX_INPUT, EquipmentSupportSnapshot, _check, _document, _identity,
)
from weft.shared_world_packaging import read_package_file
from weft.source_surface_interval_construction import _encode


PROFILE = "equipment-hand-snapshot-v1"
SCOPE = ("New named-point image comparisons from exact retained pose precursors and sealed cached binary masks. "
         "Original wrist diagnostics remain unchanged. No video/PNG decoding or new inference; "
         "image overlap does not verify physical contact, support, grasp or load.")
EXTRA_MEMBERS = {
    "inputs/raw.json": 8 << 20,
    "inputs/audit.json": 2 << 20,
    "inputs/source_manifest.json": 1 << 20,
    "inputs/mask_manifest.json": 1 << 20,
    "raster-reads.json": 512 << 10,
}


def _require(value, message):
    if not value:
        raise ValueError("Equipment hand snapshot: " + message)


def _reserve(deadline, seconds):
    _check(deadline)
    _require(deadline - time.monotonic() >= seconds,
             "insufficient remaining time for the next bounded preparation; no new stage started")


def _hand_channels(saved_annotations, provenance):
    from weft.equipment_hand_evidence import PARTS

    uncertainty = {row[0]: row[2] for row in provenance["channel_landmark_index_uncertainty_px"]}
    channels, bindings, groups = [], [], {}
    for wrist in saved_annotations["channels"]:
        side = wrist["point_role"].removesuffix("_wrist")
        _require(side in {"left", "right"} and side + "_hand" not in groups,
                 "one original wrist channel per selected hand required")
        group = {}
        for part in PARTS:
            name = side + "_" + part
            channel_id = wrist["id"] if part == "wrist" else "hand-" + side + "-" + part
            channels.append({**wrist, "id": channel_id, "point_role": name})
            bindings.append({"channel_id": channel_id, "landmark_name": name,
                             "uncertainty_px": uncertainty[wrist["id"]]})
            group[part] = channel_id
        groups[side + "_hand"] = group
    _require(len({row["id"] for row in channels}) == len(channels), "generated hand channel IDs conflict")
    return channels, bindings, groups


@dataclass(frozen=True)
class EquipmentHandSnapshot:
    snapshot: EquipmentSupportSnapshot
    evidence: object
    artifacts: dict[str, bytes]


def prepare_equipment_hand_snapshot(snapshot, *, deadline):
    """Keep the source seal, original selection and authored uncertainty unchanged."""
    _require(type(snapshot) is EquipmentSupportSnapshot, "verified source snapshot required")
    from weft.equipment_hand_evidence import MAX_SECONDS as HAND_SECONDS, prepare_equipment_hand_evidence
    from weft.equipment_silhouette_evidence import MAX_LOADS, MAX_RASTER_BYTES
    from weft.pose_precursor_custody import MAX_SECONDS as CUSTODY_SECONDS, prepare_pose_precursor_custody
    from weft.pose_surface_annotations import MAX_SECONDS as ANNOTATION_SECONDS, prepare_pose_surface_annotations

    files, identities = dict(snapshot.source_files), dict(snapshot.source_identities)
    total = snapshot.read_work["selected_bytes"]
    pins = _document(files["manifest.json"], deadline)["files"]
    def read(name, maximum):
        nonlocal total
        _check(deadline)
        _require(name not in files and name in pins, "missing or repeated sealed input: " + name)
        before = _identity(snapshot.source_root / name)
        raw = read_package_file(snapshot.source_root, name, max_bytes=min(maximum, MAX_INPUT - total),
                                expected_sha256=pins[name])
        after = _identity(snapshot.source_root / name)
        _require(before == after, "selected input changed during read")
        files[name], identities[name] = raw, after
        total += len(raw)
        return raw

    for name, maximum in EXTRA_MEMBERS.items():
        read(name, maximum)
    mask_files = _document(files["inputs/mask_manifest.json"], deadline).get("files")
    mask_pin = sha256(files["inputs/masks.json"]).hexdigest()
    _require(type(mask_files) is dict and mask_files.get("mask-evidence.json") == mask_pin,
             "original mask manifest differs from exact retained mask metadata")
    # The envelope reuses strict finite/duplicate-key JSON checks for this saved list.
    records = _document(b'{"records":' + files["raster-reads.json"] + b'}', deadline)["records"]
    _require(type(records) is list and len(records) <= MAX_LOADS, "bounded cached raster inventory required")
    inventory = {}
    for row in records:
        _check(deadline)
        _require(type(row) is dict and set(row) == {"path", "png_sha256", "png_bytes", "binary_sha256",
                 "binary_bytes", "dimensions_hw", "mask_evidence_id"}, "incomplete cached raster descriptor")
        path = row["path"]
        _require(type(path) is str and 0 < len(path) <= 160 and path.endswith(".png")
                 and not PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts
                 and "\\" not in path and str(PurePosixPath(path)) == path and path not in inventory,
                 "unique normalized relative cached raster path required")
        binary = "binary/" + path.removesuffix(".png") + ".bin"
        dims = row["dimensions_hw"]
        _require(type(dims) is list and len(dims) == 2 and all(type(v) is int and 1 <= v <= 4096 for v in dims)
                 and type(row["binary_bytes"]) is int and row["binary_bytes"] == dims[0] * dims[1]
                 and row["binary_bytes"] <= MAX_RASTER_BYTES
                 and type(row["png_bytes"]) is int and row["png_bytes"] > 0
                 and type(row["mask_evidence_id"]) is str and 0 < len(row["mask_evidence_id"]) <= 192,
                 "bounded original raster dimensions and identity required")
        _require(all(type(row[key]) is str and len(row[key]) == 64
                     and all(c in "0123456789abcdef" for c in row[key]) for key in ("png_sha256", "binary_sha256"))
                 and mask_files.get(path) == row["png_sha256"]
                 and pins.get("pngs/" + path) == row["png_sha256"]
                 and pins.get(binary) == row["binary_sha256"],
                 "cached binary/PNG ancestry differs from recorded source seals")
        inventory[path] = (row, binary)

    used, raster_bytes = set(), 0
    def cached_raster(request):
        nonlocal raster_bytes
        _check(min(deadline, request["deadline_monotonic"]))
        path = request["path"]
        _require(path in inventory and path not in used, "required cached binary mask missing or requested twice")
        row, name = inventory[path]
        _require(request["mask_document_sha256"] == mask_pin
                 and request["mask_evidence_id"] == row["mask_evidence_id"]
                 and request["raster_sha256"] == row["binary_sha256"]
                 and request["dimensions"] == row["dimensions_hw"]
                 and request["maximum_bytes"] == row["binary_bytes"]
                 and request["remaining_total_bytes"] >= row["binary_bytes"]
                 and raster_bytes + row["binary_bytes"] <= MAX_RASTER_BYTES,
                 "point request differs from the exact cached raster descriptor")
        raw = read(name, row["binary_bytes"])
        _require(len(raw) == row["binary_bytes"] and raw.count(b"\0") + raw.count(b"\1") == len(raw),
                 "cached raster must contain exact binary cells")
        used.add(path); raster_bytes += len(raw)
        return raw

    saved = snapshot.silhouette.pose_annotations
    provenance = saved.describe()["provenance"]
    channels, bindings, groups = _hand_channels(_document(saved.annotation_document(), deadline), provenance)
    _reserve(deadline, CUSTODY_SECONDS)
    custody = prepare_pose_precursor_custody(context_pose_document=saved.pose_document,
        raw_pose_document=files["inputs/raw.json"], identity_audit_document=files["inputs/audit.json"],
        manifest_document=files["inputs/source_manifest.json"])
    _reserve(deadline, ANNOTATION_SECONDS)
    annotations = prepare_pose_surface_annotations(dict(saved.input_documents), pose_document=saved.pose_document,
        source_sha256=snapshot.pose["source_sha256"], channels=channels, landmark_bindings=bindings,
        minimum_confidence=provenance["confidence_threshold_authored"],
        uncertainty_method=provenance["uncertainty_method"], uncertainty_note=provenance["uncertainty_note"],
        precursor_custody=custody)
    _reserve(deadline, HAND_SECONDS)
    evidence = prepare_equipment_hand_evidence(annotations, hand_channel_ids=groups,
        equipment_document=snapshot.silhouette.equipment_document, mask_document=snapshot.silhouette.mask_document,
        equipment_ids=snapshot.equipment_ids, raster_loader=cached_raster,
        source_equivalence=snapshot.silhouette.source_equivalence,
        construction_frames=snapshot.silhouette.describe()["construction_frames"])
    _check(deadline)
    work = {**snapshot.read_work, "selected_files": len(files), "selected_bytes": total, "scope": SCOPE,
            "cached_binary_reads": len(used), "cached_binary_bytes": raster_bytes,
            "source_decodes": 0, "raster_decodes": 0, "pose_model_calls": 0,
            "point_relations": evidence.describe()["work"]["pair_relations"]}
    artifacts = {
        "hand/annotations.json": annotations.annotation_document(),
        "hand/annotation-metadata.json": annotations._metadata + b"\n",
        "hand/annotation-frames.jsonl": b"\n".join(annotations._rows) + b"\n",
        "hand/precursor-metadata.json": custody._metadata + b"\n",
        "hand/precursor-frames.jsonl": b"\n".join(custody._source_rows) + b"\n",
        "hand/metadata.json": evidence._metadata + b"\n",
        "hand/frames.jsonl": b"\n".join(evidence._rows) + b"\n",
    }
    artifacts["hand/preparation.json"] = _encode({"schema_version": 1, "profile": PROFILE,
        "source_manifest_sha256": snapshot.source_manifest_sha256, "hand_channel_ids": groups,
        "uncertainty_policy": "Each finger inherits its saved wrist's authored pixel range; not a measured hand-error bound",
        "minimum_confidence": provenance["confidence_threshold_authored"], "landmark_bindings": bindings,
        "read_work": work, "verification_scope": SCOPE}, 32 << 10, deadline) + b"\n"
    restored = replace(snapshot, source_files=files, source_identities=identities, read_work=work)
    return EquipmentHandSnapshot(restored, evidence, artifacts)
