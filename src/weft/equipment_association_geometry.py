"""Opt-in shape hypotheses under unresolved initial identity, never identity repair."""

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time

from weft.equipment_temporal_silhouette import _propose_temporal_geometry, _sample_selected
from weft.io import canonical_hash
from weft.mask_track_hypotheses import _json, associate_source_masks
from weft.segmentation_masks import _codec_files, decode_mask
from weft.silhouette_geometry import SilhouetteEvidence
from weft.world_evidence import _freeze, _thaw, load_world_evidence


PROFILE = "conditional-association-equipment-geometry-v1"
CONTRACT = {
    "maximum_seconds": 30., "maximum_pixels": 50_000_000,
    "association": "Recompute the complete construction-only adjacent mask chain; no supplied association JSON is trusted.",
    "identity": "Initial object-to-mask correspondence remains an explicit unverified prior; original unresolved groups are never changed.",
    "chain": "Every adjacent link from declared initialization through the last selected frame must survive, with no provider-label change.",
    "derivation": "Decode every admitted original encoded mask once and require exact equality with its saved binary raster.",
    "geometry": "Reuse fixed-envelope temporal bands under explicit static-camera/static-object and exterior priors.",
    "scope": "Conditional source-image geometry only; no recovered physical identity, depth, contact, grasp, support or load.",
}


def _require(condition, message):
    if not condition:
        raise ValueError("Association geometry: " + message)


@dataclass(frozen=True, init=False)
class MaskAssociationEvidence:
    _report: object
    _proof: object

    def __init__(self):
        raise TypeError("Use prepare_mask_association_evidence")


def prepare_mask_association_evidence(source_document, mask_document, *, raster_loader, max_seconds=30.):
    """Verify source binding, full associations and original encoded-to-binary equality."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds)
             and 0 < max_seconds <= CONTRACT["maximum_seconds"], "bounded preparation time required")
    deadline = time.monotonic() + max_seconds
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Association geometry preparation deadline")
    cached, pixel_count = {}, 0
    def load(row):
        nonlocal pixel_count
        check()
        binary = raster_loader(row)
        _require(type(binary) is bytes, "saved binary raster bytes required")
        pixel_count += len(binary)
        _require(pixel_count <= CONTRACT["maximum_pixels"], "aggregate raster bound")
        cached[row["id"]] = binary
        return binary
    report = associate_source_masks(source_document, mask_document, raster_loader=load,
                                    maximum_seconds=min(15., max_seconds))
    check()
    document, masks = _json(source_document), _json(mask_document)
    construction = json.loads(load_world_evidence({sha256(source_document).hexdigest(): source_document}).construction_bytes())
    payloads = {row["id"]: row["payload"] for row in document["evidence"]}
    rows = {row["id"]: row for row in masks["observations"]}
    codec = {name: sha256(path.read_bytes()).hexdigest() for name, path in _codec_files().items()}
    records = []
    for identifier, binary in cached.items():
        check()
        old, row = payloads[identifier], rows[identifier]
        raw = old["raw_model_output"]
        _require(type(raw) is dict and raw.get("frame") == row["frame"]
                 and raw.get("object_id") == row["raw_model_id"]
                 and raw.get("status") == "returned"
                 and raw.get("source_dimensions") == [document["image"]["width_px"], document["image"]["height_px"]]
                 and raw.get("mask_dimensions") == row["raster_dimensions"]
                 and type(raw.get("mask_encoded")) is str
                 and sha256(raw["mask_encoded"].encode()).hexdigest() == row["encoded_sha256"],
                 "original encoded mask identity or dimensions differ")
        raster = decode_mask(raw)
        check()
        _require(raster.tobytes() == binary, "original encoded mask differs from saved raster")
        records.append({"id": identifier, "encoded_sha256": row["encoded_sha256"],
                        "raster_sha256": sha256(binary).hexdigest(), "pixels": len(binary)})
    check()
    proof = {"source_document_sha256": sha256(source_document).hexdigest(),
             "mask_document_sha256": sha256(mask_document).hexdigest(),
             "association_sha256": canonical_hash(report), "codec_hashes": codec,
             "encoded_to_raster_derivation_verified": True, "decoded_masks": len(records),
             "decoded_pixels": pixel_count, "records": records,
             "construction_relationships": construction["sources"][0]["relationships"],
             "identity_scope": "Global raw groups remain sealed in original inputs; only partition-filtered relationships are exported here.",
             "physical_identity_verified": False, "contact": "unknown"}
    prepared = object.__new__(MaskAssociationEvidence)
    object.__setattr__(prepared, "_report", _freeze(report))
    object.__setattr__(prepared, "_proof", _freeze(proof))
    return prepared


def propose_association_equipment_geometry(context, association, object_id, *, construction_frames,
        initial_association_prior, static_prior, conditional_exterior=False, max_parts=8, max_seconds=20.):
    """Keep raw uncertainty while testing one explicitly conditional shape hypothesis."""
    started = time.monotonic()
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 20.,
             "bounded geometry time required")
    _require(isinstance(context, SilhouetteEvidence) and isinstance(association, MaskAssociationEvidence),
             "verified silhouette and recomputed mask association contexts required")
    _require(type(initial_association_prior) is dict
             and set(initial_association_prior) == {"mode", "provenance"}
             and initial_association_prior["mode"] == "unverified_initial_object_mask_correspondence"
             and type(initial_association_prior["provenance"]) is str
             and 1 <= len(initial_association_prior["provenance"].strip())
             and len(initial_association_prior["provenance"]) <= 2000,
             "explicit unverified initial object/mask correspondence required")
    _require(type(construction_frames) is list and 2 <= len(construction_frames) <= 16
             and all(type(frame) is int for frame in construction_frames)
             and all(a < b for a, b in zip(construction_frames, construction_frames[1:])),
             "fixed increasing construction frames required")
    report, proof = _thaw(association._report), _thaw(association._proof)
    _require(proof["association_sha256"] == canonical_hash(report), "prepared association changed")
    _require(proof["source_document_sha256"] == context._document_sha256
             and proof["mask_document_sha256"] == context._hashes["mask_document_sha256"],
             "association and geometry source documents differ")
    scene = _thaw(context._scene)
    entities = {row["world_entity_id"]: row for row in scene["source_binding"]["entity_bindings"]}
    _require(object_id in entities, "existing object required")
    raw_id = entities[object_id]["observation_entity_id"]
    obj = next(row for row in scene["objects"] if row["id"] == object_id)
    ref = obj["provenance"]["source_reference"]
    slots = {frame["frame"]: next(row for row in frame["slots"] if row["raw_object_id"] == raw_id)
             for frame in report["frames"]}
    _require(ref["frame"] in slots and slots[ref["frame"]]["id"] == ref["evidence_id"],
             "original scene anchor does not match source evidence")
    anchor = slots[ref["frame"]]["anchor_id"]
    _require(anchor is not None, "original scene frame lacks an uninterrupted image anchor")
    beginnings = [frame for frame, row in slots.items() if row["id"] == anchor
                  and row["status"] == "declared_initialization"]
    _require(len(beginnings) == 1, "source anchor must initialize under the same raw provider label")
    start, last = beginnings[0], construction_frames[-1]
    _require(start <= construction_frames[0] and last < len(report["frames"]), "selected frames outside anchor span")
    chain = []
    for frame in range(start, last + 1):
        row = slots[frame]
        source = context._construction[(context._document_sha256, raw_id, frame)]
        _require(source["partition"] == "construction" and source["admission"] == "included",
                 "complete selected chain must remain construction evidence")
        _require(row["anchor_id"] == anchor and (frame == start or row["previous_id"] == slots[frame - 1]["id"]),
                 "missing, ambiguous or provider-swapped image chain cannot constrain this shape")
        original = _thaw(context._rows[(frame, raw_id)])
        _require(original["id"] == row["id"] and original["raster_sha256"] == row["mask_sha256"],
                 "prepared geometry row differs from association chain")
        chain.append({"frame": frame, "evidence_id": row["id"], "previous_id": row["previous_id"],
                      "raw_object_id": raw_id, "identity": deepcopy(original["identity"]),
                      "original_source": deepcopy(original.get("original_source"))})
    def sample(prepared, entity, frame, camera, lower, upper, deadline):
        selected = _thaw(prepared._rows[(frame, raw_id)])
        _require(selected["mask_status"] == "decoded" and selected["proposal_status"] == "returned",
                 "selected original mask unavailable")
        return _sample_selected(prepared, selected, frame, camera, lower, upper, deadline)
    result = _propose_temporal_geometry(context, object_id, construction_frames=construction_frames,
        static_prior=static_prior, conditional_exterior=conditional_exterior, max_parts=max_parts,
        max_seconds=max_seconds - (time.monotonic() - started), sample_builder=sample)
    statement = "Initial object/mask correspondence and unresolved raw identities remain unverified; geometry is conditional on the complete adjacent image association."
    result["candidate_geometry"]["assumptions"].append(statement)
    result["report"].update(profile=PROFILE, association_contract=deepcopy(CONTRACT),
        initial_association_prior=deepcopy(initial_association_prior), association_proof=proof,
        conditional_source_chain=chain, raw_identity_reclassified=False, physical_identity_verified=False)
    result["report"]["candidate_geometry"] = deepcopy(result["candidate_geometry"])
    if result["geometry_config"] is not None:
        result["geometry_config"]["assumptions"].append(statement)
        result["geometry_config"]["replacements"][0]["geometry"] = deepcopy(result["candidate_geometry"])
    _require(len(json.dumps(result, allow_nan=False).encode()) <= 4 << 20, "bounded conditional result required")
    if time.monotonic() - started >= max_seconds:
        raise TimeoutError("Association geometry deadline")
    return result
