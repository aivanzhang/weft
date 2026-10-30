"""Bind a coupled world to exact construction evidence, not metadata-only identities."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from weft.world_evidence import MAX_DOCUMENTS, MAX_INPUT_BYTES, load_world_evidence


def build_shared_world_binding(documents, *, source_sha256, world_ids, entity_refs):
    """Return source bindings and filtered construction bytes without admitting audit values."""
    registry = load_world_evidence(documents)
    construction = registry.construction_bytes()
    exported = json.loads(construction)
    if len(world_ids) != len(set(world_ids)) or "human" not in world_ids or set(entity_refs) != set(world_ids):
        raise ValueError("Source bindings must cover the human and every unique rigid entity")
    sources = {source["document_sha256"]: source for source in exported["sources"]}
    total = None
    by_entity = {}
    for sha, source in sources.items():
        contract = source["source"]
        count = contract["decoded_frames"]
        interval = contract.get("decode_interval", [0, count - 1])
        if contract["sha256"] != source_sha256:
            raise ValueError("Coupled source evidence must refer to the same original video")
        if interval != [0, count - 1] or contract.get("source_complete") is False:
            raise ValueError("Coupled source evidence must preserve the complete original frame interval")
        if total is not None and count != total:
            raise ValueError("Coupled evidence documents disagree on the source frame count")
        total = count
        for obj in source["objects"]:
            by_entity[(sha, obj["id"])] = []
        for row in source["observations"]:
            by_entity[(sha, row["object_id"])].append(row)

    rows, used = [], set()
    for wid in world_ids:
        ref = entity_refs[wid]
        sha = oid = None
        available = 0
        mode = "authored_addition"
        if ref is not None:
            if not isinstance(ref, (tuple, list)) or len(ref) != 2:
                raise ValueError("An observation reference requires its document SHA256 and object ID")
            sha, oid = ref
            key = (sha, oid)
            if key not in by_entity or key in used:
                raise ValueError("Unknown or reused document-qualified observation entity")
            used.add(key)
            slots = by_entity[key]
            available = sum(row["admission"] == "included" and row["visibility"] in {"detected", "partial"}
                            and bool(row["measurements"]) for row in slots)
            mode = "observed_track" if available else "inferred_prior"
        rows.append({"world_entity_id": wid,
                     "entity_kind": "articulated_human" if wid == "human" else "rigid_object",
                     "observation_document_sha256": sha, "observation_entity_id": oid,
                     "provenance_mode": mode, "observed_slots": available,
                     "missing_slots": total - available, "pre_init_slots": 0})
    if {sha for sha, _ in used} != set(sources):
        raise ValueError("Do not attach unreferenced observation documents to a coupled world")
    return {"mode": "source_qualified", "source_sha256": source_sha256,
            "total_source_frames": total,
            "evidence_documents": {sha: f"evidence/{sha}.json" for sha in sorted(sources)},
            "entity_bindings": rows}, construction


def verify_shared_world_sources(scene, directory):
    """Resolve pinned documents and recompute complete construction-only entity accounting."""
    binding = scene["source_binding"]
    if binding["mode"] == "analytic_fixture":
        return {"mode": "analytic_fixture", "source_evidence_verified": False,
                "scope": "Authored software fixture; no source observation documents."}
    paths = binding["evidence_documents"]
    if not isinstance(paths, dict) or not 1 <= len(paths) <= MAX_DOCUMENTS:
        raise ValueError("Source-qualified worlds require 1–8 pinned observation documents")
    directory = Path(directory).resolve()
    documents, total_bytes = {}, 0
    for sha, name in paths.items():
        if not isinstance(name, str):
            raise ValueError("Evidence paths must be safe relative JSON paths")
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or rel.as_posix() != name or rel.suffix != ".json":
            raise ValueError("Evidence paths must be safe relative JSON paths")
        path = (directory / rel).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Evidence path is missing or escapes the coupled run")
        size = path.stat().st_size
        if total_bytes + size > MAX_INPUT_BYTES:
            raise ValueError("Coupled source evidence exceeds 64 MiB")
        raw = path.read_bytes()
        total_bytes += len(raw)
        if total_bytes > MAX_INPUT_BYTES or hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("Coupled observation document differs from its pinned hash")
        documents[sha] = raw
    refs = {row["world_entity_id"]: None if row["provenance_mode"] == "authored_addition" else
            (row["observation_document_sha256"], row["observation_entity_id"])
            for row in binding["entity_bindings"]}
    expected, construction = build_shared_world_binding(
        documents, source_sha256=binding["source_sha256"],
        world_ids=["human", *(obj["id"] for obj in scene["objects"])], entity_refs=refs)
    # Paths may differ after packaging; exact hashes and all semantic bindings must not.
    expected["evidence_documents"] = paths
    if expected != binding:
        raise ValueError("Coupled source binding disagrees with complete construction evidence")
    exported = json.loads(construction)
    return {"mode": "source_qualified", "source_evidence_verified": True,
            "source_sha256": binding["source_sha256"], "source_frames": binding["total_source_frames"],
            "document_count": len(documents), "input_bytes": total_bytes,
            "construction_sha256": hashlib.sha256(construction).hexdigest(),
            "source_clocks": [{"document_sha256": source["document_sha256"], "time": source["time"],
                               "partition_counts": dict(Counter(row["partition"] for row in source["observations"]))}
                              for source in exported["sources"]],
            "scope": "Exact source identities and complete slots verified; pose/track output is not physical truth, and clocks are not implicitly aligned."}
