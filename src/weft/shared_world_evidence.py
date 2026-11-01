"""Bind a coupled world to exact construction evidence, not metadata-only identities."""

from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from types import MappingProxyType

from weft.world_evidence import MAX_DOCUMENTS, MAX_INPUT_BYTES, load_world_evidence


MAX_SOURCE_FILES = 256
MAX_SOURCE_SECONDS = 30.0
MAX_EQUIVALENCES = 7
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _check_deadline(deadline):
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("Invalid coupled source preflight deadline")
    if time.monotonic() >= deadline:
        raise ValueError("Coupled source preflight time bound exceeded")


def validate_equivalence_bindings(binding):
    """Validate optional pinned mappings without loading media or changing old bindings."""
    if "source_equivalences" not in binding:
        return []
    rows = binding["source_equivalences"]
    if (binding["mode"] != "source_qualified" or not isinstance(rows, list)
            or not 1 <= len(rows) <= MAX_EQUIVALENCES):
        raise ValueError("Source equivalences require one to seven source-qualified mappings")
    seen_media, seen_bundles = set(), set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"original_source_sha256", "derivative_source_sha256",
                                                   "bundle_manifest_sha256", "bundle_path"}:
            raise ValueError("Source equivalence binding fields differ from contract")
        for key in ("original_source_sha256", "derivative_source_sha256", "bundle_manifest_sha256"):
            if not isinstance(row[key], str) or not _DIGEST.fullmatch(row[key]):
                raise ValueError("Source equivalence requires exact SHA256 identities")
        original, derivative = row["original_source_sha256"], row["derivative_source_sha256"]
        if original != binding["source_sha256"] or derivative == original or derivative in seen_media:
            raise ValueError("Source equivalence must bind distinct derivatives to the canonical original")
        path = row["bundle_path"]
        if path != f"source-mappings/{row['bundle_manifest_sha256']}" or path in seen_bundles:
            raise ValueError("Source equivalence requires a unique safe relative source-mappings/<manifest SHA256> path")
        seen_media.add(derivative)
        seen_bundles.add(path)
    return sorted(rows, key=lambda row: row["derivative_source_sha256"])


def _original_frame_key(row):
    return (row.source_frame, row.source_native_pts, row.source_decoder_pts,
            row.source_duration_pts, row.pixel_sha256, row.decoded_bytes)


def _equivalence_index(items, source_sha256):
    from weft.source_equivalence import SourceEquivalence

    if not isinstance(items, (tuple, list)) or len(items) > MAX_EQUIVALENCES:
        raise ValueError("Use at most seven explicitly verified source equivalences")
    indexed = {}
    for item in items:
        if not isinstance(item, SourceEquivalence) or item.original_sha256 != source_sha256:
            raise ValueError("Equivalence must verify a derivative of this original source")
        if item.derivative_sha256 == source_sha256 or item.derivative_sha256 in indexed:
            raise ValueError("Duplicate, circular or redundant source equivalence")
        if indexed:
            canonical = next(iter(indexed.values()))
            if (item.dimensions != canonical.dimensions or item.time_base != canonical.time_base
                    or item.source_clock != canonical.source_clock or len(item.frames) != len(canonical.frames)
                    or any(_original_frame_key(a) != _original_frame_key(b)
                           for a, b in zip(item.frames, canonical.frames))):
                raise ValueError("Equivalence bundles disagree on the canonical original frame/clock signature")
        indexed[item.derivative_sha256] = item
    _check_input_bounds(items, {})
    return indexed


def _check_input_bounds(equivalents, documents):
    size = sum(item.total_input_bytes for item in equivalents) + sum(map(len, documents.values()))
    files = sum(len(item.file_names) for item in equivalents) + len(documents)
    if size > MAX_INPUT_BYTES:
        raise ValueError("Combined pinned source inputs exceed 64 MiB")
    if files > MAX_SOURCE_FILES:
        raise ValueError("Combined pinned source file bound exceeded")


def _verify_document_clock(source, equivalence, *, derivative):
    contract = source["source"]
    if (contract["width_px"], contract["height_px"]) != equivalence.dimensions:
        raise ValueError("Observation source dimensions differ from the verified media mapping")
    if contract["decoded_frames"] != len(equivalence.frames):
        raise ValueError("Observation source coverage differs from the verified media mapping")
    if source["observation_schema_version"] == 3:
        scale = Fraction(equivalence.time_base)
        expected = [float((row.derivative_decoder_pts if derivative else row.source_decoder_pts) * scale)
                    for row in equivalence.frames]
        actual = contract["frame_times_s"]
        if len(actual) != len(expected) or any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
                                               for a, b in zip(actual, expected)):
            raise ValueError("Observation decoder clock conflicts with the verified media mapping")
    # Legacy frame/FPS timestamps remain declared assumptions, never recast as native observations.


def build_shared_world_binding(documents, *, source_sha256, world_ids, entity_refs, source_equivalences=(), deadline=None):
    """Return source bindings and filtered construction bytes without admitting audit values."""
    deadline = time.monotonic() + MAX_SOURCE_SECONDS if deadline is None else deadline
    _check_deadline(deadline)
    deadline = min(deadline, time.monotonic() + MAX_SOURCE_SECONDS)
    equivalents = _equivalence_index(source_equivalences, source_sha256)
    if isinstance(documents, Mapping) and all(type(raw) is bytes for raw in documents.values()):
        _check_input_bounds(equivalents.values(), documents)
    registry = load_world_evidence(documents)
    construction = registry.construction_bytes(source_aliases={sha: source_sha256 for sha in equivalents})
    exported = json.loads(construction)
    if len(world_ids) != len(set(world_ids)) or "human" not in world_ids or set(entity_refs) != set(world_ids):
        raise ValueError("Source bindings must cover the human and every unique rigid entity")
    sources = {source["document_sha256"]: source for source in exported["sources"]}
    total, dimensions, clocks = None, None, {}
    by_entity = {}
    for sha, source in sources.items():
        contract = source["source"]
        count = contract["decoded_frames"]
        interval = contract.get("decode_interval", [0, count - 1])
        if contract["sha256"] != source_sha256:
            if contract["sha256"] not in equivalents:
                raise ValueError("Coupled source evidence must refer to the same original video or a verified derivative")
            _verify_document_clock(source, equivalents[contract["sha256"]], derivative=True)
        elif equivalents:
            _verify_document_clock(source, next(iter(equivalents.values())), derivative=False)
        if interval != [0, count - 1] or contract.get("source_complete") is False:
            raise ValueError("Coupled source evidence must preserve the complete original frame interval")
        if total is not None and count != total:
            raise ValueError("Coupled evidence documents disagree on the source frame count")
        total = count
        shape = (contract["width_px"], contract["height_px"])
        if dimensions is not None and shape != dimensions:
            raise ValueError("Coupled evidence documents disagree on the source dimensions")
        dimensions = shape
        clock_key = (contract["sha256"], source["time"]["basis"])
        clock = tuple(contract["frame_times_s"]) if source["observation_schema_version"] == 3 else contract["fps"]
        if clocks.setdefault(clock_key, clock) != clock:
            raise ValueError("Coupled evidence documents disagree on a source clock with the same basis")
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
    referenced_media = {source["source"]["sha256"] for source in sources.values()}
    if set(equivalents) - referenced_media:
        raise ValueError("Do not attach unused source equivalence bundles")
    binding = {"mode": "source_qualified", "source_sha256": source_sha256,
            "total_source_frames": total,
            "evidence_documents": {sha: f"evidence/{sha}.json" for sha in sorted(sources)},
            "entity_bindings": rows}
    if equivalents:
        binding["source_equivalences"] = [{"original_source_sha256": item.original_sha256,
            "derivative_source_sha256": item.derivative_sha256, "bundle_manifest_sha256": item.manifest_sha256,
            "bundle_path": f"source-mappings/{item.manifest_sha256}"} for _, item in sorted(equivalents.items())]
    _check_deadline(deadline)
    return binding, construction


def _load_equivalences(binding, directory_fd, documents, deadline):
    from weft.source_equivalence import MAX_BUNDLE_BYTES, MAX_FILES, load_source_equivalence

    rows = validate_equivalence_bindings(binding)
    result, total_bytes, total_files = [], sum(map(len, documents.values())), len(documents)
    for row in rows:
        _check_deadline(deadline)
        if MAX_INPUT_BYTES - total_bytes < 1 or MAX_SOURCE_FILES - total_files < 2:
            raise ValueError("Combined pinned source byte/file bound exceeded")
        item = load_source_equivalence(row["bundle_path"], dir_fd=directory_fd,
                                       max_input_bytes=min(MAX_BUNDLE_BYTES, MAX_INPUT_BYTES - total_bytes),
                                       max_files=min(MAX_FILES + 1, MAX_SOURCE_FILES - total_files), deadline=deadline)
        if (item.original_sha256 != row["original_source_sha256"]
                or item.derivative_sha256 != row["derivative_source_sha256"]
                or item.manifest_sha256 != row["bundle_manifest_sha256"]):
            raise ValueError("Source equivalence differs from its pinned identities")
        result.append(item)
        total_bytes += item.total_input_bytes
        total_files += len(item.file_names)
    return result


def _document_paths(binding):
    paths = binding["evidence_documents"]
    if not isinstance(paths, dict) or not 1 <= len(paths) <= MAX_DOCUMENTS:
        raise ValueError("Source-qualified worlds require 1–8 pinned observation documents")
    seen = set()
    for sha, name in paths.items():
        if not isinstance(sha, str) or not _DIGEST.fullmatch(sha):
            raise ValueError("Evidence document identity must be an exact SHA256")
        if (not isinstance(name, str) or not 1 <= len(name) <= 240 or "\\" in name or ":" in name
                or any(ord(char) < 32 for char in name)
                or any(part in ("", ".", "..") for part in name.split("/"))
                or PurePosixPath(name).is_absolute() or PurePosixPath(name).suffix != ".json"
                or PurePosixPath(name).parts[0] == "source-mappings" or name in seen):
            raise ValueError("Evidence paths must be distinct safe relative JSON paths outside source-mappings")
        seen.add(name)
    for name in seen:
        if any(parent.as_posix() in seen for parent in PurePosixPath(name).parents):
            raise ValueError("Evidence file and directory paths overlap")
    return paths


def _read_document(directory_fd, name, limit):
    fd = os.dup(directory_fd)
    try:
        parts = PurePosixPath(name).parts
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(file_fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise ValueError("Coupled source evidence exceeds its 64 MiB bound or is not a regular file")
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
            if len(raw) > limit or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Evidence file changed during bounded read")
            return raw
    except OSError as error:
        raise ValueError("Evidence path is missing or unsafe") from error
    finally:
        os.close(fd)


def verify_shared_world_sources(scene, directory, *, include_files=False, deadline=None):
    """Resolve pinned documents and recompute complete construction-only entity accounting."""
    if deadline is not None:
        _check_deadline(deadline)
    binding = scene["source_binding"]
    if binding["mode"] == "analytic_fixture":
        validate_equivalence_bindings(binding)
        report = {"mode": "analytic_fixture", "source_evidence_verified": False,
                  "scope": "Authored software fixture; no source observation documents."}
        return (report, MappingProxyType({})) if include_files else report
    default_deadline = time.monotonic() + MAX_SOURCE_SECONDS
    deadline = default_deadline if deadline is None else min(deadline, default_deadline)
    paths = _document_paths(binding)
    validate_equivalence_bindings(binding)
    _check_deadline(deadline)
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Source evidence directory is missing or unsafe")
    directory = directory.resolve()
    documents, total_bytes = {}, 0
    _check_deadline(deadline)
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for sha, name in paths.items():
            _check_deadline(deadline)
            raw = _read_document(fd, name, MAX_INPUT_BYTES - total_bytes)
            total_bytes += len(raw)
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError("Coupled observation document differs from its pinned hash")
            documents[sha] = raw
        equivalents = _load_equivalences(binding, fd, documents, deadline)
    finally:
        os.close(fd)
    refs = {row["world_entity_id"]: None if row["provenance_mode"] == "authored_addition" else
            (row["observation_document_sha256"], row["observation_entity_id"])
            for row in binding["entity_bindings"]}
    expected, construction = build_shared_world_binding(
        documents, source_sha256=binding["source_sha256"],
        world_ids=["human", *(obj["id"] for obj in scene["objects"])], entity_refs=refs,
        source_equivalences=equivalents, deadline=deadline)
    # Paths may differ after packaging; exact hashes and all semantic bindings must not.
    expected["evidence_documents"] = paths
    actual = dict(binding)
    if equivalents:
        actual["source_equivalences"] = validate_equivalence_bindings(binding)
    if expected != actual:
        raise ValueError("Coupled source binding disagrees with complete construction evidence")
    exported = json.loads(construction)
    report = {"mode": "source_qualified", "source_evidence_verified": True,
            "source_sha256": binding["source_sha256"], "source_frames": binding["total_source_frames"],
            "document_count": len(documents), "input_bytes": total_bytes,
            "total_pinned_input_bytes": total_bytes + sum(item.total_input_bytes for item in equivalents),
            "canonical_partitions": "Consistent frame reservations; empty unsampled slots retain their original withholding.",
            "construction_sha256": hashlib.sha256(construction).hexdigest(),
            "source_equivalences": [item.inspection() for item in equivalents],
            "source_clocks": [{"document_sha256": source["document_sha256"], "time": source["time"],
                               "partition_counts": dict(Counter(row["partition"] for row in source["observations"]))}
                              for source in exported["sources"]],
            "scope": "Exact source identities and complete slots verified; pose/track output is not physical truth, and clocks are not implicitly aligned."}
    _check_deadline(deadline)
    if not include_files:
        return report
    files = {name: documents[sha] for sha, name in paths.items()}
    for item in equivalents:
        files.update({f"source-mappings/{item.manifest_sha256}/{name}": item.raw_bytes(name)
                      for name in item.file_names})
    return report, MappingProxyType(files)
