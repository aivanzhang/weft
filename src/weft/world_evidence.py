"""Source-qualified evidence registry, not an executable world or physical qualification.

``load_world_evidence({exact_document_sha256: raw_json_bytes})`` validates complete
decoded object/frame coverage without changing original bytes, IDs, or clocks.
``summary()`` returns a defensive inspection snapshot; ``source_bytes(digest)``
returns the exact immutable input. These registry methods are not model exports.
``construction_bytes(allowed_partitions=("construction",))`` returns immutable
JSON containing every source slot; excluded slots have null visibility and empty
measurements/references. Its source entries contain document identity, source/time/image/
coordinate contracts, qualified objects, filtered camera claims, observations,
evidence metadata, and filtered relationships. No raw payload/provenance/limits
or scene blobs cross this boundary. All references are document-qualified.
Optional ``source_frame_selection`` restricts every aliased media group to
explicit original decoded frames; excluded partition labels are not reconciled.
"""

from collections import Counter
from collections.abc import Mapping
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType

from weft.observations import validate_observations


MAX_DOCUMENTS = 8
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_JSON_NODES = 2_000_000
MAX_JSON_DEPTH = 64
MAX_SELECTED_FRAMES_PER_SOURCE = 1800
EXPORT_PARTITIONS = frozenset({"construction", "generation", "development_validation"})
_DIGEST = re.compile(r"[0-9a-f]{64}")
_LIMITS = (
    "This is a filtered construction-evidence adapter, not an executable shared world.",
    "Exact input hashes establish identity, not physical truth or calibration.",
    "Excluded slots remain withheld; missing observations never establish absence.",
    "Model-derived coordinates remain inferred and uncalibrated.",
    "Raw payloads and unqualified global claims are omitted, not independently verified.",
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Evidence JSON must not contain nonfinite numbers")


def _check_tree(value):
    stack, count = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise ValueError("Evidence JSON exceeds its node/depth bound")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Evidence JSON must not contain nonfinite numbers")
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
            count += len(item)
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return count


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _qualified(digest, **identity):
    return {"document_sha256": digest, **identity}


def _claim(claim, allowed_ids, digest):
    references = claim["evidence_ids"]
    value = claim["value"]
    if claim["status"] == "unknown" and not references:
        return {"value": None, "status": "unknown", "evidence_refs": [], "admission": "unknown"}
    # Unreferenced global priors and arbitrary compound blobs have no safe partition boundary.
    eligible = bool(references) and all(ref in allowed_ids for ref in references)
    scalar = value is None or type(value) in (str, int, float, bool)
    if eligible and scalar:
        return {"value": value, "status": claim["status"], "admission": "included",
                "evidence_refs": [_qualified(digest, evidence_id=ref) for ref in references]}
    return {"value": None, "status": claim["status"], "admission": "withheld",
            "evidence_refs": [], "reason": "unqualified_or_excluded_evidence_or_compound_value"}


@dataclass(frozen=True, slots=True)
class _Source:
    digest: str
    raw: bytes
    document: Mapping


@dataclass(frozen=True, slots=True)
class WorldEvidence:
    """Validated immutable registry; model consumers receive only construction_bytes()."""

    _sources: tuple[_Source, ...]

    @property
    def document_sha256s(self):
        return tuple(source.digest for source in self._sources)

    @property
    def entity_keys(self):
        return tuple((source.digest, obj["id"]) for source in self._sources
                     for obj in source.document["objects"])

    def source_bytes(self, document_sha256):
        for source in self._sources:
            if source.digest == document_sha256:
                return source.raw
        raise KeyError(document_sha256)

    def summary(self):
        """Return a fresh inspection snapshot, never shared mutable registry state."""
        sources = []
        for source in self._sources:
            doc = source.document
            first, last = doc["source"].get("decode_interval", (0, doc["source"]["decoded_frames"] - 1))
            sources.append({"document_sha256": source.digest, "bytes": len(source.raw),
                            "schema_version": doc["schema_version"], "source": _thaw(doc["source"]),
                            "time": _thaw(doc["time"]), "image": _thaw(doc["image"]),
                            "coordinates": _thaw(doc["coordinates"]), "decoded_frame_interval": [first, last],
                            "entities": [_qualified(source.digest, object_id=obj["id"]) for obj in doc["objects"]],
                            "slots": len(doc["observations"]), "decoded_object_frame_coverage": "complete",
                            "partition_counts": dict(Counter(row["partition"] for row in doc["observations"])),
                            "visibility_counts": dict(Counter(row["visibility"] for row in doc["observations"]))})
        return {"profile": "world-evidence-registry-v1", "schema_version": 1,
                "physical_validation": "unverified", "sources": sources,
                "total_bytes": sum(len(source.raw) for source in self._sources),
                "total_slots": sum(len(source.document["observations"]) for source in self._sources)}

    def _checked_aliases(self, source_aliases):
        aliases = {} if source_aliases is None else source_aliases
        if (not isinstance(aliases, Mapping) or len(aliases) > MAX_DOCUMENTS
                or any(not isinstance(key, str) or not _DIGEST.fullmatch(key)
                       or not isinstance(value, str) or not _DIGEST.fullmatch(value)
                       or key == value or value in aliases for key, value in aliases.items())):
            raise ValueError("Source aliases require distinct SHA256 identities without chains or cycles")
        media = {source.document["source"]["sha256"] for source in self._sources}
        if set(aliases) - media:
            raise ValueError("Source aliases must reference included observation media")
        return dict(aliases)

    def _frame_selection(self, selection, aliases):
        if selection is None:
            return None
        if not isinstance(selection, Mapping) or not 1 <= len(selection) <= MAX_DOCUMENTS:
            raise ValueError("Frame selection must explicitly cover every canonical media group")
        intervals = {}
        for source in self._sources:
            metadata = source.document["source"]
            identity = aliases.get(metadata["sha256"], metadata["sha256"])
            intervals.setdefault(identity, []).append(metadata.get("decode_interval", (0, metadata["decoded_frames"] - 1)))
        if (any(type(key) is not str or not _DIGEST.fullmatch(key) for key in selection)
                or set(selection) != set(intervals)):
            raise ValueError("Frame selection must name exactly the canonical media groups after aliases")
        result = {}
        for identity, frames in selection.items():
            if type(frames) not in (list, tuple) or len(frames) > MAX_SELECTED_FRAMES_PER_SOURCE:
                raise ValueError("Frame selection exceeds the bounded ordered original-frame collection")
            previous = -1
            for frame in frames:
                if (type(frame) is not int or frame <= previous
                        or not any(first <= frame <= last for first, last in intervals[identity])):
                    raise ValueError("Frame selection requires ordered unique original decoded frame integers")
                previous = frame
            result[identity] = frozenset(frames)
        return result

    def _check_partitions(self, aliases, selection=None):
        partitions = {}
        for source in self._sources:
            identity = source.document["source"]["sha256"]
            identity = aliases.get(identity, identity)
            for row in source.document["observations"]:
                if row["partition"] == "unsampled":
                    if row["visibility"] != "not_sampled" or row["measurements"]:
                        raise ValueError("Unsampled partition must retain an empty not_sampled observation")
                    continue
                if selection is not None and row["frame"] not in selection[identity]:
                    continue
                # A missing sample is not a holdout; every actual frame reservation must agree.
                key = (identity, row["frame"])
                previous = partitions.setdefault(key, row["partition"])
                if previous != row["partition"]:
                    raise ValueError(f"Conflicting canonical frame partitions at frame {row['frame']}")

    def construction_bytes(self, *, allowed_partitions=("construction",), source_aliases=None, source_frame_selection=None):
        """Export typed evidence only; development access requires explicit opt-in."""
        if not isinstance(allowed_partitions, (tuple, list, set, frozenset)):
            raise ValueError("Allowed partitions must be an explicit collection")
        if (len(allowed_partitions) > len(EXPORT_PARTITIONS)
                or any(type(partition) is not str or partition not in EXPORT_PARTITIONS for partition in allowed_partitions)
                or len(set(allowed_partitions)) != len(allowed_partitions)):
            raise ValueError("Only construction, generation and development_validation may be exported")
        allowed = frozenset(allowed_partitions)
        # Verified aliases add restrictions; they never rewrite slots or grant partition access.
        aliases = self._checked_aliases(source_aliases)
        selection = self._frame_selection(source_frame_selection, aliases)
        self._check_partitions(aliases, selection)
        sources = [self._construction_source(source, allowed, None if selection is None else
                   selection[aliases.get(source.document["source"]["sha256"], source.document["source"]["sha256"])])
                   for source in self._sources]
        output = {"profile": "world-construction-evidence-v1", "schema_version": 1,
                  "physical_validation": "unverified", "allowed_partitions": sorted(allowed),
                  "sources": sources, "limits": list(_LIMITS)}
        if selection is not None:
            output["source_frame_selection"] = {
                "profile": "restrict-only-original-frame-selection-v1",
                "frames_by_canonical_source_sha256": {identity: sorted(frames) for identity, frames in selection.items()},
                "partition_consistency_scope": "selected_frames_only",
                "excluded_partition_labels": "preserved_not_reconciled",
                "source_aliases": aliases, "maximum_frames_per_source": MAX_SELECTED_FRAMES_PER_SOURCE,
            }
        return json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    @staticmethod
    def _construction_source(source, allowed, selected=None):
        doc, digest = source.document, source.digest
        admitted_ids = {identifier for row in doc["observations"] if row["partition"] in allowed
                        and (selected is None or row["frame"] in selected)
                        for identifier in row["evidence_ids"]}
        withheld_times = {obj["id"]: [] for obj in doc["objects"]}
        rows = []
        for row in doc["observations"]:
            admitted = row["partition"] in allowed and (selected is None or row["frame"] in selected)
            if not admitted:
                withheld_times[row["object_id"]].append(row["time_s"])
            rows.append({key: row[key] for key in ("id", "frame", "time_s", "object_id", "visibility", "partition")})
            rows[-1].update(entity_ref=_qualified(digest, object_id=row["object_id"]),
                            admission="included" if admitted else "withheld",
                            visibility=row["visibility"] if admitted else None,
                            measurements=_thaw(row["measurements"]) if admitted else {},
                            evidence_refs=[_qualified(digest, evidence_id=identifier) for identifier in row["evidence_ids"]] if admitted else [])
        evidence = [{key: row[key] for key in ("id", "frame", "time_s", "object_id", "kind", "status")}
                    for row in doc["evidence"] if row["id"] in admitted_ids]
        for row in evidence:
            row.update(document_sha256=digest, raw_payload_omitted=True)
        objects = [{"id": obj["id"], "entity_ref": _qualified(digest, object_id=obj["id"]),
                    "properties": {name: _claim(claim, admitted_ids, digest) for name, claim in obj["properties"].items()}}
                   for obj in doc["objects"]]
        relationships = []
        for relation in doc["relationships"]:
            refs = relation["evidence_ids"]
            if not refs or any(ref not in admitted_ids for ref in refs):
                continue
            interval = relation["interval_s"]
            # An admitted opening observation cannot expose a relation's withheld later duration.
            if any(times and (interval is None or bisect_left(times, interval[0]) < bisect_right(times, interval[1]))
                   for identifier in relation["objects"] if (times := withheld_times[identifier])):
                continue
            relationships.append({key: _thaw(relation[key]) for key in ("id", "type", "status", "interval_s")})
            relationships[-1].update(entity_refs=[_qualified(digest, object_id=identifier) for identifier in relation["objects"]],
                                     evidence_refs=[_qualified(digest, evidence_id=identifier) for identifier in refs],
                                     attributes_omitted=True)
        return {"document_sha256": digest, "observation_schema_version": doc["schema_version"],
                "source": _thaw(doc["source"]), "time": _thaw(doc["time"]), "image": _thaw(doc["image"]),
                "coordinates": _thaw(doc["coordinates"]), "objects": objects,
                "camera": {key: _claim(doc["camera"][key], admitted_ids, digest) for key in ("motion", "calibration")},
                "observations": rows, "evidence": evidence, "relationships": relationships,
                "relationships_filtered": True,
                "raw_payloads_and_global_metadata_omitted": True}


def load_world_evidence(sources):
    """Pin exact JSON bytes and require every decoded object/frame slot once."""
    if not isinstance(sources, Mapping) or not 1 <= len(sources) <= MAX_DOCUMENTS:
        raise ValueError("World evidence requires 1–8 observation documents")
    items = list(sources.items())
    total_bytes = 0
    for digest, raw in items:
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest) or type(raw) is not bytes:
            raise ValueError("Sources require exact lowercase SHA256 keys and immutable JSON bytes")
        total_bytes += len(raw)
        if total_bytes > MAX_INPUT_BYTES:
            raise ValueError("Combined evidence inputs exceed 64 MiB")
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Observation-document SHA256 does not match its exact input bytes")
    result, nodes = [], 0
    for digest, raw in sorted(items):
        try:
            doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_nonfinite)
            nodes += _check_tree(doc)
            if nodes > MAX_JSON_NODES:
                raise ValueError("Combined evidence JSON exceeds its node bound")
            doc = validate_observations(doc)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError, OverflowError, RecursionError) as error:
            raise ValueError("Malformed observation-document JSON or schema") from error
        expected = doc["source"]["decoded_frames"] * len(doc["objects"])
        # Existing validation rejects duplicate/out-of-range keys, so this count proves coverage.
        if len(doc["observations"]) != expected:
            raise ValueError(f"Incomplete decoded object/frame coverage: {len(doc['observations'])} slots, expected {expected}")
        result.append(_Source(digest, raw, _freeze(doc)))
    return WorldEvidence(tuple(result))
