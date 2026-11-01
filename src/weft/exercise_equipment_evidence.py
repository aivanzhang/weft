"""Compose ordinary sealed pose/mask runs into the existing image-evidence package."""

from hashlib import sha256
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import time
import zipfile

from weft.equipment_support_snapshot import _document, _identity, _path
from weft.io import local_time
from weft.shared_world_packaging import read_package_file, validate_package_layout, write_package_files


PROFILE = "pose-equipment-silhouette-composition-v1"
MAX_INPUT, MAX_OUTPUT, MAX_FILES = 32 << 20, 64 << 20, 384
MAX_LOADS, MAX_BINARY = 512, 4 << 20
SCOPE = ("Selected sealed pose/mask members composed as conditional image evidence. "
         "Parent failures, source clocks and unknowns remain unchanged. No video/PNG decode, "
         "inference, body construction or verified identity/contact/grasp/load is supplied.")


def _require(value, message):
    if not value:
        raise ValueError("Exercise equipment evidence: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Exercise equipment evidence cooperative deadline exceeded")


def _pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _encoded(value, deadline, maximum=1 << 20):
    from weft.source_surface_interval_construction import _encode
    return _encode(value, maximum - 1, deadline) + b"\n"


class _Inputs:
    def __init__(self, deadline):
        self.deadline, self.raw, self.identities, self.total = deadline, {}, {}, 0

    def read(self, root, name, maximum, expected=None):
        _check(self.deadline)
        key = (root, name)
        if key not in self.raw:
            remaining = min(maximum, MAX_INPUT - self.total)
            _require(remaining > 0, "selected input exceeds 32 MiB")
            before = _identity(root / name)
            value = read_package_file(root, name, max_bytes=remaining, expected_sha256=expected)
            _require(before == _identity(root / name), "selected member changed during read")
            self.identities[key] = before
            self.raw[key] = value
            self.total += len(value)
        value = self.raw[key]
        _require(len(value) <= maximum and (expected is None or sha256(value).hexdigest() == expected),
                 "selected member has conflicting size or identity")
        _check(self.deadline)
        return value

    def bundle(self, root, expected):
        raw = self.read(root, "manifest.json", 1 << 20, expected)
        manifest = _document(raw, self.deadline)
        files = manifest.get("files")
        _require(type(files) is dict and 1 <= len(files) <= 16384 and "manifest.json" not in files
                 and all(_pin(value) for value in files.values()), "bounded hash-only source file map required")
        validate_package_layout(files)
        return raw, manifest, files

    def member(self, root, files, name, maximum):
        _require(name in files, "source seal omits " + name)
        return self.read(root, name, maximum, files[name])

    def recheck(self):
        for (root, name), identity in self.identities.items():
            _check(self.deadline)
            _require(_identity(root / name) == identity, "selected member changed before publication")


class _Output:
    def __init__(self, root, deadline):
        self.root, self.deadline, self.files, self.total = root, deadline, {}, 0
        self.terminal_manifest = None

    def write(self, name, raw, *, reserve=True):
        _check(self.deadline)
        _require(type(raw) is bytes and name not in self.files, "output requires new exact bytes")
        _require(len(self.files) + 1 + (3 if reserve else 0) <= MAX_FILES
                 and self.total + len(raw) + ((256 << 10) if reserve else 0) <= MAX_OUTPUT,
                 "complete composition exceeds output file/byte bounds")
        write_package_files(self.root, {name: raw})
        self.files[name] = sha256(raw).hexdigest()
        self.total += len(raw)
        _check(self.deadline)

    def publish(self, raw):
        _check(self.deadline)
        _require(len(self.files) + 1 <= MAX_FILES and self.total + len(raw) <= MAX_OUTPUT,
                 "terminal seal exceeds output bounds")
        self.terminal_manifest = raw
        # This verified write is terminal; later caller timing cannot revoke a consumable seal.
        write_package_files(self.root, {"manifest.json": raw})

    def failure(self, error, started):
        # Failure publication never reads sources again or clears a failed parent.
        self.deadline = time.monotonic() + 2.
        failure = {"schema_version": 1, "profile": PROFILE, "status": "failed", "sealed": False,
            "error_type": type(error).__name__, "reason": str(error)[:1500],
            "elapsed_seconds": time.monotonic() - started, "scope": SCOPE}
        if (self.root / "manifest.json").exists():
            original = self.root / "manifest.json"
            raw = read_package_file(self.root, "manifest.json", max_bytes=256 << 10)
            if raw == self.terminal_manifest:
                caller = {**failure, "status": "caller_failed_after_terminal_publication",
                    "artifact_status": "completed", "output_path": str(self.root),
                    "artifact_manifest_sha256": sha256(raw).hexdigest(), "artifact_unchanged": True}
                write_package_files(self.root.parent, {self.root.name + ".caller-failure.json":
                                                       _encoded(caller, self.deadline, 8 << 10)})
                return
            # An invalid partial write is evidence, not a published terminal artifact.
            import os
            os.link(original, self.root / "attempted-manifest.json", follow_symlinks=False)
            original.unlink()
            if "manifest.json" not in self.files:
                self.total += len(raw)
            self.files["attempted-manifest.json"] = self.files.pop("manifest.json", sha256(raw).hexdigest())
            _require(sha256(raw).hexdigest() == self.files["attempted-manifest.json"], "late seal custody differs")
        self.write("failure.json", _encoded(failure, self.deadline, 8 << 10), reserve=False)
        self.write("manifest.json", _encoded({"schema_version": 1, "profile": PROFILE, "status": "failed",
            "files": dict(self.files)}, self.deadline, 256 << 10), reserve=False)


def _complete_pose(pose, raw, deadline):
    frames, original = pose.get("frames"), raw.get("frames")
    decoded, metadata, stride = (pose.get(key) for key in ("decoded_frames", "frame_count", "inference_stride"))
    _require(all(type(value) is int and 1 <= value <= 1800 for value in (decoded, metadata, stride))
             and type(frames) is list and type(original) is list and 1 <= len(frames) <= 900 and len(original) <= 900,
             "complete bounded pose slots and explicit original inference stride required")
    expected = list(range(0, decoded, stride))
    _require(len(expected) <= 900 and all(type(row) is dict and type(row.get("frame")) is int for row in frames + original)
             and [row["frame"] for row in frames] == [row["frame"] for row in original] == expected,
             "context and precursor must retain every scheduled original pose, including missing poses")
    qualification = "within_original_one_frame_container_tolerance"
    if abs(decoded - metadata) > 1:
        audit = pose.get("container_frame_audit")
        if audit is not None:
            _require(type(audit) is dict and audit.get("method") == "Matroska unlaced video-block count"
                     and type(audit.get("encoded_video_frames")) is int and audit["encoded_video_frames"] == decoded
                     and type(audit.get("decoded_frames")) is int and audit["decoded_frames"] == decoded
                     and audit.get("matches") is True, "original encoded-frame audit is inconsistent")
            qualification = "saved_original_matroska_block_audit_not_reexecuted"
        else:
            qualification = "requires_recorded_complete_frame_proof"
    _check(deadline)
    return expected[:math.floor(.6 * len(expected))], qualification


def _parent_pose(reader, output, root, expected):
    seal, parent, pins = reader.bundle(root, expected)
    output.write("inputs/source_manifest.json", seal)
    ordinary = (type(parent.get("schema_version")) is int and parent["schema_version"] == 1
                and parent.get("profile") == "exercise-body-source-v1"
                and parent.get("status") in ("construction_only", "failed")
                and parent.get("simulation_ready") is False and "schema" not in parent)
    legacy = type(parent.get("schema")) is int and parent["schema"] == 1 and "schema_version" not in parent
    _require(ordinary or legacy, "known ordinary body-source or legacy hash-only pose seal required")
    failed = ordinary and parent["status"] == "failed"
    _require(failed or "failure.json" not in pins, "successful/unknown parent contradicts its failure marker")
    if failed:
        marker = reader.member(root, pins, "failure.json", 128 << 10)
        failure = _document(marker, reader.deadline)
        _require(failure.get("profile") == "exercise-body-source-v1" and failure.get("status") == "failed",
                 "failed ordinary parent needs its exact sealed failure marker")
        output.write("inputs/pose-parent-failure.json", marker)
    names = (("evidence.json", "inputs/context.json", 8 << 20),
             ("pose-landmarks.json", "inputs/raw.json", 8 << 20),
             ("identity-audit.json", "inputs/audit.json", 2 << 20))
    supplied = {}
    for name, destination, maximum in names:
        supplied[name] = reader.member(root, pins, name, maximum)
        output.write(destination, supplied[name])
    pose, raw = (_document(supplied[name], reader.deadline) for name in ("evidence.json", "pose-landmarks.json"))
    selected, qualification = _complete_pose(pose, raw, reader.deadline)
    for name in ("source.json", "run-config.json", "model.json"):
        if ordinary and name != "model.json":
            _require(name in pins, "ordinary pose source omits " + name)
        if ordinary and not failed and name == "model.json":
            _require(name in pins, "completed ordinary source omits model metadata")
        if name not in pins:
            continue
        value = reader.member(root, pins, name, 128 << 10)
        document = _document(value, reader.deadline)
        if name == "source.json" and ordinary:
            _require(document.get("sha256") == pose.get("source_sha256")
                     and type(document.get("artifact")) is str
                     and pins.get(document["artifact"]) == pose.get("source_sha256"),
                     "original source/rights metadata and sealed media identity differ")
        if name == "run-config.json" and ordinary:
            _require(document.get("profile") == "exercise"
                     and document.get("pose_identity") == pose.get("pose_identity", {}).get("mode"),
                     "original pose options contradict the identity audit")
        if name == "model.json":
            _require(document == pose.get("model"), "external and embedded original model metadata differ")
        output.write("inputs/pose-" + name, value)
    description = {"manifest_sha256": sha256(seal).hexdigest(),
        "profile": "exercise-body-source-v1" if ordinary else "legacy_hash_only_run",
        "status": parent["status"] if ordinary else "not_declared_by_hash_only_seal",
        "failure_retained": failed, "external_model_metadata": "retained" if "model.json" in pins else "not_supplied",
        "body_admission": "not_requested_or_evaluated", "source_media_verification": "sealed_hash_join_only_no_media_read",
        "decode_completeness": qualification}
    return pose, supplied, seal, selected, description


def _retention_inventory(raw, masks, mask_pin, pins, deadline):
    document = _document(raw, deadline)
    _require(type(document.get("schema_version")) is int and document["schema_version"] == 1
             and document.get("profile") == "decoded-segmentation-binary-retention-v1"
             and document.get("mask_document_sha256") == mask_pin, "exact opt-in decoded binary inventory required")
    records = document.get("records")
    _require(type(records) is list and len(records) <= 4000, "bounded full mask retention inventory required")
    rows = masks.get("observations")
    _require(type(rows) is list and len(rows) <= 4000, "bounded original mask observations required")
    by_id = {row.get("id"): row for row in rows if type(row) is dict and type(row.get("id")) is str}
    _require(len(by_id) == len(rows), "unique original mask identities required")
    inventory = {}
    for record in records:
        _check(deadline)
        _require(type(record) is dict and set(record) == {"path", "png_sha256", "png_bytes", "binary_sha256",
                 "binary_bytes", "dimensions_hw", "mask_evidence_id"}, "exact retained binary descriptor required")
        path = record["path"]
        _require(type(path) is str and 0 < len(path) <= 160 and path.endswith(".png")
                 and not PurePosixPath(path).is_absolute() and str(PurePosixPath(path)) == path
                 and ".." not in PurePosixPath(path).parts and "\\" not in path and path not in inventory,
                 "unique normalized retained raster path required")
        dims = record["dimensions_hw"]
        _require(type(dims) is list and len(dims) == 2 and all(type(v) is int and 1 <= v <= 4096 for v in dims)
                 and type(record["binary_bytes"]) is int and record["binary_bytes"] == math.prod(dims)
                 and type(record["png_bytes"]) is int and 0 < record["png_bytes"] <= MAX_INPUT
                 and type(record["mask_evidence_id"]) is str and 0 < len(record["mask_evidence_id"]) <= 192
                 and _pin(record["png_sha256"]) and _pin(record["binary_sha256"]), "invalid retained raster bounds/pins")
        row = by_id.get(record["mask_evidence_id"])
        binary = "binary/" + path.removesuffix(".png") + ".bin"
        _require(row is not None and row.get("mask_status") == "decoded" and row.get("raster_path") == path
                 and row.get("raster_dimensions") == dims and row.get("raster_sha256") == record["binary_sha256"]
                 and pins.get(path) == record["png_sha256"] and pins.get(binary) == record["binary_sha256"],
                 "retained binary/PNG descriptor differs from original mask row/seal")
        inventory[path] = (record, binary)
    _require(set(inventory) == {row.get("raster_path") for row in rows if row.get("mask_status") == "decoded"},
             "binary retention must cover every decoded mask; missing/failed masks remain unavailable")
    return inventory


def _runtime_archive(deadline):
    # One archive preserves loaded source custody without consuming hundreds of package members.
    package, files, total = Path(__file__).parent, {}, 0
    modules = sorted((name, module) for name, module in tuple(sys.modules.items())
                     if name == "weft" or name.startswith("weft."))
    _require(len(modules) <= 128, "loaded Weft module inventory exceeds bound")
    for name, module in modules:
        _check(deadline)
        path = getattr(module, "__file__", None)
        _require(path is not None, "loaded Weft module lacks source custody")
        path = _path(path)
        _require(path.is_relative_to(package) and path.suffix == ".py", "loaded Weft source is outside package")
        relative = path.relative_to(package).as_posix()
        if relative in files:
            continue
        raw = read_package_file(package, relative, max_bytes=512 << 10)
        total += len(raw)
        _require(total <= 8 << 20, "loaded source snapshot exceeds eight MiB")
        files[relative] = raw
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, raw in files.items():
            _check(deadline)
            archive.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), raw)
    _check(deadline)
    return stream.getvalue(), {name: sha256(raw).hexdigest() for name, raw in files.items()}


def prepare_exercise_equipment_evidence(pose_run, equipment_run, masks_run, out, *, human_id,
        equipment_ids, minimum_confidence, uncertainty_px, uncertainty_method, uncertainty_note,
        source_equivalence_run=None, expected_pose_manifest_sha256=None,
        expected_equipment_manifest_sha256=None, expected_masks_manifest_sha256=None,
        max_seconds=30., progress=None):
    """Compose sealed source members; camera/contact priors and body admission remain separate."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 1 <= max_seconds <= 30,
             "cooperative bound must be 1–30 seconds")
    _require(type(human_id) is str and 0 < len(human_id) <= 80 and type(equipment_ids) in (list, tuple)
             and 1 <= len(equipment_ids) <= 8 and all(type(v) is str and 0 < len(v) <= 80 for v in equipment_ids)
             and len(set(equipment_ids)) == len(equipment_ids), "explicit original human/equipment IDs required")
    _require(all(value is None or _pin(value) for value in (expected_pose_manifest_sha256,
                 expected_equipment_manifest_sha256, expected_masks_manifest_sha256)), "invalid expected source seal")
    for value, low, high in ((minimum_confidence, 0, 1), (uncertainty_px, 0, 64)):
        _require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
                 "explicit bounded authored confidence and pixel uncertainty required")
    _require(all(type(value) is str and 0 < len(value.strip()) <= 192 and len(value) <= 192
                 for value in (uncertainty_method, uncertainty_note)), "explicit bounded uncertainty disclosure required")
    started = time.monotonic(); deadline = started + max_seconds
    pose_root, equipment_root, mask_root, target = map(_path, (pose_run, equipment_run, masks_run, out))
    proof_root = None if source_equivalence_run is None else _path(source_equivalence_run)
    roots = (pose_root, equipment_root, mask_root) + (() if proof_root is None else (proof_root,))
    _require(all(root.is_dir() for root in roots) and target.parent.is_dir() and not target.exists()
             and all(target != root and target not in root.parents and root not in target.parents for root in roots),
             "new output with existing parent must be outside every input")
    target.mkdir(mode=0o700)
    reader, output = _Inputs(deadline), _Output(target, deadline)
    def emit(stage, message):
        if progress is not None:
            progress(stage, message)
        _check(deadline)
    def reserve(seconds):
        _check(deadline)
        _require(deadline - time.monotonic() >= seconds, "insufficient remaining time for next bounded stage")
    try:
        emit("inputs", "Read exact sealed pose, equipment and retained masks; no body or media work")
        pose, supplied, parent_seal, selected, parent = _parent_pose(reader, output, pose_root, expected_pose_manifest_sha256)
        equipment_seal, _, equipment_pins = reader.bundle(equipment_root, expected_equipment_manifest_sha256)
        mask_seal, _, mask_pins = reader.bundle(mask_root, expected_masks_manifest_sha256)
        output.write("inputs/equipment_manifest.json", equipment_seal)
        output.write("inputs/mask_manifest.json", mask_seal)
        selected_members = ((equipment_root, equipment_pins, "scene-observations.json", "inputs/equipment.json", 16 << 20),
            (equipment_root, equipment_pins, "report.json", "inputs/equipment-report.json", 256 << 10),
            (mask_root, mask_pins, "mask-evidence.json", "inputs/masks.json", 8 << 20),
            (mask_root, mask_pins, "report.json", "inputs/masks-report.json", 1 << 20),
            (mask_root, mask_pins, "binary-retention.json", "inputs/binary-retention.json", 2 << 20))
        values = {}
        for root, pins, name, destination, maximum in selected_members:
            values[destination] = reader.member(root, pins, name, maximum)
            output.write(destination, values[destination])
        equipment = _document(values["inputs/equipment.json"], deadline)
        masks = _document(values["inputs/masks.json"], deadline)
        _require(_document(values["inputs/equipment-report.json"], deadline).get("status") == "evidence_only",
                 "completed canonical tracking-evidence export required")
        mask_report = _document(values["inputs/masks-report.json"], deadline)
        _require(mask_report.get("status") in ("decoded_proposals", "partial_decode"),
                 "completed/partial-decode mask evidence required, not a failed export")
        verification = mask_report.get("source_verification")
        _require(type(verification) is dict and verification.get("manifest_sha256") == sha256(equipment_seal).hexdigest()
                 and verification.get("files") == equipment_pins,
                 "mask export must retain the exact equipment parent seal and member identities")
        mask_pin = sha256(values["inputs/masks.json"]).hexdigest()
        inventory = _retention_inventory(values["inputs/binary-retention.json"], masks, mask_pin, mask_pins, deadline)
        proof = None
        if equipment.get("source", {}).get("sha256") != pose.get("source_sha256"):
            from weft.source_equivalence import load_source_equivalence
            _require(proof_root is not None, "different original/equipment sources need complete verified source equivalence")
            proof_seal, _, proof_pins = reader.bundle(proof_root, None)
            _require(1 <= len(proof_pins) <= 128, "bounded complete proof inventory required")
            proof_names = {*proof_pins, "manifest.json"}
            before_proof = {}
            for name in sorted(proof_names):
                _check(deadline)
                before_proof[name] = _identity(proof_root / name)
                _require((proof_root, name) not in reader.identities
                         or reader.identities[(proof_root, name)] == before_proof[name],
                         "proof manifest changed before verification")
            proof = load_source_equivalence(proof_root, max_input_bytes=MAX_INPUT - reader.total, max_files=129, deadline=deadline)
            _require(set(proof.file_names) == proof_names and proof.raw_bytes("manifest.json") == proof_seal,
                     "verified proof inventory changed from its bounded manifest")
            reader.total += proof.total_input_bytes
            for name in proof.file_names:
                _check(deadline)
                after = _identity(proof_root / name)
                _require(after == before_proof[name], "proof changed during verification")
                reader.identities[(proof_root, name)] = after
                output.write("source-equivalence/" + name, proof.raw_bytes(name))
        else:
            _require(proof_root is None, "same-source composition must not adopt an unrelated source mapping")
        if parent["decode_completeness"] == "requires_recorded_complete_frame_proof":
            from fractions import Fraction
            _require(proof is not None and proof.original_sha256 == pose["source_sha256"]
                     and proof.derivative_sha256 == equipment["source"]["sha256"]
                     and proof.dimensions == (pose["width"], pose["height"])
                     and len(proof.frames) == pose["decoded_frames"],
                     "larger container/decode discrepancy needs exact whole-video frame proof")
            for row in pose["frames"]:
                _check(deadline)
                stamp = row.get("decoder_timestamp_s")
                expected = float(proof.frames[row["frame"]].source_decoder_pts * Fraction(proof.time_base))
                _require(type(stamp) in (int, float) and math.isfinite(stamp)
                         and math.isclose(stamp, expected, rel_tol=0, abs_tol=1e-9),
                         "complete original sampled decoder clocks differ from source proof")
            parent["decode_completeness"] = "recorded_whole_video_source_equivalence_not_independent_decode"
        from weft.pose_precursor_custody import prepare_pose_precursor_custody
        reserve(5.)
        custody = prepare_pose_precursor_custody(context_pose_document=supplied["evidence.json"],
            raw_pose_document=supplied["pose-landmarks.json"], identity_audit_document=supplied["identity-audit.json"],
            manifest_document=parent_seal)
        from weft.observations import adapt_exercise
        emit("annotations", "Qualify the fixed original pose prefix; retain every missing and excluded slot")
        human = adapt_exercise(pose, construction_frames=selected)
        _require(human_id in {row["id"] for row in human["objects"]}, "human ID absent from original canonical pose document")
        human_raw = _encoded(human, deadline, 8 << 20); human_pin = sha256(human_raw).hexdigest()
        output.write("human-observations.json", human_raw)
        channels = [{"id": side + "-wrist", "entity_ref": {"document_sha256": human_pin, "object_id": human_id},
                     "point_role": side + "_wrist", "surface_id": "unknown"} for side in ("left", "right")]
        from weft.pose_surface_annotations import prepare_pose_surface_annotations
        reserve(10.)
        annotations = prepare_pose_surface_annotations({human_pin: human_raw}, pose_document=supplied["evidence.json"],
            source_sha256=pose["source_sha256"], channels=channels,
            landmark_bindings=[{"channel_id": row["id"], "landmark_name": row["point_role"], "uncertainty_px": uncertainty_px}
                               for row in channels], minimum_confidence=minimum_confidence,
            uncertainty_method=uncertainty_method, uncertainty_note=uncertainty_note, precursor_custody=custody)
        for name, raw in {"source-annotations.json": annotations.annotation_document(),
            "annotation-metadata.json": annotations._metadata + b"\n",
            "annotation-frames.jsonl": b"\n".join(annotations._rows) + b"\n"}.items():
            output.write(name, raw)
        reads, binary_total, used = [], 0, set()
        def retain(path):
            nonlocal binary_total
            _check(deadline)
            _require(path in inventory, "required retained mask is absent")
            row, binary = inventory[path]
            if path in used:
                return reader.raw[(mask_root, binary)]
            _require(len(used) < MAX_LOADS and binary_total + row["binary_bytes"] <= MAX_BINARY,
                     "complete eligible mask inventory exceeds selected load/byte bounds")
            raw = reader.member(mask_root, mask_pins, binary, row["binary_bytes"])
            _require(len(raw) == row["binary_bytes"] and raw.count(b"\0") + raw.count(b"\1") == len(raw), "exact binary cells required")
            png = reader.member(mask_root, mask_pins, path, row["png_bytes"])
            _require(len(png) == row["png_bytes"], "original PNG length differs; no decoder fallback")
            output.write(binary, raw); output.write("pngs/" + path, png)
            reads.append(row); used.add(path); binary_total += len(raw)
            return raw
        def raster(request):
            _check(min(deadline, request["deadline_monotonic"]))
            path = request["path"]
            _require(path in inventory, "required retained mask is absent")
            row, _ = inventory[path]
            _require(request["mask_document_sha256"] == mask_pin
                     and request["mask_evidence_id"] == row["mask_evidence_id"]
                     and request["raster_sha256"] == row["binary_sha256"] and request["dimensions"] == row["dimensions_hw"]
                     and request["maximum_bytes"] == row["binary_bytes"]
                     and request["remaining_total_bytes"] >= row["binary_bytes"]
                     and path not in used, "requested mask differs from bounded retained descriptor")
            return retain(path)
        from weft.equipment_silhouette_evidence import prepare_equipment_silhouette_evidence
        emit("silhouettes", "Compare authored wrist rectangles with saved binary cells; physical contact stays unknown")
        reserve(10.)
        silhouettes = prepare_equipment_silhouette_evidence(annotations, equipment_document=values["inputs/equipment.json"],
            mask_document=values["inputs/masks.json"], equipment_ids=equipment_ids,
            wrist_channel_ids=[row["id"] for row in channels], raster_loader=raster,
            source_equivalence=proof, construction_frames=selected)
        # Qualified fingers may need a mask even when both wrist estimates are unavailable.
        from weft.equipment_silhouette_evidence import _qualified_mask
        from weft.exercise_support_evidence import _equipment_point
        human_rows = {row["frame"]: row for row in human["observations"] if row["object_id"] == human_id}
        equip_rows = {(row["frame"], row["object_id"]): row for row in equipment["observations"]}
        mask_rows = {(row["frame"], row["object_id"]): row for row in masks["observations"]}
        records = {row["id"]: row for row in equipment["evidence"]}
        for frame in selected:
            _check(deadline)
            observed_human = human_rows.get(frame)
            if (observed_human is None or observed_human["partition"] != "construction"
                    or observed_human["visibility"] not in ("detected", "partial")):
                continue
            slot = silhouettes.frame(frame)
            supplied = {"frame": frame, "time_s": slot["pose_time_s"], "decoder_timestamp_s": slot["pose_decoder_timestamp_s"]}
            for identity in equipment_ids:
                key = (slot["equipment_source_frame"], identity)
                observed, mask = equip_rows.get(key), mask_rows.get(key)
                if (observed is None or observed["partition"] != "construction" or mask is None
                        or mask.get("mask_status") != "decoded" or mask.get("phase") == "pre_initialization"):
                    continue
                box, reason = _equipment_point(observed, supplied, equipment["schema_version"] == 3, slot["source_mapping"])
                if reason is None and _qualified_mask(mask, observed, records, box, slot["source_mapping"],
                                                      pose["source_sha256"], proof, deadline) is None:
                    retain(mask["raster_path"])
        for name, raw in {"silhouette/metadata.json": silhouettes._metadata + b"\n",
            "silhouette/frames.jsonl": b"\n".join(silhouettes._rows) + b"\n",
            "raster-reads.json": _encoded(reads, deadline, 512 << 10)}.items():
            output.write(name, raw)
        archive, runtime_pins = _runtime_archive(deadline)
        output.write("implementation-source.zip", archive)
        result = {"schema_version": 1, "profile": PROFILE, "status": "completed", "sealed": True,
            "output_path": str(target), "created_at_local": local_time(), "parent_pose": parent,
            "equipment_manifest_sha256": sha256(equipment_seal).hexdigest(), "masks_manifest_sha256": sha256(mask_seal).hexdigest(),
            "construction_frames": selected, "selection_policy": "first_floor_0.6_of_all_original_sampled_pose_rows_before_point_queries",
            "human_id": human_id, "equipment_ids": list(equipment_ids), "minimum_confidence": minimum_confidence,
            "uncertainty_px": uncertainty_px, "uncertainty_method": uncertainty_method, "uncertainty_note": uncertainty_note,
            "counts": silhouettes.describe()["counts"], "annotation_counts": annotations.describe()["counts"],
            "input_bytes": reader.total, "input_files": len(reader.identities),
            "input_read_operations": len(reader.raw) + (0 if proof is None else len(proof.file_names)),
            "proof_manifest_repeat_bytes": 0 if proof is None else len(proof_seal),
            "raster_loads": silhouettes.describe()["work"]["raster_loads"],
            "retained_binary_loads": len(reads), "binary_bytes": binary_total, "source_decodes": 0, "raster_decodes": 0,
            "pose_model_calls": 0, "native_calls": 0, "api_calls": 0, "contact": "unknown", "grasp": "unknown",
            "load": "unknown", "body_or_floor_installed": False, "physics_ready": False, "scope": SCOPE,
            "runtime_sha256s": runtime_pins, "runtime_scope": "Loaded local Python sources only; not a standalone interpreter/dependency runtime",
            "time_limit_kind": "cooperative_not_hard_process_limit", "prepared_seconds": time.monotonic() - started,
            "bounds": {"input_bytes": MAX_INPUT, "output_bytes": MAX_OUTPUT, "output_files": MAX_FILES,
                "raster_loads": MAX_LOADS, "binary_bytes": MAX_BINARY, "maximum_seconds": max_seconds}}
        output.write("result.json", _encoded({**result, "status": "prepared_pending_terminal_seal", "sealed": False},
                                            deadline, 128 << 10))
        emit("seal", "Publish a new image-evidence package; original parent status and all unknowns remain unchanged")
        reader.recheck()
        manifest = _encoded({"schema_version": 1, "profile": PROFILE, "status": "completed",
            "files": dict(output.files)}, deadline, 256 << 10)
        returned = {**result, "manifest_sha256": sha256(manifest).hexdigest()}
        output.publish(manifest)
        return returned
    except Exception as error:
        try:
            output.failure(error, started)
        except (OSError, ValueError, TimeoutError):
            pass
        raise
