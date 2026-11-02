"""Bounded, source-only Full/IMAGE crop proposals for the existing recovery verifier."""

from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import stat
import time

import cv2
import numpy as np
from PIL import Image

from weft import exercise_vision
from weft.errors import UnsupportedVideo
from weft.exercise_body_source import BodyArtifactSink, _CappedBuffer, _inventory
from weft.exercise_source import describe_source
from weft.io import local_time
from weft.pose_crop import map_crop_pose
from weft.pose_crop_selection import CONTRACT, METHOD, apply_selective_crop, assess_selective_crop, plan_selective_crop
from weft.pose_source_recovery import INPUT_FILES, SUFFIXES, _bytes, _json, _path, _read, _sha, _validate_baseline
from weft.shared_world_packaging import write_package_files


PROFILE = "exercise-pose-crop-proposals-v1"
MAX_SECONDS = 180
MAX_BYTES = 128 * 1024 * 1024
MAX_FILES = 510
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_JOURNAL_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
RESERVE_BYTES = 256 * 1024
MAX_PREVIEWS = 48
IMPLEMENTATION_FILES = ("pose_crop_pipeline.py", "exercise_body_source.py", "exercise_vision.py",
    "pose_crop_selection.py", "pose_crop.py", "pose_source_recovery.py", "exercise_source.py",
    "video_io.py", "shared_world_packaging.py", "world_evidence.py", "io.py", "errors.py")
SOURCE_NOTE = "Rights not independently verified; user-provided local video"


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Native pose proposals exceeded their shared deadline")


class _Budget:
    """One reservation covers both pose trees, candidate pixels and terminal metadata."""

    def __init__(self, root, maximum, deadline):
        self.root, self.maximum, self.deadline = root, maximum, deadline
        self.files, self.total, self.sinks = {}, 0, {}

    def sink(self, directory=""):
        if directory not in self.sinks:
            root = self.root / directory
            if directory:
                root.mkdir(mode=0o700, parents=True, exist_ok=False)
            self.sinks[directory] = _Sink(root, self)
        return self.sinks[directory]

    def committed(self, sink, name):
        key = (sink.root / name).relative_to(self.root).as_posix()
        size = sink.owned[name][2]
        self.total += size - self.files.get(key, 0)
        self.files[key] = size


class _Sink(BodyArtifactSink):
    def __init__(self, root, budget):
        super().__init__(root, deadline=budget.deadline)
        self.budget = budget

    def _buffer(self, name):
        _check(self.deadline)
        if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Proposal artifacts require a single safe filename")
        path, old = self.root / name, self.owned.get(name)
        if old is None and (path.exists() or path.is_symlink()):
            raise ValueError("Proposal output cannot replace an existing artifact")
        if old is not None:
            info = path.stat(follow_symlinks=False)
            if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != old:
                raise ValueError("Proposal artifact changed outside its owner")
        key = path.relative_to(self.budget.root).as_posix()
        if len(self.budget.files) + (key not in self.budget.files) > MAX_FILES - 3:
            raise ValueError("Proposal output exceeds its aggregate file reservation")
        capacity = min(MAX_FILE_BYTES, self.budget.maximum - RESERVE_BYTES - self.budget.total
                       + (old[2] if old else 0))
        if capacity <= 0:
            raise ValueError("Proposal output exceeds its aggregate byte reservation")
        return _CappedBuffer(capacity, self.deadline)

    def _commit(self, name, buffer):
        super()._commit(name, buffer)
        self.budget.committed(self, name)

    def write_bytes(self, name, raw):
        with self._buffer(name) as buffer:
            buffer.write(raw)
            self._commit(name, buffer)

    def append_candidate(self, value):
        name = "crop-observations.jsonl"
        old = self.owned.get(name)
        size = old[2] if old else 0
        with self._buffer(name) as reservation:
            maximum = min(MAX_LINE_BYTES, MAX_JOURNAL_BYTES - size, reservation.maximum - size)
        with _CappedBuffer(maximum, self.deadline) as buffer:
            for piece in json.JSONEncoder(sort_keys=True, allow_nan=False).iterencode(value):
                buffer.write(piece.encode())
            buffer.write(b"\n")
            # The journal preserves each completed proposal without repeatedly rewriting its prefix.
            flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            flags |= os.O_APPEND if old else os.O_CREAT | os.O_EXCL
            root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                fd = os.open(name, flags, 0o600, dir_fd=root_fd)
                with os.fdopen(fd, "ab") as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode) or (old is not None and
                            (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != old):
                        raise ValueError("Candidate journal changed before append")
                    stream.write(buffer.getvalue())
                    stream.flush()
                    info = os.fstat(stream.fileno())
                    self.owned[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                self.budget.committed(self, name)
            finally:
                os.close(root_fd)
        _check(self.deadline)


def _finish(evidence, directory, sink, raw=None):
    if raw is not None:
        sink.write_bytes("pose-landmarks.json", raw)
        exercise_vision.resolve_pose_identity(evidence, directory, "raw", artifact_sink=sink)
    error = None
    try:
        exercise_vision.finalize_evidence(evidence, directory, artifact_sink=sink)
    except UnsupportedVideo as exc:
        if evidence.get("quality", {}).get("passes") is not False:
            raise
        error = str(exc)
    sink.write_json("pose-quality.json", evidence["quality"])
    sink.write_json("report.json", {"status": "complete_pose_evidence", "quality_error": error,
        "sampled_frames": len(evidence["frames"]), "source_sha256": evidence["source_sha256"],
        "scope": "Source-only model observations; not a constructed body or physical validation"})
    return error


def _decoded_images(video, evidence, deadline, audit):
    cap = cv2.VideoCapture(str(video))
    try:
        fields = (("width", cv2.CAP_PROP_FRAME_WIDTH), ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                  ("fps", cv2.CAP_PROP_FPS), ("frame_count", cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or any(cap.get(prop) != evidence[key] for key, prop in fields):
            raise ValueError("Crop decoder metadata differs from the original baseline")
        originals = {row["frame"]: row for row in evidence["frames"]}
        for index in range(evidence["decoded_frames"] + 1):
            _check(deadline)
            ok, image = cap.read()
            _check(deadline)
            if index == evidence["decoded_frames"]:
                if ok:
                    raise ValueError("Crop decoder contains frames beyond the original interval")
                audit["complete"] = True
                break
            if not ok or image.shape != (evidence["height"], evidence["width"], 3) or image.dtype != np.uint8:
                raise ValueError("Crop decoder ended early or changed image dimensions/type")
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != index + 1:
                raise ValueError("Crop decoder frame index differs from the exact sequential source slot")
            stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            stamp = stamp if math.isfinite(stamp) else None
            audit["decoded_frames"] += 1
            if index in originals:
                expected = originals[index].get("decoder_timestamp_s")
                if stamp != expected:
                    raise ValueError("Crop decoder timestamp differs from the original source clock")
                audit["sampled_clocks_checked"] += 1
                if stamp is None:
                    audit["unknown_native_timestamps"] += 1
            yield index, image
    finally:
        cap.release()


def _failure(target, maximum, error, started, work):
    work = dict(work)
    if not work.pop("baseline_complete", False):
        raw_path = target / "whole_image/pose-landmarks.json"
        completed = None
        try:
            completed = len(_json(_read(raw_path, time.monotonic() + 1))["frames"]) if raw_path.is_file() else 0
        except (ValueError, OSError, TimeoutError, KeyError, TypeError):
            pass
        work.update(whole_completed_samples=completed, whole_inference_calls=None,
            inference_count_scope="A failed native baseline call may have started without returning a saved observation")
    failure = {"schema_version": 1, "profile": PROFILE, "status": "failed", "simulation_ready": False,
        "created_at": local_time(), "seconds": time.monotonic() - started, "work": work,
        "error": {"type": type(error).__name__, "message": str(error)[:2000]}}
    try:
        write_package_files(target, {"failure.json": _bytes(failure)})
        files, total = _inventory(target, time.monotonic() + 2)
        seal = _bytes({"schema_version": 1, "profile": PROFILE, "status": "failed",
                       "simulation_ready": False, "files": files})
        if len(files) + 1 <= MAX_FILES and total + len(seal) <= maximum:
            write_package_files(target, {"manifest.json": seal})
    except (ValueError, OSError, TimeoutError):
        pass


def prepare_pose_crop_proposals(video, out, *, pose_model, source_note=SOURCE_NOTE,
                               max_seconds=180, deadline=None, progress=None, decoder_clock_sink=None):
    """Generate one fixed Full/IMAGE/raw proposal package; body admission remains separate."""
    if decoder_clock_sink is not None and not callable(decoder_clock_sink):
        raise ValueError("Decoder clock sink must be callable")
    if (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
            or not 1 <= max_seconds <= MAX_SECONDS):
        raise ValueError("Native pose proposals require a 1–180 second bound")
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("Proposal deadline must be a finite monotonic time")
    if not isinstance(source_note, str) or not source_note.strip() or len(source_note.encode()) > 4096:
        raise ValueError("Source note must be nonempty and at most 4096 UTF-8 bytes")
    started = time.monotonic()
    deadline = min(started + max_seconds, deadline if deadline is not None else started + max_seconds)
    video, model, target = map(_path, (video, pose_model, out))
    if (video.suffix.lower() not in SUFFIXES or not video.is_file() or not 0 < video.stat().st_size <= MAX_FILE_BYTES
            or not model.is_file() or model.stat().st_size > MAX_FILE_BYTES):
        raise ValueError("Proposals require a local video up to 64 MiB and the pinned Full model")
    if target.exists() or any(target.is_relative_to(p) or p.is_relative_to(target) for p in (video, model)):
        raise ValueError("Proposal output must be new and outside the video/model inputs")
    _check(deadline)
    provenance = exercise_vision.model_provenance(model, variant="full")
    source_size = video.stat().st_size
    source_sha = _sha(_read(video, deadline))
    maximum = MAX_BYTES - source_size - RESERVE_BYTES
    snapshots = {name: _read(Path(__file__).with_name(name), deadline) for name in IMPLEMENTATION_FILES}
    target.mkdir(mode=0o700, parents=True, exist_ok=False)
    budget = _Budget(target, maximum, deadline)
    work = {"video_decodes": 0, "whole_inference_calls": 0, "crop_inference_calls": 0,
            "physics_steps": 0, "body_models": 0, "api_calls": 0, "baseline_complete": False}
    try:
        root = budget.sink()
        whole = budget.sink("whole_image")
        source = describe_source(source_sha) or {"original_name": video.name, "rights": source_note}
        root.write_json("source.json", {**source, "sha256": source_sha, "bytes": source_size,
            "source_note": source_note, "original_bytes": "External original retained unchanged; passed explicitly to recovery"})
        root.write_json("model.json", provenance)
        root.write_json("settings.json", {"profile": PROFILE, "pose_model": str(model), "pose_mode": "image",
            "pose_identity": "raw", "source_note": source_note, "method": METHOD, "contract": dict(CONTRACT),
            "maximum_seconds": deadline - started, "maximum_proposal_bytes": maximum,
            "maximum_proposal_plus_source_bytes": MAX_BYTES - RESERVE_BYTES, "maximum_files": MAX_FILES,
            "maximum_file_bytes": MAX_FILE_BYTES, "maximum_journal_bytes": MAX_JOURNAL_BYTES,
            "maximum_decoded_frames_per_pass": 1800, "maximum_baseline_inferences": 900,
            "maximum_crop_inferences": 64, "maximum_decode_passes": 2, "maximum_previews_per_pass": MAX_PREVIEWS,
            "worker_count": 1, "scope": "Explicit alternative source recipe, never a default model/mode fallback",
            "deadline_scope": "Cooperative around native calls; caller subprocess must enforce hard wall/RSS limits"})
        implementation = budget.sink("implementation-source")
        for name, raw in snapshots.items():
            implementation.write_bytes(name, raw)
        versions = {"python": platform.python_version()}
        for name in ("numpy", "mediapipe", "opencv-contrib-python", "Pillow"):
            versions[name] = importlib.metadata.version(name)
        root.write_json("implementation.json", {"files": {n: _sha(raw) for n, raw in snapshots.items()},
            "runtime_versions": versions, "scope": "Selected generation/custody modules; not a complete Python runtime closure"})

        def report(stage, detail):
            _check(deadline)
            if stage == "tracking":
                work["whole_inference_calls"] = detail["sampled_frames"]
            if progress:
                progress("whole_image" if stage == "tracking" else stage, detail)
            _check(deadline)

        work["video_decodes"] += 1
        try:
            baseline = exercise_vision.inspect_exercise(video, whole.root, model, report,
                pose_mode="image", pose_identity="raw", artifact_sink=whole, max_preview_frames=MAX_PREVIEWS,
                **({"decoder_clock_sink": decoder_clock_sink} if decoder_clock_sink is not None else {}))
        except UnsupportedVideo:
            baseline = _json(_read(whole.root / "evidence.json", deadline))
            if baseline.get("quality", {}).get("passes") is not False:
                raise
        work["whole_inference_calls"] = len(baseline["frames"])
        work["baseline_complete"] = True
        raw = _read(whole.root / "pose-landmarks.json", deadline)
        _validate_baseline(baseline, _json(raw), _json(_read(whole.root / "identity-audit.json", deadline)), source_sha)
        before_error = _finish(baseline, whole.root, whole)
        plans = {}
        for row in baseline["frames"]:
            _check(deadline)
            plans[row["frame"]] = plan_selective_crop(baseline, row)
        eligible = [frame for frame, plan in plans.items() if plan["eligible"]]
        root.write_json("crop-plan.json", {"plans": list(plans.values()), "eligible_frames": eligible,
            "bound_exceeded": len(eligible) > CONTRACT["maximum_target_frames"],
            "context": "Complete original Full/IMAGE/raw sequence; no repaired anchors, labels, geometry or task feedback"})
        report("crop_planning", {"sampled_frames": len(plans), "eligible_frames": len(eligible)})
        if len(eligible) > CONTRACT["maximum_target_frames"]:
            raise ValueError("More than 64 eligible crops; preserve the complete baseline without partial candidate selection")
        root.write_text("crop-observations.jsonl", "")
        proposals, decisions, pixels = {}, {}, []
        originals = {row["frame"]: row for row in baseline["frames"]}
        preview_frames = set(list(originals)[::max(1, math.ceil(len(originals) / MAX_PREVIEWS))])
        previews, stamps = [], []
        audit = {"required": bool(eligible), "complete": False, "decoded_frames": 0,
            "sampled_clocks_checked": 0, "unknown_native_timestamps": 0,
            "status": "pending" if eligible else "not_needed",
            "scope": "Exact sequential indices, dimensions, decoded count and original sampled native timestamps; no retiming"}
        if eligible:
            work["video_decodes"] += 1
            crop_sink = budget.sink("crop-inputs")
            _check(deadline)
            mp, detector = exercise_vision._detector(model, "image")
            frames = _decoded_images(video, baseline, deadline, audit)
            try:
                for frame, bgr in frames:
                    if frame not in originals:
                        continue
                    original = originals[frame]
                    if frame in eligible:
                        crop = plans[frame]["crop"]
                        x1, y1, x2, y2 = crop["box_xyxy"]
                        image = np.ascontiguousarray(bgr[y1:y2, x1:x2])
                        name = f"frame-{frame:04d}.png"
                        crop_sink.write_image(name, Image.fromarray(image[:, :, ::-1]))
                        pixels.append({"frame": frame, "path": "crop-inputs/" + name, "box_xyxy": crop["box_xyxy"],
                            "source_bgr8_sha256": hashlib.sha256(memoryview(bgr).cast("B")).hexdigest(),
                            "crop_bgr8_sha256": hashlib.sha256(memoryview(image).cast("B")).hexdigest(),
                            "scope": "Decoded original pixels, not independent physical measurements"})
                        tick = time.perf_counter()
                        work["crop_inference_calls"] += 1
                        observed = exercise_vision.summarize_result(detector.detect(mp.Image(
                            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
                        _check(deadline)
                        proposed = map_crop_pose(observed, crop, baseline, original, time.perf_counter() - tick)
                        proposed["crop_proposal"] = crop
                        decision = assess_selective_crop(baseline, original, proposed, plans[frame])
                        root.append_candidate({"frame": frame, "raw_crop_result": observed,
                                               "proposed": proposed, "decision": decision})
                        proposals[frame], decisions[frame] = proposed, decision
                        report("crop_inference", {"frame": frame, "completed": len(proposals),
                                                   "planned": len(eligible), "accepted": decision["accepted"]})
                    if frame in preview_frames:
                        row = proposals[frame] if decisions.get(frame, {}).get("accepted") else original
                        previews.append(exercise_vision._draw_pose(bgr, row)); stamps.append(row["time_s"])
            finally:
                try:
                    frames.close()
                finally:
                    detector.close()
            audit["status"] = "verified"
        if list(proposals) != eligible:
            raise ValueError("Every eligible crop must have exactly one retained decision")
        root.write_json("decode-audit.json", audit)
        root.write_json("crop-inputs.json", pixels)
        root.write_json("decisions.json", list(decisions.values()))
        selected = apply_selective_crop(baseline, proposals, plans, decisions)
        chosen = budget.sink("selected")
        after_error = _finish(selected, chosen.root, chosen, raw)
        for before, after in zip(baseline["frames"], selected["frames"], strict=True):
            if any(before[k] != after[k] for k in ("id", "frame", "time_s", "decoder_timestamp_s")):
                raise ValueError("Selected poses changed source identities or clocks")
            if not decisions.get(before["frame"], {}).get("accepted") and before != after:
                raise ValueError("A rejected or unplanned original pose changed")
        for name in ("background.jpg", "tracks.gif", "tracks-contact-sheet.png"):
            if (whole.root / name).exists() and (not eligible or name == "background.jpg"):
                chosen.write_bytes(name, _read(whole.root / name, deadline))
        if previews:
            exercise_vision._save_previews(previews, stamps, chosen.root, artifact_sink=chosen)
        report("selected", {"sampled_frames": len(selected["frames"]),
            "accepted_frames": len(selected["pose_reinference"]["changed_frames"]),
            "pose_quality_passes": selected["quality"]["passes"]})
        _read(video, deadline, source_sha)
        _read(model, deadline, provenance["sha256"])
        if video.stat().st_size != source_size or any(importlib.metadata.version(name) != version
                for name, version in versions.items() if name != "python"):
            raise ValueError("Source size or inference runtime changed during proposal preparation")
        for name, original in snapshots.items():
            _read(Path(__file__).with_name(name), deadline, _sha(original))
        result = {"schema_version": 1, "profile": PROFILE, "status": "source_evidence_only", "simulation_ready": False,
            "created_at": local_time(), "source_sha256": source_sha, "model": baseline["model"],
            "source_frames": baseline["decoded_frames"], "sampled_frames": len(baseline["frames"]),
            "eligible_frames": eligible, "accepted_frames": selected["pose_reinference"]["changed_frames"],
            "rejected_frames": [f for f, decision in decisions.items() if not decision["accepted"]],
            "whole_quality_error": before_error, "selected_quality_error": after_error,
            "work": work, "seconds": time.monotonic() - started,
            "limits": ["Source-only consistency proposals, not verified anatomy, contact or a ready body/lab",
                "Competing hypotheses, unavailable context and failed pose quality remain visible",
                "No model tournament, chained refinement, comparison labels, geometry, task, reward or physics feedback",
                "Native call memory/wall limits require a caller-owned subprocess; encoded artifacts and image buffers are bounded here",
                "Original video/model are pinned and rechecked, not duplicated; recovery receives the original video explicitly"]}
        root.write_json("report.json", result)
        files, total = _inventory(target, deadline)
        if (set(files) != set(budget.files) or not set(INPUT_FILES) <= files.keys()
                or total != budget.total or total > maximum - RESERVE_BYTES or len(files) + 1 > MAX_FILES):
            raise ValueError("Final proposal inventory differs from its aggregate reservations")
        seal = _bytes({"schema_version": 1, "profile": PROFILE, "status": "source_evidence_only",
            "simulation_ready": False, "source_sha256": source_sha, "files": files})
        if total + len(seal) > maximum:
            raise ValueError("Proposal manifest exceeds its reserved terminal bytes")
        _check(deadline)
        write_package_files(target, {"manifest.json": seal})
        return target
    except Exception as error:
        _failure(target, maximum, error, started, work)
        raise
