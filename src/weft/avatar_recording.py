"""Appearance sidecars preserve the original physical journal and export contract."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from weft.avatar_assets import _read, load_avatar_package
from weft.avatar_gltf import safe_relative
from weft.avatar_lab import DEFAULT_APPEARANCE, MAX_AVATARS, validate_appearance
from weft.io import canonical_hash, digest, local_time
from weft.recording import Recorder, _atomic_json, _encoded


MAX_CHANGES = 1024
MAX_CHANGE_BYTES = 1024 * 1024


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _save_catalog(root, catalog):
    directory = root / "appearance-assets"
    directory.mkdir(exist_ok=False)
    files, assets = {}, {}
    for key, package in catalog.packages.items():
        asset_root = directory / key
        asset_root.mkdir()
        payload = {"originals/" + name: raw for name, raw in package.originals.items()}
        payload.update(package.texture_files)
        payload["render.json"] = package.render_json
        for name, raw in payload.items():
            path = asset_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
            files[path.relative_to(root).as_posix()] = _sha(raw)
        assets[key] = {"asset_id": package.manifest["asset_id"],
                       "display_name": package.manifest["display_name"],
                       "render_sha256": package.render_sha256,
                       "source_manifest_sha256": key,
                       "textures": deepcopy(package.render["textures"])}
    code = directory / "viewer"
    code.mkdir()
    for name, raw in catalog.viewer_files.items():
        path = code / name
        with path.open("xb") as stream:
            stream.write(raw)
        files[path.relative_to(root).as_posix()] = _sha(raw)
    receipt = {"schema_version": 1, "profile": "weft-recorded-avatar-assets-v1",
               "assets": assets, "files": files}
    _atomic_json(root / "appearance-assets.json", receipt)
    return receipt


class AppearanceRecorder(Recorder):
    def __init__(self, root, scene, *, profile, catalog, settings, executor):
        super().__init__(root, scene, profile=profile)
        engine = scene.get("engine_scene", scene)
        if engine.get("profile") not in ("shared-world-v1", "shared-world-v2"):
            raise ValueError("Avatars currently require a shared-world body")
        self.catalog = catalog
        self.settings = validate_appearance(settings, catalog.packages)
        self._appearance_executor = executor
        self._appearance_stream = None
        self._appearance_count = 0
        self._appearance_bytes = 0
        self._last_physical_hash = None
        self.asset_future = None

    def _write(self, event):
        super()._write(event)
        self._last_physical_hash = _sha((_encoded(dict(event, index=self._count - 1)) + "\n").encode())

    def start(self, initial_snapshot, seed=0):
        result = super().start(initial_snapshot, seed)
        self._appearance_stream = (self.root / "appearance-events.jsonl").open("x", encoding="utf-8")
        self._write_appearance(self.settings)
        self.asset_future = self._appearance_executor.submit(_save_catalog, self.root, self.catalog)
        self._appearance_metadata("recording")
        return result

    def _appearance_metadata(self, status):
        metadata = {"schema_version": 1, "profile": "weft-recorded-appearance-v1", "status": status,
                    "scene_sha256": canonical_hash(self.scene), "changes": self._appearance_count,
                    "scope": "Appearance changes after recorded physical events; no actions or time are inserted",
                    "display_scope": "Exports reconstruct the selected appearance from saved states, not screen pixels. A client GPU fallback may have displayed the original body; per-frame client render status is not recorded.",
                    "limits": {"changes": MAX_CHANGES, "bytes": MAX_CHANGE_BYTES}}
        if status == "closed":
            metadata.update(events_sha256=digest(self.root / "events.jsonl"),
                            appearance_events_sha256=digest(self.root / "appearance-events.jsonl"))
        _atomic_json(self.root / "appearance.json", metadata)

    def _write_appearance(self, settings):
        if self._closed:
            raise ValueError("Appearance recording is closed")
        record = {"index": self._appearance_count, "after_event_index": self._count - 1,
                  "after_event_sha256": self._last_physical_hash, "settings": settings}
        line = _encoded(record) + "\n"
        size = len(line.encode())
        if self._appearance_count >= MAX_CHANGES or self._appearance_bytes + size > MAX_CHANGE_BYTES:
            raise ValueError("Appearance-change limit reached; stop and save before changing appearance")
        self._appearance_stream.write(line)
        self._appearance_stream.flush()
        self._appearance_count += 1
        self._appearance_bytes += size

    def select_appearance(self, settings):
        validated = validate_appearance(settings, self.catalog.packages)
        if validated != self.settings:
            self._write_appearance(validated)
            self.settings = validated

    def close(self):
        if self._closed:
            return self.summary()
        result = super().close()
        if self._appearance_stream is not None:
            self._appearance_stream.flush()
            os.fsync(self._appearance_stream.fileno())
            self._appearance_stream.close()
            self._appearance_metadata("closed")
        return result


def load_appearance(root, recording):
    root = Path(root)
    metadata_raw = _read(root, "appearance.json", 16384)
    metadata = json.loads(metadata_raw)
    change_raw = _read(root, "appearance-events.jsonl", MAX_CHANGE_BYTES)
    if (metadata.get("profile") != "weft-recorded-appearance-v1" or metadata.get("status") != "closed"
            or metadata.get("scene_sha256") != canonical_hash(recording["scene"])
            or metadata.get("events_sha256") != digest(root / "events.jsonl")
            or metadata.get("appearance_events_sha256") != _sha(change_raw)):
        raise ValueError("Appearance sidecar is not bound to this closed physical journal")
    assets = json.loads(_read(root, "appearance-assets.json", 256 * 1024))
    if assets.get("profile") != "weft-recorded-avatar-assets-v1" or len(assets.get("assets", {})) > MAX_AVATARS:
        raise ValueError("Invalid recorded avatar asset receipt")
    events = recording["events"]
    changes, previous = [], -1
    for line in change_raw.splitlines():
        if len(changes) >= MAX_CHANGES:
            raise ValueError("Appearance journal exceeds its change bound")
        change = json.loads(line)
        if not isinstance(change, dict):
            raise ValueError("Appearance change must be an object")
        index = change.get("after_event_index")
        if (set(change) != {"index", "after_event_index", "after_event_sha256", "settings"}
                or type(change["index"]) is not int or change["index"] != len(changes)
                or type(index) is not int or not previous <= index < len(events)
                or index < 0 or (not changes and index != 0)
                or change["after_event_sha256"] != _sha((_encoded(events[index]) + "\n").encode())):
            raise ValueError("Appearance change does not match its physical event")
        change["settings"] = validate_appearance(change["settings"], assets["assets"])
        changes.append(change)
        previous = index
    if not changes or type(metadata.get("changes")) is not int or metadata["changes"] != len(changes):
        raise ValueError("Appearance journal is incomplete")
    return {"metadata": metadata, "assets": assets, "changes": changes}


def settings_by_event(changes, count):
    current, index = deepcopy(DEFAULT_APPEARANCE), 0
    for event_index in range(count):
        # Paused selections stay in the sidecar; the final one at a state is rendered.
        while index < len(changes) and changes[index]["after_event_index"] <= event_index:
            current = changes[index]["settings"]
            index += 1
        yield current


def _deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("Avatar export exceeded its time bound; physical journal and partial exports are preserved")


def _saved_catalog(root, assets, deadline):
    import re
    from weft.avatar_lab import MAX_CATALOG_BYTES, VIEWER_FILES

    if (not isinstance(assets.get("assets"), dict) or not 1 <= len(assets["assets"]) <= MAX_AVATARS
            or not isinstance(assets.get("files"), dict) or len(assets["files"]) > 200):
        raise ValueError("Recorded avatar catalog exceeds its bounds")
    packages, expected, total = {}, {}, 0
    for key, entry in assets["assets"].items():
        _deadline(deadline)
        if not re.fullmatch(r"[0-9a-f]{64}", key) or not isinstance(entry, dict):
            raise ValueError("Invalid recorded avatar identity")
        prefix = "appearance-assets/" + key + "/"
        package = load_avatar_package(root / prefix / "originals", max_seconds=min(5, max(.000001, deadline-time.monotonic())))
        if (package.source_manifest_sha256 != key or entry.get("source_manifest_sha256") != key
                or entry.get("render_sha256") != package.render_sha256
                or entry.get("textures") != package.render["textures"]
                or entry.get("asset_id") != package.manifest["asset_id"]
                or entry.get("display_name") != package.manifest["display_name"]):
            raise ValueError("Recorded avatar identity or retarget mapping has changed")
        payload = {"originals/" + name: raw for name, raw in package.originals.items()}
        payload.update(package.texture_files)
        payload["render.json"] = package.render_json
        total += sum(map(len, payload.values()))
        if total > MAX_CATALOG_BYTES:
            raise ValueError("Recorded avatar catalog exceeds 128 MiB")
        for name, raw in payload.items():
            _deadline(deadline)
            expected[prefix + name] = _sha(raw)
            if _read(root, prefix + name, len(raw)) != raw:
                raise ValueError("Recorded avatar bytes differ from their validated originals: " + name)
        packages[key] = package
    viewer_files = {}
    for name in VIEWER_FILES:
        relative = "appearance-assets/viewer/" + name
        viewer_files[name] = _read(root, relative, 512 * 1024)
        expected[relative] = _sha(viewer_files[name])
    if assets["files"] != expected:
        raise ValueError("Recorded avatar file inventory or hashes differ")
    return packages, viewer_files


def export_with_appearance(recorder, export):
    # The single export executor saves assets before this task; the physical journal stands alone.
    physical = export(recorder.root)
    try:
        recorder.asset_future.result(timeout=1)
        appearance = export_avatar_recording(recorder.root)
    except Exception as error:
        _atomic_json(recorder.root / "appearance-export-failure.json", {
            "profile": "weft-appearance-export-failure-v1", "time": local_time(),
            "physical_manifest_sha256": physical.get("manifest_sha256"), "error": str(error),
            "scope": "Physical exports remain available; selected-appearance export is not complete"})
        raise RuntimeError("Physics-body exports are saved; avatar export failed: " + str(error)) from error
    return {**physical, "appearance": appearance}


def export_avatar_recording(root, *, max_seconds=180):
    import cv2
    from weft.browser_video import open_writer
    import numpy as np
    from weft.avatar_offscreen import AvatarOffscreen
    from weft.avatar_replay import build_avatar_replay
    from weft.recording import load_recording, verify_recording

    if type(max_seconds) not in (int, float) or not 0 < max_seconds <= 180:
        raise ValueError("Avatar export requires a positive time bound of at most 180 seconds")
    deadline = time.monotonic() + max_seconds
    root = Path(root).resolve()
    if (root / "appearance-manifest.json").exists():
        return verify_avatar_recording(root, max_seconds=max_seconds)
    verify_recording(root, deadline=deadline)
    data = load_recording(root, deadline=deadline)
    appearance = load_appearance(root, data)
    packages, viewer_files = _saved_catalog(root, appearance["assets"], deadline)
    outputs = ("avatar-video.mp4", "avatar-video.partial.mp4", "avatar-replay.html")
    if any((root / name).exists() for name in outputs):
        raise FileExistsError("Partial avatar export exists; preserve it before retrying in a new directory")
    scene, events = data["scene"], data["events"]
    engine_scene = scene.get("engine_scene", scene)
    temporary = root / "avatar-video.partial.mp4"
    renderer = AvatarOffscreen(engine_scene, packages, width=1080, height=420)
    writer = None
    maximum_output = 384 * 1024 * 1024

    def check():
        _deadline(deadline)
        size = sum((root / name).stat().st_size for name in outputs if (root / name).exists())
        if size > maximum_output:
            raise ValueError("Avatar media exceeded 384 MiB; action journal is preserved")

    started = time.monotonic()
    try:
        writer = open_writer(cv2, temporary, 1 / scene["dt"], (1080, 420))
        if not writer.isOpened():
            raise RuntimeError("Avatar MP4 encoder unavailable; physical journal is preserved")
        for event, settings in zip(events, settings_by_event(appearance["changes"], len(events)), strict=True):
            check()
            frame = renderer.render(event["snapshot"], settings)
            if not isinstance(frame, np.ndarray) or frame.shape != (420, 1080, 3) or frame.dtype != np.uint8:
                raise ValueError("Avatar renderer returned an invalid frame")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
        renderer.close()
    render_seconds = time.monotonic() - started
    cap = cv2.VideoCapture(str(temporary))
    decoded = 0
    try:
        while cap.read()[0]:
            check()
            decoded += 1
    finally:
        cap.release()
    if decoded != len(events):
        raise RuntimeError("Avatar video does not retain every recorded state")
    check()
    html = build_avatar_replay(scene, events, appearance["changes"], packages, viewer_files)
    with (root / "avatar-replay.html").open("x", encoding="utf-8") as stream:
        stream.write(html)
    temporary.rename(root / "avatar-video.mp4")
    check()
    files = dict(appearance["assets"]["files"])
    for name in ("appearance.json", "appearance-events.jsonl", "appearance-assets.json", "avatar-video.mp4", "avatar-replay.html"):
        files[name] = digest(root / name)
    modules = ("avatar_recording.py", "avatar_replay.py", "avatar_offscreen.py", "avatar_assets.py", "avatar_gltf.py", "assets/avatar_offscreen_bridge.cjs")
    implementation = {name: digest(Path(__file__).parent / name) for name in modules}
    manifest = {"schema_version": 1, "profile": "weft-appearance-export-v1", "created_at": local_time(),
                "physical_manifest_sha256": digest(root / "manifest.json"), "files": files,
                "video_frames": decoded, "render_seconds": render_seconds, "implementation_sha256": implementation,
                "scope": "Saved appearance over recorded simulated states; no physics or physical validation performed"}
    _atomic_json(root / "appearance-manifest.json", manifest)
    return {"directory": str(root), "video": str(root / "avatar-video.mp4"),
            "replay": str(root / "avatar-replay.html"), "video_frames": decoded,
            "render_seconds": render_seconds, "manifest_sha256": digest(root / "appearance-manifest.json")}


def verify_avatar_recording(root, *, max_seconds=180):
    from weft.recording import load_recording, verify_recording

    if type(max_seconds) not in (int, float) or not 0 < max_seconds <= 180:
        raise ValueError("Avatar verification must be bounded to at most 180 seconds")
    deadline = time.monotonic() + max_seconds
    root = Path(root).resolve()
    physical = verify_recording(root, deadline=deadline)
    data = load_recording(root, deadline=deadline)
    appearance = load_appearance(root, data)
    _saved_catalog(root, appearance["assets"], deadline)
    manifest = json.loads(_read(root, "appearance-manifest.json", 256 * 1024))
    required = set(appearance["assets"]["files"]) | {"appearance.json", "appearance-events.jsonl",
                "appearance-assets.json", "avatar-video.mp4", "avatar-replay.html"}
    if (manifest.get("profile") != "weft-appearance-export-v1" or set(manifest.get("files", {})) != required
            or manifest.get("physical_manifest_sha256") != physical["manifest_sha256"]
            or manifest.get("video_frames") != len(data["events"])):
        raise ValueError("Avatar export is not bound to the complete physical recording")
    total = 0
    for name, expected in manifest["files"].items():
        _deadline(deadline)
        safe_relative(name)
        raw = _read(root, name, 384 * 1024 * 1024)
        total += len(raw)
        if total > 512 * 1024 * 1024 or _sha(raw) != expected:
            raise ValueError("Avatar export integrity failed: " + name)
    return {"directory": str(root), "verified_files": len(required), "video_frames": len(data["events"]),
            "manifest_sha256": digest(root / "appearance-manifest.json"),
            "scope": "Appearance and physical journal integrity; no new simulation or physical validation"}
