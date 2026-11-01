"""Bounded saved-input collection; selection and authored geometry remain explicit."""

from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

from weft import exercise_surface_inputs as collector
from weft.io import local_time
from weft.shared_world_packaging import read_package_file, write_package_files


PROFILE = "exercise-surface-workflow-v1"
SELECTION_PROFILE = "exercise-surface-selection-v1"
INVENTORY_PROFILE = "exercise-surface-inventory-v1"
MAX_FILES, MAX_BYTES, MAX_INPUT_BYTES = 96, 64 << 20, 32 << 20
MAX_OBSERVATION_BYTES = 16 << 20
MAX_RUNTIME_BYTES, MAX_REPORT_BYTES, MAX_REPORT_FILES = 8 << 20, 16 << 20, 32
MAX_METADATA_BYTES, MAX_SELECTION_BYTES, FINAL_RESERVE = 1 << 20, 128 << 10, 2 << 20
RUNTIME_ROOT = Path(__file__).parent
RUNTIME_FILES = (
    "__init__.py", "exercise_surface_pipeline.py", "exercise_surface_inputs.py", "exercise_surface_report.py",
    "source_surface_fit_inputs.py", "source_surface_envelopes.py", "source_surface_interval_construction.py",
    "source_surface_anchors.py", "camera_pose.py", "camera_ground.py", "world_alignment.py", "world_evidence.py",
    "observations.py", "io.py", "shared_world_packaging.py", "shared_world_evidence.py",
    "whole_body_reference_refinement.py", "constrained_qp.py", "body_frame.py",
    "root_motion.py", "spatial_pose.py", "errors.py", "assets/brand.css",
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SCOPE = "Saved conditional construction inputs only; no fitting, body/floor installation, contact, load, grasp or physical admission"


def _require(condition, message):
    if not condition:
        raise ValueError("Exercise surface workflow: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Exercise surface workflow exceeded its cooperative deadline")


def _path(value):
    path = Path(value).absolute()
    _require(len(path.parts) <= 64 and ".." not in path.parts
             and not any(part.is_symlink() for part in (path, *path.parents)), "paths cannot traverse symlinks")
    return path


def _name(name):
    _require(type(name) is str and 0 < len(name) <= 256 and "\\" not in name and ":" not in name
             and all(ord(char) >= 32 for char in name), "unsafe saved member name")
    parts = name.split("/")
    _require(len(parts) <= 6 and all(part not in ("", ".", "..") for part in parts), "unsafe saved member path")
    return name


def _encoded(value, limit=MAX_METADATA_BYTES):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    _require(len(raw) <= limit, "metadata exceeds its byte reservation")
    return raw


def _json(raw, limit, deadline):
    return collector._json(raw, limit, deadline)


class _Reads:
    def __init__(self, deadline):
        self.deadline, self.rows = deadline, {}

    def read(self, path, limit, expected=None):
        _check(self.deadline)
        path = _path(path)
        if path not in self.rows:
            before = path.stat(follow_symlinks=False)
            raw = read_package_file(path.parent, path.name, max_bytes=limit, expected_sha256=expected)
            after = path.stat(follow_symlinks=False)
            identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
            _require(identity(before) == identity(after), "input changed while reading")
            self.rows[path] = (raw, identity(after))
        raw, _ = self.rows[path]
        _require(len(raw) <= limit and (expected is None or sha256(raw).hexdigest() == expected), "input pin or bound differs")
        _check(self.deadline)
        return raw

    def recheck(self):
        for path, (raw, expected) in self.rows.items():
            _check(self.deadline)
            path = _path(path)
            current = path.stat(follow_symlinks=False)
            _require((current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) == expected,
                     "input identity changed before publication")
            read_package_file(path.parent, path.name, max_bytes=max(1, len(raw)), expected_sha256=sha256(raw).hexdigest())
            _check(self.deadline)


def _inventory(root, deadline):
    pending, files, visits, total = [_path(root)], {}, 0, 0
    while pending:
        _check(deadline)
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                _check(deadline)
                visits += 1
                _require(visits <= MAX_FILES * 3, "saved directory inventory exceeds its bound")
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                    continue
                _require(stat.S_ISREG(info.st_mode) and info.st_size <= 32 << 20, "saved member is not a bounded regular file")
                name = _name(Path(entry.path).relative_to(root).as_posix())
                files[name] = info.st_size
                total += info.st_size
                _require(len(files) <= MAX_FILES and total <= MAX_BYTES, "saved package exceeds its file/byte bound")
    return files


def _resume(root, reads, expected):
    raw = reads.read(root / "manifest.json", MAX_METADATA_BYTES, expected)
    manifest = _json(raw, MAX_METADATA_BYTES, reads.deadline)
    _require(manifest.get("schema_version") == 1 and type(manifest["schema_version"]) is int
             and manifest.get("profile") == PROFILE and manifest.get("status") == "completed"
             and manifest.get("seal_scope") == "complete_immutable_outputs", "resume requires a completed collection workflow")
    pins = manifest.get("files")
    _require(type(pins) is dict and len(pins) < MAX_FILES
             and all(type(pin) is str and _SHA.fullmatch(pin) for pin in pins.values()), "invalid resume file inventory")
    for name in pins:
        _name(name)
    sizes = _inventory(root, reads.deadline)
    _require(set(sizes) == {*pins, "manifest.json"} and "failure.json" not in sizes, "resume has missing or unsealed members")
    files = {name: reads.read(root / name, sizes[name], pin) for name, pin in pins.items()}
    _require({"result.json", "workflow-state.json", "inventory.json", "selection.json", "plan.json"} <= files.keys(), "resume metadata is incomplete")
    result = _json(files["result.json"], MAX_METADATA_BYTES, reads.deadline)
    state = _json(files["workflow-state.json"], MAX_METADATA_BYTES, reads.deadline)
    _require(result.get("status") == state.get("status") == "completed" and result.get("sealed") is True
             and result.get("collection_status") == state.get("collection_status"), "resume terminal state differs")
    inventory = _json(files["inventory.json"], MAX_METADATA_BYTES, reads.deadline)
    plan = _json(files["plan.json"], MAX_METADATA_BYTES, reads.deadline)
    bindings = inventory.get("input_bindings", {})
    _require(type(bindings) is dict and set(bindings) == {"source_sha256", "observation_document_sha256s", "camera_raw_sha256", "body_raw_sha256"}
             and all(value.get("profile") == PROFILE and value.get("input_bindings") == bindings for value in (result, state, plan))
             and state.get("selection_sha256") == plan.get("selection_sha256") == sha256(files["selection.json"]).hexdigest(),
             "resume source or selection custody differs")
    collection_status = state["collection_status"]
    _require(collection_status in {"needs_selection", "incomplete_request", "qualified_but_unavailable", "conditional_inputs_available"},
             "unsupported resume collection status")
    if collection_status == "needs_selection":
        _require(state.get("request_sha256") is None and state.get("collection_manifest_sha256") is None,
                 "unselected resume cannot claim qualified request custody")
    else:
        _require({"collection-manifest.json", "collection-report.json", "collection-request.json", "collection-frames.jsonl"} <= files.keys(),
                 "resume collector package is incomplete")
        collection = _json(files["collection-manifest.json"], 64 << 10, reads.deadline)
        report = _json(files["collection-report.json"], 256 << 10, reads.deadline)
        members = collection.get("files")
        _require(collection.get("profile") == collector.PROFILE and collection.get("status") == report.get("status") == collection_status
                 and type(members) is dict and all(name in files and sha256(files[name]).hexdigest() == pin for name, pin in members.items())
                 and {"collection-request.json", "collection-report.json", "collection-frames.jsonl"} <= members.keys()
                 and state.get("collection_manifest_sha256") == sha256(files["collection-manifest.json"]).hexdigest()
                 and state.get("request_sha256") == plan.get("request_sha256") == report.get("request_sha256")
                     == sha256(files["collection-request.json"]).hexdigest()
                 and report.get("input_bindings") == bindings, "resume collector seal or input custody differs")
    docs = bindings.get("observation_document_sha256s")
    _require(type(docs) is list and 1 <= len(docs) <= 8 and docs == sorted(set(docs))
             and all(type(pin) is str and _SHA.fullmatch(pin) for pin in docs), "resume original observations are unavailable")
    original = {f"surface-observation-{i:02d}.json": pin for i, pin in enumerate(docs)}
    for key, name in (("camera_raw_sha256", "surface-camera.json"), ("body_raw_sha256", "original-human.json")):
        pin = bindings.get(key)
        _require(pin is None or type(pin) is str and _SHA.fullmatch(pin), "resume optional input identity differs")
        if pin is not None:
            original[name] = pin
    _require(inventory.get("profile") == INVENTORY_PROFILE and inventory.get("input_files") == plan.get("input_files") == original
             and all(name in files and sha256(files[name]).hexdigest() == pin for name, pin in original.items()),
             "resume original input inventory differs")
    _require(sum(len(files[name]) for name in original) <= MAX_INPUT_BYTES, "resume selected input snapshot exceeds 32 MiB")
    return {name: files[name] for name in original}, files["selection.json"], raw, inventory, state


def _discover(files, source_sha256, deadline):
    documents = {sha256(raw).hexdigest(): raw for name, raw in files.items() if name.startswith("surface-observation-")}
    _require(1 <= len(documents) <= 8 and sum(map(len, documents.values())) <= 16 << 20,
             "one to eight unique observation documents within 16 MiB are required")
    sources = {_json(raw, 8 << 20, deadline).get("source", {}).get("sha256") for raw in documents.values()}
    _require(len(sources) == 1 and all(type(pin) is str and _SHA.fullmatch(pin) for pin in sources),
             "all observation declarations must identify one exact original source")
    declared = next(iter(sources))
    _require(source_sha256 is None or source_sha256 == declared, "explicit source identity differs from observations")
    source, qualified, observations, _, _, _ = collector._sources(documents, declared, deadline)
    _require(max(source["metadata_frames"], source["decoded_frames"]) <= collector.MAX_FRAMES, "complete source exceeds 1800 frames")
    camera, body = files.get("surface-camera.json"), files.get("original-human.json")
    _require(sum(map(len, documents.values())) + (len(camera) if camera else 0) <= 16 << 20, "surface inputs exceed 16 MiB")
    if camera is not None:
        document = _json(camera, 4 << 20, deadline)
        collector.validate_camera(document, {"source_sha256": declared, "width": source["width_px"], "height": source["height_px"],
            "frame_count": source["metadata_frames"], "fps": source.get("fps", document.get("fps"))})
    if body is not None:
        _, reason = collector._body(_json(body, 8 << 20, deadline), declared, deadline)
        _require(reason is None, "original body belongs to a different source")
    bindings = {"source_sha256": declared, "observation_document_sha256s": sorted(documents),
                "camera_raw_sha256": sha256(camera).hexdigest() if camera else None,
                "body_raw_sha256": sha256(body).hexdigest() if body else None}
    entities = []
    for digest, document in sorted(qualified.items()):
        _check(deadline)
        grouped = {obj["id"]: [] for obj in document["objects"]}
        for row in document["observations"]:
            grouped[row["object_id"]].append(row)
        for obj in document["objects"]:
            rows = grouped[obj["id"]]
            entities.append({"entity_ref": obj["entity_ref"], "properties": obj["properties"], "source_slots": len(rows),
                "construction_annotation_slots": sum(row["admission"] == "included" and row["partition"] == "construction"
                                                       and row["visibility"] in {"detected", "partial"} for row in rows),
                "partition_counts": dict(Counter(row["partition"] for row in rows)),
                "visibility_counts": dict(Counter(row["visibility"] if row["admission"] == "included" else "withheld" for row in rows))})
    inventory = {"schema_version": 1, "profile": INVENTORY_PROFILE, "input_bindings": bindings,
        "input_files": {name: sha256(raw).hexdigest() for name, raw in sorted(files.items())},
        "source": source, "source_clocks": {digest: document["time"] for digest, document in sorted(qualified.items())},
        "entities": entities, "scope": "Construction-filtered source declarations and entity references; no source media decoded or verified"}
    _encoded(inventory)
    return inventory, documents


def _source_inputs(paths, optional, reads):
    documents, total = {}, 0
    for path in paths:
        limit = min(8 << 20, MAX_OBSERVATION_BYTES - total)
        _require(limit > 0, "observation input byte reservation exhausted")
        raw = reads.read(path, limit)
        total += len(raw)
        pin = sha256(raw).hexdigest()
        _require(pin not in documents, "duplicate observation bytes are not separate inputs")
        documents[pin] = raw
    files = {f"surface-observation-{i:02d}.json": raw for i, (_, raw) in enumerate(sorted(documents.items()))}
    for name, path in optional.items():
        limit = min(4 << 20, MAX_OBSERVATION_BYTES - total) if name == "surface-camera.json" else 8 << 20
        _require(limit > 0, "surface input byte reservation exhausted")
        files[name] = reads.read(path, limit)
    return files


def _selection(raw, bindings, deadline):
    value = _json(raw, MAX_SELECTION_BYTES, deadline)
    _require(set(value) == {"schema_version", "profile", "input_bindings", "channels", "human_entity_ref"}
             and type(value["schema_version"]) is int and value["schema_version"] == 1 and value["profile"] == SELECTION_PROFILE
             and value["input_bindings"] == bindings, "selection must retain the exact original input pins")
    _require(type(value["channels"]) is list and len(value["channels"]) <= collector.MAX_CHANNELS,
             "selection requires at most twelve existing annotation channels")
    _require(value["channels"] or value["human_entity_ref"] is None, "empty selection cannot silently choose a human")
    return value


def _runtime(reads):
    files, total = {}, 0
    for name in RUNTIME_FILES:
        raw = reads.read(RUNTIME_ROOT / name, min(16 << 10 if name.endswith(".css") else 2 << 20, MAX_RUNTIME_BYTES - total))
        total += len(raw)
        files["runtime/weft/" + name] = raw
    return files, {"python": sys.version.split()[0], "numpy": version("numpy")}


def _report(files, max_seconds):
    from weft.exercise_surface_report import build_exercise_surface_report
    return build_exercise_surface_report(files, max_seconds=max_seconds)


def _store(out, files, saved, *, terminal=False):
    new = {name: raw for name, raw in files.items() if name not in saved}
    _require(all(type(raw) is bytes and (name not in saved or saved[name] == raw) for name, raw in files.items()),
             "an immutable output cannot be replaced")
    _require(len(saved) + len(new) <= MAX_FILES - (0 if terminal else 4)
             and sum(map(len, saved.values())) + sum(map(len, new.values())) <= MAX_BYTES - (0 if terminal else FINAL_RESERVE),
             "output exceeds its file/byte reservation")
    for name, raw in new.items():
        _name(name)
        write_package_files(_path(out), {name: raw})
        saved[name] = raw


def _verify_saved(out, saved, deadline):
    sizes = _inventory(out, deadline)
    _require(set(sizes) == set(saved), "output has missing or untracked members")
    for name, raw in saved.items():
        _check(deadline)
        read_package_file(_path(out), name, max_bytes=max(1, len(raw)), expected_sha256=sha256(raw).hexdigest())
    _check(deadline)


def _preserve_terminal(out, attempted, saved):
    for name, raw in attempted.items():
        path = out / name
        if not path.exists():
            continue
        read_package_file(_path(out), name, max_bytes=max(1, len(raw)), expected_sha256=sha256(raw).hexdigest())
        before = path.stat(follow_symlinks=False)
        target = out / ("attempted-" + name)
        os.link(path, target, follow_symlinks=False)
        after = path.stat(follow_symlinks=False)
        linked = target.stat(follow_symlinks=False)
        _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                 == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                 == (linked.st_dev, linked.st_ino, linked.st_size, linked.st_mtime_ns), "terminal attempt changed before preservation")
        path.unlink()
        saved.pop(name, None)
        saved["attempted-" + name] = raw


def _terminal(out, result, saved, deadline, attempted):
    _verify_saved(out, saved, deadline)
    result.update(sealed=True, seal_scope="complete_immutable_outputs")
    attempted["result.json"] = _encoded(result)
    _store(out, {"result.json": attempted["result.json"]}, saved, terminal=True)
    _check(deadline)
    manifest = _encoded({"schema_version": 1, "profile": PROFILE, "status": result["status"],
        "seal_scope": result["seal_scope"], "files": {name: sha256(raw).hexdigest() for name, raw in sorted(saved.items())}})
    attempted["manifest.json"] = manifest
    _check(deadline)
    try:
        _store(out, {"manifest.json": manifest}, saved, terminal=True)
    except (Exception, KeyboardInterrupt) as error:
        # A valid published seal is terminal, even if completion notification fails.
        try:
            read_package_file(_path(out), "manifest.json", max_bytes=len(manifest), expected_sha256=sha256(manifest).hexdigest())
        except Exception:
            raise error
        saved["manifest.json"] = manifest
    return {**result, "manifest_sha256": sha256(manifest).hexdigest()}


def prepare_exercise_surface_collection(observations, out, *, source_sha256=None, camera=None, body=None,
        selection=None, request=None, resume=None, expected_manifest_sha256=None, max_seconds=30, progress=None):
    """Discover, explicitly select, or qualify saved inputs in a new immutable package."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 1 <= max_seconds <= 30,
             "max_seconds must be within [1,30]")
    _require(type(observations) in (list, tuple) and (selection is None or request is None), "observation list and mutually exclusive edits required")
    _require(progress is None or callable(progress), "progress must be a callback")
    _require(source_sha256 is None or type(source_sha256) is str and _SHA.fullmatch(source_sha256), "source identity must be an exact SHA256")
    if resume is None:
        _require(1 <= len(observations) <= 8 and request is None and expected_manifest_sha256 is None,
                 "fresh collection requires observations; request and manifest pins require resume")
    else:
        _require(not observations and source_sha256 is None and camera is None and body is None
                 and (selection is not None) != (request is not None), "resume requires exactly one edit and no raw-source options")
    _require(expected_manifest_sha256 is None or type(expected_manifest_sha256) is str and _SHA.fullmatch(expected_manifest_sha256),
             "expected manifest must be an exact SHA256")
    out = _path(out)
    paths = [_path(path) for path in observations]
    optional = {name: _path(path) for name, path in (("surface-camera.json", camera), ("original-human.json", body)) if path is not None}
    edit = _path(selection if selection is not None else request) if selection is not None or request is not None else None
    parent = _path(resume) if resume is not None else None
    all_paths = [*paths, *optional.values(), *([edit] if edit else []), *([parent] if parent else [])]
    _require(not out.exists() and not any(path == out or path.is_relative_to(out) or out.is_relative_to(path) for path in all_paths),
             "output must be new and cannot overlap supplied inputs")
    started = time.monotonic()
    deadline = started + max_seconds
    reads, saved, attempted = _Reads(deadline), {}, {}
    result = {"schema_version": 1, "profile": PROFILE, "status": "failed", "sealed": False,
        "created_at": local_time(), "directory": str(out), "collection_status": None, "report_path": None,
        "next_action": None, "error": None, "scope": _SCOPE, "external_api_usd": 0,
        "fit_or_simulator_admission": False, "body_or_floor_installed": False, "physical_validation": "unverified",
        "contact": "unknown", "load": "unknown", "grasp": "unknown"}
    stage = "inputs"
    out.mkdir(parents=True, mode=0o700)
    def update(name, detail):
        _check(deadline)
        if progress is not None:
            progress(name, detail)
        _check(deadline)
    try:
        parent_raw, prior_selection, prior_inventory, prior_state = None, None, None, None
        if parent is not None:
            inputs, prior_selection, parent_raw, prior_inventory, prior_state = _resume(parent, reads, expected_manifest_sha256)
            _store(out, {"parent-manifest.json": parent_raw}, saved)
        else:
            inputs = _source_inputs(paths, optional, reads)
        _require(sum(map(len, inputs.values())) <= MAX_INPUT_BYTES, "selected original input snapshot exceeds 32 MiB")
        _store(out, inputs, saved)
        edited_raw = reads.read(edit, MAX_SELECTION_BYTES if selection is not None else collector.MAX_REQUEST_BYTES) if edit else None
        if edited_raw is not None:
            _store(out, {"selection.json" if selection is not None else "collection-request.json": edited_raw}, saved)
        runtime, versions = _runtime(reads)
        _store(out, runtime, saved)
        inventory, documents = _discover(inputs, source_sha256, deadline)
        if prior_inventory is not None:
            _require(inventory == prior_inventory, "resumed original source/entity inventory changed")
        bindings = inventory["input_bindings"]
        _store(out, {"inventory.json": _encoded(inventory)}, saved)
        update("inputs", {"source_sha256": bindings["source_sha256"], "observation_documents": len(documents), "entities": len(inventory["entities"])})
        stage = "selection"
        selection_raw = edited_raw if selection is not None else prior_selection
        if selection_raw is None:
            selection_raw = _encoded({"schema_version": 1, "profile": SELECTION_PROFILE, "input_bindings": bindings,
                                     "channels": [], "human_entity_ref": None}, MAX_SELECTION_BYTES)
        chosen = _selection(selection_raw, bindings, deadline)
        _store(out, {"selection.json": selection_raw}, saved)
        operation = "qualification" if request is not None else "selection" if chosen["channels"] else "discovery"
        _require(request is None or chosen["channels"], "request resume requires a previously usable explicit selection")
        update("selection", {"channels": len(chosen["channels"]), "human_explicitly_selected": chosen["human_entity_ref"] is not None})
        meta = None
        if chosen["channels"]:
            stage = "qualification"
            prepared = collector.prepare_exercise_surface_inputs(documents, source_sha256=bindings["source_sha256"],
                channels=chosen["channels"], camera_document=inputs.get("surface-camera.json"),
                body_document=inputs.get("original-human.json"), human_entity_ref=chosen["human_entity_ref"])
            _check(deadline)
            qualified = prepared.qualify(edited_raw if request is not None else prepared.request_document())
            _check(deadline)
            meta = qualified.describe()
            _store(out, qualified.files(), saved)
        collection_status = meta["status"] if meta is not None else "needs_selection"
        next_action = ("Copy selection.json outside this immutable package, select explicit channel/entity rows, then resume into a new output."
            if meta is None else "Copy collection-request.json outside this immutable package, supply justified missing inputs, then resume into a new output."
            if collection_status == "incomplete_request" else "Inspect every unavailable source/channel reason; retained unknowns are not zero targets. Edit a copy of collection-request.json if justified."
            if collection_status == "qualified_but_unavailable" else "Inspect the complete rows and surface-config.json. Fitting remains a separate explicit command with its own qualified body source; nothing is installed.")
        update("qualification", {"collection_status": collection_status, "config_available": meta is not None and meta["config_member"] is not None})
        state = {"schema_version": 1, "profile": PROFILE, "status": "completed", "collection_status": collection_status,
            "operation": operation, "input_bindings": bindings, "selection_sha256": sha256(selection_raw).hexdigest(),
            "request_sha256": meta["request_sha256"] if meta else None,
            "collection_manifest_sha256": sha256(saved["collection-manifest.json"]).hexdigest() if meta else None,
            "parent_manifest_sha256": sha256(parent_raw).hexdigest() if parent_raw else None,
            "fit_or_simulator_admission": False, "body_or_floor_installed": False, "physical_validation": "unverified",
            "contact": "unknown", "load": "unknown", "grasp": "unknown", "scope": _SCOPE, "next_action": next_action}
        plan = {"schema_version": 1, "profile": PROFILE, "operation": operation, "input_bindings": bindings,
            "input_files": inventory["input_files"], "selection_sha256": state["selection_sha256"], "request_sha256": state["request_sha256"],
            "parent_manifest_sha256": state["parent_manifest_sha256"], "runtime_files": {name: sha256(raw).hexdigest() for name, raw in runtime.items()},
            "runtime_versions": versions, "bounds": {"cooperative_seconds": max_seconds, "failure_cleanup_seconds": 5,
                "output_files": MAX_FILES, "output_bytes": MAX_BYTES, "selected_source_bytes": MAX_INPUT_BYTES,
                "runtime_bytes": MAX_RUNTIME_BYTES, "report_files": MAX_REPORT_FILES, "report_bytes": MAX_REPORT_BYTES,
                "memory_scope": "Bounded input/cache/output sizes; no hard RSS or process-isolation claim",
                "completion_scope": "Cooperative checks precede final seal publication; a valid completed seal remains terminal"},
            "source_scope": "Exact saved declarations, not fresh media verification; prior runtime snapshots are never executed"}
        _store(out, {"workflow-state.json": _encoded(state), "plan.json": _encoded(plan)}, saved)
        stage = "report"
        update("report", {"collection_status": collection_status})
        report_inputs = {name: raw for name, raw in saved.items()
                         if not name.startswith("runtime/") and name not in {"plan.json", "parent-manifest.json"}}
        _require(len(report_inputs) <= 32 and sum(map(len, report_inputs.values())) <= 32 << 20,
                 "selected report inputs exceed their file/byte bound")
        report_files = _report(report_inputs, max_seconds=min(5., deadline - time.monotonic()))
        _check(deadline)
        _require(type(report_files) is dict and "index.html" in report_files and len(report_files) <= MAX_REPORT_FILES
                 and all(type(name) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\.(?:html|css|json)", name)
                         and type(raw) is bytes and len(raw) <= 2 << 20 for name, raw in report_files.items())
                 and sum(map(len, report_files.values())) <= MAX_REPORT_BYTES, "offline report exceeds its safe output contract")
        _store(out, {"inspection/" + name: raw for name, raw in report_files.items()}, saved)
        stage = "source_recheck"
        reads.recheck()
        result.update(status="completed", collection_status=collection_status, operation=operation, input_bindings=bindings,
            report_path=str(out / "inspection/index.html"), next_action=next_action, parent_manifest_sha256=state["parent_manifest_sha256"],
            config_path=str(out / "surface-config.json") if meta and meta["config_member"] else None,
            source_inputs_rechecked=True, seconds=time.monotonic() - started)
        stage = "seal"
        return _terminal(out, result, saved, deadline, attempted)
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="failed", sealed=False, report_path=None, config_path=None,
            error={"stage": stage, "type": type(error).__name__, "message": str(error)[:2000]}, seconds=time.monotonic() - started)
        cleanup = time.monotonic() + 5
        try:
            _preserve_terminal(out, attempted, saved)
            # Failed packages retain partial writes rather than dropping their evidence.
            for name, size in _inventory(out, cleanup).items():
                if name not in saved:
                    saved[name] = read_package_file(out, name, max_bytes=max(1, size))
            return _terminal(out, result, saved, cleanup, {})
        except Exception as seal_error:
            result.update(sealed=False, seal_scope="unsealed_failure", seal_error={"type": type(seal_error).__name__, "message": str(seal_error)[:1000]})
            try:
                write_package_files(out, {"failure.json": _encoded(result)})
            except Exception:
                pass
            return result
