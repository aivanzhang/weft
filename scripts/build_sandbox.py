"""Build a video or saved-body sandbox and its report using the existing Weft CLI."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo


PROJECT = Path(__file__).resolve().parents[1]
MAX_LOG_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_HANDOFF_BYTES = 16 * 1024 * 1024
BODY_SOURCE_PROFILE = "exercise-body-source-v1"
BODY_METADATA = ("manifest.json", "body-source.json", "source.json")


def _path(value):
    path = Path(value).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Inputs and outputs must not pass through symlinks or '..'")
    return path


def _read(path, limit=MAX_METADATA_BYTES):
    path = _path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError(f"Expected a bounded regular file: {path}")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if (len(raw) != before.st_size or len(raw) > limit
            or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
               (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise ValueError(f"Input changed while reading: {path}")
    return raw


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate metadata key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("Non-finite metadata value")

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid, parse_float=finite)
    if not isinstance(value, dict):
        raise ValueError("Expected a metadata object")
    return value


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _environment(root):
    # The local workflow needs neither provider credentials nor the caller's Python startup hooks.
    return {"PATH": str(PROJECT / ".venv-lab/bin") + os.pathsep + "/usr/bin:/bin",
            "PYTHONPATH": str(PROJECT / "src"), "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "TZ": "America/Los_Angeles", "TMPDIR": "/private/tmp" if sys.platform == "darwin" else "/tmp",
            "MPLCONFIGDIR": str(root / "plot-cache")}


def _run(command, root, name, seconds):
    started, process = time.monotonic(), None
    sizes = {"stdout": 0, "stderr": 0}
    error, interrupted = None, False
    with (root / (name + ".stdout.log")).open("xb") as out, (root / (name + ".stderr.log")).open("xb") as err:
        with selectors.DefaultSelector() as selector:
            try:
                process = subprocess.Popen(command, cwd=PROJECT, env=_environment(root), stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, bufsize=0)
                selector.register(process.stdout, selectors.EVENT_READ, (out, "stdout"))
                selector.register(process.stderr, selectors.EVENT_READ, (err, "stderr"))
                while selector.get_map() or process.poll() is None:
                    if time.monotonic() - started >= seconds:
                        raise TimeoutError(f"{name} exceeded its {seconds:g}-second process limit")
                    for key, _ in selector.select(.1):
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        stream, channel = key.data
                        remaining = MAX_LOG_BYTES - sizes[channel]
                        stream.write(chunk[:remaining])
                        sizes[channel] += min(len(chunk), remaining)
                        if len(chunk) > remaining:
                            raise ValueError(f"{name} exceeded its log limit")
                        if channel == "stderr":
                            # Native logs stay inspectable without allowing terminal control sequences.
                            text = chunk.decode("utf-8", errors="replace")
                            print("".join(c for c in text if c in "\n\t" or ord(c) >= 32 and ord(c) != 127), end="", file=sys.stderr, flush=True)
            except (Exception, KeyboardInterrupt) as exc:
                interrupted = isinstance(exc, KeyboardInterrupt)
                error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            finally:
                if process is not None:
                    # A native call may ignore Python deadlines; only this owned process group is stopped.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
                    process.stdout.close()
                    process.stderr.close()
    return {"returncode": process.returncode if process else None, "error": error,
            "interrupted": interrupted, "seconds": time.monotonic() - started, "log_bytes": sizes}


def _terminal(build):
    manifest_raw, result_raw = _read(build / "manifest.json"), _read(build / "result.json")
    manifest, result = _json(manifest_raw), _json(result_raw)
    if (not isinstance(manifest.get("files"), dict)
            or manifest["files"].get("result.json") != hashlib.sha256(result_raw).hexdigest()
            or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or type(result.get("schema_version")) is not int or result["schema_version"] != 1
            or manifest.get("seal_scope") not in ("complete_immutable_outputs", "failure_receipt_only")
            or manifest.get("seal_scope") != result.get("seal_scope")
            or manifest.get("status") != result.get("status")):
        raise ValueError("Build terminal receipt is missing or differs from its seal")
    return result, hashlib.sha256(manifest_raw).hexdigest()


def _manifest_files(document, maximum):
    files = document.get("files")
    if (type(document.get("schema_version")) is not int or document["schema_version"] != 1
            or not isinstance(files, dict) or not 1 <= len(files) <= maximum or "manifest.json" in files):
        raise ValueError("Preserved lab requires bounded versioned file manifests")
    for name, sha in files.items():
        if (not isinstance(name, str) or len(name) > 512 or "\\" in name
                or any(ord(c) < 32 or ord(c) == 127 for c in name)
                or any(part in ("", ".", "..") for part in name.split("/"))
                or not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError("Preserved lab manifest has an unsafe path or invalid hash")
    return files


def _preserved_lab(build, terminal, manifest_sha, *, expected_source_sha=None):
    # Authenticate the saved handoff, not the runtime or large source/media closure.
    deadline, used, snapshots = time.monotonic() + 5, 0, {}

    def check():
        if time.monotonic() >= deadline:
            raise ValueError("Preserved lab identity check exceeded its five-second bound")

    def read(path):
        nonlocal used
        check()
        if path not in snapshots:
            data = _read(path, min(MAX_METADATA_BYTES, MAX_HANDOFF_BYTES - used))
            used += len(data)
            snapshots[path] = data
        return snapshots[path]

    parent_raw = read(build / "manifest.json")
    parent = _json(parent_raw)
    if (hashlib.sha256(parent_raw).hexdigest() != manifest_sha
            or parent.get("seal_scope") != "complete_immutable_outputs"
            or terminal.get("seal_scope") != parent.get("seal_scope") or terminal.get("lab_status") != "ready"):
        raise ValueError("Preserved lab is not bound by a complete immutable build seal")
    parent_files = _manifest_files(parent, 6000)

    def covered(name):
        data = read(build / name)
        if hashlib.sha256(data).hexdigest() != parent_files.get(name):
            raise ValueError("Preserved lab metadata differs from its parent build seal: " + name)
        return data

    if _json(covered("result.json")) != terminal or parent.get("status") != terminal.get("status"):
        raise ValueError("Preserved lab terminal identity differs from its build")
    lab = _path(terminal.get("lab_directory", ""))
    if lab not in (build / "lab", build / "learning_workflow/lab"):
        raise ValueError("Saved lab path is outside the expected build layout")
    trained = lab == build / "learning_workflow/lab"
    prefix = lab.relative_to(build).as_posix() + "/run/"
    source_prefix = "learning_workflow/selected/" if trained else "candidate/"
    failure = lab / "failure.json"
    if failure.exists() or failure.is_symlink():
        raise ValueError("Preserved lab has an incomplete-preparation failure marker")
    manifest_raw = covered(prefix + "manifest.json")
    manifest = _json(manifest_raw)
    files = _manifest_files(manifest, 1024)
    run_sha = hashlib.sha256(manifest_raw).hexdigest()
    if (manifest.get("status") != "ready" or manifest.get("control_profile") != "shared_exercise"
            or "failure.json" in files
            or {name[len(prefix):]: sha for name, sha in parent_files.items() if name.startswith(prefix)}
               != {**files, "manifest.json": run_sha}):
        raise ValueError("Preserved lab run is incomplete or differs from its parent seal")
    stages = terminal.get("stages")
    stage_name = "learning_interactive_lab" if trained else "interactive_lab"
    completed = [row for row in stages if isinstance(row, dict) and row.get("stage") == stage_name
                 and row.get("status") == "ready"] if isinstance(stages, list) and len(stages) <= 512 else []
    if len(completed) != 1 or completed[0].get("manifest_sha256") != run_sha:
        raise ValueError("Preserved lab has no matching completed build stage")
    total = 0
    for name in files:
        check()
        info = _path(lab / "run" / name).stat()
        total += info.st_size
        if not stat.S_ISREG(info.st_mode) or info.st_size > 64 * 1024 * 1024 or total > 192 * 1024 * 1024:
            raise ValueError("Preserved lab declared files are missing, unsafe or oversized")

    contract = _json(covered(prefix + "exercise-lab.json"))
    scene = _json(covered(prefix + "scene.json"))
    task = _json(covered(prefix + "exercise-task.json"))
    source_raw = covered(prefix + "source-run/manifest.json")
    source = _json(source_raw)
    source_files = _manifest_files(source, 512)
    source_sha = hashlib.sha256(source_raw).hexdigest()
    if (covered(source_prefix + "manifest.json") != source_raw
            or source.get("status") != ("trained" if trained else "ready")
            or type(contract.get("source_files_verified")) is not int
            or contract["source_files_verified"] != len(source_files)
            or any(files.get("source-run/" + name) != sha or parent_files.get(source_prefix + name) != sha
                   for name, sha in source_files.items())
            or source_files.get("scene.json") != files.get("scene.json")
            or parent_files.get("candidate/scene.json") != files.get("scene.json")):
        raise ValueError("Preserved lab source candidate differs from its parent build")
    if "exercise-task.json" in source_files and source_files["exercise-task.json"] != files["exercise-task.json"]:
        raise ValueError("Preserved lab task differs from its selected source task")
    if trained and parent_files.get("learning_workflow/exercise-task.json") != files["exercise-task.json"]:
        raise ValueError("Preserved lab task differs from its learning workflow")
    fixed = {"schema_version": 1, "profile": "shared-world", "control_profile": "shared_exercise",
             "scene_sha256": hashlib.sha256(json.dumps(scene, sort_keys=True, allow_nan=False).encode()).hexdigest(),
             "task_sha256": hashlib.sha256(json.dumps(task, sort_keys=True, allow_nan=False).encode()).hexdigest(),
             "reset_options": {"jitter": False, "jitter_scale": 0.0}}
    config = _json(read(lab / "lab-config.json"))
    if (any(type(doc.get("schema_version")) is not int or type(doc.get("learned_policy_available")) is not bool
            or any(doc.get(key) != value for key, value in fixed.items()) for doc in (contract, config))
            or contract.get("source_manifest_sha256") != source_sha
            or contract["learned_policy_available"] is not trained or config["learned_policy_available"] is not trained
            or config.get("source_run") != str(build / source_prefix.rstrip("/"))
            or config.get("manifest_sha256") != run_sha
            or task.get("scene_sha256") != fixed["scene_sha256"]
            or any(terminal.get(key) != fixed[key] for key in ("scene_sha256", "task_sha256"))):
        raise ValueError("Preserved lab configuration, scene or task identity is inconsistent")
    policy_pins = {"policy_sha256": "policy.json", "learning_manifest_sha256": "learning-run/manifest.json",
                   "policy_evaluation_manifest_sha256": "policy-evaluation-run/manifest.json",
                   "policy_evaluation_sha256": "policy-evaluation.json"}
    if trained and any(contract.get(key) != files.get(name) or config.get(key) != files.get(name)
                       or name not in files for key, name in policy_pins.items()):
        raise ValueError("Preserved lab policy or evaluation identities are incomplete")
    binding = scene.get("source_binding")
    if (not isinstance(binding, dict) or binding.get("mode") != "source_qualified"
            or not isinstance(binding.get("source_sha256"), str) or len(binding["source_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in binding["source_sha256"])):
        raise ValueError("Preserved lab has no bound video source identity")
    if "input-preflight.json" in parent_files:
        if _json(covered("input-preflight.json")).get("source_sha256") != binding["source_sha256"]:
            raise ValueError("Preserved lab video source differs from the build input")
    if expected_source_sha is not None and binding["source_sha256"] != expected_source_sha:
        raise ValueError("Preserved lab video source differs from the workflow plan")
    # Lifecycle files are deliberately outside the immutable seal; closed is not failed preparation.
    lifecycle = _json(read(lab / "status.json")).get("status")
    if not isinstance(lifecycle, str) or not 1 <= len(lifecycle) <= 64 or not all(c.isalnum() or c in "_-" for c in lifecycle):
        raise ValueError("Preserved lab has malformed lifecycle metadata")
    for path, data in snapshots.items():
        check()
        current = _read(path, min(MAX_METADATA_BYTES, MAX_HANDOFF_BYTES - used))
        used += len(current)
        if current != data:
            raise ValueError("Preserved lab identity metadata changed during verification")
    if failure.exists() or failure.is_symlink():
        raise ValueError("Preserved lab acquired an incomplete-preparation failure marker")
    check()
    return {"status": "identity_metadata_verified", "directory": str(lab), "manifest_sha256": run_sha,
            "source_manifest_sha256": source_sha, "source_sha256": binding["source_sha256"],
            "scene_sha256": fixed["scene_sha256"], "task_sha256": fixed["task_sha256"],
            "lifecycle_status": lifecycle, "declared_run_files": len(files), "metadata_bytes_read": used,
            "runtime_admission": "not_checked_by_launcher",
            "scope": "Selected sealed identity metadata and declared file presence only; full artifact and runtime checks run on manual resume."}


def _report_identity(root, source_sha, build_status):
    raw = _read(root / "manifest.json")
    manifest = _json(raw)
    if (manifest.get("profile") != "shared-exercise-build-report-v1"
            or manifest.get("status") != "report_generated"
            or manifest.get("source_manifest_sha256") != source_sha
            or not isinstance(manifest.get("files"), dict)):
        raise ValueError("Report is missing or refers to a different build")
    for name in ("index.html", "report.json.txt"):
        contents = _read(root / name)
        if hashlib.sha256(contents).hexdigest() != manifest["files"].get(name):
            raise ValueError("Report summary or page differs from its seal")
        if name == "report.json.txt":
            summary = _json(contents)
            if summary.get("source_manifest_sha256") != source_sha or summary.get("build_status") != build_status:
                raise ValueError("Report summary contradicts its build")
    return hashlib.sha256(raw).hexdigest()


def _body_identity(body):
    # Constructors verify all body/media bytes; the launcher only pins bounded identity metadata.
    body = _path(body)
    failure = body / "failure.json"
    if failure.exists() or failure.is_symlink():
        raise ValueError("Use a successful construction-only exercise-body output, not failed body evidence")
    contents = {name: _read(body / name) for name in BODY_METADATA}
    manifest, marker, source = (_json(contents[name]) for name in BODY_METADATA)
    files = manifest.get("files")
    if (not isinstance(files, dict) or not 1 <= len(files) <= 511 or "failure.json" in files
            or any(type(doc.get("schema_version")) is not int or doc["schema_version"] != 1
                   or doc.get("profile") != BODY_SOURCE_PROFILE or doc.get("status") != "construction_only"
                   or doc.get("simulation_ready") is not False for doc in (manifest, marker))):
        raise ValueError("--human-run requires a sealed construction-only exercise-body output, not a saved lab")
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()}
    if any(files.get(name) != hashes[name] for name in BODY_METADATA[1:]):
        raise ValueError("Saved body provenance differs from its manifest")
    artifact, source_sha = marker.get("source_artifact"), marker.get("source_sha256")
    artifacts, human_sha = marker.get("artifacts"), files.get("human.json")
    valid_sha = lambda value: isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    if (not isinstance(artifact, str) or artifact not in {"source" + suffix for suffix in (".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv")}
            or not valid_sha(source_sha) or files.get(artifact) != source_sha
            or source.get("sha256") != source_sha or source.get("artifact") != artifact
            or not valid_sha(human_sha) or not isinstance(artifacts, dict)
            or artifacts.get("human.json") != human_sha or artifacts.get("source.json") != hashes["source.json"]
            or any(not valid_sha(sha) or files.get(name) != sha for name, sha in artifacts.items())):
        raise ValueError("Saved exercise-body provenance has conflicting source/body identities")
    if failure.exists() or failure.is_symlink():
        raise ValueError("Saved exercise-body input acquired a failure marker")
    return {"profile": BODY_SOURCE_PROFILE, "status": "construction_only", "source_sha256": source_sha,
            "source_artifact": artifact, "human_sha256": human_sha, "metadata_sha256": hashes,
            "scope": "Identity metadata only; full body, source and recovery proofs are verified by construction admission."}


def _saved_body_binding(build, command, pins, manifest, body_identity):
    body = _path(command[command.index("--human-run") + 1])
    current = _body_identity(body)
    if current != body_identity or any(pins.get(str(body / name)) != sha for name, sha in current["metadata_sha256"].items()):
        raise ValueError("Saved exercise-body input changed from the workflow plan")
    if manifest["seal_scope"] == "failure_receipt_only":
        return "unavailable_before_sealed_construction"
    files = manifest["files"]

    def covered(name):
        raw = _read(build / name)
        if hashlib.sha256(raw).hexdigest() != files.get(name):
            raise ValueError("Saved body consumption proof differs from the build seal")
        return raw

    if "settings.json" not in files:
        return "unavailable_before_sealed_construction"
    settings = _json(covered("settings.json"))
    if (settings.get("human_run") != str(body) or settings.get("video") is not None
            or settings.get("base_run") is not None or type(settings.get("inference_calls")) is not int
            or settings["inference_calls"] != 0):
        raise ValueError("Build consumed different saved-body settings than the workflow plan")
    person = "--person-only" in command
    archive = "candidate/inputs/body-source-manifest.json" if person else "body_seed/inputs/human-source/source-manifest.json"
    report = "candidate/reconstruction.json" if person else "body_seed/construction-seed.json"
    prefix = "candidate/" if person else "body_seed/inputs/human-source/"
    required = (archive, report, prefix + "body-source.json", prefix + "source.json", prefix + "human.json")
    if any(name not in files for name in required):
        return "unavailable_before_sealed_construction"
    hashes = current["metadata_sha256"]
    for name, expected in ((archive, hashes["manifest.json"]),
                           (prefix + "body-source.json", hashes["body-source.json"]),
                           (prefix + "source.json", hashes["source.json"])):
        if hashlib.sha256(covered(name)).hexdigest() != expected:
            raise ValueError("Build consumed different saved-body provenance than the workflow plan")
    proof = _json(covered(report))
    verification = proof.get("source_verification")
    if (proof.get("source_sha256") != current["source_sha256"]
            or files[prefix + "human.json"] != current["human_sha256"]
            or not isinstance(verification, dict)):
        raise ValueError("Build source/body identities differ from the workflow plan")
    if person:
        valid = (proof.get("profile") == "shared-world-v3" and proof.get("status") == "ready"
                 and proof.get("source_manifest_sha256") == hashes["manifest.json"]
                 and proof.get("human_file_sha256") == current["human_sha256"]
                 and settings.get("scene_scope") == "human_only")
    else:
        valid = (proof.get("profile") == "shared-world-construction-seed-v1"
                 and proof.get("status") == "construction_only" and proof.get("simulation_ready") is False
                 and proof.get("input_profile") == BODY_SOURCE_PROFILE and proof.get("source_directory") == str(body)
                 and verification.get("manifest_sha256") == hashes["manifest.json"]
                 and proof.get("human_json_sha256") == current["human_sha256"])
    if not valid:
        raise ValueError("Saved body construction provenance contradicts the workflow plan")
    return "sealed_body_source_match_plan"


def _input_identity(build, command, pins, source_sha, body_identity=None):
    raw = _read(build / "manifest.json")
    if hashlib.sha256(raw).hexdigest() != source_sha:
        raise ValueError("Build seal changed before input-identity verification")
    manifest = _json(raw)
    if "--human-run" in command:
        return _saved_body_binding(build, command, pins, manifest, body_identity)
    expected = manifest["files"].get("input-preflight.json")
    if manifest["seal_scope"] == "failure_receipt_only" or expected is None:
        return "unavailable_before_sealed_preflight"
    contents = _read(build / "input-preflight.json")
    if hashlib.sha256(contents).hexdigest() != expected:
        raise ValueError("Build input preflight differs from its seal")
    preflight = _json(contents)
    video = command[command.index("--video") + 1]
    model = command[command.index("--pose-model") + 1]
    rows = preflight.get("input_pins")
    if (not isinstance(rows, list) or len(rows) > 320 or not all(isinstance(row, dict) for row in rows)
            or preflight.get("source_sha256") != pins[video] or not isinstance(preflight.get("pose_model"), dict)
            or preflight["pose_model"].get("sha256") != pins[model]):
        raise ValueError("Build consumed different source/model bytes than the workflow plan")
    for path, sha in pins.items():
        matches = [row for row in rows if row.get("path") == path]
        if len(matches) != 1 or matches[0].get("sha256") != sha:
            raise ValueError("Build source/model paths or identities differ from the workflow plan")
    return "sealed_source_and_model_match_plan"


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Local exercise video, or explicit --profile spatial input; never uploaded")
    source.add_argument("--planar-run", help="Sealed planar reconstruction for the explicit 3D object workflow")
    source.add_argument("--human-run", help="Sealed construction-only exercise-body output; reuses saved poses without inference (not a saved lab)")
    source.add_argument("--inspect-build", help="Saved build to report without rerunning video processing")
    parser.add_argument("--person-only", action="store_true", help="Declare one person, ground-only support and no task-relevant equipment")
    parser.add_argument("--evidence-run", help="Otherwise, sealed equipment observations")
    parser.add_argument("--config", help="Explicit camera/equipment configuration, or source-pinned 3D object hypotheses")
    parser.add_argument("--profile", choices=("exercise", "spatial"), help="Exercise by default; --planar-run implies spatial")
    parser.add_argument("--scene-artifact", help="With --planar-run: exact sealed scene path (default scene.json)")
    parser.add_argument("--tracking-profile", choices=("planar", "appearance"), help="Required for spatial raw video; never selected automatically")
    parser.add_argument("--tracking-config", help="With spatial raw video: explicit existing tracking/task JSON")
    parser.add_argument("--training-max-seconds", type=float, help="Spatial --train cooperative limit, 1–180 seconds (default 60)")
    parser.add_argument("--learning", help="Spatial only: reuse a sealed same-scene policy instead of training again")
    parser.add_argument("--source-mapping", action="append", default=[])
    parser.add_argument("--pose-model", help="Local model; default .local/models/pose_landmarker_lite.task")
    parser.add_argument("--pose-recovery", choices=("selective-crop",), help="Explicit Full/IMAGE/raw recipe; no automatic model fallback")
    parser.add_argument("--root-assistance", action="store_true", help="Enable three virtual root controls; not recovered human balance")
    parser.add_argument("--train", action="store_true", help="Explicit bounded training and separate-start evaluation; resolved settings are saved")
    parser.add_argument("--policy-family", choices=("phase-residual", "feedback-residual"),
                        help="With --train: existing phase corrections (default) or opt-in joint state-feedback corrections")
    parser.add_argument("--source-note", help="Video attribution or provenance")
    parser.add_argument("--out", help="New directory; default artifacts/video-<local timestamp>-<random suffix>")
    interact = parser.add_mutually_exclusive_group()
    interact.add_argument("--serve", action="store_true", help="After success, start the local sandbox; never opens a browser")
    interact.add_argument("--console", action="store_true", help="After success, interact and record in the terminal without a server")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, help="Build deadline, within the existing pipeline's limits")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    args.spatial = args.profile == "spatial" or args.planar_run is not None
    if args.spatial:
        if args.profile == "exercise" or args.human_run or args.inspect_build:
            parser.error("Spatial workflow requires --planar-run or --profile spatial --video")
        if any((args.person_only, args.evidence_run, args.source_mapping, args.pose_model,
                args.pose_recovery, args.root_assistance, args.policy_family)):
            parser.error("Spatial workflow cannot include exercise/equipment extraction options")
        if not args.config:
            parser.error("Spatial workflow requires --config with explicit source-pinned 3D assumptions")
        if args.planar_run:
            if not args.planar_run.strip() or any((args.tracking_config, args.tracking_profile, args.source_note)):
                parser.error("--planar-run reuses saved evidence; raw tracking/source options are not allowed")
        elif not args.video or not args.tracking_profile or not args.tracking_config or args.scene_artifact:
            parser.error("Spatial video requires --tracking-profile and --tracking-config; --scene-artifact is saved-source only")
        if args.training_max_seconds is not None and not args.train:
            parser.error("--training-max-seconds requires spatial --train")
        if args.learning and args.train:
            parser.error("--learning and --train are mutually exclusive; completed learning is never repeated implicitly")
        return args
    if any((args.scene_artifact, args.tracking_profile, args.tracking_config, args.learning, args.training_max_seconds is not None)):
        parser.error("Spatial tracking, scene and training options require the spatial workflow")
    if args.human_run is not None and not args.human_run.strip():
        parser.error("--human-run requires a sealed exercise-body output directory")
    if args.human_run and any(value is not None for value in (args.pose_model, args.pose_recovery, args.source_note)):
        parser.error("--human-run reuses sealed exercise-body provenance; raw pose/model/source options are not allowed")
    if args.inspect_build:
        if any((args.person_only, args.evidence_run, args.config, args.source_mapping, args.pose_model,
                args.pose_recovery, args.root_assistance, args.train, args.policy_family, args.source_note, args.max_seconds is not None)):
            parser.error("--inspect-build cannot change or rerun saved build settings")
    elif args.person_only:
        if args.evidence_run or args.config or args.source_mapping:
            parser.error("--person-only cannot include equipment or source mappings")
    elif not (args.evidence_run and args.config):
        parser.error("Declare --person-only, or supply both --evidence-run and --config")
    if len(args.source_mapping) > 7:
        parser.error("At most seven source mappings are supported")
    if args.policy_family and not args.train:
        parser.error("--policy-family requires --train; learning is never enabled implicitly")
    return args


def main(argv=None):
    args = _arguments(argv)
    python = PROJECT / ".venv-lab/bin/python"
    if not python.is_file():
        raise ValueError("Missing .venv-lab/bin/python; this launcher does not install dependencies")
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    root = _path(args.out or PROJECT / "artifacts" / (now.strftime("video-%Y%m%d-%H%M%S-") + os.urandom(3).hex()))
    if args.spatial:
        import importlib.util
        from types import SimpleNamespace

        spec = importlib.util.spec_from_file_location("_weft_spatial_sandbox_workflow", Path(__file__).with_name("spatial_sandbox_workflow.py"))
        workflow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow)
        host = SimpleNamespace(project=PROJECT, path=_path, read=_read, parse=_json, write=_write,
                               run=_run, environment=_environment, launch=subprocess.call)
        return workflow.run_workflow(args, root, python, now.isoformat(), host)
    if root.exists():
        saved = root / "build" if (root / "build").is_dir() else root
        command = [str(python), str(Path(__file__).absolute()), "--inspect-build", str(saved)]
        raise ValueError("Output already exists. Inspect without rerunning: " + shlex.join(command))
    prefix = [str(python), "-P", "-B", "-m", "weft"]
    command, pins, body_identity = None, {}, None
    if args.inspect_build:
        build = _path(args.inspect_build)
        if root.is_relative_to(build) or build.is_relative_to(root):
            raise ValueError("Report output must be outside the saved build")
        _terminal(build)
        maximum = 0
    else:
        maximum = args.max_seconds
        if maximum is None:
            maximum = ((270 if args.train else 90) if args.human_run else
                       (600 if args.train else 450) if args.pose_recovery else (450 if args.train else 270))
        ceiling = 180 if args.human_run and not args.train else 660 if args.pose_recovery else 480
        if not math.isfinite(maximum) or not 1 <= maximum <= ceiling:
            raise ValueError("Build deadline is outside the pipeline's allowed interval")
        if args.human_run:
            body = _path(args.human_run)
            inputs = [body]
            source_flags = ["--human-run", str(body)]
        else:
            video = _path(args.video)
            variant = "full" if args.pose_recovery else "lite"
            model = _path(args.pose_model or PROJECT / ".local/models" / f"pose_landmarker_{variant}.task")
            inputs = [video, model]
            source_flags = ["--video", str(video), "--pose-model", str(model)]
        inputs += [_path(p) for p in (args.evidence_run, args.config, *args.source_mapping) if p]
        if any(root.is_relative_to(p) or p.is_relative_to(root) for p in inputs):
            raise ValueError("Output must be outside every input")
        if args.human_run:
            body_identity = _body_identity(body)
            pins.update({str(body / name): sha for name, sha in body_identity["metadata_sha256"].items()})
        else:
            for path in (video, model):
                pins[str(path)] = hashlib.sha256(_read(path, 64 * 1024 * 1024)).hexdigest()
        build = root / "build"
        command = prefix + ["shared-exercise-build", *source_flags, "--full-reference",
                            "--max-seconds", str(maximum), "--out", str(build)]
        if args.person_only:
            command += ["--person-only"]
        else:
            command += ["--evidence-run", str(_path(args.evidence_run)), "--config", str(_path(args.config))]
            for mapping in args.source_mapping:
                command += ["--source-mapping", str(_path(mapping))]
        if args.pose_recovery:
            command += ["--pose-recovery", args.pose_recovery]
        if args.source_note:
            command += ["--source-note", args.source_note]
        if args.root_assistance:
            command += ["--root-assistance"]
        if args.train:
            command += ["--train", "--training-generations", "1", "--training-population", "4",
                        "--training-seed", "1729", "--evaluation-seed", "938401", "--evaluation-trials", "3"]
            if args.policy_family:
                command += ["--training-policy-family", args.policy_family]
    root.mkdir(parents=True, mode=0o700)
    report_command = prefix + ["shared-exercise-inspect", "--run", str(build), "--out", str(root / "report")]
    _write(root / "workflow-plan.json", {"profile": "local-video-sandbox-workflow-v1", "created_at": now.isoformat(),
        "build_command": command, "report_command": report_command, "input_sha256": pins,
        **({"body_source": body_identity} if body_identity is not None else {}),
        "build_process_seconds": maximum + 25 if command else 0, "report_process_seconds": 40,
        "log_bytes_per_stream": MAX_LOG_BYTES, "api_calls": 0, "retries": 0,
        "training_evidence": "exposed development; fixed separate seeds, not a final holdout",
        "native_memory_limit": "not hard-capped; underlying frame/output buffers are bounded"})
    receipt = {"status": "stopped", "build": str(build), "report": None, "lab": None,
               "api_calls": 0, "retries": 0, "runtime_admission": "not_checked_by_launcher"}
    print(f"Artifacts: {root}\nLocal processing only; no paid API calls.", flush=True)
    if command:
        print("1/3 Build the world (stage details below)", flush=True)
        receipt["build_process"] = _run(command, root, "build", maximum + 25)
    try:
        terminal, receipt["build_manifest_sha256"] = _terminal(build)
        receipt["build_status"] = terminal.get("status")
        if receipt.get("build_process", {}).get("interrupted"):
            raise ValueError("Interrupted; preserved outputs were not rerun or reported automatically")
        print("2/3 Save the step-by-step report", flush=True)
        receipt["report_process"] = _run(report_command, root, "report", 40)
        if receipt["report_process"]["returncode"] != 0 or receipt["report_process"]["error"]:
            raise ValueError("Report generation stopped; inspect report.stderr.log")
        receipt["report_manifest_sha256"] = _report_identity(root / "report", receipt["build_manifest_sha256"], receipt["build_status"])
        if _terminal(build)[1] != receipt["build_manifest_sha256"]:
            raise ValueError("Saved build changed while reporting")
        receipt["report"] = str(root / "report/index.html")
        print(f"Report: {receipt['report']}", flush=True)
        receipt["input_binding"] = _input_identity(build, command, pins, receipt["build_manifest_sha256"], body_identity) if command else "saved_build_not_reprocessed"
        build_process = receipt.get("build_process", {"returncode": 0, "error": None})
        if (build_process["returncode"] == 0 and build_process["error"] is None
                and terminal.get("status") == "ready" and terminal.get("lab_status") == "ready"):
            expected_binding = "sealed_body_source_match_plan" if args.human_run else "sealed_source_and_model_match_plan"
            if command and receipt["input_binding"] != expected_binding:
                raise ValueError("Ready build has no sealed input identities matching the workflow plan")
            lab = _path(terminal.get("lab_directory", ""))
            if lab not in (build / "lab", build / "learning_workflow/lab"):
                raise ValueError("Saved lab path is outside the expected build layout")
            receipt.update(status="ready", lab=str(lab))
            print("3/3 Sandbox ready; task success and physical accuracy are separate checks.", flush=True)
        else:
            if terminal.get("lab_status") == "ready":
                expected_binding = "sealed_body_source_match_plan" if args.human_run else "sealed_source_and_model_match_plan"
                if command and receipt["input_binding"] != expected_binding:
                    raise ValueError("Preserved lab has no sealed input identities matching the workflow plan")
                expected_source_sha = (body_identity["source_sha256"] if body_identity is not None
                    else pins[command[command.index("--video") + 1]]) if command else None
                receipt["preserved_lab"] = _preserved_lab(build, terminal, receipt["build_manifest_sha256"],
                                                         expected_source_sha=expected_source_sha)
                lab = receipt["preserved_lab"]["directory"]
                print("3/3 Build stopped, but a prepared lab was preserved. Its identity metadata is verified; nothing was started.", flush=True)
                print("Manual resume only; full artifact and compatible-runtime checks still apply. Task success and physical accuracy are not established.", flush=True)
                print(shlex.join(prefix + ["lab", "--resume", lab, "--port", str(args.port)]), flush=True)
                print(shlex.join(prefix + ["lab-console", "--resume", lab]), flush=True)
            else:
                print("3/3 No playable sandbox: the build stopped. The report retains the reason and evidence.", flush=True)
    except (ValueError, OSError, TypeError, RecursionError) as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        print(f"Stopped: {exc}. Preserved artifacts: {root}", file=sys.stderr)
    _write(root / "workflow-result.json", receipt)
    if receipt["status"] != "ready":
        return 1
    lab_command = prefix + ["lab", "--resume", receipt["lab"], "--port", str(args.port)]
    console_command = prefix + ["lab-console", "--resume", receipt["lab"]]
    print("Sandbox → choose a control mode → Start. Record & start → Stop & save exports replay/video/actions.")
    print("Start or reopen (requires this build's compatible runtime):\n" + shlex.join(lab_command), flush=True)
    print("Or interact and record in the terminal:\n" + shlex.join(console_command), flush=True)
    if args.console:
        return subprocess.call(console_command, cwd=PROJECT, env=_environment(root))
    if args.serve:
        # Serving is explicitly user-controlled and separate from the bounded construction process.
        return subprocess.call(lab_command, cwd=PROJECT, env=_environment(root))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, TypeError, RecursionError) as error:
        print(f"weft: {error}", file=sys.stderr)
        sys.exit(1)
