"""Explicit body custody for imported worlds; no artifact code, scores or policies enter review."""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from weft.exercise_body_source import _json
from weft.io import canonical_hash, local_time
from weft.openai_access import PROJECT_ROOT
from weft.portal_lab_runtime import VALIDATION_COMMIT, prepare_validation_runtime
from weft.shared_world_packaging import read_package_file, read_package_manifest

PROFILE = "portal-imported-construction-attachment-v1"
DIRECTORY = "review-evidence"
UNAVAILABLE = "Muse review of imported labs needs a verified body-evidence attachment"
MAX_BYTES = 128 << 20
SCOPE = "Exact source/body construction evidence plus current world assumptions; no physical accuracy, task success, policy or repair qualification."
# Fixed repository code validates data; implementation-source files are never imported.
DRIVER = r'''
import hashlib,json,os,resource,sys,threading,time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(10,10))
resource.setrlimit(resource.RLIMIT_FSIZE,(65536,65536))
def monitor():
 while True:
  time.sleep(.1)
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)>1073741824: os._exit(74)
threading.Thread(target=monitor,daemon=True).start()
from weft.exercise_body_source import validate_body_source_files
from weft.shared_world_packaging import read_package_file,read_package_manifest
p=Path(sys.argv[1]); raw,pins=read_package_manifest(p)
if len(raw)>1048576 or len(pins)>512: raise ValueError("Body bounds")
files={};total=len(raw)
for n,h in pins.items():
 b=read_package_file(p,n,max_bytes=min(67108864,134217728-total),expected_sha256=h)
 total+=len(b);files[n]=b
human,observations=validate_body_source_files(files,deadline=time.monotonic()+10)
canon=lambda o: hashlib.sha256(json.dumps(o,sort_keys=True,allow_nan=False).encode()).hexdigest()
print(json.dumps({'manifest_sha256':hashlib.sha256(raw).hexdigest(),'human_sha256':canon(human),'observations_sha256':canon(observations)}))
'''


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Construction attachment exceeded its deadline")


def _body_files(body, deadline):
    raw, pins = read_package_manifest(body)
    if len(raw) > 1 << 20 or len(pins) > 512 or _json(raw).get("profile") != "exercise-body-source-v1":
        raise ValueError("Attachment requires a bounded sealed body source")
    files, total = {}, len(raw)
    for name, pin in pins.items():
        _check(deadline)
        data = read_package_file(body, name, max_bytes=min(64 << 20, MAX_BYTES - total), expected_sha256=pin)
        total += len(data)
        files[name] = data
    _check(deadline)
    return raw, files


def _validate_body(body, files, *, project, deadline):
    from weft.exercise_body_source import validate_body_source_files

    # Historical recovery has an exact algorithm/runtime guard. Other sources use today's validator.
    if "pose-recovery.json" not in files:
        human, observations = validate_body_source_files(files, deadline=deadline)
        return human, observations, {"kind": "current", "validator_sha256": sha256(Path(__file__).with_name("exercise_body_source.py").read_bytes()).hexdigest()}
    runtime = prepare_validation_runtime(project)
    pins = _json(files["recovery/implementation.json"])["files"]
    if not isinstance(pins, dict) or not 1 <= len(pins) <= 32:
        raise ValueError("Recovery validation requires bounded implementation pins")
    for name, pin in pins.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*\.py", name):
            raise ValueError("Invalid recovery implementation name")
        trusted = read_package_file(runtime / "src/weft", name, max_bytes=2 << 20, expected_sha256=pin)
        if files.get("recovery/implementation-source/" + name) != trusted:
            raise ValueError("Recovery proof differs from the reviewed historical implementation")
    _check(deadline)
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(runtime / "src"), "PYTHONNOUSERSITE": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    with tempfile.TemporaryFile() as output:
        child = subprocess.Popen([sys.executable, "-c", DRIVER, str(body)], cwd=runtime,
                                 env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=output)
        try:
            child.wait(timeout=max(.01, min(15, deadline - time.monotonic())))
        except BaseException:
            child.kill()
            child.wait()
            raise
        output.seek(0)
        raw = output.read(65537)
    if child.returncode or len(raw) > 65536:
        raise ValueError("Reviewed historical body validation failed; source proof was not changed")
    validated = _json(raw)
    human, observations = _json(files["human.json"]), _json(files["observations.json"])
    manifest = read_package_file(body, "manifest.json", max_bytes=1 << 20)
    if validated != {"manifest_sha256": sha256(manifest).hexdigest(), "human_sha256": canonical_hash(human),
                     "observations_sha256": canonical_hash(observations)}:
        raise ValueError("Historical validation result differs from exact attached bytes")
    return human, observations, {"kind": "reviewed_historical", "commit": VALIDATION_COMMIT,
                                "driver_sha256": sha256(DRIVER.encode()).hexdigest(), "recovery_modules": pins}


def _join(root, files, human, observations, deadline):
    from weft.portal_scene_context import construction_context
    from weft.shared_world_evidence import verify_shared_world_sources
    from weft.observations import validate_observations

    world, binding = construction_context(root)
    if world is None:
        raise ValueError("Attachment requires a current source-bound exercise world")
    run = root / binding["run"]
    scene = _json(read_package_file(run, "scene.json", max_bytes=16 << 20,
                                  expected_sha256=binding["input_sha256"]["scene.json"]))
    if scene["human_spec"] != human or human["source_sha256"] != binding["source_sha256"]:
        raise ValueError("Attached body must exactly equal the current scene human; implicit transforms are unsupported")
    verify_shared_world_sources(scene, run, deadline=deadline)
    rows = [row for row in scene["source_binding"]["entity_bindings"] if row["world_entity_id"] == "human"]
    if len(rows) != 1 or rows[0]["observation_entity_id"] != "person-0":
        raise ValueError("Current world has no exact human observation binding")
    document = rows[0]["observation_document_sha256"]
    name = scene["source_binding"]["evidence_documents"][document]
    observed = _json(read_package_file(run, name, max_bytes=32 << 20, expected_sha256=document))
    validate_observations(observed)
    # The original adapter may add authored topology/role/scene provenance, never change observations.
    excluded = {"objects", "relationships", "provenance"}
    if ({k: v for k, v in observations.items() if k not in excluded}
            != {k: v for k, v in observed.items() if k not in excluded}):
        raise ValueError("Attached body observations differ from current world source measurements or partitions")
    for key in ("adapter", "input_evidence_sha256", "source_metadata"):
        if observations["provenance"].get(key) != observed["provenance"].get(key):
            raise ValueError("Attached pose provenance differs from current world observations")
    evidence_hash = sha256(files["evidence.json"]).hexdigest()
    if any(doc["provenance"]["input_artifacts"].get("evidence.json") != evidence_hash for doc in (observations, observed)):
        raise ValueError("Current world observations do not name the attached original evidence bytes")
    if any(row["partition"] not in {"construction", "unsampled"} for row in observed["observations"]):
        raise ValueError("Imported review cannot expose withheld or unassigned body evidence")
    evidence = _json(files["evidence.json"])
    if "generation" in evidence:
        raise ValueError("Exercise review must use verified reference observations, not an alternate generation payload")
    clock = scene["coordinate_frame"]["clock_binding"]
    if clock["source_fps"] != evidence["fps"] or scene["source_binding"]["total_source_frames"] != evidence["decoded_frames"]:
        raise ValueError("Attached source clock differs from current world")
    _check(deadline)
    return binding, document


def validate_attachment(root, *, project=PROJECT_ROOT, max_seconds=30):
    """Action-time full custody check; never run from project listing or rendering."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 30:
        raise ValueError("Attachment validation requires at most 30 seconds")
    root = Path(root).absolute()
    deadline = time.monotonic() + max_seconds
    job = _json(read_package_file(root, "job.json", max_bytes=64 << 10))
    if job.get("kind") != "imported_lab":
        raise ValueError("Construction attachments apply only to explicit imported labs")
    raw = read_package_file(root, DIRECTORY + "/receipt.json", max_bytes=32 << 10,
                            expected_sha256=job.get("review_attachment_sha256"))
    if not job.get("review_attachment_sha256"):
        raise ValueError(UNAVAILABLE)
    receipt = _json(raw)
    body = root / DIRECTORY / "body"
    manifest, files = _body_files(body, deadline)
    human, observations, validation = _validate_body(body, files, project=project, deadline=deadline)
    world, document = _join(root, files, human, observations, deadline)
    expected = _receipt(manifest, human, world, document, validation)
    if receipt != expected:
        raise ValueError("Construction attachment receipt differs from current source/body/world custody")
    return body, {"receipt_sha256": sha256(raw).hexdigest(), **receipt}


def _receipt(manifest, human, world, document, validation):
    return {"schema_version": 1, "profile": PROFILE, "status": "verified_construction_attachment",
            "source_sha256": human["source_sha256"], "body_manifest_sha256": sha256(manifest).hexdigest(),
            "human_sha256": canonical_hash(human), "human_observation_sha256": document,
            "world": world, "validation": validation, "scope": SCOPE, "api_calls": 0,
            "physics_steps": 0, "inference_calls": 0, "repair_qualified": False}


def attachment_summary(root, job):
    """Small pinned metadata only; full source admission happens in the review worker."""
    try:
        pin = job.get("review_attachment_sha256")
        if not isinstance(pin, str) or not re.fullmatch(r"[a-f0-9]{64}", pin):
            raise ValueError(UNAVAILABLE)
        raw = read_package_file(root, DIRECTORY + "/receipt.json", max_bytes=32 << 10, expected_sha256=pin)
        receipt = _json(raw)
        if receipt.get("profile") != PROFILE or receipt.get("source_sha256") != job["source_sha256"] or receipt.get("status") != "verified_construction_attachment":
            raise ValueError(UNAVAILABLE)
        try:
            _preparation_manifest(root, job)
            prepared = True
        except (ValueError, OSError, KeyError):
            prepared = False
        return {"available": True, "status": "attached_pending_action_time_validation", "receipt": DIRECTORY + "/receipt.json",
                "prepared": prepared,
                "reason": "Attached construction evidence is revalidated before a Muse request; imported local checks and repairs remain unavailable."}
    except (ValueError, OSError, KeyError):
        return {"available": False, "reason": UNAVAILABLE}


PREPARATION_PROFILE = "portal-imported-review-preparation-v1"
PREPARED_FILES = {"context.json": 64 << 10, "request.json": 1 << 20, "advisor-call.json": 32 << 10}
PUBLIC_ARTIFACTS = frozenset({DIRECTORY + "/receipt.json", DIRECTORY + "/prepared/context.json", DIRECTORY + "/prepared/request.json"})


def _preparation_manifest(root, job):
    pin = job.get("review_preparation_sha256")
    if not isinstance(pin, str) or not re.fullmatch(r"[a-f0-9]{64}", pin):
        raise ValueError("Prepared review snapshot has no registered payload pins")
    raw = read_package_file(root, DIRECTORY + "/prepared/manifest.json", max_bytes=8192, expected_sha256=pin)
    manifest = _json(raw)
    if (set(manifest) != {"schema_version", "profile", "attachment_receipt_sha256", "source_sha256", "files"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["profile"] != PREPARATION_PROFILE
            or manifest["attachment_receipt_sha256"] != job.get("review_attachment_sha256")
            or manifest["source_sha256"] != job.get("source_sha256")
            or not isinstance(manifest["files"], dict) or set(manifest["files"]) != set(PREPARED_FILES)
            or any(not isinstance(pin, str) or not re.fullmatch(r"[a-f0-9]{64}", pin) for pin in manifest["files"].values())):
        raise ValueError("Prepared review snapshot differs from its attachment or payload contract")
    return manifest


def attachment_artifact(root, job, name):
    """Serve only three pinned review documents, never the attached body directory."""
    if name not in PUBLIC_ARTIFACTS or job.get("kind") != "imported_lab":
        raise ValueError("File is not a public construction attachment artifact")
    pin = job.get("review_attachment_sha256")
    if not isinstance(pin, str) or not re.fullmatch(r"[a-f0-9]{64}", pin):
        raise ValueError(UNAVAILABLE)
    receipt = _json(read_package_file(root, DIRECTORY + "/receipt.json", max_bytes=32 << 10, expected_sha256=pin))
    if (receipt.get("profile") != PROFILE or receipt.get("source_sha256") != job.get("source_sha256")
            or receipt.get("status") != "verified_construction_attachment"):
        raise ValueError("Construction attachment receipt identity differs")
    if name != DIRECTORY + "/receipt.json":
        manifest = _preparation_manifest(root, job)
        filename = Path(name).name
        read_package_file(root, name, max_bytes=PREPARED_FILES[filename], expected_sha256=manifest["files"][filename])
    return Path(root) / name
