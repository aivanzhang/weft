"""Build inspectable foot candidates from one verified mobile video run."""

import json
from hashlib import sha256
from pathlib import Path
import shutil
import time

from weft.benchmark import seal_suite
from weft.foot_body import audit_foot_body, foot_xml
from weft.foot_image_refinement import refine_foot_image
from weft.foot_reference import build_foot_reference
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts


def build_candidate(run, out, *, image_directions=False):
    started = time.monotonic()
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists() or out.is_relative_to(run):
        raise ValueError("Choose a new candidate directory outside the completed source run")
    if type(image_directions) is not bool:
        raise ValueError("Image-direction refinement must be an explicit boolean choice")
    verified = verify_artifacts(run)
    manifest_bytes = (run / "manifest.json").read_bytes()
    if sha256(manifest_bytes).hexdigest() != verified["manifest_sha256"]:
        raise ValueError("Source manifest changed after verification")
    manifest = json.loads(manifest_bytes)
    documents, input_bytes = {}, {}
    for name in ("human.json", "evidence.json", "pose-landmarks.json", "source.json"):
        path = run / name
        if name not in manifest["files"] or path.resolve().parent != run or path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Foot input must be a sealed, bounded file directly inside the source run")
        payload = path.read_bytes()
        if len(payload) > 32 * 1024 * 1024 or sha256(payload).hexdigest() != manifest["files"][name]:
            raise ValueError("Source input changed after verification")
        input_bytes[name] = payload
        documents[name] = json.loads(payload)
    if not all(isinstance(value, dict) for value in documents.values()):
        raise ValueError("Source input documents must be JSON objects")
    spec, evidence, raw, source = (documents[name] for name in ("human.json", "evidence.json", "pose-landmarks.json", "source.json"))
    source_hash = spec.get("source_sha256")
    artifact = source.get("artifact")
    if (not isinstance(artifact, str) or manifest["files"].get(artifact) != source_hash
            or source.get("sha256") != source_hash or evidence.get("source_sha256") != source_hash
            or raw.get("source_sha256", source_hash) != source_hash):
        raise ValueError("Body, raw poses, decoded evidence and sealed video must share source identity")
    files = _implementation_files()
    hashes = {name: digest(path) for name, path in files.items()}
    metadata = {key: evidence.get(key) for key in ("width", "height", "fps", "decoded_frames")}
    metadata.update(source_sha256=source_hash, dimensions_provenance="Original verified run's decoded evidence; no new decode")
    original = build_foot_reference(spec, raw | {"source_sha256": source_hash}, metadata)
    candidate = refine_foot_image(spec, original) if image_directions else original
    audit = audit_foot_body(spec, candidate)
    out.mkdir(parents=True)
    for name in documents:
        destination = out / "inputs" / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(input_bytes[name])
    (out / "source-manifest.json").write_bytes(manifest_bytes)
    write_json(out / "human.json", spec)
    write_json(out / "original-foot-reference.json", original)
    write_json(out / "foot-reference.json", candidate)
    write_json(out / "candidate-audit.json", audit)
    if audit["status"] != "unavailable":
        (out / "model.xml").write_text(foot_xml(spec, candidate))
    if _implementation_files() != files:
        raise RuntimeError("Implementation file set changed during candidate preparation; output remains unsealed")
    for name, path in files.items():
        destination = out / "implementation-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if digest(destination) != hashes[name]:
            raise RuntimeError("Implementation changed during candidate preparation; output remains unsealed")
    write_json(out / "implementation.json", hashes)
    result = {"created_at": local_time(), "source_run": str(run), "source_verification": verified,
              "source_manifest_sha256": verified["manifest_sha256"], "status": audit["status"],
              "image_directions": image_directions, "interactive_start_available": audit["interactive_start_available"],
              "source_frames": len(candidate["frames"]), "foot_coverage": candidate["coverage"],
              "floor_summary": audit["summary"], "accepted": False, "physical_validation": "unverified",
              "scope": "Opt-in simulation candidate with source/floor diagnostics; not an accurate human-model certificate",
              "paid_requests": 0, "runtime_seconds": time.monotonic() - started}
    write_json(out / "report.json", result)
    verification = seal_suite(out)
    return {key: value for key, value in result.items() if key != "foot_coverage"} | {"directory": str(out), "verification": verification}
