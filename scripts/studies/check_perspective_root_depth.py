"""One fixed construction-only root-depth proposal; no retries, dynamics or geometry admission."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from weft.io import canonical_hash, digest, local_time
from weft.perspective_root_depth import CONTRACT, DEFAULTS, MAX_OUTPUT_BYTES as MAX_PROPOSAL_BYTES, propose_perspective_root_depth
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files


MODULES = ("perspective_root_depth.py", "arm_reference_alignment.py", "body_refinement.py", "human.py",
           "mobile_human.py", "body_frame.py", "spatial_pose.py", "root_motion.py", "errors.py", "io.py",
           "world_evidence.py", "observations.py", "shared_world_packaging.py")
MAX_SECONDS = 45
MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_FILES = 128
METADATA_RESERVE_BYTES = 256 * 1024
MAX_FAILURE_RECEIPT_BYTES = 64 * 1024


def _bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def _reserve_output(files):
    required = sum(map(len, files.values())) + MAX_PROPOSAL_BYTES + METADATA_RESERVE_BYTES
    if required > MAX_OUTPUT_BYTES or len(files) + 8 > MAX_OUTPUT_FILES:
        raise ValueError("Complete input archive plus proposal/metadata reserves exceeds the output bound")


def _publish_manifest(out, report):
    raw = _bytes(report)
    if len(raw) > METADATA_RESERVE_BYTES // 4:
        raise ValueError("Study report exceeds its metadata reserve")
    write_package_files(out, {"report.json": raw})
    files, total = {}, 0
    for path in out.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
            files[path.relative_to(out).as_posix()] = digest(path)
            if total + MAX_FAILURE_RECEIPT_BYTES > MAX_OUTPUT_BYTES or len(files) + 3 > MAX_OUTPUT_FILES:
                raise ValueError("Study output exceeds its bounded envelope")
    raw = _bytes({"schema_version": 1, "created_at": local_time(), "status": report["status"], "files": files})
    if len(raw) > METADATA_RESERVE_BYTES // 4 or total + len(raw) + MAX_FAILURE_RECEIPT_BYTES > MAX_OUTPUT_BYTES:
        raise ValueError("Study manifest exceeds its metadata reserve")
    # A partial seal is never published under the canonical manifest name.
    write_package_files(out, {"manifest.pending.json": raw})
    (out / "manifest.pending.json").rename(out / "manifest.json")


def _failure_receipt(out, error, report, parent_manifest_sha256):
    preserved = []
    for name in ("report.json", "manifest.json"):
        path = out / name
        if path.exists():
            destination = name.replace(".json", ".publication-failed.json")
            path.rename(out / destination)
            preserved.append(destination)
    receipt = {"schema_version": 1, "created_at": local_time(), "status": "failed",
               "physical_validation": "unverified", "promoted": False, "custody": "failure_receipt_only",
               "parent_manifest_sha256": parent_manifest_sha256, "prior_proposal_status": report.get("status"),
               "error": f"{type(error).__name__}: {error}"[:2000], "preserved_publication_files": preserved,
               "unsealed_outputs_preserved": True,
               "scope": "Only failed receipts are sealed; partial inputs/proposal/publication bytes remain unmodified and unqualified."}
    raw = _bytes(receipt)
    import hashlib
    manifest = _bytes({"schema_version": 1, "status": "failed", "custody": "failure_receipt_only",
                       "unsealed_outputs_preserved": True,
                       "files": {name: hashlib.sha256(raw).hexdigest() for name in ("failure.json", "report.json")}})
    if 2 * len(raw) + len(manifest) > MAX_FAILURE_RECEIPT_BYTES:
        raise ValueError("Failed study receipt exceeds its separate publication allowance")
    write_package_files(out, {"failure.json": raw, "report.json": raw, "manifest.json": manifest})


def run(candidate, out):
    started = time.monotonic()
    source, out = Path(candidate).resolve(), Path(out).resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Choose a new study directory outside the immutable source")
    manifest_raw, hashes = read_package_manifest(source)
    if len(hashes) > 512:
        raise ValueError("Input exceeds the 512-file bound")
    total = len(manifest_raw)
    saved = {}
    names = {"human.json", "construction-evidence.json", "geometry-proposal.json", "scene.json", "source.json"}
    if not names <= set(hashes):
        raise ValueError("Source candidate lacks complete source/human/construction/camera inputs")
    for name, expected in hashes.items():
        if time.monotonic() - started >= MAX_SECONDS:
            raise TimeoutError("Root-depth input verification exceeded its bound")
        raw = read_package_file(source, name, expected_sha256=expected,
                                max_bytes=min(64 * 1024 * 1024, max(1, MAX_INPUT_BYTES - total)))
        total += len(raw)
        if total > MAX_INPUT_BYTES:
            raise ValueError("Input exceeds its aggregate byte bound")
        if name in names:
            saved[name] = raw
    scene, human = json.loads(saved["scene.json"]), json.loads(saved["human.json"])
    proposal = json.loads(saved["geometry-proposal.json"])
    if scene["human_spec"] != human or proposal["body_binding"]["human_json_sha256"] != hashes["human.json"]:
        raise ValueError("Human document differs from the camera/world construction input")
    binding = next(row for row in scene["source_binding"]["entity_bindings"] if row["world_entity_id"] == "human")
    entity = {"document_sha256": binding["observation_document_sha256"], "object_id": binding["observation_entity_id"]}
    camera = proposal["camera_hypothesis"]
    if any(obj["provenance"]["camera_hypothesis"] != camera for obj in scene["objects"]):
        raise ValueError("Shared world and fixed camera hypotheses differ")
    package = Path(__file__).resolve().parents[2] / "src/weft"
    implementation = {name: (package / name).read_bytes() for name in MODULES}
    implementation["study.py"] = Path(__file__).read_bytes()
    project = package.parent.parent
    for name in ("requirements-lab.lock", "pyproject.toml"):
        implementation["project/" + name] = (project / name).read_bytes()
    import hashlib
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    plan = {"schema_version": 1, "created_at": local_time(), "source": str(source),
            "parent_manifest_sha256": sha(manifest_raw), "verified_parent_files": len(hashes), "verified_parent_bytes": total,
            "input_files": {name: hashes[name] for name in sorted(saved)}, "human_entity_ref": entity,
            "camera_hypothesis": camera, "camera_sha256": canonical_hash(camera), "settings": DEFAULTS,
            "contract": CONTRACT, "maximum_study_seconds": MAX_SECONDS, "maximum_output_bytes": MAX_OUTPUT_BYTES,
            "implementation_sha256": {name: sha(raw) for name, raw in implementation.items()},
            "runtime_versions": {name: version(name) for name in ("numpy", "mujoco", "gymnasium")},
            "candidate_count": 1, "paid_calls": 0, "external_api_usd": 0, "physics_steps": 0,
            "scope": "Single unadmitted root-depth hypothesis, never installed into a scene/task or promoted by image fit"}
    initial_files = {"pre-fit.json": _bytes(plan), "inputs/parent-manifest.json": manifest_raw,
                     **{"inputs/" + name: raw for name, raw in saved.items()},
                     **{"implementation/" + name: raw for name, raw in implementation.items()}}
    _reserve_output(initial_files)
    out.mkdir(parents=True, mode=0o700)
    report = {"status": "failed", "physical_validation": "unverified", "promoted": False}
    previous_handler = signal.getsignal(signal.SIGALRM)

    def alarm(*_):
        raise TimeoutError("Root-depth study exceeded its overall wall-time bound")

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, MAX_SECONDS - (time.monotonic() - started)))
    try:
        write_package_files(out, initial_files)
        result = propose_perspective_root_depth(saved["human.json"], saved["construction-evidence.json"],
            human_entity_ref=entity, camera_hypothesis=camera)
        proposal_raw = _bytes(result)
        if len(proposal_raw) > MAX_PROPOSAL_BYTES:
            raise ValueError("Study proposal exceeds its reserved output bound")
        write_package_files(out, {"proposal.json": proposal_raw})
        report.update(status=result["status"], geometry_admission=result["report"]["geometry_admission"],
                      image_summary=result["report"]["image_summary"], failures=result["report"]["failures"],
                      changed_reference_poses=result["report"].get("changed_reference_poses"),
                      fit_seconds=result["report"]["seconds"], references=len(result["report"]["references"]),
                      source_slots=len(result["report"]["source_slots"]))
        for name, raw in saved.items():
            assert read_package_file(source, name, max_bytes=64 * 1024 * 1024, expected_sha256=sha(raw)) == raw
        assert read_package_file(source, "manifest.json", max_bytes=1024 * 1024, expected_sha256=sha(manifest_raw)) == manifest_raw
        for name, raw in implementation.items():
            path = (Path(__file__) if name == "study.py" else project / name.removeprefix("project/")
                    if name.startswith("project/") else package / name)
            if path.read_bytes() != raw:
                raise ValueError("Pinned study implementation/configuration changed: " + name)
        report["input_and_implementation_unchanged"] = True
        report["seconds"] = time.monotonic() - started
        _publish_manifest(out, report)
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        _failure_receipt(out, error, report, plan["parent_manifest_sha256"])
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run, args.out), indent=2, allow_nan=False))
