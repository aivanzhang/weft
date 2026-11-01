"""Freeze the selected local HD release without changing any source asset pack."""

from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

WEFT = Path(__file__).resolve().parents[2]
ROOT = WEFT / "tools/avatar-assets"
sys.path.insert(0, str(ROOT))
from avatar_assets.cli import verify_pack
from avatar_assets.safety import artifacts, new_directory, require, sha256, write_json

MODELS = {
    "panda": "muse-panda-hd-build-v3", "fox": "fox-hd-build-v2",
    "robot": "workshop-robot-hd-build-v2", "lamp": "lamp-hd-build-v2",
}
MEDIA = {
    "panda": "muse-panda-hd-media-v1", "fox": "fox-hd-media-v1",
    "robot": "workshop-robot-hd-media-v1", "lamp": "lamp-hd-media-v1",
}
PROOFS = [
    "hd-final-attribute-proof-v1/manifest.json", "hd-final-attribute-proof-v2/manifest.json",
    "hd-recipe-reproducibility-v1/manifest.json", "hd-media-verification-v1/manifest.json",
    "hd-visual-review-v1/manifest.json", "hd-gallery-browser-v1/browser-test.json",
    "muse-panda-hd-reload-v2/manifest.json", "fox-hd-reload-v2/manifest.json",
    "workshop-robot-hd-reload-v2/manifest.json", "lamp-hd-reload-v2/manifest.json",
    "hd-muse-fidelity-unsupported-v1/manifest.json", "hd-weft-profile-unsupported-v1/manifest.json",
]


def record(path, base=ROOT):
    path = Path(path)
    return {"path": path.relative_to(base).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}


def main():
    browser = json.loads((ROOT / "outputs/hd-gallery-browser-v1/browser-test.json").read_text())
    media_check = json.loads((ROOT / "outputs/hd-media-verification-v1/verification.json").read_text())
    require(browser["status"] == media_check["status"] == "passed", "Final verification is incomplete")
    require((ROOT / "browser/gallery.json").read_bytes() == (ROOT / "browser/gallery-hd-v1.json").read_bytes(), "Default gallery is not the checked HD catalog")
    out = new_directory(ROOT / "outputs/hd-release-v2")
    source_files = []
    for folder in ["avatar_assets", "scripts", "tests", "examples/scenes", "docs", "skills/avatar-asset-workflow"]:
        source_files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".json", ".md"} and "__pycache__" not in p.parts)
    source_files.extend(p for p in (ROOT / "browser").iterdir() if p.is_file() and p.suffix in {".js", ".json", ".css", ".html"})
    source_files.extend([ROOT / "browser/tests/gallery.mjs", ROOT / "README.md", ROOT / "TODO.md", ROOT / ".gitignore", ROOT / "requirements.lock", ROOT / "run"])
    require(len(source_files) < 512 and sum(p.stat().st_size for p in source_files) < 64 * 1024**2, "Source snapshot exceeds its bound")
    for path in sorted(set(source_files)):
        destination = out / "sources" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shutil.copyfile(__file__, out / "seal-source.py")
    receipts = [
        ROOT / "evidence/hd-final-checks-v1.json",
        ROOT / "outputs/hd-quality-verification-v1/python-tests.txt",
        ROOT / "outputs/hd-quality-verification-v1/initial-tests.json",
        ROOT / "outputs/hd-runner-incident-v1/incident.json",
        ROOT / "outputs/hd-render-profile-v1/comparison.json",
        WEFT / ".local/native-comb-probe-v1/result.json",
        WEFT / ".local/native-comb-probe-v1/job.json",
        WEFT / ".local/native-comb-probe-v1/final-input-guards.json",
        WEFT / ".local/native-comb-probe-v1/probe-failure.json",
        WEFT / ".local/oidn-portrait-probe-v1/report.json",
        WEFT / ".local/oidn-portrait-probe-v1/visual-review.json",
    ]
    for path in receipts:
        destination = out / "receipts" / path.parent.name / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    revisions = {
        "muse-panda-hd-build-v1": "Superseded: shorter radial coat, bare ear/forelock details; not Muse parity",
        "muse-panda-hd-build-v2": "Rejected final acceptance: outer monitor EPERM, missing verified exit/RSS; native files preserved",
        "fox-hd-build-v1": "Superseded by persistent directional native groom and explicit card-source material metadata",
        "workshop-robot-hd-build-v1": "Superseded by final metadata/runner rebuild; original files preserved",
        "lamp-hd-build-v1": "Superseded by final metadata/runner rebuild; original files preserved",
        "hd-proof-integration-v1": "Superseded temporary atlas-path fallback; final proof requires explicit source_material",
        "hd-proof-legacy-unsupported-v1": "Honest unsupported result for legacy missing source-material binding",
        "hd-release-v1": "Rejected incomplete release: assumed analytic probe had a manifest; corrected inventory in v2, source copies preserved",
    }
    rejected = []
    for name, reason in revisions.items():
        path = ROOT / "outputs" / name / "manifest.json"
        rejected.append({**record(path), "reason": reason, "full_attempt_preserved_locally": True})
        destination = out / "prior-attempt-manifests" / name / "manifest.json"
        destination.parent.mkdir(parents=True)
        shutil.copyfile(path, destination)
    intermediates = []
    for directory, reason in [
        (ROOT / "outputs/hd-card-probe-v1", "Analytic card fixture, not a completed character; it has native reports but no pack manifest"),
        (ROOT / "outputs/hd-fur-look-v1", "Superseded dense/radial coat look development; not the final combed character"),
        (WEFT / ".local/native-comb-probe-v1/attempt-2", "Failed analytic initialization probe; retained before corrected attempt-3"),
    ]:
        files = sorted(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        require(len(files) < 512, "Intermediate evidence entry bound exceeded")
        intermediates.append({"path": str(directory), "reason": reason, "artifacts": [record(p, WEFT) for p in files],
                              "preserved_locally": True, "full_files_copied_into_release": False})
    packs = []
    for identifier, name in MODELS.items():
        model = ROOT / "outputs" / name
        media = ROOT / "outputs" / MEDIA[identifier]
        portable = media.with_name(media.name + "-portable-v1")
        packs.append({"id": identifier, "model": record(model / "model.glb"),
                      "model_manifest": record(model / "manifest.json"), "editable_source": record(model / "source.blend"),
                      "full_media_manifest": record(media / "manifest.json"), "portable_media_manifest": record(portable / "manifest.json"),
                      "pose_and_mapping": "Authored rig-contract.json in model pack; not a newly calibrated Weft mapping"})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")
    release = {
        "schema_version": 1, "status": "verified_stylized_hd_upgrade_not_muse_parity", "recorded_at": stamp,
        "session": "01a18ebe-a17c-7cc0-9117-06ca476fb31c", "workspace": str(ROOT),
        "exact_user_request": "you're working on making these avatars better right? I want the same textures and hd quality as the muse ones for example fura and hari",
        "packs": packs, "proofs": [record(ROOT / "outputs" / name) for name in PROOFS],
        "prior_attempts": rejected, "intermediate_evidence": intermediates, "source_snapshot_files": len(set(source_files)),
        "reference_archive": {"path": "/Users/ivan/Downloads/Poo avatar assets.zip", "sha256": media_check["reference_archive_sha256"], "unchanged": True},
        "rights": "Newly authored geometry, groom and seeded textures. Supplied Muse reference has no independently verified license; generated-media metadata is provenance only. Hidden geometry, anatomy and rig are authored assumptions.",
        "representations": "Native Blender hair in portraits/videos and editable sources; separate skinned MASK cards in GLB. Native strands are not exported. OIDN denoises existing renders, never generates replacement art.",
        "limitations": ["Exact Muse sculpting/groom/shading parity is not demonstrated", "No dynamic hair or facial-muscle simulation", "No new Weft profile conversion or runtime alignment", "All loops uncertified", "Short retimed previews, not new eight-second Muse timing", "Bit-identical GLB rebuilds not guaranteed; preserve hashes", "Sampled visual review, not human approval or every-frame seam playback"],
        "tools": {"geometry_renderer": "Local Blender 4.5.13 CPU, four threads", "portrait_filter": "Bundled OIDN 2.3.3 CPU, isolated worker", "neural_text_to_3d_provider": None},
        "costs": {"provider_calls": 0, "provider_cost_usd": 0, "recurring_services": 0, "local_cpu_storage_only": True},
        "bounds": {"heavy_native_workers": 2, "threads_per_native_job": 4, "media_wall_seconds": 1200, "native_sampled_rss_mib": 2048, "output_mib": 512, "portrait_filter_wall_seconds": 90, "packaging_wall_seconds": 180, "packaging_rss_mib": 768, "cleanup_seconds": 7, "scope": "Sampled direct-child RSS; not instantaneous OS or descendant memory caps"},
        "job_handles": {"33572": "panda media finished", "79384": "fox media finished", "58161": "lamp media finished", "93216": "robot media finished", "23067": "robot/lamp reloads finished", "29760": "temporary browser check finished"},
        "live_owned_jobs_or_servers": [], "git": "Follow-up local and uncommitted/unpushed; earlier migration remains published. Unrelated Weft dirty work preserved.",
        "deferred_integration": "weft_metadata.py and its analytic alignment test are preserved drafts, not a runtime capture/calibrated mapping. Other Weft edits remain separately owned.",
    }
    write_json(out / "release.json", release)
    write_json(out / "manifest.json", {"schema_version": 1, "kind": "hd_release_source_and_evidence", "status": release["status"], "artifacts": artifacts(out)})
    checked = verify_pack(out)
    index = {"schema_version": 1, "recorded_at": stamp, "status": release["status"],
             "release_manifest": record(out / "manifest.json"), "release_details": "outputs/hd-release-v2/release.json",
             "browser_report": "outputs/hd-gallery-browser-v1/browser-test.json", "visual_review": "outputs/hd-visual-review-v1/review.json",
             "media_verification": "outputs/hd-media-verification-v1/verification.json", "source_snapshot": "outputs/hd-release-v2/sources",
             "provider_calls": 0, "provider_cost_usd": 0, "server_running_by_this_task": False,
             "start_from_weft_root": "./tools/avatar-assets/run gallery", "url": "http://127.0.0.1:8765/#panda"}
    write_json(ROOT / "evidence/hd-release-v1.json", index)
    print(json.dumps({"release": str(out), "source_snapshot_files": len(set(source_files)), "verified": checked, "manifest": index["release_manifest"]}))


if __name__ == "__main__":
    main()
