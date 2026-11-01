"""Bind completed HD media checks to preserved packaging evidence and exact bytes."""

import json
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2] / "tools/avatar-assets"
sys.path.insert(0, str(ROOT))
from PIL import Image
from avatar_assets.cli import verify_pack
from avatar_assets.safety import artifacts, new_directory, require, sha256, write_json

PACKS = {
    "panda": ("muse-panda-hd-build-v3", "muse-panda-hd-media-v1", 2),
    "fox": ("fox-hd-build-v2", "fox-hd-media-v1", 2),
    "robot": ("workshop-robot-hd-build-v2", "workshop-robot-hd-media-v1", 3),
    "lamp": ("lamp-hd-build-v2", "lamp-hd-media-v1", 2),
}


def check(start):
    require(time.monotonic() - start < 120, "Media evidence time bound exceeded")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(peak < 512 * 1024 * 1024, "Media evidence RSS bound exceeded")


def main():
    start = time.monotonic()
    out = new_directory(ROOT / "outputs/hd-media-verification-v1")
    results = []
    for identifier, (model_name, media_name, seconds) in PACKS.items():
        check(start)
        model, media = ROOT / "outputs" / model_name, ROOT / "outputs" / media_name
        verified = verify_pack(media)
        manifest = json.loads((media / "manifest.json").read_text())
        require(manifest["parent_3d_manifest_sha256"] == sha256(model / "manifest.json"), "Mismatched media/model")
        require(sha256(media / "sources/scene.json") == sha256(model / "sources/scene.json"), "Mismatched scene recipe")
        require(not manifest["has_transparency"] and len(manifest["clips"]) == 4, "Incorrect media declaration")
        with Image.open(media / "portrait.png") as image:
            require(image.size == (1600, 1600) and image.getchannel("A").getextrema() == (255, 255), "Portrait shape/alpha changed")
        denoise = manifest["portrait_postprocessing"]
        require(sha256(media / "portrait-raw.png") == denoise["input"]["sha256"], "Raw portrait changed")
        require(sha256(media / "portrait.png") == sha256(media / "portrait-denoise/portrait.png"), "Filtered portrait changed")
        clips = []
        for clip in manifest["clips"]:
            check(start)
            inspection = clip["inspection"]
            # Packaging already decoded every frame; matching hashes preserve that evidence.
            require(inspection["sha256"] == sha256(media / clip["path"]), "Decoded video changed")
            require((inspection["width"], inspection["height"], inspection["frames"], inspection["fps"], inspection["duration_seconds"])
                    == (720, 720, seconds * 24, 24, seconds), "Decoded dimensions/timing differ")
            require(inspection["codec"] == "h264" and not inspection["audio"] and not inspection["has_transparency"], "Wrong video format")
            require(inspection["motion_detected"] and inspection["unique_thumbnails"] > 1, "Missing actual motion")
            require(clip["loop_declaration"].startswith("not_certified"), "Unreviewed loop claim")
            first = media / "frames" / clip["id"] / "0000.png"
            require(sha256(first) == sha256(media / clip["fallback"]), "Static fallback differs from first native frame")
            clips.append({"id": clip["id"], "meaning": clip["meaning"], "inspection": inspection,
                          "fallback_equals_first_native_frame": True, "exact_endpoint_mae": clip["exact_endpoint_mae"],
                          "loop_declaration": clip["loop_declaration"], "source_clip_seconds": clip["source_clip_seconds"]})
        jobs = {}
        for name in ("job.json", "packaging-job.json"):
            job = json.loads((media / name).read_text())
            require(job["exit_code"] == 0 and not job["failure"] and job["process_exit_verified"], "Media job did not finish cleanly")
            require(job["elapsed_seconds"] <= job["limits"]["wall_seconds"], "Wall-time bound exceeded")
            require(job["sampled_peak_rss_bytes"] <= job["limits"]["rss_mb"] * 1024**2, "Sampled memory bound exceeded")
            jobs[name] = {key: job[key] for key in ("elapsed_seconds", "sampled_peak_rss_bytes", "final_output_bytes", "measurement_scope")}
        portable = media.with_name(media_name + "-portable-v1")
        portable_check = verify_pack(portable)
        distribution = json.loads((portable / "manifest.json").read_text())
        require(distribution["distribution"]["source_manifest"]["sha256"] == sha256(media / "manifest.json"), "Portable source differs")
        results.append({"id": identifier, "model": model_name, "media": media_name,
                        "model_sha256": sha256(model / "model.glb"), "manifest_sha256": sha256(media / "manifest.json"),
                        "artifact_hashes_verified": verified["artifact_hashes_verified"], "clips": clips, "jobs": jobs,
                        "portrait": manifest["portrait"], "raw_portrait_sha256": denoise["input"]["sha256"],
                        "portrait_postprocess": "Separate local OIDN CPU denoising; raw retained; no generative replacement",
                        "portable_manifest_sha256": sha256(portable / "manifest.json"),
                        "portable_artifact_hashes_verified": portable_check["artifact_hashes_verified"]})
    archive = Path("/Users/ivan/Downloads/Poo avatar assets.zip")
    archive_hash = sha256(archive)
    require(archive_hash == "cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb", "Reference archive changed")
    check(start)
    report = {"schema_version": 1, "status": "passed", "packs": results, "reference_archive_sha256": archive_hash,
              "scope": "Hash verification plus reuse of hash-bound full-frame packaging decode; separate sampled visual review is required",
              "elapsed_seconds": round(time.monotonic() - start, 3),
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "provider_calls": 0, "provider_cost_usd": 0}
    shutil.copyfile(__file__, out / "verification-source.py")
    write_json(out / "verification.json", report)
    write_json(out / "manifest.json", {"schema_version": 1, "kind": "hd_media_verification", "status": "passed", "artifacts": artifacts(out)})
    print(json.dumps({"status": report["status"], "packs": len(results), "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
