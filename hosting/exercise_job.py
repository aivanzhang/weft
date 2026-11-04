"""Run only Weft's fixed exercise pipeline, never submitted code or user-selected commands.

Produces the assets a browser needs to show the result and to drive avatar substitution
client-side: the pose evidence, the articulated body and ground, the executable task, the
trained policy with its rollout and baseline contrast, plus a downloadable environment
archive. Two distinct evaluations ship and must not be conflated: rl-evaluation.json is the
learned controller against random/no-control baselines, human-evaluation.json is the
authored reference servo's tracking error against the pose-derived motion.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import zipfile

# Enough headroom for pose inference and a bounded MuJoCo rollout; the ball job's limits
# are far too tight for either. The lab worker enforces its own 360 s wall limit with a
# better message, so the job deadline sits just above it as a backstop.
CPU_SECONDS = 420
WALL_SECONDS = 390
ADDRESS_BYTES = 3072 * 1024**2
RESULT_FILES = (
    "evidence.json", "pose-landmarks.json", "pose-quality.json", "joint-angles.json",
    "root-motion.json", "human.xml", "human.json", "body-frame.json",
    "floor-assumption.json", "task.json", "training.json", "policy.json",
    "policy-rollout.json", "human-reference.json", "human-reference-rollout.json",
    "rl-evaluation.json", "human-evaluation.json", "report.json", "manifest.json",
    "reconstruction.gif", "tracks.gif",
)


def _load(path):
    """Missing or unreadable stage artifacts stay absent; a bad one must not fail the job."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _number(value, digits=4):
    """result.json is written with allow_nan=False, so non-finite values become null."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return round(float(value), digits) if math.isfinite(value) else None


def summarize(run):
    """Flat, bounded summary for the browser; absent stages stay absent rather than faked."""
    summary = {}
    report = _load(run / "report.json") or {}
    learning = report.get("learning") or {}
    evaluation = _load(run / "rl-evaluation.json") or learning.get("evaluation") or {}
    policies = evaluation.get("policies") or {}
    if policies:
        # Per-episode trial lists are large; the browser only shows the contrast.
        summary["baselines"] = {
            name: {"successes": row.get("successes"), "episodes": row.get("episodes"),
                   "success_rate": _number(row.get("success_rate")),
                   "mean_return": _number(row.get("mean_return"), 3)}
            for name, row in sorted(policies.items())[:8] if isinstance(row, dict)}
    task = _load(run / "task.json") or {}
    if evaluation or task:
        summary["task"] = {
            "joint": evaluation.get("selected_joint") or task.get("selected_joint"),
            "target_angle_radians": _number(evaluation.get("target_angle", task.get("target_angle"))),
            "angle_tolerance_radians": _number(task.get("angle_tolerance_radians")),
            "success_definition": (evaluation.get("success_definition") or "")[:240] or None}
    training = _load(run / "training.json") or learning.get("training") or {}
    if training:
        summary["training"] = {"algorithm": training.get("algorithm"),
                               "episodes": training.get("completed_episodes"),
                               "seconds": _number(training.get("seconds"), 2)}
    tracking = report.get("reference_tracking") or _load(run / "human-evaluation.json") or {}
    if tracking:
        summary["reference_tracking"] = {
            "servo_rmse_radians": _number(tracking.get("reference_servo_rmse_radians")),
            "frames": tracking.get("frames"),
            "duration_coverage_fraction": _number(tracking.get("duration_coverage_fraction")),
            "max_floor_penetration_model_units": _number(
                tracking.get("maximum_floor_penetration_model_units"), 5)}
    quality = report.get("pose_quality") or _load(run / "pose-quality.json") or {}
    if quality:
        summary["pose"] = {"sampled_frames": quality.get("sampled_frames"),
                           "tracked_frames": quality.get("tracked_frames"),
                           "tracked_fraction": _number(quality.get("tracked_fraction"), 3),
                           "passes": quality.get("passes"),
                           "warnings": len(quality.get("warnings") or ())}
    manifest = run / "manifest.json"
    if manifest.is_file():
        payload = manifest.read_bytes()
        files = (_load(manifest) or {}).get("files") or {}
        # The manifest already holds a digest per artifact; hashing the manifest itself gives
        # the browser one value that pins the whole run.
        summary["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
        summary["manifest_files"] = len(files)
    summary["seconds"] = _number(report.get("seconds"), 1)
    summary["limits"] = [str(item)[:200] for item in (report.get("limits") or ())[:6]]
    summary["artifacts"] = [name for name in RESULT_FILES if (run / name).is_file()]
    return summary


def main():
    directory = Path(sys.argv[1]).resolve()
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_BYTES, ADDRESS_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    import cv2
    cv2.setNumThreads(1)
    import time
    from weft.lab import Lab, prepare_lab
    from weft.portal_pipeline import prepare_demo_source

    model = Path(os.environ.get("WEFT_POSE_MODEL", "/app/models/pose_landmarker_lite.task"))
    if not model.is_file():
        raise ValueError("pose_model_missing")
    # Any accepted container becomes a bounded MP4 first; the original stays beside it.
    source, normalization = prepare_demo_source(directory / "upload.mp4", directory / "source")
    root = prepare_lab(source, directory / "lab", profile="exercise", body_model="mobile",
                       pose_model=model,
                       source_note="Hosted submission; rights not independently verified")
    # prepare_lab only seals the configuration; the worker runs the stages.
    lab = Lab(root, avatars=())
    status = None
    try:
        lab.start_worker()
        deadline = time.monotonic() + WALL_SECONDS
        while True:
            status = lab.status()
            if status["status"] in ("ready", "failed", "cancelled"):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("exercise_pipeline_deadline")
            time.sleep(0.25)
    finally:
        lab.close()
    if status["status"] != "ready":
        raise RuntimeError(str((status.get("error") or {}).get("message") or "pipeline_failed"))
    run = root / "run"
    result = {"kind": "exercise", "schema_version": 1,
              "normalization": normalization, "summary": summarize(run),
              "scope": ("Conditional body and authored ground recovered from one video. "
                        "Simulated joint control, not measured anatomy or verified physics.")}
    (directory / "result.json").write_text(json.dumps(result, allow_nan=False))
    # The download is a fixed artifact allowlist, not arbitrary files or credentials.
    with zipfile.ZipFile(directory / "environment.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in RESULT_FILES:
            path = run / name
            if path.is_file():
                archive.write(path, name)
        if normalization is not None:
            archive.writestr("source-normalized.json", json.dumps(normalization, indent=2))
        archive.writestr("README.txt",
            "Weft exercise environment\n\n"
            "human.xml is the MuJoCo articulated body with its authored ground plane.\n"
            "human-reference-rollout.json replays the pose-derived reference motion and is what\n"
            "  the browser retargets onto a substituted avatar.\n"
            "task.json is the executable single-joint task; training.json is the learning curve.\n"
            "policy.json and policy-rollout.json are the trained joint controller and its rollout.\n"
            "rl-evaluation.json compares the learned policy against random and no-control baselines.\n"
            "human-evaluation.json is the authored reference servo's tracking error, a different\n"
            "  measurement from the baseline comparison above.\n"
            "manifest.json holds a sha256 for every artifact this run produced.\n"
            "Joint control is simulated; this is not measured anatomy or validated biomechanics.\n"
            "Source: https://github.com/kintsugi-software/weft\n")


if __name__ == "__main__":
    main()
