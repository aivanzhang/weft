"""Run only Weft's fixed exercise pipeline, never submitted code or user-selected commands.

Two stages ship, because each produces something the other cannot and the browser needs
both. The reinforcement-learning stage trains a joint controller and scores it against
baselines, but its scene is profile exercise-v3, which the client-side avatar retargeter
rejects. The appearance stage builds the same body as a shared-world-v3 scene, which is
the only profile an avatar can be substituted onto. Dropping either one removes a step the
product claims: without the first there is no learning result, without the second a judge
cannot replace the character in the world they just generated.

Two distinct evaluations ship and must not be conflated: rl-evaluation.json is the learned
controller against random/no-control baselines; human-evaluation.json is the authored
reference servo's tracking error against the pose-derived motion.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import zipfile

# Enough headroom for pose inference and a bounded MuJoCo rollout; the ball job's limits
# are far too tight for either. The lab worker enforces its own 360 s wall limit with a
# better message, so the job deadline sits just above it as a backstop.
CPU_SECONDS = 600
WALL_SECONDS = 420
ADDRESS_BYTES = 4096 * 1024**2
# Both stages seal packages of several hundred files, and the sealers hold a descriptor per
# directory level while they read. The ball job's 256 is far too tight for that.
OPEN_FILES = 1024
# Learning-stage artifacts, archived at the archive root.
RESULT_FILES = (
    "evidence.json", "pose-landmarks.json", "pose-quality.json", "joint-angles.json",
    "root-motion.json", "human.xml", "human.json", "body-frame.json",
    "floor-assumption.json", "task.json", "training.json", "policy.json",
    "policy-rollout.json", "human-reference.json", "human-reference-rollout.json",
    "rl-evaluation.json", "human-evaluation.json", "report.json", "manifest.json",
    "reconstruction.gif", "tracks.gif",
)
# Appearance-stage artifacts, archived under appearance/. scene.json is the shared-world-v3
# scene the browser compiles before it can retarget an avatar; the rest is its provenance.
APPEARANCE_FILES = (
    "scene.json", "human-reference.json", "exercise-task.json", "exercise-lab.json",
    "inventory.json", "reference-feasibility.json", "reconstruction.json",
    "pose-quality.json", "evidence.json", "body-source.json", "source.json",
    "manifest.json",
)


# The pipeline's own refusal wording is better than anything restated for a visitor, so a
# code only selects which advice to add; the detail text travels alongside it unchanged.
REFUSALS = (
    ("multiple_people", "multiple people or competing pose"),
    ("multiple_people", "competing pose hypotheses"),
    ("no_motion", "not a dynamic exercise video"),
    ("no_person_visible", "insufficient full-body visibility"),
    ("root_reference_gap", "reference time gaps exceed"),
    ("root_reference_gap", "mobile root reference unavailable"),
    ("body_anchor", "body anchor"),
    ("body_anchor", "visible-chain body anchor"),
    ("undecodable", "use a decodable video"),
    ("undecodable", "could not decode"),
    # A container header that claims more frames than decode: truncated or badly muxed.
    ("undecodable", "incomplete exercise decode"),
    ("too_large", "exceeds its byte bound"),
    ("pose_cpu", "pose_inference_requires_avx"),
    ("timeout", "deadline"),
    ("timeout", "wall-time limit"),
)


def classify_refusal(detail):
    """Map a refusal onto a closed code so the Worker can add advice, never invent a reason."""
    lowered = str(detail).lower()
    for code, needle in REFUSALS:
        if needle in lowered:
            return code
    return "other"


def require_pose_cpu():
    """MediaPipe's x86 inference build aborts with SIGILL when the CPU lacks AVX.

    Real Cloudflare hosts have it; an emulated amd64 container on an ARM developer machine
    does not, and the abort surfaces as an inscrutable "exited with code -4" after pose
    estimation has already started. Fail up front with the actual reason instead.
    """
    if platform.machine() not in ("x86_64", "AMD64"):
        return
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return
    supported = {flag for line in flags.splitlines() if line.startswith("flags")
                 for flag in line.split(":", 1)[-1].split()}
    if supported and "avx" not in supported:
        raise RuntimeError("pose_inference_requires_avx")


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


def build_appearance(source, out, model, intent, *, deadline):
    """Build the same body as a shared-world scene, the only kind an avatar can replace.

    Mirrors the portal's saved-body route as library calls: admitted preflight, conditional
    person inventory, human-only world, then the shared exercise lab. No Portal, no HTTP and
    no provider calls. Returns the lab's run directory, or a recorded reason it stopped.

    Both shipped source anchors are attempted, unchanged, because they refuse different
    clips at frame zero: the legacy one wants confident head, torso and limb endpoints, the
    visible-chain one wants a confidence-qualified limb chain. Neither is relaxed; the
    second is only tried when the first refuses, and the receipt records which admitted.
    """
    from weft.exercise_perception_pipeline import (LEGACY_ANCHOR_PROFILE,
                                                   VISIBLE_ANCHOR_PROFILE,
                                                   prepare_exercise_perception_preflight)
    from weft.person_inventory import make_conditional_person_inventory
    from weft.shared_exercise_lab import prepare_shared_exercise_lab
    from weft.shared_world_human import construct_human_only_run

    attempts, preflight, anchor = [], None, None
    for index, profile in enumerate((LEGACY_ANCHOR_PROFILE, VISIBLE_ANCHOR_PROFILE)):
        try:
            result = prepare_exercise_perception_preflight(
                source, out / f"preflight-{index}", pose_model=model, anchor_profile=profile,
                max_seconds=max(1, min(210, int(deadline - time.monotonic()))))
        except Exception as error:
            attempts.append({"anchor_profile": profile,
                             "refused": f"{type(error).__name__}: {error}"[:200]})
            continue
        # local_ready is the library's admitted state; awaiting_consent is the portal's own
        # wrapper around it, which this route does not go through.
        if result.get("status") != "local_ready":
            attempts.append({"anchor_profile": profile, "status": result.get("status")})
            continue
        preflight, anchor = result, profile
        attempts.append({"anchor_profile": profile, "status": "local_ready"})
        break
    if preflight is None:
        return None, {"stage": "preflight", "attempts": attempts}
    directory = out / f"preflight-{(LEGACY_ANCHOR_PROFILE, VISIBLE_ANCHOR_PROFILE).index(anchor)}"
    inventory = make_conditional_person_inventory(preflight["source_sha256"], intent)
    built = construct_human_only_run(directory / "body_source", inventory, out / "world",
                                     deadline=deadline, max_seconds=30)
    # root_assistance matches the primary exercise route; the gates themselves are unchanged.
    lab = prepare_shared_exercise_lab(out / "world", out / "lab", root_assistance=True,
                                      max_seconds=max(1, min(60, int(deadline - time.monotonic()))))
    return out / "lab/run", {"stage": "ready", "inventory": inventory,
                             "anchor_profile": anchor, "attempts": attempts,
                             "world_manifest_sha256": built["manifest_sha256"],
                             "lab_manifest_sha256": lab["manifest_sha256"],
                             "reference_feasibility": built.get("reference_feasibility")}


def summarize_appearance(run, receipt):
    """What the browser needs to know before it offers appearance replacement."""
    if run is None:
        return {"available": False, **receipt}
    scene = _load(run / "scene.json") or {}
    human = scene.get("human_spec") or {}
    bridge = (scene.get("coordinate_frame") or {}).get("scale_bridge") or {}
    return {"available": scene.get("profile") in ("shared-world-v1", "shared-world-v2",
                                                  "shared-world-v3"),
            "scene_profile": scene.get("profile"),
            "body_height_to_world_scale": _number(bridge.get("body_height_to_world_scale"), 6),
            "has_body_frame": bool(human.get("body_frame")),
            "lengths": {name: _number(value, 5) for name, value in
                        sorted((human.get("lengths") or {}).items())},
            "anchor_profile": receipt.get("anchor_profile"),
            "world_manifest_sha256": receipt.get("world_manifest_sha256"),
            "lab_manifest_sha256": receipt.get("lab_manifest_sha256"),
            "reference_feasibility": receipt.get("reference_feasibility"),
            "artifacts": [f"appearance/{name}" for name in APPEARANCE_FILES
                          if (run / name).is_file()],
            "scope": ("Appearance only: a predefined avatar is retargeted onto this body in "
                      "the browser and never drives physics.")}


def main():
    directory = Path(sys.argv[1]).resolve()
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_BYTES, ADDRESS_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (OPEN_FILES, OPEN_FILES))
    import cv2
    cv2.setNumThreads(1)
    from weft.lab import Lab, prepare_lab
    from weft.portal_inventory import exercise_intent
    from weft.portal_pipeline import prepare_demo_source

    started = time.monotonic()
    require_pose_cpu()
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
    deadline = started + WALL_SECONDS
    try:
        lab.start_worker()
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
    # The learning result already stands on its own, so a refused appearance stage is
    # recorded and returned rather than discarding a complete run.
    try:
        appearance, receipt = build_appearance(source, directory / "appearance", model,
                                               exercise_intent(), deadline=deadline)
    except Exception as error:
        appearance, receipt = None, {"stage": "failed",
                                     "reason": f"{type(error).__name__}: {error}"[:200]}
    result = {"kind": "exercise", "schema_version": 2,
              "normalization": normalization, "summary": summarize(run),
              "appearance": summarize_appearance(appearance, receipt),
              "seconds": round(time.monotonic() - started, 1),
              "scope": ("Conditional body and authored ground recovered from one video. "
                        "Simulated joint control, not measured anatomy or verified physics.")}
    (directory / "result.json").write_text(json.dumps(result, allow_nan=False))
    # The download is a fixed artifact allowlist, not arbitrary files or credentials.
    with zipfile.ZipFile(directory / "environment.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in RESULT_FILES:
            path = run / name
            if path.is_file():
                archive.write(path, name)
        if appearance is not None:
            for name in APPEARANCE_FILES:
                path = appearance / name
                if path.is_file():
                    archive.write(path, "appearance/" + name)
        archive.writestr("appearance/receipt.json", json.dumps(receipt, indent=2, default=str))
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
            "manifest.json holds a sha256 for every artifact this run produced.\n\n"
            "appearance/scene.json is the same body as a shared-world-v3 scene, which is the\n"
            "  profile the browser compiles before it can replace the character with one of\n"
            "  the predefined avatars from /api/avatars. The root scene.json above is\n"
            "  exercise-v3 and the retargeter rejects it, so use this one for appearance.\n"
            "appearance/receipt.json records the world and lab seals, or why the stage stopped.\n\n"
            "Joint control is simulated; this is not measured anatomy or validated biomechanics.\n"
            "A substituted avatar is appearance only and never drives physics.\n"
            "Source: https://github.com/kintsugi-software/weft\n")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        # A refusal the pipeline already explained must not reach a visitor as a generic
        # "could not finish". Record it where the Worker can read it, then fail as before.
        detail = str(error) or type(error).__name__
        try:
            Path(sys.argv[1]).joinpath("failure.json").write_text(json.dumps(
                {"kind": "exercise", "code": classify_refusal(detail),
                 "detail": detail[:400]}, allow_nan=False))
        except Exception:
            pass
        raise
