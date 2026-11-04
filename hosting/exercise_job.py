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
import threading
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
    # "body anchor", "Body tracking anchor" and "Visible-chain body anchor abstained" are
    # all real wordings from the same gate; match the shared part, not a guessed phrase.
    ("body_anchor", "body anchor"),
    ("body_anchor", "tracking anchor"),
    ("body_anchor", "construction pose evidence"),
    ("undecodable", "use a decodable video"),
    ("undecodable", "could not decode"),
    # A container header that claims more frames than decode: truncated or badly muxed.
    ("undecodable", "incomplete exercise decode"),
    ("too_large", "exceeds its byte bound"),
    ("pose_cpu", "pose_inference_requires_avx"),
    # The lab's own learning budget. Retryable on a shorter window: episode length tracks
    # the reference duration, so a narrower span is cheaper, not just differently framed.
    ("learning_budget", "learning/evaluation exceeded"),
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


def build_appearance(source, out, model, intent, *, deadline, body=None):
    """Build the same body as a shared-world scene, the only kind an avatar can replace.

    Mirrors the portal's saved-body route as library calls: admitted preflight, conditional
    person inventory, human-only world, then the shared exercise lab. No Portal, no HTTP and
    no provider calls. Returns the lab's run directory, or a recorded reason it stopped.

    Both shipped source anchors are attempted, unchanged, because they refuse different
    clips at frame zero: the legacy one wants confident head, torso and limb endpoints, the
    visible-chain one wants a confidence-qualified limb chain. Neither is relaxed; the
    second is only tried when the first refuses, and the receipt records which admitted.
    The anchor differs but the pose evidence does not, so the retry reuses the first
    attempt's sealed body instead of estimating pose over the same video twice.
    """
    from weft.exercise_perception_pipeline import (LEGACY_ANCHOR_PROFILE,
                                                   VISIBLE_ANCHOR_PROFILE,
                                                   prepare_exercise_perception_preflight)
    from weft.person_inventory import make_conditional_person_inventory
    from weft.shared_exercise_lab import prepare_shared_exercise_lab
    from weft.shared_world_human import construct_human_only_run

    # Each rung is a different unchanged gate or a narrower span of the same clip, never a
    # relaxed one: legacy anchor, visible-chain anchor, then both again over the longest span
    # the frozen constraints accept. A defect at frame zero refuses both anchors on the full
    # clip while leaving the rest of it perfectly usable, which is the third rung's reason.
    rungs = [(LEGACY_ANCHOR_PROFILE, None), (VISIBLE_ANCHOR_PROFILE, None)]
    attempts, preflight, anchor, windowed, admitted = [], None, None, None, None
    index = -1
    while True:
        index += 1
        if index >= len(rungs):
            break
        profile, clip = rungs[index]
        # A sealed body from an earlier attempt is reusable even though its anchor was
        # refused: the refusal is about frame zero, not about the pose evidence.
        # Only a body sealed from the same footage is reusable; a windowed rung poses its
        # own span. Reuse is about the anchor differing, not the clip.
        reuse = None if clip is not None else next(
            (candidate for candidate in
             [*(out / f"preflight-{earlier}/body_source" for earlier in range(index)
                if rungs[earlier][1] is None), body]
             if candidate is not None and (candidate / "manifest.json").is_file()), None)
        options = ({"body_source": reuse, "retain_body_source": True} if reuse is not None
                   else {"pose_model": model})
        try:
            result = prepare_exercise_perception_preflight(
                clip or source, out / f"preflight-{index}", anchor_profile=profile, **options,
                max_seconds=max(1, min(210, int(deadline - time.monotonic()))))
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            attempts.append({"anchor_profile": profile, "reused_body": reuse is not None,
                             "windowed": clip is not None, "refused": detail[:200]})
            # Once every full-clip anchor has refused for a localized reason, add the same
            # anchors again over the span the proposal accepts, then stop extending.
            if (windowed is None and index + 1 >= len(rungs)
                    and classify_refusal(detail) in LOCALIZED
                    and time.monotonic() < deadline - 90):
                evidence = next(
                    (candidate for candidate in
                     (out / f"preflight-{earlier}/body_source/evidence.json"
                      for earlier in range(len(rungs)))
                     if candidate.is_file()), None)
                if evidence is not None:
                    narrowed, windowed = propose_window(
                        evidence, source, out / "window/source.mp4")
                    if narrowed is not None:
                        rungs += [(LEGACY_ANCHOR_PROFILE, narrowed),
                                  (VISIBLE_ANCHOR_PROFILE, narrowed)]
            continue
        # local_ready is the library's admitted state; awaiting_consent is the portal's own
        # wrapper around it, which this route does not go through.
        if result.get("status") != "local_ready":
            attempts.append({"anchor_profile": profile, "reused_body": reuse is not None,
                             "windowed": clip is not None, "status": result.get("status")})
            continue
        preflight, anchor, admitted = result, profile, index
        attempts.append({"anchor_profile": profile, "reused_body": reuse is not None,
                         "windowed": clip is not None, "status": "local_ready"})
        break
    if preflight is None:
        return None, {"stage": "preflight", "attempts": attempts,
                      "supported_window": windowed}
    directory = out / f"preflight-{admitted}"
    inventory = make_conditional_person_inventory(preflight["source_sha256"], intent)
    built = construct_human_only_run(directory / "body_source", inventory, out / "world",
                                     deadline=deadline, max_seconds=30)
    # root_assistance matches the primary exercise route; the gates themselves are unchanged.
    lab = prepare_shared_exercise_lab(out / "world", out / "lab", root_assistance=True,
                                      max_seconds=max(1, min(60, int(deadline - time.monotonic()))))
    return out / "lab/run", {"stage": "ready", "inventory": inventory,
                             "anchor_profile": anchor, "attempts": attempts,
                             # A windowed appearance covers a narrower span than the
                             # learning rollout; the browser must not present them as one.
                             "supported_window": windowed,
                             "windowed": rungs[admitted][1] is not None,
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
            "windowed": receipt.get("windowed"),
            "supported_window": receipt.get("supported_window"),
            "world_manifest_sha256": receipt.get("world_manifest_sha256"),
            "lab_manifest_sha256": receipt.get("lab_manifest_sha256"),
            "reference_feasibility": receipt.get("reference_feasibility"),
            "artifacts": [f"appearance/{name}" for name in APPEARANCE_FILES
                          if (run / name).is_file()],
            "scope": ("Appearance only: a predefined avatar is retargeted onto this body in "
                      "the browser and never drives physics.")}


# Refusals that describe a moment in the clip rather than the whole file. Only these are
# worth retrying on a narrower window; an undecodable container or a wrong CPU will not
# improve by looking at less of it.
LOCALIZED = ("multiple_people", "no_person_visible", "no_motion", "body_anchor",
             "root_reference_gap", "learning_budget")
# Training and evaluation cost scales with episode length and the lab enforces its own
# 120-second learning budget, which a long window blows on hosted hardware: a 23.6 s
# proposal was refused after 313 s and a 12 s clamp still was after 277 s. Estimating the
# ratio from in-container timings underpredicted both times, so the bound is the only
# reference duration verified end to end on production hardware rather than a projection.
# Any sub-span of a satisfying span still satisfies the gap horizon, so clamping from the
# start keeps the proposal valid; a longer usable interval is simply used from its start.
WINDOW_MAX_SECONDS = 8.0


def seal_body(source, out, model, *, deadline):
    """Estimate pose once. The sealed evidence serves the window proposal and the
    appearance stage, so neither has to pose the same video again.

    Returns the directory even when the body stage refuses: the refusal is a judgement
    about the clip, and the evidence it judged is still written and still useful.
    """
    from weft.exercise_body_source import prepare_exercise_body_source
    refusal = None
    try:
        prepare_exercise_body_source(source, out, pose_model=model, native_timing=True,
                                     max_seconds=max(1, min(180, int(deadline - time.monotonic()))))
    except Exception as error:
        refusal = f"{type(error).__name__}: {error}"[:220]
    evidence = out / "evidence.json"
    if not evidence.is_file():
        found = sorted(out.rglob("evidence.json")) if out.is_dir() else []
        evidence = found[0] if found else None
    return out, evidence, refusal


def propose_window(evidence, source, target):
    """Longest interval satisfying the unchanged constraints, encoded for a second attempt.

    Uses the portal's own proposal and encoder so the hosted lane cannot drift to a
    different window policy or a different codec profile than the local route.
    """
    from weft.portal_supported_window import _encode_window, propose_supported_window
    report = propose_supported_window(json.loads(evidence.read_text()))
    proposal = report.get("proposal")
    if not proposal:
        return None, {"proposed": False, "reasons": (report.get("reasons") or [])[:4]}
    start, full_end = float(proposal["start_s"]), float(proposal["end_s"])
    end = min(full_end, start + WINDOW_MAX_SECONDS)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = _encode_window(source, target, start_s=start, end_s=end)
    return target, {"proposed": True,
                    "start_s": _number(start, 2),
                    "end_s": _number(end, 2),
                    "duration_s": _number(end - start, 2),
                    "proposed_end_s": _number(full_end, 2),
                    "clamped": end < full_end,
                    "encoded": encoded,
                    "scope": ("Longest span of the original satisfying the same frozen "
                              "constraints. Content outside it is unobserved, not absent.")}


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
    deadline = started + WALL_SECONDS

    def learn(clip, work):
        """The learning stage on one clip; raises with the lab's own refusal text."""
        # 600 training episodes, not the 1,500 default. The evaluation is untouched -- still
        # 100 held-out episodes against the frozen task -- and it returns identical successes
        # at 600 on every clip measured: waiters-squat 100/1/0, bodyweight-squats 100/0/0,
        # sots-press 100/10/0, with training 58-70% shorter. 400 degrades waiters-squat to
        # 99/100, so 600 is the floor, not a round number.
        root = prepare_lab(clip, work, profile="exercise", body_model="mobile",
                           pose_model=model, episodes=600,
                           source_note="Hosted submission; rights not independently verified")
        # prepare_lab only seals the configuration; the worker runs the stages.
        lab = Lab(root, avatars=())
        status = None
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
        return root

    # The two stages share only the source video, so they overlap. The lab already runs as
    # its own process and the appearance stage is in-process work dominated by pose and
    # MuJoCo inside released-GIL native code, so a thread genuinely overlaps rather than
    # queueing behind the interpreter. Sequential cost was the sum of both; now it is the
    # longer of the two. The appearance stage is started only for the first attempt: a
    # window retry changes the clip underneath it, so it is rerun serially in that case.
    appearance_slot = {}

    def appearance_worker(clip, out, sealed):
        try:
            appearance_slot["value"] = build_appearance(
                clip, out, model, exercise_intent(), deadline=deadline, body=sealed)
        except Exception as error:
            appearance_slot["value"] = (None, {
                "stage": "failed", "reason": f"{type(error).__name__}: {error}"[:200]})

    # The learning stage poses internally, so sealing a body up front would add a pass to
    # every clean clip: measured 147 s to 231 s in production for an unchanged result. The
    # seal is taken only on the refusal path, where the evidence is what the proposal needs.
    body, window = None, {"attempted": False}
    parallel = threading.Thread(target=appearance_worker, daemon=True,
                                args=(source, directory / "appearance", None),
                                name="weft-appearance")
    parallel.start()
    try:
        root = learn(source, directory / "lab")
    except Exception as error:
        detail = str(error) or type(error).__name__
        code = classify_refusal(detail)
        # A refusal about one moment of the clip is retried on the longest span that does
        # satisfy the same constraints. Nothing is relaxed; the first refusal is recorded.
        if code not in LOCALIZED or time.monotonic() > deadline - 90:
            raise
        body, evidence, _ = seal_body(source, directory / "body", model, deadline=deadline)
        if evidence is None:
            raise
        clip, window = propose_window(evidence, source, directory / "window/source.mp4")
        window.update(attempted=True, first_refusal={"code": code, "detail": detail[:300]})
        if clip is None:
            raise
        source = clip
        body, evidence, body_refusal = seal_body(source, directory / "window/body", model,
                                                 deadline=deadline)
        if body_refusal is not None:
            window["window_refusal"] = body_refusal
            raise RuntimeError(body_refusal)
        root = learn(source, directory / "window/lab")
        window["admitted"] = True
    run = root / "run"
    # The learning result already stands on its own, so a refused appearance stage is
    # recorded and returned rather than discarding a complete run.
    parallel.join(timeout=max(1, deadline - time.monotonic()))
    if window.get("attempted"):
        # The retry replaced the clip, so the overlapped attempt described different
        # footage. Redo it serially against the clip the learning result actually used.
        try:
            appearance, receipt = build_appearance(
                source, directory / "window/appearance", model, exercise_intent(),
                deadline=deadline, body=body)
        except Exception as error:
            appearance, receipt = None, {"stage": "failed",
                                         "reason": f"{type(error).__name__}: {error}"[:200]}
    else:
        appearance, receipt = appearance_slot.get("value") or (None, {
            "stage": "failed", "reason": "appearance stage did not finish in time"})
    result = {"kind": "exercise", "schema_version": 2,
              "normalization": normalization, "summary": summarize(run),
              "appearance": summarize_appearance(appearance, receipt),
              "supported_window": window,
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
        if window.get("attempted"):
            archive.writestr("supported-window.json", json.dumps(window, indent=2, default=str))
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
