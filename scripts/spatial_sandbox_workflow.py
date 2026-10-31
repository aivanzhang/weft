"""Bounded orchestration of existing object commands; no engine imports or inferred priors."""

import hashlib
import html
import json
import math
import os
from pathlib import PurePosixPath
import shlex
import stat
import sys
import time


PROFILE = "guided-spatial-sandbox-v1"
MAX_METADATA_BYTES = 16 << 20
MAX_MEMBER_BYTES = 2 << 20
TRAINING = {"seed": 1729, "eval_seed": 20261031, "generations": 3, "population": 8, "eval_episodes": 12}
SCOPE = "Authored 3D hypothesis. Ready means playable, not task success or physical accuracy. Development evaluation is not a final holdout."


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return _sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _relative(name):
    _require(isinstance(name, str) and 0 < len(name) <= 512 and "\\" not in name
             and not PurePosixPath(name).is_absolute()
             and all(part not in ("", ".", "..") for part in name.split("/"))
             and all(ord(c) >= 32 and ord(c) != 127 for c in name), "Unsafe artifact path")
    return name


def _text(value):
    return "".join(c for c in str(value)[:2000] if c in "\n\t" or ord(c) >= 32 and ord(c) != 127)


class _Metadata:
    def __init__(self, host, deadline):
        self.host, self.deadline, self.used, self.cache = host, deadline, 0, {}

    def check(self):
        _require(time.monotonic() < self.deadline, "Spatial workflow deadline reached; no stage retried")

    def read(self, path, limit=MAX_MEMBER_BYTES):
        self.check()
        path = self.host.path(path)
        if path not in self.cache:
            raw = self.host.read(path, min(limit, MAX_METADATA_BYTES - self.used))
            self.used += len(raw)
            self.cache[path] = raw
        raw = self.cache[path]
        _require(len(raw) <= limit, "Metadata exceeds its member bound")
        return raw

    def document(self, path, limit=MAX_MEMBER_BYTES):
        return self.host.parse(self.read(path, limit))

    def bundle(self, root):
        _require(not (root / "failure.json").exists() and not (root / "failure.json").is_symlink(),
                 "Stage has a failure marker; partial output cannot continue")
        raw = self.read(root / "manifest.json")
        manifest = self.host.parse(raw)
        _require(manifest.get("status") not in ("failed", "cancelled", "partial", "incomplete"),
                 "Stage manifest declares an unsuccessful output")
        versions = [manifest[k] for k in ("schema", "schema_version") if k in manifest]
        files = manifest.get("files")
        _require(versions and all(type(v) is int and v == 1 for v in versions)
                 and isinstance(files, dict) and 1 <= len(files) <= 6000 and "manifest.json" not in files,
                 "Stage requires a bounded versioned manifest")
        for name, digest in files.items():
            _relative(name)
            _require(_hash(digest), "Invalid sealed artifact identity")

        def covered(name):
            data = self.read(root / _relative(name))
            _require(_sha(data) == files.get(name), "Missing or changed sealed metadata: " + name)
            return self.host.parse(data)

        return _sha(raw), files, covered

    def recheck(self):
        # Recheck only consumed metadata; stage commands own full archive/media admission.
        for path, expected in self.cache.items():
            self.check()
            raw = self.host.read(path, min(len(expected), MAX_METADATA_BYTES - self.used))
            self.used += len(raw)
            _require(raw == expected, "Workflow metadata changed during execution: " + str(path))


def _video_hash(path, host, check):
    path = host.path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 100_000_000,
                 "Provide a regular local video no larger than 100 MB")
        digest, count = hashlib.sha256(), 0
        while True:
            check()
            block = stream.read(min(1 << 20, before.st_size + 1 - count))
            if not block:
                break
            count += len(block)
            _require(count <= before.st_size, "Video grew during hashing")
            digest.update(block)
        after = os.fstat(stream.fileno())
    _require(count == before.st_size and (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
             == (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Video changed during hashing")
    return digest.hexdigest(), count


def _learning_summary(document, scene_sha, policy_sha, *, expected_episodes=None):
    _require(document.get("scene_sha256") == scene_sha and document.get("policy_sha256") == policy_sha
             and document.get("physical_validation") == "unverified", "Learning identities differ from the candidate")
    output = {}
    regimes = document.get("regimes")
    if regimes is None and expected_episodes is None:
        return {"status": "completed", "evidence_partition": "exposed_development", "regimes": {},
                "reason": "This saved policy has no separated nominal/changed evaluation counts; metrics are unavailable"}
    for label, rows, expected in (
        ("nominal_exact", regimes.get("nominal_exact"), 1),
        ("changed_initial_conditions", regimes.get("changed_initial_conditions", {}).get("policies"), expected_episodes),
    ):
        _require(isinstance(rows, dict) and set(rows) == {"learned", "random", "coast", "brake"}, "Missing saved controller comparisons")
        output[label] = {}
        for name, row in rows.items():
            count, successes = row.get("episodes"), row.get("successes")
            _require(type(count) is int and 1 <= count <= 64 and (expected is None or count == expected)
                     and type(successes) is int and 0 <= successes <= count,
                     "Invalid saved evaluation counts")
            output[label][name] = {"successes": successes, "episodes": count}
    return {"status": "completed", "evidence_partition": "exposed_development", "regimes": output,
            "scope": "Saved simulator comparison; no new replay or physical trial in this summary"}


def _index(root, receipt, assumptions, host):
    esc = lambda value: html.escape(_text(value), quote=True)
    rows = []
    for stage in receipt["stages"]:
        rows.append("<tr><td>" + esc(stage["label"]) + "</td><td>" + esc(stage["status"]) + "</td><td>" +
                    esc(stage.get("error", "")) + "</td></tr>")
    artifacts = ["workflow-plan.json", "workflow-result.json", "inputs/spatial-config.json", "inputs/tracking-config.json",
                 "planar-lab/run/scene.json", "planar-lab/run/report.json", "candidate/report.json",
                 "candidate/construction-report.json", "candidate/scene.json", "candidate/inputs/input-receipt.json",
                 "candidate/source.json", "candidate/evidence.json", "candidate/tracks.gif",
                 "learning/training.json", "learning/rl-evaluation.json", "learning/policy.json", "lab/run/report.json",
                 "lab/run/scene.json", "lab/run/rl-evaluation.json", "lab/run/policy.json", "lab/run/training.json"]
    artifacts += [s["id"] + suffix for s in receipt["stages"] for suffix in (".stdout.log", ".stderr.log")]
    links = []
    for name in artifacts:
        target = root / name
        if target.is_file() and not target.is_symlink():
            host.path(target)
            links.append('<li><a href="' + esc(name) + '">' + esc(name) + "</a></li>")
    learning = receipt["learning"]
    summary = "<p>Learning not requested; manual controls only.</p>"
    if learning["status"] == "requested_not_completed":
        summary = "<p>The requested policy was not completed or verified. No manual lab was substituted.</p>"
    elif learning["status"] == "completed":
        summary = "<h2>Saved simulator results</h2><p>Exposed development seeds; not physical validation.</p><ul>"
        for regime, controllers in learning["regimes"].items():
            for name, counts in controllers.items():
                summary += "<li>" + esc(regime + " / " + name) + f": {counts['successes']}/{counts['episodes']} successful trials</li>"
        summary += "</ul>"
        if learning.get("reason"):
            summary += "<p>" + esc(learning["reason"]) + "</p>"
    commands = "".join("<pre>" + esc(command) + "</pre>" for command in receipt.get("resume_commands", {}).values())
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>Weft object workflow</title><style>body{font:16px system-ui;max-width:960px;margin:2rem auto;padding:0 1rem;background:#F5EADC;color:#214F50}td,th{padding:.5rem;text-align:left;border-bottom:1px solid #aaa}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#965644}</style>'
            '<h1>Object video → 3D sandbox</h1><p><strong>' + esc(receipt["status"]) + '</strong> — ' + esc(SCOPE) + '</p>'
            '<p>' + esc(receipt.get("error", {}).get("message", "")) + '</p><table><tr><th>Stage</th><th>Result</th><th>Reason</th></tr>' +
            "".join(rows) + '</table><h2>Your explicit assumptions</h2><ul>' +
            "".join("<li>" + esc(item) + "</li>" for item in assumptions) + '</ul>' + summary +
            '<h2>Interact and record</h2><p>In the console: controls → record start → action right → step 3 → record stop. '
            'Use mode learned only when a completed policy is attached. Recording creates new sessions without changing run/.</p>' + commands +
            '<h2>Stage artifacts and logs</h2><ul>' + "".join(links) + '</ul></html>')
    _require(len(page.encode()) <= 128 << 10, "Workflow index exceeds its bound")
    with (root / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(page)


def run_workflow(args, root, python, created_at, host):
    maximum = args.max_seconds if args.max_seconds is not None else (300 if args.video else 180)
    training_seconds = args.training_max_seconds if args.training_max_seconds is not None else 60
    _require(math.isfinite(maximum) and 1 <= maximum <= 360, "Spatial workflow deadline must be 1–360 seconds")
    _require(math.isfinite(training_seconds) and 1 <= training_seconds <= 180, "Spatial training deadline must be 1–180 seconds")
    _require(not root.exists(), "Output already exists; inspect its index.html / workflow-result.json or choose a new directory")
    inputs = [host.path(value) for value in (args.planar_run, args.video, args.config, args.tracking_config, args.learning) if value]
    _require(all(not root.is_relative_to(p) and not p.is_relative_to(root) for p in inputs), "Output must be outside all inputs")
    scene_name = _relative(args.scene_artifact or "scene.json")
    prefix = [str(python), "-P", "-B", "-m", "weft"]
    started = time.monotonic()
    reader = _Metadata(host, started + maximum - .25)
    root.mkdir(parents=True, mode=0o700)
    (root / "inputs").mkdir()
    receipt = {"schema_version": 1, "profile": PROFILE, "status": "stopped", "created_at": created_at,
               "stages": [], "input_sha256": {}, "lab": None, "source": None,
               "learning": {"status": "requested_not_completed" if args.train or args.learning else "not_requested"},
               "physical_validation": "unverified", "promoted": False, "api_calls": 0, "retries": 0,
               "scope": SCOPE, "metadata_validation": "Selected sealed metadata; existing child commands own full source and runtime admission"}
    stage_plan = ([{"id": "tracking", "label": "Extract the explicit planar scene", "cap_seconds": 120}] if args.video else [])
    stage_plan += [{"id": "construction", "label": "Construct the authored 3D hypothesis", "cap_seconds": 25}]
    if args.train:
        stage_plan += [{"id": "training", "label": "Train and compare controllers", "cap_seconds": training_seconds + 5}]
    elif args.learning:
        stage_plan += [{"id": "learning", "label": "Reuse the sealed same-scene policy", "cap_seconds": 0}]
    stage_plan += [{"id": "lab", "label": "Admit and package the interactive lab", "cap_seconds": 45}]
    host.write(root / "workflow-plan.json", {"schema_version": 1, "profile": PROFILE, "created_at": created_at,
        "inputs": [str(p) for p in inputs], "stages": stage_plan, "maximum_seconds": maximum,
        "cleanup_reserve_seconds": 5, "training": {**TRAINING, "max_seconds": training_seconds} if args.train else None,
        "learning_source": str(host.path(args.learning)) if args.learning else None,
        "source_scene_artifact": scene_name, "log_bytes_per_stream": 1 << 20, "metadata_read_bytes": MAX_METADATA_BYTES,
        "video_hash_bytes": 100_000_000 if args.video else 0, "native_memory_limit": "not hard-capped",
        "api_calls": 0, "retries": 0, "training_evidence": "exposed development, not final holdout"})
    assumptions, current = [], None

    def stage(name, command):
        nonlocal current
        reader.check()
        current = dict(next(item for item in stage_plan if item["id"] == name))
        current.update(status="running", command=command)
        receipt["stages"].append(current)
        seconds = min(current["cap_seconds"], reader.deadline - time.monotonic() - 5)
        _require(seconds > 0, "No remaining process budget; next stage was not started")
        print(f"{len(receipt['stages'])}/{len(stage_plan)} {current['label']}", flush=True)
        current["process"] = host.run(command, root, name, seconds)
        result = current["process"]
        _require(result.get("returncode") == 0 and not result.get("error") and not result.get("interrupted"),
                 name + " stopped; inspect its saved logs and partial output (no retry)")

    try:
        receipt["implementation_sha256"] = {
            name: _sha(reader.read(host.project / "scripts" / name))
            for name in ("build_sandbox.py", "spatial_sandbox_workflow.py")
        }
        config_path = host.path(args.config)
        config_raw = reader.read(config_path, 64 << 10)
        config = host.parse(config_raw)
        assumptions = config.get("assumptions", [])
        _require(config.get("profile") == "planar-to-spatial-hypothesis-v1" and _hash(config.get("source_scene_sha256"))
                 and isinstance(assumptions, list) and 1 <= len(assumptions) <= 32
                 and all(isinstance(item, str) and 0 < len(item) <= 2000 for item in assumptions),
                 "Provide source-pinned 3D hypotheses with explicit assumptions; none are inferred")
        (root / "inputs/spatial-config.json").write_bytes(config_raw)
        receipt["input_sha256"][str(config_path)] = _sha(config_raw)
        if args.video:
            tracking_path = host.path(args.tracking_config)
            tracking_raw = reader.read(tracking_path, 64_000)
            host.parse(tracking_raw)
            (root / "inputs/tracking-config.json").write_bytes(tracking_raw)
            video = host.path(args.video)
            video_sha, video_bytes = _video_hash(video, host, reader.check)
            receipt["input_sha256"].update({str(video): video_sha, str(tracking_path): _sha(tracking_raw)})
            receipt["video_hash_bytes"] = video_bytes
            source = root / "planar-lab/run"
            command = prefix + ["lab", "--video", str(video), "--profile", args.tracking_profile,
                "--config", str(root / "inputs/tracking-config.json"), "--out", str(source.parent), "--no-train", "--no-serve", "--seed", "1729"]
            if args.source_note:
                command += ["--source-note", args.source_note]
            stage("tracking", command)
            _require(reader.document(source.parent / "status.json").get("status") == "ready", "Planar processing did not finish")
            _, _, tracking_member = reader.bundle(source)
            source_report = tracking_member("report.json")
            _require(source_report.get("source_sha256") == video_sha and source_report.get("learning", {}).get("status") == "not_requested",
                     "Planar source/video identity or no-training contract differs")
            current["status"] = "completed"
        else:
            source = host.path(args.planar_run)
        source_sha, source_files, source_member = reader.bundle(source)
        source_scene = source_member(scene_name)
        receipt["source"] = {"directory": str(source), "manifest_sha256": source_sha,
                             "scene_artifact": scene_name, "scene_sha256": _canonical(source_scene)}
        _require(_canonical(source_scene) == config["source_scene_sha256"],
                 "3D config source_scene_sha256 does not match the extracted planar scene. Inspect the saved scene and supply explicit matching assumptions; the config was not rewritten.")
        candidate = root / "candidate"
        stage("construction", prefix + ["spatial-scene", "--run", str(source), "--config", str(root / "inputs/spatial-config.json"),
            "--scene-artifact", scene_name, "--manifest-sha256", source_sha, "--max-seconds", "20", "--out", str(candidate)])
        candidate_sha, candidate_files, member = reader.bundle(candidate)
        report, scene = member("report.json"), member("scene.json")
        scene_sha = _canonical(scene)
        _require(report.get("status") == "constructed_hypothesis" and report.get("physical_validation") == "unverified"
                 and report.get("scene_sha256") == scene_sha and scene.get("profile") == "spatial-rigid-v1"
                 and report.get("source_manifest_sha256") == source_sha
                 and candidate_files.get("inputs/planar/manifest.json") == source_sha
                 and candidate_files.get("inputs/original-scene.json") == source_files[scene_name]
                 and candidate_files.get("inputs/authored-config.json") == _sha(config_raw), "Construction identities or completion differ")
        current.update(status="completed", manifest_sha256=candidate_sha, scene_sha256=scene_sha)
        learning = host.path(args.learning) if args.learning else root / "learning"
        learning_sha, policy_sha = None, None
        if args.train:
            command = prefix + ["spatial-train", "--run", str(candidate), "--out", str(learning), "--max-seconds", str(training_seconds)]
            for key, value in TRAINING.items():
                command += ["--" + key.replace("_", "-"), str(value)]
            stage("training", command)
        elif args.learning:
            current = dict(next(item for item in stage_plan if item["id"] == "learning"))
            current.update(status="running", directory=str(learning), new_training=False)
            receipt["stages"].append(current)
            print(f"{len(receipt['stages'])}/{len(stage_plan)} {current['label']}", flush=True)
        if args.train or args.learning:
            learning_sha, learning_files, member = reader.bundle(learning)
            policy = member("policy.json")
            _require(policy.get("scene_sha256") == scene_sha and _canonical(member("scene.json")) == scene_sha,
                     "Trained policy uses a different scene/task")
            policy_sha = learning_files["policy.json"]
            receipt["learning"] = _learning_summary(member("rl-evaluation.json"), scene_sha, policy_sha,
                expected_episodes=TRAINING["eval_episodes"] if args.train else None)
            receipt["learning"]["mode"] = "trained" if args.train else "reused"
            current.update(status="completed", manifest_sha256=learning_sha, policy_sha256=policy_sha)
        lab = root / "lab"
        command = prefix + ["spatial-lab", "--run", str(candidate), "--out", str(lab)]
        if args.train or args.learning:
            command += ["--learning", str(learning)]
        stage("lab", command)
        lab_sha, lab_files, member = reader.bundle(lab / "run")
        lab_report = member("report.json")
        _require(reader.document(lab / "status.json").get("status") == "ready"
                 and lab_report.get("status") == "ready_spatial_lab" and lab_report.get("physical_validation") == "unverified"
                 and lab_report.get("learned_policy_available") is bool(args.train or args.learning)
                 and _canonical(member("scene.json")) == scene_sha
                 and lab_files.get("source-input-manifest.json") == candidate_sha,
                 "Prepared lab does not match the completed candidate and requested controls")
        if args.train or args.learning:
            _require(lab_files.get("learning-input-manifest.json") == learning_sha and lab_files.get("policy.json") == policy_sha,
                     "Lab does not carry the exact completed policy")
        else:
            _require("policy.json" not in lab_files, "Manual lab unexpectedly contains a policy")
        current.update(status="completed", manifest_sha256=lab_sha)
        reader.recheck()
        receipt.update(status="ready", lab=str(lab), resume_commands={
            "browser": shlex.join(prefix + ["lab", "--resume", str(lab), "--port", str(args.port)]),
            "console": shlex.join(prefix + ["lab-console", "--resume", str(lab)])})
    except (ValueError, OSError, TypeError, KeyError, AttributeError, RecursionError) as error:
        receipt["error"] = {"type": type(error).__name__, "message": _text(error)}
        if current is not None and current["status"] == "running":
            current.update(status="stopped", error=_text(error))
        print("Stopped: " + _text(error) + ". Preserved outputs: " + str(root), file=sys.stderr)
    receipt.update(seconds=time.monotonic() - started, metadata_bytes_read=reader.used)
    host.write(root / "workflow-result.json", receipt)
    _index(root, receipt, [item for item in assumptions[:32] if isinstance(item, str)] if isinstance(assumptions, list) else [], host)
    print(f"Step-by-step index: {root / 'index.html'}", flush=True)
    if receipt["status"] != "ready":
        return 1
    print("Sandbox ready. " + SCOPE, flush=True)
    for command in receipt["resume_commands"].values():
        print(command, flush=True)
    if args.console or args.serve:
        command = prefix + (["lab-console", "--resume", str(lab)] if args.console else
                            ["lab", "--resume", str(lab), "--port", str(args.port)])
        return host.launch(command, cwd=host.project, env=host.environment(root))
    return 0
