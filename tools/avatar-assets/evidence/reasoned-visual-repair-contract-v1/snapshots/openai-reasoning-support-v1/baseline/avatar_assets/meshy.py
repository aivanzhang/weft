"""Explicit, bounded Meshy stages with immutable inputs and resumable task identities."""

import base64
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlsplit, urlunsplit

from PIL import Image

from avatar_assets.meshy_access import Client, Ledger, MeshyError, canonical, digest, now
from avatar_assets.safety import artifacts, local_file, new_directory, read_json, require, sha256, slug, write_json

TERMINAL = {"SUCCEEDED", "FAILED", "CANCELED"}
MODEL = "meshy-7.1"
PRICING = "https://docs.meshy.ai/en/api/pricing"


def _text(value, limit, label):
    require(isinstance(value, str) and 0 < len(value.strip()) <= limit, f"Invalid {label}")
    return value


def validate_config(config):
    require(isinstance(config, dict) and config.get("schema_version") == 1, "Expected Meshy schema_version 1")
    require(set(config) <= {"schema_version", "id", "subject", "style", "prompt", "reference", "reference_sha256",
                           "generation", "states", "rig", "height_meters", "max_credits", "rights"}, "Unknown config field")
    slug(config.get("id"))
    for field, limit in (("subject", 400), ("style", 800), ("prompt", 800), ("rights", 4000)):
        _text(config.get(field), limit, field)
    generation = config.get("generation")
    require(isinstance(generation, dict) and set(generation) ==
            {"mode", "geometry_resolution", "texture_resolution", "target_polycount", "pose_mode"}, "Invalid generation settings")
    require(generation["mode"] in ("image", "text"), "Generation mode must be image or text")
    require(generation["geometry_resolution"] in ("standard", "2k", "4k"), "Invalid geometry resolution")
    require(generation["texture_resolution"] in ("2k", "4k"), "This bounded profile supports 2K/4K textures")
    require(type(generation["target_polycount"]) is int and 1000 <= generation["target_polycount"] <= 100000,
            "Target must be 1000–100000 triangles; output still requires validation")
    require(generation["pose_mode"] in ("", "a-pose", "t-pose"), "Invalid requested rest pose")
    reference = config.get("reference")
    require((generation["mode"] == "image") == bool(reference), "Image mode requires one local reference; text mode takes none")
    if reference:
        _text(reference, 1000, "reference path")
        require(re.fullmatch(r"[a-f0-9]{64}", config.get("reference_sha256", "")), "Pin the selected reference SHA256")
    require(config.get("rig") in ("humanoid", "none"), "Meshy rigging supports humanoids; other subjects need local articulation")
    height = config.get("height_meters")
    require(type(height) in (int, float) and math.isfinite(height) and .05 <= height <= 10, "Invalid authored height")
    states = config.get("states")
    require(isinstance(states, list) and len(states) <= 8, "At most eight requested motion states")
    require(not states or config["rig"] == "humanoid", "Generated motion requires a humanoid rig")
    seen = set()
    for state in states:
        require(isinstance(state, dict) and set(state) == {"id", "prompt", "duration"}, "Invalid state fields")
        name = slug(state.get("id"))
        require(name not in seen, "Duplicate state ID")
        seen.add(name)
        _text(state.get("prompt"), 400, "motion prompt")
        duration = state.get("duration")
        require(type(duration) in (int, float) and math.isfinite(duration) and 2 <= duration <= 10
                and duration*2 == int(duration*2), "Motion duration must be 2–10 seconds in half-second increments")
    estimate = 30 + (5 if generation["geometry_resolution"] != "standard" else 0)
    estimate += (5 if config["rig"] == "humanoid" else 0) + 13*len(states)
    require(type(config.get("max_credits")) is int and estimate <= config["max_credits"] <= 500,
            "Job credit ceiling must cover its estimate and be at most 500")
    return estimate


def prepare(config_path, output):
    config_path = local_file(config_path)
    config = read_json(config_path)
    estimate = validate_config(config)
    reference = None
    if config.get("reference"):
        reference = local_file(config_path.parent/config["reference"], 8*1024*1024)
        require(reference.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"), "Unsupported reference format")
        require(sha256(reference) == config["reference_sha256"], "Reference bytes changed")
        with Image.open(reference) as image:
            expected = {'.png':'PNG', '.jpg':'JPEG', '.jpeg':'JPEG', '.webp':'WEBP'}[reference.suffix.lower()]
            require(image.format == expected, 'Reference content format differs from its extension')
            require(image.width*image.height <= 16_777_216 and getattr(image, "n_frames", 1) == 1,
                    "Reference must be one bounded still image")
            image.verify()
    out = new_directory(output)
    (out/"inputs").mkdir()
    (out/"stages").mkdir()
    (out/"sources").mkdir()
    shutil.copyfile(config_path, out/"inputs/config.json")
    resolved = dict(config)
    if reference:
        target = out/"inputs"/("reference"+reference.suffix.lower())
        shutil.copyfile(reference, target)
        resolved["reference"] = target.relative_to(out).as_posix()
    for name in ("meshy.py", "meshy_access.py"):
        shutil.copyfile(Path(__file__).parent/name, out/"sources"/name)
    plan = {"schema_version": 1, "id": config["id"], "created": now(), "config": resolved,
            "model": MODEL, "max_credits": config["max_credits"], "estimated_credits": estimate,
            "estimated_paid_calls": (1 if reference else 2)+(1 if config["rig"] == "humanoid" else 0)+2*len(config["states"]),
            "pricing_source": PRICING, "pricing_checked": "2026-10-31", "usd_cost": None,
            "usd_reason": "Account credit purchase price is not exposed by the balance endpoint",
            "automatic_paid_retries": 0, "reference_reconstruction": False,
            "limits": {"reference_bytes": 8*1024*1024, "download_bytes": 64*1024*1024,
                       "output_bytes": 512*1024*1024, "polls_per_stage": 120, "min_poll_seconds": 15,
                       "request_deadline_seconds": 45, "download_deadline_seconds": 120},
            "sources": artifacts(out/"sources")}
    write_json(out/"plan.json", plan)
    return {"status": "prepared_no_paid_call", "out": str(out), "estimated_credits": estimate,
            "estimated_paid_calls": plan["estimated_paid_calls"], "plan_sha256": sha256(out/"plan.json")}


def _plan(folder):
    root = Path(folder).resolve()
    require(not Path(folder).is_symlink(), "Job folder cannot be a symlink")
    plan = read_json(root/"plan.json")
    validate_config(plan["config"])
    require(plan["model"] == MODEL and plan["id"] == plan["config"]["id"]
            and plan["max_credits"] == plan["config"]["max_credits"], "Inconsistent job plan")
    if plan["config"].get("reference"):
        path = local_file(root/plan["config"]["reference"], 8*1024*1024)
        require(path.is_relative_to(root/"inputs") and sha256(path) == plan["config"]["reference_sha256"],
                "Prepared reference changed or escaped inputs")
    return root, plan


@contextmanager
def _lock(root):
    with (root/"job.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Job is already owned by another process") from None
        yield


def _dependency(ledger, plan, step):
    row = ledger.call(plan["id"], step)
    require(row and row["state"] == "SUCCEEDED" and row["task_id"], f"Stage {step} must succeed first")
    return row["task_id"]


def stage_request(root, plan, stage, ledger):
    slug(stage)
    cfg, gen = plan["config"], plan["config"]["generation"]
    image_mode = gen["mode"] == "image"
    if stage == "mesh":
        body = {"ai_model": MODEL, "model_type": "standard", "geometry_resolution": gen["geometry_resolution"],
                "should_remesh": True, "target_polycount": gen["target_polycount"], "topology": "triangle",
                "pose_mode": gen["pose_mode"], "target_formats": ["glb"]}
        price = 20+(5 if gen["geometry_resolution"] != "standard" else 0)
        if image_mode:
            reference = root/cfg["reference"]
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}[reference.suffix]
            body.update(image_url="data:"+mime+";base64,"+base64.b64encode(reference.read_bytes()).decode(),
                        should_texture=True, enable_pbr=True, texture_resolution=gen["texture_resolution"],
                        texture_prompt=cfg["prompt"])
            return "/openapi/v1/image-to-3d", body, price+10
        body.update(mode="preview", prompt=cfg["prompt"])
        return "/openapi/v2/text-to-3d", body, price
    if stage == "texture":
        require(not image_mode, "Image-to-3D already includes texturing")
        return "/openapi/v2/text-to-3d", {"mode": "refine", "ai_model": MODEL,
                "preview_task_id": _dependency(ledger, plan, "mesh"), "enable_pbr": True,
                "texture_resolution": gen["texture_resolution"], "texture_prompt": cfg["style"], "target_formats": ["glb"]}, 10
    require(cfg["rig"] == "humanoid", "Rig/motion not enabled for this subject")
    if stage == "rig":
        return "/openapi/v1/rigging", {"input_task_id": _dependency(ledger, plan, "mesh" if image_mode else "texture"),
                                      "height_meters": cfg["height_meters"]}, 5
    rig_id = _dependency(ledger, plan, "rig")
    for state in cfg["states"]:
        if stage == "motion_"+state["id"]:
            return "/openapi/v1/text-to-motion", {"prompt": state["prompt"], "duration": state["duration"], "mode": "prime"}, 10
        if stage == "animate_"+state["id"]:
            return "/openapi/v1/animations", {"rig_task_id": rig_id,
                    "motion_task_id": _dependency(ledger, plan, "motion_"+state["id"])}, 3
    raise ValueError("Unknown configured stage")


def public_value(value):
    if isinstance(value, dict):
        return {k: public_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [public_value(v) for v in value]
    if isinstance(value, str) and value.startswith("https://"):
        parts = urlsplit(value)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    if isinstance(value, str) and value.startswith("data:"):
        return {"embedded_input_omitted": True, "sha256": digest(value)}
    return value


def _receipt(root, stage, kind, value):
    folder = root/"stages"/slug(stage)
    folder.mkdir(exist_ok=True)
    sources = folder/"tools"
    sources.mkdir(exist_ok=True)
    hashes = {}
    for name in ("meshy.py", "meshy_access.py"):
        source = Path(__file__).parent/name
        checksum = sha256(source)
        target = sources/(checksum[:16]+"-"+name)
        if not target.exists():
            shutil.copyfile(source, target)
        require(sha256(target) == checksum, "Execution source snapshot changed")
        hashes[name] = checksum
    number = len(list(folder.glob(kind+"-*.json")))
    require(number < 128, "Stage receipt count exceeded")
    write_json(folder/f"{kind}-{number:03d}.json", {**public_value(value), "execution_sources": hashes})


def submit(folder, stage, client=None, ledger=None):
    root, plan = _plan(folder)
    owned = ledger is None
    ledger = ledger or Ledger()
    try:
        with _lock(root):
            ledger.bind_job(plan, root)
            old = ledger.call(plan["id"], stage)
            if old:
                require(old["state"] not in ("submitting", "uncertain"), "Uncertain submission: reconcile its task ID; never retry POST")
                return {"status": old["state"], "task_id": old["task_id"], "new_paid_call": False}
            endpoint, body, credits = stage_request(root, plan, stage, ledger)
            client = client or Client()
            balance = client.balance()
            require(balance["balance"] >= credits, "Insufficient Meshy credits; no generation submitted")
            request = {"endpoint": endpoint, "body": public_value(body), "request_sha256": digest(body),
                       "estimated_credits": credits, "created": now(), "balance_before": balance}
            if "image_url" in body:
                request["body"]["image_url"] = {"local_reference": plan["config"]["reference"],
                    "sha256": plan["config"]["reference_sha256"], "encoding": "base64 data URI; original bytes"}
            _receipt(root, stage, "request", request)
            ledger.reserve(plan["id"], stage, endpoint, request, credits)
            try:
                response = client.request("POST", endpoint, body)
                task_id = response.get("result")
                require(isinstance(task_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_id), "Missing task identity")
                ledger.update(plan["id"], stage, "PENDING", task_id, response)
            except Exception as error:
                certain_rejection = isinstance(error, MeshyError) and error.status in (400, 401, 402, 403, 404, 422, 429)
                state = "rejected" if certain_rejection else "uncertain"
                ledger.update(plan["id"], stage, state, response={"error": str(error), "http_status": getattr(error, "status", None)})
                _receipt(root, stage, "failure", {"state": state, "error": str(error), "recorded": now()})
                raise
            _receipt(root, stage, "submission", {"task_id": task_id, "created": now(), "estimated_credits": credits})
            return {"status": "PENDING", "task_id": task_id, "new_paid_call": True, "reserved_credits": credits}
    finally:
        if owned:
            ledger.close()


def status(folder, stage, client=None, ledger=None):
    root, plan = _plan(folder)
    owned = ledger is None
    ledger = ledger or Ledger()
    try:
        with _lock(root):
            ledger.bind_job(plan, root)
            row = ledger.call(plan["id"], slug(stage))
            require(row and row["task_id"], "No task ID; submission cannot be polled or safely repeated")
            cached = json.loads(row["response"] or "{}")
            polls = sorted((root/"stages"/stage).glob("status-*.json"))
            if row["state"] in TERMINAL or polls and time.time()-polls[-1].stat().st_mtime < 15:
                result = cached
            else:
                require(len(polls) < 120, "Poll bound reached; inspect the saved task before extending the watch")
                result = (client or Client()).request("GET", row["endpoint"]+"/"+row["task_id"])
                require(result.get("id") == row["task_id"] and result.get("status") in
                        TERMINAL|{"PENDING", "IN_PROGRESS"}, "Unexpected Meshy task identity or state")
                ledger.update(plan["id"], stage, result["status"], response=result)
                _receipt(root, stage, "status", {"observed": now(), "task": result})
            return {"status": result.get("status", row["state"]), "task_id": row["task_id"],
                    "progress": result.get("progress"), "consumed_credits": result.get("consumed_credits"),
                    "error": public_value(result.get("task_error"))}
    finally:
        if owned:
            ledger.close()


def fetch(folder, stage, client=None, ledger=None):
    root, plan = _plan(folder)
    owned = ledger is None
    ledger = ledger or Ledger()
    try:
        with _lock(root):
            ledger.bind_job(plan, root)
            row = ledger.call(plan["id"], slug(stage))
            require(row and row["state"] == "SUCCEEDED", "Only successful jobs can be fetched")
            data = json.loads(row["response"])
            target = root/"stages"/stage/"downloads"
            if (target/"manifest.json").exists():
                manifest = read_json(target/"manifest.json")
                for item in manifest["artifacts"]:
                    require(sha256(target/item["path"]) == item["sha256"], "Downloaded artifact changed")
                return {"status": "already_downloaded", "out": str(target)}
            require(not target.exists(), "Partial download preserved; inspect it before choosing a new download attempt")
            target.mkdir()
            urls = {}
            if data.get("model_urls", {}).get("glb"):
                urls["model.glb"] = data["model_urls"]["glb"]
            result = data.get("result") or {}
            for key in ("rigged_character_glb_url", "animation_glb_url"):
                if result.get(key):
                    urls["model.glb"] = result[key]
            for key in ("animation_fbx_url", "animation_bvh_url"):
                if result.get(key) and "model.glb" not in urls:
                    urls["motion."+("fbx" if "fbx" in key else "bvh")] = result[key]
            if result.get("motion_url"):
                require(result.get("motion_format") in ("fbx", "bvh"), "Unsupported raw motion format")
                urls["motion."+result["motion_format"]] = result["motion_url"]
            if data.get("thumbnail_url"):
                urls["thumbnail.png"] = data["thumbnail_url"]
            require(urls, "Task returned no recognized downloadable artifacts")
            records = {}
            client = client or Client()
            for filename, url in urls.items():
                records[filename] = client.download(url, target/filename)
            _receipt(root, stage, "download", {"recorded": now(), "files": records})
            write_json(target/"manifest.json", {"schema_version": 1, "kind": "meshy_original_downloads",
                       "status": "downloaded_not_visually_accepted", "task_id": row["task_id"], "stage": stage,
                       "plan_sha256": sha256(root/"plan.json"), "downloads": records, "artifacts": artifacts(target)})
            artifacts(root)
            return {"status": "downloaded_not_visually_accepted", "out": str(target), "files": records}
    finally:
        if owned:
            ledger.close()


def run(folder, max_seconds=900, *, client=None, ledger=None, clock=time.monotonic, sleep=time.sleep):
    """Resume a finite dependency graph; uncertain submissions never receive a second POST."""
    require(type(max_seconds) is int and 1 <= max_seconds <= 3600, 'Run deadline must be 1–3600 seconds')
    root, plan = _plan(folder)
    cfg = plan['config']
    stages = ['mesh']+(['texture'] if cfg['generation']['mode'] == 'text' else [])
    if cfg['rig'] == 'humanoid':
        stages += ['rig']
    for state in cfg['states']:
        stages += ['motion_'+state['id'], 'animate_'+state['id']]
    started, completed = clock(), []
    for stage in stages:
        if clock()-started >= max_seconds:
            return {'status': 'checkpointed', 'next_stage': stage, 'completed': completed, 'out': str(root)}
        submit(root, stage, client=client, ledger=ledger)
        while True:
            result = status(root, stage, client=client, ledger=ledger)
            if result['status'] == 'SUCCEEDED':
                fetch(root, stage, client=client, ledger=ledger)
                completed.append(stage)
                break
            require(result['status'] not in ('FAILED', 'CANCELED'), 'Provider stage failed; retained without paid retry: '+stage)
            if clock()-started+15 >= max_seconds:
                return {'status': 'checkpointed', 'next_stage': stage, 'task_id': result.get('task_id'),
                        'completed': completed, 'out': str(root)}
            sleep(15)
    return {'status': 'generated_downloaded_visual_review_pending', 'completed': completed, 'out': str(root),
            'elapsed_seconds': round(clock()-started, 3), 'quality_accepted': False}


def add_commands(parent):
    parser = parent.add_parser("meshy", help="Explicit authorized Meshy generation and resumable tasks")
    _commands(parser)


def _commands(parser):
    sub = parser.add_subparsers(dest="meshy_command", required=True)
    sub.add_parser("account", help="Free key/balance check; never prints the key")
    budget = sub.add_parser("budget", help="Set a finite, non-resetting credit ceiling")
    budget.add_argument("--credits", type=int, required=True)
    sub.add_parser("ledger", help="Read local reserved and reported usage")
    authorize = sub.add_parser('authorize', help='Append recorded user approval for one new job; does not purchase credits')
    authorize.add_argument('folder')
    authorize.add_argument('--approval-record', required=True)
    plan = sub.add_parser("plan", help="Prepare local immutable inputs; no provider request")
    plan.add_argument("config")
    plan.add_argument("--out", required=True)
    run_parser = sub.add_parser('run', help='Resume all configured stages within a bounded time; no automatic paid retry')
    run_parser.add_argument('folder')
    run_parser.add_argument('--max-seconds', type=int, default=900)
    for name in ("submit", "status", "fetch"):
        item = sub.add_parser(name)
        item.add_argument("folder")
        item.add_argument("--stage", required=True)
def main(argv=None):
    import argparse
    integrated = isinstance(argv, argparse.Namespace)
    if integrated:
        args = argv
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        _commands(parser)
        args = parser.parse_args(argv)
    command = args.meshy_command
    try:
        if command == "account":
            result = Client().balance()
        elif command in ("budget", "ledger"):
            ledger = Ledger()
            try:
                result = ledger.configure(args.credits) if command == "budget" else ledger.summary()
            finally:
                ledger.close()
        elif command == "plan":
            result = prepare(args.config, args.out)
        elif command == 'authorize':
            root, plan = _plan(args.folder)
            ledger = Ledger()
            try:
                result = ledger.authorize_job(plan, root, read_json(args.approval_record))
            finally:
                ledger.close()
        elif command == 'run':
            result = run(args.folder, args.max_seconds)
        else:
            result = {"submit": submit, "status": status, "fetch": fetch}[command](args.folder, args.stage)
        if integrated:
            return result
        print(json.dumps(result, sort_keys=True, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
