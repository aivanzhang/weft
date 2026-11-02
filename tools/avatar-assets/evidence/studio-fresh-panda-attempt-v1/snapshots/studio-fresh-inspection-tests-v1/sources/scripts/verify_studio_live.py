"""Serve the real Studio temporarily, or read its canonical ownership without dispatch."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import sys
import threading
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAX_JSON = 1024**2
MAX_FILE = 64 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path, maximum):
    require(path == path.resolve() and path.is_file() and not path.is_symlink(),
            "Expected a canonical regular file")
    info = path.stat()
    require(0 <= info.st_size <= maximum, "File exceeds its inspection bound")
    return info


def relative_file(root, name, maximum=MAX_JSON):
    require(isinstance(name, str) and name and not Path(name).is_absolute()
            and all(part not in ("", ".", "..") for part in name.split("/")), "Invalid relative artifact path")
    path = root / name
    require(path.is_relative_to(root), "Artifact escapes its root")
    regular(path, maximum)
    return path


def parse_json(value):
    def unique(items):
        result = {}
        for key, item in items:
            require(key not in result, "Duplicate JSON field")
            result[key] = item
        return result

    def nonfinite(_value):
        raise ValueError("Nonfinite JSON value")

    require(len(value.encode("utf8") if isinstance(value, str) else value) <= MAX_JSON,
            "JSON exceeds its inspection bound")
    try:
        return json.loads(value, object_pairs_hook=unique, parse_constant=nonfinite)
    except RecursionError:
        raise ValueError("JSON nesting exceeds its inspection bound") from None


def read_json(path):
    regular(path, MAX_JSON)
    with path.open("rb") as stream:
        return parse_json(stream.read(MAX_JSON + 1))


def sha(path, maximum=MAX_FILE):
    before = regular(path, maximum)
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(min(1024**2, maximum + 1 - count)):
            count += len(chunk)
            require(count <= maximum, "Hash input exceeds its inspection bound")
            digest.update(chunk)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            and count == before.st_size, "Hash input changed during inspection")
    return digest.hexdigest()


def pinned_sources(root, entries, expected, *, check_current=True):
    require(isinstance(entries, list) and 0 < len(entries) <= 256, "Invalid source inventory")
    names, total = set(), 0
    for item in entries:
        require(isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"}, "Invalid source receipt")
        name = item["path"]
        require(isinstance(name, str) and name in expected and name not in names, "Unexpected or duplicate source pin")
        names.add(name)
        require(type(item["bytes"]) is int and 0 < item["bytes"] <= MAX_JSON, "Invalid source byte bound")
        total += item["bytes"]
        require(total <= 16 * MAX_JSON, "Aggregate source inspection bound exceeded")
        saved = relative_file(root, name)
        require(saved.stat().st_size == item["bytes"] and sha(saved, MAX_JSON) == item["sha256"]
                and (not check_current or sha(expected[name], MAX_JSON) == item["sha256"]),
                "Prepared source pin changed")
    require(names == set(expected) or not check_current, "Prepared job lacks current source pins")


def fresh_generation(state, job_id, output, request, prepared, plan, request_digest):
    from avatar_assets import meshy
    from avatar_assets.pipeline import _pipeline_sources
    from avatar_assets.pipeline_prompts import validate_plan
    from avatar_assets.studio_worker import _validate_prepared_request

    require(request["reference"] is None and prepared.get("reference") is None,
            "Fresh inspection currently admits text-only jobs")
    _validate_prepared_request(prepared, request, output)
    require(validate_plan(plan, prepared)["status"] == "ready", "Fresh generation needs a ready planner result")
    request_hash = sha(output / "request.json", MAX_JSON)
    receipt = read_json(output / "input-receipt.json")
    require(isinstance(receipt, dict) and set(receipt) == {"request_sha256", "sources"}
            and receipt["request_sha256"] == request_hash, "Prepared request hash differs")
    binding_path = state / "jobs" / job_id / "pipeline-request.json"
    require(read_json(binding_path) == {"schema_version": 1, "studio_request_digest": request_digest,
                                      "pipeline_request_sha256": request_hash},
            "Prepared request is not bound to the canonical Studio request")
    sealed = (output / "delivery-receipt.json").is_file() and (output / "delivery.json").is_file()
    pinned_sources(output / "sources", receipt["sources"], _pipeline_sources(), check_current=not sealed)
    config_path, generation_path = output / "generation-config.json", output / "generation/plan.json"
    config, generation = read_json(config_path), read_json(generation_path)
    estimate = meshy.validate_config(config)
    require(set(config) == {"schema_version", "id", "subject", "style", "prompt", "height_meters", "rig",
                            "states", "max_credits", "rights", "generation"}, "Unexpected fresh generation fields")
    for name, expected in {"id": prepared["id"] + "-mesh", "subject": plan["subject"], "style": plan["style"],
                           "prompt": plan["generation_prompt"], "height_meters": plan["height_meters"],
                           "rig": "none", "states": [], "max_credits": 35,
                           "generation": {"mode": "text", "geometry_resolution": "4k", "texture_resolution": "4k",
                                          "target_polycount": 35000, "pose_mode": ""}}.items():
        require(config[name] == expected, "Fresh generation differs from the prepared plan: " + name)
    require(read_json(output / "generation/inputs/config.json") == config and generation.get("config") == config,
            "Generation config copies differ")
    generation_hash = sha(generation_path, MAX_JSON)
    require(read_json(output / "generation-receipt.json") == {"plan_sha256": generation_hash},
            "Generation plan hash differs")
    for name, expected in {"schema_version": 1, "id": config["id"], "model": meshy.MODEL, "max_credits": 35,
                           "estimated_credits": estimate, "estimated_paid_calls": 2, "automatic_paid_retries": 0,
                           "reference_reconstruction": False, "usd_cost": None}.items():
        require(type(generation.get(name)) is type(expected) and generation.get(name) == expected,
                "Unexpected fresh generation plan field: " + name)
    require(estimate == 35, "Unexpected fresh generation estimate")
    pinned_sources(output / "generation/sources", generation.get("sources"),
                   {name: Path(meshy.__file__).parent / name for name in ("meshy.py", "meshy_access.py")},
                   check_current=not sealed)
    files = [output / "input-receipt.json", binding_path, config_path, output / "generation/inputs/config.json",
             generation_path, output / "generation-receipt.json"]
    return generation, {"model": generation["model"], "mode": "text", "settings": config["generation"],
                        "estimated_credits": estimate, "estimated_paid_calls": 2, "automatic_paid_retries": 0,
                        "usd_cost": None, "config_sha256": sha(config_path, MAX_JSON),
                        "plan_sha256": generation_hash, "maximum_agent_calls": prepared["max_agent_calls"],
                        "authorization_checked": False, "execution_gate": "canonical_ledger_and_explicit_start"}, files


def reused_source(output, request, prepared):
    source, selected = prepared.get("reuse_model"), request.get("reuse_model")
    require(isinstance(source, dict) and isinstance(selected, dict), "Prepared source selection differs")
    original = Path(selected["path"])
    copied = relative_file(output, source["path"], MAX_FILE)
    require(original.is_relative_to(ROOT / "outputs"), "Reused source escapes local outputs")
    info = regular(original, MAX_FILE)
    require({"bytes": info.st_size, "device": info.st_dev, "inode": info.st_ino, "mtime_ns": info.st_mtime_ns}
            == {key: selected[key] for key in ("bytes", "device", "inode", "mtime_ns")}
            and source["original_path"] == str(original) and info.st_size > 0
            and copied.stat().st_size == info.st_size and source["bytes"] == info.st_size
            and sha(original) == source["sha256"] and sha(copied) == source["sha256"],
            "Prepared reused source bytes/hash differ from the canonical file identity")
    return {"sha256": source["sha256"], "bytes": source["bytes"], "path": source["path"]}


def local_time(value):
    return datetime.fromtimestamp(value, ZoneInfo("America/Los_Angeles")).isoformat() if value else None


def inspect(job_id):
    from avatar_assets.studio_jobs import _digest, process_identity

    require(isinstance(job_id, str) and re.fullmatch(r"[a-f0-9]{24}", job_id), "Invalid canonical job ID")
    binding = read_json(ROOT / ".local/studio-registry.json")
    require(isinstance(binding, dict) and set(binding) == {"schema_version", "state_dir"}
            and binding["schema_version"] == 1, "Invalid canonical registry binding")
    state = Path(binding["state_dir"])
    if state != state.resolve() or not state.is_relative_to(ROOT / ".local"):
        raise ValueError("Canonical Studio registry escapes the workspace")
    database = state / "jobs.sqlite3"
    regular(database, MAX_FILE)
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=1) as db:
        db.row_factory = sqlite3.Row
        lengths = db.execute("SELECT length(CAST(request AS BLOB)),length(CAST(data AS BLOB)),length(output),length(status) FROM jobs WHERE id=?", (job_id,)).fetchone()
        require(lengths is not None, "Selected job is absent from the canonical registry")
        require(all(type(size) is int and size <= maximum for size, maximum in zip(lengths, (MAX_JSON, MAX_JSON, 80, 80))),
                "Canonical job row exceeds its inspection bound")
        row = db.execute("SELECT request,data,request_digest,output,status FROM jobs WHERE id=?", (job_id,)).fetchone()
        queued = [item[0] for item in db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY queued_at,id LIMIT 257")]
        require(len(queued) <= 256, "Canonical queue exceeds its inspection bound")
        active = db.execute("SELECT job FROM active WHERE id=1").fetchone()
    data, request = parse_json(row["data"]), parse_json(row["request"])
    require(isinstance(data, dict) and isinstance(request, dict)
            and set(request) == {"prompt", "style", "custom_actions", "preset", "reference", "reuse_model"},
            "Invalid canonical Studio request")
    if _digest(request) != row["request_digest"]:
        raise ValueError("Canonical immutable request digest differs")
    if read_json(state / "jobs" / job_id / "request.json") != request:
        raise ValueError("Canonical saved request differs")
    require(row["output"] == "studio-" + job_id,
            "Invalid canonical output name")
    output = ROOT / "outputs" / row["output"]
    if output != output.resolve() or not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Job output escapes the workspace")
    prepared_path, plan_path = output / "request.json", output / "plan.json"
    prepared, plan = read_json(prepared_path), read_json(plan_path)
    for key in ("prompt", "style", "custom_actions", "preset"):
        if prepared.get(key) != request.get(key):
            raise ValueError("Prepared request differs from the Studio input: " + key)
    files = [prepared_path, plan_path]
    source_reuse, source_generation = None, None
    if request["reuse_model"] is not None:
        source_reuse = reused_source(output, request, prepared)
        generation_path = output / "generation/plan.json"
        generation = read_json(generation_path) if generation_path.exists() else None
        mode = "reused_generated_source"
    else:
        generation, source_generation, extra = fresh_generation(state, job_id, output, request, prepared, plan,
                                                               row["request_digest"])
        files.extend(extra)
        mode = "new_meshy_generation"
    plan_id = _digest({"request": row["request_digest"], "plan": plan,
                       "generation": generation})
    for key in ("plan_id", "expected_plan_id"):
        require(data.get(key) in (None, plan_id), "Recorded canonical plan ID differs")
    owner = {key: data.get(key) for key in ("owner_pid", "owner_identity", "operation", "invocations",
                                           "invocation_dir", "cancel_requested")}
    owner.update(active_lease_job=active[0] if active else None, queued_jobs=queued,
                 heartbeat_local=local_time(data.get("heartbeat")), ended_local=local_time(data.get("ended")))
    owner["identity_verified_live"] = False
    if type(owner["owner_pid"]) is int and owner["owner_pid"] > 0:
        try:
            owner["identity_verified_live"] = process_identity(owner["owner_pid"]) == owner["owner_identity"]
        except (ValueError, OSError, ProcessLookupError):
            pass
    return {"schema_version": 1, "provenance": "actual_pipeline", "id": job_id,
            "status": row["status"], "state_dir": str(state.relative_to(ROOT)),
            "output": str(output.relative_to(ROOT)), "plan_id": plan_id, "owner": owner,
            "request": {key: request[key] for key in ("prompt", "style", "custom_actions", "preset")},
            "generation_mode": mode, "reference_selected": request["reference"] is not None,
            "source_reuse": source_reuse, "source_generation": source_generation,
            "files": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p, MAX_JSON)} for p in files],
            "recorded_local": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
            "ownership_scope": "Read-only canonical registry and selected PID identity; no dispatch, cancellation or signal"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--serve", action="store_true", help="Start the real ephemeral HTTP server; no implicit test Start")
    parser.add_argument("--max-seconds", type=int, default=915)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{24}", args.job) or not 30 <= args.max_seconds <= 1830:
        raise ValueError("Invalid job ID or bounded lifetime")
    before = inspect(args.job)
    if not args.serve:
        print(json.dumps(before, sort_keys=True), flush=True)
        return
    if before["owner"]["queued_jobs"]:
        raise ValueError("Refuse server startup while queued work could be implicitly dispatched")
    if before["owner"]["active_lease_job"] not in (None, args.job):
        raise ValueError("Another job owns the worker; preserve it and defer this test")
    from avatar_assets.studio import create_server

    server = create_server(port=0, root=ROOT)
    if str(server.manager.state_dir.relative_to(ROOT)) != before["state_dir"]:
        raise ValueError("Server did not use the canonical registry")
    timer = threading.Timer(args.max_seconds, server.shutdown)
    timer.daemon = True
    timer.start()

    def stop_server(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    print(json.dumps({"port": server.server_address[1], "pid": os.getpid(),
                      "job_id": args.job, "canonical_state_dir": before["state_dir"]}), flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        timer.cancel()
        server.server_close()
        # Detached workers retain their own lease and bounds after this server exits.


if __name__ == "__main__":
    main()
