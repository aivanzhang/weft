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


def read_json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError("Expected a bounded regular JSON file")
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_time(value):
    return datetime.fromtimestamp(value, ZoneInfo("America/Los_Angeles")).isoformat() if value else None


def inspect(job_id):
    from avatar_assets.studio_jobs import _digest, process_identity

    binding = read_json(ROOT / ".local/studio-registry.json")
    state = Path(binding["state_dir"])
    if state != state.resolve() or not state.is_relative_to(ROOT / ".local"):
        raise ValueError("Canonical Studio registry escapes the workspace")
    database = state / "jobs.sqlite3"
    if database.is_symlink() or not database.is_file():
        raise ValueError("Canonical Studio database is unavailable")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=1) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("Selected job is absent from the canonical registry")
        queued = [item[0] for item in db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY queued_at,id")]
        active = db.execute("SELECT job FROM active WHERE id=1").fetchone()
    data, request = json.loads(row["data"]), json.loads(row["request"])
    if _digest(request) != row["request_digest"]:
        raise ValueError("Canonical immutable request digest differs")
    if read_json(state / "jobs" / job_id / "request.json") != request:
        raise ValueError("Canonical saved request differs")
    output = ROOT / "outputs" / row["output"]
    if output != output.resolve() or not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Job output escapes the workspace")
    prepared_path, plan_path = output / "request.json", output / "plan.json"
    prepared, plan = read_json(prepared_path), read_json(plan_path)
    for key in ("prompt", "style", "custom_actions", "preset"):
        if prepared.get(key) != request.get(key):
            raise ValueError("Prepared request differs from the Studio input: " + key)
    if not prepared.get("reuse_model") or not request.get("reuse_model"):
        raise ValueError("This verifier accepts explicit source reuse only; it grants no Meshy spend")
    source = prepared["reuse_model"]
    selected = request["reuse_model"]
    original = Path(selected["path"])
    copied = output / source["path"]
    if (original != original.resolve() or not original.is_relative_to(ROOT / "outputs")
            or copied != copied.resolve() or not copied.is_relative_to(output)
            or not original.is_file() or not copied.is_file()):
        raise ValueError("Reused source must stay within its pinned local output paths")
    info = original.stat()
    if ({"bytes": info.st_size, "device": info.st_dev, "inode": info.st_ino, "mtime_ns": info.st_mtime_ns}
            != {key: selected[key] for key in ("bytes", "device", "inode", "mtime_ns")}
            or source["original_path"] != str(original) or not 0 < info.st_size <= 64 * 1024**2
            or copied.stat().st_size != info.st_size or source["bytes"] != info.st_size
            or sha(original) != source["sha256"] or sha(copied) != source["sha256"]):
        raise ValueError("Prepared reused source bytes/hash differ from the canonical file identity")
    generation = output / "generation/plan.json"
    plan_id = _digest({"request": row["request_digest"], "plan": plan,
                       "generation": read_json(generation) if generation.exists() else None})
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
            "source_reuse": {"sha256": source["sha256"], "bytes": source["bytes"], "path": source["path"]},
            "files": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in (prepared_path, plan_path)],
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
