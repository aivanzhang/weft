"""The studio bridge drives plan/grant/start/adopt against the studio contract only."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from weft.portal_avatar_studio import AvatarStudioBridge, list_library, read_library_zip


class FakeStudio:
    """Scripted studio: each GET advances the job the way the real pipeline would."""

    def __init__(self, tmp_path, *, authorization_reason=None):
        self.jobs = {}
        self.calls = []
        self.model_bytes = b"GLB" * 2000
        self.authorization_reason = authorization_reason
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, payload):
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path.endswith("/files/model"):
                    return self._reply(200, fake.model_bytes)
                identifier = self.path.rsplit("/", 1)[-1]
                job = fake.jobs[identifier]
                fake.calls.append(("GET", identifier, job["status"]))
                payload = {"id": identifier, "status": job["status"], "plan_id": job.get("plan_id"),
                           "elapsed_seconds": 1.0,
                           "stages": [{"stage": "generation", "status": "running"}],
                           "plan": {"authorization": {"can_start": job.get("can_start", False),
                                                       "reason": fake.authorization_reason,
                                                       "estimated_credits": 35}},
                           "result": ({"model_url": f"/api/studio/jobs/{identifier}/files/model"}
                                      if job["status"] == "complete" else None),
                           "error": job.get("error")}
                self._reply(200, payload)
                job["status"] = job.get("next", {}).get(job["status"], job["status"])

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                if self.path == "/api/studio/jobs":
                    identifier = "a" * 24
                    fake.jobs[identifier] = {"status": "created"}
                    fake.calls.append(("POST", "jobs", json.loads(body)["prompt"]))
                    return self._reply(201, {"id": identifier, "status": "created"})
                operation = self.path.rsplit("/", 1)[-1]
                identifier = self.path.split("/")[-2]
                fake.calls.append(("POST", identifier, operation))
                job = fake.jobs[identifier]
                if operation == "plan":
                    job.update(status="planned", plan_id="plan-1",
                               can_start=fake.authorization_reason is None)
                elif operation == "start":
                    job.update(status="running", next={"running": "complete"})
                return self._reply(202, {"id": identifier, "status": job["status"]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def bridge_for(tmp_path, fake, monkeypatch):
    (tmp_path / "tools/avatar-assets/avatar_assets").mkdir(parents=True)
    python = tmp_path / "tools/avatar-assets/.venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    submitted = []
    bridge = AvatarStudioBridge(tmp_path / "portal", tmp_path, submitted.append)
    bridge.port = fake.port
    monkeypatch.setattr(bridge, "ensure", lambda timeout=30: fake.port)
    monkeypatch.setattr(bridge, "_grant", lambda entry: None)
    return bridge, submitted


def adopt_stub(bridge, tmp_path):
    def fake_adopt(key, entry, payload):
        raw = bridge._request("GET", payload["result"]["model_url"])
        archive = tmp_path / "packed.zip"
        archive.write_bytes(b"ZIP" + raw[:8])
        bridge.submit(archive.read_bytes())
        with bridge.lock:
            bridge._registry[key].update(state="ready", asset_id="created-x", error=None)
            bridge._save()
    return fake_adopt


def test_create_plans_starts_and_adopts_through_the_studio_contract(tmp_path, monkeypatch):
    fake = FakeStudio(tmp_path)
    try:
        bridge, submitted = bridge_for(tmp_path, fake, monkeypatch)
        monkeypatch.setattr(bridge, "_adopt", adopt_stub(bridge, tmp_path))
        bridge.create("a turquoise axolotl explorer", request_id="b" * 32,
                      spend_approval="Approved: 35 credits for this generation")
        assert ("POST", "jobs", "a turquoise axolotl explorer") in fake.calls
        states = []
        for _ in range(8):
            summary = bridge.poll()
            states.append(summary["creations"][0]["state"])
            if summary["creations"][0]["state"] == "ready":
                break
            for future in list(bridge._executor._threads or []):
                pass
            import time
            time.sleep(0.05)
        assert "generating" in states and states[-1] == "ready"
        assert submitted and submitted[0].startswith(b"ZIP")
        reloaded = AvatarStudioBridge(tmp_path / "portal", tmp_path, submitted.append)
        assert reloaded.summary()["creations"][0]["state"] == "ready"
        reloaded.close()
        bridge.close()
    finally:
        fake.close()


def test_insufficient_allowance_surfaces_the_studio_reason_after_grant_failure(tmp_path, monkeypatch):
    fake = FakeStudio(tmp_path, authorization_reason="Existing canonical allowance is insufficient")
    try:
        bridge, _ = bridge_for(tmp_path, fake, monkeypatch)
        monkeypatch.setattr(bridge, "_grant",
                            lambda entry: (_ for _ in ()).throw(RuntimeError("meshy account empty")))
        bridge.create("a prompt", request_id="c" * 32, spend_approval="Approved")
        summary = bridge.poll()
        row = summary["creations"][0]
        assert row["state"] == "failed"
        assert "meshy account empty" in row["error"] and "insufficient" in row["error"]
        bridge.close()
    finally:
        fake.close()


def test_creation_requires_prompt_request_id_and_approval(tmp_path, monkeypatch):
    fake = FakeStudio(tmp_path)
    try:
        bridge, _ = bridge_for(tmp_path, fake, monkeypatch)
        with pytest.raises(ValueError, match="Describe the character"):
            bridge.create("", request_id="d" * 32, spend_approval="yes")
        with pytest.raises(ValueError, match="request id"):
            bridge.create("a prompt", request_id="short", spend_approval="yes")
        with pytest.raises(ValueError, match="approval"):
            bridge.create("a prompt", request_id="d" * 32, spend_approval="")
        bridge.close()
    finally:
        fake.close()


def test_library_listing_and_selection_are_bounded(tmp_path):
    folder = tmp_path / ".local/avatar-library"
    folder.mkdir(parents=True)
    (folder / "panda.zip").write_bytes(b"P" * 128)
    (folder / "panda.json").write_text(json.dumps({"display_name": "Panda"}))
    (folder / "ignored.txt").write_text("not a zip")
    rows = list_library(tmp_path)
    assert rows == [{"id": "panda", "display_name": "Panda", "bytes": 128}]
    assert read_library_zip(tmp_path, "panda") == b"P" * 128
    with pytest.raises(ValueError, match="not available"):
        read_library_zip(tmp_path, "missing")
    with pytest.raises(ValueError, match="listed id"):
        read_library_zip(tmp_path, "../escape")
