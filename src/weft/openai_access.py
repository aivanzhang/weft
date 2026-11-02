"""Project-key Responses transport and a crash-conservative portal spending ledger."""

from contextlib import contextmanager
import http.client
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import time

from weft.io import local_time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL = "gpt-5.4-mini-2026-03-17"
PRICING = {"model": MODEL, "checked": "2026-11-02", "input_usd_per_million": .75,
           "output_usd_per_million": 4.50, "context_tokens": 400_000,
           "source": "https://developers.openai.com/api/docs/models/gpt-5.4-mini"}
MAX_OUTPUT_TOKENS = 3000
# Reserve the full context ceiling, not a tokenizer guess; unused cost releases only on known usage.
RESERVATION_MICROUSD = 313_500
MAX_RESPONSE_BYTES = 128_000


def key_path(project=PROJECT_ROOT):
    return Path(project) / ".local/credentials/openai-api-key"


def _key_file(path, *, read=False):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("OpenAI credential path cannot contain symlinks")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or not 10 <= info.st_size <= 4096):
            raise ValueError("OpenAI key must be an owner-only, bounded regular file")
        return stream.read(4097).strip().decode("ascii") if read else None


def read_key(project=PROJECT_ROOT):
    key = os.environ.get("OPENAI_API_KEY") or _key_file(key_path(project), read=True)
    if not 10 <= len(key) <= 4096 or not all(33 <= ord(c) <= 126 for c in key):
        raise ValueError("Invalid project OpenAI key")
    return key


def credential_status(project=PROJECT_ROOT):
    try:
        if os.environ.get("OPENAI_API_KEY"):
            read_key(project)
            source = "environment"
        else:
            _key_file(key_path(project))
            source = "project_file"
        return {"configured": True, "source": source, "live_verified": False}
    except (OSError, ValueError, UnicodeError):
        return {"configured": False, "source": None, "live_verified": False}


class OpenAIBudget:
    """One ledger per project, independent of portal output folders and Meta's allowance."""

    def __init__(self, project=PROJECT_ROOT):
        self.folder = Path(project) / ".local/portal-openai"
        if any(p.is_symlink() for p in (self.folder, *self.folder.parents)):
            raise ValueError("Budget path cannot contain symlinks")
        self.folder.mkdir(parents=True, mode=0o700, exist_ok=True)
        if self.folder.stat().st_uid != os.getuid() or stat.S_IMODE(self.folder.stat().st_mode) != 0o700:
            raise ValueError("OpenAI budget directory must be owner-only")
        self.path = self.folder / "ledger.sqlite3"
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        info = os.fstat(fd)
        os.close(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("OpenAI budget ledger must be owner-only")
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS budget(id INTEGER PRIMARY KEY CHECK(id=1), total INTEGER NOT NULL, note TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY, request_sha256 TEXT NOT NULL,
                    reserved INTEGER NOT NULL, charged INTEGER NOT NULL, state TEXT NOT NULL,
                    created TEXT NOT NULL, response_id TEXT, usage TEXT);
                CREATE TABLE IF NOT EXISTS responses(request_sha256 TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                    call_id TEXT NOT NULL REFERENCES calls(id), response TEXT NOT NULL,
                    PRIMARY KEY(request_sha256,source_sha256));
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=3)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def configure(self, total_usd, *, note):
        if type(total_usd) not in (int, float) or not math.isfinite(total_usd) or not 0 < total_usd <= 100:
            raise ValueError("Configure a positive total OpenAI allowance no greater than $100")
        total = math.floor(total_usd * 1e6)
        if not isinstance(note, str) or not 1 <= len(note) <= 500:
            raise ValueError("Record the administrator's spending instruction")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            spent = db.execute("SELECT COALESCE(SUM(charged),0) FROM calls").fetchone()[0]
            if total < spent:
                raise ValueError("Total cannot be below existing or uncertain charges")
            db.execute("INSERT INTO budget VALUES(1,?,?) ON CONFLICT(id) DO UPDATE SET total=excluded.total,note=excluded.note", (total, note))
        return self.status()

    def status(self):
        with self.connect() as db:
            row = db.execute("SELECT total FROM budget WHERE id=1").fetchone()
            spent, calls, unknown = db.execute("SELECT COALESCE(SUM(charged),0),COUNT(*),COALESCE(SUM(state!='settled'),0) FROM calls").fetchone()
        total = row[0] if row else 0
        return {"configured": row is not None, "total_usd": total / 1e6, "committed_usd": spent / 1e6,
                "remaining_usd": max(0, total - spent) / 1e6, "calls": calls, "unsettled_calls": unknown,
                "reservation_usd": RESERVATION_MICROUSD / 1e6, "provider_hard_cap": False}

    def reserve(self, identity, request_sha256):
        import re
        if not re.fullmatch(r"[a-f0-9]{32}", identity) or not re.fullmatch(r"[a-f0-9]{64}", request_sha256):
            raise ValueError("Invalid API call identity")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            total = db.execute("SELECT total FROM budget WHERE id=1").fetchone()
            charged, count = db.execute("SELECT COALESCE(SUM(charged),0),COUNT(*) FROM calls").fetchone()
            overage = db.execute("SELECT 1 FROM calls WHERE state='overage' LIMIT 1").fetchone()
            if total is None or overage or charged + RESERVATION_MICROUSD > total[0] or count >= 10_000:
                raise ValueError("OpenAI allowance is unconfigured or exhausted; no request sent")
            if db.execute("SELECT 1 FROM calls WHERE id=?", (identity,)).fetchone():
                raise ValueError("This job already reserved its API call; automatic retries are disabled")
            db.execute("INSERT INTO calls VALUES(?,?,?,?,?,?,NULL,NULL)",
                       (identity, request_sha256, RESERVATION_MICROUSD, RESERVATION_MICROUSD, "uncertain", local_time()))

    def settle(self, identity, response):
        usage = response.get("usage", {})
        counts = [usage.get(k) for k in ("input_tokens", "output_tokens")]
        if any(type(n) is not int or n < 0 for n in counts) or not isinstance(response.get("id"), str):
            raise ValueError("Incomplete OpenAI usage; full reservation retained")
        charged = math.ceil(counts[0] * .75 + counts[1] * 4.5)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM calls WHERE id=?", (identity,)).fetchone()
            if row is None or row["state"] != "uncertain":
                raise ValueError("OpenAI reservation is missing or already settled")
            # Overages remain visible and block later work rather than being clipped away.
            db.execute("UPDATE calls SET charged=?,state=?,response_id=?,usage=? WHERE id=?",
                       (charged, "settled" if charged <= row["reserved"] else "overage",
                        response["id"], json.dumps(usage), identity))
        if charged > row["reserved"]:
            raise ValueError("OpenAI usage exceeded reservation; further calls stopped")
        return {"cost_usd_upper_estimate": charged / 1e6, "usage": usage, "response_id": response["id"]}

    def remember_response(self, identity, source_sha256, response):
        import re
        raw = json.dumps(response, allow_nan=False, sort_keys=True)
        if not re.fullmatch(r"[a-f0-9]{64}", source_sha256) or len(raw.encode()) > MAX_RESPONSE_BYTES:
            raise ValueError("Invalid cached scene-analysis identity")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            call = db.execute("SELECT * FROM calls WHERE id=?", (identity,)).fetchone()
            if (call is None or call["state"] != "settled" or response.get("status") != "completed"
                    or response.get("model") != MODEL or response.get("id") != call["response_id"]
                    or response.get("usage") != json.loads(call["usage"])):
                raise ValueError("Only a completed, settled response can be reused")
            db.execute("INSERT OR IGNORE INTO responses VALUES(?,?,?,?)",
                       (call["request_sha256"], source_sha256, identity, raw))

    def cached_response(self, request_sha256, source_sha256):
        with self.connect() as db:
            row = db.execute("SELECT response,call_id,charged FROM responses JOIN calls ON calls.id=responses.call_id "
                "WHERE responses.request_sha256=? AND source_sha256=? AND calls.state='settled'",
                (request_sha256, source_sha256)).fetchone()
        if row is None:
            return None
        return json.loads(row["response"]), {"cost_usd_upper_estimate": 0., "usage": None,
            "reused_call_id": row["call_id"], "original_cost_usd_upper_estimate": row["charged"] / 1e6}


def post_response(payload, key, *, timeout=60):
    """Fixed TLS endpoint, no account-login flow, tools, redirects, proxies or retries."""
    raw = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    if len(raw) > 3_000_000:
        raise ValueError("OpenAI request exceeds 3 MB")
    deadline = time.monotonic() + timeout
    conn = http.client.HTTPSConnection("api.openai.com", timeout=timeout)
    try:
        conn.connect()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("OpenAI connection deadline exceeded")
        conn.sock.settimeout(remaining)
        conn.request("POST", "/v1/responses", raw, {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        if conn.sock:
            conn.sock.settimeout(max(.01, deadline - time.monotonic()))
        response = conn.getresponse()
        if response.status != 200:
            raise ValueError(f"OpenAI returned HTTP {response.status}; no automatic retry, reservation retained")
        result = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("OpenAI response deadline exceeded")
            if conn.sock:
                conn.sock.settimeout(remaining)
            chunk = response.read1(min(16384, MAX_RESPONSE_BYTES + 1 - len(result)))
            if not chunk:
                break
            result.extend(chunk)
            if len(result) > MAX_RESPONSE_BYTES:
                raise ValueError("OpenAI response exceeds 128 KB")
        return json.loads(result)
    except (OSError, http.client.HTTPException):
        raise ValueError("OpenAI transport failed; reservation retained, no automatic retry") from None
    finally:
        conn.close()
