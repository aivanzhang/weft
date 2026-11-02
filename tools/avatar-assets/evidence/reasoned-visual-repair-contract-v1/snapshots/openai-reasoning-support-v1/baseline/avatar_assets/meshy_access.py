"""Private Meshy transport and a crash-conservative, project-wide credit journal."""

from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

from avatar_assets.safety import require


PROJECT = Path(__file__).resolve().parents[3]
API = "https://api.meshy.ai"
MAX_RESPONSE = 2 * 1024 * 1024


def now():
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def private_directory(path):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Private path cannot contain symlinks")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700,
            "Private directory requires current-user ownership and mode 0700")
    return path


def load_key(project=PROJECT):
    path = Path(project) / ".local/credentials/meshy-api-key"
    require(not any(p.is_symlink() for p in path.parents), "Credential parents cannot be symlinks")
    require(stat.S_IMODE(path.parent.stat().st_mode) == 0o700, "Credential directory requires mode 0700")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and stat.S_IMODE(info.st_mode) == 0o600 and 1 <= info.st_size <= 1024,
                "Meshy key must be an owner-only regular file of 1–1024 bytes")
        key = os.read(fd, 1025).decode("ascii").strip()
    finally:
        os.close(fd)
    require(re.fullmatch(r"msy_[A-Za-z0-9_-]{8,1000}", key), "Invalid Meshy key format")
    return key


@contextmanager
def deadline(seconds):
    # Socket inactivity alone does not bound a slow trickle of response bytes.
    require(not signal.getitimer(signal.ITIMER_REAL)[0], "Cannot replace an active process deadline")
    def expired(*_):
        raise TimeoutError("Meshy request deadline exceeded")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class MeshyError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def api_path(path):
    return bool(re.fullmatch(r"/openapi/(?:v1/(?:balance|(?:rigging|animations|image-to-3d|"
                            r"text-to-motion)(?:/[A-Za-z0-9_-]{1,128})?)|"
                            r"v2/text-to-3d(?:/[A-Za-z0-9_-]{1,128})?)", path))


class Client:
    def __init__(self, project=PROJECT):
        self._key = load_key(project)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, method, path, payload=None):
        require(api_path(path) and method in ("GET", "POST"), "Unapproved Meshy endpoint or method")
        require((method == "POST") == (payload is not None), "Invalid request payload")
        raw = canonical(payload).encode() if payload is not None else None
        maximum = 28 * 1024 * 1024 if path == "/openapi/v1/image-to-3d" else 16384
        require(raw is None or len(raw) <= maximum, "Request exceeds payload bound")
        request = urllib.request.Request(API + path, data=raw, method=method,
                  headers={"Authorization": "Bearer " + self._key, "Accept": "application/json",
                           "Content-Type": "application/json"})
        try:
            with deadline(45), self._opener.open(request, timeout=15) as response:
                body = response.read(MAX_RESPONSE + 1)
                require(len(body) <= MAX_RESPONSE, "Meshy response exceeds bound")
                result = json.loads(body.decode().replace(self._key, "[redacted]"))
                require(isinstance(result, (dict, list)), "Invalid Meshy response")
                return result
        except urllib.error.HTTPError as error:
            # Provider error bodies can echo request data; expose only the HTTP status.
            raise MeshyError(error.code, f"Meshy HTTP {error.code}; no automatic retry") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            suffix = "; submission may be unknown" if method == "POST" else ""
            raise MeshyError(None, f"Meshy transport/response failure ({type(error).__name__}){suffix}") from None

    def balance(self):
        data = self.request("GET", "/openapi/v1/balance")
        require(isinstance(data, dict) and type(data.get("balance")) is int and data["balance"] >= 0,
                "Invalid Meshy balance")
        return {name: data[name] for name in ("balance", "monthly_credits", "permanent_credits", "monthly_resets_at")
                if name in data and (data[name] is None or type(data[name]) is int)}

    def download(self, url, path):
        parts = urllib.parse.urlsplit(url)
        require(parts.scheme == "https" and parts.hostname in ("assets.meshy.ai", "cdn.meshy.ai")
                and parts.port in (None, 443) and not parts.username and not parts.password and not parts.fragment,
                "Asset URL must use an approved Meshy HTTPS asset host")
        path = Path(path)
        require(not path.exists() and not path.is_symlink(), "Asset destination already exists")
        total, checksum = 0, hashlib.sha256()
        started = time.monotonic()
        # Asset requests never carry the API key and never follow redirects.
        with deadline(120), self._opener.open(url, timeout=15) as response, path.open("xb") as handle:
            expected = response.headers.get("Content-Length")
            require(expected is None or 0 < int(expected) <= 64 * 1024 * 1024, "Asset exceeds 64 MiB")
            while chunk := response.read(256 * 1024):
                total += len(chunk)
                require(total <= 64 * 1024 * 1024, "Asset exceeds 64 MiB")
                handle.write(chunk)
                checksum.update(chunk)
            require(total > 0 and (expected is None or total == int(expected)), "Incomplete asset download")
            handle.flush()
            os.fsync(handle.fileno())
        return {"bytes": total, "sha256": checksum.hexdigest(), "elapsed_seconds": round(time.monotonic() - started, 3)}


class Ledger:
    def __init__(self, project=PROJECT):
        folder = private_directory(Path(project) / ".local/meshy")
        path = folder / "ledger.sqlite3"
        if not path.exists():
            try:
                os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600))
            except FileExistsError:
                pass
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o600,
                "Unsafe Meshy ledger permissions")
        self.db = sqlite3.connect(path, timeout=5)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY CHECK(id=1), credit_limit INTEGER, created TEXT);
            CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, output TEXT UNIQUE, plan_hash TEXT, credit_limit INTEGER);
            CREATE TABLE IF NOT EXISTS calls (job TEXT, step TEXT, endpoint TEXT, request TEXT, credits INTEGER,
                state TEXT, task_id TEXT, response TEXT, created TEXT, updated TEXT, PRIMARY KEY(job,step));
            CREATE TABLE IF NOT EXISTS budget_grants (job TEXT PRIMARY KEY, credits INTEGER, plan_hash TEXT,
                approval TEXT, created TEXT);
        """)

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def configure(self, limit):
        require(type(limit) is int and 1 <= limit <= 10000, "Use a finite 1–10000 credit allowance")
        with self.transaction():
            row = self.db.execute("SELECT credit_limit FROM budget WHERE id=1").fetchone()
            require(row is None or row[0] == limit, "Existing project budget is immutable; do not reset or silently expand it")
            if row is None:
                self.db.execute("INSERT INTO budget VALUES(1,?,?)", (limit, now()))
        return self.summary()

    def bind_job(self, plan, output):
        value = (plan["id"], str(Path(output).resolve()), digest(plan), plan["max_credits"])
        with self.transaction():
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (plan["id"],)).fetchone()
            require(row is None or tuple(row) == value, "Job plan or location changed after binding")
            if row is None:
                self.db.execute("INSERT INTO jobs VALUES(?,?,?,?)", value)

    def call(self, job, step):
        row = self.db.execute("SELECT * FROM calls WHERE job=? AND step=?", (job, step)).fetchone()
        return dict(row) if row else None

    def authorize_job(self, plan, output, approval):
        """Append explicit user approval for one immutable job; never rewrite the base budget."""
        require(isinstance(approval, dict) and approval.get('decision') == 'approved'
                and approval.get('job_id') == plan['id'] and approval.get('plan_digest') == digest(plan),
                'Approval must identify the exact prepared job')
        credits = approval.get('additional_credits')
        require(type(credits) is int and 1 <= credits <= 500 and credits == plan['max_credits'],
                'Additional allowance must match the finite job ceiling')
        require(isinstance(approval.get('user_reply'), str) and 0 < len(approval['user_reply']) <= 4000
                and len(canonical(approval)) <= 8192, 'Preserve the explicit user approval')
        self.bind_job(plan, output)
        value = (plan['id'], credits, digest(plan), canonical(approval))
        with self.transaction():
            require(self.db.execute('SELECT 1 FROM budget WHERE id=1').fetchone(), 'Configure the base budget first')
            row = self.db.execute('SELECT job,credits,plan_hash,approval FROM budget_grants WHERE job=?', (plan['id'],)).fetchone()
            require(row is None or tuple(row) == value, 'Existing job allowance is immutable')
            if row is None:
                require(not self.db.execute('SELECT 1 FROM calls WHERE job=?', (plan['id'],)).fetchone(),
                        'Record new-job approval before any paid submission')
                self.db.execute('INSERT INTO budget_grants VALUES(?,?,?,?,?)', (*value, now()))
        return self.summary()

    def reserve(self, job, step, endpoint, payload, credits):
        require(type(credits) is int and credits > 0, "Invalid credit reservation")
        with self.transaction():
            require(self.call(job, step) is None, "Stage already reserved; never resubmit an uncertain request")
            budget = self.db.execute("SELECT credit_limit FROM budget WHERE id=1").fetchone()
            bound = self.db.execute("SELECT credit_limit FROM jobs WHERE id=?", (job,)).fetchone()
            require(self.summary()['pricing_overruns'] == 0,
                    'Provider charge exceeded its estimate; reconcile pricing before new paid work')
            usage = dict(self.db.execute('SELECT job,SUM(credits) FROM calls GROUP BY job').fetchall())
            grants = dict(self.db.execute('SELECT job,credits FROM budget_grants').fetchall())
            used = usage.get(job, 0)
            usage[job] = used + credits
            # A targeted grant cannot subsidize another job or reopen past charges.
            base_usage = sum(max(0, amount-grants.get(name, 0)) for name, amount in usage.items())
            require(budget is not None and bound is not None, "Configure a project budget and bind the plan first")
            require(base_usage <= budget[0] and used + credits <= bound[0], "Meshy credit allowance exhausted")
            self.db.execute("INSERT INTO calls VALUES(?,?,?,?,?,'submitting',NULL,NULL,?,?)",
                            (job, step, endpoint, canonical(payload), credits, now(), now()))

    def update(self, job, step, state, task_id=None, response=None):
        with self.transaction():
            row = self.call(job, step)
            require(row is not None, "Unreserved Meshy request")
            require(task_id is None or row["task_id"] in (None, task_id), "Task identity changed")
            encoded = canonical(response) if response is not None else row["response"]
            reported = response.get('consumed_credits') if isinstance(response, dict) else None
            require(reported is None or type(reported) is int and reported >= 0, 'Invalid provider credit accounting')
            # Unexpected charges increase the floor; later refunds never reopen it.
            floor = max(row['credits'], reported or 0)
            self.db.execute("UPDATE calls SET state=?,task_id=?,response=?,credits=?,updated=? WHERE job=? AND step=?",
                            (state, task_id or row["task_id"], encoded, floor, now(), job, step))

    def summary(self):
        row = self.db.execute("SELECT credit_limit FROM budget WHERE id=1").fetchone()
        grants = self.db.execute('SELECT job,credits FROM budget_grants ORDER BY job').fetchall()
        additional = sum(item['credits'] for item in grants)
        calls = self.db.execute("SELECT credits,state,response,request FROM calls").fetchall()
        confirmed = [json.loads(x["response"]).get("consumed_credits") for x in calls
                     if x["response"] and x["state"] in ("SUCCEEDED", "FAILED", "CANCELED")]
        return {"project_credit_limit": row[0]+additional if row else None,
                "base_project_credit_limit": row[0] if row else None,
                "approved_additional_credits": additional,
                "job_scoped_allowances": {item['job']:item['credits'] for item in grants},
                "reserved_credits": sum(x["credits"] for x in calls), "paid_submissions": len(calls),
                "unknown_submissions": sum(x["state"] in ("submitting", "uncertain") for x in calls),
                "provider_reported_credits": sum(x for x in confirmed if type(x) is int),
                "settled_tasks": sum(type(x) is int for x in confirmed),
                "pricing_overruns": sum(x['credits'] > json.loads(x['request']).get('estimated_credits', x['credits']) for x in calls),
                "refunds_reopen_allowance": False}
