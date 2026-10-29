"""Project-local Meta credentials and crash-conservative application spend accounting."""

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import fcntl
import getpass
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import warnings
from uuid import uuid4

from weft.io import local_time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_LEDGER_BYTES = 5_000_000
MAX_RESERVATIONS = 10_000


def _root(project_root=None):
    root = Path(project_root or PROJECT_ROOT).resolve()
    if not root.is_dir():
        raise ValueError("The project root must already exist")
    return root


def _owned_file(fd):
    value = os.fstat(fd)
    if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid() or stat.S_IMODE(value.st_mode) != 0o600:
        raise ValueError("Meta private files must be owned by this user with mode 0600")


def _private_dir(root, name):
    local = root / ".local"
    if local.is_symlink():
        raise ValueError("Project .local directory cannot be a symlink")
    local.mkdir(mode=0o700, exist_ok=True)
    if local.stat().st_uid != os.getuid() or stat.S_IMODE(local.stat().st_mode) & 0o022:
        raise ValueError("Project .local must be user-owned and not writable by other users")
    directory = local / name
    if directory.is_symlink():
        raise ValueError("Meta private directory cannot be a symlink")
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.stat().st_uid != os.getuid():
        raise ValueError("Meta private directory must be owned by this user")
    directory.chmod(0o700)
    return directory


def _fsync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_private(path, maximum):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        _owned_file(fd)
        if os.fstat(fd).st_size > maximum:
            raise ValueError("Meta private file exceeds its size bound")
        with os.fdopen(fd, "r", closefd=False) as stream:
            return stream.read(maximum + 1)
    finally:
        os.close(fd)


def _key(value):
    if not isinstance(value, str) or not 8 <= len(value) <= 4096 or any(not 33 <= ord(char) <= 126 for char in value):
        raise ValueError("Meta key must be a nonempty printable token without whitespace")
    return value


def resolve_key(project_root=None):
    """Return a key only to the caller; never include it in status or artifacts."""
    environment = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
    if environment:
        return _key(environment)
    root = _root(project_root)
    local, directory = root / ".local", root / ".local" / "credentials"
    if local.is_symlink() or directory.is_symlink():
        raise ValueError("Meta credential directories cannot be symlinks")
    path = directory / "meta-api-key"
    if not path.exists() and not path.is_symlink():
        raise ValueError("Meta key absent; configure a project key or META_API_KEY / MODEL_API_KEY")
    if directory.stat().st_uid != os.getuid() or stat.S_IMODE(directory.stat().st_mode) != 0o700:
        raise ValueError("Meta credential directory must have mode 0700")
    return _key(_read_private(path, 4096).rstrip("\n"))


def _money(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("Budget and cost values must be finite nonnegative USD numbers")
    try:
        return int((Decimal(str(value)) * 1_000_000).to_integral_value(rounding=ROUND_CEILING))
    except (InvalidOperation, OverflowError):
        raise ValueError("Invalid USD amount") from None


def _validate_ledger(value):
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or type(value.get("total_budget_microusd")) is not int or value["total_budget_microusd"] <= 0
            or not isinstance(value.get("reservations"), dict) or len(value["reservations"]) > MAX_RESERVATIONS):
        raise ValueError("Invalid Meta budget ledger; no request authorized")
    for reservation_id, entry in value["reservations"].items():
        if (not re.fullmatch(r"[a-f0-9]{32}", reservation_id) or not isinstance(entry, dict)
                or entry.get("status") not in ("reserved", "settled", "unknown")
                or type(entry.get("reserved_microusd")) is not int or entry["reserved_microusd"] <= 0
                or type(entry.get("charged_microusd")) is not int or entry["charged_microusd"] < 0):
            raise ValueError("Invalid Meta reservation; no request authorized")
        if entry["status"] in ("reserved", "unknown") and entry["charged_microusd"] < entry["reserved_microusd"]:
            raise ValueError("Unsettled Meta reservations cannot release estimated charges")
    return value


def _write_ledger(directory, ledger):
    data = json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if len(data.encode()) > MAX_LEDGER_BYTES:
        raise ValueError("Meta ledger reached its bounded size; request not authorized")
    temporary = directory / f"ledger-{uuid4().hex}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(fd)
        os.replace(temporary, directory / "ledger.json")
        _fsync_directory(directory)
    finally:
        os.close(fd)


def _load_ledger(directory):
    path = directory / "ledger.json"
    if not path.exists() and not path.is_symlink():
        raise ValueError("No total Meta budget configured; explicit total budget is required")
    return _validate_ledger(json.loads(_read_private(path, MAX_LEDGER_BYTES)))


@contextmanager
def _locked(project_root):
    directory = _private_dir(_root(project_root), "meta-budget")
    fd = os.open(directory / "ledger.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        _owned_file(fd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield directory
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _summary(ledger):
    committed = sum(item["charged_microusd"] for item in ledger["reservations"].values())
    total = ledger["total_budget_microusd"]
    return {"total_budget_usd": total / 1e6, "committed_usd": committed / 1e6,
            "remaining_usd": max(0, total - committed) / 1e6,
            "budget_exceeded": committed > total, "reservation_count": len(ledger["reservations"]),
            "unsettled_count": sum(item["status"] != "settled" for item in ledger["reservations"].values()),
            "provider_hard_cap": False, "scope": "Application-side estimated budget; unknown charges retain their reservation"}


def configure_access(project_root=None, *, total_budget_usd, key=None, prompt=False):
    """Set an explicit lifetime total without discarding prior reservations or charges."""
    root = _root(project_root)
    total = _money(total_budget_usd, positive=True)
    if key is not None and prompt:
        raise ValueError("Choose a supplied key or hidden prompt, not both")
    if prompt:
        if not sys.stdin.isatty() or not sys.stderr.isatty():
            raise ValueError("Hidden key entry requires an interactive terminal; no echoed fallback is allowed")
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("Meta API key (hidden; stored only in this project): ")
    if key is not None:
        key = _key(key)
    with _locked(root) as directory:
        path = directory / "ledger.json"
        ledger = _load_ledger(directory) if path.exists() or path.is_symlink() else {
            "schema_version": 1, "created_at": local_time(), "total_budget_microusd": total, "reservations": {}}
        committed = sum(entry["charged_microusd"] for entry in ledger["reservations"].values())
        if total < committed:
            raise ValueError("Total budget cannot be lower than already committed charges")
        if key is not None:
            credentials = _private_dir(root, "credentials")
            fd = os.open(credentials / "meta-api-key", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                with os.fdopen(fd, "w", closefd=False) as stream:
                    stream.write(key + "\n")
                    stream.flush()
                    os.fsync(fd)
                _fsync_directory(credentials)
            finally:
                os.close(fd)
        ledger.update(total_budget_microusd=total, configured_at=local_time())
        _write_ledger(directory, ledger)
        return _summary(ledger)


def access_status(project_root=None):
    """Inspect configuration without returning credentials or contacting Meta."""
    root = _root(project_root)
    try:
        environment = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
        if environment:
            _key(environment)
        else:
            local, directory = root / ".local", root / ".local" / "credentials"
            if local.is_symlink() or directory.is_symlink():
                raise ValueError("Meta credential directories cannot be symlinks")
            info = directory.stat()
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise ValueError("Unsafe Meta credential directory permissions")
            info = (directory / "meta-api-key").lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or not 8 <= info.st_size <= 4097):
                raise ValueError("Unsafe Meta key file metadata")
        credential_present, credential_error = True, None
    except (ValueError, OSError) as error:
        credential_present, credential_error = False, type(error).__name__
    result = {"credential_present": credential_present, "credential_error": credential_error,
              "credential_source": ("environment" if os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY") else "project_file") if credential_present else None,
              "live_access_verified": False, "budget_configured": False, "provider_hard_cap": False}
    path = root / ".local" / "meta-budget" / "ledger.json"
    if path.exists() or path.is_symlink():
        with _locked(root) as directory:
            result.update(_summary(_load_ledger(directory)), budget_configured=True)
    return result


def reserve_call(project_root=None, *, estimate_usd, request_sha256):
    """Durably reserve before transport; a process crash leaves the reservation charged."""
    estimate = _money(estimate_usd, positive=True)
    if not isinstance(request_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", request_sha256):
        raise ValueError("A request SHA-256 identity is required")
    with _locked(project_root) as directory:
        ledger = _load_ledger(directory)
        committed = sum(item["charged_microusd"] for item in ledger["reservations"].values())
        if len(ledger["reservations"]) >= MAX_RESERVATIONS:
            raise ValueError("Meta reservation count reached its bound")
        if committed + estimate > ledger["total_budget_microusd"]:
            raise ValueError("Insufficient remaining total Meta budget; no request made")
        reservation_id = uuid4().hex
        ledger["reservations"][reservation_id] = {"status": "reserved", "created_at": local_time(),
            "request_sha256": request_sha256, "reserved_microusd": estimate, "charged_microusd": estimate,
            "actual_cost_usd": None, "outcome": "request_may_have_been_sent"}
        _write_ledger(directory, ledger)
        return {"reservation_id": reservation_id, "reserved_usd": estimate / 1e6, **_summary(ledger)}


def settle_call(project_root=None, *, reservation_id, actual_cost_usd=None, outcome):
    """Release an estimate only when complete usage is known; ambiguity stays charged."""
    actual = None if actual_cost_usd is None else _money(actual_cost_usd)
    if outcome not in ("proposed", "failed", "completed"):
        raise ValueError("Unknown Meta call outcome")
    with _locked(project_root) as directory:
        ledger = _load_ledger(directory)
        if reservation_id not in ledger["reservations"]:
            raise ValueError("Meta reservation not found")
        entry = ledger["reservations"][reservation_id]
        if entry["status"] != "reserved":
            raise ValueError("Meta reservation was already finalized")
        entry.update(status="unknown" if actual is None else "settled", settled_at=local_time(),
                     charged_microusd=entry["reserved_microusd"] if actual is None else actual,
                     actual_cost_usd=None if actual is None else actual / 1e6, outcome=outcome)
        _write_ledger(directory, ledger)
        return {"reservation_id": reservation_id, **_summary(ledger)}
