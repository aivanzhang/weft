"""One supervised portal job; no agent CLI or browser-login credentials."""

import resource
import os
from pathlib import Path
import sys
import threading
import time


def _limit(kind, requested):
    # A supervisor can impose a tighter ceiling; workers must never widen it.
    inherited = resource.getrlimit(kind)
    limit = min([requested, *(value for value in inherited if value != resource.RLIM_INFINITY)])
    resource.setrlimit(kind, (limit, limit))


def main():
    repairs = len(sys.argv) == 4 and sys.argv[3] == "repairs"
    cpu_limit = 65 if repairs else 360
    _limit(resource.RLIMIT_CPU, cpu_limit)
    _limit(resource.RLIMIT_FSIZE, 512 * 1024**2)
    from weft.lab_worker import atomic_json
    root = Path(sys.argv[1])
    review = len(sys.argv) == 4 and sys.argv[3] == "review"
    checks = len(sys.argv) == 4 and sys.argv[3] == "checks"
    supports = len(sys.argv) == 4 and sys.argv[3] == "supports"
    work = root / "muse/repairs" if repairs else root / "muse/checks" if checks else root / "muse" if review else root
    stopped = threading.Event()
    started = time.monotonic()

    def rss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)

    def monitor():
        while not stopped.wait(.25):
            if rss() > 1536 * 1024**2:
                atomic_json(work / "status.json", {"status": "failed", "stage": "resource_limit",
                    "error": "Worker exceeded the sampled 1.5 GiB memory allowance", "events": []})
                os._exit(74)

    threading.Thread(target=monitor, name="weft-resource-limit", daemon=True).start()
    if repairs:
        from weft.portal_repair_jobs import execute_repair_job
        result = execute_repair_job(root)
    elif supports:
        from weft.portal_supports import execute_support_job
        result = execute_support_job(root)
    elif checks:
        from weft.portal_checks import execute_checks
        result = execute_checks(root)
    elif review:
        from weft.portal_review import execute_review
        result = execute_review(root, project=Path(sys.argv[2]))
    else:
        from weft.portal_pipeline import execute_job
        result = execute_job(root, project=Path(sys.argv[2]))
    stopped.set()
    atomic_json(work / "worker-receipt.json", {"exit_code": result, "seconds": time.monotonic() - started,
        "peak_rss_bytes": rss(), "rss_limit_kind": "250ms sampled peak, not a kernel allocation limit"})
    return result


if __name__ == "__main__":
    raise SystemExit(main())
