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
    moving = len(sys.argv) == 4 and sys.argv[3] == "moving"
    repairs = len(sys.argv) == 4 and sys.argv[3] == "repairs"
    equipment = len(sys.argv) == 4 and sys.argv[3] == "equipment"
    perception = len(sys.argv) == 4 and sys.argv[3] == "perception"
    tasks = len(sys.argv) == 4 and sys.argv[3] == "tasks"
    saved_body = len(sys.argv) == 4 and sys.argv[3] == "saved-body"
    cpu_limit = 185 if moving or tasks else 125 if saved_body else 110 if equipment else 65 if repairs else 360
    _limit(resource.RLIMIT_CPU, cpu_limit)
    _limit(resource.RLIMIT_FSIZE, 512 * 1024**2)
    from weft.lab_worker import atomic_json
    root = Path(sys.argv[1])
    review = len(sys.argv) == 4 and sys.argv[3] == "review"
    checks = len(sys.argv) == 4 and sys.argv[3] == "checks"
    supports = len(sys.argv) == 4 and sys.argv[3] == "supports"
    work = root / "moving-equipment" if moving else root / "perception" if perception else root / "equipment" if equipment else root / "muse/repairs" if repairs else root / "muse/checks" if checks else root / "muse" if review else root
    receipt_name = ("preparation-worker-receipt.json" if (work / "request.json").is_file() else "preflight-worker-receipt.json") if perception else "worker-receipt.json"
    if moving:
        from weft.portal_moving_equipment import read_status, storage_limit
        storage_limit(root)
        receipt_name = read_status(root)["phase"] + "-worker-receipt.json"
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
    if saved_body:
        from weft.portal_saved_body import execute_job
        result = execute_job(root, project=Path(sys.argv[2]))
    elif tasks:
        from weft.portal_task_jobs import execute_job
        result = execute_job(root, project=Path(sys.argv[2]))
    elif moving:
        from weft.portal_moving_equipment import execute_job
        result = execute_job(root, project=Path(sys.argv[2]))
    elif perception:
        from weft.portal_perception_jobs import execute_perception_job
        result = execute_perception_job(root, project=Path(sys.argv[2]))
    elif equipment:
        from weft.portal_equipment_candidates import execute_equipment_job
        result = execute_equipment_job(root, project=Path(sys.argv[2]))
    elif repairs:
        from weft.portal_repair_jobs import execute_repair_job
        result = execute_repair_job(root, project=Path(sys.argv[2]))
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
    atomic_json(work / receipt_name, {"exit_code": result, "seconds": time.monotonic() - started,
        "peak_rss_bytes": rss(), "rss_limit_kind": "250ms sampled peak, not a kernel allocation limit"})
    return result


if __name__ == "__main__":
    raise SystemExit(main())
