"""One supervised portal job; no agent CLI or browser-login credentials."""

import resource
import os
from pathlib import Path
import sys
import threading
import time


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (360, 360))
    resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024**2, 512 * 1024**2))
    from weft.portal_pipeline import execute_job
    from weft.lab_worker import atomic_json
    root = Path(sys.argv[1])
    review = len(sys.argv) == 4 and sys.argv[3] == "review"
    work = root / "muse" if review else root
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
    if review:
        from weft.portal_review import execute_review
        result = execute_review(root, project=Path(sys.argv[2]))
    else:
        result = execute_job(root, project=Path(sys.argv[2]))
    stopped.set()
    atomic_json(work / "worker-receipt.json", {"exit_code": result, "seconds": time.monotonic() - started,
        "peak_rss_bytes": rss(), "rss_limit_kind": "250ms sampled peak, not a kernel allocation limit"})
    return result


if __name__ == "__main__":
    raise SystemExit(main())
