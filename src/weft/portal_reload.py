"""Local-development restarts defer to jobs and in-memory worlds."""
import signal
import threading
import time

from weft.portal import PortalServer

RELOAD_EXIT = 75


class ReloadablePortalServer(PortalServer):
    def __init__(self, address, portal):
        super().__init__(address, portal)
        self.reload_pending = False
        self.next_reload_check = 0
        self.previous_signal = signal.signal(signal.SIGUSR1, self.request_reload)

    def request_reload(self, *_):
        if not self.reload_pending:
            print('Backend edit detected; restarting when idle. Loaded live worlds require a manual restart when finished.', flush=True)
        self.reload_pending = True

    def service_actions(self):
        if not self.reload_pending or time.monotonic() < self.next_reload_check:
            return
        self.next_reload_check = time.monotonic() + 1
        portal = self.portal
        # No accepted request may enqueue work between the idle check and shutdown.
        acquired = 0
        try:
            for _ in range(8):
                if not self.slots.acquire(blocking=False):
                    return
                acquired += 1
            with portal.lock:
                if portal.active or portal.labs:
                    return
                # Extra threads include demo orchestration, avatar jobs and exports.
                allowed = {threading.current_thread(), portal.thread}
                manager = getattr(portal, 'avatar_manager', None)
                if manager is not None:
                    with manager._lock:
                        if manager._future is not None and not manager._future.done():
                            return
                    # This single-worker pool stays alive after its tracked future completes.
                    allowed.update(t for t in threading.enumerate() if t.name.startswith('weft-avatar_'))
                if any(t.is_alive() and t not in allowed for t in threading.enumerate()):
                    return
                if portal._pending_jobs():
                    return
                for identity in portal.jobs:
                    for read in (portal._review_status, portal._check_status, portal._repair_status):
                        if read(identity).get('status') in {'queued', 'running', 'cancelling'}:
                            return
                portal.stopping.set()
                # Ctrl+C during a reload must still let the existing finally block finish.
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                raise SystemExit(RELOAD_EXIT)
        finally:
            for _ in range(acquired):
                self.slots.release()

    def server_close(self):
        signal.signal(signal.SIGUSR1, self.previous_signal)
        super().server_close()
