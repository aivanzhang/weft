"""Probe existing ownership without starting a server or changing saved state."""

import errno
import fcntl
import json
import os
from pathlib import Path
import socket
import sys


def inspect_demo(root, ports):
    locked = False
    try:
        handle = os.open(Path(root) / "portal.lock", os.O_RDWR | os.O_NOFOLLOW)
    except FileNotFoundError:
        handle = None
    if handle is not None:
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                locked = True
        finally:
            os.close(handle)
    busy = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                if error.errno != errno.EADDRINUSE:
                    raise
                busy.append(port)
    return {"directory_in_use": locked, "busy_ports": busy}


if __name__ == "__main__":
    print(json.dumps(inspect_demo(sys.argv[1], [int(port) for port in sys.argv[2:]])))
