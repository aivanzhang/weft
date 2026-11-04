import fcntl
from pathlib import Path
import runpy
import socket

inspect_demo = runpy.run_path(str(Path(__file__).parents[1] / "sandbox/preflight.py"))["inspect_demo"]


def test_preflight_detects_lock_without_deleting_or_replacing_it(tmp_path):
    lock = tmp_path / "portal.lock"
    lock.write_bytes(b"unchanged")
    with lock.open("r+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert inspect_demo(tmp_path, []) == {"directory_in_use": True, "busy_ports": []}
        assert lock.read_bytes() == b"unchanged"
    assert inspect_demo(tmp_path, [])["directory_in_use"] is False


def test_preflight_missing_collection_does_not_create_it(tmp_path):
    root = tmp_path / "missing"
    assert inspect_demo(root, []) == {"directory_in_use": False, "busy_ports": []}
    assert not root.exists()


def test_preflight_detects_listener_without_connecting_or_stopping_it(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert inspect_demo(tmp_path, [port])["busy_ports"] == [port]
        assert listener.getsockname()[1] == port
    assert inspect_demo(tmp_path, [port])["busy_ports"] == []
