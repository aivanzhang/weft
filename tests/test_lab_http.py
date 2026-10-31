from contextlib import contextmanager
from http.client import HTTPConnection
import json
from pathlib import Path
import threading

import pytest

from weft.io import digest, write_json
from weft.lab import Lab, LabServer


def make_ready_lab(root):
    root = Path(root)
    run = root / "run"
    run.mkdir(parents=True)
    scene = {
        "schema_version": 1, "profile": "planar-rigid-v1", "width": 1, "height": 0.6,
        "dt": 1 / 30, "gravity": [0, 0], "damping": 0.9, "restitution": 0.65,
        "objects": [{"id": "actor", "label": "Actor", "role": "active", "shape": "circle",
                     "position": [0.2, 0.3], "velocity": [0, 0], "radius": 0.03,
                     "color": "#DDCC00", "mass": 1, "provenance": {"geometry": "synthetic"}}],
        "controlled_id": "actor", "target": [0.8, 0.3],
        "task": {"position_tolerance": 0.06, "speed_tolerance": 0.15, "max_steps": 300, "acceleration": 2},
        "relationships": [], "assumptions": ["Synthetic HTTP integration fixture, not real video evidence"],
    }
    write_json(root / "lab-config.json", {"profile": "planar"})
    write_json(root / "status.json", {"status": "ready", "events": []})
    write_json(run / "scene.json", scene)
    write_json(run / "policy.json", {"schema_version": 1, "algorithm": "tabular_q_learning",
                                   "error_edges": [-0.1, 0.1], "velocity_edges": [-0.1, 0.1], "q_values": {}})
    (run / "preview.bin").write_bytes(b"0123456789abcdef")
    (run / "lab.html").write_text("<html>untrusted artifact</html>")
    write_json(run / "manifest.json", {"files": {p.name: digest(p) for p in run.iterdir()}})
    return Lab(root)


def test_failed_interval_summary_does_not_claim_full_source_verification(tmp_path):
    lab=make_ready_lab(tmp_path/'lab')
    lab.config.update(profile='appearance',scene_config={'tracking':{
        'decode_interval':[2580,2760],'initialization_frame':2620,'other_private_field':'not returned'}})
    write_json(lab.root/'status.json',{'status':'failed','error':{'message':'Incomplete source window'},'events':[]})
    try:
        result=lab.summary()
        assert result['status']=='failed'
        interval=result['requested_source_interval']
        assert set(interval)=={'first_frame','last_frame','initialization_frame','scope'}
        assert interval['first_frame']==2580 and interval['initialization_frame']==2620
        assert 'Requested' in interval['scope'] and 'not whole-video verification' in interval['scope']
        assert 'snapshot' not in result
    finally:
        lab.close()


@contextmanager
def serving(lab):
    server = LabServer(("127.0.0.1", 0), lab)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        lab.close()


def request(server, path, *, method="GET", data=None, body=None, token=True, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    payload = json.dumps(data).encode() if data is not None else body
    all_headers = {"Host": f"127.0.0.1:{server.server_port}"}
    if token:
        all_headers["Authorization"] = "Bearer " + server.lab.token
    if payload is not None:
        all_headers["Content-Type"] = "application/json"
        all_headers["Content-Length"] = str(len(payload))
    all_headers.update(headers or {})
    try:
        connection.putrequest(method, path, skip_host=True)
        for name, value in all_headers.items():
            connection.putheader(name, value)
        connection.endheaders(payload)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


@pytest.fixture
def server(tmp_path):
    with serving(make_ready_lab(tmp_path / "lab")) as instance:
        yield instance


def test_status_requires_token_host_and_same_origin(server):
    assert request(server, "/api/status", token=False)[0] == 401
    assert request(server, "/api/status", headers={"Authorization": "Bearer wrong"})[0] == 401
    assert request(server, "/api/status", headers={"Host": "attacker.example"})[0] == 403
    assert request(server, "/api/status", headers={"Origin": "https://attacker.example"})[0] == 403
    code, headers, body = request(server, "/api/status", headers={"Origin": f"http://127.0.0.1:{server.server_port}"})
    assert code == 200
    result = json.loads(body)
    assert result["status"] == "ready" and result["snapshot"]["steps"] == 0
    assert result["scene"]["controlled_id"] == "actor"
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


def test_query_token_allows_artifact_links_but_unicode_token_is_rejected(server):
    assert request(server, f"/api/status?token={server.lab.token}", token=False)[0] == 200
    assert request(server, "/api/status?token=%C3%A9", token=False)[0] == 401
    assert request(server, "/api/status", headers={"Authorization": "Bearer é"})[0] == 401


def test_exercise_coverage_endpoint_requires_the_same_local_authentication(server):
    assert request(server, '/api/exercise-coverage', token=False)[0] == 401
    assert request(server, '/api/exercise-coverage', headers={'Origin': 'https://attacker.example'})[0] == 403
    code, headers, body = request(server, '/api/exercise-coverage')
    assert code == 200 and json.loads(body) == {'status': 'not_applicable'}
    assert headers['Cache-Control'] == 'no-store'


def test_inspection_endpoint_is_authenticated_cached_and_does_not_start_physics(server):
    assert request(server, '/api/inspection', token=False)[0] == 401
    assert request(server, '/api/inspection', headers={'Host': 'attacker.example'})[0] == 403
    assert request(server, '/api/inspection', headers={'Origin': 'https://attacker.example'})[0] == 403
    assert server.lab.inspection_future is None
    code, headers, body = request(server, '/api/inspection')
    assert code == 200 and json.loads(body)['status'] in ('loading', 'available')
    future = server.lab.inspection_future
    assert future.result(timeout=3)['status'] == 'available'
    code, _, body = request(server, '/api/inspection')
    result = json.loads(body)
    assert code == 200 and result['actors'][0]['id'] == 'actor'
    assert len(body) <= 64 * 1024 and headers['Cache-Control'] == 'no-store'
    assert server.lab.inspection_future is future and server.lab.session is None


def test_inspection_failure_does_not_enable_or_rewrite_a_run(server):
    scene_path = server.lab.root / 'run' / 'scene.json'
    scene_path.write_text('{"unsealed":true}')
    request(server, '/api/inspection')
    server.lab.inspection_future.result(timeout=3)
    code, _, body = request(server, '/api/inspection')
    assert code == 200 and json.loads(body)['status'] == 'unavailable'
    assert scene_path.read_text() == '{"unsealed":true}'
    assert server.lab.session is None and server.lab.status()['status'] == 'ready'


def test_public_assets_need_no_token_but_never_bypass_host_check(server):
    assert request(server, "/", token=False)[0] == 200
    assert request(server, "/lab.js", token=False)[0] == 200
    assert request(server, "/", token=False, headers={"Host": "localhost.attacker.example"})[0] == 403
    assert request(server, "/files/run/scene.json", token=False)[0] == 401


def test_artifact_paths_cannot_escape_allowed_roots(server, tmp_path):
    assert request(server, "/files/run/../lab-config.json")[0] == 400
    assert request(server, "/files/run/%2e%2e/lab-config.json")[0] == 400
    assert request(server, "/files/run/../../outside.txt")[0] == 400
    assert request(server, "/files/lab-config.json")[0] == 400
    (server.lab.root / "run" / "config-link.json").symlink_to(server.lab.root / "lab-config.json")
    assert request(server, "/files/run/config-link.json")[0] == 400


def test_binary_byte_ranges_and_untrusted_html_download(server):
    code, headers, body = request(server, "/files/run/preview.bin", headers={"Range": "bytes=3-7"})
    assert (code, body) == (206, b"34567")
    assert headers["Content-Range"] == "bytes 3-7/16"
    assert headers["Content-Length"] == "5"
    assert request(server, "/files/run/preview.bin", headers={"Range": "bytes=14-999"})[2] == b"ef"
    assert request(server, "/files/run/preview.bin", headers={"Range": "bytes=16-"})[0] == 416
    assert request(server, "/files/run/preview.bin", headers={"Range": "bytes=1-2,4-5"})[0] == 416
    code, headers, _ = request(server, "/files/run/lab.html")
    assert code == 200 and headers["Content-Disposition"].startswith("attachment;")


@pytest.mark.parametrize("body,headers", [
    (b"{", {}), (b"[]", {}), (b"{}", {"Content-Type": "text/plain"}),
    (b"x" * 4097, {}), (b"{}", {"Content-Length": "nonsense"}),
])
def test_malformed_requests_fail_without_mutating_state(server, body, headers):
    assert request(server, "/api/step", method="POST", body=body, headers=headers)[0] == 400
    status = json.loads(request(server, "/api/status")[2])
    assert status["snapshot"]["steps"] == 0


@pytest.mark.parametrize("data", [
    {"steps": 7}, {"steps": 0}, {"steps": True}, {"action": 6}, {"action": -1},
    {"action": True}, {"policy": "true"}, {"steps": float("nan")},
])
def test_control_bounds_reject_invalid_inputs(server, data):
    assert request(server, "/api/step", method="POST", data=data)[0] == 400
    assert json.loads(request(server, "/api/status")[2])["snapshot"]["steps"] == 0


def test_authoritative_steps_reset_and_completed_episode(server):
    code, _, body = request(server, "/api/step", method="POST", data={"action": 2, "steps": 6})
    result = json.loads(body)
    assert code == 200 and len(result["transitions"]) == 6
    assert result["snapshot"]["steps"] == 6
    assert result["snapshot"]["objects"][0]["position"][0] > 0.2
    assert request(server, "/api/reset", method="POST", data={"seed": True})[0] == 400
    reset = json.loads(request(server, "/api/reset", method="POST", data={"seed": 8})[2])
    assert reset["snapshot"]["steps"] == 0
    assert reset["snapshot"]["objects"][0]["position"] == [0.2, 0.3]
    for _ in range(50):
        result = json.loads(request(server, "/api/step", method="POST", data={"action": 0, "steps": 6})[2])
    assert result["snapshot"]["truncated"]
    result = json.loads(request(server, "/api/step", method="POST", data={"action": 2})[2])
    assert result["transitions"] == [] and result["snapshot"]["steps"] == 300


def test_pending_run_cannot_be_controlled(server):
    write_json(server.lab.root / "status.json", {"status": "running", "events": []})
    assert request(server, "/api/step", method="POST", data={"action": 0})[0] == 400
    assert json.loads(request(server, "/api/status")[2])["status"] == "running"


def test_tampered_run_is_rejected_before_loading_session(server):
    (server.lab.root / "run" / "scene.json").write_text("{}")
    assert request(server, "/api/status")[0] == 400
    assert server.lab.session is None


def test_worker_start_failure_closes_log_and_does_not_inherit_credentials(tmp_path, monkeypatch):
    lab = make_ready_lab(tmp_path / "lab")
    captured = {}
    monkeypatch.setenv("EXAMPLE_SERVICE_SECRET", "test-only-do-not-inherit")

    def unavailable(command, **kwargs):
        captured.update(kwargs)
        raise OSError("Deliberate process-start failure")

    monkeypatch.setattr("weft.lab.subprocess.Popen", unavailable)
    try:
        with pytest.raises(OSError, match="Deliberate"):
            lab.start_worker()
        assert lab.worker_log.closed
        assert "EXAMPLE_SERVICE_SECRET" not in captured["env"]
        assert set(captured["env"]) <= {"PATH", "PYTHONUNBUFFERED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MPLCONFIGDIR",
                                        "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONSAFEPATH", "PYTHONDONTWRITEBYTECODE"}
        import weft.lab as lab_module
        assert captured["env"]["PYTHONPATH"] == str(Path(lab_module.__file__).resolve().parent.parent)
        assert Path(captured["env"]["MPLCONFIGDIR"]).is_relative_to(lab.root)
    finally:
        lab.close()


@pytest.mark.parametrize("terminal", ["ready", "failed"])
def test_close_preserves_terminal_status_while_worker_is_exiting(tmp_path, terminal):
    lab = make_ready_lab(tmp_path / "lab")
    write_json(lab.root / "status.json", {"status": terminal, "events": []})

    class FinishingWorker:
        def poll(self):
            return None

        def wait(self, timeout):
            return 0

        def terminate(self):
            raise AssertionError("Do not cancel a pipeline that already published its terminal result")

    lab.worker = FinishingWorker()
    lab.close()
    assert lab.status()["status"] == terminal


def test_worker_exit_without_terminal_status_preserves_failure_evidence(tmp_path):
    lab = make_ready_lab(tmp_path / "lab")
    write_json(lab.root / "status.json", {"status": "running", "events": [{"stage": "source", "index": 0}]})

    class ExitedWorker:
        def wait(self, timeout):
            return 7

        def poll(self):
            return 7

    lab.worker = ExitedWorker()
    lab.worker_log = (lab.root / "worker.log").open("wb")
    try:
        lab._watch_worker()
        status = lab.status()
        assert status["status"] == "failed"
        assert status["events"] == [{"stage": "source", "index": 0}]
        assert "code 7" in status["error"]["message"]
        assert (lab.root / "failure.json").is_file()
        assert lab.worker_log.closed
    finally:
        lab.close()


def test_cancelled_worker_watcher_does_not_overwrite_cancellation(tmp_path):
    lab = make_ready_lab(tmp_path / "lab")

    class StoppedWorker:
        def wait(self, timeout):
            return -15

        def poll(self):
            return -15

    lab.worker = StoppedWorker()
    lab.worker_log = (lab.root / "worker.log").open("wb")
    lab.cancelled = True
    lab.fail("User cancelled the run", "cancelled")
    try:
        lab._watch_worker()
        assert lab.status()["status"] == "cancelled"
        assert lab.status()["error"]["message"] == "User cancelled the run"
        assert lab.worker_log.closed
    finally:
        lab.close()
