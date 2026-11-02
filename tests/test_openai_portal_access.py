import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from weft.openai_access import MODEL, RESERVATION_MICROUSD, OpenAIBudget, credential_status, read_key
from weft.portal_pipeline import analyze_frames, validate_plan
from weft.io import digest


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    folder = tmp_path / ".local/credentials"
    folder.mkdir(parents=True, mode=0o700)
    file = folder / "openai-api-key"
    file.write_text("sk-test-not-a-real-key")
    file.chmod(0o600)
    return tmp_path


def response(plan=None, **updates):
    return {"id": "response-test", "model": MODEL, "status": "completed",
        "usage": {"input_tokens": 2000, "output_tokens": 300},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(plan or {
            "route": "exercise", "summary": "One person exercising", "people": 1,
            "camera": "fixed", "objects": [], "warnings": ["Muscles are illustrative"]})}]}], **updates}


def test_key_is_server_only_and_permission_checked(project):
    assert credential_status(project) == {"configured": True, "source": "project_file", "live_verified": False}
    assert read_key(project) == "sk-test-not-a-real-key"
    path = project / ".local/credentials/openai-api-key"
    path.chmod(0o644)
    assert not credential_status(project)["configured"]
    with pytest.raises(ValueError, match="owner-only"):
        read_key(project)


def test_reject_symlinked_key(project):
    path = project / ".local/credentials/openai-api-key"
    saved = path.with_name("saved")
    path.rename(saved)
    path.symlink_to(saved)
    assert not credential_status(project)["configured"]


def test_budget_reservation_survives_restarts_and_blocks_duplicate(project):
    budget = OpenAIBudget(project)
    assert not budget.status()["configured"]
    with pytest.raises(ValueError, match="unconfigured"):
        budget.reserve("a" * 32, "b" * 64)
    budget.configure(5, note="User approved $5 for portal development")
    budget.reserve("a" * 32, "b" * 64)
    reopened = OpenAIBudget(project)
    assert reopened.status()["committed_usd"] == RESERVATION_MICROUSD / 1e6
    with pytest.raises(ValueError, match="already reserved"):
        reopened.reserve("a" * 32, "b" * 64)
    with pytest.raises(ValueError, match="existing"):
        reopened.configure(.1, note="Not allowed to erase charges")
    receipt = reopened.settle("a" * 32, response())
    assert receipt["cost_usd_upper_estimate"] == .00285
    assert reopened.status()["unsettled_calls"] == 0
    assert reopened.status()["remaining_usd"] == 4.99715


def test_concurrent_reservations_cannot_overspend(project):
    budget = OpenAIBudget(project)
    budget.configure(.4, note="Concurrency test")
    def reserve(i):
        try:
            OpenAIBudget(project).reserve(str(i) * 32, "c" * 64)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(reserve, [1, 2])) == [False, True]
    assert budget.status()["calls"] == 1


def test_missing_usage_and_overage_fail_closed(project):
    budget = OpenAIBudget(project)
    budget.configure(5, note="Test")
    budget.reserve("a" * 32, "b" * 64)
    with pytest.raises(ValueError, match="usage"):
        budget.settle("a" * 32, response(usage={}))
    assert budget.status()["unsettled_calls"] == 1
    with pytest.raises(ValueError, match="exceeded"):
        budget.settle("a" * 32, response(usage={"input_tokens": 500_000, "output_tokens": 3000}))
    with pytest.raises(ValueError, match="exhausted"):
        budget.reserve("d" * 32, "b" * 64)


def sample(project):
    folder = project / "analysis"
    folder.mkdir()
    image = folder / "frame-00000.jpg"
    image.write_bytes(b"not-a-real-jpeg; transport is mocked")
    return folder, {"width": 100, "height": 200, "fps": 30, "decoded_frame_count": 30, "source_sha256": "c"*64,
        "frames": [{"frame": 0, "decoder_time_s": .007, "nominal_time_s": 0.,
                    "artifact": image.name, "sha256": digest(image)}]}


def test_structured_api_artifacts_and_no_login_or_tools(project):
    OpenAIBudget(project).configure(5, note="Test")
    folder, frames = sample(project)
    def transport(payload, key):
        assert key == "sk-test-not-a-real-key"
        assert payload["store"] is False and "tools" not in payload
        assert payload["text"]["format"]["strict"] is True
        assert payload["model"] == MODEL
        assert payload["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
        assert OpenAIBudget(project).status()["unsettled_calls"] == 1
        return response()
    plan = analyze_frames(frames, folder, identity="a" * 32, project=project, transport=transport)
    assert plan["route"] == "exercise"
    assert json.loads((folder / "receipt.json").read_text())["authentication"] == "server_project_api_key"
    assert all("sk-test" not in p.read_text() for p in folder.glob("*.json"))


def test_identical_source_and_request_reuses_paid_analysis(project):
    budget = OpenAIBudget(project)
    budget.configure(5, note="Test")
    folder, frames = sample(project)
    analyze_frames(frames, folder, identity="a" * 32, project=project, transport=lambda *args: response())
    def forbidden(*args):
        raise AssertionError("Identical analysis must not incur another paid call")
    plan = analyze_frames(frames, folder, identity="b" * 32, project=project, transport=forbidden)
    assert plan["route"] == "exercise" and budget.status()["calls"] == 1
    receipt = json.loads((folder / "receipt.json").read_text())
    assert receipt["provider_request_sent"] is False and receipt["cost_usd_upper_estimate"] == 0
    assert budget.cached_response(receipt["request_sha256"], "d" * 64) is None


def test_transport_failure_retains_cost_without_retry(project):
    OpenAIBudget(project).configure(5, note="Test")
    folder, frames = sample(project)
    calls = []
    def transport(payload, key):
        calls.append(payload)
        raise TimeoutError("Test uncertain request")
    with pytest.raises(TimeoutError):
        analyze_frames(frames, folder, identity="a" * 32, project=project, transport=transport)
    assert len(calls) == 1 and OpenAIBudget(project).status()["unsettled_calls"] == 1


def test_transport_uses_only_fixed_tls_api_endpoint(monkeypatch):
    import weft.openai_access as access
    calls = []
    class Socket:
        def settimeout(self, value):
            assert 0 < value <= 60
    class Response:
        status = 200
        body = b'{"status":"completed"}'
        def read1(self, count):
            result, self.body = self.body[:count], self.body[count:]
            return result
    class Connection:
        sock = Socket()
        def __init__(self, host, timeout):
            assert host == "api.openai.com" and timeout == 60
        def connect(self): calls.append("connect")
        def request(self, method, path, body, headers):
            calls.append(path)
            assert method == "POST" and path == "/v1/responses"
            assert headers["Authorization"] == "Bearer sk-test-not-a-real-key"
            assert json.loads(body)["store"] is False
        def getresponse(self): return Response()
        def close(self): calls.append("close")
    monkeypatch.setattr(access.http.client, "HTTPSConnection", Connection)
    assert access.post_response({"store":False}, "sk-test-not-a-real-key") == {"status":"completed"}
    assert calls == ["connect", "/v1/responses", "close"]


@pytest.mark.parametrize("change", [{"command": "execute arbitrary code"}, {"camera": "invented"},
    {"people": True}, {"objects": [{"label": "x", "shape": "box", "role": "movable", "box_xyxy": [.5, 0, .2, 1]}]}])
def test_untrusted_model_output_cannot_expand_pipeline(change):
    plan = json.loads(response()["output"][0]["content"][0]["text"])
    with pytest.raises(ValueError):
        validate_plan({**plan, **change})
