from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError

import cv2
import numpy as np
import pytest

from weft.io import digest, write_json
from weft import scene_advisor as advisor
from weft import meta_access


@pytest.fixture(autouse=True)
def isolated_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(meta_access, "PROJECT_ROOT", tmp_path)
    meta_access.configure_access(tmp_path, total_budget_usd=.2)


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    video = root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (80, 60))
    for i in range(4):
        writer.write(np.full((60, 80, 3), i * 40, dtype=np.uint8))
    writer.release()
    source_hash = digest(video)
    rows = [{"id": f"{source_hash}:frame:{i}", "object_id": "ball-0", "time_s": i / 10,
             "frame": i, "x_px": 10 + i, "y_px": 30} for i in range(4)]
    write_json(root / "evidence.json", {"source_sha256": source_hash, "generation": rows,
        "development_validation": [{"secret": "DO_NOT_EXPOSE_DEVELOPMENT"}],
        "audit_episodes": [[{"secret": "DO_NOT_EXPOSE_AUDIT"}]], "limits": ["Uncalibrated image scale"]})
    return root


def proposal(context):
    return {"summary": "Rolling motion has multiple possible physical explanations.",
            "hypotheses": [{"id": "support", "concept": "contact", "status": "hypothesis",
                "claim": "A support surface could constrain vertical motion.",
                "evidence_ids": [context["observations"][0]["id"]], "object_ids": ["ball-0"],
                "confidence": .5, "alternatives": ["Camera projection hides vertical motion."],
                "required_tests": ["Measure the support plane and collect a second viewpoint."]}],
            "unanswered_questions": ["What is the metric scale?"]}


def reply(context, **overrides):
    result = {"id": "test-response", "model": advisor.MODEL,
              "usage": {"prompt_tokens": 400, "completion_tokens": 200},
              "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(proposal(context))}}]}
    result.update(overrides)
    return BytesIO(json.dumps(result).encode())


def test_prepare_never_calls_network_or_exposes_audit(monkeypatch, run, tmp_path):
    monkeypatch.setattr(advisor, "_open", lambda *_: pytest.fail("Unexpected network request"))
    before = {p.name: digest(p) for p in run.iterdir()}
    out = tmp_path / "prepared"
    result = advisor.advise_run(run, out)
    payload = json.loads((out / "request.json").read_text())
    assert payload['reasoning_effort'] == 'low'
    assert payload['max_completion_tokens'] == 4096
    assert result["status"] == "prepared" and result["request_count"] == 0
    assert not result["live_access_verified"] and not result["applied"]
    assert "DO_NOT_EXPOSE" not in json.dumps(payload)
    assert len([item for item in payload["messages"][1]["content"] if item["type"] == "image_url"]) == 3
    assert payload["response_format"]["json_schema"]["schema"] == advisor.PROPOSAL_SCHEMA
    assert before == {p.name: digest(p) for p in run.iterdir()}
    assert result["implementation_sha256"] == digest(Path(advisor.__file__))


@pytest.mark.parametrize('mobile', [False, True])
def test_spatial_body_context_includes_authored_limits_not_evaluation(run, mobile):
    evidence = json.loads((run / "evidence.json").read_text())
    evidence["reference"] = evidence.pop("generation")
    evidence["limits"].append("The simulator uses a simplified anchored eight-hinge skeleton, not a complete human musculoskeletal model.")
    write_json(run / "evidence.json", evidence)
    write_json(run / "human.json", {"schema_version": 3 if mobile else 2, "profile": "exercise-v3" if mobile else "exercise-v2",
        "controller": "bias_compensated_pd", "motor_limits": {"torso_flex": 120},
        "assumptions": ["Virtual support, not measured forces"],
        "reference_evidence": {"secret": "DO_NOT_EXPOSE"},
        "mobility": {"base_height": .8, "floor_fit": {"secret": "DO_NOT_EXPOSE"}},
        "root_reference": [{"secret": "DO_NOT_EXPOSE"}],
        "reconstruction_coverage": {"secret": "DO_NOT_EXPOSE"}})
    context = advisor.prepare_context(run)
    assert context["body_model"]["profile"] == ("exercise-v3" if mobile else "exercise-v2")
    if mobile:
        assert context["body_model"]["mobile_support"]["base_height"] == .8
        axes = context['body_model']['mobile_support']['coordinate_axes']
        assert axes['root_z'].startswith('vertical')
        assert axes['root_y'].startswith('depth')
        assert 'not independently calibrated' in axes['position_units']
    else:
        assert 'mobile_support' not in context['body_model']
    assert context["body_model"]["motor_limits"] == {"torso_flex": 120}
    assert "DO_NOT_EXPOSE" not in json.dumps(context)
    assert "eight-hinge" not in json.dumps(context)


def test_actual_request_payload_and_usage_are_preserved_without_key(monkeypatch, run, tmp_path):
    context = advisor.prepare_context(run)
    sent = []
    def respond(request):
        sent.append(request)
        assert request.full_url == "https://api.meta.ai/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer fixture-secret"
        return reply(context)
    monkeypatch.setenv("MODEL_API_KEY", "fixture-secret")
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setattr(advisor, "_open", respond)
    out = tmp_path / "proposal"
    result = advisor.advise_run(run, out, execute=True, max_cost_usd=.06)
    assert len(sent) == 1 and result["request_count"] == 1 and result["retries"] == 0
    assert result["status"] == "proposed" and result["live_access_verified"]
    assert result['schema_validated'] and result['finish_reason'] == 'stop'
    assert result["cost_usd"] == pytest.approx((400 * 1.25 + 200 * 4.25) / 1e6)
    assert not result["applied"] and result["independent_verification"] == "not performed"
    assert json.loads((out / "hypotheses.json").read_text())["accepted"] is False
    assert all("fixture-secret" not in p.read_text() for p in out.iterdir())


@pytest.mark.parametrize("budget", [None, 0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_budget_cannot_create_request(monkeypatch, run, tmp_path, budget):
    monkeypatch.setenv("MODEL_API_KEY", "fixture-secret")
    monkeypatch.setattr(advisor, "_open", lambda *_: pytest.fail("Unexpected network request"))
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="budget"):
        advisor.advise_run(run, out, execute=True, max_cost_usd=budget)
    assert not out.exists()


def test_missing_key_never_falls_back(monkeypatch, run, tmp_path):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="absent"):
        advisor.advise_run(run, tmp_path / "missing", execute=True, max_cost_usd=.06)


@pytest.mark.parametrize("change", [
    lambda value: value.update(patch={"gravity": [0, 0]}),
    lambda value: value["hypotheses"][0].update(evidence_ids=["audit:frame:999"]),
    lambda value: value["hypotheses"][0].update(object_ids=["unseen-person"]),
    lambda value: value["hypotheses"][0].update(confidence=float("nan")),
    lambda value: value["hypotheses"][0].update(confidence=True),
    lambda value: value["hypotheses"][0].update(status="verified"),
    lambda value: value["hypotheses"][0].update(required_tests=[]),
    lambda value: value["hypotheses"].append(deepcopy(value["hypotheses"][0])),
])
def test_untrusted_output_rejects_invalid_claims_or_references(run, change):
    context = advisor.prepare_context(run)
    value = proposal(context)
    change(value)
    with pytest.raises(ValueError):
        advisor.validate_advice(value, [row["id"] for row in context["observations"]], ["ball-0"])


def test_rejected_response_retained_and_not_applied(monkeypatch, run, tmp_path):
    context = advisor.prepare_context(run)
    result = json.loads(reply(context).read())
    result["choices"][0]["finish_reason"] = "length"
    monkeypatch.setenv("MODEL_API_KEY", "fixture-secret")
    monkeypatch.setattr(advisor, "_open", lambda _: BytesIO(json.dumps(result).encode()))
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="did not finish"):
        advisor.advise_run(run, out, execute=True, max_cost_usd=.06)
    record = json.loads((out / "advisor-call.json").read_text())
    assert record["status"] == "failed" and record["request_count"] == 1
    assert (out / "response.json").exists() and not (out / "hypotheses.json").exists()


def test_http_failure_logged_without_retry(monkeypatch, run, tmp_path):
    def fail(request):
        raise HTTPError(request.full_url, 429, "rate limit", {}, None)
    monkeypatch.setenv("MODEL_API_KEY", "fixture-secret")
    monkeypatch.setattr(advisor, "_open", fail)
    out = tmp_path / "rate-limit"
    with pytest.raises(RuntimeError, match="429"):
        advisor.advise_run(run, out, execute=True, max_cost_usd=.06)
    record = json.loads((out / "advisor-call.json").read_text())
    assert record["request_count"] == 1 and record["retries"] == 0 and record["cost_usd"] is None


def test_unknown_usage_is_not_free():
    assert advisor._cost(None) is None
    assert advisor._cost({"prompt_tokens": 42}) is None
    assert advisor._cost({"prompt_tokens": True, "completion_tokens": 1}) is None


def test_source_and_destination_integrity(run, tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        advisor.advise_run(run, run / "advice")
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(FileExistsError):
        advisor.advise_run(run, out)
    with (run / "source.mp4").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        advisor.advise_run(run, tmp_path / "changed")


def test_exercise_input_preserves_pose_uncertainty(run):
    evidence = json.loads((run / "evidence.json").read_text())
    evidence["reference"] = [{"evidence_id": row["id"], "time_s": row["time_s"],
                               "joints": {"left_knee": .3}, "joint_confidence": {"left_knee": .2},
                               "pose_status": "tracked_partial"} for row in evidence.pop("generation")]
    write_json(run / "evidence.json", evidence)
    context = advisor.prepare_context(run)
    assert context["profile"] == "exercise" and context["objects"][0]["id"] == "person"
    assert context["observations"][0]["joint_confidence"]["left_knee"] == .2
    assert context["observations"][0]["pose_status"] == "tracked_partial"


def test_redirect_cannot_forward_credentials():
    with pytest.raises(ValueError, match="not forwarded"):
        advisor._NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://other.example")


def test_missing_aggregate_budget_blocks_network(monkeypatch, run, tmp_path):
    project = tmp_path / "unconfigured"
    project.mkdir()
    monkeypatch.setenv("MODEL_API_KEY", "fixture-secret")
    monkeypatch.setattr(advisor, "_open", lambda *_: pytest.fail("Unexpected request"))
    with pytest.raises(ValueError, match="total Meta budget"):
        advisor.advise_run(run, tmp_path / "not-authorized", execute=True, max_cost_usd=.06, project_root=project)


def test_failure_consumes_aggregate_reservation(monkeypatch, run, tmp_path):
    monkeypatch.setenv("MODEL_API_KEY", "fixture-secret")
    meta_access.configure_access(tmp_path, total_budget_usd=.06)
    monkeypatch.setattr(advisor, "_open", lambda _: (_ for _ in ()).throw(TimeoutError("unknown charge")))
    with pytest.raises(TimeoutError):
        advisor.advise_run(run, tmp_path / "timeout", execute=True, max_cost_usd=.06)
    status = meta_access.access_status(tmp_path)
    assert status["committed_usd"] == advisor.ESTIMATE and status["unsettled_count"] == 1
    monkeypatch.setattr(advisor, "_open", lambda *_: pytest.fail("Exceeded aggregate budget"))
    with pytest.raises(ValueError, match="Insufficient"):
        advisor.advise_run(run, tmp_path / "second", execute=True, max_cost_usd=.06)
    assert json.loads((tmp_path / "second/advisor-call.json").read_text())["request_count"] == 0
