from copy import deepcopy
import json

import pytest

from weft.evidence_scorecard import build_evidence_scorecard
from test_lab_console import fake_lab, run_commands


def saved_inspection(*, partial=False):
    identities = {"scene_sha256": "a" * 64, "task_sha256": "b" * 64}
    card = build_evidence_scorecard({"source_manifest_sha256": "c" * 64, **identities,
        "learning": {**identities, "evaluation": {"trials_per_controller": 3,
            "controllers": {name: {"successes": 3 if name in ("learned", "authored") else 0,
                **({} if partial else {"completed_intervals": 3, "reset_failures": 0, "mean_score": None})}
                for name in ("learned", "authored", "coast", "random")}}}})
    return {"status": "available", "source_manifest_sha256": "c" * 64, "evidence_scorecard": card}


def test_evidence_overview_and_detail_never_move_or_change_controls(fake_lab, monkeypatch):
    cls, root = fake_lab
    calls = []
    def inspect(self):
        calls.append(True)
        return saved_inspection()
    monkeypatch.setattr(cls, "inspection", inspect, raising=False)
    _, output = run_commands(root, ["motor hip .25", "evidence", "evidence 12", "controls", "quit"])
    lab = cls.instances[0]
    assert lab.calls == [("/api/reset", {"seed": 1729})]
    assert len(calls) == 2
    assert len([line for line in output if line.split(".")[0].isdigit()]) == 15
    objects = [json.loads(line) for line in output if line.startswith("{")]
    assert next(row for row in objects if row.get("measurement") == "learned saved simulator successes")["out_of"] == 3
    detail = next(row for row in objects if "category" in row)
    assert detail["category"]["id"] == "task_learning_utility"
    assert detail["saved_run"] == str(root / "run")
    assert detail["category"]["metrics"][0]["value"] is None
    assert objects[-1]["selected_manual_action"] == [0., .25]
    assert "Simulator outcomes do not establish physical accuracy." in output


def test_loading_returns_without_polling_or_blocking_then_reuses_ready_summary(fake_lab, monkeypatch):
    cls, root = fake_lab
    replies = iter([{"status": "loading"}, saved_inspection()])
    monkeypatch.setattr(cls, "inspection", lambda _: next(replies), raising=False)
    _, output = run_commands(root, ["evidence", "status", "evidence", "quit"])
    assert sum("Evidence is loading" in line for line in output) == 1
    assert any("1. Execution and action validity" in line for line in output)
    assert cls.instances[0].calls == [("/api/reset", {"seed": 1729})]


@pytest.mark.parametrize("response", [{"status": "available"},
    {"status": "unavailable", "reason": "Saved metadata\n\x1b[2Jmust remain text"}])
def test_unavailable_or_older_inspection_does_not_fake_a_scorecard(fake_lab, monkeypatch, response):
    cls, root = fake_lab
    monkeypatch.setattr(cls, "inspection", lambda _: response, raising=False)
    _, output = run_commands(root, ["evidence", "quit"])
    assert "Evidence summary is unavailable for this saved run." in output
    assert not any("1. Execution" in line for line in output)
    assert all("\x1b" not in line for line in output)
    assert cls.instances[0].calls == [("/api/reset", {"seed": 1729})]


@pytest.mark.parametrize("argument", ["0", "15", "-1", "nan", "1.5"])
def test_bad_category_never_starts_inspection_or_control(fake_lab, monkeypatch, argument):
    cls, root = fake_lab
    def forbidden(_):
        raise AssertionError("Invalid category must not start work")
    monkeypatch.setattr(cls, "inspection", forbidden, raising=False)
    _, output = run_commands(root, ["evidence " + argument, "quit"])
    assert any("Evidence category must be an integer" in line for line in output)
    assert cls.instances[0].calls == [("/api/reset", {"seed": 1729})]


@pytest.mark.parametrize("change", ["missing_row", "missing_scope", "missing_metrics", "null_metric", "bad_label", "nan_value", "different_source"])
def test_malformed_scorecard_is_rejected_without_stopping_an_active_recording(fake_lab, monkeypatch, change):
    cls, root = fake_lab
    response = deepcopy(saved_inspection())
    card = response["evidence_scorecard"]
    if change == "missing_row":
        card["rows"].pop()
    elif change == "missing_scope":
        card.pop("scope")
    elif change == "missing_metrics":
        card["rows"][0].pop("metrics")
    elif change == "null_metric":
        card["rows"][0]["metrics"][0] = None
    elif change == "bad_label":
        card["rows"][0]["metrics"][0]["label"] = {}
    elif change == "nan_value":
        card["rows"][0]["metrics"][0]["value"] = float("nan")
    else:
        response["source_manifest_sha256"] = "d" * 64
    monkeypatch.setattr(cls, "inspection", lambda _: response, raising=False)
    result, output = run_commands(root, ["record start", "evidence", "mode coast", "step", "record stop", "quit"])
    assert "Saved evidence summary has an unsupported format" in output
    assert result["recording"]["export"] == "ready"
    assert [row["type"] for row in cls.instances[0].events] == ["start", "step"]


def test_success_only_lab_receipt_does_not_invent_completion_or_reward(fake_lab, monkeypatch):
    cls, root = fake_lab
    monkeypatch.setattr(cls, "inspection", lambda _: saved_inspection(partial=True), raising=False)
    _, output = run_commands(root, ["evidence", "evidence 12", "quit"])
    objects = [json.loads(line) for line in output if line.startswith("{")]
    detail = next(row["category"] for row in objects if "category" in row)
    metrics = {metric["id"]: metric for metric in detail["metrics"]}
    assert (metrics["learned_successes"]["value"], metrics["learned_successes"]["denominator"]) == (3, 3)
    assert metrics["learned_completed_intervals"]["value"] is None
    assert metrics["learned_completed_intervals"]["denominator"] is None
    assert metrics["learned_mean_reward"]["value"] is None
    assert cls.instances[0].calls == [("/api/reset", {"seed": 1729})]
