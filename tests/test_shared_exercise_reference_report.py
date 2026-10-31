from copy import deepcopy
from html.parser import HTMLParser

import pytest

from weft.shared_exercise_reference_report import PROFILE, write_reference_trial_report


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.links, self.tags, self.text = [], [], []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "a":
            self.links.append(dict(attrs)["href"])

    def handle_data(self, data):
        self.text.append(data)


def result():
    return {"schema_version": 1, "profile": PROFILE, "status": "completed", "stage": "complete",
            "same_initial_state": True, "scene_sha256": "a" * 64, "task_sha256": "b" * 64,
            "trials": [{"mode": mode, "status": "verified", "steps": 254,
                        "metrics": {"success": False, "completed_interval": True, "mean_reward": .3,
                                    "simulated_duration_s": 254 / 30, "angular_rmse_rad": .2,
                                    "failed_gates": ["equipment_stability:working_object"]},
                        "action_replay": {"status": "passed"}}
                       for mode in ("authored", "reference_proposal")]}


def artifact(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("saved")


def page(root, report):
    assert write_reference_trial_report(root, report) == "index.html"
    return Page((root / "index.html").read_text())


def test_completed_delivery_keeps_both_task_failures_and_fixed_scope(tmp_path):
    report = result()
    original = deepcopy(report)
    for mode in ("authored", "reference_proposal"):
        artifact(tmp_path, f"recordings/{mode}/replay.html")
        artifact(tmp_path, f"proofs/{mode}-action-replay.json")
    parsed = page(tmp_path, report)
    text = " ".join(parsed.text)
    assert "Comparison saved" in text and text.count("Task outcome: Failed") == 2
    assert "does not mean the simulated task passed" in text
    assert "programmed controller, not a learned policy or a repaired body" in text
    assert "Identical initial states verified" in text
    assert text.count("Verified exact actions and states") == 2
    assert len(parsed.links) == 4 and "script" not in parsed.tags
    assert report == original


def test_partial_failure_keeps_existing_replay_but_does_not_invent_proof(tmp_path):
    report = result()
    report.update(status="failed", stage="export", error={"message": "Time bound reached"})
    report["trials"] = report["trials"][:1]
    artifact(tmp_path, "recordings/authored/replay.html")
    parsed = page(tmp_path, report)
    text = " ".join(parsed.text)
    assert "Comparison stopped" in text and "Stopped at export: Time bound reached" in text
    assert "Not verified" in text and "Verified exact actions" not in text
    assert parsed.links == ["recordings/authored/replay.html"]


def test_text_and_report_paths_never_become_html_or_remote_links(tmp_path):
    report = result()
    report.update(status="failed", stage='<img src="remote">', error={"message": "<script>bad()</script>"})
    report["trials"][0]["recording"] = {"directory": "https://example.invalid/"}
    report["trials"][0]["metrics"]["failed_gates"] = ["<iframe src='x'>"]
    report["scene_sha256"] = '<img src="remote">'
    parsed = page(tmp_path, report)
    assert not set(parsed.tags) & {"script", "img", "iframe"}
    assert not parsed.links
    assert "<script>bad()</script>" in " ".join(parsed.text)


def test_missing_values_are_unavailable_not_zero_or_success(tmp_path):
    report = result()
    report.update(status="failed", same_initial_state=None, scene_sha256=None, task_sha256=None)
    report["trials"] = [{"mode": "authored", "metrics": {"mean_reward": float("nan")}}]
    parsed = page(tmp_path, report)
    text = " ".join(parsed.text)
    assert "Task outcome: Not evaluated" in text
    assert "Matched initial states not yet verified" in text
    assert "Unavailable" in text and "nan" not in text


def test_links_do_not_follow_artifact_symlinks(tmp_path):
    artifact(tmp_path, "real/replay.html")
    (tmp_path / "recordings").mkdir()
    (tmp_path / "recordings/authored").symlink_to(tmp_path / "real", target_is_directory=True)
    assert not page(tmp_path, result()).links


def test_output_is_exclusive_and_does_not_change_existing_report(tmp_path):
    original = b"preserved output"
    (tmp_path / "index.html").write_bytes(original)
    with pytest.raises(FileExistsError):
        write_reference_trial_report(tmp_path, result())
    assert (tmp_path / "index.html").read_bytes() == original


@pytest.mark.parametrize("value", [None, "unavailable"])
def test_partial_null_or_unknown_metrics_are_not_a_report_failure(tmp_path, value):
    report = result()
    report.update(status="failed", trials=[{"mode": "authored", "metrics": value, "action_replay": value}])
    text = " ".join(page(tmp_path, report).text)
    assert "Task outcome: Not evaluated" in text and "Not verified" in text


@pytest.mark.parametrize("count", [0, 1])
def test_completed_status_does_not_invent_missing_replay_checks(tmp_path, count):
    report = result()
    report["trials"] = report["trials"][:count]
    text = " ".join(page(tmp_path, report).text)
    assert "Both recordings" not in text and "Verified exact actions and states" not in text
    assert "workflow finished" in text


@pytest.mark.parametrize("change", ["nonterminal", "duplicate", "third"])
def test_rejects_incomplete_or_unbounded_summary_contract(tmp_path, change):
    report = result()
    if change == "nonterminal":
        report["status"] = "running"
    elif change == "duplicate":
        report["trials"][1]["mode"] = "authored"
    else:
        report["trials"].append(report["trials"][0])
    with pytest.raises(ValueError):
        write_reference_trial_report(tmp_path, report)
    assert not (tmp_path / "index.html").exists()
