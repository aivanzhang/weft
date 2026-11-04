from weft.portal_demo import DemoManager


def test_live_elapsed_advances_without_stage_changes_and_keeps_terminal_duration(tmp_path, monkeypatch):
    demo = DemoManager(tmp_path / "portal", tmp_path, "local-test-token")
    demo.run = {"id": "run-test", "status": "running", "created": 100,
                "elapsed_seconds": 2, "steps": [], "total_estimate_seconds": 60,
                "error": None, "source": {}}
    monkeypatch.setattr("weft.portal_demo.time.time", lambda: 112.3)
    assert demo.summary()["elapsed_seconds"] == 12.3
    monkeypatch.setattr("weft.portal_demo.time.time", lambda: 120.8)
    assert demo.summary()["elapsed_seconds"] == 20.8
    assert demo.run["elapsed_seconds"] == 2
    demo.run.update(status="complete", elapsed_seconds=19.7)
    assert demo.summary()["elapsed_seconds"] == 19.7
