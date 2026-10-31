"""The opt-in controller uses the same workflow without changing defaults or task budgets."""

from copy import deepcopy
import json
import sys

import pytest

from weft import cli, exercise_learning_pipeline as workflow, exercise_scene_pipeline
from weft import shared_exercise_feedback_learning as feedback, shared_exercise_feedback_residual as actor
from weft import shared_exercise_policies as policies
from test_build_sandbox import launcher, mocked_worker, project
from test_exercise_learning_pipeline import SETTINGS, _files, _stub_stages
from test_person_only_downstream import source_package
from test_shared_exercise_run import source_fixture


@pytest.mark.parametrize("family", ["feedback", "v4", None, True, [], {}])
def test_unknown_policy_family_rejects_before_io(tmp_path, family):
    with pytest.raises(ValueError, match="policy_family"):
        workflow.prepare_exercise_learning_pipeline("missing", tmp_path / "out",
            train_settings={"evaluation_seed": 6173, "policy_family": family})
    assert not (tmp_path / "out").exists()


def test_default_settings_and_explicit_phase_keep_original_contract():
    unchanged = workflow.validate_train_settings(SETTINGS)
    assert "policy_family" not in unchanged
    explicit = workflow.validate_train_settings({**SETTINGS, "policy_family": "phase-residual"})
    assert explicit.pop("policy_family") == "phase-residual"
    assert explicit == unchanged
    supplied = {**SETTINGS, "policy_family": "feedback-residual"}
    before = deepcopy(supplied)
    assert workflow.validate_train_settings(supplied)["policy_family"] == "feedback-residual"
    assert supplied == before


@pytest.mark.parametrize("person", [False, True])
def test_feedback_uses_all_existing_workflow_stages_without_fallback(tmp_path, monkeypatch, person):
    source = source_package(tmp_path)[0] if person else source_fixture(tmp_path)[0]
    original = _files(source)
    calls = _stub_stages(monkeypatch)
    monkeypatch.setattr(feedback, "train_shared_exercise_feedback_policy", workflow.residual.train_shared_exercise_residual_policy)
    monkeypatch.setattr(feedback, "select_shared_exercise_feedback_policy", workflow.select_feasible_shared_exercise_policy)
    monkeypatch.setattr(feedback, "evaluate_shared_exercise_feedback_policy", workflow.residual.evaluate_shared_exercise_residual_policy)
    def wrong_family(*args, **options):
        pytest.fail("The explicit feedback choice cannot fall back to phase learning")
    monkeypatch.setattr(workflow.residual, "train_shared_exercise_residual_policy", wrong_family)
    monkeypatch.setattr(workflow.residual, "evaluate_shared_exercise_residual_policy", wrong_family)
    monkeypatch.setattr(workflow, "select_feasible_shared_exercise_policy", wrong_family)
    out = tmp_path / "workflow"
    result = workflow.prepare_exercise_learning_pipeline(source, out,
        train_settings={**SETTINGS, "policy_family": "feedback-residual"}, max_seconds=30)
    plan = json.loads((out / "plan.json").read_bytes())
    assert calls == ["training", "training_selection", "independent_evaluation", "interactive_lab"]
    assert plan["policy_profile"] == actor.POLICY_PROFILE and plan["training_algorithm"] == actor.ALGORITHM
    scene = json.loads((out / "scene.json").read_bytes())
    assert plan["policy_runtime_sha256"] == actor.runtime_hashes(scene)
    assert plan["maximum_training_steps"] == 8 * plan["horizon_steps"]
    assert plan["maximum_evaluation_steps"] == 8 * plan["horizon_steps"]
    assert set(plan["training_seeds"]).isdisjoint(plan["evaluation_seeds"])
    assert plan["evaluation_feedback_to_optimizer"] is False
    assert result["status"] == "ready" and result["physical_validation"] == "unverified"
    assert result["evaluation"]["comparison_complete"] is False
    assert _files(source) == original


def test_feedback_training_failure_never_runs_an_alternative_or_later_stage(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path)
    calls = _stub_stages(monkeypatch)
    def fail(run, out, **options):
        calls.append("feedback_failed")
        out.mkdir()
        (out / "partial.json").write_text('{"status":"interrupted"}')
        raise RuntimeError("injected feedback failure")
    monkeypatch.setattr(feedback, "train_shared_exercise_feedback_policy", fail)
    out = tmp_path / "workflow"
    with pytest.raises(RuntimeError, match="injected feedback failure"):
        workflow.prepare_exercise_learning_pipeline(source, out,
            train_settings={**SETTINGS, "policy_family": "feedback-residual"})
    assert calls == ["feedback_failed"]
    assert (out / "training/partial.json").exists()
    assert not (out / "selected").exists() and not (out / "lab").exists()
    assert json.loads((out / "result.json").read_bytes())["status"] == "failed"


def test_frontdoor_explicit_feedback_preserves_eight_episode_budget(project, monkeypatch):
    calls = mocked_worker(monkeypatch, ready=True)
    assert launcher.main(["--video", str(project / "video.webm"), "--person-only", "--train",
                          "--policy-family", "feedback-residual"]) == 0
    command = calls[0][0]
    assert command[command.index("--training-policy-family") + 1] == "feedback-residual"
    assert command[command.index("--training-generations") + 1] == "1"
    assert command[command.index("--training-population") + 1] == "4"
    assert "--serve" not in command and "--full-reference" in command


@pytest.mark.parametrize("mode", ["video", "inspect-build"])
def test_frontdoor_family_does_not_implicitly_enable_learning(project, monkeypatch, mode):
    monkeypatch.setattr(launcher, "_read", lambda *args: pytest.fail("Invalid options must fail before input reads"))
    with pytest.raises(SystemExit):
        launcher.main(["--" + mode, "missing", "--person-only", "--policy-family", "feedback-residual"])
    assert not (project / "artifacts").exists()


def test_build_cli_passes_opt_in_family_without_other_default_changes(tmp_path, monkeypatch, capsys):
    received = {}
    def build(*args, **options):
        received.update(options)
        return {"status": "ready"}
    monkeypatch.setattr(exercise_scene_pipeline, "prepare_exercise_scene", build)
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-build", "--person-only", "--video", "source",
        "--out", str(tmp_path / "out"), "--train", "--evaluation-seed", "6173",
        "--training-policy-family", "feedback-residual"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    assert received["train_settings"] == {"evaluation_seed": 6173, "policy_family": "feedback-residual"}


def test_build_cli_family_requires_explicit_training(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(exercise_scene_pipeline, "prepare_exercise_scene", lambda *args, **kw: pytest.fail("No implicit training"))
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-build", "--person-only", "--video", "source",
        "--out", str(tmp_path / "out"), "--training-policy-family", "feedback-residual"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "require --train" in capsys.readouterr().err


def test_direct_train_cli_dispatches_only_explicit_feedback_method(monkeypatch, capsys):
    received = {}
    def train(run, out, **options):
        received.update(run=run, out=out, **options)
        return {"status": "trained"}
    monkeypatch.setattr(feedback, "train_shared_exercise_feedback_policy", train)
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-train", "--run", "source", "--out", "out",
                                     "--policy-family", "feedback-residual"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "trained"
    assert received["generations"] == 2 and received["population"] == 8
    assert received["root_assistance"] is False


def test_policy_dispatch_is_versioned_and_reports_zero_anchor(monkeypatch, tmp_path):
    received = []
    policy = {"profile": actor.POLICY_PROFILE, "feedback_coefficients": [[0., 0.]] * 3}
    monkeypatch.setattr(actor, "validate_shared_exercise_feedback_policy", lambda *args: received.append("validate") or policy)
    monkeypatch.setattr(actor, "prepare_shared_exercise_feedback_policy", lambda *args: received.append("prepare") or "actor")
    monkeypatch.setattr(feedback, "evaluate_shared_exercise_feedback_policy", lambda *args, **kw: received.append("evaluate") or kw)
    assert policies.validate_policy(policy, {}, {}) == policy
    assert policies.prepare_policy(policy, object()) == "actor"
    assert policies.selection_kind(policy) == ("authored_unchanged", True, False)
    policy["feedback_coefficients"] = [[.1, 0.], [0., 0.], [0., 0.]]
    assert policies.selection_kind(policy) == ("searched_feedback_residuals", False, False)
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    assert policies.evaluate_policy(tmp_path, tmp_path / "new", seed=6173) == {"seed": 6173}
    assert received == ["validate", "prepare", "evaluate"]


@pytest.mark.parametrize("profile", ["shared-exercise-policy-v2", "shared-exercise-policy-v4"])
def test_direct_selection_cli_dispatches_sealed_selector_by_profile(tmp_path, monkeypatch, capsys, profile):
    from weft import shared_exercise_selection

    (tmp_path / "policy.json").write_text(json.dumps({"profile": profile}))
    calls = []
    def choose(run, out, **options):
        calls.append((run, out, options))
        return {"status": "trained", "selected_profile": profile}
    if profile.endswith("v4"):
        monkeypatch.setattr(feedback, "select_shared_exercise_feedback_policy", choose)
        monkeypatch.setattr(shared_exercise_selection, "select_feasible_shared_exercise_policy",
                            lambda *a, **k: pytest.fail("V4 must not use the phase selector"))
    else:
        monkeypatch.setattr(shared_exercise_selection, "select_feasible_shared_exercise_policy", choose)
        monkeypatch.setattr(feedback, "select_shared_exercise_feedback_policy",
                            lambda *a, **k: pytest.fail("V2 must keep its original selector"))
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-select", "--learning", str(tmp_path),
                                     "--out", str(tmp_path / "selected"), "--max-seconds", "12"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["selected_profile"] == profile
    assert calls == [(str(tmp_path), str(tmp_path / "selected"), {"max_seconds": 12.})]


@pytest.mark.parametrize("profile", ["shared-exercise-policy-v1", "shared-exercise-policy-v3", "arbitrary"])
def test_selection_rejects_other_profiles_without_optimizer(tmp_path, monkeypatch, profile):
    (tmp_path / "policy.json").write_text(json.dumps({"profile": profile}))
    monkeypatch.setattr(feedback, "select_shared_exercise_feedback_policy", lambda *a, **k: pytest.fail("Unsupported profile"))
    with pytest.raises(ValueError, match="selection supports|Unsupported"):
        policies.select_policy(tmp_path, tmp_path / "selected")
    assert not (tmp_path / "selected").exists()
