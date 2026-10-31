"""Versioned human-only policy dispatch without reinterpreting older artifacts."""


def _profile(policy):
    if not isinstance(policy, dict) or policy.get("profile") not in {
        "shared-exercise-policy-v1", "shared-exercise-policy-v2", "shared-exercise-policy-v3", "shared-exercise-policy-v4",
    }:
        raise ValueError("Unsupported human-only exercise policy profile")
    return policy["profile"]


def validate_policy(policy, scene, task):
    if isinstance(scene, dict) and scene.get("profile") == "shared-world-v4":
        raise ValueError("Static-support learning requires a separately qualified policy contract")
    # Lazy imports keep the learners' lab-helper dependency acyclic.
    if _profile(policy) == "shared-exercise-policy-v4":
        from weft.shared_exercise_feedback_residual import validate_shared_exercise_feedback_policy
        return validate_shared_exercise_feedback_policy(policy, scene, task)
    if _profile(policy) == "shared-exercise-policy-v1":
        from weft.shared_exercise_learning import validate_shared_exercise_policy
        return validate_shared_exercise_policy(policy, scene, task)
    if _profile(policy) == "shared-exercise-policy-v3":
        from weft.shared_exercise_feasibility_learning import validate_shared_exercise_feasibility_policy
        return validate_shared_exercise_feasibility_policy(policy, scene, task)
    from weft.shared_exercise_residual_learning import validate_shared_exercise_residual_policy
    return validate_shared_exercise_residual_policy(policy, scene, task)


def prepare_policy(policy, env):
    if env.scene.get("profile") == "shared-world-v4":
        raise ValueError("Static-support learning requires a separately qualified policy contract")
    if _profile(policy) == "shared-exercise-policy-v4":
        from weft.shared_exercise_feedback_residual import prepare_shared_exercise_feedback_policy
        return prepare_shared_exercise_feedback_policy(policy, env)
    if _profile(policy) == "shared-exercise-policy-v1":
        from weft.shared_exercise_learning import prepare_shared_exercise_policy
        return prepare_shared_exercise_policy(policy, env)
    if _profile(policy) == "shared-exercise-policy-v3":
        from weft.shared_exercise_feasibility_learning import prepare_shared_exercise_feasibility_policy
        return prepare_shared_exercise_feasibility_policy(policy, env)
    from weft.shared_exercise_residual_learning import prepare_shared_exercise_residual_policy
    return prepare_shared_exercise_residual_policy(policy, env)


def selection_kind(policy):
    """Describe an already validated policy, independently of training's claimed label."""
    if _profile(policy) == "shared-exercise-policy-v4":
        authored = all(value == 0 for row in policy["feedback_coefficients"] for value in row)
        return ("authored_unchanged" if authored else "searched_feedback_residuals"), authored, False
    if _profile(policy) in {"shared-exercise-policy-v2", "shared-exercise-policy-v3"}:
        authored = all(value == 0 for row in policy["residual_coefficients"] for value in row)
        return ("authored_unchanged" if authored else "searched_joint_residuals"), authored, False
    authored = all(value == 1 for row in policy["weights"] for value in row)
    coast = all(value == 0 for row in policy["weights"] for value in row)
    return ("authored_unchanged" if authored else "coast_unchanged" if coast else "searched_gains"), authored, coast


def evaluate_policy(learning_dir, out, **options):
    from pathlib import Path

    from weft.shared_exercise_lab import _json
    from weft.shared_world_packaging import read_package_file

    # The selected evaluator verifies the entire package before any physics is run.
    policy = _json(read_package_file(Path(learning_dir), "policy.json", max_bytes=64 * 1024))
    if _profile(policy) == "shared-exercise-policy-v4":
        from weft.shared_exercise_feedback_learning import evaluate_shared_exercise_feedback_policy
        return evaluate_shared_exercise_feedback_policy(learning_dir, out, **options)
    if _profile(policy) == "shared-exercise-policy-v1":
        from weft.shared_exercise_learning import evaluate_shared_exercise_policy
        return evaluate_shared_exercise_policy(learning_dir, out, **options)
    if _profile(policy) == "shared-exercise-policy-v3":
        from weft.shared_exercise_feasibility_evaluation import evaluate_shared_exercise_feasibility_policy
        return evaluate_shared_exercise_feasibility_policy(learning_dir, out, **options)
    from weft.shared_exercise_residual_learning import evaluate_shared_exercise_residual_policy
    return evaluate_shared_exercise_residual_policy(learning_dir, out, **options)


def select_policy(learning_dir, out, **options):
    from pathlib import Path

    from weft.shared_exercise_lab import _json
    from weft.shared_world_packaging import read_package_file

    policy = _json(read_package_file(Path(learning_dir), "policy.json", max_bytes=64 * 1024))
    if _profile(policy) == "shared-exercise-policy-v4":
        from weft.shared_exercise_feedback_learning import select_shared_exercise_feedback_policy
        return select_shared_exercise_feedback_policy(learning_dir, out, **options)
    if _profile(policy) == "shared-exercise-policy-v2":
        from weft.shared_exercise_selection import select_feasible_shared_exercise_policy
        return select_feasible_shared_exercise_policy(learning_dir, out, **options)
    raise ValueError("Saved-only selection supports phase-residual v2 or feedback-residual v4 training")
