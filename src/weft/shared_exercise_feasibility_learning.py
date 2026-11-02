"""Refine the same bounded actor using training feasibility, without re-admitting old policies."""

import ast
from copy import deepcopy
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np

from weft.io import canonical_hash, digest, local_time
from weft import shared_exercise_residual_learning as actor
from weft.shared_exercise_lab import _json, _runtime_versions, _verified_inputs
from weft.shared_exercise_learning import _bytes, _context, _integer, _output_path, _package, _score, _seconds, _sha, _write
from weft.shared_exercise_selection import BODY_GATES, EQUIPMENT_GATES, _history, _trial as validate_trial
from weft.shared_exercise_task import GATES, GATES_SHA256, SharedExerciseTaskEnv, prepare_shared_exercise_snapshot_validator
from weft.shared_world import HUMAN_ONLY_PROFILE
from weft.shared_world_packaging import collect_shared_source_files, read_package_file, read_package_manifest


POLICY_PROFILE = "shared-exercise-policy-v3"
ALGORITHM = "shared_exercise_phase_feasibility_cem_v1"
ACTOR_ALGORITHM = "shared_exercise_phase_residual_actor_v1"
RANK_PROFILE = "shared-exercise-feasibility-rank-v1"
IMPORT_PROFILE = "shared-exercise-original-generation-import-v1"
ARCHIVE = "imported-training"
POLICY_RUNTIME_FILES = (*actor.POLICY_RUNTIME_FILES, "shared_exercise_selection.py",
                       "shared_exercise_feasibility_learning.py", "shared_exercise_feasibility_evaluation.py")
RULE = ("Complete candidates passing every unchanged gate on both original training starts first; otherwise "
        "minimize maximum positive normalized gate excess, then mean excess, then maximize original mean reward. "
        "Incomplete candidates are ineligible; stable candidate order breaks exact ties.")
SCOPE = ("Training-only feasibility-first refinement of unchanged bounded phase-conditioned joint residuals. "
         "Authored reference feedback, root assistance, simulator state and bias forces remain privileged; "
         "equipment forces remain forbidden. Original reward and all task gates are unchanged. Imported original "
         "training trials are verified historical evidence, not permission to execute their old policy. "
         "This is not new-video generalization, recovered grasp, measured muscles or physical validation.")
MAX_NEW_EPISODES = 12
MAX_NEW_CONTROL_STEPS = 3048
MAX_IMPORTED_EPISODES = 16
MAX_IMPORTED_CONTROL_STEPS = 4064
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_FILES = 512
_HEX = re.compile(r"[0-9a-f]{64}")
_ACTOR_FUNCTIONS = ("_trial", "_action", "phase_basis", "_coefficients")


def _runtime_hashes(scene=None):
    names = POLICY_RUNTIME_FILES
    if scene is not None and "native_clock" in scene.get("human_spec", {}):
        names = (*names, "source_clock.py")
    return {name: digest(Path(__file__).parent / name) for name in names}


def _scene_runtime_hashes(scene):
    return _runtime_hashes(scene) if "native_clock" in scene.get("human_spec", {}) else _runtime_hashes()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _excess(value, threshold, *, lower=False, tolerance=1e-12):
    if not _finite(value) or value < 0 or not _finite(threshold) or threshold <= 0:
        raise ValueError("Feasibility ranks require finite nonnegative measurements and positive fixed gates")
    delta = threshold - value if lower else value - threshold
    result = delta / threshold if delta > tolerance else 0.
    if not math.isfinite(result):
        raise ValueError("Feasibility normalized excess exceeds finite bounds")
    return result


def optimization_rank(trials):
    """Rank validated training outcomes; reward is retained, never relabeled as feasibility."""
    if not isinstance(trials, list) or len(trials) != 2:
        raise ValueError("Feasibility ranking requires both original training starts")
    score = _score(trials)
    if not _finite(score):
        raise ValueError("Feasibility ranking requires finite original reward")
    complete = all(row.get("status") == "completed" and row.get("metrics") is not None
                   and row["metrics"].get("completed_interval") is True and row["metrics"].get("failure") is None
                   for row in trials)
    terms = []
    if complete:
        for trial in trials:
            metrics = trial["metrics"]
            terms.extend(_excess(metrics[key], GATES[key]) for key in BODY_GATES)
            for equipment in metrics["equipment"]:
                terms.extend(_excess(equipment[key], GATES["equipment_" + key]) for key in EQUIPMENT_GATES)
            for contact in metrics["contact_requirements"]:
                required, observed = contact["required_duration_s"], contact["observed_duration_s"]
                if not _finite(required) or required <= 0 or not _finite(observed) or observed < 0:
                    raise ValueError("Feasibility contact coverage requires a positive declared duration")
                terms.extend((0. if math.isclose(observed, required, abs_tol=1e-8) else abs(observed - required) / required,
                              _excess(contact["qualifying_fraction"], GATES["contact_duration_fraction"], lower=True),
                              _excess(contact["max_gap_s"], GATES["max_contact_gap_s"])))
    feasible = complete and all(row["metrics"]["success"] for row in trials)
    if complete and feasible != (not any(terms)):
        raise ValueError("Feasibility terms disagree with the unchanged task gates")
    return {"profile": RANK_PROFILE, "complete": complete, "feasible": feasible,
            "max_normalized_excess": max(terms) if terms else None,
            "mean_normalized_excess": float(np.mean(terms)) if terms else None,
            "violated_terms": sum(value > 0 for value in terms), "gate_terms": len(terms), "reward_score": score}


def _rank_key(rank):
    return (0 if rank["feasible"] else 1 if rank["complete"] else 2,
            rank["max_normalized_excess"] or 0., rank["mean_normalized_excess"] or 0., -rank["reward_score"])


def _validate_rank(rank, scene, task, score):
    keys = {"profile", "complete", "feasible", "max_normalized_excess", "mean_normalized_excess",
            "violated_terms", "gate_terms", "reward_score"}
    count = 2 * (len(BODY_GATES) + len(EQUIPMENT_GATES) * len(task["equipment_ids"]) + 3 * len(task["contact_requirements"]))
    if (not isinstance(rank, dict) or set(rank) != keys or rank["profile"] != RANK_PROFILE
            or rank["complete"] is not True or type(rank["feasible"]) is not bool
            or type(rank["gate_terms"]) is not int or rank["gate_terms"] != count
            or type(rank["violated_terms"]) is not int or not 0 <= rank["violated_terms"] <= count
            or not all(_finite(rank[key]) and rank[key] >= 0 for key in ("max_normalized_excess", "mean_normalized_excess"))
            or rank["mean_normalized_excess"] > rank["max_normalized_excess"]
            or rank["feasible"] != (rank["violated_terms"] == 0)
            or rank["feasible"] != (rank["max_normalized_excess"] == 0)
            or not _finite(rank["reward_score"]) or rank["reward_score"] != score):
        raise ValueError("Selected v3 policy requires an eligible, consistent frozen feasibility rank")


def validate_shared_exercise_feasibility_policy(policy, scene, task):
    scene, task, count = _context(scene, task)
    if scene["profile"] == HUMAN_ONLY_PROFILE:
        raise ValueError("Person-only scenes require the v2 residual learner, not the v3 feasibility policy")
    keys = {"schema_version", "profile", "algorithm", "actor_algorithm", "scene_sha256", "task_sha256", "evaluator_sha256",
            "gates_sha256", "runtime_sha256", "runtime_versions", "action_names", "feature_names", "residual_coefficients",
            "residual_bound", "basis_profile", "root_assistance", "training_seed", "training_score", "optimization_rank",
            "selection_objective", "source_training_manifest_sha256", "import_proof_sha256", "privileged_simulator_features", "scope"}
    if (not isinstance(policy, dict) or set(policy) != keys or type(policy["schema_version"]) is not int
            or policy["schema_version"] != 1 or policy["profile"] != POLICY_PROFILE or policy["algorithm"] != ALGORITHM
            or policy["actor_algorithm"] != ACTOR_ALGORITHM or len(_bytes(policy)) > 64 * 1024):
        raise ValueError("Feasibility policy differs from its explicit v3 actor/trainer contract")
    actor._coefficients(policy["residual_coefficients"])
    hashes = _scene_runtime_hashes(scene)
    if (policy["scene_sha256"] != canonical_hash(scene) or policy["task_sha256"] != canonical_hash(task)
            or policy["gates_sha256"] != GATES_SHA256 or policy["runtime_sha256"] != hashes
            or policy["evaluator_sha256"] != hashes["shared_exercise_task.py"] or policy["runtime_versions"] != _runtime_versions()):
        raise ValueError("Feasibility policy task, scene, evaluator or complete current runtime differs")
    if (policy["action_names"] != scene["action_contract"]["names"][:count] or policy["feature_names"] != list(actor.FEATURE_NAMES)
            or policy["basis_profile"] != actor.BASIS_PROFILE or type(policy["residual_bound"]) not in (int, float)
            or policy["residual_bound"] != actor.RESIDUAL_BOUND or policy["root_assistance"] != task["control"]["root_assistance"]
            or policy["privileged_simulator_features"] is not True or policy["scope"] != SCOPE
            or policy["selection_objective"] != RULE):
        raise ValueError("Feasibility policy cannot change the actor, assistance, objective or privilege disclosure")
    _integer(policy["training_seed"], "training seed", 0, 2**31 - 1)
    if not _finite(policy["training_score"]):
        raise ValueError("Feasibility policy must retain finite original reward")
    for name in ("source_training_manifest_sha256", "import_proof_sha256"):
        if not isinstance(policy[name], str) or not _HEX.fullmatch(policy[name]):
            raise ValueError("Feasibility policy requires exact original training and import-proof hashes")
    _validate_rank(policy["optimization_rank"], scene, task, policy["training_score"])
    return deepcopy(policy)


def prepare_shared_exercise_feasibility_policy(policy, env):
    policy = validate_shared_exercise_feasibility_policy(policy, env.scene, env.task)
    if (policy["evaluator_sha256"] != env.evaluator_sha256
            or env.observation_contract()["profile"] != "shared-exercise-observation-v1"):
        raise ValueError("Feasibility actor requires the unchanged task evaluator and observation contract")
    coefficients = actor._coefficients(policy["residual_coefficients"])
    return lambda: actor._action(coefficients, env)


def _actor_ast(raw):
    try:
        tree = ast.parse(raw)
    except (ValueError, SyntaxError, UnicodeError) as error:
        raise ValueError("Imported actor implementation is not valid Python source") from error
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if not set(_ACTOR_FUNCTIONS) <= set(functions):
        raise ValueError("Imported actor lacks the complete direct training-action path")
    names = {node.id for node in ast.walk(functions["_trial"]) if isinstance(node, ast.Name)}
    if "_action" not in names or names & {"prepare_policy", "policy_action", "validate_policy"}:
        raise ValueError("Imported training did not directly use the frozen actor")
    return {name: _sha(ast.dump(functions[name], include_attributes=False).encode()) for name in _ACTOR_FUNCTIONS}


def _verify_original(manifest_raw, files, deadline):
    manifest = _json(manifest_raw)
    if (not isinstance(manifest, dict) or manifest.get("status") != "trained" or not isinstance(manifest.get("files"), dict)
            or set(manifest["files"]) != set(files) or any(_sha(files[name]) != expected for name, expected in manifest["files"].items())
            or any(name.split("/")[0] in {"selection.json", "selection-source", ARCHIVE, "import-proof.json"}
                   or any(part in {"evaluation.json", "pre-evaluation-freeze.json", "comparison.json"}
                          or part.startswith("policy-evaluation") for part in Path(name).parts) for name in files)):
        raise ValueError("Import accepts only a complete original trained package, never selection or evaluation evidence")
    required = {"scene.json", "exercise-task.json", "policy.json", "settings.json", "training.json", "input-manifest.json"}
    if not required <= set(files) or len(files["policy.json"]) > 64 * 1024:
        raise ValueError("Import requires complete bounded original training documents")
    scene, task, count = _context(_json(files["scene.json"]), _json(files["exercise-task.json"]))
    if scene["profile"] == HUMAN_ONLY_PROFILE:
        raise ValueError("Person-only scenes require the v2 residual learner, not the v3 feasibility trainer")
    policy = _json(files["policy.json"])
    hashes = policy.get("runtime_sha256") if isinstance(policy, dict) else None
    if not isinstance(hashes, dict) or set(hashes) != set(actor._scene_runtime_hashes(scene)):
        raise ValueError("Import requires the original v2 complete runtime closure")
    same = {}
    for name, expected in hashes.items():
        actor._check(deadline)
        if (not isinstance(expected, str) or not _HEX.fullmatch(expected)
                or "implementation/" + name not in files or _sha(files["implementation/" + name]) != expected):
            raise ValueError("Imported original runtime bytes differ from their pinned identities")
        if name != "shared_exercise_policies.py":
            actual = digest(Path(__file__).parent / name)
            if actual != expected:
                raise ValueError("Historical trials cannot be imported after a physics, reset or actor runtime change")
            same[name] = actual
    original_ast = _actor_ast(files["implementation/shared_exercise_residual_learning.py"])
    if original_ast != _actor_ast(Path(actor.__file__).read_bytes()):
        raise ValueError("Imported and current actor ASTs differ")
    metadata = SimpleNamespace(scene=scene, scene_sha256=canonical_hash(scene), task_sha256=canonical_hash(task),
        evaluator_sha256=hashes["shared_exercise_task.py"], action_names=scene["action_contract"]["names"][:count], task=task)
    coefficients = actor._coefficients(policy.get("residual_coefficients"))
    _integer(policy.get("training_seed"), "original training seed", 0, 2**31 - 1)
    if not _finite(policy.get("training_score")) or policy != actor._policy(metadata, coefficients,
            policy["training_seed"], policy["training_score"], hashes):
        raise ValueError("Import document is not the exact original reward-trained v2 policy contract")
    settings, training, candidates, authored_score = _history(files, scene, task, policy, deadline)
    if (settings["generations"] != 2 or settings["population"] != 8 or not 1 <= settings["horizon_steps"] <= 254
            or training["policy_sha256"] != _sha(files["policy.json"])):
        raise ValueError("Import requires the original two-generation/eight-candidate search and at most 254 fixed steps")
    authored = not bool(np.any(coefficients))
    if (training["selected_authored_baseline"] is not authored or training["selected_coast_baseline"] is not False
            or training["selected_feedback_kind"] != ("authored_unchanged" if authored else "searched_joint_residuals")
            or training["training_score_improved_over_authored"] is not (policy["training_score"] > authored_score)):
        raise ValueError("Original training selection labels differ from its reward-selected coefficients")
    first = training["history"][0]["candidates"]
    if any(row["cached"] or not all(trial["metrics"] is not None and trial["metrics"]["completed_interval"]
                                  for trial in row["trials"]) for row in first):
        raise ValueError("Import requires sixteen complete original first-generation episodes; no implicit rerun")
    proof = {"schema_version": 1, "profile": IMPORT_PROFILE, "original_training_manifest_sha256": _sha(manifest_raw),
        "original_policy_sha256": _sha(files["policy.json"]), "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
        "original_training_algorithm": actor.ALGORITHM, "new_training_algorithm": ALGORITHM,
        "actor_algorithm": ACTOR_ALGORITHM, "actor_ast_sha256": original_ast,
        "unchanged_training_execution_runtime_sha256": same, "runtime_versions": _runtime_versions(),
        "original_dispatcher_sha256": hashes["shared_exercise_policies.py"],
        "current_dispatcher_sha256": digest(Path(__file__).with_name("shared_exercise_policies.py")),
        "dispatcher_scope": "Original training _trial calls _action directly. Archived dispatcher is not executed or admitted; current v3 policies require the complete current runtime.",
        "imported_generation": 0, "imported_candidates": 8, "imported_episodes": 16,
        "imported_control_steps": sum(trial["control_steps"] for row in first for trial in row["trials"]),
        "imported_candidate_sha256": [canonical_hash(row["residual_coefficients"]) for row in first],
        "original_full_history_verified": True, "rng_history_verified": True, "original_resets_verified": True,
        "new_physics_steps_during_import": 0, "evaluation_used": False, "selection_objective": RULE,
        "archive": ARCHIVE, "archive_files": len(files), "archive_bytes": sum(map(len, files.values())) + len(manifest_raw)}
    return scene, task, policy, settings, training, authored_score, proof


def _verify_refinement(files, scene, task, original_policy, original_settings, original_training, proof):
    required = {"policy.json", "training.json", "settings.json", "input-manifest.json"}
    if not required <= set(files):
        raise ValueError("Feasibility package lacks its selected policy and complete training history")
    policy = validate_shared_exercise_feasibility_policy(_json(files["policy.json"]), scene, task)
    settings, training = _json(files["settings.json"]), _json(files["training.json"])
    pins = {"algorithm": ALGORITHM, "actor_algorithm": ACTOR_ALGORITHM, "selection_objective": RULE,
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "gates_sha256": GATES_SHA256,
        "runtime_sha256": policy["runtime_sha256"], "runtime_versions": _runtime_versions(),
        "source_manifest_sha256": proof["original_training_manifest_sha256"],
        "source_training_manifest_sha256": proof["original_training_manifest_sha256"],
        "import_proof_sha256": _sha(files["import-proof.json"]), "seed": original_settings["seed"],
        "training_seeds": original_settings["training_seeds"], "training_reset_options": original_settings["training_reset_options"],
        "horizon_steps": task["interval"]["horizon_steps"], "generations": 2, "population": 8, "elite_count": 2,
        "initial_std": .08, "minimum_std": .02, "coefficient_shape": [2, 19], "basis_profile": actor.BASIS_PROFILE,
        "residual_bound": actor.RESIDUAL_BOUND, "root_residual_channels": [], "equipment_force_channels": [],
        "candidate_anchors": ["zero_residual_authored", "previous_feasibility_best"], "maximum_new_episodes": 12,
        "maximum_control_steps": 12 * task["interval"]["horizon_steps"],
        "maximum_physics_substeps": 12 * task["interval"]["horizon_steps"] * scene["substeps"],
        "imported_episodes": 16, "imported_control_steps": proof["imported_control_steps"],
        "maximum_output_bytes": MAX_OUTPUT_BYTES, "workers": 1, "external_api_usd": 0, "evaluation_used": False,
        "source_scope": task["source_scope"], "scope": SCOPE}
    if (not isinstance(settings, dict) or set(settings) != {*pins, "schema_version", "created_at", "max_seconds"}
            or settings.get("schema_version") != 1 or any(settings.get(name) != value for name, value in pins.items())
            or _sha(files["input-manifest.json"]) != proof["original_training_manifest_sha256"]
            or policy["source_training_manifest_sha256"] != proof["original_training_manifest_sha256"]
            or policy["import_proof_sha256"] != _sha(files["import-proof.json"])
            or policy["training_seed"] != original_settings["seed"]):
        raise ValueError("Feasibility settings/policy differ from the frozen objective, budget or original import proof")
    _seconds(settings["max_seconds"], 60)
    if (not isinstance(training, dict) or training.get("status") != "trained" or training.get("algorithm") != ALGORITHM
            or training.get("policy_sha256") != _sha(files["policy.json"]) or training.get("completed_generations") != 2
            or training.get("failure") is not None or training.get("active_candidates") != []
            or not isinstance(training.get("history"), list) or len(training["history"]) != 2
            or not _finite(training.get("seconds")) or not 0 <= training["seconds"] <= settings["max_seconds"]):
        raise ValueError("Feasibility package requires a completed, bounded two-generation training history")
    initial_rows = []
    for original_row in original_training["history"][0]["candidates"]:
        row = deepcopy(original_row)
        row.update(origin="imported_generation_zero", optimization_rank=optimization_rank(row["trials"]))
        initial_rows.append(row)
    selected = _selected(initial_rows)
    elites = sorted(initial_rows, key=lambda row: _rank_key(row["optimization_rank"]))[:2]
    elite = np.asarray([row["residual_coefficients"] for row in elites])
    mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), .02)
    expected_first = {"generation": 0, "origin": "imported_original_trials_re_ranked", "candidates": initial_rows,
        "next_mean": mean.tolist(), "next_std": std.tolist(), "selected_reward_score": selected["score"],
        "selected_optimization_rank": selected["optimization_rank"]}
    if training["history"][0] != expected_first:
        raise ValueError("Feasibility imported generation or elite update differs from its exact original evidence")
    rng = np.random.default_rng(original_settings["seed"])
    rng.normal(np.zeros((2, 19)), np.full((2, 19), .08), (8, 2, 19))
    samples = np.clip(rng.normal(mean, std, (8, 2, 19)), -actor.RESIDUAL_BOUND, actor.RESIDUAL_BOUND)
    samples[0], samples[1] = 0., selected["residual_coefficients"]
    cache = {canonical_hash(row["residual_coefficients"]): row for row in initial_rows}
    record = training["history"][1]
    if (not isinstance(record, dict) or set(record) != {"generation", "origin", "candidates", "selected_reward_score", "selected_optimization_rank"}
            or record["generation"] != 1 or record["origin"] != "new_feasibility_refinement"
            or not isinstance(record["candidates"], list) or len(record["candidates"]) != 8):
        raise ValueError("Feasibility refinement generation is missing candidates")
    initial = {row["seed"]: row for row in initial_rows[0]["trials"]}
    episodes = steps = hits = 0
    for index, (coefficients, row) in enumerate(zip(samples, record["candidates"], strict=True)):
        key = canonical_hash(coefficients.tolist())
        if key in cache:
            expected = deepcopy(cache[key])
            expected.update(candidate=index, cached=True, origin="verified_training_cache")
            if row != expected:
                raise ValueError("Feasibility cached candidate differs from its exact earlier training trials")
            hits += 1
            continue
        keys = {"candidate", "residual_coefficients", "trials", "complete", "cached", "origin", "score", "loss", "optimization_rank"}
        if (not isinstance(row, dict) or set(row) != keys or row["candidate"] != index or row["residual_coefficients"] != coefficients.tolist()
                or row["complete"] is not True or row["cached"] is not False or row["origin"] != "new_training"
                or not isinstance(row["trials"], list) or len(row["trials"]) != 2):
            raise ValueError("Feasibility new candidate differs from the frozen RNG and same bounded action family")
        for seed, reset, trial in zip(settings["training_seeds"], actor.TRAIN_RESETS, row["trials"], strict=True):
            validate_trial(trial, scene, task, original_policy, seed, reset)
            if (trial["initial_state_sha256"] != initial[seed]["initial_state_sha256"]
                    or trial.get("initial_metrics") != initial[seed]["initial_metrics"]):
                raise ValueError("Feasibility candidate changed the original training reset state")
            episodes += 1
            steps += trial["control_steps"]
        if row["score"] != _score(row["trials"]) or row["loss"] != -row["score"] or row["optimization_rank"] != optimization_rank(row["trials"]):
            raise ValueError("Feasibility candidate reward or optimization rank differs from its fixed-gate metrics")
        cache[key] = row
    selected = _selected(list(cache.values()))
    authored = not bool(np.any(selected["residual_coefficients"]))
    authored_score = initial_rows[0]["score"]
    expected = {"imported_episodes": 16, "imported_control_steps": proof["imported_control_steps"],
        "evaluated_episodes": episodes, "control_steps": steps, "cache_hits": hits,
        "selected_optimization_rank": selected["optimization_rank"], "selected_reward_score": selected["score"],
        "selection_objective": RULE, "import_proof_sha256": _sha(files["import-proof.json"]),
        "selected_authored_baseline": authored, "selected_coast_baseline": False,
        "selected_feedback_kind": "authored_unchanged" if authored else "searched_joint_residuals",
        "improvement_over_authored_training": selected["score"] - authored_score,
        "training_score_improved_over_authored": selected["score"] > authored_score}
    if (any(training.get(name) != value for name, value in expected.items()) or episodes > MAX_NEW_EPISODES or steps > MAX_NEW_CONTROL_STEPS
            or record["selected_reward_score"] != selected["score"] or record["selected_optimization_rank"] != selected["optimization_rank"]
            or policy["training_score"] != selected["score"] or policy["optimization_rank"] != selected["optimization_rank"]
            or policy["residual_coefficients"] != selected["residual_coefficients"]):
        raise ValueError("Feasibility selected policy, rank or budget differs from the complete training-only search")
    return policy


def verify_imported_policy_baseline(files, scene, task):
    """Attest original coefficients for a labeled comparison; never prepare/admit their old v2 policy."""
    archived = {name[len(ARCHIVE) + 1:]: raw for name, raw in files.items() if name.startswith(ARCHIVE + "/")}
    manifest_raw = archived.pop("manifest.json", None)
    if manifest_raw is None or "import-proof.json" not in files:
        raise ValueError("Original comparison baseline requires its complete imported archive and proof")
    old_scene, old_task, policy, settings, training, _, proof = _verify_original(manifest_raw, archived, time.monotonic() + 15)
    if old_scene != scene or old_task != task or _json(files["import-proof.json"]) != proof:
        raise ValueError("Original comparison baseline scene/task or execution-equivalence proof differs")
    _verify_refinement(files, scene, task, policy, settings, training, proof)
    return {"policy": policy, "policy_sha256": proof["original_policy_sha256"],
        "training_manifest_sha256": proof["original_training_manifest_sha256"], "actor_algorithm": ACTOR_ALGORITHM,
        "training_algorithm": actor.ALGORITHM, "source_path": ARCHIVE + "/policy.json",
        "import_proof_sha256": _sha(files["import-proof.json"]),
        "actor_equivalence": {"actor_ast_sha256": proof["actor_ast_sha256"],
            "unchanged_training_execution_runtime_sha256": proof["unchanged_training_execution_runtime_sha256"],
            "old_policy_admitted": False, "comparison_method": "Original coefficients through the current identical actor, not old-policy runtime migration"}}


class _Writer:
    def __init__(self, out):
        self.out = out
        self.sizes = {path.relative_to(out).as_posix(): path.stat().st_size for path in out.rglob("*") if path.is_file()}

    def write(self, name, value):
        size = len(_bytes(value))
        if (size > 4 * 1024 * 1024 or sum(self.sizes.values()) - self.sizes.get(name, 0) + size > MAX_OUTPUT_BYTES - 65536
                or len(set(self.sizes) | {name}) >= MAX_OUTPUT_FILES):
            raise ValueError("Feasibility metadata would exceed reserved output bounds; refused before write")
        _write(self.out, name, value)
        self.sizes[name] = size


def _selected(rows):
    eligible = [row for row in rows if row["optimization_rank"]["complete"]]
    if not eligible:
        raise ValueError("No complete candidate is eligible; no policy can be published")
    return min(eligible, key=lambda row: _rank_key(row["optimization_rank"]))


def _policy(env, selected, seed, hashes, proof):
    policy = actor._policy(env, actor._coefficients(selected["residual_coefficients"]), seed, selected["score"], hashes)
    policy.update(profile=POLICY_PROFILE, algorithm=ALGORITHM, actor_algorithm=ACTOR_ALGORITHM, scope=SCOPE,
                  selection_objective=RULE, optimization_rank=deepcopy(selected["optimization_rank"]),
                  source_training_manifest_sha256=proof["original_training_manifest_sha256"], import_proof_sha256=_sha(_bytes(proof)))
    return policy


def train_shared_exercise_feasibility_policy(original_training_dir, out, *, max_seconds=60):
    """Import generation zero; spend at most twelve new full-interval training episodes."""
    _seconds(max_seconds, 60)
    source, out = Path(original_training_dir).absolute(), _output_path(original_training_dir, out)
    started, deadline = time.monotonic(), time.monotonic() + max_seconds
    manifest_raw, names = read_package_manifest(source)
    if any(any(part in {"evaluation.json", "pre-evaluation-freeze.json", "comparison.json"}
               or part.startswith("policy-evaluation") for part in Path(name).parts) for name in names):
        raise ValueError("Feasibility import cannot read evaluation evidence")
    verified, files = _verified_inputs(source, deadline=deadline)
    if verified != manifest_raw:
        raise ValueError("Original training manifest changed during import")
    scene, task, original_policy, old_settings, original, authored_score, proof = _verify_original(manifest_raw, files, deadline)
    source_proof = collect_shared_source_files(scene, source, expected_manifest_sha256=_sha(manifest_raw))
    source_proof.validate_layout(reserved_paths={ARCHIVE, "import-proof.json"})
    hashes, horizon = _scene_runtime_hashes(scene), task["interval"]["horizon_steps"]
    maximum = MAX_NEW_EPISODES * horizon
    if maximum > MAX_NEW_CONTROL_STEPS or proof["imported_control_steps"] > MAX_IMPORTED_CONTROL_STEPS:
        raise ValueError("Feasibility refinement exceeds its new/imported action bounds")
    extra = {ARCHIVE + "/manifest.json": manifest_raw, **{ARCHIVE + "/" + name: raw for name, raw in files.items()},
             "import-proof.json": _bytes(proof), **actor._source_views(source, manifest_raw, scene)}
    _package(out, files["scene.json"], files["exercise-task.json"], manifest_raw, source_proof, extra=extra)
    writer = _Writer(out)
    settings = {"schema_version": 1, "algorithm": ALGORITHM, "actor_algorithm": ACTOR_ALGORITHM, "created_at": local_time(),
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "gates_sha256": GATES_SHA256,
        "runtime_sha256": hashes, "runtime_versions": _runtime_versions(), "source_manifest_sha256": _sha(manifest_raw),
        "source_training_manifest_sha256": _sha(manifest_raw), "import_proof_sha256": _sha(_bytes(proof)),
        "seed": old_settings["seed"], "training_seeds": old_settings["training_seeds"],
        "training_reset_options": old_settings["training_reset_options"], "horizon_steps": horizon,
        "generations": 2, "population": 8, "elite_count": 2, "initial_std": .08, "minimum_std": .02,
        "coefficient_shape": [2, 19], "basis_profile": actor.BASIS_PROFILE, "residual_bound": actor.RESIDUAL_BOUND,
        "root_residual_channels": [], "equipment_force_channels": [], "selection_objective": RULE,
        "candidate_anchors": ["zero_residual_authored", "previous_feasibility_best"],
        "maximum_new_episodes": MAX_NEW_EPISODES, "maximum_control_steps": maximum,
        "maximum_physics_substeps": maximum * scene["substeps"], "imported_episodes": 16,
        "imported_control_steps": proof["imported_control_steps"], "maximum_output_bytes": MAX_OUTPUT_BYTES,
        "max_seconds": max_seconds, "workers": 1, "external_api_usd": 0, "evaluation_used": False,
        "source_scope": task["source_scope"], "scope": SCOPE}
    training = {"status": "running", "algorithm": ALGORITHM, "history": [], "active_candidates": [],
        "imported_episodes": 16, "imported_control_steps": proof["imported_control_steps"], "evaluated_episodes": 0,
        "control_steps": 0, "cache_hits": 0, "completed_generations": 0, "failure": None}
    env, budget = None, actor._Budget(maximum)
    try:
        writer.write("settings.json", settings)
        rows, cache = [], {}
        for original_row in original["history"][0]["candidates"]:
            row = deepcopy(original_row)
            row.update(origin="imported_generation_zero", optimization_rank=optimization_rank(row["trials"]))
            rows.append(row)
            cache[canonical_hash(row["residual_coefficients"])] = row
        selected = _selected(rows)
        elites = sorted(rows, key=lambda row: _rank_key(row["optimization_rank"]))[:2]
        elite = np.asarray([row["residual_coefficients"] for row in elites])
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), .02)
        training["history"].append({"generation": 0, "origin": "imported_original_trials_re_ranked",
            "candidates": rows, "next_mean": mean.tolist(), "next_std": std.tolist(),
            "selected_reward_score": selected["score"], "selected_optimization_rank": selected["optimization_rank"]})
        training["completed_generations"] = 1
        writer.write("training.json", training)
        rng = np.random.default_rng(settings["seed"])
        rng.normal(np.zeros((2, 19)), np.full((2, 19), .08), (8, 2, 19))
        samples = np.clip(rng.normal(mean, std, (8, 2, 19)), -actor.RESIDUAL_BOUND, actor.RESIDUAL_BOUND)
        samples[0], samples[1] = 0., selected["residual_coefficients"]
        env = SharedExerciseTaskEnv(scene, task)
        validator = prepare_shared_exercise_snapshot_validator(scene, task)
        initial = {row["seed"]: row for row in rows[0]["trials"]}
        active = training["active_candidates"]
        for index, coefficients in enumerate(samples):
            actor._check(deadline)
            key = canonical_hash(coefficients.tolist())
            if key in cache:
                row = deepcopy(cache[key])
                row.update(candidate=index, cached=True, origin="verified_training_cache")
                active.append(row)
                training["cache_hits"] += 1
            else:
                row = {"candidate": index, "residual_coefficients": coefficients.tolist(), "trials": [],
                       "complete": False, "cached": False, "origin": "new_training"}
                active.append(row)
                for seed, reset in zip(settings["training_seeds"], actor.TRAIN_RESETS, strict=True):
                    if training["evaluated_episodes"] >= MAX_NEW_EPISODES:
                        raise ValueError("Feasibility new-episode budget exhausted; no retries")
                    training["evaluated_episodes"] += 1
                    try:
                        trial = actor._trial(env, "learned", seed, reset, coefficients, deadline, budget, validator=validator)
                    except Exception as error:
                        if hasattr(error, "partial_trial"):
                            row["trials"].append(error.partial_trial)
                        raise
                    row["trials"].append(trial)
                    validate_trial(trial, scene, task, original_policy, seed, reset)
                    if (trial["initial_state_sha256"] != initial[seed]["initial_state_sha256"]
                            or trial.get("initial_metrics") != initial[seed]["initial_metrics"]):
                        raise ValueError("New training reset differs from the verified original initial state")
                    training["control_steps"] = budget.used
                score = _score(row["trials"])
                row.update(score=score, loss=-score, complete=True, optimization_rank=optimization_rank(row["trials"]))
                cache[key] = deepcopy(row)
            selected = _selected(list(cache.values()))
            writer.write("training.json", training)
        training["history"].append({"generation": 1, "origin": "new_feasibility_refinement", "candidates": active,
            "selected_reward_score": selected["score"], "selected_optimization_rank": selected["optimization_rank"]})
        training.update(active_candidates=[], completed_generations=2)
        actor._input_unchanged(source, manifest_raw, deadline)
        if _scene_runtime_hashes(scene) != hashes:
            raise ValueError("Feasibility actor/evaluator/optimizer runtime changed during training")
        policy = _policy(env, selected, settings["seed"], hashes, proof)
        validate_shared_exercise_feasibility_policy(policy, scene, task)
        writer.write("policy.json", policy)
        authored = not bool(np.any(selected["residual_coefficients"]))
        training.update(status="trained", seconds=time.monotonic() - started, policy_sha256=digest(out / "policy.json"),
            selected_authored_baseline=authored, selected_coast_baseline=False,
            selected_feedback_kind="authored_unchanged" if authored else "searched_joint_residuals",
            improvement_over_authored_training=selected["score"] - authored_score,
            training_score_improved_over_authored=selected["score"] > authored_score,
            selected_optimization_rank=selected["optimization_rank"], selected_reward_score=selected["score"],
            selection_objective=RULE, import_proof_sha256=_sha(_bytes(proof)))
        writer.write("training.json", training)
        seal = actor._seal(out, "trained", deadline, maximum_bytes=MAX_OUTPUT_BYTES, maximum_files=MAX_OUTPUT_FILES)
        return {"directory": str(out), "manifest_sha256": seal, **training}
    except Exception as error:
        training.update(status="failed", control_steps=budget.used, seconds=time.monotonic() - started,
                        failure=f"{type(error).__name__}: {error}"[:2000])
        actor._failure(out, "training.json", training, error)
        raise
    finally:
        if env is not None:
            env.close()
