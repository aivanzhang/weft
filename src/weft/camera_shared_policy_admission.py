"""Admit an unchanged research actor through explicitly pinned integration-only changes."""

import ast
from hashlib import sha256
import math
from pathlib import Path
import time

from weft import camera_shared_policy as actor
from weft.camera_shared_policy_receipt import (
    ADMISSION_PROFILE, ALLOWED_CHANGED_FILES, COMPATIBILITY_PROFILE, EXECUTION_RUNTIME_REQUIRED,
    ORIGINAL_RUNTIME_REQUIRED, SCOPE, _json, _runtime,
    policy_runtime_requirements,
    validate_camera_policy_admission as validate_saved_admission,
)
from weft.camera_shared_receipt import IDENTITIES
from weft.io import canonical_hash


RUNTIME_FILES = tuple(sorted(EXECUTION_RUNTIME_REQUIRED - ORIGINAL_RUNTIME_REQUIRED))
MAX_RUNTIME_BYTES = 16 << 20
MAX_SOURCE_BYTES = 2 << 20


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_deadline(deadline):
    _require(type(deadline) in (int, float) and math.isfinite(deadline), "Camera admission requires a finite deadline")
    if time.monotonic() >= deadline:
        raise TimeoutError("Camera policy admission exceeded its cooperative deadline")


def _runtime_sources(scene):
    _require(isinstance(scene, dict) and scene.get("profile") == "shared-world-v6",
             "Camera policy execution requires its actual camera world")
    root, total, sources = Path(__file__).parent, 0, {}
    _, required = policy_runtime_requirements(scene)
    _require(len(required) <= 96, "Camera execution runtime exceeds its file bound")
    for name in sorted(required):
        path = root / name
        _require(path.is_file() and not path.is_symlink(), "Camera execution runtime needs regular source files")
        size = path.stat().st_size
        _require(size <= MAX_SOURCE_BYTES and total + size <= MAX_RUNTIME_BYTES,
                 "Camera execution runtime exceeds its source byte bound")
        with path.open("rb") as stream:
            raw = stream.read(MAX_SOURCE_BYTES + 1)
        total += len(raw)
        _require(len(raw) == size and total <= MAX_RUNTIME_BYTES, "Camera execution source changed while reading")
        sources[name] = raw
    return sources


def execution_runtime_hashes(scene):
    return {name: sha256(raw).hexdigest() for name, raw in _runtime_sources(scene).items()}


def _ast(raw):
    _require(type(raw) is bytes and len(raw) <= MAX_SOURCE_BYTES, "Saved verifier source exceeds its bound")
    try:
        return ast.parse(raw)
    except (SyntaxError, UnicodeError, RecursionError) as error:
        raise ValueError("Saved verifier source must be bounded Python syntax") from error


def _dump(tree):
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _normalized_verifier_ast(raw):
    """Remove only the declared saved-verifier context plumbing; all other syntax remains exact."""
    tree = _ast(raw)
    names = ("verify_camera_shared_training", "verify_camera_shared_evaluation")
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}
    _require(set(functions) == set(names), "Saved verifier context requires both original verification functions")
    for function in functions.values():
        args = function.args
        _require(args.kwonlyargs and args.kwonlyargs[-1].arg == "historical_runtime_sha256"
                 and args.kwonlyargs[-1].annotation is None and isinstance(args.kw_defaults[-1], ast.Constant)
                 and args.kw_defaults[-1].value is None,
                 "Saved verifier compatibility permits only the optional historical runtime context")
        args.kwonlyargs.pop()
        args.kw_defaults.pop()
    expected = ast.parse("hashes = actor.runtime_hashes(scene) if historical_runtime_sha256 is None else historical_runtime_sha256").body[0]
    replacements = [node for node in functions[names[0]].body if _dump(node) == _dump(expected)]
    _require(len(replacements) == 1, "Saved training verifier runtime selection differs from the declared adapter")
    replacement = ast.parse("hashes = actor.runtime_hashes(scene)").body[0]
    function = functions[names[0]]
    function.body[function.body.index(replacements[0])] = replacement
    calls = [node for node in ast.walk(functions[names[1]]) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == names[0]]
    _require(len(calls) == 1, "Saved evaluation must use exactly one unchanged training verifier")
    keywords = [entry for entry in calls[0].keywords if entry.arg == "historical_runtime_sha256"]
    _require(len(keywords) == 1 and isinstance(keywords[0].value, ast.Name)
             and keywords[0].value.id == "historical_runtime_sha256",
             "Saved evaluation runtime forwarding differs from the declared adapter")
    calls[0].keywords.remove(keywords[0])
    return _dump(tree)


def _verifier_delta(original, current):
    normalized = _normalized_verifier_ast(current)
    _require(normalized == _dump(_ast(original)), "Camera compatibility cannot change training, evaluation or actor arithmetic")
    return {"profile": "camera-shared-saved-verifier-runtime-context-v1",
            "original_sha256": sha256(original).hexdigest(), "current_sha256": sha256(current).hexdigest(),
            "unchanged_ast_sha256": sha256(normalized.encode()).hexdigest()}


def _bounded_package(manifest_raw, files):
    _require(type(manifest_raw) is bytes and len(manifest_raw) <= 4 << 20
             and isinstance(files, dict) and 1 <= len(files) <= 512,
             "Camera admission requires a bounded complete evidence package")
    total = len(manifest_raw)
    for name, raw in files.items():
        _require(isinstance(name, str) and len(name) <= 512 and type(raw) is bytes,
                 "Camera admission evidence requires exact named bytes")
        total += len(raw)
        _require(total <= 128 << 20, "Camera admission evidence exceeds 128 MiB")


def _compatibility(policy, files, sources, scene):
    old = _runtime(policy.get("runtime_sha256"))
    original_required, execution_required = policy_runtime_requirements(scene)
    _require(original_required <= old.keys() <= execution_required,
             "Camera policy original runtime dependency closure differs")
    current = {name: sha256(raw).hexdigest() for name, raw in sources.items()}
    changed = {name: {"original_sha256": pin, "current_sha256": current[name]}
               for name, pin in old.items() if pin != current[name]}
    _require(set(changed) <= set(ALLOWED_CHANGED_FILES), "Camera actor, physics, scoring or source runtime changed")
    for name, pin in old.items():
        original = files.get("implementation/" + name)
        _require(type(original) is bytes and len(original) <= MAX_SOURCE_BYTES
                 and sha256(original).hexdigest() == pin, "Camera admission requires every exact original implementation")
    delta = (_verifier_delta(files["implementation/camera_shared_learning.py"], sources["camera_shared_learning.py"])
             if "camera_shared_learning.py" in changed else None)
    return current, {"profile": COMPATIBILITY_PROFILE, "allowed_changed_files": list(ALLOWED_CHANGED_FILES),
                     "changed_files": changed, "verifier_delta": delta}


def prepare_camera_policy_admission(scene, task, *, training_manifest_raw, training_files,
                                    evaluation_manifest_raw=None, evaluation_files=None, deadline):
    """Verify saved evidence once; this is not a new physical/source admission or training run."""
    from weft.camera_shared_learning import (
        TRAINING_ALIASES, _json as package_json, verify_camera_shared_evaluation, verify_camera_shared_training,
    )

    _check_deadline(deadline)
    _bounded_package(training_manifest_raw, training_files)
    _require((evaluation_manifest_raw is None) == (evaluation_files is None), "Camera evaluation needs its exact manifest and members")
    policy_raw = training_files.get("policy.json")
    policy = _json(policy_raw)
    sources = _runtime_sources(scene)
    current, compatibility = _compatibility(policy, training_files, sources, scene)
    _check_deadline(deadline)
    if evaluation_files is None:
        verify_camera_shared_training(training_manifest_raw, training_files, scene, task, deadline=deadline,
                                      historical_runtime_sha256=policy["runtime_sha256"])
    else:
        _bounded_package(evaluation_manifest_raw, evaluation_files)
        _require(evaluation_files.get("input-manifest.json") == training_manifest_raw,
                 "Camera evaluation must retain the exact selected training manifest")
        inventory = package_json(training_manifest_raw).get("files")
        _require(isinstance(inventory, dict) and inventory.keys() == training_files.keys()
                 and all(evaluation_files.get(TRAINING_ALIASES.get(name, name)) == raw
                         for name, raw in training_files.items()),
                 "Camera evaluation changed or omitted selected training evidence")
        verify_camera_shared_evaluation(evaluation_manifest_raw, evaluation_files, scene, task, deadline=deadline,
                                        historical_runtime_sha256=policy["runtime_sha256"])
    weights = actor.coefficients(policy["feedback_coefficients"])
    admission = {"schema_version": 1, "profile": ADMISSION_PROFILE, "status": "admitted",
        "scene_sha256": task["scene_sha256"], "task_sha256": canonical_hash(task),
        **{key: task[key] for key in IDENTITIES}, "policy_sha256": sha256(policy_raw).hexdigest(),
        "training_manifest_sha256": sha256(training_manifest_raw).hexdigest(),
        "evaluation_manifest_sha256": None if evaluation_manifest_raw is None else sha256(evaluation_manifest_raw).hexdigest(),
        "original_runtime_sha256": policy["runtime_sha256"], "execution_runtime_sha256": current,
        "runtime_versions": policy["runtime_versions"], "compatibility": compatibility,
        "compatibility_sha256": canonical_hash(compatibility),
        "summary": {"selected_feedback_kind": actor.selection_kind(policy), "nonzero_coefficients": int((weights != 0).sum()),
                    "root_assistance": task["control"]["root_assistance"], "learning_qualification": "unqualified_research"},
        "privileged_simulator_features": True, "scope": SCOPE}
    result = validate_saved_admission(admission, scene, task, policy_document=policy_raw,
                                     execution_runtime_sha256=current)
    _check_deadline(deadline)
    return result


def validate_camera_policy_admission(admission, scene, task, policy_document):
    """Validate the current actor/adapter boundary; retained history is verified by preparation."""
    sources = _runtime_sources(scene)
    current = {name: sha256(raw).hexdigest() for name, raw in sources.items()}
    result = validate_saved_admission(admission, scene, task, policy_document=policy_document,
                                     execution_runtime_sha256=current)
    delta = result["compatibility"]["verifier_delta"]
    if delta is not None:
        normalized = _normalized_verifier_ast(sources["camera_shared_learning.py"])
        _require(sha256(normalized.encode()).hexdigest() == delta["unchanged_ast_sha256"],
                 "Camera saved-verifier adapter differs from admitted arithmetic")
    actor.validate_prepared_policy(_json(policy_document), scene, task, hashes=result["original_runtime_sha256"])
    return result


def prepare_camera_policy_sampler(policy_document, admission, env):
    """Freeze exact coefficients and authority once; each decision performs no file I/O or hashing."""
    admitted = validate_camera_policy_admission(admission, env.scene, env.task, policy_document)
    policy = actor.validate_prepared_policy(_json(policy_document), env.scene, env.task,
                                           hashes=admitted["original_runtime_sha256"])
    _require(env.evaluator_sha256 == policy["evaluator_sha256"]
             and env.observation_contract()["profile"] == policy["observation_profile"]
             and list(env.action_names) == policy["action_names"], "Camera policy environment or action authority differs")
    weights, groups = actor.coefficients(policy["feedback_coefficients"]), actor.joint_groups(env.action_names)

    def sample():
        action, _, authored, _, phase = actor.action_sample(weights, groups, env)
        return action, {"phase": phase, "authored_action": authored.tolist(),
                        "applied_joint_delta": (action[:19] - authored[:19]).tolist()}

    return sample
