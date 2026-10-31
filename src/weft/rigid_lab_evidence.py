"""Pure saved spatial-evaluation checks; no policy execution or physics validation."""

import json
import math
from pathlib import PurePosixPath
import re

from weft.io import canonical_hash


PROFILE = "spatial-rigid-evidence-v1"
MAX_EVALUATION_BYTES = 4 * 1024 * 1024
MAX_OUTPUT_BYTES = 32 * 1024
_SHA = re.compile(r"[0-9a-f]{64}")
_CONTROLLERS = ("learned", "random", "coast", "brake")
_VERSIONS = ((1, "spatial-policy-v1", "spatial_cem_v1"), (2, "spatial-policy-v2", "spatial_cem_v2"))
_SCOPE = "Saved simulator summaries and trial arithmetic only; no replay, fresh-final evaluation or physical validation."


def _require(condition, message):
    if not condition:
        raise ValueError("Invalid saved rigid evidence: " + message)


def _number(value, name, *, low=0, high=1e9, integer=False):
    _require(type(value) in ((int,) if integer else (int, float))
             and low <= value <= high and math.isfinite(value), name)
    return value


def _document(value, name, limit):
    _require(isinstance(value, dict), name + " must be an object")
    try:
        raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("Invalid saved rigid evidence: " + name + " JSON") from error
    _require(len(raw) <= limit, name + " byte bound")
    return value


def _text(value, name, limit=128):
    _require(isinstance(value, str) and 0 < len(value) <= limit
             and all(ord(char) >= 32 or char in "\t\n\r" for char in value), name)
    return value


def _close(actual, expected, name):
    _number(actual, name, low=-1e9)
    _require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), name + " contradicts saved trials")


def _trial_bounds(evaluation):
    # Reject oversized trial lists before serializing or hashing their contents.
    regimes = evaluation.get("regimes")
    _require(isinstance(regimes, dict) and set(regimes) == {"nominal_exact", "changed_initial_conditions"},
             "evaluation regimes")
    changed = regimes["changed_initial_conditions"]
    _require(isinstance(changed, dict), "changed-condition regime")
    for rows, lower, upper in ((regimes["nominal_exact"], 1, 1),
                               (changed.get("policies"), 4, 64), (evaluation.get("policies"), 5, 65)):
        _require(isinstance(rows, dict) and set(rows) == set(_CONTROLLERS), "evaluation controller bound")
        for row in rows.values():
            _require(isinstance(row, dict), "controller summary")
            count = _number(row.get("episodes"), "early episode bound", integer=True, low=lower, high=upper)
            trials = row.get("trials")
            _require(isinstance(trials, list) and len(trials) == count, "early trial bound")


def _pointers(values, names):
    _require(isinstance(values, list) and len(values) <= 128, "metadata pointer bound")
    available = {}
    for row in values:
        _require(isinstance(row, dict), "metadata pointer")
        name, path, digest = row.get("source_name"), row.get("path"), row.get("sha256")
        for value in (name, path):
            _require(isinstance(value, str) and len(value) <= 512 and value
                     and not PurePosixPath(value).is_absolute() and ":" not in value and "\\" not in value
                     and all(ord(char) >= 32 for char in value)
                     and not any(part in ("", ".", "..") for part in value.split("/")), "unsafe metadata pointer")
        _require(isinstance(digest, str) and _SHA.fullmatch(digest) and name not in available, "metadata identity")
        available[name] = {"source_name": name, "path": path, "sha256": digest}
    _require(all(name in available for name in names), "required metadata was not hash-verified")
    return [available[name] for name in names]


def _trial(row, *, jitter, dynamic_ids, max_steps):
    _require(isinstance(row, dict), "trial must be an object")
    seed = _number(row.get("seed"), "trial seed", integer=True, high=2**31 - 1)
    _number(row.get("steps"), "trial steps", integer=True, high=max_steps)
    total = _number(row.get("return"), "episode return", low=-1e9)
    for key in ("success", "terminated", "truncated", "reset_jitter"):
        _require(type(row.get(key)) is bool, "trial " + key)
    _require(row["reset_jitter"] is jitter, "regime reset protocol")
    failure = row.get("failure")
    if failure is not None:
        _text(failure, "execution failure", 2000)
        _require(not row["success"], "failed execution claims success")
    else:
        _require(row["terminated"] or row["truncated"], "unterminated saved trial")
    perturbation = row.get("perturbation")
    if not jitter:
        _require(perturbation is None, "nominal reset has a perturbation")
    else:
        _require(isinstance(perturbation, dict) and set(perturbation) == dynamic_ids, "perturbation actors")
        for value in perturbation.values():
            _require(isinstance(value, dict) and type(value.get("fallback_unperturbed")) is bool,
                     "perturbation fallback")
            for key in ("position_delta", "velocity_delta"):
                vector = value.get(key)
                _require(isinstance(vector, list) and len(vector) == 3, "perturbation vector")
                for number in vector:
                    _number(number, "perturbation component", low=-10, high=10)
    return seed, total, int(row["success"])


def _regime(rows, *, identifier, episodes, jitter, dynamic_ids, max_steps):
    _require(isinstance(rows, dict) and set(rows) == set(_CONTROLLERS), "regime controllers")
    result, signatures, examples, failed = {}, None, [], 0
    for name in _CONTROLLERS:
        row = rows[name]
        _require(isinstance(row, dict), "controller summary")
        count = _number(row.get("episodes"), "episodes", integer=True, low=1, high=65)
        _require(count == episodes, "regime episode count")
        trials = row.get("trials")
        _require(isinstance(trials, list) and len(trials) == count, "trial count")
        values = [_trial(trial, jitter=jitter, dynamic_ids=dynamic_ids, max_steps=max_steps) for trial in trials]
        current = [(value[0], trial.get("perturbation")) for value, trial in zip(values, trials, strict=True)]
        _require(signatures is None or current == signatures, "controllers do not share the saved reset conditions")
        signatures = current
        successes = sum(value[2] for value in values)
        _require(_number(row.get("successes"), "successes", integer=True, high=count) == successes, "success count")
        _close(row.get("success_rate"), successes / count, "success rate")
        mean_return = math.fsum(value[1] for value in values) / count
        _close(row.get("mean_return"), mean_return, "mean episode return")
        result[name] = {"episodes": count, "successes": successes, "mean_episode_return": row["mean_return"]}
        for trial in trials:
            if not trial["success"]:
                failed += 1
                if len(examples) < 8:
                    failure = trial.get("failure")
                    examples.append({"regime": identifier, "controller": name, "seed": trial["seed"],
                        "kind": "execution_failure" if failure is not None else "task_not_successful",
                        "failure": failure[:512] if failure is not None else None,
                        "failure_text_truncated": failure is not None and len(failure) > 512})
    seeds = [seed for seed, _ in signatures]
    return {"episodes_per_controller": episodes, "controllers": result, "seeds": seeds,
            "distinct_seed_count": len(set(seeds)), "failure_examples": examples,
            "failure_examples_omitted": failed - len(examples)}


def normalize_rigid_lab_evidence(*, scene, task=None, policy=None, evaluation=None, verified_metadata):
    """Normalize exact caller-verified documents, preserving regimes and return units."""
    def absent(status, reason):
        return {"profile": PROFILE, "status": status, "reason": reason,
                "physical_validation": "unverified", "scope": _SCOPE, "evidence": []}

    _require(isinstance(scene, dict), "scene must be an object")
    if scene.get("profile") != "spatial-rigid-v1":
        return absent("unsupported", "This adapter supports only saved spatial-rigid-v1 evaluation.")
    if policy is None or evaluation is None:
        return absent("unavailable", "A hash-verified spatial policy and evaluation are both required.")
    _require(isinstance(policy, dict) and isinstance(evaluation, dict), "policy/evaluation must be objects")
    version = (policy.get("schema_version"), policy.get("profile"), policy.get("algorithm"))
    if type(version[0]) is not int or version not in _VERSIONS:
        return absent("unsupported", "Spatial policy version/algorithm is not supported by this saved-metadata adapter.")
    if type(evaluation.get("schema_version")) is not int or evaluation["schema_version"] != 1:
        return absent("unsupported", "Spatial evaluation schema is not supported by this saved-metadata adapter.")
    _trial_bounds(evaluation)
    _document(scene, "scene", 2 * 1024 * 1024)
    _document(policy, "policy", 64 * 1024)
    _document(evaluation, "evaluation", MAX_EVALUATION_BYTES)
    names = ["scene.json", "policy.json", "rl-evaluation.json"]
    if task is not None:
        _document(task, "task sidecar", 64 * 1024)
        names.append("task.json")
    evidence = _pointers(verified_metadata, names)
    policy_sha = next(row["sha256"] for row in evidence if row["source_name"] == "policy.json")
    scene_sha = canonical_hash(scene)
    _require(type(scene.get("schema_version")) is int and scene["schema_version"] == 1
             and evaluation.get("scene_profile") == scene["profile"]
             and evaluation.get("profile") == policy["profile"]
             and evaluation.get("physical_validation") == policy.get("physical_validation") == "unverified"
             and evaluation.get("scene_sha256") == policy.get("scene_sha256") == scene_sha
             and evaluation.get("policy_sha256") == policy_sha, "evaluation/policy/scene binding")
    units = scene.get("units")
    _require(isinstance(units, dict) and units.get("length") == "scene_width" and units.get("time") == "second",
             "normalized scene units")
    objects = scene.get("objects")
    _require(isinstance(objects, list) and 1 <= len(objects) <= 12 and all(isinstance(obj, dict) for obj in objects),
             "object bound")
    ids = [_text(obj.get("id"), "object identity") for obj in objects]
    _require(len(set(ids)) == len(ids) and policy.get("object_ids") == ids
             and all(obj.get("role") in ("active", "passive", "static") for obj in objects), "policy object ordering or roles")
    controlled = scene.get("controlled_id")
    index = policy.get("controlled_index")
    _require(type(index) is int and 0 <= index < len(ids) and ids[index] == controlled
             and objects[index].get("role") in ("active", "passive")
             and evaluation.get("controlled_id") == policy.get("controlled_id") == controlled,
             "controlled object binding")
    _require(type(policy.get("observation_dim")) is int and policy["observation_dim"] == 13 * len(ids) + 4,
             "policy observation dimension")
    weights = policy.get("weights")
    _require(isinstance(weights, list) and len(weights) == 11, "policy weight bound")
    for weight in weights:
        _number(weight, "policy weight", low=-6, high=6)
    embedded = scene.get("task")
    _require(isinstance(embedded, dict) and set(embedded) == {"position_tolerance", "speed_tolerance", "acceleration", "max_steps"},
             "embedded task contract")
    for key, lower, upper in (("position_tolerance", .01, .2), ("speed_tolerance", .01, .5)):
        _number(embedded[key], key, low=lower, high=upper)
        _require(type(policy.get(key)) in (int, float) and policy[key] == embedded[key], "policy task tolerance")
    _number(embedded["acceleration"], "task acceleration", low=.1, high=10)
    max_steps = _number(embedded["max_steps"], "task horizon", integer=True, low=10, high=900)
    target = scene.get("target")
    _require(isinstance(target, list) and len(target) == 3, "task target")
    for value in target:
        _number(value, "task target coordinate", high=2)
    if task is not None:
        expected = {"profile": scene["profile"], "controlled_id": controlled, "target": scene.get("target"), **embedded}
        _require(all(key in task and canonical_hash({key: task[key]}) == canonical_hash({key: value})
                     for key, value in expected.items()), "task sidecar differs from embedded task")
    regimes = evaluation.get("regimes")
    _require(isinstance(regimes, dict) and set(regimes) == {"nominal_exact", "changed_initial_conditions"}, "evaluation regimes")
    changed = regimes["changed_initial_conditions"]
    _require(isinstance(changed, dict) and isinstance(changed.get("policies"), dict), "changed-condition regime")
    pos = _number(changed.get("position_jitter_m"), "position jitter", high=.15)
    vel = _number(changed.get("velocity_jitter_mps"), "velocity jitter", high=1)
    first = changed["policies"].get("learned")
    _require(isinstance(first, dict), "changed-condition learned summary")
    count = _number(first.get("episodes"), "changed-condition episodes", integer=True, low=4, high=64)
    dynamic_ids = {obj["id"] for obj in objects if obj.get("role") != "static"}
    nominal = _regime(regimes["nominal_exact"], identifier="nominal_exact", episodes=1, jitter=False,
                      dynamic_ids=dynamic_ids, max_steps=max_steps)
    altered = _regime(changed["policies"], identifier="changed_initial_conditions", episodes=count, jitter=bool(pos or vel),
                      dynamic_ids=dynamic_ids, max_steps=max_steps)
    aggregate = evaluation.get("policies")
    _require(isinstance(aggregate, dict) and set(aggregate) == set(_CONTROLLERS), "aggregate controllers")
    normalized_aggregate = {}
    for name in _CONTROLLERS:
        row = aggregate[name]
        _require(isinstance(row, dict), "aggregate summary")
        original, perturbed = regimes["nominal_exact"][name], changed["policies"][name]
        trials = original["trials"] + perturbed["trials"]
        _require(canonical_hash(row["trials"]) == canonical_hash(trials), "aggregate trial order/content")
        successes = original["successes"] + perturbed["successes"]
        for key, expected in (("episodes", count + 1), ("successes", successes),
                              ("changed_init_episodes", count), ("changed_init_successes", perturbed["successes"])):
            _require(_number(row.get(key), key, integer=True, high=65) == expected, "aggregate counts")
        _require(row.get("nominal_success") is original["trials"][0]["success"], "aggregate nominal outcome")
        _close(row.get("success_rate"), successes / (count + 1), "aggregate success rate")
        _close(row.get("mean_return"), math.fsum(trial["return"] for trial in trials) / (count + 1), "aggregate return")
        normalized_aggregate[name] = {"episodes": count + 1, "successes": successes, "mean_episode_return": row["mean_return"]}
    altered.update(position_jitter=pos, velocity_jitter=vel,
                   position_jitter_unit="scene_width", velocity_jitter_unit="scene_width_per_second")
    result = {"profile": PROFILE, "status": "available", "scene_profile": scene["profile"], "scene_sha256": scene_sha,
        "policy_sha256": policy_sha, "controlled_id": controlled,
        "task_binding": "matching_task_sidecar" if task is not None else "scene_embedded_task",
        "regimes": {"nominal_exact": nominal, "changed_initial_conditions": altered},
        "aggregate": {"controllers": normalized_aggregate}, "return_unit": "episode_return",
        "physical_validation": "unverified", "evidence_kind": "simulated_development", "partition": "exposed_development",
        "evidence": evidence, "scope": _SCOPE,
        "limitations": ["Return is summed reward per episode, not mean reward per step.",
            "Scene lengths and jitter are normalized scene-width units; historical _m keys do not establish metres.",
            "Changed-condition labels describe the saved reset protocol; zero jitter, fallback placement or repeated seeds do not establish distinct starts.",
            "Task identity is bound through the exact scene and optional matching task sidecar; no exercise task hash is invented.",
            "No completion/reset counts, calibrated geometry or physical transfer measurements are inferred."]}
    _document(result, "normalized output", MAX_OUTPUT_BYTES)
    return result
