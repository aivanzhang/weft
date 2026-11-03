"""Muse-linked controller candidates are tested against a frozen world, task and scorer."""

from copy import deepcopy
import base64
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np

from weft.io import canonical_hash, digest, local_time
from weft.scene_advisor import validate_advice
from weft.shared_exercise_feedback_residual import coefficients, runtime_hashes
from weft.shared_exercise_feedback_learning import _trial
from weft.shared_exercise_learning import _context, _score
from weft.shared_exercise_lab import _runtime_versions
from weft.shared_exercise_residual_learning import _Budget
from weft.shared_exercise_reset import human_jitter_options
from weft.shared_exercise_task import SharedExerciseTaskEnv, prepare_shared_exercise_snapshot_validator

PROFILE = "portal-controller-repair-comparison-v1"
PLAN_PROFILE = "portal-controller-repair-plan-v1"
MAX_SECONDS = 60.
MAX_CANDIDATES = 3
MAX_CONTROL_STEPS = 7200
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MIN_SCORE_IMPROVEMENT = .01
DELTAS = (-.1, .1, .2)
CHOICE_PROFILE = "muse-controller-choice-v1"
CHOICE_QUESTIONS = (
    "Choose a falsifiable SIMULATOR control hypothesis, or abstain. Available local tests add the same bounded joint velocity-feedback correction to three existing joint groups: feedback-1=-0.1, feedback-2=+0.1, feedback-3=+0.2. These numbers are authored options, not measurements or model-generated code.",
    "For each selected option, put exactly weft:test:feedback-1, weft:test:feedback-2 or weft:test:feedback-3 as a required_tests item in an evidence-linked control hypothesis about person/world:human. Select at most three. If none is justified, return no such token; do not force a choice.",
    "Explain why the selected correction could improve tracking and what would falsify it. The independent test keeps the original world, full reference interval, force limits, task, scorer and gates unchanged, uses paired starts, and rejects absent strict improvement. No physical validation or automatic application follows.",
    "Keep gravity/contact/anatomy and omitted scene inventory uncertainties explicit. A body-only approximation does not establish absence of equipment or other people. No scores, previous candidate outcomes or held-out measurements are supplied.",
)
SOURCE_SCOPE = "construction_only; no held-out scores or audit observations supplied"
SCOPE = ("Authored numerical controller candidates prompted by evidence-linked Muse hypotheses. "
         "Only bounded joint velocity-feedback corrections vary; world, reference, task, gates, "
         "root assistance and equipment force authority stay unchanged. Paired exposed-development "
         "simulation is not recovered physics, measured force, a new-video holdout or physical validation. "
         "Muse's physical measurement requests remain unresolved; no selected candidate is automatically applied.")


class MuseAbstained(ValueError):
    """A valid response selected no supported simulator test."""


def _require(condition, message):
    if not condition:
        raise ValueError("Portal repair comparison: " + message)


def _bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()


def _runtime(scene):
    return {**runtime_hashes(scene), "portal_repairs.py": digest(Path(__file__)),
            "portal_checks.py": digest(Path(__file__).with_name("portal_checks.py"))}


def prepare_repair_plan(scene, task, advice, context):
    """Freeze locally authored candidates; model prose never becomes executable code or geometry."""
    scene, task, _ = _context(scene, task)
    _require(scene["source_binding"]["mode"] == "source_qualified", "a source-qualified exercise world is required")
    source = scene["source_binding"]["source_sha256"]
    _require(isinstance(context, dict) and len(_bytes(context)) <= 32 * 1024
             and context.get("profile") == "exercise" and context.get("source_sha256") == source
             and context.get("evidence_scope") == SOURCE_SCOPE, "review must bind the same construction-only exercise source")
    observations, objects = context.get("observations"), context.get("objects")
    _require(isinstance(observations, list) and 1 <= len(observations) <= 12 and isinstance(objects, list)
             and 1 <= len(objects) <= 14, "bounded reviewed observations and object identities required")
    _require(all(isinstance(row, dict) and isinstance(row.get("id"), str)
                 and row["id"].startswith(source + ":frame:") for row in observations), "reviewed frame source differs")
    _require(all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in objects), "invalid reviewed object IDs")
    advice = validate_advice(deepcopy(advice), [row["id"] for row in observations], [row["id"] for row in objects])
    eligible = [row for row in advice["hypotheses"] if row["status"] == "hypothesis"
                and row["concept"] in {"control", "articulation"} and row["evidence_ids"]
                and set(row["object_ids"]) & {"person", "world:human"}]
    choice_mode = context.get("questions") == list(CHOICE_QUESTIONS)
    if choice_mode and not eligible:
        raise MuseAbstained("Muse selected no supported controller hypothesis; no simulation will run")
    _require(eligible, "no supported evidence-linked control/articulation hypothesis; physical measurements cannot become simulator repairs")
    selections = {}
    if choice_mode:
        for row in eligible:
            for test in row["required_tests"]:
                if test.startswith("weft:test:"):
                    candidate = test.removeprefix("weft:test:")
                    _require(row["concept"] == "control" and candidate in {"feedback-1", "feedback-2", "feedback-3"}
                             and candidate not in selections, "invalid or duplicate explicit Muse candidate selection")
                    selections[candidate] = row
        if not selections:
            raise MuseAbstained("Muse abstained from the bounded controller options; no simulation will run")
    hypothesis = eligible[0]
    candidates = []
    for index, delta in enumerate(DELTAS):
        if choice_mode:
            if f"feedback-{index + 1}" not in selections:
                continue
            hypothesis = selections[f"feedback-{index + 1}"]
        weights = [[0., delta] for _ in range(3)]
        coefficients(weights)
        candidates.append({"id": f"feedback-{index + 1}", "kind": "joint_velocity_feedback_sensitivity",
            "hypothesis_id": hypothesis["id"], "evidence_ids": list(hypothesis["evidence_ids"]),
            "object_ids": list(hypothesis["object_ids"]), "feedback_coefficients": weights,
            "parameter_origin": "fixed_local_authored_grid_not_muse_numbers_or_measurements"})
    horizon = task["interval"]["horizon_steps"]
    maximum = (1 + len(candidates)) * 2 * horizon
    _require(1 <= horizon <= 900 and maximum <= MAX_CONTROL_STEPS, "complete comparison exceeds the action bound; intervals cannot be shortened")
    protocol = {"profile": "paired-frozen-feedback-comparison-v1", "baseline": "authored_unchanged",
        "trials": [{"seed": 1729, "reset_options": human_jitter_options(0.)},
                   {"seed": 92401, "reset_options": human_jitter_options(.05)}],
        "horizon_steps": horizon, "maximum_control_steps": maximum,
        "acceptance": "all_original_task_gates_on_both_trials_and_mean_reward_score_improvement",
        "minimum_score_improvement": MIN_SCORE_IMPROVEMENT,
        "score": "existing_shared_exercise_learning._score", "tie_break": "first_frozen_candidate",
        "evidence_partition": "exposed_development_simulation_not_independent_physical_or_new_video_evidence"}
    return {"schema_version": 1, "profile": PLAN_PROFILE, "source_sha256": source,
        **({"model_choice": {"profile": CHOICE_PROFILE, "selected_ids": list(selections),
                            "authority": "Muse selected authored options; independent scorer decides acceptance"}} if choice_mode else {}),
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
        "advice_sha256": canonical_hash(advice), "context_sha256": canonical_hash(context),
        "runtime_sha256": _runtime(scene), "runtime_versions": _runtime_versions(), "candidates": candidates, "protocol": protocol,
        "physical_hypotheses": [{"id": row["id"], "status": "unresolved", "required_tests": list(row["required_tests"])}
                                for row in advice["hypotheses"]], "scope": SCOPE}


def _comparison(baseline, candidate):
    complete = all(row["status"] == "completed" and row.get("metrics", {}).get("completed_interval") is True
                   and row["metrics"].get("failure") is None for row in candidate)
    passes = complete and all(row["metrics"].get("success") is True for row in candidate)
    before, after = _score(baseline), _score(candidate)
    improvement = after - before
    accepted = passes and improvement >= MIN_SCORE_IMPROVEMENT
    return {"baseline_score": before, "candidate_score": after, "score_improvement": improvement,
            "all_original_gates_passed": passes, "accepted_in_simulation": accepted,
            "reason": "unchanged simulator gates and strict improvement passed" if accepted else
                      "original task gate failure or insufficient simulator score improvement"}


class _Writer:
    def __init__(self, out):
        self.out, self.bytes = out, 0

    def save(self, name, document):
        raw = _bytes(document)
        _require(self.bytes + len(raw) <= MAX_OUTPUT_BYTES, "artifact byte bound exceeded")
        with (self.out / name).open("xb") as stream:
            stream.write(raw)
        self.bytes += len(raw)


def run_repair_plan(scene, task, advice, context, plan, out, *, max_seconds=60,
                    _source_receipt=None, _before_seal=None):
    """Run all predeclared candidates on the same paired starts; retain every rejected candidate."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= MAX_SECONDS,
             "maximum runtime must be positive and at most 60 seconds")
    started, deadline = time.monotonic(), time.monotonic() + max_seconds
    scene, task, advice, context = deepcopy((scene, task, advice, context))
    expected = prepare_repair_plan(scene, task, advice, context)
    _require(canonical_hash(plan) == canonical_hash(expected), "plan differs from its frozen candidates, scorer or input identities")
    out = Path(out).absolute()
    _require(".." not in out.parts and not any(path.is_symlink() for path in (out, *out.parents)),
             "output links or traversal are unsupported")
    out.mkdir(parents=True, exist_ok=False)
    writer = _Writer(out)
    writer.save("plan.json", expected)
    implementation = {name: Path(__file__).with_name(name).read_bytes() for name in expected["runtime_sha256"]}
    _require(all(sha256(raw).hexdigest() == expected["runtime_sha256"][name] for name, raw in implementation.items()),
             "runtime changed before source snapshot")
    writer.save("runtime-source.json", {"runtime_sha256": expected["runtime_sha256"],
        "runtime_versions": expected["runtime_versions"], "files": {name: raw.decode() for name, raw in implementation.items()}})
    writer.save("inputs.json", {"scene": scene, "task": task, "advice": advice, "context": context})
    if _source_receipt is not None:
        writer.save("source-verification.json", _source_receipt)
    writer.save("work-plan.json", {"created_at": local_time(), "maximum_seconds": float(max_seconds),
        "maximum_candidates": MAX_CANDIDATES, "maximum_control_steps": expected["protocol"]["maximum_control_steps"],
        "maximum_output_bytes": MAX_OUTPUT_BYTES, "external_api_usd": 0, "inference_calls": 0,
        "workers": 1, "plan_sha256": canonical_hash(expected)})
    budget = _Budget(expected["protocol"]["maximum_control_steps"])
    runs, env, active = [], None, None
    try:
        env = SharedExerciseTaskEnv(scene, task)
        validator = prepare_shared_exercise_snapshot_validator(scene, task)
        conditions = [{"id": "baseline", "feedback_coefficients": [[0., 0.]] * 3}, *expected["candidates"]]
        initial = {}
        for condition in conditions:
            trials = []
            for index, trial in enumerate(expected["protocol"]["trials"]):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Portal repair comparison exceeded its deadline")
                active = f"{condition['id']}-trial-{index + 1}.json"
                result = _trial(env, "learned", trial["seed"], trial["reset_options"],
                                coefficients(condition["feedback_coefficients"]), deadline, budget,
                                trace=True, validator=validator)
                writer.save(active, result)
                active = None
                identity = result["initial_state_sha256"]
                if condition["id"] == "baseline":
                    _require(identity is not None, "baseline reset failed; comparison cannot change initial states")
                    initial[index] = identity
                else:
                    _require(identity == initial[index], "paired initial state differs across candidates")
                trials.append(result)
            row = {"id": condition["id"], "trials": trials}
            if condition["id"] != "baseline":
                row["comparison"] = _comparison(runs[0]["trials"], trials)
            runs.append(row)
        _require(_runtime(scene) == expected["runtime_sha256"] and _runtime_versions() == expected["runtime_versions"]
                 and canonical_hash(env.scene) == expected["scene_sha256"]
                 and canonical_hash(env.task) == expected["task_sha256"], "runtime, world or task changed during comparison")
        if _before_seal is not None:
            _before_seal()
        if time.monotonic() >= deadline:
            raise TimeoutError("Portal repair comparison exceeded its deadline before publication")
        accepted = [row for row in runs[1:] if row["comparison"]["accepted_in_simulation"]]
        selected = max(accepted, key=lambda row: row["comparison"]["candidate_score"]) if accepted else None
        report = {"schema_version": 1, "profile": PROFILE, "status": "complete", "created_at": local_time(),
            **({"model_choice": expected["model_choice"]} if "model_choice" in expected else {}),
            "plan_sha256": canonical_hash(expected), "scene_sha256": expected["scene_sha256"],
            "task_sha256": expected["task_sha256"], "runtime_sha256": expected["runtime_sha256"],
            "selected_candidate": selected["id"] if selected else None, "applied": False,
            "accepted_in_simulation": selected is not None, "physical_validation": "unverified",
            "candidates": [{"id": row["id"], **row["comparison"]} for row in runs[1:]],
            "physical_hypotheses": expected["physical_hypotheses"], "control_steps": budget.used,
            "seconds": time.monotonic() - started, "external_api_usd": 0, "scope": SCOPE}
        if "model_choice" in expected:
            proposal_cost = (_source_receipt or {}).get("provider_call", {}).get("cost_usd")
            report.update(proposal_external_api_usd=proposal_cost, comparison_external_api_usd=0,
                          external_api_usd=proposal_cost)
        writer.save("report.json", report)
        writer.save("manifest.json", {"schema_version": 1, "status": "complete",
                    "files": {path.name: digest(path) for path in sorted(out.iterdir()) if path.is_file()}})
        return report
    except Exception as error:
        if active is not None and hasattr(error, "partial_trial"):
            writer.save(active, error.partial_trial)
        writer.save("failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"[:2000],
                    "control_steps": budget.used, "partial_evidence_preserved": True, "applied": False})
        raise
    finally:
        if env is not None:
            env.close()


def run_review_repairs(root, out, *, max_seconds=60):
    """Read an existing bound Muse review; never call a provider or alter its project."""
    from weft.portal_checks import inputs, read_json
    from weft.shared_world_packaging import collect_shared_source_files

    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= MAX_SECONDS,
             "maximum runtime must be positive and at most 60 seconds")
    deadline = time.monotonic() + max_seconds
    root, out = Path(root).absolute(), Path(out).absolute()
    _require(not any(path.is_symlink() for path in (root, *root.parents)) and ".." not in root.parts + out.parts,
             "project/output paths cannot contain links or traversal")
    _require(out != root and not root.is_relative_to(out), "repair output cannot contain or replace its source project")
    _require(not out.is_relative_to(root) or out.is_relative_to(root / "muse/repairs"),
             "project-local repair output must stay under muse/repairs, outside immutable runs and reviews")
    scene, task, advice, binding = inputs(root)
    _require(task is not None, "object-only worlds need a separately qualified repair adapter")
    context, _ = read_json(root / "muse/advice/context.json", 32 * 1024)
    proof = collect_shared_source_files(scene, root / binding["run"], expected_manifest_sha256=binding["run_manifest_sha256"])
    plan = prepare_repair_plan(scene, task, advice, context)
    def unchanged():
        _, _, _, after = inputs(root)
        _require(after == binding, "review/source changed during comparison; results cannot be admitted")
        current = collect_shared_source_files(scene, root / binding["run"],
                    expected_manifest_sha256=binding["run_manifest_sha256"])
        _require(current.files == proof.files, "source evidence changed during comparison")
    return run_repair_plan(scene, task, advice, context, plan, out, max_seconds=deadline - time.monotonic(),
        _source_receipt={"review_binding": binding, "source_verification": proof.report}, _before_seal=unchanged)


def request_controller_review(root, out, *, project, execute=False):
    """Keep the optional paid controller-choice review separate from the original physical advice."""
    from weft.portal_checks import inputs
    from weft.portal_review import review_input
    from weft.scene_advisor import advise_run, ESTIMATE
    from weft.shared_world_packaging import collect_shared_source_files

    root, out = Path(root).absolute(), Path(out).absolute()
    _require(not out.exists() and not any(p.is_symlink() for p in (out, *out.parents)), "fresh unlinked review output required")
    _require(not root.is_relative_to(out) and (not out.is_relative_to(root)
             or out.is_relative_to(root / "muse/repairs")), "review output must not replace original project evidence")
    scene, task, _, binding = inputs(root)
    _require(task is not None, "controller choice requires an exercise task")
    _context(scene, task)
    _require(task["interval"]["horizon_steps"] * 8 <= MAX_CONTROL_STEPS, "full comparison exceeds action bound")
    proof = collect_shared_source_files(scene, root / binding["run"], expected_manifest_sha256=binding["run_manifest_sha256"])
    source, construction, world = review_input(root, project=project)
    _require(construction == binding["construction"], "construction changed before controller review")
    out.mkdir(parents=True, exist_ok=False)
    (out / "binding.json").write_bytes(_bytes({"profile": CHOICE_PROFILE, "review_binding": binding,
        "source_verification": proof.report, "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task)}))
    record = advise_run(source, out / "advice", questions=list(CHOICE_QUESTIONS), execute=execute,
                        max_cost_usd=ESTIMATE if execute else None, project_root=project, scene_context=world)
    _require(inputs(root)[3] == binding, "construction changed during controller review")
    return record


def run_controller_review_repairs(root, review, out, *, max_seconds=60):
    """Validate the exact separate provider response, then test only its explicit allowed choices."""
    from weft.portal_checks import inputs, read_json
    from weft.portal_review import review_input
    from weft.scene_advisor import prepare_context, MODEL, SYSTEM, PROPOSAL_SCHEMA, OUTPUT_TOKEN_LIMIT, _sample, _cost
    from weft.shared_world_packaging import collect_shared_source_files

    _require(type(max_seconds) in (int, float) and 0 < max_seconds <= MAX_SECONDS, "invalid comparison deadline")
    deadline = time.monotonic() + max_seconds
    root, review, out = Path(root).absolute(), Path(review).absolute(), Path(out).absolute()
    _require(out != review and not review.is_relative_to(out) and not out.is_relative_to(review), "comparison must be separate from advice")
    _require(not root.is_relative_to(out) and (not out.is_relative_to(root)
             or out.is_relative_to(root / "muse/repairs")), "comparison cannot replace project evidence")
    scene, task, _, binding = inputs(root)
    source, _, world = review_input(root)
    stored, stored_pin = read_json(review / "binding.json", 128 << 10)
    _require(stored.get("profile") == CHOICE_PROFILE, "controller review profile differs")
    _require(stored.get("review_binding") == binding and stored.get("scene_sha256") == canonical_hash(scene)
             and stored.get("task_sha256") == canonical_hash(task), "controller review source/world/task differs")
    pins = {"binding.json": stored_pin}
    documents = {}
    for name, limit in (("context.json", 32 << 10), ("request.json", 4 << 20), ("response.json", 140000),
                        ("hypotheses.json", 140000), ("advisor-call.json", 128 << 10)):
        documents[name], pins["advice/" + name] = read_json(review / "advice" / name, limit)
    context, request, response, proposal, call = (documents[n] for n in
        ("context.json", "request.json", "response.json", "hypotheses.json", "advisor-call.json"))
    _require(context == prepare_context(source, list(CHOICE_QUESTIONS), scene_context=world), "controller context differs from original construction")
    _require(set(request) == {"model", "messages", "response_format", "max_completion_tokens", "reasoning_effort"}
             and request["model"] == MODEL and request["max_completion_tokens"] == OUTPUT_TOKEN_LIMIT
             and request["reasoning_effort"] == "low" and request["response_format"] ==
             {"type": "json_schema", "json_schema": {"name": "WeftSceneAdvice", "schema": PROPOSAL_SCHEMA}}, "controller request contract differs")
    messages = request["messages"]
    _require(isinstance(messages, list) and len(messages) == 2 and messages[0] == {"role": "developer", "content": SYSTEM}
             and messages[1].get("role") == "user" and isinstance(messages[1].get("content"), list)
             and messages[1]["content"] and messages[1]["content"][0] ==
             {"type": "text", "text": json.dumps(context, sort_keys=True, allow_nan=False)}, "controller request context differs")
    _require(call.get("status") == "proposed" and call.get("schema_validated") is True
             and call.get("request_count") == 1 and call.get("retries") == 0 and call.get("live_access_verified") is True
             and call.get("model") == MODEL and call.get("model_returned") == MODEL and response.get("model") == MODEL
             and call.get("usage") == response.get("usage") and call.get("cost_usd") == _cost(response.get("usage"))
             and call.get("request_sha256") == canonical_hash(request)
             and call.get("source_sha256") == binding["source_sha256"]
             and call.get("simulation_world_sha256") == canonical_hash(world), "completed controller provider receipt required")
    rows = _sample(list({row["frame"]: row for row in context["observations"]}.values()), 3)
    content = messages[1]["content"]
    _require(len(content) == 1 + 2 * len(rows) and call.get("frame_ids") == [row["id"] for row in rows]
             and isinstance(call.get("frame_jpeg_sha256"), list) and len(call["frame_jpeg_sha256"]) == len(rows),
             "controller images differ from the declared construction frames")
    for index, row in enumerate(rows):
        caption, image = content[1 + 2 * index:3 + 2 * index]
        _require(caption == {"type": "text", "text": f"Construction frame {row['frame']}; evidence {row['id']}"}
                 and isinstance(image, dict) and set(image) == {"type", "image_url"} and image["type"] == "image_url"
                 and isinstance(image["image_url"], dict) and set(image["image_url"]) == {"url", "detail"}
                 and image["image_url"]["detail"] == "low", "controller frame caption or image fields differ")
        url = image["image_url"]["url"]
        _require(isinstance(url, str) and url.startswith("data:image/jpeg;base64,") and len(url) <= 175000,
                 "controller image must be a bounded inline JPEG")
        raw = base64.b64decode(url.split(",", 1)[1], validate=True)
        _require(0 < len(raw) <= 131072 and sha256(raw).hexdigest() == call["frame_jpeg_sha256"][index],
                 "controller image differs from the provider receipt")
    choices = response.get("choices")
    _require(isinstance(choices, list) and len(choices) == 1 and choices[0].get("finish_reason") == "stop"
             and not choices[0].get("message", {}).get("refusal"), "controller provider response incomplete")
    _require(proposal.get("source") == "model_proposal" and proposal.get("accepted") is False
             and proposal.get("advice") == json.loads(choices[0]["message"]["content"]), "controller advice differs from provider response")
    advice = proposal["advice"]
    plan = prepare_repair_plan(scene, task, advice, context)
    proof = collect_shared_source_files(scene, root / binding["run"], expected_manifest_sha256=binding["run_manifest_sha256"])
    def unchanged():
        _require(inputs(root)[3] == binding, "original review changed during comparison")
        for name, pin in pins.items():
            _require(digest(review / name) == pin, "controller review changed during comparison")
        current = collect_shared_source_files(scene, root / binding["run"], expected_manifest_sha256=binding["run_manifest_sha256"])
        _require(current.files == proof.files, "source proof changed during comparison")
    return run_repair_plan(scene, task, advice, context, plan, out, max_seconds=deadline - time.monotonic(),
        _source_receipt={"review_binding": binding, "controller_review_files": pins, "source_verification": proof.report,
                         "provider_call": {key: call.get(key) for key in
                             ("model", "request_sha256", "request_count", "cost_usd", "usage", "budget_reservation")}},
        _before_seal=unchanged)
