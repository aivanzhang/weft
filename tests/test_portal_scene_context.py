"""Review projection custody; analytical inputs never claim physical evidence."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from weft.io import canonical_hash, digest, write_json
from weft.portal_scene_context import construction_context, validate_context
from weft.shared_support_fixture import static_support_fixture
from weft.shared_support_task import build_shared_support_task
from weft.shared_world import validate_shared_world


def make_job(root, *, source_sha256="a" * 64, support=True):
    root = Path(root)
    root.mkdir(exist_ok=True)
    scene, _ = static_support_fixture()
    scene["human_spec"]["source_sha256"] = source_sha256
    obj = scene["objects"][0]
    obj["provenance"] = {"mode": "authored_addition", "calibration_status": "estimated_prior",
        "physical_validation": "unverified", "source_contact_verified": False,
        "assumptions": ["Analytical fixture; no observed equipment or recovered contact"]}
    scene["source_binding"] = {"mode": "source_qualified", "source_sha256": source_sha256, "total_source_frames": 61,
        "evidence_documents": {"b" * 64: "evidence/body.json"}, "entity_bindings": [
            {"world_entity_id": "human", "entity_kind": "articulated_human", "observation_entity_id": "person",
             "observation_document_sha256": "b" * 64, "provenance_mode": "observed_track",
             "observed_slots": 61, "missing_slots": 0, "pre_init_slots": 0},
            {"world_entity_id": obj["id"], "entity_kind": "rigid_object", "observation_entity_id": None,
             "observation_document_sha256": None, "provenance_mode": "authored_addition",
             "observed_slots": 0, "missing_slots": 61, "pre_init_slots": 0}]}
    scene = validate_shared_world(scene)
    requirements = [{"object_id": obj["id"], "human_segment": "torso", "direction_world": [0., 0., 1.],
                     "minimum_load_body_weight_fraction": .2, "start_step": 6, "end_step": 60}]
    task = build_shared_support_task(scene, root_assistance=False, support_requirements=requirements)
    run = root / "lab/run"
    run.mkdir(parents=True)
    write_json(run / "scene.json", scene)
    write_json(run / "exercise-task.json", task)
    write_json(run / "evaluation.json", {"secret_marker": "DO_NOT_READ_EVALUATION"})
    write_json(run / "policy.json", {"secret_marker": "DO_NOT_READ_POLICY"})
    write_json(run / "manifest.json", {"status": "ready", "files": {p.name: digest(p) for p in run.iterdir()}})
    write_json(run.parent / "lab-config.json", {"control_profile": "shared_exercise", "manifest_sha256": digest(run / "manifest.json"),
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task)})
    job = json.loads((root / "job.json").read_bytes()) if (root / "job.json").exists() else {}
    job.update(source_sha256=source_sha256, kind="support_experiment" if support else "video")
    if support:
        prior = {"schema_version": 1, "profile": "shared-static-support-priors-v1", "source_sha256": source_sha256,
            "objects": scene["objects"], "support_requirements": requirements, "root_assistance": False, "scale": 1.}
        write_json(root / "support-priors.json", prior)
        job["support_priors_sha256"] = digest(root / "support-priors.json")
    write_json(root / "job.json", job)
    write_json(root / "status.json", {"status": "ready", "route": "exercise", "lab": "lab"})
    return root


@pytest.fixture
def job(tmp_path):
    return make_job(tmp_path / "project")


def test_exact_closed_context_excludes_policy_results_and_preserves_prior_scope(job, monkeypatch):
    import weft.portal_scene_context as module
    before = {p.relative_to(job).as_posix(): digest(p) for p in job.rglob("*") if p.is_file()}
    original = module.read_package_file
    seen = []
    def read(root, name, **kwargs):
        seen.append(name)
        assert name not in {"evaluation.json", "policy.json"}
        return original(root, name, **kwargs)
    monkeypatch.setattr(module, "read_package_file", read)
    context, binding = construction_context(job)
    assert "DO_NOT_READ" not in json.dumps(context)
    assert context["objects"][0]["provenance"]["source_contact_verified"] is False
    assert context["task_assumptions"]["support_requirements"][0]["human_segment"] == "torso"
    assert context["body"]["mobility"]["ground_method"] == "authored_horizontal_floor"
    assert len(json.dumps(context).encode()) < 4000
    assert set(binding["input_sha256"]) == {"scene.json", "exercise-task.json"}
    assert binding["support_priors_sha256"] == digest(job / "support-priors.json")
    assert before == {p.relative_to(job).as_posix(): digest(p) for p in job.rglob("*") if p.is_file()}


@pytest.mark.parametrize("file", ["scene.json", "exercise-task.json"])
def test_changed_sealed_input_rejects(job, file):
    (job / "lab/run" / file).write_text('{}')
    with pytest.raises(ValueError, match="pinned SHA256"):
        construction_context(job)


@pytest.mark.parametrize("change", ["bytes", "source", "position", "root", "load"])
def test_changed_authored_support_prior_rejects_even_if_rehashed(job, change):
    path = job / "support-priors.json"
    prior = json.loads(path.read_bytes())
    if change == "bytes": path.write_bytes(path.read_bytes() + b' ')
    else:
        if change == "source": prior["source_sha256"] = "c" * 64
        elif change == "position": prior["objects"][0]["position"][0] += .1
        elif change == "root": prior["root_assistance"] = True
        else: prior["support_requirements"][0]["minimum_load_body_weight_fraction"] = .3
        write_json(path, prior)
        config = json.loads((job / "job.json").read_bytes())
        config["support_priors_sha256"] = digest(path)
        write_json(job / "job.json", config)
    with pytest.raises(ValueError): construction_context(job)


@pytest.mark.parametrize("change", ["source", "task_binding", "manifest_binding", "traversal", "absolute", "link"])
def test_mismatched_identity_and_unsafe_paths_reject(job, tmp_path, change):
    path = job / "lab/lab-config.json"
    value = json.loads(path.read_bytes())
    if change == "source":
        path = job / "job.json";value = json.loads(path.read_bytes());value["source_sha256"] = "d" * 64
    elif change == "task_binding": value["task_sha256"] = "d" * 64
    elif change == "manifest_binding": value["manifest_sha256"] = "d" * 64
    elif change in {"traversal", "absolute"}:
        path = job / "status.json";value = json.loads(path.read_bytes());value["lab"] = "../lab" if change == "traversal" else str(job / "lab")
    else:
        destination = tmp_path / "linked-scene.json"
        (job / "lab/run/scene.json").rename(destination)
        (job / "lab/run/scene.json").symlink_to(destination)
    if change != "link": write_json(path, value)
    with pytest.raises(ValueError): construction_context(job)


@pytest.mark.parametrize("change", ["top_level", "nested", "object", "requirement", "nan", "oversized", "source"])
def test_advisor_context_rejects_extra_fields_nonfinite_and_wrong_source(job, change):
    context, _ = construction_context(job)
    if change == "top_level": context["evaluation"] = {"score": 1}
    elif change == "nested": context["body"]["mobility"]["score"] = 1
    elif change == "object": context["objects"][0]["provenance"]["policy"] = {}
    elif change == "requirement": context["task_assumptions"]["support_requirements"][0]["object_id"] = "missing"
    elif change == "nan": context["gravity"][0] = float('nan')
    elif change == "oversized": context["objects"][0]["label"] = 'x' * 13000
    else: context["source_sha256"] = "d" * 64
    with pytest.raises(ValueError): validate_context(context, source_sha256="a" * 64)


def test_original_exercise_does_not_require_support_prior(tmp_path):
    root = make_job(tmp_path / "original", support=False)
    context, binding = construction_context(root)
    assert context["source_sha256"] == "a" * 64
    assert "support_priors_sha256" not in binding


def test_nonexercise_and_nonready_paths_are_explicit(tmp_path):
    root = tmp_path / "rigid";root.mkdir()
    write_json(root / "job.json", {"source_sha256": "a" * 64})
    write_json(root / "status.json", {"status": "ready", "route": "objects"})
    assert construction_context(root) == (None, None)
    write_json(root / "status.json", {"status": "running", "route": "exercise"})
    with pytest.raises(ValueError, match="ready"): construction_context(root)


def test_large_coupled_request_uniformly_samples_observations_without_dropping_world(job,tmp_path):
    from weft.scene_advisor import prepare_context, _sample
    world,_=construction_context(job)
    source=tmp_path/'advisor';source.mkdir()
    rows=[{'id':f"{'a'*64}:frame:{i}",'time_s':i/30,'joints':{'joint-'+str(n):i/100 for n in range(70)},
           'joint_confidence':{'joint-'+str(n):.75 for n in range(70)}} for i in range(121)]
    write_json(source/'evidence.json',{'source_sha256':'a'*64,'reference':rows,'assumptions':['unverified'],'audit':[{'secret':'DO_NOT_SEND'}]})
    before=(source/'evidence.json').read_bytes()
    context=prepare_context(source,scene_context=world)
    count=context['observation_sampling']['selected']
    assert count in {8,4,3} and context['observation_sampling']['available']==121
    assert [row['id'] for row in context['observations']]==[row['id'] for row in _sample(rows,count)]
    assert context['observations'][0]['frame']==0 and context['observations'][-1]['frame']==120
    assert context['simulation_world']==world and len(json.dumps(context).encode())<=20000
    assert 'DO_NOT_SEND' not in json.dumps(context) and (source/'evidence.json').read_bytes()==before
    assert prepare_context(source,scene_context=world)==context


def test_world_is_not_truncated_when_minimum_observation_sample_still_overflows(job,tmp_path):
    from weft.scene_advisor import prepare_context
    world,_=construction_context(job)
    source=tmp_path/'oversized';source.mkdir()
    write_json(source/'evidence.json',{'source_sha256':'a'*64,'reference':[{'id':f"{'a'*64}:frame:{i}",'joints':{'large': 'x'*8000}} for i in range(12)]})
    with pytest.raises(ValueError,match='20 KB'):prepare_context(source,scene_context=world)
