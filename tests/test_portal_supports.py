import json
from pathlib import Path
import time

import pytest

from weft.io import digest, write_json
from weft.portal_supports import copy_body, execute_support_job
from test_portal import portal, upload, request, serving
from test_static_support_build import priors


@pytest.fixture
def parent(portal):
    job = upload(portal)
    root = portal.root / job["id"]
    body = root / "build/body_source"
    body.mkdir(parents=True)
    (body / "source.webm").write_bytes((root / "source.webm").read_bytes())
    write_json(body / "human.json", {"scope": "Metadata fixture, native build is stubbed"})
    write_json(body / "evidence.json", {"source_sha256": job["source_sha256"]})
    write_json(body / "manifest.json", {"schema_version": 1, "profile": "exercise-body-source-v1", "status": "construction_only",
        "files": {p.name: digest(p) for p in body.iterdir()}})
    write_json(root / "status.json", {"status": "ready", "route": "exercise", "lab": "lab", "events": []})
    return portal.summary(job["id"])


def payload(parent, **changes):
    return json.dumps({"request_id": "b" * 32, "priors": priors(parent["source_sha256"]), "train": False, **changes}).encode()


def test_authored_variant_is_separate_deduplicated_and_requires_no_api_key(portal, parent):
    root = portal.root / parent["id"]
    before = {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}
    child = portal.support_experiment(parent["id"], payload(parent))
    assert child["id"] != parent["id"] and child["kind"] == "support_experiment"
    assert child["parent_id"] == parent["id"] and not child["source_available"]
    assert portal.support_experiment(parent["id"], payload(parent))["id"] == child["id"]
    assert before == {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="different experiment"):
        portal.support_experiment(parent["id"], payload(parent, train=True))
    with pytest.raises(ValueError, match="original exercise"):
        portal.support_experiment(child["id"], payload(parent, request_id="c" * 32))


@pytest.mark.parametrize("kind", ["source", "observed", "moving", "unknown", "bool", "oversize"])
def test_bad_support_request_cannot_create_job(portal, parent, kind):
    value = json.loads(payload(parent))
    if kind == "source": value["priors"]["source_sha256"] = "f" * 64
    elif kind == "observed": value["priors"]["objects"][0]["provenance"]["source_contact_verified"] = True
    elif kind == "moving": value["priors"]["objects"][0]["role"] = "passive"
    elif kind == "unknown": value["command"] = "do not execute"
    elif kind == "bool": value["train"] = "false"
    else: value["padding"] = "x" * 128_000
    with pytest.raises(ValueError):
        portal.support_experiment(parent["id"], json.dumps(value).encode())
    assert len(portal.jobs) == 1


@pytest.mark.parametrize("train", [False, True])
def test_worker_reuses_exact_body_and_existing_builder_without_inference(portal, parent, monkeypatch, train):
    child = portal.support_experiment(parent["id"], payload(parent, train=train))
    folder = portal.root / child["id"]
    source = portal.root / parent["id"] / "build/body_source"
    called = []
    def build(base, evidence, config, out, **options):
        called.append(options)
        assert (base, evidence, config) == (None, None, None)
        assert options["scene_scope"] == "human_static_supports" and "video" not in options
        assert digest(options["human_run"] / "manifest.json") == digest(source / "manifest.json")
        assert options["full_reference"] is True
        assert bool(options["train_settings"]) == train
        lab = out / "lab"
        lab.mkdir(parents=True)
        return {"status": "ready", "lab_directory": str(lab)}
    monkeypatch.setattr("weft.exercise_scene_pipeline.prepare_exercise_scene", build)
    assert execute_support_job(folder) == 0 and len(called) == 1
    assert portal.summary(child["id"])["status"] == "ready"
    assert portal.summary(parent["id"])["status"] == "ready"
    assert digest(folder / "source.webm") == parent["source_sha256"]
    status_before = digest(folder / "status.json")
    assert execute_support_job(folder) == 0 and len(called) == 1
    assert digest(folder / "status.json") == status_before
    manifest = json.loads((source / "manifest.json").read_text())
    assert all(digest(folder / "body_source" / name) == sha for name, sha in manifest["files"].items())


def test_changed_priors_fail_before_body_or_build(portal, parent, monkeypatch):
    child = portal.support_experiment(parent["id"], payload(parent))
    folder = portal.root / child["id"]
    (folder / "support-priors.json").write_text('{}')
    monkeypatch.setattr("weft.exercise_scene_pipeline.prepare_exercise_scene", lambda *_a, **_k: pytest.fail("No build allowed"))
    assert execute_support_job(folder) == 1
    assert not (folder / "body_source").exists()
    assert portal.summary(parent["id"])["status"] == "ready"


def test_copy_rejects_changed_source_and_expired_deadline(portal, parent, tmp_path):
    source = portal.root / parent["id"] / "build/body_source"
    manifest = digest(source / "manifest.json")
    with pytest.raises(TimeoutError):
        copy_body(source, tmp_path / "expired", expected_manifest=manifest, deadline=time.monotonic() - 1)
    (source / "human.json").write_text('{}')
    with pytest.raises(ValueError):
        copy_body(source, tmp_path / "changed", expected_manifest=manifest, deadline=time.monotonic() + 5)


def test_support_endpoint_auth_bounds_and_saved_settings(portal, parent):
    with serving(portal) as server:
        path = f"/api/projects/{parent['id']}/supports"
        assert request(server, path, "POST", payload(parent), auth=False)[0] == 401
        assert request(server, path, "POST", payload(parent), headers={"Origin": "https://attacker.invalid"})[0] == 403
        assert request(server, path, "POST", b"x" * 128_001)[0] == 400
        status, _, raw = request(server, path, "POST", payload(parent))
        assert status == 202
        child = json.loads(raw)
        assert request(server, f"/api/projects/{child['id']}/files/support-priors.json")[0] == 200
        assert request(server, f"/api/projects/{child['id']}/files/job.json")[0] == 400
