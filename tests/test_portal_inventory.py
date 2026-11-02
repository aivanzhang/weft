"""Explicit scene scope gates: no inference, physics, network, or listening server required."""
from copy import deepcopy
from io import BytesIO
import json
from types import SimpleNamespace

import pytest

from weft.io import digest, write_json
from weft.portal import Portal, PortalHandler
from weft import portal_pipeline as pipeline
from weft.portal_inventory import exercise_intent, inventory_report, validate_intent


@pytest.mark.parametrize("inventory", [None, "unknown", "equipment_or_obstacles"])
def test_absence_is_never_fabricated(inventory):
    intent = exercise_intent(inventory, "simulation")
    report = inventory_report(intent, "a" * 64)
    assert report["route"] == "needs_input" and report["absence_verified"] is False
    assert intent["basis"] == ("not_declared" if inventory is None else "user_declared")


def test_only_explicit_ground_declaration_can_enter_person_simulation():
    intent = exercise_intent("ground_only", "simulation")
    assert inventory_report(intent, "a" * 64)["route"] == "human_only"
    assert inventory_report(intent, "a" * 64, ai_inventory="equipment_or_obstacles")["route"] == "needs_input"
    assert inventory_report(exercise_intent("unknown", "simulation"), "a" * 64, ai_inventory="ground_only")["route"] == "needs_input"


@pytest.mark.parametrize("change", [lambda d:d.update(basis="ai_verified"), lambda d:d.update(absence_verified=True),
    lambda d:d.update(inventory="ground_only",basis="not_declared"), lambda d:d.update(schema_version=True),
    lambda d:d.update(extra="field"), lambda d:d.update(output="full_recovered_scene")])
def test_bad_declaration_rejected(change):
    intent=exercise_intent("unknown", "body_preview");change(intent)
    with pytest.raises(ValueError): validate_intent(intent)


def make_job(root, monkeypatch, *, intent=None, ai=False, mode="exercise", ai_inventory=None):
    root.mkdir(); (root/"source.webm").write_bytes(b"mocked video; no decoding")
    config={"id":"a"*32,"source_file":"source.webm","source_sha256":digest(root/"source.webm"),"ai":ai,"mode":mode}
    if intent is not None: config["exercise_intent"]=intent
    write_json(root/"job.json",config)
    calls=[]
    def sample(video, out):
        calls.append("sample");out.mkdir();return {}
    def analyze(*args,**kwargs):
        calls.append("ai")
        return {"route":"exercise","summary":"Mock scene","camera":"fixed","people":1,"warnings":[],"objects":[],"exercise_inventory":ai_inventory or "unknown"}
    monkeypatch.setattr(pipeline,"sample_video",sample)
    monkeypatch.setattr(pipeline,"analyze_frames",analyze)
    import weft.exercise_scene_pipeline as scene
    import weft.exercise_body_source as body
    import weft.lab_worker as worker
    def construct(*args,**kwargs):
        calls.append(("simulation",kwargs));return {"status":"ready","lab_directory":str(root/"build/lab")}
    def preview(video, out, **kwargs):
        calls.append(("preview",kwargs));out.mkdir(parents=True)
        for name in ("human.json","human.xml","evidence.json","native-clock.json","manifest.json","tracks.gif"):
            (out/name).write_bytes(b"analytical fixture")
    monkeypatch.setattr(scene,"prepare_exercise_scene",construct)
    monkeypatch.setattr(body,"prepare_exercise_body_source",preview)
    monkeypatch.setattr(worker,"execute",lambda *_:pytest.fail("No unrelated physics or training"))
    return config,calls


@pytest.mark.parametrize("intent", [None,exercise_intent("unknown","simulation"),exercise_intent("equipment_or_obstacles","simulation")])
def test_missing_mixed_inventory_stops_before_api_sampling_or_inference(tmp_path,monkeypatch,intent):
    root=tmp_path/"job";config,calls=make_job(root,monkeypatch,intent=intent,ai=True)
    original=(root/"source.webm").read_bytes()
    assert pipeline.execute_job(root,project=tmp_path)==0
    assert not calls
    assert json.loads((root/"status.json").read_bytes())["status"]=="needs_input"
    report=json.loads((root/"analysis/scene-inventory.json").read_bytes())
    assert report["source_sha256"]==config["source_sha256"] and report["route"]=="needs_input"
    assert (root/"source.webm").read_bytes()==original and not (root/"failure.json").exists()


@pytest.mark.parametrize("inventory", ["ground_only","equipment_or_obstacles","unknown"])
def test_preview_keeps_body_only_without_any_simulator_or_inventory_absence(tmp_path,monkeypatch,inventory):
    root=tmp_path/"job";_,calls=make_job(root,monkeypatch,intent=exercise_intent(inventory,"body_preview"))
    assert pipeline.execute_job(root,project=tmp_path)==0
    assert [item[0] for item in calls if type(item) is tuple]==["preview"]
    options=calls[-1][1]
    assert options["native_timing"] is True and options["pose_identity"]=="raw" and options["max_seconds"]==180
    assert not {"scene_scope","train_settings","root_assistance"}&options.keys()
    assert json.loads((root/"status.json").read_bytes())["status"]=="preview"
    assert not (root/"build/candidate").exists()


def test_ground_declaration_reaches_original_native_full_reference_builder(tmp_path,monkeypatch):
    root=tmp_path/"job";_,calls=make_job(root,monkeypatch,intent=exercise_intent("ground_only","simulation"))
    assert pipeline.execute_job(root,project=tmp_path)==0
    assert calls[-1][0]=="simulation"
    options=calls[-1][1]
    assert options["scene_scope"]=="human_only" and options["native_timing"] and options["full_reference"]
    assert options["train_settings"]["trials"]==3


def test_ai_equipment_conflict_cannot_be_dropped_after_classification(tmp_path,monkeypatch):
    root=tmp_path/"job";_,calls=make_job(root,monkeypatch,intent=exercise_intent("ground_only","simulation"),ai=True,ai_inventory="equipment_or_obstacles")
    assert pipeline.execute_job(root,project=tmp_path)==0 and calls==["sample","ai"]
    assert json.loads((root/"analysis/scene-inventory.json").read_bytes())["declaration_conflict"]


def test_upload_persists_exact_intent_and_preview_survives_restart(tmp_path):
    root=tmp_path/"portal"
    portal=Portal(root,project=tmp_path,start=False)
    try:
        job=portal.upload(BytesIO(b"example"),7,name="clip.webm",mode="exercise",ai=False,request_id="c"*32,
                          exercise_inventory="equipment_or_obstacles",exercise_output="body_preview")
        folder=root/job["id"]
        assert job["exercise_intent"]==exercise_intent("equipment_or_obstacles","body_preview")
        write_json(folder/"status.json",{"status":"preview","route":"exercise"})
        (folder/"build/body_source").mkdir(parents=True);(folder/"build/body_source/tracks.gif").write_bytes(b"preview")
    finally:portal.close()
    reopened=Portal(root,project=tmp_path,start=False)
    try:
        summary=reopened.summary(job["id"])
        assert summary["status"]=="preview" and summary["sandbox"] is None
        assert summary["body_preview"]["image"].endswith("tracks.gif")
        assert reopened.cancel(job["id"])["status"]=="preview"
        with pytest.raises(ValueError,match="not ready"):reopened.lab(job["id"])
    finally:reopened.close()


@pytest.mark.parametrize("operation",["review","check","repair"])
def test_imported_mixed_lab_cannot_claim_unattached_body_review(tmp_path,operation):
    portal=Portal(tmp_path/"portal",project=tmp_path,start=False)
    try:
        job=portal.upload(BytesIO(b"example"),7,name="clip.webm",mode="exercise",ai=False,request_id="d"*32)
        config=portal.jobs[job["id"]];config.update(kind="imported_lab",inventory={"method":"saved_world"})
        write_json(portal.root/job["id"]/"status.json",{"status":"ready","route":"exercise"})
        with pytest.raises(ValueError,match="verified body-evidence attachment"):getattr(portal,operation)(job["id"])
        summary=portal.summary(job["id"])
        assert summary["inventory"]=={"method":"saved_world"} and not summary["support_experiment_available"]
        assert not (portal.root/job["id"]/"muse").exists()
    finally:portal.close()


def test_http_upload_parser_preserves_new_fields_without_server():
    seen={}
    def upload(stream,size,**kwargs):seen.update(kwargs);return {"status":"queued"}
    handler=SimpleNamespace(path="/api/projects?name=clip.webm&mode=exercise&ai=false&request_id="+"a"*32+"&exercise_inventory=unknown&exercise_output=body_preview",
        headers={"Content-Length":"7"},rfile=BytesIO(b"example"),connection=SimpleNamespace(settimeout=lambda _:None),
        server=SimpleNamespace(portal=SimpleNamespace(upload=upload)),route_lab=lambda:None,authorized=lambda:True,
        reply=lambda result,status=200:(status,result))
    assert PortalHandler.do_POST(handler)[0]==202
    assert seen["exercise_inventory"]=="unknown" and seen["exercise_output"]=="body_preview"
    handler.path+="&exercise_inventory=ground_only"
    assert PortalHandler.do_POST(handler)[0]==400


def test_new_ai_inventory_schema_remains_proposal_only_and_old_saved_plans_work():
    base={"route":"exercise","summary":"Person","camera":"fixed","people":1,"warnings":[],"objects":[]}
    assert pipeline.validate_plan(base)==base
    proposed=deepcopy(base);proposed["exercise_inventory"]="equipment_or_obstacles"
    assert pipeline.validate_plan(proposed)==proposed
    proposed["exercise_inventory"]="absence_verified"
    with pytest.raises(ValueError):pipeline.validate_plan(proposed)
