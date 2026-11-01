"""Saved-pose annotation joins and collector composition without native work."""

from copy import deepcopy
from hashlib import sha256
import importlib
import json
import time

import pytest

from test_multiview_landmark_evidence import denied, local_only


@pytest.fixture
def api():
    return importlib.import_module("weft.pose_surface_annotations")


def raw(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def fixture(count=6, *, pose_change=None, observation_change=None, names=("left_ankle",)):
    source = "a"*64
    poses = []
    for index in range(0, count-1, 2):
        poses.append({"id":f"{source}:frame:{index}","frame":index,"time_s":index/30,
            "decoder_timestamp_s":index/30+.001,"status":"tracked","pose_count":1,
            "landmarks":[{"x":.2+i*.01,"y":.4+i*.001,"z":900.,"visibility":.9,"presence":.8} for i in range(33)]})
    pose = {"schema_version":1,"profile":"exercise-pose-v1","source_sha256":source,
        "fps":30.,"frame_count":count,"decoded_frames":count-1,"width":100,"height":80,"frames":poses,
        "pose_model":{"scope":"analytical source estimates, no inference","sha256":"b"*64},"limits":[]}
    if pose_change is not None: pose_change(pose)
    observations = importlib.import_module("weft.observations").adapt_exercise(pose)
    if observation_change is not None: observation_change(observations)
    observation_raw=raw(observations); digest=sha256(observation_raw).hexdigest()
    channels=[{"id":"channel-"+str(i),"entity_ref":{"document_sha256":digest,"object_id":"person-0"},
               "point_role":name,"surface_id":"surface-0"} for i,name in enumerate(names)]
    return {"observation_documents":{digest:observation_raw},"pose_document":raw(pose),"source_sha256":source,
        "channels":channels,"landmark_bindings":[{"channel_id":c["id"],"landmark_name":c["point_role"],"uncertainty_px":2.}
                                                 for c in channels],
        "minimum_confidence":.5,"uncertainty_method":"Explicit analytical two-pixel range",
        "uncertainty_note":"Authored assumption, not a measured accuracy bound or derived from confidence"}


def test_exact_pixels_and_all_original_slots_preserve_uncertainty_distinction(api):
    inputs=fixture(); result=api.prepare_pose_surface_annotations(**inputs)
    annotation=json.loads(result.annotation_document()); rows=annotation["annotations"]
    assert len(rows)==3
    assert rows[0]["pixel"]==pytest.approx([47.,34.16])
    assert rows[0]["uncertainty_px"]==2.
    detail=result.frame(0)["channels"][0]
    assert detail["source_confidence"]==.8 and detail["source_visibility_score"]==.9
    assert detail["source_presence_score"]==.8 and detail["authored_uncertainty_px"]==2.
    assert result.frame(0)["decoder_timestamp_s"]==.001 and result.frame(0)["pose_time_s"]==0.
    assert [result.frame(i)["channels"][0]["reason"] for i in range(6)] == [None,"source_not_sampled",None,"source_not_sampled",None,"metadata_only_not_decoded"]
    meta=result.describe()
    assert meta["counts"]["retained_source_frames"]==6 and meta["counts"]["channel_slots"]==6
    assert meta["counts"]["qualified_record_payload_checks"]==3
    assert not meta["raw_identity_authenticated"] and not meta["confidence_is_pixel_error_bound"]
    assert meta["geometry_evaluations"]==0 and meta["contact"]=="unknown"
    provenance=json.loads(annotation["provenance"])
    assert provenance["pose_raw_sha256"]==sha256(inputs["pose_document"]).hexdigest()
    assert provenance["uncertainty_status"]=="authored_pixel_range_not_calibrated_from_confidence"
    assert "not_authenticated" in provenance["coordinate_policy"]


def test_heel_and_ankle_keep_their_own_exact_indices(api):
    result=api.prepare_pose_surface_annotations(**fixture(names=("left_heel","left_ankle","right_wrist")))
    details=result.frame(0)["channels"]
    assert [d["landmark_index"] for d in details]==[29,27,16]
    assert [d["pixel"][0] for d in details]==pytest.approx([49.,47.,36.])
    assert result.describe()["counts"]["qualified_record_payload_checks"]==3


@pytest.mark.parametrize("name", ["left_heel","equipment_handle","unknown"])
def test_no_implicit_anatomical_or_equipment_substitution(api,name):
    inputs=fixture(); inputs["landmark_bindings"][0]["landmark_name"]=name
    with pytest.raises(ValueError,match="endpoint substitution"):
        api.prepare_pose_surface_annotations(**inputs)


def test_withheld_and_unavailable_source_never_supply_landmark_payloads(api):
    def change(observations):
        observations["observations"][0]["partition"]="audit"
        observations["evidence"][0]["payload"]["landmarks"]="withheld_payload_not_an_array"
        observations["observations"][4]["visibility"]="not_detected"
        observations["observations"][4]["measurements"]={}
    result=api.prepare_pose_surface_annotations(**fixture(observation_change=change))
    assert result.frame(0)["channels"][0]["reason"]=="excluded_evidence_partition"
    assert result.frame(0)["channels"][0]["source_confidence"] is None
    assert result.frame(0)["channels"][0]["representation"]=="not_read_unqualified_source"
    assert result.frame(4)["channels"][0]["reason"]=="source_entity_unavailable"
    assert result.describe()["counts"]["qualified_record_payload_checks"]==1
    assert len(json.loads(result.annotation_document())["annotations"])==1


@pytest.mark.parametrize("field,reason", [
    ("confidence","below_authored_confidence_threshold"),
    ("outside","outside_or_unavailable_source_image"),
    ("missing_score","unknown_landmark_confidence"),
    ("multiple","unavailable_or_ambiguous_pose"),
])
def test_unusable_estimates_are_unknown_not_filled(api,field,reason):
    def change(pose):
        frame=pose["frames"][0]; point=frame["landmarks"][27]
        if field=="confidence": point["visibility"]=.2
        if field=="outside": point["x"]=1.1
        if field=="missing_score": point["presence"]=None
        if field=="multiple": frame["pose_count"]=2
    result=api.prepare_pose_surface_annotations(**fixture(pose_change=change))
    detail=result.frame(0)["channels"][0]
    assert detail["reason"]==reason and detail["pixel"] is None
    annotation=json.loads(result.annotation_document())["annotations"][0]
    assert annotation["status"]=="unknown" and annotation["pixel"] is annotation["uncertainty_px"] is None
    assert detail["authored_uncertainty_px"]==2.


@pytest.mark.parametrize("multiplicity", ["missing",True,1.0,1.5,2])
def test_pose_multiplicity_requires_explicit_integer_one(api,multiplicity):
    def change(pose):
        if multiplicity=="missing": pose["frames"][0].pop("pose_count")
        else: pose["frames"][0]["pose_count"]=multiplicity
    result=api.prepare_pose_surface_annotations(**fixture(pose_change=change))
    detail=result.frame(0)["channels"][0]
    assert detail["reason"]=="unavailable_or_ambiguous_pose" and detail["pixel"] is None
    annotation=json.loads(result.annotation_document())["annotations"][0]
    assert annotation["status"]=="unknown" and annotation["pixel"] is None
    assert result.describe()["counts"]["copied_estimates"]==2
    assert result.describe()["counts"]["channel_slots"]==6


@pytest.mark.parametrize("field,value", [("frame_count",True),("frame_count",1.0),("frame_count","missing"),
                                         ("decoded_frames",True),("decoded_frames",1.0),("decoded_frames","missing")])
def test_source_denominators_require_explicit_integers(api,field,value):
    inputs=fixture(2)
    pose=json.loads(inputs["pose_document"]); pose["frame_count"]=1
    if value=="missing": pose.pop(field)
    else: pose[field]=value
    observation=json.loads(next(iter(inputs["observation_documents"].values())))
    observation["source"]["metadata_frames"]=1
    observation["provenance"]["input_evidence_sha256"]=api.canonical_hash(pose)
    encoded=raw(observation); digest=sha256(encoded).hexdigest()
    inputs.update(pose_document=raw(pose),observation_documents={digest:encoded})
    inputs["channels"][0]["entity_ref"]["document_sha256"]=digest
    with pytest.raises(ValueError,match="denominators"):
        api.prepare_pose_surface_annotations(**inputs)


@pytest.mark.parametrize("marker,reason",[("identity_raw","identity_derived_requires_separate_original_proof"),
                                         ("image_refinement","crop_derived_requires_separate_original_proof")])
def test_derived_arrays_are_not_silently_original_pixels(api,marker,reason):
    def change(pose):
        frame=pose["frames"][0]
        frame[marker]=deepcopy(frame) if marker=="identity_raw" else {"method":"unverified_crop_fixture"}
        frame["landmarks"][27]["x"]=.9
    result=api.prepare_pose_surface_annotations(**fixture(pose_change=change))
    assert result.frame(0)["channels"][0]["reason"]==reason
    assert result.frame(0)["channels"][0]["pixel"] is None
    assert result.describe()["counts"]["copied_estimates"]==2


def declared_raw_pose(pose):
    pose["pose_identity"]={"mode":"raw","audit_artifact":"identity-audit.json","repaired_frames":0,
                           "raw_artifact_sha256":"c"*64}
    for frame in pose["frames"]:
        frame["identity_status"]="single_model_hypothesis_not_verified_person_count"


def test_known_raw_producer_label_retains_unverified_identity(api):
    result=api.prepare_pose_surface_annotations(**fixture(pose_change=declared_raw_pose))
    assert result.describe()["counts"]["copied_estimates"]==3
    detail=result.frame(0)["channels"][0]
    assert detail["pixel"]==pytest.approx([47.,34.16])
    assert detail["representation"]=="declared_raw_model_estimate_not_authenticated_identity"
    provenance=json.loads(json.loads(result.annotation_document())["provenance"])
    assert provenance["identity_declaration"]["status"]=="consistent_raw_declaration_not_authenticated"
    assert provenance["identity_declaration"]["mode"]=="raw"
    assert provenance["identity_declaration"]["repaired_frames"]==0
    assert not result.describe()["raw_identity_authenticated"]


def test_raw_producer_no_hypothesis_slot_stays_unavailable(api):
    def change(pose):
        declared_raw_pose(pose)
        pose["frames"][1].update(status="no_pose", pose_count=0, landmarks=[],
                                 identity_status="no_model_hypothesis")
    result=api.prepare_pose_surface_annotations(**fixture(pose_change=change))
    detail=result.frame(2)["channels"][0]
    assert detail["reason"]=="source_entity_unavailable" and detail["pixel"] is None
    assert detail["representation"]=="not_read_unqualified_source"
    assert result.describe()["counts"]["copied_estimates"]==2
    assert result.describe()["counts"]["qualified_record_payload_checks"]==2
    assert result.describe()["counts"]["retained_source_frames"]==6


@pytest.mark.parametrize("declaration", ["missing",None,{"mode":"temporal","repaired_frames":0},
    {"mode":"unknown","repaired_frames":0},{"mode":"raw","repaired_frames":1},
    {"mode":"raw","repaired_frames":True},{"mode":"raw","repaired_frames":False},
    {"mode":"raw","repaired_frames":0.0},{"mode":"raw"}])
def test_known_raw_label_requires_consistent_explicit_declaration(api,declaration):
    def change(pose):
        declared_raw_pose(pose)
        if declaration=="missing": pose.pop("pose_identity")
        else: pose["pose_identity"]=declaration
    result=api.prepare_pose_surface_annotations(**fixture(pose_change=change))
    assert result.describe()["counts"]["copied_estimates"]==0
    assert result.frame(0)["channels"][0]["reason"] in {
        "raw_identity_declaration_required","unsupported_or_inconsistent_identity_declaration"}


@pytest.mark.parametrize("key,value", [("identity_raw",{}),("identity_derivation_id",None),
    ("identity_recompute_required",False),("image_refinement",{}),("image_refinement_raw",{}),
    ("identity_status","temporally_reassigned_not_verified_anatomy"),("identity_status","unknown"),
    ("identity_status",None),("identity_status","")])
def test_raw_label_cannot_hide_derivation_or_unknown_status(api,key,value):
    def change(pose):
        declared_raw_pose(pose); pose["frames"][0][key]=value
    result=api.prepare_pose_surface_annotations(**fixture(pose_change=change))
    assert result.describe()["counts"]["copied_estimates"]==2
    assert result.frame(0)["channels"][0]["pixel"] is None
    assert result.frame(0)["channels"][0]["reason"] is not None


@pytest.mark.parametrize("declaration", [None,{"mode":"temporal","repaired_frames":0},
    {"mode":"unknown","repaired_frames":0},{"mode":"raw","repaired_frames":1}])
def test_unmarked_legacy_rows_do_not_override_bad_top_identity(api,declaration):
    result=api.prepare_pose_surface_annotations(**fixture(pose_change=lambda pose:pose.update(pose_identity=declaration)))
    assert result.describe()["counts"]["copied_estimates"]==0
    assert result.frame(0)["channels"][0]["reason"]=="unsupported_or_inconsistent_identity_declaration"


@pytest.mark.parametrize("multiplicity", ["missing",True,1.0,2])
def test_declared_raw_label_does_not_override_exact_integer_count(api,multiplicity):
    def change(pose):
        declared_raw_pose(pose)
        if multiplicity=="missing": pose["frames"][0].pop("pose_count")
        else: pose["frames"][0]["pose_count"]=multiplicity
    result=api.prepare_pose_surface_annotations(**fixture(pose_change=change))
    assert result.frame(0)["channels"][0]["reason"]=="unavailable_or_ambiguous_pose"
    assert result.describe()["counts"]["copied_estimates"]==2


@pytest.mark.parametrize("change,reason",[
    (lambda d:d["provenance"].update(input_evidence_sha256="f"*64),"canonical_pose_document_binding_differs"),
    (lambda d:d["evidence"][0]["payload"]["landmarks"][27].update(x=.8),"source_pose_payload_differs"),
    (lambda d:d["evidence"][0].update(kind="equipment_hypothesis"),"source_pose_entity_evidence_differs"),
])
def test_exact_document_and_row_custody_required(api,change,reason):
    result=api.prepare_pose_surface_annotations(**fixture(observation_change=change))
    assert result.frame(0)["channels"][0]["reason"]==reason
    assert result.frame(0)["channels"][0]["pixel"] is None


def test_source_pixels_are_not_silently_rescaled_from_different_image(api):
    def change(pose): pose.update(source_width=200,source_height=160)
    result=api.prepare_pose_surface_annotations(**fixture(pose_change=change))
    assert result.describe()["counts"]["copied_estimates"]==0
    assert result.frame(0)["channels"][0]["reason"]=="pose_dimensions_not_original_source"


@pytest.mark.parametrize("key,value",[("minimum_confidence",True),("minimum_confidence",1.1),
                                     ("uncertainty_method",""),("uncertainty_note","x"*193)])
def test_explicit_authored_policy_is_bounded(api,key,value):
    inputs=fixture(); inputs[key]=value
    with pytest.raises(ValueError,match="explicit"):
        api.prepare_pose_surface_annotations(**inputs)


def test_uncertainty_is_never_derived_from_changed_confidence(api):
    low=fixture(pose_change=lambda pose:pose["frames"][0]["landmarks"][27].update(presence=.6))
    high=fixture(pose_change=lambda pose:pose["frames"][0]["landmarks"][27].update(presence=.99))
    first=api.prepare_pose_surface_annotations(**low).frame(0)["channels"][0]
    second=api.prepare_pose_surface_annotations(**high).frame(0)["channels"][0]
    assert first["source_confidence"]!=second["source_confidence"]
    assert first["authored_uncertainty_px"]==second["authored_uncertainty_px"]==2.


@pytest.mark.parametrize("raw_declaration", [False,True], ids=["unmarked","declared_raw"])
def test_annotation_provenance_survives_actual_collector_and_envelope(api,raw_declaration):
    inputs=fixture(pose_change=declared_raw_pose if raw_declaration else None)
    result=api.prepare_pose_surface_annotations(**inputs)
    collection_api=importlib.import_module("weft.exercise_surface_inputs")
    base=importlib.import_module("test_source_surface_envelopes").fixture(6)
    camera_raw=raw(base[1]); surface=base[2]
    surface["source_sha256"]=inputs["source_sha256"]; surface["calibration_sha256"]=sha256(camera_raw).hexdigest()
    surface["surfaces"][0]["id"]="surface-0"
    names=importlib.import_module("weft.source_surface_interval_construction").NAMES
    body={"schema_version":3,"profile":"exercise-v3","source_sha256":inputs["source_sha256"],"dt":1/30,
        "reference":[{"time_s":i/30,"joints":{name:0. for name in names}} for i in (0,2,4)],
        "reference_evidence":[{"time_s":i/30,"evidence_id":f"{inputs['source_sha256']}:frame:{i}"} for i in (0,2,4)],
        "root_reference":[{"time_s":i/30,"translation":[0.,0.,0.]} for i in (0,2,4)]}
    collection=collection_api.prepare_exercise_surface_inputs(inputs["observation_documents"],source_sha256=inputs["source_sha256"],
        channels=inputs["channels"],camera_document=camera_raw,body_document=raw(body),
        human_entity_ref=inputs["channels"][0]["entity_ref"])
    request=json.loads(collection.request_document()); request.update(weight=2.,surface_intervals=surface,
        annotations=json.loads(result.annotation_document()))
    claim={"status":"assumed","method":"Explicit analytical body/world map",
           "uncertainty":"Authored fixture, not physical calibration or anatomical proof","evidence_refs":[]}
    request["body_binding"]["body_to_world"]={"schema_version":1,"profile":"world-alignment-v1",
        "from_frame":collection.describe()["body_coordinate_frame"],"to_frame":surface["world_frame"],
        "spatial":{"scale":1.,"rotation_wxyz":[1.,0.,0.,0.],"translation":[0.,0.,0.]},
        "clock":{"rate":1.,"source_origin_s":0.,"target_origin_s":0.},"claims":{"spatial":claim,"clock":deepcopy(claim)}}
    request["body_binding"]["channels"][0].update(body_point="left_ankle",status="assumed",
        method="Explicit same named ankle correspondence",uncertainty="Not verified anatomy or contact")
    qualified=collection.qualify(raw(request)); files=qualified.files()
    preserved=json.loads(files["surface-annotations.json"])
    assert preserved==json.loads(result.annotation_document())
    assert preserved["provenance"]==json.loads(result.annotation_document())["provenance"]
    if raw_declaration:
        assert json.loads(preserved["provenance"])["identity_declaration"]["status"]=="consistent_raw_declaration_not_authenticated"
    assert qualified.describe()["conditional_envelopes"]==3
    assert qualified.describe()["conditional_body_channel_terms"]==3
    assert qualified.frame(0)["body_join"]["channels"][0]["status"]=="conditional_interval_ready"
    assert qualified.frame(0)["image_admission"]["status"]=="construction_image_source_ready"
    assert qualified.frame(0)["envelope"]["anchors"][0]["status"]=="conditional_surface_envelope"
    assert json.loads(files["surface-config.json"])["annotations"]["sha256"]==sha256(files["surface-annotations.json"]).hexdigest()
    assert qualified.describe()["contact"]=="unknown" and not qualified.describe()["fit_or_simulator_admission"]


def test_input_bounds_and_cached_queries_are_no_reprocessing(api,monkeypatch):
    inputs=fixture(); result=api.prepare_pose_surface_annotations(**inputs)
    for name in ("_sources","_document","canonical_hash"):
        monkeypatch.setattr(api,name,denied)
    for i in range(6): assert result.frame(i)["source_frame"]==i
    detached=result.frame(0); detached["channels"].clear()
    assert len(result.frame(0)["channels"])==1
    inputs["pose_document"]=b" "*(api.MAX_POSE_BYTES+1)
    with pytest.raises(ValueError,match="eight MiB"):
        api.prepare_pose_surface_annotations(**inputs)


def test_timeout_does_not_publish_partial_rows(api,monkeypatch):
    inputs=fixture(); monkeypatch.setattr(api,"MAX_SECONDS",0.)
    with pytest.raises(TimeoutError): api.prepare_pose_surface_annotations(**inputs)


def test_critical_annotation_provenance_cannot_be_silently_truncated(api):
    inputs=fixture(names=tuple(api.LANDMARKS)[:12])
    for i,(channel,binding) in enumerate(zip(inputs["channels"],inputs["landmark_bindings"],strict=True)):
        channel["id"]=binding["channel_id"]="channel-"+"x"*68+str(i)
    inputs.update(uncertainty_method="m"*192,uncertainty_note="u"*192)
    with pytest.raises(ValueError,match="complete output exceeds"):
        api.prepare_pose_surface_annotations(**inputs)


def test_complete_large_input_and_query_cost(api,record_property):
    inputs=fixture(1800,names=("left_ankle","right_wrist"))
    # Compact the identical synthetic JSON, not the source coverage or byte cap.
    document=json.loads(next(iter(inputs["observation_documents"].values())))
    compact=json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    digest=sha256(compact).hexdigest(); inputs["observation_documents"]={digest:compact}
    for channel in inputs["channels"]: channel["entity_ref"]["document_sha256"]=digest
    started=time.monotonic(); result=api.prepare_pose_surface_annotations(**inputs); seconds=time.monotonic()-started
    started=time.monotonic()
    for i in range(1000): assert result.frame(i)["source_frame"]==i
    query_seconds=time.monotonic()-started
    meta=result.describe()
    assert meta["counts"]["retained_pose_slots"]==900 and meta["counts"]["retained_source_frames"]==1800
    assert meta["counts"]["channel_slots"]==3600 and meta["counts"]["copied_estimates"]==1800
    assert meta["counts"]["qualified_record_payload_checks"]==900
    assert meta["cached_row_bytes"]+meta["annotation_bytes"]<=api.MAX_CACHE_BYTES
    measurement={"frames":1800,"pose_slots":900,"channels":2,"prepare_seconds":seconds,
        "cached_1000_queries_seconds":query_seconds,"source_input_bytes":sum(map(len,inputs["observation_documents"].values()))+len(inputs["pose_document"]),
        "cached_row_bytes":meta["cached_row_bytes"],"annotation_bytes":meta["annotation_bytes"],"geometry_evaluations":0}
    record_property("bounded_pose_annotation_measurement",json.dumps(measurement,sort_keys=True))
    print(json.dumps(measurement,sort_keys=True))
