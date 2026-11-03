"""Synthetic provider fixtures verify custody and real offline MP4 request preparation."""

import base64
from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from weft import exercise_perception_pipeline as bridge
from weft.io import canonical_hash, digest, write_json
from weft.portal_pipeline import sample_video, INSTRUCTIONS, SCHEMA
from weft.openai_access import MODEL


@pytest.fixture
def inputs(tmp_path):
    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (160, 100))
    assert writer.isOpened()
    for i in range(16):
        frame = np.full((100, 160, 3), 220, np.uint8)
        cv2.rectangle(frame, (20+i, 40), (40+i, 70), (40, 80, 100), -1)
        writer.write(frame)
    writer.release()
    analysis = _fixture_analysis(video, tmp_path / "analysis")
    return video, analysis, tmp_path / "output"


def _fixture_analysis(video, analysis):
    frames = sample_video(video, analysis)
    plan = {"route":"exercise", "camera":"fixed", "people":1,
        "summary":"Synthetic response fixture; no real person or provider call", "warnings":[],
        "exercise_inventory":"equipment_or_obstacles",
        "objects":[{"label":"Synthetic support", "shape":"box", "role":"static", "box_xyxy":[.123,.4,.26,.71]}]}
    text = json.dumps({"video":{k:frames[k] for k in ("width","height","fps","decoded_frame_count")},
        "frames":[{k:r[k] for k in ("frame","decoder_time_s","nominal_time_s")} for r in frames["frames"]]})
    content = [{"type":"input_text", "text":text}]
    for row in frames["frames"]:
        content.append({"type":"input_image", "detail":"high", "image_url":"data:image/jpeg;base64,"+
                        base64.b64encode((analysis/row["artifact"]).read_bytes()).decode()})
    request = {"model":MODEL, "instructions":INSTRUCTIONS, "store":False,
        "input":[{"role":"user","content":content}], "max_output_tokens":3000,
        "text":{"format":{"type":"json_schema","name":"weft_scene_plan","strict":True,"schema":SCHEMA}}}
    pin = canonical_hash(request)
    saved = deepcopy(request);saved["input"][0]["content"] = content[:1]
    saved.update(images=frames["frames"], request_sha256=pin, scope="Synthetic fixture")
    response = {"id":"synthetic-response", "model":MODEL, "status":"completed",
        "output":[{"type":"message","content":[{"type":"output_text","text":json.dumps(plan)}]}]}
    receipt = {"request_sha256":pin, "response_id":"synthetic-response", "model":MODEL,
        "provider":"openai", "authentication":"server_project_api_key", "fixture":True}
    for name,value in (("plan.json",plan),("request.json",saved),("response.json",response),("receipt.json",receipt)):
        write_json(analysis/name,value)
    return analysis


def test_real_offline_request_uses_exact_source_and_outward_rounded_regions(inputs, monkeypatch):
    video, analysis, out = inputs
    import weft.meta_segmentation as meta
    monkeypatch.setattr(meta,"resolve_key",lambda *a:pytest.fail("Offline preparation cannot read credentials"))
    before = digest(video)
    result = bridge.prepare_exercise_perception(video, analysis, out)
    assert result["status"] == "prepared_not_sent" and result["api_calls"] == result["physics_steps"] == 0
    assert not result["simulation_ready"] and digest(video) == before
    prompt = json.loads((out/"plan.json").read_bytes())["prompts"]
    assert prompt["boxes"] == [[1,0,19,40,23,31]]
    payload = json.loads((out/"tracking/request.json").read_bytes())
    assert payload["model"] == "sam-3.1" and payload["store"] is False
    encoded = payload["input"][0]["content"][1]["video_url"].split(",",1)[1]
    assert base64.b64decode(encoded) == video.read_bytes()
    manifest = json.loads((out/"manifest.json").read_bytes())
    assert all(digest(out/name)==pin for name,pin in manifest["files"].items())
    assert manifest["children"]["tracking"] == digest(out/"tracking/manifest.json")


@pytest.mark.parametrize("name,field,value", [
    ("plan.json","people",2), ("plan.json","objects",[]),
    ("plan.json","exercise_inventory","unknown"),
    ("receipt.json","request_sha256","0"*64), ("response.json","status","incomplete"),
    ("frames.json","decoded_frame_count",501), ("request.json","store",True),
])
def test_changed_analysis_stops_before_output_or_provider(inputs, name, field, value):
    _, analysis, out = inputs
    obj=json.loads((analysis/name).read_bytes());obj[field]=value;write_json(analysis/name,obj)
    with pytest.raises(ValueError):bridge.read_equipment_analysis(analysis)
    assert not out.exists()


def test_modified_source_fails_before_preparation(inputs):
    video, analysis, out=inputs
    video.write_bytes(video.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="exact analyzed source"):
        bridge.prepare_exercise_perception(video,analysis,out)
    assert not out.exists()


def test_tracking_error_preserves_failure_without_retry(inputs, monkeypatch):
    video,analysis,out=inputs
    import weft.segmentation_lab as tracking
    calls=[]
    def fail(*args,**kwargs):
        calls.append(kwargs);raise RuntimeError("Synthetic no-call preparation failure")
    monkeypatch.setattr(tracking,"track_video",fail)
    with pytest.raises(RuntimeError):bridge.prepare_exercise_perception(video,analysis,out)
    assert len(calls)==1 and calls[0]["execute"] is False
    saved=json.loads((out/"failure.json").read_bytes())
    assert saved["retry"] is False and saved["stage"]=="prepare_tracking"
    assert (out/"plan.json").exists() and not (out/"result.json").exists()


def test_existing_tracking_must_match_ai_prompt_before_body_work(inputs,monkeypatch):
    video,analysis,out=inputs
    import weft.segmentation_evidence as segmentation
    monkeypatch.setattr(segmentation,"adapt_tracking_run",lambda _: {
        "source":{"submitted_video_sha256":digest(video)},
        "objects":[{"raw_model_id":"1","initialization":{"frame":0,"roi":[0,0,40,40]}}]})
    with pytest.raises(ValueError,match="initialization"):
        bridge.prepare_exercise_perception(video,analysis,out,saved_tracking=analysis)
    assert not out.exists()


def test_explicit_execution_needs_budget_and_body_before_output(inputs):
    video,analysis,out=inputs
    with pytest.raises(ValueError,match="allowance"):
        bridge.prepare_exercise_perception(video,analysis,out,execute_tracking=True,max_meta_cost_usd=.01)
    with pytest.raises(ValueError,match="pose model"):
        bridge.prepare_exercise_perception(video,analysis,out,execute_tracking=True,max_meta_cost_usd=.15)
    assert not out.exists()


def test_even_consistently_rehashed_different_localization_prompt_is_rejected(inputs):
    _,analysis,_=inputs
    request=json.loads((analysis/"request.json").read_bytes())
    request["instructions"]="Coordinates describe the LAST image, not the first"
    payload={k:v for k,v in request.items() if k not in {"images","request_sha256","scope"}}
    for row in request["images"]:
        payload["input"][0]["content"].append({"type":"input_image","detail":"high","image_url":
            "data:image/jpeg;base64,"+base64.b64encode((analysis/row["artifact"]).read_bytes()).decode()})
    pin=canonical_hash(payload)
    request["input"][0]["content"]=request["input"][0]["content"][:1]
    request["request_sha256"]=pin
    write_json(analysis/"request.json",request)
    receipt=json.loads((analysis/"receipt.json").read_bytes());receipt["request_sha256"]=pin
    write_json(analysis/"receipt.json",receipt)
    with pytest.raises(ValueError,match="payload"):
        bridge.read_equipment_analysis(analysis)


@pytest.mark.parametrize("changed",[False,True,"body_anchor"])
def test_saved_tracking_export_rejoins_exact_source_before_ready(inputs,monkeypatch,changed):
    video,analysis,out=inputs
    import weft.segmentation_evidence as segmentation
    import weft.segmentation_observations as observations
    import weft.shared_world_lab as worlds
    tracking=video.parent/"saved_tracking";tracking.mkdir();write_json(tracking/"manifest.json",{"fixture":True})
    body=video.parent/"body";body.mkdir()
    document={"source":{"submitted_video_sha256":digest(video)},
        "objects":[{"raw_model_id":"1","initialization":{"frame":0,"roi":[19,40,23,31]}},
                   {"raw_model_id":"2","initialization":{"frame":0,"roi":[50,10,40,80]}}]}
    monkeypatch.setattr(bridge,"body_tracking_anchor",lambda *a: {"prompt":[2,0,50,10,40,80],
        "scope":"Synthetic fixture only", "physical_identity_verified":False})
    monkeypatch.setattr(segmentation,"adapt_tracking_run",lambda _:deepcopy(document))
    def seed(source,target,**kwargs):
        target.mkdir();write_json(target/"scene.json",{"source_binding":{"source_sha256":digest(video)}})
        write_json(target/"manifest.json",{"fixture":True});return target
    monkeypatch.setattr(worlds,"prepare_shared_world_seed",seed)
    def export(source,target):
        target.mkdir();value=deepcopy(document)
        if changed:value["objects"][1 if changed=="body_anchor" else 0]["initialization"]["roi"][0]+=1
        write_json(target/"segmentation-evidence.json",value);write_json(target/"manifest.json",{"fixture":True})
    monkeypatch.setattr(observations,"export_tracking_evidence",export)
    if changed:
        with pytest.raises(ValueError,match="initialization"):
            bridge.prepare_exercise_perception(video,analysis,out,body_source=body,saved_tracking=tracking)
        assert not (out/"result.json").exists() and (out/"failure.json").exists()
    else:
        result=bridge.prepare_exercise_perception(video,analysis,out,body_source=body,saved_tracking=tracking)
        assert result["status"]=="evidence_ready" and result["api_calls"]==0 and result["source_mappings"]==[]
        assert result["tracking_manifest_sha256"]==digest(tracking/"manifest.json")


def _rehash_analysis(analysis):
    frames=json.loads((analysis/'frames.json').read_bytes())
    request=json.loads((analysis/'request.json').read_bytes())
    request['images']=frames['frames']
    request['input'][0]['content'][0]['text']=json.dumps({'video':{k:frames[k] for k in ('width','height','fps','decoded_frame_count')},
        'frames':[{k:r[k] for k in ('frame','decoder_time_s','nominal_time_s')} for r in frames['frames']]})
    payload=deepcopy({k:v for k,v in request.items() if k not in {'images','request_sha256','scope'}})
    for row in frames['frames']:
        payload['input'][0]['content'].append({'type':'input_image','detail':'high','image_url':'data:image/jpeg;base64,'+
            base64.b64encode((analysis/row['artifact']).read_bytes()).decode()})
    pin=canonical_hash(payload);request['request_sha256']=pin;write_json(analysis/'request.json',request)
    receipt=json.loads((analysis/'receipt.json').read_bytes());receipt['request_sha256']=pin;write_json(analysis/'receipt.json',receipt)


@pytest.mark.parametrize('change',['pixels','time','frame','fps'])
def test_every_rehashed_sample_must_match_actual_source(inputs,change):
    video,analysis,out=inputs
    frames=json.loads((analysis/'frames.json').read_bytes());row=frames['frames'][-1]
    if change=='pixels':
        (analysis/row['artifact']).write_bytes((analysis/frames['frames'][0]['artifact']).read_bytes())
        row['sha256']=digest(analysis/row['artifact'])
    elif change=='time':row['decoder_time_s']+=.1
    elif change=='frame':row['frame']=frames['decoded_frame_count']+2;row['nominal_time_s']=row['frame']/frames['fps']
    else:
        frames['fps']+=1
        for row in frames['frames']:row['nominal_time_s']=row['frame']/frames['fps']
    write_json(analysis/'frames.json',frames);_rehash_analysis(analysis)
    with pytest.raises(ValueError,match='source'):
        bridge.prepare_exercise_perception(video,analysis,out)
    assert not out.exists()


def _anchor_fixture(tmp_path,proposal,change=None):
    seed=tmp_path/'seed';seed.mkdir()
    points=[{'index':i,'position':[60+(i%3)*8,12+i*2], 'presence':.9,'visibility':.9} for i in range(33)]
    row={'frame':0,'object_id':'person-0','partition':'construction','visibility':'detected','id':'pose-0',
         'evidence_ids':['source-frame-0'],'measurements':{'landmarks':{'coordinate_system':'image','unit':'pixels','status':'inferred','value':points}}}
    document={'source':{'sha256':proposal['source_sha256']},'image':{'width_px':160,'height_px':100},'observations':[row]}
    if change:change(document)
    write_json(seed/'observations.json',document);pin=digest(seed/'observations.json')
    scene={'source_binding':{'source_sha256':proposal['source_sha256'],'evidence_documents':{pin:'observations.json'},
        'entity_bindings':[{'world_entity_id':'human','observation_document_sha256':pin,'observation_entity_id':'person-0'}]}}
    write_json(seed/'scene.json',scene)
    write_json(seed/'manifest.json',{'schema_version':1,'files':{'observations.json':pin,'scene.json':digest(seed/'scene.json')}})
    return seed,scene


def test_sealed_person_anchor_is_source_bound_and_never_a_rigid_box(inputs):
    video,analysis,out=inputs
    proposal,_=bridge.read_equipment_analysis(analysis);seed,scene=_anchor_fixture(video.parent,proposal)
    before=(seed/'observations.json').read_bytes()
    anchor=bridge.body_tracking_anchor(seed,scene,proposal)
    assert anchor['prompt']==[2,0,56,8,24,72]
    assert anchor['landmark_indices']==list(range(33)) and anchor['padding_pixels']==4
    assert not anchor['physical_identity_verified'] and 'Never a rigid body box' in anchor['scope']
    assert before==(seed/'observations.json').read_bytes()
    assert anchor['body_seed_manifest_sha256']==digest(seed/'manifest.json')


@pytest.mark.parametrize('change',[
    lambda d:d['observations'][0].update(partition='audit'),
    lambda d:d['observations'][0].update(frame=1),
    lambda d:d['observations'][0]['measurements']['landmarks']['value'][27].update(visibility=.2),
    lambda d:d['observations'][0]['measurements']['landmarks']['value'][11].update(position=[-1,20]),
    lambda d:d['source'].update(sha256='0'*64),
])
def test_anchor_abstains_without_original_usable_construction_landmarks(inputs,change):
    video,analysis,_=inputs;proposal,_=bridge.read_equipment_analysis(analysis)
    seed,scene=_anchor_fixture(video.parent,proposal,change)
    with pytest.raises(ValueError):bridge.body_tracking_anchor(seed,scene,proposal)


def test_four_equipment_regions_do_not_truncate_or_start_body_work(inputs,monkeypatch):
    video,analysis,out=inputs
    original=bridge.read_equipment_analysis
    def four(path):
        proposal,files=original(path);proposal['boxes']=[[i,0,19,40,23,31] for i in range(1,5)];return proposal,files
    monkeypatch.setattr(bridge,'read_equipment_analysis',four)
    import weft.segmentation_lab as tracking
    monkeypatch.setattr(tracking,'inspect_video',lambda *a,**k:pytest.fail('Count preflight must precede decode/body'))
    with pytest.raises(ValueError,match='three'):
        bridge.prepare_exercise_perception(video,analysis,out,execute_tracking=True,max_meta_cost_usd=.15)
    assert not out.exists()


def test_existing_body_prepares_person_plus_equipment_without_inference_or_api(inputs,monkeypatch):
    video,analysis,out=inputs;proposal,_=bridge.read_equipment_analysis(analysis)
    source,scene=_anchor_fixture(video.parent,proposal)
    import weft.shared_world_lab as worlds
    import weft.exercise_body_source as body
    import weft.meta_segmentation as meta
    from shutil import copytree
    def seed(source,target,**kwargs):
        copytree(source,target);write_json(target/'scene.json',scene);return target
    monkeypatch.setattr(worlds,'prepare_shared_world_seed',seed)
    monkeypatch.setattr(body,'prepare_exercise_body_source',lambda *a,**k:pytest.fail('No inference'))
    monkeypatch.setattr(meta,'resolve_key',lambda *a,**k:pytest.fail('No credential read'))
    result=bridge.prepare_exercise_perception(video,analysis,out,body_source=source)
    assert result['status']=='prepared_not_sent' and not result['assembly_inputs_ready']
    request=json.loads((out/'tracking/request.json').read_bytes())
    text=request['input'][0]['content'][0]['text']
    assert '1<|box' in text and '2<|box' in text
    manifest=json.loads((out/'manifest.json').read_bytes())
    assert manifest['files']['body-anchor.json']==digest(out/'body-anchor.json')


@pytest.fixture
def local_preflight_source(tmp_path):
    from test_exercise_body_source import pose_run, write_pose_files, seal
    from weft.exercise_body_source import prepare_exercise_body_source_from_evidence
    video=tmp_path/'original.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480));assert writer.isOpened()
    for _ in range(63):writer.write(np.full((480,640,3),180,np.uint8))
    writer.release()
    poses=pose_run(tmp_path/'poses')
    (poses/'source.webm').write_bytes(video.read_bytes())
    write_json(poses/'source.json',{'artifact':'source.webm','sha256':digest(video),'rights':'Analytical fixture, no physical person'})
    write_pose_files(poses,dropped=());seal(poses)
    body=tmp_path/'body';prepare_exercise_body_source_from_evidence(poses,body)
    return video,body


def test_real_sealed_preflight_revalidates_without_decode_inference_or_seed_rebuild(local_preflight_source,tmp_path,monkeypatch):
    import time
    import weft.exercise_body_source as bodies
    import weft.shared_world_lab as worlds
    import weft.segmentation_lab as tracking
    video,body=local_preflight_source
    prepared=bridge.prepare_exercise_perception_preflight(video,tmp_path/'preflight',body_source=body)
    assert prepared['status']=='local_ready' and prepared['inference_calls']==prepared['api_calls']==0
    assert prepared['body_anchor']['prompt'][0]==1 and prepared['metadata']['frame_count']==63
    def forbidden(*a,**k):pytest.fail('Validation cannot decode, infer or rebuild seed')
    monkeypatch.setattr(bodies,'prepare_exercise_body_source',forbidden)
    monkeypatch.setattr(worlds,'prepare_shared_world_seed',forbidden)
    monkeypatch.setattr(tracking,'inspect_video',forbidden)
    monkeypatch.setattr(cv2,'VideoCapture',forbidden)
    validated=bridge.validate_perception_preflight(video,tmp_path/'preflight',deadline=time.monotonic()+10)
    assert validated==prepared
    assert validated['base_manifest_sha256']==digest(tmp_path/'preflight/body_seed/manifest.json')


@pytest.mark.parametrize('changed',['video','metadata','anchor','seed','failure','implementation'])
def test_preflight_tampering_stops_reuse(local_preflight_source,tmp_path,changed):
    import time
    video,body=local_preflight_source;p=tmp_path/'preflight'
    bridge.prepare_exercise_perception_preflight(video,p,body_source=body)
    if changed=='video':video.write_bytes(video.read_bytes()+b'changed')
    elif changed=='failure':write_json(p/'failure.json',{'error':'preserved failure'})
    else:
        path={'metadata':p/'metadata.json','anchor':p/'body-anchor.json','seed':p/'body_seed/human.json',
              'implementation':p/'implementation/exercise_perception_pipeline.py'}[changed]
        path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError):bridge.validate_perception_preflight(video,p,deadline=time.monotonic()+10)


def test_preflight_no_model_stops_before_decode_or_output(inputs,monkeypatch):
    video,_,out=inputs
    import weft.segmentation_lab as tracking
    monkeypatch.setattr(tracking,'inspect_video',lambda *a,**k:pytest.fail('Missing model precedes decoding'))
    with pytest.raises(ValueError,match='Local pose model'):
        bridge.prepare_exercise_perception_preflight(video,out)
    assert not out.exists()


@pytest.mark.parametrize('field,value',[('width',32),('height',2200),('frame_count',501),('fps',0),('fps',1)])
def test_preflight_intersects_portal_sampling_envelope(inputs,monkeypatch,field,value):
    video,_,out=inputs;body=video.parent/'body';body.mkdir()
    import weft.segmentation_lab as tracking
    metadata={'source_sha256':digest(video),'width':160,'height':100,'frame_count':63,'fps':30.,
              'decoder_timestamps_s':[i/30 for i in range(63)]};metadata[field]=value
    monkeypatch.setattr(tracking,'inspect_video',lambda *a,**k:metadata)
    with pytest.raises(ValueError,match='metadata'):
        bridge.prepare_exercise_perception_preflight(video,out,body_source=body)
    assert not out.exists()


def test_bridge_reuses_exact_preflight_without_copy_or_reconstruction(local_preflight_source,tmp_path,monkeypatch):
    import weft.exercise_body_source as bodies
    import weft.shared_world_lab as worlds
    import weft.meta_segmentation as meta
    video,body=local_preflight_source;p=tmp_path/'preflight'
    prepared=bridge.prepare_exercise_perception_preflight(video,p,body_source=body)
    analysis=_fixture_analysis(video,tmp_path/'analysis')
    before={f.relative_to(p).as_posix():digest(f) for f in p.rglob('*') if f.is_file()}
    def forbidden(*a,**k):pytest.fail('Prepared seed must be reused without inference, rebuild or API')
    monkeypatch.setattr(bodies,'prepare_exercise_body_source',forbidden)
    monkeypatch.setattr(worlds,'prepare_shared_world_seed',forbidden)
    monkeypatch.setattr(meta,'resolve_key',forbidden)
    events=[];out=tmp_path/'bridge'
    result=bridge.prepare_exercise_perception(video,analysis,out,preflight=p,progress=events.append)
    assert result['status']=='prepared_not_sent' and result['preflight_descriptor']=='preflight-descriptor.json'
    descriptor=json.loads((out/'preflight-descriptor.json').read_bytes())
    assert descriptor['preflight_manifest_sha256']==prepared['manifest_sha256']
    assert descriptor['base_run']==str(p/'body_seed') and not (out/'body_seed').exists()
    anchor=json.loads((out/'body-anchor.json').read_bytes())
    assert anchor=={**prepared['body_anchor'],'prompt':[2,*prepared['body_anchor']['prompt'][1:]]}
    manifest=json.loads((out/'manifest.json').read_bytes())
    assert 'body_seed' not in manifest['children'] and manifest['files']['preflight-descriptor.json']==digest(out/'preflight-descriptor.json')
    assert before=={f.relative_to(p).as_posix():digest(f) for f in p.rglob('*') if f.is_file()}
    assert len(events)<=10 and 'validating_preflight' in [row['stage'] for row in events]
    assert all(set(row)=={'stage','message','seconds'} for row in events)


def test_preflight_cannot_be_combined_with_body_override(inputs):
    video,analysis,out=inputs
    with pytest.raises(ValueError,match='owns its original'):
        bridge.prepare_exercise_perception(video,analysis,out,preflight=video.parent/'preflight',body_source=video.parent/'body')
    assert not out.exists()


def test_generated_preflight_prepares_body_once_and_preserves_failed_child(local_preflight_source,tmp_path,monkeypatch):
    import time
    from shutil import copytree
    import weft.exercise_body_source as bodies
    import weft.meta_segmentation as meta
    video,body=local_preflight_source;model=tmp_path/'local-model.task';model.write_bytes(b'analytical stub only')
    calls=[]
    def fake_pose(source,out,**kwargs):
        calls.append(kwargs);copytree(body,out);return out
    monkeypatch.setattr(bodies,'prepare_exercise_body_source',fake_pose)
    monkeypatch.setattr(meta,'resolve_key',lambda *a:pytest.fail('No provider credential access'))
    p=tmp_path/'preflight';result=bridge.prepare_exercise_perception_preflight(video,p,pose_model=model)
    assert len(calls)==1 and calls[0]['native_timing'] is True
    assert result['inference_calls']==1 and result['api_calls']==0
    assert bridge.validate_perception_preflight(video,p,deadline=time.monotonic()+10)==result
    assert len(calls)==1
    write_json(p/'body_source/failure.json',{'error':'late local failure'})
    with pytest.raises(ValueError,match='failure marker'):
        bridge.validate_perception_preflight(video,p,deadline=time.monotonic()+10)


def test_preflight_seed_failure_marker_is_not_ignored(local_preflight_source,tmp_path):
    import time
    video,body=local_preflight_source;p=tmp_path/'preflight'
    bridge.prepare_exercise_perception_preflight(video,p,body_source=body)
    write_json(p/'body_seed/failure.json',{'error':'late failure'})
    with pytest.raises(ValueError,match='construction-only'):
        bridge.validate_perception_preflight(video,p,deadline=time.monotonic()+10)


def test_progress_callback_cannot_bypass_source_or_body_options(inputs):
    video,analysis,out=inputs
    with pytest.raises(ValueError,match='callback'):
        bridge.prepare_exercise_perception(video,analysis,out,progress='not callable')
    assert not out.exists()


@pytest.mark.parametrize('change',['anchor','metadata','api_type'])
def test_resealed_preflight_cannot_reinterpret_body_source(local_preflight_source,tmp_path,change):
    import time
    video,body=local_preflight_source;p=tmp_path/'preflight'
    bridge.prepare_exercise_perception_preflight(video,p,body_source=body)
    result=json.loads((p/'result.json').read_bytes())
    if change=='anchor':
        anchor=json.loads((p/'body-anchor.json').read_bytes());anchor['prompt'][2]+=1
        write_json(p/'body-anchor.json',anchor);result['body_anchor']=anchor
    elif change=='metadata':
        metadata=json.loads((p/'metadata.json').read_bytes());metadata['fps']=29.
        write_json(p/'metadata.json',metadata);result['metadata']=metadata
    else:result['api_calls']=False
    write_json(p/'result.json',result)
    manifest=json.loads((p/'manifest.json').read_bytes())
    manifest['files']={name:digest(p/name) for name in manifest['files']};write_json(p/'manifest.json',manifest)
    with pytest.raises(ValueError):bridge.validate_perception_preflight(video,p,deadline=time.monotonic()+10)
