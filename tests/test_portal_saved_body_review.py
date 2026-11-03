"""Saved-body review custody and scope, using pure analytical metadata only."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import pytest
from weft import portal_review as review
from weft.io import digest,write_json
from weft.portal_inventory import exercise_intent
from weft.person_inventory import make_conditional_person_inventory


def seal(root,**extra):
    write_json(root/'manifest.json',{'schema_version':1,**extra,'files':{p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file() and p!=root/'manifest.json'}})
    return digest(root/'manifest.json')


@pytest.fixture
def job(tmp_path,monkeypatch):
    root=tmp_path/('a'*32);world=root/'saved-body/world';world.mkdir(parents=True)
    (world/'source.mp4').write_bytes(b'Analytical bytes, not decoded media')
    source=digest(world/'source.mp4');intent=exercise_intent('unknown','simulation')
    inventory=make_conditional_person_inventory(source,intent)
    write_json(world/'human.json',{'profile':'exercise-v3','source_sha256':source})
    write_json(world/'evidence.json',{'source_sha256':source,'reference':[{'frame':0,'id':source+':frame:0'}]})
    write_json(world/'inventory.json',inventory);write_json(world/'scene.json',{'fixture':'not source-admitted'})
    (world/'inputs').mkdir()
    original={'profile':'exercise-body-source-v1','status':'construction_only','simulation_ready':False,'files':{n:digest(world/n) for n in ('source.mp4','human.json','evidence.json')}}
    write_json(world/'inputs/body-source-manifest.json',original);body_pin=digest(world/'inputs/body-source-manifest.json')
    world_pin=seal(world,profile='shared-world-v3',status='ready')
    request={'profile':'portal-saved-body-v1','child_id':root.name,'parent_id':'b'*32,'parent_files':{'job.json':'c'*64},'body_manifest_sha256':body_pin,'identity':'d'*64,'original_exercise_intent':intent,'payload':{'request_id':'e'*32,'preflight_manifest_sha256':'f'*64,'modeling_scope':'body_authored_ground_only'}}
    write_json(root/'saved-body/request.json',request)
    receipt={'profile':'portal-saved-body-v1','parent_id':'b'*32,'source_sha256':source,'parent_files':request['parent_files'],'body_manifest_sha256':body_pin,'world_manifest_sha256':world_pin,'inventory':inventory}
    write_json(root/'saved-body/receipt.json',receipt)
    lab=root/'lab/run';lab.mkdir(parents=True);(lab/'scene.json').write_bytes((world/'scene.json').read_bytes())
    write_json(lab/'policy.json',{'secret':'DO_NOT_SEND_POLICY'});lab_pin=seal(lab)
    write_json(root/'lab/lab-config.json',{'manifest_sha256':lab_pin})
    write_json(root/'job.json',{'id':root.name,'kind':'saved_body','parent_id':'b'*32,'source_sha256':source,'saved_body_identity':'d'*64,'saved_body_request_sha256':digest(root/'saved-body/request.json')})
    write_json(root/'status.json',{'status':'ready','route':'exercise','lab':'lab','receipt_sha256':digest(root/'saved-body/receipt.json')})
    monkeypatch.setattr(review,'construction_context',lambda _:({'objects':[],'scope':'authored analytical world'}, {'manifest_sha256':lab_pin}))
    return root


def test_ready_child_without_source_file_reviews_exact_original_evidence(job):
    before={p.relative_to(job).as_posix():p.read_bytes() for p in job.rglob('*') if p.is_file()}
    source,binding,world=review.review_input(job)
    assert source==job/'saved-body/world'
    assert set(binding['input_sha256'])=={'source.mp4','human.json','evidence.json'}
    assert binding['review_scope']=='conditional_body_authored_ground'
    assert binding['saved_body_attachment']['inventory']['original_exercise_intent']['inventory']=='unknown'
    assert len(binding['review_questions'])==4 and len(binding['review_questions'][-1])<=500
    assert 'omission is not absence' in binding['review_questions'][-1]
    assert 'DO_NOT_SEND_POLICY' not in json.dumps(binding)
    assert all((job/n).read_bytes()==raw for n,raw in before.items())


@pytest.mark.parametrize('path',['source.mp4','human.json','evidence.json','inventory.json','inputs/body-source-manifest.json'])
def test_changed_sealed_source_refuses(job,path):
    (job/'saved-body/world'/path).write_bytes(b'{}')
    with pytest.raises(ValueError):review.review_input(job)


@pytest.mark.parametrize('change',['receipt','request','lab_scene','missing_member','original_body','inventory','source_identity','wrong_world','not_ready','no_world'])
def test_coordinated_custody_or_scope_changes_refuse(job,monkeypatch,change):
    def edit(path,fn):
        obj=json.loads(path.read_bytes());fn(obj);write_json(path,obj)
    if change in ('receipt','request'):
        edit(job/f'saved-body/{change}.json',lambda x:x.update(parent_id='0'*32))
    elif change=='lab_scene':
        write_json(job/'lab/run/scene.json',{'different':True});pin=seal(job/'lab/run');write_json(job/'lab/lab-config.json',{'manifest_sha256':pin})
    elif change=='missing_member':
        edit(job/'saved-body/world/manifest.json',lambda x:x['files'].pop('inventory.json'))
    elif change=='original_body':
        edit(job/'saved-body/world/inputs/body-source-manifest.json',lambda x:x['files'].update({'human.json':'1'*64}))
    elif change=='inventory':
        edit(job/'saved-body/world/inventory.json',lambda x:x.update(absence_verified=True))
    elif change=='source_identity':
        edit(job/'saved-body/world/human.json',lambda x:x.update(source_sha256='0'*64))
    elif change=='wrong_world':monkeypatch.setattr(review,'construction_context',lambda _:({'objects':[{'id':'equipment'}]},{}))
    elif change=='not_ready':edit(job/'status.json',lambda x:x.update(status='failed'))
    elif change=='no_world':monkeypatch.setattr(review,'construction_context',lambda _:(None,None))
    if change in ('missing_member','original_body','inventory','source_identity'):
        if change!='missing_member':seal(job/'saved-body/world')
        edit(job/'saved-body/receipt.json',lambda x:x.update(world_manifest_sha256=digest(job/'saved-body/world/manifest.json')))
        edit(job/'status.json',lambda x:x.update(receipt_sha256=digest(job/'saved-body/receipt.json')))
    with pytest.raises(ValueError):review.review_input(job)


def test_review_questions_retain_equipment_declaration(job):
    intent=exercise_intent('equipment_or_obstacles','simulation');source=json.loads((job/'job.json').read_bytes())['source_sha256']
    inventory=make_conditional_person_inventory(source,intent)
    request=json.loads((job/'saved-body/request.json').read_bytes());request['original_exercise_intent']=intent;write_json(job/'saved-body/request.json',request)
    config=json.loads((job/'job.json').read_bytes());config['saved_body_request_sha256']=digest(job/'saved-body/request.json');write_json(job/'job.json',config)
    write_json(job/'saved-body/world/inventory.json',inventory);world_pin=seal(job/'saved-body/world')
    receipt=json.loads((job/'saved-body/receipt.json').read_bytes());receipt.update(inventory=inventory,world_manifest_sha256=world_pin);write_json(job/'saved-body/receipt.json',receipt)
    state=json.loads((job/'status.json').read_bytes());state['receipt_sha256']=digest(job/'saved-body/receipt.json');write_json(job/'status.json',state)
    _,binding,_=review.review_input(job)
    assert 'equipment_or_obstacles' in binding['review_questions'][-1]
    assert binding['saved_body_attachment']['inventory']['absence_verified'] is False
