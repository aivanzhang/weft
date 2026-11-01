"""Analytical offline configuration pages; no source or simulator execution."""

import builtins
from copy import deepcopy
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import socket
import subprocess
import time

import pytest

from weft import shared_exercise_config_report as report


def raw(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


@pytest.fixture(autouse=True)
def no_external_work(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'mujoco', 'gymnasium', 'cv2', 'torch', 'mediapipe', 'requests', 'httpx'}, name
        return original(name, *args, **kwargs)
    def denied(*args, **kwargs):
        pytest.fail('Offline report attempted network access')
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


class Page(HTMLParser):
    def __init__(self, value):
        super().__init__(); self.links, self.scripts, self.current = [], [], None
        self.feed(value.decode())
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ('href', 'src'):
            if key in attrs: self.links.append(attrs[key])
        if tag == 'script': self.current = []; self.scripts.append(self.current)
    def handle_data(self, value):
        if self.current is not None: self.current.append(value)
    def handle_endtag(self, tag):
        if tag == 'script': self.current = None
    @property
    def data(self):
        return json.loads(''.join(self.scripts[0]))


_DOM = r'''
import fs from'node:fs';import vm from'node:vm';import assert from'node:assert/strict';import{performance}from'node:perf_hooks';
const html=fs.readFileSync(process.argv[2],'utf8'),scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];let created=0,writes=0,downloads=[],revoked=[],blobs=[];const events=new Map();
class E{constructor(tag=''){this.tag=tag;this.children=[];this.listeners=new Map();this.value='';this.checked=false;this.disabled=false;this.hidden=false;this.own='';created++;}set textContent(v){this.own=String(v);this.children=[];writes++;}get textContent(){return this.own+this.children.map(n=>n.textContent).join('');}set innerHTML(v){assert.fail('unsafe HTML');}append(...v){this.children.push(...v);}replaceChildren(...v){this.children=[...v];this.own='';writes++;}addEventListener(k,f){this.listeners.set(k,f);}fire(k){this.listeners.get(k)?.();}click(){downloads.push({name:this.download,url:this.href});}}
const ids=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(v=>[v[1],new E()])),el=id=>{assert(ids.has(id),id);return ids.get(id)};el('data').textContent=scripts[0][1];
vm.runInNewContext(scripts[1][1],{document:{getElementById:el,createElement:t=>new E(t)},window:{addEventListener:(k,f)=>events.set(k,f)},URL:{createObjectURL:b=>{blobs.push(b.value);return'blob:'+blobs.length;},revokeObjectURL:u=>revoked.push(u)},Blob:class{constructor(v){this.value=v.join('');}},fetch:()=>assert.fail('network'),setTimeout:()=>assert.fail('timer')},{timeout:1000});
'''


def run_dom(tmp_path, page, code):
    filename = tmp_path/'page.html'; filename.write_bytes(page)
    result = subprocess.run(['node', '--input-type=module', '-', str(filename)], input=_DOM+code,
                            text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout) if result.stdout else None


def package(*, status='needs_input', frames=5, decoded=4, unavailable=False, derivative=False, person='person-1'):
    source = 'a'*64
    if status=='config_ready': person='person-0'
    references = [{'index':i, 'source_frame':frame, 'evidence_id':f'pose-{frame}', 'time_s':frame/30}
                  for i,frame in enumerate(range(0,decoded,2))]
    by_frame = {r['source_frame']:r['index'] for r in references}
    def custody(role, pin):
        return {'directory':'sources/'+role, 'manifest_sha256':pin, 'profile':'authored-source-v1',
                'status':'construction_only', 'files':{'evidence.json':'e'*64}}
    inputs = {'human':custody('human','1'*64), 'equipment':custody('equipment','2'*64), 'source_mappings':[]}
    if derivative:
        inputs['source_mappings'] = [{'directory':'sources/mappings/'+'3'*64, 'manifest_sha256':'3'*64,
            'profile':'authored-source-equivalence-v1', 'status':'completed', 'files':{'mapping.json':'4'*64},
            'original_sha256':source, 'derivative_sha256':'f'*64, 'frame_count':decoded, 'time_base':[1,30],
            'verification':'full_saved_bundle_verified_no_decode'}]
    sources = []
    for role, pin, ids in [('human','b'*64,[person]), ('equipment','c'*64,['box-1','box-2'])]:
        aligned = not (role=='equipment' and unavailable)
        slots = []
        for frame in range(decoded):
            for object_id in ids:
                body = aligned and frame in by_frame
                slots.append({'source_frame':frame, 'original_frame':frame if aligned else None, 'time_s':frame/30,
                    'object_id':object_id, 'observation_id':f'{object_id}-{frame}', 'partition':'construction',
                    'visibility':'detected', 'admission':'included', 'reason':None if body else 'missing_body_reference' if aligned else 'unavailable_source_mapping',
                    'evidence_ids':[f'{object_id}-evidence-{frame}'], 'bbox_available':True,
                    'body_reference_available':body, 'selection_eligible':body})
        sources.append({'role':role, 'document_sha256':pin, 'source_sha256':'f'*64 if role=='equipment' and (unavailable or derivative) else source,
            'image':{'width':640,'height':480}, 'time':{'fps':30,'time_base':[1,30]}, 'decoded_frames':decoded, 'metadata_frames':frames,
            'alignment':'unavailable' if not aligned else 'verified_derivative' if role=='equipment' and derivative else 'original',
            'entities':[{'object_id':name, 'label':name, 'kind':None} for name in ids], 'slots':slots})
    counts = {'original_source_slots':frames,'decoded_source_slots':decoded,'body_references':len(references),
              'human_entity_slots':decoded,'equipment_entity_slots':decoded*2,'metadata_tail_slots':frames-decoded}
    snapshot = {'schema_version':1,'profile':'shared-exercise-config-snapshot-v1','source_sha256':source,'inputs':inputs,
        'body':{'document_sha256':'d'*64,'profile':'exercise-v3','reference_count':len(references),'references':references,
                'validation':'deferred_to_existing_builder'},
        'source_slots':[{'source_frame':i,'source_status':'decoded' if i<decoded else 'metadata_only_not_decoded',
                         'nominal_time_s':i/30,'body_reference_indices':[by_frame[i]] if i in by_frame else []} for i in range(frames)],
        'observation_sources':sources,'issues':[],'counts':counts,'world_admission':'not_evaluated',
        'contact':'unknown','grasp':'unknown','load':'unknown'}
    request = {'schema_version':1,'profile':'shared-exercise-config-request-v1','snapshot_sha256':sha256(raw(snapshot)).hexdigest(),
        'human_entity_ref':None,'camera':None,'geometry_prior':None,'equipment':[],'target_object_id':None,
        'build_options':{'body_scale':1.0,'root_assistance':False}}
    config = None
    if status=='config_ready':
        request.update(human_entity_ref={'document_sha256':'b'*64,'object_id':person},
            camera={'mode':'fixed_pelvis_perspective_prior','distance_model_units':2.5},geometry_prior='floor_box_depth_equals_width',
            equipment=[{'object_id':'box-1','source_frame':0}],target_object_id='box-1',
            build_options={'body_scale':1.25,'root_assistance':True},frame_translation_xy=[1,0])
        config = {k:request[k] for k in ('camera','geometry_prior','equipment','target_object_id','frame_translation_xy')}
        config.update(schema_version=1,profile='source-equipment-proposal-config-v1',source_sha256=source)
    files = {'snapshot.json':raw(snapshot),'request.json':raw(request)}
    if config is not None: files['config.json'] = raw(config)
    metadata = {'schema_version':1,'profile':'shared-exercise-config-metadata-v1','status':status,
        'snapshot_sha256':sha256(files['snapshot.json']).hexdigest(),'request_sha256':sha256(files['request.json']).hexdigest(),
        'config_sha256':None if config is None else sha256(files['config.json']).hexdigest(),
        'missing_fields':[] if config else ['human_entity_ref','camera','geometry_prior','equipment','target_object_id'],
        'issues':[],'counts':counts,'scope':'Authored analytical snapshot; not independent source authentication',
        'world_admission':'not_evaluated','contact':'unknown','grasp':'unknown','load':'unknown','build_options':request['build_options']}
    if unavailable: metadata['issues']=[{'code':'source_mapping','message':'Equipment source mapping is unavailable.'}]
    files['metadata.json']=raw(metadata)
    return files


def reseal(files, *, snapshot=False):
    metadata = json.loads(files['metadata.json'])
    if snapshot:
        request = json.loads(files['request.json']); request['snapshot_sha256']=sha256(files['snapshot.json']).hexdigest()
        files['request.json']=raw(request)
    metadata.update(snapshot_sha256=sha256(files['snapshot.json']).hexdigest(),request_sha256=sha256(files['request.json']).hexdigest(),
                    config_sha256=sha256(files['config.json']).hexdigest() if 'config.json' in files else None)
    files['metadata.json']=raw(metadata)


def check_links(output, inputs):
    for name, value in output.items():
        if name.endswith('.html'):
            for target in Page(value).links:
                assert target[3:] in inputs if target.startswith('../') else target in output


def test_discovery_preserves_all_slots_raw_downloads_and_no_source_defaults():
    files=package(); before=deepcopy(files); output=report.build_shared_exercise_config_report(files)
    assert files==before and set(output)=={'index.html','frames-001.html','report.json','brand.css'}
    assert output['brand.css']==(Path(report.__file__).parent/'assets/brand.css').read_bytes()
    summary=json.loads(output['report.json']); data=Page(output['index.html']).data
    assert data['request']==json.loads(files['request.json']) and data['request']['equipment']==[]
    assert data['request']['human_entity_ref'] is None and data['request']['camera'] is None and data['request']['target_object_id'] is None
    assert summary['input_files']=={k:sha256(v).hexdigest() for k,v in files.items()}
    assert summary['source_choice_defaults']=='none' and summary['world_admission']=='not_evaluated'
    rows=Page(output['frames-001.html']).data['rows']
    assert [r['frame'] for r in rows]==list(range(5)) and rows[4]['original']['source_status']=='metadata_only_not_decoded'
    assert rows[4]['views']==[None,None] and rows[1]['original']['body_references']==[]
    assert b'PRINTED_MANIFEST_SHA256' in output['index.html'] and b'--resume PATH_TO_CONFIG_PACKAGE' in output['index.html']
    assert b'shared-exercise-build' not in output['index.html']; check_links(output,files)


def test_ready_exact_projection_keeps_authored_options_and_builder_boundary():
    files=package(status='config_ready',derivative=True); output=report.build_shared_exercise_config_report(files)
    page=output['index.html']; data=Page(page).data
    assert b'Configuration ready; world validation still required' in page
    assert b'--full-reference' in page and b'--body-scale 1.25' in page and b'--root-assistance' in page
    assert b'--source-mapping PATH_TO_CONFIG_PACKAGE/sources/mappings/' in page
    assert data['request']['frame_translation_xy']==[1,0] and data['request']['build_options']['root_assistance'] is True
    check_links(output,files)


def test_unaligned_source_keeps_own_complete_frames_and_explicit_unknowns():
    files=package(status='unavailable',unavailable=True); output=report.build_shared_exercise_config_report(files)
    data=Page(output['frames-001.html']).data
    assert data['views'][1]['alignment']=='unavailable'
    slots=[slot for r in data['rows'] for slot in (r['views'][1] or [])]
    assert len(slots)==8 and all(s['original_frame'] is None and not s['selection_eligible'] for s in slots)
    assert [s['time_s'] for s in slots]==[i/30 for i in range(4) for _ in range(2)]
    assert b'shared-exercise-build' not in output['index.html']


def test_invalid_authored_options_preserve_named_failure_and_exact_request_link():
    files=package(); request=json.loads(files['request.json']); request['build_options']['body_scale']=True; files['request.json']=raw(request)
    meta=json.loads(files['metadata.json']);meta['build_options']=None;meta['issues']=[{'code':'invalid_build_options','message':'Body size must be 0.5–2.'}]
    files['metadata.json']=raw(meta);reseal(files)
    output=report.build_shared_exercise_config_report(files)
    assert not json.loads(output['report.json'])['request_editor_available']
    assert not Page(output['index.html']).scripts and b'../request.json' in output['index.html']
    assert b'Body size must be' in output['index.html'] and b'No replacement' not in output['index.html']


@pytest.mark.parametrize('change,match', [('missing_slot','complete observation'),('duplicate_slot','duplicate frame/entity'),
    ('wrong_reference','reference join'),('false_admission','cannot claim'),('changed_config','config changes'),
    ('unavailable_ready','ready equipment'),('foreign_request','snapshot identity'),('options_mismatch','build options differ')])
def test_consistently_resealed_contradictions_fail_closed(change,match):
    files=package(status='config_ready');snapshot=json.loads(files['snapshot.json']);metadata=json.loads(files['metadata.json'])
    if change=='missing_slot':snapshot['observation_sources'][1]['slots'].pop()
    elif change=='duplicate_slot':snapshot['observation_sources'][1]['slots'][1]=deepcopy(snapshot['observation_sources'][1]['slots'][0])
    elif change=='wrong_reference':snapshot['source_slots'][0]['body_reference_indices']=[]
    elif change=='false_admission':metadata['world_admission']='admitted'
    elif change=='changed_config':
        config=json.loads(files['config.json']);config['camera']['distance_model_units']=3;files['config.json']=raw(config)
    elif change=='unavailable_ready':snapshot['observation_sources'][1]['slots'][0]['selection_eligible']=False
    elif change=='options_mismatch':metadata['build_options']['root_assistance']=False
    files['snapshot.json']=raw(snapshot);files['metadata.json']=raw(metadata);reseal(files,snapshot=True)
    if change=='foreign_request':
        request=json.loads(files['request.json']);request['snapshot_sha256']='0'*64;files['request.json']=raw(request);reseal(files)
    with pytest.raises(ValueError,match=match):report.build_shared_exercise_config_report(files)


def test_strict_json_and_resource_bounds_fail_before_report_work(monkeypatch):
    files=package()
    with pytest.raises(ValueError,match='fixed saved-input'):report.build_shared_exercise_config_report(files|{'../extra':b'{}'})
    with pytest.raises(ValueError,match='snapshot.json exceeds'):report.build_shared_exercise_config_report(files|{'snapshot.json':b' '*(8*1024*1024+1)})
    with pytest.raises(ValueError,match='finite strict JSON'):report.build_shared_exercise_config_report(files|{'metadata.json':b'{"x":NaN}'})
    with pytest.raises(ValueError,match='finite strict JSON'):report.build_shared_exercise_config_report(files|{'metadata.json':b'{"x":1,"x":2}'})
    with pytest.raises(ValueError,match='five seconds'):report.build_shared_exercise_config_report(files,max_seconds=True)
    clock=iter((0.,6.));monkeypatch.setattr(report.time,'monotonic',lambda:next(clock))
    with pytest.raises(TimeoutError):report.build_shared_exercise_config_report(files)


def test_scripted_editor_downloads_exact_choices_and_never_defaults_source_fields(tmp_path):
    files=package(person='person-0');output=report.build_shared_exercise_config_report(files)
    measured=run_dom(tmp_path,output['index.html'],r'''
const D=JSON.parse(scripts[0][1]);assert.equal(el('human').value,'');assert.equal(el('object').value,'');assert.equal(el('source-frame').value,'');assert.equal(el('target').value,'');assert.equal(el('camera-mode').value,'');assert.equal(el('distance').value,'');assert.equal(el('geometry').value,'');assert.equal(el('body-scale').value,'1');assert.equal(el('assistance').value,'false');
el('download-request').fire('click');assert.deepEqual(JSON.parse(blobs[0]),D.request);assert.equal(downloads[0].name,'shared-exercise-request.json');
el('object').value='0';el('source-frame').value='1';el('source-frame').fire('input');assert.equal(el('add-equipment').disabled,true);el('add-equipment').fire('click');assert.equal(el('selected-equipment').children.length,0);
el('source-frame').value='0';el('source-frame').fire('input');assert.equal(el('add-equipment').disabled,false);el('add-equipment').fire('click');assert.equal(el('target').value,'');assert.equal(el('object').value,'');assert.equal(el('source-frame').value,'');
for(const[id,value]of[['human','0'],['camera-mode','fixed_pelvis_perspective_prior'],['distance','2.5'],['geometry','floor_box_depth_equals_width'],['target','box-1'],['body-scale','1.5'],['assistance','true']]){el(id).value=value;el(id).fire('change');}
el('translation').checked=true;el('shift-x').value='1';el('shift-y').value='0';el('translation').fire('change');el('download-request').fire('click');
const saved=JSON.parse(blobs[1]);assert.equal(saved.snapshot_sha256,D.request.snapshot_sha256);assert.deepEqual(saved.human_entity_ref,{document_sha256:D.humans[0].document_sha256,object_id:'person-0'});assert.deepEqual(saved.equipment,[{object_id:'box-1',source_frame:0}]);assert.equal(saved.target_object_id,'box-1');assert.deepEqual(saved.camera,{mode:'fixed_pelvis_perspective_prior',distance_model_units:2.5});assert.deepEqual(saved.build_options,{body_scale:1.5,root_assistance:true});assert.deepEqual(saved.frame_translation_xy,[1,0]);assert.equal(revoked.length,1);
const baseline=writes,nodes=created;for(let i=0;i<1000;i++)el('distance').fire('input');assert.equal(writes,baseline);assert.equal(created,nodes);
el('distance').value='Infinity';el('distance').fire('input');assert.equal(el('download-request').disabled,true);el('download-request').fire('click');assert.equal(downloads.length,2);events.get('pagehide')();assert.equal(revoked.length,2);
process.stdout.write(JSON.stringify({unchanged_updates:1000,text_writes:writes-baseline-2,downloaded_requests:downloads.length}));
''')
    assert measured['downloaded_requests']==2


def test_safe_markup_and_small_frame_navigation_noops(tmp_path):
    files=package();snapshot=json.loads(files['snapshot.json']);snapshot['observation_sources'][1]['entities'][0]['label']='</script><img src=x onerror=alert(1)>'
    files['snapshot.json']=raw(snapshot);reseal(files,snapshot=True);output=report.build_shared_exercise_config_report(files)
    assert b'</script><img' not in output['index.html'] and b'\\u003c/script>' in output['index.html']
    measured=run_dom(tmp_path,output['frames-001.html'],r'''
const D=JSON.parse(scripts[0][1]);assert.equal(el('frame').value,'');assert.equal(el('view').value,'');
el('frame').value='0';el('view').value='1';el('view').fire('change');el('entity').value='0';el('entity').fire('change');assert.deepEqual(JSON.parse(el('source-record').textContent),D.rows[0].views[1][0]);
const baseline=writes,nodes=created;for(let i=0;i<1000;i++)el('frame').fire('change');assert.equal(writes,baseline);assert.equal(created,nodes);
el('frame').value='4';el('frame').fire('change');assert.match(el('availability').textContent,/Outside this observation/);el('view').value='original';el('view').fire('change');assert.match(el('clock').textContent,/metadata_only_not_decoded/);el('frame').value='999';el('frame').fire('change');assert.equal(el('frame-title').textContent,'Choose a frame and view');process.stdout.write(JSON.stringify({unchanged_updates:1000,unchanged_writes:0}));
''')
    assert measured['unchanged_writes']==0


def test_representative_maximum_frames_preserve_all_slots_and_bounded_navigation(tmp_path,record_property):
    files=package(frames=1800,decoded=1800);start=time.perf_counter();output=report.build_shared_exercise_config_report(files);elapsed=time.perf_counter()-start
    summary=json.loads(output['report.json']);assert len(summary['pages'])==18 and len(output)==21
    assert sum(map(len,output.values()))<=16<<20 and max(len(v) for k,v in output.items() if k.endswith('.html'))<=2<<20
    rows=[row for p in summary['pages'] for row in Page(output[p['path']]).data['rows']]
    assert [r['frame'] for r in rows]==list(range(1800))
    assert sum(len(view) for r in rows for view in r['views'] if view is not None)==5400
    measured=run_dom(tmp_path,output['frames-018.html'],r'''
const D=JSON.parse(scripts[0][1]);let slots=0;for(let fi=0;fi<D.rows.length;fi++){el('frame').value=String(fi);for(let vi=0;vi<D.views.length;vi++){el('view').value=String(vi);el('view').fire('change');for(let ei=0;ei<D.views[vi].entities.length;ei++){el('entity').value=String(ei);el('entity').fire('change');assert.deepEqual(JSON.parse(el('source-record').textContent),D.rows[fi].views[vi][ei]);slots++;}}}
const baseline=writes;for(let i=0;i<1000;i++)el('entity').fire('change');assert.equal(writes,baseline);const nodes=created,start=performance.now();for(let i=0;i<1000;i++){el('frame').value=String(i%D.rows.length);el('frame').fire('change');}assert.equal(created,nodes);process.stdout.write(JSON.stringify({selected_slots:slots,changed_updates:1000,changed_ms:performance.now()-start,new_nodes:created-nodes,unchanged_updates:1000,unchanged_writes:0}));
''')
    assert measured['selected_slots']==300 and measured['new_nodes']==0
    record_property('input_bytes',sum(map(len,files.values())));record_property('output_bytes',sum(map(len,output.values())))
    record_property('preparation_seconds',elapsed);record_property('scripted_navigation',json.dumps(measured,sort_keys=True))


def test_person_zero_boundary_preserves_other_identities_without_selecting_or_renaming(tmp_path):
    files=package();request=json.loads(files['request.json']);request['human_entity_ref']={'document_sha256':'b'*64,'object_id':'person-1'}
    files['request.json']=raw(request);reseal(files);before=deepcopy(files)
    output=report.build_shared_exercise_config_report(files)
    assert files==before and Page(output['index.html']).data['request']['human_entity_ref']==request['human_entity_ref']
    assert Page(output['frames-001.html']).data['views'][0]['entities'][0]['object_id']=='person-1'
    run_dom(tmp_path,output['index.html'],r"""
assert.equal(el('human').value,'0');assert.equal(el('human').children[1].disabled,true);assert.equal(el('download-request').disabled,true);assert.match(el('request-state').textContent,/requires saved body ID person-0/);assert.equal(JSON.parse(el('request-preview').textContent).human_entity_ref.object_id,'person-1');el('download-request').fire('click');assert.equal(downloads.length,0);
""")
    files=package(status='config_ready');request=json.loads(files['request.json']);request['human_entity_ref']['object_id']='person-1'
    snapshot=json.loads(files['snapshot.json']);snapshot['observation_sources'][0]['entities'][0]['object_id']='person-1'
    for row in snapshot['observation_sources'][0]['slots']:row['object_id']='person-1'
    files['snapshot.json']=raw(snapshot);files['request.json']=raw(request);reseal(files,snapshot=True)
    with pytest.raises(ValueError,match='person-0 builder'):report.build_shared_exercise_config_report(files)


def test_explicit_null_translation_is_disabled_and_preserved_in_ready_requests(tmp_path):
    files=package(status='config_ready')
    for name in ('request.json','config.json'):
        document=json.loads(files[name]);document['frame_translation_xy']=None;files[name]=raw(document)
    reseal(files);before=deepcopy(files)
    output=report.build_shared_exercise_config_report(files)
    assert files==before and json.loads(output['report.json'])['status']=='config_ready'
    assert Page(output['index.html']).data['request']['frame_translation_xy'] is None
    assert json.loads(output['report.json'])['input_files']['config.json']==sha256(files['config.json']).hexdigest()
    run_dom(tmp_path,output['index.html'],r'''
const D=JSON.parse(scripts[0][1]);assert.equal(el('translation').checked,false);assert.equal(el('shift-x').value,'');assert.equal(el('shift-y').value,'');assert.equal(el('shift-x').disabled,true);assert.equal(el('shift-y').disabled,true);assert.equal(el('download-request').disabled,false);
el('download-request').fire('click');assert.deepEqual(JSON.parse(blobs[0]),D.request);assert.equal(JSON.parse(el('request-preview').textContent).frame_translation_xy,null);
el('translation').checked=true;el('shift-x').value='2';el('shift-y').value='0';el('translation').fire('change');el('download-request').fire('click');assert.deepEqual(JSON.parse(blobs[1]).frame_translation_xy,[2,0]);
el('translation').checked=false;el('translation').fire('change');el('download-request').fire('click');assert.equal(JSON.parse(blobs[2]).frame_translation_xy,null);assert.equal(el('shift-x').disabled,true);
''')
