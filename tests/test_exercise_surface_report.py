"""Saved analytical collection pages; no browser, media or native execution."""

import builtins
from copy import deepcopy
from hashlib import sha256
from html.parser import HTMLParser
import importlib
import json
from pathlib import Path
import socket
import subprocess

import pytest

from weft import exercise_surface_report as report


def raw(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def pin(value):
    return sha256(value).hexdigest()


@pytest.fixture(autouse=True)
def no_external_work(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'mujoco', 'gymnasium', 'cv2', 'torch', 'mediapipe', 'requests', 'httpx'}, name
        return original(name, *args, **kwargs)
    def denied(*args, **kwargs):
        pytest.fail('Report attempted network work')
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def package(mode='discovery', *, body=True):
    api = importlib.import_module('weft.exercise_surface_inputs')
    fixture = importlib.import_module('test_exercise_surface_inputs')
    prepared, request, kwargs = fixture.fixture(api, with_body=body)
    meta = prepared.describe(); bindings = meta['input_bindings']
    original = {f'surface-observation-{i:02d}.json': value for i, (_, value) in enumerate(sorted(kwargs['observation_documents'].items()))}
    if kwargs['camera_document'] is not None: original['surface-camera.json'] = kwargs['camera_document']
    if kwargs['body_document'] is not None: original['original-human.json'] = kwargs['body_document']
    observation = json.loads(original['surface-observation-00.json']); entity = request['annotations']['channels'][0]['entity_ref']
    inventory = {'schema_version': 1, 'profile': 'exercise-surface-inventory-v1', 'input_bindings': bindings,
        'input_files': {name: pin(value) for name, value in original.items()}, 'source': observation['source'],
        'source_clocks': meta['source_clocks'], 'entities': [{'entity_ref': entity, 'properties': {}, 'source_slots': 4,
            'construction_annotation_slots': 4, 'partition_counts': {'construction': 4}, 'visibility_counts': {'detected': 4}}],
        'scope': 'Analytical inventory wrapper over actual collector bytes; not source authentication'}
    selection = {'schema_version': 1, 'profile': 'exercise-surface-selection-v1', 'input_bindings': bindings,
        'channels': [] if mode == 'discovery' else request['annotations']['channels'], 'human_entity_ref': None}
    files = dict(original)
    if mode != 'discovery':
        if mode == 'unknown': request['surface_intervals']['surfaces'][0].update(height_interval_m=None, quality='unknown')
        qualified = prepared.qualify(prepared.request_document() if mode == 'draft' else raw(request))
        files.update(qualified.files()); meta = qualified.describe()
    state = {'schema_version': 1, 'profile': 'exercise-surface-workflow-v1', 'status': 'completed',
        'collection_status': 'needs_selection' if mode == 'discovery' else meta['status'],
        'operation': 'discovery' if mode == 'discovery' else 'selection' if mode == 'draft' else 'qualification',
        'input_bindings': bindings, 'selection_sha256': pin(raw(selection)),
        'request_sha256': None if mode == 'discovery' else pin(files['collection-request.json']),
        'collection_manifest_sha256': None if mode == 'discovery' else pin(files['collection-manifest.json']),
        'parent_manifest_sha256': None, 'fit_or_simulator_admission': False, 'body_or_floor_installed': False,
        'physical_validation': 'unverified', 'contact': 'unknown', 'load': 'unknown', 'grasp': 'unknown', 'next_action': 'Inspect explicit saved inputs'}
    files.update({'inventory.json': raw(inventory), 'selection.json': raw(selection), 'workflow-state.json': raw(state)})
    return files


def reseal(files):
    state = json.loads(files['workflow-state.json'])
    state['selection_sha256'] = pin(files['selection.json'])
    if 'collection-manifest.json' in files:
        seal = json.loads(files['collection-manifest.json'])
        seal['files'] = {name: pin(value) for name, value in files.items() if name not in {
            'inventory.json', 'selection.json', 'workflow-state.json', 'collection-manifest.json'}}
        files['collection-manifest.json'] = raw(seal)
        state['collection_manifest_sha256'] = pin(files['collection-manifest.json'])
    files['workflow-state.json'] = raw(state)


class Page(HTMLParser):
    def __init__(self, data):
        super().__init__(); self.links = []; self.scripts = []; self.inside = None
        self.feed(data.decode())
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'script': self.inside = []; self.scripts.append(self.inside)
        for key in ('href', 'src'):
            if key in attrs: self.links.append(attrs[key])
    def handle_data(self, value):
        if self.inside is not None: self.inside.append(value)
    def handle_endtag(self, tag):
        if tag == 'script': self.inside = None
    @property
    def data(self):
        return json.loads(''.join(self.scripts[0]))


def test_discovery_has_no_default_identity_and_all_links_resolve():
    files = package(); before = deepcopy(files); result = report.build_exercise_surface_report(files)
    assert files == before and set(result) == {'index.html', 'brand.css', 'report.json'}
    assert result['brand.css'] == (Path(report.__file__).parent/'assets/brand.css').read_bytes()
    page = Page(result['index.html']); data = page.data
    assert data['input_bindings'] == json.loads(files['selection.json'])['input_bindings']
    assert data['has_body'] is True
    assert b'--selection PATH_TO_DOWNLOADED_SELECTION' in result['index.html']
    assert b'body-reference-propose' not in result['index.html']
    assert all(name in result if not name.startswith('../') else name[3:] in files for name in page.links)
    assert json.loads(result['report.json'])['selection_defaults'] == 'none'


@pytest.mark.parametrize('mode,body,expected', [('draft', True, 'incomplete_request'), ('qualified', True, 'conditional_inputs_available'),
    ('unknown', True, 'qualified_but_unavailable'), ('qualified', False, 'conditional_inputs_available')])
def test_actual_collector_rows_preserve_unknowns_templates_units_and_next_steps(mode, body, expected):
    files = package(mode, body=body); result = report.build_exercise_surface_report(files)
    summary = json.loads(result['report.json']); assert summary['collection_status'] == expected
    assert summary['frame_count'] == 4 and summary['channel_count'] == 1 and summary['fit_or_simulator_admission'] is False
    page = Page(result['frames-001.html']); rows = page.data['rows']
    original = [json.loads(line) for line in files['collection-frames.jsonl'].splitlines()]
    assert [row['frame'] for row in rows] == list(range(4))
    assert rows[0]['channels'][0]['template'] == original[0]['source']['channels'][0]['annotation_template']
    assert rows[0]['channels'][0]['template']['pixel'] is None
    assert rows[1]['body_reference'] is None
    if mode == 'unknown':
        assert rows[0]['channels'][0]['envelope']['reason'] == 'unknown_surface_height_interval'
        assert page.data['surfaces'][0]['height_interval_m'] is None
    for name, value in result.items():
        if name.endswith('.html'):
            assert all(link in result if not link.startswith('../') else link[3:] in files for link in Page(value).links)
    text = result['index.html'].decode()
    assert 'PATH_TO_EDITED_COLLECTION_REQUEST' in text and '../collection-request.json' in text
    if mode == 'draft': assert 'Download surface-config.json' not in text
    else: assert '--run PATH_TO_SEPARATELY_QUALIFIED_BODY' in text and 'fitter may still reject' in text
    if not body: assert 'No original body was supplied' in text


@pytest.mark.parametrize('change,match', [
    ('source_hash', 'input bytes'), ('missing_frame', 'complete original'), ('foreign_template', 'template changes'),
    ('false_admission', 'physical or fit admission'), ('frame_denominator', 'denominators'), ('missing_member', 'input bytes'),
])
def test_resealed_saved_contradictions_fail_closed(change, match):
    files = package('draft')
    if change == 'source_hash': files['surface-observation-00.json'] += b' '
    elif change in {'missing_frame', 'foreign_template'}:
        rows = [json.loads(line) for line in files['collection-frames.jsonl'].splitlines()]
        if change == 'missing_frame': rows.pop()
        else: rows[0]['source']['channels'][0]['annotation_template']['time_s'] = .5
        files['collection-frames.jsonl'] = b'\n'.join(map(raw, rows))+b'\n'
    elif change == 'false_admission':
        state = json.loads(files['workflow-state.json']); state['fit_or_simulator_admission'] = True; files['workflow-state.json'] = raw(state)
    elif change == 'frame_denominator':
        meta = json.loads(files['collection-report.json']); meta['source_counts']['metadata_frames'] += 1; files['collection-report.json'] = raw(meta)
    else: del files['surface-camera.json']
    if change != 'missing_member': reseal(files)
    with pytest.raises(ValueError, match=match): report.build_exercise_surface_report(files)


def test_unsafe_names_nonfinite_json_and_cooperative_deadline_fail(monkeypatch):
    files = package()
    with pytest.raises(ValueError, match='fixed input'): report.build_exercise_surface_report(files | {'../foreign.json': b'{}'})
    with pytest.raises(ValueError, match='finite strict JSON'): report.build_exercise_surface_report(files | {'inventory.json': b'{"v":NaN}'})
    with pytest.raises(ValueError, match='five seconds'): report.build_exercise_surface_report(files, max_seconds=True)
    clock = iter((0., 6.)); monkeypatch.setattr(report.time, 'monotonic', lambda: next(clock))
    with pytest.raises(TimeoutError): report.build_exercise_surface_report(files)


def large_package():
    files = package('draft', body=False)
    inventory, selection, meta, request = (json.loads(files[n]) for n in ('inventory.json', 'selection.json', 'collection-report.json', 'collection-request.json'))
    observation = json.loads(files['surface-observation-00.json']); observation['source'].update(decoded_frames=1800, metadata_frames=1800)
    files['surface-observation-00.json'] = raw(observation); document_sha = pin(files['surface-observation-00.json'])
    bindings = inventory['input_bindings']; old = bindings['observation_document_sha256s'][0]; bindings['observation_document_sha256s'] = [document_sha]
    inventory['input_files']['surface-observation-00.json'] = document_sha
    inventory['source'].update(decoded_frames=1800, metadata_frames=1800)
    inventory['source_clocks'] = {document_sha: inventory['source_clocks'][old]}
    entity = {'document_sha256': document_sha, 'object_id': inventory['entities'][0]['entity_ref']['object_id']}
    inventory['entities'][0].update(entity_ref=entity, source_slots=1800, construction_annotation_slots=0,
        partition_counts={'unsampled':1800},visibility_counts={'withheld':1800})
    channels = [{'id':f'channel-{i}', 'point_role':'Unmeasured image point', 'surface_id':'surface-0','entity_ref':entity} for i in range(12)]
    selection.update(input_bindings=bindings,channels=channels)
    request['input_bindings'] = bindings; request['annotations']['channels'] = channels; request['annotations']['annotations'] = []
    files['collection-request.json'] = raw(request)
    meta.update(input_bindings=bindings,request_sha256=pin(files['collection-request.json']),source_clocks=inventory['source_clocks'])
    meta['source_counts'].update(decoded_frames=1800,metadata_frames=1800,retained_source_frames=1800,channel_slots=21600,status_counts={'source_not_sampled':21600})
    rows = []
    for frame in range(1800):
        cells = [{'channel_id':c['id'],'entity_ref':entity,'surface_id':'surface-0','reason':'source_not_sampled','annotation_template':None,
                  'source_time_s':frame/30,'source_partition':'unsampled','source_visibility':None,'source_evidence_refs':[]} for c in channels]
        rows.append(raw({'source':{'source_frame':frame,'source_status':'decoded','original_body_reference':None,'channels':cells},
                         'envelope':None,'body_join':None,'image_admission':None}))
    files['collection-frames.jsonl'] = b'\n'.join(rows)+b'\n'
    files.update({'inventory.json':raw(inventory),'selection.json':raw(selection),'collection-report.json':raw(meta)})
    state = json.loads(files['workflow-state.json']); state.update(input_bindings=bindings,request_sha256=pin(files['collection-request.json']))
    files['workflow-state.json'] = raw(state); reseal(files)
    return files


_DOM = r'''
import fs from'node:fs';import vm from'node:vm';import assert from'node:assert/strict';import{performance}from'node:perf_hooks';
const html=fs.readFileSync(process.argv[2],'utf8'),scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];let created=0,downloads=[],revoked=[],blobs=[];const events=new Map();
class E{constructor(tag=''){this.tag=tag;this.children=[];this.listeners=new Map();this.value='';this.checked=false;this.disabled=false;this.own='';created++;}set textContent(v){this.own=String(v);this.children=[];}get textContent(){return this.own+this.children.map(n=>n.textContent).join('');}set innerHTML(v){assert.fail('unsafe HTML');}append(...v){this.children.push(...v);}replaceChildren(...v){this.children=[...v];this.own='';}addEventListener(k,f){this.listeners.set(k,f);}fire(k){this.listeners.get(k)?.();}click(){downloads.push({name:this.download,url:this.href});}}
const ids=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(v=>[v[1],new E()])),el=id=>{assert(ids.has(id),id);return ids.get(id)};el('data').textContent=scripts[0][1];
vm.runInNewContext(scripts[1][1],{document:{getElementById:el,createElement:t=>new E(t)},window:{addEventListener:(k,f)=>events.set(k,f)},URL:{createObjectURL:b=>{blobs.push(b.value);return'blob:'+blobs.length;},revokeObjectURL:u=>revoked.push(u)},Blob:class{constructor(v){this.value=v.join('');}},fetch:()=>assert.fail('network'),setTimeout:()=>assert.fail('timer')},{timeout:1000});
'''


def run_dom(tmp_path, page, code):
    filename = tmp_path/'page.html'; filename.write_bytes(page)
    result = subprocess.run(['node','--input-type=module','-',str(filename)],input=_DOM+code,text=True,capture_output=True,timeout=5)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout) if result.stdout else None


def test_scripted_selection_requires_explicit_choices_and_downloads_exact_schema(tmp_path):
    files = package(); output = report.build_exercise_surface_report(files)
    result = run_dom(tmp_path, output['index.html'], r'''
assert.equal(downloads.length,0);assert.equal(el('entity').value,'');assert.equal(el('source-ack').checked,false);assert.equal(el('download-selection').disabled,true);el('download-selection').fire('click');assert.equal(downloads.length,0);
el('entity').value='0';el('entity').fire('change');el('channel-id').value='chosen_point';el('point-role').value='Chosen ankle image estimate';el('surface-id').value='unnamed_support';el('add-channel').fire('click');assert.equal(el('download-selection').disabled,true);
el('source-ack').checked=true;el('source-ack').fire('change');el('download-selection').fire('click');const first=JSON.parse(blobs[0]);assert.equal(first.human_entity_ref,null);assert.equal(first.channels.length,1);assert.equal(first.channels[0].surface_id,'unnamed_support');assert.equal(first.profile,'exercise-surface-selection-v1');
el('set-human').fire('click');el('download-selection').fire('click');const second=JSON.parse(blobs[1]);assert.deepEqual(second.input_bindings,JSON.parse(scripts[0][1]).input_bindings);assert.deepEqual(second.human_entity_ref,second.channels[0].entity_ref);assert.equal(revoked.length,1);events.get('pagehide')();assert.equal(revoked.length,2);process.stdout.write(JSON.stringify({downloads:downloads.length,selection:first}));
''')
    assert result['downloads'] == 2
    assert set(result['selection']) == {'schema_version','profile','input_bindings','channels','human_entity_ref'}


def test_scripted_annotation_download_keeps_exact_clock_and_unknown_values(tmp_path):
    files = package('qualified'); before = deepcopy(files); output = report.build_exercise_surface_report(files)
    value = run_dom(tmp_path, output['frames-001.html'], r'''
assert.match(el('source-clock').textContent,/Source time: 0 s/);assert.match(el('body-clock').textContent,/Original body knot: 0 s/);assert.equal(el('download-template').disabled,false);el('download-template').fire('click');const annotation=JSON.parse(blobs[0]);assert.equal(annotation.time_s,0);assert.equal(annotation.pixel,null);assert.equal(annotation.uncertainty_px,null);assert.equal(annotation.status,'unknown');assert.match(el('supplied').textContent,/annotated/);el('frame').value='1';el('frame').fire('change');assert.match(el('body-clock').textContent,/No original body reference/);events.get('pagehide')();assert.equal(revoked.length,1);process.stdout.write(JSON.stringify(annotation));
''')
    original = json.loads(files['collection-frames.jsonl'].splitlines()[0])
    assert value == original['source']['channels'][0]['annotation_template'] and files == before


def test_report_output_caps_reject_instead_of_dropping_source_slots(monkeypatch):
    files = package('draft')
    with monkeypatch.context() as patch:
        patch.setattr(report, 'MAX_HTML', 1024)
        with pytest.raises(ValueError, match='complete report'): report.build_exercise_surface_report(files)
    with monkeypatch.context() as patch:
        patch.setattr(report, 'MAX_OUTPUT', 1024)
        with pytest.raises(ValueError, match='16 MiB'): report.build_exercise_surface_report(files)


def test_large_complete_pages_and_repeated_selected_frame_rendering(tmp_path, record_property):
    import time
    files = large_package(); started = time.perf_counter(); output = report.build_exercise_surface_report(files); elapsed = time.perf_counter()-started
    summary = json.loads(output['report.json']); assert summary['frame_count'] == 1800 and summary['channel_count'] == 12
    pages = [Page(output[p['file']]).data for p in summary['pages']]
    assert [row['frame'] for page in pages for row in page['rows']] == list(range(1800))
    assert sum(len(row['channels']) for page in pages for row in page['rows']) == 21600
    assert len(output) == 21 and max(len(v) for n,v in output.items() if n.endswith('.html')) <= 2<<20
    measurement = run_dom(tmp_path, output['frames-018.html'], r'''
assert.match(el('frame-title').textContent,/1700/);assert.equal(el('download-template').disabled,true);assert.match(el('body-clock').textContent,/No original body reference/);const before=created,start=performance.now();
for(let i=0;i<1000;i++){el('frame').value=String(i%100);el('channel').value=String(i%12);el('frame').fire('change');}const elapsed=(performance.now()-start)/1000;assert.equal(created,before);assert.match(el('frame-title').textContent,/1799/);assert.match(el('envelope').textContent,/Unavailable/);el('download-template').fire('click');assert.equal(downloads.length,0);process.stdout.write(JSON.stringify({updates:1000,new_nodes:created-before,seconds:elapsed,frame_options:el('frame').children.length,channel_options:el('channel').children.length}));
''')
    record_property('large_saved_projection', json.dumps({'frames':1800,'channels':12,'input_bytes':sum(map(len,files.values())),
        'output_bytes':sum(map(len,output.values())),'files':len(output),'prepare_seconds':elapsed,'scripted_dom':measurement,
        'scope':'Synthetic bounded saved rows and Node mock DOM; no browser performance or source qualification claim'},sort_keys=True))


def test_script_data_escapes_untrusted_text_and_preserves_unknowns():
    files = package(); inventory = json.loads(files['inventory.json'])
    inventory['entities'][0]['properties'] = {'note': {'value':'</script><img src=x onerror=alert(1)>','status':'authored','evidence_refs':[]}}
    files['inventory.json'] = raw(inventory)
    page = report.build_exercise_surface_report(files)['index.html']
    assert b'<img src=x' not in page and len(Page(page).scripts) == 2
    assert Page(page).data['entities'][0]['properties']['note']['value'].startswith('</script>')
