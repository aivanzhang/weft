"""Bounded saved-input worksheets; no geometry, media or source qualification."""

from hashlib import sha256
import html
import json
import math
from pathlib import Path
import re
import time


PROFILE = 'exercise-surface-report-v1'
MAX_INPUT, MAX_OUTPUT, MAX_HTML, MAX_FILES = 32 << 20, 16 << 20, 2 << 20, 32
MAX_FRAMES, MAX_CHANNELS, PAGE_FRAMES, PAGE_BYTES = 1800, 12, 100, 1536 << 10
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_INPUT = re.compile(r'(?:inventory|selection|workflow-state|collection-request|collection-report|collection-manifest|surface-config|surface-camera|surface-intervals|surface-annotations|surface-body-binding|original-human)\.json\Z|collection-frames\.jsonl\Z|surface-observation-0[0-7]\.json\Z')
_BINDINGS = {'source_sha256', 'observation_document_sha256s', 'camera_raw_sha256', 'body_raw_sha256'}
_STATUSES = {'needs_selection', 'incomplete_request', 'conditional_inputs_available', 'qualified_but_unavailable'}
_TITLES = {'needs_selection': 'Choose what to annotate', 'incomplete_request': 'Fill the missing inputs',
            'qualified_but_unavailable': 'Inputs saved; usable ranges are still unavailable',
            'conditional_inputs_available': 'Conditional input files are available'}
_COMMAND = '.venv-lab/bin/python -m weft '


def _require(value, message):
    if not value:
        raise ValueError('Exercise surface report: ' + message)


def _encoded(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    try:
        result = json.loads(raw, object_pairs_hook=unique)
        _encoded(result)
    except (ValueError, RecursionError) as error:
        raise ValueError('Exercise surface report: finite strict JSON required') from error
    _require(type(result) is dict, 'JSON object required')
    return result


def _text(value, bound=512):
    _require(type(value) is str and 0 < len(value) <= bound and all(ord(c) >= 32 for c in value), 'bounded text required')
    return value


def _count(value, bound=21600):
    _require(type(value) is int and 0 <= value <= bound, 'invalid bounded count')
    return value


def _number(value):
    _require(value is None or type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e9,
             'finite saved number or explicit unknown required')
    return value


def _ref(value, documents):
    _require(type(value) is dict and set(value) == {'document_sha256', 'object_id'}
             and value['document_sha256'] in documents, 'foreign entity identity')
    _text(value['object_id'], 120)
    return (value['document_sha256'], value['object_id'])


def _template(value, source, frame, channel, row):
    if value is None:
        return None
    _require(type(value) is dict and set(value) == {'id', 'channel_id', 'frame', 'time_s', 'observation_id', 'status', 'pixel', 'uncertainty_px'}
             and value['id'] == f"{source}:frame:{frame}:surface-anchor:{channel['id']}"
             and value['channel_id'] == channel['id'] and type(value['frame']) is int and value['frame'] == frame
             and value['time_s'] == row['source_time_s'] and value['observation_id'] == row['source_observation_id']
             and value['status'] == 'unknown' and value['pixel'] is None and value['uncertainty_px'] is None
             and row['reason'] is None and row['source_partition'] == 'construction'
             and row['source_visibility'] in {'detected', 'partial'}, 'annotation template changes source identity or invents a value')
    return value


def _project_row(value, index, channels, source, supplied):
    row = value.get('source')
    _require(type(row) is dict and type(row.get('source_frame')) is int and row['source_frame'] == index
             and row.get('source_status') in {'decoded', 'metadata_only_not_decoded'}
             and type(row.get('channels')) is list and len(row['channels']) == len(channels), 'complete original frame/channel slots required')
    body = row.get('original_body_reference')
    if body is not None:
        _require(type(body) is dict and set(body) == {'reference_index', 'evidence_id', 'body_time_s'}, 'invalid original body clock')
        _count(body['reference_index'], 900); _number(body['body_time_s']); _text(body['evidence_id'])
    envelope, joined = value.get('envelope'), value.get('body_join')
    def index_rows(document, name):
        if document is None:
            return {}
        rows = document.get(name)
        _require(document.get('source_frame') == index and type(rows) is list and len(rows) == len(channels), 'saved geometry slots differ')
        result = {r['channel_id']: r for r in rows}
        _require(set(result) == {c['id'] for c in channels} and len(result) == len(rows), 'saved geometry channel identities differ')
        return result
    anchors, body_rows = index_rows(envelope, 'anchors'), index_rows(joined, 'channels')
    projected = []
    for channel, cell in zip(channels, row['channels']):
        _require(type(cell) is dict and cell.get('channel_id') == channel['id'] and cell.get('entity_ref') == channel['entity_ref']
                 and cell.get('surface_id') == channel['surface_id'], 'source channel identity differs')
        _number(cell.get('source_time_s'))
        refs = cell.get('source_evidence_refs')
        _require(type(refs) is list and len(refs) <= 128, 'bounded source evidence links required')
        for evidence in refs:
            _require(type(evidence) is dict and set(evidence) == {'document_sha256', 'evidence_id'}
                     and evidence['document_sha256'] == channel['entity_ref']['document_sha256'], 'foreign source evidence link')
            _text(evidence['evidence_id'])
        template = _template(cell.get('annotation_template'), source, index, channel, cell)
        if row['source_status'] != 'decoded':
            _require(template is None, 'metadata-only frame cannot acquire a template')
        anchor, match = anchors.get(channel['id']), body_rows.get(channel['id'])
        projected.append({**{k: cell.get(k) for k in ('source_observation_id', 'source_time_s', 'source_partition',
            'source_visibility', 'reason', 'source_evidence_refs')}, 'template': template,
            'supplied_annotation': supplied.get((index, channel['id'])),
            'envelope': None if anchor is None else {k: anchor.get(k) for k in ('status', 'reason', 'world_envelope_m')},
            'body_join': None if match is None else {k: match.get(k) for k in ('status', 'reason', 'body_point')}})
    admission = value.get('image_admission')
    return {'frame': index, 'status': row['source_status'], 'body_reference': body, 'channels': projected,
            'image_admission': None if admission is None else {k: admission.get(k) for k in
                ('status', 'reason', 'source_time_s', 'source_partition', 'body_reference', 'evidence_ref', 'observation_ref')}}


_STYLE = '''body{margin:0;background:var(--weft-canvas);color:var(--weft-ink);font:16px/1.5 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:28px 20px}h1{line-height:1.15}section{background:var(--weft-surface);border:1px solid var(--weft-border);border-radius:12px;padding:20px;margin:18px 0}label{display:block;margin:12px 0}input,select,button{font:inherit;padding:8px;border:1px solid var(--weft-border);border-radius:6px;max-width:100%;box-sizing:border-box}input[type=text],select{width:100%}button{background:var(--weft-primary);color:var(--weft-primary-foreground);cursor:pointer;margin:4px}button:disabled{opacity:.5;cursor:default}a{color:var(--weft-primary)}pre,code{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.9em}pre{max-height:20em;overflow:auto;background:var(--weft-canvas);padding:12px}table{border-collapse:collapse;width:100%}th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--weft-border);padding:8px;overflow-wrap:anywhere}.scroll{overflow:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}.muted{color:var(--weft-muted)}.notice{border-left:4px solid var(--weft-accent);padding-left:12px}:focus-visible{outline:3px solid var(--weft-ink);outline-offset:2px}nav{display:flex;flex-wrap:wrap;gap:16px}details{margin:12px 0}summary{cursor:pointer}'''

_JS_COMMON = ''''use strict';const D=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id);let downloadURL=null;
function text(id,value){el(id).textContent=value;}function json(value){return JSON.stringify(value,null,2);}function shown(value){return value===null||value===undefined?'Unknown':typeof value==='object'?json(value):String(value);}function node(tag,value){const n=document.createElement(tag);if(value!==undefined)n.textContent=value;return n;}function download(value,name){if(downloadURL)URL.revokeObjectURL(downloadURL);downloadURL=URL.createObjectURL(new Blob([json(value)+'\\n'],{type:'application/json'}));const a=node('a');a.href=downloadURL;a.download=name;a.click();}window.addEventListener('pagehide',()=>{if(downloadURL)URL.revokeObjectURL(downloadURL);downloadURL=null;});
'''

_JS_SELECT = '''let choices=[],human=null,entityPage=0,visible=[];const pageSize=50;
function entityName(e){return e.entity_ref.object_id+' · '+e.entity_ref.document_sha256.slice(0,12);}
function entities(){const query=el('entity-search').value.trim().toLowerCase();visible=D.entities.map((e,i)=>[e,i]).filter(([e])=>entityName(e).toLowerCase().includes(query));const pages=Math.max(1,Math.ceil(visible.length/pageSize));entityPage=Math.min(entityPage,pages-1);const options=[node('option','Choose an observed ID')];options[0].value='';for(const[e,i]of visible.slice(entityPage*pageSize,(entityPage+1)*pageSize)){const o=node('option',entityName(e));o.value=String(i);options.push(o);}el('entity').replaceChildren(...options);el('entity').value='';text('entity-count',visible.length+' matching IDs · page '+(entityPage+1)+' of '+pages);el('entity-prev').disabled=entityPage===0;el('entity-next').disabled=entityPage+1===pages;details();}
function selected(){const value=el('entity').value;return value===''?null:D.entities[Number(value)];}
function details(){const e=selected();text('entity-details',e?json(e):'No entity selected. Labels are saved claims, not verified identity.');el('set-human').disabled=!D.has_body||!e;}
function update(){const rows=[];choices.forEach((c,i)=>{const tr=node('tr');for(const v of[c.id,c.entity_ref.object_id,c.point_role,c.surface_id])tr.append(node('td',v));const td=node('td'),button=node('button','Remove');button.type='button';button.addEventListener('click',()=>{choices.splice(i,1);update();});td.append(button);tr.append(td);rows.push(tr);});el('chosen').replaceChildren(...rows);text('human-choice',human?json(human):'Unknown — no body entity chosen');el('download-selection').disabled=!el('source-ack').checked||choices.length===0;el('add-channel').disabled=choices.length>=12;}
el('entity-search').addEventListener('input',()=>{entityPage=0;entities();});el('entity-prev').addEventListener('click',()=>{entityPage--;entities();});el('entity-next').addEventListener('click',()=>{entityPage++;entities();});el('entity').addEventListener('change',details);el('source-ack').addEventListener('change',update);
el('set-human').addEventListener('click',()=>{const e=selected();if(D.has_body&&e){human={...e.entity_ref};update();}});el('clear-human').addEventListener('click',()=>{human=null;update();});
el('add-channel').addEventListener('click',()=>{const e=selected(),id=el('channel-id').value.trim(),role=el('point-role').value.trim(),surface=el('surface-id').value.trim();if(!e||!id||id.length>80||!role||role.length>120||!surface||surface.length>80||choices.some(c=>c.id===id)||choices.length>=12){text('selection-error','Choose an observed ID and enter a unique channel name, point label and surface name.');return;}choices.push({id,entity_ref:{...e.entity_ref},point_role:role,surface_id:surface});text('selection-error','');update();});
el('download-selection').addEventListener('click',()=>{if(!el('source-ack').checked||!choices.length)return;download({schema_version:1,profile:'exercise-surface-selection-v1',input_bindings:D.input_bindings,channels:choices,human_entity_ref:human},'exercise-surface-selection.json');});entities();update();
'''

_JS_FRAME = '''let previous='';function render(){const key=el('frame').value+':'+el('channel').value;if(key===previous)return;previous=key;const r=D.rows[Number(el('frame').value)],i=Number(el('channel').value),c=r.channels[i],decl=D.channels[i];text('frame-title','Original frame '+r.frame+' · '+r.status);text('source-clock','Source time: '+shown(c.source_time_s)+' s · '+shown(c.source_partition)+' · '+shown(c.source_visibility));text('body-clock',r.body_reference?'Original body knot: '+shown(r.body_reference.body_time_s)+' s · '+r.body_reference.evidence_id:'No original body reference at this frame; none is interpolated.');text('availability',c.reason||'Construction annotation template available; values still require explicit input.');text('envelope',c.envelope?json(c.envelope):'Unavailable — the request has not produced a surface envelope.');text('body-join',c.body_join?json(c.body_join):'Unavailable — no conditional body-to-surface join.');text('image-admission',r.image_admission?json(r.image_admission):'Unavailable — image-source admission has not been prepared.');text('source-links',json({entity_ref:decl.entity_ref,observation_id:c.source_observation_id,evidence_refs:c.source_evidence_refs}));text('template',c.template?json(c.template):'No template for this excluded, unsampled, unavailable or metadata-only slot.');text('supplied',c.supplied_annotation?json(c.supplied_annotation):'Unknown — no annotation supplied for this frame/channel.');text('surface-range',json(D.surfaces.find(s=>s.id===decl.surface_id)||{id:decl.surface_id,status:'unknown'}));el('download-template').disabled=c.template===null;}
const options=D.rows.map((r,i)=>{const o=node('option','Frame '+r.frame+' · '+r.status);o.value=String(i);return o;});el('frame').replaceChildren(...options);el('frame').value='0';el('channel').replaceChildren(...D.channels.map((c,i)=>{const o=node('option',c.id+' · '+c.point_role);o.value=String(i);return o;}));el('channel').value='0';el('frame').addEventListener('change',render);el('channel').addEventListener('change',render);el('download-template').addEventListener('click',()=>{const r=D.rows[Number(el('frame').value)],i=Number(el('channel').value),template=r.channels[i].template;if(template)download(template,'annotation-'+String(r.frame).padStart(4,'0')+'-'+String(i+1).padStart(2,'0')+'.json');});render();
'''


def _page(title, content, data=None, script=''):
    payload = '' if data is None else '<script id="data" type="application/json">' + _encoded(data).decode().replace('<', '\\u003c') + '</script><script>' + _JS_COMMON + script + '</script>'
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'self\' \'unsafe-inline\'; script-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
        f'<title>{html.escape(title)} · Weft</title><link rel="stylesheet" href="brand.css"><style>{_STYLE}</style></head>'
        f'<body><main><p class="muted">Weft · saved surface inputs</p><h1>{html.escape(title)}</h1>{content}'
        '<p class="muted">Offline saved-data view. No video decoding, fitting, simulation or network request runs here.</p></main>' + payload + '</body></html>').encode()


def _frame_page(rows, channels, surfaces, index, count):
    nav = ['<a href="index.html">Overview</a>']
    if index: nav.append(f'<a href="frames-{index:03d}.html">Previous page</a>')
    if index + 1 < count: nav.append(f'<a href="frames-{index+2:03d}.html">Next page</a>')
    content = '<nav>' + ''.join(nav) + f'</nav><p>Page {index+1} of {count}; original frames {rows[0]["frame"]}–{rows[-1]["frame"]}. Every original slot is retained.</p>'
    content += '''<section><div class="grid"><label>Original frame<select id="frame"></select></label><label>Chosen channel<select id="channel"></select></label></div><h2 id="frame-title"></h2><p id="source-clock"></p><p id="body-clock"></p><p id="availability" role="status"></p></section>
<section><h2>Supply a pixel annotation</h2><p>Use this exact source-frame template. Fill pixel coordinates and their uncertainty only when justified. Keep unknown or occluded states explicit. A visible point is not proof of contact.</p><button id="download-template" type="button">Download this annotation template</button><details><summary>Exact template</summary><pre id="template"></pre></details><details><summary>Already supplied annotation</summary><pre id="supplied"></pre></details><p>Add edited rows to <code>annotations.annotations</code> in <a href="../collection-request.json">collection-request.json</a>; templates do not change the saved request.</p></section>
<section><h2>Supplied surface range</h2><p>Height and extents are world-metre assumptions. Pixel uncertainty is separate. Unknown bounds are not zero height or a contact measurement.</p><pre id="surface-range"></pre><details><summary>Saved envelope status</summary><pre id="envelope"></pre></details><details><summary>Saved body correspondence</summary><pre id="body-join"></pre></details><details><summary>Separate image-source admission</summary><pre id="image-admission"></pre></details></section>
<section><h2>Exact source links</h2><pre id="source-links"></pre><a href="../collection-frames.jsonl">Complete original frame/channel records</a></section>'''
    return _page('Inspect original frame inputs', content, {'rows': rows, 'channels': channels, 'surfaces': surfaces}, _JS_FRAME)


def build_exercise_surface_report(files, *, max_seconds=5):
    """Return fixed relative-name bytes; the caller owns sealing and publication."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 5, 'time bound must be within five seconds')
    deadline = time.monotonic() + max_seconds
    def check():
        if time.monotonic() >= deadline: raise TimeoutError('Exercise surface report deadline exceeded')
    check()
    _require(type(files) is dict and 3 <= len(files) <= 32 and all(type(name) is str and _INPUT.fullmatch(name)
             and type(value) is bytes for name, value in files.items()), 'bounded fixed input names/bytes required')
    _require(sum(map(len, files.values())) <= MAX_INPUT, 'saved report inputs exceed 32 MiB')
    for name in ('inventory.json', 'selection.json', 'workflow-state.json'):
        _require(name in files and 0 < len(files[name]) <= 2 << 20, 'missing bounded workflow metadata')
    inventory, selection, state = (_json(files[name]) for name in ('inventory.json', 'selection.json', 'workflow-state.json'))
    bindings = inventory.get('input_bindings')
    _require(type(bindings) is dict and set(bindings) == _BINDINGS and type(bindings['source_sha256']) is str and _SHA.fullmatch(bindings['source_sha256'])
             and type(bindings['observation_document_sha256s']) is list and 1 <= len(bindings['observation_document_sha256s']) <= 8,
             'exact original input bindings required')
    documents = bindings['observation_document_sha256s']
    _require(documents == sorted(set(documents)) and all(type(v) is str and _SHA.fullmatch(v) for v in documents), 'invalid observation identities')
    for key in ('camera_raw_sha256', 'body_raw_sha256'):
        _require(bindings[key] is None or type(bindings[key]) is str and _SHA.fullmatch(bindings[key]), 'invalid optional input identity')
    _require(inventory.get('schema_version') == 1 and inventory.get('profile') == 'exercise-surface-inventory-v1'
             and state.get('schema_version') == 1 and state.get('profile') == 'exercise-surface-workflow-v1'
             and state.get('status') == 'completed' and state.get('collection_status') in _STATUSES
             and state.get('input_bindings') == bindings and selection.get('input_bindings') == bindings,
             'workflow profiles or original-input identities differ')
    _require(state.get('fit_or_simulator_admission') is False and state.get('body_or_floor_installed') is False
             and state.get('physical_validation') == 'unverified'
             and all(state.get(k) == 'unknown' for k in ('contact', 'load', 'grasp')), 'collection cannot claim physical or fit admission')
    hashes = {name: sha256(raw).hexdigest() for name, raw in files.items()}; check()
    _require(state.get('selection_sha256') == hashes['selection.json'], 'saved selection pin differs')
    input_pins = inventory.get('input_files')
    source = inventory.get('source')
    _require(type(source) is dict and source.get('sha256') == bindings['source_sha256'], 'source summary identity differs')
    source_frames = max(_count(source.get('decoded_frames'), MAX_FRAMES), _count(source.get('metadata_frames'), MAX_FRAMES))
    _require(source_frames > 0 and type(inventory.get('source_clocks')) is dict
             and set(inventory['source_clocks']) == set(documents), 'complete source coverage and original clocks required')
    expected_inputs = {f'surface-observation-{i:02d}.json': pin for i, pin in enumerate(documents)}
    for name, key in (('surface-camera.json', 'camera_raw_sha256'), ('original-human.json', 'body_raw_sha256')):
        if bindings[key] is not None: expected_inputs[name] = bindings[key]
    _require(input_pins == expected_inputs and all(hashes.get(name) == pin for name, pin in input_pins.items()), 'exact original input bytes differ')
    entities = inventory.get('entities')
    _require(type(entities) is list and len(entities) <= 1024, 'bounded exact entity choices required')
    entity_keys = set(); display_entities = []
    for entity in entities:
        check(); key = _ref(entity.get('entity_ref'), documents)
        _require(key not in entity_keys, 'duplicate entity choice'); entity_keys.add(key)
        for name in ('source_slots', 'construction_annotation_slots'): _count(entity.get(name), MAX_FRAMES)
        _require(type(entity.get('properties')) is dict, 'construction-filtered property claims required')
        display_entities.append({k: entity[k] for k in ('entity_ref', 'properties', 'source_slots', 'construction_annotation_slots', 'partition_counts', 'visibility_counts')})
    _require(set(selection) == {'schema_version', 'profile', 'input_bindings', 'channels', 'human_entity_ref'}
             and type(selection['schema_version']) is int and selection['schema_version'] == 1
             and selection['profile'] == 'exercise-surface-selection-v1' and type(selection['channels']) is list
             and len(selection['channels']) <= MAX_CHANNELS, 'explicit existing-schema selection required')
    channels = selection['channels']; channel_ids = set()
    for channel in channels:
        _require(type(channel) is dict and set(channel) == {'id', 'entity_ref', 'point_role', 'surface_id'}
                 and _ref(channel['entity_ref'], documents) in entity_keys, 'channel must name an existing observed entity')
        for name, bound in (('id', 80), ('point_role', 120), ('surface_id', 80)): _text(channel[name], bound)
        _require(channel['id'] not in channel_ids, 'duplicate channel identity'); channel_ids.add(channel['id'])
    if selection['human_entity_ref'] is not None:
        _require(bindings['body_raw_sha256'] is not None and _ref(selection['human_entity_ref'], documents) in entity_keys,
                 'body entity must be explicitly selected from original observations')
    status = state['collection_status']; metadata = request = None; pages = []; surfaces = []
    if status == 'needs_selection':
        _require(not channels and selection['human_entity_ref'] is None and state.get('request_sha256') is None
                 and state.get('collection_manifest_sha256') is None and not any(n.startswith('collection-') for n in files),
                 'discovery cannot invent selected channels or a qualified request')
    else:
        required = {'collection-report.json', 'collection-request.json', 'collection-frames.jsonl', 'collection-manifest.json'}
        _require(channels and required <= files.keys() and len(files['collection-frames.jsonl']) <= 8 << 20,
                 'selected collection needs complete bounded saved rows')
        metadata = _json(files['collection-report.json']); request = _json(files['collection-request.json']); seal = _json(files['collection-manifest.json'])
        _require(state.get('request_sha256') == hashes['collection-request.json'] == metadata.get('request_sha256')
                 and state.get('collection_manifest_sha256') == hashes['collection-manifest.json']
                 and seal.get('profile') == metadata.get('profile') == 'exercise-surface-input-collection-v1'
                 and seal.get('status') == metadata.get('status') == status and metadata.get('input_bindings') == request.get('input_bindings') == bindings,
                 'saved request/collection identities differ')
        _require(type(seal.get('files')) is dict and set(seal['files']) == set(files) - {'inventory.json', 'selection.json', 'workflow-state.json', 'collection-manifest.json'}
                 and all(hashes.get(name) == pin for name, pin in seal['files'].items()), 'collection member inventory or bytes differ')
        _require(metadata.get('fit_or_simulator_admission') is False and metadata.get('body_or_floor_installed') is False
                 and metadata.get('body_geometry_evaluations') == 0 and metadata.get('physical_validation') == 'unverified'
                 and all(metadata.get(k) == 'unknown' for k in ('contact', 'load', 'grasp')), 'saved collection overclaims qualification')
        _require(request.get('profile') == 'exercise-surface-input-request-v1' and request.get('annotations', {}).get('channels') == channels,
                 'request channel selection differs')
        surfaces = request['surface_intervals']['surfaces']
        _require(type(surfaces) is list and len(surfaces) <= MAX_CHANNELS, 'bounded saved surface declarations required')
        supplied = {}
        for annotation in request['annotations']['annotations']:
            check(); key = (annotation.get('frame'), annotation.get('channel_id'))
            _require(type(key[0]) is int and 0 <= key[0] < MAX_FRAMES and key[1] in channel_ids and key not in supplied,
                     'duplicate or foreign supplied annotation')
            supplied[key] = annotation
        counts = metadata.get('source_counts', {}); count = _count(counts.get('retained_source_frames'), MAX_FRAMES)
        _require(count == source_frames and counts.get('channel_slots') == count * len(channels)
                 and counts.get('decoded_frames') == source['decoded_frames'] and counts.get('metadata_frames') == source['metadata_frames']
                 and metadata.get('source_clocks') == inventory['source_clocks'], 'original slot denominators or clocks differ')
        lines = files['collection-frames.jsonl'].splitlines()
        _require(len(lines) == count and all(0 < len(raw) <= 128 << 10 for raw in lines), 'complete original frame rows required')
        current, size = [], 0
        for index, raw in enumerate(lines):
            check(); row = _project_row(_json(raw), index, channels, bindings['source_sha256'], supplied); encoded_size = len(_encoded(row))
            _require(encoded_size <= PAGE_BYTES, 'one frame exceeds the report page bound; no rows dropped')
            if current and (len(current) >= PAGE_FRAMES or size + encoded_size > PAGE_BYTES): pages.append(current); current, size = [], 0
            current.append(row); size += encoded_size
        if current: pages.append(current)
        _require(len(pages) + 3 <= MAX_FILES, 'complete report exceeds page bound; no rows dropped')
        config_present = 'surface-config.json' in files
        _require(metadata.get('config_schema_qualified') is config_present and metadata.get('config_member') == ('surface-config.json' if config_present else None), 'saved config readiness differs')
    check()
    esc = lambda value: html.escape(str(value), quote=True)
    content = '<p class="notice">This collects explicit assumptions and observations. It does not recover ground, verify contact, install a body or admit inputs to a fitter.</p>'
    content += '<section><h2>Your source</h2><p>' + str(source_frames) + ' original frame slots; ' + str(source['decoded_frames']) + ' declared decoded frames. '
    content += ('Original body supplied. ' if bindings['body_raw_sha256'] is not None else 'No original body supplied. ')
    content += ('Camera metadata supplied.' if bindings['camera_raw_sha256'] is not None else 'Camera metadata unavailable.')
    content += '</p><p>Source and body clocks stay separate; missing values remain unknown.</p><details><summary>Exact source identity and clock declarations</summary><pre>'
    content += esc(json.dumps({'input_bindings': bindings, 'source': source, 'source_clocks': inventory['source_clocks']}, indent=2)) + '</pre></details></section>'
    data, script = None, ''
    if status == 'needs_selection':
        content += '''<section><h2>1. Choose observed IDs and point labels</h2><p>Choose only the IDs and image points you intend to annotate. A surface name does not classify it as ground or stationary equipment.</p><label><input id="source-ack" type="checkbox"> I want to use the exact source shown above</label><label>Find an observed ID<input id="entity-search" type="text" maxlength="120"></label><p id="entity-count"></p><button id="entity-prev" type="button">Previous IDs</button><button id="entity-next" type="button">Next IDs</button><label>Observed entity<select id="entity"></select></label><details><summary>Saved entity claims and coverage</summary><pre id="entity-details"></pre></details><div class="grid"><label>Unique channel name<input id="channel-id" type="text" maxlength="80" placeholder="For example: ankle_point"></label><label>Image point label<input id="point-role" type="text" maxlength="120" placeholder="Describe the point you will annotate"></label><label>Surface name<input id="surface-id" type="text" maxlength="80" placeholder="For example: support_a"></label></div><button id="add-channel" type="button">Add this channel</button><p id="selection-error" role="alert"></p><div class="scroll"><table><thead><tr><th>Channel</th><th>Observed ID</th><th>Point label</th><th>Surface name</th><th>Action</th></tr></thead><tbody id="chosen"></tbody></table></div></section>
<section><h2>2. Optional body identity</h2><p>If an original body was supplied, explicitly choose which observed entity it represents. This does not verify that identity.</p><button id="set-human" type="button">Use selected entity for the body</button><button id="clear-human" type="button">Leave body identity unknown</button><pre id="human-choice"></pre></section><section><h2>3. Download your selection</h2><button id="download-selection" type="button" disabled>Download selection JSON</button><p>No files change here. Use the downloaded selection to create a new draft:</p>'''
        command = _COMMAND + 'surface-inputs --resume PATH_TO_THIS_COLLECTION --selection PATH_TO_DOWNLOADED_SELECTION --out NEW_DRAFT_DIRECTORY'
        content += '<pre>' + esc(command) + '</pre></section>'
        data = {'entities': display_entities, 'input_bindings': bindings, 'has_body': bindings['body_raw_sha256'] is not None}; script = _JS_SELECT
    else:
        def items(values):
            _require(type(values) is list and len(values) <= 128, 'bounded missing-field guidance required')
            return '<ul>' + ''.join('<li>' + esc(_text(v, 2048)) + '</li>' for v in values) + '</ul>' if values else '<p>None reported. This is not fit or physical admission.</p>'
        content += '<section><h2>What still needs input</h2>' + items(metadata['missing_fields']) + '<h3>Explicit inputs and unavailable conditions</h3>' + items(metadata['required_explicit_inputs'])
        content += '<p>Download <a href="../collection-request.json">collection-request.json</a>, edit only justified fields, and keep unknowns explicit. Pixel uncertainty, world-height ranges, body correspondence and fitting weight are different inputs.</p><pre>'
        content += esc(_COMMAND + 'surface-inputs --resume PATH_TO_THIS_COLLECTION --request PATH_TO_EDITED_COLLECTION_REQUEST --out NEW_COLLECTION_DIRECTORY') + '</pre></section>'
        content += '<section><h2>Inspect original frames</h2><p>' + str(count) + ' frames × ' + str(len(channels)) + ' channels. Every original slot is available; each page renders only the selected frame/channel.</p><nav>'
        content += ''.join(f'<a href="frames-{i+1:03d}.html">Frames {page[0]["frame"]}–{page[-1]["frame"]}</a>' for i, page in enumerate(pages)) + '</nav></section>'
        content += '<section><h2>Chosen channels and body identity</h2><pre>' + esc(json.dumps({'channels': channels, 'original_selection_human_entity_ref': selection['human_entity_ref'],
            'edited_request_human_entity_ref': (request.get('body_binding') or {}).get('human_entity_ref')}, indent=2)) + '</pre><p>An edited body identity is shown separately; no identity is inferred from a channel name.</p></section>'
        if metadata['config_schema_qualified']:
            body = _json(files['original-human.json']) if 'original-human.json' in files else None
            camera_mode = body is not None and isinstance(body.get('camera_reference'), dict)
            objective = 'camera-surface-intervals' if camera_mode else 'surface-intervals'
            command = _COMMAND + 'body-reference-propose --run PATH_TO_SEPARATELY_QUALIFIED_BODY --objective ' + objective
            command += ' --surface-config PATH_TO_COLLECTION/surface-config.json'
            if camera_mode: command += ' --camera-image-scale AUTHORED_VALUE'
            command += ' --out NEW_FIT_DIRECTORY'
            content += '<section><h2>Saved input configuration</h2><p><a href="../surface-config.json">Download surface-config.json</a>. The input schema is qualified; a fitter may still reject the body, camera or mapping. This collection is not a body run.</p>'
            if body is None: content += '<p>No original body was supplied. Conditional source envelopes do not establish a body-to-surface join.</p>'
            content += '<p>After obtaining a separately qualified original body source, the next command is:</p><pre>' + esc(command) + '</pre><p>Replace every uppercase placeholder explicitly. This page does not run that command or choose an image-loss scale.</p></section>'
    content += '<section><h2>Saved evidence</h2><ul>' + ''.join('<li><a href="../' + esc(name) + '">' + esc(name) + '</a></li>' for name in sorted(files)) + '</ul><p>These are exact saved inputs and receipts, not fresh media authentication. Contact, load, grasp and physical accuracy remain unknown or unverified.</p></section>'
    summary = {'schema_version': 1, 'profile': PROFILE, 'collection_status': status, 'input_bindings': bindings,
        'input_files': hashes, 'frame_count': sum(map(len, pages)), 'channel_count': len(channels),
        'pages': [{'file': f'frames-{i+1:03d}.html', 'first_frame': page[0]['frame'], 'last_frame': page[-1]['frame'], 'frames': len(page)} for i, page in enumerate(pages)],
        'selection_defaults': 'none', 'fit_or_simulator_admission': False, 'body_or_floor_installed': False,
        'contact': 'unknown', 'physical_validation': 'unverified', 'scope': 'Saved projected metadata only; full original records remain linked'}
    palette = Path(__file__).parent / 'assets/brand.css'
    with palette.open('rb') as stream: brand = stream.read(16385)
    _require(len(brand) <= 16384 and b'</' not in brand, 'bounded original brand stylesheet required')
    output = {'index.html': _page(_TITLES[status], content, data, script), 'brand.css': brand, 'report.json': _encoded(summary)}
    for index, page in enumerate(pages):
        check(); output[f'frames-{index+1:03d}.html'] = _frame_page(page, channels, surfaces, index, len(pages))
        _require(sum(map(len, output.values())) <= MAX_OUTPUT, 'report exceeds 16 MiB; no slots dropped')
    _require(len(output) <= MAX_FILES and sum(map(len, output.values())) <= MAX_OUTPUT
             and all(len(raw) <= MAX_HTML for name, raw in output.items() if name.endswith('.html')), 'complete report exceeds its file or HTML bounds')
    check()
    return output
