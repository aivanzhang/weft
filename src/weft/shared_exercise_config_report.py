"""Offline worksheets for explicit shared-exercise choices, never world admission."""

from hashlib import sha256
import html
import json
import math
from pathlib import Path, PurePosixPath
import re
import time

from weft.equipment_proposal_config import (
    DEPTH_RATIO_PRIOR, MAX_CONFIG_BYTES, MAX_DEPTH_PROVENANCE_CHARS, MAX_DEPTH_TO_WIDTH_RATIO,
    MIN_DEPTH_TO_WIDTH_RATIO, validate_depth_prior,
)


PROFILE = 'shared-exercise-config-report-v1'
MAX_INPUT, MAX_OUTPUT, MAX_HTML, MAX_FILES = 16 << 20, 16 << 20, 2 << 20, 32
MAX_FRAMES, PAGE_FRAMES, PAGE_BYTES = 1800, 100, 1 << 20
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _require(value, message):
    if not value:
        raise ValueError('Shared exercise configuration report: ' + message)


def _encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def _document(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    try:
        result = json.loads(raw, object_pairs_hook=unique)
        _encode(result)
    except (ValueError, RecursionError, UnicodeError) as error:
        raise ValueError('Shared exercise configuration report: finite strict JSON required') from error
    _require(type(result) is dict, 'JSON object required')
    return result


def _text(value, maximum=512):
    _require(type(value) is str and 0 < len(value) <= maximum and all(ord(c) >= 32 for c in value), 'bounded text required')
    return value


def _count(value, maximum=1800):
    _require(type(value) is int and 0 <= value <= maximum, 'bounded integer required')
    return value


def _number(value):
    _require(value is None or type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e9,
             'finite number or explicit unknown required')
    return value


def _pin(value):
    _require(type(value) is str and _SHA.fullmatch(value), 'exact SHA256 required')
    return value


def _nullable_text(value, maximum=512):
    return None if value is None else _text(value, maximum)


def _issues(value):
    _require(type(value) is list and len(value) <= 128, 'bounded issues required')
    for row in value:
        _require(type(row) is dict and set(row) == {'code', 'message'}, 'named issue required')
        _text(row['code'], 120); _text(row['message'], 2048)


def _options(value):
    _require(type(value) is dict and set(value) == {'body_scale', 'root_assistance'}, 'explicit build options required')
    scale = value['body_scale']
    _require(type(scale) in (int, float) and math.isfinite(scale) and .5 <= scale <= 2
             and type(value['root_assistance']) is bool, 'scale must be 0.5–2 and root assistance boolean')
    return value


def _request_header(value, snapshot_pin):
    fields = {'schema_version', 'profile', 'snapshot_sha256', 'human_entity_ref', 'camera', 'geometry_prior',
              'equipment', 'target_object_id', 'build_options'}
    _require(fields <= set(value) <= fields | {'frame_translation_xy'} and type(value['schema_version']) is int
             and value['schema_version'] == 1 and value['profile'] == 'shared-exercise-config-request-v1'
             and value['snapshot_sha256'] == snapshot_pin, 'request schema or snapshot identity differs')


def _request(value, snapshot_pin):
    _request_header(value, snapshot_pin)
    human = value['human_entity_ref']
    if human is not None:
        _require(type(human) is dict and set(human) == {'document_sha256', 'object_id'}, 'explicit human identity required')
        _pin(human['document_sha256']); _text(human['object_id'], 120)
    camera = value['camera']
    if camera is not None:
        _require(type(camera) is dict and set(camera) == {'mode', 'distance_model_units'}
                 and camera['mode'] == 'fixed_pelvis_perspective_prior', 'unsupported camera choice')
        distance = camera['distance_model_units']
        _require(type(distance) in (int, float) and math.isfinite(distance) and .1 <= distance <= 50,
                 'camera distance must be 0.1–50 model units')
    _require(value['geometry_prior'] in (None, 'floor_box_depth_equals_width', DEPTH_RATIO_PRIOR), 'unsupported geometry choice')
    equipment = value['equipment']
    _require(type(equipment) is list and len(equipment) <= 12, 'at most twelve equipment choices allowed')
    names = set()
    for row in equipment:
        fields = {'object_id', 'source_frame'} | ({'depth_prior'} if value['geometry_prior'] == DEPTH_RATIO_PRIOR else set())
        _require(type(row) is dict and set(row) == fields, 'explicit object/frame pair and selected depth-prior fields required')
        _text(row['object_id'], 80); _count(row['source_frame'], MAX_FRAMES-1)
        if value['geometry_prior'] == DEPTH_RATIO_PRIOR:
            validate_depth_prior(row['depth_prior'])
        _require(row['object_id'] not in names and row['object_id'] != 'human', 'duplicate or reserved equipment ID')
        names.add(row['object_id'])
    _require(value['target_object_id'] is None or value['target_object_id'] in names, 'target must be explicitly selected equipment')
    if value.get('frame_translation_xy') is not None:
        shift = value['frame_translation_xy']
        _require(type(shift) is list and len(shift) == 2 and all(type(v) in (int, float) and math.isfinite(v)
                 and 0 <= v <= 6 for v in shift) and any(v > 0 for v in shift), 'optional XY translation must be nonzero within 0–6')
    _options(value['build_options'])
    return value


def _input_custody(value):
    _require(type(value) is dict and set(value) == {'human', 'equipment', 'source_mappings'}, 'saved input custody required')
    mappings = value['source_mappings']
    _require(type(mappings) is list and len(mappings) <= 7, 'bounded source mappings required')
    for role, row in [('human', value['human']), ('equipment', value['equipment'])] + [('mapping', r) for r in mappings]:
        fields = {'directory', 'manifest_sha256', 'files', 'profile', 'status'} | (set() if role != 'mapping' else
            {'original_sha256', 'derivative_sha256', 'frame_count', 'time_base', 'verification'})
        _require(type(row) is dict and set(row) == fields, 'saved source inventory fields differ')
        pin = _pin(row['manifest_sha256'])
        _require(row['directory'] == ('sources/' + role if role != 'mapping' else 'sources/mappings/' + pin),
                 'saved source directory differs')
        if role == 'mapping':
            _pin(row['original_sha256']); _pin(row['derivative_sha256']); _count(row['frame_count'])
            _require(row['verification'] == 'full_saved_bundle_verified_no_decode', 'unsupported source-mapping claim')
        _nullable_text(row['profile']); _nullable_text(row['status'])
        files = row['files']
        _require(type(files) is dict and len(files) <= 2048, 'bounded source member pins required')
        for name, member_pin in files.items():
            _text(name, 256); path = PurePosixPath(name)
            _require(not path.is_absolute() and '..' not in path.parts and str(path) == name
                     and '\\' not in name and name != 'manifest.json', 'unsafe or duplicate manifest member')
            _pin(member_pin)


def _snapshot(value, check):
    fields = {'schema_version', 'profile', 'source_sha256', 'inputs', 'body', 'source_slots', 'observation_sources',
              'issues', 'counts', 'world_admission', 'contact', 'grasp', 'load'}
    _require(type(value) is dict and set(value) == fields and type(value['schema_version']) is int
             and value['schema_version'] == 1 and value['profile'] == 'shared-exercise-config-snapshot-v1', 'snapshot schema differs')
    _pin(value['source_sha256']); _input_custody(value['inputs']); _issues(value['issues'])
    _require(value['world_admission'] == 'not_evaluated' and all(value[k] == 'unknown' for k in ('contact', 'grasp', 'load')),
             'snapshot cannot claim world or contact admission')
    counts = value['counts']
    _require(type(counts) is dict and set(counts) == {'original_source_slots', 'decoded_source_slots', 'body_references',
             'human_entity_slots', 'equipment_entity_slots', 'metadata_tail_slots'}, 'complete snapshot denominators required')
    for number in counts.values(): _count(number, 43200)
    frames, decoded = counts['original_source_slots'], counts['decoded_source_slots']
    _require(0 < frames <= MAX_FRAMES and decoded <= frames and counts['metadata_tail_slots'] == frames-decoded,
             'original frame denominators differ')
    body = value['body']
    _require(type(body) is dict and set(body) == {'document_sha256', 'profile', 'reference_count', 'references', 'validation'}
             and body['validation'] == 'deferred_to_existing_builder', 'body validation must remain deferred')
    _pin(body['document_sha256']); _nullable_text(body['profile']); _count(body['reference_count'], 900)
    references = body['references']
    _require(type(references) is list and len(references) == body['reference_count'] == counts['body_references'], 'body reference count differs')
    by_frame = [[] for _ in range(frames)]
    for index, reference in enumerate(references):
        check()
        _require(type(reference) is dict and set(reference) == {'index', 'source_frame', 'evidence_id', 'time_s'}
                 and type(reference['index']) is int and reference['index'] == index, 'body reference identity differs')
        _count(reference['source_frame'], frames-1); _text(reference['evidence_id']); _number(reference['time_s'])
        _require(reference['time_s'] is not None and reference['time_s'] >= 0, 'body time must remain explicit')
        by_frame[reference['source_frame']].append(index)
    original = value['source_slots']
    _require(type(original) is list and len(original) == frames, 'complete original source slots required')
    for index, row in enumerate(original):
        check()
        _require(type(row) is dict and set(row) == {'source_frame', 'source_status', 'nominal_time_s', 'body_reference_indices'}
                 and type(row['source_frame']) is int and row['source_frame'] == index
                 and row['source_status'] == ('decoded' if index < decoded else 'metadata_only_not_decoded')
                 and row['body_reference_indices'] == by_frame[index]
                 and all(type(v) is int for v in row['body_reference_indices']), 'original source slot or reference join differs')
        _number(row['nominal_time_s'])
    sources = value['observation_sources']
    _require(type(sources) is list and len(sources) == 2 and {s.get('role') for s in sources} == {'human', 'equipment'},
             'one human and one equipment observation source required')
    _require(len({s.get('document_sha256') for s in sources}) == 2, 'observation documents must stay distinct')
    projected, role_counts = [], {}
    source_fields = {'role', 'document_sha256', 'source_sha256', 'image', 'time', 'decoded_frames', 'metadata_frames',
                     'alignment', 'entities', 'slots'}
    slot_fields = {'source_frame', 'original_frame', 'time_s', 'object_id', 'observation_id', 'partition', 'visibility',
                   'admission', 'reason', 'evidence_ids', 'bbox_available', 'body_reference_available', 'selection_eligible'}
    for source in sources:
        check(); _require(set(source) == source_fields, 'observation-source fields differ')
        _pin(source['document_sha256']); _pin(source['source_sha256'])
        _count(source['decoded_frames']); _count(source['metadata_frames'])
        _require(type(source['image']) is dict and type(source['time']) is dict
                 and len(_encode({'image':source['image'], 'time':source['time']})) <= 16384, 'bounded image and clock metadata required')
        _require(source['alignment'] in {'original', 'verified_derivative', 'unavailable'}, 'unknown source alignment')
        if source['alignment'] == 'original':
            _require(source['source_sha256'] == value['source_sha256'], 'original source identity differs')
        if source['alignment'] == 'verified_derivative':
            _require(any(m['original_sha256'] == value['source_sha256'] and m['derivative_sha256'] == source['source_sha256']
                         for m in value['inputs']['source_mappings']), 'verified derivative has no saved mapping pin')
        entities = source['entities']
        _require(type(entities) is list and len(entities) <= 12, 'at most twelve entities per source')
        names = {}
        for i, entity in enumerate(entities):
            _require(type(entity) is dict and set(entity) == {'object_id', 'label', 'kind'}, 'entity identity fields differ')
            name = _text(entity['object_id'], 120); _nullable_text(entity['label']); _nullable_text(entity['kind'])
            _require(name not in names, 'duplicate entity identity'); names[name] = i
        slots = source['slots']; role_counts[source['role']] = len(slots) if type(slots) is list else -1
        _require(type(slots) is list and len(slots) == source['decoded_frames']*len(entities), 'complete observation frame/entity grid required')
        grouped = [[None]*len(entities) for _ in range(source['decoded_frames'])]
        for slot in slots:
            check(); _require(type(slot) is dict and set(slot) == slot_fields, 'observation slot fields differ')
            frame = _count(slot['source_frame'], source['decoded_frames']-1)
            _require(slot['object_id'] in names, 'foreign slot entity'); entity_index = names[slot['object_id']]
            _require(grouped[frame][entity_index] is None, 'duplicate frame/entity slot'); grouped[frame][entity_index] = slot
            original_frame = slot['original_frame']
            if original_frame is not None: _count(original_frame, frames-1)
            _require(source['alignment'] != 'original' or original_frame == frame, 'original frame index differs')
            _require(source['alignment'] != 'unavailable' or original_frame is None, 'unaligned source invents original frame')
            _number(slot['time_s'])
            for key in ('observation_id', 'partition', 'visibility', 'admission', 'reason'): _nullable_text(slot[key], 1024)
            _require(type(slot['evidence_ids']) is list and len(slot['evidence_ids']) <= 128, 'bounded evidence links required')
            for evidence in slot['evidence_ids']: _text(evidence)
            _require(all(type(slot[key]) is bool for key in ('bbox_available', 'body_reference_available', 'selection_eligible')),
                     'availability must be explicit booleans')
            _require(not slot['selection_eligible'] or original_frame is not None, 'unaligned slot cannot be eligible')
            _require(slot['body_reference_available'] == (original_frame is not None and bool(by_frame[original_frame])),
                     'body-reference availability differs from original knots')
        projected.append({**{k:v for k,v in source.items() if k != 'slots'}, 'rows':grouped})
    _require(role_counts['human'] == counts['human_entity_slots'] and role_counts['equipment'] == counts['equipment_entity_slots']
             and sum(role_counts.values()) <= 43200, 'observation denominators differ')
    return projected


_STYLE = '''body{margin:0;background:var(--weft-canvas);color:var(--weft-ink);font:16px/1.5 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:28px 20px}h1{line-height:1.15}section{background:var(--weft-surface);border:1px solid var(--weft-border);border-radius:12px;padding:20px;margin:18px 0}label{display:block;margin:12px 0}input,select,textarea,button{font:inherit;padding:8px;border:1px solid var(--weft-border);border-radius:6px;max-width:100%;box-sizing:border-box}input:not([type=checkbox]),select,textarea{width:100%}button{background:var(--weft-primary);color:var(--weft-primary-foreground);cursor:pointer;margin:4px}button:disabled{opacity:.5;cursor:default}a{color:var(--weft-primary)}pre,code{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.9em}pre{max-height:24em;overflow:auto;background:var(--weft-canvas);padding:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}.muted{color:var(--weft-muted)}.notice{border-left:4px solid var(--weft-accent);padding-left:12px}:focus-visible{outline:3px solid var(--weft-ink);outline-offset:2px}nav{display:flex;flex-wrap:wrap;gap:16px}details{margin:12px 0}summary{cursor:pointer}'''


def _page(title, content, data=None, script=''):
    payload = '' if data is None else ('<script id="data" type="application/json">'
        + _encode(data).decode().replace('<', '\\u003c') + '</script><script>' + script + '</script>')
    raw = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'self\' \'unsafe-inline\'; script-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
        f'<title>{html.escape(title)} · Weft</title><link rel="stylesheet" href="brand.css"><style>{_STYLE}</style></head>'
        f'<body><main><p class="muted">Weft · shared exercise configuration</p><h1>{html.escape(title)}</h1>{content}'
        '<p class="muted">Offline saved-data guide. No fitting, simulation, media decoding or network request runs here.</p></main>'
        + payload + '</body></html>').encode()
    _require(len(raw) <= MAX_HTML, 'HTML exceeds two MiB')
    return raw


_JS_COMMON = """'use strict';const D=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id);
function text(id,value){const n=el(id),s=String(value);if(n.textContent!==s)n.textContent=s;}function pretty(v){return JSON.stringify(v,null,2);}function node(tag,value){const n=document.createElement(tag);if(value!==undefined)n.textContent=value;return n;}function option(value,label){const n=node('option',label);n.value=String(value);return n;}function index(value,length){if(!/^(0|[1-9][0-9]*)$/.test(value))return null;const i=Number(value);return i<length?i:null;}function show(v){return v===null||v===undefined?'Unknown':String(v);}
"""

_JS_EDITOR = r"""let equipment=D.request.equipment.map(r=>({object_id:r.object_id,source_frame:r.source_frame,ratio:r.depth_prior?String(r.depth_prior.depth_to_width_ratio):'',provenance:r.depth_prior?r.depth_prior.provenance:''})),revision=0,previous='',edited=null,downloadURL=null,renderedGeometry=D.request.geometry_prior||'';
const fields=['human','camera-mode','distance','geometry','body-scale','assistance','translation','shift-x','shift-y','target'];
function fill(id,options,value){el(id).replaceChildren(...options);el(id).value=value;}
const humans=[option('','Choose body evidence'),...D.humans.map((h,i)=>{const o=option(i,(h.label||h.object_id)+(h.object_id==='person-0'?'':' — unsupported body ID'));o.disabled=h.object_id!=='person-0';return o;})];
let humanValue='';if(D.request.human_entity_ref){humanValue=String(D.humans.findIndex(h=>h.object_id===D.request.human_entity_ref.object_id&&h.document_sha256===D.request.human_entity_ref.document_sha256));if(humanValue==='-1'){humans.push(option('saved','Saved ID not found in this inventory'));humanValue='saved';}}
fill('human',humans,humanValue);fill('object',[option('','Choose equipment'),...D.objects.map((o,i)=>option(i,o.label||o.object_id))],'');el('source-frame').value='';el('source-frame').max=String(Math.max(0,D.eligibility.length-1));el('source-frame').disabled=D.eligibility.length===0;
el('camera-mode').value=D.request.camera?'fixed_pelvis_perspective_prior':'';el('distance').value=D.request.camera?String(D.request.camera.distance_model_units):'';el('geometry').value=D.request.geometry_prior||'';
el('body-scale').value=String(D.request.build_options.body_scale);el('assistance').value=String(D.request.build_options.root_assistance);el('translation').checked=Array.isArray(D.request.frame_translation_xy);el('shift-x').value=el('translation').checked?String(D.request.frame_translation_xy[0]):'';el('shift-y').value=el('translation').checked?String(D.request.frame_translation_xy[1]):'';
function selectedRows(target){const rows=[];equipment.forEach((r,i)=>{const line=node('div'),remove=node('button','Remove');remove.type='button';line.append(node('span',r.object_id+' · equipment frame '+r.source_frame+' '));remove.addEventListener('click',()=>{const oldTarget=el('target').value;equipment.splice(i,1);revision++;selectedRows(oldTarget);update();choice();});line.append(remove);if(el('geometry').value===D.depth_prior.mode){const ratioLabel=node('label',r.object_id+' depth ÷ width (dimensionless)'),ratio=node('input'),provenanceLabel=node('label',r.object_id+' ratio source or reason (supplied, unverified)'),provenance=node('textarea');ratio.type='number';ratio.min=String(D.depth_prior.minimum);ratio.max=String(D.depth_prior.maximum);ratio.step='any';ratio.value=r.ratio;provenance.rows=2;provenance.maxLength=2*D.depth_prior.maximum_provenance_characters;provenance.value=r.provenance;for(const event of ['input','change']){ratio.addEventListener(event,()=>{if(r.ratio===ratio.value)return;r.ratio=ratio.value;revision++;update();});provenance.addEventListener(event,()=>{if(r.provenance===provenance.value)return;r.provenance=provenance.value;revision++;update();});}ratioLabel.append(ratio);provenanceLabel.append(provenance);line.append(ratioLabel,provenanceLabel);}rows.push(line);});el('selected-equipment').replaceChildren(...rows);fill('target',[option('','Choose the task target'),...equipment.map(r=>option(r.object_id,r.object_id))],equipment.some(r=>r.object_id===target)?target:'');}
function equipmentRequest(geometry,errors){return equipment.map(r=>{const row={object_id:r.object_id,source_frame:r.source_frame};if(geometry===D.depth_prior.mode){const value=r.ratio.trim()===''?null:Number(r.ratio);if(value===null||!Number.isFinite(value)||value<D.depth_prior.minimum||value>D.depth_prior.maximum)errors.push(r.object_id+' needs a supplied depth/width ratio between '+D.depth_prior.minimum+' and '+D.depth_prior.maximum+'.');if(!r.provenance.trim()||Array.from(r.provenance).length>D.depth_prior.maximum_provenance_characters)errors.push(r.object_id+' needs a ratio source or reason (1–'+D.depth_prior.maximum_provenance_characters+' characters).');row.depth_prior={depth_to_width_ratio:value,provenance:r.provenance};}return row;});}
function bounded(id,lo,hi,errors,label){const raw=el(id).value.trim();if(raw==='')return null;const v=Number(raw);if(!Number.isFinite(v)||v<lo||v>hi){errors.push(label+' must be '+lo+'–'+hi+'.');return null;}return v;}
function update(){const key=fields.map(id=>id==='translation'?el(id).checked:el(id).value).join('|')+'|'+revision;if(key===previous)return;previous=key;const errors=[],missing=[],humanIndex=index(el('human').value,D.humans.length),human=humanIndex===null?(el('human').value==='saved'?D.request.human_entity_ref:null):{document_sha256:D.humans[humanIndex].document_sha256,object_id:D.humans[humanIndex].object_id};
 const distance=bounded('distance',.1,50,errors,'Camera distance'),scale=bounded('body-scale',.5,2,errors,'Body size'),mode=el('camera-mode').value,camera=mode==='fixed_pelvis_perspective_prior'&&distance!==null?{mode,distance_model_units:distance}:null,geometry=['floor_box_depth_equals_width',D.depth_prior.mode].includes(el('geometry').value)?el('geometry').value:null;
 if(mode!==''&&mode!=='fixed_pelvis_perspective_prior')errors.push('Choose the supported camera assumption.');if(el('geometry').value!==''&&!geometry)errors.push('Choose the supported geometry assumption.');if(scale===null)errors.push('Choose a body size.');const assistance=el('assistance').value;if(!['true','false'].includes(assistance))errors.push('Choose assistance on or off.');
 if(human&&human.object_id!=='person-0')errors.push('This builder requires saved body ID person-0. Other identities remain inspectable.');if(!human)missing.push('body evidence');if(!camera)missing.push('camera assumption and distance');if(!geometry)missing.push('floor-resting box assumption');if(!equipment.length)missing.push('equipment and frame');if(!el('target').value)missing.push('target');
 edited={schema_version:1,profile:'shared-exercise-config-request-v1',snapshot_sha256:D.request.snapshot_sha256,human_entity_ref:human,camera,geometry_prior:geometry,equipment:equipmentRequest(geometry,errors),target_object_id:el('target').value||null,build_options:{body_scale:scale,root_assistance:assistance==='true'}};
 el('shift-x').disabled=el('shift-y').disabled=!el('translation').checked;if(el('translation').checked){const x=bounded('shift-x',0,6,errors,'X translation'),y=bounded('shift-y',0,6,errors,'Y translation');if(x===null||y===null||x+y===0)errors.push('Translation needs two numbers and at least one positive value.');else edited.frame_translation_xy=[x,y];}else if(D.request.frame_translation_xy===null)edited.frame_translation_xy=null;
 if(edited.target_object_id&&!equipment.some(r=>r.object_id===edited.target_object_id))errors.push('Target must be selected equipment.');const preview=pretty(edited),asciiBytes=preview.length+5*(preview.match(/[\u007f-\uffff]/g)||[]).length+1;if(asciiBytes+1024>D.depth_prior.maximum_request_bytes)errors.push('The complete request exceeds its saved-file budget. Shorten provenance text; no equipment or text is dropped automatically.');text('request-state',errors.length?errors.join(' '):missing.length?'Still to choose: '+missing.join(', ')+'. You can save this incomplete request.':'Choices recorded. Run the setup command to validate them.');text('request-preview',preview);el('download-request').disabled=errors.length>0;
}
let previousChoice='';function choice(){const key=el('object').value+'|'+el('source-frame').value+'|'+revision;if(key===previousChoice)return;previousChoice=key;const oi=index(el('object').value,D.objects.length),fi=index(el('source-frame').value,D.eligibility.length);let message='Choose equipment and its video frame; nothing is selected automatically.',valid=false;if(oi!==null&&fi!==null){const cell=D.eligibility[fi][oi];valid=cell[1]&&D.objects[oi].object_id!=='human'&&D.objects[oi].object_id.length<=80&&!equipment.some(r=>r.object_id===D.objects[oi].object_id)&&equipment.length<12;message='Equipment frame '+fi+' · original frame '+show(cell[0])+'. '+(cell[1]?'Saved inputs allow this choice; world geometry is still unchecked.':'Not eligible in the saved inputs. Inspect the frame for its reason.');el('inspect-choice').href='frames-'+String(D.frame_pages[fi]+1).padStart(3,'0')+'.html';}text('choice-state',message);el('add-equipment').disabled=!valid;}
el('add-equipment').addEventListener('click',()=>{choice();if(el('add-equipment').disabled)return;const oi=index(el('object').value,D.objects.length),fi=index(el('source-frame').value,D.eligibility.length);equipment.push({object_id:D.objects[oi].object_id,source_frame:fi,ratio:'',provenance:''});revision++;selectedRows(el('target').value);el('object').value='';el('source-frame').value='';update();choice();});
for(const id of fields)for(const event of ['change','input'])el(id).addEventListener(event,id==='geometry'?()=>{if(renderedGeometry===el('geometry').value)return;renderedGeometry=el('geometry').value;selectedRows(el('target').value);update();choice();}:update);for(const id of ['object','source-frame'])for(const event of ['change','input'])el(id).addEventListener(event,choice);
el('download-request').addEventListener('click',()=>{update();if(el('download-request').disabled||!edited)return;if(downloadURL)URL.revokeObjectURL(downloadURL);downloadURL=URL.createObjectURL(new Blob([pretty(edited)+'\n'],{type:'application/json'}));const a=node('a');a.href=downloadURL;a.download='shared-exercise-request.json';a.click();});window.addEventListener('pagehide',()=>{if(downloadURL)URL.revokeObjectURL(downloadURL);downloadURL=null;});selectedRows(D.request.target_object_id||'');update();choice();
"""

_JS_FRAME = r"""let previous='',previousView='';const noEntity=[option('','No object choice for this view')],entityOptions=D.views.map(view=>[option('','Choose an object'),...view.entities.map((e,i)=>option(i,e.label||e.object_id))]);
el('frame').replaceChildren(option('','Choose a frame'),...D.rows.map((r,i)=>option(i,'Frame '+r.frame)));el('frame').value='';el('view').replaceChildren(option('','Choose a view'),option('original','Original body frames'),...D.views.map((v,i)=>option(i,v.role==='human'?'Body observations':'Equipment observations')));el('view').value='';el('entity').replaceChildren(option('','Choose a view first'));el('entity').value='';
function render(){const viewValue=el('view').value,vi=index(viewValue,D.views.length);if(viewValue!==previousView){previousView=viewValue;el('entity').replaceChildren(...(vi===null?noEntity:entityOptions[vi]));el('entity').value='';el('entity').disabled=vi===null;}
 const key=el('frame').value+'|'+viewValue+'|'+el('entity').value;if(key===previous)return;previous=key;const fi=index(el('frame').value,D.rows.length),ei=vi===null?null:index(el('entity').value,D.views[vi].entities.length);if(fi===null||(viewValue!=='original'&&vi===null)){text('frame-title','Choose a frame and view');text('availability','Inspection does not select a frame for the request.');for(const id of ['clock','body-reference','source-record','view-metadata'])text(id,'');return;}
 const row=D.rows[fi];text('frame-title','Frame '+row.frame);if(viewValue==='original'){text('view-metadata','Original body source');text('source-record',row.original?pretty(row.original):'Outside this source’s original frame coverage.');text('clock',row.original?'Original source time: '+show(row.original.nominal_time_s)+' s · '+row.original.source_status:'No original source slot.');text('body-reference',row.original?pretty(row.original.body_references):'Unavailable');text('availability',row.original&&row.original.body_references.length?'Original body reference exists; world validation is still deferred.':'No original body reference at this frame; none is interpolated.');return;}
 const view=D.views[vi],slots=row.views[vi];text('view-metadata',pretty(view));if(ei===null){text('availability','Choose an object to inspect its saved availability.');text('clock','This view uses its own source-frame clock.');text('source-record','');text('body-reference','');return;}const slot=slots&&slots[ei];if(!slot){text('availability','Outside this observation source’s decoded-frame coverage; no observation was invented.');text('clock','');text('source-record','');text('body-reference','');return;}
 text('clock','Observation time: '+show(slot.time_s)+' s · original frame: '+show(slot.original_frame));text('availability',slot.selection_eligible?'Eligible as a saved input only. Building still checks geometry and collisions.':slot.reason||'Unavailable for selection.');text('body-reference',slot.body_reference_available?'Original body reference available':'No matching original body reference');text('source-record',pretty(slot));
}
for(const id of ['frame','view','entity'])el(id).addEventListener('change',render);render();
"""


def _frame_page(rows, views, index, count):
    links = ['<a href="index.html">Configuration</a>']
    if index: links.append(f'<a href="frames-{index:03d}.html">Previous page</a>')
    if index+1 < count: links.append(f'<a href="frames-{index+2:03d}.html">Next page</a>')
    content = '<nav>' + ''.join(links) + '</nav>'
    content += f'<p>Frames {rows[0]["frame"]}–{rows[-1]["frame"]} · page {index+1} of {count}. Each view keeps its own clock. Inspecting a frame does not select it for the request.</p>'
    content += '''<section><div class="grid"><label>Frame<select id="frame"></select></label><label>View<select id="view"></select></label><label>Object<select id="entity"></select></label></div><h2 id="frame-title"></h2><p id="clock"></p><p id="availability" role="status"></p><p id="body-reference"></p><details><summary>Exact saved slot</summary><pre id="source-record"></pre></details><details><summary>Source identity and clock</summary><pre id="view-metadata"></pre></details></section><p>Image overlap is not contact. A source-eligible frame is not a validated 3D object.</p><a href="../snapshot.json">Complete saved availability</a>'''
    return _page('Inspect frames', content, {'rows':rows, 'views':views}, _JS_COMMON + _JS_FRAME)


_FORM = '''<section><h2>Choose the scene inputs</h2><p>No body evidence, equipment, frame, target or camera assumption is chosen for you. The current builder requires saved body ID <code>person-0</code>; select it explicitly. Other identities remain inspectable in the frame view.</p><div class="grid"><label>Body evidence<select id="human"></select></label><label>Camera assumption<select id="camera-mode"><option value="">Choose a camera assumption</option><option value="fixed_pelvis_perspective_prior">Fixed camera relative to the body</option></select></label><label>Camera distance (original body-model units)<input id="distance" type="number" min="0.1" max="50" step="any"></label><label>Equipment shape assumption<select id="geometry"><option value="">Choose a shape assumption</option><option value="floor_box_depth_equals_width">Floor-resting boxes; depth equals width</option><option value="floor_box_explicit_depth_ratio">Floor-resting boxes; supply each depth/width ratio</option></select></label></div><p>The box shape and camera distance are authored assumptions, not recovered measurements. Boxes rest on the unchanged floor; image overlap does not establish contact.</p><p>Optional depth ratios must be supplied for each selected object with a source or reason. Use assumed 3D depth divided by width, not the image-box aspect ratio. A ratio of 0.5 means half as deep as wide. No ratio is filled in automatically, and supplied values do not establish depth, contact or load.</p></section>
<section><h2>Equipment and target</h2><div class="grid"><label>Equipment<select id="object"></select></label><label>Equipment video frame<input id="source-frame" type="number" min="0" step="1"></label></div><p id="choice-state" role="status"></p><a id="inspect-choice" href="frames-001.html">Inspect frame availability</a><button id="add-equipment" type="button">Add equipment at this frame</button><div id="selected-equipment"></div><label>Task target<select id="target"></select></label><p>The target is one selected object, not an invented target position.</p></section>
<section><h2>Body size and assistance</h2><p>Authored defaults: unchanged size and assistance off. These options do not alter the saved body or its source evidence.</p><div class="grid"><label>Body size multiplier<input id="body-scale" type="number" min="0.5" max="2" step="any"></label><label>Body-position assistance<select id="assistance"><option value="false">Off</option><option value="true">On — programmed assistance</option></select></label></div><details><summary>Optional coordinate-frame translation</summary><p>This moves the authored coordinate frame; it does not measure motion or change the source.</p><label><input id="translation" type="checkbox"> Supply a positive XY translation</label><div class="grid"><label>X (model units)<input id="shift-x" type="number" min="0" max="6" step="any"></label><label>Y (model units)<input id="shift-y" type="number" min="0" max="6" step="any"></label></div></details></section>
<section><h2>Save your choices</h2><p id="request-state" role="status"></p><button id="download-request" type="button">Download edited request</button><p>This downloads a new file. It does not modify the saved package or start a build.</p><details><summary>Exact request</summary><pre id="request-preview"></pre></details></section>'''


def build_shared_exercise_config_report(files, *, max_seconds=5):
    """Return inspection/ member bytes; the caller preserves inputs and seals the package."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 5, 'time bound must be within five seconds')
    deadline = time.monotonic() + max_seconds
    def check():
        if time.monotonic() >= deadline: raise TimeoutError('Shared exercise configuration report deadline exceeded')
    check()
    _require(type(files) is dict and {'metadata.json', 'snapshot.json', 'request.json'} <= set(files)
             <= {'metadata.json', 'snapshot.json', 'request.json', 'config.json'} and all(type(v) is bytes for v in files.values()),
             'fixed saved-input files required')
    _require(sum(map(len, files.values())) <= MAX_INPUT, 'input bytes exceed bound')
    for name, limit in {'metadata.json':256<<10, 'snapshot.json':8<<20, 'request.json':128<<10, 'config.json':64<<10}.items():
        if name in files: _require(0 < len(files[name]) <= limit, name + ' exceeds its byte bound')
    metadata, snapshot = (_document(files[n]) for n in ('metadata.json', 'snapshot.json'))
    pins = {name:sha256(raw).hexdigest() for name, raw in files.items()}
    metadata_fields = {'schema_version', 'profile', 'status', 'snapshot_sha256', 'request_sha256', 'config_sha256',
                       'missing_fields', 'issues', 'counts', 'scope', 'world_admission', 'contact', 'grasp', 'load', 'build_options'}
    _require(set(metadata) == metadata_fields and type(metadata['schema_version']) is int and metadata['schema_version'] == 1
             and metadata['profile'] == 'shared-exercise-config-metadata-v1'
             and metadata['status'] in {'needs_input', 'config_ready', 'unavailable', 'failed'}, 'metadata schema differs')
    _require(metadata['snapshot_sha256'] == pins['snapshot.json'] and metadata['request_sha256'] == pins['request.json']
             and metadata['config_sha256'] == pins.get('config.json'), 'saved input bytes do not match metadata pins')
    _require(('config.json' in files) == (metadata['status'] == 'config_ready'), 'configuration-ready status requires exact config bytes')
    _require(metadata['world_admission'] == 'not_evaluated' and all(metadata[k] == 'unknown' for k in ('contact', 'grasp', 'load')),
             'configuration cannot claim world or contact admission')
    _require(type(metadata['missing_fields']) is list and len(metadata['missing_fields']) <= 128, 'bounded missing fields required')
    for field in metadata['missing_fields']: _text(field)
    _text(metadata['scope'], 4096); _issues(metadata['issues'])
    views = _snapshot(snapshot, check)
    _require(metadata['counts'] == snapshot['counts'], 'metadata and snapshot denominators differ')
    request_error = None
    supplied_request = _document(files['request.json']); _request_header(supplied_request, pins['snapshot.json'])
    try:
        request = _request(supplied_request, pins['snapshot.json'])
    except ValueError as error:
        _require(metadata['status'] in {'needs_input', 'failed'} and metadata['issues'], 'invalid choices require explicit saved issues')
        request, request_error = None, str(error)
    if metadata['build_options'] is not None: _options(metadata['build_options'])
    try:
        expected_options = _options(supplied_request['build_options'])
    except ValueError:
        expected_options = None
    _require(metadata['build_options'] == expected_options, 'saved build options differ')
    if metadata['status'] == 'config_ready':
        _require(request is not None and not metadata['missing_fields'] and request['human_entity_ref'] is not None
                 and request['camera'] is not None and request['geometry_prior'] is not None and request['equipment']
                 and request['target_object_id'] is not None, 'ready configuration still has missing choices')
        expected = {k:request[k] for k in ('camera', 'geometry_prior', 'equipment', 'target_object_id')}
        expected.update(schema_version=1, profile='source-equipment-proposal-config-v1', source_sha256=snapshot['source_sha256'])
        if 'frame_translation_xy' in request: expected['frame_translation_xy'] = request['frame_translation_xy']
        _require(_document(files['config.json']) == expected, 'config changes the explicit request')
        human = next(v for v in views if v['role'] == 'human'); equipment = next(v for v in views if v['role'] == 'equipment')
        chosen = request['human_entity_ref']
        _require(chosen['object_id'] == 'person-0', 'ready body identity is incompatible with the current person-0 builder')
        _require(chosen['document_sha256'] == human['document_sha256'] and any(e['object_id'] == chosen['object_id'] for e in human['entities']),
                 'ready body choice is absent from the saved inventory')
        entity_indices = {e['object_id']:i for i,e in enumerate(equipment['entities'])}
        for choice in request['equipment']:
            _require(choice['object_id'] in entity_indices and choice['source_frame'] < len(equipment['rows'])
                     and equipment['rows'][choice['source_frame']][entity_indices[choice['object_id']]]['selection_eligible'],
                     'ready equipment choice is unavailable in the saved inventory')
    original, references = snapshot['source_slots'], snapshot['body']['references']
    maximum = max(len(original), *(len(view['rows']) for view in views))
    rows = []
    for frame in range(maximum):
        check(); row = original[frame] if frame < len(original) else None
        rows.append({'frame':frame, 'original':None if row is None else {**row, 'body_references':[references[i] for i in row['body_reference_indices']]},
                     'views':[v['rows'][frame] if frame < len(v['rows']) else None for v in views]})
    pages, page, size, frame_pages = [], [], 0, []
    for row in rows:
        check(); amount = len(_encode(row))
        _require(amount <= PAGE_BYTES, 'one complete frame exceeds display page bound')
        if page and (len(page) >= PAGE_FRAMES or size+amount > PAGE_BYTES): pages.append(page); page, size = [], 0
        frame_pages.append(len(pages)); page.append(row); size += amount
    if page: pages.append(page)
    view_metadata = [{k:v for k,v in view.items() if k != 'rows'} for view in views]
    css = (Path(__file__).parent/'assets/brand.css').read_bytes()
    _require(len(css) <= 32<<10, 'brand stylesheet exceeds bound')
    output, page_index = {'brand.css':css}, []
    for i, page in enumerate(pages):
        check(); name = f'frames-{i+1:03d}.html'; output[name] = _frame_page(page, view_metadata, i, len(pages))
        page_index.append({'path':name, 'first_frame':page[0]['frame'], 'last_frame':page[-1]['frame'], 'frames':len(page)})
    titles = {'needs_input':'Choose the scene inputs', 'config_ready':'Configuration ready; world validation still required',
              'unavailable':'Saved inputs need attention', 'failed':'The configuration needs correction'}
    counts = snapshot['counts']; status = metadata['status']
    content = f'<p>{counts["body_references"]} body references · {counts["original_source_slots"]} original frame slots. No simulation has been admitted by this setup.</p>'
    if metadata['missing_fields'] or metadata['issues'] or snapshot['issues']:
        content += '<section><h2>What needs attention</h2><ul>'
        for field in metadata['missing_fields']:
            label = {'human_entity_ref':'body evidence', 'geometry_prior':'equipment shape assumption', 'equipment':'equipment and its frame', 'target_object_id':'the task target', 'build_options':'body size and assistance'}.get(field, field.replace('_', ' '))
            content += '<li>Choose ' + html.escape(label) + '.</li>'
        for issue in metadata['issues'] + snapshot['issues']: content += '<li>' + html.escape(issue['message']) + '</li>'
        content += '</ul></section>'
    content += '<section><h2>Inspect saved frames</h2><p>All original and observation slots are retained, including missing and excluded inputs. Frame inspection does not make a request choice.</p><nav>'
    content += ''.join(f'<a href="{p["path"]}">{p["first_frame"]}–{p["last_frame"]}</a>' for p in page_index) + '</nav></section>'
    editor = None
    if request is not None:
        human = next(v for v in views if v['role'] == 'human'); equipment = next(v for v in views if v['role'] == 'equipment')
        editor = {'request':request, 'humans':[{**e,'document_sha256':human['document_sha256']} for e in human['entities']],
                  'objects':equipment['entities'], 'eligibility':[[[s['original_frame'],s['selection_eligible']] for s in row] for row in equipment['rows']],
                  'frame_pages':frame_pages, 'depth_prior':{'mode':DEPTH_RATIO_PRIOR,
                      'minimum':MIN_DEPTH_TO_WIDTH_RATIO, 'maximum':MAX_DEPTH_TO_WIDTH_RATIO,
                      'maximum_provenance_characters':MAX_DEPTH_PROVENANCE_CHARS,
                      'maximum_request_bytes':MAX_CONFIG_BYTES}}
        content += _FORM
    else:
        content += '<section><h2>Correct the saved request</h2><p>' + html.escape(request_error) + '</p><p>Use the reported issue to correct a copy of <a href="../request.json">request.json</a>; no replacement values were invented.</p></section>'
    prefix = '.venv-lab/bin/python -m weft '
    resume = prefix + 'shared-exercise-config --resume PATH_TO_CONFIG_PACKAGE --request PATH_TO_DOWNLOADED_REQUEST.json --manifest-sha256 PRINTED_MANIFEST_SHA256 --out NEW_CONFIG_DIRECTORY'
    content += '<section><h2>Next step</h2><p>Replace the path placeholders and use the manifest SHA printed by the CLI. Save into a new directory; keep the sealed package unchanged.</p><pre>' + html.escape(resume) + '</pre>'
    if status == 'config_ready':
        build = prefix + 'shared-exercise-build --human-run PATH_TO_CONFIG_PACKAGE/sources/human --evidence-run PATH_TO_CONFIG_PACKAGE/sources/equipment --config PATH_TO_CONFIG_PACKAGE/config.json'
        build += ' --body-scale ' + str(metadata['build_options']['body_scale']) + ' --full-reference --out NEW_BUILD_DIRECTORY'
        for mapping in snapshot['inputs']['source_mappings']: build += ' --source-mapping PATH_TO_CONFIG_PACKAGE/' + mapping['directory']
        if metadata['build_options']['root_assistance']: build += ' --root-assistance'
        content += '<p>The saved configuration is ready. Building still checks the body, geometry and collisions; it can reject these assumptions.</p><a href="../config.json">Download the saved configuration</a><pre>' + html.escape(build) + '</pre>'
    content += '</section><details><summary>Technical identities and exact files</summary><ul>'
    for name in sorted(files): content += f'<li><a href="../{name}">{name}</a> · {pins[name]}</li>'
    content += '</ul><pre>' + html.escape(json.dumps({'counts':counts,'inputs':snapshot['inputs'],'world_admission':'not_evaluated'}, indent=2)) + '</pre></details>'
    output['index.html'] = _page(titles[status], content, editor, _JS_COMMON+_JS_EDITOR if editor is not None else '')
    summary = {'schema_version':1, 'profile':PROFILE, 'status':status, 'input_files':pins, 'asset_files':{'brand.css':sha256(css).hexdigest()},
        'counts':counts, 'pages':page_index, 'request_editor_available':editor is not None, 'source_choice_defaults':'none',
        'build_options':metadata['build_options'], 'world_admission':'not_evaluated', 'contact':'unknown', 'grasp':'unknown', 'load':'unknown',
        'source_verification':'caller_owned_not_performed_here', 'geometry_recomputed':False}
    if request is not None and request['geometry_prior'] == DEPTH_RATIO_PRIOR:
        summary['equipment_depth_priors'] = request['equipment']
        summary['depth_prior_scope'] = 'supplied_unverified_not_recovered'
    output['report.json'] = _encode(summary)+b'\n'; check()
    _require(len(output) <= MAX_FILES and sum(map(len, output.values())) <= MAX_OUTPUT, 'complete report exceeds file or byte bounds')
    return output
