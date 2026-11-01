"""Inspect bounded saved silhouette relations without loading images or admitting physics."""

from collections import Counter
from fractions import Fraction
from hashlib import sha256
import html
from io import BytesIO
import json
import math
from pathlib import Path
import re
import time


PROFILE = 'equipment-silhouette-report-v1'
EVIDENCE_PROFILE = 'equipment-silhouette-evidence-v1'
MAX_FRAMES, MAX_PAIRS, MAX_FILES = 1800, 16, 32
MAX_CONTENT, MAX_INPUT = 8 << 20, (8 << 20) + MAX_FRAMES + 1
MAX_METADATA, MAX_ROW, MAX_OUTPUT, MAX_HTML = 256 << 10, 128 << 10, 16 << 20, 2 << 20
PAGE_FRAMES, PAGE_BYTES = 100, 1 << 20
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_WRISTS = {'left_wrist': 15, 'right_wrist': 16}
_ROW = set('source_frame source_status pose_evidence_id pose_time_s pose_decoder_timestamp_s equipment_source_frame source_mapping pairs'.split())
_PAIR = set('wrist_channel_id wrist_name equipment_id human_entity_ref human_observation_id human_partition equipment_observation_id equipment_time_s equipment_partition mask_evidence_id mask_phase mask_status mask_time_s model_identity status reason pixel source_confidence source_visibility_score source_presence_score representation authored_uncertainty_px intersects distance_infimum_squared_px2 contact'.split())
_RASTER = {'raster_sha256', 'raster_dimensions', 'image_box_px'}
_MAPPING = set('original_frame derivative_frame original_decoder_time_s derivative_decoder_time_s original_native_pts derivative_native_pts original_duration_pts derivative_duration_pts pixel_sha256'.split())
_META = set('schema_version profile status source_sha256 equipment_source_sha256 equipment_document_sha256 mask_document_sha256 pose_provenance source_equivalence source_clocks construction_frames selection_policy counts work cached_row_bytes pixel_cell_mapping geometry_convention scope contact physical_noncontact load grasp physical_validation body_or_floor_installed native_calls image_decodes_in_adapter bounds'.split())


def _require(value, message):
    if not value:
        raise ValueError('Equipment silhouette report: ' + message)


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique)
        _encode(value)
    except (ValueError, RecursionError, UnicodeError) as error:
        raise ValueError('Equipment silhouette report: finite strict JSON required') from error
    _require(type(value) is dict, 'JSON object required')
    return value


def _text(value, maximum=512, *, nullable=False):
    _require(nullable and value is None or type(value) is str and 0 < len(value) <= maximum
             and all(ord(c) >= 32 for c in value), 'bounded text required')
    return value


def _integer(value, maximum=MAX_FRAMES, minimum=0):
    _require(type(value) is int and minimum <= value <= maximum, 'bounded integer required')
    return value


def _number(value, maximum=86400., *, nullable=False):
    _require(nullable and value is None or type(value) in (int, float)
             and 0 <= value <= maximum and math.isfinite(value), 'finite nonnegative saved number required')
    return value


def _pin(value):
    _require(type(value) is str and _SHA.fullmatch(value), 'exact SHA-256 required')
    return value


def _metadata(meta):
    _require(_META <= meta.keys() and set(meta) <= _META | {'canonical_frame_selection'}
             and type(meta['schema_version']) is int and meta['schema_version'] == 1
             and meta['profile'] == EVIDENCE_PROFILE and meta['status'] == 'conditional_image_evidence_only', 'unsupported evidence schema')
    for key in ('source_sha256', 'equipment_source_sha256', 'equipment_document_sha256', 'mask_document_sha256'):
        _pin(meta[key])
    _require(all(meta[k] == 'unknown' for k in ('contact', 'physical_noncontact', 'load', 'grasp'))
             and meta['physical_validation'] == 'unverified' and meta['body_or_floor_installed'] is False
             and type(meta['native_calls']) is int and meta['native_calls'] == 0
             and type(meta['image_decodes_in_adapter']) is int and meta['image_decodes_in_adapter'] == 0,
             'image evidence cannot claim physical validation or installation')
    counts = meta['counts']
    _require(type(counts) is dict and set(counts) == {'source_frames', 'decoded_frames', 'metadata_frames', 'pose_slots', 'pair_slots', 'status_counts'}, 'complete coverage counts required')
    frames = _integer(counts['source_frames'], minimum=1)
    _require(frames == max(_integer(counts['decoded_frames']), _integer(counts['metadata_frames'])), 'original source coverage differs')
    _integer(counts['pose_slots'], min(frames, 900))
    pairs = _integer(counts['pair_slots'], MAX_FRAMES * MAX_PAIRS, 1)
    _require(pairs % frames == 0 and 1 <= pairs // frames <= MAX_PAIRS, 'complete pair denominator required')
    statuses = counts['status_counts']
    _require(type(statuses) is dict and 1 <= len(statuses) <= 128, 'bounded status counts required')
    for name, count in statuses.items():
        _text(name); _integer(count, pairs, 1)
    _require(sum(statuses.values()) == pairs, 'status denominator differs')
    clocks = meta['source_clocks']
    _require(type(clocks) is dict and 2 <= len(clocks) <= 8
             and meta['equipment_document_sha256'] in clocks, 'exact source-document clocks required')
    for pin, value in clocks.items():
        _pin(pin); _require(type(value) is dict, 'saved source clock object required')
    provenance = meta['pose_provenance']
    _require(type(provenance) is dict and provenance.get('profile') == 'source-pose-surface-annotations-v1'
             and provenance.get('source_sha256') == meta['source_sha256']
             and provenance.get('uncertainty_status') == 'authored_pixel_range_not_calibrated_from_confidence'
             and provenance.get('contact_load_grasp') == 'unknown' and provenance.get('ground_or_equipment_inferred') is False,
             'source-pose identity or authored uncertainty scope differs')
    _pin(provenance.get('pose_raw_sha256')); _pin(provenance.get('pose_canonical_sha256'))
    declarations = provenance.get('channel_landmark_index_uncertainty_px')
    _require(type(declarations) is list and 1 <= len(declarations) <= 2, 'named wrist uncertainty declarations required')
    channels = {}
    for item in declarations:
        _require(type(item) is list and len(item) == 3, 'invalid wrist declaration')
        name, index, uncertainty = item
        _text(name, 80); _integer(index, 16, 15); _number(uncertainty, 64)
        _require(name not in channels and index not in [v[0] for v in channels.values()], 'duplicate wrist declaration')
        channels[name] = (index, uncertainty)
    selected = meta['construction_frames']
    if selected is not None:
        _require(type(selected) is list and len(selected) <= frames, 'bounded explicit frame selection required')
        for frame in selected: _integer(frame, frames - 1)
        _require(selected == sorted(set(selected)), 'ordered unique explicit selection required')
    proof = meta['source_equivalence']
    if meta['source_sha256'] == meta['equipment_source_sha256']:
        _require(proof is None, 'same-source inputs cannot claim unused derivative equivalence')
    else:
        _require(type(proof) is dict and proof.get('profile') == 'whole-video-source-equivalence-v1'
                 and proof.get('original_sha256') == meta['source_sha256']
                 and proof.get('derivative_sha256') == meta['equipment_source_sha256']
                 and proof.get('frame_count') == counts['decoded_frames'], 'saved derivative identity or coverage differs')
        _pin(proof.get('manifest_sha256'))
    if 'canonical_frame_selection' in meta:
        selection = meta['canonical_frame_selection']
        _require(type(selection) is dict and set(selection) == {'profile', 'frames_by_canonical_source_sha256',
                 'partition_consistency_scope', 'excluded_partition_labels', 'source_aliases', 'maximum_frames_per_source'}
                 and selection['profile'] == 'restrict-only-original-frame-selection-v1'
                 and selection['partition_consistency_scope'] == 'selected_frames_only'
                 and selection['excluded_partition_labels'] == 'preserved_not_reconciled'
                 and type(selection['maximum_frames_per_source']) is int
                 and selection['maximum_frames_per_source'] == MAX_FRAMES,
                 'selected-only partition scope or bound differs')
        frame_map = selection['frames_by_canonical_source_sha256']
        _require(selected is not None and type(frame_map) is dict and set(frame_map) == {meta['source_sha256']}
                 and type(frame_map[meta['source_sha256']]) is list
                 and frame_map[meta['source_sha256']] == selected, 'canonical frame selection differs from explicit original frames')
        for frame in frame_map[meta['source_sha256']]:
            _integer(frame, counts['decoded_frames'] - 1)
        aliases = {} if proof is None else {meta['equipment_source_sha256']: meta['source_sha256']}
        _require(type(selection['source_aliases']) is dict and selection['source_aliases'] == aliases,
                 'canonical frame selection aliases differ from original/derivative identities')
    _integer(meta['cached_row_bytes'], MAX_CONTENT)
    for key in ('selection_policy', 'pixel_cell_mapping', 'geometry_convention', 'scope'): _text(meta[key], 2048)
    _require(type(meta['work']) is dict and type(meta['bounds']) is dict, 'saved work and limits required')
    return channels


def _distance(pair):
    value = pair['distance_infimum_squared_px2']
    _require(type(value) is dict and set(value) == {'numerator', 'denominator', 'outward_interval'}, 'saved exact squared distance required')
    numerator, denominator = value['numerator'], value['denominator']
    _require(type(numerator) is int and type(denominator) is int and numerator >= 0 and denominator > 0
             and max(numerator.bit_length(), denominator.bit_length()) <= 4096, 'bounded rational distance required')
    interval = value['outward_interval']
    _require(type(interval) is list and len(interval) == 2, 'distance interval required')
    for v in interval: _number(v, 2 * 4096 ** 2)
    distance = Fraction(numerator, denominator)
    _require(Fraction(interval[0]) <= distance <= Fraction(interval[1]), 'distance interval excludes its saved exact value')
    rounded = float(distance)
    expected = [math.nextafter(rounded, -math.inf) if Fraction(rounded) > distance else rounded,
                math.nextafter(rounded, math.inf) if Fraction(rounded) < distance else rounded]
    _require(interval == expected, 'distance interval differs from outward rounding of the saved rational')
    _require(type(pair.get('touches_excluded_edge')) is bool
             and pair['touches_excluded_edge'] == (not pair['intersects'] and numerator == 0)
             and (not pair['intersects'] or numerator == 0), 'zero-distance boundary is not an intersection')


def _pair(pair, row, meta, channels):
    _require(type(pair) is dict and _PAIR <= pair.keys() and set(pair) <= _PAIR | _RASTER | {'touches_excluded_edge'}, 'unsupported pair fields')
    channel, wrist, equipment = pair['wrist_channel_id'], pair['wrist_name'], pair['equipment_id']
    _text(equipment, 80)
    _require(channel in channels and wrist in _WRISTS and channels[channel][0] == _WRISTS[wrist], 'named wrist differs from its declaration')
    ref = pair['human_entity_ref']
    _require(type(ref) is dict and set(ref) == {'document_sha256', 'object_id'}
             and ref['document_sha256'] in meta['source_clocks']
             and ref['document_sha256'] != meta['equipment_document_sha256'], 'human entity identity differs')
    _text(ref['object_id'], 120)
    for key in ('human_observation_id', 'equipment_observation_id', 'mask_evidence_id', 'human_partition', 'equipment_partition', 'mask_phase', 'mask_status', 'representation'):
        _text(pair[key], nullable=True)
    for key in ('equipment_time_s', 'mask_time_s'): _number(pair[key], nullable=True)
    for key in ('source_confidence', 'source_visibility_score', 'source_presence_score'): _number(pair[key], 1, nullable=True)
    _number(pair['authored_uncertainty_px'], 64, nullable=True)
    _require(pair['authored_uncertainty_px'] in (None, channels[channel][1]), 'pixel uncertainty differs from its authored declaration')
    pixel = pair['pixel']
    if pixel is not None:
        _require(type(pixel) is list and len(pixel) == 2 and row['source_status'] == 'decoded'
                 and pair['human_partition'] == 'construction', 'excluded human slot cannot expose a source pixel')
        for v in pixel: _number(v, 4096)
    status = pair['status']
    _require(pair['contact'] == 'unknown' and status in {'unavailable', 'conditional_image_intersection', 'conditional_image_exclusion'}, 'image relation cannot become physical contact')
    if status == 'unavailable':
        _text(pair['reason'])
        _require(pair['intersects'] is None and pair['distance_infimum_squared_px2'] is None
                 and pair['model_identity'] is None and 'touches_excluded_edge' not in pair, 'unavailable relation cannot invent distance or identity')
    else:
        _require(pair['reason'] is None and type(pair['intersects']) is bool
                 and pair['intersects'] == (status == 'conditional_image_intersection')
                 and pixel is not None and pair['authored_uncertainty_px'] is not None
                 and row['pose_evidence_id'] is not None
                 and pair['equipment_partition'] == 'construction', 'conditional image relation lacks its saved source inputs')
        identity = pair['model_identity']
        _require(type(identity) is dict and set(identity) == {'status', 'unresolved_group_id', 'gap_seen', 'physical_identity_verified'}
                 and identity['status'] in {'assumed_model_identity', 'unresolved_association'}
                 and type(identity['gap_seen']) is bool and identity['physical_identity_verified'] is False, 'model association is not physical identity')
        _text(identity['unresolved_group_id'], 160, nullable=True)
        _distance(pair)
        _require(_RASTER <= pair.keys() and pair['equipment_time_s'] == pair['mask_time_s'], 'conditional mask clock or raster identity differs')
        mapping = row['source_mapping']
        if mapping is not None:
            _require(row['pose_decoder_timestamp_s'] is not None
                     and abs(row['pose_decoder_timestamp_s'] - mapping['original_decoder_time_s']) <= 1e-9
                     and pair['mask_time_s'] is not None
                     and math.isclose(pair['mask_time_s'], mapping['derivative_decoder_time_s'], abs_tol=1e-6), 'conditional source/derivative decoder clocks differ')
    if set(pair) & _RASTER:
        _require(_RASTER <= pair.keys() and (status != 'unavailable' or pair['reason'] == 'empty_model_silhouette'), 'partial or excluded raster descriptor')
        _require(row['source_status'] == 'decoded' and row['pose_evidence_id'] is not None
                 and pair['human_partition'] == pair['equipment_partition'] == 'construction', 'excluded slot cannot expose raster geometry')
        _pin(pair['raster_sha256'])
        dims, box = pair['raster_dimensions'], pair['image_box_px']
        _require(type(dims) is list and len(dims) == 2 and type(box) is list and len(box) == 4, 'bounded raster shape required')
        for v in dims: _integer(v, 4096, 1)
        for v in box: _integer(v, 4096)
        _require(box[0] < box[2] and box[1] < box[3], 'half-open image box required')
    identity = (channel, wrist, equipment, ref['document_sha256'], ref['object_id'])
    projected = {key: pair[key] for key in ('status', 'reason', 'pixel', 'authored_uncertainty_px', 'source_confidence', 'source_visibility_score', 'source_presence_score', 'representation', 'model_identity', 'equipment_time_s', 'mask_time_s', 'intersects', 'distance_infimum_squared_px2')}
    projected['source_links'] = {key: pair[key] for key in ('human_observation_id', 'human_partition', 'equipment_observation_id', 'equipment_partition', 'mask_evidence_id', 'mask_phase', 'mask_status')}
    projected['raster'] = None if not _RASTER <= pair.keys() else {key: pair[key] for key in sorted(_RASTER)}
    projected['touches_excluded_edge'] = pair.get('touches_excluded_edge')
    return identity, projected


def _row(row, index, meta, channels, selected):
    counts = meta['counts']
    _require(set(row) == _ROW and type(row['source_frame']) is int and row['source_frame'] == index
             and row['source_status'] == ('decoded' if index < counts['decoded_frames'] else 'metadata_only_not_decoded'), 'complete original frame order/coverage required')
    _text(row['pose_evidence_id'], nullable=True)
    for key in ('pose_time_s', 'pose_decoder_timestamp_s'): _number(row[key], nullable=True)
    _require(row['pose_evidence_id'] is not None or row['pose_time_s'] is row['pose_decoder_timestamp_s'] is None, 'unsampled pose slot has invented clocks')
    mapping = row['source_mapping']
    if meta['source_equivalence'] is None:
        _require(mapping is None and type(row['equipment_source_frame']) is int and row['equipment_source_frame'] == index, 'same-source frame mapping differs')
    elif index >= counts['decoded_frames']:
        _require(mapping is None and row['equipment_source_frame'] is None, 'metadata tail cannot invent a derivative mapping')
    else:
        _require(type(mapping) is dict and set(mapping) == _MAPPING and type(mapping['original_frame']) is int
                 and mapping['original_frame'] == index and mapping['derivative_frame'] == row['equipment_source_frame'], 'exact original/derivative frame mapping required')
        _integer(mapping['derivative_frame'], counts['decoded_frames'] - 1)
        for key in ('original_decoder_time_s', 'derivative_decoder_time_s'): _number(mapping[key])
        for key in ('original_native_pts', 'derivative_native_pts', 'original_duration_pts', 'derivative_duration_pts'):
            _require(type(mapping[key]) is int and abs(mapping[key]) <= 2 ** 63 - 1, 'bounded source timestamp integer required')
        _pin(mapping['pixel_sha256'])
    pairs = row['pairs']
    _require(type(pairs) is list and len(pairs) == counts['pair_slots'] // counts['source_frames'], 'complete frame pair slots required')
    identities, projected = [], []
    common = ('metadata_only_not_decoded' if index >= counts['decoded_frames'] else
              'excluded_by_explicit_frame_selection' if selected is not None and index not in selected else
              'source_pose_not_sampled' if row['pose_evidence_id'] is None else None)
    for pair in pairs:
        identity, view = _pair(pair, row, meta, channels)
        if common is not None:
            _require(pair['status'] == 'unavailable' and pair['reason'] == common and pair['pixel'] is None,
                     'excluded/unsampled source slot or reason priority differs')
        identities.append(identity); projected.append(view)
    result = {key: row[key] for key in _ROW - {'pairs'}}
    result['pairs'] = projected
    return identities, result


_STYLE = '''body{margin:0;background:var(--weft-canvas);color:var(--weft-ink);font:16px/1.5 system-ui,sans-serif}main{max-width:1060px;margin:auto;padding:28px 20px}section{background:var(--weft-surface);border:1px solid var(--weft-border);border-radius:12px;padding:20px;margin:18px 0}a{color:var(--weft-primary)}label{display:block;margin:12px 0}select{font:inherit;padding:8px;max-width:100%;width:100%}pre,code{white-space:pre-wrap;overflow-wrap:anywhere}pre{max-height:24em;overflow:auto;background:var(--weft-canvas);padding:12px;font-size:.85em}nav{display:flex;flex-wrap:wrap;gap:16px}.notice{border-left:4px solid var(--weft-accent);padding-left:12px}.muted{color:var(--weft-muted)}:focus-visible{outline:3px solid var(--weft-ink)}li{margin:6px 0}'''
_SCRIPT = ''''use strict';const D=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id),text=(id,value)=>{el(id).textContent=value;},shown=v=>v===null?'Unknown':String(v),json=v=>JSON.stringify(v,null,2);let previous='';
function options(id,rows,label){el(id).replaceChildren(...rows.map((r,i)=>{const n=document.createElement('option');n.value=String(i);n.textContent=label(r);return n;}));el(id).value='0';}
function render(){const key=el('frame').value+':'+el('pair').value;if(key===previous)return;previous=key;const r=D.rows[Number(el('frame').value)],p=r.pairs[Number(el('pair').value)],name=D.pairs[Number(el('pair').value)];text('frame-title','Original frame '+r.source_frame+' · '+r.source_status);text('pair-title',name.wrist+' / '+name.equipment+' · channel '+name.channel);text('relation',p.status==='unavailable'?'Unavailable':p.intersects?'Conditional image overlap':'Conditional image separation');text('reason',p.reason===null?'Saved image comparison available; physical contact remains unknown.':p.reason);text('distance',p.distance_infimum_squared_px2===null?'Unknown — no distance was evaluated.':'Squared image distance (px²): '+p.distance_infimum_squared_px2.outward_interval.map(shown).join(' to '));text('boundary',p.touches_excluded_edge?'Zero distance at an excluded half-open edge is not image overlap.':'Image overlap or separation does not establish physical contact or non-contact.');text('pixel',p.pixel===null?'Unknown — no source wrist pixel supplied.':json(p.pixel)+' px');text('uncertainty',p.authored_uncertainty_px===null?'Unknown — no pixel range available.':'±'+shown(p.authored_uncertainty_px)+' px, authored; not calibrated from confidence.');text('clocks',json({original_source_frame:r.source_frame,pose_time_s:r.pose_time_s,pose_decoder_timestamp_s:r.pose_decoder_timestamp_s,equipment_source_frame:r.equipment_source_frame,equipment_time_s:p.equipment_time_s,mask_time_s:p.mask_time_s,source_mapping:r.source_mapping}));text('source-links',json({human_entity_ref:name.human_entity_ref,pose_evidence_id:r.pose_evidence_id,...p.source_links}));text('details',json({source_confidence:p.source_confidence,source_visibility_score:p.source_visibility_score,source_presence_score:p.source_presence_score,representation:p.representation,model_identity:p.model_identity,raster:p.raster,distance_infimum_squared_px2:p.distance_infimum_squared_px2}));}
options('frame',D.rows,r=>'Frame '+r.source_frame+' · '+r.source_status);options('pair',D.pairs,p=>p.wrist+' / '+p.equipment+' · '+p.channel);el('frame').addEventListener('change',render);el('pair').addEventListener('change',render);render();
'''


def _page(title, content, data=None):
    payload = '' if data is None else '<script id="data" type="application/json">' + _encode(data).decode().replace('<', '\\u003c') + '</script><script>' + _SCRIPT + '</script>'
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'self\' \'unsafe-inline\'; script-src \'unsafe-inline\'; connect-src \'none\'; base-uri \'none\'; form-action \'none\'">'
        f'<title>{html.escape(title)} · Weft</title><link rel="stylesheet" href="brand.css"><style>{_STYLE}</style></head>'
        f'<body><main><p class="muted">Weft · saved equipment image evidence</p><h1>{html.escape(title)}</h1>{content}'
        '<p class="muted">Offline saved-data inspection only. No image decode, source qualification, fitting, simulation or network request.</p></main>' + payload + '</body></html>').encode()


def _frame_page(rows, pairs, index, count, selection=None):
    links = ['<a href="index.html">Overview</a>']
    if index: links.append(f'<a href="frames-{index:03d}.html">Previous page</a>')
    if index + 1 < count: links.append(f'<a href="frames-{index+2:03d}.html">Next page</a>')
    content = '<nav>' + ''.join(links) + f'</nav><p>Page {index+1} of {count} · original frames {rows[0]["source_frame"]}–{rows[-1]["source_frame"]}</p>'
    if selection is not None:
        content += '<p class="notice">Partition consistency was checked only on the explicitly selected original frames. Excluded partition labels were preserved, not reconciled. This is not a full-source partition-consistency check.</p>'
    content += '''<p class="notice">A wrist estimate and a model silhouette are image evidence, not proof of touch. Physical contact, support, load and grasp remain unknown.</p>
<section><label>Original source frame<select id="frame"></select></label><label>Named wrist / model object<select id="pair"></select></label><h2 id="frame-title"></h2><p id="pair-title"></p><h3 id="relation"></h3><p id="reason"></p><p id="distance"></p><p id="boundary"></p></section>
<section><h2>Source wrist range</h2><p id="pixel"></p><p id="uncertainty"></p><p>Unknown is not zero. Wrist position is not a hand-contact point.</p></section>
<section><h2>Original and derivative clocks</h2><pre id="clocks"></pre><p>Pose, decoder and derivative times stay separate; no clock is inferred or retimed here.</p></section>
<section><h2>Saved evidence identities</h2><pre id="source-links"></pre><details><summary>Confidence, model association, raster identity and exact squared distance</summary><pre id="details"></pre></details><a href="equipment-frames.jsonl">Complete exact frame/pair records</a></section>'''
    return _page('Inspect wrist / object image evidence', content, {'rows': rows, 'pairs': pairs})


def build_equipment_silhouette_report(metadata_raw: bytes, frames_raw: bytes, *, max_seconds=5):
    """Return fixed local-name bytes; the caller authenticates input seals and writes output."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 5, 'time bound must be within five seconds')
    deadline = time.monotonic() + max_seconds
    def check():
        if time.monotonic() >= deadline: raise TimeoutError('Equipment silhouette report deadline exceeded')
    check()
    _require(type(metadata_raw) is bytes and type(frames_raw) is bytes and 0 < len(metadata_raw) <= MAX_METADATA
             and 0 < len(frames_raw) and len(metadata_raw) + len(frames_raw) <= MAX_INPUT, 'saved metadata/rows exceed input bounds')
    meta = _json(metadata_raw); channels = _metadata(meta); check()
    selected = None if meta['construction_frames'] is None else frozenset(meta['construction_frames'])
    stream, statuses, poses, cached, identities = BytesIO(frames_raw), Counter(), 0, 0, None
    pages, current, page_bytes, derivative_frames = [], [], 0, set()
    for index in range(meta['counts']['source_frames']):
        check(); raw = stream.readline(MAX_ROW + 2)
        _require(0 < len(raw) <= MAX_ROW + 1 and raw.strip(), 'missing or oversized original frame row')
        row = _json(raw); cached += len(_encode(row))
        keys, view = _row(row, index, meta, channels, selected)
        if identities is None:
            identities = keys
            equipment = list(dict.fromkeys(k[2] for k in keys))
            wrists = list(dict.fromkeys((k[0], k[1], k[3], k[4]) for k in keys))
            _require(1 <= len(equipment) <= 8 and len(wrists) == len(channels)
                     and keys == [(w[0], w[1], e, w[2], w[3]) for e in equipment for w in wrists], 'complete ordered wrist/object pairs required')
        _require(keys == identities, 'frame pair identities or order changed')
        statuses.update(p['reason'] or p['status'] for p in row['pairs'])
        poses += row['pose_evidence_id'] is not None
        if row['source_mapping'] is not None:
            _require(row['equipment_source_frame'] not in derivative_frames, 'duplicate derivative source frame')
            derivative_frames.add(row['equipment_source_frame'])
        size = len(_encode(view))
        if current and (len(current) >= PAGE_FRAMES or page_bytes + size > PAGE_BYTES):
            pages.append(current); current, page_bytes = [], 0
        current.append(view); page_bytes += size
    _require(not stream.read(1), 'extra source frame rows are not allowed')
    if current: pages.append(current)
    _require(dict(statuses) == meta['counts']['status_counts'] and poses == meta['counts']['pose_slots']
             and cached == meta['cached_row_bytes'], 'saved coverage/status/cache counts differ from complete rows')
    _require(cached + len(_encode(meta)) <= MAX_CONTENT, 'complete producer cache exceeds eight MiB')
    check()
    asset = Path(__file__).parent / 'assets' / 'brand.css'
    with asset.open('rb') as stream:
        css = stream.read((32 << 10) + 1)
    _require(len(css) <= 32 << 10, 'brand asset exceeds its bound'); check()
    pairs = [{'channel': c, 'wrist': w, 'equipment': e, 'human_entity_ref': {'document_sha256': d, 'object_id': o}} for c, w, e, d, o in identities]
    files = {'equipment-metadata.json': metadata_raw, 'equipment-frames.jsonl': frames_raw, 'brand.css': css}
    navigation, page_index = [], []
    for i, rows in enumerate(pages):
        check(); name = f'frames-{i+1:03d}.html'; files[name] = _frame_page(rows, pairs, i, len(pages), meta.get('canonical_frame_selection'))
        navigation.append(f'<li><a href="{name}">Original frames {rows[0]["source_frame"]}–{rows[-1]["source_frame"]}</a></li>')
        page_index.append({'path': name, 'first_source_frame': rows[0]['source_frame'], 'last_source_frame': rows[-1]['source_frame'], 'frames': len(rows)})
    summary = {'schema_version': 1, 'profile': PROFILE, 'status': 'saved_metadata_inspected', 'source_sha256': meta['source_sha256'],
        'equipment_source_sha256': meta['equipment_source_sha256'], 'counts': meta['counts'], 'pairs': pairs, 'pages': page_index,
        'input_files': {'equipment-metadata.json': sha256(metadata_raw).hexdigest(), 'equipment-frames.jsonl': sha256(frames_raw).hexdigest()},
        'asset_files': {'brand.css': sha256(css).hexdigest()}, 'source_seal_verification': 'caller_owned_not_performed_here',
        'geometry_recomputed': False, 'physical_validation': 'unverified', 'contact': 'unknown', 'support': 'unknown', 'load': 'unknown', 'grasp': 'unknown',
        'scope': 'Saved source-image relations only; no source acquisition, raster validation, contact qualification, body/floor installation or simulator admission.'}
    if 'canonical_frame_selection' in meta:
        summary['canonical_frame_selection'] = meta['canonical_frame_selection']
    files['report.json'] = _encode(summary) + b'\n'
    esc = lambda value: html.escape(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False))
    content = '<p class="notice">Image overlap is not physical contact. Image separation is not verified physical non-contact. Support, load and grasp remain unknown.</p>'
    content += f'<section><h2>Complete original coverage</h2><pre>{esc(meta["counts"])}</pre><p>Every original frame and named pair is retained, including unsampled, excluded and metadata-only slots. Raw reason priority is preserved; exclusion is not reclassified as missing sampling.</p><ul>{"".join(navigation)}</ul></section>'
    if 'canonical_frame_selection' in meta:
        content += '<section><h2>Selected-frame consistency only</h2><p class="notice">Partition consistency was checked only on the explicitly selected original frames. Excluded partition labels were preserved, not reconciled. This is not a full-source partition-consistency check.</p>' + f'<pre>{esc(meta["canonical_frame_selection"])}</pre></section>'
    content += f'<section><h2>Selected wrists and model objects</h2><pre>{esc(pairs)}</pre><p>Model object IDs are not authenticated physical identity. No hand contact or ground/support geometry is inferred.</p></section>'
    content += f'<section><h2>Source clocks and authored uncertainty</h2><details><summary>Exact saved source/derivative clocks and pose provenance</summary><pre>{esc({k: meta[k] for k in ("source_sha256", "equipment_source_sha256", "source_clocks", "source_equivalence", "pose_provenance", "construction_frames")})}</pre></details><p>Pixel uncertainty is an authored image range, not a confidence-calibrated physical error. Distance is squared image distance in px²; zero distance at an excluded half-open cell edge does not imply image overlap.</p></section>'
    content += '<section><h2>Exact saved records</h2><ul><li><a href="equipment-metadata.json">Original evidence metadata</a></li><li><a href="equipment-frames.jsonl">Every original frame/pair record</a></li><li><a href="report.json">Inspector coverage and byte identities</a></li></ul><p>The caller verifies the selected input seal. This inspector checks saved joins/counts/arithmetic, not original rasters or source authenticity.</p></section>'
    files['index.html'] = _page('Equipment image evidence', content)
    check()
    _require(len(files) <= MAX_FILES and sum(map(len, files.values())) <= MAX_OUTPUT
             and all(len(raw) <= MAX_HTML for name, raw in files.items() if name.endswith('.html')), 'complete offline output exceeds its bound; no slots dropped')
    return files
