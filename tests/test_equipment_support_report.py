"""Saved-data arithmetic, complete coverage and bounded scripted navigation."""

import builtins
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import socket
import subprocess
import time

import pytest

from weft import equipment_support_report as report


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def canonical(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@pytest.fixture(autouse=True)
def saved_only(monkeypatch):
    imported = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.startswith(('numpy', 'mujoco', 'cv2', 'gymnasium', 'weft.')):
            raise AssertionError('No native or evidence-producing imports in saved inspection')
        return imported(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: pytest.fail('network'))
    popen = subprocess.Popen
    def process(args, *a, **k):
        assert type(args) is list and args[0] == 'node', 'Only scripted local DOM checks may execute'
        return popen(args, *a, **k)
    monkeypatch.setattr(subprocess, 'Popen', process)


def fixture(count=10, objects=2, *, derivative=False):
    """Authored saved selector records, not recovered video or independently sealed history."""
    source, equipment = 'a' * 64, ('b' if derivative else 'a') * 64
    cutoff = int(.6 * count); decoded = min(1800, count * 2 + 1); declared = min(1800, decoded + 1)
    keys = [(f'equipment-{obj}', side + '_wrist') for obj in range(objects) for side in ('left', 'right')]
    rows, ground_counts, baseline_counts = [], Counter(), Counter()
    def identity(i): return f'{source}:frame:{i*2}'
    for i in range(count):
        current = i < cutoff
        row = {'id': identity(i), 'frame': i * 2, 'time_s': i * 2 / 120, 'decoder_timestamp_s': i * 2 / 120 + .002,
               'partition': 'construction' if current else 'excluded_later', 'ground_points': [], 'equipment_pairs': []}
        if derivative:
            row['equipment_source_mapping'] = {'original_frame': i * 2, 'derivative_frame': i * 2,
                'original_decoder_time_s': i * 2 / 120 + .002, 'derivative_decoder_time_s': i * 2 / 120 + .004,
                'original_native_pts': i * 20, 'derivative_native_pts': i * 20 + 2,
                'original_duration_pts': 10, 'derivative_duration_pts': 10, 'pixel_sha256': 'f' * 64}
        for index, name in ((29, 'left_heel'), (30, 'right_heel'), (31, 'left_foot_index'), (32, 'right_foot_index')):
            status = 'unavailable' if current else 'excluded_later_partition'
            row['ground_points'].append({'index': index, 'name': name, 'status': status, 'pixel': None,
                                         'confidence': None, 'verified_contact': False})
            if current: ground_counts[status] += 1
        if current:
            left, last = max(0, i - 12), min(cutoff - 1, i + 12)
            row['landmark_origin'] = 'authored_analytical_fixture'
            row['window'] = {'first_evidence_id': identity(left), 'last_evidence_id': identity(last),
                'samples': last - left + 1, 'span_seconds': last * 2 / 120 - left * 2 / 120, 'reasons': []}
        for j, (obj, hand) in enumerate(keys):
            status = ('image_separation' if i == 3 else 'projected_co_motion_candidate') if current else 'excluded_later_partition'
            pair = {'object_id': obj, 'hand': hand, 'status': status, 'verified_contact': False, 'grasp': 'unknown'}
            if current:
                pair.update(reasons=[] if status == 'projected_co_motion_candidate' else ['outside_authored_window'],
                    window_missing_points=0, window_image_bounds_px=[1., 2., 5., 6.], range_torso_fraction=.01,
                    object_observation_id=f'{obj}-{i*2}', object_evidence_ids=[f'mask-{obj}-{i*2}'],
                    object_source_frame=i*2, object_time_s=i*2/120+.004, equipment_slot_reason=None,
                    hand_image_confidence=.8, image_box_px=[1., 2., 5., 6.], hand_pixel=[3., 4.], image_gap_px=0.,
                    maximum_window_gap_torso_fraction=0.)
                baseline_counts[status] += 1
            row['equipment_pairs'].append(pair)
        rows.append(row)
    result = {'schema_version': 1, 'profile': 'exercise-support-evidence-v1', 'status': 'candidates_only',
        'source_sha256': source, 'pose_evidence_canonical_sha256': 'c' * 64,
        'equipment_observations_sha256': 'd' * 64, 'equipment_source_sha256': equipment,
        'equipment_object_ids': [f'equipment-{i}' for i in range(objects)],
        'equipment_source_equivalence': None if not derivative else {'profile': 'whole-video-source-equivalence-v1',
            'original_sha256': source, 'derivative_sha256': equipment, 'manifest_sha256': 'e' * 64,
            'limitations': ['Authored test proof metadata; not independently authenticated history']},
        'camera_prior': {'mode': 'fixed_source_camera', 'status': 'authored_unverified', 'source_sha256': source,
                         'rationale': 'Explicit fixed-camera assumption for saved analytical records'},
        'contract': {'generation_fraction': .6, 'scope': 'Image-only co-motion candidates, contact unknown'},
        'rows': rows, 'image_scale': {'median_torso_span_px': 80., 'usable_construction_rows': cutoff,
                                    'total_construction_rows': cutoff, 'scope': 'Image normalization, not metric scale'},
        'counts': {'supplied_pose_rows': count, 'construction_rows': cutoff, 'excluded_later_rows': count-cutoff,
                   'ground_point_slots': cutoff*4, 'retained_ground_point_slots': count*4,
                   'ground_status_counts': dict(ground_counts), 'equipment_pair_slots': cutoff*len(keys),
                   'retained_equipment_pair_slots': count*len(keys), 'equipment_status_counts': dict(baseline_counts)},
        'coverage': {'declared_frame_count': declared, 'decoded_frames': decoded,
            'unsampled_decoded_frame_indices': [i for i in range(decoded) if i not in {r['frame'] for r in rows}],
            'container_only_frame_indices': list(range(decoded, declared)),
            'decoded_beyond_container_metadata_indices': list(range(declared, decoded))},
        'contact_state': 'unknown', 'ground_height': None, 'support_force': None, 'promoted': False,
        'physical_validation': 'unverified', 'limits': ['No source, native physics or raster work in this authored fixture']}
    baseline_hash = canonical(result)
    points, point_counts = [], Counter()
    for i in range(cutoff):
        pairs = []
        for j, (obj, hand) in enumerate(keys):
            status = 'exclusion' if i == 0 and j == 1 else 'unavailable' if i == 0 and j == 2 else 'intersection'
            pairs.append({'object_id': obj, 'hand': hand, 'status': status,
                'reason': 'missing_saved_image_point' if status == 'unavailable' else None,
                'mask_evidence_id': None if status == 'unavailable' else f'mask-{obj}-{i*2}'})
            point_counts[status] += 1
        points.append({'source_frame': i*2, 'pose_evidence_id': identity(i), 'pairs': pairs})
    decisions, active = Counter(), Counter()
    for i, row in enumerate(rows):
        for j, pair in enumerate(row['equipment_pairs']):
            before = pair['status']
            decision = {'status': 'excluded_later' if i >= cutoff else 'not_candidate', 'reason': before,
                        'window_point_range': None, 'window_image_counts': None}
            if before == 'projected_co_motion_candidate':
                left, right = max(0, i-12), min(cutoff, i+13)
                counts = {name: sum(points[k]['pairs'][j]['status'] == name for k in range(left, right))
                          for name in ('intersection', 'exclusion', 'unavailable')}
                status = 'unavailable' if counts['unavailable'] else 'excluded' if counts['exclusion'] else 'kept'
                decision.update(status=status, reason={'kept': None, 'excluded': 'conditional_image_exclusion_in_original_window',
                    'unavailable': 'required_window_image_point_unavailable'}[status], window_point_range=[left, right], window_image_counts=counts)
                pair['status'] = {'kept': before, 'excluded': 'image_silhouette_exclusion', 'unavailable': 'unavailable'}[status]
            pair.update(co_motion_baseline_status=before, silhouette_constraint=decision)
            decisions[decision['status']] += 1
            if i < cutoff: active[pair['status']] += 1
    result['counts']['equipment_status_counts'] = dict(active)
    candidates = sum(decisions[k] for k in ('kept', 'excluded', 'unavailable'))
    point_bytes = sum(len(json.dumps(p, sort_keys=True, allow_nan=False).encode()) for p in points)
    entries = (cutoff+1)*len(keys)*3
    result['equipment_silhouette_constraint'] = {'schema_version': 1, 'profile': report.CONSTRAINT_PROFILE,
        'status': 'applied_restrict_only', 'baseline_profile': result['profile'], 'baseline_canonical_sha256': baseline_hash,
        'baseline_equipment_status_counts': dict(baseline_counts), 'input_bindings': {
            'source_sha256': source, 'equipment_source_sha256': equipment, 'pose_evidence_canonical_sha256': 'c'*64,
            'pose_raw_sha256': '1'*64, 'equipment_document_sha256': 'd'*64, 'mask_document_sha256': '2'*64,
            'annotation_document_sha256': '3'*64, 'observation_document_sha256s': ['4'*64],
            'silhouette_metadata_sha256': '5'*64, 'silhouette_rows_length_prefixed_sha256': '6'*64,
            'rows_digest_encoding': 'Each original cached row prefixed by its unsigned 8-byte big-endian byte length; not a source-file hash',
            'source_equivalence_manifest_sha256': 'e'*64 if derivative else None},
        'construction_frames': [r['frame'] for r in rows[:cutoff]], 'point_rows': points,
        'counts': {'source_frames': max(decoded, declared), 'construction_pair_slots': cutoff*len(keys),
            'retained_pose_pair_slots': count*len(keys), 'baseline_candidates': candidates, 'kept_candidates': decisions['kept'],
            'excluded_candidates': decisions['excluded'], 'unavailable_candidates': decisions['unavailable'],
            'unchanged_construction_noncandidates': decisions['not_candidate'], 'excluded_later_pose_pairs': decisions['excluded_later'],
            'point_status_counts': dict(point_counts)},
        'work': {'selected_frame_queries': cutoff, 'point_slots': cutoff*len(keys), 'window_queries': candidates,
            'serialized_point_cache_bytes': point_bytes, 'prefix_counter_entries': entries, 'prefix_counter_buffer_bytes': entries*4,
            'prepared_row_bytes': cutoff*len(keys)*900, 'raster_calls': 0, 'pose_annotation_queries': 0},
        'bounds': {'cooperative_seconds': 5., 'within_original_deadline': True, 'additional_cache_bytes': 2<<20,
            'complete_output_bytes': 8<<20, 'construction_poses': 540, 'equipment_ids': 8, 'wrists': 2},
        'contact': 'unknown', 'physical_noncontact': 'unknown', 'grasp': 'unknown', 'load': 'unknown',
        'physical_validation': 'unverified', 'body_or_floor_installed': False,
        'scope': 'Saved analytical image constraint, not physical support', 'baseline_recovery': 'Restore original statuses and counts'}
    return result


class Page(HTMLParser):
    def __init__(self, value):
        super().__init__(); self.links, self.scripts, self.inside = [], [], None
        self.feed(value.decode())
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for name in ('href', 'src'):
            if name in attrs: self.links.append(attrs[name])
        if tag == 'script': self.inside = []; self.scripts.append(self.inside)
    def handle_data(self, value):
        if self.inside is not None: self.inside.append(value)
    def handle_endtag(self, tag):
        if tag == 'script': self.inside = None
    @property
    def data(self): return json.loads(''.join(self.scripts[0]))


def test_exact_baseline_decisions_clocks_and_complete_links_are_preserved():
    value = fixture(derivative=True); original = raw(value)+b'\n'
    output = report.build_equipment_support_report(original)
    assert output['support-evidence.json'] == original
    data = Page(output['frames-001.html']).data
    assert [r['source_frame'] for r in data['rows']] == list(range(22))
    assert data['rows'][1]['source_status'] == 'unsampled_decoded_frame' and data['rows'][1]['pose'] is None
    assert data['rows'][21]['source_status'] == 'metadata_only_not_decoded'
    assert data['rows'][0]['pose']['equipment_source_mapping']['original_decoder_time_s'] == .002
    assert data['rows'][0]['pose']['equipment_source_mapping']['derivative_decoder_time_s'] == .004
    summary = json.loads(output['report.json'])
    assert all(summary['constraint_counts'][k] > 0 for k in ('kept_candidates', 'excluded_candidates', 'unavailable_candidates'))
    assert summary['original_counts']['equipment_status_counts'] != summary['active_counts']['equipment_status_counts']
    assert summary['baseline_reconstruction_verified'] and summary['geometry_recomputed'] is False
    assert all(summary[k] == 'unknown' for k in ('contact', 'physical_noncontact', 'support', 'grasp', 'load'))
    assert summary['asset_files']['brand.css'] == sha256(output['brand.css']).hexdigest()
    for name, value in output.items():
        if name.endswith('.html'): assert all(link.split('#')[0] in output for link in Page(value).links)
    window_index = json.loads(output['window-index.js'].decode().removeprefix('window.WeftWindowIndex=').removesuffix(';\n'))
    assert [r['source_frame'] for r in window_index] == list(range(0, 12, 2))
    assert all(r['page'] in output for r in window_index)


@pytest.mark.parametrize('fault', ['baseline_hash', 'source_pin', 'missing_pose', 'missing_point', 'pair_order',
    'window_range', 'window_counts', 'unavailable_precedence', 'active_counts', 'later_window', 'physical_claim', 'work_count'])
def test_contradictory_saved_decisions_reject(fault):
    value = fixture(); c = value['equipment_silhouette_constraint']; p = value['rows'][0]['equipment_pairs'][0]
    if fault == 'baseline_hash': c['baseline_canonical_sha256'] = '0'*64
    elif fault == 'source_pin': c['input_bindings']['source_sha256'] = '0'*64
    elif fault == 'missing_pose': value['rows'].pop()
    elif fault == 'missing_point': c['point_rows'].pop()
    elif fault == 'pair_order': c['point_rows'][0]['pairs'].reverse()
    elif fault == 'window_range': p['silhouette_constraint']['window_point_range'][1] -= 1
    elif fault == 'window_counts': p['silhouette_constraint']['window_image_counts']['intersection'] -= 1
    elif fault == 'unavailable_precedence': value['rows'][0]['equipment_pairs'][2]['silhouette_constraint']['status'] = 'excluded'
    elif fault == 'active_counts': value['counts']['equipment_status_counts']['image_silhouette_exclusion'] += 1
    elif fault == 'later_window': value['rows'][-1]['window'] = deepcopy(value['rows'][0]['window'])
    elif fault == 'physical_claim': c['physical_noncontact'] = 'verified'
    elif fault == 'work_count': c['work']['window_queries'] += 1
    with pytest.raises(ValueError): report.build_equipment_support_report(raw(value))


def test_strict_input_and_expired_budget_reject_before_asset_read(monkeypatch):
    valid = raw(fixture())
    monkeypatch.setattr(report.Path, 'open', lambda *a, **k: pytest.fail('premature asset read'))
    for value in (valid.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1', 1),
                  valid.replace(b'"raster_calls":0', b'"raster_calls":NaN'), b'x'*(report.MAX_INPUT+1)):
        with pytest.raises(ValueError): report.build_equipment_support_report(value)
    ticks = iter([0., 1.]); monkeypatch.setattr(report.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError): report.build_equipment_support_report(valid, max_seconds=.1)


@pytest.mark.parametrize('limit', ['MAX_HTML', 'MAX_OUTPUT', 'MAX_FILES'])
def test_output_limits_reject_without_dropping_evidence(monkeypatch, limit):
    monkeypatch.setattr(report, limit, 1)
    with pytest.raises(ValueError, match='bound'): report.build_equipment_support_report(raw(fixture()))


_DOM = r'''
import fs from'node:fs';import vm from'node:vm';import assert from'node:assert/strict';import{performance}from'node:perf_hooks';
const html=fs.readFileSync(process.argv[2],'utf8'),index=fs.readFileSync(process.argv[3],'utf8'),scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];let created=0,writes=0;
class E{constructor(){this.children=[];this.listeners=new Map();this.value='';this.own='';this.hidden=false;created++;}set textContent(v){writes++;this.own=String(v);this.children=[];}get textContent(){return this.own;}set innerHTML(v){assert.fail('unsafe markup');}appendChild(v){this.children.push(v);}addEventListener(k,f){this.listeners.set(k,f);}fire(k){this.listeners.get(k)?.();}}
const ids=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(v=>[v[1],new E()])),el=id=>{assert(ids.has(id),id);return ids.get(id)};el('data').textContent=scripts[0][1];
const listeners=new Map(),location={hash:''},window={addEventListener:(k,f)=>listeners.set(k,f)},context={window,location,document:{getElementById:el,createElement:()=>new E()},fetch:()=>assert.fail('network'),setTimeout:()=>assert.fail('timer')};
vm.runInNewContext(index,context,{timeout:1000});vm.runInNewContext(scripts[2][1],context,{timeout:1000});
'''


def dom(tmp_path, output, assertions, page='frames-001.html'):
    path = tmp_path/'page.html'; path.write_bytes(output[page]); index = tmp_path/'index.js'; index.write_bytes(output['window-index.js'])
    result = subprocess.run(['node', '--input-type=module', '-', str(path), str(index)], input=_DOM+assertions,
                            text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout) if result.stdout else None


def test_scripted_selected_windows_unknowns_safe_text_and_same_page_links(tmp_path):
    value = fixture()
    value['rows'][0]['equipment_pairs'][0]['object_observation_id'] = '</script><img src=x onerror=alert(1)>'
    baseline = deepcopy(value); extra = baseline.pop('equipment_silhouette_constraint')
    baseline['counts']['equipment_status_counts'] = extra['baseline_equipment_status_counts']
    for row in baseline['rows']:
        for pair in row['equipment_pairs']:
            pair['status'] = pair.pop('co_motion_baseline_status'); pair.pop('silhouette_constraint')
    value['equipment_silhouette_constraint']['baseline_canonical_sha256'] = canonical(baseline)
    output = report.build_equipment_support_report(raw(value))
    assert b'<img src=x' not in output['frames-001.html']
    dom(tmp_path, output, r'''
assert.match(el('baseline').textContent,/<\/script><img/);assert.match(el('decision').textContent,/"status": "kept"/);el('pair').value='1';el('pair').fire('change');assert.match(el('decision').textContent,/"status": "excluded"/);el('pair').value='2';el('pair').fire('change');assert.match(el('decision').textContent,/"status": "unavailable"/);
assert.equal(el('window-links').children.filter(n=>!n.hidden).length,6);assert.match(el('window-links').children[0].children[0].href,/#frame=0&pair=2$/);
location.hash='#frame=2&pair=0';listeners.get('hashchange')();assert.equal(el('frame').value,'2');assert.equal(el('pair').value,'0');assert.match(el('clock').textContent,/"source_frame": 2/);
el('frame').value='1';el('frame').fire('change');assert.match(el('decision').textContent,/No window or pair value was invented/);assert.equal(el('window-links').children.filter(n=>!n.hidden).length,0);
el('frame').value='18';el('frame').fire('change');assert.match(el('decision').textContent,/excluded_later/);assert.match(el('point').textContent,/later pose partition/);
''')


def test_maximum_complete_coverage_and_bounded_repeated_navigation(tmp_path, record_property):
    oversized = raw(fixture(count=900, objects=8))
    with pytest.raises(ValueError, match='eight MiB'): report.build_equipment_support_report(oversized)
    source = raw(fixture(count=900, objects=6)); began = time.perf_counter()
    output = report.build_equipment_support_report(source); elapsed = time.perf_counter()-began
    summary = json.loads(output['report.json'])
    assert summary['complete_source_frames'] == 1800 and summary['active_counts']['retained_equipment_pair_slots'] == 10800
    assert sum(row['frames'] for row in summary['pages']) == 1800
    assert len(output) <= 32 and sum(map(len, output.values())) <= 16<<20
    measured = dom(tmp_path, output, r'''
assert.equal(el('frame').children.length,100);assert.equal(el('pair').children.length,12);
el('frame').value='24';el('frame').fire('change');assert.equal(el('window-links').children.filter(n=>!n.hidden).length,20);el('window-page').value='1';el('window-page').fire('change');assert.equal(el('window-links').children.filter(n=>!n.hidden).length,5);
const count=created,start=performance.now();for(let i=0;i<1000;i++){el('frame').value=String(i%100);el('pair').value=String(i%12);el('frame').fire('change');}const ms=performance.now()-start;assert.equal(created,count);const w=writes;for(let i=0;i<1000;i++)el('pair').fire('change');assert.equal(writes,w);
process.stdout.write(JSON.stringify({updates:1000,milliseconds:ms,new_nodes:created-count,repeated_unchanged_updates:1000,unchanged_writes:writes-w}));
''')
    record_property('bounded_saved_report', json.dumps({'source_frames': 1800, 'retained_pose_pairs': 10800,
        'rejected_oversized_fixture_bytes': len(oversized), 'input_bytes': len(source), 'output_bytes': sum(map(len, output.values())), 'files': len(output),
        'maximum_html_bytes': max(len(v) for k,v in output.items() if k.endswith('.html')),
        'prepare_seconds': elapsed, 'scripted_dom': measured,
        'scope': 'Authored saved records and Node mock DOM only; not browser, source, native or RSS evidence'}, sort_keys=True))


def test_all_sixteen_named_pairs_remain_in_a_legal_width_fixture():
    output = report.build_equipment_support_report(raw(fixture(objects=8)))
    page = Page(output['frames-001.html']).data
    assert len(page['pairs']) == 16
    assert all(len(row['pose']['equipment_pairs']) == 16 for row in page['rows'] if row['pose'] is not None)
    assert json.loads(output['report.json'])['active_counts']['retained_equipment_pair_slots'] == 160


def test_preserved_actual_analytical_producer_payload_without_rerun(record_property):
    path = Path('.local/equipment-silhouette-constraint-analytic-01/support-evidence.json')
    if not path.is_file(): pytest.skip('Optional preserved local producer payload is not shipped with tests')
    with path.open('rb') as stream: source = stream.read((64<<10)+1)
    assert len(source) == 52979 and sha256(source).hexdigest() == 'ce3c8a7055cb8b23df8aff5e719493146cb9a146f30c60047872f03afb46ece9'
    output = report.build_equipment_support_report(source)
    summary = json.loads(output['report.json'])
    assert output['support-evidence.json'] == source and summary['complete_source_frames'] == 31
    assert summary['constraint_counts']['baseline_candidates'] == summary['constraint_counts']['kept_candidates'] == 18
    assert summary['original_counts']['supplied_pose_rows'] == 15 and summary['original_counts']['construction_rows'] == 9
    record_property('saved_producer_schema', json.dumps({'sha256': sha256(source).hexdigest(), 'bytes': len(source),
        'source_frames': 31, 'pose_rows': 15, 'construction_rows': 9, 'kept_candidates': 18,
        'exact_download_retained': True, 'producer_rerun': False, 'scope': 'One already-saved analytical producer result; no source/native work'}))
