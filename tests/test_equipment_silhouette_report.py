"""Saved metadata inspection, plus one tiny analytical producer/schema composition."""

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

from weft import equipment_silhouette_report as report


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


@pytest.fixture(autouse=True)
def metadata_only(monkeypatch):
    imported = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.startswith(('numpy', 'mujoco', 'cv2', 'gymnasium')) or name.startswith('weft.'):
            raise AssertionError('Inspector cannot import native/source/evidence-producing code')
        return imported(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: pytest.fail('network'))
    popen = subprocess.Popen
    def process(args, *a, **k):
        assert type(args) is list and args[0] == 'node', 'Only the local scripted DOM is allowed'
        return popen(args, *a, **k)
    monkeypatch.setattr(subprocess, 'Popen', process)
    return imported


def fixture(count=7, objects=1, *, derivative=True):
    source, equipment_source, human_doc, equipment_doc = 'a' * 64, ('b' if derivative else 'a') * 64, 'c' * 64, 'd' * 64
    decoded = count - 1 if count < 1800 else count - 2
    selected = list(range(0, min(decoded, 501), 2))
    rows = []
    for frame in range(count):
        sampled = frame < decoded and frame % 2 == 0
        stamp = frame / 30
        reason = ('metadata_only_not_decoded' if frame >= decoded else
                  'excluded_by_explicit_frame_selection' if frame not in selected else None)
        pairs = []
        for object_index in range(objects):
            for side in ('left', 'right'):
                available = reason is None
                intersection = frame % 6 == 4
                zero_edge = frame % 6 == 2
                relation = 'conditional_image_intersection' if intersection else 'conditional_image_exclusion'
                pair = {'wrist_channel_id': side, 'wrist_name': side + '_wrist', 'equipment_id': f'object-{object_index}',
                    'human_entity_ref': {'document_sha256': human_doc, 'object_id': 'human-0'},
                    'human_observation_id': f'human-{frame}' if frame < decoded else None,
                    'human_partition': 'construction' if frame < decoded else None,
                    'equipment_observation_id': f'object-{object_index}-{frame}' if frame < decoded else None,
                    'equipment_time_s': stamp + .002 if frame < decoded else None,
                    'equipment_partition': 'construction' if frame < decoded else None,
                    'mask_evidence_id': f'mask-{object_index}-{frame}' if frame < decoded else None,
                    'mask_phase': 'post_initialization' if frame < decoded else None,
                    'mask_status': 'decoded' if frame < decoded else None,
                    'mask_time_s': stamp + .002 if frame < decoded else None,
                    'model_identity': {'status': 'assumed_model_identity', 'unresolved_group_id': None,
                        'gap_seen': False, 'physical_identity_verified': False} if available else None,
                    'status': relation if available else 'unavailable', 'reason': reason,
                    'pixel': [35., 33.] if available else None,
                    'source_confidence': .8 if available else None,
                    'source_visibility_score': .9 if available else None,
                    'source_presence_score': .8 if available else None,
                    'representation': 'declared_raw_model_estimate_not_authenticated_identity' if available else None,
                    'authored_uncertainty_px': .25 if available else None,
                    'intersects': intersection if available else None,
                    'distance_infimum_squared_px2': None, 'contact': 'unknown'}
                if available:
                    q = 0 if intersection or zero_edge else 9
                    pair.update(touches_excluded_edge=zero_edge,
                        distance_infimum_squared_px2={'numerator': q, 'denominator': 16, 'outward_interval': [q / 16, q / 16]},
                        raster_sha256='e' * 64, raster_dimensions=[3, 3], image_box_px=[32, 30, 38, 36])
                pairs.append(pair)
        mapping = None if not derivative or frame >= decoded else {'original_frame': frame, 'derivative_frame': frame,
            'original_decoder_time_s': stamp, 'derivative_decoder_time_s': stamp + .002,
            'original_native_pts': frame * 33 + 3, 'derivative_native_pts': frame * 33 + 5,
            'original_duration_pts': 33, 'derivative_duration_pts': 33, 'pixel_sha256': 'f' * 64}
        rows.append({'source_frame': frame, 'source_status': 'decoded' if frame < decoded else 'metadata_only_not_decoded',
            'pose_evidence_id': f'pose-{frame}' if sampled else None, 'pose_time_s': stamp if sampled else None,
            'pose_decoder_timestamp_s': stamp if sampled else None,
            'equipment_source_frame': frame if not derivative or frame < decoded else None,
            'source_mapping': mapping, 'pairs': pairs})
    provenance = {'profile': 'source-pose-surface-annotations-v1', 'pose_raw_sha256': '1' * 64,
        'pose_canonical_sha256': '2' * 64, 'source_sha256': source, 'coordinate_policy': 'Authored analytical saved rows, not source admission',
        'uncertainty_status': 'authored_pixel_range_not_calibrated_from_confidence',
        'uncertainty_method': 'authored', 'uncertainty_note': 'Analytical range, not a measurement',
        'channel_landmark_index_uncertainty_px': [['left', 15, .25], ['right', 16, .25]],
        'contact_load_grasp': 'unknown', 'ground_or_equipment_inferred': False}
    metadata = {'schema_version': 1, 'profile': report.EVIDENCE_PROFILE, 'status': 'conditional_image_evidence_only',
        'source_sha256': source, 'equipment_source_sha256': equipment_source, 'equipment_document_sha256': equipment_doc,
        'mask_document_sha256': '3' * 64, 'pose_provenance': provenance,
        'source_equivalence': {'profile': 'whole-video-source-equivalence-v1', 'original_sha256': source,
            'derivative_sha256': equipment_source, 'manifest_sha256': '4' * 64, 'frame_count': decoded,
            'time_base': '1/1000', 'limitations': ['Authored metadata only; no source verification']} if derivative else None,
        'source_clocks': {human_doc: {'basis': 'pose_source_time', 'fps': 30.}, equipment_doc: {'basis': 'decoder_time', 'fps': 30.}},
        'construction_frames': selected, 'selection_policy': 'Explicit construction frames; exclusion precedes sampling status',
        'counts': {'source_frames': count, 'decoded_frames': decoded, 'metadata_frames': count},
        'work': {'pose_frame_queries': len(selected), 'raster_loads': len(selected) * objects,
            'binary_raster_bytes': len(selected) * objects * 9, 'foreground_runs': len(selected) * objects * 4,
            'pair_relations': len(selected) * objects * 2},
        'cached_row_bytes': 0, 'pixel_cell_mapping': 'Half-open image cells',
        'geometry_convention': 'Closed wrist rectangle versus half-open model cells; zero distance does not imply intersection',
        'scope': 'Analytical saved fixture; no source, PNG, native or inference run', 'contact': 'unknown',
        'physical_noncontact': 'unknown', 'load': 'unknown', 'grasp': 'unknown', 'physical_validation': 'unverified',
        'body_or_floor_installed': False, 'native_calls': 0, 'image_decodes_in_adapter': 0,
        'bounds': {'result_cache_bytes': 8 << 20}}
    return metadata, rows


def encode(metadata, rows):
    metadata = deepcopy(metadata)
    encoded = [raw(row) for row in rows]
    metadata['cached_row_bytes'] = sum(map(len, encoded))
    metadata['counts'].update(pose_slots=sum(r['pose_evidence_id'] is not None for r in rows),
        pair_slots=sum(len(r['pairs']) for r in rows),
        status_counts=dict(Counter(p['reason'] or p['status'] for r in rows for p in r['pairs'])))
    return raw(metadata), b''.join(row + b'\n' for row in encoded)


class Page(HTMLParser):
    def __init__(self, raw):
        super().__init__(); self.links, self.scripts, self.inside = [], [], None
        self.feed(raw.decode())
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'href' in attrs: self.links.append(attrs['href'])
        if tag == 'script': self.inside = []; self.scripts.append(self.inside)
    def handle_data(self, data):
        if self.inside is not None: self.inside.append(data)
    def handle_endtag(self, tag):
        if tag == 'script': self.inside = None
    @property
    def data(self): return json.loads(''.join(self.scripts[0]))


def test_complete_exact_slots_links_and_conditional_boundary_are_preserved():
    inputs = encode(*fixture())
    output = report.build_equipment_silhouette_report(*inputs)
    assert output['equipment-metadata.json'] == inputs[0] and output['equipment-frames.jsonl'] == inputs[1]
    data = Page(output['frames-001.html']).data
    assert [r['source_frame'] for r in data['rows']] == list(range(7))
    assert data['rows'][1]['pose_evidence_id'] is None
    assert data['rows'][1]['pairs'][0]['reason'] == 'excluded_by_explicit_frame_selection'
    assert data['rows'][2]['pairs'][0]['touches_excluded_edge'] is True
    assert data['rows'][2]['pairs'][0]['intersects'] is False
    assert data['rows'][4]['pairs'][0]['intersects'] is True
    assert data['rows'][6]['pairs'][0]['distance_infimum_squared_px2'] is None
    assert data['rows'][0]['pose_decoder_timestamp_s'] == 0
    assert data['rows'][0]['pairs'][0]['equipment_time_s'] == .002
    for name, value in output.items():
        if name.endswith('.html'):
            assert all(link in output for link in Page(value).links)
    summary = json.loads(output['report.json'])
    assert summary['geometry_recomputed'] is False and summary['contact'] == summary['support'] == 'unknown'
    assert summary['counts']['pair_slots'] == 14
    assert summary['asset_files']['brand.css'] == sha256(output['brand.css']).hexdigest()


@pytest.mark.parametrize('fault', ['missing_frame', 'duplicate_frame', 'pair_identity', 'status_total', 'excluded_pixel', 'claim', 'uncertainty', 'distance', 'zero_edge', 'mapping_clock'])
def test_saved_contradictions_reject_without_reprocessing(fault):
    meta, rows = fixture()
    p = rows[0]['pairs'][0]
    if fault == 'missing_frame': rows.pop()
    elif fault == 'duplicate_frame': rows[1]['source_frame'] = 0
    elif fault == 'pair_identity': rows[2]['pairs'][0]['equipment_id'] = 'other-object'
    elif fault == 'excluded_pixel': rows[1]['pairs'][0]['pixel'] = [35., 33.]
    elif fault == 'claim': p['contact'] = 'verified'
    elif fault == 'uncertainty': p['authored_uncertainty_px'] = .5
    elif fault == 'distance': p['distance_infimum_squared_px2']['outward_interval'] = [0, 1]
    elif fault == 'zero_edge': rows[2]['pairs'][0]['intersects'] = True
    elif fault == 'mapping_clock': rows[0]['source_mapping']['original_decoder_time_s'] = .01
    metadata, frames = encode(meta, rows)
    if fault == 'status_total':
        value = json.loads(metadata); value['counts']['status_counts']['unavailable'] = 1; metadata = raw(value)
    with pytest.raises(ValueError): report.build_equipment_silhouette_report(metadata, frames)


def test_same_source_and_unavailable_clock_mismatch_remain_explicit():
    meta, rows = fixture(derivative=False)
    output = report.build_equipment_silhouette_report(*encode(meta, rows))
    assert Page(output['frames-001.html']).data['rows'][-1]['equipment_source_frame'] == 6
    meta, rows = fixture()
    p = rows[0]['pairs'][0]
    p.update(status='unavailable', reason='pose_decoder_clock_unavailable_or_different_from_equivalence',
             intersects=None, distance_infimum_squared_px2=None, model_identity=None)
    for key in (*report._RASTER, 'touches_excluded_edge'): p.pop(key)
    rows[0]['pose_decoder_timestamp_s'] = .1
    # Both named wrists retain the failed clock, not a repaired timestamp.
    rows[0]['pairs'][1] = deepcopy(p) | {'wrist_channel_id': 'right', 'wrist_name': 'right_wrist'}
    output = report.build_equipment_silhouette_report(*encode(meta, rows))
    assert Page(output['frames-001.html']).data['rows'][0]['pose_decoder_timestamp_s'] == .1


def test_strict_json_limits_and_deadline_stop_without_asset_reads(monkeypatch):
    metadata, frames = encode(*fixture())
    read = report.Path.open
    monkeypatch.setattr(report.Path, 'open', lambda *a: pytest.fail('asset read before invalid-input rejection'))
    huge = json.loads(metadata)
    huge['pose_provenance']['channel_landmark_index_uncertainty_px'][0][2] = 10 ** 4000
    for bad in (metadata.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1'),
                metadata.replace(b'"native_calls":0', b'"native_calls":NaN'), raw(huge)):
        with pytest.raises(ValueError): report.build_equipment_silhouette_report(bad, frames)
    with pytest.raises(ValueError): report.build_equipment_silhouette_report(metadata, frames + b'{}\n')
    with monkeypatch.context() as patch:
        patch.setattr(report, 'MAX_INPUT', 100)
        with pytest.raises(ValueError, match='input bounds'): report.build_equipment_silhouette_report(metadata, frames)
    ticks = iter((0., 1.))
    with monkeypatch.context() as patch:
        patch.setattr(report.time, 'monotonic', lambda: next(ticks))
        with pytest.raises(TimeoutError): report.build_equipment_silhouette_report(metadata, frames, max_seconds=.5)
    monkeypatch.setattr(report.Path, 'open', read)


@pytest.mark.parametrize('bound', ['MAX_HTML', 'MAX_OUTPUT', 'MAX_FILES'])
def test_output_bound_rejects_instead_of_dropping_slots(monkeypatch, bound):
    monkeypatch.setattr(report, bound, 1)
    with pytest.raises(ValueError, match='no slots dropped'):
        report.build_equipment_silhouette_report(*encode(*fixture()))


_DOM = r'''
import fs from'node:fs';import vm from'node:vm';import assert from'node:assert/strict';import{performance}from'node:perf_hooks';
const html=fs.readFileSync(process.argv[2],'utf8'),scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];let created=0,writes=0;
class E{constructor(){this.children=[];this.listeners=new Map();this.value='';this.own='';created++;}set textContent(v){writes++;this.own=String(v);this.children=[];}get textContent(){return this.own;}set innerHTML(v){assert.fail('unsafe markup');}replaceChildren(...v){this.children=v;}addEventListener(k,f){this.listeners.set(k,f);}fire(k){this.listeners.get(k)?.();}}
const ids=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(v=>[v[1],new E()])),el=id=>{assert(ids.has(id),id);return ids.get(id)};el('data').textContent=scripts[0][1];
vm.runInNewContext(scripts[1][1],{document:{getElementById:el,createElement:()=>new E()},fetch:()=>assert.fail('network'),setTimeout:()=>assert.fail('timer')},{timeout:1000});
'''


def dom(tmp_path, page, assertions):
    path = tmp_path / 'page.html'; path.write_bytes(page)
    result = subprocess.run(['node', '--input-type=module', '-', str(path)], input=_DOM + assertions,
                            text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout) if result.stdout else None


def test_scripted_view_keeps_zero_edge_unknowns_and_untrusted_text_safe(tmp_path):
    meta, rows = fixture()
    for row in rows:
        for pair in row['pairs']: pair['equipment_id'] = '</script><img src=x onerror=alert(1)>'
    output = report.build_equipment_silhouette_report(*encode(meta, rows))
    page = output['frames-001.html']
    assert b'<img src=x' not in page and len(Page(page).scripts) == 2
    dom(tmp_path, page, r'''
assert.match(el('pair-title').textContent,/<\/script><img/);el('frame').value='2';el('frame').fire('change');assert.equal(el('relation').textContent,'Conditional image separation');assert.match(el('boundary').textContent,/not image overlap/);el('frame').value='6';el('frame').fire('change');assert.equal(el('relation').textContent,'Unavailable');assert.match(el('distance').textContent,/Unknown/);assert.match(el('uncertainty').textContent,/Unknown/);
''')


def test_maximum_frame_coverage_and_repeated_selected_detail_cost(tmp_path, record_property):
    inputs = encode(*fixture(count=1800, objects=2))
    began = time.perf_counter(); output = report.build_equipment_silhouette_report(*inputs); elapsed = time.perf_counter() - began
    summary = json.loads(output['report.json'])
    assert summary['counts']['source_frames'] == 1800 and summary['counts']['pair_slots'] == 7200
    assert [p['first_source_frame'] for p in summary['pages']] == list(range(0, 1800, 100))
    assert sum(p['frames'] for p in summary['pages']) == 1800
    assert len(output) <= report.MAX_FILES and sum(map(len, output.values())) <= report.MAX_OUTPUT
    measurement = dom(tmp_path, output['frames-018.html'], r'''
assert.equal(el('frame').children.length,100);assert.equal(el('pair').children.length,4);const before=created,start=performance.now();for(let i=0;i<1000;i++){el('frame').value=String(i%100);el('pair').value=String(i%4);el('frame').fire('change');}const elapsed=performance.now()-start;assert.equal(created,before);assert.match(el('frame-title').textContent,/1799/);const w=writes;for(let i=0;i<1000;i++)el('frame').fire('change');assert.equal(writes,w);process.stdout.write(JSON.stringify({updates:1000,milliseconds:elapsed,new_nodes:created-before,repeated_unchanged_updates:1000,unchanged_writes:writes-w}));
''')
    record_property('maximum_saved_projection', json.dumps({'source_frames': 1800, 'pair_slots': 7200,
        'input_bytes': sum(map(len, inputs)), 'output_bytes': sum(map(len, output.values())), 'output_files': len(output),
        'prepare_seconds': elapsed, 'maximum_html_bytes': max(len(v) for k, v in output.items() if k.endswith('.html')),
        'scripted_dom': measurement, 'scope': 'Analytical saved rows and Node mock DOM; no browser, source admission or native performance claim'}, sort_keys=True))


def test_maximum_pair_width_is_not_truncated():
    output = report.build_equipment_silhouette_report(*encode(*fixture(objects=8)))
    data = Page(output['frames-001.html']).data
    assert len(data['pairs']) == 16 and all(len(row['pairs']) == 16 for row in data['rows'])
    assert json.loads(output['report.json'])['counts']['pair_slots'] == 112


def test_actual_analytical_producer_schema_and_exact_raw_retention(monkeypatch, metadata_only, record_property):
    # Only this preparation uses the released fixture's nine-byte analytical raster.
    blocked = ('mujoco', 'cv2', 'gymnasium', 'torch', 'mediapipe', 'tensorflow', 'requests', 'httpx',
               'weft.human', 'weft.foot_body', 'weft.shared_world')
    def imported(name, *args, **kwargs):
        if any(name == item or name.startswith(item + '.') for item in blocked):
            raise AssertionError('No native, media, source acquisition or model work')
        return metadata_only(name, *args, **kwargs)
    with monkeypatch.context() as preparation:
        preparation.setattr(builtins, '__import__', imported)
        preparation.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('producer subprocess'))
        import importlib
        from test_equipment_silhouette_evidence import fixture as producer_fixture
        producer = importlib.import_module('weft.equipment_silhouette_evidence')
        annotations, options, calls, _ = producer_fixture(producer, count=7)
        evidence = producer.prepare_equipment_silhouette_evidence(annotations, **options)
        metadata = evidence.describe()
        metadata_raw = raw(metadata)
        frames_raw = b''.join(raw(evidence.frame(i)) + b'\n' for i in range(metadata['counts']['source_frames']))
    # The renderer again runs under the stricter no-producer-import guard.
    output = report.build_equipment_silhouette_report(metadata_raw, frames_raw)
    assert output['equipment-metadata.json'] == metadata_raw
    assert output['equipment-frames.jsonl'] == frames_raw
    assert len(calls) == 3 and metadata['work']['binary_raster_bytes'] == 27
    summary = json.loads(output['report.json'])
    assert summary['counts'] == metadata['counts']
    assert summary['counts']['source_frames'] == 7 and summary['counts']['pair_slots'] == 14
    data = Page(output['frames-001.html']).data
    assert [row['source_frame'] for row in data['rows']] == list(range(7))
    assert data['rows'][1]['pairs'][0]['reason'] == 'source_pose_not_sampled'
    assert data['rows'][0]['pairs'][0]['distance_infimum_squared_px2'] == {
        'numerator': 9, 'denominator': 16, 'outward_interval': [.5625, .5625]}
    record_property('producer_schema_composition', json.dumps({'source_frames': 7, 'pair_slots': 14,
        'analytical_raster_loads': len(calls), 'analytical_raster_bytes': 27,
        'metadata_sha256': sha256(metadata_raw).hexdigest(), 'frames_sha256': sha256(frames_raw).hexdigest(),
        'input_bytes': len(metadata_raw) + len(frames_raw), 'exact_returned_raw_bytes': True,
        'scope': 'One existing analytical fixture; no source/PNG decode, model, physics or network'}, sort_keys=True))


def selected_only(metadata):
    original, equipment = metadata['source_sha256'], metadata['equipment_source_sha256']
    return {'profile': 'restrict-only-original-frame-selection-v1',
        'frames_by_canonical_source_sha256': {original: list(metadata['construction_frames'])},
        'partition_consistency_scope': 'selected_frames_only', 'excluded_partition_labels': 'preserved_not_reconciled',
        'source_aliases': {} if equipment == original else {equipment: original}, 'maximum_frames_per_source': 1800}


@pytest.mark.parametrize('derivative', [False, True])
def test_selected_only_scope_retains_excluded_partition_labels_and_exact_bindings(derivative):
    meta, rows = fixture(derivative=derivative)
    meta['canonical_frame_selection'] = selected_only(meta)
    rows[1]['pairs'][0]['human_partition'] = 'audit'
    inputs = encode(meta, rows)
    output = report.build_equipment_silhouette_report(*inputs)
    assert output['equipment-metadata.json'] == inputs[0] and output['equipment-frames.jsonl'] == inputs[1]
    assert json.loads(output['report.json'])['canonical_frame_selection'] == meta['canonical_frame_selection']
    for name in ('index.html', 'frames-001.html'):
        assert b'checked only on the explicitly selected original frames' in output[name]
        assert b'preserved, not reconciled' in output[name]
        assert b'not a full-source partition-consistency check' in output[name]
    saved = Page(output['frames-001.html']).data['rows'][1]['pairs'][0]
    assert saved['source_links']['human_partition'] == 'audit'
    assert saved['source_links']['equipment_partition'] == 'construction'
    assert saved['reason'] == 'excluded_by_explicit_frame_selection'


def test_explicit_empty_selection_retains_all_unavailable_frame_slots():
    meta, rows = fixture()
    meta['construction_frames'] = []
    meta['canonical_frame_selection'] = selected_only(meta)
    for row in rows[:-1]:
        for pair in row['pairs']:
            pair.update(status='unavailable', reason='excluded_by_explicit_frame_selection', intersects=None,
                distance_infimum_squared_px2=None, model_identity=None, pixel=None, authored_uncertainty_px=None,
                source_confidence=None, source_visibility_score=None, source_presence_score=None, representation=None)
            for key in (*report._RASTER, 'touches_excluded_edge'): pair.pop(key, None)
    output = report.build_equipment_silhouette_report(*encode(meta, rows))
    summary = json.loads(output['report.json'])
    assert summary['counts']['source_frames'] == 7 and summary['counts']['pair_slots'] == 14
    assert summary['canonical_frame_selection']['frames_by_canonical_source_sha256'] == {meta['source_sha256']: []}
    assert all(pair['status'] == 'unavailable' for row in Page(output['frames-001.html']).data['rows'] for pair in row['pairs'])


def test_selected_only_binding_and_scope_mismatches_reject():
    original, rows = fixture()
    original['canonical_frame_selection'] = selected_only(original)
    for fault in ('frames', 'boolean_frame', 'aliases', 'full_scope', 'labels', 'bound', 'metadata_tail', 'no_explicit_selection'):
        meta = deepcopy(original); selection = meta['canonical_frame_selection']; source = meta['source_sha256']
        if fault == 'frames': selection['frames_by_canonical_source_sha256'][source] = [0, 4]
        elif fault == 'boolean_frame': selection['frames_by_canonical_source_sha256'][source][0] = False
        elif fault == 'aliases': selection['source_aliases'] = {source: meta['equipment_source_sha256']}
        elif fault == 'full_scope': selection['partition_consistency_scope'] = 'full_source'
        elif fault == 'labels': selection['excluded_partition_labels'] = 'reconciled'
        elif fault == 'bound': selection['maximum_frames_per_source'] = 1801
        elif fault == 'metadata_tail':
            meta['construction_frames'].append(6); selection['frames_by_canonical_source_sha256'][source].append(6)
        else: meta['construction_frames'] = None
        with pytest.raises(ValueError): report.build_equipment_silhouette_report(*encode(meta, rows))


def test_no_selection_field_preserves_every_legacy_output_byte():
    # Captured from frozen e8da5e8 before the optional field was supported.
    expected = {
        'equipment-metadata.json': 'ba883704682f95fe4140d4b0bd4ed2b3fcbdf14087f8654d82bbb5575432ebcc',
        'equipment-frames.jsonl': 'ce0ed39c997982365f248ba429e1e4f7b36cd93cfb293f0c6780479a702465aa',
        'brand.css': '53ed61a8f32aab25c9fb6caf01aeb2ee17bbf092c411c2c6429f2c6ab65d8156',
        'frames-001.html': '0247c12c6a2a4378c31035022092f122aa0b0c073881e716da0f3c72dad66ef2',
        'report.json': '63ef220d7d9fed7748a859975fac6b0a9cf9707105737011915828b689d4a83e',
        'index.html': '4e693487afcfe1f81be729be7251c66651de424143ae49601c434e5074f16a15'}
    output = report.build_equipment_silhouette_report(*encode(*fixture()))
    assert {name: sha256(value).hexdigest() for name, value in output.items()} == expected
