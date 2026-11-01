"""Offline depth-prior editor checks; local fake DOM only, no browser or physics."""

from copy import deepcopy
import json

import pytest

from test_shared_exercise_config_report import Page, no_external_work, package, raw, reseal, run_dom
from weft import shared_exercise_config_report as report
from weft.equipment_proposal_config import DEPTH_RATIO_PRIOR


def configured(*, provenance='  Supplied estimate from object dimensions; not verified.  '):
    files = package(status='config_ready')
    request = json.loads(files['request.json'])
    request['geometry_prior'] = DEPTH_RATIO_PRIOR
    request['equipment'][0]['depth_prior'] = {'depth_to_width_ratio': .5, 'provenance': provenance}
    config = json.loads(files['config.json'])
    config.update(geometry_prior=DEPTH_RATIO_PRIOR, equipment=deepcopy(request['equipment']))
    files.update({'request.json': raw(request), 'config.json': raw(config)})
    reseal(files)
    return files


def test_report_preserves_supplied_ratio_provenance_and_unknowns_without_changing_inputs():
    files = configured(provenance='  <script>not code</script>\nUser estimate 🐼  ')
    before = deepcopy(files)
    output = report.build_shared_exercise_config_report(files)
    summary, data = json.loads(output['report.json']), Page(output['index.html']).data
    assert files == before
    assert summary['equipment_depth_priors'] == json.loads(files['request.json'])['equipment']
    assert summary['depth_prior_scope'] == 'supplied_unverified_not_recovered'
    assert data['request'] == json.loads(files['request.json'])
    assert all(summary[name] == 'unknown' for name in ('contact', 'grasp', 'load'))
    assert summary['geometry_recomputed'] is False and summary['world_admission'] == 'not_evaluated'
    assert b'--config PATH_TO_CONFIG_PACKAGE/config.json' in output['index.html']
    assert b'No ratio is filled in automatically' in output['index.html']
    assert b'<script>not code</script>' not in output['index.html']


def test_editor_requires_each_object_prior_preserves_exact_text_and_roundtrips_legacy(tmp_path):
    output = report.build_shared_exercise_config_report(configured())
    measured = run_dom(tmp_path, output['index.html'], r'''
const D=JSON.parse(scripts[0][1]),initial=JSON.stringify(D.request),fields=i=>{const r=el('selected-equipment').children[i];return{ratio:r.children[2].children[0],provenance:r.children[3].children[0]};};
assert.equal(fields(0).ratio.value,'0.5');assert.equal(fields(0).provenance.value,D.request.equipment[0].depth_prior.provenance);
el('download-request').fire('click');assert.deepEqual(JSON.parse(blobs[0]),D.request);
el('object').value='1';el('source-frame').value='2';el('source-frame').fire('input');el('add-equipment').fire('click');
assert.equal(fields(1).ratio.value,'');assert.equal(fields(1).provenance.value,'');assert.equal(el('download-request').disabled,true);
fields(1).ratio.value='2';fields(1).ratio.fire('input');assert.equal(el('download-request').disabled,true);
fields(1).provenance.value='  My second object estimate 🐼  ';fields(1).provenance.fire('input');assert.equal(el('download-request').disabled,false);
el('download-request').fire('click');const saved=JSON.parse(blobs[1]);assert.equal(saved.equipment[1].depth_prior.depth_to_width_ratio,2);assert.equal(saved.equipment[1].depth_prior.provenance,'  My second object estimate 🐼  ');assert.equal(saved.target_object_id,'box-1');
const baseline=writes,nodes=created;for(let i=0;i<1000;i++){fields(1).ratio.fire('input');el('geometry').fire('change');}assert.equal(writes,baseline);assert.equal(created,nodes);
el('geometry').value='floor_box_depth_equals_width';el('geometry').fire('change');assert.equal(el('selected-equipment').children[0].children.length,2);el('download-request').fire('click');assert.deepEqual(JSON.parse(blobs[2]).equipment,[{object_id:'box-1',source_frame:0},{object_id:'box-2',source_frame:2}]);
el('geometry').value=D.depth_prior.mode;el('geometry').fire('change');assert.equal(fields(1).ratio.value,'2');assert.equal(fields(1).provenance.value,'  My second object estimate 🐼  ');assert.equal(JSON.stringify(D.request),initial);
process.stdout.write(JSON.stringify({downloads:downloads.length,unchanged_events:2000}));
''')
    assert measured == {'downloads': 3, 'unchanged_events': 2000}


def test_opt_in_from_legacy_has_no_automatic_ratio_and_rejects_invalid_inputs(tmp_path):
    output = report.build_shared_exercise_config_report(package(status='config_ready'))
    run_dom(tmp_path, output['index.html'], r'''
const D=JSON.parse(scripts[0][1]);el('geometry').value=D.depth_prior.mode;el('geometry').fire('change');
const r=el('selected-equipment').children[0],ratio=r.children[2].children[0],provenance=r.children[3].children[0];assert.equal(ratio.value,'');assert.equal(provenance.value,'');assert.equal(el('download-request').disabled,true);
for(const value of ['','0','121','Infinity']){ratio.value=value;ratio.fire('input');provenance.value='User supplied';provenance.fire('input');assert.equal(el('download-request').disabled,true);}
ratio.value='1';ratio.fire('input');provenance.value='  ';provenance.fire('input');assert.equal(el('download-request').disabled,true);
provenance.value='🐼'.repeat(2001);provenance.fire('input');assert.equal(el('download-request').disabled,true);
provenance.value='🐼'.repeat(2000);provenance.fire('input');assert.equal(el('download-request').disabled,false);
el('download-request').fire('click');assert.equal(Array.from(JSON.parse(blobs[0]).equipment[0].depth_prior.provenance).length,2000);
''')


def test_encoded_request_budget_blocks_download_without_trimming_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'MAX_CONFIG_BYTES', 5000)
    output = report.build_shared_exercise_config_report(configured())
    run_dom(tmp_path, output['index.html'], r'''
const row=el('selected-equipment').children[0],provenance=row.children[3].children[0];
provenance.value='🐼'.repeat(1000);provenance.fire('input');assert.equal(el('download-request').disabled,true);
assert.match(el('request-state').textContent,/saved-file budget/);assert.equal(JSON.parse(el('request-preview').textContent).equipment[0].depth_prior.provenance,provenance.value);
el('download-request').fire('click');assert.equal(downloads.length,0);
''')


@pytest.mark.parametrize('prior', [None, {}, {'depth_to_width_ratio': .5},
    {'depth_to_width_ratio': True, 'provenance': 'User'}, {'depth_to_width_ratio': 0, 'provenance': 'User'},
    {'depth_to_width_ratio': 121, 'provenance': 'User'}, {'depth_to_width_ratio': '1', 'provenance': 'User'},
    {'depth_to_width_ratio': 1, 'provenance': '   '}, {'depth_to_width_ratio': 1, 'provenance': 'x' * 2001},
    {'depth_to_width_ratio': 1, 'provenance': 'User', 'measured': True}])
def test_invalid_ready_depth_prior_is_not_presented_as_valid(prior):
    files = configured()
    for name in ('request.json', 'config.json'):
        document = json.loads(files[name]); document['equipment'][0]['depth_prior'] = prior
        files[name] = raw(document)
    reseal(files)
    with pytest.raises(ValueError):
        report.build_shared_exercise_config_report(files)


def test_ready_report_rejects_config_that_drops_or_changes_supplied_depth():
    files = configured()
    config = json.loads(files['config.json'])
    config['equipment'][0]['depth_prior']['depth_to_width_ratio'] = 1
    files['config.json'] = raw(config); reseal(files)
    with pytest.raises(ValueError, match='config changes'):
        report.build_shared_exercise_config_report(files)


def test_legacy_report_keeps_absent_prior_and_discovery_keeps_all_source_choices_unset():
    files = package(status='config_ready')
    output = report.build_shared_exercise_config_report(files)
    summary = json.loads(output['report.json'])
    assert 'equipment_depth_priors' not in summary and 'depth_prior_scope' not in summary
    assert Page(output['index.html']).data['request'] == json.loads(files['request.json'])
    discovery = report.build_shared_exercise_config_report(package())
    request = Page(discovery['index.html']).data['request']
    assert request['geometry_prior'] is None and request['equipment'] == []


def test_twelve_object_maximum_provenance_edits_keep_rows_and_handlers_stable(tmp_path, record_property):
    files = configured()
    snapshot = json.loads(files['snapshot.json'])
    equipment = snapshot['observation_sources'][1]
    names = [f'box-{index + 1}' for index in range(12)]
    equipment['entities'] = [{'object_id': name, 'label': name, 'kind': None} for name in names]
    originals = {row['source_frame']: row for row in equipment['slots'] if row['object_id'] == 'box-1'}
    equipment['slots'] = [{**deepcopy(originals[frame]), 'object_id': name,
        'observation_id': f'{name}-{frame}', 'evidence_ids': [f'{name}-evidence-{frame}']}
        for frame in range(equipment['decoded_frames']) for name in names]
    snapshot['counts']['equipment_entity_slots'] = len(equipment['slots'])
    files['snapshot.json'] = raw(snapshot)
    choices = [{'object_id': name, 'source_frame': 0,
                'depth_prior': {'depth_to_width_ratio': .5, 'provenance': 'x' * 2000}} for name in names]
    for filename in ('request.json', 'config.json'):
        document = json.loads(files[filename]); document['equipment'] = choices; files[filename] = raw(document)
    metadata = json.loads(files['metadata.json']); metadata['counts'] = snapshot['counts']; files['metadata.json'] = raw(metadata)
    reseal(files, snapshot=True)
    output = report.build_shared_exercise_config_report(files)
    measured = run_dom(tmp_path, output['index.html'], r'''
const rows=el('selected-equipment').children.slice(),countHandlers=n=>n.listeners.size+n.children.reduce((sum,c)=>sum+countHandlers(c),0),handlers=countHandlers(el('selected-equipment')),nodes=created;
assert.equal(rows.length,12);const started=performance.now();
for(let pass=0;pass<10;pass++)for(let i=0;i<12;i++){const ratio=rows[i].children[2].children[0],provenance=rows[i].children[3].children[0];ratio.value=String(.5+(pass*12+i+1)/1000);ratio.fire('input');provenance.value='x'.repeat(1997)+String(pass*12+i).padStart(3,'0');provenance.fire('input');}
const elapsed=performance.now()-started;assert.ok(elapsed<2000);assert.equal(created,nodes);assert.equal(countHandlers(el('selected-equipment')),handlers);assert.ok(rows.every((row,i)=>row===el('selected-equipment').children[i]));assert.equal(el('download-request').disabled,false);
const request=JSON.parse(el('request-preview').textContent);assert.equal(request.equipment.length,12);assert.ok(request.equipment.every(row=>row.depth_prior.provenance.length===2000));
process.stdout.write(JSON.stringify({objects:12,provenance_characters_each:2000,changed_input_events:240,batch_ms:elapsed,new_nodes:created-nodes,retained_selected_row_handlers:handlers}));
''')
    assert measured['new_nodes'] == 0 and measured['changed_input_events'] == 240
    for key, value in measured.items(): record_property(key, value)
