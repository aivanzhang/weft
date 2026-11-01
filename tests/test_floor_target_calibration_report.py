"""Analytical saved metadata and scripted DOM only; no calibration execution."""

import builtins
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import socket
import subprocess
import time

import pytest

from weft import floor_target_calibration_report as report


SOURCE, OBS, MEASURE = 'a'*64, 'b'*64, 'c'*64
I = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


def raw(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def pin(value):
    return sha256(value).hexdigest()


def canonical(value):
    return pin(json.dumps(value,sort_keys=True,allow_nan=False).encode())


@pytest.fixture(autouse=True)
def no_external_work(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'cv2', 'mujoco', 'numpy', 'gymnasium', 'torch', 'mediapipe', 'requests', 'httpx'}, name
        assert name != 'weft.floor_target_calibration', 'report must not import its solver'
        return original(name, *args, **kwargs)
    def denied(*args, **kwargs):
        pytest.fail('Report attempted network work')
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def fixture(*, count=6, supplied=False, corners=4, used=(0, 2)):
    camera = {'schema_version':1, 'source_sha256':SOURCE, 'width':640, 'height':480, 'fps':30., 'frame_count':count,
        'world_axes':'right_handed_z_up', 'units':'m', 'quality':'measured' if supplied else 'reconstructed',
        'provenance':'Explicit authored test fixture; no recovered physical calibration',
        'K':[[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]], 'radial_k1':0.,
        'frames':[{'frame':f, 'time_s':f/30, 'R':I, 't':[0., 0., 2.]} for f in range(count)]}
    ground = {'schema_version':1, 'source_sha256':SOURCE, 'calibration_sha256':canonical(camera),
        'world_axes':'right_handed_z_up', 'units':'m', 'height_m':.25,
        'quality':'measured' if supplied else 'reconstructed', 'provenance':'Authored nonzero ground fixture'}
    camera_raw, ground_raw = raw(camera)+b'\n', raw(ground)+b'\n'
    def ref(name):return {'document_sha256':MEASURE, 'record_id':name}
    request = {'schema_version':1, 'profile':'floor-target-calibration-request-v1', 'source_sha256':SOURCE,
        'mode':'supplied' if supplied else 'planar_target', 'observation_sha256s':[OBS], 'measurement_sha256s':[MEASURE],
        'supplied_camera_sha256':pin(camera_raw) if supplied else None, 'supplied_ground_sha256':pin(ground_raw) if supplied else None,
        'world_frame':{'id':'authored-world', 'length_unit':'m', 'axes':'x_right_y_away_z_up'},
        'anchor_frame':None if supplied else used[0], 'check_frames':[] if supplied else list(used[1:]),
        'evaluation_frames':[] if supplied else [max(used)+2], 'fixed_interval':None if supplied else
            {'first_frame':0, 'last_frame':count-1, 'camera_static':True, 'target_static':True,
             'method':'Explicit authored fixture', 'uncertainty':'Declared only; not propagated',
             'evidence_refs':[{'document_sha256':OBS, 'evidence_id':'global'}]},
        'measurement_refs':{'camera':ref('camera'), 'floor':ref('floor'),
                            'target':None if supplied else ref('target'), 'target_to_world':None if supplied else ref('alignment')}}
    records = []
    def header(name, kind):
        uncertainty = {key:('bounded hand annotation' if key == 'pixel_error_note' else .01) for key in report._UNCERTAINTY[kind]}
        uncertainty['interpretation'] = 'declared_absolute_bound'
        row = {'document_sha256':MEASURE, 'id':name, 'kind':kind, 'quality':'measured', 'method':'Authored fixture method',
            'provenance':'Metadata fixture only', 'uncertainty':uncertainty,
            'evidence_refs':[{'document_sha256':OBS, 'evidence_id':'global'}], 'target_units':'mm' if kind == 'planar_target' else None}
        records.append(row); return row
    for name, kind in [('camera', 'camera_calibration'), ('floor', 'floor_plane')]+([] if supplied else [('target', 'planar_target'), ('alignment', 'target_to_world')]):
        header(name, kind)
    frames = []
    for f in range(count):
        role = 'anchor' if f == request['anchor_frame'] else 'construction_check' if f in request['check_frames'] else 'evaluation_not_used' if f in request['evaluation_frames'] else 'not_selected'
        row = {'source_frame':f, 'source_status':'decoded' if f < count-1 else 'metadata_only_not_decoded',
            'nominal_time_s':f/30, 'role':role, 'partition':None, 'observation_id':None, 'source_time_s':None,
            'measurement_ref':None, 'status':'unavailable', 'reason':'missing_corner_measurement', 'points':[]}
        if role in ('anchor', 'construction_check', 'evaluation_not_used'):
            header('corners-'+str(f), 'target_corners')
            row.update(partition='construction', observation_id='target-'+str(f), source_time_s=f/30+.001,
                measurement_ref=ref('corners-'+str(f)), status='observed' if role == 'evaluation_not_used' else 'prepared',
                reason='evaluation_not_used' if role == 'evaluation_not_used' else None,
                points=[] if role == 'evaluation_not_used' else [{'corner_id':'corner-'+str(i), 'status':'observed', 'reason':None} for i in range(corners)])
        frames.append(row)
    projections = []
    if not supplied:
        for f in used:
            points = []
            for i in range(corners):
                xy = [20.+i%8*30, 20.+i//8*30]; predicted = [xy[0]+.1, xy[1]+.1]
                delta = [a-b for a,b in zip(predicted, xy)]; error = math.hypot(*delta)
                points.append({'corner_id':'corner-'+str(i), 'pixel':xy, 'uncertainty_px':.5, 'predicted_pixel':predicted,
                    'residual_px':delta, 'error_px':error, 'depth_m':2., 'status':'projected'})
            errors = [p['error_px'] for p in points]
            projections.append({'source_frame':f, 'measurement_ref':ref('corners-'+str(f)), 'points':points,
                'rms_px':math.sqrt(sum(v*v for v in errors)/len(errors)), 'maximum_px':max(errors)})
    errors = [p['error_px'] for row in projections for p in row['points']]
    alternatives = [] if supplied else [
        {'index':0, 'R_ct':I, 't_ct':[0., 0., 2.], 'R_cw':I, 't_cw':[0., 0., 2.], 'duplicate_of':None,
         'status':'nominal_checks_passed', 'reasons':[], 'checks':{'proper_rotation':True, 'depth':True, 'projection_domain':True,
             'construction_residual':True, 'rms_px':math.sqrt(sum(v*v for v in errors)/len(errors)), 'maximum_px':max(errors)}, 'projections':projections},
        {'index':1, 'R_ct':None, 't_ct':None, 'R_cw':None, 't_cw':None, 'duplicate_of':None, 'status':'rejected',
         'reasons':['invalid_solver_pose'], 'checks':{'proper_rotation':False, 'depth':False, 'projection_domain':False,
             'construction_residual':False, 'rms_px':None, 'maximum_px':None}, 'projections':[]}]
    result = {'schema_version':1, 'profile':'floor-target-calibration-result-v1', 'status':'qualified_nominal', 'mode':request['mode'],
        'reason':'supplied_exact_nominal_pair_not_authenticated' if supplied else 'one_distinct_nominal_solution_under_fixed_checks',
        'input_bindings':{}, 'source':{'sha256':SOURCE, 'width':640, 'height':480, 'fps':30., 'native_frame_times_s':None, 'decoded_frames':count-1,
            'metadata_frames':count, 'retained_frames':count, 'time':{OBS:{'unit':'s', 'origin':'source'}}},
        'request':request, 'measurement_records':records, 'frames':frames, 'alternatives':alternatives,
        'selected_index':None if supplied else 0, 'artifact_bindings':None, 'counts':{}, 'work':{},
        'limits':{'maximum_source_frames':1800, 'maximum_corners':64, 'maximum_alternatives':2, 'maximum_point_reprojections':1024,
            'maximum_rms_px':1., 'maximum_error_px':2., 'minimum_depth_m':.2, 'maximum_depth_m':100.,
            'duplicate_rotation_max_abs':1e-9, 'duplicate_translation_max_abs_m':1e-9},
        'uncertainty':{'propagation':'not_propagated', 'interpretation':'Nominal consistency only; declared bounds or standard deviations are not propagated or certified', 'measurement_authenticity':'unverified'},
        'claims':{'physical_validation':False, 'contact':'unknown', 'grasp':'unknown', 'load':'unknown', 'geometry_modified':False,
            'body_or_floor_installed':False, 'evaluation_used':False, 'source_measurement_authentication':False}}
    files = {'camera.json':camera_raw, 'ground.json':ground_raw}
    return save(result, files)


def save(result, files):
    request = result['request']; request_raw = raw(request)
    members = [{'role':'observation', 'path':'inputs/observations/'+OBS+'.json', 'sha256':OBS, 'bytes':100},
        {'role':'measurement', 'path':'inputs/measurements/'+MEASURE+'.json', 'sha256':MEASURE, 'bytes':500},
        {'role':'request', 'path':'inputs/request.json', 'sha256':pin(request_raw), 'bytes':len(request_raw)}]
    if request['mode'] == 'supplied':
        for name in ('camera', 'ground'):
            members.append({'role':name, 'path':'inputs/'+name+'.json', 'sha256':pin(files[name+'.json']), 'bytes':len(files[name+'.json'])})
    result['input_bindings'] = {'source_sha256':SOURCE, 'request_sha256':pin(request_raw), 'observation_document_sha256s':[OBS],
        'measurement_document_sha256s':[MEASURE], 'supplied_camera_sha256':request['supplied_camera_sha256'], 'supplied_ground_sha256':request['supplied_ground_sha256']}
    prepared = [r for r in result['frames'] if r['status'] == 'prepared']
    known = sum(p['status'] == 'observed' for r in prepared for p in r['points'])
    passing = [r for r in result['alternatives'] if r['status'] == 'nominal_checks_passed']
    result['counts'] = {'source_slots':len(result['frames']), 'measurement_records':len(result['measurement_records']),
        'construction_frames_requested':(request['anchor_frame'] is not None)+len(request['check_frames']),
        'evaluation_frames_not_used':len(request['evaluation_frames']), 'prepared_construction_frames':len(prepared),
        'used_corner_observations':known, 'alternatives':len(result['alternatives']), 'passing_alternatives':len(passing),
        'distinct_passing_alternatives':sum(r['duplicate_of'] is None for r in passing),
        'frame_status_counts':dict(Counter(r['reason'] or r['status'] for r in result['frames']))}
    result['work'] = {'input_bytes':sum(r['bytes'] for r in members), 'solver_calls':0 if request['mode'] == 'supplied' or result['status'] == 'unavailable' else 1,
        'undistorted_points':known, 'point_reprojections':sum(len(p['points']) for a in result['alternatives'] for p in a['projections']), 'prepared_construction_frames':len(prepared)}
    if result['status'] == 'qualified_nominal':
        camera = json.loads(files['camera.json']); ground = json.loads(files['ground.json'])
        result['artifact_bindings'] = {'camera_raw_sha256':pin(files['camera.json']), 'camera_canonical_sha256':canonical(camera),
            'ground_raw_sha256':pin(files['ground.json']), 'qualities':{'camera':camera['quality'], 'ground':ground['quality']},
            'generation':'supplied_exact_bytes' if request['mode'] == 'supplied' else 'planar_nominal'}
    else:
        result['artifact_bindings'] = None
        files = {k:v for k,v in files.items() if k not in ('camera.json', 'ground.json')}
    files.update({'calibration.json':raw(result), 'inputs/request.json':request_raw,
        'input-files.json':raw({'schema_version':1, 'profile':'floor-target-calibration-input-files-v1', 'files':members})})
    return files


class Page(HTMLParser):
    def __init__(self, value):
        super().__init__(); self.links=[]; self.scripts=[]; self.current=None; self.feed(value.decode())
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if tag == 'a':self.links.append(attrs['href'])
        if tag == 'script':self.current=[];self.scripts.append(self.current)
    def handle_data(self, value):
        if self.current is not None:self.current.append(value)
    def handle_endtag(self, tag):
        if tag == 'script':self.current=None
    @property
    def data(self):return json.loads(''.join(self.scripts[0]))


def test_qualified_target_preserves_inputs_unknowns_alternatives_and_links():
    files=fixture(); before=deepcopy(files); output=report.build_floor_target_calibration_report(files)
    assert files == before
    index=Page(output['index.html']); model=json.loads(output['report.json'])
    assert model['admitted_downloads'] == ['camera.json', 'ground.json']
    assert model['counts']['alternatives'] == 2 and model['counts']['source_slots'] == 6
    assert all('../'+name in index.links for name in ('camera.json', 'ground.json', 'calibration.json', 'inputs/request.json'))
    assert b'>mm<' in output['index.html'] and b'0.25 m' in output['index.html'] and b'invalid_solver_pose' in output['index.html']
    rows=Page(output['frames-001.html']).data['rows']
    assert [r['frame']['source_frame'] for r in rows] == list(range(6))
    assert rows[4]['frame']['role'] == 'evaluation_not_used' and rows[4]['projections'] == [None,None]
    assert rows[5]['frame']['source_status'] == 'metadata_only_not_decoded'
    assert output['brand.css'] == (Path(report.__file__).parent/'assets/brand.css').read_bytes()


def test_supplied_sidecars_need_no_target_and_keep_exact_quality_clock_and_bytes():
    files=fixture(supplied=True); result=json.loads(files['calibration.json']); camera=json.loads(files['camera.json'])
    camera['frames'][2]['time_s'] += 5e-8; files['camera.json']=raw(camera)+b'\n'
    ground=json.loads(files['ground.json']);ground['calibration_sha256']=canonical(camera);files['ground.json']=raw(ground)+b'\n'
    for key in ('camera','ground'):result['request']['supplied_'+key+'_sha256']=pin(files[key+'.json'])
    files=save(result,files); output=report.build_floor_target_calibration_report(files)
    assert b'No target pose was solved' in output['index.html'] and b'Camera quality: measured' in output['index.html']
    assert json.loads(output['report.json'])['counts']['used_corner_observations'] == 0


@pytest.mark.parametrize('status', ['unavailable','rejected','ambiguous'])
def test_unadmitted_outcomes_keep_all_reasons_and_never_link_result_sidecars(status):
    files=fixture(); result=json.loads(files['calibration.json']); result['status']=status;result['selected_index']=None
    if status == 'unavailable':
        result['alternatives']=[];result['reason']='missing_floor_measurement'
        for row in result['frames']:
            if row['status']=='prepared':row.update(status='observed',reason=None,points=[])
    elif status == 'rejected':
        result['alternatives']=[dict(result['alternatives'][1],index=0)];result['reason']='no_admissible_nominal_solution'
    else:
        other=deepcopy(result['alternatives'][0]);other.update(index=1,t_ct=[.1,0.,2.],t_cw=[.1,0.,2.]);result['alternatives'][1]=other
        result['reason']='multiple_distinct_nominal_solutions'
    output=report.build_floor_target_calibration_report(save(result,files)); index=Page(output['index.html'])
    assert '../camera.json' not in index.links and '../ground.json' not in index.links
    assert json.loads(output['report.json'])['status'] == status
    assert len(Page(output['frames-001.html']).data['rows']) == 6


@pytest.mark.parametrize('mutation', ['missing_output','foreign_camera','counts','missing_frame','evaluation_payload','pixel_error','negative_depth','uncertainty_claim'])
def test_forged_saved_joins_fail_closed(mutation):
    files=fixture();result=json.loads(files['calibration.json'])
    if mutation=='missing_output':del files['ground.json']
    elif mutation=='foreign_camera':files['camera.json']+=b' '
    elif mutation=='counts':result['counts']['source_slots']-=1
    elif mutation=='missing_frame':result['frames'].pop()
    elif mutation=='evaluation_payload':result['frames'][4]['points']=[{'corner_id':'hidden','status':'observed','reason':None}]
    elif mutation=='pixel_error':result['alternatives'][0]['projections'][0]['points'][0]['error_px']+=1
    elif mutation=='negative_depth':result['alternatives'][0]['projections'][0]['points'][0]['depth_m']=-1
    else:result['uncertainty']['propagation']='certified'
    files['calibration.json']=raw(result)
    with pytest.raises(ValueError):report.build_floor_target_calibration_report(files)


def test_strict_json_safe_text_paths_and_resource_limits():
    files=fixture(); result=json.loads(files['calibration.json'])
    result['measurement_records'][0]['method']='</script><img src=x onerror=alert(1)>\nsecond line'
    files=save(result,files); output=report.build_floor_target_calibration_report(files)
    assert b'<img src=x' not in output['index.html'] and b'&lt;img src=x' in output['index.html']
    for bad in (b'{"x":1,"x":2}',b'{"x":NaN}'):
        with pytest.raises(ValueError):report.build_floor_target_calibration_report({**files,'calibration.json':bad})
    inventory=json.loads(files['input-files.json']);inventory['files'][0]['path']='../source.json'
    with pytest.raises(ValueError):report.build_floor_target_calibration_report({**files,'input-files.json':raw(inventory)})
    with pytest.raises(ValueError):report.build_floor_target_calibration_report({**files,'calibration.json':b' '*(report.MAX_RESULT+1)})
    with pytest.raises(ValueError):report.build_floor_target_calibration_report(files,max_seconds=6)
    with pytest.raises(TimeoutError):report.build_floor_target_calibration_report(files,max_seconds=1e-12)


def test_missing_nominal_clock_retains_native_times_without_manufacturing_fps():
    files=fixture(); result=json.loads(files['calibration.json'])
    result.update(status='unavailable',reason='missing_source_nominal_clock',alternatives=[],selected_index=None)
    times=[.001+f/29.97 for f in range(6)]
    result['source'].update(fps=None,native_frame_times_s=times,decoded_frames=6)
    for row in result['frames']:
        row.update(nominal_time_s=None,source_time_s=times[row['source_frame']],source_status='decoded',points=[])
        if row['status']=='prepared':row.update(status='observed',reason='missing_source_nominal_clock')
    files=save(result,files);output=report.build_floor_target_calibration_report(files)
    saved=Page(output['frames-001.html']).data['rows']
    assert all(row['frame']['nominal_time_s'] is None for row in saved)
    assert [row['frame']['source_time_s'] for row in saved]==times
    assert json.loads(output['report.json'])['admitted_downloads']==[]
    result['frames'][0]['nominal_time_s']=0.;files['calibration.json']=raw(result)
    with pytest.raises(ValueError,match='unknown'):report.build_floor_target_calibration_report(files)


def test_request_inventory_order_is_preserved_while_result_pins_are_sorted():
    files=fixture();result=json.loads(files['calibration.json']);request=result['request']
    request['observation_sha256s']=['d'*64,OBS];request['measurement_sha256s']=['e'*64,MEASURE]
    request_raw=raw(request);files['inputs/request.json']=request_raw
    result['source']['time']['d'*64]=deepcopy(result['source']['time'][OBS])
    result['input_bindings'].update(request_sha256=pin(request_raw),observation_document_sha256s=sorted(request['observation_sha256s']),
                                   measurement_document_sha256s=sorted(request['measurement_sha256s']))
    inventory=json.loads(files['input-files.json'])
    inventory['files'].extend([{'role':'observation','path':'inputs/observations/'+('d'*64)+'.json','sha256':'d'*64,'bytes':100},
                               {'role':'measurement','path':'inputs/measurements/'+('e'*64)+'.json','sha256':'e'*64,'bytes':100}])
    request_row=next(r for r in inventory['files'] if r['role']=='request');request_row.update(sha256=pin(request_raw),bytes=len(request_raw))
    result['work']['input_bytes']=sum(row['bytes'] for row in inventory['files'])
    files.update({'input-files.json':raw(inventory),'calibration.json':raw(result)})
    before=deepcopy(files);output=report.build_floor_target_calibration_report(files)
    assert files==before and json.loads(output['report.json'])['input_files']['inputs/request.json']==pin(request_raw)
    assert result['request']['observation_sha256s']==['d'*64,OBS]
    result['input_bindings']['observation_document_sha256s']=request['observation_sha256s']
    with pytest.raises(ValueError,match='bindings'):report.build_floor_target_calibration_report({**files,'calibration.json':raw(result)})


def test_early_abstention_retains_named_headers_without_used_points_or_projection():
    files=fixture();result=json.loads(files['calibration.json'])
    result.update(status='unavailable',reason='unknown_pixel_uncertainty',alternatives=[],selected_index=None)
    for row in result['frames']:
        if row['status']=='prepared':
            row['status']='observed'
            if row['source_frame']!=result['request']['anchor_frame']:row['points']=[]
    files=save(result,files);output=report.build_floor_target_calibration_report(files)
    model=json.loads(output['report.json']);rows=Page(output['frames-001.html']).data['rows']
    assert len(rows[0]['frame']['points'])==4 and rows[0]['frame']['status']=='observed'
    assert rows[0]['projections']==[] and model['counts']['used_corner_observations']==0
    assert model['counts']['prepared_construction_frames']==0 and model['admitted_downloads']==[]
    result['frames'][4]['points']=deepcopy(result['frames'][0]['points'])
    with pytest.raises(ValueError,match='unprepared|excluded'):report.build_floor_target_calibration_report({**files,'calibration.json':raw(result)})


def test_projection_order_follows_explicit_anchor_then_checks_not_source_sort():
    files=fixture(used=(2,0));result=json.loads(files['calibration.json'])
    assert [row['source_frame'] for row in result['alternatives'][0]['projections']]==[2,0]
    output=report.build_floor_target_calibration_report(files);rows=Page(output['frames-001.html']).data['rows']
    assert [row['frame']['source_frame'] for row in rows]==list(range(6))
    assert rows[2]['frame']['role']=='anchor' and rows[0]['frame']['role']=='construction_check'
    assert rows[0]['projections'][0]['source_frame']==0 and rows[2]['projections'][0]['source_frame']==2
    result['alternatives'][0]['projections'].sort(key=lambda row:row['source_frame'])
    with pytest.raises(ValueError,match='projections'):report.build_floor_target_calibration_report({**files,'calibration.json':raw(result)})


_DOM = r'''
import vm from'node:vm';import assert from'node:assert/strict';import{performance}from'node:perf_hooks';
let created=0,writes=0;class E{constructor(tag=''){this.tag=tag;this.children=[];this.listeners=new Map();this.value='';this.own='';this.attrs={};created++;}set textContent(v){this.own=String(v);this.children=[];writes++;}get textContent(){return this.own+this.children.map(n=>n.textContent).join('');}set innerHTML(v){assert.fail('unsafe markup');}append(...v){this.children.push(...v);}replaceChildren(...v){this.children=[...v];writes++;}setAttribute(k,v){this.attrs[k]=String(v);writes++;}addEventListener(k,f){this.listeners.set(k,f);}fire(k){this.listeners.get(k)?.();}}
const scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)],ids=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(v=>[v[1],new E()])),el=id=>{assert(ids.has(id));return ids.get(id)};el('data').textContent=scripts[0][1];
vm.runInNewContext(scripts[1][1],{document:{getElementById:el,createElement:t=>new E(t),createElementNS:(_,t)=>new E(t)},fetch:()=>assert.fail('network'),setTimeout:()=>assert.fail('timer')},{timeout:1000});
const D=JSON.parse(scripts[0][1]);assert.equal(el('frame').value,'');assert.equal(el('alternative').value,'');
const started=performance.now();let frames=0,points=0,maxPlot=0;
for(let f=0;f<D.rows.length;f++){el('frame').value=String(f);el('frame').fire('change');assert.deepEqual(JSON.parse(el('frame-record').textContent),D.rows[f].frame);frames++;
 for(let a=0;a<D.alternatives.length;a++){el('alternative').value=String(a);el('alternative').fire('change');const p=D.rows[f].projections[a];assert.equal(el('point-rows').children.length,p===null?0:p.points.length);points+=p===null?0:p.points.length;maxPlot=Math.max(maxPlot,el('plot-points').children.length);}}
const navigation_ms=performance.now()-started,before=writes,nodes=created;for(let i=0;i<1000;i++)el('alternative').fire('change');assert.equal(writes,before);assert.equal(created,nodes);
el('frame').value='';el('frame').fire('change');assert.equal(el('frame-clock').textContent,'');assert.equal(el('measurement-record').textContent,'');assert.equal(el('plot-points').children.length,0);
process.stdout.write(JSON.stringify({frames,points,max_svg_points:maxPlot,navigation_ms,unchanged_updates:1000,unchanged_writes:0}));
'''


def test_large_complete_report_and_bounded_selected_frame_navigation(record_property):
    files=fixture(count=1800,corners=64,used=(0,2,4,6,8,10,12,14))
    start=time.monotonic();output=report.build_floor_target_calibration_report(files);elapsed=time.monotonic()-start
    model=json.loads(output['report.json']); assert len(model['pages']) == 9
    frames=[row['frame']['source_frame'] for page in model['pages'] for row in Page(output[page['path']]).data['rows']]
    assert frames == list(range(1800)) and len(output)<=16 and sum(map(len,output.values()))<=4<<20
    assert all(len(value)<=2<<20 for name,value in output.items() if name.endswith('.html'))
    html_value=output['frames-001.html'].decode()
    program='const html='+json.dumps(html_value)+';\n'+_DOM
    measured=subprocess.run(['node','--input-type=module','-'],input=program,text=True,capture_output=True,timeout=5)
    assert measured.returncode == 0,measured.stderr
    behavior=json.loads(measured.stdout);assert behavior['frames']==200 and behavior['points']==512 and behavior['max_svg_points']==128
    record_property('measurement',json.dumps({'scope':'Authored Python preparation and Node fake DOM, not browser performance',
        'input_bytes':sum(map(len,files.values())),'output_bytes':sum(map(len,output.values())),'prepare_seconds':elapsed,**behavior},sort_keys=True))
