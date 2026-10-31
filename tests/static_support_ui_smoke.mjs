/** Scripted DOM checks using the real pure support-task producer; no browser or physics. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {Element, html, script, drain, fixture as exerciseFixture} from './lab_ui_smoke.mjs';

const root=fileURLToPath(new URL('..',import.meta.url));
const python=`import importlib.abc,json,sys
from dataclasses import replace
sys.path[:0]=['src','tests']
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'mujoco','gymnasium','cv2','mediapipe'}:raise AssertionError('No native dependency: '+name)
sys.meta_path.insert(0,Guard())
from test_shared_support_task import support_fixture,accumulator,sample,finish
from weft.shared_support_task import build_shared_support_task
names=['torso_twist','torso_lateral','torso_flex']+[s+'_'+j for s in ('left','right') for j in ('shoulder_twist','shoulder_lateral','shoulder','elbow','hip_twist','hip_lateral','hip','knee')]
output=[]
for assisted in (False,True):
 scene,task=support_fixture(root=assisted)
 scene.update(bounds=[3.,2.,2.5],assumptions=['Authored analytical UI fixture'],joint_names=names)
 scene['objects'][0].update(label='Authored support',color='#D8C9BA')
 task=build_shared_support_task(scene,support_requirements=task['support_requirements'],root_assistance=assisted)
 acc=accumulator(scene,task);initial=acc.metrics()
 for i in range(acc.substeps):acc.consume(sample(acc,i))
 active=acc.metrics()
 for i in range(acc.substeps,acc.maximum_samples):acc.consume(sample(acc,i))
 passed=acc.metrics();failed=finish(accumulator(scene,task),lambda value,i:replace(value,contacts=()))
 output.append(dict(scene=scene,task=task,action_contract=dict(names=names+(['root_x','root_y','root_z'] if assisted else []),root_assistance=task['control']['root_assistance'],object_force_channels=[]),metrics=dict(initial=initial,active=active,passed=passed,failed=failed)))
print(json.dumps(output,allow_nan=False))
`;
const fixtures=JSON.parse(execFileSync(`${root}/.venv-lab/bin/python`,['-I','-B','-c',python],{
  cwd:root,timeout:5000,maxBuffer:128*1024,env:{PATH:process.env.PATH||'/usr/bin:/bin',PYTHONDONTWRITEBYTECODE:'1'},encoding:'utf8'}));
const copy=value=>JSON.parse(JSON.stringify(value));
const content=node=>[node.textContent,...node.children.map(content)].join('\n');
function attachedSupportPolicy(kind='searched_joint_residuals',overrides={}){
  return {profile:'shared-support-policy-v1',policy_sha256:'c'.repeat(64),selected_feedback_kind:kind,
    privileged_simulator_features:true,training_score:.7,training_score_improved_over_authored:kind!=='authored_unchanged',
    evaluation_status:'evaluated',evaluation_successes:Object.fromEntries(['coast','random','authored','learned'].map(name=>[name,{episodes:2,successes:['authored','learned'].includes(name)?1:0}])),
    evaluation_reset_protocol:{profile:'shared-support-reset-v1',human_jitter_scale:.1,object_position_jitter_world:0,
      object_velocity_jitter_world_s:0,equipment_stability_reference:'not_applicable',static_support_reset_state:'unchanged_scene_pose_and_velocity'},
    ...overrides};
}
const withPolicy=(data=fixtures[0],policy=attachedSupportPolicy())=>({...copy(data),learned_policy_available:true,exercise_policy:policy});
function elementsFrom(source){
  const elements=[],ids=new Map();
  for(const match of source.matchAll(/<([a-z][a-z0-9-]*)\b([^>]*)>/gi)){
    const attributes={};for(const attr of match[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g))attributes[attr[1]]=attr[2]??'';
    const element=new Element(match[1],attributes);elements.push(element);if(element.id)ids.set(element.id,element);
  }
  return {elements,ids};
}
function geometry(data,metrics){
  return {profile:'shared-world-state-v1',controlled_id:'human',objects:copy(data.scene.objects),target:null,
    steps:Math.round(metrics.simulated_duration_s*30),time_s:metrics.simulated_duration_s,source_time_s:metrics.simulated_duration_s,
    substeps:8,contact_details:[],contact_count:0,contact_summary:{human_object:0},coupled_metrics:{object_distance_world:9.876},
    human:{height:1.8,segments:[{id:'left_forearm',a:[.5,.5,.7],b:[.5,.5,.5],radius:.04}],muscles:[],
      control_mode:'direct_motor_actions',angles:Object.fromEntries(data.action_contract.names.map(n=>[n,0])),
      mobility:{floor_height:0,support_force:[0,0,0],translation:[0,0,0]}}};
}
function liveFixture(original=fixtures[0],metric='initial'){
  const data=copy(original),{elements,ids}=elementsFrom(html),requests=[],pending=[],timers=new Map();let timer=0,recording=false;
  let currentMetric=typeof metric==='string'?data.metrics[metric]:metric,geometryChange=()=>{};
  const state=()=>{const engine=geometry(data,currentMetric);geometryChange(engine);return {profile:'shared-exercise-state-v1',engine_state:engine,engine_state_status:'current',
    task_metrics:currentMetric,terminated:currentMetric.success||Boolean(currentMetric.failure),truncated:currentMetric.completed_interval&&!currentMetric.success&&!currentMetric.failure};};
  const status=()=>({status:'ready',profile:'shared-world',control_profile:'shared_exercise',scene:data.scene,exercise_task:data.task,
    action_contract:data.action_contract,learned_policy_available:data.learned_policy_available??false,exercise_policy:data.exercise_policy??null,snapshot:state(),recording,sessions:[],
    events:[],artifacts:['human-reference.json','repair-ledger.json','reconstruction.json','exercise-task.json',...(data.exercise_policy?['policy.json']:[])].map(name=>({name,bytes:100}))});
  const summary=new Element('summary'),header=new Element('tr'),hint=new Element('p'),equipmentWrap=new Element('div');
  ids.get('exercise-metrics-details').querySelector=selector=>selector==='summary'?summary:null;
  ids.get('exercise-equipment').closest=selector=>selector==='.table-wrap'?equipmentWrap:null;
  ids.get('exercise-contacts').closest=()=>({querySelector:()=>header});
  ids.get('exercise-results').querySelector=()=>hint;
  const select=selector=>selector==='.panel'?elements.filter(e=>e.classes.has('panel')):selector==='[data-action]'?elements.filter(e=>'action'in e.dataset)
    :selector==='[data-json]'?elements.filter(e=>'json'in e.dataset):selector==='[data-action][aria-pressed=true]'?elements.filter(e=>'action'in e.dataset&&e.getAttribute('aria-pressed')==='true')
    :selector.startsWith('#')?[ids.get(selector.slice(1))].filter(Boolean):assert.fail(`Unknown selector ${selector}`);
  ids.get('stages').querySelectorAll=()=>elements.filter(e=>'panel'in e.dataset);
  const document={hidden:false,getElementById:id=>{assert(ids.has(id),id);return ids.get(id);},createElement:tag=>new Element(tag),querySelectorAll:select,querySelector:s=>select(s)[0]||null,addEventListener(){}};
  const response=value=>({ok:true,status:200,headers:{get:()=>null},json:async()=>copy(value)});
  const fetch=async(path,options={})=>{
    const row={path,data:options.body?JSON.parse(options.body):null};requests.push(row);
    if(path==='/api/status')return response(status());
    if(path==='/api/inspection'||path==='/api/exercise-coverage')return response({status:'unavailable'});
    if(path.startsWith('/files/')){const name=decodeURIComponent(path.split('/').at(-1).split('?')[0]);
      if(name==='human-reference.json')return response({frames:[{...geometry(data,currentMetric),source_time_s:0,reference_projection:true}]});
      if(name==='exercise-task.json')return response(data.task);
      if(name==='repair-ledger.json')return response({proposals:[]});
      if(name==='reconstruction.json')return response({scope:'Authored test'});
      assert.fail(`Unexpected artifact ${path}`);}
    return new Promise(resolve=>pending.push({...row,resolve}));
  };
  vm.runInNewContext(script,{document,window:{addEventListener(){}},location:{hash:'',pathname:'/'},
    sessionStorage:{getItem:()=>null,setItem(){}},history:{replaceState(){}},URLSearchParams,AbortSignal,fetch,
    setTimeout:(callback,delay)=>{timers.set(++timer,{callback,delay});return timer;},clearTimeout:id=>timers.delete(id),performance:{now:()=>0},console},{filename:'lab.js'});
  return {ids,data,requests,pending,timers,summary,header,hint,equipmentWrap,
    changeGeometry(change){geometryChange=change;},
    async ready(){await drain();assert.match(ids.get('status').textContent,/ready/);elements.find(e=>e.dataset.panel==='play').click();await drain();},
    action(value){select('[data-action]').find(e=>e.dataset.action===String(value)&&!e.hidden).click();},
    open(){ids.get('exercise-metrics-details').open=true;ids.get('exercise-metrics-details').dispatch('toggle');},
    async reply(path,nextMetric=currentMetric){const i=pending.findIndex(r=>r.path===path);assert(i>=0,`${path} not pending`);const row=pending.splice(i,1)[0];
      currentMetric=typeof nextMetric==='string'?data.metrics[nextMetric]:nextMetric;
      if(path==='/api/record/start')recording=true;if(path==='/api/record/stop')recording=false;
      row.resolve(response({snapshot:state(),...(path.startsWith('/api/record/')?{recording}:{}),...(path==='/api/record/stop'?{export:'running'}:{})}));await drain();},
    async inject(nextMetric){ids.get('reset').click();await drain();await this.reply('/api/reset',nextMetric);},
  };
}

const replayHtml=fs.readFileSync(new URL('../src/weft/assets/shared-world-replay.html',import.meta.url),'utf8');
const replayScript=[...replayHtml.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)].at(-1)[1];
function replayFixture(original=fixtures[0],metrics=['initial','active','failed'],editEvents=()=>{}){
  const data=copy(original),events=metrics.map((value,i)=>{const m=typeof value==='string'?data.metrics[value]:value;
    return {type:i?'step':'start',controller:'manual',exercise_metrics:m,snapshot:geometry(data,m||data.metrics.initial),engine_state_status:'current'};});
  editEvents(events);
  const raw=JSON.stringify({scene:data.scene,exercise_task:data.task,events}),{ids}=elementsFrom(replayHtml),workers=[];let blob='';
  ids.get('data').textContent=raw;ids.get('data').remove=()=>{};ids.get('view').width=1080;ids.get('view').height=420;
  class Worker{constructor(){this.queue=[];this.context=vm.createContext({postMessage:value=>this.onmessage({data:value})});vm.runInContext(blob,this.context);workers.push(this);}
    postMessage(value){this.queue.push(value);}flush(){assert(this.queue.length);this.context.onmessage({data:this.queue.shift()});}terminate(){}}
  vm.runInNewContext(replayScript,{document:{hidden:false,getElementById:id=>{assert(ids.has(id),id);return ids.get(id);},createElement:tag=>new Element(tag),addEventListener(){}},
    window:{addEventListener(){}},location:{reload(){}},Blob:class{constructor(parts){blob=parts.join('');}},URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}},Worker,
    setTimeout(){throw Error('Inspection must not start playback');},clearTimeout(){},console});
  workers[0].flush();workers[0].flush();assert.equal(workers[0].queue.length,0);
  return {ids,raw,events,workers,select(index){ids.get('scrub').value=String(index);ids.get('scrub').dispatch('input');workers[0].flush();},
    turn(){ids.get('yaw').dispatch('input');}};
}

async function declaredSupportScopeAndUnknownInitialLoads(){
  const ui=liveFixture();await ui.ready();ui.open();
  assert.equal(ui.ids.get('toggle').disabled,false);assert.equal(ui.ids.get('actor-controls').hidden,true);
  assert.equal(ui.ids.get('joint').children.length,19);assert.equal(ui.ids.get('policy').disabled,true);
  assert.match(ui.ids.get('world-heading').textContent,/authored static supports/);
  assert.match(ui.ids.get('exercise-task-heading').textContent,/authored static support/);
  assert.match(ui.ids.get('exercise-assistance').textContent,/OFF.*19 joint/);
  assert.match(ui.summary.textContent,/declared support loads/);assert(ui.equipmentWrap.hidden);
  assert.deepEqual(ui.header.children.map(e=>e.textContent),['Body segment / static support','Authored load requirement','Observed coverage','Unqualified gaps','Current load / result']);
  const details=content(ui.ids.get('exercise-contacts'));
  assert.match(details,/left forearm.*Authored support/);assert.match(details,/minimum 5\.0% of simulated body weight/);
  assert.match(details,/Current unavailable.*peak unavailable/);assert.match(details,/Qualifying fraction unavailable/);
  assert(details.includes(ui.data.task.source_scope.supports[0].object_sha256));
  assert.doesNotMatch(ui.ids.get('exercise-equipment-summary').textContent,/Maximum displacement|0\.000.*drift/);
  assert(!ui.ids.get('live').drawCalls.some(c=>c.name==='setLineDash'&&c.args[0]?.length));
  assert(!ui.requests.some(r=>r.path==='/api/step'||r.path==='/api/actor'));
}
async function manualCoastAndReferenceKeepExactHumanOnlyPayloads(){
  for(const data of fixtures){const ui=liveFixture(data);await ui.ready();
    assert.equal(ui.ids.get('joint').children.length,data.action_contract.names.length);
    ui.action(1);await drain();const request=ui.pending.find(r=>r.path==='/api/step');assert(request);
    assert.equal(request.data.action.length,data.action_contract.names.length);assert.equal(request.data.action[0],.25);assert(request.data.action.slice(1).every(v=>v===0));
    await ui.reply('/api/step','active');
    ui.ids.get('mode-coast').click();ui.ids.get('step').click();await drain();assert.deepEqual(ui.pending[0].data,{steps:1,controller:'coast'});await ui.reply('/api/step');
    ui.ids.get('reference-servo').click();ui.ids.get('step').click();await drain();assert.deepEqual(ui.pending[0].data,{steps:1,controller:'reference'});await ui.reply('/api/step');
    assert(!ui.requests.some(r=>r.path==='/api/actor'||r.path==='/api/joint'));
  }
}
async function recordResetAndExportStillUseExistingEndpoints(){
  const ui=liveFixture();await ui.ready();ui.ids.get('record').click();await drain();assert(ui.pending.some(r=>r.path==='/api/record/start'));
  await ui.reply('/api/record/start','initial');assert.equal(ui.ids.get('record').disabled,true);assert.equal(ui.ids.get('actor').disabled,true);
  ui.ids.get('reset').click();await drain();assert(!ui.pending.some(r=>r.path==='/api/reset'));
  await ui.reply('/api/step','active');await ui.reply('/api/reset','initial');
  ui.ids.get('stop-record').click();await drain();assert(ui.pending.some(r=>r.path==='/api/record/stop'));await ui.reply('/api/record/stop','active');
  assert(!ui.requests.some(r=>/train|replay|policy\.json/.test(r.path)));
}
async function supportPolicySelectionDisclosesTrainingAndUnchangedSupports(){
  for(const data of fixtures)for(const kind of ['searched_joint_residuals','authored_unchanged']){
    const ui=liveFixture(withPolicy(data,attachedSupportPolicy(kind)));await ui.ready();
    assert.equal(ui.ids.get('policy').disabled,false);assert.equal(ui.ids.get('start-learned').disabled,false);
    assert.equal(ui.ids.get('policy').textContent,kind==='authored_unchanged'?'Selected authored baseline':'Searched joint corrections');
    assert.match(ui.ids.get('exercise-policy-summary').textContent,kind==='authored_unchanged'
      ?/unchanged programmed feedback, not new learned gains/:/phase-dependent joint corrections.*root assistance remains programmed, not learned/);
    assert.match(ui.ids.get('learning-description').textContent,/simulator-state and reference features, not video pixels/);
    assert.match(ui.ids.get('start-help').textContent,/attached selected policy/);
    assert.equal(ui.ids.get('joint').children.length,data.action_contract.names.length);
    assert.equal(ui.ids.get('actor-controls').hidden,true);assert.equal(ui.ids.get('actor').disabled,true);
    assert.match(ui.ids.get('exercise-evaluation-summary').textContent,/matched starts, frozen policy/);
    assert.match(ui.ids.get('exercise-evaluation-summary').textContent,/human-only perturbations.*static supports keep their exact scene poses.*not a changed-support test/);
    assert.equal(ui.ids.get('scores').children.length,4);assert.match(content(ui.ids.get('scores')),/1\/2/);
    assert.match(ui.ids.get('exercise-outcome').textContent,/In progress/);
    assert.equal(ui.ids.get('learning-curve').hidden,true);assert.equal(ui.ids.get('exercise-policy-links').children.length,1);
    ui.ids.get('policy').click();ui.ids.get('step').click();await drain();
    assert.deepEqual(ui.pending[0].data,{steps:1,policy:true});await ui.reply('/api/step','active');
    assert.match(ui.ids.get('mode-description').textContent,/Equipment has no force controls/);
    assert(!ui.requests.some(r=>/\/api\/(actor|joint)|\/(policy|training|rl-evaluation)\.json/.test(r.path)));
  }
  const unevaluated=liveFixture(withPolicy(fixtures[0],attachedSupportPolicy('authored_unchanged',{evaluation_status:'not_attached',evaluation_successes:null,evaluation_reset_protocol:null})));
  await unevaluated.ready();assert.equal(unevaluated.ids.get('policy').disabled,false);
  assert.equal(unevaluated.ids.get('scores').children.length,0);assert.match(unevaluated.ids.get('exercise-evaluation-summary').textContent,/Independent evaluation unavailable.*Training is not an evaluation pass/);
}
async function invalidSupportPolicyDisablesOnlyLearnedControls(){
  for(const policy of [attachedSupportPolicy('searched_joint_residuals',{profile:'shared-exercise-policy-v2'}),attachedSupportPolicy('searched_joint_residuals',{policy_sha256:'invalid'})]){
    const ui=liveFixture(withPolicy(fixtures[0],policy));await ui.ready();
    assert.equal(ui.ids.get('policy').disabled,true);assert.equal(ui.ids.get('start-learned').disabled,true);
    for(const id of ['mode-manual','mode-coast','reference-servo','record','toggle'])assert.equal(ui.ids.get(id).disabled,false,id);
    assert.equal(ui.ids.get('exercise-policy-links').hidden,true);assert.equal(ui.ids.get('scores').children.length,0);
    assert.match(ui.ids.get('exercise-policy-summary').textContent,/No validated human-only policy/);
    const requests=ui.requests.length;ui.ids.get('policy').click();ui.ids.get('start-learned').click();await drain();assert.equal(ui.requests.length,requests);
    ui.action(1);await drain();assert.equal(ui.pending[0].data.action.length,19);assert.equal(ui.pending[0].data.action[0],.25);
    assert(ui.pending[0].data.action.slice(1).every(value=>value===0));await ui.reply('/api/step','active');
    assert.match(ui.ids.get('exercise-outcome').textContent,/In progress/);
  }
  const wrongScene=exerciseFixture('shared-world',{sharedExercise:true,exercisePolicyAvailable:true,exercisePolicy:attachedSupportPolicy()});
  await wrongScene.ready();assert.equal(wrongScene.ids.get('policy').disabled,true);assert.equal(wrongScene.ids.get('reference-servo').disabled,false);
}
async function learnedSupportControlsSerializeFreshStartsAndRecording(){
  const ui=liveFixture(withPolicy());await ui.ready();ui.open();
  ui.ids.get('start-learned').click();ui.ids.get('start-learned').click();await drain();
  assert.deepEqual(ui.pending.map(r=>[r.path,r.data]),[['/api/reset',{}]]);
  assert.equal(ui.ids.get('record').disabled,true);assert.equal(ui.ids.get('policy').disabled,true);
  await ui.reply('/api/reset');assert.deepEqual(ui.pending.map(r=>r.data),[{steps:3,policy:true}]);
  ui.ids.get('step').click();ui.ids.get('step').click();ui.ids.get('record').click();await drain();
  assert.deepEqual(ui.pending.map(r=>r.path),['/api/step']);assert.equal(ui.ids.get('start-learned').disabled,true);
  await ui.reply('/api/step','active');assert.deepEqual(ui.pending.map(r=>[r.path,r.data]),[['/api/record/start',{}]]);
  await ui.reply('/api/record/start','initial');assert.deepEqual(ui.pending.map(r=>r.data),[{steps:3,policy:true}]);
  assert.equal(ui.ids.get('record').disabled,true);assert.equal(ui.ids.get('joint').disabled,true);
  ui.ids.get('reset').click();ui.ids.get('reset').click();await drain();assert.deepEqual(ui.pending.map(r=>r.path),['/api/step']);
  await ui.reply('/api/step','active');await ui.reply('/api/reset','initial');
  assert.match(ui.ids.get('record-status').textContent,/Recording/);
  let row;const requests=ui.requests.length,started=performance.now();
  for(let i=0;i<50;i++){
    ui.ids.get('step').click();ui.ids.get('step').click();await drain();assert.deepEqual(ui.pending.map(r=>r.data),[{steps:1,policy:true}]);
    await ui.reply('/api/step','active');row??=ui.ids.get('exercise-contacts').children[0];assert.equal(ui.ids.get('exercise-contacts').children[0],row);
  }
  assert.equal(ui.requests.length-requests,50);assert(ui.timers.size<=1);assert.equal(ui.pending.length,0);
  console.log(`MEASURE ${JSON.stringify({case:'same-support-metrics-learned-steps',updates:50,elapsed_ms:Number((performance.now()-started).toFixed(3)),requests:50,max_pending_steps:1,retained_timers:ui.timers.size,reused_detail_row:true,scope:'scripted DOM only; no browser or physics'})}`);
  ui.ids.get('step').click();ui.ids.get('stop-record').click();ui.ids.get('stop-record').click();await drain();
  assert.deepEqual(ui.pending.map(r=>r.path),['/api/step']);assert.equal(ui.ids.get('policy').disabled,true);
  await ui.reply('/api/step','active');assert.deepEqual(ui.pending.map(r=>[r.path,r.data]),[['/api/record/stop',{}]]);
  await ui.reply('/api/record/stop','active');assert.equal(ui.pending.length,0);assert.match(ui.ids.get('record-status').textContent,/Saving recording/);
  const mutations=ui.requests.filter(r=>r.data!==null);
  assert(mutations.every(r=>['/api/reset','/api/step','/api/record/start','/api/record/stop'].includes(r.path)));
  assert(mutations.filter(r=>r.path==='/api/step').every(r=>r.data.policy===true&&Object.keys(r.data).length===2));
  assert.equal(mutations.filter(r=>r.path==='/api/record/start').length,1);assert.equal(mutations.filter(r=>r.path==='/api/record/stop').length,1);
}
async function realActiveAndTerminalMetricsStaySeparateFromLegacyScores(){
  const ui=liveFixture();await ui.ready();ui.open();await ui.inject('active');
  assert.match(content(ui.ids.get('exercise-contacts')),/Current 10\.0%.*qualifying now/);
  assert.match(content(ui.ids.get('exercise-contacts')),/50\.0% qualifying/);
  assert.match(ui.ids.get('exercise-outcome').textContent,/In progress/);
  await ui.inject('failed');assert.match(ui.ids.get('exercise-outcome').textContent,/Not passed.*failed gates/);
  assert.match(content(ui.ids.get('exercise-contacts')),/Current 0\.0%.*active, not qualifying/);
  assert.match(ui.ids.get('exercise-gates').textContent,/support_requirement:0/);
  await ui.inject('passed');assert.match(ui.ids.get('exercise-outcome').textContent,/Passed.*complete/);
}
async function malformedOrCrossProfileSupportCannotEnableControls(){
  const changes=[d=>d.task.support_requirements=[],d=>d.task.profile='shared-exercise-task-v1',
    d=>d.task.source_scope.supports[0].object_sha256='bad',d=>d.task.support_requirements[0].human_segment='left_palm',
    d=>d.scene.objects[0].role='passive',d=>d.action_contract.names.push('object_x')];
  for(const change of changes){const data=copy(fixtures[0]);change(data);const ui=liveFixture(data);await ui.ready();
    assert.equal(ui.ids.get('toggle').disabled,true);assert.equal(ui.ids.get('record').disabled,true);
    assert.match(ui.ids.get('exercise-outcome').textContent,/unavailable/);assert(!ui.requests.some(r=>r.path==='/api/step'));
  }
}
async function missingOrForeignMetricsPauseWithoutInventingZeros(){
  for(const change of [m=>delete m.support_requirements,m=>delete m.support_requirements[0].load_body_weight_fraction,
    m=>m.support_requirements[0].minimum_load_body_weight_fraction=.1,m=>m.equipment.push({object_id:'fake'}),
    m=>m.gate_applicability.declared_support='not_applicable',m=>m.source_scope.supports[0].object_sha256='b'.repeat(64)]){
    const ui=liveFixture();await ui.ready();const metric=copy(ui.data.metrics.active);change(metric);await ui.inject(metric);
    assert.equal(ui.ids.get('toggle').disabled,true);assert.match(ui.ids.get('exercise-duration').textContent,/Unavailable; no zero/);
    assert.equal(ui.ids.get('exercise-contacts').children.length,0);
  }
}
async function repeatedDetailsAreStableAndTextIsSafe(){
  const data=copy(fixtures[0]);data.scene.objects[0].label='<img src=x onerror=alert(1)>';
  const ui=liveFixture(data);await ui.ready();ui.open();const first=ui.ids.get('exercise-contacts').children[0],requests=ui.requests.length;
  for(let i=0;i<100;i++)ui.ids.get('exercise-metrics-details').dispatch('toggle');
  assert.equal(ui.ids.get('exercise-contacts').children[0],first);assert.equal(ui.requests.length,requests);
  assert.match(content(first),/<img src=x onerror=alert\(1\)>/);
  assert(first.children.every(e=>e.tagName==='TD'));
}
function replayShowsSavedSupportLoadsAndNoObjectGoal(){
  const ui=replayFixture();assert.match(ui.ids.get('recording-title').textContent,/Authored static-support/);
  assert.match(content(ui.ids.get('support-requirements')),/Current load unavailable.*peak unavailable/);
  assert.match(ui.ids.get('exercise-scope').textContent,/disabled.*19 joint motors/);
  assert(!ui.ids.get('view').drawCalls.some(c=>c.name==='setLineDash'&&c.args[0]?.length));
  ui.select(1);assert.match(content(ui.ids.get('support-requirements')),/Current load 10\.0%.*peak 10\.0%/);
  assert.match(ui.ids.get('status').textContent,/in progress.*incomplete/);assert.doesNotMatch(ui.ids.get('status').textContent,/drift|object dist/);
  ui.select(2);assert.match(ui.ids.get('status').textContent,/not passed.*complete/);assert.match(content(ui.ids.get('support-requirements')),/longest gap 0\.067s/);
}
function replayMalformedMetricIsUnavailableNotASuccess(){
  for(const change of [m=>delete m.support_requirements,m=>delete m.simulated_duration_s,m=>m.simulated_duration_s='bad',
    m=>delete m.support_requirements[0].qualifying_fraction,m=>m.support_requirements[0].qualifying_fraction=null,
    m=>m.source_scope.supports[0].object_sha256='b'.repeat(64),m=>m.gate_applicability.declared_support='not_applicable',m=>m.equipment.push({})]){
    const m=copy(fixtures[0].metrics.passed);change(m);const ui=replayFixture(fixtures[0],[m]);
    assert.match(ui.ids.get('status').textContent,/unavailable/);assert.equal(ui.ids.get('support-requirements').children.length,0);
  }
}
async function unknownCoverageCannotClaimSatisfiedSupport(){
  const ui=liveFixture();await ui.ready();const m=copy(ui.data.metrics.passed);m.support_requirements[0].qualifying_fraction=null;await ui.inject(m);
  assert.equal(ui.ids.get('toggle').disabled,true);assert.match(ui.ids.get('exercise-outcome').textContent,/unavailable/);
}
const malformedObjects=[state=>delete state.objects,state=>state.objects=null,state=>state.objects=[],
  state=>state.objects=[null],state=>state.objects.push(copy(state.objects[0])),state=>state.objects[0].id='other',
  state=>state.objects[0].role='passive',state=>state.objects[0].position[2]+=.1,state=>state.objects[0].quaternion=[0,1,0,0]];
async function malformedStaticObjectsPauseLiveControls(){
  for(const change of malformedObjects){const ui=liveFixture();await ui.ready();ui.open();ui.changeGeometry(change);await ui.inject('active');
    assert.equal(ui.ids.get('toggle').disabled,true);assert.equal(ui.ids.get('record').disabled,true);
    assert.match(ui.ids.get('exercise-outcome').textContent,/unavailable/);assert.equal(ui.ids.get('exercise-contacts').children.length,0);
    assert(!ui.requests.some(r=>r.path==='/api/step'));
  }
  const data=copy(fixtures[0]);data.scene.objects=[null];const ui=liveFixture(data);await ui.ready();
  assert.equal(ui.ids.get('toggle').disabled,true);assert.match(content(ui.ids.get('objects')),/geometry unavailable/);
}
function changedReplayObjectsInvalidateCachedSupport(){
  for(const change of malformedObjects){const ui=replayFixture(fixtures[0],['passed','passed'],events=>change(events[1].snapshot));
    assert.match(ui.ids.get('status').textContent,/passed.*complete/);ui.select(1);
    assert.match(ui.ids.get('status').textContent,/unavailable/);assert.equal(ui.ids.get('support-requirements').children.length,0);
    assert.equal(ui.workers[0].queue.length,0);
  }
  const data=copy(fixtures[0]);data.scene.objects=[null];const ui=replayFixture(data,['passed']);
  assert.match(ui.ids.get('status').textContent,/unavailable/);assert.equal(ui.ids.get('support-requirements').children.length,0);
}
function replayRepeatedViewChangesDoNotRebuildMetricsOrAdvance(){
  const ui=replayFixture(fixtures[1]);const first=ui.ids.get('support-requirements').children[0];
  for(let i=0;i<100;i++)ui.turn();assert.equal(ui.ids.get('support-requirements').children[0],first);
  assert.equal(ui.workers[0].queue.length,0);assert.equal(ui.ids.get('scrub').value,'0');
  assert.match(ui.ids.get('exercise-scope').textContent,/22 channels/);
}

const checks=[declaredSupportScopeAndUnknownInitialLoads,manualCoastAndReferenceKeepExactHumanOnlyPayloads,
  supportPolicySelectionDisclosesTrainingAndUnchangedSupports,invalidSupportPolicyDisablesOnlyLearnedControls,learnedSupportControlsSerializeFreshStartsAndRecording,
  recordResetAndExportStillUseExistingEndpoints,realActiveAndTerminalMetricsStaySeparateFromLegacyScores,
  malformedOrCrossProfileSupportCannotEnableControls,missingOrForeignMetricsPauseWithoutInventingZeros,unknownCoverageCannotClaimSatisfiedSupport,repeatedDetailsAreStableAndTextIsSafe,
  replayShowsSavedSupportLoadsAndNoObjectGoal,replayMalformedMetricIsUnavailableNotASuccess,replayRepeatedViewChangesDoNotRebuildMetricsOrAdvance,
  malformedStaticObjectsPauseLiveControls,changedReplayObjectsInvalidateCachedSupport];
const selected=new Set(process.argv.slice(2));
assert([...selected].every(name=>checks.some(check=>check.name===name)),'Unknown selected check');
for(const check of checks.filter(check=>!selected.size||selected.has(check.name))){
  await check();console.log(`PASS ${check.name}`);
}
