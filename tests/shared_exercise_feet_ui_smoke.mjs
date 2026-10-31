/** Existing lab/replay UI with saved analytical task fixtures; no browser or native work. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {performance} from 'node:perf_hooks';
import {pathToFileURL} from 'node:url';
import {Element, html, script, drain} from './lab_ui_smoke.mjs';

const direct=process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href;
const fixtures=direct?JSON.parse(fs.readFileSync(process.argv[2], 'utf8')):[];
if(direct)assert.equal(fixtures.length,6);
const copy=value=>JSON.parse(JSON.stringify(value));
const content=node=>[node.textContent,...node.children.map(content)].join('\n');
function elementsFrom(source){
  const elements=[],ids=new Map();
  for(const match of source.matchAll(/<([a-z][a-z0-9-]*)\b([^>]*)>/gi)){
    const attributes={};for(const attr of match[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g))attributes[attr[1]]=attr[2]??'';
    const element=new Element(match[1],attributes);elements.push(element);if(element.id)ids.set(element.id,element);
  }
  return {elements,ids};
}
function state(data,name){
  const value=copy(typeof name==='string'?data.states[name]:name),engine=value.engine_state;
  if(data.actual_saved)return value;
  // The task fixture deliberately lacks renderer geometry; these authored lines test only drawing/presentation.
  Object.assign(engine,{controlled_id:'human',source_time_s:engine.time_s,substeps:data.scene.substeps,contact_count:0,
    contact_details:[],contact_summary:{human_object:0},coupled_metrics:{},target:data.scene.target});
  Object.assign(engine.human,{height:1.8,segments:[{id:'left_foot',a:[1,1,.1],b:[1,1.1,.1],radius:.04}],muscles:[],
    control_mode:'direct_motor_actions',angles:Object.fromEntries(data.action_contract.names.map(name=>[name,0])),
    mobility:{floor_height:0,support_force:[0,0,0],translation:[0,0,0]}});
  return value;
}
function liveFixture(original=fixtures[0],change=()=>{}){
  const data=copy(original);change(data);
  const {elements,ids}=elementsFrom(html),requests=[],pending=[],timers=new Map();let timer=0,recording=false;
  let currentState=state(data,'initial');
  const status=()=>({status:'ready',profile:'shared-world',control_profile:'shared_exercise',scene:data.scene,exercise_task:data.task,
    action_contract:data.action_contract,capabilities:data.capabilities,learned_policy_available:true,
    exercise_policy:{profile:'shared-exercise-policy-v4',policy_sha256:'c'.repeat(64)},appearance:{enabled:true},snapshot:currentState,
    recording,sessions:[],events:[],artifacts:['human-reference.json','repair-ledger.json','reconstruction.json','exercise-task.json'].map(name=>({name,bytes:100})),
    ...(data.status_overlay||{})});
  ids.get('exercise-metrics-details').querySelector=()=>new Element('summary');
  ids.get('exercise-equipment').closest=()=>new Element('div');
  ids.get('exercise-contacts').closest=()=>({querySelector:()=>new Element('tr')});
  ids.get('exercise-results').querySelector=()=>new Element('p');
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
      if(name==='human-reference.json')return response({frames:[{...currentState.engine_state,source_time_s:0,reference_projection:true}]});
      if(name==='exercise-task.json')return response(data.task);
      if(name==='repair-ledger.json')return response({proposals:[]});
      if(name==='reconstruction.json')return response({scope:'Authored analytical test'});
      assert.fail(`Unexpected artifact ${path}`);}
    return new Promise(resolve=>pending.push({...row,resolve}));
  };
  const context={document,window:{addEventListener(){}},location:{hash:'',pathname:'/'},sessionStorage:{getItem:()=>null,setItem(){}},history:{replaceState(){}},
    URLSearchParams,AbortSignal,fetch,setTimeout:(callback,delay)=>{timers.set(++timer,{callback,delay});return timer;},clearTimeout:id=>timers.delete(id),performance:{now:()=>0},console,
    WeftLabAppearance:{create(){assert.fail('Feet avatars are unsupported');}}};
  vm.runInNewContext(script.replace(/\}\)\(\);\s*$/, 'globalThis.feetTest={updateExerciseSnapshot,recordingEvidenceValid,applyRecordingEvidence};})();'),context,{filename:'lab.js'});
  return {ids,data,requests,pending,timers,context,
    async ready(){await drain();assert.match(ids.get('status').textContent,/ready/);elements.find(e=>e.dataset.panel==='play').click();await drain();},
    action(){select('[data-action]').find(e=>e.dataset.action==='1'&&!e.hidden).click();},
    motor(index){ids.get('joint').value=data.action_contract.names[index];ids.get('joint').dispatch('change');},
    update(name){currentState=state(data,name);context.feetTest.updateExerciseSnapshot(currentState);},
    async reply(path,name='active'){const i=pending.findIndex(row=>row.path===path);assert(i>=0,path);const row=pending.splice(i,1)[0];currentState=state(data,name);
      if(path==='/api/record/start')recording=true;if(path==='/api/record/stop')recording=false;
      row.resolve(response({snapshot:currentState,...(path.startsWith('/api/record/')?{recording}:{}),...(path==='/api/record/stop'?{export:'running'}:{})}));await drain();},
    panel(){return ids.get('exercise-results').children.find(row=>row.children.some(child=>child.textContent==='Unscored ankles and foot contacts'));},
  };
}

async function controlsUseExplicitAnklesAndPreserveBaseCriteria(){
  for(const data of fixtures){const ui=liveFixture(data);await ui.ready();
    assert.equal(ui.ids.get('toggle').disabled,false);assert.equal(ui.ids.get('joint').children.length,data.action_contract.names.length);
    assert.equal(ui.ids.get('actor-controls').hidden,true);assert.equal(ui.ids.get('policy').disabled,true);assert.equal(ui.ids.get('start-learned').disabled,true);
    assert.match(ui.ids.get('exercise-assistance').textContent,/25 joint motors.*19 body \+ 6 ankle/);
    assert.match(ui.ids.get('play-boundary').textContent,/19-joint.*unchanged.*no score or support credit/);
    ui.motor(19);ui.action();await drain();const action=ui.pending[0].data.action;
    assert.equal(action.length,data.action_contract.names.length);assert.equal(action[19],.25);assert(action.every((v,i)=>i===19||v===0));
    await ui.reply('/api/step');ui.ids.get('mode-coast').click();ui.ids.get('step').click();await drain();assert.deepEqual(ui.pending[0].data,{steps:1,controller:'coast'});await ui.reply('/api/step');
    ui.ids.get('reference-servo').click();assert.match(ui.ids.get('mode-description').textContent,/original-start.*do not follow video/);
    ui.ids.get('step').click();await drain();assert.deepEqual(ui.pending[0].data,{steps:1,controller:'reference'});await ui.reply('/api/step');
    assert(!ui.requests.some(row=>/api\/(actor|joint|appearance)/.test(row.path)));
  }
}
async function identityAndMappingMismatchesPauseWithoutLegacyFallback(){
  for(const change of [data=>data.task.base_scene_sha256='0'.repeat(64),data=>data.action_contract.engine_motor_indices[19]=19,
    data=>data.states.initial.base_task_sha256='0'.repeat(64),data=>data.states.initial.ankle_diagnostics.scored=true]){
    const ui=liveFixture(fixtures[0],change);await ui.ready();assert.equal(ui.ids.get('toggle').disabled,true);assert.equal(ui.ids.get('record').disabled,true);
    assert.match(ui.ids.get('exercise-outcome').textContent,/unavailable/);ui.action();await drain();assert(!ui.pending.length);
  }
}
async function diagnosticsSeparateUnknownAcceptedAndFailedClocks(){
  const ui=liveFixture(fixtures.find(data=>data.scene.base_profile==='shared-world-v4'));await ui.ready();
  assert.match(content(ui.panel()),/mean unavailable.*No substep observed yet/s);
  ui.update('active');assert.match(content(ui.panel()),/8 accepted \/ 8 observed.*8 excluded foot\/object/);
  const details=ui.panel().children.find(row=>row.tagName==='DETAILS');details.open=true;details.dispatch('toggle');
  assert.match(content(details),/left foot.*8 records/);
  ui.update('failed');assert.match(content(ui.panel()),/11 accepted \/ 12 observed.*failed\/invalid.*last valid boundary/);
  assert.match(ui.ids.get('exercise-failure').textContent,/last valid geometry/);assert.equal(ui.ids.get('step').disabled,true);
  assert(!ui.ids.get('exercise-body-errors').textContent.includes('0.300'));
}
async function recordingSerializesActionsAndResets(){
  const ui=liveFixture(fixtures[1]);await ui.ready();ui.ids.get('record').click();await drain();assert.equal(ui.pending[0].path,'/api/record/start');
  await ui.reply('/api/record/start','initial');assert.equal(ui.pending.filter(row=>row.path==='/api/step').length,1);
  ui.ids.get('reset').click();await drain();assert(!ui.pending.some(row=>row.path==='/api/reset'));
  await ui.reply('/api/step');await ui.reply('/api/reset','initial');
  ui.ids.get('stop-record').click();await drain();assert(ui.pending.some(row=>row.path==='/api/record/stop'));await ui.reply('/api/record/stop');
  assert(!ui.requests.some(row=>/train|appearance|policy.json/.test(row.path)));
}
function receipt(ui){
  const name='session-123456abcdef',hash='a'.repeat(64),pins=ui.data.capabilities.task_contract.identity;
  const identities={source_run_manifest_sha256:hash,recording_manifest_sha256:hash,receipt_manifest_sha256:hash,receipt_sha256:hash,
    recording_scene_sha256:hash,engine_scene_sha256:pins.scene_sha256,task_sha256:pins.task_sha256,policy_sha256:null,
    ...Object.fromEntries(['base_scene_sha256','base_task_sha256','projection_contract_sha256','adapter_implementation_sha256','foot_reference_sha256'].map(key=>[key,pins[key]]))};
  return {schema_version:1,profile:'recording-evidence-v1',status:'available',session:name,partition:'exposed_development',physical_validation:'unverified',identities,
    recorded:{events:2,actions:1,resets:0},replay:{status:'passed',protocol:'shared-exercise-feet-full-state-replay-v1',events_checked:2,actions_checked:1,resets_checked:0,learned_actions_verified:null},
    verified_metadata:['run/manifest.json','run/scene.json','run/exercise-task.json',`sessions/${name}/manifest.json`,`sessions/${name}/scene.json`,`sessions/${name}/evidence/manifest.json`,`sessions/${name}/evidence/receipt.json`].map(path=>({path,source_name:path,sha256:hash,bytes:1}))};
}
async function receiptRequiresFeetProtocolAndAllBoundIdentities(){
  const ui=liveFixture();await ui.ready();const saved=receipt(ui),api=ui.context.feetTest;
  assert(api.recordingEvidenceValid(saved,saved.session));api.applyRecordingEvidence(saved);
  assert.match(ui.ids.get('recording-evidence-replay').textContent,/unscored ankle diagnostics matched/);
  for(const mutate of [value=>value.replay.protocol='shared-exercise-exact-action-state-task-v1',value=>value.identities.foot_reference_sha256='0'.repeat(64),
    value=>value.identities.policy_sha256='0'.repeat(64)]){const changed=copy(saved);mutate(changed);assert.equal(api.recordingEvidenceValid(changed,changed.session),false);}
}

const replayHtml=fs.readFileSync(new URL('../src/weft/assets/shared-world-replay.html',import.meta.url),'utf8');
const replayScript=[...replayHtml.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)].at(-1)[1];
function replayFixture(data){
  const events=data.replay_bundle?.events??['initial','active','failed'].map((name,i)=>{const snapshot=state(data,name);return {type:i?'step':'start',controller:'manual',snapshot:snapshot.engine_state,
    exercise_metrics:snapshot.task_metrics,ankle_diagnostics:snapshot.ankle_diagnostics,engine_state_status:snapshot.engine_state_status};});
  const {ids}=elementsFrom(replayHtml),workers=[];let blob='';ids.get('data').textContent=JSON.stringify(data.replay_bundle??{scene:data.scene,exercise_task:data.task,events});ids.get('data').remove=()=>{};
  class Worker{constructor(){this.queue=[];this.context=vm.createContext({postMessage:value=>this.onmessage({data:value})});vm.runInContext(blob,this.context);workers.push(this);}
    postMessage(value){this.queue.push(value);}flush(){this.context.onmessage({data:this.queue.shift()});}terminate(){}}
  vm.runInNewContext(replayScript,{document:{hidden:false,getElementById:id=>{assert(ids.has(id),id);return ids.get(id);},createElement:tag=>new Element(tag),addEventListener(){}},
    window:{addEventListener(){}},location:{reload(){}},Blob:class{constructor(parts){blob=parts.join('');}},URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}},Worker,
    setTimeout(){throw Error('Replay inspection cannot start playback');},clearTimeout(){},console});
  workers[0].flush();workers[0].flush();
  return {ids,select(index){ids.get('scrub').value=String(index);ids.get('scrub').dispatch('input');workers[0].flush();},turn(){ids.get('yaw').dispatch('input');}};
}
function savedReplayPreservesBaseSupportAndUnscoredFeet(){
  for(const data of fixtures){const replay=replayFixture(data);assert.match(replay.ids.get('recording-title').textContent,/Body and feet/);
    assert.match(replay.ids.get('feet-status').textContent,/mean error unavailable/);replay.select(1);assert.match(replay.ids.get('feet-status').textContent,/8 accepted \/ 8 observed/);
    if(data.scene.base_profile==='shared-world-v4')assert.match(replay.ids.get('support-status').textContent,/simulated/);
    replay.select(2);assert.match(replay.ids.get('feet-status').textContent,/last valid boundary/);
  }
  const data=copy(fixtures.find(value=>value.scene.base_profile==='shared-world-v4'));data.states.initial.ankle_diagnostics.foot_support_credit=true;
  const invalid=replayFixture(data);assert.match(invalid.ids.get('status').textContent,/unavailable/);assert.equal(invalid.ids.get('support-panel').hidden,true);
}
async function repeatedUpdatesKeepDiagnosticNodesAndUnchangedText(){
  const ui=liveFixture(fixtures.find(data=>data.scene.base_profile==='shared-world-v4'));await ui.ready();ui.update('active');
  const panel=ui.panel(),details=panel.children.at(-1);details.open=true;details.dispatch('toggle');
  const tracked=[panel.children[1],panel.children[2],details.children[1]];let writes=0;
  for(const element of tracked){let saved=element.textContent;Object.defineProperty(element,'textContent',{get:()=>saved,set:value=>{writes++;saved=value;}});}
  const start=performance.now();for(let i=0;i<200;i++)ui.update('active');const liveMs=performance.now()-start;
  assert.equal(ui.panel(),panel);assert.equal(writes,0);assert.equal(ui.pending.length,0);
  const replay=replayFixture(ui.data);replay.select(1);const output=replay.ids.get('feet-status');let saved=output.textContent,replayWrites=0;
  Object.defineProperty(output,'textContent',{get:()=>saved,set:value=>{saved=value;replayWrites++;}});
  const replayStart=performance.now();for(let i=0;i<200;i++)replay.turn();const replayMs=performance.now()-replayStart;assert.equal(replayWrites,0);
  console.log(JSON.stringify({measurement:'scripted_200_unchanged_updates',live_ms:liveMs,replay_ms:replayMs,diagnostic_text_writes:writes,replay_diagnostic_text_writes:replayWrites,scope:'scripted DOM only, not browser/native performance'}));
}

const checks=[controlsUseExplicitAnklesAndPreserveBaseCriteria,identityAndMappingMismatchesPauseWithoutLegacyFallback,
  diagnosticsSeparateUnknownAcceptedAndFailedClocks,recordingSerializesActionsAndResets,receiptRequiresFeetProtocolAndAllBoundIdentities,
  savedReplayPreservesBaseSupportAndUnscoredFeet,repeatedUpdatesKeepDiagnosticNodesAndUnchangedText];
export {liveFixture,replayFixture,content};
if(direct){
  const requested=new Set(process.argv.slice(3));let failures=0;
  for(const check of checks.filter(check=>!requested.size||requested.has(check.name))){try{await check();console.log(`PASS ${check.name}`);}catch(error){failures++;console.error(`FAIL ${check.name}: ${error.stack}`);}}
  if(failures)process.exitCode=1;
}
