/** Saved-shaped analytical DOM/worker checks; no native replay, browser or source admission. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {performance} from 'node:perf_hooks';
import {pathToFileURL} from 'node:url';
import {cameraFixture, replayFixture} from './camera_shared_ui_smoke.mjs';
import {policyFixture} from './camera_shared_policy_ui_smoke.mjs';

const hash=char=>char.repeat(64),copy=value=>JSON.parse(JSON.stringify(value));
function referenceFixture(roots=false){
  const data=cameraFixture(roots),bundle=data.replay_bundle;
  bundle.reference_controller={profile:'camera-source-fit-controller-view-v1',sha256:hash('e'),
    kind:'source_fit_authored_controller',scene_sha256:bundle.camera_identity.scene_sha256,
    task_sha256:bundle.camera_identity.task_sha256,gates_sha256:bundle.exercise_task.gates_sha256,
    physical_validation:'unverified',scope:'Fitted joint controls; original camera, body, task and scoring remain unchanged'};
  Object.assign(bundle.events[1],{controller:'reference_proposal',reference_controller_sha256:hash('e'),
    action:Array(roots?22:19).fill(0)});
  return data;
}
function paused(replay){
  assert.equal(replay.ids.get('play').disabled,true);
  assert.match(replay.ids.get('camera-status').textContent,/unavailable/);
  assert.match(replay.ids.get('controller').textContent,/unavailable/);
}
function referenceReplayPreservesCameraIdentityAndGates(){
  for(const roots of [false,true]){
    const data=referenceFixture(roots),before=JSON.stringify(data),replay=replayFixture(data);
    assert.equal(replay.ids.get('play').disabled,false);
    assert.match(replay.ids.get('camera-status').textContent,/mean unavailable/);
    replay.select(1);assert.equal(replay.ids.get('play').disabled,false);
    assert.match(replay.ids.get('controller').textContent,/fitted-reference.*original task unchanged.*eeeeeeeeeeee.*programmed, not learned/);
    assert.match(replay.ids.get('camera-status').textContent,/9 valid \/ 9 observed.*not action replay or per-frame cryptographic identity/);
    assert.match(replay.ids.get('exercise-scope').textContent,/fitted joint controller.*original body and task stay unchanged/);
    replay.select(2);assert.match(replay.ids.get('camera-status').textContent,/last valid boundary/);
    replay.select(3);assert.match(replay.ids.get('status').textContent,/passed.*interval complete/);
    assert.equal(JSON.stringify(data),before);
  }
}
function referenceSummaryRequiresExactBoundedIdentity(){
  const mutations=[
    r=>r.profile='source-fit-controller-v1',r=>r.sha256='e'.repeat(63),
    r=>r.scene_sha256=hash('0'),r=>r.task_sha256=hash('0'),r=>r.gates_sha256=hash('0'),
    r=>r.kind='learned_controller',r=>r.physical_validation='verified',
    r=>r.scope='',r=>r.scope='x'.repeat(513),r=>r.scope=['unchanged'],r=>r.scope='Verified physical recovery',
    r=>delete r.gates_sha256,r=>r.document_utf8='not a compact view'
  ];
  for(const mutate of mutations){
    const data=referenceFixture();mutate(data.replay_bundle.reference_controller);
    const replay=replayFixture(data);paused(replay);
    assert(!replay.ids.get('view').drawCalls.some(row=>row.name==='stroke'));
  }
  for(const value of [null,undefined]){
    const data=referenceFixture();data.replay_bundle.reference_controller=value;
    const replay=replayFixture(data);replay.select(1);paused(replay);
  }
}
function referenceMetadataCannotCrossControllerModes(){
  for(const mode of ['manual','coast','authored','learned','unknown']){
    const data=referenceFixture();data.replay_bundle.events[1].controller=mode;
    const replay=replayFixture(data);replay.select(1);paused(replay);
  }
  for(const type of ['start','reset']){
    const data=referenceFixture(),row=data.replay_bundle.events[0];
    row.type=type;row.reference_controller_sha256=hash('e');
    paused(replayFixture(data));
    row.controller='reference_proposal';paused(replayFixture(data));
  }
  for(const mutate of [
    row=>delete row.reference_controller_sha256,row=>row.reference_controller_sha256=hash('0'),
    row=>row.reference_controller={sha256:hash('e')},row=>row.reference_recipe={},
    row=>row.policy_sha256=hash('f'),row=>row.policy_sample={}
  ]){
    const data=referenceFixture();mutate(data.replay_bundle.events[1]);
    const replay=replayFixture(data);replay.select(1);paused(replay);
  }
  for(const field of ['reference_controller','reference_recipe','reference_controller_document']){
    const data=cameraFixture();data.replay_bundle.events[1][field]={};
    const replay=replayFixture(data);replay.select(1);paused(replay);
  }
  for(const change of [
    data=>data.replay_bundle.events[1].exercise_metrics.gates_sha256=hash('0'),
    data=>data.replay_bundle.events[1].snapshot.profile='shared-world-v3',
    data=>data.replay_bundle.camera_identity.source_clock_sha256=hash('0')
  ]){
    const data=referenceFixture();change(data);
    const replay=replayFixture(data);replay.select(1);paused(replay);
  }
}
function policyAndReferenceSummariesCannotMix(){
  const mixed=policyFixture(true);
  mixed.replay_bundle.reference_controller=referenceFixture(true).replay_bundle.reference_controller;
  paused(replayFixture(mixed));
  for(const field of ['camera_policy_admission','camera_policy_summary','policy_sha256','policy_admission_sha256']){
    const data=referenceFixture();data.replay_bundle[field]=field.endsWith('sha256')?hash('f'):{};
    paused(replayFixture(data));
  }
}
function workerForwardsOnlyTheCompactReferenceView(){
  const html=fs.readFileSync(new URL('../src/weft/assets/shared-world-replay.html',import.meta.url),'utf8');
  const expression=html.match(/const workerSource=([\s\S]*?);\nconst url=/)?.[1];
  assert(expression);
  const helper=html.match(/function compactCameraReference\([\s\S]*?\n}/)?.[0]||'';
  const source=vm.runInNewContext(helper+'\n'+expression),messages=[];
  const context=vm.createContext({postMessage:value=>messages.push(value)});
  vm.runInContext(source,context);
  const data=referenceFixture();
  context.onmessage({data:{type:'init',raw:JSON.stringify(data.replay_bundle)}});
  assert.deepEqual(copy(messages[0].reference_controller),data.replay_bundle.reference_controller);
  assert(Buffer.byteLength(JSON.stringify(messages[0].reference_controller))<1024);
  const invalid=copy(data.replay_bundle);invalid.reference_controller.document_utf8='x'.repeat(1<<20);
  context.onmessage({data:{type:'init',raw:JSON.stringify(invalid)}});
  assert.deepEqual(copy(messages[1].reference_controller),{invalid:true});
  assert(Buffer.byteLength(JSON.stringify(messages[1].reference_controller))<64);
}
function repeatedReferenceViewsReuseValidationAndDiagnosticText(){
  const replay=replayFixture(referenceFixture(true));replay.select(1);
  assert.equal(replay.ids.get('play').disabled,false);
  const output=replay.ids.get('camera-status');let saved=output.textContent,writes=0;
  Object.defineProperty(output,'textContent',{get:()=>saved,set:value=>{saved=value;writes++;}});
  const start=performance.now();for(let i=0;i<1000;i++)replay.turn();const elapsed=performance.now()-start;
  assert.equal(writes,0);
  console.log(JSON.stringify({measurement:'camera_reference_1000_unchanged_views',elapsed_ms:elapsed,
    diagnostic_text_writes:writes,scope:'Mocked DOM only; not browser or native performance'}));
}

const tests={referenceReplayPreservesCameraIdentityAndGates,referenceSummaryRequiresExactBoundedIdentity,
  referenceMetadataCannotCrossControllerModes,policyAndReferenceSummariesCannotMix,
  workerForwardsOnlyTheCompactReferenceView,repeatedReferenceViewsReuseValidationAndDiagnosticText};
export {referenceFixture};
if(import.meta.url===pathToFileURL(process.argv[1]||'').href){
  const selected=process.argv.length>2?process.argv.slice(2):Object.keys(tests);let failed=0;
  for(const name of selected){assert(tests[name]);try{tests[name]();console.log('PASS '+name);}
    catch(error){failed++;console.error('FAIL '+name+'\n'+error.stack);}}
  console.log(JSON.stringify({passed:selected.length-failed,failed,scope:'Analytical saved-shaped camera reference replay UI'}));
  process.exitCode=failed?1:0;
}
