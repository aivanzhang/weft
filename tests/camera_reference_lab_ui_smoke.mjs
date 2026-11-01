/** Analytical lab presentation only; no browser, native engine, source processing or network. */
import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';
import {pathToFileURL} from 'node:url';
import {cameraFixture,cameraReceipt,liveFixture} from './camera_shared_ui_smoke.mjs';
import {policyFixture} from './camera_shared_policy_ui_smoke.mjs';
import {drain} from './lab_ui_smoke.mjs';

const copy=value=>JSON.parse(JSON.stringify(value)),hash=char=>char.repeat(64);
const node=(ui,id)=>{
  const find=element=>element.id===id?element:element.children.map(find).find(Boolean);
  return ui.ids.get(id)||find(ui.ids.get('exercise-task-card'));
};
function fittedFixture(roots=false){
  const data=cameraFixture(roots),summary={profile:'camera-reference-lab-attachment-v1',
    reference_controller_sha256:hash('d'),reference_trial_manifest_sha256:hash('e'),reference_proposal_manifest_sha256:hash('f'),
    kind:'source_fit_authored_controller',learned:false,root_assistance:data.task.control.root_assistance,
    saved_trial_outcomes:{authored:{steps:3,success:true,mean_reward:.83},reference_proposal:{steps:3,success:true,mean_reward:.80}},
    saved_replay_scope:'Retained matched-trial receipts; attachment does not rerun or independently prove replay',
    physical_validation:'unverified',scope:'Fitted joint targets, not a learned policy or necessarily better control. Original body, camera, root assistance, floor, task and scoring stay unchanged. Contact and physical accuracy are unverified.'};
  data.capabilities.controllers.fitted_reference_controller=true;
  data.capabilities.fitted_reference=copy(summary);
  data.status_overlay={learned_policy_available:false,exercise_policy:null,exercise_reference:copy(summary)};
  return data;
}
function fittedReceipt(data){
  const value=cameraReceipt(data),summary=data.status_overlay.exercise_reference;
  value.identities.reference_controller_sha256=summary.reference_controller_sha256;
  for(const [path,sha256] of [['run/reference-controller.json',summary.reference_controller_sha256],
    ['run/reference-trial/manifest.json',summary.reference_trial_manifest_sha256],['run/reference-attachment.json',hash('9')]]){
    value.verified_metadata.push({source_name:path,path,sha256,bytes:100});
  }
  Object.assign(value.replay,{protocol:'camera-shared-reference-full-state-replay-v1',reference_actions_verified:1});
  return value;
}

async function fittedControlsRecordDistinctModesAndSerializeReset(){
  for(const roots of [false,true]){
    const ui=liveFixture(fittedFixture(roots));await ui.ready();const button=node(ui,'mode-fitted');
    assert(button,'Fitted control must be present');assert.equal(button.disabled,false);
    assert.equal(ui.ids.get('policy').disabled,true);assert.equal(ui.ids.get('joint').children.length,roots?22:19);
    assert.match(ui.ids.get('reference-servo').textContent,/Original joint targets/);
    assert.match(node(ui,'fitted-reference-summary').textContent,/not learned.*not necessarily better/);
    assert.match(node(ui,'fitted-reference-summary').textContent,roots?/assistance stays programmed/:/assistance is disabled/);
    button.click();assert.equal(button.getAttribute('aria-pressed'),'true');
    ui.ids.get('step').click();await drain();assert.deepEqual(ui.pending[0].data,{steps:1,controller:'reference_proposal'});
    ui.ids.get('record').click();await drain();assert.equal(ui.pending.length,1);
    await ui.reply('/api/step');assert.equal(ui.pending[0].path,'/api/record/start');
    await ui.reply('/api/record/start','initial');assert.deepEqual(ui.pending[0].data,{steps:3,controller:'reference_proposal'});
    ui.ids.get('reset').click();await drain();assert(!ui.pending.some(row=>row.path==='/api/reset'));
    await ui.reply('/api/step');await ui.reply('/api/reset','initial');
    ui.ids.get('reference-servo').click();ui.ids.get('step').click();await drain();
    assert.deepEqual(ui.pending[0].data,{steps:1,controller:'reference'});await ui.reply('/api/step');
    ui.ids.get('mode-coast').click();ui.ids.get('step').click();await drain();
    assert.deepEqual(ui.pending[0].data,{steps:1,controller:'coast'});await ui.reply('/api/step');
    ui.ids.get('mode-manual').click();ui.motor(roots?21:18);ui.action();await drain();
    assert.equal(ui.pending[0].data.action.length,roots?22:19);assert.equal(ui.pending[0].data.action.at(-1),.25);
    assert(ui.pending[0].data.action.slice(0,-1).every(v=>v===0));await ui.reply('/api/step');
    ui.ids.get('stop-record').click();await drain();await ui.reply('/api/record/stop');
    assert(!ui.requests.some(row=>/train|appearance|reference-controller.json|api\/actor/.test(row.path)));
    assert(ui.requests.filter(row=>row.path==='/api/step').every(row=>!('reference_controller_sha256' in row.data)));
  }
}

async function missingOrMismatchedAttachmentCannotEnableFitted(){
  const missing=liveFixture(cameraFixture());await missing.ready();
  assert.equal(node(missing,'mode-fitted')?.disabled,true);
  assert.match(node(missing,'fitted-reference-summary').textContent,/No fitted controller is attached/);
  for(const mutate of [d=>delete d.status_overlay.exercise_reference,
    d=>d.status_overlay.exercise_reference.reference_controller_sha256=hash('0'),
    d=>d.capabilities.fitted_reference.reference_trial_manifest_sha256='invalid',
    d=>d.capabilities.fitted_reference.root_assistance='explicit_3_axis',
    d=>d.capabilities.controllers.fitted_reference_controller='true',
    d=>d.capabilities.fitted_reference.learned=true,
    d=>d.capabilities.fitted_reference.saved_trial_outcomes.reference_proposal.mean_reward=Infinity,
    d=>d.status_overlay.exercise_reference.saved_trial_outcomes.reference_proposal.steps=4]){
    const data=fittedFixture();mutate(data);const ui=liveFixture(data);await ui.ready();
    const button=node(ui,'mode-fitted');assert.equal(button.disabled,true);assert.equal(ui.ids.get('step').disabled,false);
    button.click();ui.ids.get('step').click();await drain();
    assert(Array.isArray(ui.pending[0].data.action));assert.equal(ui.pending[0].data.controller,undefined);
  }
  const mixed=policyFixture();Object.assign(mixed.capabilities.controllers,{fitted_reference_controller:true});
  mixed.capabilities.fitted_reference=fittedFixture().capabilities.fitted_reference;
  mixed.status_overlay.exercise_reference=copy(mixed.capabilities.fitted_reference);
  const ui=liveFixture(mixed);await ui.ready();assert.equal(ui.ids.get('step').disabled,true);
  assert.equal(node(ui,'mode-fitted').disabled,true);assert.equal(ui.ids.get('policy').disabled,true);
}

async function fittedReceiptsRequireRecipeAndDistinctCounts(){
  const ui=liveFixture(fittedFixture(true));await ui.ready();const saved=fittedReceipt(ui.data),api=ui.context.feetTest;
  assert(api.recordingEvidenceValid(saved,saved.session),'Exact fitted receipt must be accepted');
  api.applyRecordingEvidence(saved);
  assert.match(ui.ids.get('recording-evidence-replay').textContent,/Fitted decisions matched: 1/);
  assert.match(ui.ids.get('recording-evidence-replay').textContent,/not a learned/);
  assert.match(ui.ids.get('recording-evidence-status').textContent,/not rerun/);
  assert(ui.ids.get('recording-evidence-identities').children.some(row=>row.children.some(c=>c.textContent===saved.identities.reference_controller_sha256)));
  for(const mutate of [v=>v.replay.protocol='camera-shared-full-state-replay-v1',
    v=>v.replay.reference_actions_verified=null,v=>v.replay.reference_actions_verified=3,
    v=>v.replay.reference_actions_verified=true,v=>v.replay.learned_actions_verified=0,
    v=>v.identities.reference_controller_sha256=hash('0'),v=>v.identities.policy_sha256=hash('0'),
    v=>v.verified_metadata.find(row=>row.path==='run/reference-controller.json').sha256=hash('0'),
    v=>v.verified_metadata.find(row=>row.path==='run/reference-trial/manifest.json').sha256=hash('0'),
    v=>v.verified_metadata=v.verified_metadata.filter(row=>row.path!=='run/reference-attachment.json')]){
    const changed=copy(saved);mutate(changed);assert.equal(api.recordingEvidenceValid(changed,changed.session),false);
  }
  const manual=copy(saved);manual.replay.reference_actions_verified=0;
  assert(api.recordingEvidenceValid(manual,manual.session));api.applyRecordingEvidence(manual);
  assert.match(ui.ids.get('recording-evidence-replay').textContent,/Fitted decisions matched: 0/);
  const unverified=copy(saved);Object.assign(unverified.replay,{status:'not_performed',protocol:null,
    actions_checked:null,events_checked:null,resets_checked:null,learned_actions_verified:null,reference_actions_verified:null});
  assert(api.recordingEvidenceValid(unverified,unverified.session));api.applyRecordingEvidence(unverified);
  assert.match(ui.ids.get('recording-evidence-replay').textContent,/not performed/);
  unverified.replay.reference_actions_verified=0;assert.equal(api.recordingEvidenceValid(unverified,unverified.session),false);
  const unattached=liveFixture(cameraFixture(true));await unattached.ready();
  assert.equal(unattached.context.feetTest.recordingEvidenceValid(saved,saved.session),false);
}

async function repeatedFittedUpdatesDoNotRewriteStaticControlsOrEvidence(){
  const data=fittedFixture(true);
  for(const state of Object.values(data.states)){
    state.engine_state.human.segments=Array.from({length:32},(_,i)=>({...state.engine_state.human.segments[0],id:`analytic_${i}`}));
    state.engine_state.human.muscles=Array.from({length:38},(_,i)=>({name:`analytic_${i}`,a:[1,1,.8],b:[1,1,1.5],activation:.1}));
    state.engine_state.contact_details=Array.from({length:256},()=>({kind:'human_floor',position:[1,1,0]}));
  }
  const ui=liveFixture(data);await ui.ready();ui.update('active');const button=node(ui,'mode-fitted');button.click();
  const saved=fittedReceipt(ui.data),api=ui.context.feetTest;api.applyRecordingEvidence(saved);
  let writes=0,attributes=0;
  for(const output of [node(ui,'fitted-reference-summary'),node(ui,'fitted-reference-comparison'),button,
    ui.ids.get('mode-description'),ui.ids.get('recording-evidence-replay'),ui.ids.get('recording-evidence-identities')]){
    let value=output.textContent;Object.defineProperty(output,'textContent',{get:()=>value,set:next=>{writes++;value=next;}});
  }
  const original=button.setAttribute.bind(button);button.setAttribute=(...args)=>{attributes++;original(...args);};
  const childCount=ui.ids.get('exercise-task-card').children.length,started=performance.now();
  for(let i=0;i<200;i++){ui.update('active');api.applyRecordingEvidence(saved);}
  assert.equal(writes,0);assert.equal(attributes,0);assert.equal(ui.ids.get('exercise-task-card').children.length,childCount);
  assert.equal(ui.pending.length,0);
  console.log(JSON.stringify({measurement:'200_unchanged_fitted_camera_scripted_updates',elapsed_ms:performance.now()-started,
    fitted_text_writes:writes,fitted_attribute_writes:attributes,fixture:'32 segments / 38 illustrative muscles / 256 contacts',
    scope:'Node fake DOM only; not browser responsiveness or native control evidence'}));
}

const checks={fittedControlsRecordDistinctModesAndSerializeReset,missingOrMismatchedAttachmentCannotEnableFitted,
  fittedReceiptsRequireRecipeAndDistinctCounts,repeatedFittedUpdatesDoNotRewriteStaticControlsOrEvidence};
export {fittedFixture,fittedReceipt,node};
if(import.meta.url===pathToFileURL(process.argv[1]||'').href){
  const names=process.argv.slice(2),selected=names.length?names:Object.keys(checks);let failed=0;
  for(const name of selected){try{assert(checks[name],name);await checks[name]();console.log(`PASS ${name}`);}
    catch(error){failed++;console.error(`FAIL ${name}`,error);}}
  console.log(JSON.stringify({passed:selected.length-failed,failed,scope:'Analytical fitted-reference GUI only'}));process.exitCode=failed?1:0;
}
