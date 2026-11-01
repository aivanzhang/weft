/** Explicit v5 summaries and outbound control requests; mocked DOM, no browser/physics. */
import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';
import {fixture, drain} from './lab_ui_smoke.mjs';

function policy(overrides = {}) {
  return {profile:'shared-exercise-policy-v5',policy_sha256:'c'.repeat(64),
    selected_feedback_kind:'searched_feedback_residuals',privileged_simulator_features:true,
    training_score:.7,training_score_improved_over_authored:false,training_feasible:false,
    selection_objective:'training-only feasibility',evaluation_status:'not_attached',
    evaluation_successes:null,evaluation_reset_protocol:null,...overrides};
}
const create = (options = {}) => fixture('shared-world',{sharedExercise:true,sharedProfile:'shared-world-v2',
  exercisePolicyAvailable:true,exercisePolicy:policy(),...options});

async function splitFeedbackKeepsHumanControlsAndSerializedRecording() {
  for(const rootAssistance of [false,true]) {
    const ui=create({rootAssistance});await ui.ready();
    assert.equal(ui.ids.get('joint').children.length,rootAssistance?22:19);
    assert.equal(ui.ids.get('actor').disabled,true);
    assert.equal(ui.ids.get('policy').textContent,'Separate arm/leg feedback');
    assert.equal(ui.ids.get('start-learned').textContent,'Try separate arm/leg feedback');
    assert.match(ui.ids.get('exercise-policy-summary').textContent,/torso, shoulder, elbow, hip and knee.*current position and velocity errors.*Five groups.*ten bounded.*±0\.25.*root assistance remains programmed, not learned/);
    assert.match(ui.ids.get('learning-description').textContent,/simulator-state and reference features, not video pixels/);
    assert.match(ui.ids.get('exercise-evaluation-summary').textContent,/Independent evaluation unavailable/);
    const before=ui.backendTaskState();ui.ids.get('policy').click();
    assert.deepEqual(ui.backendTaskState(),before);
    ui.ids.get('step').click();await drain();
    assert.deepEqual(ui.pending[0].data,{steps:1,policy:true});ui.resolveNext('/api/step');await drain();
    assert.match(ui.ids.get('mode-description').textContent,/Equipment has no force controls/);
    ui.ids.get('record').click();await drain();
    assert.deepEqual(ui.pending[0].data,{});assert.equal(ui.ids.get('policy').disabled,true);
    ui.resolveNext('/api/record/start');await drain();
    assert.deepEqual(ui.pending[0].data,{steps:3,policy:true});
    ui.ids.get('reset').click();await drain();assert(!ui.pending.some(row=>row.path==='/api/reset'));
    ui.resolveNext('/api/step');await drain();ui.resolveNext('/api/reset');await drain();
    assert.equal(ui.backendTaskState().task_metrics.observed_substeps,0);
    assert.match(ui.ids.get('record-status').textContent,/Recording/);
    ui.ids.get('stop-record').click();await drain();ui.resolveNext('/api/record/stop');await drain();
    assert.equal(ui.maxActiveSteps(),1);
    assert(!ui.requests.some(row=>['/api/actor','/api/joint'].includes(row.path)));
    assert(!ui.requests.some(row=>/\/(policy|training|rl-evaluation)\.json/.test(row.path)));
  }
}

async function zeroSplitPolicyIsStillAnAuthoredBaseline() {
  const ui=create({exercisePolicy:policy({selected_feedback_kind:'authored_unchanged'})});await ui.ready();
  assert.equal(ui.ids.get('policy').textContent,'Selected authored baseline');
  assert.match(ui.ids.get('exercise-policy-summary').textContent,/unchanged programmed feedback, not new learned gains/);
  assert.doesNotMatch(ui.ids.get('exercise-policy-summary').textContent,/Separate arm\/leg feedback|Training searched/);
  assert.match(ui.ids.get('training-caption').textContent,/At least one training start still failed/);
}

async function splitTrainingFeasibilityNeverOverwritesChangedStartFailures() {
  const scores=Object.fromEntries(['coast','random','authored','learned'].map(name=>[name,{episodes:3,successes:0}]));
  const ui=create({exercisePolicy:policy({training_feasible:true,evaluation_status:'evaluated',evaluation_successes:scores})});
  await ui.ready();
  assert.match(ui.ids.get('training-caption').textContent,/Both training starts passed every task limit.*Independent results are shown separately/);
  assert.deepEqual(ui.ids.get('scores').children.map(row=>row.children[0].textContent),Array(4).fill('0/3'));
  assert.match(ui.ids.get('scores').children[3].children[1].textContent,/Separate arm\/leg feedback/);
  assert.match(ui.ids.get('exercise-outcome').textContent,/In progress/);
  assert.match(ui.ids.get('learning-boundary').textContent,/not a new-video test/);
}

async function incompleteSplitMetadataDisablesOnlyItsLearnedMode() {
  for(const overrides of [{training_feasible:undefined},{training_feasible:'false'},
    {selected_feedback_kind:'searched_joint_residuals'},{selected_feedback_kind:'searched_gains'},
    {profile:'shared-exercise-policy-v6'},{policy_sha256:'not-a-pin'},{privileged_simulator_features:false}]) {
    const ui=create({exercisePolicy:policy(overrides)});await ui.ready();
    assert.equal(ui.ids.get('policy').disabled,true);assert.equal(ui.ids.get('start-learned').disabled,true);
    assert.match(ui.ids.get('exercise-policy-summary').textContent,/No validated human-only policy/);
    ui.ids.get('mode-coast').click();ui.ids.get('step').click();await drain();
    assert.deepEqual(ui.pending[0].data,{steps:1,controller:'coast'});
    ui.resolveNext('/api/step');await drain();
  }
}

async function unchangedSplitStatusDoesNotRewritePolicyOrFetchProofs() {
  const ui=create();await ui.ready();
  let writes=0;
  for(const id of ['policy','start-learned','exercise-policy-summary','learning-description','training-caption']) {
    const element=ui.ids.get(id);let value=element.textContent;
    Object.defineProperty(element,'textContent',{get:()=>value,set:next=>{writes++;value=next;}});
  }
  const options=ui.ids.get('joint').children, before=ui.requests.length, started=performance.now();
  for(let index=0;index<200;index++) {
    ui.visibility(true);ui.visibility(false);await ui.nextTimer();
  }
  const elapsed=performance.now()-started;
  assert.equal(writes,0);assert.strictEqual(ui.ids.get('joint').children,options);
  assert(ui.requests.slice(before).every(row=>row.path==='/api/status'));
  assert.equal(ui.pending.length,0);assert.equal(ui.timers.size,0);
  console.log(JSON.stringify({scope:'mocked unchanged status, not browser performance',updates:200,
    milliseconds:elapsed,policy_text_writes:writes,status_requests:ui.requests.length-before}));
}

let failed=0;
for(const check of [splitFeedbackKeepsHumanControlsAndSerializedRecording,zeroSplitPolicyIsStillAnAuthoredBaseline,
  splitTrainingFeasibilityNeverOverwritesChangedStartFailures,incompleteSplitMetadataDisablesOnlyItsLearnedMode,
  unchangedSplitStatusDoesNotRewritePolicyOrFetchProofs]) {
  try {await check();console.log(`PASS ${check.name}`);}
  catch(error) {failed++;console.error(`FAIL ${check.name}: ${error.stack}`);}
}
if(failed) process.exitCode=1;
