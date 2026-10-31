/** V4 presentation and outbound controls; analytical payloads, no physics or browser. */
import assert from 'node:assert/strict';
import {fixture, drain} from './lab_ui_smoke.mjs';

function policy(overrides = {}) {
  return {profile:'shared-exercise-policy-v4',policy_sha256:'c'.repeat(64),
    selected_feedback_kind:'searched_feedback_residuals',privileged_simulator_features:true,
    training_score:.7,training_score_improved_over_authored:false,training_feasible:false,
    selection_objective:'training-only feasibility',evaluation_status:'not_attached',
    evaluation_successes:null,evaluation_reset_protocol:null,...overrides};
}

async function feedbackUsesCurrentErrorsWithoutClaimingLearnedRootOrPhaseBias() {
  for(const rootAssistance of [false,true]) {
    const ui=fixture('shared-world',{sharedExercise:true,rootAssistance,
      sharedProfile:'shared-world-v2',
      exercisePolicyAvailable:true,exercisePolicy:policy()});await ui.ready();
    assert.equal(ui.ids.get('policy').disabled,false);
    assert.equal(ui.ids.get('policy').textContent,'Searched joint feedback');
    assert.equal(ui.ids.get('start-learned').textContent,'Try searched joint feedback');
    assert.match(ui.ids.get('exercise-policy-summary').textContent,/current position and velocity errors.*programmed feedback.*root assistance remains programmed, not learned/);
    assert.doesNotMatch(ui.ids.get('exercise-policy-summary').textContent,/phase-dependent/);
    assert.match(ui.ids.get('learning-description').textContent,/simulator-state and reference features, not video pixels/);
    assert.match(ui.ids.get('training-caption').textContent,/prioritized task limits before reward.*At least one training start still failed/);
    assert.match(ui.ids.get('exercise-evaluation-summary').textContent,/Independent evaluation unavailable/);
    const before=ui.backendTaskState();ui.ids.get('inspect-sandbox').click();ui.ids.get('policy').click();
    assert.deepEqual(ui.backendTaskState(),before);assert.match(ui.ids.get('mode-description').textContent,/current position and velocity errors/);
    ui.ids.get('step').click();await drain();
    const request=ui.pending.find(row=>row.path==='/api/step');assert(request);
    assert.deepEqual(request.data,{steps:1,policy:true});
    assert(!ui.requests.some(row=>/\/(training|policy|rl-evaluation)\.json/.test(row.path)));
  }
}

async function zeroFeedbackRemainsAnUnchangedAuthoredBaseline() {
  const ui=fixture('shared-world',{sharedExercise:true,exercisePolicyAvailable:true,
    exercisePolicy:policy({selected_feedback_kind:'authored_unchanged'})});await ui.ready();
  assert.equal(ui.ids.get('policy').textContent,'Selected authored baseline');
  assert.match(ui.ids.get('exercise-policy-summary').textContent,/unchanged programmed feedback, not new learned gains/);
  assert.doesNotMatch(ui.ids.get('exercise-policy-summary').textContent,/searched small joint corrections/);
  assert.match(ui.ids.get('training-caption').textContent,/At least one training start still failed/);
}

async function feasibleTrainingDoesNotOverwriteIndependentFailures() {
  const scores=Object.fromEntries(['coast','random','authored','learned'].map(name=>[name,{episodes:3,successes:0}]));
  const ui=fixture('shared-world',{sharedExercise:true,exercisePolicyAvailable:true,
    exercisePolicy:policy({training_feasible:true,evaluation_status:'evaluated',evaluation_successes:scores})});await ui.ready();
  assert.match(ui.ids.get('training-caption').textContent,/Both training starts passed every task limit.*Independent results are shown separately/);
  assert.deepEqual(ui.ids.get('scores').children.map(row=>row.children[0].textContent),Array(4).fill('0/3'));
  assert.match(ui.ids.get('exercise-evaluation-summary').textContent,/Attached independent changed-start evaluation/);
  assert.match(ui.ids.get('scores').children[3].children[1].textContent,/Searched joint feedback/);
  assert.match(ui.ids.get('exercise-outcome').textContent,/In progress/);
}

async function incompleteOrCrossProfileFeedbackMetadataCannotEnableControl() {
  for(const overrides of [{training_feasible:undefined},{training_feasible:'false'},
    {selected_feedback_kind:'searched_joint_residuals'},{selected_feedback_kind:'searched_gains'},
    {selected_feedback_kind:'coast_unchanged'},{profile:'shared-exercise-policy-v2'},
    {profile:'shared-exercise-policy-v3'},{privileged_simulator_features:false}]) {
    const ui=fixture('shared-world',{sharedExercise:true,exercisePolicyAvailable:true,exercisePolicy:policy(overrides)});await ui.ready();
    assert.equal(ui.ids.get('policy').disabled,true);
    assert.equal(ui.ids.get('start-learned').disabled,true);
    assert.match(ui.ids.get('exercise-policy-summary').textContent,/No validated human-only policy/);
    assert(!ui.requests.some(row=>row.path==='/api/step'));
  }
}

for(const check of [feedbackUsesCurrentErrorsWithoutClaimingLearnedRootOrPhaseBias,
  zeroFeedbackRemainsAnUnchangedAuthoredBaseline,feasibleTrainingDoesNotOverwriteIndependentFailures,
  incompleteOrCrossProfileFeedbackMetadataCannotEnableControl]) {
  await check();console.log(`PASS ${check.name}`);
}
