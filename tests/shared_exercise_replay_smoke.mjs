/** Human-only replay presentation; task validation and generic playback have separate suites. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync(new URL('../src/weft/assets/shared-world-replay.html', import.meta.url), 'utf8');
const script = [...html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)].at(-1)[1];
const object = {id: 'equipment', role: 'passive', shape: 'box', size: [.1, .1, .1],
  position: [.8, .6, .4], quaternion: [1, 0, 0, 0], color: '#965644'};

function fixture({rootAssistance = 'disabled', metrics = {}, stateTime = 0, engineStateStatus = 'current', eventTags = [{}], sceneProfile = 'shared-world-v1', horizon = 3} = {}) {
  const scene = {profile: sceneProfile, bounds: [3, 2, 2.5], dt: 1 / 30,
    task: {object_goal: {position_tolerance: .04}}};
  const exerciseTask = {profile: 'shared-exercise-task-v1',
    control: {root_assistance: rootAssistance, object_forces: 'forbidden'},
    equipment_ids: ['equipment'], interval: {start_offset_s: 0, horizon_steps: horizon},
    contact_requirements: [{object_id: 'equipment', wrist: 'left', kind: 'upward_support', start_step: 0, end_step: horizon}]};
  const taskMetrics = {profile: 'shared-exercise-task-v1', simulated_duration_s: 0, declared_duration_s: horizon / 30,
    observed_substeps: 0, valid_substeps: 0, completed_interval: false, success: false, failure: null,
    failed_gates: [], angular_rmse_rad: null, peak_angular_rmse_rad: 0,
    equipment: [{object_id: 'equipment', displacement_model_units: 0}],
    virtual_object_forces: 'forbidden', root_assistance: rootAssistance, ...metrics};
  exerciseTask.equipment_ids = taskMetrics.equipment.map(row => row.object_id);
  // The engine's old target score is deliberately different from exercise task results.
  const event = {type: taskMetrics.failure ? 'simulation_failure' : 'step', action: Array(rootAssistance === 'disabled' ? 19 : 22).fill(0),
    engine_state_status: engineStateStatus, exercise_metrics: taskMetrics,
    snapshot: {objects: taskMetrics.equipment.map((row, index) => ({...object, id: row.object_id,
      position: [object.position[0] + index * .2, ...object.position.slice(1)]})), target: [2, 1, .4], success: !taskMetrics.success,
      human: {segments: [{id: 'torso', a: [.8, .6, .5], b: [.8, .6, 1.1], radius: .1}], muscles: [], mobility: {floor_height: .1}},
      contact_details: [{kind: 'human_object', objects: ['human:forearm', 'equipment'], position: [.8, .6, .45]}],
      contact_count: 1, contact_summary: {human_object: 1}, coupled_metrics: {object_distance_world: 9.876},
      time_s: stateTime, substeps: 8}};
  const raw = JSON.stringify({scene, exercise_task: exerciseTask, events: eventTags.map(tags => ({...event, ...tags}))});
  const elements = new Map(), calls = [], workers = [];
  let blobSource = '';
  const context = new Proxy({}, {get: (object, name) => object[name] || ((...args) => calls.push({name, args})),
    set: (object, name, value) => (object[name] = value, true)});
  const element = id => {
    if (!elements.has(id)) elements.set(id, {id, value: id === 'zoom' ? '1' : id === 'yaw' ? '.5' : '0',
      checked: false, disabled: true, hidden: ['exercise-scope','controller'].includes(id), open: false, width: 1080, height: 420,
      textContent: id === 'data' ? raw : '', listeners: {},
      getContext: () => context, addEventListener(name, handler) { this.listeners[name] = handler; }, remove() {}});
    return elements.get(id);
  };
  class MockWorker {
    constructor() {
      this.queue = [];
      this.context = vm.createContext({postMessage: data => this.onmessage({data})});
      vm.runInContext(blobSource, this.context); workers.push(this);
    }
    postMessage(data) { this.queue.push(data); }
    flush() { assert(this.queue.length); this.context.onmessage({data: this.queue.shift()}); }
    terminate() {}
  }
  vm.runInNewContext(script, {document: {getElementById: element, addEventListener() {}},
    window: {addEventListener() {}}, location: {reload() {}},
    Blob: class { constructor(parts) { blobSource = parts.join(''); } },
    URL: {createObjectURL: () => 'blob:exercise-replay-test', revokeObjectURL() {}}, Worker: MockWorker,
    setTimeout() { throw new Error('Inspection must not start playback'); }, clearTimeout() {}});
  assert.equal(workers.length, 1);
  workers[0].flush(); workers[0].flush();
  assert.equal(workers[0].queue.length, 0);
  assert.equal(element('play').disabled, false);
  return {element, calls, original: JSON.parse(raw), selectFrame(index) {
    element('scrub').value = String(index); element('scrub').listeners.input(); workers[0].flush();
  }, selected: () => {
    element('details').open = true; element('details').listeners.toggle();
    return JSON.parse(element('detail').textContent);
  }};
}

function exerciseTitleAndAssistanceNeverImplyObjectActuatorsOrGrasp() {
  for (const rootAssistance of ['disabled', 'explicit_3_axis']) {
    const ui = fixture({rootAssistance});
    assert.equal(ui.element('recording-title').textContent, 'Human-only exercise task replay');
    assert.equal(ui.element('exercise-scope').hidden, false);
    const scope = ui.element('exercise-scope').textContent;
    assert.match(scope, /Equipment stays passive: no virtual object forces/);
    assert.match(scope, rootAssistance === 'disabled' ? /Root assistance: disabled/ : /Root assistance: three explicit virtual controls/);
    assert.match(scope, /throughout the declared interval/);
    assert.match(scope, /authored simulation task, not verified grasp or biomechanics/);
    assert.match(ui.element('contact-explanation').textContent, /not necessarily qualifying wrist-region support/);
    assert.match(ui.element('contact-explanation').textContent, /Only the declared wrist-region requirements count/);
    assert.doesNotMatch(ui.element('contact-explanation').textContent, /authored 3D object target/);
  }
}

function wholeIntervalHistoryIsNotReplacedByCurrentGeometryOrLegacyScore() {
  const ui = fixture({stateTime: 8 / 240, metrics: {simulated_duration_s: 8 / 240, observed_substeps: 8, valid_substeps: 8,
    angular_rmse_rad: .2, peak_angular_rmse_rad: .4, failed_gates: ['equipment_stability:equipment'],
    equipment: [{object_id: 'equipment', displacement_model_units: .180}, {object_id: 'other', displacement_model_units: .120}]}});
  assert.match(ui.element('status').textContent, /Human-only task · 0\.033s · in progress/);
  assert.match(ui.element('status').textContent, /maximum equipment drift 0\.180 model units · 8 measured substeps/);
  assert.doesNotMatch(ui.element('status').textContent, /object dist|9\.876|passed/);
  assert(!ui.calls.some(call => call.name === 'setLineDash' && call.args[0]?.length), 'Exercise replay must not draw the old object target');
  assert.equal(ui.calls.filter(call => call.name === 'arc').length, 3, 'Only the recorded body contact appears in the three views');
  const selected = ui.selected();
  assert.equal(selected.exercise_metrics.peak_angular_rmse_rad, .4);
  assert.equal(selected.exercise_metrics.equipment[0].displacement_model_units, .18);
  assert.deepEqual(selected.snapshot.objects[0].position, object.position, 'The equipment has returned to its reset position, but past drift remains reported');
  assert.deepEqual(selected, ui.original.events[0], 'Inspection preserves all selected event metrics and states');
}

function completedSuccessAndGateFailureUseExerciseOutcomeNotEngineTarget() {
  for (const success of [true, false]) {
    const ui = fixture({stateTime: .1, metrics: {simulated_duration_s: .1, observed_substeps: 24, valid_substeps: 24,
      completed_interval: true, success, failed_gates: success ? [] : ['equipment_stability:equipment'],
      equipment: [{object_id: 'equipment', displacement_model_units: success ? .01 : .08}]}});
    assert.match(ui.element('status').textContent, success ? / · passed · / : / · not passed · /);
    assert.match(ui.element('status').textContent, /0\.100s/);
    assert.doesNotMatch(ui.element('status').textContent, /in progress|object dist|9\.876/);
    assert.equal(ui.selected().exercise_metrics.success, success);
    assert.equal(ui.selected().snapshot.success, !success, 'Legacy engine success must not decide exercise outcome');
  }
}

function interruptedFailureShowsMeasuredClockAndLastValidGeometryExplicitly() {
  const failure = 'engine_failure:invalid_substep_observation';
  const ui = fixture({stateTime: 8 / 240, engineStateStatus: 'last_valid_control_boundary', eventTags: [{type:'step',controller:'learned',policy_sha256:'a'.repeat(64)}], metrics: {
    simulated_duration_s: 9 / 240, observed_substeps: 9, valid_substeps: 8, completed_interval: false, success: false,
    failure, failed_gates: [failure], equipment: [{object_id: 'equipment', displacement_model_units: .02}]}});
  assert.match(ui.element('status').textContent, / · not passed/);
  const displayedTime = Number(ui.element('status').textContent.match(/Human-only task · ([\d.]+)s/)[1]);
  assert(Math.abs(displayedTime - 9 / 240) <= .0005001, 'The displayed clock must use measured exercise time, not stale geometry time');
  assert.match(ui.element('status').textContent, /9 measured substeps/);
  assert.match(ui.element('status').textContent, /showing last valid geometry after simulation failure/);
  assert.match(ui.element('controller').textContent, /recorded policy · SHA256 aaaaaaaaaaaa/);
  assert.doesNotMatch(ui.element('status').textContent, /in progress|object dist/);
  const selected = ui.selected();
  assert.equal(selected.exercise_metrics.failure, failure);
  assert.equal(selected.exercise_metrics.valid_substeps, 8);
  assert.equal(selected.snapshot.time_s, 8 / 240);
  assert.equal(selected.engine_state_status, 'last_valid_control_boundary');
  assert.deepEqual(selected, ui.original.events[0], 'A failed event must not relabel stale geometry as a new terminal state');
}

function controllerLabelsFollowRecordedIntentAcrossFramesAndResets() {
  const hash = '0123456789ab' + 'f'.repeat(52);
  const tags = [{type:'start',action:undefined},{controller:'manual'},{controller:'coast'},
    {controller:'authored'},{controller:'learned',policy_sha256:hash},{type:'reset',action:undefined},{}];
  const ui = fixture({eventTags:tags});
  const labels = [/^Recording start · no action applied$/, /manual human action/, /coast · all human commands zero/,
    /authored feedback · programmed, not learned/, /recorded policy · SHA256 0123456789ab/, /^Seeded reset · no action applied$/,
    /^Controller unspecified/];
  for(let index=0;index<tags.length;index++) {
    if(index) ui.selectFrame(index);
    assert.equal(ui.element('controller').hidden,false);
    assert.match(ui.element('controller').textContent,labels[index]);
    assert.match(ui.element('status').textContent,/in progress/);
    assert.deepEqual(ui.selected(),ui.original.events[index],'Controller display must not rewrite the recorded event');
    if(index!==4) assert.doesNotMatch(ui.element('controller').textContent,/SHA256/,'Prior policy identity must not survive a manual action or reset');
  }
  ui.selectFrame(4);
  assert.match(ui.element('controller').textContent,/not a claim of learned improvement/);
  assert(!ui.element('controller').textContent.includes(hash),'The compact label uses a short hash; details preserve the complete identity');
  assert.equal(ui.selected().policy_sha256,hash);
}

function untaggedOldEventsNeverInferCoastAndMissingPolicyIdentityStaysExplicit() {
  for(const tag of [{},{controller:'unknown'}]) {
    const ui=fixture({eventTags:[tag]});
    assert(ui.selected().action.every(value=>value===0));
    assert.match(ui.element('controller').textContent,/^Controller unspecified/);
    assert.doesNotMatch(ui.element('controller').textContent,/coast|manual|recorded policy/);
  }
  for(const policy_sha256 of [undefined,'not-a-hash']) {
    const ui=fixture({eventTags:[{controller:'learned',policy_sha256}]});
    assert.match(ui.element('controller').textContent,/recorded policy · policy identity unavailable/);
    assert.doesNotMatch(ui.element('controller').textContent,/SHA256/);
  }
}

function referenceProposalNamesFixedTaskAndPreservesExactRecipeIdentity() {
  const hash='abcd01234567'+'8'.repeat(52);
  const ui=fixture({eventTags:[{controller:'reference_proposal',reference_controller_sha256:hash},{type:'reset',action:undefined}]});
  const label=ui.element('controller').textContent;
  assert.match(label,/fitted-reference alternative · original task unchanged/);
  assert.match(label,/SHA256 abcd01234567 · programmed, not learned/);
  assert(!label.includes(hash));
  assert.equal(ui.selected().reference_controller_sha256,hash);
  assert.deepEqual(ui.selected(),ui.original.events[0]);
  assert.match(ui.element('status').textContent,/in progress/);
  ui.selectFrame(1);
  assert.equal(ui.element('controller').textContent,'Seeded reset · no action applied');
  for(const reference_controller_sha256 of [undefined,'<img src=x>']) {
    const missing=fixture({eventTags:[{controller:'reference_proposal',reference_controller_sha256}]});
    assert.match(missing.element('controller').textContent,/controller identity unavailable/);
    assert.doesNotMatch(missing.element('controller').textContent,/SHA256|<img/);
  }
}

function fullReferenceV2ReplaysUnclippedIntervalAndIndependentFailure() {
  const ui = fixture({sceneProfile: 'shared-world-v2', horizon: 254, rootAssistance: 'explicit_3_axis',
    stateTime: 254 / 30, metrics: {simulated_duration_s: 254 / 30, observed_substeps: 254 * 16,
      valid_substeps: 254 * 16, completed_interval: true, success: false, failed_gates: ['body_tracking']}});
  assert.match(ui.element('status').textContent, /8\.467s · not passed/);
  assert.match(ui.element('status').textContent, /4064 measured substeps/);
  assert.doesNotMatch(ui.element('status').textContent, /object dist|9\.876|in progress/);
  assert.equal(ui.original.scene.profile, 'shared-world-v2');
  assert.equal(ui.original.exercise_task.interval.horizon_steps, 254);
  assert.equal(ui.selected().exercise_metrics.declared_duration_s, 254 / 30);
  assert.equal(ui.selected().action.length, 22);
  assert(ui.calls.some(call => call.name === 'lineTo'), 'V2 retains the same geometry rendering');
}

const checks = [exerciseTitleAndAssistanceNeverImplyObjectActuatorsOrGrasp,
  wholeIntervalHistoryIsNotReplacedByCurrentGeometryOrLegacyScore,
  completedSuccessAndGateFailureUseExerciseOutcomeNotEngineTarget,
  interruptedFailureShowsMeasuredClockAndLastValidGeometryExplicitly,
  controllerLabelsFollowRecordedIntentAcrossFramesAndResets,
  untaggedOldEventsNeverInferCoastAndMissingPolicyIdentityStaysExplicit,
  referenceProposalNamesFixedTaskAndPreservesExactRecipeIdentity,
  fullReferenceV2ReplaysUnclippedIntervalAndIndependentFailure];
const requested = new Set(process.argv.slice(2));
const selected = requested.size ? checks.filter(check => requested.has(check.name)) : checks;
if (requested.size && selected.length !== requested.size) throw new Error('Unknown replay check name');
let failures = 0;
for (const check of selected) {
  try { check(); console.log(`PASS ${check.name}`); }
  catch (error) { failures++; console.error(`FAIL ${check.name}: ${error.stack}`); }
}
if (failures) process.exitCode = 1;
