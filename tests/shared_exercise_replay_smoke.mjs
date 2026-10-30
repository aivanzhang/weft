/** Human-only replay presentation; task validation and generic playback have separate suites. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync(new URL('../src/weft/assets/shared-world-replay.html', import.meta.url), 'utf8');
const script = [...html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)].at(-1)[1];
const object = {id: 'equipment', role: 'passive', shape: 'box', size: [.1, .1, .1],
  position: [.8, .6, .4], quaternion: [1, 0, 0, 0], color: '#965644'};

function fixture({rootAssistance = 'disabled', metrics = {}, stateTime = 0, engineStateStatus = 'current'} = {}) {
  const scene = {profile: 'shared-world-v1', bounds: [3, 2, 2.5], dt: 1 / 30,
    task: {object_goal: {position_tolerance: .04}}};
  const exerciseTask = {profile: 'shared-exercise-task-v1',
    control: {root_assistance: rootAssistance, object_forces: 'forbidden'},
    equipment_ids: ['equipment'], interval: {start_offset_s: 0, horizon_steps: 3},
    contact_requirements: [{object_id: 'equipment', wrist: 'left', kind: 'upward_support', start_step: 0, end_step: 3}]};
  const taskMetrics = {profile: 'shared-exercise-task-v1', simulated_duration_s: 0, declared_duration_s: .1,
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
  const raw = JSON.stringify({scene, exercise_task: exerciseTask, events: [event]});
  const elements = new Map(), calls = [], workers = [];
  let blobSource = '';
  const context = new Proxy({}, {get: (object, name) => object[name] || ((...args) => calls.push({name, args})),
    set: (object, name, value) => (object[name] = value, true)});
  const element = id => {
    if (!elements.has(id)) elements.set(id, {id, value: id === 'zoom' ? '1' : id === 'yaw' ? '.5' : '0',
      checked: false, disabled: true, hidden: id === 'exercise-scope', open: false, width: 1080, height: 420,
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
  return {element, calls, original: JSON.parse(raw), selected: () => {
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
  const ui = fixture({stateTime: 8 / 240, engineStateStatus: 'last_valid_control_boundary', metrics: {
    simulated_duration_s: 9 / 240, observed_substeps: 9, valid_substeps: 8, completed_interval: false, success: false,
    failure, failed_gates: [failure], equipment: [{object_id: 'equipment', displacement_model_units: .02}]}});
  assert.match(ui.element('status').textContent, / · not passed/);
  const displayedTime = Number(ui.element('status').textContent.match(/Human-only task · ([\d.]+)s/)[1]);
  assert(Math.abs(displayedTime - 9 / 240) <= .0005001, 'The displayed clock must use measured exercise time, not stale geometry time');
  assert.match(ui.element('status').textContent, /9 measured substeps/);
  assert.match(ui.element('status').textContent, /showing last valid geometry after simulation failure/);
  assert.doesNotMatch(ui.element('status').textContent, /in progress|object dist/);
  const selected = ui.selected();
  assert.equal(selected.exercise_metrics.failure, failure);
  assert.equal(selected.exercise_metrics.valid_substeps, 8);
  assert.equal(selected.snapshot.time_s, 8 / 240);
  assert.equal(selected.engine_state_status, 'last_valid_control_boundary');
  assert.deepEqual(selected, ui.original.events[0], 'A failed event must not relabel stale geometry as a new terminal state');
}

const checks = [exerciseTitleAndAssistanceNeverImplyObjectActuatorsOrGrasp,
  wholeIntervalHistoryIsNotReplacedByCurrentGeometryOrLegacyScore,
  completedSuccessAndGateFailureUseExerciseOutcomeNotEngineTarget,
  interruptedFailureShowsMeasuredClockAndLastValidGeometryExplicitly];
let failures = 0;
for (const check of checks) {
  try { check(); console.log(`PASS ${check.name}`); }
  catch (error) { failures++; console.error(`FAIL ${check.name}: ${error.stack}`); }
}
if (failures) process.exitCode = 1;
