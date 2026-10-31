/** Person-only live and offline display without a browser, server or simulation. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {Element, html, script, drain} from './lab_ui_smoke.mjs';

const clone = value => JSON.parse(JSON.stringify(value));
const replayHtml = fs.readFileSync(new URL('../src/weft/assets/shared-world-replay.html', import.meta.url), 'utf8');
const replayScript = [...replayHtml.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)].at(-1)[1];
function bundle({assistance = false, task = true, badMetrics = false} = {}) {
  const names = [...Array(19)].map((_, i) => `joint_${i}`).concat(['root_x', 'root_y', 'root_z']);
  const scene = {profile: 'shared-world-v3', bounds: [3, 2, 2.5], dt: 1 / 30,
    controlled_id: 'human', target: null, objects: [], task: {type: 'human_motion_v1', object_goal: null},
    action_contract: {names, physical_limits: Array(22).fill(30)}, assumptions: ['Analytic UI fixture']};
  const exerciseTask = {profile: 'shared-exercise-task-v2',
    control: {root_assistance: assistance ? 'explicit_3_axis' : 'disabled', object_forces: 'forbidden'},
    interval: {horizon_steps: 200, start_offset_s: 0}, equipment_ids: [], contact_requirements: []};
  const geometry = {profile: 'shared-world-v3', objects: [], controlled_id: 'human', target: null,
    steps: 0, time_s: 0, substeps: 8, terminated: false, truncated: false, success: false,
    normalized_action: Array(22).fill(0), contact_count: 1,
    contact_summary: {human_object: 0, human_floor: 1, object_object: 0},
    contact_details: [{kind: 'human_floor', objects: ['human:left_foot', 'mobile_floor'], position: [.8, .6, 0]}],
    coupled_metrics: {angular_rmse_rad: .1, root_rmse_model_units: .01, object_distance_world: null},
    human: {angles: Object.fromEntries(names.slice(0, 19).map(name => [name, 0])),
      segments: [{id: 'torso', a: [.8, .6, .5], b: [.8, .6, 1.1], radius: .1}], muscles: [],
      mobility: {floor_height: 0, support_force: [0, 0, 0]}}};
  const metrics = {profile: 'shared-exercise-task-v2', simulated_duration_s: 0, declared_duration_s: 200 / 30,
    valid_substeps: 0, observed_substeps: 0, completed_interval: false, success: false, failure: null,
    clock_status: 'observed', equipment: [], contact_requirements: [], failed_gates: [], gates_sha256: 'f'.repeat(64),
    angular_rmse_rad: null, root_rmse_model_units: null, peak_angular_rmse_rad: 0,
    peak_root_rmse_model_units: 0, max_floor_penetration_model_units: 0,
    gate_applicability: {human_tracking: 'applied', equipment_stability: badMetrics ? 'passed' : 'not_applicable', contact_requirements: 'not_applicable'},
    reward_profile: 'human_tracking_effort_v1'};
  const state = {profile: 'shared-exercise-state-v1', engine_state: geometry, engine_state_status: 'current',
    task_metrics: metrics, terminated: false, truncated: false};
  return {scene, task: task ? exerciseTask : null, geometry, metrics, state};
}

function replayFixture(options = {}) {
  const data = bundle(options), elements = new Map(), workers = [], timers = new Map();
  const events = ['start', 'reset'].map((type, index) => ({index, type, seed: 11, snapshot: clone(data.geometry),
    ...(data.task ? {exercise_metrics: clone(data.metrics), engine_state_status: 'current'} : {})}));
  const input = {scene: data.scene, exercise_task: data.task, events};
  const element = id => {
    if (!elements.has(id)) elements.set(id, new Element(id === 'view' ? 'canvas' : 'div', {id}));
    return elements.get(id);
  };
  element('data').textContent = JSON.stringify(input);
  element('data').remove = () => {};
  element('view').width = 1080; element('view').height = 420;
  element('zoom').value = '1'; element('yaw').value = '.5';
  let blobSource, nextTimer = 0;
  const document = {hidden: false, getElementById: element, addEventListener(name, handler) { this[name] = handler; }};
  class Worker {
    constructor() {
      this.queue = []; workers.push(this);
      this.context = vm.createContext({postMessage: value => this.onmessage({data: value})});
      vm.runInContext(blobSource, this.context);
    }
    postMessage(value) { this.queue.push(value); }
    flush() { this.context.onmessage({data: this.queue.shift()}); }
    terminate() {}
  }
  vm.runInNewContext(replayScript, {document, window: {addEventListener() {}}, Worker,
    Blob: class { constructor(parts) { blobSource = parts.join(''); } },
    URL: {createObjectURL: () => 'blob:test', revokeObjectURL() {}},
    setTimeout(callback) { const id = ++nextTimer; timers.set(id, callback); return id; },
    clearTimeout: id => timers.delete(id)});
  workers[0].flush(); workers[0].flush();
  return {data, input, element, document, timers, worker: workers[0]};
}

for (const task of [false, true]) for (const assistance of [false, true]) {
  const ui = replayFixture({task, assistance}), el = ui.element;
  assert.equal(el('recording-title').textContent, 'Person-only 3D task replay');
  assert.match(el('world-scope').textContent, /object absence is not visually verified/);
  assert.match(el('status').textContent, /equipment.*not applicable/);
  assert.doesNotMatch(el('status').textContent, /object dist|maximum equipment drift/);
  assert(!el('view').drawCalls.some(call => call.name === 'setLineDash' && call.args[0]?.length));
  assert.equal(el('view').drawCalls.filter(call => call.name === 'arc').length, 3, 'Only ground contact dots in three panes');
  if (task) {
    assert.match(el('exercise-scope').textContent, /not applicable and add no reward/);
    assert.match(el('exercise-scope').textContent, assistance ? /three explicit/ : /disabled/);
  }
  el('scrub').value = '1'; el('scrub').dispatch('input'); ui.worker.flush();
  el('details').open = true; el('details').dispatch('toggle');
  assert.deepEqual(JSON.parse(el('detail').textContent), ui.input.events[1], 'Saved reset and exact state are retained');
  el('yaw').value = '1.3'; el('yaw').dispatch('input');
  assert.deepEqual(JSON.parse(el('detail').textContent), ui.input.events[1], 'Camera does not change recorded states');
  el('play').click(); assert.equal(ui.timers.size, 1);
  ui.document.hidden = true; ui.document.visibilitychange(); assert.equal(ui.timers.size, 0);
}

async function liveFixture(options = {}) {
  const data = bundle(options), elements = [], ids = new Map(), requests = [], timers = new Map();
  for (const match of html.matchAll(/<([a-z][a-z0-9-]*)\b([^>]*)>/gi)) {
    const attributes = {};
    for (const attr of match[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g)) attributes[attr[1]] = attr[2] ?? '';
    const element = new Element(match[1], attributes); elements.push(element); if (element.id) ids.set(element.id, element);
  }
  const select = selector => {
    if (selector === '.panel') return elements.filter(el => el.classes.has('panel'));
    if (selector === '[data-action]') return elements.filter(el => 'action' in el.dataset);
    if (selector === '[data-json]') return elements.filter(el => 'json' in el.dataset);
    if (selector === '[data-action][aria-pressed=true]') return select('[data-action]').filter(el => el.getAttribute('aria-pressed') === 'true');
    if (selector.startsWith('#')) return [ids.get(selector.slice(1))].filter(Boolean);
    throw new Error(`Unexpected selector ${selector}`);
  };
  ids.get('stages').querySelectorAll = () => elements.filter(el => 'panel' in el.dataset);
  const document = {hidden: false, getElementById(id) { assert(ids.has(id), id); return ids.get(id); },
    createElement: tag => new Element(tag), querySelectorAll: select, querySelector: selector => select(selector)[0],
    addEventListener(name, handler) { this[name] = handler; }};
  const summary = {status: 'ready', profile: 'shared-world', control_profile: 'shared_exercise', directory: 'analytic-fixture',
    scene: data.scene, snapshot: data.state, exercise_task: data.task, learned_policy_available: false,
    action_contract: {names: data.scene.action_contract.names.slice(0, options.assistance ? 22 : 19),
      root_assistance: data.task.control.root_assistance, object_force_channels: []},
    events: [], artifacts: [], sessions: [], recording: false};
  const response = (value, ok = true) => ({ok, headers: {get: () => '500'}, json: async () => clone(value)});
  const fetch = async (path, options = {}) => {
    requests.push({path, body: options.body && JSON.parse(options.body)});
    if (path === '/api/status') return response(summary);
    if (path === '/api/step') return response({snapshot: data.state});
    if (path === '/api/reset') return response({snapshot: data.state});
    if (path === '/api/record/start') return response({snapshot: data.state, recording: true});
    if (path.startsWith('/api/')) return response({status: 'unavailable'});
    if (path.includes('human-reference.json')) return response({frames: [clone(data.geometry), clone(data.geometry)]});
    return response({}, false);
  };
  let timerId = 0;
  vm.runInNewContext(script, {document, window: {addEventListener() {}}, fetch,
    location: {hash: '#token=synthetic', pathname: '/'}, sessionStorage: {getItem() {}, setItem() {}}, history: {replaceState() {}},
    URLSearchParams, AbortSignal, performance: {now: () => 0}, console,
    setTimeout(callback) { const id = ++timerId; timers.set(id, callback); return id; }, clearTimeout: id => timers.delete(id)});
  await drain(); await drain();
  return {ids, requests, document, timers, select};
}

for (const assistance of [false, true]) {
  const ui = await liveFixture({assistance}), el = id => ui.ids.get(id);
  assert.equal(el('toggle').disabled, false, el('error').textContent);
  assert.match(el('world-heading').textContent, /human body and ground/);
  assert.match(el('exercise-equipment-summary').textContent, /not applicable/);
  assert.equal(el('objects').children.length, 0);
  assert.equal(el('actor').children.length, 1);
  assert.equal(el('actor-controls').hidden, true);
  assert.equal(el('joint').children.length, assistance ? 22 : 19);
  assert(!el('live').drawCalls.some(call => call.name === 'setLineDash' && call.args[0]?.length));
  el('exercise-metrics-details').open = true; el('exercise-metrics-details').ontoggle();
  assert.match(el('exercise-equipment').children[0].children[0].textContent, /Not applicable/);
  el('step').click(); await drain();
  const step = ui.requests.find(row => row.path === '/api/step');
  assert.equal(step.body.action.length, assistance ? 22 : 19);
  assert(!ui.requests.some(row => row.path === '/api/actor'));
  assert.equal(ui.timers.size, 0, 'Inspection does not keep idle polling alive');
}
const invalid = await liveFixture({badMetrics: true});
assert.equal(invalid.ids.get('toggle').disabled, true);
assert.match(invalid.ids.get('exercise-outcome').textContent, /unavailable/);
console.log('person-only UI/replay: 7 scenarios passed; no server, browser or physics');
