/** Saved-source UI presentation and outbound selectors; no browser, server or physics. */
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {Element, html, script} from './lab_ui_smoke.mjs';

const study = fs.realpathSync(process.argv[2] || 'artifacts/shared-exercise-lab-integration-01');
const sha = raw => crypto.createHash('sha256').update(raw).digest('hex');
const inputs = new Map(), artifacts = new Map();
let inputBytes = 0;
function readRaw(name, limit, expected, root = study) {
  const filename = fs.realpathSync(path.join(root, name));
  assert(filename.startsWith(root + path.sep), 'Payload must stay in its explicitly selected root');
  const fd = fs.openSync(filename, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK);
  let raw;
  try {
    const before = fs.fstatSync(fd);
    assert(before.isFile() && before.size <= limit, 'Payload exceeds its bound');
    assert(inputBytes + before.size <= 32 * 1024 * 1024, 'Combined inspected metadata exceeds 32 MiB');
    const buffer = Buffer.alloc(before.size + 1); let length = 0, count;
    while (length < buffer.length && (count = fs.readSync(fd, buffer, length, buffer.length - length, null))) length += count;
    const after = fs.fstatSync(fd);
    assert.equal(length, before.size, 'Payload size changed');
    assert.equal(after.mtimeMs, before.mtimeMs, 'Payload changed during read');
    assert.equal(after.ctimeMs, before.ctimeMs, 'Payload changed during read');
    raw = buffer.subarray(0, length);
  } finally { fs.closeSync(fd); }
  if (expected) assert.equal(sha(raw), expected, `Saved input changed: ${name}`);
  const label = root === study ? name : 'lab/run/' + name;
  inputBytes += raw.length; inputs.set(label, {filename, sha256: sha(raw), bytes: raw.length, root, name});
  return raw;
}
function readJson(name, limit, expected, root = study) {
  const raw = readRaw(name, limit, expected, root);
  return {value: JSON.parse(raw), bytes: raw.length};
}
const manifest = readJson('manifest.json', 1024 * 1024).value;
assert.equal(manifest.status, 'verified', 'Wait for the root study to finish and seal');
const readSealed = (name, limit) => {
  assert.equal(typeof manifest.files[name], 'string', `Study does not seal ${name}`);
  return readJson(name, limit, manifest.files[name]);
};
const initial = readSealed('initial-status.json', 4 * 1024 * 1024).value;
const final = readSealed('final-task-state.json', 4 * 1024 * 1024).value;
const inspection = readSealed('inspection.json', 64 * 1024).value;
const settings = process.argv[3] ? readSealed('settings.json', 64 * 1024).value : null;
const lab = fs.realpathSync(process.argv[3] || path.join(study, 'lab'));
if (settings) assert.equal(fs.realpathSync(settings.lab_directory), lab, 'External lab must match the sealed study settings');
const run = fs.realpathSync(path.join(lab, 'run'));
assert(run.startsWith(lab + path.sep));
const runHash = settings ? settings.run_manifest_sha256 : manifest.files['lab/run/manifest.json'];
assert.match(runHash, /^[0-9a-f]{64}$/);
const runManifest = readJson('manifest.json', 1024 * 1024, runHash, run).value;
assert.equal(readRaw('lab-implementation/assets/lab.html', 256 * 1024, runManifest.files['lab-implementation/assets/lab.html'], run).toString(), html);
assert.equal(readRaw('lab-implementation/assets/lab.js', 256 * 1024, runManifest.files['lab-implementation/assets/lab.js'], run).toString(), script);
assert.equal(sha(html), runManifest.files['lab-implementation/assets/lab.html']);
assert.equal(sha(script), runManifest.files['lab-implementation/assets/lab.js']);
assert.equal(initial.status, 'ready'); assert.equal(initial.control_profile, 'shared_exercise');
assert.equal(fs.realpathSync(initial.directory), lab);
const rootAssistance=initial.exercise_task.control.root_assistance;
assert(['explicit_3_axis','disabled'].includes(rootAssistance));
assert.equal(initial.action_contract.names.length, rootAssistance==='explicit_3_axis'?22:19);
assert.equal(final.profile, 'shared-exercise-state-v1'); assert.equal(final.task_sha256, initial.snapshot.task_sha256);
assert.equal(typeof final.task_metrics.completed_interval, 'boolean'); assert.equal(typeof final.task_metrics.success, 'boolean');
assert.equal(inspection.status, 'available'); assert(inspection.geometry.panels.length > 0);
const hasPolicy = initial.learned_policy_available === true;
const policyLabels={authored_unchanged:'Selected authored baseline',coast_unchanged:'Selected coast baseline',searched_gains:'Searched human feedback'};
const policyDescriptions={authored_unchanged:/unchanged programmed feedback, not new learned gains/,
  coast_unchanged:/unchanged zero-command baseline, not new learned gains/,searched_gains:/searched feedback gains using simulator state and the known body reference/};
let selectedKind=null,expectedScores=[];
if (hasPolicy) {
  selectedKind=initial.exercise_policy.selected_feedback_kind;assert(Object.hasOwn(policyLabels,selectedKind));
  assert(['evaluated','not_attached'].includes(initial.exercise_policy.evaluation_status));
  assert.equal(typeof initial.exercise_policy.training_score_improved_over_authored, 'boolean');
  assert.equal(runManifest.files['policy.json'], initial.exercise_policy.policy_sha256);
  const policy = readJson('policy.json', 64 * 1024, initial.exercise_policy.policy_sha256, run).value;
  assert.equal(policy.profile, 'shared-exercise-policy-v1'); assert.equal(policy.task_sha256, initial.snapshot.task_sha256);
  if(initial.exercise_policy.evaluation_status==='evaluated')for(const name of ['coast','random','authored','learned']) {
    const row=initial.exercise_policy.evaluation_successes[name];
    assert(Number.isInteger(row.episodes)&&row.episodes>=1&&row.episodes<=6);
    assert.equal(row.episodes,initial.exercise_policy.evaluation_successes.coast.episodes);
    assert(Number.isInteger(row.successes)&&row.successes>=0&&row.successes<=row.episodes);
    expectedScores.push(`${row.successes}/${row.episodes}`);
  }
}

const copy = value => JSON.parse(JSON.stringify(value));
const response = (value, bytes) => ({ok: true, headers: {get: name => name === 'Content-Length' ? String(bytes) : null}, json: async () => copy(value)});
async function drain() { for (let i = 0; i < 60; i++) await Promise.resolve(); }
function harness() {
  const elements = [], ids = new Map(), requests = [], pending = [], timers = new Map(), windowListeners = new Map();
  let timer = 0;
  for (const match of html.matchAll(/<([a-z][a-z0-9-]*)\b([^>]*)>/gi)) {
    const attributes = {};
    for (const attr of match[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g)) attributes[attr[1]] = attr[2] ?? '';
    const element = new Element(match[1], attributes); elements.push(element);
    if (element.id) ids.set(element.id, element);
  }
  const select = selector => {
    if (selector === '.panel') return elements.filter(element => element.classes.has('panel'));
    if (selector === '[data-action]') return elements.filter(element => 'action' in element.dataset);
    if (selector === '[data-json]') return elements.filter(element => 'json' in element.dataset);
    if (selector === '[data-action][aria-pressed=true]') return select('[data-action]').filter(element => element.getAttribute('aria-pressed') === 'true');
    if (selector.startsWith('#')) return [ids.get(selector.slice(1))].filter(Boolean);
    throw new Error(`Unsupported DOM selector ${selector}`);
  };
  ids.get('stages').querySelectorAll = () => elements.filter(element => 'panel' in element.dataset);
  const document = {hidden: false, getElementById: id => { assert(ids.has(id), `Missing DOM element ${id}`); return ids.get(id); },
    querySelectorAll: select, querySelector: selector => select(selector)[0] || null,
    createElement: tag => new Element(tag), addEventListener() {}};
  function fetch(url, options = {}) {
    const request = {url, data: options.body ? JSON.parse(options.body) : null}; requests.push(request);
    if (url === '/api/status') return Promise.resolve(response(initial, JSON.stringify(initial).length));
    if (url === '/api/inspection') return Promise.resolve(response(inspection, JSON.stringify(inspection).length));
    if (url.startsWith('/files/run/')) {
      const pathname = new URL(url, 'http://local-mock.invalid').pathname;
      const relative = pathname.slice('/files/run/'.length).split('/').map(decodeURIComponent).join('/');
      assert(!relative.startsWith('/') && !relative.split('/').some(part => !part || part === '.' || part === '..'));
      assert.equal(typeof runManifest.files[relative], 'string', 'Only sealed saved artifacts may supply fetch data');
      assert(relative.endsWith('.json'), 'This harness does not decode videos or images');
      if (!artifacts.has(relative)) artifacts.set(relative, readJson(relative,
        relative === 'human-reference.json' ? 8_000_000 : 2_000_000, runManifest.files[relative], run));
      const artifact = artifacts.get(relative); return Promise.resolve(response(artifact.value, artifact.bytes));
    }
    assert.equal(url, '/api/step', 'No unexpected mutation or external request');
    return new Promise(resolve => pending.push({...request, resolve}));
  }
  const storage = new Map();
  vm.runInNewContext(script, {document, window: {addEventListener(name, listener) { windowListeners.set(name, listener); }},
    location: {hash: '#token=local-mock-only', pathname: '/'}, sessionStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)},
    history: {replaceState() {}}, URLSearchParams, AbortSignal, fetch, performance: {now: () => 0},
    setTimeout(fn, delay) { timers.set(++timer, {fn, delay}); return timer; }, clearTimeout(id) { timers.delete(id); }}, {filename: 'lab.js'});
  return {ids, elements, requests, pending, timers, shutdown() { windowListeners.get('pagehide')(); }};
}

const ui = harness(); await drain();
assert.equal(ui.ids.get('error').hidden, true, ui.ids.get('error').textContent);
assert.equal(ui.ids.get('exercise-task-card').hidden, false);
assert.match(ui.ids.get('exercise-assistance').textContent, rootAssistance==='explicit_3_axis'?/3 explicit virtual controls/:/Root assistance: OFF/);
assert.equal(ui.ids.get('joint').children.length, initial.action_contract.names.length);
assert.deepEqual(ui.ids.get('joint').children.map(option => option.value), initial.action_contract.names);
assert.equal(ui.ids.get('actor-controls').hidden, true); assert.equal(ui.ids.get('policy').disabled, !hasPolicy);
if (hasPolicy) {
  assert.equal(ui.ids.get('policy').textContent, policyLabels[selectedKind]);
  assert.match(ui.ids.get('exercise-policy-summary').textContent, policyDescriptions[selectedKind]);
  assert.match(ui.ids.get('learning-description').textContent, /simulator-state and reference features, not video pixels/);
  assert.match(ui.ids.get('training-caption').textContent, initial.exercise_policy.training_score_improved_over_authored
    ? /training score was above the authored baseline.*not an independent evaluation result/
    : /No higher training score than the authored baseline/);
  assert.match(ui.ids.get('exercise-evaluation-summary').textContent, expectedScores.length
    ? /Attached independent changed-start evaluation/:/Independent evaluation unavailable/);
  if(expectedScores.length)assert.match(ui.ids.get('exercise-evaluation-summary').textContent,
    initial.exercise_policy.evaluation_reset_protocol?.profile==='shared-exercise-human-jitter-v1'
      ? /human-only perturbations.*exact scene pose and velocity.*not a changed-equipment test/
      : initial.exercise_policy.evaluation_reset_protocol?.profile==='shared-exercise-reset-v1'?/Legacy reset/:/Reset protocol unspecified/);
  assert.deepEqual(ui.ids.get('scores').children.map(row => row.children[0].textContent), expectedScores);
  assert.equal(ui.ids.get('exercise-policy-links').children.length, 1);
  assert(ui.ids.get('exercise-policy-links').children[0].href.includes('/files/run/policy.json'));
}
assert.match(ui.ids.get('exercise-outcome').textContent, /In progress/);
assert(!ui.requests.some(request => /\/(training|rl-evaluation|policy)\.json/.test(request.url)));
assert.equal(ui.ids.get('measurements-inspect').disabled, true);
assert.match(ui.ids.get('inspection-source').textContent, new RegExp(`${inspection.source.decoded_frames} decoded source frames`));
assert(ui.ids.get('source').src.includes('/files/run/' + inspection.source.video));
assert.equal(runManifest.files[inspection.source.video], inspection.source.sha256);
assert.equal(ui.ids.get('geometry-preview').src, undefined, 'Geometry image should remain lazy');
ui.ids.get('inspect-geometry').click();
assert(ui.ids.get('geometry-preview').src.includes('/files/run/' + inspection.geometry.panels[0].name));
assert.match(ui.ids.get('geometry-caption').textContent, /not synchronized to video playback/);
assert(!ui.ids.get('world-canvas').drawCalls.some(call => call.name === 'setLineDash' && call.args[0]?.length));
const taskButton = ui.elements.find(element => element.dataset.json === 'exercise-task.json');
assert(taskButton); assert.equal(taskButton.disabled, false);
taskButton.click(); await drain();
assert.equal(ui.ids.get('inspector').hidden, false);
assert.equal(ui.ids.get('inspect-title').textContent, 'exercise-task.json');
assert.deepEqual(JSON.parse(ui.ids.get('json').textContent), initial.exercise_task);
ui.ids.get('inspect-sandbox').click();
ui.ids.get('mode-coast').click(); ui.ids.get('step').click(); await drain();
assert.equal(ui.pending.length, 1); assert.deepEqual(ui.pending[0].data, {steps: 1, controller: 'coast'});
ui.shutdown();

const authored = harness(); await drain();
authored.ids.get('inspect-sandbox').click(); authored.ids.get(hasPolicy ? 'policy' : 'reference-servo').click(); authored.ids.get('step').click(); await drain();
assert.equal(authored.pending.length, 1); assert.deepEqual(authored.pending[0].data, hasPolicy ? {steps: 1, policy: true} : {steps: 1, controller: 'reference'});
// Only the transport envelope is mocked; this exact saved final state is not a one-step physics claim.
authored.pending.shift().resolve(response({snapshot: final}, JSON.stringify(final).length)); await drain();
assert.match(authored.ids.get('exercise-outcome').textContent, final.task_metrics.success?/^Passed/
  :final.task_metrics.failure?/^Not passed · simulation failed/:final.task_metrics.completed_interval?/^Not passed · interval complete/:/^In progress/);
assert.equal(authored.ids.get('play-state').textContent, final.task_metrics.success?'Goal reached'
  :final.terminated||final.truncated||final.task_metrics.failure?'Trial finished':'Paused — ready to try');
assert.match(authored.ids.get('exercise-duration').textContent, new RegExp(`${final.task_metrics.valid_substeps} valid / ${final.task_metrics.observed_substeps} observed`));
assert.doesNotMatch(authored.ids.get('readout').textContent, /object.*dist|target:|TASK SUCCESS/);
authored.ids.get('exercise-metrics-details').open = true; authored.ids.get('exercise-metrics-details').dispatch('toggle');
for (const gate of final.task_metrics.failed_gates) assert(authored.ids.get('exercise-gates').textContent.includes(gate));
assert(!authored.ids.get('live').drawCalls.some(call => call.name === 'setLineDash' && call.args[0]?.length));
assert(!authored.requests.some(request => /\/(training|rl-evaluation|policy)\.json/.test(request.url)));
if (hasPolicy) {
  assert.match(authored.ids.get('mode-description').textContent, policyDescriptions[selectedKind]);
  assert.deepEqual(authored.ids.get('scores').children.map(row => row.children[0].textContent), expectedScores);
}
authored.shutdown();
const verifiedInputs = [...inputs];
inputBytes = 0;
for (const [name, item] of verifiedInputs) readRaw(item.name, item.bytes, item.sha256, item.root);
console.log(JSON.stringify({status: 'passed', study, lab, study_manifest_sha256: inputs.get('manifest.json').sha256, lab_run_manifest_sha256: runHash,
  payload_hashes: Object.fromEntries(verifiedInputs.filter(([name]) => !name.startsWith('lab/run/')).map(([name, item]) => [name, item.sha256])),
  asset_sha256: {html: sha(html), js: sha(script)}, controls: initial.action_contract.names.length,
  source_frames: inspection.source.decoded_frames, geometry_panels: inspection.geometry.panels.length,
  policy: hasPolicy ? initial.exercise_policy : null, metadata_bytes_rechecked: inputBytes,
  final_outcome: {success: final.task_metrics.success, completed_interval: final.task_metrics.completed_interval, failed_gates: final.task_metrics.failed_gates},
  scope: 'Actual saved-payload UI presentation and outbound coast plus selected controller selectors only. Final snapshot injected unchanged; no new physics, browser, image/video decoding, media copy or server.'}, null, 2));
