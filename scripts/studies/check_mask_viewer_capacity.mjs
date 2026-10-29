import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {performance} from 'node:perf_hooks';
import {fileURLToPath} from 'node:url';
import vm from 'node:vm';

const root = fileURLToPath(new URL('../../', import.meta.url));
const output = process.argv[2];
assert(output, 'Provide a new output directory');
const out = path.resolve(output);
fs.mkdirSync(out, {recursive: false});
const script = fs.readFileSync(path.join(root, 'src/weft/assets/mask-viewer.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'src/weft/assets/mask-viewer.html'), 'utf8');
const samplePath = path.join(root, 'artifacts/meta-exercise-silhouette-masks-02/mask-evidence.json');
const sampleBytes = fs.readFileSync(samplePath);
const sample = JSON.parse(sampleBytes);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const snapshots = {
  'mask-viewer.js': script, 'mask-viewer.html': html,
  'check_mask_viewer_capacity.mjs': fs.readFileSync(fileURLToPath(import.meta.url), 'utf8'),
};
for (const [name, contents] of Object.entries(snapshots)) fs.writeFileSync(path.join(out, name), contents);
const contract = {
  frames: 500, objects: 8, trials: 3, rapidSelections: 500, sequentialSelections: 100, layerChanges: 200,
  idleMilliseconds: 100, metadataSourceSha256: sha(sampleBytes),
  scope: 'Synthetic maximum-count metadata in Node with stub images/canvas; no browser, image decoding, layout or paint measurement',
};
fs.writeFileSync(path.join(out, 'pre-run.json'), JSON.stringify({contract, snapshots: Object.fromEntries(
  Object.entries(snapshots).map(([name, contents]) => [name, sha(contents)]))}, null, 2));

// Real row structure, synthetic repetitions: this tests capacity, not another physical scene.
const data = {source: {...sample.source, frame_count: 500}, preview: sample.preview, frames: [], observations: []};
for (let i = 0; i < contract.frames; i++) {
  const indices = [];
  for (let j = 0; j < contract.objects; j++) {
    indices.push(data.observations.length);
    data.observations.push({...sample.observations[0], frame: i, time_s: i / 25,
      id: `capacity:frame:${i}:model-object:${j + 1}`, object_id: `sam-${j + 1}`, raw_model_id: String(j + 1),
      raster_path: `masks/${i}-${j}.png`});
  }
  data.frames.push({frame: i, time_s: i / 25, preview: `frames/${i}.jpg`, observation_indices: indices});
}
const payload = JSON.stringify(data);
const tick = () => new Promise(resolve => setImmediate(resolve));
const samples = [];
for (let trial = 0; trial < contract.trials; trial++) {
  let requested = 0, active = 0, peak = 0, draws = 0;
  const pending = [], timers = new Map();
  let timerId = 0;
  const elements = new Map([...html.matchAll(/id="([\w-]+)"/g)].map(match => [match[1], {
    textContent: '', value: '0', checked: true, events: {},
    setAttribute() {}, addEventListener(name, handler) {this.events[name] = handler;},
    getContext() {return new Proxy({}, {get: (_, key) => () => {if (key === 'drawImage') draws++;},
      set: () => true});},
  }]));
  elements.get('mask-data').textContent = payload;
  class StubImage {
    set src(value) {requested++; active++; peak = Math.max(peak, active); pending.push(this);}
  }
  const context = vm.createContext({document: {getElementById: id => elements.get(id)},
    window: {addEventListener() {}}, Image: StubImage,
    setTimeout: fn => {timers.set(++timerId, fn); return timerId;}, clearTimeout: id => timers.delete(id)});
  async function complete() {
    const batch = pending.splice(0);
    batch.forEach(picture => {active--; picture.onload();});
    await tick();
  }
  const start = performance.now();
  vm.runInContext(script, context);
  const startupMs = performance.now() - start;
  for (let i = 0; i < contract.rapidSelections; i++) vm.runInContext(`selectMaskFrame(${i})`, context);
  assert.equal(requested, 9);
  await complete();
  assert.equal(draws, 0);
  assert.equal(requested, 18);
  await complete();
  assert.match(elements.get('position').textContent, /^Frame 499 \/ 499/);
  const navigationMs = [];
  for (let i = 0; i < contract.sequentialSelections; i++) {
    const before = performance.now();
    vm.runInContext(`selectMaskFrame(${i})`, context);
    await complete();
    navigationMs.push(performance.now() - before);
  }
  const requestsBeforeToggle = requested;
  for (let i = 0; i < contract.layerChanges; i++) {
    const layer = elements.get('masks'); layer.checked = !layer.checked; layer.events.change();
  }
  assert.equal(requested, requestsBeforeToggle);
  const idleBefore = {requested, draws};
  await new Promise(resolve => setTimeout(resolve, contract.idleMilliseconds));
  assert.deepEqual({requested, draws}, idleBefore);
  assert.equal(timers.size, 0);
  assert.equal(active, 0);
  assert.equal(peak, 9);
  assert.equal(requested, 18 + contract.sequentialSelections * 9);
  navigationMs.sort((a, b) => a - b);
  samples.push({startup_ms: startupMs, scripted_navigation_median_ms: navigationMs[50],
    scripted_navigation_p95_ms: navigationMs[94], maximum_active_images: peak,
    image_requests: requested, idle_image_requests: 0, idle_draws: 0, layer_change_image_requests: 0});
}
assert.equal(fs.readFileSync(path.join(root, 'src/weft/assets/mask-viewer.js'), 'utf8'), script);
assert.equal(fs.readFileSync(path.join(root, 'src/weft/assets/mask-viewer.html'), 'utf8'), html);
const report = {status: 'scripted_capacity_checks_passed', contract, metadata_bytes: Buffer.byteLength(payload),
  observation_rows: data.observations.length, samples, actual_browser_interaction: false};
fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
process.stdout.write(JSON.stringify(report, null, 2) + '\n');
