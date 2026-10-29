import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const script = fs.readFileSync(new URL('../src/weft/assets/mask-viewer.js', import.meta.url), 'utf8');
const html = fs.readFileSync(new URL('../src/weft/assets/mask-viewer.html', import.meta.url), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));

function fixture({missing = false} = {}) {
  const data = {source: {width_px: 100, height_px: 80}, preview: {dimensions: [100, 80]}, frames: [], observations: []};
  for (let i = 0; i < 3; i++) {
    data.frames.push({frame: i, time_s: i / 10, preview: `frames/${i}.jpg`, observation_indices: [i]});
    data.observations.push({object_id: 'sam-1', phase: i ? 'post_initialization' : 'initialization',
      original_source: {frame: 100 + i}, color: [0, 215, 255], box: missing ? null : {box_xyxy_halfopen: [10, 20, 30, 40]},
      raster_path: missing ? null : `masks/${i}.png`, mask_status: missing ? 'not_returned' : 'decoded',
      silhouette: missing ? null : {centroid_px: [20, 30], foreground_pixels: 80, components_8_connected: 2}});
  }
  const drawCalls = [], pending = [], allImages = [], timers = new Map(), listeners = new Map();
  let timer = 0;
  const elements = new Map([...html.matchAll(/id="([\w-]+)"/g)].map(match => [match[1], {
    textContent: '', value: '0', max: 0, checked: true, hidden: true, disabled: false, attributes: {}, events: {},
    addEventListener(event, handler) {this.events[event] = handler;},
    setAttribute(key, value) {this.attributes[key] = value;},
    getContext() {return new Proxy({}, {get: (target, key) => (...args) => drawCalls.push([key, ...args]),
      set: (target, key, value) => true});}
  }]));
  elements.get('mask-data').textContent = JSON.stringify(data);
  class FakeImage {
    set src(value) {this.path = value; pending.push(this); allImages.push(this);}
  }
  const context = vm.createContext({document: {getElementById: id => {
    assert(elements.has(id), `Missing real HTML element ${id}`); return elements.get(id);
  }}, window: {addEventListener: (name, handler) => listeners.set(name, handler)}, Image: FakeImage,
  setTimeout: callback => {timers.set(++timer, callback); return timer;}, clearTimeout: id => timers.delete(id)});
  vm.runInContext(script, context);
  return {data, elements, drawCalls, pending, allImages, timers, listeners, context,
    select: index => vm.runInContext(`selectMaskFrame(${index})`, context),
    async complete() {const batch = pending.splice(0); batch.forEach(picture => picture.onload()); await tick();},
    execute: source => vm.runInContext(source, context)};
}

let checks = 0;
async function test(name, fn) {await fn(); checks++; process.stdout.write(`ok ${checks} - ${name}\n`);}

await test('Loads one frame and uses its original frame identity', async () => {
  const f = fixture(); assert.equal(f.pending.length, 2);
  await f.complete();
  assert.match(f.elements.get('position').textContent, /^Frame 0 \/ 2/);
  assert.match(f.elements.get('summary').textContent, /Original source frame 100/);
  assert.equal(f.elements.get('canvas').hidden, false);
  assert.equal(f.elements.get('stage').attributes['aria-busy'], 'false');
});
await test('Rapid selection coalesces to the latest frame without stale drawing', async () => {
  const f = fixture(); f.select(1); f.select(2);
  assert.equal(f.pending.length, 2); await f.complete();
  assert.equal(f.drawCalls.length, 0);
  assert.deepEqual(f.pending.map(p => p.path), ['frames/2.jpg', 'masks/2.png']);
  await f.complete(); assert.match(f.elements.get('position').textContent, /^Frame 2/);
  assert.equal(f.allImages.length, 4);
});
await test('Layer toggles reuse only the currently displayed images', async () => {
  const f = fixture(); await f.complete(); f.drawCalls.length = 0;
  f.elements.get('masks').checked = false; f.elements.get('masks').events.change();
  assert.equal(f.drawCalls.filter(call => call[0] === 'drawImage').length, 1);
  assert.equal(f.allImages.length, 2);
});
await test('A failed image waits for the rest of its batch before advancing', async () => {
  const f = fixture(); f.pending.shift().onerror(); await tick(); f.select(1);
  assert.equal(f.allImages.length, 2);
  await f.complete(); assert.equal(f.allImages.length, 4);
  await f.complete(); assert.match(f.elements.get('position').textContent, /^Frame 1/);
});
await test('Failed saved images never retain a previous visible frame', async () => {
  const f = fixture(); await f.complete(); f.select(1);
  f.pending.shift().onerror(); await f.complete();
  assert.equal(f.elements.get('canvas').hidden, true);
  assert.match(f.elements.get('position').textContent, /Could not load/);
  assert.equal(f.allImages.length, 4);
});
await test('Missing masks stay absent and do not request phantom images', async () => {
  const f = fixture({missing: true}); assert.equal(f.pending.length, 1); await f.complete();
  assert.match(f.elements.get('summary').textContent, /not returned/);
  assert.equal(f.drawCalls.filter(call => call[0] === 'strokeRect').length, 0);
});
await test('Page hiding prevents queued work and stale rendering', async () => {
  const f = fixture(); f.select(2); f.listeners.get('pagehide')(); await f.complete();
  assert.equal(f.pending.length, 0); assert.equal(f.drawCalls.length, 0);
  f.listeners.get('pageshow')(); assert.equal(f.pending.length, 2); await f.complete();
  assert.match(f.elements.get('position').textContent, /^Frame 2/);
});
await test('Slider input keeps only one debounced selection', async () => {
  const f = fixture(); await f.complete();
  for (let i = 0; i < 100; i++) {f.elements.get('frame').value = String(i % 3); f.elements.get('frame').events.input();}
  assert.equal(f.timers.size, 1); assert.equal(f.pending.length, 0);
  const callback = [...f.timers.values()][0]; f.timers.clear(); callback();
  assert.equal(f.pending.length, 2); await f.complete();
});
await test('Frame endpoints clamp and expose disabled controls', async () => {
  const f = fixture(); f.select(-5); await f.complete(); await f.complete();
  assert.equal(f.elements.get('previous').disabled, true);
  f.select(100); await f.complete(); assert.equal(f.elements.get('next').disabled, true);
});
await test('Offline page exposes raw evidence and makes no API requests', async () => {
  assert.match(html, /original-evidence\.json/); assert.match(html, /No API calls/);
  assert(!script.includes('fetch(')); assert(!html.includes('https://'));
  assert.match(html, /Silhouette centers are not centers of mass/);
});
process.stdout.write(`${checks} deterministic UI checks passed; browser interaction not performed.\n`);
