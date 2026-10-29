import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync('src/weft/assets/spatial-replay.html', 'utf8');
const script = [...html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)].at(-1)[1];
const body = {id: 'block', role: 'active', shape: 'box', size: [.1, .1, .1], position: [.2, .3, .2], quaternion: [1, 0, 0, 0], color: '#AB7E48'};
const scene = {bounds: [1, .6, .6], dt: 1/60, task: {position_tolerance: .025}};
const support = {...body, id: 'support', role: 'static', position: [.5, .3, .05], size: [.8, .5, .05]};
const events = [0, 1, 2].map(i => ({type: i ? 'step' : 'start', snapshot: {objects: [{...body, position: [.2+.1*i, .3, .2]}, support], target: [.8, .3, .2], contact_details: [], contact_count: 0, substeps: 17, time_s: i/60}}));
const original = JSON.stringify({scene, events});
const elements = new Map(), timers = new Map(), documentListeners = {}, windowListeners = {};
let clips = 0, fills = 0, timerId = 0, blobSource = '', worker, reloads = 0;
const context2d = new Proxy({}, {get: (_, key) => (...args) => {
  if (key === 'clip') clips++;
  if (key === 'fill') fills++;
  if (['arc', 'moveTo', 'lineTo'].includes(key)) assert.ok(args.every(Number.isFinite));
}, set: () => true});
function element(id) {
  if (!elements.has(id)) elements.set(id, {id, value: id === 'zoom' ? '1' : id === 'yaw' ? '.5' : '0', checked: false, disabled: true, open: false,
    width: 1080, height: 400, textContent: id === 'data' ? original : '', listeners: {},
    addEventListener(name, handler) {this.listeners[name] = handler;}, remove() {this.removed = true;}, getContext() {return context2d;}});
  return elements.get(id);
}
class FakeWorker {
  constructor() {
    this.queue = []; worker = this;
    this.context = vm.createContext({postMessage: data => this.onmessage({data})});
    vm.runInContext(blobSource, this.context);
  }
  postMessage(data) {this.queue.push(data);}
  flush() {this.context.onmessage({data: this.queue.shift()});}
  terminate() {this.terminated = true;}
}
const document = {hidden: false, getElementById: element, addEventListener(name, handler) {documentListeners[name] = handler;}};
const sandbox = vm.createContext({document, window: {addEventListener(name, handler) {windowListeners[name] = handler;}},
  location: {reload() {reloads++;}}, Blob: class {constructor(parts) {blobSource = parts.join('');}},
  URL: {createObjectURL() {return 'blob:local-test';}, revokeObjectURL() {}}, Worker: FakeWorker,
  setTimeout(fn) {timers.set(++timerId, fn); return timerId;}, clearTimeout(id) {timers.delete(id);}});
vm.runInContext(script, sandbox);
assert.equal(worker.queue[0].type, 'init');
assert.equal(element('data').removed, true);
worker.flush(); worker.flush();
assert.equal(clips, 3);
assert.equal(fills, 18);
assert.equal(element('frame').textContent, '1 / 3');
assert.equal(element('detail').textContent, '');
assert.equal(element('play').disabled, false);

element('scrub').value = '1'; element('scrub').listeners.input();
element('scrub').value = '2'; element('scrub').listeners.input();
assert.equal(worker.queue.length, 1);
worker.flush();
assert.equal(clips, 3);
assert.equal(worker.queue.length, 1);
worker.flush();
assert.equal(element('frame').textContent, '3 / 3');
assert.equal(clips, 6);

element('details').open = true; element('details').listeners.toggle();
assert.equal(JSON.parse(element('detail').textContent).snapshot.time_s, 2/60);
element('zoom').value = '3'; element('zoom').listeners.input();
element('follow').checked = true; element('follow').listeners.change();
assert.equal(JSON.stringify({scene, events}), original);
const rotated = vm.runInContext('rotate([1,0,0],[Math.SQRT1_2,0,0,Math.SQRT1_2])', sandbox);
assert.ok(Math.abs(rotated[0]) < 1e-12 && Math.abs(rotated[1]-1) < 1e-12 && Math.abs(rotated[2]) < 1e-12);

element('play').listeners.click(); assert.equal(timers.size, 1);
document.hidden = true; documentListeners.visibilitychange(); assert.equal(timers.size, 0);
windowListeners.pagehide(); assert.equal(worker.terminated, true);
windowListeners.pageshow({persisted: true}); assert.equal(reloads, 1);
console.log('Spatial replay checks passed: worker parsing, bounded latest-frame queue, three clipped views, deferred details, camera-only controls, quaternion axes, hidden pause, worker cleanup, cached-page recovery. No browser was used.');
