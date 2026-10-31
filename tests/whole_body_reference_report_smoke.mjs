/** Saved JSON and a scripted DOM only; no browser, network, media or native runtime. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {performance} from 'node:perf_hooks';

const path = process.argv[2];
assert(path && fs.statSync(path).isFile() && fs.statSync(path).size <= 2 * 1024 * 1024);
const html = fs.readFileSync(path, 'utf8');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 2);
const data = JSON.parse(scripts[0][1]);
let created = 0, fetches = 0, timers = 0;

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.children = []; this.attributes = {}; this.listeners = new Map();
    this.dataset = {}; this.style = {}; this.value = ''; this.checked = false; this.disabled = false; this.ownText = ''; created++;
  }
  set textContent(value) { this.ownText = String(value); this.children = []; }
  get textContent() { return this.ownText + this.children.map(child => child.textContent).join(''); }
  set innerHTML(_) { assert.fail('Untrusted metadata must not enter innerHTML'); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.ownText = ''; this.children = [...children]; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  getAttribute(key) { return this.attributes[key]; }
  addEventListener(kind, handler) { this.listeners.set(kind, handler); }
  fire(kind) { this.listeners.get(kind)?.({target:this}); }
  click() { if (!this.disabled) this.fire('click'); }
}

const ids = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(match => [match[1], new Element()]));
const el = id => { assert(ids.has(id), `Missing static report element ${id}`); return ids.get(id); };
el('data').textContent = scripts[0][1]; el('mode').value = 'source'; el('affected').checked = true;
const document = {getElementById:el, createElement:tag => new Element(tag), createElementNS:(_, tag) => new Element(tag)};
const sandbox = {document, fetch:() => { fetches++; assert.fail('Offline report requested network'); },
  setTimeout:() => { timers++; assert.fail('Offline report created a timer'); },
  setInterval:() => { timers++; assert.fail('Offline report created a poll'); }};
const start = performance.now();
vm.runInNewContext(scripts[1][1], sandbox, {timeout:1000});
const rowText = id => el(id).children.map(row => row.children.map(cell => cell.textContent));
const countNodes = node => 1 + node.children.reduce((n, child) => n + countNodes(child), 0);
const liveNodes = () => [...ids.values()].reduce((n, node) => n + countNodes(node), 0);

// A retained source slot without a reference stays selectable and explicitly unknown.
assert.equal(el('sample').children.length, data.source.length);
assert.equal(el('previous').disabled, true);
assert.match(el('identity').textContent, /Retained source slot 1\/3.*decoder time Unknown/);
assert.match(el('source-id').textContent, /:frame:0$/);
assert.equal(rowText('points')[1][4], 'Unknown');
// One source-level descriptor is visible only for the selected crop, never for grid/unknown targets.
const origin = data.source[0].target_origin;
if (origin) {
  assert.equal(el('target-label').textContent, 'Selected crop estimate, mapped to source image');
  assert.equal(el('target-details').hidden, false);
  assert.equal(el('target-metadata').children.length, 12);
  for (const key of ['method', 'original_observation_sha256', 'proposal_sha256', 'decision_sha256']) {
    assert(el('target-metadata').textContent.includes(origin[key]));
  }
  assert(el('target-metadata').textContent.includes('Landmark JSON pathlandmarks'));
  assert(!el('target-metadata').textContent.includes('image_refinement.landmarks'));
  el('target-details').open = true;
  const provenanceNodes = liveNodes();
  el('sample').fire('change');
  assert.equal(el('target-details').open, true);
  assert.equal(liveNodes(), provenanceNodes);
} else {
  assert.equal(el('target-label').textContent, 'Inferred whole-frame estimate');
  assert.equal(el('target-details').hidden, true);
}
const lines = el('lines').children.slice();
assert.equal(lines.length, 3);
assert.equal((lines[1].getAttribute('d').match(/ M/g) || []).length, 2, 'Missing geometry breaks the chart path');
el('next').click();
assert.equal(el('sample').value, 1);
assert.match(el('context').textContent, /No body reference/);
assert.match(el('penetration').textContent, /Input body: Unknown; Proposed body: Unknown/);
assert.equal(el('target-details').hidden, true);
assert.equal(el('target-metadata').children.length, 0);
assert.match(el('target-label').textContent, /Unavailable/);
assert.equal(el('before-dot').style.display, 'none');
assert.equal(el('after-dot').style.display, 'none');
assert(rowText('gaps').every(cells => cells[2] === 'Unknown' && cells[3] === 'Unknown'));
el('next').click(); el('next').click();
assert.equal(el('next').disabled, true);
assert.equal(el('sample').value, 2);
assert.equal(el('lines').children[1], lines[1], 'Changing a sample must not rebuild chart lines');

// Segment names are explicit, while endpoint signs and unknown primitive IDs stay exact.
el('previous').click(); el('previous').click();
assert.match(rowText('gaps')[0][0], /^Right shin capsule · endpoint −1$/);
assert.match(rowText('gaps')[0][1], /right_knee_geom:-1\.0 · right_knee_geom/);
const details = el('gaps').children[0].children[1].children[0];
assert.equal(details.tagName, 'DETAILS');
assert.equal(details.children[0].tagName, 'SUMMARY');
el('affected').checked = false; el('affected').fire('change');
assert.equal(el('gaps').children.length, data.support_names.length);
assert.match(rowText('gaps')[2][0], /^Unknown saved primitive · center sample$/);
assert.equal(details.children[0].textContent, 'Exact saved IDs');
assert.match(rowText('gaps')[2][1], /<\/script><img src=x onerror=/, 'Untrusted IDs remain literal text');
assert.equal(el('gaps').children.flatMap(row => row.children).filter(cell => cell.tagName === 'IMG').length, 0);

// Coincident control samples do not acquire source-image targets or direct observation IDs.
el('mode').value = 'grid'; el('mode').fire('change');
assert.equal(el('sample').value, '0');
assert.equal(el('sample').children.length, data.grid.length);
assert.match(el('identity').textContent, /Control-grid check 1\/3/);
assert.match(el('source-id').textContent, /No direct source observation ID/);
assert.match(el('context').textContent, /Body reference .*:frame:0 at 0\.00000 s/);
assert.match(rowText('points')[0][0], /Unavailable.*no source endpoint targets/);
assert.equal(el('target-details').hidden, true);
assert.equal(el('target-metadata').children.length, 0);
assert.match(el('target-label').textContent, /Unavailable/);
assert.notEqual(el('lines').children[1], lines[1]);
el('next').click();
assert.match(el('context').textContent, /:frame:0.*:frame:2/);
assert.equal(el('before-dot').style.display, '');

// Repeated detail navigation stays bounded and makes no requests, polls or control calls.
const gridLine = el('lines').children[1], baseline = liveNodes();
for (let i = 0; i < 40; i++) { el('sample').value = '1'; el('sample').fire('change'); }
assert.equal(liveNodes(), baseline);
assert.equal(el('lines').children[1], gridLine);
assert.equal(fetches, 0); assert.equal(timers, 0);
assert(!/\b(?:fetch|XMLHttpRequest|WebSocket|setInterval|requestAnimationFrame)\s*\(/.test(scripts[1][1]));
process.stdout.write(JSON.stringify({checks:5, fetches, timers, live_nodes:liveNodes(), created_nodes:created,
  seconds:(performance.now() - start) / 1000, measurement:'scripted DOM only; not browser responsiveness'}) + '\n');
