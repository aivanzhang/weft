import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const source = await readFile(new URL('../src/weft/assets/demo_focus.js', import.meta.url), 'utf8');

test('run views are exclusive and release hidden simulation and video resources', () => {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, {
      hidden: false, dataset: {}, attributes: {}, listeners: {}, paused: 0,
      setAttribute(name, value) { this.attributes[name] = value; },
      removeAttribute(name) { delete this[name]; delete this.attributes[name]; },
      addEventListener(name, listener) { this.listeners[name] = listener; },
      pause() { this.paused += 1; }, load() {}, focus() {},
    });
    return elements.get(id);
  };
  const tabs = ['world', 'recording', 'results'].map((name) => {
    const tab = element('tab-' + name);
    tab.dataset.runView = name;
    return tab;
  });
  const saved = new Map();
  let cleanURL;
  const context = {
    window: {}, URLSearchParams,
    location: { hash: '#token=secret', pathname: '/demo', search: '' },
    sessionStorage: { getItem: (key) => saved.get(key), setItem: (key, value) => saved.set(key, value) },
    history: { replaceState: (_, __, url) => { cleanURL = url; } },
    document: { getElementById: element, querySelectorAll: () => tabs, activeElement: element('opener') },
  };
  vm.runInNewContext(source, context);
  assert.equal(cleanURL, '/demo');
  assert.equal(saved.get('weft-portal-token'), 'secret');
  const viewer = context.window.WeftRunFocus();
  viewer.open({ world: 'a'.repeat(32), video: '/recording.mp4' });
  assert.equal(element('world-frame').hidden, false);
  assert.equal(element('result-media').hidden, true);
  assert.equal(element('studio-results').hidden, true);
  assert.match(element('world-view').src, /#token=secret&embed=1$/);
  const pauses = element('studio-video').paused;
  tabs[0].listeners.click();
  assert.equal(element('studio-video').paused, pauses, 'selecting the active view does no work');
  tabs[1].listeners.click();
  assert.equal(element('world-view').src, undefined);
  assert.equal(element('world-frame').hidden, true);
  assert.equal(element('result-media').hidden, false);
  assert.equal(element('studio-video').src, '/recording.mp4');
  tabs[2].listeners.click();
  assert.equal(element('result-media').hidden, true);
  assert.equal(element('studio-video').src, undefined);
  assert.equal(element('studio-results').hidden, false);
  viewer.close();
  assert.equal(element('world-view').src, undefined);
  assert.equal(element('studio-video').src, undefined);
  context.location.hash = '';
  vm.runInNewContext(source, context);
  assert.equal(context.window.WeftDemoSession.token, 'secret');
});

test('access links remain usable without browser storage', () => {
  const context = {
    window: {}, URLSearchParams,
    location: { hash: '#token=private', pathname: '/demo', search: '' },
    sessionStorage: { getItem() { throw new Error('disabled'); }, setItem() { throw new Error('disabled'); } },
    history: { replaceState() {} },
  };
  vm.runInNewContext(source, context);
  assert.equal(context.window.WeftDemoSession.token, 'private');
  assert.equal(context.window.WeftDemoSession.stored, false);
});

test('code links prefill without logging in and can accept a later authenticated session', () => {
  let cleaned;
  const stored = new Map();
  const context = {
    window: {}, URLSearchParams,
    location: { hash: '', pathname: '/demo', search: '?access_code=DEMO-EXAMPLE&keep=yes' },
    sessionStorage: { getItem: (key) => stored.get(key), setItem: (key, value) => stored.set(key, value) },
    history: { replaceState: (_, __, value) => { cleaned = value; } },
  };
  vm.runInNewContext(source, context);
  const session = context.window.WeftDemoSession;
  assert.equal(session.accessCode, 'DEMO-EXAMPLE');
  assert.equal(session.token, '');
  assert.equal(cleaned, '/demo?keep=yes');
  session.setToken('authenticated');
  assert.equal(session.token, 'authenticated');
  assert.equal(session.stored, true);
  assert.equal(stored.get('weft-portal-token'), 'authenticated');
});

test('a fresh code link fills the gate without fetching protected APIs or submitting', async () => {
  const nodes = new Map();
  const node = (id) => {
    if (!nodes.has(id)) nodes.set(id, { value: '', listeners: {},
      addEventListener(type, listener) { this.listeners[type] = listener; } });
    return nodes.get(id);
  };
  let requests = 0;
  const context = {
    window: {}, URLSearchParams,
    location: { hash: '', pathname: '/demo', search: '?access_code=DEMO-EXAMPLE' },
    sessionStorage: { getItem: () => '', setItem() {} },
    history: { replaceState() {} },
    document: { getElementById: node, querySelectorAll: () => [], addEventListener() {} },
    setTimeout, clearTimeout,
    fetch() { requests += 1; throw new Error('Unexpected request before unlock'); },
  };
  vm.runInNewContext(source, context);
  const build = await readFile(new URL('../src/weft/assets/demo_build.js', import.meta.url), 'utf8');
  vm.runInNewContext(build, context);
  assert.equal(node('gate-code').value, 'DEMO-EXAMPLE');
  assert.equal(typeof node('gate-form').listeners.submit, 'function');
  assert.equal(requests, 0);
});

test('elapsed time ticks between polls, never rewinds live, and releases hidden or finished timers', () => {
  const listeners = new Map(), timers = new Map(), paints = [];
  let time = 1000, nextTimer = 0;
  const context = {
    window: {}, URLSearchParams,
    location: { hash: '', pathname: '/demo', search: '' },
    sessionStorage: { getItem: () => '' },
    document: { hidden: false,
      addEventListener: (event, fn) => listeners.set(event, fn),
      removeEventListener: (event) => listeners.delete(event) },
  };
  vm.runInNewContext(source, context);
  const clock = context.window.WeftElapsedClock((elapsed, running) => paints.push([elapsed, running]), {
    now: () => time,
    schedule: (fn, delay) => { assert.equal(delay, 100); timers.set(++nextTimer, fn); return nextTimer; },
    cancel: (id) => timers.delete(id),
  });
  const tick = () => { const [id, fn] = timers.entries().next().value; timers.delete(id); fn(); };
  clock.update({ id: 'live', elapsed: 5, running: true });
  time += 100; tick();
  assert.equal(paints.at(-1)[0], 5.1);
  time += 1100;
  clock.update({ id: 'live', elapsed: 5.2, running: true });
  assert.equal(paints.at(-1)[0], 6.2, 'a stale poll cannot rewind the clock');
  assert.equal(timers.size, 1);
  context.document.hidden = true;
  listeners.get('visibilitychange')();
  assert.equal(timers.size, 0);
  time += 10000;
  context.document.hidden = false;
  listeners.get('visibilitychange')();
  assert.equal(paints.at(-1)[0], 16.2);
  assert.equal(timers.size, 1);
  clock.update({ id: 'live', elapsed: 16, running: false });
  assert.equal(timers.size, 0);
  assert.deepEqual(paints.at(-1), [16, false], 'terminal duration is the actual recorded value');
  clock.update({ id: 'new', elapsed: 0, running: true });
  assert.equal(paints.at(-1)[0], 0);
  clock.dispose();
  assert.equal(timers.size, 0);
  assert.equal(listeners.size, 0);
});

test('saved results summarize recorded metrics without replaying step status or timing', () => {
  const context = { window: {}, URLSearchParams, location: { hash: '', search: '', pathname: '/demo' },
    sessionStorage: { getItem: () => '' } };
  vm.runInNewContext(source, context);
  const record = { steps: [
    { id: 'world', status: 'done', elapsed_seconds: 12, metrics: { physics: 'MuJoCo' } },
    { id: 'avatar', status: 'done', elapsed_seconds: 5, metrics: { 'task success': 'true', 'tracking error': '0.04 rad' } },
  ] };
  const before = JSON.stringify(record);
  const entries = Array.from(context.window.WeftRunResults(record), (entry) => Array.from(entry));
  assert.deepEqual(entries, [['physics', 'MuJoCo'], ['task success', 'true'], ['tracking error', '0.04 rad']]);
  assert.equal(JSON.stringify(record), before);
});

test('an already completed run loads results without displaying processing steps or a progress timer', async () => {
  const nodes = new Map();
  const makeNode = () => ({ children: [], style: {}, hidden: false, textContent: '',
    classList: { add() {}, remove() {} },
    addEventListener() {}, setAttribute() {},
    replaceChildren(...children) { this.children = children; },
    append(...children) { this.children.push(...children); },
  });
  const node = (id) => { if (!nodes.has(id)) nodes.set(id, makeNode()); return nodes.get(id); };
  let statesRead = 0;
  const context = {
    window: {}, URLSearchParams, setTimeout, clearTimeout, performance,
    location: { hash: '#token=connected', pathname: '/demo', search: '', origin: 'http://127.0.0.1:8765' },
    sessionStorage: { getItem: () => '', setItem() {} },
    localStorage: { getItem: () => '[]', setItem() {} },
    history: { replaceState() {} },
    document: { getElementById: node, querySelectorAll: () => [], addEventListener() {},
      createElement: makeNode, createTextNode: (text) => ({ textContent: text }) },
    fetch: async (path) => ({ ok: true, json: async () => {
      if (path === '/api/demo/state') {
        statesRead += 1;
        return { status: 'complete', run: 'saved', elapsed_seconds: 47,
          steps: [{ id: 'avatar', status: 'done', metrics: { 'task success': 'true' } }] };
      }
      if (path === '/api/demo/gallery') return { items: [{
        id: 'saved-example', title: 'Completed exercise', video_url: '/saved.mp4', metrics: 'task success',
      }] };
      return { steps: [], featured: [], clips: [], items: [] };
    } }),
  };
  vm.runInNewContext(source, context);
  const build = await readFile(new URL('../src/weft/assets/demo_build.js', import.meta.url), 'utf8');
  vm.runInNewContext(build, context);
  for (let turn = 0; turn < 5; turn += 1) await new Promise(setImmediate);
  assert.equal(statesRead, 1);
  assert.equal(node('steps').hidden, true);
  assert.equal(node('steps').children.length, 0);
  assert.equal(node('progress').hidden, true);
  assert.equal(node('finished-content').hidden, false);
  assert.equal(node('finished-empty').hidden, true);
  assert.equal(node('demo-title').textContent, 'Completed exercise');
  assert.equal(node('open-studio').textContent, 'Watch recording');
});
