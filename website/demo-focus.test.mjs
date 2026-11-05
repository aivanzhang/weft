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
    window: {}, URLSearchParams, AbortSignal,
    location: { hash: '#token=secret', pathname: '/demo', search: '' },
    sessionStorage: { getItem: (key) => saved.get(key), setItem: (key, value) => saved.set(key, value) },
    history: { replaceState: (_, __, url) => { cleanURL = url; } },
    document: { getElementById: element, querySelectorAll: () => tabs, activeElement: element('opener') },
  };
  vm.runInNewContext(source, context);
  assert.equal(cleanURL, '/demo');
  assert.equal(saved.get('weft-portal-token'), 'secret');
  const viewer = context.window.WeftRunFocus();
  viewer.open({ world: 'a'.repeat(32), video: '/recording.mp4', source: '/original.mp4', view: 'world',
    comparison: { result_label: 'Controlled body', result_note: 'A visual stand-in for the person.' } });
  assert.equal(element('studio-result-label').textContent, 'Controlled body');
  assert.equal(element('studio-result-note').textContent, '');
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
  assert.equal(element('studio-source-video').src, '/original.mp4');
  tabs[2].listeners.click();
  assert.equal(element('result-media').hidden, true);
  assert.equal(element('studio-video').src, undefined);
  assert.equal(element('studio-results').hidden, false);
  assert.equal(element('studio-source-video').src, undefined);
  viewer.close();
  assert.equal(element('world-view').src, undefined);
  assert.equal(element('studio-video').src, undefined);
  viewer.open({ video: '/recording.mp4', source: '/original.mp4' });
  assert.equal(viewer.active, 'recording', 'studio opens on the comparison by default');
  assert.equal(tabs[0].hidden, true, 'unavailable live controls do not clutter saved examples');
  assert.equal(element('studio-original').hidden, false);
  viewer.close();
  viewer.open({ video: '/recording-only.mp4' });
  assert.equal(element('studio-original').hidden, true);
  assert.equal(element('studio-source-video').src, undefined, 'missing source never reuses earlier footage');
  assert.equal(tabs[1].textContent, 'Recording');
  viewer.close();
  context.location.hash = '';
  vm.runInNewContext(source, context);
  assert.equal(context.window.WeftDemoSession.token, 'secret');
});

test('access links remain usable without browser storage', () => {
  const context = {
    window: {}, URLSearchParams, AbortSignal,
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
    window: {}, URLSearchParams, AbortSignal,
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

test('visitors can see curated recordings while a prefilled access form never submits or fetches private APIs', async () => {
  const nodes = new Map();
  const makeNode = () => ({ value: '', hidden: false, children: [], listeners: {}, attributes: {},
    pause() {}, load() {}, removeAttribute() {},
    addEventListener(type, listener) { this.listeners[type] = listener; },
    replaceChildren(...children) { this.children = children; },
    setAttribute(name, value) { this.attributes[name] = value; } });
  const node = (id) => { if (!nodes.has(id)) nodes.set(id, makeNode()); return nodes.get(id); };
  const requests = [];
  const context = {
    window: {}, URLSearchParams, AbortSignal,
    location: { hash: '', pathname: '/demo', search: '?access_code=DEMO-EXAMPLE' },
    sessionStorage: { getItem: () => '', setItem() {} },
    history: { replaceState() {} },
    document: { getElementById: node, querySelectorAll: () => [], addEventListener() {}, createElement: makeNode },
    setTimeout, clearTimeout, AbortSignal,
    fetch: async (path) => {
      requests.push(path);
      assert.equal(path, '/api/demo/showcase', 'visitors never request private APIs');
      return { ok: true, json: async () => ({ home_url: 'http://127.0.0.1:3001/', items: [{
        id: 'curated', title: 'Recorded exercise', video_url: '/api/demo/showcase/curated/video', readonly: true,
      }] }) };
    },
  };
  node('builder').hidden = true;
  vm.runInNewContext(source, context);
  const build = await readFile(new URL('../src/weft/assets/demo_build.js', import.meta.url), 'utf8');
  vm.runInNewContext(build, context);
  for (let turn = 0; turn < 3; turn += 1) await new Promise(setImmediate);
  assert.equal(node('gate-code').value, 'DEMO-EXAMPLE');
  assert.equal(typeof node('gate-form').listeners.submit, 'function');
  assert.equal(node('access-menu').attributes['data-expanded'], 'true');
  assert.equal(node('gate-panel').inert, false);
  assert.deepEqual(requests, ['/api/demo/showcase']);
  assert.equal(node('finished-content').hidden, false);
  assert.equal(node('demo-title').textContent, 'Recorded exercise');
  assert.equal(node('builder').hidden, true);
  assert.equal(node('home-link').href, 'http://127.0.0.1:3001/');
});

test('elapsed time ticks between polls, never rewinds live, and releases hidden or finished timers', () => {
  const listeners = new Map(), timers = new Map(), paints = [];
  let time = 1000, nextTimer = 0;
  const context = {
    window: {}, URLSearchParams, AbortSignal,
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
  const context = { window: {}, URLSearchParams, AbortSignal, location: { hash: '', search: '', pathname: '/demo' },
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

test('the examples gallery loads saved results without polling the builder or displaying its progress', async () => {
  const nodes = new Map();
  const makeNode = () => ({ children: [], style: {}, hidden: false, textContent: '',
    classList: { add() {}, remove() {} },
    addEventListener() {}, setAttribute() {}, pause() {}, load() {}, removeAttribute() {},
    replaceChildren(...children) { this.children = children; },
    append(...children) { this.children.push(...children); },
  });
  const node = (id) => { if (!nodes.has(id)) nodes.set(id, makeNode()); return nodes.get(id); };
  let statesRead = 0;
  const context = {
    window: {}, URLSearchParams, AbortSignal, setTimeout, clearTimeout, performance, AbortSignal,
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
      if (path === '/api/demo/featured') return { featured: ['run-deadbeef'] };
      if (path === '/api/demo/gallery') return { items: [{
        id: 'run-deadbeef', title: 'Older private robot run', video_url: '/private.mp4',
      }, {
        id: 'saved-example', title: 'Completed exercise', video_url: '/saved.mp4', metrics: 'task success',
      }] };
      return { steps: [], featured: [], clips: [], items: [] };
    } }),
  };
  vm.runInNewContext(source, context);
  const build = await readFile(new URL('../src/weft/assets/demo_build.js', import.meta.url), 'utf8');
  vm.runInNewContext(build, context);
  for (let turn = 0; turn < 5; turn += 1) await new Promise(setImmediate);
  assert.equal(statesRead, 0);
  assert.equal(node('builder').hidden, true);
  assert.equal(node('steps').children.length, 0);
  assert.equal(node('finished-content').hidden, false);
  assert.equal(node('finished-empty').hidden, true);
  assert.equal(node('demo-title').textContent, 'Completed exercise');
  assert.equal(node('open-studio').hidden, false);
  assert.equal(typeof node('open-studio').onclick, 'function');
});


function galleryFixture() {
  const nodes = new Map(), calls = [], timeouts = [];
  const node = (id) => {
    if (!nodes.has(id)) nodes.set(id, {
      hidden: false, children: [], listeners: {}, attributes: {},
      addEventListener(name, callback) { this.listeners[name] = callback; },
      setAttribute(name, value) { this.attributes[name] = value; },
      pause() {}, load() {}, removeAttribute() {},
      replaceChildren(...items) { this.children = items; },
    });
    return nodes.get(id);
  };
  const context = {
    location: { pathname: "/demo" }, clearTimeout,
    window: { WeftDemoSession: { token: '', accessCode: '' },
      WeftRunFocus: () => ({}), WeftElapsedClock: () => ({}), WeftCarousel() {}, WeftComparisonCopy() {} },
    document: { getElementById: node, querySelectorAll: () => [], addEventListener() {},
      createElement: () => node(Symbol()) },
    AbortSignal: { timeout(ms) {
      const controller = new AbortController();
      timeouts.push({ ms, controller });
      return controller.signal;
    } },
    fetch: (path, options) => new Promise((resolve, reject) => {
      calls.push({ path, options, resolve, reject });
      options.signal.addEventListener('abort', () => reject(new Error('Timed out')));
    }),
  };
  return { node, calls, timeouts, context };
}
const settleGallery = async () => { for (let i = 0; i < 4; i += 1) await new Promise(setImmediate); };

test('gallery timeout ends loading; retry is deduplicated and replaces the placeholder with recordings', async () => {
  const f = galleryFixture();
  const build = await readFile(new URL('../src/weft/assets/demo_build.js', import.meta.url), 'utf8');
  vm.runInNewContext(build, f.context);
  assert.equal(f.node('finished-loading').hidden, false);
  assert.equal(f.node('explore').attributes['aria-busy'], 'true');
  assert.equal(f.timeouts[0].ms, 15000);
  f.timeouts[0].controller.abort();
  await settleGallery();
  assert.equal(f.node('finished-loading').hidden, true);
  assert.equal(f.node('finished-retry').hidden, false);
  assert.equal(f.node('explore').attributes['aria-busy'], 'false');
  f.node('finished-retry').listeners.click();
  f.node('finished-retry').listeners.click();
  assert.equal(f.calls.length, 2);
  assert.equal(f.node('finished-loading').hidden, false);
  f.calls[1].resolve({ ok: true, json: async () => ({ items: [
    { id: 'example', title: 'Example world', video_url: '/example.mp4' },
  ] }) });
  await settleGallery();
  assert.equal(f.node('finished-loading').hidden, true);
  assert.equal(f.node('finished-content').hidden, false);
  assert.equal(f.node('finished-empty').hidden, true);
  assert.equal(f.node('finished-nav').hidden, true, 'one recording needs no previous/next buttons');
  assert.equal(f.node('demo-title').textContent, 'Example world');
  assert.ok(f.calls.every((call) => call.path === '/api/demo/showcase' && !call.options.method));
});

test('an empty gallery shows an honest empty state instead of an endless loader', async () => {
  const f = galleryFixture();
  const build = await readFile(new URL('../src/weft/assets/demo_build.js', import.meta.url), 'utf8');
  vm.runInNewContext(build, f.context);
  f.calls[0].resolve({ ok: true, json: async () => ({ items: [] }) });
  await settleGallery();
  assert.equal(f.node('finished-loading').hidden, true);
  assert.equal(f.node('finished-content').hidden, true);
  assert.equal(f.node('finished-empty').hidden, false);
  assert.equal(f.node('finished-retry').hidden, true);
  assert.match(f.node('finished-message-detail').textContent, /Finished experiments will appear here/);
});

async function inlineGateFixture() {
  const build = await readFile(new URL('../src/weft/assets/demo_build.js', import.meta.url), 'utf8');
  const gate = build.slice(build.indexOf('  /* ---------------- gate'), build.indexOf('  /* ---------------- my runs'));
  const nodes = new Map(), events = {}, requests = [];
  let focused, scrolled, loaded = 0;
  const node = (id) => {
    if (!nodes.has(id)) nodes.set(id, {
      hidden: false, attributes: {}, listeners: {}, value: '',
      setAttribute(name, value) { this.attributes[name] = value; },
      addEventListener(name, callback) { this.listeners[name] = callback; },
      focus() { focused = id; }, scrollIntoView() { scrolled = id; },
      contains(target) { return [...nodes.values()].includes(target); },
    });
    return nodes.get(id);
  };
  const context = {
    $: node, token: '', studioId: null, buildPage: true, auth: {}, root: { WeftDemoSession: { accessCode: '', setToken() {} } },
    document: { addEventListener(name, callback) { events[name] = callback; } },
    sessionStorage: { setItem() {} }, keepToken() {}, renderMine() {}, loadDemo() { loaded += 1; },
    api: (path, options) => new Promise((resolve, reject) => requests.push({ path, options, resolve, reject })),
  };
  vm.runInNewContext(gate, context);
  return { node, events, requests, status: () => ({ focused, scrolled, loaded }) };
}

test('Build reveals an inline code field; Escape and outside clicks close it without submitting', async () => {
  const f = await inlineGateFixture();
  assert.equal(f.node('gate-panel').inert, true);
  f.node('show-builder').listeners.click();
  assert.equal(f.node('show-builder').attributes['aria-expanded'], 'true');
  assert.equal(f.node('gate-panel').inert, false);
  assert.equal(f.status().focused, 'gate-code');
  assert.equal(f.node('show-builder').hidden, true);
  f.events.keydown({ key: 'Escape' });
  assert.equal(f.node('gate-panel').inert, true);
  assert.equal(f.status().focused, 'show-builder');
  assert.equal(f.node('show-builder').hidden, false);
  f.node('show-builder').listeners.click();
  f.events.pointerdown({ target: f.node('gate-code') });
  assert.equal(f.node('gate-panel').inert, false);
  f.events.pointerdown({ target: {} });
  assert.equal(f.node('gate-panel').inert, true);
  assert.equal(f.requests.length, 0);
});

test('inline unlock deduplicates submission, preserves errors, and becomes a direct Build action after success', async () => {
  const f = await inlineGateFixture();
  f.node('show-builder').listeners.click();
  f.node('gate-code').value = 'MY-CODE';
  const submit = () => f.node('gate-form').listeners.submit({ preventDefault() {} });
  const failed = submit();
  await submit();
  assert.equal(f.requests.length, 1);
  assert.equal(f.node('gate-submit').disabled, true);
  assert.equal(f.requests[0].path, '/api/demo/unlock');
  assert.deepEqual(JSON.parse(f.requests[0].options.body), { code: 'MY-CODE' });
  f.requests[0].reject(new Error('That code is not valid.'));
  await failed;
  assert.equal(f.node('gate-error').textContent, 'That code is not valid.');
  assert.equal(f.node('gate-code').value, 'MY-CODE');
  assert.equal(f.node('gate-submit').disabled, false);
  const success = submit();
  f.requests[1].resolve({ token: 'test-session' });
  await success;
  assert.equal(f.node('gate-panel').hidden, true);
  assert.equal(f.node('gate-panel').inert, true);
  assert.equal(f.node('gate-form').attributes['aria-busy'], 'false');
  assert.equal(f.node('show-builder').attributes['aria-controls'], 'builder');
  assert.equal(f.status().loaded, 1);
  f.node('show-builder').listeners.click();
  assert.equal(f.status().scrolled, 'builder');
  assert.equal(f.requests.length, 2);
});

test('carousel supports directional swipes and keyboard arrows without taking vertical scroll or controls', () => {
  const listeners = {}, moves = [];
  const element = { addEventListener: (name, listener) => { listeners[name] = listener; } };
  const context = { window: {}, URLSearchParams, location: { hash: '', search: '', pathname: '/demo' },
    sessionStorage: { getItem: () => '' } };
  vm.runInNewContext(source, context);
  context.window.WeftCarousel(element, (direction) => moves.push(direction));
  const target = { closest: () => null };
  const down = (x = 200, y = 100, extra = {}) => listeners.pointerdown({ target, button: 0, pointerId: 1, clientX: x, clientY: y, ...extra });
  const up = (x, y = 100) => listeners.pointerup({ pointerId: 1, clientX: x, clientY: y });
  down(); up(80); // Left advances.
  assert.deepEqual(moves, [1]);
  let prevented = 0;
  listeners.click({ preventDefault() { prevented++; }, stopPropagation() {} });
  assert.equal(prevented, 1, 'a swipe cannot also open the studio');
  down(); up(280);
  down(); up(180, 240); // Vertical scroll.
  down(); up(180); // Tap jitter.
  down(); listeners.pointercancel(); up(80);
  down(200, 100, { target: { closest: () => ({}) } }); up(80);
  assert.deepEqual(moves, [1, -1]);
  listeners.keydown({ target, key: 'ArrowRight', preventDefault() {} });
  listeners.keydown({ target, key: 'ArrowLeft', preventDefault() {} });
  listeners.keydown({ target: { closest: () => ({}) }, key: 'ArrowRight' });
  assert.deepEqual(moves, [1, -1, 1, -1]);
});

test('example carousel navigates to specific studio URLs and clears stale previews', async () => {
  const nodes = new Map(), navigations = [];
  const makeNode = () => ({ children: [], hidden: false, attributes: {}, listeners: {}, dataset: {},
    addEventListener(name, listener) { this.listeners[name] = listener; },
    setAttribute(name, value) { this.attributes[name] = value; },
    removeAttribute(name) { delete this[name]; },
    replaceChildren(...children) { this.children = children; },
    append(...children) { this.children.push(...children); },
    pause() {}, load() {}, focus() {}, scrollTo() {},
  });
  const node = (id) => { if (!nodes.has(id)) nodes.set(id, makeNode()); return nodes.get(id); };
  const tabs = ['recording', 'world', 'results'].map((view) => {
    const tab = makeNode(); tab.dataset.runView = view; return tab;
  });
  const context = { window: {}, URLSearchParams, AbortSignal, setTimeout, clearTimeout,
    location: { hash: '', search: '', pathname: '/demo', assign: (url) => navigations.push(url) }, sessionStorage: { getItem: () => '' },
    document: { getElementById: node, createElement: makeNode, addEventListener() {},
      querySelectorAll: (selector) => selector === '[data-run-view]' ? tabs : [],
      body: { classList: { add() {}, remove() {} } } },
    fetch: async (path) => {
      assert.equal(path, '/api/demo/showcase', 'opening curated examples never calls a private API');
      return { ok: true, json: async () => ({ items: [
        { id: 'rolling-motion', title: 'Rolling ball', video_url: '/ball.mp4' },
        { id: 'paired', title: 'Movement', caption: 'A controlled simulation.', metrics: '246 control steps',
          video_url: '/result.mp4', source_video_url: '/original.mp4', source_credit: 'Author · license' },
        { id: 'single', title: 'Recording only', video_url: '/single.mp4' },
      ] }) };
    } };
  vm.runInNewContext(source, context);
  vm.runInNewContext(await readFile(new URL('../src/weft/assets/demo_build.js', import.meta.url), 'utf8'), context);
  await settleGallery();
  assert.equal(node('demo-original').hidden, false);
  assert.equal(node('demo-source-video').src, '/original.mp4#t=0.1');
  node('demo-video').listeners.play?.();
  assert.deepEqual(navigations, [], 'starting a video must never navigate to Studio');
  await node('open-studio').onclick();
  assert.deepEqual(navigations, ['/demo/studio/paired']);
  node('next').listeners.click();
  assert.equal(node('example-count').textContent, '2 / 3');
  assert.equal(node('demo-original').hidden, true);
  assert.equal(node('demo-source-video').src, undefined);
  await node('open-studio').onclick();
  assert.deepEqual(navigations, ['/demo/studio/paired', '/demo/studio/single']);
  node('next').listeners.click();
  assert.equal(node('example-count').textContent, '3 / 3');
  assert.equal(node('demo-title').textContent, 'Rolling ball', 'ball is last even with a stale server catalog');
});

async function directStudioFixture(identity, authenticated = false) {
  const nodes = new Map(), calls = [], opened = [], modes = [];
  const node = (id) => {
    if (!nodes.has(id)) nodes.set(id, {
      hidden: false, attributes: {}, listeners: {}, children: [], style: {},
      setAttribute(key, value) { this.attributes[key] = value; },
      addEventListener(key, listener) { this.listeners[key] = listener; },
      replaceChildren(...items) { this.children = items; },
      append(...items) { this.children.push(...items); },
      scrollTo() {}, focus() {},
    });
    return nodes.get(id);
  };
  const example = { id: authenticated ? 'run-deadbeef' : 'paired', title: 'Movement',
    video_url: '/recording.mp4', source_video_url: '/source.mp4' };
  const context = {
    window: {
      WeftDemoSession: { token: authenticated ? 'private-session' : '', accessCode: '', stored: true },
      WeftRunFocus: (mode) => { modes.push(mode); return { open: (record) => opened.push(record), close() {} }; },
      WeftCarousel() {}, WeftElapsedClock: () => ({}), WeftStudioResults() {},
    },
    location: { pathname: `/demo/studio/${identity}` }, AbortSignal,
    document: { getElementById: node, addEventListener() {}, querySelectorAll: () => [],
      createElement: () => node(Symbol()), body: { classList: { add() { throw new Error('Studio must not lock the page'); } } } },
    fetch: async (path, options) => {
      calls.push({ path, options });
      if (path === '/api/demo/studio/run-deadbeef') return { ok: true, json: async () => ({ id: example.id, title: 'Saved run', steps: [] }) };
      assert.equal(path, authenticated ? '/api/demo/gallery' : '/api/demo/showcase');
      return { ok: true, json: async () => ({ items: [example] }) };
    },
  };
  vm.runInNewContext(await readFile(new URL('../src/weft/assets/demo_build.js', import.meta.url), 'utf8'), context);
  await settleGallery();
  return { node, calls, opened, modes, context };
}

test('a direct public studio URL loads its specific comparison without a gallery or modal lock', async () => {
  const f = await directStudioFixture('paired');
  assert.equal(f.modes[0].page, true);
  assert.equal(f.node('studio-title').textContent, 'Movement');
  assert.equal(f.context.document.title, 'Movement · Weft');
  assert.equal(f.opened[0].video, '/recording.mp4');
  assert.equal(f.opened[0].source, '/source.mp4');
  assert.equal(f.node('studio-modal').attributes['aria-busy'], 'false');
  assert.deepEqual(f.calls.map((call) => call.path), ['/api/demo/showcase']);
});

test('missing studios show a recoverable error; private studios require access before making requests', async () => {
  const missing = await directStudioFixture('missing');
  assert.equal(missing.node('studio-title').textContent, 'Studio unavailable');
  assert.match(missing.node('studio-error').textContent, /no longer available/);
  const locked = await directStudioFixture('run-deadbeef');
  assert.equal(locked.calls.length, 0);
  assert.equal(locked.node('gate-panel').inert, false);
  assert.equal(locked.node('show-builder').hidden, true);
  assert.match(locked.node('studio-error').textContent, /access code/);
});

test('an authenticated private studio reloads saved results and recording through protected APIs', async () => {
  const f = await directStudioFixture('run-deadbeef', true);
  assert.equal(f.node('studio-title').textContent, 'Saved run');
  assert.equal(f.node('studio-error').hidden, true);
  assert.equal(f.calls.length, 2);
  assert.ok(f.calls.every((call) => call.options.headers.Authorization === 'Bearer private-session'));
  assert.match(f.opened[0].video, /^\/recording.mp4\?token=/);
});

test('building keeps stage previews mounted across status polling and after completion', async () => {
  const nodes = new Map(), timeouts = new Map();
  let nextTimer = 0, mounts = 0, disposals = 0;
  const makeNode = () => ({ children: [], style: {}, value: '', dataset: {}, listeners: {}, attributes: {}, classList: { add() {}, remove() {} },
    append(...children) { this.children.push(...children); }, replaceChildren(...children) { this.children = children; },
    addEventListener(key, fn) { this.listeners[key] = fn; }, setAttribute(key, value) { this.attributes[key] = value; }, removeAttribute(key) { delete this[key]; }, pause() {}, load() {}, focus() {} });
  const node = id => { if (!nodes.has(id)) nodes.set(id, makeNode()); return nodes.get(id); };
  const visuals = [{ kind: 'video', url: '/physics.mp4', label: 'Physics replay' }];
  let state = { status: 'idle', steps: [] };
  const context = { URLSearchParams, AbortSignal,
    location: { pathname: '/demo/build' },
    localStorage: { getItem: () => '[]', setItem() {} },
    setTimeout(fn) { timeouts.set(++nextTimer, fn); return nextTimer; }, clearTimeout(id) { timeouts.delete(id); },
    window: { WeftDemoSession: { token: 'test', accessCode: '' }, WeftRunFocus: () => ({}),
      WeftElapsedClock: () => ({ update() {} }), WeftCarousel() {},
      WeftStageVisuals(host) { mounts++; const video = makeNode(); host.append(video); return () => { disposals++; }; } },
    document: { getElementById: node, createElement: makeNode, querySelectorAll: () => [], addEventListener() {} },
    fetch: async path => ({ ok: true, json: async () => {
      if (path.includes('/clips')) return { clips: [{ id: 'bodyweight-squats', title: 'Squat', character: { prompt: 'Panda' } }] };
      if (path.includes('/start')) state = { status: 'running', run: 'run-active', steps: [{ id: 'rollout', status: 'done', detail: 'Recorded', visuals }] };
      return path.includes('/state') || path.includes('/start') ? structuredClone(state) : {};
    } }),
  };
  vm.runInNewContext(await readFile(new URL('../src/weft/assets/demo_build.js', import.meta.url), 'utf8'), context);
  await settleGallery();
  assert.equal(node('run').disabled, true, 'a video or example is required');
  node('clip-row').children[0].listeners.click();
  assert.equal(node('run').disabled, false);
  assert.equal(node('file-label').textContent, 'Squat');
  assert.match(node('input-preview').src, /bodyweight-squats/);
  node('toggle-runs').listeners.click(); assert.equal(node('mine').hidden, false);
  assert.equal(node('toggle-runs').attributes['aria-expanded'], 'true');
  node('close-runs').listeners.click(); assert.equal(node('mine').hidden, true);
  await node('run').listeners.click(); await settleGallery();
  assert.equal(mounts, 1);
  const video = node('steps').children[0].children[1].children[0];
  const poll = timeouts.values().next().value;
  state.steps[0].detail = 'Recording finished';
  await poll();
  assert.equal(mounts, 1);
  assert.equal(node('steps').children[0].children[1].children[0], video);
  state.status = 'complete';
  await [...timeouts.values()].at(-1)();
  assert.equal(node('steps').hidden, false, 'the just-finished build retains inspectable outputs');
  assert.equal(mounts, 1);
  assert.equal(disposals, 0);
});
