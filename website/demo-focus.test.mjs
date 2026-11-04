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
  const tabs = ['world', 'recording', 'evidence'].map((name) => {
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
  assert.equal(element('studio-steps').hidden, true);
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
  assert.equal(element('studio-steps').hidden, false);
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
    fetch() { requests += 1; throw new Error('Unexpected request before unlock'); },
  };
  vm.runInNewContext(source, context);
  const build = await readFile(new URL('../src/weft/assets/demo_build.js', import.meta.url), 'utf8');
  vm.runInNewContext(build, context);
  assert.equal(node('gate-code').value, 'DEMO-EXAMPLE');
  assert.equal(typeof node('gate-form').listeners.submit, 'function');
  assert.equal(requests, 0);
});
