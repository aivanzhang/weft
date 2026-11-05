import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import { performance } from 'node:perf_hooks';

const source = await readFile(new URL('../src/weft/assets/demo_studio.js', import.meta.url), 'utf8');
const model = JSON.parse(await readFile(new URL('../assets/demo/studio/body-model.json', import.meta.url), 'utf8'));
const manifest = JSON.parse(await readFile(new URL('../assets/demo/studio/waiters-squat-panda.json', import.meta.url), 'utf8'));

function fixture() {
  const nodes = [], observers = [], timers = new Map(), events = {}, requests = [], points = [];
  let next = 0;
  const ctx = { clearRect() {}, fillRect() {}, beginPath() {}, stroke() {}, fill() {},
    moveTo(...p) { points.push(p); }, lineTo(...p) { points.push(p); }, arc(...p) { points.push(p.slice(0, 2)); } };
  function create(tag = 'div') {
    const item = { tag, children: [], attributes: {}, listeners: {}, hidden: false,
      append(...items) { items.forEach(x => { x.parent = this; this.children.push(x); }); },
      replaceChildren(...items) { this.children = []; this.append(...items); },
      remove() { if (this.parent) this.parent.children = this.parent.children.filter(x => x !== this); },
      setAttribute(k, v) { this.attributes[k] = v; }, removeAttribute(k) { delete this.attributes[k]; delete this[k]; },
      addEventListener(k, fn) { this.listeners[k] = fn; }, pause() { this.paused = true; }, load() {},
      getContext() { return ctx; }, get firstElementChild() { return this.children[0]; }, get lastChild() { return this.children.at(-1); } };
    nodes.push(item); return item;
  }
  const body = create(); body.id = 'modal-body'; body.append(create());
  const context = { window: { addEventListener(k, fn) { events[k] = fn; }, WeftDemoSession: { token: 'secret' } },
    AbortController, AbortSignal, console, performance,
    document: { hidden: false, createElement: create, getElementById: id => nodes.find(x => x.id === id),
      addEventListener(k, fn) { events[k] = fn; }, removeEventListener(k) { delete events[k]; } },
    IntersectionObserver: class { constructor(callback) { this.callback = callback; this.targets = []; observers.push(this); } observe(target) { this.targets.push(target); } disconnect() { this.disconnected = true; } },
    requestAnimationFrame(fn) { timers.set(++next, fn); return next; }, cancelAnimationFrame(id) { timers.delete(id); },
    fetch: async (url) => { requests.push(url); return { ok: true, json: async () => url.endsWith('body-model.json') ? model : manifest }; },
  };
  vm.runInNewContext(source, context);
  return { context, body, nodes, observers, timers, events, requests, points, create,
    tick(now) { const [id, fn] = timers.entries().next().value; timers.delete(id); fn(now); } };
}
const settle = async () => { for (let i = 0; i < 4; i++) await new Promise(setImmediate); };

test('studio starts with a recorded model and video, then actual ordered evidence without eager JSON previews', async () => {
  const f = fixture();
  f.context.window.WeftStudioPage.open({ record: { id: 'waiters-squat-panda' },
    entry: { id: 'waiters-squat-panda', video_url: '/result.mp4', source_video_url: '/source.mp4' }, signed: value => value });
  await settle();
  const tour = f.body.children.find(x => x.id === 'studio-tour');
  assert.equal(tour.children[1].id, 'finished-result');
  const panels = tour.children[1].children;
  assert.deepEqual(panels.map(panel => panel.children[0].children[0].textContent), ['00 / INITIAL', '01 / MODEL', '02 / RECORDING']);
  assert.equal(panels[0].children[1].src, '/source.mp4');
  assert.equal(panels[2].children[1].src, '/result.mp4');
  const knobs = f.nodes.find(item => item.className === 'studio-knobs');
  assert.equal(knobs.parent.className, 'studio-model-host');
  assert.equal(knobs.attributes['aria-label'], 'Model controls');
  assert.equal(knobs.children.length, 3);
  assert.equal(f.nodes.some(item => item.className === 'studio-settings'), false);
  assert.equal(tour.children[2].id, 'pipeline');
  assert.equal(f.nodes.filter(x => x.tag === 'article').length, 4);
  assert.equal(f.nodes.filter(x => x.tag === 'video').length, 4);
  assert.ok(f.nodes.filter(x => x.tag === 'video').every(x => x.poster || x.preload === 'metadata'), 'every video has a poster or loads a preview frame');
  assert.equal(f.nodes.some(x => x.textContent === 'Inspect the initial input ↑'), false);
  assert.deepEqual(f.requests, ['/api/demo/exhibit/waiters-squat-panda.json', '/api/demo/exhibit/body-model.json']);
  assert.equal(f.timers.size, 0, 'opening a page must not start playback');
  const details = f.nodes.find(x => x.className === 'studio-asset-inspect');
  details.open = true; await details.listeners.toggle();
  assert.equal(f.requests.length, 3);
  await details.listeners.toggle(); assert.equal(f.requests.length, 3, 'opening the same asset reuses its preview');
  f.events.pagehide();
  assert.ok(f.observers.every(x => x.disconnected));
  assert.ok(f.nodes.filter(x => x.tag === 'video').every(x => x.paused && !x.src));
});

test('skeleton playback is bounded, user-started and cancelled offscreen; repeated play does not overlap', () => {
  const f = fixture(), host = f.create();
  const stop = f.context.window.WeftStudioPage.modelViewer(host, model);
  const play = host.children[1].children[0];
  assert.equal(f.timers.size, 0);
  play.listeners.click(); assert.equal(f.timers.size, 1);
  const start = performance.now();
  for (let i = 0; i < 180; i++) f.tick(i * 33.333);
  const duration = performance.now() - start;
  assert.ok(f.points.every(p => p.every(Number.isFinite)));
  assert.equal(f.timers.size, 1);
  f.observers[0].callback([{ isIntersecting: false }]);
  assert.equal(f.timers.size, 0);
  assert.equal(play.attributes['aria-pressed'], 'false');
  stop(); assert.equal(f.timers.size, 0);
  console.log(`Recorded skeleton: 180 mocked-canvas frames in ${duration.toFixed(1)} ms; no idle RAF. Browser/GPU latency not measured.`);
});

test('missing model keeps the saved video and private stage downloads, with an explicit unavailable state', async () => {
  const f = fixture();
  f.context.fetch = async () => ({ ok: false });
  f.context.window.WeftStudioPage.open({ record: { id: 'run-1234', model_url: '/api/demo/studio/run-1234/model',
    steps: [{ id: 'intake', status: 'failed', detail: 'Pose gaps', artifact: { name: 'intake.zip' } }] },
    entry: { video_url: '/private.mp4' }, signed: value => value + '?token=secret' });
  await settle();
  assert.ok(f.nodes.find(x => x.tag === 'video').src.startsWith('/private.mp4'));
  assert.ok(f.nodes.some(x => x.href === '/api/demo/artifact/intake/run-1234?token=secret'));
  assert.ok(f.nodes.some(x => x.className === 'studio-stage-status' && x.textContent === 'failed'));
  assert.ok(f.nodes.some(x => x.className === 'studio-placeholder' && /not available/.test(x.textContent)));
  assert.equal(f.timers.size, 0);
  f.events.pagehide();
});

test('rolling controls reuse the shared physics engine and stop on page visibility changes', async () => {
  const f = fixture();
  const rolling = JSON.parse(await readFile(new URL('../assets/demo/studio/rolling-model.json', import.meta.url), 'utf8'));
  vm.runInNewContext(await readFile(new URL('../src/weft/assets/engine.js', import.meta.url), 'utf8'), f.context);
  f.context.window.WeftPhysics = f.context.WeftPhysics;
  const stop = f.context.window.WeftStudioPage.modelViewer(f.create(), rolling);
  f.nodes.find(x => x.attributes['aria-label'] === 'Try an authored push right').listeners.click();
  for (let i = 0; i < 40; i++) f.tick(i * 16.667);
  assert.ok(f.points.every(p => p.every(Number.isFinite)));
  f.context.document.hidden = true; f.events.visibilitychange();
  assert.equal(f.timers.size, 0); stop();
});


test('back-forward restoration remounts released media and model resources', async () => {
  const f = fixture();
  f.context.window.WeftStudioPage.open({ record: { id: 'waiters-squat-panda' },
    entry: { id: 'waiters-squat-panda', video_url: '/result.mp4' }, signed: value => value });
  await settle();
  f.events.pagehide();
  assert.ok(f.nodes.filter(x => x.tag === 'video').every(x => !x.src));
  f.events.pageshow({ persisted: true });
  await settle();
  assert.equal(f.body.children.filter(x => x.id === 'studio-tour').length, 1);
  assert.equal(f.nodes.filter(x => x.tag === 'video' && x.src).length, 2);
  assert.equal(f.timers.size, 0);
  f.events.pagehide();
});

test('pose controls edit connected limbs without changing the source or quietly claiming scored physics', () => {
  const f = fixture(), api = f.context.window.WeftStudioPage;
  const before = JSON.stringify(model.frames[0]);
  const original = model.frames[0].segments, ids = model.segment_ids;
  const altered = api.poseExperiment(original, original, ids, { motion: 1, arms: 1.2, legs: 1, joints: { left_elbow: 30 } });
  const upper = ids.indexOf('left_upper_arm'), lower = ids.indexOf('left_forearm');
  const length = segment => Math.hypot(...segment[1].map((value, index) => value-segment[0][index]));
  assert.ok(Math.abs(length(altered[upper]) / length(original[upper]) - 1.2) < 1e-10);
  assert.ok(Math.abs(length(altered[lower]) / length(original[lower]) - 1.2) < 1e-10);
  assert.equal(JSON.stringify(altered[upper][1]), JSON.stringify(altered[lower][0]));
  assert.equal(JSON.stringify(model.frames[0]), before);
  const stop = api.modelViewer(f.create(), model);
  const arm = f.nodes.find(item => item.attributes['aria-label'] === 'Arm length');
  arm.value = '1.2'; arm.listeners.input();
  assert.ok(f.nodes.some(item => item.className === 'studio-edit-note' && item.textContent === 'Modified pose'));
  stop();
});

function galleryFixture() {
  const f = fixture(), callbacks = new Map(), hints = new Map(), storage = new Map();
  const add = (key, fn) => { if (!callbacks.has(key)) callbacks.set(key, []); callbacks.get(key).push(fn); };
  f.context.window.addEventListener = (key, fn) => add('window-' + key, fn);
  f.context.document.addEventListener = (key, fn) => add('document-' + key, fn);
  f.context.setTimeout = (fn, delay) => { hints.set(1, { fn, delay }); return 1; };
  f.context.clearTimeout = id => hints.delete(id);
  f.context.sessionStorage = { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) };
  const elements = {};
  for (const id of ['demo-source-video', 'demo-video', 'gallery-play', 'gallery-play-label', 'gallery-seek', 'gallery-time', 'gallery-reset', 'gallery-interact', 'gallery-model', 'gallery-transport', 'gallery-status', 'demo-result-note', 'demo-comparison', 'finished-content', 'gallery-interact-hint', 'example-picker']) {
    const element = f.create(id.includes('video') ? 'video' : 'div'); element.id = id; elements[id] = element;
    const handlers = new Map();
    element.addEventListener = (key, fn) => {
      if (!handlers.has(key)) handlers.set(key, []); handlers.get(key).push(fn);
      element.listeners[key] = event => { const results = handlers.get(key).map(handler => handler(event)); return results.find(value => value?.then); };
    };
  }
  elements['gallery-play'].append(f.create('span'));
  for (const id of ['demo-source-video', 'demo-video']) {
    Object.assign(elements[id], { currentTime: 0, duration: 5, readyState: 4, ended: false, paused: true,
      play: async function () { this.paused = false; }, pause() { this.paused = true; } });
  }
  const gallery = f.context.window.WeftGallery();
  const record = { id: 'waiters-squat-panda', source_video_url: '/source.mp4', video_url: '/result.mp4', result_note: 'Saved result' };
  gallery.select(record, value => value);
  return { ...f, elements, gallery, record, hints, storage,
    emit(key, event) { for (const fn of callbacks.get(key) || []) fn(event); } };
}

test('paired playback seeks together, corrects drift, pauses while buffering and stops offscreen', async () => {
  const f = galleryFixture(), e = f.elements;
  assert.ok(f.observers[0].targets.includes(e['finished-content']), 'visibility includes the playback controls');
  e['gallery-play'].listeners.click(); await settle();
  assert.equal(e['demo-video'].paused, false); assert.equal(e['demo-source-video'].paused, false);
  assert.equal(f.timers.size, 1);
  e['demo-video'].currentTime = 2; e['demo-source-video'].currentTime = 1.8;
  f.tick(16); assert.equal(e['demo-source-video'].currentTime, 2);
  e['gallery-seek'].value = '3.2'; e['gallery-seek'].listeners.input();
  assert.equal(e['demo-video'].currentTime, 3.2); assert.equal(e['demo-source-video'].currentTime, 3.2);
  assert.equal(f.timers.size, 0);
  e['gallery-play'].listeners.click(); await settle();
  e['demo-source-video'].readyState = 2; e['demo-source-video'].listeners.waiting();
  assert.equal(e['demo-video'].paused, true); assert.equal(f.timers.size, 0);
  e['demo-source-video'].readyState = 4; e['demo-source-video'].listeners.canplay(); await settle();
  assert.equal(e['demo-video'].paused, false); assert.equal(f.timers.size, 1);
  f.observers[0].callback([{ isIntersecting: false }]);
  assert.equal(e['demo-video'].paused, true); assert.equal(f.timers.size, 0);
  f.emit('window-pagehide');
});

test('gallery inspection is optional, model-cached, and released before changing examples', async () => {
  const f = galleryFixture(), e = f.elements;
  assert.equal(f.requests.length, 0);
  await e['gallery-interact'].listeners.click(); await settle();
  assert.equal(e['gallery-interact'].attributes['aria-pressed'], 'true');
  assert.equal(e['demo-video'].hidden, true); assert.equal(e['gallery-transport'].hidden, true);
  assert.ok(f.nodes.some(item => item.attributes['aria-label'] === 'Joint bend offset'));
  assert.equal(f.requests.length, 2);
  e['gallery-interact'].listeners.click();
  assert.equal(e['demo-video'].hidden, false); assert.equal(e['gallery-model'].hidden, true);
  await e['gallery-interact'].listeners.click(); await settle();
  assert.equal(f.requests.length, 2, 'same example reuses its model');
  f.gallery.select({ id: 'run-abc', video_url: '/private.mp4', result_note: 'New result' }, value => value);
  assert.equal(e['gallery-interact'].hidden, true);
  assert.equal(e['gallery-model'].children.length, 0);
  assert.equal(f.timers.size, 0);
  f.emit('window-pagehide');
});

test('interaction hint appears once when visible and dismisses after 4.5 seconds', () => {
  const f = galleryFixture(), e = f.elements;
  const observer = f.observers.find(item => item.targets.includes(e['gallery-interact']));
  observer.callback([{ isIntersecting: true }]);
  assert.equal(e['gallery-interact-hint'].hidden, false);
  assert.equal(f.hints.get(1).delay, 4500);
  f.hints.get(1).fn();
  assert.equal(e['gallery-interact-hint'].hidden, true);
  assert.equal(e['gallery-interact'].attributes['aria-describedby'], undefined);
  observer.callback([{ isIntersecting: true }]);
  assert.equal(f.hints.size, 0);
  assert.equal(f.storage.get('weft-model-hint-seen'), 'yes');
  f.emit('window-pagehide');
});


test('chooser cues match each example and leave no animation running after exit, closure or reduced motion', () => {
  const f = fixture();
  f.context.document.createElementNS = (_, tag) => f.create(tag);
  const picker = f.create(); picker.open = true;
  const motions = f.context.window.WeftExampleMotions(picker);
  const ids = ['rolling-motion', 'bodyweight-squats', 'incline-press', 'deadlift', 'jumping-jacks', 'overhead-squat', 'waiters-squat', 'sots-press', 'lunge-twist', 'single-leg-squat'];
  const buttons = ids.map(id => { const button = f.create('button'); motions.decorate(button, id); return button; });
  assert.equal(new Set(buttons.map(button => button.children[0].children[1].attributes.d)).size, 10);
  assert.equal(f.timers.size, 0, 'opening the chooser does not animate all ten options');
  const ball = buttons[0], squat = buttons[1];
  ball.listeners.pointerenter({ pointerType: 'mouse' });
  f.tick(0); const start = ball.children[0].children[2].attributes.cx;
  f.tick(600); assert.notEqual(ball.children[0].children[2].attributes.cx, start);
  squat.listeners.focus();
  assert.equal(ball.attributes['data-motion'], 'false');
  assert.equal(f.timers.size, 1, 'only the active option animates');
  f.tick(0); const standing = squat.children[0].children[1].attributes.d;
  f.tick(1200); assert.notEqual(squat.children[0].children[1].attributes.d, standing);
  f.tick(2400); assert.equal(f.timers.size, 0, 'one finite movement per hover');
  squat.listeners.blur(); assert.equal(squat.attributes['data-motion'], 'false');
  ball.listeners.focus(); picker.open = false; picker.listeners.toggle();
  assert.equal(f.timers.size, 0);
  picker.open = true; ball.listeners.focus(); f.context.document.hidden = true; f.events.visibilitychange();
  assert.equal(f.timers.size, 0);
  f.context.document.hidden = false; f.context.window.matchMedia = () => ({ matches: true });
  squat.listeners.focus(); assert.equal(f.timers.size, 0, 'reduced motion shows a still cue');
  assert.equal(squat.attributes['data-motion'], 'true');
  f.events.pagehide(); assert.equal(squat.attributes['data-motion'], 'false');
  assert.equal(f.requests.length, 0, 'symbolic previews never download videos or models');
});

test('buffering during the initial play request resumes both videos instead of cancelling playback', async () => {
  const f = galleryFixture(), e = f.elements;
  for (const video of [e['demo-source-video'], e['demo-video']]) {
    let rejectPending;
    video.readyState = 2;
    video.play = function () {
      this.paused = false;
      if (this.readyState >= 3) return Promise.resolve();
      return new Promise((_, reject) => { rejectPending = reject; });
    };
    video.pause = function () {
      this.paused = true;
      if (rejectPending) { rejectPending(Object.assign(new Error('Play interrupted by pause'), { name: 'AbortError' })); rejectPending = null; }
    };
  }
  e['gallery-play'].listeners.click();
  assert.equal(e['demo-source-video'].preload, 'auto', 'explicit play allows buffering beyond metadata');
  assert.equal(e['demo-video'].preload, 'auto');
  e['demo-source-video'].listeners.waiting();
  await settle();
  assert.equal(e['gallery-play'].attributes['aria-pressed'], 'true', 'buffering retains the user’s play intent');
  assert.equal(e['gallery-status'].textContent, 'Buffering the comparison…');
  e['demo-source-video'].readyState = e['demo-video'].readyState = 4;
  e['demo-video'].listeners.canplay(); await settle();
  assert.equal(e['demo-source-video'].paused, false);
  assert.equal(e['demo-video'].paused, false);
  assert.equal(f.timers.size, 1);
  e['demo-source-video'].readyState = 2; e['demo-source-video'].listeners.waiting();
  e['gallery-play'].listeners.click();
  e['demo-source-video'].readyState = 4; e['demo-source-video'].listeners.canplay(); await settle();
  assert.equal(e['demo-video'].paused, true, 'explicit pause cancels automatic buffer recovery');
  assert.equal(f.timers.size, 0);
});

test('character popover prefills measured choices, preserves custom text and marks filled values', () => {
  const f = fixture();
  const elements = Object.fromEntries(['character-picker', 'prompt', 'character-trigger', 'character-indicator', 'character-note'].map(id => {
    const element = f.create(); element.id = id; element.value = ''; element.focus = () => { element.focused = true; }; return [id, element];
  }));
  const preference = f.context.window.WeftCharacterPreference();
  preference.prefill({ prompt: 'Panda' });
  assert.equal(elements.prompt.value, 'Panda');
  assert.equal(elements['character-trigger'].attributes['data-filled'], 'true');
  assert.match(elements['character-note'].textContent, /Reuses the existing character/);
  elements.prompt.value = 'A fox'; elements.prompt.listeners.input();
  preference.prefill({ prompt: 'Robot' });
  assert.equal(elements.prompt.value, 'A fox', 'a measured suggestion must not overwrite the user’s text');
  elements.prompt.value = ''; elements.prompt.listeners.input();
  assert.equal(elements['character-trigger'].attributes['data-filled'], 'false');
  preference.prefill({ prompt: 'Robot' }); assert.equal(elements.prompt.value, 'Robot');
  elements['character-picker'].open = true; elements['character-picker'].listeners.toggle();
  assert.equal(elements.prompt.focused, true);
  f.events.keydown({ key: 'Escape' });
  assert.equal(elements['character-picker'].open, false);
  assert.equal(elements['character-trigger'].focused, true);
});

test('private pipeline uses each stage’s own video and releases inline media', async () => {
  const f = fixture();
  f.context.window.WeftStudioPage.open({
    record: { id: 'run-test', steps: [
      { id: 'rollout', visuals: [{ kind: 'video', url: '/physics.mp4', poster: '/physics.jpg', label: 'Physics replay' }] },
      { id: 'avatar', visuals: [{ kind: 'video', url: '/appearance.mp4', label: 'Character render' }] },
    ] },
    entry: { video_url: '/final.mp4', source_video_url: '/source.mp4' }, signed: value => value,
  });
  await settle();
  const inline = f.nodes.filter(node => node.tag === 'video' && node.parent.className === 'stage-output');
  assert.deepEqual(inline.map(video => video.src), ['/physics.mp4', '/appearance.mp4']);
  assert.equal(inline[0].poster, '/physics.jpg');
  assert.ok(inline[1].listeners.loadedmetadata, 'legacy videos seek a preview frame when no poster exists');
  assert.equal(f.nodes.filter(node => node.tag === 'video' && node.src === '/final.mp4').length, 1);
  assert.ok(inline.every(video => video.controls && video.preload === 'metadata'));
  f.events.pagehide();
  assert.ok(inline.every(video => video.paused && !video.src));
});

test('step seven uses the appearance section video, poster and matching evidence instead of the imported preview', async () => {
  const f = fixture();
  const appearance = { id: 'avatar', status: 'done', detail: 'Character swap',
    visuals: [{ kind: 'video', url: '/appearance.mp4', poster: '/appearance.jpg', label: 'Character swap' }],
    artifact: { name: 'weft-avatar.zip' }, metrics: { swap: 'step 120' } };
  const imported = { id: 'create', status: 'done', experimental: true,
    visuals: [{ kind: 'video', url: '/broken-character.mp4' }], artifact: { name: 'weft-create.zip' } };
  const original = JSON.stringify([appearance, imported]);
  const steps = f.context.window.WeftPresentationSteps([appearance, imported]);
  assert.equal(steps[1].visuals, appearance.visuals);
  assert.equal(steps[1].artifact, appearance.artifact);
  assert.equal(steps[1].evidenceStep, 'avatar');
  assert.equal(steps[1].experimental, false);
  assert.equal(steps[1].title, 'Try another character.');
  assert.equal(JSON.stringify([appearance, imported]), original);
  f.context.window.WeftStudioPage.open({ record: { id: 'run-test', steps: [appearance, imported] },
    entry: { video_url: '/result.mp4' }, signed: value => value });
  await settle();
  assert.equal(f.nodes.filter(n => n.tag === 'video' && n.src === '/appearance.mp4').length, 2);
  assert.equal(f.nodes.some(n => n.src === '/broken-character.mp4'), false);
  assert.equal(f.nodes.filter(n => n.href === '/api/demo/artifact/avatar/run-test').length, 2);
  assert.equal(f.context.window.WeftPresentationSteps([imported])[0].visuals.length, 0, 'missing swap never falls back to the broken preview');
});
