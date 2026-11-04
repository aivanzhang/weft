import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const site = await readFile(new URL('./site.js', import.meta.url), 'utf8');
const source = site.slice(site.indexOf('// Keep explanations within the viewport'));

function fixture() {
  const docEvents = {}, winEvents = {};
  const timers = new Map();
  let timerId = 0;
  const terms = ['interactive', 'worlds', 'video'].map((name) => {
    const styles = {}, events = {};
    const anchor = { left: 260, top: 600, bottom: 650, width: 80 };
    const tip = {
      dataset: {},
      getBoundingClientRect: () => ({ width: 260, height: 100 }),
      style: { setProperty: (key, value) => { styles[key] = value; } },
    };
    const term = {
      styles, events, tip, anchor, dataset: { headlineTerm: name },
      querySelector: () => tip,
      getBoundingClientRect: () => anchor,
      addEventListener: (type, listener) => { events[type] = listener; },
      contains: (target) => target === term || target === tip,
    };
    return term;
  });
  const document = {
    hidden: false,
    querySelectorAll: () => terms,
    addEventListener: (type, listener) => { docEvents[type] = listener; },
  };
  vm.runInNewContext(source, {
    document,
    window: { innerWidth: 375, innerHeight: 700,
      addEventListener: (type, listener) => { winEvents[type] = listener; } },
    setTimeout: (callback) => { timers.set(++timerId, callback); return timerId; },
    clearTimeout: (id) => timers.delete(id),
  });
  return { terms, document, docEvents, winEvents, timers };
}

test('hover, focus and tap explanations stay onscreen and only one opens at a time', () => {
  const { terms: [first, second, third], docEvents } = fixture();
  first.events.pointerenter({ pointerType: 'mouse' });
  assert.equal(first.dataset.tipOpen, 'true');
  assert.equal(first.styles['--tip-x'], '99px');
  assert.equal(first.styles['--tip-y'], '486px');
  assert.equal(first.styles['--tip-arrow-x'], '201px');
  assert.equal(first.tip.dataset.placement, 'above');
  second.events.focus();
  assert.equal(first.dataset.tipOpen, undefined);
  assert.equal(second.dataset.tipOpen, 'true');
  third.events.click();
  assert.equal(second.dataset.tipOpen, undefined);
  assert.equal(third.dataset.tipOpen, 'true');
  docEvents.keydown({ key: 'Escape' });
  assert.equal(third.dataset.tipOpen, undefined);
});

test('tooltip hover grace is cancellable and dismissal cleans pending work', () => {
  const { terms: [term], document, docEvents, winEvents, timers } = fixture();
  term.events.focus();
  term.events.pointerleave();
  assert.equal(timers.size, 1);
  term.events.pointerenter({ pointerType: 'mouse' });
  assert.equal(timers.size, 0);
  docEvents.pointerdown({ target: term });
  assert.equal(term.dataset.tipOpen, 'true');
  docEvents.pointerdown({ target: {} });
  assert.equal(term.dataset.tipOpen, undefined);
  for (const event of ['scroll', 'resize', 'pagehide']) {
    term.events.focus();
    term.events.pointerleave();
    winEvents[event]();
    assert.equal(term.dataset.tipOpen, undefined);
    assert.equal(timers.size, 0);
  }
  term.events.focus();
  document.hidden = true;
  docEvents.visibilitychange();
  assert.equal(term.dataset.tipOpen, undefined);
});


test('tooltip pointers follow the word center, with room-aware placement at screen edges', () => {
  const { terms: [interactive, worlds, video] } = fixture();
  Object.assign(interactive.anchor, { left: 120, width: 80, top: 200, bottom: 250 });
  interactive.events.focus();
  assert.equal(interactive.styles['--tip-x'], '30px');
  assert.equal(interactive.styles['--tip-arrow-x'], '130px');
  assert.equal(interactive.styles['--tip-y'], '86px');
  Object.assign(worlds.anchor, { left: 5, width: 70, top: 30, bottom: 80 });
  worlds.events.focus();
  assert.equal(worlds.tip.dataset.placement, 'below');
  assert.equal(worlds.styles['--tip-x'], '16px');
  assert.equal(worlds.styles['--tip-arrow-x'], '24px');
  Object.assign(video.anchor, { left: 200, width: 100, top: 200, bottom: 250 });
  video.events.focus();
  assert.equal(video.tip.dataset.placement, 'below');
  assert.equal(video.styles['--tip-y'], '264px');
});
