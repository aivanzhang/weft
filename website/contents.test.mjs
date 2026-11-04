import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const source = await readFile(new URL('./contents.js', import.meta.url), 'utf8');

test('contents is icon-only, follows page order, and links to real unique sections', async () => {
  const html = await readFile(new URL('./index.html', import.meta.url), 'utf8');
  const menu = html.match(/<details class="page-contents"[\s\S]*?<\/details>/)[0];
  const summary = menu.match(/<summary[\s\S]*?<\/summary>/)[0];
  assert.match(summary, /aria-label="Table of contents"/);
  assert.match(summary, /<svg/);
  assert.equal(summary.replace(/<[^>]*>/g, '').trim(), '');
  const targets = [...menu.matchAll(/href="#([^"]+)"/g)].map((match) => match[1]);
  assert.deepEqual(targets, ['overview', 'experiment', 'research', 'about', 'impact']);
  let previous = -1;
  for (const id of targets) {
    assert.equal([...html.matchAll(new RegExp(`id="${id}"`, 'g'))].length, 1, id);
    const position = html.indexOf(`id="${id}"`);
    assert.ok(position > previous, id);
    previous = position;
  }
});

function fixture() {
  const menuEvents = {}, documentEvents = {}, windowEvents = {};
  let triggerFocus = 0, targetFocus = 0;
  const section = { attributes: {},
    setAttribute(key, value) { this.attributes[key] = value; },
    focus(options) { assert.equal(options.preventScroll, true); targetFocus += 1; } };
  const contents = { open: true,
    querySelector: () => ({ focus() { triggerFocus += 1; } }),
    addEventListener: (type, listener) => { menuEvents[type] = listener; },
    contains: (target) => target === contents };
  vm.runInNewContext(source, {
    document: { querySelector: () => contents,
      getElementById: (id) => id === 'research' ? section : null,
      addEventListener: (type, listener) => { documentEvents[type] = listener; } },
    window: { addEventListener: (type, listener) => { windowEvents[type] = listener; } },
  });
  return { contents, section, menuEvents, documentEvents, windowEvents,
    focusCounts: () => [triggerFocus, targetFocus] };
}

test('section selection closes the menu and transfers focus without replacing native scrolling', () => {
  const f = fixture();
  f.menuEvents.click({ target: { closest: () => ({ hash: '#research' }) } });
  assert.equal(f.contents.open, false);
  assert.equal(f.section.attributes.tabindex, '-1');
  assert.deepEqual(f.focusCounts(), [0, 1]);
});

test('Escape restores the trigger; outside clicks close without stealing focus', () => {
  const f = fixture();
  let prevented = false;
  f.documentEvents.keydown({ key: 'Escape', preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  assert.equal(f.contents.open, false);
  assert.deepEqual(f.focusCounts(), [1, 0]);
  f.contents.open = true;
  f.documentEvents.pointerdown({ target: {} });
  assert.equal(f.contents.open, false);
  assert.deepEqual(f.focusCounts(), [1, 0]);
});

test('modified navigation, clicks inside the trigger, and page exit keep native behavior', () => {
  const f = fixture();
  f.menuEvents.click({ ctrlKey: true, target: { closest: () => ({ hash: '#research' }) } });
  assert.equal(f.contents.open, true);
  assert.deepEqual(f.focusCounts(), [0, 0]);
  f.documentEvents.pointerdown({ target: f.contents });
  assert.equal(f.contents.open, true);
  f.windowEvents.pagehide();
  assert.equal(f.contents.open, false);
});
