import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import test from 'node:test';

test('hot refresh polls serially, pauses hidden pages and reloads only a ready revision', async () => {
  const timers = new Map(), events = {}, requests = [];
  let next = 0, reloads = 0;
  const document = { hidden: false, currentScript: { dataset: { version: 'old' } }, addEventListener: (key, fn) => { events[key] = fn; } };
  const context = { document, window: { addEventListener: (key, fn) => { events[key] = fn; } },
    location: { reload: () => reloads++ }, AbortController,
    setTimeout: (fn, ms) => { timers.set(++next, { fn, ms }); return next; }, clearTimeout: id => timers.delete(id),
    fetch: (_, { signal }) => new Promise((resolve, reject) => {
      requests.push(state => resolve({ ok: true, json: async () => state }));
      signal.addEventListener('abort', () => reject(new Error('aborted')));
    }),
  };
  const settle = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
  vm.runInNewContext(await readFile(new URL('./hot-refresh.js', import.meta.url), 'utf8'), context);
  events.pageshow(); assert.equal(requests.length, 1);
  requests[0]({ revision: 'new', ready: false }); await settle();
  assert.equal(reloads, 0);
  document.hidden = true; events.visibilitychange();
  assert.equal(timers.size, 0);
  document.hidden = false; events.visibilitychange();
  assert.equal(requests.length, 2);
  requests[1]({ revision: 'new', ready: true }); await settle();
  assert.equal(reloads, 1); assert.equal(timers.size, 0);
  events.pagehide(); assert.equal(timers.size, 0);
});
