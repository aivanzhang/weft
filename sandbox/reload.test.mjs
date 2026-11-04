import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough, Writable } from 'node:stream';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { changeKind, reloadState, watchSources } from './reload.mjs';
import { restartingBackend } from './reload-backend.mjs';

const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
test('watch only source assets, coalesce atomic saves, ignore evidence and caches', async () => {
  for (const name of ['.local/job/status.json', 'src/weft/__pycache__/portal.pyc', 'website/dev.test.mjs', 'website/.save.css']) assert.equal(changeKind(name), null);
  assert.equal(changeKind('src/weft/portal.py'), 'backend');
  assert.equal(changeKind('src/weft/assets/demo_examples.json'), 'backend');
  assert.equal(changeKind('assets/demo/example/source.mp4'), 'page');
  const root = await mkdtemp(path.join(os.tmpdir(), 'weft-watch-'));
  for (const folder of ['website', 'src/weft/assets', 'assets/demo', 'assets/brand/exports']) await mkdir(path.join(root, folder), { recursive: true });
  const changes = [];
  const close = watchSources(root, kind => changes.push(kind), { delay: 30 });
  try {
    await writeFile(path.join(root, 'website/styles.css'), 'body{}');
    await writeFile(path.join(root, 'src/weft/assets/demo_examples.json'), '[]');
    await writeFile(path.join(root, 'src/weft/model.py'), '# changed');
    for (let i = 0; !changes.length && i < 100; i++) await wait(10);
    assert.deepEqual(changes, ['backend']);
  } finally { close(); }
  await writeFile(path.join(root, 'website/styles.css'), 'body{color:red}');
  await wait(80);
  assert.deepEqual(changes, ['backend']);
});

test('restart waits for child exit, coalesces edits, and Ctrl+C cannot respawn it', () => {
  const children = [], state = reloadState();
  const output = new Writable({ write(chunk, encoding, done) { done(); } });
  const spawnChild = () => {
    const child = new EventEmitter();
    child.stdout = new PassThrough(); child.stderr = new PassThrough(); child.kills = [];
    child.kill = signal => child.kills.push(signal); children.push(child); return child;
  };
  const backend = restartingBackend({ args: [], port: 3000 }, state, { spawnChild, output, errors: output, log: { log() {} } });
  const exits = []; backend.on('exit', (...args) => exits.push(args));
  children[0].stdout.write('Weft por'); children[0].stdout.write('tal: ready\n');
  assert.equal(state.status().ready, true);
  const original = state.status().revision;
  backend.reload(); backend.reload();
  assert.deepEqual(children[0].kills, ['SIGUSR1']);
  assert.equal(children.length, 1);
  state.refresh(); assert.equal(state.status().ready, false);
  children[0].emit('exit', 75, null);
  assert.equal(children.length, 2);
  children[1].stdout.write('Weft portal: ready\n');
  assert.equal(state.status().ready, true);
  assert.notEqual(state.status().revision, original);
  backend.reload(); backend.kill(); backend.kill();
  children[1].emit('exit', 75, null);
  assert.deepEqual(children[1].kills, ['SIGUSR1', 'SIGINT']);
  assert.equal(children.length, 2);
  assert.deepEqual(exits, [[75, null]]);
});

test('a broken backend edit waits for the next save instead of a restart loop', () => {
  const children = [], state = reloadState();
  const output = new Writable({ write(chunk, encoding, done) { done(); } });
  const backend = restartingBackend({ args: [], port: 3000 }, state, {
    output, errors: output, log: { log() {} },
    spawnChild: () => {
      const child = new EventEmitter(); child.stdout = new PassThrough(); child.stderr = new PassThrough(); child.kill = () => {};
      children.push(child); return child;
    },
  });
  children[0].stdout.write('Weft portal: ready');
  backend.reload(); children[0].emit('exit', 75, null);
  children[1].emit('exit', 1, null);
  assert.equal(state.status().ready, false);
  assert.equal(children.length, 2);
  backend.reload(); assert.equal(children.length, 3);
  children[2].stdout.write('Weft portal: ready');
  assert.equal(state.status().ready, true);
  backend.kill(); children[2].emit('exit', 0, null);
});

test('partial watcher setup releases watchers already opened', () => {
  let closed = 0, calls = 0;
  assert.throws(() => watchSources('/unused', () => {}, { watchFile: () => {
    if (++calls === 2) throw new Error('watch unavailable');
    return { on() {}, close() { closed++; } };
  } }), /watch unavailable/);
  assert.equal(closed, 1);
});
