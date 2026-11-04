import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtemp, mkdir } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { demoCommand, supervise } from './demo.mjs';

test('demo reuses the prepared collection and supports an explicit root and matched ports', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'weft-demo-test-'));
  assert.equal((await demoCommand([], root)).args[4], '.local/portal');
  await mkdir(path.join(root, '.local/demo-portal-04'), { recursive: true });
  const command = await demoCommand([], root);
  assert.equal(command.args[4], '.local/demo-portal-04');
  assert.equal(command.port, 3000);
  assert.equal(command.sandboxPort, 8765);
  const custom = await demoCommand(['--root', 'saved worlds/$(literal)', '--port=3001', '--sandbox-port', '8767', '--check'], root);
  assert.equal(custom.args[4], 'saved worlds/$(literal)');
  assert.equal(custom.args[6], '8767');
  assert.equal(custom.port, 3001);
  assert.equal(custom.check, true);
  assert.equal(custom.cwd, root);
});

test('invalid options and conflicting ports fail before starting either process', async () => {
  for (const args of [['--port=0'], ['--sandbox-port=65536'], ['--port=x'], ['--root'],
    ['--port=8765'], ['--root', '--check'], ['--host=0.0.0.0']]) {
    await assert.rejects(demoCommand(args));
  }
});

function fixture() {
  const child = new EventEmitter();
  const server = new EventEmitter();
  const signals = new EventEmitter();
  const kills = [];
  let closes = 0;
  child.kill = (signal) => { kills.push(signal); };
  server.close = (done) => { closes += 1; done(); };
  server.closeAllConnections = () => {};
  return { child, server, signals, kills, closes: () => closes };
}

test('Ctrl+C stops both once, waits for Python, and removes signal handlers', async () => {
  const f = fixture();
  const done = supervise(f.child, f.server, { signals: f.signals });
  f.signals.emit('SIGINT');
  f.signals.emit('SIGTERM');
  assert.deepEqual(f.kills, ['SIGINT']);
  assert.equal(f.closes(), 1);
  f.child.emit('exit', 0, null);
  assert.equal(await done, 0);
  assert.equal(f.signals.listenerCount('SIGINT'), 0);
  assert.equal(f.signals.listenerCount('SIGTERM'), 0);
});

test('a sandbox failure closes the landing page and preserves the failure exit code', async () => {
  const f = fixture();
  const done = supervise(f.child, f.server, { signals: f.signals });
  f.child.emit('exit', 7, null);
  assert.equal(await done, 7);
  assert.equal(f.closes(), 1);
  assert.deepEqual(f.kills, []);
});

test('spawn errors close the landing page without trying to kill a nonexistent process', async () => {
  const f = fixture();
  const errors = [];
  const done = supervise(f.child, f.server, { signals: f.signals, log: { error: (message) => errors.push(message) } });
  f.child.emit('error', new Error('Python could not start'));
  assert.equal(await done, 1);
  assert.equal(f.closes(), 1);
  assert.deepEqual(f.kills, []);
  assert.deepEqual(errors, ['Python could not start']);
});

test('a stuck child has a bounded shutdown', async () => {
  const f = fixture();
  f.child.kill = (signal) => {
    f.kills.push(signal);
    if (signal === 'SIGKILL') f.child.emit('exit', null, signal);
  };
  const keepAlive = setTimeout(() => {}, 1000);
  try {
    const done = supervise(f.child, f.server, { signals: f.signals, shutdownMs: 5 });
    f.signals.emit('SIGTERM');
    assert.equal(await done, 1);
    assert.deepEqual(f.kills, ['SIGINT', 'SIGKILL']);
  } finally { clearTimeout(keepAlive); }
});
