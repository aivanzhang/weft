import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { EventEmitter, once } from 'node:events';
import { access, mkdtemp, mkdir, readFile, realpath, writeFile } from 'node:fs/promises';
import { createServer } from 'node:net';
import { Writable } from 'node:stream';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { demoCommand, supervise } from './demo.mjs';
import { reloadState, watchSources } from './reload.mjs';
import { restartingBackend } from './reload-backend.mjs';
import { landingServer } from '../website/dev.mjs';

const repository = path.resolve(import.meta.dirname, '..');
const until = async check => {
  const deadline = Date.now() + 10000;
  while (!check()) { assert.ok(Date.now() < deadline, 'Reload must finish'); await new Promise(r => setTimeout(r, 20)); }
};

test('real edits refresh HTML, restart Python once, retain access and shut down cleanly', { timeout: 20000 }, async t => {
  try { await access(path.join(repository, '.venv-lab/bin/python')); } catch { t.skip('Lab Python required'); return; }
  const root = await mkdtemp(path.join(await realpath(os.tmpdir()), 'weft-hot-refresh-'));
  const watched = path.join(root, 'watched');
  for (const folder of ['website', 'src/weft/assets', 'assets/demo', 'assets/brand/exports']) await mkdir(path.join(watched, folder), { recursive: true });
  const probe = createServer().listen(0, '127.0.0.1'); await once(probe, 'listening');
  const sandboxPort = probe.address().port; await new Promise(r => probe.close(r));
  const state = reloadState();
  const server = landingServer({ sandboxPort, hotReload: state }).listen(0, '127.0.0.1');
  await once(server, 'listening');
  const port = server.address().port, base = `http://127.0.0.1:${port}`;
  const command = await demoCommand(['--root', path.join(root, 'jobs'), '--port', String(port), '--sandbox-port', String(sandboxPort)]);
  const children = [];
  const output = new Writable({ write(chunk, encoding, done) { done(); } });
  const backend = restartingBackend(command, state, {
    spawnChild: (...args) => { const child = spawn(...args); children.push(child); return child; },
    output, errors: output, log: { log() {} },
  });
  const signals = new EventEmitter();
  const done = supervise(backend, server, { signals });
  const close = watchSources(watched, kind => kind === 'backend' ? backend.reload() : state.refresh());
  backend.once('stopping', close);
  try {
    await until(() => state.status().ready);
    const token = await readFile(path.join(root, 'jobs/.access-token'), 'utf8');
    for (const route of ['/website/index.html', '/demo', '/demo/studio/bodyweight-squats']) {
      const response = await fetch(base + route), html = await response.text();
      assert.equal(response.status, 200);
      assert.match(html, /src="\/__weft_reload.js" data-version="[a-z0-9-]+"/);
      assert.equal(Number(response.headers.get('content-length')), Buffer.byteLength(html));
    }
    assert.equal((await fetch(base + '/__weft_reload.js')).status, 200);
    assert.equal((await fetch(base + '/__weft_reload', { headers: { Origin: 'https://other.example' } })).status, 403);
    const original = state.status().revision;
    await writeFile(path.join(watched, 'website/styles.css'), 'body{}');
    await until(() => state.status().revision !== original);
    assert.equal(children.length, 1, 'CSS must not restart Python');
    const beforeRestart = state.status().revision, start = performance.now();
    await writeFile(path.join(watched, 'src/weft/model.py'), '# edited');
    await writeFile(path.join(watched, 'src/weft/assets/demo_examples.json'), '[]');
    await until(() => children.length === 2 && state.status().ready && state.status().revision !== beforeRestart);
    assert.notEqual(children[0].pid, children[1].pid);
    assert.equal(children[0].exitCode, 75);
    assert.equal(await readFile(path.join(root, 'jobs/.access-token'), 'utf8'), token);
    assert.equal((await fetch(base + '/api/demo/showcase')).status, 200);
    assert.deepEqual(await (await fetch(base + '/__weft_reload')).json(), state.status());
    console.log(`Debounced edit to restarted backend ready: ${(performance.now() - start).toFixed(0)} ms (isolated idle collection).`);
  } finally {
    close(); signals.emit('SIGINT');
    assert.equal(await done, 0);
  }
  assert.equal(server.listening, false);
  assert.equal(children.at(-1).exitCode, 0);
});
