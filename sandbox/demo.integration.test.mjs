import assert from 'node:assert/strict';
import { spawn, execFile } from 'node:child_process';
import { once } from 'node:events';
import { access, mkdtemp, readFile, realpath, writeFile } from 'node:fs/promises';
import { createServer } from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import test from 'node:test';

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const python = path.join(repository, '.venv-lab/bin/python');

test('real Ctrl+C exits both demo processes and releases ports and the collection lock', { timeout: 20000 }, async (t) => {
  try { await access(python); } catch { t.skip('Requires the installed lab Python'); return; }
  const probes = [createServer(), createServer()];
  await Promise.all(probes.map(async (server) => { server.listen(0, '127.0.0.1'); await once(server, 'listening'); }));
  const ports = probes.map((server) => server.address().port);
  await Promise.all(probes.map((server) => new Promise((resolve) => server.close(resolve))));
  const root = await mkdtemp(path.join(await realpath(os.tmpdir()), 'weft-demo-shutdown-'));
  await writeFile(path.join(root, 'saved-evidence.txt'), 'preserve this');
  const child = spawn(process.execPath, [path.join(repository, 'sandbox/demo.mjs'),
    '--root', root, '--port', String(ports[0]), '--sandbox-port', String(ports[1])],
  { cwd: repository, stdio: ['ignore', 'pipe', 'pipe'] });
  let stdout = '', stderr = '';
  child.stdout.on('data', (data) => { stdout += data; });
  child.stderr.on('data', (data) => { stderr += data; });
  const exited = once(child, 'exit');
  try {
    const deadline = Date.now() + 12000;
    while (!stdout.includes('Weft portal:')) {
      assert.equal(child.exitCode, null, stderr.slice(-2000));
      assert.ok(Date.now() < deadline, 'Both servers must become ready');
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    const origin = `http://127.0.0.1:${ports[0]}`;
    const publicIndex = await fetch(origin + '/api/demo/showcase');
    assert.equal(publicIndex.status, 200);
    const showcase = await publicIndex.json();
    assert.equal(showcase.home_url, origin + '/');
    assert.ok(showcase.items.every((item) => item.readonly && !item.project));
    if (showcase.items.length) {
      const recording = await fetch(origin + showcase.items[0].video_url, { headers: { Range: 'bytes=0-31' } });
      assert.equal(recording.status, 206);
      assert.equal((await recording.arrayBuffer()).byteLength, 32);
      if (showcase.items[0].source_video_url) {
        const original = await fetch(origin + showcase.items[0].source_video_url, { headers: { Range: 'bytes=0-31' } });
        assert.equal(original.status, 206);
        assert.equal((await original.arrayBuffer()).byteLength, 32);
      }
    }
    const page = await fetch(origin + '/demo?access_code=example');
    assert.equal(page.status, 200);
    assert.equal(new URL(page.url).port, String(ports[0]));
    assert.match(await page.text(), /demo_build.js/);
    for (const asset of ['/demo_build.js', '/demo.css', '/brand.css', '/demo-font.woff2']) {
      const response = await fetch(origin + asset);
      assert.equal(response.status, 200, asset);
      assert.ok((await response.arrayBuffer()).byteLength > 0);
    }
    assert.equal((await fetch(origin + '/api/demo/steps')).status, 401);
    const accessCode = (await readFile(path.join(repository, '.local/demo-access-code'), 'utf8')).trim();
    const unlock = await fetch(origin + '/api/demo/unlock', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Origin: origin },
      body: JSON.stringify({ code: accessCode }),
    });
    assert.equal(unlock.status, 200);
    const session = await unlock.json();
    assert.equal(session.ok, true);
    const steps = await fetch(origin + '/api/demo/steps', { headers: { Authorization: 'Bearer ' + session.token } });
    assert.equal(steps.status, 200);
    assert.equal((await steps.json()).steps.length, 7);
    child.kill('SIGINT');
    const [code, signal] = await exited;
    assert.equal(signal, null);
    assert.equal(code, 0, stderr.slice(-2000));
    assert.match(stdout, /Weft demo stopped/);
    const result = await promisify(execFile)(python, [path.join(repository, 'sandbox/preflight.py'), root, ...ports.map(String)]);
    assert.deepEqual(JSON.parse(result.stdout), { directory_in_use: false, busy_ports: [] });
    assert.equal(await readFile(path.join(root, 'saved-evidence.txt'), 'utf8'), 'preserve this');
  } finally {
    if (child.exitCode === null && child.signalCode === null) child.kill('SIGINT');
  }
});
