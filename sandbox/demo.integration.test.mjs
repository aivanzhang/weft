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
    const origin = `http://127.0.0.1:${ports[1]}`;
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
