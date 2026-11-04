import { execFile, spawn } from 'node:child_process';
import { access, stat } from 'node:fs/promises';
import { constants } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { landingServer } from '../website/dev.mjs';
import { sandboxCommand } from './run.mjs';
import { reloadState, watchSources } from './reload.mjs';
import { restartingBackend } from './reload-backend.mjs';

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const execute = promisify(execFile);

export function startupConflict(result, command) {
  const messages = [];
  if (result.directory_in_use) {
    messages.push(`The demo collection (${command.args[4]}) is already owned by another sandbox. Stop that sandbox with Ctrl+C; do not delete portal.lock.`);
  }
  for (const port of result.busy_ports) {
    messages.push(`Port ${port} is already in use. Find its listening process with: lsof -nP -iTCP:${port} -sTCP:LISTEN`);
  }
  if (messages.length) messages.push('No demo servers were started. Stop the old servers, then run npm run demo again.');
  return messages.join('\n');
}

export async function demoCommand(args, root = repository) {
  let projectRoot, port = 3000, sandboxPort = 8765, check = false, help = false, watch = true;
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === '--check') { check = true; continue; }
    if (argument === '--no-watch') { watch = false; continue; }
    if (argument === '--help' || argument === '-h') { help = true; continue; }
    const match = /^(--root|--port|--sandbox-port)(?:=(.*))?$/.exec(argument);
    if (!match) throw new Error(`Unknown demo option: ${argument}`);
    const value = match[2] ?? args[++index];
    if (!value || value.startsWith('--')) throw new Error(`${match[1]} requires a value`);
    if (match[1] === '--root') projectRoot = value;
    else {
      if (!/^\d+$/.test(value) || Number(value) < 1 || Number(value) > 65535) {
        throw new Error(`${match[1]} must be an integer from 1 to 65535`);
      }
      if (match[1] === '--port') port = Number(value);
      else sandboxPort = Number(value);
    }
  }
  if (port === sandboxPort) throw new Error('The landing page and sandbox need different ports');
  if (!projectRoot) {
    // Keep the prepared recordings visible instead of silently opening an empty collection.
    const prepared = await stat(path.join(root, '.local/demo-portal-04')).catch((error) => {
      if (error.code === 'ENOENT') return null;
      throw error;
    });
    projectRoot = prepared?.isDirectory() ? '.local/demo-portal-04' : '.local/portal';
  }
  return { ...sandboxCommand(['--root', projectRoot, '--port', String(sandboxPort)], root),
    port, sandboxPort, check, help, watch };
}

export function supervise(child, server, { signals = process, shutdownMs = 15000, log = console } = {}) {
  return new Promise((resolve) => {
    let stopping = false, exited = false, closed = false, code = 0, deadline;
    const finish = () => {
      if (!exited || !closed) return;
      clearTimeout(deadline);
      signals.removeListener('SIGINT', stop);
      signals.removeListener('SIGTERM', stop);
      server.removeListener('error', fail);
      resolve(code);
    };
    const stop = () => {
      if (stopping) return;
      stopping = true;
      server.close(() => { closed = true; finish(); });
      server.closeAllConnections();
      if (!exited) {
        child.kill('SIGINT');
        // Slow exports must finish without an automatic hard kill.
        deadline = setTimeout(() => log.warn('Still waiting for the sandbox to save its state and stop…'), shutdownMs);
        deadline.unref();
      }
    };
    const fail = (error) => { log.error(error.message); code = 1; stop(); };
    signals.on('SIGINT', stop);
    signals.on('SIGTERM', stop);
    server.once('error', fail);
    child.once('error', (error) => { exited = true; fail(error); finish(); });
    child.once('exit', (exitCode, signal) => {
      exited = true;
      if (!stopping) code = exitCode ?? (signal ? 1 : 0);
      else if (signal === 'SIGKILL') code = 1;
      stop();
      finish();
    });
  });
}

async function main() {
  const command = await demoCommand(process.argv.slice(2));
  if (command.help) {
    console.log('Usage: npm run demo -- [--root PATH] [--port 3000] [--sandbox-port 8765] [--no-watch] [--check]');
    console.log('Starts the landing page and sandbox together. Ctrl+C stops both and keeps saved runs.');
    return;
  }
  await access(command.executable, constants.X_OK).catch(() => {
    throw new Error('Sandbox Python is unavailable. Follow sandbox/README.md to set up .venv-lab.');
  });
  const { stdout } = await execute(command.executable, [path.join(repository, 'sandbox/preflight.py'),
    path.resolve(command.cwd, command.args[4]), String(command.port), String(command.sandboxPort)],
  { cwd: command.cwd, timeout: 5000, maxBuffer: 65536 });
  const conflict = startupConflict(JSON.parse(stdout), command);
  if (conflict) throw new Error(conflict);
  if (command.check) { console.log(JSON.stringify(command, null, 2)); return; }
  const hotReload = command.watch ? reloadState() : null;
  const server = landingServer({ sandboxPort: command.sandboxPort, hotReload });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(command.port, '127.0.0.1', () => {
      server.removeListener('error', reject);
      resolve();
    });
  }).catch((error) => {
    if (error.code === 'EADDRINUSE') throw new Error(`Port ${command.port} is already in use. Stop the existing landing server first.`);
    throw error;
  });
  console.log(`Weft landing page: http://127.0.0.1:${command.port}/`);
  console.log(`Weft walkthrough: http://127.0.0.1:${command.port}/demo`);
  console.log('Ctrl+C stops both servers. Saved runs and budgets are preserved.');
  // A separate process group prevents terminal Ctrl+C from interrupting Python twice.
  const child = hotReload ? restartingBackend(command, hotReload) : spawn(command.executable, command.args, {
    cwd: command.cwd, stdio: 'inherit', detached: process.platform !== 'win32',
    env: { ...process.env, PYTHONUNBUFFERED: '1', WEFT_HOME_PORT: String(command.port) },
  });
  let stopWatching = () => {};
  if (hotReload) {
    try {
      stopWatching = watchSources(repository, kind => {
        if (kind === 'backend') child.reload();
        else hotReload.refresh();
      });
      console.log('Hot refresh enabled. Page edits refresh automatically; backend edits restart when idle. --no-watch disables it.');
    } catch (error) { console.warn(`Hot refresh unavailable: ${error.message}`); }
    child.once('stopping', () => stopWatching());
  }
  process.exitCode = await supervise(child, server);
  stopWatching();
  if (process.exitCode === 0) console.log('Weft demo stopped. Saved runs are preserved.');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
