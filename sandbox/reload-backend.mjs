import { spawn } from 'node:child_process';
import { EventEmitter } from 'node:events';

// Present one child lifecycle to the shutdown supervisor across safe restarts.
export function restartingBackend(command, state, { spawnChild = spawn, output = process.stdout, errors = process.stderr, log = console } = {}) {
  const lifecycle = new EventEmitter();
  let child, stopping = false, ready = false, generation = 0, signalled = false, started = false;
  const request = () => {
    if (!stopping && ready && !signalled) {
      signalled = true;
      child.kill('SIGUSR1');
    }
  };
  function start() {
    ready = false; signalled = false;
    const loaded = generation;
    let tail = '';
    child = spawnChild(command.executable, command.args, {
      cwd: command.cwd, stdio: ['ignore', 'pipe', 'pipe'], detached: process.platform !== 'win32',
      env: { ...process.env, PYTHONUNBUFFERED: '1', WEFT_HOME_PORT: String(command.port), WEFT_DEMO_RELOAD: '1' },
    });
    child.stdout.pipe(output, { end: false });
    child.stderr.pipe(errors, { end: false });
    child.stdout.on('data', chunk => {
      if (ready) return;
      tail = (tail + chunk.toString()).slice(-2048);
      if (tail.includes('Weft portal:')) {
        ready = true; started = true;
        if (generation !== loaded) request();
        else state.publish();
      }
    });
    child.once('error', error => lifecycle.emit('error', error));
    child.once('exit', (code, signal) => {
      if (!stopping && code === 75) {
        log.log('Reloading the sandbox…');
        start();
      } else if (!stopping && started) {
        child = null; ready = false; state.pending();
        log.log('Sandbox stopped. Fix the error and save a Python file to retry. Ctrl+C exits.');
      } else lifecycle.emit('exit', code, signal);
    });
  }
  lifecycle.reload = () => {
    if (stopping) return;
    generation += 1; state.pending();
    if (!child) start();
    else request();
  };
  lifecycle.kill = () => {
    if (stopping) return;
    stopping = true;
    lifecycle.emit('stopping');
    if (child) child.kill('SIGINT');
    else lifecycle.emit('exit', 0, null);
  };
  start();
  return lifecycle;
}
